"""Bounded F32 QK/PV reductions and ordinary BF16 cuBLAS, after cache replay.

Runs with the shared ABI/context from check-nvrtc.py and immutable capture 10.
--attention-torch uses the pinned venv and matching wheel libraries; otherwise
the same cuBLAS calls use the runtime toolkit libraries, without importing Torch.
"""
import os

stages_report = dict(status="running", diagnostic=True, cases=[])
def stages_persist():
    (output_directory / "attention-stages.json").write_text(json.dumps(stages_report,indent=2)+"\n")

def save_stage(name, data, expected):
    (output_directory / (name+".bf16")).write_bytes(data)
    cmp = comparison(data,expected["data"],3)
    stages_report["cases"].append(dict(name=name,comparison=cmp))
    stages_persist()
    print(name, "exact",cmp["exact"],"/",cmp["elements"],"beyond",cmp["beyondAllowance"],flush=True)

def narrow_f32(data):
    return b"".join(struct.pack("<H", ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 65535)
                    for (bits,) in struct.iter_unpack("<I",data))

candidate_source = r'''
extern "C" __global__ void et_attention_dot_stages(CudaKernelArgs a) {
    et_u64 i = et_thread(); if (i >= a.elements) return;
    et_u64 positions = a.integers[13], tokens = a.integers[7], qheads = a.integers[11], dim = a.integers[10];
    et_u64 item = i / (a.operation ? dim : positions);
    et_u64 sequence = item / (tokens*qheads), head = (item/tokens)%qheads;
    et_u64 kh = head * et_shape(a,1)[1] / qheads;
    unsigned int count = a.integers[14];
    float p[32] = {};
    et_u64 length = a.operation ? positions : dim;
    for (et_u64 j=0;j<length;++j) {
        float x = a.operation ? et_load<float>(a.inputs[5],a.input_dtypes[5],item*positions+j)
                              : et_load<float>(a.inputs[0],a.input_dtypes[0],item*dim+j);
        float y = a.operation ? et_cache_load(a,1,sequence,j,kh,i%dim,dim)
                              : et_cache_load(a,0,sequence,i%positions,kh,j,dim);
        p[j%count] += x*y;
    }
    if (a.integers[15]) {
        for (unsigned int step=count/2;step;step>>=1)
            for (unsigned int j=0;j<step;++j) p[j] += p[j+step];
    } else for (unsigned int j=1;j<count;++j) p[0] += p[j];
    ((float*)a.output)[i] = p[0];
}
'''
(output_directory / "dot-stages.cu").write_text(candidate_source)
modules["attention_dot_stages"] = compile_module("attention_dot_stages",header+text("cache.cu")+candidate_source)

try:
    if options.attention_torch:
        import torch
        assert torch.__version__ == "2.10.0+cu128"
        assert not torch.backends.cuda.matmul.allow_tf32
        assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        torch.set_grad_enabled(False)
        def gpu(t):
            contiguous=torch.frombuffer(bytearray(t["data"]),dtype=torch.bfloat16).reshape(t["shape"]).cuda()
            restored=torch.empty_strided(t["shape"],t["original"]["stride"],dtype=torch.bfloat16,device="cuda")
            return restored.copy_(contiguous)
        def raw(t): return t.contiguous().cpu().view(torch.uint8).numpy().tobytes()
        qt,kt,vt,pt = gpu(q),gpu(k),gpu(v),gpu(probabilities)
        kt,vt = kt.repeat_interleave(qheads//kheads,dim=1),vt.repeat_interleave(qheads//kheads,dim=1)
        qt_result = torch.matmul(qt,kt.transpose(-1,-2))
        pv_result = torch.matmul(pt,vt).transpose(1,2)
        assert raw(qt_result)==rounded_qk["data"]
        assert raw(pv_result)==context["data"]
        save_stage("torch-qk",raw(qt_result),rounded_qk)
        save_stage("torch-pv",raw(pv_result),context)
        save_stage("torch-contiguous-qk",raw(torch.matmul(qt.contiguous(),kt.transpose(-1,-2))),rounded_qk)
        for label,left,right,expected in [("qk",qt,kt.transpose(-1,-2),rounded_qk),("pv",pt,vt,context)]:
            # TF32 remains off; this is an independent F32 GEMM diagnostic.
            result = torch.matmul(left.float(),right.float())
            if label=="pv": result=result.transpose(1,2)
            (output_directory/("torch-f32-"+label+".f32")).write_bytes(raw(result))
            save_stage("torch-f32-"+label,raw(result.to(torch.bfloat16)),expected)
        stages_report["torch"] = dict(version=torch.__version__,tf32=False,bf16ReducedPrecisionReduction=True,workspaceConfig=os.environ["CUBLAS_WORKSPACE_CONFIG"])
    for operation,expected,label in [(0,rounded_qk,"qk"),(1,context,"pv")]:
        for partials in [1,2,4,8,16,32]:
            for tree in ([False] if partials==1 else [False,True]):
                d = Args.from_buffer_copy(a); d.operation=operation
                d.elements=batch*qheads*tokens*(dim if operation else positions)
                d.inputs[5]=upload(probabilities["data"]);d.input_dtypes[5]=3
                d.integers[14]=partials;d.integers[15]=int(tree)
                d.output=protected(bytes(d.elements*4))
                launch("attention_dot_stages","et_attention_dot_stages",d)
                data=read(d.output,d.elements*4)
                name=f"f32-{label}-partials{partials}-"+("tree" if tree else "serial")
                if operation: data=token_first(data,qheads,tokens,4)
                (output_directory/(name+".f32")).write_bytes(data)
                narrowed=narrow_f32(data)
                if not operation and partials==1:
                    assert narrowed==(output_directory/"native-rounded-qk.bin").read_bytes()
                if operation and partials==32 and tree:
                    assert narrowed==(output_directory/"official-probabilities-native-pv.bin").read_bytes()
                save_stage(name,narrowed,expected)
    # Ordinary BF16 strided batched GEMM with explicit F32 accumulation.
    # Expanded GQA banks retain each head in ordinary [B,H,T,D] matrix layout.
    assert batch==1 and q["dtype"]==3
    blas=library("cublas"); handle=c.c_void_p(); checked(blas.cublasCreate_v2(c.byref(handle)))
    version=c.c_int();checked(blas.cublasGetVersion_v2(handle,c.byref(version)))
    stages_report["cublasVersion"]=version.value
    workspace=upload(bytes(32<<20))
    def expanded(t):
        size=positions*dim*2
        return b"".join(t["data"][(h*kheads//qheads)*size:(h*kheads//qheads+1)*size] for h in range(qheads))
    kp,vp=upload(expanded(k)),upload(expanded(v))
    qp,pp=upload(q["data"]),upload(probabilities["data"])
    qp_original=upload(token_first(q["data"],qheads,tokens,2))
    try:
        for label,ap,bp,n,m,inner,trans,lda,ldb,stridea,strideb,expected in [
            ("qk",kp,qp,positions,tokens,dim,1,dim,dim,positions*dim,tokens*dim,rounded_qk),
            ("qk-original-layout",kp,qp_original,positions,tokens,dim,1,dim,qheads*dim,positions*dim,dim,rounded_qk),
            ("pv",vp,pp,dim,tokens,positions,0,dim,positions,positions*dim,tokens*positions,context)]:
            for mode in [0,16]:
                for dtype,width in [(14,2),(0,4)]:
                    checked(blas.cublasSetMathMode(handle,mode))
                    checked(blas.cublasSetWorkspace_v2(handle,c.c_void_p(workspace),c.c_size_t(32<<20)))
                    dest=protected(bytes(qheads*m*n*width)); alpha,beta=c.c_float(1),c.c_float(0)
                    checked(blas.cublasGemmStridedBatchedEx(handle,trans,0,n,m,inner,c.byref(alpha),
                        c.c_void_p(ap),14,lda,c.c_longlong(stridea),c.c_void_p(bp),14,ldb,c.c_longlong(strideb),
                        c.byref(beta),c.c_void_p(dest),dtype,n,c.c_longlong(m*n),qheads,68,-1))
                    checked(driver.cuCtxSynchronize()); data=read(dest,qheads*m*n*width)
                    if label=="pv": data=token_first(data,qheads,tokens,width)
                    name=f"cublas-{label}-mode{mode}-"+("bf16" if width==2 else "f32")
                    if width==4:
                        (output_directory/(name+".f32")).write_bytes(data)
                        data=narrow_f32(data)
                    save_stage(name,data,expected)
        if options.attention_padding:
            checked(blas.cublasSetMathMode(handle,16))
            checked(blas.cublasSetWorkspace_v2(handle,c.c_void_p(workspace),c.c_size_t(32<<20)))
            for padded in [positions,304,320,512,528,544,560,1024]:
                def pad_rows(data,rows,width,padded_rows):
                    return b"".join(data[h*rows*width*2:(h+1)*rows*width*2]+bytes((padded_rows-rows)*width*2) for h in range(qheads))
                padded_k=upload(pad_rows(expanded(k),positions,dim,padded))
                padded_v=upload(pad_rows(expanded(v),positions,dim,padded))
                scores=protected(bytes(qheads*tokens*padded*4))
                alpha,beta=c.c_float(1),c.c_float(0)
                checked(blas.cublasGemmStridedBatchedEx(handle,1,0,padded,tokens,dim,c.byref(alpha),
                    c.c_void_p(padded_k),14,dim,c.c_longlong(padded*dim),c.c_void_p(qp),14,dim,c.c_longlong(tokens*dim),
                    c.byref(beta),c.c_void_p(scores),0,padded,c.c_longlong(tokens*padded),qheads,68,-1))
                checked(driver.cuCtxSynchronize())
                scores_raw=read(scores,qheads*tokens*padded*4)
                compact=b"".join(scores_raw[r*padded*4:(r*padded+positions)*4] for r in range(qheads*tokens))
                score_bf16=narrow_f32(compact)
                save_stage(f"pipeline-pad{padded}-qk",score_bf16,rounded_qk)
                soft=Args.from_buffer_copy(scores_args);soft.inputs[6]=upload(score_bf16);soft.input_dtypes[6]=3
                soft.output=protected(bytes(len(probabilities["data"])))
                launch_diagnostic("et_replay_softmax",soft,qheads*tokens*32)
                probs=read(soft.output,len(probabilities["data"]))
                save_stage(f"pipeline-pad{padded}-probabilities",probs,probabilities)
                padded_probs=upload(b"".join(probs[r*positions*2:(r+1)*positions*2]+bytes((padded-positions)*2) for r in range(qheads*tokens)))
                dest=protected(bytes(qheads*tokens*dim*4))
                checked(blas.cublasGemmStridedBatchedEx(handle,0,0,dim,tokens,padded,c.byref(alpha),
                    c.c_void_p(padded_v),14,dim,c.c_longlong(padded*dim),c.c_void_p(padded_probs),14,padded,c.c_longlong(tokens*padded),
                    c.byref(beta),c.c_void_p(dest),0,dim,c.c_longlong(tokens*dim),qheads,68,-1))
                checked(driver.cuCtxSynchronize())
                data=token_first(read(dest,qheads*tokens*dim*4),qheads,tokens,4)
                (output_directory/f"pipeline-pad{padded}-pv.f32").write_bytes(data)
                save_stage(f"pipeline-pad{padded}-context",narrow_f32(data),context)
    finally:
        checked(blas.cublasDestroy_v2(handle))
    for ptr,size in guarded:
        assert read(ptr-16,16)==guards and read(ptr+size,16)==guards
    stages_report["storageGuardsIntact"]=True
    stages_report["loadedLibraries"]=sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if any(name in line for name in ["libcublas","libcuda.so","libcudart"])})
    stages_report["status"]="completed"
except BaseException as error:
    stages_report["status"]="failed";stages_report["error"]=repr(error);raise
finally:
    stages_persist()
