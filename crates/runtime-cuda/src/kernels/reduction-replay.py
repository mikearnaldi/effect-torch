"""Bounded same-input F32 Sum diagnostic, invoked by check-nvrtc.py.

Production kernels compile unchanged. Only the diagnostic Sum schedule varies.
No model execution or checkpoint GEMM runs here; all 278 routing rows remain.
After integration, pass --reduction-old-source pointing to archived run-46
compute.cu to retain the old-kernel control. sum_tests.rs tests production.
"""
import hashlib
import json
import os
import torch

def option(name): return Path(sys.argv[sys.argv.index(name)+1])
capture=option("--reduction-replay")
effect=option("--reduction-effect")
boundary=option("--reduction-boundary")
checkpoint=option("--reduction-checkpoint")
output=option("--reduction-output")
output.mkdir(parents=True,exist_ok=False)
report=dict(status="running",diagnostic=True,stages={},generic=[])
def persist(): (output/"report.json").write_text(json.dumps(report,indent=2)+chr(10))
def raw(value): return value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
def decode(data): return [v[0] for v in struct.iter_unpack("<f",data)]
def pack(values): return struct.pack("<"+"f"*len(values),*values)
def save(name,data):
    (output/(name+".f32")).write_bytes(data)
    return hashlib.sha256(data).hexdigest()
def compare(name,data,expected,exact=False):
    a,b=decode(data),decode(expected);assert len(a)==len(b)
    diffs=[i for i,(x,y) in enumerate(zip(a,b)) if struct.pack("<f",x)!=struct.pack("<f",y)]
    report["stages"][name]=dict(elements=len(a),different=len(diffs),maxAbsoluteError=max([abs(a[i]-b[i]) for i in diffs] or [0]),sha256=save(name,data),first=[dict(index=i,actual=a[i],expected=b[i]) for i in diffs[:8]])
    persist();print(name,"diff",len(diffs),"/",len(a),flush=True)
    if exact: assert data==expected,name
def actual(name): return (effect/("encoder."+name+".f32")).read_bytes()
def tensor(path,name):
    with path.open("rb") as f:
        n=struct.unpack("<Q",f.read(8))[0];h=json.loads(f.read(n));entry=h[name]
        start,end=entry["data_offsets"];f.seek(8+n+start);data=f.read(end-start)
    return entry,data
def expanded(entry,data):
    if entry["dtype"]=="F32":return data
    assert entry["dtype"]=="BF16"
    return b"".join(struct.pack("<I",v[0]<<16) for v in struct.iter_unpack("<H",data))
def ref(name):return tensor(capture/"layer0.safetensors","encoder.layer0."+name)
def gpu(data,shape):return torch.frombuffer(bytearray(data),dtype=torch.float32).reshape(shape).cuda()

guards=[]
def buffer(size):
    size=max(size,4);allocation=upload(bytes([165])*16+bytes(size)+bytes([165])*16)
    guards.append((allocation,size));return allocation+16
def args_for(shapes,out_shape,tail=()):
    a=Args();a.elements=math.prod(out_shape);a.output_dtype=a.compute_dtype=1
    for i in range(len(shapes)):a.input_dtypes[i]=1
    a.output=buffer(a.elements*4);a.scratch[3]=upload(bytes(4))
    meta=[len(out_shape)]+[len(s) for s in shapes]+[0]*(8-len(shapes))+list(out_shape)+sum([list(s) for s in shapes],[])+list(tail)
    a.metadata=upload(struct.pack("<"+"Q"*len(meta),*meta))
    return a
def execute(module,name,a,work=None):
    if a.elements:
        if work is None:launch(module,name,a)
        else:
            function=c.c_void_p();checked(driver.cuModuleGetFunction(c.byref(function),modules[module],name.encode()))
            params=(c.c_void_p*1)(c.addressof(a))
            checked(driver.cuLaunchKernel(function,min(65535,(work+255)//256),1,1,256,1,1,0,None,params,None))
            checked(driver.cuCtxSynchronize())
    assert read(a.scratch[3],4)==bytes(4),name
    return read(a.output,a.elements*4)
def reduce(data,shape,dims,warp=False,op=0):
    dims=[d%len(shape) for d in dims];out_shape=[1 if i in dims else n for i,n in enumerate(shape)]
    a=args_for([shape],out_shape,dims);a.inputs[0]=upload(data or bytes(4));a.operation=op;a.integers[0]=len(dims)
    return execute("reduction_candidate" if warp else "tensor_f32","et_sum_warp" if warp else "et_reduce",a,a.elements*32 if warp else None)
def binary(left,lshape,right,rshape,op):
    a=args_for([lshape,rshape],lshape);a.inputs[0]=upload(left);a.inputs[1]=upload(right);a.operation=op
    return execute("typed","et_binary",a)
def exponent(data,shape):
    a=args_for([shape],shape);a.inputs[0]=upload(data);a.operation=3
    return execute("typed","et_unary",a)
def gather(data,indices,width):
    return b"".join(data[(i//8*width+expert)*4:(i//8*width+expert+1)*4] for i,expert in enumerate(indices))

diagnostic=r'''
extern "C" __global__ void et_sum_warp(CudaKernelArgs a) {
    unsigned int rank=et_meta(a)[1],lane=threadIdx.x&31U;
    if (rank>64) { et_error(a,4);return; }
    const et_u64 *shape=et_shape(a,0),*dims=et_tail(a);
    et_u64 mask=0,count=1;
    for(et_u64 i=0;i<a.integers[0];++i) mask|=1ULL<<dims[i];
    for(unsigned int i=0;i<rank;++i) if(mask&(1ULL<<i)) count*=shape[i];
    for(et_u64 row=et_thread()/32;row<a.elements;row+=(et_u64)gridDim.x*(blockDim.x/32)) {
        float sum=0.0f;
        for(et_u64 r=lane;r<count;r+=32) {
            et_u64 out=row,reduced=r,source=0,stride=1;
            for(int d=(int)rank-1;d>=0;--d) {
                bool selected=mask&(1ULL<<d);
                et_u64 coordinate=selected?reduced%shape[d]:out%shape[d];
                if(selected) reduced/=shape[d];else out/=shape[d];
                source+=coordinate*stride;stride*=shape[d];
            }
            sum+=((const float*)a.inputs[0])[source];
        }
        for(unsigned int offset=16;offset;offset>>=1) sum+=__shfl_down_sync(0xffffffffU,sum,offset);
        if(!lane) ((float*)a.output)[row]=sum;
    }
}
'''

try:
    persist()
    assert torch.__version__=="2.10.0+cu128"
    assert torch.version.git_version=="449b1768410104d3ed79d3bcfe4ba1d65c7f22c0"
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is True
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG")==":4096:8"
    torch.set_grad_enabled(False)
    report["torch"]=dict(version=torch.__version__,git=torch.version.git_version,cuda=torch.version.cuda,tf32=False,bf16ReducedPrecisionReduction=True,workspaceConfig=os.environ["CUBLAS_WORKSPACE_CONFIG"])
    for name in ["reduction-replay.py","check-nvrtc.py","tensor.cu","typed.cuh","typed.cu","common.cuh","compute.cu"]:
        (output/name).write_bytes((ROOT/name).read_bytes())
    (output/"device.rs").write_bytes((ROOT.parent/"device.rs").read_bytes())
    (output/"old-control-compute.cu").write_text(wrapper)
    report["oldControlSourceSha256"]=hashlib.sha256(wrapper.encode()).hexdigest()
    (output/"diagnostic.cu").write_text(header+diagnostic)
    modules["reduction_candidate"]=compile_module("reduction_candidate",header+diagnostic)
    rows,width,top=278,128,8
    scores=actual("native.scores");indices=list(map(int,decode(actual("native.indices"))))
    official=expanded(*ref("router.output.0"));official_weights=expanded(*ref("router.output.1"))
    ids_entry,ids_raw=ref("router.output.2");assert ids_entry["dtype"]=="I64"
    official_ids=[v[0] for v in struct.iter_unpack("<q",ids_raw)]
    (output/"official-indices.i64").write_bytes(ids_raw)
    (output/"native-indices.u32").write_bytes(struct.pack("<"+"I"*len(indices),*indices))
    aligned=b"".join(official_weights[(r*top+official_ids[r*top:(r+1)*top].index(e))*4:(r*top+official_ids[r*top:(r+1)*top].index(e)+1)*4] for r in range(rows) for e in indices[r*top:(r+1)*top])
    assert aligned==actual("officialSelectedWeights.weights")
    assert actual("native.hidden")==(boundary/"production.encoder.layer1.inputNorm.input.f32").read_bytes()
    assert actual("officialSelectedWeights.hidden")==expanded(*ref("output"))
    report["hiddenGuards"]=dict(native44EqualsActual43=hashlib.sha256(actual("native.hidden")).hexdigest(),selectedWeightResetEqualsOfficial=hashlib.sha256(actual("officialSelectedWeights.hidden")).hexdigest(),elements=782848,ffnNotRerun=True)
    name="model.decoder.layers.0.router.per_expert_scale"
    index=json.loads((checkpoint/"model.safetensors.index.json").read_text())
    scale_entry,scale_raw=tensor(checkpoint/index["weight_map"][name],name)
    assert scale_entry==dict(dtype="BF16",shape=[128],data_offsets=scale_entry["data_offsets"])
    (output/"checkpoint-expert-scale.bf16").write_bytes(scale_raw)
    scale_all=expanded(scale_entry,scale_raw)
    scales=b"".join(scale_all[e*4:(e+1)*4] for e in indices)
    save("selectedScale",scales)
    report["scaleProvenance"]=dict(name=name,shard=index["weight_map"][name],metadata=scale_entry,sha256=hashlib.sha256(scale_raw).hexdigest())
    compare("old-scores",scores,expanded(*ref("router.proj.output")),True)
    maximum=reduce(scores,[rows,width],[-1],op=2)
    compare("old-maximum",maximum,actual("native.softmaxMaximum"),True)
    shifted=binary(scores,[rows,width],maximum,[rows,1],1)
    exps=exponent(shifted,[rows,width])
    compare("old-exp",exps,actual("native.softmaxExp"),True)
    tx=gpu(scores,[rows,width]);torch_prob=raw(torch.softmax(tx,dim=-1))
    compare("torch-softmax",torch_prob,official,True)
    compare("torch-separate-exp",raw(torch.exp(tx-tx.max(-1,keepdim=True).values)),exps,True)
    save("torch-unfused-sum128",raw(gpu(exps,[rows,width]).sum(-1,keepdim=True)))
    for soft in [False,True]:
        den=reduce(exps,[rows,width],[-1],warp=soft)
        compare("warp-denominator" if soft else "old-denominator",den,actual("native.softmaxDenominator"),not soft)
        probabilities=binary(exps,[rows,width],den,[rows,1],3)
        compare("warp-probabilities" if soft else "old-probabilities",probabilities,official if soft else actual("native.probabilities"),True)
        numbers=decode(probabilities)
        assert indices==sum([sorted(range(width),key=lambda e:(-numbers[r*width+e],e))[:top] for r in range(rows)],[])
        selected=gather(probabilities,indices,width)
        compare("warp-selected" if soft else "old-selected",selected,actual("officialProbabilities.selected" if soft else "native.selected"),True)
        for renorm in [False,True]:
            label=f"sum{int(soft)}-renorm{int(renorm)}"
            total=reduce(selected,[rows,top],[-1],warp=renorm)
            norm=binary(selected,[rows,top],total,[rows,1],3)
            weights=binary(norm,[rows,top],scales,[rows,top],2)
            if not renorm:
                branch="officialProbabilities" if soft else "native"
                compare(label+"-total",total,actual(branch+".totalWeight"),True)
                compare(label+"-normalized",norm,actual(branch+".normalizedWeights"),True)
                compare(label+"-weights-control",weights,actual(branch+".weights"),True)
            else:
                compare(label+"-total",total,raw(gpu(selected,[rows,top]).sum(-1,keepdim=True)),True)
                save(label+"-normalized",norm)
            compare(label+"-weights",weights,aligned,soft and renorm)
    # Independent rational witness, without model geometry or expert scales.
    witness_bits=[1040849724,1036246331,1016280329,1016130760,1015983510,1015910743,1015007911,1014115770]
    witness=struct.pack("<8I",*witness_bits)
    expected=pack([355686261/1073741824])
    old=reduce(witness,[1,8],[-1]);new=reduce(witness,[1,8],[-1],warp=True)
    assert old!=expected and new==expected
    report["independentWitness"]=dict(inputBits=witness_bits,exactNumerator=355686261,exactDenominator=1073741824,old=decode(old),candidate=decode(new))
    # Exact dyadic sums cover tails, zero widths, multiple axes and materialized
    # noncontiguous views. Float opmath then final half narrowing is explicit.
    for width in [0,1,2,3,7,8,9,15,16,17,31,32,33,63,64,65,127,128,129,255,256,257,511,512,513,1023,1024,1025,2048,2049,4096,65537]:
        for rows in [1,3]:
            for dtype in [torch.float32,torch.float16,torch.bfloat16]:
                x=((torch.arange(rows*width,dtype=torch.int64)%17)-8).to(dtype).reshape(rows,width)
                data=raw(x.float());new=reduce(data,list(x.shape),[-1],warp=True)
                expected=x.float().sum(-1,keepdim=True)
                assert new==raw(expected),(rows,width,dtype)
                assert raw(torch.frombuffer(bytearray(new),dtype=torch.float32).to(dtype))==raw(expected.to(dtype))
                report["generic"].append(dict(shape=list(x.shape),dtype=str(dtype),dims=[1],exact=True))
    x=((torch.arange(3*7*5,dtype=torch.int64)%17)-8).to(torch.float32).reshape(3,7,5)
    for view in [x,x.permute(2,0,1),x[:1].expand(4,7,5),x[:,::2,:]]:
        for dims in [[0],[1],[2],[0,2],[0,1,2]]:
            new=reduce(raw(view),list(view.shape),dims,warp=True)
            assert new==raw(view.sum(tuple(dims),keepdim=True)),(view.shape,view.stride(),dims)
            report["generic"].append(dict(shape=list(view.shape),strides=list(view.stride()),dims=dims,exact=True))
    for values in [[0.,-0.,-0.],[float('inf'),1.,2.],[-float('inf'),1.,2.],[float('nan'),1.,2.],[1e-40,-1e-40,1e-40]]:
        old=decode(reduce(pack(values),[1,3],[-1]))[0];new=decode(reduce(pack(values),[1,3],[-1],warp=True))[0]
        assert (math.isnan(old) and math.isnan(new)) or struct.pack('<f',old)==struct.pack('<f',new)
    for address,size in guards:
        assert read(address,16)==bytes([165])*16 and read(address+16+size,16)==bytes([165])*16
    report["storageGuardsIntact"]=True
    report["loadedLibraries"]=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if any(n in line for n in ['libnvrtc','libcuda.so','libcublas','libtorch_cuda'])})
    report["status"]="completed"
    print("PASSED",len(report['generic']),"generic cases; independent old-fails/new-passes witness; captured controls and both sums",flush=True)
except BaseException as error:
    report["status"]="failed";report["error"]=repr(error);raise
finally:
    persist()
