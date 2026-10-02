// Store-only argument copy: integer[12]=T, [13]=H, [14]=D, input[6]=raw token-major V.
extern "C" __global__ void et_vnorm_store56(CudaKernelArgs a) {
    et_u64 tokens=a.integers[12],heads=a.integers[13],dim=a.integers[14];
    et_u64 lane=blockIdx.x/tokens,token=blockIdx.x%tokens;
    if(lane>=a.integers[5]*a.integers[2])return;
    unsigned int count=((const unsigned int*)a.scratch[0])[lane];
    if(count>tokens){et_error(a,1);return;}
    if(token>=count)return;
    const unsigned int*cursors=(const unsigned int*)a.inputs[7];
    et_u64 sequence=lane/a.integers[2],position=cursors[lane]+token;
    const et_u64*table=(const et_u64*)a.inputs[3],*header=table+sequence*4;
    if(position<header[1]||position>=header[2]){et_error(a,1);return;}
    const et_u64*pointers=table+header[3]+(position-header[0])*4;
    for(et_u64 hd=threadIdx.x;hd<heads*dim;hd+=blockDim.x){
        et_u64 head=hd/dim,d=hd%dim,source=((lane*heads+head)*tokens+token)*dim+d;
        et_store(pointers[0],3,hd,et_load<float>(a.inputs[1],1,source));
    }
    unsigned int wl=threadIdx.x&31U;
    for(et_u64 head=threadIdx.x/32;head<heads;head+=blockDim.x/32){
        et_u64 source=((lane*tokens+token)*heads+head)*dim;
        float partial[4]={0,0,0,0};
        for(et_u64 k=wl*4;k<dim;k+=128){
            #pragma unroll
            for(int j=0;j<4;++j)if(k+j<dim){float v=et_load<float>(a.inputs[6],3,source+k+j);partial[j]+=v*v;}
        }
        float sum=((partial[0]+partial[1])+partial[2])+partial[3];
        for(unsigned int offset=16;offset;offset>>=1)sum+=__shfl_down_sync(0xffffffffU,sum,offset);
        sum=__shfl_sync(0xffffffffU,sum,0);
        float mean=sum*(1.0f/(float)dim),inverse=rsqrtf(mean+(float)a.scalars[1]);
        for(et_u64 d=wl;d<dim;d+=32){
            float value=et_load<float>(a.inputs[6],3,source+d)*inverse;
            // Preserve norm BF16 storage, conversion's F32 widening, then cache BF16 storage.
            unsigned short normalized=et_to16(value,true);
            float converted=et_bfloat_float(normalized);
            et_store(pointers[1],3,head*dim+d,converted);
        }
    }
}
