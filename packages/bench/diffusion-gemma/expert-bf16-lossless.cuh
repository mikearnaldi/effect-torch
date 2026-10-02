// Standalone lossless storage experiment. Semantic operands remain exact BF16.
// Current CUTLASS B accesses contain eight aligned words. Both expert inner
// sizes and every exact split boundary are multiples of eight, so an access
// cannot cross a 64-word block. Each lane owns its 16-byte shared destination.
#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>
#ifndef ET_LOSSLESS_FIXED
#define ET_LOSSLESS_FIXED 0
#endif

struct EtLosslessContext {
  const uint32_t* headers;
  const uint8_t* payload;
  const void* dense;
  const uint8_t* tails;
};
__device__ EtLosslessContext* et_lossless_contexts;
__device__ unsigned long long et_lossless_dense_base;
__device__ unsigned long long et_lossless_expert_bytes;
inline size_t et_lossless_host_dense_bytes=0,et_lossless_host_packed_bytes=0,et_lossless_host_raw_blocks=0;

__device__ __forceinline__ void et_lossless_expand(void* destination,
    uint32_t sm0,uint32_t sm1,uint32_t exponents,uint32_t base){
  uint32_t result[4];
  #pragma unroll
  for(int pair=0;pair<4;++pair){
    uint32_t signs_mantissas=(pair<2?sm0:sm1)>>((pair%2)*16);
    uint32_t deltas=exponents>>(pair*8);
    uint32_t a=(signs_mantissas&127)|((signs_mantissas&128)<<8)|((base+(deltas&15))<<7);
    uint32_t b=((signs_mantissas>>8)&127)|((signs_mantissas&32768))|((base+((deltas>>4)&15))<<7);
    result[pair]=a|(b<<16);
  }
  *reinterpret_cast<uint4*>(destination)=make_uint4(result[0],result[1],result[2],result[3]);
}

template<int Bytes, bool ZeroFill>
__device__ __forceinline__ void et_lossless_copy(void* destination,const void* source,
    bool valid,EtLosslessContext context) {
  static_assert(Bytes==16,"Eight exact BF16 words per CUTLASS global access");
  if(!valid){if(ZeroFill)*reinterpret_cast<uint4*>(destination)=make_uint4(0,0,0,0);return;}
  auto index=(reinterpret_cast<uintptr_t>(source)-reinterpret_cast<uintptr_t>(context.dense))/2;
#if ET_LOSSLESS_FIXED
  auto* packed=reinterpret_cast<const uint32_t*>(context.payload+(index/8)*12);
  uint32_t a=packed[0],b=packed[1],c=packed[2],header=context.headers[index/64];
  if(header&256){
    uint32_t d=*reinterpret_cast<const uint32_t*>(context.tails+size_t(header>>9)*4+(index%64)/8*4);
    *reinterpret_cast<uint4*>(destination)=make_uint4(a,b,c,d);
  }else et_lossless_expand(destination,a,b,c,header&255);
#else
  uint32_t header=context.headers[index/64];
  auto* block=context.payload+size_t(header>>9)*4;
  if(header&256){
    *reinterpret_cast<uint4*>(destination)=*reinterpret_cast<const uint4*>(block+(index%64)*2);
    return;
  }
  auto* packed=reinterpret_cast<const uint32_t*>(block+(index%64)/8*12);
  et_lossless_expand(destination,packed[0],packed[1],packed[2],header&255);
#endif
}

template<int Bytes, bool ZeroFill>
__device__ __forceinline__ unsigned et_lossless_copy_async(void* destination,const void* source,
    bool valid,EtLosslessContext context){
  static_assert(Bytes==16,"Eight exact BF16 words per CUTLASS global access");
  if(!valid){
    if(ZeroFill)cutlass::arch::cp_async_zfill<16,cutlass::arch::CacheOperation::Always>(destination,source,false);
    return 256;
  }
  auto index=(reinterpret_cast<uintptr_t>(source)-reinterpret_cast<uintptr_t>(context.dense))/2;
#if ET_LOSSLESS_FIXED
  auto* packed=context.payload+(index/8)*12;
  cutlass::arch::cp_async<4,cutlass::arch::CacheOperation::Always>(destination,packed,true);
  cutlass::arch::cp_async<4,cutlass::arch::CacheOperation::Always>(static_cast<uint8_t*>(destination)+4,packed+4,true);
  cutlass::arch::cp_async<4,cutlass::arch::CacheOperation::Always>(static_cast<uint8_t*>(destination)+8,packed+8,true);
  uint32_t header=context.headers[index/64];
  if(header&256)cutlass::arch::cp_async<4,cutlass::arch::CacheOperation::Always>(static_cast<uint8_t*>(destination)+12,
    context.tails+size_t(header>>9)*4+(index%64)/8*4,true);
#else
  uint32_t header=context.headers[index/64];
  auto* block=context.payload+size_t(header>>9)*4;
  if(header&256){
    cutlass::arch::cp_async<16,cutlass::arch::CacheOperation::Always>(destination,block+(index%64)*2,true);
  }else{
    auto* packed=block+(index%64)/8*12;
    // Twelve-byte groups are four-byte aligned, including odd groups.
    cutlass::arch::cp_async<4,cutlass::arch::CacheOperation::Always>(destination,packed,true);
    cutlass::arch::cp_async<4,cutlass::arch::CacheOperation::Always>(static_cast<uint8_t*>(destination)+4,packed+4,true);
    cutlass::arch::cp_async<4,cutlass::arch::CacheOperation::Always>(static_cast<uint8_t*>(destination)+8,packed+8,true);
  }
#endif
  return header&511;
}

__device__ __forceinline__ void et_lossless_decode_shared(void* destination,unsigned metadata){
  if(metadata&256)return;
  auto* packed=static_cast<uint32_t*>(destination);
  et_lossless_expand(destination,packed[0],packed[1],packed[2],metadata&255);
}

__device__ __forceinline__ EtLosslessContext et_lossless_context(const void* expert) {
  auto index=(reinterpret_cast<unsigned long long>(expert)-et_lossless_dense_base)/et_lossless_expert_bytes;
  return et_lossless_contexts[index];
}

// Generated from the installed CUTLASS header by the explicit build helper.
// Only operand-B global copies change; exact MMA order and stages are preserved.
#include "expert-bf16-lossless-mma.generated.cuh"

template<class T> struct EtLosslessMma;
template<class Shape,class IA,class SA,cutlass::arch::CacheOperation::Kind CA,
    class IB,class SB,cutlass::arch::CacheOperation::Kind CB,class EC,class LC,
    class Policy,int Stages,cutlass::gemm::SharedMemoryClearOption Clear,class Enable>
struct EtLosslessMma<cutlass::gemm::threadblock::MmaMultistage<Shape,IA,SA,CA,IB,SB,CB,EC,LC,Policy,Stages,Clear,Enable>> {
  using Type=cutlass::gemm::threadblock::EtLosslessMmaMultistage<Shape,IA,SA,CA,IB,SB,CB,EC,LC,Policy,Stages,Clear,Enable>;
};

inline EtLosslessContext et_lossless_upload(const std::vector<unsigned short>& words,const void* dense) {
  if(words.size()%64)std::abort();
  std::vector<uint32_t> headers;
  std::vector<uint8_t> payload;
  std::vector<uint8_t> tails;
  for(size_t offset=0;offset<words.size();offset+=64){
    unsigned low=255,high=0;
    for(size_t i=0;i<64;++i){unsigned e=(words[offset+i]>>7)&255;low=std::min(low,e);high=std::max(high,e);}
    bool raw=high-low>15;
    et_lossless_host_raw_blocks+=raw;
    if(payload.size()%16||payload.size()/4>0x7fffff)std::abort();
#if ET_LOSSLESS_FIXED
    headers.push_back(uint32_t(tails.size()/4)<<9|unsigned(raw)<<8|low);
    if(raw){
      for(size_t group=0;group<64;group+=8){
        auto* bytes=reinterpret_cast<const uint8_t*>(words.data()+offset+group);
        payload.insert(payload.end(),bytes,bytes+12);
        tails.insert(tails.end(),bytes+12,bytes+16);
      }
    }else{
#else
    headers.push_back(uint32_t(payload.size()/4)<<9|unsigned(raw)<<8|low);
    if(raw){
      auto* bytes=reinterpret_cast<const uint8_t*>(words.data()+offset);
      payload.insert(payload.end(),bytes,bytes+128);
    }else{
#endif
      for(size_t group=0;group<64;group+=8){
        uint8_t encoded[12]={};
        for(size_t i=0;i<8;++i){
          unsigned word=words[offset+group+i];
          encoded[i]=(word&127)|((word>>8)&128);
          encoded[8+i/2]|=(((word>>7)&255)-low)<<((i%2)*4);
        }
        payload.insert(payload.end(),encoded,encoded+12);
      }
    }
  }
  uint32_t* device_headers=nullptr;uint8_t* device_payload=nullptr;
  et_lossless_host_dense_bytes+=words.size()*2;
  et_lossless_host_packed_bytes+=headers.size()*4+payload.size()+tails.size();
  et_cutlass_cuda(cudaMalloc(&device_headers,headers.size()*4));
  et_cutlass_cuda(cudaMalloc(&device_payload,payload.size()));
  et_cutlass_cuda(cudaMemcpy(device_headers,headers.data(),headers.size()*4,cudaMemcpyHostToDevice));
  et_cutlass_cuda(cudaMemcpy(device_payload,payload.data(),payload.size(),cudaMemcpyHostToDevice));
  uint8_t* device_tails=nullptr;
  if(!tails.empty()){
    et_cutlass_cuda(cudaMalloc(&device_tails,tails.size()));
    et_cutlass_cuda(cudaMemcpy(device_tails,tails.data(),tails.size(),cudaMemcpyHostToDevice));
  }
  return {device_headers,device_payload,dense,device_tails};
}

inline void et_lossless_bind(const std::vector<EtLosslessContext>& contexts,const void* dense,size_t expert_bytes){
  std::fprintf(stderr,"losslessBF16 blockWords=64 exponentBits=4 experts=%zu denseBytes=%zu packedBytes=%zu rawBlocks=%zu\n",
    contexts.size(),et_lossless_host_dense_bytes,et_lossless_host_packed_bytes,et_lossless_host_raw_blocks);
  EtLosslessContext* device_contexts=nullptr;
  et_cutlass_cuda(cudaMalloc(&device_contexts,contexts.size()*sizeof(EtLosslessContext)));
  et_cutlass_cuda(cudaMemcpy(device_contexts,contexts.data(),contexts.size()*sizeof(EtLosslessContext),cudaMemcpyHostToDevice));
  auto address=reinterpret_cast<unsigned long long>(dense);
  auto bytes=static_cast<unsigned long long>(expert_bytes);
  et_cutlass_cuda(cudaMemcpyToSymbol(et_lossless_contexts,&device_contexts,sizeof(device_contexts)));
  et_cutlass_cuda(cudaMemcpyToSymbol(et_lossless_dense_base,&address,sizeof(address)));
  et_cutlass_cuda(cudaMemcpyToSymbol(et_lossless_expert_bytes,&bytes,sizeof(bytes)));
}
