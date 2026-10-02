#include <cassert>
#include <cstdint>
#include <vector>
#define __global__
struct Index { unsigned int x; } blockIdx, blockDim, threadIdx;
static unsigned long long atomicCAS(unsigned long long* p, unsigned long long expected, unsigned long long desired) {
    auto old=*p; if(old==expected)*p=desired; return old;
}
#include "fused_moe75_guard.cu"
int main() {
    for(unsigned int tokens: {64U,256U}) {
        std::vector<unsigned int> original(tokens*8), weights(tokens*8), result_weights(tokens*8+2,0xa5a5a5a5U);
        std::vector<int> result(tokens*8+2,-123);
        for(unsigned int i=0;i<tokens*8;++i){original[i]=(i%8)*13;weights[i]=0x7fc00000U+i;}
        auto run=[&](unsigned long long& status, bool null_inputs) {
            blockDim.x=128;
            for(unsigned int i=0;i<((tokens+127)/128)*128;++i) {
                blockIdx.x=i/128;threadIdx.x=i%128;
                et_fused_moe75_guard(null_inputs?nullptr:original.data(), null_inputs?nullptr:weights.data(),
                    result.data()+1,result_weights.data()+1,&status,17,tokens);
            }
            assert(result.front()==-123 && result.back()==-123);
            assert(result_weights.front()==0xa5a5a5a5U && result_weights.back()==0xa5a5a5a5U);
        };
        unsigned long long status=0;run(status,false);assert(status==0);
        for(unsigned int i=0;i<tokens*8;++i){assert(result[i+1]==int(original[i]));assert(result_weights[i+1]==weights[i]);}
        for(unsigned int bad: {128U,0xffffffffU,0U}) {
            original[7]=bad;status=0;run(status,false);assert(status==((17ULL<<32)|1));
            for(unsigned int i=0;i<8;++i){assert(result[i+1]==int(i));assert(result_weights[i+1]==0);}
        }
        status=(3ULL<<32)|5;run(status,true);assert(status==((3ULL<<32)|5));
        for(unsigned int i=0;i<tokens*8;++i){assert(result[i+1]==int(i%8));assert(result_weights[i+1]==0);}
        original[7]=91;status=0;run(status,false);assert(status==0);
        for(unsigned int i=0;i<tokens*8;++i){assert(result[i+1]==int(original[i]));assert(result_weights[i+1]==weights[i]);}
    }
}
