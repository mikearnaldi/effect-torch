"""Prepare a narrowly adapted CUTLASS mainloop and optionally build an exact prototype."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--include', type=Path, default=Path('/root/.cache/effect-torch/vllm-0.24.0/lib/python3.12/site-packages/flashinfer/data/cutlass/include'))
    parser.add_argument('--header', type=Path)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--decode', choices=['register','async'], default='register')
    parser.add_argument('--fixed', action='store_true', help='fixed12-byte slots with sparse raw tails')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.header or args.include/'cutlass/gemm/threadblock/mma_multistage.h'
    original = path.read_text()
    modified = original.replace('MmaMultistage', 'EtLosslessMmaMultistage')
    replacements = {
        'SmemIteratorB smem_iterator_B_;': 'SmemIteratorB smem_iterator_B_;\n  ::EtLosslessContext lossless_;',
        'int lane_idx\n    ):': 'int lane_idx,\n      ::EtLosslessContext lossless\n    ):',
        'smem_write_stage_idx_(0),': 'lossless_(lossless),\n      smem_write_stage_idx_(0),',
    }
    for old, new in replacements.items():
        assert modified.count(old) == 1, f'CUTLASS structure changed: {old}'
        modified = modified.replace(old,new)
    pattern = r'cutlass::arch::cp_async(_zfill)?<kSrcBytes, kCacheOpB>\(\s*dst_ptr \+ v, (gmem_ptr|iterator_B.get\(\)), iterator_B.valid\(\)\);'
    def replace_copy(match):
        zero = 'true' if match[1] else 'false'
        if args.decode == 'async':
            index = '(group_start_B+j)' if match[2] == 'gmem_ptr' else 'j'
            return (f'lossless_set_header({index}*IteratorB::kAccessesPerVector+v, '
                    f'::et_lossless_copy_async<kSrcBytes, {zero}>(dst_ptr + v, {match[2]}, iterator_B.valid(), lossless_));')
        return f'::et_lossless_copy<kSrcBytes, {zero}>(dst_ptr + v, {match[2]}, iterator_B.valid(), lossless_);'
    modified, count = re.subn(pattern,replace_copy,modified)
    assert count == 3, f'expected exactly three B-copy sites, got{count}'
    if args.decode == 'async':
        modified=modified.replace('::EtLosslessContext lossless_;',
            '::EtLosslessContext lossless_;\n'
            '''  unsigned lossless_headers_[Stages][Detail::AsyncCopyIterationsPerStageB*IteratorB::kAccessesPerVector]{};
  int lossless_decode_stage_=0;
  CUTLASS_DEVICE void lossless_set_header(int index,unsigned value){
    CUTLASS_PRAGMA_UNROLL
    for(int stage=0;stage<Stages;++stage)if(stage==smem_write_stage_idx_){
      CUTLASS_PRAGMA_UNROLL
      for(int access=0;access<Detail::AsyncCopyIterationsPerStageB*IteratorB::kAccessesPerVector;++access)
        if(access==index)lossless_headers_[stage][access]=value;
    }
  }
  CUTLASS_DEVICE unsigned lossless_get_header(int index){
    unsigned value=256;
    CUTLASS_PRAGMA_UNROLL
    for(int stage=0;stage<Stages;++stage)if(stage==lossless_decode_stage_){
      CUTLASS_PRAGMA_UNROLL
      for(int access=0;access<Detail::AsyncCopyIterationsPerStageB*IteratorB::kAccessesPerVector;++access)
        if(access==index)value=lossless_headers_[stage][access];
    }
    return value;
  }''')
        old='cutlass::arch::cp_async_wait<Base::kStages - 2>();\n    __syncthreads();'
        assert modified.count(old)==1
        new='''cutlass::arch::cp_async_wait<Base::kStages - 2>();
    // Each lane expands its own completed asynchronous copies before the
    // existing CTA barrier publishes the decoded BF16 stage to MMA readers.
    SmemIteratorB decoder(this->smem_iterator_B_);
    decoder.add_tile_offset({lossless_decode_stage_-smem_write_stage_idx_,0});
    decoder.set_iteration_index(0);
    CUTLASS_PRAGMA_UNROLL
    for(int j=0;j<Detail::AsyncCopyIterationsPerStageB;++j){
      auto* destination=reinterpret_cast<typename IteratorB::AccessType*>(decoder.get());
      CUTLASS_PRAGMA_UNROLL
      for(int v=0;v<IteratorB::kAccessesPerVector;++v)
        ::et_lossless_decode_shared(destination+v,lossless_get_header(j*IteratorB::kAccessesPerVector+v));
      ++decoder;
    }
    lossless_decode_stage_=(lossless_decode_stage_+1)%Stages;
    __syncthreads();'''
        modified=modified.replace(old,new)
    generated = args.output/'expert-bf16-lossless-mma.generated.cuh'
    generated.write_text(modified)
    source = Path(__file__).resolve().parent
    command = [os.environ.get('NVCC','nvcc'),'-O3','-std=c++17','-arch=sm_120',
               '--expt-relaxed-constexpr','-Xptxas=-v','-I',str(args.include),'-I',str(args.output),
               '-DET_CUTLASS_M=32','-DET_CUTLASS_N=64','-DET_CUTLASS_K=64',
               '-DET_CUTLASS_WARP_M=16','-DET_CUTLASS_WARP_N=32','-DET_CUTLASS_STAGES=3',
               '-DET_DEVICE_LOSSLESS=1',str(source/'expert-cutlass-device.cu'),'-lcublas','-lcuda',
               '-o',str(args.output/f'lossless-{args.decode}-m32n64-s3')]
    if args.fixed:
        command.insert(1,'-DET_LOSSLESS_FIXED=1')
    record = dict(originalHeader=str(path), originalHeaderSha256=hashlib.sha256(original.encode()).hexdigest(),
                  generatedHeaderSha256=hashlib.sha256(modified.encode()).hexdigest(), replacementCopySites=count,
                  command=command, decode=args.decode, fixedSlots=args.fixed, built=False)
    if args.build:
        with (args.output/'build.log').open('x') as log:
            result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        record.update(built=result.returncode==0,exitCode=result.returncode)
    (args.output/'build-plan.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))
    if args.build and result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
