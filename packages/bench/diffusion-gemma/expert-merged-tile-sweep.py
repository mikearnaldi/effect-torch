"""Prepare/build/test merged exact-K16 expert tile variants; GPU phases are explicit."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import time

# (tile M,N,K; warp M,N,K; pipeline stages), with no inter-warp K split.
VARIANTS = {
    'baseline-m32n64-s3': (32, 64, 64, 16, 32, 64, 3),
    'm32n64-s2': (32, 64, 64, 16, 32, 64, 2),
    'm16n64-s2': (16, 64, 64, 16, 32, 64, 2),
    'm16n64-s3': (16, 64, 64, 16, 32, 64, 3),
    'm16n64-s4': (16, 64, 64, 16, 32, 64, 4),
    'm16n128-w32-s2': (16, 128, 64, 16, 32, 64, 2),
    'm16n128-w64-s2': (16, 128, 64, 16, 64, 64, 2),
    'm32n128-w64-s2': (32, 128, 64, 16, 64, 64, 2),
}
BOUNDARIES = [0, 1, 2, 15, 16, 17, 30, 31, 32, 33, 42, 43, 57, 58, 64, 65,
              128, 129, 192, 193, 256, 257, 416, 417, 448, 449, 512]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['plan', 'build', 'proof', 'timing'])
    parser.add_argument('output', type=Path)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--variants', default=','.join(VARIANTS))
    parser.add_argument('--vectors', type=Path)
    parser.add_argument('--grids', default='2,3,4')
    parser.add_argument('--patterns', default='0,2,6,9')
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--no-m1', action='store_true', help='zero one-row experts for tensor-only isolation')
    parser.add_argument('--tag', help='fresh evidence filename stem for another pass')
    args = parser.parse_args()
    assert args.tag is None or re.fullmatch(r'[A-Za-z0-9_-]+', args.tag)
    variants = {name: VARIANTS[name] for name in args.variants.split(',')}
    grids = [int(value) for value in args.grids.split(',')]
    assert all(1 <= value <= 8 for value in grids)
    assert args.rounds > 0
    source = args.source.resolve()
    include = os.environ.get('CUTLASS_INCLUDE', '/root/.cache/effect-torch/vllm-0.24.0/lib/python3.12/site-packages/flashinfer/data/cutlass/include')
    args.output.mkdir(parents=True, exist_ok=True)
    specs = {}
    for name, (m, n, k, wm, wn, wk, stages) in variants.items():
        assert k == wk == 64 and m % wm == n % wn == 0
        specs[name] = {'tile': [m,n,k], 'warp': [wm,wn,wk], 'stages': stages,
                       'threads': (m//wm)*(n//wn)*32,
                       'command': [os.environ.get('NVCC', 'nvcc'), '-O3', '-std=c++17', '-arch=sm_120',
                                   '--expt-relaxed-constexpr', '-Xptxas=-v', '-I', include,
                                   f'-DET_CUTLASS_M={m}', f'-DET_CUTLASS_N={n}', f'-DET_CUTLASS_K={k}',
                                   f'-DET_CUTLASS_WARP_M={wm}', f'-DET_CUTLASS_WARP_N={wn}',
                                   f'-DET_CUTLASS_STAGES={stages}', str(source/'expert-cutlass-device.cu'),
                                   '-lcublas', '-lcuda', '-o', str(args.output/name)]}
    if args.mode == 'plan':
        hashes = {name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in [
            'expert-cutlass-device.cu', 'expert-cutlass-device.cuh', 'expert-cutlass-grouped.cuh', 'expert-gemm-batching.cu']}
        with (args.output/'plan.json').open('x') as output:
            json.dump({'variants': specs, 'sourceSha256': hashes, 'gridFactors': grids,
                       'exactness': 'K16 MMA sequence, identical split boundaries and BF16 partial rounding; GPU bitwise proof required',
                       'limitations': ['component experiment only', 'aggregate histogram vectors, not exact layer routes',
                                       'hybrid M1 host routing omits production first-projection count fence']}, output, indent=2)
        print(json.dumps({'variants': len(specs), 'gpuExecuted': False}))
        return
    if args.mode == 'build':
        failed = False
        with (args.output/'build.jsonl').open('x') as evidence:
            for name, spec in specs.items():
                started = time.monotonic()
                with (args.output/f'{name}-build.log').open('x') as log:
                    result = subprocess.run(spec['command'], stdout=log, stderr=subprocess.STDOUT)
                failed = failed or result.returncode != 0
                record = {'variant': name, 'exitCode': result.returncode, 'seconds': time.monotonic()-started}
                evidence.write(json.dumps(record)+'\n'); evidence.flush()
                print(json.dumps(record), flush=True)
        if failed:
            raise SystemExit(1)
        return
    if args.mode == 'proof':
        cases = [('boundaries', BOUNDARIES, projection, pattern, 73, 2, 0)
                 for projection in range(2) for pattern in map(int, args.patterns.split(','))]
    else:
        assert args.vectors is not None, '--vectors required for timing'
        vectors = json.loads(args.vectors.read_text())['vectors']
        cases = [(vector['label'], vector['rows'], projection, 0, 71, grid, repeat)
                 for repeat in range(args.rounds) for vector in vectors
                 for projection in range(2) for grid in grids]
    with (args.output/f'{args.tag or args.mode}.jsonl').open('x') as evidence:
        for label, rows, projection, pattern, seed, grid, repeat in cases:
            if args.no_m1:
                rows = [0 if value == 1 else value for value in rows]
                label += '-no-m1'
            names = list(specs)
            # Rotate order across repeats instead of always running baseline cold.
            names = names[repeat % len(names):] + names[:repeat % len(names)]
            for name in names:
                binary = args.output/name
                if not binary.exists():
                    raise FileNotFoundError(f'missing built variant {binary}; inspect compile evidence')
                env = dict(os.environ, ET_DEVICE_MERGE_SLICES='1', ET_DEVICE_CUBLAS_M1='1',
                           ET_DEVICE_COMPACT='1', ET_DEVICE_BLOCKS_PER_SM=str(grid),
                           ET_CACHED_COUNTS='1' if projection else '0')
                result = subprocess.run([str(binary), str(projection), str(seed), str(pattern), ','.join(map(str, rows))],
                                        env=env, text=True, capture_output=True)
                records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
                record = {'variant': name, 'label': label, 'rows': rows, 'projection': projection,
                          'pattern': pattern, 'blocksPerSM': grid, 'repeat': repeat,
                          'exitCode': result.returncode, 'records': records, 'stderr': result.stderr}
                evidence.write(json.dumps(record)+'\n'); evidence.flush()
                if result.returncode or len(records) != 2 or any(
                        value['exact'] != value['elements'] or value['graphExact'] != value['elements'] for value in records):
                    raise RuntimeError(f'bitwise proof failed: {name}, {label}, projection {projection}, pattern {pattern}')
                candidate = next(value for value in records if value['mode'] == 'gpu_metadata')
                print(json.dumps({'variant': name, 'label': label, 'projection': projection,
                                  'blocksPerSM': grid, 'graphMilliseconds': candidate['graphMedianMs']}), flush=True)


if __name__ == '__main__':
    main()
