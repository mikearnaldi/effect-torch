"""Compare complete exact expert pipelines with raw and lossless weight loads."""
import argparse
import json
import os
from pathlib import Path
import subprocess

BOUNDARIES=[0,1,2,15,16,17,30,31,32,33,42,43,57,58,64,65,128,129,192,193,256,257,416,417,448,449,512]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['proof','timing'])
    parser.add_argument('output',type=Path)
    parser.add_argument('--binary',action='append',required=True,help='label=/absolute/binary')
    parser.add_argument('--vectors',type=Path)
    parser.add_argument('--patterns',default='0,2,6,9')
    parser.add_argument('--rounds',type=int,default=2)
    parser.add_argument('--grids',default='2')
    args=parser.parse_args()
    binaries=dict(value.split('=',1) for value in args.binary)
    assert len(binaries)==len(args.binary)
    for binary in binaries.values():assert Path(binary).is_file(),binary
    if args.mode=='proof':
        cases=[('boundaries',BOUNDARIES,projection,pattern,73,2,0) for projection in range(2) for pattern in map(int,args.patterns.split(','))]
    else:
        vectors=json.loads(args.vectors.read_text())['vectors']
        cases=[(v['label'],v['rows'],projection,0,71,grid,repeat) for repeat in range(args.rounds)
               for v in vectors for projection in range(2) for grid in map(int,args.grids.split(','))]
    with args.output.open('x') as output:
        for label,rows,projection,pattern,seed,grid,repeat in cases:
            names=list(binaries);names=names[repeat%len(names):]+names[:repeat%len(names)]
            for name in names:
                env=dict(os.environ,ET_DEVICE_MERGE_SLICES='1',ET_DEVICE_CUBLAS_M1='1',ET_DEVICE_COMPACT='1',
                         ET_DEVICE_BLOCKS_PER_SM=str(grid),ET_CACHED_COUNTS='1' if projection else '0')
                result=subprocess.run([binaries[name],str(projection),str(seed),str(pattern),','.join(map(str,rows))],env=env,capture_output=True,text=True)
                records=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
                record=dict(variant=name,binary=binaries[name],label=label,rows=rows,projection=projection,
                            pattern=pattern,seed=seed,blocksPerSM=grid,repeat=repeat,exitCode=result.returncode,
                            records=records,stderr=result.stderr)
                output.write(json.dumps(record)+'\n');output.flush()
                if result.returncode or len(records)!=2 or any(v['exact']!=v['elements'] or v['graphExact']!=v['elements'] for v in records):
                    raise RuntimeError(f'bitwise GPU proof failed:{name},{label},projection{projection},pattern{pattern}')
                candidate=next(v for v in records if v['mode']=='gpu_metadata')
                print(json.dumps(dict(variant=name,label=label,projection=projection,pattern=pattern,blocksPerSM=grid,
                                      graphMilliseconds=candidate['graphMedianMs'])),flush=True)


if __name__=='__main__':
    main()
