"""CPU cross-language contracts; optional full-tensor CUDA evidence comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import unittest

from shared_rng import CanvasStream, uniform_f32


class CpuContracts(unittest.TestCase):
    def test_canvas_matches_javascript_imul_and_rejection(self):
        source = r"""
const out=[];
for (const seed of [0,12345,0xffffffff]) for(const vocab of [262144,2147483649]) {
 let state=seed;
 const next=()=>{state=(state+0x6d2b79f5)>>>0;let t=Math.imul(state^(state>>>15),state|1);t^=t+Math.imul(t^(t>>>7),t|61);return(t^(t>>>14))>>>0;};
 const limit=0x100000000-0x100000000%vocab;
 for(let draw=0;draw<5;draw++){let values=[];for(let i=0;i<256;i++){let t=next();while(t>=limit)t=next();values.push(t%vocab);}out.push({seed,vocab,draw,values});}
}
process.stdout.write(JSON.stringify(out));
"""
        records = json.loads(subprocess.check_output(["node", "-e", source], text=True))
        streams = {}
        for record in records:
            key = record["seed"], record["vocab"]
            stream = streams.setdefault(key, CanvasStream(record["seed"]))
            self.assertEqual(stream.canvas(256, record["vocab"]), record["values"])
        self.assertGreater(streams[(0, 2147483649)].words, 5 * 256)

    def test_uniform_matches_javascript_bigint_and_float32(self):
        source = r"""
const mask=(1n<<64n)-1n,g=0x9e3779b97f4a7c15n,out=[];
for(const seed of [0,12345,0xffffffff])for(const draw of [0,1,19,0xffffffff])for(const i of [0,1,31,4096,67108863]){
 let x=(BigInt(seed)+BigInt(draw)*g+2n*BigInt(i)+g)&mask;
 x=((x^(x>>30n))*0xbf58476d1ce4e5b9n)&mask;x=((x^(x>>27n))*0x94d049bb133111ebn)&mask;x^=x>>31n;
 const value=Math.fround((Number(x>>11n)+0.5)/9007199254740992);out.push({seed,draw,i,value});
}process.stdout.write(JSON.stringify(out));
"""
        records = json.loads(subprocess.check_output(["node", "-e", source], text=True))
        for record in records:
            self.assertEqual(uniform_f32(record["seed"], record["draw"], record["i"]), record["value"])


def verify_gpu(directory: Path) -> None:
    import numpy as np
    import torch
    from shared_rng import install_gpu, uniform_like

    install_gpu()
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["freshCompileResetsStream"]
    streams = {}
    for record in manifest["canvases"]:
        key = record["seed"], record["vocabSize"]
        stream = streams.setdefault(key, CanvasStream(record["seed"]))
        assert stream.canvas(len(record["values"]), record["vocabSize"]) == record["values"]
    for record in manifest["records"]:
        expected = (directory / record["file"]).read_bytes()
        assert hashlib.sha256(expected).hexdigest() == record["sha256"]
        template = torch.empty(record["shape"], dtype=torch.float32, device="cuda")
        seeds = torch.tensor([record["seed"]], dtype=torch.int64, device="cuda")
        draws = torch.tensor([record["draw"]], dtype=torch.int64, device="cuda")
        actual = uniform_like(template, seeds, draws).cpu().numpy().tobytes()
        if actual != expected:
            a = np.frombuffer(actual, dtype=np.uint32)
            b = np.frombuffer(expected, dtype=np.uint32)
            bad = np.flatnonzero(a != b)
            raise AssertionError(f"uniform mismatch {record}: count={len(bad)}, first={bad[:8].tolist()}")
        print(json.dumps({"uniformExact": True, **record}), flush=True)
    # Request ordering and captured graph repetition cannot implicitly advance
    # the stateless helper's stream; all advancement belongs to request state.
    template = torch.empty((2, 4099), dtype=torch.float32, device="cuda")
    seeds = torch.tensor([12345, 0], dtype=torch.int64, device="cuda")
    draws = torch.tensor([2, 0], dtype=torch.int64, device="cuda")
    compiled = torch.compile(uniform_like, fullgraph=True)
    observed = compiled(template, seeds, draws).cpu().numpy()
    for row, (seed, draw) in enumerate([(12345, 2), (0, 0)]):
        expected = np.array([uniform_f32(seed, draw, i) for i in range(4099)], dtype=np.float32)
        assert observed[row].tobytes() == expected.tobytes()
    assert compiled(template, seeds, draws).cpu().numpy().tobytes() == observed.tobytes()
    print(json.dumps({"canvasExact": True, "requestOrderExact": True, "torchCompileExact": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(CpuContracts))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.evidence:
        verify_gpu(args.evidence)
