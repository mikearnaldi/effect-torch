"""Production grouped subset versus direct-output CUTLASS, same fallback work."""
import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("vectors")
parser.add_argument("output")
parser.add_argument("--binary-prefix", default="/root/expert-cutlass-subset-")
parser.add_argument("--tiles", default="16x64,16x128,32x64,32x128")
parser.add_argument("--stress", action="store_true")
parser.add_argument("--projections", default="0,1")
parser.add_argument("--gemv-stress", action="store_true",
                    help="Device-metadata harness only: cancellation, subnormal and midpoint patterns")
parser.add_argument("--special-stress", action="store_true")
parser.add_argument("--split-stress", action="store_true",
                    help="All split geometry boundaries with broad numerical patterns")
args = parser.parse_args()
vectors = json.loads(Path(args.vectors).read_text())["vectors"]
cases = [(v["label"], v["rows"], 71, 0) for v in vectors]
if args.stress:
    # All direct-output eligibility boundaries, plus mixed fallback rows.
    cases = [("eligibility-boundaries", list(range(1, 33)) + [63, 64, 127, 128, 129, 256, 257, 416, 417, 512], seed, pattern)
             for seed in (17, 313) for pattern in range(3)]
    cases += [("below-group-threshold", [1, 2, 3, 4, 129, 512], 71, 0),
              ("all-grouped", [2, 3, 7, 16], 71, 0)]
if args.gemv_stress:
    cases = [(label, rows, seed, pattern)
             for label, rows in (("all-m1", [1] * 33),
                                 ("mixed-m1-zero", [0, 1, 1, 0, 2, 17, 33, 128, 257, 512]))
             for seed in (17, 313) for pattern in range(8)]
if args.split_stress:
    cases = [("split-boundaries", [0, 1, 2, 16, 17, 30, 31, 32, 33, 42, 43, 57, 58,
                                   64, 65, 128, 129, 192, 193, 256, 257, 416, 417,
                                   448, 449, 512], seed, pattern)
             for seed in (17, 313) for pattern in range(8)]
if args.special_stress:
    cases = [("special-values", [0, 1, 2, 16, 17, 31, 33, 43, 58, 65, 129, 193, 257, 417, 449], 17, pattern)
             for pattern in range(8, 12)]
with Path(args.output).open("x") as output:
    for tile in args.tiles.split(","):
        for label, rows, seed, pattern in cases:
            for projection in map(int, args.projections.split(",")):
                result = subprocess.run([args.binary_prefix + tile, str(projection), str(seed), str(pattern),
                    ",".join(map(str, rows))], text=True, capture_output=True)
                records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
                evidence = dict(tile=tile, histogram=label, projection=projection, rows=rows,
                    seed=seed, pattern=pattern, exitCode=result.returncode, records=records, stderr=result.stderr)
                output.write(json.dumps(evidence) + "\n")
                output.flush()
                if result.returncode or len(records) != 2 or any(
                    record["exact"] != record["elements"] or record["graphExact"] != record["elements"]
                    for record in records):
                    raise SystemExit(f"Mismatch/failure: {evidence}")
                print(tile, label, projection, [(r["mode"], r["gpuMedianMs"], r["graphMedianMs"])
                                              for r in records], flush=True)
