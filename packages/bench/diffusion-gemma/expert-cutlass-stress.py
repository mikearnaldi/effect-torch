"""Run an already-built CUTLASS prototype against cuBLAS and eager fixtures.

Timing is disabled. Each case fails closed on BF16 bit mismatches. Use a separate
GPU slot from full-model gates and timing runs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("binary")
parser.add_argument("output")
parser.add_argument("--suite", choices=["smoke", "boundaries", "all"], default="boundaries")
parser.add_argument("--fixtures", default="/root/components-eager-20260930/expert-components")
args = parser.parse_args()
fixture_root = Path(args.fixtures)
if args.suite == "smoke":
    rows = [2, 17, 149, 262]
elif args.suite == "all":
    rows = list(range(2, 513))
else:
    rows = [2, 3, 15, 16, 17, 30, 31, 32, 33, 42, 43, 57, 58, 64, 65,
            127, 128, 129, 149, 192, 193, 256, 257, 262, 416, 417, 448, 449, 511, 512]
cases = []
for columns, inner in [(1408, 2816), (2816, 704)]:
    for row in rows:
        for pattern in range(3):
            cases.append((row, columns, inner, row + pattern * 71, "-", pattern))
for fixture in json.loads((fixture_root / "report.json").read_text())["cases"]:
    row, inner = fixture["shape"]
    if row > 1:
        cases.append((row, fixture["weightShape"][0], inner, 0,
                      str(fixture_root / fixture["name"]), 0))
with Path(args.output).open("x") as evidence:
    for index, (row, columns, inner, seed, fixture, pattern) in enumerate(cases):
        env = dict(os.environ, ET_PARTIAL_BATCH="1", ET_PARTIAL_NO_TIMING="1",
                   ET_PARTIAL_PATTERN=str(pattern))
        result = subprocess.run([args.binary, str(row), str(columns), str(inner),
                                 str(seed), fixture], env=env, text=True, capture_output=True)
        records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
        record = {"binary": args.binary, "case": index, "pattern": pattern,
                  "fixture": fixture, "exitCode": result.returncode,
                  "records": records, "stderr": result.stderr}
        evidence.write(json.dumps(record) + "\n")
        evidence.flush()
        if result.returncode != 0 or len(records) != 1 or records[0]["exact"] != records[0]["elements"]:
            raise SystemExit(f"Mismatch/failure case {index}: {record}")
        if index % 20 == 0:
            print(f"Passed {index + 1}/{len(cases)}", flush=True)
print(f"Passed all {len(cases)} cases", flush=True)
