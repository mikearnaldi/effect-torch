"""Compare CUTLASS grouped slices with cuBLAS32 streams on labeled histograms."""
import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("vectors")
parser.add_argument("output")
parser.add_argument("--binary-prefix", default="/root/expert-cutlass-")
parser.add_argument("--tiles", default="16x64x32,16x128x32,32x64x32,32x128x32,16x64x64,16x128x64,32x64x64,32x128x64")
args = parser.parse_args()
vectors = json.loads(Path(args.vectors).read_text())["vectors"]
with Path(args.output).open("x") as evidence:
    for tile in args.tiles.split(","):
        for vector in vectors:
            for projection in range(2):
                result = subprocess.run(
                    [args.binary_prefix + tile + "-heterogeneous", str(projection), "71",
                     str(len(vector["rows"])), "0", ",".join(map(str, vector["rows"]))],
                    text=True, capture_output=True)
                records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
                record = {"tile": tile, "histogram": vector["label"],
                          "projection": projection, "exitCode": result.returncode,
                          "records": records, "stderr": result.stderr}
                evidence.write(json.dumps(record) + "\n")
                evidence.flush()
                if result.returncode != 0 or len(records) != 3 or any(
                    item["exact"] != item["elements"] or
                    item.get("graphExact", item["elements"]) != item["elements"] for item in records
                ):
                    raise SystemExit(f"Mismatch/failure: {record}")
                print(tile, vector["label"], projection,
                      [(v["mode"], v.get("graphMedianMs")) for v in records], flush=True)
