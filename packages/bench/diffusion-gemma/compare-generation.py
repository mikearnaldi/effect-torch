#!/usr/bin/env python3
"""Reject unmatched generation runs before comparing end-to-end batch latency."""

import argparse
import json
import math
import statistics
from pathlib import Path


def read(path: Path) -> dict:
    records = {}
    for line in path.read_text().splitlines():
        item = json.loads(line)
        key = tuple(item[name] for name in (
            "targetPromptTokens", "requestedOutputTokens", "concurrency", "run"
        ))
        if key in records:
            raise ValueError(f"duplicate run {key} in {path}")
        records[key] = item
    if not records:
        raise ValueError(f"empty benchmark {path}")
    return records


def compare(left: dict, right: dict) -> dict:
    differences = []
    if not left or not right:
        differences.append({"field": "runKeys", "error": "empty benchmark"})
    for key in left.keys() | right.keys():
        if (len(key) != 4 or any(type(value) is not int for value in key)
                or any(value <= 0 for value in key[:3]) or key[3] < 0):
            return {"matched": False, "differences": [
                {"field": "runKeys", "error": "invalid measurement dimensions", "run": key}
            ]}
    if left.keys() != right.keys():
        differences.append({"field": "runKeys", "effect": sorted(left), "vllm": sorted(right)})
    fields = (
        "boundary", "model", "generation", "promptTokenIds", "requestSeeds",
        "generatedTokenIds", "generatedTokensPerRequest", "refinementsPerRequest",
        "blocksPerRequest", "finishReasons", "deployment",
    )
    for key in sorted(left.keys() & right.keys()):
        workload = left[key].get("workload", "natural-generation")
        if workload not in ("natural-generation", "controlled-trajectory-replay"):
            differences.append({"run": key, "field": "workload", "error": "unknown workload mode"})
        if workload != right[key].get("workload", "natural-generation"):
            differences.append({"run": key, "field": "workload", "error": "different workload modes"})
        if workload == "controlled-trajectory-replay":
            modes = [side[key].get("replaySeedMode", "fixed") for side in (left, right)]
            if modes[0] != modes[1] or modes[0] not in ("fixed", "distinct"):
                differences.append({"run": key, "field": "replaySeedMode", "error": "different or invalid seed schedules"})
            for side in (left, right):
                if side[key].get("replaySeedMode") == "distinct":
                    generation = side[key].get("generation")
                    seed = generation.get("seed") if isinstance(generation, dict) else None
                    expected = None if type(seed) is not int else [
                        (seed + key[0] * 101 + key[1] * 17 + key[2] * 13 + key[3] * 7 + request) & 0xffffffff
                        for request in range(key[2])
                    ]
                    if expected is None or side[key].get("requestSeeds") != expected:
                        differences.append({"run": key, "field": "requestSeeds",
                                            "error": "invalid distinct replay seed schedule", "expected": expected})
                    refinements = side[key].get("refinementsPerRequest")
                    if (isinstance(refinements, list) and len(refinements) == key[2]
                            and all(type(count) is int and count > 0 for count in refinements)):
                        execution = {
                            "repetitionCooldownMilliseconds": 0,
                            "prefillInvocationsPerRequest": [1] * key[2],
                            "decodeCallsPerRequest": [count + 1 for count in refinements],
                            "samplerInvocationsPerRequest": refinements,
                            "modelReadInvocationsPerRequest": refinements,
                        }
                        for field, expected in execution.items():
                            if side[key].get(field) != expected:
                                differences.append({"run": key, "field": field,
                                                    "error": "invalid measured replay execution", "expected": expected})
            for field in ("replayManifestSha256", "replayFinalCommit", "encoderCommitsPerRequest", "prefixCache"):
                if field not in left[key] or field not in right[key] or left[key][field] != right[key][field]:
                    differences.append({"run": key, "field": field, "error": "missing or different replay contract"})
            for side in (left, right):
                for field, expected in (("replayFinalCommit", "included"),
                                        ("encoderCommitsPerRequest", [1] * key[2]),
                                        ("prefixCache", "disabled")):
                    if side[key].get(field) != expected:
                        differences.append({"run": key, "field": field,
                                            "error": "invalid controlled replay contract", "expected": expected})
                digest = side[key].get("replayManifestSha256")
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    differences.append({"run": key, "field": "replayManifestSha256", "error": "invalid manifest digest"})
        for side, expected in ((left, "effect-torch"), (right, "vllm")):
            if side[key].get("engine") != expected:
                differences.append({"run": key, "field": "engine", "expected": expected,
                                    "actual": side[key].get("engine")})
            elapsed = side[key].get("elapsedMilliseconds")
            if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0:
                differences.append({"run": key, "field": "elapsedMilliseconds", "error": "expected positive finite timing"})
            for field in ("refinementsPerRequest", "blocksPerRequest"):
                counts = side[key].get(field)
                if (not isinstance(counts, list) or len(counts) != key[2]
                        or any(type(count) is not int or count <= 0 for count in counts)):
                    differences.append({"run": key, "field": field, "error": "missing positive per-request execution evidence"})
            if side[key].get("diagnosticReadbacks", False):
                differences.append({"run": key, "field": "diagnosticReadbacks", "error": "diagnostic timing excluded"})
        if any(side[key].get("measurementMode", "timing") != "timing" for side in (left, right)):
            differences.append({"run": key, "field": "measurementMode", "error": "diagnostic timing excluded"})
        for field in fields:
            if field not in left[key] or field not in right[key]:
                differences.append({"run": key, "field": field, "error": "missing comparison evidence"})
            elif left[key][field] != right[key][field]:
                differences.append({"run": key, "field": field,
                                    "effect": left[key][field], "vllm": right[key][field]})
    for side in (left, right):
        for point in {key[:3] for key in side}:
            contracts = [
                (row.get("workload", "natural-generation"), row.get("replaySeedMode", "fixed"))
                for key, row in side.items() if key[:3] == point
            ]
            if any(contract != contracts[0] for contract in contracts[1:]):
                differences.append({"point": point, "field": "workload",
                                    "error": "cannot aggregate different workload or seed modes"})
        points = {key[:3] for key, row in side.items() if row.get("replaySeedMode") == "distinct"}
        for point in points:
            rows = [(key, row) for key, row in side.items() if key[:3] == point]
            if {key[3] for key, _ in rows} != set(range(5)):
                differences.append({"point": point, "field": "runKeys", "error": "distinct replay requires measured runs 0 through 4"})
            digests = [row.get("replayManifestSha256") for _, row in rows]
            if any(not isinstance(digest, str) for digest in digests) or len(set(digests)) != len(rows):
                differences.append({"point": point, "field": "replayManifestSha256", "error": "distinct trajectories require distinct manifests"})
    result = {"matched": not differences, "differences": differences,
              "timingBoundary": "end-to-end direct batch wall time; tokenization and initialization excluded"}
    if not differences:
        points = sorted({key[:3] for key in left})
        result["measurements"] = [{
            "promptTokens": point[0], "outputTokens": point[1], "concurrency": point[2],
            "workload": next(value.get("workload", "natural-generation")
                             for key, value in left.items() if key[:3] == point),
            "replaySeedMode": next(value.get("replaySeedMode", "fixed")
                                   if value.get("workload") == "controlled-trajectory-replay" else None
                                   for key, value in left.items() if key[:3] == point),
            "effectMedianMs": statistics.median(value["elapsedMilliseconds"]
                                                for key, value in left.items() if key[:3] == point),
            "vllmMedianMs": statistics.median(value["elapsedMilliseconds"]
                                              for key, value in right.items() if key[:3] == point),
        } for point in points]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("effect", type=Path)
    parser.add_argument("vllm", type=Path)
    args = parser.parse_args()
    result = compare(read(args.effect), read(args.vllm))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["matched"] else 1)


if __name__ == "__main__":
    main()
