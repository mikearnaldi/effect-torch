"""Validate and summarize paired Effect Torch and vLLM benchmark JSONL files."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from common import load_manifest


Key = tuple[str, int, int, int]
DIRECTORY = Path(__file__).resolve().parent


def read_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if line.strip():
                    record = json.loads(line)
                    record["_source"] = f"{path}:{line_number}"
                    records.append(record)
    return records


def group_key(record: dict[str, Any]) -> Key:
    return (
        record["boundary"],
        record["targetPromptTokens"],
        record["requestedOutputTokens"],
        record["concurrency"],
    )


def validate_record(record: dict[str, Any]) -> None:
    if any(
        count != record["targetPromptTokens"]
        for count in record["actualPromptTokens"]
    ):
        raise ValueError(f'{record["_source"]}: prompt token count missed target')
    counts = record["generatedTokensPerRequest"]
    if len(counts) != record["concurrency"]:
        raise ValueError(f'{record["_source"]}: generated request count differs')
    if sum(counts) != record["generatedTokens"]:
        raise ValueError(f'{record["_source"]}: generated token total differs')
    if any(
        count <= 0 or count > record["requestedOutputTokens"] for count in counts
    ):
        raise ValueError(f'{record["_source"]}: generated token count is out of range')
    reasons = record["finishReasons"]
    if len(reasons) != record["concurrency"] or any(
        reason not in {"stop", "length"} for reason in reasons
    ):
        raise ValueError(f'{record["_source"]}: finish reason is invalid')
    if record["boundary"] == "http":
        pages = record["pageMillisecondsPerRequest"]
        sizes = record["pageUtf8BytesPerRequest"]
        if len(pages) != record["concurrency"] or len(sizes) != len(pages):
            raise ValueError(f'{record["_source"]}: page request count differs')
        if any(not request_pages for request_pages in pages):
            raise ValueError(f'{record["_source"]}: request emitted no text pages')
        if any(
            len(request_pages) != len(request_sizes)
            for request_pages, request_sizes in zip(pages, sizes)
        ):
            raise ValueError(f'{record["_source"]}: page timestamps and sizes differ')


def median(records: list[dict[str, Any]], field: str) -> float:
    return statistics.median(float(record[field]) for record in records)


def latency_percentile(records: list[dict[str, Any]], quantile: float) -> float:
    if records[0]["boundary"] == "http":
        values = sorted(
            float(value)
            for record in records
            for value in record["requestMilliseconds"]
        )
    else:
        values = sorted(float(record["elapsedMilliseconds"]) for record in records)
    return values[min(len(values) - 1, int(quantile * len(values)))]


def latency_sample_count(records: list[dict[str, Any]]) -> int:
    if records[0]["boundary"] == "http":
        return sum(len(record["requestMilliseconds"]) for record in records)
    return len(records)


def first_text_summary(records: list[dict[str, Any]]) -> str:
    if records[0]["boundary"] != "http":
        return "n/a"
    values = sorted(
        float(value)
        for record in records
        for value in record["firstTextMilliseconds"]
    )
    return "/".join(
        f"{values[min(len(values) - 1, int(quantile * len(values)))]:.2f}"
        for quantile in (0.5, 0.95, 0.99)
    )


def page_count_median(records: list[dict[str, Any]]) -> float:
    counts = [
        len(request_pages)
        for record in records
        for request_pages in record["pageMillisecondsPerRequest"]
    ]
    return statistics.median(counts)


def page_interval_summary(records: list[dict[str, Any]]) -> str:
    intervals = sorted(
        current - previous
        for record in records
        for request_pages in record["pageMillisecondsPerRequest"]
        for previous, current in zip(request_pages, request_pages[1:])
    )
    if not intervals:
        return "n/a"
    return "/".join(
        f"{intervals[min(len(intervals) - 1, int(quantile * len(intervals)))]:.2f}"
        for quantile in (0.5, 0.95, 0.99)
    )


def read_telemetry(specifications: list[str]) -> dict[str, dict[str, Any]]:
    telemetry: dict[str, dict[str, Any]] = {}
    for specification in specifications:
        engine, separator, path_text = specification.partition("=")
        if separator == "" or engine not in {"effect-torch", "vllm"}:
            raise ValueError("telemetry must use effect-torch=PATH or vllm=PATH")
        metadata: dict[str, Any] | None = None
        exit_record: dict[str, Any] | None = None
        samples: list[dict[str, Any]] = []
        with Path(path_text).open(encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                if record.get("type") == "metadata":
                    metadata = record
                elif record.get("type") == "sample":
                    samples.append(record)
                elif record.get("type") == "exit":
                    exit_record = record
        if metadata is None:
            raise ValueError(f"{path_text}: no telemetry metadata")
        if not samples:
            raise ValueError(f"{path_text}: no telemetry samples")
        if exit_record is None or exit_record.get("exitCode") != 0:
            raise ValueError(f"{path_text}: measured command did not exit successfully")
        telemetry[engine] = {"metadata": metadata, "samples": samples}
    return telemetry


def validate_telemetry_pair(telemetry: dict[str, dict[str, Any]]) -> None:
    if set(telemetry) != {"effect-torch", "vllm"}:
        raise ValueError("telemetry must include effect-torch and vllm")
    effect = telemetry["effect-torch"]["metadata"]
    vllm = telemetry["vllm"]["metadata"]
    for path in (
        ("gpu", "uuid"),
        ("gpu", "driverVersion"),
        ("gpu", "powerLimitWatts"),
        ("repository", "commit"),
        ("repository", "sourceTreeSha256"),
    ):
        left = effect
        right = vllm
        for key in path:
            left = left[key]
            right = right[key]
        if left != right:
            raise ValueError(f"telemetry metadata differs at {'.'.join(path)}")


def attach_telemetry(
    records: list[dict[str, Any]],
    telemetry: dict[str, dict[str, Any]],
) -> None:
    for record in records:
        engine = record["engine"]
        if engine not in telemetry:
            raise ValueError(f"missing telemetry for {engine}")
        distribution = record.get("engineDistribution")
        if distribution is not None and (
            distribution["environmentSha256"]
            != telemetry[engine]["metadata"]["python"]["packagesSha256"]
        ):
            raise ValueError(f'{record["_source"]}: Python environment hash differs')
        ended = datetime.fromisoformat(record["timestamp"]).timestamp()
        elapsed_seconds = float(record["elapsedMilliseconds"]) / 1000
        started = ended - elapsed_seconds
        window = [
            sample
            for sample in telemetry[engine]["samples"]
            if started <= float(sample["unixSeconds"]) <= ended
        ]
        if not window:
            raise ValueError(f'{record["_source"]}: no telemetry samples in timed window')
        memory = [
            float(sample["memoryUsedBytes"])
            for sample in window
            if sample["memoryUsedBytes"] is not None
        ]
        utilization = [
            float(sample["utilizationPercent"])
            for sample in window
            if sample["utilizationPercent"] is not None
        ]
        power = [
            float(sample["powerWatts"])
            for sample in window
            if sample["powerWatts"] is not None
        ]
        if not memory or not utilization or not power:
            raise ValueError(f'{record["_source"]}: incomplete telemetry in timed window')
        energy_joules = statistics.mean(power) * elapsed_seconds
        record["peakGpuMemoryBytes"] = max(memory)
        record["meanGpuUtilizationPercent"] = statistics.mean(utilization)
        record["meanPowerWatts"] = statistics.mean(power)
        record["energyJoules"] = energy_joules
        record["energyJoulesPerToken"] = energy_joules / float(
            record["generatedTokens"]
        )


def validate_pair(
    key: Key,
    effect: list[dict[str, Any]],
    vllm: list[dict[str, Any]],
) -> None:
    if len(effect) != len(vllm):
        raise ValueError(f"{key}: run count differs: {len(effect)} vs {len(vllm)}")
    for left, right in zip(
        sorted(effect, key=lambda record: record["run"]),
        sorted(vllm, key=lambda record: record["run"]),
    ):
        for field in (
            "model",
            "generation",
            "deployment",
            "promptIds",
            "promptContentSha256",
            "actualPromptTokens",
        ):
            if left[field] != right[field]:
                raise ValueError(
                    f'{key} run {left["run"]}: {field} differs between engines'
                )


def validate_matrix(
    grouped: dict[Key, dict[str, list[dict[str, Any]]]],
    manifest: dict[str, Any],
    allow_partial: bool,
) -> None:
    boundaries = {key[0] for key in grouped}
    if len(boundaries) != 1:
        raise ValueError(f"expected one boundary, got {sorted(boundaries)}")
    boundary = next(iter(boundaries))
    if not allow_partial:
        expected = {
            (boundary, prompt, output, concurrency)
            for prompt in manifest["matrix"]["promptTargets"]
            for output in manifest["matrix"]["outputTokens"]
            for concurrency in manifest["matrix"]["concurrencies"]
        }
        if set(grouped) != expected:
            missing = sorted(expected - set(grouped))
            extra = sorted(set(grouped) - expected)
            raise ValueError(f"benchmark matrix differs: missing={missing}, extra={extra}")

    expected_runs = list(range(manifest["matrix"]["measuredRuns"]))
    for key, engines in grouped.items():
        if set(engines) != {"effect-torch", "vllm"}:
            raise ValueError(
                f"{key}: expected effect-torch and vllm, got {sorted(engines)}"
            )
        if not allow_partial:
            for engine, records in engines.items():
                runs = sorted(record["run"] for record in records)
                if runs != expected_runs:
                    raise ValueError(f"{key} {engine}: expected runs {expected_runs}, got {runs}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--telemetry", action="append", default=[])
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(os.environ.get("MANIFEST", DIRECTORY / "manifest.json"))
    manifest = load_manifest(manifest_path)

    records = read_records(args.files)
    for record in records:
        validate_record(record)
    distributions = {
        json.dumps(record["engineDistribution"], sort_keys=True)
        for record in records
        if "engineDistribution" in record
    }
    if len(distributions) > 1:
        raise ValueError("vLLM distribution differs across result records")
    telemetry = read_telemetry(args.telemetry)
    if telemetry:
        validate_telemetry_pair(telemetry)
        attach_telemetry(records, telemetry)

    grouped: dict[Key, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        grouped[group_key(record)][record["engine"]].append(record)
    validate_matrix(grouped, manifest, args.allow_partial)

    lines = [
        "| Boundary | Prompt | Output | C | N | Effect tok/s | vLLM tok/s | Effect/vLLM | Effect p50/p95/p99 ms | vLLM p50/p95/p99 ms | Effect first text ms | vLLM first text ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in sorted(grouped):
        engines = grouped[key]
        effect = engines["effect-torch"]
        vllm = engines["vllm"]
        validate_pair(key, effect, vllm)
        effect_tps = median(effect, "aggregateTokensPerSecond")
        vllm_tps = median(vllm, "aggregateTokensPerSecond")
        effect_latency = "/".join(
            f"{latency_percentile(effect, quantile):.2f}"
            for quantile in (0.5, 0.95, 0.99)
        )
        vllm_latency = "/".join(
            f"{latency_percentile(vllm, quantile):.2f}"
            for quantile in (0.5, 0.95, 0.99)
        )
        samples = latency_sample_count(effect)
        if samples != latency_sample_count(vllm):
            raise ValueError(f"{key}: latency sample count differs between engines")
        boundary, prompt, output, concurrency = key
        lines.append(
            f"| {boundary} | {prompt} | {output} | {concurrency} | {samples} | "
            f"{effect_tps:.2f} | {vllm_tps:.2f} | {effect_tps / vllm_tps:.3f} | "
            f"{effect_latency} | {vllm_latency} | {first_text_summary(effect)} | "
            f"{first_text_summary(vllm)} |"
        )

    http_keys = [key for key in sorted(grouped) if key[0] == "http"]
    if http_keys:
        lines.extend(
            [
                "",
                "| Prompt | Output | C | Effect pages/request | vLLM pages/request | Effect inter-page p50/p95/p99 ms | vLLM inter-page p50/p95/p99 ms |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for key in http_keys:
            effect = grouped[key]["effect-torch"]
            vllm = grouped[key]["vllm"]
            _, prompt, output, concurrency = key
            lines.append(
                f"| {prompt} | {output} | {concurrency} | "
                f"{page_count_median(effect):.1f} | {page_count_median(vllm):.1f} | "
                f"{page_interval_summary(effect)} | {page_interval_summary(vllm)} |"
            )

    if telemetry:
        lines.extend(
            [
                "",
                "| Boundary | Prompt | Output | C | Effect J/token | vLLM J/token | Effect peak GiB | vLLM peak GiB | Effect util % | vLLM util % | Effect W | vLLM W |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for key in sorted(grouped):
            effect = grouped[key]["effect-torch"]
            vllm = grouped[key]["vllm"]
            boundary, prompt, output, concurrency = key
            lines.append(
                f"| {boundary} | {prompt} | {output} | {concurrency} | "
                f'{median(effect, "energyJoulesPerToken"):.4f} | '
                f'{median(vllm, "energyJoulesPerToken"):.4f} | '
                f'{median(effect, "peakGpuMemoryBytes") / 1024**3:.2f} | '
                f'{median(vllm, "peakGpuMemoryBytes") / 1024**3:.2f} | '
                f'{median(effect, "meanGpuUtilizationPercent"):.1f} | '
                f'{median(vllm, "meanGpuUtilizationPercent"):.1f} | '
                f'{median(effect, "meanPowerWatts"):.1f} | '
                f'{median(vllm, "meanPowerWatts"):.1f} |'
            )

    report = "\n".join(lines) + "\n"
    if args.output is None:
        print(report, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
