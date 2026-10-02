"""Run a command while recording one NVIDIA GPU to JSONL."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from common import load_manifest, python_packages_sha256


DIRECTORY = Path(__file__).resolve().parent
REPO_ROOT = DIRECTORY.parents[2]
MANIFEST = load_manifest(DIRECTORY / "manifest.json")


METADATA_FIELDS = (
    "index",
    "name",
    "uuid",
    "memory.total",
    "driver_version",
    "power.limit",
    "clocks.max.sm",
    "clocks.max.memory",
)
SAMPLE_FIELDS = (
    "index",
    "memory.used",
    "utilization.gpu",
    "power.draw",
    "clocks.sm",
    "clocks.mem",
    "temperature.gpu",
)


def query(fields: tuple[str, ...]) -> list[list[str]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [
        [value.strip() for value in row]
        for row in csv.reader(output.splitlines())
        if row
    ]


def number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def repository_identity(root: Path) -> dict[str, Any]:
    supplied_commit = os.environ.get("BENCH_REPOSITORY_COMMIT")
    supplied_tree = os.environ.get("BENCH_SOURCE_TREE_SHA256")
    supplied_dirty = os.environ.get("BENCH_REPOSITORY_DIRTY")
    supplied = (supplied_commit, supplied_tree, supplied_dirty)
    if any(value is not None for value in supplied):
        if not all(value is not None for value in supplied):
            raise RuntimeError("repository identity overrides must be supplied together")
        if len(supplied_commit) != 40 or any(
            character not in "0123456789abcdef" for character in supplied_commit
        ):
            raise RuntimeError("BENCH_REPOSITORY_COMMIT must be a lowercase SHA-1")
        if len(supplied_tree) != 64 or any(
            character not in "0123456789abcdef" for character in supplied_tree
        ):
            raise RuntimeError("BENCH_SOURCE_TREE_SHA256 must be a lowercase SHA-256")
        if supplied_dirty not in {"0", "1"}:
            raise RuntimeError("BENCH_REPOSITORY_DIRTY must be 0 or 1")
        return {
            "commit": supplied_commit,
            "dirty": supplied_dirty == "1",
            "sourceTreeSha256": supplied_tree,
        }

    command = ["git", "-C", str(root)]
    commit = subprocess.check_output(command + ["rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(command + ["status", "--porcelain=v1", "-z"])
    listed = subprocess.check_output(
        command + ["ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    )
    digest = hashlib.sha256()
    for relative_bytes in sorted(path for path in listed.split(b"\0") if path):
        relative = os.fsdecode(relative_bytes)
        source = root / relative
        digest.update(relative_bytes)
        digest.update(b"\0")
        if not source.exists() and not source.is_symlink():
            digest.update(b"deleted\0")
            continue
        stat = source.lstat()
        digest.update(str(stat.st_mode).encode())
        digest.update(b"\0")
        if source.is_symlink():
            digest.update(os.fsencode(os.readlink(source)))
        elif source.is_dir():
            nested = ["git", "-C", str(source)]
            digest.update(subprocess.check_output(nested + ["rev-parse", "HEAD"]))
            digest.update(
                subprocess.check_output(nested + ["status", "--porcelain=v1", "-z"])
            )
        else:
            with source.open("rb") as contents:
                while chunk := contents.read(1024 * 1024):
                    digest.update(chunk)
        digest.update(b"\0")
    return {
        "commit": commit,
        "dirty": bool(status),
        "sourceTreeSha256": digest.hexdigest(),
    }


def python_identity() -> dict[str, str]:
    return {
        "executable": sys.executable,
        "version": platform.python_version(),
        "packagesSha256": python_packages_sha256(),
    }


def write_line(output: TextIO, record: dict[str, Any]) -> None:
    output.write(json.dumps(record, separators=(",", ":")) + "\n")
    output.flush()


def collect(
    process: subprocess.Popen[str], output: TextIO, ready: threading.Event
) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        row = next(csv.reader([line]))
        values = [value.strip() for value in row]
        if len(values) != len(SAMPLE_FIELDS):
            continue
        observed = time.time()
        memory_mib = number(values[1])
        write_line(
            output,
            {
                "type": "sample",
                "timestamp": datetime.fromtimestamp(observed, timezone.utc).isoformat(),
                "unixSeconds": observed,
                "gpuIndex": int(values[0]),
                "memoryUsedBytes": (
                    None if memory_mib is None else int(memory_mib * 1024 * 1024)
                ),
                "utilizationPercent": number(values[2]),
                "powerWatts": number(values[3]),
                "smClockMHz": number(values[4]),
                "memoryClockMHz": number(values[5]),
                "temperatureCelsius": number(values[6]),
            },
        )
        ready.set()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if args.interval_ms < 20:
        parser.error("--interval-ms must be at least 20")

    metadata = query(METADATA_FIELDS)
    if len(metadata) != 1:
        raise RuntimeError(f"expected one visible NVIDIA GPU, found {len(metadata)}")
    values = metadata[0]
    if len(values) != len(METADATA_FIELDS):
        raise RuntimeError(f"unexpected nvidia-smi metadata: {values}")
    if MANIFEST["hardware"]["gpu"] not in values[1]:
        raise RuntimeError(
            f'expected {MANIFEST["hardware"]["gpu"]}, got {values[1]}'
        )
    memory_bytes = int(float(values[3]) * 1024 * 1024)
    if memory_bytes < MANIFEST["hardware"]["minimumMemoryBytes"]:
        raise RuntimeError(
            f'GPU memory {memory_bytes} is below the required '
            f'{MANIFEST["hardware"]["minimumMemoryBytes"]}'
        )

    repository = repository_identity(REPO_ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        write_line(
            output,
            {
                "type": "metadata",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "command": command,
                "repository": repository,
                "python": python_identity(),
                "intervalMilliseconds": args.interval_ms,
                "gpu": {
                    "index": int(values[0]),
                    "name": values[1],
                    "uuid": values[2],
                    "memoryTotalBytes": memory_bytes,
                    "driverVersion": values[4],
                    "powerLimitWatts": number(values[5]),
                    "maxSmClockMHz": number(values[6]),
                    "maxMemoryClockMHz": number(values[7]),
                },
            },
        )

        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(SAMPLE_FIELDS)}",
                "--format=csv,noheader,nounits",
                f"--loop-ms={args.interval_ms}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        ready = threading.Event()
        collector = threading.Thread(
            target=collect, args=(monitor, output, ready), daemon=True
        )
        collector.start()
        if not ready.wait(timeout=5):
            monitor.terminate()
            monitor.wait(timeout=5)
            raise RuntimeError("nvidia-smi did not produce a telemetry sample")

        try:
            measured = subprocess.run(command, check=False)
        finally:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
            collector.join(timeout=5)

        write_line(
            output,
            {
                "type": "exit",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "exitCode": measured.returncode,
            },
        )
    return measured.returncode


if __name__ == "__main__":
    sys.exit(main())
