from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        manifest = json.load(source)
    if manifest.get("schemaVersion") != 1:
        raise ValueError(f"unsupported manifest schema: {manifest.get('schemaVersion')}")
    return manifest


def prompt_cases(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[str] = manifest["promptWords"]
    cases: list[dict[str, Any]] = []
    for target in manifest["matrix"]["promptTargets"]:
        for prompt in manifest["prompts"]:
            padding_tokens = (
                target
                - prompt["baseTokens"]
                - manifest["promptPaddingOverheadTokens"]
            )
            if padding_tokens < 1:
                raise ValueError(f'prompt target {target} is too small for {prompt["id"]}')
            padding = " ".join(
                words[(prompt["offset"] + index) % len(words)]
                for index in range(padding_tokens)
            )
            content = f'{prompt["question"]}\n\nContext: {padding}'
            cases.append(
                {
                    "id": f'{prompt["id"]}-p{target}',
                    "targetTokens": target,
                    "messages": [{"role": "user", "content": content}],
                    "contentSha256": hashlib.sha256(content.encode()).hexdigest(),
                }
            )
    return cases


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")


def numeric_version(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in version.split("."):
        digits = "".join(character for character in part if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def python_packages_sha256() -> str:
    packages = sorted(
        f'{distribution.metadata.get("Name", "unknown").lower()}=={distribution.version}'
        for distribution in importlib.metadata.distributions()
    )
    return hashlib.sha256("\n".join(packages).encode()).hexdigest()


def verify_vllm_distribution(manifest: dict[str, Any]) -> dict[str, str]:
    version = importlib.metadata.version("vllm")
    target = manifest["target"]
    if version != target["minimumVersion"]:
        raise RuntimeError(f'expected vLLM {target["minimumVersion"]}, got {version}')

    build_commit = os.environ.get("VLLM_BUILD_COMMIT")
    if build_commit == target["imageCommit"]:
        kind = "container"
        identity = build_commit
    else:
        distribution = importlib.metadata.distribution("vllm")
        direct_text = distribution.read_text("direct_url.json")
        if direct_text is None:
            raise RuntimeError(
                "vLLM is neither the pinned container build nor the pinned wheel"
            )
        direct = json.loads(direct_text)
        archive = direct.get("archive_info") or {}
        hashes = archive.get("hashes") or {}
        wheel_sha256 = hashes.get("sha256") or str(
            archive.get("hash", "")
        ).removeprefix("sha256=")
        if direct.get("url") != target["wheelUrl"]:
            raise RuntimeError("installed vLLM wheel URL differs")
        if wheel_sha256 and wheel_sha256 != target["wheelSha256"]:
            raise RuntimeError("installed vLLM wheel SHA-256 differs")
        marker_path = Path(sys.prefix) / "effect-torch-vllm.json"
        if not marker_path.is_file():
            raise RuntimeError("pinned vLLM installation marker is missing")
        with marker_path.open(encoding="utf-8") as source:
            marker = json.load(source)
        expected_marker = {
            "url": target["wheelUrl"],
            "sha256": target["wheelSha256"],
            "version": target["minimumVersion"],
        }
        if marker != expected_marker:
            raise RuntimeError("pinned vLLM installation marker differs")
        kind = "wheel"
        identity = target["wheelSha256"]

    return {
        "kind": kind,
        "identity": identity,
        "version": version,
        "environmentSha256": python_packages_sha256(),
    }
