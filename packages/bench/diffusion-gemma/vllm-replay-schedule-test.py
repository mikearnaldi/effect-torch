"""CPU-only schedule validation for fixed and distinct controlled replay."""
import ast
import copy
import os
import json
import hashlib
import re
import struct
import tempfile
from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest.mock import patch
from typing import Any

source = Path(__file__).with_name("vllm-direct.py")
names = {"request_seed", "validate_replay_schedule", "validate_replay_execution"}
namespace = {"os": os, "Any": Any, "Path": Path, "json": json, "hashlib": hashlib, "re": re, "struct": struct}
module = ast.Module(body=[node for node in ast.parse(source.read_text()).body
                         if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)

class ReplayScheduleTests(unittest.TestCase):
    def setUp(self):
        namespace["MANIFEST"] = {"generation": {"seed": 20260924}, "matrix": {
            "promptTargets": [32], "outputTokens": [64], "concurrencies": [1],
            "warmupRuns": 2, "measuredRuns": 5}}
        self.prompts = [{"targetTokens": 32, "ids": list(range(32))}]
        self.environ = patch.dict(os.environ, {"ET_VLLM_REPLAY_DIR": "/unused", "ET_VLLM_REPLAY_DISTINCT_SEEDS": "1"})
        self.environ.start()
        self.addCleanup(self.environ.stop)

    def records(self):
        result = {}
        for run in [-1, -2, 0, 1, 2, 3, 4]:
            seed = namespace["request_seed"](32, 64, 1, run, 0)
            result[seed] = ({"request": {"seed": seed, "promptTokenIds": list(range(32)), "maxNewTokens": 64}}, "a" * 64)
        return result

    def test_distinct_schedule_accepts_seven_cases(self):
        records = self.records()
        self.assertEqual(len(records), 7)
        namespace["validate_replay_schedule"](records, self.prompts)

    def test_missing_warmup_rejected_before_generation(self):
        records = self.records()
        del records[namespace["request_seed"](32, 64, 1, -2, 0)]
        with self.assertRaisesRegex(ValueError, "run -2"):
            namespace["validate_replay_schedule"](records, self.prompts)

    def test_warmup_identity_mismatch_rejected(self):
        for field, value in [("promptTokenIds", [0]), ("maxNewTokens", 63)]:
            records = copy.deepcopy(self.records())
            records[namespace["request_seed"](32, 64, 1, -1, 0)][0]["request"][field] = value
            with self.assertRaisesRegex(ValueError, "differs from fixture"):
                namespace["validate_replay_schedule"](records, self.prompts)

    def test_actual_calls_require_terminal_commit_prefill_and_every_sampler(self):
        output = SimpleNamespace(request_id="request-1", outputs=[SimpleNamespace(token_ids=[7], finish_reason="length")])
        record = ({"steps": [{}, {}], "outputTokenIds": [7]}, "a" * 64)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            internal = "request-1-deadbeef"
            (directory / "requests.jsonl").write_text(json.dumps({"internalRequestId": internal}) + "\n")
            counter = directory / hashlib.sha256(internal.encode()).hexdigest()
            counter.write_bytes(struct.pack("<QQQ", 3, 1, 2))
            namespace["validate_replay_execution"]([output], record, temporary)
            for counts in [(2, 1, 2), (3, 0, 2), (3, 1, 1), (4, 1, 2)]:
                counter.write_bytes(struct.pack("<QQQ", *counts))
                with self.assertRaisesRegex(RuntimeError, "actual.*calls"):
                    namespace["validate_replay_execution"]([output], record, temporary)

    def test_fixed_mode_keeps_one_repeated_case(self):
        os.environ["ET_VLLM_REPLAY_DISTINCT_SEEDS"] = "0"
        records = self.records()
        self.assertEqual(len(records), 1)
        namespace["validate_replay_schedule"](records, self.prompts)

if __name__ == "__main__":
    unittest.main()
