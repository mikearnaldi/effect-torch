"""Regression tests for refusing misleading benchmark comparisons."""

import copy
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "compare_generation", Path(__file__).with_name("compare-generation.py")
)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        self.key = (32, 64, 1, 0)
        record = {
            "engine": "effect-torch", "boundary": "direct",
            "model": {"sha256": "checkpoint"}, "generation": {"threshold": 1},
            "deployment": {"maxTokens": 2048}, "promptTokenIds": [[1, 2]],
            "requestSeeds": [42], "generatedTokenIds": [[3, 4]],
            "generatedTokensPerRequest": [2], "refinementsPerRequest": [15],
            "blocksPerRequest": [1], "finishReasons": ["length"],
            "elapsedMilliseconds": 900, "measurementMode": "timing",
            "workload": "controlled-trajectory-replay",
            "replayManifestSha256": "a" * 64, "replayFinalCommit": "included",
            "encoderCommitsPerRequest": [1], "prefixCache": "disabled",
        }
        self.left = {self.key: record}
        self.right = copy.deepcopy(self.left)
        self.right[self.key].update(engine="vllm", elapsedMilliseconds=600)

    def test_matching_work_compares_same_wall_time(self):
        result = comparison.compare(self.left, self.right)
        self.assertTrue(result["matched"])
        self.assertEqual(result["measurements"][0]["effectMedianMs"], 900)
        self.assertEqual(result["measurements"][0]["vllmMedianMs"], 600)
        self.assertEqual(result["measurements"][0]["workload"], "controlled-trajectory-replay")
        self.assertEqual(result["measurements"][0]["replaySeedMode"], "fixed")

    def test_cannot_mix_natural_generation_and_replay_in_a_median(self):
        for side in (self.left, self.right):
            natural = copy.deepcopy(side[self.key])
            natural["workload"] = "natural-generation"
            side[(32, 64, 1, 1)] = natural
        result = comparison.compare(self.left, self.right)
        self.assertFalse(result["matched"])
        self.assertNotIn("measurements", result)

    def test_unknown_workload_is_rejected(self):
        for side in (self.left, self.right):
            side[self.key]["workload"] = "unknown"
        self.assertFalse(comparison.compare(self.left, self.right)["matched"])

    def test_empty_and_invalid_measurement_dimensions(self):
        self.assertFalse(comparison.compare({}, {})["matched"])
        for key in ((32, 64, 0, 0), (32, 64, 1, -1), ("32", 64, 1, 0)):
            self.assertFalse(comparison.compare(
                {key: self.left[self.key]}, {key: self.right[self.key]}
            )["matched"])

    def test_mismatched_or_missing_evidence_has_no_latency_table(self):
        changes = {
            "engine": "effect-torch", "generatedTokenIds": [[3, 5]],
            "requestSeeds": [43], "refinementsPerRequest": [16],
            "prefixCache": "enabled", "replayManifestSha256": "b" * 64,
            "measurementMode": "diagnostic-profile", "diagnosticReadbacks": True,
            "replaySeedMode": "distinct",
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                right = copy.deepcopy(self.right)
                right[self.key][field] = value
                result = comparison.compare(self.left, right)
                self.assertFalse(result["matched"])
                self.assertNotIn("measurements", result)
        del self.right[self.key]["blocksPerRequest"]
        self.assertFalse(comparison.compare(self.left, self.right)["matched"])

    def test_identically_invalid_contracts_are_rejected(self):
        for field, value in (
            ("replayFinalCommit", "omitted"), ("encoderCommitsPerRequest", [0]),
            ("prefixCache", "enabled"), ("replayManifestSha256", "invalid"),
            ("refinementsPerRequest", [None]), ("elapsedMilliseconds", float("nan")),
            ("elapsedMilliseconds", 0),
        ):
            with self.subTest(field=field, value=value):
                left, right = copy.deepcopy(self.left), copy.deepcopy(self.right)
                left[self.key][field] = right[self.key][field] = value
                result = comparison.compare(left, right)
                self.assertFalse(result["matched"])
                self.assertNotIn("measurements", result)

    def distinct_pair(self):
        pair = []
        for original in (self.left, self.right):
            rows = {}
            for run in range(5):
                row = copy.deepcopy(original[self.key])
                row.update(generation={"seed": 20260924}, replaySeedMode="distinct",
                           requestSeeds=[20260924 + 32 * 101 + 64 * 17 + 13 + run * 7],
                           replayManifestSha256=f"{run:064x}", repetitionCooldownMilliseconds=0,
                           prefillInvocationsPerRequest=[1], decodeCallsPerRequest=[16],
                           samplerInvocationsPerRequest=[15], modelReadInvocationsPerRequest=[15])
                rows[(32, 64, 1, run)] = row
            pair.append(rows)
        return pair

    def test_distinct_schedule(self):
        self.assertTrue(comparison.compare(*self.distinct_pair())["matched"])

    def test_distinct_schedule_rejects_invalid_evidence(self):
        for field, value in (("requestSeeds", [20265257]),
                             ("replayManifestSha256", "0" * 64),
                             ("replayManifestSha256", []),
                             ("repetitionCooldownMilliseconds", 2000),
                             ("decodeCallsPerRequest", [17]),
                             ("prefillInvocationsPerRequest", [0]),
                             ("samplerInvocationsPerRequest", [16]),
                             ("modelReadInvocationsPerRequest", [16])):
            with self.subTest(field=field, value=value):
                pair = self.distinct_pair()
                for rows in pair:
                    rows[(32, 64, 1, 1)][field] = value
                self.assertFalse(comparison.compare(*pair)["matched"])
        pair = self.distinct_pair()
        for rows in pair:
            del rows[(32, 64, 1, 4)]
        self.assertFalse(comparison.compare(*pair)["matched"])


if __name__ == "__main__":
    unittest.main()
