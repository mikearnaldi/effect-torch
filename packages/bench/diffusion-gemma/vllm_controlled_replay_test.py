"""CPU-only integrity and request-isolation checks; no torch dependency."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from vllm_controlled_replay import Bank, ReplayRequest, SCHEMA, validate_manifest


class ReplayValidationTest(unittest.TestCase):
    def fixture(self, directory):
        payload = b"stand-in integrity payload; preload validates actual safetensors"
        (directory / "feedback.safetensors").write_bytes(payload)
        steps = []
        for index in range(2):
            steps.append(dict(index=index, block=0, step=index,
                canvasTokenIds=[index] * 256, postCanvasTokenIds=[index + 1] * 256,
                draftTokenIds=[3] * 256, temperature=0.8, rngDraw=index, done=index == 1,
                feedbackInput=None if index == 0 else dict(file="feedback.safetensors",
                    sha256=hashlib.sha256(payload).hexdigest(), dtype="BF16", shape=[1, 256, 262144])))
        return dict(schema=SCHEMA, label="controlled-trajectory-replay-not-natural-generation",
            request=dict(seed=42, promptTokenIds=[2, 7], maxNewTokens=64),
            steps=steps, outputTokenIds=[3] * 64, committedCanvasTokenIds=[3] * 256)

    def test_valid_and_corrupt_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = self.fixture(directory)
            path = directory / "manifest.json"
            path.write_text(json.dumps(manifest))
            actual, digest = validate_manifest(path)
            self.assertEqual(actual, manifest)
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            mutations = [
                lambda m: m["steps"][1].update(canvasTokenIds=[9] * 256),
                lambda m: m["steps"][1].update(rngDraw=0),
                lambda m: m["steps"][0].update(done=True),
                lambda m: m["steps"][1]["feedbackInput"].update(sha256="0" * 64),
                lambda m: m["steps"][1]["feedbackInput"].update(file="../outside.safetensors"),
                lambda m: m["steps"][1]["feedbackInput"].update(dtype="F32"),
                lambda m: m.update(label="natural-generation"),
                lambda m: m["request"].update(seed=-1),
            ]
            for mutate in mutations:
                changed = copy.deepcopy(manifest)
                mutate(changed)
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    validate_manifest(path)

    def test_bank_request_identity(self):
        request = ReplayRequest(42, (2, 7), (3,), 64, (), "hash", None)
        bank = Bank([request])
        self.assertIs(bank.get_request(42, [2, 7]), request)
        self.assertIsNone(bank.get_request(43))
        with self.assertRaises(ValueError):
            bank.get_request(42, [2, 8])
        with self.assertRaises(ValueError):
            Bank([request, request])


if __name__ == "__main__":
    unittest.main()
