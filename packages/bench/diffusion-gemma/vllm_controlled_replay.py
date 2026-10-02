"""Validated, eagerly preloaded inputs for the controlled trajectory benchmark.

No model output is substituted: the caller still executes every model and
sampler step, then selects the recorded next input outside that computation.
Importing this module does not import torch or initialize CUDA.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA = "effect-torch-controlled-trajectory-v1"
CANVAS = 256
VOCAB = 262144


def _integer(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"invalid {name}")
    return value


def _tokens(value, name, length=None):
    if not isinstance(value, list) or (length is not None and len(value) != length):
        raise ValueError(f"invalid {name} length")
    return tuple(_integer(token, name, maximum=VOCAB - 1) for token in value)


def validate_manifest(path):
    """Validate CPU metadata and file integrity before any GPU allocation."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    document = json.loads(raw)
    if document.get("schema") != SCHEMA:
        raise ValueError("unsupported controlled trajectory schema")
    if document.get("label") != "controlled-trajectory-replay-not-natural-generation":
        raise ValueError("missing controlled replay measurement label")
    request = document["request"]
    _integer(request["seed"], "seed", maximum=0xFFFFFFFF)
    prompt = _tokens(request["promptTokenIds"], "promptTokenIds")
    if not prompt:
        raise ValueError("empty replay prompt")
    maximum = _integer(request["maxNewTokens"], "maxNewTokens", 1, CANVAS)
    output = _tokens(document["outputTokenIds"], "outputTokenIds")
    committed = _tokens(document["committedCanvasTokenIds"], "committedCanvasTokenIds", CANVAS)
    if not output or len(output) > maximum:
        raise ValueError("invalid replay output length")
    if output != committed[:len(output)]:
        raise ValueError("published output differs from committed canvas prefix")
    steps = document["steps"]
    if not isinstance(steps, list) or not steps:
        raise ValueError("empty replay steps")
    block = steps[0]["block"]
    first_step = steps[0]["step"]
    if block != 0 or first_step != 0:
        raise ValueError("replay must begin at block zero, step zero")
    for index, step in enumerate(steps):
        if step["index"] != index or step["block"] != block or step["step"] != first_step + index:
            raise ValueError("replay requires contiguous steps in one block")
        for field in ("canvasTokenIds", "postCanvasTokenIds", "draftTokenIds"):
            _tokens(step[field], field, CANVAS)
        temperature = step["temperature"]
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("invalid temperature")
        if _integer(step["rngDraw"], "rngDraw") != index:
            raise ValueError("replay denoise draw counter must start at zero")
        if type(step["done"]) is not bool or step["done"] != (index == len(steps) - 1):
            raise ValueError("only the final replay step may be done")
        feedback = step["feedbackInput"]
        if (feedback is None) != (index == 0):
            raise ValueError("feedback must be absent exactly on the initial step")
        if feedback is not None:
            if feedback["dtype"] != "BF16" or feedback["shape"] != [1, CANVAS, VOCAB]:
                raise ValueError("invalid replay feedback dtype/shape")
            file = (path.parent / feedback["file"]).resolve()
            if not file.is_relative_to(path.parent):
                raise ValueError("feedback file escapes capture directory")
            if hashlib.sha256(file.read_bytes()).hexdigest() != feedback["sha256"]:
                raise ValueError("replay feedback SHA-256 mismatch")
        if index and step["canvasTokenIds"] != steps[index - 1]["postCanvasTokenIds"]:
            raise ValueError("replay canvas transition differs from prior step")
    return document, hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ReplayStep:
    canvas: Any
    post_canvas: Any
    draft: Any
    feedback: Any
    temperature: float
    rng_draw: int
    done: bool


@dataclass(frozen=True)
class ReplayRequest:
    seed: int
    prompt_token_ids: tuple
    output_token_ids: tuple
    max_new_tokens: int
    steps: tuple[ReplayStep, ...]
    manifest_sha256: str
    committed_canvas: Any


class Bank:
    def __init__(self, requests):
        self.requests = {}
        for request in requests:
            if request.seed in self.requests:
                raise ValueError("ambiguous duplicate replay seed")
            self.requests[request.seed] = request

    def get_request(self, seed, prompt_ids=None):
        request = self.requests.get(seed)
        if request is not None and prompt_ids is not None and tuple(prompt_ids) != request.prompt_token_ids:
            raise ValueError("replay seed matched a different prompt")
        return request


def preload(directory, device):
    """Load every canonical tensor before the benchmark measurement starts."""
    import torch
    from safetensors.torch import load_file

    directory = Path(directory)
    paths = [directory] if directory.is_file() else sorted(directory.glob("**/manifest.json"))
    if not paths:
        raise ValueError("no controlled trajectory manifests found")
    validated = [(path, *validate_manifest(path)) for path in paths]
    requests = []
    for path, document, digest in validated:
        steps = []
        for step in document["steps"]:
            feedback = None
            if step["feedbackInput"] is not None:
                tensors = load_file(str(path.parent / step["feedbackInput"]["file"]), device="cpu")
                if set(tensors) != {"feedback"}:
                    raise ValueError("replay safetensors must contain only feedback")
                feedback = tensors["feedback"]
                if feedback.dtype != torch.bfloat16 or list(feedback.shape) != [1, CANVAS, VOCAB]:
                    raise ValueError("safetensors feedback differs from declared dtype/shape")
                feedback = feedback.to(device=device)
            tensor = lambda field: torch.tensor(step[field], dtype=torch.int64, device=device)
            steps.append(ReplayStep(tensor("canvasTokenIds"), tensor("postCanvasTokenIds"),
                tensor("draftTokenIds"), feedback, float(step["temperature"]), step["rngDraw"], step["done"]))
        request = document["request"]
        requests.append(ReplayRequest(request["seed"], tuple(request["promptTokenIds"]),
            tuple(document["outputTokenIds"]), request["maxNewTokens"], tuple(steps), digest,
            torch.tensor(document["committedCanvasTokenIds"], dtype=torch.int64, device=device)))
    bank = Bank(requests)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return bank
