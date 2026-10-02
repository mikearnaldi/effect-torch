"""Rebuild the pinned checkpoint and independent official-model CUDA oracle.

Download uses only Python's standard library. Oracle generation requires Torch
2.10.0 and Transformers at 93ebf6b11127967f2725cf4d012aae55c3654f5a.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import types
import urllib.request

MODEL = "google/diffusiongemma-26B-A4B-it"
REVISION = "f7f5b7f5fa82ffc52addd066915886d497f5517b"
TRANSFORMERS_REVISION = "93ebf6b11127967f2725cf4d012aae55c3654f5a"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def download(destination):
    destination.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(url) as response:
        metadata = json.load(response)
    if metadata["sha"] != REVISION:
        raise RuntimeError("checkpoint revision mismatch")
    files = [entry for entry in metadata["siblings"] if entry["rfilename"].endswith(
        (".safetensors", ".json", ".jinja", ".model")
    )]

    def fetch(entry):
        name = entry["rfilename"]
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        lfs = entry.get("lfs", {})
        expected = lfs.get("sha256")
        if path.exists() and expected and digest(path) == expected:
            return {"file": name, "bytes": path.stat().st_size, "sha256": expected}
        temporary = path.with_suffix(path.suffix + ".partial")
        with urllib.request.urlopen(f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}") as response:
            with temporary.open("wb") as output:
                for chunk in iter(lambda: response.read(8 << 20), b""):
                    output.write(chunk)
        actual = digest(temporary)
        if expected and actual != expected:
            raise RuntimeError(f"checkpoint hash mismatch: {name}")
        if entry.get("size") is not None and temporary.stat().st_size != entry["size"]:
            raise RuntimeError(f"checkpoint size mismatch: {name}")
        temporary.replace(path)
        print(f"verified {name}: {actual}", flush=True)
        return {"file": name, "bytes": path.stat().st_size, "sha256": actual,
                "officialSha256": expected}

    with ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(fetch, files))
    write_json(destination / "verified-checkpoint.json", {
        "model": MODEL, "revision": REVISION, "files": verified,
    })


def reference(checkpoint, output, state_only=False, components_only=False):
    import importlib.metadata
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, DiffusionGemmaConfig, DiffusionGemmaForBlockDiffusion
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaTextRotaryEmbedding
    from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
        DiffusionGemmaGenerationConfig, EntropyBoundSampler, EntropyBoundSamplerConfig,
    )

    if (torch.__version__ != "2.10.0+cu128" or torch.version.cuda != "12.8"
            or torch.version.git_version != "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0"):
        raise RuntimeError("Torch reference pin mismatch")
    distribution = importlib.metadata.distribution("transformers")
    source = json.loads(distribution.read_text("direct_url.json"))
    archive = f"https://github.com/huggingface/transformers/archive/{TRANSFORMERS_REVISION}.tar.gz"
    if source["url"] != archive:
        raise RuntimeError("Transformers source pin mismatch")
    if output.exists():
        raise RuntimeError("preserve existing oracle; choose a new output directory")
    output.mkdir(parents=True)
    (output / "producer.py").write_bytes(Path(__file__).read_bytes())
    state_directory = output / "initialized-state"
    state_directory.mkdir(exist_ok=True)
    config = DiffusionGemmaConfig.from_pretrained(checkpoint, local_files_only=True)
    rotary = DiffusionGemmaTextRotaryEmbedding(config.text_config)
    tensors = {
        f"model.decoder.rotary_emb.{kind}_inv_freq": getattr(rotary, f"{kind}_inv_freq").contiguous()
        for kind in sorted(set(config.text_config.layer_types))
    }
    state_file = state_directory / "initialized-rope.safetensors"
    save_file(tensors, str(state_file))
    state = {
        "status": "passed", "kind": "diffusion-gemma-initialized-rope-v1",
        "configSha256": digest(checkpoint / "config.json"), "file": state_file.name,
        "sha256": digest(state_file), "payloadBytes": sum(t.numel() * t.element_size() for t in tensors.values()),
        "producerSha256": digest(Path(__file__)), "torchVersion": torch.__version__,
        "torchCommit": torch.version.git_version, "cpuCapability": torch.backends.cpu.get_cpu_capability(),
        "sourcePins": {"model": {"revision": REVISION}, "transformers": {"archive_url": archive}},
        "tensors": [{"name": name, "dtype": "f32", "shape": list(t.shape),
                     "sha256": hashlib.sha256(t.numpy().tobytes()).hexdigest()} for name, t in tensors.items()],
    }
    write_json(state_directory / "manifest.json", state)
    print("initialized state:", state["sha256"], flush=True)
    if state_only:
        return

    torch.manual_seed(20260919)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    prompt = None
    for count in range(300):
        text = "Explain why the sky appears blue to a curious student." + " harbor" * count
        encoded = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                                add_generation_prompt=True, enable_thinking=False)
        ids = encoded["input_ids"]
        if len(ids) == 278:
            prompt = ids
            break
    if prompt is None:
        raise RuntimeError("could not construct 278-token templated prompt")
    labels = [tokenizer.encode(label, add_special_tokens=False)[0] for label in ["A", "B", "C", "D"]]
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        checkpoint, dtype=torch.bfloat16, attn_implementation="eager", experts_implementation="eager",
        local_files_only=True,
    ).eval().to("cuda")
    # CPU constructor buffers must survive BF16 module casting unchanged.
    for kind in sorted(set(config.text_config.layer_types)):
        frequency = tensors[f"model.decoder.rotary_emb.{kind}_inv_freq"].to("cuda")
        for embedding in [model.model.decoder.rotary_emb, model.model.encoder.language_model.rotary_emb]:
            setattr(embedding, f"{kind}_inv_freq", frequency.clone())
            setattr(embedding, f"{kind}_original_inv_freq", frequency.clone())
    prompt_tensor = torch.tensor([prompt], dtype=torch.long, device="cuda")
    if components_only:
        capture_components(model, prompt_tensor, output / "expert-components", torch)
        return
    inputs = {"case": {"promptIds": prompt, "labelIds": labels, "slot": 0}, "reads": []}
    records = []
    with torch.inference_mode():
        prefix = model.model.encoder(input_ids=prompt_tensor).past_key_values
        for index in range(4):
            canvas = torch.randint(0, config.text_config.vocab_size, (1, config.canvas_length), device="cuda")
            logits = model(decoder_input_ids=canvas, past_key_values=prefix).logits
            filename = f"read-{index}.safetensors"
            save_file({"logits.answer": logits[:, :1].cpu().contiguous()}, str(output / filename))
            records.append({"file": filename, "sha256": digest(output / filename)})
            inputs["reads"].append({"canvas_ids": canvas[0].cpu().tolist(), "index": index})
            print(f"saved independent read {index}", flush=True)
        del prefix, logits
        write_json(output / "inputs.json", inputs)
        generate_reference(model, prompt_tensor, output, labels, inputs["reads"][0]["canvas_ids"],
                           torch, DiffusionGemmaGenerationConfig, EntropyBoundSampler, EntropyBoundSamplerConfig)
    write_json(output / "manifest.json", {
        "model": MODEL, "revision": REVISION, "inputs_sha256": digest(output / "inputs.json"),
        "status": "passed", "reads": records, "torchVersion": torch.__version__,
        "torchCommit": torch.version.git_version, "transformersSource": source,
        "attentionImplementation": "eager", "expertsImplementation": model.get_experts_implementation(),
        "cudaVersion": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "producerSha256": digest(Path(__file__)), "checkpointManifestSha256": digest(checkpoint / "verified-checkpoint.json"),
    })


def capture_components(model, prompt, directory, torch):
    directory.mkdir(exist_ok=True)
    original_linear = torch.nn.functional.linear
    phase = "encoder"
    cases = []
    counts = {"encoder": 0, "decoder": 0}

    def linear(x, weight, bias=None):
        result = original_linear(x, weight, bias)
        if tuple(weight.shape) in [(1408, 2816), (2816, 704)] and counts[phase] < 10:
            name = f"{phase}-{counts[phase]}"
            records = {}
            for suffix, tensor in [("input.bf16", x), ("weight.bf16", weight), ("official.bf16", result)]:
                if tensor.dtype != torch.bfloat16:
                    raise RuntimeError("expert component dtype differs")
                path = directory / f"{name}.{suffix}"
                path.write_bytes(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
                records[suffix] = {"bytes": path.stat().st_size, "sha256": digest(path)}
            cases.append({"name": name, "shape": list(x.shape), "weightShape": list(weight.shape), "files": records})
            counts[phase] += 1
        return result

    torch.nn.functional.linear = linear
    try:
        with torch.inference_mode():
            prefix = model.model.encoder(input_ids=prompt).past_key_values
            phase = "decoder"
            canvas = torch.randint(0, model.config.text_config.vocab_size, (1, model.config.canvas_length), device="cuda")
            model(decoder_input_ids=canvas, past_key_values=prefix)
    finally:
        torch.nn.functional.linear = original_linear
    if counts != {"encoder": 10, "decoder": 10}:
        raise RuntimeError("expert component capture incomplete")
    write_json(directory / "report.json", {"status": "passed", "cases": cases,
        "model": MODEL, "revision": REVISION, "torchVersion": torch.__version__,
        "expertsImplementation": model.get_experts_implementation(), "producerSha256": digest(Path(__file__))})


def generate_reference(model, prompt, output, labels, decision_canvas, torch, config_type, sampler_type, sampler_config_type):
    directory = output / "generation"
    directory.mkdir(exist_ok=True)
    canvases, blocks, draws = [], [], []
    original_initialize = sampler_type.initialize_canvas
    original_multinomial = torch.multinomial
    original_step = model._denoising_step

    def initialize(sampler, *args, **kwargs):
        canvas = original_initialize(sampler, *args, **kwargs)
        canvases.append(canvas[0].cpu().tolist())
        return canvas

    def multinomial(probabilities, num_samples, *args, **kwargs):
        if num_samples != 1 or args or kwargs:
            raise RuntimeError("unexpected official multinomial call")
        before = torch.cuda.get_rng_state()
        sampled = original_multinomial(probabilities, num_samples)
        after = torch.cuda.get_rng_state()
        try:
            torch.cuda.set_rng_state(before)
            exponentials = torch.empty_like(probabilities).exponential_()
            replay = torch.argmax(probabilities / exponentials, dim=-1, keepdim=True)
            if not torch.equal(replay, sampled) or not torch.equal(torch.cuda.get_rng_state(), after):
                raise RuntimeError("exponential capture does not reproduce official multinomial and RNG state")
            filename = f"exponentials-{len(draws)}.f32"
            path = directory / filename
            exponentials.cpu().numpy().tofile(path)
            draws.append({"file": filename, "bytes": path.stat().st_size, "sha256": digest(path)})
        finally:
            torch.cuda.set_rng_state(after)
        return sampled

    def step(_model, *args, **kwargs):
        if kwargs["cur_step"] == settings["max_denoising_steps"]:
            blocks.append({"steps": []})
        result = original_step(*args, **kwargs)
        blocks[-1]["steps"].append({"argmax_tokens": result[1][0].cpu().tolist(), "exponentials": draws[-1]})
        print(f"saved generation block {len(blocks)-1} step {len(blocks[-1]['steps'])-1}", flush=True)
        return result

    settings = {
        "max_new_tokens": 257, "max_denoising_steps": 3, "t_min": 0.4, "t_max": 0.8,
        "stability_threshold": 3, "confidence_threshold": 0.005, "eos_token_id": None,
        "pad_token_id": 0, "sampler_config": {"entropy_bound": 0.1},
    }
    generation_config = config_type(**{**settings, "sampler_config": sampler_config_type(entropy_bound=0.1)},
                                  disable_compile=True, return_dict_in_generate=True)
    sampler_type.initialize_canvas = initialize
    torch.multinomial = multinomial
    model._denoising_step = types.MethodType(step, model)
    try:
        result = model.generate(input_ids=prompt, generation_config=generation_config)
    finally:
        sampler_type.initialize_canvas = original_initialize
        torch.multinomial = original_multinomial
        model._denoising_step = original_step
    if len(blocks) != 2 or any(len(block["steps"]) != 3 for block in blocks) or len(canvases) != 8:
        raise RuntimeError("reference did not exercise two blocks and six steps")
    write_json(directory / "generation.json", {
        "status": "passed", "prompt": prompt[0].cpu().tolist(), "canvas_length": model.config.canvas_length,
        "vocab_size": model.config.text_config.vocab_size, "config": settings,
        "random_canvases": canvases, "blocks": blocks, "sequences": result.sequences[0].cpu().tolist(),
    })
    write_json(output / "generation-inputs.json", {
        "promptIds": prompt[0].cpu().tolist(), "labelIds": labels, "answerRow": 0,
        "canvasIds": decision_canvas, "maxNewTokens": 257, "maxSteps": 3,
        "stabilityThreshold": 3, "eosTokenIds": [], "outputLimit": "whole-block", "seed": 20260919,
        "generationReference": str((directory / "generation.json").resolve()), "prefillChunks": [256, 278],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "state", "oracle", "components"])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    if args.command == "download":
        download(args.checkpoint)
    elif args.output is None:
        parser.error("state/oracle requires an output directory")
    else:
        reference(args.checkpoint, args.output, state_only=args.command == "state", components_only=args.command == "components")


if __name__ == "__main__":
    main()
