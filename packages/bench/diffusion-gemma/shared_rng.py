"""Benchmark overlay RNG matching Effect Torch's existing CUDA/Mulberry32 streams.

No production RNG changes: the GPU stream reproduces Tensor.uniform(F32), and
canvas draws reproduce DiffusionGemma.generationRandom(seed). Call install_gpu()
before warmup/measurement, then uniform_like inside the sampler's timed work.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
_GPU_UNIFORM: Any = None


def mix64(value: int) -> int:
    value = (value + GOLDEN) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def uniform_f32(seed: int, draw: int, index: int, source: int = 0) -> float:
    """Scalar oracle; integer overflow and F64 midpoint occur before F32 cast."""
    mixed = mix64((seed + source + draw * GOLDEN + 2 * index) & MASK64)
    value = ((mixed >> 11) + 0.5) * (1.0 / 9007199254740992.0)
    return struct.unpack("<f", struct.pack("<f", value))[0]


@dataclass
class CanvasStream:
    """Request-local JS Mulberry32 canvas stream with rejection sampling."""

    state: int
    draws: int = 0
    words: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.state <= MASK32:
            raise ValueError("seed must be a u32")

    def next_u32(self) -> int:
        self.state = (self.state + 0x6D2B79F5) & MASK32
        value = ((self.state ^ (self.state >> 15)) * (self.state | 1)) & MASK32
        value ^= (value + ((value ^ (value >> 7)) * (value | 61))) & MASK32
        self.words += 1
        return (value ^ (value >> 14)) & MASK32

    def canvas(self, length: int, vocab_size: int) -> list[int]:
        if length < 0 or not 1 <= vocab_size <= MASK32:
            raise ValueError("invalid canvas geometry")
        limit = (1 << 32) - (1 << 32) % vocab_size
        result: list[int] = []
        for _ in range(length):
            value = self.next_u32()
            while value >= limit:
                value = self.next_u32()
            result.append(value % vocab_size)
        self.draws += 1
        return result


def install_gpu() -> Any:
    """Register a torch.compile-compatible pure custom op; compile during warmup.

    Inputs: contiguous F32 [requests,...] template, I64 request seeds and logical
    denoise draw counters, both [requests] on the same GPU. Draw zero is the
    first sampler invocation. Encoder-only commits must not advance this stream.
    """
    global _GPU_UNIFORM, torch, triton, tl
    if _GPU_UNIFORM is not None:
        return _GPU_UNIFORM
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(out, seeds, draws, count: tl.constexpr, per_request: tl.constexpr, BLOCK: tl.constexpr):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = index < count
        request = index // per_request
        local_index = index % per_request
        seed = tl.load(seeds + request, mask=valid, other=0).to(tl.uint64)
        draw = tl.load(draws + request, mask=valid, other=0).to(tl.uint64)
        value = seed + draw * tl.full((), 0x9E3779B97F4A7C15, tl.uint64) + local_index.to(tl.uint64) * 2
        value += tl.full((), 0x9E3779B97F4A7C15, tl.uint64)
        value = (value ^ (value >> 30)) * tl.full((), 0xBF58476D1CE4E5B9, tl.uint64)
        value = (value ^ (value >> 27)) * tl.full((), 0x94D049BB133111EB, tl.uint64)
        value ^= value >> 31
        uniform = ((value >> 11).to(tl.float64) + 0.5) * (1.0 / 9007199254740992.0)
        tl.store(out + index, uniform.to(tl.float32), mask=valid)

    @torch.library.custom_op("effect_torch_shared_rng::uniform", mutates_args=())
    def gpu_uniform(template: torch.Tensor, seeds: torch.Tensor, draws: torch.Tensor) -> torch.Tensor:
        if template.dtype != torch.float32 or not template.is_cuda or not template.is_contiguous():
            raise ValueError("uniform template must be contiguous CUDA F32")
        if template.ndim < 1 or template.shape[0] == 0:
            raise ValueError("uniform requires at least one request")
        for metadata in (seeds, draws):
            if metadata.dtype != torch.int64 or metadata.device != template.device or not metadata.is_contiguous():
                raise ValueError("seed/draw metadata must be contiguous CUDA I64 on template device")
            if metadata.shape != (template.shape[0],):
                raise ValueError("seed/draw metadata must have one value per request")
        result = torch.empty_like(template)
        count = template.numel()
        kernel[(triton.cdiv(count, 256),)](
            result, seeds, draws, count, count // template.shape[0], 256,
            enable_fp_fusion=False,
        )
        return result

    @gpu_uniform.register_fake
    def fake(template: torch.Tensor, seeds: torch.Tensor, draws: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(template)

    _GPU_UNIFORM = gpu_uniform
    return gpu_uniform


def uniform_like(template: Any, seeds: Any, draws: Any) -> Any:
    """Use install_gpu() before entering torch.compile or a measured request."""
    if _GPU_UNIFORM is None:
        raise RuntimeError("call shared_rng.install_gpu() before sampler compilation")
    return _GPU_UNIFORM(template, seeds, draws)
