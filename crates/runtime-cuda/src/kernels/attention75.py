"""Offline Triton source: BF16 attention over Effect's immutable row table.

Online-softmax algorithm follows the retained vLLM unified-attention audit;
the row-pointer loads and end-relative window implement Effect's cache ABI.
No import of this module is needed by the eventual Rust runtime.
"""
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["LANE", "SEQUENCE", "TOKENS", "ROWS", "POSITIONS", "WINDOW", "SCALE", "ERROR_CONTEXT", "NONCAUSAL", "TOKEN_MAJOR"])
def attention75_native(Q, TABLE, OUT, STATUS, COUNTS, CURSORS, META,
                       LANE, SEQUENCE, TOKENS, ROWS, POSITIONS, WINDOW,
                       SCALE, ERROR_CONTEXT, NONCAUSAL, TOKEN_MAJOR,
                       DIM: tl.constexpr,
                       BLOCK_M: tl.constexpr = 16, BLOCK_N: tl.constexpr = 32):
    # Header row offsets remain relative to the complete original table.
    HEADS: tl.constexpr = 16
    VALID = tl.load(COUNTS + LANE)
    CURSOR = tl.load(CURSORS + LANE).to(tl.uint64)
    rank = tl.load(META + 1)
    k_shape = 9 + tl.load(META) + rank
    KV_HEADS = tl.load(META + k_shape + rank - 3)
    Q = Q + LANE.to(tl.uint64) * HEADS * TOKENS * DIM
    OUT = OUT + LANE.to(tl.uint64) * HEADS * TOKENS * DIM
    tile = tl.program_id(0)
    head = tl.program_id(1)
    retained = tl.load(TABLE + SEQUENCE * 4 + 0)
    committed = tl.load(TABLE + SEQUENCE * 4 + 1)
    end_all = tl.load(TABLE + SEQUENCE * 4 + 2)
    row_offset = tl.load(TABLE + SEQUENCE * 4 + 3)
    invalid = ((KV_HEADS == 0) | (KV_HEADS > HEADS) | (HEADS % tl.maximum(KV_HEADS, 1) != 0) | (VALID > ROWS) | (VALID < 0) | (VALID > TOKENS) | (end_all < retained)
               | (end_all - retained != POSITIONS) | (committed < retained)
               | (committed > end_all) | (CURSOR < committed)
               | (CURSOR + VALID > end_all))
    # Uniform branch precedes all pointer dereferences; preserve first error.
    if invalid:
        tl.atomic_cas(STATUS, tl.full((), 0, tl.uint64),
                      (ERROR_CONTEXT.to(tl.uint64) << 32) | tl.full((), 1, tl.uint64))
        return
    row = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.arange(0, DIM)
    active = (row < ROWS) & (row < VALID)
    end = tl.where(NONCAUSAL != 0, end_all, CURSOR + row + 1)
    start = tl.maximum(retained, tl.where(WINDOW > 0, tl.maximum(end, WINDOW) - WINDOW, retained))
    q = tl.load(Q + (head * TOKENS + row[:, None]) * DIM + d[None, :],
                mask=active[:, None], other=0).to(tl.bfloat16)
    maximum = tl.full((BLOCK_M,), float('-inf'), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, DIM), tl.float32)
    kv_head = head * KV_HEADS // HEADS
    for block in range(tl.cdiv(POSITIONS, BLOCK_N)):
        p = block * BLOCK_N + tl.arange(0, BLOCK_N)
        present = p < POSITIONS
        kaddr = tl.load(TABLE + row_offset + p * 4, present, other=0)
        vaddr = tl.load(TABLE + row_offset + p * 4 + 1, present, other=0)
        kptr = kaddr.to(tl.pointer_type(tl.bfloat16))
        vptr = vaddr.to(tl.pointer_type(tl.bfloat16))
        k = tl.load(kptr[None, :] + kv_head * DIM + d[:, None],
                    mask=present[None, :], other=0)
        v = tl.load(vptr[:, None] + kv_head * DIM + d[None, :],
                    mask=present[:, None], other=0)
        score = tl.dot(q, k) * SCALE
        allowed = (active[:, None] & present[None, :]
                   & (retained + p[None, :] >= start[:, None])
                   & (retained + p[None, :] < end[:, None]))
        score = tl.where(allowed, score, float('-inf'))
        new_max = tl.maximum(maximum, tl.max(score, 1))
        # Empty prefixes/tiles never evaluate an effective -inf - -inf.
        finite_max = tl.where(new_max == float('-inf'), 0.0, new_max)
        alpha = tl.exp(maximum - finite_max)
        prob = tl.exp(score - finite_max[:, None])
        denominator = denominator * alpha + tl.sum(prob, 1)
        acc = acc * alpha[:, None]
        acc += tl.dot(prob.to(tl.bfloat16), v)
        maximum = new_max
    result = acc / tl.where(denominator > 0, denominator, 1.0)[:, None]
    # Both layouts are supported without an extra permutation launch.
    if TOKEN_MAJOR:
        offset = (row[:, None] * HEADS + head) * DIM + d[None, :]
    else:
        offset = (head * TOKENS + row[:, None]) * DIM + d[None, :]
    tl.store(OUT + offset, result.to(tl.bfloat16), mask=row[:, None] < TOKENS)
