"""Pinned-size paged attention parity and optional before/after measurements.

Run via check-nvrtc.py --cache-only --cache-full [--cache-baseline SOURCE].
The independent reference exploits sparse queries and periodic values, not the
kernel reduction implementation. Every production head dimension is executed.
"""
import json
import time

baseline = None
if "--cache-baseline" in sys.argv:
    baseline = Path(sys.argv[sys.argv.index("--cache-baseline") + 1]).read_text()
    modules["cache_baseline"] = compile_module("cache_baseline", header + "\n" + baseline)


def bf16(value):
    return decode(encode(f32(value), 3), 3)


def full_case(dim, kheads, bidirectional):
    begin = len(allocations)
    tokens, qheads, context, count, capacity = 64, 16, 64, 61, 544
    end = context + count
    q = [((t + head) % 7 - 3) / 4 if d == 0 else 0.0
         for head in range(qheads) for t in range(tokens) for d in range(dim)]

    def key(pos, head, d):
        return ((pos * 3 + head + d) % 11 - 5) / 4

    def value(pos, head, d):
        return ((pos + head + d % 8) % 13 - 6) / 4

    def payload(values):
        return b"".join(encode(x, 3) for x in values)

    a = Args()
    a.elements = qheads * tokens * dim
    a.inputs[0] = upload(payload(q))
    for role, fn in enumerate([key, value], 1):
        a.inputs[role] = upload(payload(fn(context + t, head, d)
                                       for head in range(kheads) for t in range(tokens) for d in range(dim)))
        a.input_dtypes[role] = 3
    a.input_dtypes[0] = a.output_dtype = 3
    a.compute_dtype = 1
    guards = bytes([165]) * 16
    stores, initial, expected_stores = [], [], []
    row_bytes = kheads * dim * 2
    for fn in [key, value]:
        prefix = payload(fn(pos, head, d) for pos in range(context) for head in range(kheads) for d in range(dim))
        current = payload(fn(pos, head, d) for pos in range(context, end) for head in range(kheads) for d in range(dim))
        tail = bytes([165]) * (tokens * row_bytes)
        stores.append(upload(guards + prefix + tail + guards) + 16)
        initial.append(prefix + tail)
        expected_stores.append(prefix + current + tail[len(current):])
    table = [0, context, end, 4]
    for pos in range(end):
        table.extend([stores[0] + pos * row_bytes, stores[1] + pos * row_bytes, 0, 0])
    a.inputs[3] = upload(struct.pack("<" + "Q" * len(table), *table))
    stride = capacity + tokens
    score_bytes = qheads * tokens * stride * 4
    score_allocation = upload(guards + bytes(score_bytes) + guards)
    a.inputs[4] = score_allocation + 16
    a.inputs[7] = upload(struct.pack("<I", context))
    a.scratch[0] = upload(struct.pack("<I", count))
    a.scratch[3] = upload(bytes(4))
    out_allocation = upload(guards + bytes(a.elements * 2) + guards)
    a.output = out_allocation + 16
    shape, kvshape = [1, qheads, tokens, dim], [1, kheads, tokens, dim]
    meta = [4, 4, 4, 4, 0, 0, 0, 0, 0] + shape * 2 + kvshape * 2
    a.metadata = upload(struct.pack("<" + "Q" * len(meta), *meta))
    a.integers[0], a.integers[1], a.integers[2] = capacity, 3, 1
    a.integers[3], a.integers[4], a.integers[5] = 0 if bidirectional else 1024, bidirectional, 1
    a.integers[6], a.integers[7], a.integers[8] = 1, tokens, 3
    a.integers[9], a.integers[10] = stride, dim
    a.scalars[0] = f32(.30157)
    expected = bytearray()
    for head in range(qheads):
        kh = head * kheads // qheads
        for t in range(tokens):
            if t >= count:
                expected.extend(bytes(dim * 2))
                continue
            positions = range(end if bidirectional else context + t + 1)
            query = ((t + head) % 7 - 3) / 4
            scores = [bf16(bf16(query * key(pos, kh, 0)) * a.scalars[0]) for pos in positions]
            maximum = max(scores)
            weights = [f32(math.exp(f32(score - maximum))) for score in scores]
            total = f32(sum(weights))
            probabilities = [bf16(weight / total) for weight in weights]
            periodic = [bf16(sum(p * value(pos, kh, d) for p, pos in zip(probabilities, positions))) for d in range(8)]
            expected.extend(payload(periodic[d % 8] for d in range(dim)))
    launch("cache", "et_kv_attention", a)
    actual = read(a.output, a.elements * 2)
    assert actual == expected, (dim, bidirectional, "independent BF16 attention reference differs")
    for role in range(2):
        assert read(stores[role] - 16, len(initial[role]) + 32) == guards + expected_stores[role] + guards
    assert read(out_allocation, 16) == guards and read(a.output + a.elements * 2, 16) == guards
    assert read(score_allocation, 16) == guards and read(a.inputs[4] + score_bytes, 16) == guards
    assert read(a.scratch[3], 4) == bytes(4)
    elapsed = []
    for _ in range(3):
        start = time.perf_counter()
        launch("cache", "et_kv_attention", a)
        elapsed.append((time.perf_counter() - start) * 1000)
        assert read(a.output, a.elements * 2) == actual
    report = dict(queries=tokens, queryHeads=qheads, kvHeads=kheads, headDim=dim,
                  context=context, bidirectional=bidirectional, queryWarps=qheads * tokens,
                  scoreWorkspaceBytes=score_bytes, milliseconds=min(elapsed))
    if baseline is not None:
        fn = c.c_void_p()
        checked(driver.cuModuleGetFunction(c.byref(fn), modules["cache_baseline"], b"et_kv_attention"))
        params = (c.c_void_p * 1)(c.addressof(a))
        start = time.perf_counter()
        checked(driver.cuLaunchKernel(fn, 1, 1, 1, 256, 1, 1, 0, None, params, None))
        checked(driver.cuCtxSynchronize())
        report["baselineMilliseconds"] = (time.perf_counter() - start) * 1000
        report["speedup"] = report["baselineMilliseconds"] / report["milliseconds"]
        assert read(a.output, a.elements * 2) == actual, (dim, "previous kernel differs")
        assert report["speedup"] > 5, report
    print(json.dumps(report), flush=True)
    for ptr in allocations[begin:]:
        checked(driver.cuMemFree_v2(ptr))
    del allocations[begin:]


for dim, heads in [(256, 8), (512, 2)]:
    for bidirectional in [False, True]:
        full_case(dim, heads, bidirectional)
print("PASSED full-geometry BF16 causal/canvas parity, padding, storage guards and replay")
