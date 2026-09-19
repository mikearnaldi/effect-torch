"""GPU KV regressions, run by check-nvrtc.py --cache-only.

The driver helpers come from that runner. The CPU reference independently
appends raw cache rows and evaluates stable softmax over their decoded values.
"""

def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def encode(value, dtype):
    if dtype == 0:
        return struct.pack("<d", value)
    if dtype == 1:
        return struct.pack("<f", value)
    if dtype == 2:
        return struct.pack("<e", value)
    if not value:
        return struct.pack("<H", 0x8000 if math.copysign(1, value) < 0 else 0)
    exponent = math.frexp(abs(value))[1] - 1
    quantum = math.ldexp(1.0, max(exponent, -126) - 7)
    rounded = round(value / quantum) * quantum
    return struct.pack("<H", struct.unpack("<I", struct.pack("<f", rounded))[0] >> 16)


def decode(payload, dtype):
    if dtype == 3:
        return struct.unpack("<f", bytes(2) + payload)[0]
    return struct.unpack("<" + {0: "d", 1: "f", 2: "e"}[dtype], payload)[0]


def run_case(activation, cache, batch, rows, tokens, dim, qheads, kheads, capacity, context, window, bidirectional):
    begin_allocations = len(allocations)
    lanes = batch * rows
    aw = 8 if activation == 0 else 4 if activation == 1 else 2
    cw = 4 if cache == 1 else 1 if cache == 6 else 2
    compute = 0 if activation == 0 else 1
    output_dtype = compute
    ow = 8 if compute == 0 else 4
    scale = 1 / math.sqrt(dim)
    scale = scale if compute == 0 else f32(scale)
    values = []
    payloads = []
    for role, heads in enumerate([qheads, kheads, kheads]):
        data = [math.sin(i * .739 + role * 1.17) * .83 for i in range(lanes * heads * tokens * dim)]
        payload = b"".join(encode(v, activation) for v in data)
        payloads.append(payload)
        values.append([decode(payload[i:i + aw], activation) for i in range(0, len(payload), aw)])
    cache_rows = batch * capacity * kheads
    raw = [bytearray(cache_rows * dim * cw), bytearray(cache_rows * dim * cw)]
    scales = [[1.0] * cache_rows, [1.0] * cache_rows]

    def write_row(role, row, data):
        if cache == 6:
            data = [f32(x) for x in data]
            maximum = max(abs(x) for x in data)
            scaling = f32(maximum / 127) if maximum else 1.0
            scales[role][row] = scaling
            payload = bytes(max(-127, min(127, round(f32(x / scaling)))) + 128 for x in data)
        else:
            payload = b"".join(encode(x, cache) for x in data)
        start = row * dim * cw
        raw[role][start:start + dim * cw] = payload

    def load(role, row, d):
        i = row * dim + d
        if cache == 6:
            return f32((raw[role][i] - 128) * scales[role][row])
        return decode(raw[role][i * cw:(i + 1) * cw], cache)

    for role in range(2):
        for row in range(cache_rows):
            write_row(role, row, [math.cos((row * dim + d) * .313 + role) * .7 for d in range(dim)])
    initial = [bytes(b) for b in raw]
    initial_scales = [struct.pack("<" + "f" * cache_rows, *s) for s in scales]
    valid, cursors = [], []
    for sequence in range(batch):
        cursor = context + sequence % 3
        for row in range(rows):
            count = tokens if row == 0 else max(0, tokens - row % (tokens + 1))
            if sequence % 5 == 4:
                count = 0
            valid.append(count)
            cursors.append(cursor)
            cursor += count
    expected = [0.0] * (lanes * qheads * tokens * dim)
    for sequence in range(batch):
        for lane in range(sequence * rows, (sequence + 1) * rows):
            count = valid[lane]

            def append(token):
                for head in range(kheads):
                    source = ((lane * kheads + head) * tokens + token) * dim
                    dest = (sequence * capacity + (cursors[lane] + token) % capacity) * kheads + head
                    for role in range(2):
                        write_row(role, dest, values[role + 1][source:source + dim])

            if bidirectional:
                for token in range(count):
                    append(token)
            for token in range(count):
                if not bidirectional:
                    append(token)
                end = cursors[lane] + (count if bidirectional else token + 1)
                start = max(0, end - capacity, end - window if window else 0)
                for head in range(qheads):
                    query = ((lane * qheads + head) * tokens + token) * dim
                    positions = [(sequence * capacity + pos % capacity) * kheads + head * kheads // qheads for pos in range(start, end)]
                    scores = [sum(values[0][query + d] * load(0, row, d) for d in range(dim)) * scale for row in positions]
                    maximum = max(scores)
                    weights = [math.exp(score - maximum) for score in scores]
                    denominator = sum(weights)
                    for d in range(dim):
                        expected[query + d] = sum(w * load(1, row, d) for w, row in zip(weights, positions)) / denominator
    a = Args()
    a.elements = len(expected)
    a.compute_dtype = compute
    a.output_dtype = output_dtype
    guards = bytes([165]) * 16
    out_allocation = upload(guards + bytes([165]) * (len(expected) * ow) + guards)
    a.output = out_allocation + 16
    for role in range(3):
        a.inputs[role] = upload(payloads[role])
        a.input_dtypes[role] = activation
    for role in range(2):
        a.inputs[3 + role] = upload(guards + initial[role] + guards) + 16
        a.inputs[5 + role] = upload(guards + initial_scales[role] + guards) + 16
    a.inputs[7] = upload(struct.pack("<" + "I" * lanes, *cursors))
    a.scratch[0] = upload(struct.pack("<" + "I" * lanes, *valid))
    a.scratch[3] = upload(bytes(4))
    shape = [lanes, qheads, tokens, dim]
    kvshape = [lanes, kheads, tokens, dim]
    meta = [4, 4, 4, 4, 0, 0, 0, 0, 0] + shape * 2 + kvshape * 2
    a.metadata = upload(struct.pack("<" + "Q" * len(meta), *meta))
    a.integers[0] = capacity
    a.integers[1] = cache
    a.integers[2] = rows
    a.integers[3] = window
    a.integers[4] = bidirectional
    a.integers[5] = batch
    a.scalars[0] = scale
    launch("cache", "et_kv_attention", a)
    assert read(a.scratch[3], 4) == bytes(4)
    output = read(out_allocation, len(expected) * ow + 32)
    assert output[:16] == guards and output[-16:] == guards
    actual = struct.unpack("<" + ("d" if compute == 0 else "f") * len(expected), output[16:-16])
    tolerance = 2e-12 if compute == 0 else 3e-6
    error = max(abs(x - y) for x, y in zip(actual, expected))
    assert error <= tolerance, (activation, cache, dim, context, window, bidirectional, error)
    for role in range(2):
        assert read(a.inputs[3 + role] - 16, len(raw[role]) + 32) == guards + bytes(raw[role]) + guards
        expected_scales = struct.pack("<" + "f" * cache_rows, *scales[role])
        assert read(a.inputs[5 + role] - 16, len(expected_scales) + 32) == guards + expected_scales + guards
    # Restore the original state and repeat to verify deterministic execution.
    for role in range(2):
        checked(driver.cuMemcpyHtoD_v2(c.c_uint64(a.inputs[3 + role]), initial[role], c.c_size_t(len(initial[role]))))
        checked(driver.cuMemcpyHtoD_v2(c.c_uint64(a.inputs[5 + role]), initial_scales[role], c.c_size_t(len(initial_scales[role]))))
    launch("cache", "et_kv_attention", a)
    assert read(a.output, len(expected) * ow) == output[16:-16]
    for ptr in allocations[begin_allocations:]:
        checked(driver.cuMemFree_v2(ptr))
    del allocations[begin_allocations:]


cases = 0
for activation in [0, 1, 2, 3]:
    for cache in [1, 2, 3, 6]:
        for bidirectional in [False, True]:
            # More than eight sequences requires more than one 256-thread block.
            run_case(activation, cache, 9, 3, 3, 5, 4, 2, 7, 6, 0, bidirectional)
            run_case(activation, cache, 2, 4, 2, 1, 2, 1, 37, 71, 17, bidirectional)
            run_case(activation, cache, 1, 2, 2, 33, 2, 1, 67, 130, 0, bidirectional)
            cases += 3
print(f"PASSED {cases} warp KV dtype/ring/padding/GQA/bidirectional cases, exact cache bytes and deterministic replay")
