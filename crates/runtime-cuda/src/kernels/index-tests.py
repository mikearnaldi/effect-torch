"""Deterministic scatter-add reference and traced-workload timing.

Run through check-nvrtc.py --index-only [--index-baseline SOURCE]. The reference
pushes source updates in order; it does not duplicate the output-slice kernel.
"""
import json
import time

baseline = "--index-baseline" in sys.argv
if baseline:
    source = Path(sys.argv[sys.argv.index("--index-baseline") + 1]).read_text()
    modules["index_baseline"] = compile_module("index_baseline", header + source)


def packed(value, dtype):
    if dtype == 3:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        return struct.pack("<H", (bits + 0x7fff + ((bits >> 16) & 1)) >> 16)
    if dtype >= 4:
        bits = {4: 64, 5: 32, 6: 8}[dtype]
        value = int(value) % (1 << bits)
        if dtype == 4 and value >= 1 << 63:
            value -= 1 << 64
    return struct.pack("<" + {0: "d", 1: "f", 2: "e", 4: "q", 5: "I", 6: "B"}[dtype], value)


def unpacked(value, dtype):
    if dtype == 3:
        return struct.unpack("<f", bytes(2) + value)[0]
    return struct.unpack("<" + {0: "d", 1: "f", 2: "e", 4: "q", 5: "I", 6: "B"}[dtype], value)[0]


def coordinates(index, shape):
    result = []
    for width in reversed(shape):
        result.append(index % width)
        index //= width
    return list(reversed(result))


def scatter_case(shape, src_shape, axis, dtype, index_dtype, duplicates=True, timing=False):
    begin = len(allocations)
    elements, updates = math.prod(shape), math.prod(src_shape)
    anchor = {0: 1.0, 1: 1.0, 2: 2048.0, 3: 256.0, 4: (1 << 53) + 1, 5: (1 << 32) - 1, 6: 255}[dtype]
    original = [unpacked(packed(anchor if i % 3 else -anchor, dtype), dtype) for i in range(elements)]
    sources = [unpacked(packed([1, 1, -1, -1, 2, -2, 0, 1][i % 8], dtype), dtype) for i in range(updates)]
    indexes = []
    expected = original.copy()
    for i, update in enumerate(sources):
        coord = coordinates(i, src_shape)
        selected = ((coord[axis] // 2 if duplicates else coord[axis] * 3) + coord[-1] % 3) % shape[axis]
        indexes.append(selected)
        coord[axis] = selected
        target = 0
        for position, width in zip(coord, shape):
            target = target * width + position
        expected[target] = unpacked(packed(expected[target] + update, dtype), dtype)
    expected_bytes = b"".join(packed(v, dtype) for v in expected)
    a = Args()
    a.elements, a.operation, a.integers[0] = elements, 5, axis
    a.inputs[0] = upload(b"".join(packed(v, dtype) for v in original))
    index_bytes = b"".join(packed(v, index_dtype) for v in indexes)
    a.inputs[1] = upload(index_bytes)
    a.inputs[2] = upload(b"".join(packed(v, dtype) for v in sources))
    a.input_dtypes[0] = a.input_dtypes[2] = a.output_dtype = dtype
    a.input_dtypes[1] = index_dtype
    guards = bytes([165]) * 16
    output = upload(guards + bytes(len(expected_bytes)) + guards)
    a.output = output + 16
    a.scratch[3] = upload(bytes(4))
    rank = len(shape)
    meta = [rank, rank, rank, rank, 0, 0, 0, 0, 0] + shape * 2 + src_shape * 2
    a.metadata = upload(struct.pack("<" + "Q" * len(meta), *meta))
    launch("typed", "et_index", a)
    actual = read(a.output, len(expected_bytes))
    assert actual == expected_bytes, (shape, src_shape, axis, dtype, index_dtype, "ordered reference")
    assert read(a.scratch[3], 4) == bytes(4)
    assert read(output, 16) == guards and read(a.output + len(actual), 16) == guards
    start = time.perf_counter()
    launch("typed", "et_index", a)
    milliseconds = (time.perf_counter() - start) * 1000
    assert read(a.output, len(actual)) == actual
    report = dict(shape=shape, sourceShape=src_shape, axis=axis, dtype=dtype, duplicates=duplicates, milliseconds=milliseconds)
    if baseline:
        start = time.perf_counter()
        launch("index_baseline", "et_index", a)
        report["baselineMilliseconds"] = (time.perf_counter() - start) * 1000
        report["speedup"] = report["baselineMilliseconds"] / milliseconds
        assert read(a.output, len(actual)) == actual, (shape, axis, dtype, "previous kernel")
        if timing:
            assert report["speedup"] > 5, report
    # Invalid indices must still fail, including signed negatives and unsigned
    # maxima in a different slice from the first output element.
    invalid = packed(-1 if index_dtype == 4 else (1 << 32) - 1, index_dtype)
    offset = len(index_bytes) - len(invalid)
    checked(driver.cuMemcpyHtoD_v2(c.c_uint64(a.inputs[1] + offset), invalid, c.c_size_t(len(invalid))))
    launch("typed", "et_index", a)
    assert struct.unpack("<I", read(a.scratch[3], 4))[0] == 1
    if timing:
        print(json.dumps(report), flush=True)
    for ptr in allocations[begin:]:
        checked(driver.cuMemFree_v2(ptr))
    del allocations[begin:]


cases = 0
for dtype in range(7):
    for axis in range(3):
        for index_dtype in [4, 5]:
            source = [2, 3, 5]
            source[axis] += 2
            scatter_case([2, 3, 5], source, axis, dtype, index_dtype)
            cases += 1
    scatter_case([2, 3, 5], [1, 4, 5], 1, dtype, 5)
    cases += 1
print(f"PASSED {cases} scatter dtype/axis/duplicate/partial-shape/invalid-index cases", flush=True)
if "--index-only" in sys.argv:
    scatter_case([64, 8, 2816], [64, 8, 2816], 1, 3, 5, duplicates=False, timing=True)
    scatter_case([64, 8, 2817], [64, 8, 2817], 1, 3, 5, duplicates=True, timing=True)
    print("PASSED 2 full-geometry scatter cases", flush=True)
