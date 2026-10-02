"""CPU-only lossless BF16 exponent packing feasibility; never quantizes weights."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import struct


def encode_block(words, exponent_bits):
    exponents = [(word >> 7) & 255 for word in words]
    base = min(exponents)
    if max(exponents) - base >= 1 << exponent_bits:
        return None, struct.pack('<' + 'H' * len(words), *words)
    bits = 8 + exponent_bits
    packed = 0
    for index, word in enumerate(words):
        # Preserve sign and all seven mantissa bits, including NaN payloads.
        code = ((word >> 8) & 128) | (word & 127) | ((exponents[index] - base) << 8)
        packed |= code << (index * bits)
    return base, packed.to_bytes((len(words) * bits + 7) // 8, 'little')


def decode_block(base, payload, count, exponent_bits):
    if base is None:
        return list(struct.unpack('<' + 'H' * count, payload))
    bits = 8 + exponent_bits
    packed = int.from_bytes(payload, 'little')
    words = []
    for index in range(count):
        code = (packed >> (index * bits)) & ((1 << bits) - 1)
        words.append(((code & 128) << 8) | (code & 127) | ((base + (code >> 8)) << 7))
    return words


def analyze(chunks, block_words, exponent_bits):
    raw_blocks = compressed_blocks = total_bytes = aligned_bytes = words_count = 0
    histogram = Counter()
    for words in chunks:
        assert len(words) % block_words == 0
        for offset in range(0, len(words), block_words):
            block = words[offset:offset + block_words]
            base, payload = encode_block(block, exponent_bits)
            assert decode_block(base, payload, len(block), exponent_bits) == block
            raw_blocks += base is None
            compressed_blocks += base is not None
            # Four-byte block metadata: base8, raw flag1, payload word-offset23.
            # This bounds each independent expert payload to32MiB.
            total_bytes += 4 + len(payload)
            aligned_bytes += 4 + (len(payload) + 15) // 16 * 16
            words_count += len(block)
            exponents = [(value >> 7) & 255 for value in block]
            histogram[max(exponents) - min(exponents)] += 1
    return dict(blockWords=block_words, exponentBits=exponent_bits,
                rawBlocks=raw_blocks, compressedBlocks=compressed_blocks,
                rawBlockFraction=raw_blocks / (raw_blocks + compressed_blocks),
                bitsPerWeight=total_bytes * 8 / words_count,
                aligned16BitsPerWeight=aligned_bytes * 8 / words_count,
                byteReduction=1 - total_bytes / (words_count * 2),
                aligned16ByteReduction=1 - aligned_bytes / (words_count * 2),
                exponentRangeHistogram=dict(sorted(histogram.items())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('samples', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.samples / 'manifest.json').read_text())
    raw = (args.samples / 'samples.bf16').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest['sha256']
    # Every possible BF16 pattern, including signed zeros, subnormals, infinities,
    # and all NaN payloads, round-trips as integer bits through both layouts.
    all_patterns = list(range(65536))
    for width in (3, 4, 5):
        for count in (32, 64, 128, 256):
            for offset in range(0, len(all_patterns), count):
                words = all_patterns[offset:offset + count]
                base, payload = encode_block(words, width)
                assert decode_block(base, payload, count, width) == words
            words = [(index * 4051 + 8191) % 65536 for index in range(count)]
            base, payload = encode_block(words, width)
            assert base is None and decode_block(base, payload, count, width) == words
    chunks = []
    grouped = defaultdict(list)
    exponent_histogram = Counter()
    zeros = nonfinite = 0
    for record in manifest['records']:
        offset, count = record['sampleByteOffset'], record['sampleWords']
        payload = raw[offset:offset + count * 2]
        assert hashlib.sha256(payload).hexdigest() == record['sha256']
        words = list(struct.unpack('<' + 'H' * count, payload))
        chunks.append(words)
        grouped[record['tensor']].append(words)
        exponent_histogram.update((word >> 7) & 255 for word in words)
        zeros += sum((word & 32767) == 0 for word in words)
        nonfinite += sum(((word >> 7) & 255) == 255 for word in words)
    result = dict(sampleManifestSha256=hashlib.sha256((args.samples/'manifest.json').read_bytes()).hexdigest(),
                  sampledWords=len(raw)//2, sampledTensors=len(grouped), zeros=zeros,
                  nonfinite=nonfinite, exponentHistogram=dict(sorted(exponent_histogram.items())),
                  exactRoundtrip=True, exhaustiveBF16PatternRoundtrip=True,
                  candidates=[analyze(chunks, count, width) for count in (32,64,128,256) for width in (3,4,5)],
                  perTensor={name: [analyze(values, count, 4) for count in (64,128)] for name,values in grouped.items()},
                  limitations=['Sparse deterministic checkpoint sample, not full archive statistics.',
                               'Four-byte block metadata assumes separate payloads of at most32MiB perexpert.',
                               'No GPU decode kernel or runtime integration; savings exclude decode/descriptor/launch costs.'])
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: value for key,value in result.items() if key not in ('perTensor','exponentHistogram','candidates')}))
    for row in result['candidates']:
        print(json.dumps({key: value for key,value in row.items() if key != 'exponentRangeHistogram'}))


if __name__ == '__main__':
    main()
