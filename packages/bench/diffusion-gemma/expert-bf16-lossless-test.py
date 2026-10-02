"""Bit-level proof of the standalone CUDA prototype's grouped12-bit layout."""
import hashlib
import json
from pathlib import Path
import struct
import sys


def encode(words):
    assert len(words)%64==0
    headers=[];payload=bytearray()
    for offset in range(0,len(words),64):
        block=words[offset:offset+64]
        exponents=[(word>>7)&255 for word in block]
        base=min(exponents);raw=max(exponents)-base>15
        assert len(payload)%16==0 and len(payload)//4<=0x7fffff
        headers.append((len(payload)//4)<<9|int(raw)<<8|base)
        if raw:
            payload.extend(struct.pack('<64H',*block))
        else:
            for group in range(0,64,8):
                encoded=bytearray(12)
                for index,word in enumerate(block[group:group+8]):
                    encoded[index]=(word&127)|((word>>8)&128)
                    encoded[8+index//2]|=(((word>>7)&255)-base)<<((index%2)*4)
                payload.extend(encoded)
    return headers,payload


def decode(headers,payload,count):
    result=[]
    for index in range(0,count,8):
        header=headers[index//64];offset=(header>>9)*4
        if header&256:
            result.extend(struct.unpack_from('<8H',payload,offset+(index%64)*2))
            continue
        sm0,sm1,exponents=struct.unpack_from('<3I',payload,offset+(index%64)//8*12)
        base=header&255
        for pair in range(4):
            signs_mantissas=(sm0 if pair<2 else sm1)>>((pair%2)*16)
            deltas=exponents>>(pair*8)
            a=(signs_mantissas&127)|((signs_mantissas&128)<<8)|((base+(deltas&15))<<7)
            b=((signs_mantissas>>8)&127)|(signs_mantissas&32768)|((base+((deltas>>4)&15))<<7)
            result.extend([a,b])
    return result


def fixed_roundtrip(words):
    original_headers,original_payload=encode(words)
    fixed=bytearray();tails=bytearray();headers=[]
    for header in original_headers:
        offset=(header>>9)*4
        headers.append((len(tails)//4)<<9|(header&511))
        if header&256:
            for group in range(8):
                fixed.extend(original_payload[offset+group*16:offset+group*16+12])
                tails.extend(original_payload[offset+group*16+12:offset+group*16+16])
        else:
            fixed.extend(original_payload[offset:offset+96])
    result=[]
    for block,header in enumerate(headers):
        if header&256:
            for group in range(8):
                prefix=fixed[block*96+group*12:block*96+group*12+12]
                tail_offset=(header>>9)*4+group*4
                result.extend(struct.unpack('<8H',prefix+tails[tail_offset:tail_offset+4]))
        else:
            result.extend(decode([header&255],fixed[block*96:block*96+96],64))
    assert result==words
    assert len(fixed)+len(tails)==len(original_payload)


def main():
    patterns=list(range(65536))
    assert decode(*encode(patterns),len(patterns))==patterns
    fixed_roundtrip(patterns)
    # Full raw fallback and alternating compressed/raw blocks at changing offsets.
    scrambled=[(index*4051+8191)%65536 for index in range(65536)]
    assert all(header&256 for header in encode(scrambled)[0])
    assert decode(*encode(scrambled),len(scrambled))==scrambled
    fixed_roundtrip(scrambled)
    mixed=[]
    for offset in range(0,65536,64):
        mixed.extend((patterns if offset%128==0 else scrambled)[offset:offset+64])
    assert decode(*encode(mixed),len(mixed))==mixed
    fixed_roundtrip(mixed)
    directory=Path(sys.argv[1]);manifest=json.loads((directory/'manifest.json').read_text())
    raw=(directory/'samples.bf16').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==manifest['sha256']
    words=list(struct.unpack('<'+'H'*(len(raw)//2),raw))
    headers,payload=encode(words)
    assert decode(headers,payload,len(words))==words
    fixed_roundtrip(words)
    result=dict(exhaustivePatterns=65536,rawPatterns=65536,mixedPatterns=65536,
                sampledWords=len(words),encodedBytes=len(payload)+len(headers)*4,
                rawBlocks=sum(bool(header&256) for header in headers),exact=True,fixedSlotsExact=True)
    print(json.dumps(result))


if __name__=='__main__':
    main()
