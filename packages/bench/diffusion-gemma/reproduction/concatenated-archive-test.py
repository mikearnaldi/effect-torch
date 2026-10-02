"""Verify and restore concatenated zstd/tar frames using tiny CPU fixtures."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        inventory = []
        frames = []
        expected = {'root/baseline.txt': b'primary baseline',
                    'nix/store/fixture/node_modules/required.txt': b'supplemental runtime bytes'}
        for index, (name, data) in enumerate(expected.items()):
            archive = root / f'part{index}.tar'
            with tarfile.open(archive, 'w') as output:
                member = tarfile.TarInfo(name)
                member.size = len(data)
                output.addfile(member, io.BytesIO(data))
            frame = root / f'part{index}.tar.zst'
            subprocess.run(['zstd', '--quiet', str(archive), '-o', str(frame)], check=True)
            frames.append(frame.read_bytes())
            inventory.append({'relativePath': name, 'sizeBytes': len(data),
                              'sha256': hashlib.sha256(data).hexdigest()})
        combined = root / 'combined.tar.zst'
        combined.write_bytes(b''.join(frames))
        listing = root / 'inventory.jsonl'
        listing.write_text(''.join(json.dumps(row) + '\n' for row in inventory))
        report = root / 'verification.json'
        subprocess.run([sys.executable, str(HERE / 'verify-inventory.py'), '--archive', str(combined),
                        '--inventory', str(listing), '--output', str(report)], check=True,
                       stdout=subprocess.DEVNULL)
        result = json.loads(report.read_text())
        assert result['status'] == 'passed' and result['verifiedFiles'] == 2
        record = root / 'record.json'
        record.write_text(json.dumps({'sizeBytes': combined.stat().st_size,
                                      'sha256': hashlib.sha256(combined.read_bytes()).hexdigest()}))
        restored = root / 'restored'
        subprocess.run([sys.executable, str(HERE / 'external-assets.py'), 'restore',
                        '--archive', str(combined), '--record', str(record),
                        '--destination', str(restored)], check=True, stdout=subprocess.DEVNULL)
        assert all((restored / name).read_bytes() == data for name, data in expected.items())
    print(json.dumps({'status': 'passed', 'zstdFrames': 2, 'tarStreams': 2,
                      'verifiedFiles': 2, 'restoredFiles': 2}))


if __name__ == '__main__':
    main()
