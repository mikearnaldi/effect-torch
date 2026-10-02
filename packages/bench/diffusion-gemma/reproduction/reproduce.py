#!/usr/bin/env python3
"""Verify preserved measurements or unpack the immutable reproduction sources.

No network, pod lifecycle, GPU execution, overwrite, or credential handling.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def extract(archive, destination):
    if destination.exists():
        raise ValueError(f'Create-only destination already exists: {destination}')
    destination.mkdir(parents=True)
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            name = Path(member.name)
            if name.is_absolute() or '..' in name.parts or not (member.isfile() or member.isdir()):
                raise ValueError(f'Unsafe archive member: {member.name}')
        source.extractall(destination, filter='data')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    verify = commands.add_parser('verify', help='CPU-only recomputation of accepted contracts and medians')
    verify.add_argument('--output', type=Path, required=True, help='Fresh receipt path')
    unpack = commands.add_parser('unpack', help='Unpack sources/entry/evidence into a fresh staging directory')
    unpack.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((HERE / 'manifest.json').read_text())
    for name, expected in manifest['archiveSha256'].items():
        if sha(HERE / 'archives' / name) != expected:
            raise ValueError(f'Archive hash mismatch: {name}')
    if args.command == 'unpack':
        if args.destination.exists():
            raise ValueError('Staging destination must not exist')
        args.destination.mkdir(parents=True)
        for name, folder in [
            ('combined97-100-101-release-runner-v1.tar.gz', 'combined97-100-101-release-build-v1'),
            ('combined97-100-101-release-entry-v1.tar.gz', 'combined97-100-101-release-entry-v1'),
            ('accepted-release-norm98-evidence.tar.gz', 'accepted-evidence'),
        ]:
            extract(HERE / 'archives' / name, args.destination / folder)
        source = args.destination / 'combined97-100-101-release-build-v1' / 'source-v1.tar.gz'
        if sha(source) != manifest['sourceArchiveSha256']:
            raise ValueError('Nested immutable source archive hash mismatch')
        print(json.dumps({'status': 'unpacked', 'destination': str(args.destination.resolve()),
                          'sourceArchiveSha256': sha(source), 'externalDataRequired': True}))
        return
    if args.output.exists():
        raise ValueError('Validation output must not exist')
    with tempfile.TemporaryDirectory(prefix='effect-torch-accepted-') as temporary:
        evidence = Path(temporary) / 'evidence'
        extract(HERE / 'archives' / 'accepted-release-norm98-evidence.tar.gz', evidence)
        accepted = evidence / 'accepted-norm98'
        subprocess.run([
            sys.executable, str(evidence / 'validate-measured-goal.py'),
            '--quality', str(accepted / 'natural'),
            '--timing32', str(accepted / 'timing32'),
            '--timing128', str(accepted / 'timing128'),
            '--hardware', str(evidence / 'build' / 'assessment.json'),
            '--norm98-hardware', str(accepted / 'hardware-norm98' / 'assessment.json'),
            '--build-receipt', str(evidence / 'build' / 'build-receipt.json'),
            '--comparator', str(REPO / 'packages/bench/diffusion-gemma/compare-generation.py'),
            '--output', str(args.output.resolve()),
        ], check=True)


if __name__ == '__main__':
    main()
