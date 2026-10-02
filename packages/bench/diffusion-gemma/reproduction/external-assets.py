#!/usr/bin/env python3
"""Checksum and restore the separately retained dependency archive; audit restored guard pins."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(archive, record):
    expected = json.loads(record.read_text())
    if archive.stat().st_size != expected['sizeBytes'] or sha(archive) != expected['sha256']:
        raise ValueError('External archive size or SHA256 differs from the reviewed backup record')
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ['verify-archive', 'restore']:
        item = commands.add_parser(command)
        item.add_argument('--archive', type=Path, required=True)
        item.add_argument('--record', type=Path, default=HERE / 'external-backup.json')
        if command == 'restore':
            item.add_argument('--destination', type=Path, required=True,
                              help='Fresh staging root; archive root/... members are placed beneath this')
    audit = commands.add_parser('audit-root')
    audit.add_argument('--root-prefix', type=Path, required=True, help='Staging directory, or / for installed paths')
    args = parser.parse_args()
    if args.command in ['verify-archive', 'restore']:
        expected = verify_archive(args.archive, args.record)
        if args.command == 'restore':
            if args.destination.exists():
                raise ValueError('Restore destination must not exist; overwrite is prohibited')
            args.destination.mkdir(parents=True)
            # Only an archive matching the separately reviewed checksum reaches extraction.
            # GNU tar rejects absolute and parent traversal member names by default.
            subprocess.run(['tar', '--zstd', '--extract', '--ignore-zeros', '--file', str(args.archive.resolve()),
                            '--directory', str(args.destination.resolve()), '--no-same-owner',
                            '--keep-old-files'], check=True)
        print(json.dumps({'status': 'verified' if args.command == 'verify-archive' else 'restored',
                          'sha256': expected['sha256'], 'sizeBytes': expected['sizeBytes']}))
        return
    plan = json.loads((HERE / 'external-assets-plan.json').read_text())
    def mapped(remote):
        path = Path(remote)
        if not path.is_absolute() or '..' in path.parts:
            raise ValueError(f'Invalid remote path: {remote}')
        return args.root_prefix / path.relative_to('/')
    failures = []
    for remote, expected in plan['guardedFiles'].items():
        path = mapped(remote)
        if not path.is_file() or sha(path) != expected:
            failures.append({'path': remote, 'reason': 'missing or hash mismatch'})
    for item in plan['requiredDirectories']:
        if not mapped(item['path']).is_dir():
            failures.append({'path': item['path'], 'reason': 'required directory absent'})
    for item in plan['additionalRequiredFiles']:
        if not mapped(item['path']).is_file():
            failures.append({'path': item['path'], 'reason': 'required dependency absent'})
    result = {'status': 'failed' if failures else 'guard-pins-passed',
              'guardedFiles': len(plan['guardedFiles']), 'failures': failures,
              'scope': 'Guard hashes and mandatory paths only; full backup inventory, bank payloads, model download and ELF dependency closure require their separate verification.'}
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
