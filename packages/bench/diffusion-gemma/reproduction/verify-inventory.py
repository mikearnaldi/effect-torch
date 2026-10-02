#!/usr/bin/env python3
"""Stream a tar.zst archive, verifying its complete inventory without extraction."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


def canonical(name):
    if name.startswith('/') or '..' in PurePosixPath(name).parts:
        raise ValueError(f'Unsafe member path: {name}')
    result = str(PurePosixPath(name))
    if result in ['', '.']:
        raise ValueError(f'Empty member path: {name}')
    return result


def verify(archive, inventory_path):
    expected = {}
    errors = []
    with inventory_path.open() as inventory:
        for number, line in enumerate(inventory, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            name = canonical(item['relativePath'])
            if name in expected:
                raise ValueError(f'Duplicate inventory path at line {number}: {name}')
            kind = item.get('type', 'file')
            if kind not in ['file', 'regular', 'directory', 'symlink']:
                raise ValueError(f'Unsupported inventory type: {kind}')
            expected[name] = item
    seen = set()
    verified_files = {}
    counts = {'regularFiles': 0, 'hardlinks': 0, 'symlinks': 0, 'directories': 0}
    payload_bytes = 0
    logical_bytes = 0
    process = subprocess.Popen(['zstd', '--decompress', '--stdout', '--quiet', str(archive)],
                               stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|', ignore_zeros=True) as stream:
            for member in stream:
                name = canonical(member.name)
                if name in seen:
                    errors.append({'path': name, 'error': 'duplicate archive member'})
                seen.add(name)
                record = expected.get(name)
                if record is None:
                    errors.append({'path': name, 'error': 'extra archive member'})
                expected_type = None if record is None else record.get('type', 'file')
                actual = None
                if member.isfile():
                    counts['regularFiles'] += 1
                    digest = hashlib.sha256()
                    count = 0
                    source = stream.extractfile(member)
                    if source is None:
                        raise ValueError(f'Cannot read regular member: {name}')
                    with source:
                        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                            digest.update(chunk)
                            count += len(chunk)
                    payload_bytes += count
                    actual = (count, digest.hexdigest())
                    if count != member.size:
                        errors.append({'path': name, 'error': 'tar payload size mismatch'})
                elif member.islnk():
                    counts['hardlinks'] += 1
                    target = canonical(member.linkname)
                    actual = verified_files.get(target)
                    if actual is None:
                        errors.append({'path': name, 'error': 'hardlink target was not previously verified',
                                       'target': target})
                elif member.issym():
                    counts['symlinks'] += 1
                    if record is not None and (expected_type != 'symlink' or record.get('linkTarget') != member.linkname):
                        errors.append({'path': name, 'error': 'symlink type or target mismatch'})
                elif member.isdir():
                    counts['directories'] += 1
                    if record is not None and expected_type != 'directory':
                        errors.append({'path': name, 'error': 'directory type mismatch'})
                else:
                    errors.append({'path': name, 'error': 'unsupported archive member type'})
                if actual is not None:
                    logical_bytes += actual[0]
                    if record is not None and expected_type in ['file', 'regular'] and actual == (record.get('sizeBytes'), record.get('sha256')):
                        verified_files[name] = actual
                    else:
                        errors.append({'path': name, 'error': 'regular file type, size or SHA256 mismatch',
                                       'actualSizeBytes': actual[0], 'actualSha256': actual[1]})
        # Consume the compressed stream through EOF so zstd verifies its checksum
        # and cannot remain blocked on tar's trailing padding.
        while process.stdout.read(8 * 1024 * 1024):
            pass
        code = process.wait()
        if code != 0:
            errors.append({'error': 'zstd decompression failed', 'exitCode': code})
    except Exception as error:
        errors.append({'error': 'stream verification failed', 'detail': str(error)})
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    missing = sorted(set(expected) - seen)
    errors.extend({'path': name, 'error': 'missing archive member'} for name in missing)
    return {'status': 'passed' if not errors else 'failed', 'archive': str(archive),
            'inventory': str(inventory_path), 'inventoryEntries': len(expected),
            'archiveEntries': len(seen), 'counts': counts, 'payloadBytes': payload_bytes,
            'logicalFileBytes': logical_bytes, 'verifiedFiles': len(verified_files),
            'errors': errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Verification report must not exist')
    try:
        report = verify(args.archive, args.inventory)
    except Exception as error:
        report = {'status': 'failed', 'archive': str(args.archive),
                  'inventory': str(args.inventory), 'errors': [{'error': str(error)}]}
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'errors'}))
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
