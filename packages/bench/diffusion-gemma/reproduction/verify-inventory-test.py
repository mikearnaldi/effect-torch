"""Small CPU fixtures for the streaming backup verifier; no remote archive reads."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

spec = importlib.util.spec_from_file_location('inventory_verifier', Path(__file__).with_name('verify-inventory.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(root, members, inventory):
    archive = root / 'fixture.tar'
    with tarfile.open(archive, 'w') as target:
        for name, kind, data in members:
            info = tarfile.TarInfo(name)
            if kind == 'file':
                info.size = len(data)
                target.addfile(info, io.BytesIO(data))
            else:
                info.type = {'directory': tarfile.DIRTYPE, 'symlink': tarfile.SYMTYPE, 'hardlink': tarfile.LNKTYPE}[kind]
                if kind != 'directory':
                    info.linkname = data
                target.addfile(info)
    compressed = root / 'fixture.tar.zst'
    subprocess.run(['zstd', '--quiet', '--force', str(archive), '-o', str(compressed)], check=True)
    listing = root / 'inventory.jsonl'
    listing.write_text(''.join(json.dumps(row) + '\n' for row in inventory))
    return module.verify(compressed, listing)


def main():
    data = b'fixed accepted bytes'
    digest = hashlib.sha256(data).hexdigest()
    members = [('root', 'directory', None), ('root/data', 'file', data),
               ('root/link', 'hardlink', 'root/data'), ('root/symlink', 'symlink', '/root/data')]
    inventory = [{'relativePath': 'root', 'type': 'directory'},
                 {'relativePath': 'root/data', 'sizeBytes': len(data), 'sha256': digest},
                 {'relativePath': 'root/link', 'sizeBytes': len(data), 'sha256': digest},
                 {'relativePath': 'root/symlink', 'type': 'symlink', 'linkTarget': '/root/data'}]
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        result = fixture(root, members, inventory)
        assert result['status'] == 'passed' and result['verifiedFiles'] == 2
        assert result['payloadBytes'] == len(data) and result['logicalFileBytes'] == 2 * len(data)
        cases = [
            (members, [{**r, 'sha256': '0' * 64} if r['relativePath'] == 'root/data' else r for r in inventory]),
            (members[:-1], inventory),
            (members + [('root/extra', 'file', b'extra')], inventory),
            (members[:-1] + [('root/symlink', 'symlink', '/root/wrong')], inventory),
            ([members[0], members[2], members[1], members[3]], inventory),
            (members + [members[1]], inventory),
            (members[:-1] + [('root/symlink', 'directory', None)], inventory),
            (members, [{**r, 'sizeBytes': 99} if r['relativePath'] == 'root/link' else r for r in inventory]),
        ]
        for index, (test_members, test_inventory) in enumerate(cases):
            result = fixture(root, test_members, test_inventory)
            assert result['status'] == 'failed', index
        duplicate_inventory = inventory + [inventory[0]]
        try:
            fixture(root, members, duplicate_inventory)
        except ValueError:
            pass
        else:
            raise AssertionError('duplicate inventory accepted')
    print(json.dumps({'status': 'passed', 'positiveCases': 1, 'negativeCases': 9,
                      'coverage': 'regular/directory/symlink/hardlink, missing/extra/hash/size/type/duplicate/forward-link rejection'}))


if __name__ == '__main__':
    main()
