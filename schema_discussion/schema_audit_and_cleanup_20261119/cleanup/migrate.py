"""One-time archival relocation for this cleanup; never touches runtime state."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
CASE = Path('schema_discussion/schema_audit_and_cleanup_20261119')
CLEAN = CASE / 'cleanup'
ARCHIVE = Path('schema_discussion/archive')


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def write_json(path, value):
    (ROOT / path).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    assert not (ROOT / CLEAN / 'LOCATIONS.json').exists(), 'Already migrated'
    operators = sorted((ROOT / 'schema_wiki/aka-cuda-operators-20261030/operators').rglob('root.yaml'))
    current_files = [p for r in operators for p in r.parent.rglob('*') if p.is_file()]
    issue_files = sorted((ROOT / 'schema_extract/issues').rglob('*.json'))
    edits = [Path(p) for p in ['README.md', 'schema_wiki/README.md', 'schema_extract/README.md',
                              'schema_extract/issues/README.md', 'schema_extract/issues/archive/README.md']]
    edits += [r.parent.relative_to(ROOT) / 'dispatch.md' for r in operators]
    baseline = {
        'current_files': {str(p.relative_to(ROOT)): sha(p) for p in current_files},
        'issue_files': {str(p.relative_to(ROOT)): sha(p) for p in issue_files},
        'issue_decisions': {json.loads(p.read_text())['id']: json.loads(p.read_text())['decision']['outcome'] for p in issue_files},
        'navigation_and_dispatch': {str(p): sha(p) for p in edits},
    }
    if not (ROOT / CLEAN / 'BEFORE.json').exists():
        write_json(CLEAN / 'BEFORE.json', baseline)
        (ROOT / CLEAN / 'git-status-before.txt').write_text(subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True))
    operations = []
    for src in edits:
        dst = CLEAN / 'before' / src
        (ROOT / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / src, ROOT / dst)
        operations.append({'operation': 'preserve_before_edit', 'from': str(src), 'to': str(dst), 'sha256': sha(src)})
    moves = [
        (Path('schema_wiki/archive'), ARCHIVE / 'schema_wiki'),
        (Path('schema_extract/archive'), ARCHIVE / 'schema_extract'),
        (Path('schema_docs/archived'), ARCHIVE / 'schema_docs'),
    ]
    for name in ['TRIAGE.md', 'DESIGN_REGION_DERIVATION.md']:
        moves.append((Path('schema_extract/issues/archive') / name, ARCHIVE / 'issue_discussions' / name))
    for r in operators:
        for p in sorted((r.parent / '.extraction').iterdir()):
            assert p.is_file() and p.suffix == '.json', p
            moves.append((p.relative_to(ROOT), ARCHIVE / 'schema_wiki/extraction_records' / r.parent.name / p.name))
    directory_moves = {}
    for src, dst in moves:
        if not (ROOT / src).exists() and src.name in ['TRIAGE.md', 'DESIGN_REGION_DERIVATION.md']:
            # Both were already deleted before cleanup, despite live README links.
            original = subprocess.check_output(['git', 'show', f'HEAD:{src}'], cwd=ROOT)
            (ROOT / dst).parent.mkdir(parents=True, exist_ok=True)
            (ROOT / dst).write_bytes(original)
            operations.append({'operation': 'archive_git_head_preexisting_deletion', 'from': str(src), 'to': str(dst), 'sha256': sha(dst), 'bytes': len(original), 'git_ref': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
            continue
        if not (ROOT / src).exists() and (ROOT / dst).is_dir():
            # Resume the three directory moves completed before missing issue docs were found.
            directory_moves[str(src)] = str(dst)
            for p in sorted((ROOT / dst).rglob('*')):
                if p.is_file():
                    before, after = src / p.relative_to(ROOT / dst), p.relative_to(ROOT)
                    operations.append({'operation': 'move', 'from': str(before), 'to': str(after), 'sha256': sha(after), 'bytes': p.stat().st_size})
            continue
        assert not (ROOT / dst).exists(), dst
        if (ROOT / src).is_dir():
            directory_moves[str(src)] = str(dst)
            payload = [(p.relative_to(ROOT), dst / p.relative_to(ROOT / src)) for p in sorted((ROOT / src).rglob('*')) if p.is_file()]
        else:
            payload = [(src, dst)]
        for before, after in payload:
            operations.append({'operation': 'move', 'from': str(before), 'to': str(after), 'sha256': sha(before), 'bytes': (ROOT / before).stat().st_size})
        (ROOT / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(ROOT / src, ROOT / dst)
    for record in operations:
        assert sha(Path(record['to'])) == record['sha256'], record
        if record['operation'] == 'move':
            assert not (ROOT / record['from']).exists(), record
    write_json(CLEAN / 'LOCATIONS.json', {
        'format': 'kernelschema.formal_history_locations.v1',
        'path_base': 'repository_root',
        'historical_text_policy': 'Archived bytes, historical paths, hashes and approvals retain their original scope. Resolve old paths with this map; do not treat relocated receipts as current approval.',
        'directory_moves': directory_moves,
        'operations': operations,
        'deletions': [],
        'deduplication': 'None: every historical file is preserved byte-for-byte; no evidence was deleted.',
    })
    print(json.dumps({'moves': sum(r['operation'] == 'move' for r in operations), 'before_copies': len(edits), 'bytes_moved': sum(r.get('bytes', 0) for r in operations)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
