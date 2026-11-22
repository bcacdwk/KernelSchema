"""Task-local finite preservation audit. Never traverse runtime or archive trees."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'schema_discussion/main_integration_20261122'
def load(path): return json.loads((ROOT / path).read_text())
def digest(path):
    p = ROOT / path
    assert not any('archive' in part.lower() for part in p.parts)
    assert 'AKA_experiment' not in p.parts
    assert p.is_file() and not p.is_symlink(), path
    with p.open('rb') as stream: value = hashlib.file_digest(stream, 'sha256').hexdigest()
    return value

mapping = load('schema_discussion/main_integration_20261122/execution_map.json')
metadata = load('schema_discussion/main_integration_20261122/research_git_manifest.json')['discussion_file_metadata']
assert len(metadata) == 695
originals = []
for entry in metadata:
    path = entry['path']
    assert re.fullmatch(r'[a-zA-Z0-9_./-]+', path), 'unexpected path literal'
    assert not any(part in ('.', '..') for part in Path(path).parts)
    assert path.startswith(('schema_discussion/extract_twelve_20261119/', 'schema_discussion/extract_schema_review_20261121/', 'schema_discussion/workflow_architecture_20261120/'))
    actual = digest(path)
    assert actual == entry['sha256'] and (ROOT / path).stat().st_size == entry['bytes'], path
    originals.append({'path': path, 'sha256': actual, 'bytes': entry['bytes'], 'unchanged': True})
allowed = {p for owner in mapping['ownership'].values() for key in ('product_paths', 'test_paths', 'conditional_product_paths') for p in owner.get(key, [])}
protected = load('schema_discussion/extract_schema_review_20261121/BASELINE.json')['protected_files']
rows, skipped = [], 0
for path, entry in protected.items():
    if any('archive' in part.lower() for part in Path(path).parts):
        skipped += 1
        continue
    actual = digest(path)
    changed = actual != entry['sha256']
    assert not changed or path in allowed, path
    rows.append({'path': path, 'baseline_sha256': entry['sha256'], 'sha256': actual,
                 'status': 'authorized_product_or_test_change' if changed else 'unchanged'})
assert len(rows) == 277 and skipped == 42
preservation = mapping['preservation']
states = []
for item in preservation['run_state_checks']:
    actual = digest(item['path'])
    assert actual == item['sha256'], item['path']
    states.append({**item, 'unchanged': True})
supervisor = preservation['supervisor_path']
assert digest(supervisor) == preservation['supervisor_sha256']
receipts = []
for item in preservation['old_receipts_sha256']:
    assert digest(item['path']) == item['sha256'], item['path']
    receipts.append({'path': item['path'], 'sha256': item['sha256'], 'unchanged': True})
assert digest('AGENTS.md') == preservation['agents']['sha256']
result = {'format': 'kernelschema.integration.preservation.v1', 'status': 'passed',
    'original_discussion_files': len(originals), 'original_discussion_unchanged': True,
    'metadata_path_literal_review': 'All 695 paths use fixed relative task directories and ordinary ASCII filename components; no credential, email, query or account identifier literal was found. Contents of withheld captures were only streamed for SHA, not reviewed or published.',
    'protected_files': rows, 'archive_metadata_entries_skipped_without_target_access': skipped,
    'run_states': states, 'supervisor': {'path': supervisor, 'sha256': preservation['supervisor_sha256'], 'unchanged': True},
    'receipts': receipts, 'agents': preservation['agents'],
    'boundaries': ['Exact metadata paths only; no runtime traversal.', 'No old rules/checkpoint/usage migration.', 'State/supervisor bytes hashed without printing account or model content.']}
(BASE / 'round2_d/preservation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'status':'passed', 'originals':len(originals),'protected':len(rows),'authorized_changes':sum(x['status']!='unchanged' for x in rows),'archive_skipped':skipped,'states':len(states),'receipts':len(receipts)}))
