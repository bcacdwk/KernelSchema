"""Refresh only Sparse and aggregate bindings after the successful targeted check."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[4]
CASE = Path('schema_discussion/schema_audit_and_cleanup_20261119')
CLEAN = CASE / 'cleanup'
FOLLOWUP = CLEAN / 'sparse_followup'
DIRECTORY = Path('schema_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16')


def read(p):
    return json.loads((ROOT / p).read_text())


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def pointer(p):
    return {'path': str(p), 'sha256': sha(p)}


def write(p, value):
    (ROOT / p).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


for path, digest in read(FOLLOWUP / 'before.json')['must_remain_unchanged'].items():
    assert sha(path) == digest, path
assert 'Normalization preview: 4 YAML; 0 would change; no files written' in (ROOT / FOLLOWUP / 'check.log').read_text()
changes = read(FOLLOWUP / 'changes.json')
assert sha(DIRECTORY / 'dispatch.md') == changes['after_sha256']
current_path = DIRECTORY / '.extraction/current.json'
current = read(current_path)
assert current['files']['dispatch.md']['sha256'] == changes['before_sha256']
current['files']['dispatch.md']['sha256'] = changes['after_sha256']
current['acceptance']['dispatch_editorial_followup'] = pointer(FOLLOWUP / 'changes.json')
current['acceptance']['structure_check']['output'] = pointer(FOLLOWUP / 'check.log')
write(current_path, current)
checks = read(CLEAN / 'directory_checks.json')
record = next(x for x in checks if x['operator_id'] == 'dual_cache_latent_bf16')
assert record['manifest']['dispatch.md'] == changes['before_sha256']
record['manifest']['dispatch.md'] = changes['after_sha256']
record['output'] = pointer(FOLLOWUP / 'check.log')
write(CLEAN / 'directory_checks.json', checks)
index_path = CASE / 'CURRENT_SCHEMA_INDEX.json'
index = read(index_path)
item = next(x for x in index['operators'] if x['operator_id'] == 'dual_cache_latent_bf16')
assert item['files']['dispatch.md']['sha256'] == changes['before_sha256']
item['files']['dispatch.md']['sha256'] = changes['after_sha256']
item['current_metadata'] = pointer(current_path)
item['dispatch_editorial_followup'] = pointer(FOLLOWUP / 'changes.json')
index['directory_checks'] = pointer(CLEAN / 'directory_checks.json')
write(index_path, index)
for path, digest in read(FOLLOWUP / 'before.json')['must_remain_unchanged'].items():
    assert sha(path) == digest, path
print(json.dumps({'sparse_dispatch': pointer(DIRECTORY / 'dispatch.md'), 'sparse_current': pointer(current_path),
                  'directory_checks': pointer(CLEAN / 'directory_checks.json'), 'current_index': pointer(index_path)}, ensure_ascii=False))
