"""Single-paragraph Sparse cleanup authorized after independent verification."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[4]
CASE = Path('schema_discussion/schema_audit_and_cleanup_20261119')
CLEAN = CASE / 'cleanup'
FOLLOWUP = CLEAN / 'sparse_followup'
DIRECTORY = Path('schema_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16')


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def write(p, value):
    (ROOT / p).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


assert not (ROOT / FOLLOWUP / 'before.json').exists()
files = [DIRECTORY / 'dispatch.md', DIRECTORY / '.extraction/current.json', CLEAN / 'directory_checks.json',
         CASE / 'CURRENT_SCHEMA_INDEX.json', CLEAN / 'verification.json', CLEAN / 'REPORT.md']
preserved = []
for p in files:
    dst = FOLLOWUP / 'before' / p
    (ROOT / dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / p, ROOT / dst)
    assert sha(p) == sha(dst)
    preserved.append({'path': str(p), 'backup': str(dst), 'sha256': sha(p)})
unchanged = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'schema_wiki').rglob('*.yaml')}
unchanged.update({str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'schema_wiki').rglob('dispatch.md') if p.parent != ROOT / DIRECTORY})
unchanged.update({str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'schema_wiki').rglob('current.json') if p.parent.parent != ROOT / DIRECTORY})
write(FOLLOWUP / 'before.json', {'preserved': preserved, 'must_remain_unchanged': unchanged})
p = ROOT / DIRECTORY / 'dispatch.md'
text = p.read_text()
old = '认证与 GPU 证据只覆盖 L20D 上的 fixed 十用例与 domain 十采样点；源码注释中的 dv_* job 与百分比是冻结来源自带的设计依据声明，本目录据实转述为取舍理由，本 run 未运行 GPU、未复测。外部 comparator 定义缺失沿用既有 source_gap 记录，保留在 root.yaml 的'
new = '现有认证与 GPU 来源证据覆盖 L20D 上的 fixed 十用例与 domain 十采样点；源码注释中的 dv_* job 与百分比是固定来源的设计依据声明，本目录据此说明取舍理由。这些样本与来源声明不能证明四份设计及其完整适用域均已独立验证。来源材料未提供完整的外部 comparator 定义，已知阈值保留在 root.yaml 的'
assert text.count(old) == 1
before_sha = sha(DIRECTORY / 'dispatch.md')
p.write_text(text.replace(old, new))
write(FOLLOWUP / 'changes.json', {
    'scope': 'One paragraph of Sparse dispatch; current evidence coverage and comparator gap only.',
    'path': str(DIRECTORY / 'dispatch.md'), 'before_sha256': before_sha,
    'after_sha256': sha(DIRECTORY / 'dispatch.md'), 'replacements': [{'from': old, 'to': new}],
    'previous_editorial_chain': str(CLEAN / 'content_changes.json'),
    'before_record': {'path': str(FOLLOWUP / 'before.json'), 'sha256': sha(FOLLOWUP / 'before.json')},
    'yaml_and_other_formal_text_changed': False,
})
print(before_sha, '->', sha(DIRECTORY / 'dispatch.md'))
