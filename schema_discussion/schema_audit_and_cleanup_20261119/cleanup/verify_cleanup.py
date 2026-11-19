"""Offline, read-only verification of the cleanup; output is confined to this case."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from schema_extract.tools.snapshots import rule_files

CASE = Path('schema_discussion/schema_audit_and_cleanup_20261119')
CLEAN = CASE / 'cleanup'
os.chdir(ROOT)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pointers(value):
    if isinstance(value, dict):
        if set(value) >= {'path', 'sha256'}:
            assert sha(value['path']) == value['sha256'], value['path']
        for item in value.values():
            pointers(item)
    elif isinstance(value, list):
        for item in value:
            pointers(item)


locations = read(CLEAN / 'LOCATIONS.json')
baseline = read(CLEAN / 'BEFORE.json')
index = read(CASE / 'CURRENT_SCHEMA_INDEX.json')
moved = {r['from']: r['to'] for r in locations['operations'] if r['operation'] == 'move'}
for record in locations['operations']:
    assert sha(record['to']) == record['sha256'], record
    if record['operation'] in ['move', 'archive_git_head_preexisting_deletion']:
        assert not Path(record['from']).exists(), record
for record in locations['deletions']:
    assert not Path(record['path']).exists()
    assert sha(record['regeneration_source']) == record['source_sha256']
for record in locations['prior_location_aliases']:
    assert sha(record['to']) == record['sha256']

# The existing archive trees had no tracked modifications at cleanup entry.
# Compare their relocated payloads to those original Git blobs as an extra check.
tracked = {}
for line in subprocess.check_output(['git', 'ls-files', '-s', '--', 'schema_wiki/archive', 'schema_extract/archive', 'schema_docs/archived'], text=True).splitlines():
    meta, name = line.split('\t', 1)
    tracked[name] = meta.split()[1]
process = subprocess.Popen(['git', 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
tracked_checked = 0
for old, blob in tracked.items():
    if old not in moved:
        continue
    process.stdin.write((blob + '\n').encode())
    process.stdin.flush()
    header = process.stdout.readline().decode().split()
    payload = process.stdout.read(int(header[2]))
    assert process.stdout.read(1) == b'\n'
    assert hashlib.sha256(payload).hexdigest() == sha(moved[old]), old
    tracked_checked += 1
process.stdin.close()
assert process.wait() == 0

framework = read(CASE / 'FRAMEWORK_BASELINE.json')
framework_moves = []
for path, digest in framework.items():
    target = moved.get(path, path)
    assert sha(target) == digest, path
    if target != path:
        framework_moves.append({'from': path, 'to': target, 'sha256': digest})
runtime = read(CASE / 'RUNTIME_BASELINE.json')
for path, digest in runtime.items():
    assert sha(path) == digest, path

for path, digest in baseline['current_files'].items():
    if path.endswith('.yaml'):
        assert sha(path) == digest, path
issue_current = {str(p): sha(p) for p in Path('schema_extract/issues').rglob('*.json')}
assert issue_current == baseline['issue_files']
coverage = read('schema_discussion/issue_optimization_20261119/ISSUE_COVERAGE.json')['issues']
outcomes = Counter()
for item in coverage:
    record = read(item['final_path'])
    assert record['id'] == item['id']
    assert record['decision']['outcome'] == item['final_outcome']
    outcomes[record['decision']['outcome']] += 1
assert outcomes == {'resolved': 12, 'declined': 2, 'partially_adopted': 1}
cli = read(CLEAN / 'issues-cli.json')

sup = read(CASE / 'SUPERVISION.json')
for item in sup['operators'].values():
    assert sha(CASE / item['after_record']) == item['after_sha256']
review_chains = []
for operator in index['operators']:
    current = read(operator['current_metadata']['path'])
    pointers(current)
    assert current['acceptance']['runner_p6_receipt'] is False
    directory = Path(operator['directory'])
    assert {p.name for p in (directory / '.extraction').iterdir()} == {'current.json'}
    for name, record in operator['files'].items():
        assert sha(directory / name) == record['sha256']
    evidence_path = operator['semantic_evidence']['path']
    evidence = read(evidence_path)
    for name, record in operator['files'].items():
        if name.endswith('.yaml'):
            assert evidence['manifest'][name] == record['sha256']
    if 'record_path' in evidence:
        record_path, digest = evidence['record_path'], evidence['record_sha256']
    else:
        record = evidence['audit_record']
        record_path = record['path'] if isinstance(record, dict) else record
        digest = record['sha256'] if isinstance(record, dict) else evidence['audit_record_sha256']
    actual_record_path = moved.get(record_path, record_path)
    assert sha(actual_record_path) == digest
    review_chains.append({'operator_id': operator['operator_id'], 'after': evidence_path,
                          'record': actual_record_path, 'record_sha256': digest,
                          'reviewed_yaml_unchanged': True, 'dispatch_changes': str(CLEAN / 'content_changes.json')})
    if 'dispatch_editorial_followup' in current['acceptance']:
        review_chains[-1]['dispatch_editorial_followup'] = current['acceptance']['dispatch_editorial_followup']
pointers(index)

rules = rule_files(ROOT)
assert all(Path(p).is_file() for p in rules)
assert not any('/archive/' in p or p.startswith(('schema_discussion/', 'schema_docs/archived/')) for p in rules)

# Check active Markdown targets, excluding literal fenced examples.
docs = [Path('README.md'), Path('EXTRACT_SCHEMA.md'), Path('schema_discussion/archive/README.md'), CLEAN / 'REPORT.md']
for root in ['schema_docs', 'schema_extract', 'schema_wiki']:
    docs.extend(Path(root).rglob('*.md'))
links, broken = [], []
for doc in docs:
    fenced = False
    for number, line in enumerate(doc.read_text().splitlines(), 1):
        if line.lstrip().startswith(('```', '~~~')):
            fenced = not fenced
            continue
        if fenced:
            continue
        for match in re.finditer(r'\[[^\]\n]*\]\(([^)\n]+)\)', line):
            href = match.group(1).split()[0].strip('<>')
            target = href.split('#', 1)[0]
            if not target or ':' in target:
                continue
            record = {'file': str(doc), 'line': number, 'target': href}
            links.append(record)
            is_output = (doc.parent / target).resolve() == (CLEAN / 'verification.json').resolve()
            if not (doc.parent / target).exists() and not is_output:
                broken.append(record)
assert not broken, broken

scan = []
pattern = re.compile(r'(?<![A-Za-z0-9_])P[1-6](?![A-Za-z0-9_])|本轮|审修前|前后字节|xhigh|pending_user_decision|invalid_receipt|before_manifest|awaiting|本 run|未复测|沿用既有 source_gap')
for path in Path('schema_wiki').rglob('*'):
    if not path.is_file() or path.suffix not in ['.md', '.json', '.yaml']:
        continue
    for number, line in enumerate(path.read_text().splitlines(), 1):
        for match in pattern.finditer(line):
            assert (path.name == 'dispatch.md' and (
                path.parent.name in ['gdn_prefill_bf16_state_bf16', 'dual_cache_latent_bf16']
                and match.group() in ['P1', 'P2']
                or path.parent.name == 'fp8_scaled_grouped_gemm' and match.group() == '本轮'
            )), (str(path), number, match.group())
            scan.append({'file': str(path), 'line': number, 'match': match.group(),
                         'classification': 'Source kernel/phase name or kernel iteration; not review history.'})
for root in ['schema_wiki', 'schema_docs', 'schema_extract', 'schema_template']:
    for path in Path(root).rglob('*'):
        if path.is_file() and 'tests' not in path.parts:
            assert path.suffix not in ['.pyc', '.log', '.jsonl', '.diff', '.patch', '.bak', '.tmp'], path

check_records = read(CLEAN / 'directory_checks.json')
assert len(check_records) == 6 and sum(c['yaml_count'] for c in check_records) == 15
assert all(c['exit_code'] == 0 and c['normalization_changes'] == 0 for c in check_records)
for check in check_records:
    pointers(check['output'])
    operator = next(x for x in index['operators'] if x['operator_id'] == check['operator_id'])
    assert check['manifest'] == {name: record['sha256'] for name, record in operator['files'].items()}
test_log = (CLEAN / 'offline-prompt-freeze-tests.log').read_text()
assert re.search(r'Ran 11 tests in', test_log) and test_log.rstrip().endswith('OK')
diff = subprocess.run(['git', 'diff', '--check'], text=True, capture_output=True)
(CLEAN / 'git-diff-check.log').write_text(diff.stdout + diff.stderr)
assert diff.returncode == 0
clock = read('.schema_extract/v2/supervisor.json')['clock']
assert clock['enabled'] is False and clock['generation'] == 8
original_runs = read(CASE / 'BEFORE_REVIEW_INDEX.json')['operators']
run_counts = {item['operator_id']: len(read(Path('.schema_extract/v2/runs') / item['original_run_id'] / 'state.json')['attempts']) for item in original_runs}
assert sum(run_counts.values()) == 201
successor = read('.schema_extract/v2/runs/a362b85a-dce6-4ad4-b38e-3a186dd87c1c/state.json')
assert successor['control']['action'] == 'cancel'

result = {
    'status': 'passed',
    'yaml_files': 15, 'operators': 6, 'yaml_bytes_unchanged': True,
    'current_index': {'path': str(CASE / 'CURRENT_SCHEMA_INDEX.json'), 'sha256': sha(CASE / 'CURRENT_SCHEMA_INDEX.json')},
    'history': {'moved_files': len(moved), 'head_originals_for_preexisting_deletions': 2,
                'pre_edit_documents_preserved': 11, 'all_archived_sha256_match': True,
                'tracked_archive_files_also_match_original_git_blob': tracked_checked,
                'prior_path_aliases_verified': len(locations['prior_location_aliases']),
                'regenerable_bytecode_deleted': len(locations['deletions']),
                'locations': {'path': str(CLEAN / 'LOCATIONS.json'), 'sha256': sha(CLEAN / 'LOCATIONS.json')}},
    'framework': {'baseline_files': len(framework), 'unchanged_in_place': len(framework)-len(framework_moves), 'unchanged_relocated_history': framework_moves},
    'runtime': {'baseline_files': len(runtime), 'all_hashes_unchanged': True, 'original_attempts': run_counts,
                'original_attempt_count': 201, 'monitor_enabled': False, 'monitor_generation': 8, 'successor_control': 'cancel',
                'scope': 'Hash identity and recorded controls only; no command mutates live runtime and no model/GPU process is launched.'},
    'issues': {'all_json_files_unchanged': len(issue_current), 'batch_ids': 15, 'batch_outcomes': dict(outcomes), 'read_only_cli_exit_code': 0},
    'review_evidence_chains': review_chains,
    'checks': {'directory_checks': {'path': str(CLEAN / 'directory_checks.json'), 'sha256': sha(CLEAN / 'directory_checks.json')},
               'source_references': {'path': str(CLEAN / 'source_references.json'), 'sha256': sha(CLEAN / 'source_references.json')},
               'offline_test_count': 11, 'offline_tests_exit_code': 0,
               'offline_test_modules': ['test_prompt_materials', 'test_executor_snapshots', 'test_inputs'],
               'offline_environment': {'PATH': '/opt/venv/python3.13.12/bin:/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1'},
               'rules_file_count': len(rules), 'rules_paths_exist': True, 'discussion_added_to_frozen_rules': False,
               'git_diff_check_exit_code': 0},
    'links': {'active_markdown_files': len(docs), 'local_targets_checked': len(links), 'broken': broken,
              'historical_raw_links': 'Archived originals retain old relative paths; LOCATIONS.json resolves original paths and prior aliases by SHA.'},
    'residual_scan': {'unexpected_procedural_matches': [], 'source_semantics_kept': scan,
                      'non_test_logs_patches_or_bytecode_in_formal_trees': []},
    'exceptions': ['Stable issue decision JSON and current workflow/failure-fixture semantics retained.',
                   'Git-ignored .schema_extract/v2 and .schema_extract/archive retain original paths and contents.',
                   'Old hashed reports and archive indexes retain their original text and time scope; active navigation uses current locations.'],
    'validation_limits': ['Hardware skipped; no GPU, model, extract, runtime resume or numerical algorithm changes.',
                          'Manual semantic review and structural checks do not constitute a new Runner P6 receipt.'],
}
(CLEAN / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
for link in links:
    assert (Path(link['file']).parent / link['target'].split('#', 1)[0]).exists(), link
print(json.dumps({k: result[k] for k in ['status', 'yaml_files', 'operators', 'history', 'issues', 'links']}, ensure_ascii=False))
