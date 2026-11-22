"""Lightweight document/coverage/hash verification; never runs product code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone


REVIEW = Path(__file__).resolve().parents[1]
REPO = REVIEW.parents[1]


def read_json(relative):
    return json.loads((REVIEW / relative).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes(expected):
    missing, changed = [], []
    for name, record in expected.items():
        path = REPO / name
        wanted = record['sha256'] if isinstance(record, dict) else record
        if not path.is_file():
            missing.append(name)
        elif sha(path) != wanted:
            changed.append(name)
    return {'checked': len(expected), 'missing': missing, 'changed': changed}


def recap_items():
    result, section = [], None
    for line in (REPO / 'schema_discussion/extract_twelve_20261119/ISSUE_RECAP.md').read_text().splitlines():
        if line.startswith('## '):
            section = line[3:]
        elif section in ('已发生的 schema 内容修正', '流程、回执与辅助工具问题'):
            if line.startswith('|') and not line.startswith(('| ---', '| 类型')):
                result.append((section, line.split('|')[1].strip()))
        elif section == '已发现但本轮没有修复的来源风险与观察' and line.startswith('- '):
            result.append((section, line[2:]))
    return result


def main():
    errors = []

    def require(ok, message):
        if not ok:
            errors.append(message)

    baseline = read_json('BASELINE.json')
    guard = read_json('round4/INPUT_GUARD.json')
    product = check_hashes(baseline['protected_files'])
    prior = check_hashes(guard['prior_rounds'])
    external = check_hashes(guard['external'])
    actual_external = {'AGENTS.md'} | {
        str(path.relative_to(REPO))
        for path in (REPO / 'schema_discussion/workflow_architecture_20261120').rglob('*')
        if path.is_file()
    }
    external['added'] = sorted(actual_external - set(guard['external']))
    for label, result in [('protected product', product), ('prior rounds', prior), ('external guard', external)]:
        require(not result['missing'] and not result['changed'] and not result.get('added'), label + ' changed')
    require(product['checked'] == 319, 'expected 319 protected files')

    coverage = read_json('COVERAGE.json')
    trim = read_json('round3/TRIM_DECISIONS.json')
    x = read_json('round1/EXTRACT_FINDINGS.json')
    s = read_json('round1/SCHEMA_FINDINGS.json')
    second = read_json('round2/schema_contract_evidence/result.json')
    expected_findings = {f['id'] for f in x['findings'] + s['findings']}
    found = [f['id'] for f in coverage['findings']]
    require(len(found) == len(set(found)) == 26 and set(found) == expected_findings, '26 finding IDs mismatch')
    first_counts = {c['id']: len(c['receipts']) for c in s['framework_candidate_mapping']}
    second_counts = second['coverage']['candidate_counts']
    final_counts = {c['id']: c['occurrences'] for c in coverage['framework_candidates']}
    require(len(coverage['framework_candidates']) == len(final_counts) == 24, '24 candidate IDs mismatch')
    require(first_counts == second_counts == final_counts, 'candidate occurrence counts mismatch')
    require(sum(final_counts.values()) == 104, 'expected 104 candidate occurrences')
    expected_recap = recap_items()
    actual_recap = [(c['source_section'], c['source_item']) for c in coverage['issue_recap']]
    require(actual_recap == expected_recap and len(actual_recap) == 20, 'ISSUE_RECAP coverage mismatch')
    require({t['id'] for t in coverage['reviewed_trims']} == {t['id'] for t in trim['trimmed_proposals']} and len(coverage['reviewed_trims']) == 16, 'trim coverage mismatch')
    require(all(t['final_review'] == 'uphold' for t in coverage['reviewed_trims']), 'unexpected trim disposition')
    require(coverage['required_user_decisions'] == [] and coverage['product_changes_implemented'] is False, 'scope/decision status mismatch')
    packages = set(coverage['work_packages'])
    for field in ('findings', 'framework_candidates', 'issue_recap', 'existing_corrections', 'final_clarifications'):
        for row in coverage[field]:
            require(set(row['work_packages']) <= packages, field + ' has unknown work package')
            refs = row.get('findings', [row['finding']] if 'finding' in row else [])
            require(set(refs) <= expected_findings, field + ' has unknown finding')
    amendments = coverage['required_yaml_amendments']
    dispatches = coverage['required_dispatch_amendments']
    require(len(amendments) == 9 and len({a['path'] for a in amendments}) == 9 and len(dispatches) == 2, '9 YAML / 2 dispatch scope mismatch')
    for amendment in amendments:
        require(amendment['suggested_next_version'] == amendment['current_version'] + 1, 'non-incrementing proposed Version')
    for name in [a['path'] for a in amendments] + dispatches:
        require(name.startswith('schema_wiki/') and (REPO / name).is_file(), 'invalid product amendment path: ' + name)

    json_files = sorted(REVIEW.rglob('*.json'))
    for path in json_files:
        try:
            json.loads(path.read_text())
        except (ValueError, OSError) as exc:
            errors.append(str(path.relative_to(REVIEW)) + ': ' + str(exc))
    links, broken = [], []
    for name in ('README.md', 'FINAL_PLAN.md', 'FINAL_REVIEW.md'):
        path = REVIEW / name
        for target in re.findall(r'\[[^\]\n]*\]\(([^)\n]+)\)', path.read_text()):
            if '://' in target or target.startswith('#'):
                continue
            target = target.split('#', 1)[0]
            links.append({'file': name, 'target': target})
            if not (path.parent / target).resolve().exists():
                broken.append({'file': name, 'target': target})
    require(not broken, 'broken final document links')
    readme = (REVIEW / 'README.md').read_text()
    require('2+2 max → 1 high → 1 xhigh' in readme and '已完成' in readme, 'README lacks final process/status')
    require(not any(t in readme for t in ('正在整合', '正在第一轮', '正在第二轮', '随后由新的')), 'README retains stale progress')

    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    require(head == baseline['head'], 'HEAD changed from baseline')
    final_names = ('README.md', 'FINAL_PLAN.md', 'FINAL_REVIEW.md', 'COVERAGE.json', 'round4/INPUT_GUARD.json', 'round4/validate.py')
    result = {
        'format': 'extract_schema_review.final_validation.v1',
        'status': 'passed' if not errors else 'failed',
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'command': 'PYTHONDONTWRITEBYTECODE=1 python3 -B schema_discussion/extract_schema_review_20261121/round4/validate.py',
        'baseline_head': baseline['head'], 'current_head': head,
        'baseline_sha256': sha(REVIEW / 'BASELINE.json'),
        'protected_product_files': product, 'preserved_prior_round_files': prior,
        'preserved_agents_and_workflow_discussion': external,
        'coverage': {'findings': len(found), 'framework_candidates': len(final_counts), 'candidate_occurrences': sum(final_counts.values()), 'issue_recap_items': len(actual_recap), 'trim_groups': 16, 'yaml_amendments_pending': len(amendments), 'dispatch_amendments_pending': len(dispatches)},
        'json_files_parsed': len(json_files), 'final_markdown_links_checked': len(links),
        'broken_links': broken, 'errors': errors,
        'final_document_sha256': {name: sha(REVIEW / name) for name in final_names},
        'round4_product_tests_run': 0, 'round4_new_reproduction_experiments': 0,
        'real_model_calls': 0, 'gpu_calls': 0, 'real_extract_resume_monitor_controls': 0,
        'product_changes_implemented': False,
        'scope_note': 'Hash guard proves the 319 BASELINE-listed files unchanged; prior reports and 86 external guard files also checked. Only final discussion files were written. No new product approval or all-source correctness claim.',
    }
    (REVIEW / 'FINAL_VALIDATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
