#!/usr/bin/env python3
"""Verify library content and source/problem/measurement bindings without a GPU."""
import ast
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
from library_metadata import build_summary, extract_interface
from refresh_metadata import dependencies

ROOT = Path(__file__).resolve().parents[1]
PROBLEM_FILES = {'input.py', 'reference.py', 'metadata.json', 'shape_cases.json',
                 'shape_range.json', 'solution.py', 'roofline.json'}
PROVISIONAL = {'fmha_nvfp4_qkpv', 'gdn_prefill_bf16_state_fp32',
               'paged_latent_mxfp8_nvfp4'}
CATEGORY_COUNTS = {'gemm': 15, 'moe': 5, 'fmha': 8, 'mla': 4,
                   'sparse_attention': 4, 'linear_attention': 8}
GEMM_COUNTS = {'dense': 6, 'grouped': 5, 'grouped_masked': 4}
SUMMARY_KEYS = {'schema', 'operator', 'solution_origin', 'candidate_manifest_sha256',
                'acceptance_status', 'performance_waiver', 'problem_verification',
                'primary_measurement', 'source_records', 'measurements', 'notes'}
RUN_KEYS = {'id', 'metric', 'unit', 'source_pointer', 'boundary', 'comparison', 'protocol',
            'environment', 'ratio_statistic', 'aggregate', 'rows'}
ROW_KEYS = {'shape_id', 'candidate_us', 'baseline_us', 'candidate_over_baseline',
            'ci95', 'by_order', 'status'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(folder):
    return {p.relative_to(folder).as_posix(): sha(p)
            for p in sorted(folder.rglob('*')) if p.is_file()}


def canonical(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def expected_category(op, metadata):
    if metadata['operator_family'] == 'gemm':
        subcategory = ('grouped_masked' if 'grouped_gemm_masked' in op
                       else 'grouped' if 'grouped_gemm' in op else 'dense')
        return 'gemm', subcategory
    if metadata['operator_family'] == 'moe':
        return 'moe', None
    if 'linear_attention' in metadata.get('tags', []):
        return 'linear_attention', None
    if 'sparse_attention' in metadata.get('tags', []):
        return 'sparse_attention', None
    if op.startswith('fmha_'):
        return 'fmha', None
    if op.startswith('mla_'):
        return 'mla', None
    raise ValueError(f'Unclassified operator: {op}')


def main():
    manifest = read(ROOT / 'MANIFEST.json')
    entries = manifest['operators']
    refs = set(manifest['input_operator_sets']['reference36'])
    akas = set(manifest['input_operator_sets']['aka20'])
    names = {x['operator'] for x in entries}
    check(manifest['schema'] == 'aka.cuda_operator_library.v3', 'Normalized manifest schema')
    check(manifest['conventions'] == 'CONVENTIONS.md', 'Conventions entry')
    check(len((ROOT / 'CONVENTIONS.md').read_text().splitlines()) <= 35, 'Concise conventions')
    check((len(refs), len(akas), len(refs & akas)) == (36, 20, 12), 'Input operator sets')
    check(names == refs | akas and len(entries) == 44, 'Must contain the 44-operator union')
    check(manifest['layout'] == 'categorized_by_operator_function', 'Categorized layout')
    check({p.name for p in (ROOT / 'operators').iterdir() if p.is_dir()}
          == set(CATEGORY_COUNTS), 'Six category directories')
    declared_directories = {x['directory'] for x in entries}
    discovered_directories = {
        p.parent.parent.relative_to(ROOT).as_posix()
        for p in (ROOT / 'operators').rglob('candidate/solution.py')}
    check(declared_directories == discovered_directories and len(declared_directories) == 44,
          'Complete, unique categorized operator directories')
    check(Counter(e['category'] for e in entries) == CATEGORY_COUNTS, 'Category counts')
    check(Counter(e['subcategory'] for e in entries if e['category'] == 'gemm')
          == GEMM_COUNTS, 'GEMM subcategory counts')
    check({c['id']: c['operator_count'] for c in manifest['categories']} == CATEGORY_COUNTS,
          'Category catalog')
    for category in manifest['categories']:
        check(category['directory'] == 'operators/' + category['id'], 'Category catalog path')
        check((ROOT / category['directory'] / 'README.md').is_file(), 'Category index')
        if category['id'] == 'gemm':
            check({c['id']: c['operator_count'] for c in category['subcategories']} == GEMM_COUNTS,
                  'GEMM subcategory catalog')
            for subcategory in category['subcategories']:
                check(subcategory['directory'] == 'operators/gemm/' + subcategory['id'],
                      'GEMM subcategory path')
                check((ROOT / subcategory['directory'] / 'README.md').is_file(), 'GEMM index')
    audit = {x['operator']: x for x in read(ROOT / 'PROBLEM_AUDIT.json')['operators']}
    check(set(audit) == names, 'Problem audit coverage')
    check({x['operator'] for x in entries if x['problem_requires_latest_receipt_confirmation']}
          == PROVISIONAL, 'Three provisional problem flags')
    candidate_count = 0
    measurement_count = 0
    for entry in entries:
        op = entry['operator']
        folder = ROOT / entry['directory']
        source = hashes(folder / 'candidate')
        problem = hashes(folder / 'problem')
        candidate_count += len(source)
        check(source == entry['candidate_files'], f'{op}: source hashes')
        check(set(problem) == PROBLEM_FILES, f'{op}: complete seven-file problem')
        check(problem == entry['problem_files'] == audit[op]['problem_files_sha256'],
              f'{op}: problem hashes')
        expected_origin = 'AKA' if op in akas else 'REFERENCE'
        check(entry['solution_origin'] == expected_origin, f'{op}: selection priority')
        check(entry['problem_verification'] == audit[op]['status'], f'{op}: problem status')
        check((folder / 'candidate/solution.py').is_file(), f'{op}: entrypoint')
        check(entry['entrypoint'] == entry['directory'] + '/candidate/solution.py:Model',
              f'{op}: relocated entrypoint')
        check(audit[op]['problem_directory'] == entry['directory'] + '/problem',
              f'{op}: relocated problem audit')
        check(set(entry['performance']) == {'summary', 'metric', 'status', 'record'},
              f'{op}: uniform performance registry')
        for key in ['record', 'summary']:
            if key in entry['performance']:
                record_path = entry['performance'][key]
                check(record_path.startswith(entry['directory'] + '/performance/')
                      and (ROOT / record_path).is_file(), f'{op}: relocated {key}')
        cases = read(folder / 'problem/shape_cases.json')
        metadata = read(folder / 'problem/metadata.json')
        check(bool(cases) and bool(metadata['benchmark_contract']), f'{op}: cases/contract')
        check(entry['interface'] == extract_interface(folder), f'{op}: complete source-declared interface')
        check(entry['dependencies'] == dependencies(folder), f'{op}: declared build dependencies')
        summary = read(folder / 'performance/SUMMARY.json')
        check(set(summary) == SUMMARY_KEYS, f'{op}: uniform summary schema')
        check(summary == build_summary(folder, entry), f'{op}: summary matches original records')
        check(summary['candidate_manifest_sha256'] == canonical(source), f'{op}: summary source identity')
        check(summary['acceptance_status'] == entry['performance']['status'], f'{op}: summary status')
        for record in summary['source_records']:
            check(sha(folder / 'performance' / record['path']) == record['sha256'], f'{op}: raw record hash')
        runs = summary['measurements']
        measurement_count += len(runs)
        primary = next(r for r in runs if r['id'] == summary['primary_measurement'])
        check(entry['performance']['metric'] == primary['metric'], f'{op}: primary metric')
        for run in runs:
            check(set(run) == RUN_KEYS, f'{op}: uniform measurement schema')
            check(run['unit'] == 'us' and run['metric'] in {'cuda_event_forward', 'ncu_kernel_sum'},
                  f'{op}: metric and unit')
            check(run['comparison']['kind'] in {'production', 'historical_candidate'}, f'{op}: comparator identity')
            check(isinstance(run['environment'], list), f'{op}: environment records')
            source_path, pointer = run['source_pointer'].split('#/', 1)
            referenced = read(folder / 'performance' / source_path)
            for component in pointer.split('/'):
                referenced = referenced[component]
            check({r['shape_id'] for r in run['rows']} == set(cases)
                  and len(run['rows']) == len(cases), f'{op}: summary shape coverage')
            for row in run['rows']:
                check(set(row) == ROW_KEYS, f'{op}: uniform shape schema')
                for key in ['candidate_us', 'baseline_us', 'candidate_over_baseline']:
                    value = row[key]
                    check(value is None or (isinstance(value, (float, int))
                                           and math.isfinite(value) and value > 0), f'{op}: valid {key}')
                if row['status'] == 'baseline_unsupported':
                    check(row['baseline_us'] is None and row['candidate_over_baseline'] is None,
                          f'{op}: no fabricated unsupported comparison')
                if run['ratio_statistic'] == 'balanced_order_ratio':
                    check(row['candidate_us'] is None and row['baseline_us'] is None,
                          f'{op}: no cross-job absolute aggregation')
                intervals = [row['ci95']] + [x['ci95'] for x in row['by_order'].values()]
                for interval in intervals:
                    check(interval is None or (len(interval) == 2 and 0 < interval[0] <= interval[1]
                                               and all(math.isfinite(x) for x in interval)), f'{op}: valid interval')
        readme = (folder / 'README.md').read_text()
        check(len(readme.splitlines()) <= 22, f'{op}: concise README')
        for label in ['来源', '入口', '初始化', '题目', '范围', '依赖', '性能', '状态', '补充']:
            check(f'| {label} |' in readme, f'{op}: README field {label}')
        category, subcategory = expected_category(op, metadata)
        check((entry['category'], entry['subcategory']) == (category, subcategory),
              f'{op}: category matches the problem function')
        group = Path('operators') / category
        if subcategory:
            group /= subcategory
        check(entry['directory'] == (group / op).as_posix(), f'{op}: categorized path')
        index = ROOT / group / 'README.md'
        check(f'[{op}]({op}/README.md)' in index.read_text(), f'{op}: category navigation')
        if op in akas:
            check('AKA 的解' in (folder / 'README.md').read_text(), f'{op}: AKA label')
            selection = read(folder / 'performance/SELECTION.json')
            certificate_path = folder / 'performance/CERTIFIED_BEST.json'
            certificate = read(certificate_path)
            measured = read(folder / 'performance/ncu/result.json')
            check(source == selection['candidate_files'] == measured['source_files'],
                  f'{op}: measured source binding')
            check(problem == selection['contract_files'] == measured['contract_sha256'],
                  f'{op}: measured problem binding')
            check(certificate['tree'] == selection['candidate_tree'] == measured['formal_candidate_tree'],
                  f'{op}: certificate/source identity')
            check(sha(certificate_path) == selection['certificate_sha256'], f'{op}: certificate bytes')
            check(certificate['fixed']['full_pass'] and certificate['domain']['full_pass'],
                  f'{op}: preserved fixed/domain certification')
            check(set(cases) == set(measured['shapes']), f'{op}: measured shape coverage')
            geomean = math.exp(statistics.mean(math.log(v['kernel_sum_us'])
                                               for v in measured['shapes'].values()))
            check(math.isclose(geomean, primary['aggregate']['candidate_geomean_us'], rel_tol=1e-12),
                  f'{op}: NCU geometric mean')
            with (folder / 'performance/shapes.csv').open() as f:
                rows = list(csv.DictReader(f))
            check(len(rows) == len(cases), f'{op}: exported shape rows')
            for row in rows:
                check(row['operator'] == op and row['selected_arm'] == selection['selected_arm'],
                      f'{op}: CSV selected arm')
                check(json.loads(row['parameters']) == cases[row['shape_id']], f'{op}: CSV shape parameters')
                check(math.isclose(float(row['selected_us']), measured['shapes'][row['shape_id']]['kernel_sum_us'],
                                   rel_tol=1e-10), f'{op}: CSV measured latency')
        else:
            acceptance_path = folder / 'performance/ACCEPTANCE.json'
            acceptance = read(acceptance_path)
            check(source == acceptance['candidate_files'], f'{op}: reference acceptance binding')
            check(source == read(folder / 'performance/DELIVERY.json')['candidate_files'],
                  f'{op}: final delivery binding')
            check(entry['performance']['status'] == acceptance['status'], f'{op}: acceptance status preserved')
            evidence = audit[op]
            if op not in PROVISIONAL:
                check(sha(acceptance_path) == evidence['acceptance_sha256'], f'{op}: audited acceptance bytes')
                binding = evidence['binding']
                if evidence['binding_kind'] == 'frozen_stage_receipt':
                    check(binding['operator_id'] == op, f'{op}: staged operator')
                    check(canonical({k: problem[k] for k in ['input.py', 'reference.py']})
                          == binding['operator_source_sha256'], f'{op}: frozen input/reference pair')
                    check(canonical(metadata['benchmark_contract']) == binding['benchmark_contract_sha256'],
                          f'{op}: frozen numerical contract')
                    check({k: canonical(v) for k, v in cases.items()} == binding['cases'],
                          f'{op}: frozen shape cases')
                else:
                    check(problem['input.py'] == binding['input_sha256']
                          and problem['reference.py'] == binding['reference_sha256'],
                          f'{op}: validation input/reference pair')
                    check(canonical(metadata) == binding['metadata_canonical_sha256'],
                          f'{op}: validated metadata')
                    check(canonical(cases) == binding['cases_canonical_sha256'], f'{op}: validated cases')

    files = [p for p in ROOT.rglob('*') if p.is_file()]
    check(not any(p.is_symlink() for p in ROOT.rglob('*')), 'Unexpected symlink')
    forbidden = {'.git', '__MACOSX', '__pycache__', '.ruff_cache', '.DS_Store'}
    check(not any(forbidden.intersection(p.relative_to(ROOT).parts) for p in files), 'Cache/history contamination')
    python_count = 0
    for path in files:
        if path.suffix == '.py':
            compile(path.read_bytes(), str(path), 'exec', dont_inherit=True)
            python_count += 1
        elif path.suffix == '.md':
            text = re.sub(r'```.*?```', '', path.read_text(), flags=re.S)
            for target in re.findall(r'\]\(([^)]+)\)', text):
                target = target.split('#', 1)[0]
                if not target or re.match(r'^[a-zA-Z]+:', target):
                    continue
                check((path.parent / target).exists(), f'Broken Markdown link: {path}: {target}')
    checksum_path = ROOT / 'CHECKSUMS.sha256'
    expected = {}
    for line in checksum_path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name] = digest
    actual = {p.relative_to(ROOT).as_posix(): sha(p) for p in files if p != checksum_path}
    check(actual == expected, 'Complete-library checksum manifest')
    print(json.dumps({'status': 'passed', 'operators': len(entries),
                      'solutions': dict(Counter(x['solution_origin'] for x in entries)),
                      'categories': dict(Counter(x['category'] for x in entries)),
                      'gemm_subcategories': dict(Counter(x['subcategory'] for x in entries
                                                         if x['category'] == 'gemm')),
                      'candidate_files': candidate_count, 'problem_files': 44 * 7,
                      'interfaces_checked': len(entries), 'performance_summaries': len(entries),
                      'measurement_groups': measurement_count,
                      'forward_aliases': {e['operator']: e['interface']['forward_target'] for e in entries
                                          if e['interface']['forward_target'] != 'forward'},
                      'max_operator_readme_lines': max(len((ROOT / e['directory'] / 'README.md').read_text().splitlines())
                                                       for e in entries),
                      'python_files_syntax_checked': python_count,
                      'problem_verification': dict(Counter(x['problem_verification'] for x in entries)),
                      'files': len(files), 'bytes': sum(p.stat().st_size for p in files),
                      'gpu_evaluation_rerun': False}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
