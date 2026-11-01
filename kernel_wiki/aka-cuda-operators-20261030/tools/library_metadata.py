"""Read interface declarations and normalize existing performance records (stdlib only)."""
import ast
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def interface_from_text(candidate, inputs):
    tree = ast.parse(candidate)
    models = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Model']
    if len(models) != 1:
        raise ValueError('Expected one Model class')
    methods, aliases = {}, {}
    for node in models[0].body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods[node.name] = node
            aliases.pop(node.name, None)
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    aliases[target.id] = node.value.id
                    methods.pop(target.id, None)
    target, seen = 'forward', set()
    while target in aliases:
        if target in seen:
            raise ValueError('Cyclic forward alias')
        seen.add(target)
        target = aliases[target]
    if target not in methods:
        raise ValueError('Cannot statically resolve Model.forward')
    hooks = {n.name: n for n in ast.parse(inputs).body if isinstance(n, ast.FunctionDef)}
    if '_make_inputs' not in hooks:
        raise ValueError('Missing _make_inputs')
    return {
        '__init__': ast.unparse(methods['__init__'].args) if '__init__' in methods else None,
        'forward': ast.unparse(methods[target].args),
        'forward_target': target,
        '_make_inputs': ast.unparse(hooks['_make_inputs'].args),
        'prepare_model': ast.unparse(hooks['prepare_model'].args) if 'prepare_model' in hooks else None,
    }


def extract_interface(folder):
    return interface_from_text((folder / 'candidate/solution.py').read_text(),
                               (folder / 'problem/input.py').read_text())


def empty_row(shape):
    return {'shape_id': str(shape), 'candidate_us': None, 'baseline_us': None,
            'candidate_over_baseline': None, 'ci95': None, 'by_order': {},
            'status': 'not_reported'}


def event_rows(record, shape_ids, balanced=False):
    rows = []
    for sid in shape_ids:
        row = empty_row(sid)
        source = record.get('shapes', {}).get(sid)
        if source is not None:
            row.update(status='ratio_only' if balanced else 'measured',
                       candidate_over_baseline=source.get('balanced_R_over_N0' if balanced
                                                         else 'median_paired_R_over_N0'),
                       ci95=source.get('bootstrap_95pct_interval'))
            if not balanced:
                row.update(candidate_us=source.get('R_median_us'), baseline_us=source.get('N0_median_us'))
            # Balanced results combine allocation-order ratios, never absolute timings
            # from different jobs/GPUs. Keep the two order strata separately.
            for source_key, order in [('N0_first_median_ratio', 'N0_first'),
                                      ('R_first_median_ratio', 'R_first')]:
                if source_key in source:
                    row['by_order'][order] = {'candidate_over_baseline': source[source_key], 'ci95': None}
        rows.append(row)
    return rows


def ncu_rows(record, shape_ids):
    rows = []
    unsupported = record.get('unavailable_comparisons', {})
    for sid in shape_ids:
        row = empty_row(sid)
        source = record.get('shapes', {}).get(sid)
        if source is not None:
            c = source.get('candidate_us', source.get('candidate', {}).get('kernel_sum_us'))
            b = source.get('production_us', source.get('production', {}).get('kernel_sum_us'))
            row.update(candidate_us=c, baseline_us=b,
                       candidate_over_baseline=c / b if c is not None and b is not None else None,
                       status=source.get('status', 'measured'))
            for order, values in source.get('orders', {}).items():
                row['by_order'][order] = {'candidate_over_baseline': values['C_over_P'],
                                          'ci95': values.get('bootstrap_95pct')}
            for order, ratio in source.get('ratios', {}).items():
                row['by_order'].setdefault(order, {'candidate_over_baseline': ratio, 'ci95': None})
        elif sid in unsupported:
            row['status'] = 'baseline_unsupported'
        rows.append(row)
    return rows


def aggregate(candidate=None, baseline=None, ratio=None):
    return {'candidate_geomean_us': candidate, 'baseline_geomean_us': baseline,
            'candidate_over_baseline_geomean': ratio}


def measurement(name, metric, pointer, comparator, protocol, environment, statistic, rows,
                totals=None, boundary=None):
    return {
        'id': name, 'metric': metric, 'unit': 'us', 'source_pointer': pointer,
        'boundary': boundary or ('Full forward, according to the original event protocol.'
                                if metric == 'cuda_event_forward' else
                                'Sum of launched GPU kernels; excludes CPU/compilation and copy-engine DMA.'),
        'comparison': comparator, 'protocol': protocol, 'environment': environment,
        'ratio_statistic': statistic, 'aggregate': totals or aggregate(), 'rows': rows,
    }


def build_summary(folder, entry):
    perf = folder / 'performance'
    cases = read(folder / 'problem/shape_cases.json')
    ids = sorted(cases, key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
    source_paths = []
    runs = []
    if entry['solution_origin'] == 'AKA':
        selection = read(perf / 'SELECTION.json')
        cert = read(perf / 'CERTIFIED_BEST.json')
        result = read(perf / 'ncu/result.json')
        with (perf / 'shapes.csv').open() as f:
            csv_rows = {r['shape_id']: r for r in csv.DictReader(f)}
        rows = []
        for sid in ids:
            raw, tab = result['shapes'][sid], csv_rows[sid]
            c, b = raw['kernel_sum_us'], float(tab['production_us'])
            row = empty_row(sid)
            row.update(candidate_us=c, baseline_us=b, candidate_over_baseline=c / b,
                       status=raw.get('status', 'measured'))
            rows.append(row)
        arm = selection['selected_arm']
        runs.append(measurement(
            'ncu', 'ncu_kernel_sum', 'ncu/result.json#/shapes',
            {'kind': 'production', 'identity': selection.get('production_evidence_original'),
             'design': 'independent_jobs'}, result['protocol'], [result['environment']],
            'ratio_of_reported_medians', rows,
            aggregate(selection['geomean_us'][arm], selection['geomean_us']['production'],
                      1 / selection['speedup_over_production'])))
        status, primary = cert['status'], 'ncu'
        source_paths = ['CERTIFIED_BEST.json', 'SELECTION.json', 'ncu/result.json', 'shapes.csv']
        notes = [selection.get('comparison_limit') or
                 'Production and selected-candidate measurements come from independent jobs; baseline environment is not included.']
        # The archive-wide comparison limitation is repeated in each selection's
        # provenance only indirectly. It is never presented as a same-job comparison.
        waiver = False
    else:
        acceptance = read(perf / 'ACCEPTANCE.json')
        status = acceptance['status']
        waiver = bool(acceptance.get('user_acceptance', {}).get('explicit'))
        source_paths = ['ACCEPTANCE.json']
        notes = list(acceptance.get('limitations', []))
        if acceptance.get('metric', '').startswith('NCU'):
            record = acceptance['performance']
            environment = {k: record[k] for k in ['gpu_uuid', 'clock_mhz', 'locked_sm_mhz'] if k in record}
            if acceptance.get('environment'):
                environment['historical_environment_reference'] = acceptance['environment']
            runs.append(measurement(
                'ncu', 'ncu_kernel_sum', 'ACCEPTANCE.json#/performance',
                {'kind': 'production', 'identity': acceptance.get('comparison'),
                 'design': 'as_recorded'}, record.get('protocol'), [environment],
                'ratio_of_reported_medians', ncu_rows(record, ids), boundary=record.get('boundary')))
            primary = 'ncu'
            if record.get('unavailable_comparisons'):
                notes.append('Unsupported production shapes retain null timings; no comparison is invented.')
        else:
            jobs = acceptance.get('jobs', {})
            base = acceptance.get('comparison_baseline', {})
            identity = base.get('tree_sha256') or acceptance.get('baseline_sha256')
            comparator = {'kind': 'historical_candidate', 'identity': identity, 'design': 'paired_jobs'}
            for key, job_keys, balanced in [
                ('paired_performance', ['paired'], False),
                ('paired_reverse_diagnostic', ['paired_reverse'], False),
                ('balanced_order_performance', ['paired', 'paired_reverse'], True),
            ]:
                if key not in acceptance:
                    continue
                record = acceptance[key]
                environments = []
                for job_key in job_keys:
                    job = jobs.get(job_key, {})
                    env = dict(job.get('environment', {}))
                    if job.get('nvcc'):
                        env['nvcc'] = job['nvcc']
                    env['source_job_key'] = job_key
                    if job.get('job_id'):
                        env['job_id'] = job['job_id']
                    environments.append(env)
                runs.append(measurement(
                    key, 'cuda_event_forward', 'ACCEPTANCE.json#/' + key, comparator,
                    record.get('method'), environments,
                    'balanced_order_ratio' if balanced else 'median_of_paired_ratios',
                    event_rows(record, ids, balanced),
                    aggregate(ratio=record.get('geomean_R_over_N0'))))
            primary = acceptance.get('performance_decision', {}).get('basis')
            if primary not in {r['id'] for r in runs}:
                raise ValueError(f'Unknown authoritative performance basis: {entry["operator"]}: {primary}')
            notes.append('Legacy N0 denotes the identified historical candidate, not production; balanced results contain ratios only.')
    return {
        'schema': 'aka.performance_summary.v1', 'operator': entry['operator'],
        'solution_origin': entry['solution_origin'],
        'candidate_manifest_sha256': canonical(entry['candidate_files']),
        'acceptance_status': status, 'performance_waiver': waiver,
        'problem_verification': entry['problem_verification'],
        'primary_measurement': primary,
        'source_records': [{'path': p, 'sha256': sha(perf / p)} for p in source_paths],
        'measurements': runs, 'notes': notes,
    }
