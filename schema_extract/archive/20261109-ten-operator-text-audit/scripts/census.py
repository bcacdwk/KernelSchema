#!/usr/bin/env python3
"""Freeze the selected corpus and inventory text, without running extraction.

Counts are Unicode characters after whitespace normalization, not token counts.
Semantic labels are supplied by human/model review in annotations/, never inferred
by this script. YAML aliases and cross-document repetition have separate metrics.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

import yaml

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[2]
INVENTORY = REPO / '.schema_extract/supervision/pause-closeout-20261109/inventory.json'
TEMPLATES = ['gdn_prefill', 'fa4_prefill', 'fa4_decode', 'qsa_prefill']
EXTRACTS = ['gdn', 'gemm', 'mla', 'sparse', 'fmha', 'moe']
REGIONS = ['math_reshape', 'work_schedule', 'tile_pipeline']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def norm(value):
    return re.sub(r'\s+', ' ', value).strip()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    temporary.replace(path)


def relative(path):
    return str(path.resolve().relative_to(REPO))


def freeze():
    inv = json.loads(INVENTORY.read_text())
    manifest = {'format': 'kernelschema.text_audit.corpus.v1',
                'inventory': relative(INVENTORY), 'inventory_sha256': digest(INVENTORY.read_bytes()),
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                'scope': 'Current template tasks and current-plan pause-selected six-operator artifacts; no historical lineage mixing.',
                'operators': {}, 'rules': []}
    paths = [REPO / 'README.md', REPO / 'EXTRACT_SCHEMA.md', REPO / 'schema_extract/REVIEW.md',
             REPO / 'schema_extract/phases/03_root.md', REPO / 'schema_template/README.md',
             REPO / 'schema_template/examples/README.md'] + sorted((REPO / 'schema_docs').glob('*.md'))
    for p in paths:
        target = BASE / 'rules' / p.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
        manifest['rules'].append({'path': relative(p), 'snapshot': relative(target), 'sha256': digest(p.read_bytes())})
    for name in TEMPLATES:
        opid = 'template_' + name
        directory = REPO / 'schema_template/examples' / name
        files = []
        for p in sorted(directory.glob('*.yaml')):
            d = yaml.safe_load(p.read_text())
            files.append({'node': p.stem, 'file': p.name, 'source_file': str(p),
                          'category': 'template', 'authored': True, 'in_current_plan': True,
                          'own_sha256': digest(p.read_bytes()), 'node_version': d['meta_data'].get('version'),
                          'parent': d['meta_data'].get('dispatch', {}).get('parent')})
        manifest['operators'][opid] = {'group': 'template', 'name': name, 'planned_nodes': len(files),
            'nodes': files, 'dispatch_source': str(directory / 'dispatch.md')}
    for name in EXTRACTS:
        row = next(x for x in inv['operators'] if x['operator'] == name)
        nodes = [{k: x.get(k) for k in ['node', 'file', 'source_file', 'category', 'authored',
                 'in_current_plan', 'own_sha256', 'node_version', 'parent', 'selected_checkpoint', 'selected_basis']}
                 for x in inv['nodes'] if x['operator'] == name and x.get('in_current_plan')]
        manifest['operators']['extract_' + name] = {'group': 'extract', 'name': name,
            'planned_nodes': row['planned_nodes'], 'run_id': row['run_id'],
            'working_checkpoint': row['working_checkpoint'], 'extract_status': row['status'],
            'executor': row['executor'], 'semantic_revision': row.get('semantic_revision'),
            'nodes': nodes, 'dispatch_source': str(Path(row['working_artifacts']) / 'dispatch.md')}
    for opid, op in manifest['operators'].items():
        for node in op['nodes']:
            p = Path(node['source_file']) if node.get('source_file') else None
            node['material_present'] = bool(p and p.is_file())
            # Root approval is tracked by stages; its node.authored flag may be false.
            node['semantic_eligible'] = bool(node['material_present'] and
                (node.get('authored') or node['category'] in {'approved', 'review_candidate', 'template'}))
            if not node['material_present']:
                continue
            data = p.read_bytes()
            if node.get('own_sha256') and digest(data) != node['own_sha256']:
                raise ValueError(f'Pause identity mismatch: {p}')
            subdir = 'inputs' if node['semantic_eligible'] else 'unwritten_scaffolds'
            target = BASE / subdir / opid / node['file']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            node['snapshot'] = relative(target)
            node['sha256'] = digest(data)
            node['bytes'] = len(data)
            node['source_file'] = relative(p)
        p = Path(op['dispatch_source'])
        target = BASE / 'inputs' / opid / 'dispatch.md'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
        op['dispatch_source'] = relative(p)
        op['dispatch_snapshot'] = relative(target)
        op['dispatch_sha256'] = digest(p.read_bytes())
    write_json(BASE / 'data/manifest.json', manifest)
    return manifest


def bucket(path):
    p = '.'.join(map(str, path))
    if p.startswith('contract.algorithm_requirement.'):
        return 'algorithm_contract'
    if p.startswith('contract.hardware_specification.'):
        return 'hardware_contract'
    if p.startswith('design.math_reshape.'):
        return 'math'
    if p.startswith('design.work_schedule.'):
        return 'schedule'
    if p.startswith('design.tile_pipeline.'):
        return 'pipeline'
    if p.startswith('realization.'):
        return 'realization'
    return 'metadata'


def text_family(path, value):
    """Inventory declared open-text locations; this is NOT semantic classification."""
    keys = [str(x) for x in path]
    p = '.'.join(keys)
    last = keys[-1] if keys else ''
    if p == 'meta_data.dispatch.when':
        return 'dispatch.when'
    if p == 'meta_data.describe':
        return 'metadata.describe'
    if '.info.verb.' in p and last != 'execution_mode':
        return 'pipeline.verb'
    if p.startswith('realization.') and any(x in keys for x in ['choice', 'reason', 'feedback']):
        return 'realization.' + next(x for x in ['choice', 'reason', 'feedback'] if x in keys)
    if '.worker_assignment.refill.' in p:
        return 'schedule.refill'
    if '.worker_assignment.exit' in p and value != 'assigned_done':
        return 'schedule.exit'
    if p.startswith('contract.algorithm_requirement.golden_reference.tolerance.'):
        return 'contract.tolerance'
    if p.startswith('design.math_reshape.') and '.arithmetic.' in p:
        return 'math.arithmetic'
    if p.startswith('design.math_reshape.ledger.'):
        return 'math.ledger'
    if p.startswith('contract.hardware_specification.feature.'):
        return 'hardware.feature'
    if p.startswith('contract.hardware_specification.stack.'):
        return 'hardware.stack'
    if p.startswith('contract.hardware_specification.execution.'):
        return 'hardware.execution'
    if p.startswith('contract.algorithm_requirement.compute.') and last == 'name':
        return 'contract.compute_name'
    if p.startswith('design.math_reshape.rewrite.') and last in {'name', 'step'}:
        return 'math.' + last
    if last in {'describe', 'meaning', 'thesis', 'strategy', 'math', 'gap', 'note', 'reason', 'range',
                'keep_value', 'precision', 'constraint', 'condition'}:
        return bucket(path) + '.' + last
    return None


def walk_ast(root):
    """Every value occurrence, with the same node identity for aliased declarations."""
    def walk(node, path, ancestors):
        if id(node) in ancestors:
            raise ValueError('cyclic YAML alias')
        if isinstance(node, yaml.ScalarNode):
            if node.tag == 'tag:yaml.org,2002:str':
                yield path, node
        elif isinstance(node, yaml.MappingNode):
            for key, value in node.value:
                yield from walk(value, path + [key.value], ancestors | {id(node)})
        elif isinstance(node, yaml.SequenceNode):
            for i, child in enumerate(node.value):
                yield from walk(child, path + [i], ancestors | {id(node)})
    yield from walk(root, [], set())


def field_path(path):
    out = ''
    for key in path:
        out += f'[{key}]' if isinstance(key, int) else ('.' if out else '') + key
    return out


def ordinary_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from ordinary_strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from ordinary_strings(v)


def run_census(manifest):
    all_metrics = {}
    for opid, op in manifest['operators'].items():
        scalars, units, metrics = [], {}, []
        node_map = {n['node']: n for n in op['nodes'] if n['semantic_eligible']}
        raw = {name: yaml.safe_load((REPO / n['snapshot']).read_text()) for name, n in node_map.items()}
        def add_unit(family, text, occurrence):
            uid = 't_' + digest((family + '\0' + text).encode())[:16]
            u = units.setdefault(uid, {'id': uid, 'family': family, 'text': text, 'chars': len(text), 'occurrences': []})
            u['occurrences'].append(occurrence)
            return uid
        for name, record in node_map.items():
            source = REPO / record['snapshot']
            body = source.read_text()
            tree = yaml.compose(body)
            seen, seen_target = set(), set()
            decl = collections.Counter()
            inst = collections.Counter()
            targets = collections.Counter()
            target_instances = collections.Counter()
            verb_decls, verb_instances = [], []
            for path, node in walk_ast(tree):
                text = norm(node.value)
                b, fam, physical = bucket(path), text_family(path, text), id(node)
                is_decl = physical not in seen
                inst[b] += len(text)
                if is_decl:
                    decl[b] += len(text)
                    seen.add(physical)
                row = {'node': name, 'status': record['category'], 'path': field_path(path), 'bucket': b,
                       'family': fam, 'line': node.start_mark.line + 1, 'end_line': node.end_mark.line + 1,
                       'chars': len(text), 'declared_here': is_decl, 'text': text,
                       'snapshot': record['snapshot'], 'source_file': record['source_file']}
                if fam:
                    target_instances[fam] += len(text)
                    target_decl = physical not in seen_target
                    row['target_declaration'] = target_decl
                    if target_decl:
                        targets[fam] += len(text)
                        seen_target.add(physical)
                        row['occurrence_id'] = name + ':' + str(node.start_mark.index)
                        occurrence = {k: row[k] for k in ['occurrence_id', 'node', 'status', 'path', 'line', 'end_line', 'snapshot', 'source_file']}
                        row['unit_id'] = add_unit(fam, text, occurrence)
                    if fam == 'pipeline.verb':
                        verb_instances.append(len(text))
                        if target_decl:
                            verb_decls.append(len(text))
                scalars.append(row)
            def provider(n, region, seen_names=()):
                if n in seen_names:
                    raise ValueError('cyclic provider chain')
                v = raw[n].get('design', {}).get(region)
                if isinstance(v, dict) and 'from_parent' in v:
                    return provider(v['from_parent'], region, seen_names + (n,))
                return n, v
            effective, origins = {}, {}
            err = None
            try:
                for region in REGIONS:
                    origin, value = provider(name, region)
                    origins[region] = origin
                    effective[region] = sum(len(norm(x)) for x in ordinary_strings(value))
            except (KeyError, ValueError) as e:
                err = str(e)
            metric = {'node': name, 'status': record['category'], 'version': record.get('node_version'),
                      'sha256': record['sha256'], 'bytes': record['bytes'], 'lines': len(body.splitlines()),
                      'raw_string_declaration_chars': dict(decl), 'raw_string_instance_chars': dict(inst),
                      'open_text_declaration_chars': dict(targets), 'open_text_instance_chars': dict(target_instances),
                      'verb_declaration_count': len(verb_decls), 'verb_instance_count': len(verb_instances),
                      'verb_declaration_chars': sum(verb_decls), 'verb_instance_chars': sum(verb_instances),
                      'verb_instance_median': statistics.median(verb_instances) if verb_instances else 0,
                      'verb_instance_max': max(verb_instances, default=0),
                      'effective_design_string_instance_chars': effective, 'effective_provider': origins,
                      'effective_resolution_error': err}
            metrics.append(metric)
        md = REPO / op['dispatch_snapshot']
        lines = md.read_text().splitlines()
        paragraphs, start = [], None
        def emit_md():
            nonlocal paragraphs, start
            if paragraphs:
                text = norm(' '.join(paragraphs))
                add_unit('dispatch.md', text, {'occurrence_id': 'dispatch.md:' + str(start), 'node': 'dispatch.md',
                    'status': 'navigation', 'path': 'dispatch.md', 'line': start, 'end_line': start + len(paragraphs)-1,
                    'snapshot': op['dispatch_snapshot'], 'source_file': op['dispatch_source']})
            paragraphs, start = [], None
        for i, line in enumerate(lines, 1):
            if not line.strip():
                emit_md()
            elif line.lstrip().startswith('|'):
                emit_md()
                if not re.fullmatch(r'[\s|:\-]+', line):
                    start, paragraphs = i, [line]
                    emit_md()
            else:
                if start is None:
                    start = i
                paragraphs.append(line)
        emit_md()
        ordered_units = sorted(units.values(), key=lambda u: (u['family'], u['id']))
        out = BASE / 'data' / opid
        jsonl(out / 'scalars.jsonl', scalars)
        jsonl(out / 'units.jsonl', ordered_units)
        write_json(out / 'nodes.json', metrics)
        index_lines = [f'# {opid}: full open-text inventory', '',
            'These are exact-text groups, not proof of semantic equivalence across contexts.', '']
        for u in ordered_units:
            index_lines += [f"## {u['id']} | {u['family']} | {u['chars']} chars | {len(u['occurrences'])} declarations", '',
                '; '.join(f"{x['node']}:{x['line']} {x['path']}" for x in u['occurrences']), '', u['text'], '']
        (out / 'TEXT_INVENTORY.md').write_text('\n'.join(index_lines))
        status_counts = collections.Counter(n['category'] for n in op['nodes'])
        summary = {'group': op['group'], 'planned_nodes': op['planned_nodes'], 'status_counts': dict(status_counts),
                   'semantic_yaml_count': len(metrics), 'scaffold_only_count': sum(n['material_present'] and not n['semantic_eligible'] for n in op['nodes']),
                   'unique_text_units': len(units), 'unique_text_chars': sum(u['chars'] for u in units.values()),
                   'raw_open_text_declaration_chars': sum(u['chars'] * len(u['occurrences']) for u in units.values() if u['family'] != 'dispatch.md'),
                   'dispatch_md_chars': sum(u['chars'] * len(u['occurrences']) for u in units.values() if u['family'] == 'dispatch.md'),
                   'family_unique_counts': dict(collections.Counter(u['family'] for u in units.values())),
                   'root': next(m for m in metrics if m['node'] == 'root')}
        all_metrics[opid] = summary
        write_json(out / 'summary.json', summary)
    write_json(BASE / 'data/census_summary.json', all_metrics)
    return all_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true', help='Create the immutable corpus at the beginning of the audit.')
    args = parser.parse_args()
    manifest = freeze() if args.freeze else json.loads((BASE / 'data/manifest.json').read_text())
    stats = run_census(manifest)
    for name, s in stats.items():
        print(name, 'yaml=', s['semantic_yaml_count'], 'units=', s['unique_text_units'],
              'unique_chars=', s['unique_text_chars'], 'declared_chars=', s['raw_open_text_declaration_chars'])


if __name__ == '__main__':
    main()
