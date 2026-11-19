#!/usr/bin/env python3
"""Validate complete manual coverage and derive auditable aggregate statistics."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[2]
CATEGORIES = {'DUPLICATE', 'NONCORE', 'STRUCTURABLE', 'ESSENTIAL_TEXT', 'FORMAL',
              'SOURCE_LIMIT', 'EXPLANATION', 'MIXED', 'UNRESOLVED'}
CORES = {'none', 'some', 'all', 'uncertain'}
STRUCTURES = {'already_present', 'existing_structure', 'open_semantics', 'new_structure_candidate',
              'formula_or_predicate', 'mixed', 'not_applicable', 'uncertain'}
KINDS = {'redundancy', 'noncore', 'existing_structure', 'structural_gap', 'source_limitation', 'correctness', 'style'}
GAPS = {'none', 'open_semantics', 'representation_cost', 'missing_field_or_event', 'unresolved'}


def norm(text):
    return re.sub(r'\s+', ' ', text).strip()


def load_lines(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def check_evidence(evidence, errors, where):
    if not isinstance(evidence, list) or not evidence:
        errors.append(f'{where}: needs at least one evidence anchor')
        return
    for item in evidence:
        if not isinstance(item, str):
            errors.append(f'{where}: evidence must be a path:line string')
            continue
        match = re.fullmatch(r'(.+?)(?::|#L)(?:L)?(\d+)(?:[-–](?:L)?(\d+))?', item)
        if not match:
            errors.append(f'{where}: malformed evidence {item!r}')
            continue
        path, start, end = match.group(1), int(match.group(2)), int(match.group(3) or match.group(2))
        if path.startswith('git:'):
            try:
                _, ref, source = path.split(':', 2)
                raw = subprocess.check_output(['git', 'show', ref + ':' + source], cwd=REPO, stderr=subprocess.DEVNULL)
                length = len(raw.splitlines())
            except (ValueError, subprocess.CalledProcessError):
                errors.append(f'{where}: missing git evidence {item}')
                continue
        else:
            choices = [Path(path)] if Path(path).is_absolute() else [BASE / path, REPO / path]
            found = next((p for p in choices if p.is_file()), None)
            if found is None:
                errors.append(f'{where}: missing evidence file {item}')
                continue
            length = len(found.read_text(errors='replace').splitlines())
        if not (1 <= start <= end <= length):
            errors.append(f'{where}: evidence line outside file {item}, length={length}')


def validate_operator(opid):
    manifest = json.loads((BASE / 'data/manifest.json').read_text())['operators'][opid]
    units_list = load_lines(BASE / 'data' / opid / 'units.jsonl')
    units = {x['id']: x for x in units_list}
    path = BASE / 'annotations' / (opid + '.json')
    report = json.loads(path.read_text())
    annotations = report.get('unit_annotations', [])
    ids = [x.get('id') for x in annotations]
    errors = []
    if report.get('operator') != opid:
        errors.append('operator identity mismatch')
    if report.get('prompt_version') != 'v1.1':
        errors.append('prompt_version must be v1.1 (all rounds use the final calibrated scope)')
    duplicates = [x for x, c in Counter(ids).items() if c > 1]
    if duplicates:
        errors.append('duplicate annotations: ' + str(duplicates))
    if set(ids) != set(units):
        errors.append(f"unit coverage missing={sorted(set(units)-set(ids))}, extra={sorted(set(ids)-set(units))}")
    findings = report.get('findings', [])
    finding_ids = [f.get('id') for f in findings]
    if len(finding_ids) != len(set(finding_ids)):
        errors.append('duplicate finding IDs')
    for finding in findings:
        name = str(finding.get('id'))
        if finding.get('kind') not in KINDS or finding.get('gap_kind') not in GAPS:
            errors.append(name + ': invalid finding kind/gap_kind')
        if not finding.get('claim') or not finding.get('unit_ids'):
            errors.append(name + ': requires claim and unit_ids')
        if not set(finding.get('unit_ids', [])) <= set(units):
            errors.append(name + ': unknown unit IDs')
        if finding.get('kind') == 'existing_structure' and not finding.get('existing_structural_candidate'):
            errors.append(name + ': existing-structure claim requires an actual candidate')
        if finding.get('kind') == 'structural_gap' and not finding.get('deletion_counterexample'):
            errors.append(name + ': structural-gap claim requires a deletion counterexample')
        check_evidence(finding.get('evidence'), errors, name)
    categories = defaultdict(Counter)
    category_by_status = defaultdict(lambda: defaultdict(Counter))
    family_categories = defaultdict(lambda: defaultdict(Counter))
    structures = defaultdict(Counter)
    core_counts = defaultdict(Counter)
    annotated_by_id = {}
    for ann in annotations:
        uid = ann.get('id')
        if uid not in units:
            continue
        annotated_by_id[uid] = ann
        unit = units[uid]
        cat, core, structure = ann.get('category'), ann.get('core'), ann.get('structure')
        if cat not in CATEGORIES or core not in CORES or structure not in STRUCTURES:
            errors.append(f'{uid}: invalid category/core/structure')
            continue
        if not ann.get('reason'):
            errors.append(uid + ': missing reason')
        if not ann.get('contexts_checked') and cat != 'UNRESOLVED':
            errors.append(uid + ': unexamined contexts must be UNRESOLVED')
        if not set(ann.get('finding_ids', [])) <= set(finding_ids):
            errors.append(uid + ': unknown finding ID')
        if cat in {'STRUCTURABLE', 'ESSENTIAL_TEXT'} and not ann.get('finding_ids'):
            errors.append(uid + ': substantive core classification needs a finding')
        check_evidence(ann.get('evidence'), errors, uid)
        categories[cat].update(unit_count=1, unique_chars=unit['chars'])
        family_categories[unit['family']][cat].update(unit_count=1, unique_chars=unit['chars'])
        structures[structure].update(unit_count=1, unique_chars=unit['chars'])
        core_counts[core].update(unit_count=1, unique_chars=unit['chars'])
        occurrences = {x['occurrence_id']: x for x in unit['occurrences']}
        override_list = ann.get('occurrence_overrides', [])
        overrides = {x['occurrence_id']: x for x in override_list}
        if len(overrides) != len(override_list):
            errors.append(uid + ': duplicate occurrence override')
        if not set(overrides) <= set(occurrences):
            errors.append(uid + ': override refers to unknown occurrence')
        for oid, override in overrides.items():
            if override.get('core') not in CORES or override.get('structure') not in STRUCTURES or not override.get('reason'):
                errors.append(uid + ': invalid override core/structure/reason')
            check_evidence(override.get('evidence'), errors, uid + '/' + oid)
        for oid, occ in occurrences.items():
            effective_cat = overrides.get(oid, ann).get('category')
            if effective_cat not in CATEGORIES:
                errors.append(uid + ': invalid override category')
                continue
            categories[effective_cat].update(declarations=1, declared_chars=unit['chars'])
            category_by_status[occ['status']][effective_cat].update(declarations=1, declared_chars=unit['chars'])
            family_categories[unit['family']][effective_cat].update(declarations=1, declared_chars=unit['chars'])
    expected_nodes = {n['node'] for n in manifest['nodes'] if n['semantic_eligible']}
    reviews = report.get('node_review', [])
    reviewed = {n['node'] for n in reviews if n.get('reviewed') and n.get('effective_binding_checked')}
    if reviewed != expected_nodes or len(reviews) != len(expected_nodes):
        errors.append(f'node coverage missing={sorted(expected_nodes-reviewed)}, extra={sorted(reviewed-expected_nodes)}')
    saved, covered, candidates = Counter(), defaultdict(list), []
    for i, candidate in enumerate(report.get('removal_candidates', [])):
        label = f'removal[{i}]'
        uid = candidate.get('unit_id')
        if uid not in units:
            errors.append(label + ': unknown unit')
            continue
        unit = units[uid]
        quote, replacement = candidate.get('quote', ''), candidate.get('replacement')
        if not quote or unit['text'].count(quote) != 1 or not isinstance(replacement, str):
            errors.append(label + ': quote must occur exactly once and replacement must be a string')
            continue
        start, end = unit['text'].index(quote), unit['text'].index(quote) + len(quote)
        gain = len(quote) - len(norm(replacement))
        if gain < 0:
            errors.append(label + ': replacement is longer; record structural proposals as findings')
        actual = {x['occurrence_id']: x for x in unit['occurrences']}
        applies = candidate.get('applies_to_occurrences')
        selected = list(actual) if applies == 'all' else applies
        if not isinstance(selected, list) or not selected or not set(selected) <= set(actual):
            errors.append(label + ': invalid occurrence coverage')
            continue
        selected = list(dict.fromkeys(selected))
        for oid in selected:
            key = (uid, oid)
            if any(start < b and a < end for a, b in covered[key]):
                errors.append(label + ': overlaps another candidate at ' + oid)
            covered[key].append((start, end))
            status = actual[oid]['status']
            saved[status] += gain
        check_evidence(candidate.get('evidence'), errors, label)
        candidates.append({'index': i, 'unit_id': uid, 'family': unit['family'], 'start': start, 'end': end,
                           'saved_chars_per_declaration': gain, 'declarations': len(selected),
                           'saved_declaration_chars': gain * len(selected), 'kind': candidate.get('kind')})
    scopes = defaultdict(Counter)
    unresolved_context_groups, uncertain_core_groups = set(), set()
    unresolved_declarations, uncertain_core_declarations = 0, 0
    for uid, unit in units.items():
        if uid not in annotated_by_id:
            continue
        ann = annotated_by_id[uid]
        family = unit['family']
        labels = ['all', 'dispatch_md' if family == 'dispatch.md' else 'yaml']
        if family.startswith(('schedule.', 'pipeline.', 'realization.')):
            labels.append('main_WTR')
        if family.startswith('math.'):
            labels.append('math')
        if family.startswith('hardware.'):
            labels.append('hardware')
        if family == 'dispatch.when':
            labels.append('dispatch_when')
        resolved_annotations = [next((o for o in ann.get('occurrence_overrides', [])
                              if o['occurrence_id'] == occ['occurrence_id']), ann) for occ in unit['occurrences']]
        core_bearing = any(o.get('core') in {'some', 'all'} and o.get('category') != 'FORMAL' for o in resolved_annotations)
        if any(o.get('category') == 'UNRESOLVED' for o in resolved_annotations):
            unresolved_context_groups.add(uid)
        if any(o.get('core') == 'uncertain' for o in resolved_annotations):
            uncertain_core_groups.add(uid)
        unresolved_declarations += sum(o.get('category') == 'UNRESOLVED' for o in resolved_annotations)
        uncertain_core_declarations += sum(o.get('core') == 'uncertain' for o in resolved_annotations)
        for label in labels:
            scopes[label].update(unique_units=1, unique_chars=unit['chars'],
                                 declarations=len(unit['occurrences']),
                                 declared_chars=unit['chars'] * len(unit['occurrences']))
            if core_bearing:
                scopes[label].update(nonformal_core_bearing_units=1)
            for occ in unit['occurrences']:
                override = next((o for o in ann.get('occurrence_overrides', [])
                                 if o['occurrence_id'] == occ['occurrence_id']), ann)
                if override.get('core') in {'some', 'all'} and override.get('category') != 'FORMAL':
                    scopes[label].update(nonformal_core_bearing_declarations=1,
                                         nonformal_core_bearing_declared_chars=unit['chars'])
    for candidate in candidates:
        labels = ['all', 'dispatch_md' if candidate['family'] == 'dispatch.md' else 'yaml']
        if candidate['family'].startswith(('schedule.', 'pipeline.', 'realization.')):
            labels.append('main_WTR')
        if candidate['family'].startswith('math.'):
            labels.append('math')
        if candidate['family'].startswith('hardware.'):
            labels.append('hardware')
        if candidate['family'] == 'dispatch.when':
            labels.append('dispatch_when')
        for label in labels:
            scopes[label].update(candidate_saved_declaration_chars=candidate['saved_declaration_chars'])
    result = {'operator': opid, 'valid': not errors, 'errors': errors,
              'coverage': {'expected_units': len(units), 'annotations': len(annotations),
                           'expected_nodes': len(expected_nodes), 'reviewed_nodes': len(reviewed),
                           'unresolved_units': categories['UNRESOLVED'].get('unit_count', 0),
                           'groups_with_unresolved_contexts': len(unresolved_context_groups),
                           'unresolved_declarations': unresolved_declarations,
                           'groups_with_uncertain_core_importance': len(uncertain_core_groups),
                           'uncertain_core_declarations': uncertain_core_declarations},
              'category_statistics': dict(categories), 'category_by_status': dict(category_by_status),
              'family_category_statistics': dict(family_categories), 'structure_statistics': dict(structures),
              'core_presence_statistics': dict(core_counts), 'finding_count': len(findings),
              'scope_statistics': dict(scopes),
              'finding_kinds': dict(Counter(f.get('kind') for f in findings)),
              'gap_kinds': dict(Counter(f.get('gap_kind') for f in findings)),
              'candidate_count': len(candidates), 'candidate_saved_chars_by_status': dict(saved),
              'candidate_saved_declaration_chars': sum(saved.values()), 'candidates': candidates,
              'warning': 'Category character areas are not removable/irreducible semantic bytes. Savings are proposed prose edits, not independently certified semantic equivalence.'}
    write_json(BASE / 'data' / opid / 'audit_summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--operator')
    args = parser.parse_args()
    manifest = json.loads((BASE / 'data/manifest.json').read_text())
    names = [args.operator] if args.operator else list(manifest['operators'])
    reports = {}
    failed = False
    for opid in names:
        try:
            result = validate_operator(opid)
        except (OSError, ValueError, KeyError, TypeError) as error:
            result = {'operator': opid, 'valid': False, 'errors': [str(error)]}
        reports[opid] = result
        print(opid, 'VALID' if result['valid'] else 'INVALID', result.get('coverage', {}))
        for e in result.get('errors', [])[:30]:
            print(' ', e)
        failed |= not result['valid']
    if not args.operator:
        write_json(BASE / 'data/audit_summary.json', reports)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
