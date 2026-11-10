#!/usr/bin/env python3
"""Combine already validated operator annotations; no automatic semantic labeling."""
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

BASE = Path(__file__).resolve().parents[1]


def csv_out(path, rows):
    if not rows:
        return
    columns = list(rows[0])
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    census = json.loads((BASE / 'data/census_summary.json').read_text())
    rows, categories, findings, cohort = [], [], [], {}
    for op, c in census.items():
        summary_path = BASE / 'data' / op / 'audit_summary.json'
        if not summary_path.is_file():
            continue
        a = json.loads(summary_path.read_text())
        if not a.get('valid'):
            continue
        raw = json.loads((BASE / 'annotations' / (op + '.json')).read_text())
        units = {x['id']: x for x in map(json.loads, (BASE / 'data' / op / 'units.jsonl').open())}
        main_stats = a['scope_statistics']['main_WTR']
        root_nodes = json.loads((BASE / 'data' / op / 'nodes.json').read_text())
        root = next(x for x in root_nodes if x['node'] == 'root')
        root_wtr = sum(v for k, v in root['open_text_declaration_chars'].items()
                       if k.startswith(('schedule.', 'pipeline.', 'realization.')))
        rows.append({'operator': op, 'group': c['group'], 'planned_nodes': c['planned_nodes'],
            'authored_nodes': c['semantic_yaml_count'], 'unique_text_units': c['unique_text_units'],
            'unique_text_chars': c['unique_text_chars'],
            'yaml_open_declaration_chars': c['raw_open_text_declaration_chars'],
            'dispatch_md_chars': c['dispatch_md_chars'],
            'root_WTR_open_chars': root_wtr,
            'root_verb_median': root['verb_instance_median'],
            'WTR_groups': main_stats['unique_units'],
            'WTR_unique_chars': main_stats['unique_chars'],
            'WTR_declarations': main_stats['declarations'],
            'WTR_declaration_chars': main_stats['declared_chars'],
            'WTR_nonformal_core_groups': main_stats.get('nonformal_core_bearing_units', 0),
            'WTR_nonformal_core_declarations': main_stats.get('nonformal_core_bearing_declarations', 0),
            'WTR_core_containing_char_area': main_stats.get('nonformal_core_bearing_declared_chars', 0),
            'proposed_yaml_saved_chars': a['scope_statistics']['yaml'].get('candidate_saved_declaration_chars', 0),
            'proposed_WTR_saved_chars': main_stats.get('candidate_saved_declaration_chars', 0),
            'proposed_dispatch_saved_chars': a['scope_statistics']['dispatch_md'].get('candidate_saved_declaration_chars', 0),
            'removal_candidates': a['candidate_count'], 'findings': a['finding_count'],
            'unresolved_groups': a['coverage']['unresolved_units'],
            'groups_with_unresolved_contexts': a['coverage']['groups_with_unresolved_contexts'],
            'unresolved_declarations': a['coverage']['unresolved_declarations'],
            'groups_with_uncertain_core_importance': a['coverage']['groups_with_uncertain_core_importance']})
        for cat, stats in a['category_statistics'].items():
            categories.append({'operator': op, 'category': cat, **{k: stats.get(k, 0) for k in
                ['unit_count', 'unique_chars', 'declarations', 'declared_chars']}})
        for finding in raw['findings']:
            ids = set(finding['unit_ids'])
            occurrences = [o for uid in ids for o in units[uid]['occurrences']]
            findings.append({'operator': op, **finding, 'unique_group_count': len(ids),
                'declaration_count': len(occurrences),
                'yaml_node_count': len({o['node'] for o in occurrences if o['node'] != 'dispatch.md'}),
                'unique_char_area': sum(units[uid]['chars'] for uid in ids),
                'declaration_char_area': sum(units[uid]['chars'] * len(units[uid]['occurrences']) for uid in ids)})
    for group in ['template', 'extract']:
        members = [r for r in rows if r['group'] == group]
        totals = Counter()
        for row in members:
            totals.update({k: v for k, v in row.items() if isinstance(v, (int, float)) and k != 'root_verb_median'})
        cohort[group] = {'operators': len(members), 'sums': dict(totals),
            'median_of_operator_root_verb_medians': statistics.median(r['root_verb_median'] for r in members) if members else None,
            'median_of_operator_WTR_core_group_fractions': statistics.median(r['WTR_nonformal_core_groups'] / r['WTR_groups'] for r in members) if members else None,
            'note': 'Large trees dominate sums. Compare each operator and root separately; candidate savings are not exhaustive lossless compression rates.'}
    csv_out(BASE / 'data/operator_statistics.csv', rows)
    csv_out(BASE / 'data/category_statistics.csv', categories)
    (BASE / 'data/findings_catalog.json').write_text(json.dumps(findings, ensure_ascii=False, indent=2) + '\n')
    (BASE / 'data/cohort_statistics.json').write_text(json.dumps(cohort, ensure_ascii=False, indent=2) + '\n')
    total_keys = ['authored_nodes', 'planned_nodes', 'unique_text_units', 'unique_text_chars',
        'yaml_open_declaration_chars', 'dispatch_md_chars', 'WTR_groups', 'WTR_unique_chars',
        'WTR_declarations', 'WTR_declaration_chars', 'WTR_nonformal_core_groups',
        'WTR_nonformal_core_declarations', 'WTR_core_containing_char_area',
        'proposed_yaml_saved_chars', 'proposed_WTR_saved_chars', 'proposed_dispatch_saved_chars',
        'removal_candidates', 'findings', 'groups_with_unresolved_contexts',
        'unresolved_declarations', 'groups_with_uncertain_core_importance']
    totals = {key: sum(row[key] for row in rows) for key in total_keys}
    category_totals = {}
    for row in categories:
        target = category_totals.setdefault(row['category'], Counter())
        target.update({key: row[key] for key in ['unit_count', 'unique_chars', 'declarations', 'declared_chars']})
    key_results = {'totals': totals, 'categories': category_totals,
        'finding_kinds': dict(Counter(f['kind'] for f in findings)),
        'gap_kinds': dict(Counter(f['gap_kind'] for f in findings))}
    (BASE / 'data/key_results.json').write_text(json.dumps(key_results, ensure_ascii=False, indent=2) + '\n')
    print('validated operators', len(rows), 'findings', len(findings),
          'annotated unique groups', sum(r['unique_text_units'] for r in rows))


if __name__ == '__main__':
    main()
