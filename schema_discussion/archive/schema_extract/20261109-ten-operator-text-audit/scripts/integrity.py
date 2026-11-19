#!/usr/bin/env python3
"""Final file identity/link checks, independent of semantic annotation judgments."""
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((BASE / 'data/manifest.json').read_text())
    errors, input_count, unchanged, original_changed = [], 0, [], []
    for opid, op in manifest['operators'].items():
        records = [(n['snapshot'], n['source_file'], n['sha256']) for n in op['nodes'] if n['material_present']]
        records.append((op['dispatch_snapshot'], op['dispatch_source'], op['dispatch_sha256']))
        for snapshot, original, expected in records:
            input_count += 1
            if not (REPO / snapshot).is_file() or sha(REPO / snapshot) != expected:
                errors.append('snapshot identity mismatch: ' + snapshot)
            if (REPO / original).is_file() and sha(REPO / original) == expected:
                unchanged.append(original)
            else:
                original_changed.append(original)
        for suffix, directory in [('.md', 'operators'), ('.json', 'annotations')]:
            if not (BASE / directory / (opid + suffix)).is_file():
                errors.append('missing deliverable: ' + directory + '/' + opid + suffix)
    for record in manifest['rules']:
        if sha(REPO / record['snapshot']) != record['sha256']:
            errors.append('frozen rule mismatch: ' + record['snapshot'])
    docs = [BASE / 'README.md', BASE / 'METHOD.md'] + sorted((BASE / 'operators').glob('*.md'))
    links = 0
    for doc in docs:
        if not doc.is_file():
            errors.append('missing document: ' + str(doc.relative_to(BASE)))
            continue
        for target in re.findall(r'\]\(([^)]+)\)', doc.read_text()):
            target = target.strip('<>')
            if '://' in target or target.startswith(('mailto:', '#')):
                continue
            target = unquote(target.split('#', 1)[0])
            target = re.sub(r':\d+(?:[-–]\d+)?$', '', target)
            if not target:
                continue
            links += 1
            path = Path(target) if Path(target).is_absolute() else doc.parent / target
            # This report is created at the end of the current verification run.
            if path.resolve() == (BASE / 'supervision/final_integrity.json').resolve():
                continue
            if not path.exists():
                errors.append(f'broken link in {doc.relative_to(BASE)}: {target}')
    summaries = {}
    for opid in manifest['operators']:
        path = BASE / 'data' / opid / 'audit_summary.json'
        if path.is_file():
            summary = json.loads(path.read_text())
            summaries[opid] = summary.get('coverage', {})
            if not summary.get('valid'):
                errors.append('invalid annotation summary: ' + opid)
        else:
            errors.append('missing annotation summary: ' + opid)
    result = {'ok': not errors, 'snapshot_files_verified': input_count,
              'original_files_unchanged': len(unchanged), 'original_files_changed_since_freeze': original_changed,
              'frozen_rule_files': len(manifest['rules']), 'document_links_checked': links,
              'operator_coverage': summaries, 'errors': errors}
    (BASE / 'supervision/final_integrity.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
