"""Finite W2 CPU evidence: existing checker, raw/preview equality and resolver closure.

Reads only the nine approved directories and their fixed source files. No model,
hardware, extraction, monitor, repository mutation or historical receipt write.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
from tools import schema

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'schema_discussion/main_integration_20261122/round2_c'
MAP = json.loads((ROOT / 'schema_discussion/main_integration_20261122/execution_map.json').read_text())
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
           PYTHONPATH=f'{ROOT}/schema_template:{ROOT}',
           PATH=f'{Path(sys.executable).parent}:/usr/bin:/bin')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check(entry):
    directory = ROOT / entry['directory']
    before = {name: sha(directory / name) for name in entry['baseline_files']}
    command = entry['check_command']
    result = subprocess.run(command, cwd=ROOT, env=ENV, capture_output=True, text=True)
    log = OUT / f"check_{entry['operator_id']}.log"
    log.write_text(result.stdout + result.stderr)
    row = {'directory': entry['directory'], 'operator_id': entry['operator_id'],
           'command': command, 'cwd': str(ROOT), 'exit_code': result.returncode,
           'read_only': True, 'skip_hardware': True,
           'output': {'path': str(log.relative_to(ROOT)), 'sha256': sha(log)},
           'actual_file_sha256': before}
    if result.returncode == 0:
        plan = schema.normalize_directory(directory, skip_hardware=True)
        equality = {name + '.yaml': plan.files[name].read_bytes() == document.source.encode('utf-8')
                    for name, document in plan.documents.items()}
        assert len(equality) == len(plan.files)
        row['raw_bytes_equal_preview'] = equality
        row['raw_preview_hashes'] = {name + '.yaml': {
            'raw_sha256': sha(plan.files[name]),
            'preview_sha256': hashlib.sha256(document.source.encode('utf-8')).hexdigest(),
            'equal': equality[name + '.yaml']} for name, document in plan.documents.items()}
        assert all(equality.values()), equality
        files, docs = schema.read_directory(directory)
        resolved = schema.resolve_documents(files, docs)
        schema.validate_dispatch(files)
        row['dispatch_check'] = {'api': 'tools.schema.validate_dispatch', 'passed': True}
        row['dependencies'] = {
            name + '.yaml': {
                'implicit_contract_provider': None if name == 'root' else 'root.yaml',
                'nodes': sorted(node + '.yaml' for node in resolved.dependency_nodes[name]),
                'source_manifest': resolved.source_manifest(name),
                'region_origins': resolved.origins_dict(name),
                'effective_identity': resolved.effective_identity(name),
            } for name in files}
    assert before == {name: sha(directory / name) for name in entry['baseline_files']}
    return row

if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(check, MAP['w2']['affected_complete_directories']))
    result = {'format': 'main_integration.w2_structure_checks.v1', 'interpreter': sys.executable,
              'environment': {key: ENV[key] for key in ('PYTHONDONTWRITEBYTECODE','PYTHONPATH','PATH')},
              'directories': rows, 'passed': all(row['exit_code'] == 0 for row in rows),
              'scope_limits': 'Finite structure and dispatch checks, with hardware explicitly skipped; not numerical or GPU validation.'}
    (OUT / 'structure_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for row in rows:
        print(row['operator_id'], 'exit', row['exit_code'], flush=True)
    sys.exit(0 if result['passed'] else 1)
