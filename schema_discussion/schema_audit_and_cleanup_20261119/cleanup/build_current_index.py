"""Write the current six-operator manifests after the recorded directory checks."""
from pathlib import Path
import hashlib
import json
import subprocess
import yaml

ROOT = Path(__file__).resolve().parents[3]
CASE = Path('schema_discussion/schema_audit_and_cleanup_20261119')
CLEAN = CASE / 'cleanup'
OPS = Path('schema_wiki/aka-cuda-operators-20261030/operators')
COMMIT = '24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2'
OPERATORS = {
    'gemm': 'gemm/grouped/fp8_scaled_grouped_gemm',
    'moe': 'moe/nvfp4_fused_moe',
    'mla': 'mla/mla_expanded_fp8',
    'gdn': 'linear_attention/gdn_prefill_bf16_state_bf16',
    'fmha': 'fmha/fmha_mxfp8_qkv',
    'sparse': 'sparse_attention/dual_cache_latent_bf16',
}


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def pointer(path):
    return {'path': str(path), 'sha256': sha(path)}


def write(path, value):
    (ROOT / path).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


entries, checks, source_checks = [], [], {}
before = json.loads((ROOT / CLEAN / 'BEFORE.json').read_text())
for short, relative in OPERATORS.items():
    directory = OPS / relative
    operator = directory.name
    if short in ['fmha', 'sparse']:
        review = Path('schema_discussion/issue_optimization_20261119/schema_audits') / short / 'after.json'
        report = Path(f'schema_discussion/issue_optimization_20261119/SCHEMA_AUDIT_{short.upper()}.md')
    else:
        review = CASE / 'reviews' / short / 'after.json'
        report = CASE / 'reviews' / short / 'REVIEW.md'
    reviewed = json.loads((ROOT / review).read_text())
    files = {}
    for p in sorted((ROOT / directory).glob('*.yaml')):
        relative_path = p.relative_to(ROOT)
        metadata = yaml.safe_load(p.read_text())['META_DATA']
        digest = sha(relative_path)
        assert digest == before['current_files'][str(relative_path)]
        assert digest == reviewed['manifest'][p.name]
        reviewed_version = reviewed['versions'].get(p.name, reviewed['versions'].get(p.stem))
        if isinstance(reviewed_version, dict):
            reviewed_version = reviewed_version['after']
        assert metadata['Version'] == reviewed_version
        source = metadata['Source']
        assert source.startswith('KernelSchema@' + COMMIT + ':')
        source_path = source.split(':', 1)[1].split('::', 1)[0]
        if source_path not in source_checks:
            fixed = subprocess.check_output(['git', 'show', f'{COMMIT}:{source_path}'], cwd=ROOT)
            assert fixed == (ROOT / source_path).read_bytes(), source_path
            source_checks[source_path] = hashlib.sha256(fixed).hexdigest()
        files[p.name] = {'sha256': digest, 'version': metadata['Version'], 'source': source,
                         'source_file_sha256': source_checks[source_path]}
    files['dispatch.md'] = {'sha256': sha(directory / 'dispatch.md')}
    log = CLEAN / f'check-{short}.log'
    text = (ROOT / log).read_text()
    assert f'Normalization preview: {len(files)-1} YAML; 0 would change; no files written' in text
    check = {
        'command': ['/opt/venv/python3.13.12/bin/python3', '-B', 'schema_template/check_yaml.py', str(directory / 'root.yaml'), '--read-only', '--skip-hardware'],
        'exit_code': 0, 'yaml_count': len(files)-1, 'normalization_changes': 0, 'output': pointer(log),
        'manifest': {name: record['sha256'] for name, record in files.items()},
    }
    checks.append({'operator_id': operator, **check})
    receipt = Path('schema_discussion/archive/schema_wiki/extraction_records') / operator / 'receipt.json'
    current = {
        'format': 'schema_wiki.current_schema_manifest.v1',
        'path_base': 'repository_root',
        'operator_id': operator,
        'files': files,
        'acceptance': {
            'basis': 'direct_manual_source_review_and_read_only_structure_check',
            'runner_p6_receipt': False,
            'semantic_evidence': pointer(review),
            'semantic_report': str(report),
            'reviewed_yaml_bytes_unchanged': True,
            'dispatch_editorial_changes': pointer(CLEAN / 'content_changes.json'),
            'structure_check': {'exit_code': 0, 'read_only': True, 'skip_hardware': True, 'output': pointer(log)},
            'scope_limits': 'Hardware placeholders and source/comparator/caller limitations remain in YAML and dispatch; these checks do not establish GPU or full-domain numerical correctness.',
        },
        'historical_receipt': {**pointer(receipt), 'coverage': 'only_the_original_files_named_in_that_receipt'},
    }
    manifest = directory / '.extraction/current.json'
    write(manifest, current)
    entries.append({'operator_id': operator, 'directory': str(directory), 'design_count': len(files)-1,
                    'files': files, 'current_metadata': pointer(manifest), 'semantic_evidence': pointer(review),
                    'semantic_report': str(report), 'historical_receipt': pointer(receipt)})
write(CLEAN / 'directory_checks.json', checks)
write(CLEAN / 'source_references.json', {'commit': COMMIT, 'checked_current_against_git': source_checks})
write(CASE / 'CURRENT_SCHEMA_INDEX.json', {
    'format': 'schema_wiki.current_schema_index.v1', 'path_base': 'repository_root',
    'operator_count': len(entries), 'design_count': sum(x['design_count'] for x in entries),
    'acceptance_basis': 'manual_source_review_and_read_only_structure_check',
    'runner_p6_for_current_bytes': False, 'hardware_validation': False, 'gpu_validation': False,
    'operators': entries, 'directory_checks': pointer(CLEAN / 'directory_checks.json'),
    'historical_locations': str(CLEAN / 'LOCATIONS.json'),
})
print('Wrote six current manifests and 15-design index; YAML/Source/review hashes match.')
