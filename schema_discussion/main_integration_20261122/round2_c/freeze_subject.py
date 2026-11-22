"""Freeze the finite W2 review subject after CPU checks; never create current."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import yaml
from tools import text_policy

ROOT = Path(__file__).resolve().parents[3]
REC = ROOT / 'schema_discussion/main_integration_20261122'
OUT = REC / 'round2_c'
EM = json.loads((REC / 'execution_map.json').read_text())
BASE = json.loads((OUT / 'baseline.json').read_text())
CHECKS = json.loads((OUT / 'structure_checks.json').read_text())

def digest(data):
    return hashlib.sha256(data).hexdigest()

def file_record(path):
    p = ROOT / path
    return {'path': path, 'sha256': digest(p.read_bytes()), 'bytes': p.stat().st_size}

def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

def source_record(source):
    repo_ref, location = source.split(':', 1)
    repository, ref = repo_ref.split('@', 1)
    path, symbol = location.split('::', 1)
    assert repository == 'KernelSchema' and ref == '484902855fe7be82a74fe7c394503d1d7d222dbb'
    assert not {'archive', 'AKA_experiment'} & set(Path(path).parts)
    p = ROOT / path
    assert all(not q.is_symlink() for q in [p, *p.parents])
    blob = subprocess.check_output(['git', 'show', f'{ref}:{path}'], cwd=ROOT)
    assert p.read_bytes() == blob
    return {'locator': source, 'repository': repository, 'ref': ref, 'path': path,
            'symbol': symbol, 'sha256': digest(blob)}

assert CHECKS['passed'] and len(CHECKS['directories']) == 9
assert sum(len(row['raw_bytes_equal_preview']) for row in CHECKS['directories']) == 19
assert all(all(row['raw_bytes_equal_preview'].values()) for row in CHECKS['directories'])
changed = {row['path']: row for row in EM['w2']['required_yaml_amendments']}
changed_dispatch = set(EM['w2']['required_dispatch_amendments'])
dirs = []
for entry, check in zip(EM['w2']['affected_complete_directories'], CHECKS['directories']):
    assert check['directory'] == entry['directory']
    files = {}
    for name, old in entry['baseline_files'].items():
        path = entry['directory'] + '/' + name
        record = file_record(path)
        assert record['sha256'] == check['actual_file_sha256'][name]
        record.update(baseline_sha256=old['sha256'], changed=path in changed or path in changed_dispatch)
        if not record['changed']:
            assert record['sha256'] == old['sha256']
        if name.endswith('.yaml'):
            data = yaml.safe_load((ROOT / path).read_text())
            meta = data['META_DATA']
            assert meta['Source'] == old['source']
            assert meta['Version'] == (changed[path]['suggested_next_version'] if path in changed else old['version'])
            source = source_record(meta['Source'])
            assert source['sha256'] == old['source_file_sha256']
            explicit = [{'region':'DESIGN.' + k, 'provider':v['from']}
                        for k,v in data.get('DESIGN',{}).items() if isinstance(v,dict) and 'from' in v]
            assert explicit == old['explicit_from']
            record.update(version=meta['Version'], source=meta['Source'], source_file_sha256=source['sha256'],
                          fixed_source=source, implicit_contract_provider=check['dependencies'][name]['implicit_contract_provider'],
                          explicit_from=explicit, dependency_closure=check['dependencies'][name])
            if 'CONTRACT' in data:
                record['golden_reference_source'] = source_record(data['CONTRACT']['Algorithm_requirement']['Golden_reference']['Source'])
        files[name] = record
    receipt = entry['historical_receipt']
    assert file_record(receipt['path'])['sha256'] == receipt['sha256']
    dirs.append({'operator_id':entry['operator_id'], 'directory':entry['directory'], 'files':files,
                 'structure_check':check, 'historical_receipt':receipt})

preserved = BASE['historical_inventory']['preserved_files']
unchanged = []
for path, old in preserved.items():
    if path in changed or path in changed_dispatch:
        continue
    actual = file_record(path)
    assert actual['sha256'] == old, path
    unchanged.append(actual)
assert sum(p['path'].endswith('.yaml') for p in unchanged) == 24
assert sum(p['path'].endswith('/receipt.json') for p in unchanged) == 12

criteria = []
for op in EM['w2']['tolerance_sources']:
    golden = yaml.safe_load((ROOT/op['root']).read_text())['CONTRACT']['Algorithm_requirement']['Golden_reference']
    for c in op['criteria']:
        source = ROOT/c['source_file']
        assert all(not q.is_symlink() for q in [source,*source.parents])
        assert digest(source.read_bytes()) == c['source_sha256']
        value = json.loads(source.read_text())
        for part in c['first_pointer'].strip('/').split('/'):
            value = value[int(part)] if isinstance(value,list) else value[part]
        assert value == c['values']
    outputs = []
    for name,record in golden['Tolerance'].items():
        c = op['criteria'][1] if name == 'attention_lse' else op['criteria'][0]
        for field,value in c['values'].items():
            exact = f'{field}={json.dumps(value) if not isinstance(value,str) else value}'
            assert exact in record, (name, exact, record)
        assert 'Source only' in record and 'comparator definition unavailable' in record and 'no defaults' in record
        assert text_policy.budget_message(record,300,2) is None
        outputs.append({'output':name,'text':record,'unicode_characters':len(record),
                        'logical_lines':len(record.splitlines()),'source_record':c})
    criteria.append({'operator_id':op['operator_id'],'root':op['root'],'outputs':outputs})
assert sum(len(o['outputs']) for o in criteria) == 9
old_policy = subprocess.check_output(['git','show','f46b14a7ccf58bcb623f781471321a6bf97b765f:schema_template/tools/text_policy.py'],cwd=ROOT).decode()
assert (ROOT/'schema_template/tools/text_policy.py').read_text() == old_policy.replace(
    'Budget(A + ("Golden_reference", "Tolerance", None), 100, 1)',
    'Budget(A + ("Golden_reference", "Tolerance", None), 300, 2)')
write(OUT/'tolerance_and_preservation.json', {'format':'main_integration.w2_tolerance_preservation.v1',
    'tolerance_records':criteria,'other_budgets_unchanged':True,'unchanged_originals':unchanged,
    'counts':{'source_json':5,'tolerance_outputs':9,'unchanged_yaml':24,'unchanged_receipts':12},
    'interpretation':'Source records retain all labels and explicit null; no comparator or default semantics inferred.'})

test_paths = EM['ownership']['C']['test_paths']
tests = {'format':'main_integration.w2_template_tests.v1','interpreter':sys.executable,
    'command':[sys.executable,'-B','-m','unittest','tools.tests.test_yaml','tools.tests.test_schema','tools.tests.test_dispatch','tools.tests.test_design_resolution'],
    'cwd':str(ROOT),'environment':CHECKS['environment'],'exit_code':0,'tests':64,
    'stdout_stderr':file_record(str((OUT/'template_tests.log').relative_to(ROOT))),
    'capture_note':'Tool stdout/stderr chunks concatenated in order; test process ran once.',
    'test_inputs':[file_record(p) for p in test_paths],
    'assertions':['Complete six-field source record accepted','300 Unicode characters accepted',
                  '300 characters across two logical lines accepted','301 characters rejected',
                  'Three logical lines including interior blank rejected','Unknown output remains rejected','Gap retains 100/1 budget'],
    'not_run':['pytest','GPU','model','Extract','monitor','Runner P6']}
write(OUT/'template_tests.json',tests)

tracked = subprocess.check_output(['git','ls-files','schema_template','schema_docs'],cwd=ROOT,text=True).splitlines()
rule_paths = [p for p in tracked if p.startswith('schema_docs/') or p == 'schema_template/check_yaml.py'
              or p.startswith('schema_template/tools/') and '/tests/' not in p and p.endswith('.py')]
rule_paths += EM['ownership']['C']['product_paths'][4:6]
rule_records = [file_record(p) for p in sorted(set(rule_paths))]
subject = {'format':'main_integration.w2_subject.v1','path_base':'repository_root','status':'frozen_for_independent_review',
    'baseline_commit':subprocess.check_output(['git','rev-parse','41b7db0'],cwd=ROOT,text=True).strip(),
    'head_before_w2':'f46b14a7ccf58bcb623f781471321a6bf97b765f','owner':'C','independent_reviewer':'D',
    'counts':EM['w2']['counts'],'directories':dirs,'rules_and_checker':rule_records,
    'evidence':[file_record('schema_discussion/main_integration_20261122/round2_c/'+n)
                for n in ['baseline.json','candidate_normalization.json','structure_checks.json','check_subject.py',
                          'template_tests.json','template_tests.log','tolerance_and_preservation.json','freeze_subject.py',
                          'independent_review_correction.json']],
    'reused_finite_evidence':EM['w2']['reuse_evidence'],
    'tolerance_sources':EM['w2']['tolerance_sources'],
    'acceptance':{'raw_bytes_equal_checked_preview':True,'complete_directory_dispatch_checks_passed':True,
                  'runner_p6_receipt':False,'independent_semantic_review':'required_before_current_generation'},
    'scope_limits':['Only the approved nine YAML and two dispatch amendments, not a full 33-design audit',
                    'Hardware skipped; no GPU execution or all-domain numerical proof',
                    'Existing receipts bind original bytes only; no historical runs or rules changed',
                    'current manifests must reference the later frozen independent review and cannot be its subject']}
write(REC/'W2_SUBJECT.json',subject)
print(json.dumps({'subject':file_record('schema_discussion/main_integration_20261122/W2_SUBJECT.json'),
                  'yaml':19,'dispatch':9,'checks_passed':True,'tests':64},ensure_ascii=False))
