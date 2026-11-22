"""Read-only targeted diagnosis; all writes go to this evidence directory."""
import ast
import collections
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

REPO = Path('/home/shaohanyong.shy/KernelSchema')
OUT = Path(__file__).resolve().parent
ROOT = REPO / '.schema_extract/v2'
RUN = '2b64ba84-64b3-4d10-86fe-65c351ce571c'
SCOPE = 'P5:split_k_reduce'
R = ROOT / 'runs' / RUN
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
def copy(src, dest):
    target = OUT / dest
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, target)

s = read(R / 'state.json')
lock = read(R / 'inputs.lock.json')
frozen = Path(s['snapshot']) / 'rules'
sys.path.insert(0, str(frozen))
from schema_extract.tools.store import Store
from schema_extract.tools.recovery import scoped_restart_evidence, termination_observed, verified_result
from schema_extract.tools.processes import process_table, process_alive
from schema_extract.tools.receipts import _anchor
from schema_extract.tools.common import source_root
from schema_extract.tools.maintenance import publication_barrier

store = Store(ROOT)
ledger = store.read_supervisor()
aid = s['nodes']['split_k_reduce']['reason']['attempt_id']
assert aid == 'dafb1609-4488-4cc3-9d70-0bc770cc508f'
A = R / 'attempts' / aid
a, result, response, receipt = [read(A / f) for f in ['attempt.json', 'result.json', 'response.json', 'receipt.json']]
copy(R / 'state.json', 'state.before.json')
copy(R / 'inputs.lock.json', 'inputs.lock.before.json')
save('supervisor.before.json', ledger)
table = process_table()
evidence = scoped_restart_evidence(store, s, SCOPE, table=table)
native = verified_result(a, result, result['termination'])
save('native_result_check.json', {k: native.get(k) for k in ['attempt_id','outcome','exit_code','native_status','response_validation','session_id','resume_session_id','session_status','termination','trace','prompt_sha256','response_schema_digest']})
assert native['native_status']['success'] is True
assert native['response'] == response
assert native['session_id'] == a['session_id']
assert native['response']['attempt_id'] == aid
assert s['attempt_usage'][aid]['settled'] is True
assert not evidence['settlement_required']
assert ledger['active_hold'] is None
assert not publication_barrier(store, s, ledger)
assert not store.control_requests(RUN)
assert not s['control']['action']
assert ledger['controls'].get(RUN, {}).get('action') not in {'pause','cancel'}
assert not any(x.get('run_id') == RUN and x.get('status') == 'authorized' for x in ledger['retries'].values())

anchors = []
for field, directory, hashes in [('source_evidence', source_root(lock), lock['source']['files']), ('artifact_evidence', Path(a['workspace']), {p.name: sha(p) for p in Path(a['workspace']).iterdir() if p.is_file()})]:
    for index, item in enumerate(receipt[field]):
        _anchor(item, directory, hashes)
        anchors.append({'field':field,'index':index,'category':item.get('category'),'path':item['path'],'symbol':item.get('symbol'),'line_start':item['line_start'],'line_end':item['line_end'],'valid':True})
binding = lock['source']['bindings']['entrypoint']
entryfile = source_root(lock) / binding['path']
tree = ast.parse(entryfile.read_text())
model = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'Model')
forward = next(x for x in model.body if isinstance(x, ast.FunctionDef) and x.name == 'forward')
entry = {**binding, 'line_start':forward.lineno, 'line_end':forward.end_lineno, 'sha256':sha(entryfile)}
required_impl = [x for x in receipt['source_evidence'] if x['category'] == 'implementation' and x['path'] == binding['path']]
assert not required_impl
for field in ['source_evidence', 'artifact_evidence']:
    assert len(response[field]) == len(receipt[field])
    for raw, normalized in zip(response[field], receipt[field]):
        assert all(raw.get(k) == normalized.get(k) for k in ['category','path','symbol','line_start','line_end','note'])
prior = read(R / 'attempts/e0fbf76f-397d-446b-a789-4dc22344a61b/response.json')
diagnosis = {
    'classification':'ordinary_missing_required_evidence_category_at_bound_entry',
    'bound_implementation_entry':entry,
    'implementation_entry_anchor_present':False,
    'bound_path_anchors':[{**x,'index':i} for i,x in enumerate(receipt['source_evidence']) if x['path'] == binding['path']],
    'category_paths':dict(collections.Counter(x['category'] + ':' + x['path'] for x in receipt['source_evidence'])),
    'all_current_anchors_valid':True,
    'proposal_to_receipt_coordinates_and_categories_preserved':True,
    'normalization_note':'Runner supplies source_id, sha256 and actual quote, replacing proposal quote_offset; anchor coordinates and categories are preserved.',
    'prior_tail': [x for x in prior['artifact_evidence'] if x['path']=='split_k_reduce.yaml' and x['line_start']>=272],
    'current_tail':[x for x in receipt['artifact_evidence'] if x['path']=='split_k_reduce.yaml' and x['line_start']>=272],
    'raw_lines':len((A/'workspace/split_k_reduce.yaml').read_text().splitlines()),
    'handoff':{'characters':len(receipt['handoff']), 'nonempty_lines':len([x for x in receipt['handoff'].splitlines() if x.strip()]),'mechanical_1200_limit_applies':False,'reason':'Frozen receipts.py applies the short-handoff gate only to approve in P2/P3/P4; this is P5.'},
    'validator':str(frozen/'schema_extract/tools/receipts.py'),
    'validator_sha256':sha(frozen/'schema_extract/tools/receipts.py'),
    'prompt_rule':str(frozen/'schema_extract/prompt/operations/response.md'),
    'anchors':anchors,
}
save('diagnosis.json', diagnosis)
manifest = {}
for old in s['attempts']:
    directory = R/'attempts'/old
    for name in ['attempt.json','result.json','response.json','receipt.json','repair.json','validated.json','normalization.json','exit.json','cli-exit.json']:
        p = directory/name
        if p.is_file(): manifest[str(p.relative_to(R))] = sha(p)
    if s['attempt_usage'].get(old,{}).get('node') == 'split_k_reduce':
        for sub in ['original','normalized']:
            if (directory/sub).exists():
                for p in (directory/sub).iterdir():
                    if p.is_file(): manifest[str(p.relative_to(R))] = sha(p)
for old in {s['checkpoint'],s['merged_checkpoint'],s['nodes']['split_k_reduce']['draft_checkpoint']}:
    for p in (R/'checkpoints'/old).rglob('*'):
        if p.is_file(): manifest[str(p.relative_to(R))] = sha(p)
save('immutable_manifest.before.json', manifest)
for name in ['attempt.json','invocation.json','launch.json','process.json','cli-exit.json','exit.json','result.json','response.json','response.schema.json','receipt.json','repair.json','finished.json','prompt.txt']:
    copy(A/name,'failed_attempt/'+name)
copy(A/'original/split_k_reduce.yaml','failed_attempt/original/split_k_reduce.yaml')
copy(entryfile,'bound_entry/solution.py')
for name in ['receipts.py','recovery.py']:
    copy(frozen/'schema_extract/tools'/name,'frozen_protocol/'+name)
copy(frozen/'schema_extract/prompt/operations/response.md','frozen_protocol/response.md')
sibling = s['active_attempts'].get('P5:pair_two_pass')
sibling_alive = None
if sibling:
    for name in ['attempt.json','invocation.json','process.json','launch.json']:
        copy(R/'attempts'/sibling/name,'sibling_before/'+name)
    sibling_alive = process_alive(read(R/'attempts'/sibling/'process.json')['identity'], table)
old_proc = read(A/'process.json')
preflight = {'observed_at':time.time(),'scope_evidence':evidence,'counter':s['nodes']['split_k_reduce']['stage'], 'limit':s['config']['max_calls'], 'call_additions':s['call_additions'],'node_instance':s['nodes']['split_k_reduce']['instance'],'generation':s['nodes']['split_k_reduce']['generation'],'plan_revision':s['plan_revision'],'writing_started':s['nodes']['split_k_reduce']['writing_started'],'source_context':s['nodes']['split_k_reduce']['source_context'],'raw_from_lines':[x for x in (A/'original/split_k_reduce.yaml').read_text().splitlines() if x.strip().startswith('from:')],'termination_observed':termination_observed(a,result,table=table),'old_process_alive':[{'identity':x,'alive':process_alive(x,table)} for x in [old_proc['identity'],*old_proc.get('descendants',[])]],'control_clear':True,'active_hold':ledger['active_hold'],'maintenance_barrier':False,'sibling_attempt':sibling,'sibling_alive':sibling_alive,'imported_frozen_store':__import__('schema_extract.tools.store',fromlist=['x']).__file__}
save('preflight.json',preflight)
print(json.dumps({'scope_evidence':evidence,'calls':s['nodes']['split_k_reduce']['stage']['calls'],'limit':s['config']['max_calls'],'invalid_receipts':s['nodes']['split_k_reduce']['stage']['invalid_receipts'],'receipt_repairs':s['nodes']['split_k_reduce']['stage']['receipt_repairs'],'binding':entry,'native_verified':True,'control_clear':True,'termination_observed':preflight['termination_observed'],'sibling_alive':sibling_alive,'handoff':diagnosis['handoff']},ensure_ascii=False,indent=2))
