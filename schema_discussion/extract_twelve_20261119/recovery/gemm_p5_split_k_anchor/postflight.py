"""Confirm this one scoped recovery; write evidence only."""
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
R = ROOT / 'runs' / RUN
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name, value):
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def copy(src,dest):
    target = OUT / dest
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(src,target)

before = read(OUT/'state.before.json')
s = read(R/'state.json')
ledger = read(ROOT/'supervisor.json')
request = ledger['retries']['5c214e5e-794d-4078-a0ec-4edd2c2d764b']
assert request['status'] == 'consumed'
new_ids = [x for x in s['attempts'] if x not in before['attempts']]
scoped_new = [x for x in new_ids if read(R/'attempts'/x/'attempt.json')['node']=='split_k_reduce']
assert len(scoped_new)==1
aid = scoped_new[0]
A = R/'attempts'/aid
a, invocation, launch, proc = [read(A/f) for f in ['attempt.json','invocation.json','launch.json','process.json']]
sys.path.insert(0,str(Path(s['snapshot'])/'rules'))
from schema_extract.tools.processes import process_table, process_alive
table = process_table()
new_alive = process_alive(proc['identity'],table)
assert new_alive and launch['status']=='released'
session = '7601fdcb-278c-4db2-903e-cc57432a32e4'
assert a['role']=='reviewer' and a['session_id']==session and invocation['resume_session_id']==session
native_init = None
with (A/'events.jsonl').open() as stream:
    offset = 0
    for line_no,line in enumerate(stream,1):
        try: event=json.loads(line)
        except ValueError:
            offset+=len(line.encode());continue
        if event.get('type')=='system' and event.get('subtype')=='init':
            native_init={'event':event,'line':line_no,'offset':offset,'line_sha256':hashlib.sha256(line.encode()).hexdigest()}
            break
        offset+=len(line.encode())
assert native_init and native_init['event']['session_id']==session
prompt=(A/'prompt.txt').read_text()
diagnostic='missing bound implementation entry anchor'
matches=[i+1 for i,line in enumerate(prompt.splitlines()) if diagnostic in line]
assert matches and sha(A/'prompt.txt')==invocation['prompt_sha256']
copy(R/'state.json','state.after.json')
save('supervisor.after.json',ledger)
copy(A/'prompt.txt','new_attempt/prompt.txt')
for name in ['attempt.json','invocation.json','launch.json','process.json','materials/check.json']:
    copy(A/name,'new_attempt/'+name)
save('new_attempt/native_start.evidence.json',native_init)
save('new_attempt/prompt.evidence.json',{'path':str(A/'prompt.txt'),'sha256':sha(A/'prompt.txt'),'diagnostic':diagnostic,'lines':matches})
manifest = read(OUT/'immutable_manifest.before.json')
changed=[name for name,digest in manifest.items() if not (R/name).is_file() or sha(R/name)!=digest]
settled_ids=[x for x in before['attempts'] if before['attempt_usage'].get(x,{}).get('settled')]
settled_changed=[x for x in settled_ids if before['attempt_usage'][x]!=s['attempt_usage'].get(x)]
core_keys=['config','executor','input_fingerprint','snapshot','source_identity','selected_designs','selection_signature','writing_plan','writing_signature','plan_revision','call_additions','checkpoint','merged_checkpoint','runner']
core={k:before[k]==s[k] for k in core_keys}
approved_stages={k:before['stages'][k]['approval']==s['stages'][k]['approval'] and s['stages'][k]['status']=='approved' for k in ['P1','P2','P3','P4']}
approved_nodes={k:all(v.get(field)==s['nodes'][k].get(field) for field in ['approval','approved_generation','approved_sha256','generation']) for k,v in before['nodes'].items() if v.get('approval')}
accepted={k:s['accepted_receipts'].get(k)==v for k,v in before['accepted_receipts'].items()}
target_keys=['instance','generation','authored','authored_generation','draft_checkpoint','source_context','writing_started']
target={k:before['nodes']['split_k_reduce'][k]==s['nodes']['split_k_reduce'][k] for k in target_keys}
old_sibling=before['active_attempts']['P5:pair_two_pass']
sibling_proc_before=read(OUT/'sibling_before/process.json')
sibling_proc_after=read(R/'attempts'/old_sibling/'process.json')
sibling={'attempt_id':old_sibling,'still_active':s['active_attempts'].get('P5:pair_two_pass')==old_sibling,'identity_unchanged':sibling_proc_before['identity']==sibling_proc_after['identity'],'alive':process_alive(sibling_proc_after['identity'],table),'calls_before':before['nodes']['pair_two_pass']['stage']['calls'],'calls_after':s['nodes']['pair_two_pass']['stage']['calls'],'no_control_action':s['control']==before['control']}
counts={k:[before['nodes']['split_k_reduce']['stage'].get(k),s['nodes']['split_k_reduce']['stage'].get(k)] for k in ['calls','invalid_receipts','receipt_repairs','revisions','adjudications','signoffs']}
checks={'immutable_files_unchanged':not changed,'settled_usage_unchanged':not settled_changed,'inputs_lock_unchanged':sha(R/'inputs.lock.json')==sha(OUT/'inputs.lock.before.json'),'core_unchanged':all(core.values()),'approved_stages_retained':all(approved_stages.values()),'approved_nodes_retained':all(approved_nodes.values()),'accepted_receipts_retained':all(accepted.values()),'target_identity_and_draft_retained':all(target.values()),'calls_increased_by_one':counts['calls']==[4,5],'other_target_counts_retained':all(v[0]==v[1] for k,v in counts.items() if k!='calls'),'only_one_new_attempt':len(new_ids)==1,'sibling_preserved':sibling['still_active'] and sibling['identity_unchanged'] and sibling['alive'] and sibling['calls_before']==sibling['calls_after'] and sibling['no_control_action']}
result={'observed_at':time.time(),'run_id':RUN,'request':request,'new_attempt_id':aid,'new_attempt_alive':new_alive,'new_gate_identity':proc['identity'],'session_id':session,'native_init_confirmed':True,'diagnostic_prompt_lines':matches,'new_attempt_ids':new_ids,'counts':counts,'calls_limit':s['config']['max_calls'],'core':core,'approved_stages':approved_stages,'approved_nodes':approved_nodes,'accepted_receipts_count':len(accepted),'target_preservation':target,'sibling':sibling,'immutable_file_count':len(manifest),'changed_immutable_files':changed,'settled_usage_changed':settled_changed,'checks':checks,'status':s['status'],'stage':s['stage'],'target_status':s['nodes']['split_k_reduce']['status'],'active_attempts':s['active_attempts'],'checkpoint':s['checkpoint'],'draft_checkpoint':s['nodes']['split_k_reduce']['draft_checkpoint']}
save('postflight.json',result)
assert all(checks.values()),checks
print(json.dumps({k:result[k] for k in ['new_attempt_id','new_attempt_alive','session_id','diagnostic_prompt_lines','counts','calls_limit','sibling','immutable_file_count','checks','checkpoint','draft_checkpoint']},ensure_ascii=False,indent=2))
