"""Orchestration behavior through real fake-CLI subprocesses; no model/SDK."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

MODULE = Path(__file__).resolve().parents[2] / 'workflow.py'
spec = importlib.util.spec_from_file_location('standalone_workflow_test', MODULE)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)

FAKE = r'''
import json, os, sys
from pathlib import Path
kind = Path(__file__).stem
args = sys.argv[1:]
def option(key, default=None):
    return args[args.index(key)+1] if key in args else default
out = Path(option('--resume') or option('--out'))
plan = Path(os.environ['WORKFLOW_FAKE_PLAN'])
data = json.loads(plan.read_text())
step = data['steps'].pop(0)
assert step.pop('module') == kind, (kind, step)
events = data.setdefault('events', [])
events.append({'module':kind, 'out':str(out), 'argv':args})
plan.write_text(json.dumps(data))
out.mkdir(parents=True, exist_ok=True)
old = json.loads((out/'state.json').read_text()) if (out/'state.json').exists() else {'calls':0, 'attempt':0, 'jobs':0}
status = step.get('status','complete')
reason = step.get('reason','done' if status=='complete' else 'implementation_failed')
value = {'status':status,'stop_reason':reason,'calls_used':step.get('calls',old['calls']+1 if kind=='implement' else 1)}
if kind=='implement':
    if old.get('pending'):
        old['pending'] = False
    else:
        old['attempt'] += 1
        old['jobs'] += 1
    attempt = out/'attempts'/f"{old['attempt']:02d}"
    attempt.mkdir(parents=True, exist_ok=True)
    schema = out/'inputs/schema'
    schema.mkdir(parents=True, exist_ok=True)
    (schema/'root.yaml').write_text('fixed schema')
    value.update(schema='inputs/schema',kernel=None,feedback=None)
    if step.get('kernel',True):
        kernel = attempt/'kernel'
        kernel.mkdir(exist_ok=True)
        (kernel/'entry.py').write_text('complete source '+str(old['attempt']))
        value['kernel'] = str(kernel.relative_to(out))
    if status!='interrupted':
        (attempt/'feedback.json').write_text(json.dumps({'diagnosis':reason,'job':'original-job'}))
        value['feedback'] = str((attempt/'feedback.json').relative_to(out))
    old.update(calls=value['calls_used'],pending=status=='interrupted')
    (out/'state.json').write_text(json.dumps(old))
else:
    (out/'notes.md').write_text('concrete decision evidence')
    schema = out/'schema'
    schema.mkdir(exist_ok=True)
    (schema/'root.yaml').write_text('schema '+out.name)
    value.update(schema='schema',notes='notes.md')
    if kind=='optimize':
        value.update(decision=step.get('decision','keep'),required_fix=step.get('required',False))
(out/'result.json').write_text(json.dumps(value))
print(str(out/'result.json') if not step.get('bad_stdout') else str(out/'wrong.json'))
sys.exit(step.get('code',{'complete':0,'incomplete':1,'interrupted':2}[status]))
'''


@pytest.fixture
def setup(tmp_path, monkeypatch):
    scripts = {}
    for name in ('design', 'implement', 'optimize'):
        path = tmp_path / (name + '.py')
        path.write_text(FAKE)
        scripts[name] = path
    monkeypatch.setattr(w, 'SCRIPTS', scripts)
    plan = tmp_path/'plan.json'
    monkeypatch.setenv('WORKFLOW_FAKE_PLAN',str(plan))
    (tmp_path/'task.md').write_text('fixed task')
    schema=tmp_path/'schema'
    schema.mkdir()
    (schema/'root.yaml').write_text('initial schema')
    kernel=tmp_path/'seed'
    kernel.mkdir()
    (kernel/'entry.py').write_text('seed source')
    (tmp_path/'feedback.json').write_text('{}')
    def launch(steps, start='schema', extra=()):
        plan.write_text(json.dumps({'steps':steps}))
        args=['--task',str(tmp_path/'task.md'),'--out',str(tmp_path/'run'), *extra]
        if start!='task':
            args+=['--schema',str(schema)]
        if start=='external':
            args+=['--kernel',str(kernel),'--feedback',str(tmp_path/'feedback.json')]
        code=w.main(args)
        return code, w.read(tmp_path/'run/result.json')
    return tmp_path,plan,launch


def I(**kw): return {'module':'implement',**kw}
def O(**kw): return {'module':'optimize',**kw}


@pytest.mark.parametrize('start', ['task','schema','external'])
def test_three_starts_complete_and_do_not_replay(setup,start,capsys):
    base,plan,launch=setup
    steps=([{'module':'design'}] if start=='task' else [O()] if start=='external' else []) + [I(),O()]
    code,result=launch(steps,start)
    assert code==0 and result['stop_reason']=='done'
    before=w.read(plan)
    assert not before['steps']
    assert w.main(['--resume',str(base/'run')])==0
    assert w.read(plan)==before
    pair=result['last_usable']
    assert '/attempts/01/kernel' in pair['kernel'] and pair['feedback'].endswith('/attempts/01/feedback.json')
    assert set(result)=={'status','stop_reason','last_usable','latest_attempt'}
    events=before['events']
    if start=='external':
        assert [e['module'] for e in events]==['optimize','implement','optimize']
        assert [e['argv'][e['argv'].index('--max-calls')+1] for e in events if e['module']=='optimize']==['1','1']
    assert capsys.readouterr().out.splitlines()==[str(base/'run/result.json')]*2


def test_repair_resumes_one_directory_and_accumulates_calls(setup):
    base,plan,launch=setup
    code,result=launch([I(),O(decision='repair',required=True),I(calls=2),O()])
    assert code==0
    events=w.read(plan)['events']
    assert events[0]['out']==events[2]['out'] and '--resume' in events[2]['argv']
    assert '--feedback' in events[2]['argv']
    assert result['last_usable']['kernel'].endswith('attempts/02/kernel')
    assert w.read(base/'run/state.json')['versions'][0]['implementation']['calls_used']==2


def test_external_repair_first_build_and_reserved_optimize(setup):
    base,plan,launch=setup
    code,_=launch([O(decision='repair',required=True),I(),O()], 'external')
    assert code==0
    event=w.read(plan)['events'][1]
    assert '--out' in event['argv'] and '--resume' not in event['argv'] and '--kernel' in event['argv']


def test_revise_new_version_retests_and_uses_latest_source(setup):
    base,plan,launch=setup
    code,result=launch([I(),O(decision='revise'),I(),O()])
    assert code==0 and result['last_usable']['kernel'].startswith('v002/')
    events=w.read(plan)['events']
    assert events[2]['argv'][events[2]['argv'].index('--kernel')+1].endswith('v001/implement/attempts/01/kernel')
    assert events[2]['argv'][events[2]['argv'].index('--schema')+1].endswith('v001/optimize/01/schema')


def test_design_gap_without_source_can_revise(setup):
    base,plan,launch=setup
    code,_=launch([I(status='incomplete',reason='design_gap',kernel=False),O(decision='revise',required=True),I(),O()])
    assert code==0
    assert '--kernel' not in w.read(plan)['events'][1]['argv']


@pytest.mark.parametrize('required,expected',[(False,0),(True,1)])
def test_last_version_optional_proposal_and_required_defect(setup,required,expected):
    _,_,launch=setup
    code,result=launch([I(),O(decision='revise',required=required)],extra=['--max-versions','1'])
    assert code==expected and result['stop_reason']=='limit'
    assert bool(result['last_usable']) is not required
    assert result['latest_attempt'].endswith('optimize/01/result.json')


def test_keep_after_failure_does_not_rebuild_or_claim_success(setup):
    _,plan,launch=setup
    code,result=launch([I(status='incomplete',reason='implementation_failed'),O()])
    assert code==1 and result['stop_reason']=='implementation_failed'
    assert len(w.read(plan)['events'])==2


def test_limit_does_not_reset_coder_or_clear_required_problem(setup):
    base,plan,launch=setup
    code,result=launch([I(calls=2),O(decision='repair',required=True)],extra=['--max-calls-per-schema','2'])
    assert code==1 and result['stop_reason']=='limit' and result['last_usable'] is None
    assert len(w.read(plan)['events'])==2
    assert len(w.read(base/'run/state.json')['versions'])==1


def test_optimize_allowance_sum_and_chain_completion(setup):
    base,plan,launch=setup
    code,result=launch([I(),O(decision='repair',required=True,calls=2),I(calls=2)])
    assert code==0 and result['stop_reason']=='limit'
    assert len(w.read(plan)['events'])==3
    assert len(w.read(base/'run/state.json')['versions'][0]['optimize'])==1


def test_no_success_feedback_decision_is_not_complete(setup):
    _,_,launch=setup
    code,result=launch([I(status='incomplete',reason='design_gap',kernel=False),O(decision='repair',required=True,calls=2),I(calls=2)])
    assert code==1 and result['stop_reason']=='limit'


def test_optional_new_version_failure_retains_original_pair(setup):
    _,_,launch=setup
    code,result=launch([I(),O(decision='revise'),I(status='incomplete',reason='evaluation_failed')])
    assert code==1 and result['stop_reason']=='evaluation_failed'
    assert result['last_usable']['schema'].startswith('v001/')
    assert result['latest_attempt'].startswith('v002/')


def test_original_job_resume_and_completed_stages_not_replayed(setup):
    base,plan,launch=setup
    code,_=launch([I(status='interrupted'),I(calls=1),O()])
    assert code==2
    assert w.main(['--resume',str(base/'run')])==0
    events=w.read(plan)['events']
    assert events[0]['out']==events[1]['out'] and '--resume' in events[1]['argv']
    assert w.read(base/'run/v001/implement/state.json')['jobs']==1


def test_partial_writer_is_not_replayed_or_counted_as_zero(setup):
    base,plan,launch=setup
    code,_=launch([I(),O(status='interrupted')])
    assert code==2
    before=w.read(plan)
    assert w.main(['--resume',str(base/'run')])==2
    assert w.read(plan)==before


@pytest.mark.parametrize('bad', [{'bad_stdout':True},{'code':1}])
def test_child_stdout_and_status_are_checked(setup,bad):
    _,_,launch=setup
    code,result=launch([I(**bad)])
    assert code==2 and result['last_usable'] is None


def test_immutable_pair_change_detected_on_completed_resume(setup):
    base,plan,launch=setup
    _,result=launch([I(),O()])
    (base/'run'/result['last_usable']['kernel']/'entry.py').write_text('changed')
    before=w.read(plan)
    assert w.main(['--resume',str(base/'run')])==1
    assert w.read(base/'run/result.json')['stop_reason']=='input_changed'
    assert w.read(plan)==before


def test_real_implement_feedback_is_accepted_by_real_optimize(tmp_path, monkeypatch):
    """Exercise the actual cross-module material bindings; only model/GPU are fake."""
    from schema_optimize.tools import runner as optimize
    entry = MODULE.parents[1] / 'schema_implement/implement_kernel.py'
    impl_spec = importlib.util.spec_from_file_location('workflow_interface_implement', entry)
    impl = importlib.util.module_from_spec(impl_spec)
    impl_spec.loader.exec_module(impl)
    example = MODULE.parent / 'examples/vector_add'
    run = tmp_path / 'implement'
    run.mkdir()
    args = impl.parser().parse_args(['--task',str(example/'task.md'),'--schema',str(example/'schema'),
            '--eval-task',str(example/'eval'),'--gpu','L20D','--out',str(run)])
    state = {'config':{'model':'qwen3.8-max[1m]','effort':'max','max_calls':2,'timeout_s':1,'gpu':'L20D'},
             'calls_used':0,'manifest':{},'feedback_ids':[],'feedback_paths':[],
             'consumed_feedback':0,'action':None,'initialized':False}
    impl.initialize(run,args,state)
    config,reference = impl.agate.load_eval(run/'inputs/eval')
    class Client:
        def submit_job(self,kind,request): return {'job_id':'cpu-original-job'}
        def get_job(self,job_id,**kw):
            ids=reference['shapes']
            return {'job_id':job_id,'status':'succeeded','cleanup_pending':False,'result':{
                'passed':{stage:{i:{'status':'passed'} for i in ids} for stage in ('compile','correctness','performance')},
                'correctness':{'shapes':{i:{'cases':[{'outputs':[{'passed':True,'error':None}],'error':None}]} for i in ids}},
                'performance':{'shapes':{i:{'timing_stats':{'median_ms':0.01,'runs':100},'benchmark_mode':'eager'} for i in ids}},
                'environment':{'gpu_name':'CPU fake'}}}
    def writer(prompt,run,attempt,state,persist,dirs):
        state['calls_used'] += 1
        persist()
        return {'files':[{'path':'entry.py','content':'class Model:\n    def forward(self,x,y): return x+y\n'}],
                'entry_point':'entry.py','notes':'CPU fixture realizes both fixed dispatch branches; no real GPU claim','gaps':[]}
    # Request construction uses the installed official SDK; only submission/results are fake.
    result=impl.execute(run,state,client_factory=lambda _: (Client(),'cpu-fake-transport'),writer=writer)
    assert result['status']=='complete'
    out=tmp_path/'optimize'
    args=optimize.parser('optimize').parse_args(['--task',str(example/'task.md'), '--schema',str(example/'schema'),
            '--kernel',str(run/result['kernel']),'--feedback',str(run/'result.json'),'--out',str(out)])
    out.mkdir()
    execution=optimize.Run(args,'optimize')
    def decide(prompt,output_schema,phase):
        assert 'entry.py' in prompt and 'cpu-original-job' in prompt and 'CPU fake' in prompt
        execution.result['calls_used'] += 1
        return {'decision':'keep','required_fix':False,'notes':'CPU protocol decision on fixed material', 'files':[], 'missing':[]}
    monkeypatch.setattr(execution,'call',decide)
    assert optimize.execute(execution)==0
    assert execution.result['decision']=='keep'


def test_recover_committed_child_without_relaunch_or_double_count(setup):
    base,plan,launch=setup
    code,_=launch([I(status='interrupted'),I(calls=1),O()])
    assert code==2
    state=w.read(base/'run/state.json')
    workflow=w.Workflow(base/'run',state)
    value=workflow.child()  # Child finishes, then the coordinator stops before consuming it.
    assert value['status']=='complete'
    before=w.read(plan)['events']
    assert w.main(['--resume',str(base/'run')])==0
    events=w.read(plan)['events']
    assert events[:len(before)]==before and len(events)==len(before)+1
    assert w.read(base/'run/state.json')['versions'][0]['implementation']['calls_used']==1
