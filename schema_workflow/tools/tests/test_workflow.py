"""Orchestration behavior through real fake-CLI subprocesses; no model/SDK."""
import importlib.util
import hashlib
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
    assert all('--reference' not in event['argv'] for event in events)
    if start=='external':
        assert [e['module'] for e in events]==['optimize','implement','optimize']
        assert [e['argv'][e['argv'].index('--max-calls')+1] for e in events if e['module']=='optimize']==['1','1']
    assert capsys.readouterr().out.splitlines()==[str(base/'run/result.json')]*2


@pytest.mark.parametrize('changed', ['environment.json', 'neighbor/dispatch.md'])
@pytest.mark.parametrize('neighbor_entry', ['neighbor', 'neighbor/root.yaml'])
def test_references_reach_design_and_only_materials_reach_implement(setup, monkeypatch, changed, neighbor_entry):
    base,plan,launch=setup
    reference=base/'environment.json'
    reference.write_text('{"source":"cache","arch":"sm_103"}')
    neighbor=base/'neighbor'
    neighbor.mkdir()
    (neighbor/'root.yaml').write_text('reference schema')
    (neighbor/'dispatch.md').write_text('reference dispatch')
    monkeypatch.chdir(base)
    code,_=launch([{'module':'design'},I(),O()],start='task',
                  extra=['--reference','environment.json','--reference',neighbor_entry])
    assert code==0
    events=w.read(plan)['events']
    args=events[0]['argv']
    expected=[str(reference),str(base/neighbor_entry)]
    assert [args[index+1] for index,key in enumerate(args) if key=='--reference']==expected
    implement_args=events[1]['argv']
    assert [implement_args[index+1] for index,key in enumerate(implement_args) if key=='--reference']==[str(reference)]
    assert '--feedback' not in implement_args and '--reference' not in events[2]['argv']
    state=w.read(base/'run/state.json')
    assert state['config']['reference']==expected
    assert str(reference) in state['inputs'] and str(neighbor) in state['inputs']
    before=w.read(plan)
    assert w.main(['--resume',str(base/'run')])==0
    (base/changed).write_text('changed reference bytes')
    assert w.main(['--resume',str(base/'run')])==1
    assert w.read(base/'run/result.json')['stop_reason']=='input_changed'
    assert w.read(base/'run/result.json')['last_usable'] is None
    assert w.read(plan)==before


@pytest.mark.parametrize('start', ['schema','external'])
def test_reference_requires_the_design_start(setup,start):
    base,_,launch=setup
    with pytest.raises(SystemExit) as error:
        launch([],start=start,extra=['--reference',str(base/'feedback.json')])
    assert error.value.code==2
    assert not (base/'run').exists()


def test_resume_cannot_replace_references(setup):
    base,plan,launch=setup
    launch([{'module':'design'},I(),O()],start='task',extra=['--reference',str(base/'feedback.json')])
    before=w.read(plan)
    with pytest.raises(SystemExit) as error:
        w.main(['--resume',str(base/'run'),'--reference',str(base/'task.md')])
    assert error.value.code==2
    assert w.read(plan)==before


def test_reference_cannot_contain_output_directory(setup):
    base,_,launch=setup
    with pytest.raises(SystemExit) as error:
        launch([],start='task',extra=['--reference',str(base)])
    assert error.value.code==2
    assert not (base/'run').exists()


def test_real_design_snapshots_environment_and_delivers_it_to_writer(setup,monkeypatch):
    """Run the actual Design CLI material path, replacing only its model call."""
    base,_,launch=setup
    script=base/'design_materials.py'
    script.write_text(f'''
import sys
sys.path.insert(0, {str(MODULE.parents[1])!r})
from schema_design.tools import runner as design
def inspect(self, prompt, output_schema, phase):
    self.result['calls_used'] += 1
    self.last_reads = []
    if phase == 'select':
        return {{'references': [], 'notes': 'CPU material delivery check'}}
    (self.out/'cpu_writer_prompt.txt').write_text(prompt)
    return {{'files': [], 'notes': 'CPU material delivery check; no model or GPU invocation',
             'missing': ['CPU fixture stops after inspecting the fixed input']}}
design.Run.call = inspect
raise SystemExit(design.main())
''')
    monkeypatch.setitem(w.SCRIPTS,'design',script)
    reference=MODULE.parent/'examples/vector_add/environment.json'
    code,result=launch([],start='task',extra=['--reference',str(reference)])
    assert code==1 and result['stop_reason']=='needs_input'
    out=base/'run/design'
    manifest=w.read(out/'input/manifest.json')
    record=next(item for item in manifest['files'] if item['source']==str(reference))
    original=reference.read_bytes()
    assert (out/record['snapshot']).read_bytes()==original
    assert record['sha256']==hashlib.sha256(original).hexdigest()
    assert record['snapshot'] in manifest['bindings']['explicit_reference_files']
    assert original.decode() in (out/'cpu_writer_prompt.txt').read_text()
    assert w.read(out/'result.json')['schema'] is None


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


def test_fixed_materials_reach_each_new_version_without_readding_on_repair(setup):
    base,plan,launch=setup
    reference=base/'language.json'
    reference.write_text('{"language":"CPU fixture"}')
    neighbor=base/'neighbor'
    neighbor.mkdir()
    (neighbor/'root.yml').write_text('Schema reference')
    (neighbor/'dispatch.md').write_text('reference boundary')
    code,_=launch([{'module':'design'},I(),O(decision='repair',required=True),I(calls=2),
                   O(decision='revise'),I(),O()],start='task',
                  extra=['--reference',str(reference),'--reference',str(neighbor/'dispatch.md')])
    assert code==0
    attempts=[event['argv'] for event in w.read(plan)['events'] if event['module']=='implement']
    for args in (attempts[0],attempts[2]):
        assert [args[i+1] for i,key in enumerate(args) if key=='--reference']==[str(reference)]
    assert '--resume' in attempts[1] and '--reference' not in attempts[1]
    assert len(w.read(base/'run/state.json')['versions'])==2


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


@pytest.mark.parametrize('recover_rejected',[False,True])
def test_real_workflow_implement_and_optimize_preserve_reference_materials(tmp_path,monkeypatch,recover_rejected):
    """Use real business CLIs/material paths; Design output, model and GPU are CPU fixtures."""
    example=MODULE.parent/'examples/vector_add'
    language=example/'triton_3_7_1_language.json'
    bundle=tmp_path/'manual'
    bundle.mkdir()
    (bundle/'index.md').write_text('[local facts](facts.json)')
    (bundle/'facts.json').write_text('{"fact":"CPU local-link material"}')
    source=tmp_path/'source.py'
    source.write_text('def cpu_source_context(): return 1\n')
    materials=[language,example/'environment.json',bundle,source]
    script=r'''
import json, shutil, sys
from pathlib import Path
sys.path.insert(0, CHECKOUT)
from schema_implement.tools.tests.test_implement import impl, FakeClient, reply, writer
mode=Path(__file__).stem
recover_rejected=RECOVER_REJECTED
example=Path(EXAMPLE)
language=example/'triton_3_7_1_language.json'
if mode=='design':
    args=sys.argv[1:]
    out=Path(args[args.index('--out')+1])
    out.mkdir()
    shutil.copytree(example/'schema',out/'schema')
    (out/'notes.md').write_text('CPU Schema fixture; no Design/model acceptance claim')
    (out/'result.json').write_text(json.dumps({'status':'complete','stop_reason':'done',
        'calls_used':1,'schema':'schema','notes':'notes.md'}))
    print(out/'result.json')
    raise SystemExit(0)
if mode=='implement':
    _,reference=impl.agate.load_eval(example/'eval')
    impl.agate.connect=lambda _: (FakeClient(reference),'CPU fake transport')
    if recover_rejected and '--resume' not in sys.argv:
        from atrex_gateway_client.errors import GatewayError
        original_build=impl.agate.build_request
        def legacy(*args,**kwargs):
            wire=original_build(*args,**kwargs)
            wire['harness']='atrex'
            return wire
        impl.agate.build_request=legacy
        class Reject(FakeClient):
            def submit_job(self,kind,request):
                raise GatewayError(422,{'reason':'request_validation_failed','error_class':'validation'})
        impl.agate.connect=lambda _: (Reject(reference),'CPU fake transport')
    def inspect(prompt,run,attempt,state,persist,dirs):
        assert '--recover-rejected-request' not in sys.argv, 'request recovery must not call coder'
        assert 'Fixed supplementary references' in prompt
        records=[state['manifest'][path] for path in state['reference_paths']]
        assert len(records)==4 and not any(Path(item['origin']).name=='schema' for item in records)
        for path,item in zip(state['reference_paths'],records):
            original=Path(item['origin'])
            root=original if original.is_dir() else original.parent
            for name,digest in item['files'].items():
                raw=(run/path/name).read_bytes()
                assert raw==(root/name).read_bytes() and impl.coder.sha(raw)==digest
        (run/'cpu_material_read.json').write_text(json.dumps(records))
        return writer([reply()])(prompt,run,attempt,state,persist,dirs)
    impl.coder.invoke=inspect
    raise SystemExit(impl.main())
from schema_optimize.tools import runner as optimize
def decide(self,prompt,output_schema,phase):
    assert language.read_text() in prompt and 'CPU local-link material' in prompt
    assert 'job-original' in prompt and 'observed fake device' in prompt
    records=list(self.inputs.records.values())
    fixed=next(item for item in records if Path(item['source']).name==language.name)
    assert '/inputs/reference/' in fixed['source'] and fixed['source']!=str(language)
    assert (self.out/fixed['snapshot']).read_bytes()==language.read_bytes()
    feedback=next(json.loads((self.out/item['snapshot']).read_text()) for item in records
                  if Path(item['source']).name=='feedback.json' and 'materials' in json.loads((self.out/item['snapshot']).read_text()))
    assert any(item['origin']==str(language) and item['files'][language.name]==fixed['sha256']
               for item in feedback['materials'])
    if recover_rejected:
        assert feedback['diagnosis']['kind']=='done' and feedback['job']['status']=='succeeded'
        assert not any('/attempts/01/' in item['source'] for item in records)
    (self.out/'cpu_material_read.json').write_text(json.dumps({'record':fixed,'materials':feedback['materials']}))
    self.result['calls_used']+=1
    return {'decision':'keep','required_fix':False,'notes':'CPU fixture verifies material delivery only',
            'files':[],'missing':[]}
optimize.Run.call=decide
raise SystemExit(optimize.main())
'''.replace('CHECKOUT',repr(str(MODULE.parents[1]))).replace('EXAMPLE',repr(str(example))).replace('RECOVER_REJECTED',repr(recover_rejected))
    for name in ('design','implement','optimize'):
        path=tmp_path/(name+'.py')
        path.write_text(script)
        monkeypatch.setitem(w.SCRIPTS,name,path)
    original={str(path):w.digest(path) for path in (example/'task.md',example/'eval',*materials)}
    run=tmp_path/'run'
    args=['--task',str(example/'task.md'),'--eval-task',str(example/'eval'),
          '--gpu','L20D','--out',str(run),'--reference',str(example/'schema')]
    for path in materials:
        args+=['--reference',str(path)]
    assert w.main(args)==(1 if recover_rejected else 0)
    if recover_rejected:
        rejected=w.read(run/'result.json')
        assert rejected['stop_reason']=='evaluation_failed'
        old=w.digest(run/'v001/implement/attempts/01')
        design=w.digest(run/'design')
        assert w.main(['--resume',str(run),'--recover-rejected-request'])==0
        state=w.read(run/'state.json')
        assert len(state['versions'])==1 and state['versions'][0]['implementation']['calls_used']==1
        assert state['last_usable']['kernel']=='v001/implement/attempts/02/kernel'
        assert w.digest(run/'v001/implement/attempts/01')==old and w.digest(run/'design')==design
        assert w.main(['--resume',str(run),'--recover-rejected-request'])==0
    result=w.read(run/'result.json')
    assert result['status']=='complete'
    observed=w.read(run/'v001/optimize/01/cpu_material_read.json')
    assert len(observed['materials'])==len(materials)
    assert observed['record']['sha256']==hashlib.sha256(language.read_bytes()).hexdigest()
    assert len(w.read(run/'v001/implement/cpu_material_read.json'))==len(materials)
    assert {str(path):w.digest(path) for path in (example/'task.md',example/'eval',*materials)}==original


def test_workflow_recovery_interruption_then_queries_original_action(setup):
    base,plan,launch=setup
    code,_=launch([I(status='incomplete',reason='evaluation_failed'),I(status='interrupted',calls=1),I(calls=1),O()])
    assert code==1
    run=base/'run'
    assert w.main(['--resume',str(run),'--recover-rejected-request'])==2
    state_before=(run/'state.json').read_bytes()
    assert w.main(['--resume',str(run),'--recover-rejected-request'])==2
    assert (run/'state.json').read_bytes()==state_before
    assert w.main(['--resume',str(run)])==0
    events=w.read(plan)['events']
    assert [event['module'] for event in events]==['implement','implement','implement','optimize']
    assert len({event['out'] for event in events[:3]})==1
    assert '--recover-rejected-request' in events[1]['argv']
    assert '--recover-rejected-request' not in events[2]['argv']
    state=w.read(run/'state.json')
    assert len(state['versions'])==1 and state['versions'][0]['implementation']['calls_used']==1


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
