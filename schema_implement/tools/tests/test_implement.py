"""CPU behavior tests; no model, GPU, torch, or old runtime required."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

MODULE = Path(__file__).resolve().parents[2] / 'implement_kernel.py'
spec = importlib.util.spec_from_file_location('ks_implement_test_entry', MODULE)
impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impl)
coder, agate = impl.coder, impl.agate
EXAMPLE = MODULE.parents[1] / 'schema_workflow/examples/vector_add'


def response(reference, *, failed=None, missing_timing=None, status='succeeded'):
    ids = reference['shapes']
    return {'job_id':'job-original','status':status,'cleanup_pending':False,
            'result':{'passed':{s:{i:{'status':'failed' if failed == i and s == 'correctness' else 'passed'} for i in ids}
                                for s in ('compile','correctness','performance')},
                      'correctness':{'shapes':{i:{'cases':[{'outputs':[{'passed':i != failed,'error':None}],
                                                        'error':None}]} for i in ids}},
                      'performance':{'shapes':{i:{'timing_stats':{'median_ms':0.01,'runs':100},'benchmark_mode':'eager'}
                                              for i in ids if i != missing_timing}},
                      'environment':{'gpu_name':'observed fake device'}}}


def reply(*, gaps=None, multi=False):
    return {'files':[{'path':'entry.py','content':'class Model:\n    def forward(self, x, y):\n        return x + y\n'}] +
                    ([{'path':'pkg/helper.py','content':'VALUE = 1\n'}] if multi else []),
            'entry_point':'entry.py','notes':'root 1..1024 and tiled 1025..65536 dispatch realized in entry.py; CPU fake source',
            'gaps':gaps or []}


class FakeClient:
    def __init__(self, reference, outcomes=None):
        self.reference = reference
        self.outcomes = outcomes or [response(reference)]
        self.submitted, self.queries, self.uploads = [], [], []
        self.fail_submit_once = False
        self.fail_query_once = False
        self._next = 0
    def submit_job(self, kind, request):
        self.submitted.append(copy.deepcopy(request))
        if self.fail_submit_once:
            self.fail_submit_once = False
            raise ConnectionError('ack lost')
        return {'job_id':'job-original','status':'queued'}
    def get_job(self, job_id, **kwargs):
        self.queries.append((job_id,kwargs))
        if self.fail_query_once:
            self.fail_query_once = False
            raise ConnectionError('query disconnected')
        r = self.outcomes[min(self._next, len(self.outcomes)-1)]
        self._next += 1
        return copy.deepcopy(r)
    def prepare_uploads(self, gpu, files, kind):
        self.uploads.append(files)
        return {'job_id':'job-original','uploads':[{'path':'candidate.zip','put_url':'https://upload.invalid/private?secret=hidden','upload_ref':'opaque-source-ref'}]}
    def upload_file(self,url,path):
        import zipfile
        self.uploaded_members = zipfile.ZipFile(path).namelist()


@pytest.fixture
def run(tmp_path, monkeypatch):
    folder = tmp_path / 'run'
    folder.mkdir()
    args = impl.parser().parse_args(['--task',str(EXAMPLE/'task.md'),'--schema',str(EXAMPLE/'schema'),
                                    '--eval-task',str(EXAMPLE/'eval'),'--gpu','L20D','--out',str(folder)])
    state = {'config':{'model':'qwen3.8-max[1m]','effort':'max','max_calls':3,'timeout_s':1,'gpu':'L20D'},
             'calls_used':0,'manifest':{},'feedback_ids':[],'feedback_paths':[],
             'consumed_feedback':0,'action':None,'initialized':False}
    impl.initialize(folder,args,state)
    config,reference = agate.load_eval(folder / 'inputs/eval')
    def build(kernel,entry,config,reference,gpu,key,name):
        files = coder.inventory(kernel)
        return {'candidate':(kernel/entry).read_text() if len(files)==1 else {'archive':'candidate.zip','entry_point':entry},
                'reference':reference,'options':config['options'],'spec':{'target_hardware':[gpu]},'idempotency_key':key}
    monkeypatch.setattr(agate,'build_request',build)
    return folder,state,config,reference


def writer(answers):
    remaining = iter(answers)
    def call(prompt,run,attempt,state,persist,dirs):
        assert 'DOC0:' in prompt and 'Effective schemas' in prompt
        state['calls_used'] += 1
        persist()
        value = copy.deepcopy(next(remaining))
        coder.save(attempt/'draft.json',{'structured_output':value,'reads':[{'tool':'Read','input':{'file_path':str(run/'inputs/schema/root.yaml')}}]})
        return value
    return call


def test_full_cases_and_complete_resume_no_replay(run):
    folder,state,config,reference = run
    client = FakeClient(reference)
    factory=lambda _: (client,'fake-official-interface')
    value=impl.execute(folder,state,client_factory=factory,writer=writer([reply()]))
    assert set(value)=={'status','stop_reason','calls_used','schema','kernel','feedback'}
    assert value['status']=='complete' and state['calls_used']==1
    assert len(client.submitted)==1
    f=coder.read_json(folder/value['feedback'])
    assert len(f['cases'])==7 and all(row['passed'] for row in f['cases'])
    assert f['job']['status']=='succeeded'
    assert f['environment']=={'gpu_name':'observed fake device'}
    assert impl.execute(folder,state,client_factory=lambda _:pytest.fail('completed run touched SDK'))==value


def test_same_schema_repair_uses_all_cases_and_preserves_attempt(run):
    folder,state,config,reference=run
    client=FakeClient(reference,[response(reference,failed='3'),response(reference)])
    value=impl.execute(folder,state,client_factory=lambda _:(client,'fake'),writer=writer([reply(),reply()]))
    assert value['status']=='complete' and state['calls_used']==2
    assert len(client.submitted)==2
    assert client.submitted[0]['reference']==client.submitted[1]['reference']
    first=coder.read_json(folder/'attempts/01/feedback.json')
    assert first['diagnosis']['kind']=='implementation_failed' and not first['cases'][3]['passed']
    assert first['schema']['sha256']==coder.read_json(folder/'attempts/02/feedback.json')['schema']['sha256']


def test_missing_timing_and_infrastructure_do_not_trigger_coder(run):
    folder,state,config,reference=run
    client=FakeClient(reference,[response(reference,missing_timing='0')])
    value=impl.execute(folder,state,client_factory=lambda _:(client,'fake'),writer=writer([reply()]))
    assert value['stop_reason']=='evaluation_failed' and state['calls_used']==1
    assert not coder.read_json(folder/value['feedback'])['cases'][0]['passed']


def test_known_required_gap_cannot_submit_or_pass(run):
    folder,state,config,reference=run
    client=FakeClient(reference)
    state['config']['max_calls']=1
    value=impl.execute(folder,state,client_factory=lambda _:(client,'fake'),writer=writer([
        reply(gaps=[{'kind':'implementation_failed','detail':'tiled branch unfinished'}])]))
    assert value['status']=='incomplete' and not client.submitted


def test_design_gap_without_kernel(run):
    folder,state,config,reference=run
    r=reply(gaps=[{'kind':'design_gap','detail':'Missing required dispatcher decision'}]);r.update(files=[],entry_point=None)
    value=impl.execute(folder,state,client_factory=lambda _:(FakeClient(reference),'fake'),writer=writer([r]))
    assert value['kernel'] is None and value['stop_reason']=='design_gap'
    assert coder.read_json(folder/value['feedback'])['diagnosis']['detail']=='Missing required dispatcher decision'


def test_multifile_package_and_no_signed_url_artifact(run):
    folder,state,config,reference=run
    client=FakeClient(reference)
    value=impl.execute(folder,state,client_factory=lambda _:(client,'fake'),writer=writer([reply(multi=True)]))
    assert value['status']=='complete'
    assert set(client.uploaded_members)=={'entry.py','pkg/helper.py'}
    assert client.submitted[0]['candidate']=={'archive':'candidate.zip','entry_point':'entry.py'}
    assert client.submitted[0]['oss_files']==[{'path':'candidate.zip','upload_ref':'opaque-source-ref'}]
    assert 'secret=hidden' not in (folder/'state.json').read_text()


def test_unknown_submission_resumes_same_key_and_request(run):
    folder,state,config,reference=run
    client=FakeClient(reference);client.fail_submit_once=True
    with pytest.raises(coder.Problem):
        impl.execute(folder,state,client_factory=lambda _:(client,'fake'),writer=writer([reply()]))
    value=impl.execute(folder,state,client_factory=lambda _:(client,'fake'),writer=writer([]))
    assert value['status']=='complete' and state['calls_used']==1
    assert client.submitted[0]==client.submitted[1]


def test_known_job_resumes_only_original_query(run):
    folder,state,config,reference=run
    client=FakeClient(reference);client.fail_query_once=True
    with pytest.raises(coder.Problem,match='original job'):
        impl.execute(folder,state,client_factory=lambda _:(client,'fake'),writer=writer([reply()]))
    value=impl.execute(folder,state,client_factory=lambda _:(client,'fake'),writer=writer([]))
    assert value['status']=='complete' and len(client.submitted)==1
    assert all(job=='job-original' and kwargs['include_spec'] and kwargs['timeout']<=30 for job,kwargs in client.queries)


def test_cancellation_cleanup_unknown_stays_interrupted(run):
    folder,state,config,reference=run
    view=response(reference,status='cancelled');view['cleanup_pending']=True
    client=FakeClient(reference,[view])
    with pytest.raises(coder.Problem,match='cleanup'):
        impl.execute(folder,state,client_factory=lambda _:(client,'fake'),writer=writer([reply()]))
    assert state['action']['job_id']=='job-original' and len(client.submitted)==1


def test_changed_fixed_source_stops_before_sdk(run):
    folder,state,config,reference=run
    (folder/'inputs/eval/input.py').write_text('changed')
    with pytest.raises(coder.Problem,match='Fixed input'):
        impl.execute(folder,state,client_factory=lambda _:pytest.fail('SDK called'))


def test_feedback_relative_dependencies_are_fixed(tmp_path):
    source=tmp_path/'old';source.mkdir();(source/'kernel').mkdir()
    (source/'kernel/helper.py').write_text('value=3')
    coder.save(source/'feedback.json',{'kernel':{'path':'kernel'}})
    m=coder.freeze_feedback(source/'feedback.json',tmp_path/'fixed')
    assert len(m)==2
    assert any('helper.py' in v['files'] for v in m.values())


def test_unsafe_reply_retained_before_rejection(tmp_path):
    r=reply();r['files'][0]['path']='../escape.py'
    coder.save(tmp_path/'draft.json',r)
    with pytest.raises(coder.Problem):coder.materialize(r,tmp_path/'kernel')
    assert (tmp_path/'draft.json').exists() and not (tmp_path/'kernel').exists()


def test_qwen_environment_preserves_connection_without_persisting_it(tmp_path,monkeypatch):
    config=tmp_path/'auth';config.mkdir()
    coder.save(config/'settings.json',{'env':{'ANTHROPIC_API_KEY':'test-only-config-key',
                                          'ANTHROPIC_BASE_URL':'https://configured.invalid',
                                          'HTTPS_PROXY':'https://proxy.invalid','UNRELATED':'do-not-add'}})
    monkeypatch.setenv('CLAUDE_CONFIG_DIR',str(config))
    monkeypatch.setenv('ANTHROPIC_API_KEY','process-value')
    monkeypatch.delenv('ANTHROPIC_BASE_URL',raising=False)
    env=coder.cli_environment('max')
    assert env['ANTHROPIC_API_KEY']=='process-value' and env['ANTHROPIC_BASE_URL']=='https://configured.invalid'
    assert env.get('UNRELATED')!='do-not-add' and env['CLAUDE_CODE_EFFORT_LEVEL']=='max'
    assert 'process-value' not in coder.redact('process-value',env)


def test_stream_retains_reads_and_final_not_thinking():
    events=[{'type':'assistant','message':{'content':[{'type':'thinking','thinking':'private'},
             {'type':'tool_use','name':'Read','input':{'file_path':'/fixed/root.yaml'}}]}},
            {'type':'result','structured_output':reply(),'result':'final text'}]
    value,draft=coder.parse_stream('\n'.join(json.dumps(x) for x in events))
    assert value==reply() and draft['reads'][0]['input']['file_path']=='/fixed/root.yaml'
    assert 'private' not in json.dumps(draft)


def test_help_another_cwd_and_missing_material_result(tmp_path):
    help_result=subprocess.run([sys.executable,str(MODULE),'--help'],cwd=tmp_path,capture_output=True,text=True)
    assert help_result.returncode==0 and '--resume' in help_result.stdout
    run=tmp_path/'missing'
    r=subprocess.run([sys.executable,str(MODULE),'--task',str(EXAMPLE/'task.md'),'--out',str(run)],cwd=tmp_path,capture_output=True,text=True)
    assert r.returncode==1 and r.stdout.strip()==str(run/'result.json')
    assert coder.read_json(run/'result.json')['stop_reason']=='needs_input'


def test_real_sdk_builder_uses_fixed_explicit_options(tmp_path):
    pytest.importorskip('atrex_gateway_client')
    config,ref=agate.load_eval(EXAMPLE/'eval')
    kernel=tmp_path/'kernel';kernel.mkdir();(kernel/'entry.py').write_text('class Model: pass')
    wire=agate.build_request(kernel,'entry.py',config,ref,'L20D','key','test')
    assert wire['options']==config['options'] and len(wire['reference']['shapes'])==7
    assert wire['spec']['languages']==['triton']


def test_completed_candidate_digest_is_checked_on_resume(run):
    folder,state,config,reference=run
    client=FakeClient(reference)
    value=impl.execute(folder,state,client_factory=lambda _:(client,'fake'),writer=writer([reply()]))
    (folder/value['kernel']/'entry.py').write_text('class Model: pass')
    with pytest.raises(coder.Problem,match='Fixed input differs'):
        impl.execute(folder,state,client_factory=lambda _:pytest.fail('corrupt completed source touched SDK'))


def test_new_feedback_keeps_cumulative_allowance_and_original_success(run,tmp_path):
    folder,state,config,reference=run
    client=FakeClient(reference)
    value=impl.execute(folder,state,client_factory=lambda _:(client,'fake'),writer=writer([reply()]))
    first=(folder/value['feedback']).read_bytes()
    f=tmp_path/'repair.md';f.write_text('Fix explicit wrapper dispatch in the same design.')
    assert impl.add_feedback(folder,state,[f])
    assert not impl.add_feedback(folder,state,[f])
    second=impl.execute(folder,state,new_feedback=True,client_factory=lambda _:(client,'fake'),writer=writer([reply()]))
    assert second['calls_used']==2 and second['kernel']!=value['kernel']
    assert (folder/value['feedback']).read_bytes()==first
    assert len(client.submitted)==2


def test_dead_coder_timeout_can_resume_with_remaining_calls(run):
    folder,state,config,reference=run
    state['calls_used']=1
    state['result']={'status':'interrupted','stop_reason':'interrupted','calls_used':1,'schema':'inputs/schema','kernel':None,'feedback':None}
    client=FakeClient(reference)
    value=impl.execute(folder,state,client_factory=lambda _:(client,'fake'),writer=writer([reply()]))
    assert value['status']=='complete' and value['calls_used']==2


def test_actual_cli_adapter_argv_environment_and_no_thinking(run,tmp_path,monkeypatch):
    folder,state,config,reference=run
    bindir=tmp_path/'bin';bindir.mkdir()
    fake=bindir/'claude'
    fake.write_text('#!'+sys.executable+'\nimport json,os,sys\n'
                    'args=sys.argv[1:]\n'
                    'assert args[args.index("--tools")+1]=="Read,Glob,Grep"\n'
                    'assert args[args.index("--permission-mode")+1]=="dontAsk"\n'
                    'assert args[args.index("--settings")+1]==\'{"disableAllHooks":true}\'\n'
                    'assert os.environ["ANTHROPIC_BASE_URL"]=="https://cpu-qwen.invalid"\n'
                    'prompt=sys.stdin.read()\nassert "DOC0:" in prompt\n'
                    'print(json.dumps({"type":"assistant","message":{"content":[{"type":"thinking","thinking":"hidden"},{"type":"tool_use","name":"Read","input":{"file_path":"/fixed/schema/root.yaml"}}]}}))\n'
                    'print(json.dumps({"type":"result","structured_output":'+repr(reply())+',"result":"final only"}))\n')
    fake.chmod(0o755)
    monkeypatch.setenv('PATH',str(bindir)+os.pathsep+os.environ['PATH'])
    monkeypatch.setenv('ANTHROPIC_BASE_URL','https://cpu-qwen.invalid')
    attempt=impl.next_attempt(folder,state)
    returned=coder.invoke(impl.prompt(folder,state,attempt),folder,attempt,state,
                          lambda:coder.save(folder/'state.json',state),[folder/'inputs'])
    draft=coder.read_json(attempt/'draft.json')
    assert returned==reply() and state['calls_used']==1 and state['action'] is None
    assert draft['reads'][0]['tool']=='Read' and 'hidden' not in json.dumps(draft)
    assert 'cpu-qwen.invalid' not in (attempt/'draft.json').read_text()


def test_bad_output_is_counted_retained_and_retryable(run):
    folder,state,config,reference=run
    bad=reply();bad['files'][0]['path']='/escape.py'
    client=FakeClient(reference)
    value=impl.execute(folder,state,client_factory=lambda _:(client,'fake'),writer=writer([bad,reply()]))
    assert value['status']=='complete' and value['calls_used']==2
    assert coder.read_json(folder/'attempts/01/feedback.json')['diagnosis']['kind']=='write_failed'
    assert coder.read_json(folder/'attempts/01/draft.json')['structured_output']==bad
    assert len(client.submitted)==1


def test_service_failure_cannot_be_reinterpreted_as_repair(run):
    folder,state,config,reference=run
    broken=response(reference,failed='0',status='failed')
    broken['error']={'error_class':'infra','reason':'worker_lost'}
    client=FakeClient(reference,[broken])
    value=impl.execute(folder,state,client_factory=lambda _:(client,'fake'),writer=writer([reply()]))
    assert value['stop_reason']=='evaluation_failed'
    assert impl.execute(folder,state,new_feedback=True,client_factory=lambda _:pytest.fail('infra sent to coder'))==value


def test_observed_comparator_mismatch_does_not_pass(run):
    folder,state,config,reference=run
    view=response(reference);view['result']['runner_config']={'atol':0.5,'rtol':0.5}
    rows,env,reason,detail=agate.interpret(view,reference,config)
    assert reason=='evaluation_failed' and 'differs' in detail


def test_pending_job_is_settled_before_new_repair_feedback(run,tmp_path):
    folder,state,config,reference=run
    client=FakeClient(reference);client.fail_query_once=True
    with pytest.raises(coder.Problem):
        impl.execute(folder,state,client_factory=lambda _:(client,'fake'),writer=writer([reply()]))
    note=tmp_path/'repair.md';note.write_text('Resolve wrapper branch coverage in the same Schema.')
    impl.add_feedback(folder,state,[note])
    value=impl.execute(folder,state,new_feedback=True,client_factory=lambda _:(client,'fake'),writer=writer([reply()]))
    assert value['status']=='complete' and value['calls_used']==2
    assert state['consumed_feedback']==1 and len(client.submitted)==2
    assert coder.read_json(folder/'attempts/01/feedback.json')['diagnosis']['kind']=='done'


def test_repair_with_exhausted_calls_gets_new_failure_diagnosis(run,tmp_path):
    folder,state,config,reference=run
    state['config']['max_calls']=1
    client=FakeClient(reference)
    old=impl.execute(folder,state,client_factory=lambda _:(client,'fake'),writer=writer([reply()]))
    note=tmp_path/'repair.md';note.write_text('Required code fix after observed candidate.')
    impl.add_feedback(folder,state,[note])
    value=impl.execute(folder,state,new_feedback=True,client_factory=lambda _:(client,'fake'),writer=writer([]))
    assert value['status']=='incomplete' and value['calls_used']==1 and value['kernel'] is None
    assert value['feedback']!=old['feedback']
    assert coder.read_json(folder/value['feedback'])['diagnosis']['kind']=='implementation_failed'
    assert coder.read_json(folder/old['feedback'])['diagnosis']['kind']=='done'
