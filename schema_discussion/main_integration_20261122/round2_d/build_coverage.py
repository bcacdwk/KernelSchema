"""Project the approved decisions into actual implementation/evidence identities."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'schema_discussion/main_integration_20261122'
def load(p):return json.loads((ROOT/p).read_text())
def ref(p):return {'path':p,'sha256':hashlib.sha256((ROOT/p).read_bytes()).hexdigest()}
oldpath='schema_discussion/extract_schema_review_20261121/COVERAGE.json';old=load(oldpath);em=load('schema_discussion/main_integration_20261122/execution_map.json')
a=load('schema_discussion/main_integration_20261122/round1_a/final_validation.json');b=load('schema_discussion/main_integration_20261122/round1_b/coverage.json')
prefix='schema_discussion/main_integration_20261122/'
commits={'W1':'1a221e440f6f64cb83c4179f1cb9d22b92c3e812','W3':'d444101c789a8c8da6c198e59003d09acb28ec17','W4':'f46b14a7ccf58bcb623f781471321a6bf97b765f','W2':'Git commit containing this implementation coverage and the frozen W2 product manifest'}
root='schema_wiki/aka-cuda-operators-20261030/operators/'
wp=em['ownership']['C']['product_paths'];Y=[x['path'] for x in em['w2']['required_yaml_amendments']]
by={
'S-01':[Y[0],root+'linear_attention/kda_raw_prefill_state_bf16/prepare_chain.yaml'],
'S-02':[Y[0]],'S-03':[root+'fmha/fmha_fp8_per_token_head/root.yaml',root+'fmha/fmha_fp8_per_token_head/decode_split_kv.yaml','schema_docs/5.design.tile_pipeline.md'],
'S-05':[root+'linear_attention/kda_raw_prefill_state_bf16/prepare_chain.yaml','schema_docs/1.contract.algorithm_requirement.md'],
'S-07':['schema_template/tools/text_policy.py']+[x['root'] for x in em['w2']['tolerance_sources']],
'S-09':wp[4:6]+em['w2']['required_dispatch_amendments']+[x for x in wp if x.endswith('current.json') or x.endswith('CURRENT_SCHEMA_INDEX.json')],
'S-10':['schema_docs/5.design.tile_pipeline.md'],'S-11':['schema_docs/4.design.work_schedule.md','schema_docs/5.design.tile_pipeline.md']}
maps={}
for item in old['findings']:
 i=item['id'];packages=item['work_packages']; paths=[];tests=[];evidence=[]
 if i in a['coverage']:
  owner=em['ownership']['A'];paths=owner['product_paths'] if i.startswith('X-0') else (['schema_template/tools/normalization.py'] if i=='S-06' else ['schema_extract/tools/runner.py','schema_extract/tools/prompts.py'])
  tests=a['coverage'][i]['tests'];evidence=[prefix+'round1_a/final_validation.json']
 elif i=='X-08' or i in ['X-09','X-10','X-11','X-12']:
  paths=em['ownership']['B']['product_paths'] if i=='X-08' else ({'X-09':['schema_extract/tools/store.py'],'X-10':['schema_extract/tools/monitor.py','schema_extract/tools/cli.py'],'X-11':['schema_extract/tools/wakeup.py'],'X-12':['schema_extract/README.md']}[i])
  key='W1:X-08' if i=='X-08' else 'W4:'+i
  tests=b[key] if isinstance(b[key],list) else [x for v in b[key].values() if isinstance(v,list) for x in v]
  evidence=[prefix+'round1_b/final_verification.json',prefix+'round1_b/coverage.json']
 elif i in by:
  paths=by[i];tests=['tools.tests.test_yaml','tools.tests.test_schema','tools.tests.test_dispatch','tools.tests.test_design_resolution'];evidence=[prefix+'W2_REVIEW.json',prefix+'W2_SUBJECT.json',prefix+'round2_c/template_tests.json',prefix+'round2_d/current_verification.json']
 else:
  evidence=[prefix+'round2_d/preservation.json',prefix+'round1_a/final_validation.json']
  tests=['Retained behavior only; existing named boundary tests and precise byte preservation, not a new claim of source/GPU correctness.']
 if i=='X-06':paths=['schema_template/tools/normalization.py']+[x for x in wp if x.endswith('current.json')];evidence += [prefix+'W2_REVIEW.json']
 state='implemented_and_verified' if item['implementation_status']=='pending' else 'explicitly_excluded' if item['decision']=='excluded' else 'retained_no_standalone_change'
 maps[i]={'id':i,'approved_decision':item['decision'],'state':state,'work_packages':packages,'disposition':item['disposition'],'products':[ref(p) for p in dict.fromkeys(paths)],'actual_tests_or_preservation':tests,'evidence':[ref(p) for p in dict.fromkeys(evidence)],'commit_binding':{p:commits[p] for p in packages}}
result={'format':'kernelschema.integration.implementation_coverage.v1','status':'all_authorized_implementation_verified','approved_coverage':ref(oldpath),'counts':{'findings':26,'framework_candidates':24,'candidate_occurrences':104,'issue_recap':20,'existing_corrections':11,'reviewed_trims':16,'final_clarifications':2},'counts_are':'Approved decision coverage counts, not tests run or independent issues. Historical study test counts are not added to implementation results.','commit_binding':commits,'findings':list(maps.values())}
for name in ['framework_candidates','issue_recap','existing_corrections','reviewed_trims','final_clarifications']:
 rows=[]
 for item in old[name]:
  ids=item.get('findings',[]) or ([item['finding']] if item.get('finding') else [])
  if name=='final_clarifications':ids=['S-09'] if item['id']=='FC-01' else ['X-04']
  rows.append({**item,'implementation_state':('approved_exclusion_preserved' if item.get('action') in ['deleted','excluded'] or item.get('decision','').startswith(('no-change','excluded','exclude-')) else 'implemented_or_retained_as_approved'),'implementation_findings':ids,'evidence_paths':sorted({x['path'] for i in ids for x in maps[i]['evidence']})})
 result[name]=rows
result['explicit_exclusions']=old['excluded_work']
result['deferred_questions']=old['deferred_questions']
result['preservation_and_integration']={'historical_baseline_commit':'41b7db0dd636f01a1d8bc490bc159c2d65b7b034','workflow_merge':'6c3859e58cbc25e4094590c5e50f39ffcdb773a3','workflow_natural_commits':37,'evidence':[ref(prefix+p) for p in ['round2_d/workflow_verification.json','round2_d/current_verification.json','round2_d/preservation.json','round2_d/compatibility_correction.json']],'navigation':'README.md and schema_wiki/README.md retain original six/15 and add twelve/33 = 18/48; nine current versus three unchanged receipt directories.','conditional_compatibility_change':'Root assigned examples/workflows/row_softmax/task/materials.json to D after pinned DOC1 failure. Only doc_1/doc_4/doc_5 sizes and hashes refreshed; no workflow parser, policy, api, iterate or receipt implementation changed.'}
result['new_implementation_results']={'round1_A_distinct_tests':186,'round1_B_tests':46,'round2_C_template_tests':64,'round2_C_complete_directory_checks':9,'round2_D_workflow':ref(prefix+'round2_d/workflow_verification.json'),'round2_D_native_provenance_cases':27,'historical_masked_cases_reused_not_rerun':4616,'real_model_gpu_extract_monitor_calls':0}
assert len(result['findings'])==26 and len(result['framework_candidates'])==24 and sum(x['occurrences'] for x in result['framework_candidates'])==104
assert [len(result[k]) for k in ['issue_recap','existing_corrections','reviewed_trims','final_clarifications']]==[20,11,16,2]
result['mandatory_pending']=[]
(BASE/'IMPLEMENTATION_COVERAGE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(result['counts'])
