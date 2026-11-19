from pathlib import Path
import json,subprocess,hashlib,os
HERE=Path(__file__).parent;PY='/opt/venv/python3.13.12/bin/python3'
B=Path('schema_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16')
records=[]
def run(label,args,env=None):
 p=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
 path=HERE/(label+'.txt');path.write_text(p.stdout)
 record={'name':label,'yaml_manifest':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in B.glob('*.yaml')},'command':args,'exit_code':p.returncode,'output_path':str(path),'output_sha256':hashlib.sha256(path.read_bytes()).hexdigest()};records.append(record)
 print(label,p.returncode,p.stdout[-600:])
 if p.returncode:raise SystemExit(p.returncode)
def graph(label,node,params):
 args=[PY,'-B','schema_template/check_yaml.py',str(B/node),'--read-only','--skip-hardware','--graph','check']
 for k,v in params.items():args+=['--set',f'{k}={v}']
 args+=['--worktiles','1'];run(label,args)
run('final_directory_check',[PY,'-B','schema_template/check_yaml.py',str(B/'root.yaml'),'--read-only','--skip-hardware'])
run('cpu_check_run',[PY,'-B',str(HERE/'cpu_checks.py')],dict(os.environ,OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES=''))
run('abi_cpu_check_run',[PY,'-B',str(HERE/'abi_cpu_checks.py')],dict(os.environ,OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES=''))
root=dict(total_tokens=256,key_head_count=4,value_head_count=32,request_count=1,group_ratio=8,chunk_count=4,total_chunks=4,slot=0,hv=0,output_final_state=1,use_qk_l2norm_in_kernel=1)
graph('graph_root_four_chunks','root.yaml',root)
root.update(total_tokens=64,chunk_count=1,total_chunks=1,slot=1,output_final_state=0,use_qk_l2norm_in_kernel=0)
graph('graph_root_empty_prep_one_chunk','root.yaml',root)
seg=dict(total_tokens=3072,key_head_count=2,value_head_count=4,request_count=1,group_ratio=2,nseg=16,total_chunks=48,slot=0,hv=0,output_final_state=1,use_qk_l2norm_in_kernel=0,seg_count=3,sg=2,weak=0)
graph('graph_segmented_composed','segmented.yaml',seg)
seg.update(sg=1,weak=1)
graph('graph_segmented_weak_exit','segmented.yaml',seg)
for node in ['root','segmented']:
 run('effective_'+node,[PY,'-B','schema_template/check_yaml.py',str(B/(node+'.yaml')),'--show-effective'])
(HERE/'checks.json').write_text(json.dumps({'checks':records,'all_exit_zero':True,'scope':'Current directory checker, finite representative-worker graphs, CPU source-derived examples and effective provenance; no GPU acceptance.'},indent=2)+'\n')
