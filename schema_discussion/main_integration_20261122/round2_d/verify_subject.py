"""Independent finite W2 subject audit; reads only approved subject/source paths."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
from tools import schema
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'schema_discussion/main_integration_20261122/round2_d'
def load(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def git_bytes(commit,p):
    assert not any('archive' in x.lower() for x in Path(p).parts)
    return subprocess.check_output(['git','show',commit+':'+p],cwd=ROOT)
subject_path='schema_discussion/main_integration_20261122/W2_SUBJECT.json'
s=load(subject_path);m=load('schema_discussion/main_integration_20261122/execution_map.json')
assert s['status']=='frozen_for_independent_review'
for row in s['rules_and_checker']+s['evidence']:assert sha(row['path'])==row['sha256'],row['path']
rows=[];consumers=[];differences=[]
def changes(a,b,path=()):
    if a==b:return []
    if isinstance(a,dict) and isinstance(b,dict):
        return sum((changes(a.get(k),b.get(k),path+(str(k),)) for k in sorted(a.keys()|b.keys())),[])
    if isinstance(a,list) and isinstance(b,list) and len(a)==len(b):
        return sum((changes(x,y,path+(str(i),)) for i,(x,y) in enumerate(zip(a,b))),[])
    return ['.'.join(path)]
for d in s['directories']:
    directory=ROOT/d['directory'];files,docs=schema.read_directory(directory)
    expected={Path(n).stem for n in d['files'] if n.endswith('.yaml')}
    assert set(files)==expected
    plan=schema.normalize_directory(directory,skip_hardware=True)
    resolved=schema.resolve_documents(files,docs)
    schema.validate_dispatch(files)
    log=d['structure_check']['output'];assert sha(log['path'])==log['sha256'] and d['structure_check']['exit_code']==0
    for name,item in d['files'].items():
        assert sha(item['path'])==item['sha256'],item['path']
        if not name.endswith('.yaml'):continue
        node=Path(name).stem;raw=files[node].read_bytes()
        assert plan.candidates.get(node,raw.decode()).encode()==raw,node
        doc=docs[node].value;assert doc['META_DATA']['Version']==item['version'] and doc['META_DATA']['Source']==item['source']
        fixed=item['fixed_source'];assert sha(fixed['path'])==fixed['sha256']==hashlib.sha256(git_bytes(fixed['ref'],fixed['path'])).hexdigest()
        assert resolved.source_manifest(node)==item['dependency_closure']['source_manifest']
        assert resolved.origins_dict(node)==item['dependency_closure']['region_origins']
        assert resolved.effective_identity(node)==item['dependency_closure']['effective_identity']
        assert sorted(n+'.yaml' for n in resolved.dependency_nodes[node])==item['dependency_closure']['nodes']
        old=schema.yaml_source.loads(git_bytes(s['baseline_commit'],item['path']).decode(),node).value
        fields=changes(old,doc)
        if fields:differences.append({'path':item['path'],'changed_fields':fields})
        assert (doc['META_DATA']['Version']==old['META_DATA']['Version']+1) if item['changed'] else raw==git_bytes(s['baseline_commit'],item['path'])
        assert doc['META_DATA']['Source']==old['META_DATA']['Source']
        if node!='root':assert resolved.effective_document(node)['CONTRACT']==docs['root'].value['CONTRACT']
        rows.append({'path':item['path'],'sha256':item['sha256'],'version':item['version'],'raw_equals_preview':True,'dependency_nodes':sorted(resolved.dependency_nodes[node]),'implicit_contract_provider':None if node=='root' else 'root','region_origins':resolved.origins_dict(node)})
    if any(x['operator_id']==d['operator_id'] for x in m['w2']['tolerance_sources']):consumers += [d['directory']+'/'+node+'.yaml' for node in sorted(files)]
assert len(rows)==19 and len(consumers)==10 and len(differences)==9
aliases={'bf16_fused_moe':{'y_result':0},'mla_absorbed_fp8':{'attention_output':0,'attention_lse':1},'mla_nvfp4':{'out':0,'lse':0},'paged_indexed_latent_fp8':{'out':0},'kda_raw_prefill_state_bf16':{'attention_output':0,'final_state':0,'checkpoint_states':0}}
tolerances=[]
for spec in m['w2']['tolerance_sources']:
    doc=schema.yaml_source.loads((ROOT/spec['root']).read_text(),'root').value
    contract=doc['CONTRACT']['Algorithm_requirement'];actual=contract['Golden_reference']['Tolerance']
    assert set(actual)==set(aliases[spec['operator_id']])
    for output,index in aliases[spec['operator_id']].items():
        c=spec['criteria'][index];source=load(c['source_file']);assert sha(c['source_file'])==c['source_sha256']
        fixed=source
        for part in c['first_pointer'].strip('/').split('/'):fixed=fixed[int(part)] if isinstance(fixed,list) else fixed[part]
        assert fixed==c['values'] and output in contract['Output']
        value=actual[output];labels=dict(re.findall(r'(atol|rtol|max_mismatch_fraction|max_rel_l2|relative_to|mode)=([^,;\s]+)',value))
        for key,expected in fixed.items():
            assert key in labels
            observed=labels[key]
            assert observed=='null' if expected is None else (float(observed)==expected if isinstance(expected,(int,float)) else observed==expected)
        assert 'Source only' in value and 'comparator definition unavailable' in value and 'no defaults' in value
        assert len(value)<=300 and len(value.splitlines())<=2
        records=[]
        def walk(v,p=''):
            if isinstance(v,dict):
                if 'tolerance' in v and v['tolerance']==fixed and 'name' in v:records.append({'output':v['name'],'pointer':p+'/tolerance'})
                for k,x in v.items():walk(x,p+'/'+str(k))
            elif isinstance(v,list):
                for i,x in enumerate(v):walk(x,p+'/'+str(i))
        walk(source)
        assert set(c['source_outputs'])<=set(x['output'] for x in records)
        tolerances.append({'operator':spec['operator_id'],'schema_output':output,'text':value,'values':fixed,'source_file':c['source_file'],'source_sha256':c['source_sha256'],'pointer':c['first_pointer'],'source_aliases':c['source_outputs'],'matching_source_records':len(records),'status':'passed'})
assert len(tolerances)==9
facts=load('schema_discussion/main_integration_20261122/round2_d/source_facts.json')
for path,item in facts['fixed_sources'].items():
    assert sha(path)==item['sha256']==hashlib.sha256(git_bytes(facts['source_commit'],path)).hexdigest()
    item['matches_fixed_git_source']=True
(OUT/'source_facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2)+'\n')
result={'format':'kernelschema.integration.independent_subject_checks.v1','status':'passed','subject':{'path':subject_path,'sha256':sha(subject_path)},'actual_checks':{'yaml':19,'dispatch':9,'changed_yaml':9,'tolerance_outputs':9,'effective_tolerance_consumers':10,'raw_equals_preview':True,'actual_checker_evidence_hashes_match':True,'fixed_sources_match_git':True},'documents':rows,'tolerances':tolerances,'tolerance_consumers':consumers,'semantic_diff_fields':differences,'limits':['No full 33-design audit; only the approved changes and real dependencies.','Normalization API only previews; no product mutation.','Comparator/GPU outcomes not inferred from recorded labels.']}
(OUT/'independent_subject_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result['actual_checks']));print(json.dumps(differences,ensure_ascii=False,indent=2))
