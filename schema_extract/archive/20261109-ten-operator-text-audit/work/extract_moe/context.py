import json,pathlib,yaml,hashlib,collections
B=pathlib.Path('schema_extract/archive/20261109-ten-operator-text-audit');W=B/'work/extract_moe'; manifest=json.loads((B/'data/manifest.json').read_text())['operators']['extract_moe']; census={n['node']:n for n in json.loads((B/'data/extract_moe/nodes.json').read_text())}; docs={}
for n in manifest['nodes']:
 if n['semantic_eligible']:docs[n['node']]=yaml.safe_load((B/'inputs/extract_moe'/n['file']).read_text())
remove={'thesis','strategy','describe'}
def stripped(x):
 if isinstance(x,dict):
  return {k:({a:{'execution_mode':z['execution_mode']} for a,z in v.items() if isinstance(z,dict) and 'execution_mode' in z} if k=='verb' else stripped(v)) for k,v in x.items() if k not in remove}
 if isinstance(x,list):return [stripped(v) for v in x]
 return x
G={};records=[]
for node,d in docs.items():
 p=census[node]['effective_provider']; areas={a:docs[p[a]]['design'][a] for a in p};w=areas['work_schedule'];t=areas['tile_pipeline']; signatures=[]
 for phase in w['phase']:
  data={'work':stripped(w['phase'][phase]),'pipeline':stripped(t['phase'][phase])};raw=json.dumps(data,sort_keys=True,separators=(',',':'));h=hashlib.sha256(raw.encode()).hexdigest()
  if h not in G:G[h]={'i':len(G),'data':data,'contexts':[]}
  G[h]['contexts'].append(node+':'+phase);signatures.append([phase,G[h]['i']])
 records.append({'node':node,'status':census[node]['status'],'parent':d.get('meta_data',{}).get('dispatch',{}).get('parent','root'),'providers':p,'phase_groups':signatures,'flow':w['flow'],'realization_items':{k:[x['item'] for x in v] for k,v in d.get('realization',{}).items()},'math_attributes':{s: {k:v for k,v in val.items() if k!='meaning'} for r in areas['math_reshape']['rewrite'] for s,val in r.get('symbol',{}).items()}})
(W/'phase_structures.json').write_text(json.dumps(list(G.values()),ensure_ascii=False,indent=2));(W/'context_records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2));print('nodes',len(records),'unique phases',len(G))
# preserve all occurrence bindings for a separate closed coverage table
U=[json.loads(x) for x in (B/'data/extract_moe/units.jsonl').read_text().splitlines()];(W/'occurrences.json').write_text(json.dumps({x['id']:x['occurrences'] for x in U},ensure_ascii=False))
