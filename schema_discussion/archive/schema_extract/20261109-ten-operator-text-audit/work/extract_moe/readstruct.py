import json,pathlib,sys
W=pathlib.Path(__file__).parent;G=json.loads((W/'struct_diff.json').read_text());a,b=map(int,sys.argv[1:]);rep={'work.decomposition':'D','work.mapping':'M','work.schedule':'S','pipeline.resource':'R','pipeline.role':'L','pipeline.domain':'I','pipeline.milestone':'Milestone','member_slot_shape':'shape','worker_assignment':'assign','execution_mode':'mode','single_tile':'out'}
def short(p):
 for k,v in rep.items():p=p.replace(k,v)
 return p
for x in G[a:b+1]:
 print('#P'+str(x['i'])+' base='+str(x['base'])+' contexts='+','.join(x['contexts']))
 print('SET '+json.dumps({short(k):v for k,v in x['set'].items()},separators=(',',':')))
 if x['delete']:print('DELETE '+json.dumps([short(k) for k in x['delete']],separators=(',',':')))
