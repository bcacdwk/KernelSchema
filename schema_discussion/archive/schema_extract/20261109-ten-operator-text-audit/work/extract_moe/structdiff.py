import json,pathlib
W=pathlib.Path(__file__).parent;G=json.loads((W/'phase_structures.json').read_text());out=[]
def flat(x,p=''):
 if isinstance(x,dict) and x:return {a:v for k,z in x.items() for a,v in flat(z,p+'.'+k if p else k).items()}
 if isinstance(x,list) and x:return {a:v for i,z in enumerate(x) for a,v in flat(z,p+'['+str(i)+']').items()}
 return {p:x}
prev=[]
for x in G:
 f=flat(x['data']);best={'set':f,'delete':[]};base=None;length=len(json.dumps(best))
 for j,b in enumerate(prev):
  changes={'set':{k:v for k,v in f.items() if k not in b or b[k]!=v},'delete':[k for k in b if k not in f]};n=len(json.dumps(changes))
  if n<length:best=changes;length=n;base=j
 out.append({'i':x['i'],'contexts':x['contexts'],'base':base,**best});prev.append(f)
(W/'struct_diff.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print('phases',len(out),'compact delta chars',sum(len(json.dumps(x)) for x in out))
