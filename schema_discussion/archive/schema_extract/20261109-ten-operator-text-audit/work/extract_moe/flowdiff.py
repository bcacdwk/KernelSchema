import pathlib,json
W=pathlib.Path(__file__).parent;g=json.loads((W/'flow_groups.json').read_text());out=[];prev=[]
def flat(x,p=''):
 if isinstance(x,dict) and x:return {a:v for k,z in x.items() for a,v in flat(z,p+'.'+k if p else k).items()}
 if isinstance(x,list) and x:return {a:v for i,z in enumerate(x) for a,v in flat(z,p+'['+str(i)+']').items()}
 return {p:x}
for i,x in enumerate(g):
 f=flat(x['flow']);best={'set':f,'delete':[]};base=None;length=len(json.dumps(best))
 for j,b in enumerate(prev):
  z={'set':{k:v for k,v in f.items() if k not in b or b[k]!=v},'delete':[k for k in b if k not in f]};n=len(json.dumps(z))
  if n<length:best=z;length=n;base=j
 out.append({'i':i,'nodes':x['nodes'],'base':base,**best});prev.append(f)
(W/'flow_diff.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print('flow delta chars',sum(len(json.dumps(x)) for x in out))
