import json,pathlib
W=pathlib.Path(__file__).parent
U=json.loads((W/'indexed.json').read_text()); P=W/'manual.json'; A=json.loads(P.read_text()) if P.exists() else {}
def mark(spec,cat,core,structure,reason,tags=()):
 for part in spec.split(','):
  parts=list(map(int,part.split('-'))); ids=range(parts[0],parts[-1]+1)
  for i in ids:
   if str(i) in A:raise ValueError('duplicate '+str(i))
   A[str(i)]={'category':cat,'core':core,'structure':structure,'reason':reason,'finding_tags':list(tags)}
def save():
 P.write_text(json.dumps(A,ensure_ascii=False,indent=2)+'\n');(W/'progress.json').write_text(json.dumps({'reviewed_count':len(A),'reviewed_ids':[U[int(i)]['id'] for i in A],'missing_indices':[i for i in range(len(U)) if str(i) not in A]},indent=2)+'\n');print('saved',len(A))
