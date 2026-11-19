import sys,json,pathlib,re
P=pathlib.Path(__file__).resolve().parents[2]; W=P/'work/extract_fmha'; U=json.loads((W/'indexed_units.json').read_text()); R=json.loads((W/'progress.json').read_text()); N={n:i+1 for i,n in enumerate(sorted(set(o['node'] for u in U for o in u['occurrences'])))}
if sys.argv[1]=='read':
 a,b=map(int,sys.argv[2:4])
 for i in range(a,b+1):
  x=U[i-1]; occ=x['occurrences']; contexts={o['node']+':'+o['path'] for o in occ}
  print(f"[{i}] {x['id']} {x['family']} ({len(occ)} occurrences)")
  print(x['text']);group={}
  for o in occ: group.setdefault(o['path'],[]).append(o['node'])
  print('USES '+ ' | '.join(k+' ['+','.join(str(N[n]) for n in sorted(set(v)))+']' for k,v in group.items()))
 R['units_read']=sorted(set(R['units_read'])|set(range(a,b+1)))
if sys.argv[1]=='mark':
 inds=[]
 for v in sys.argv[2].split(','):
  ab=v.split('-');inds.extend(range(int(ab[0]),int(ab[-1])+1))
 for i in inds:
  assert i in R['units_read'],i
  R['classified'][str(i)]={'category':sys.argv[3],'core':sys.argv[4],'structure':sys.argv[5],'finding_ids':sys.argv[6].split(',') if sys.argv[6]!='-' else [],'reason':sys.argv[7]}
 print('classified',len(R['classified']),'read',len(R['units_read']))
(W/'progress.json').write_text(json.dumps(R,ensure_ascii=False,indent=2))
