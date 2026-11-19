import json,pathlib,difflib,collections,re,hashlib
B=pathlib.Path('schema_extract/archive/20261109-ten-operator-text-audit'); W=B/'work/extract_moe'
u=[json.loads(s) for s in (B/'data/extract_moe/units.jsonl').read_text().splitlines()]
fams=list(dict.fromkeys(x['family'] for x in u))
# Order only; does not infer any classification.
u=sorted(u,key=lambda x:(fams.index(x['family']),not any(o['node']=='root' for o in x['occurrences']),x['occurrences'][0]['node'],x['occurrences'][0]['line']))
for i,x in enumerate(u): x['review_index']=i
(W/'indexed.json').write_text(json.dumps(u,ensure_ascii=False))
prior=collections.defaultdict(list); pack=[]
for x in u:
 i=x['review_index'];words=x['text'].split(' '); candidates=prior[x['family']]; best=None;score=0
 for p in candidates:
  ratio=len(set(words)&p['ws'])/max(len(set(words)|p['ws']),1)
  if ratio>score: score=ratio;best=p
 header=f"#{i} {x['id']} {x['family']} | "+';'.join(sorted({o['node']+':'+str(o['line'])+':'+o['path'] for o in x['occurrences']}))
 body=x['text'];base=None;ops=None
 if best and score>.42:
  sm=difflib.SequenceMatcher(None,best['words'],words,autojunk=False);ops=sm.get_opcodes();chunks=[]
  for tag,a,b,c,d in ops:
   if tag!='equal': chunks.append(f"[{a}:{b}] {json.dumps(' '.join(best['words'][a:b]),ensure_ascii=False)} => {json.dumps(' '.join(words[c:d]),ensure_ascii=False)}")
  delta=f"BASE #{best['i']} exact unchanged words; "+' ; '.join(chunks)
  if len(delta)<len(body)*.92:body=delta;base=best['i']
 pack.append({'index':i,'unit_id':x['id'],'header':header,'body':body,'base':base})
 prior[x['family']].append({'i':i,'words':words,'ws':set(words)})
(W/'packs.json').write_text(json.dumps(pack,ensure_ascii=False))
print('units',len(u),'text',sum(len(x['text']) for x in u),'read body',sum(len(x['body']) for x in pack),'delta units',sum(x['base'] is not None for x in pack))
for f in fams:
 inds=[x['review_index'] for x in u if x['family']==f];print(f,min(inds),max(inds))
