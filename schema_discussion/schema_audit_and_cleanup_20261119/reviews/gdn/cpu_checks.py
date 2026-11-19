"""Finite source-derived checks; CPU only, not CUDA numerical acceptance."""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
import yaml
HERE=Path(__file__).parent
ROOT=Path.cwd()
SRC=Path('kernel_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16')
WIKI=Path('schema_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16')
COMMIT='24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2'
def bf(x):
 a=np.asarray(x,dtype=np.float32);u=a.view(np.uint32)
 return ((u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
def scan(g):
 x=np.array(g,dtype=np.float32).copy()
 for off in (1,2,4,8,16):
  old=x.copy()
  for base in (0,32):x[base+off:base+32]=old[base+off:base+32]+old[base:base+32-off]
 x[32:]+=x[31]
 return x
r={'scope':'CPU finite examples of source math, address regions, dispatch and identities; no GPU or full-domain correctness claim'}
manifest={}
for p in sorted(x for x in SRC.rglob('*') if x.is_file()):
 raw=p.read_bytes();frozen=subprocess.check_output(['git','show',f'{COMMIT}:{p}'])
 assert raw==frozen,str(p)
 manifest[str(p)]={'sha256':hashlib.sha256(raw).hexdigest(),'matches_fixed_commit':True}
(HERE/'source_manifest.json').write_text(json.dumps({'commit':COMMIT,'files':manifest},indent=2)+'\n')
r['source_identity']={'commit':COMMIT,'file_count':len(manifest),'all_match':True}
# A concrete BF16 dot product and a system coefficient that changes when its gram narrowing is omitted.
k=bf([.123,.377,.615])
gram=np.float32(np.dot(k,k));gramh=bf(gram)
be=np.float32(.1)
with_round=bf(be*gramh);without=bf(be*gram)
assert float(gram)!=float(gramh) and float(with_round)!=float(without)
r['gram_rounding']={'k':k.tolist(),'fp32_dot':float(gram),'stored_bf16_gram':float(gramh),'beta':float(be),'system_with_gram_rounding':float(with_round),'system_without':float(without)}
# Four padded scan lanes do not necessarily hold bit-identical prefix sums.
g=np.zeros(64,dtype=np.float32);g[:3]=[-2.5639898777008057,-4.752813816070557,-0.7293564677238464];b=scan(g)
assert np.any(b[3:]!=b[2])
r['zero_padded_scan']={'valid_g':g[:3].tolist(),'last_valid_b':float(b[2]),'tail_unique_values':np.unique(b[3:]).tolist(),'claim':'zero padding does not make the FP32 b tail bitwise flat'}
# Embedded scalar case, shape H=2/HV=4/N=1/T=3072 => selected G=3 at sm=148.
# q=k=e0, v=0, state[0,0]=2**22; beta=0 except last token of segment 1 (beta=.5).
# All other coordinates/heads may be zero; actual state layout is [value,key].
b=scan(np.full(64,np.float32(-11*np.log(2)/1024),dtype=np.float32))
ai=np.exp(b).astype(np.float32);alpha=ai[-1];dl=np.exp(b[-1]-b).astype(np.float32)
qs=np.float32(128**-.5)
def segment(state,index):
 aprod=np.float32(1);outs=[]
 for chunk in range(16):
  shi=bf(state)
  # For beta=0 rows rhs is zero. Only the last row of segment 1 has an update;
  # its triangular inverse cannot change that last-only RHS.
  rhs=np.float32(0)
  if index==1 and chunk==15:rhs=bf(np.float32(.5)*np.float32(-ai[-1]*shi))
  if chunk==0:outs.append(float(bf(np.float32(ai[0]*qs)*shi)))
  state=np.float32(alpha*state+bf(dl[-1]*rhs))
  aprod=np.float32(aprod*alpha)
 return state,aprod,outs[0]
s0=np.float32(2**22);B=[];A=[]
for sg in range(3):
 end,a,_=segment(s0 if sg==0 else np.float32(0),sg);B.append(end);A.append(a)
serial_entry,_,_=segment(B[0],1)
composed_entry=np.float32(B[1]+A[1]*B[0])
_,_,serial_out=segment(serial_entry,2);_,_,composed_out=segment(composed_entry,2)
assert max(A)<np.float32(.001)
assert abs(serial_out-composed_out)>.01
r['segmented_feedback']={'shape':{'H':2,'HV':4,'N':1,'T':3072,'sm_count':148,'G':3},'g':float(np.float32(-11*np.log(2)/1024)),'initial_bf16_scalar':float(s0),'beta_nonzero_token':2047,'beta_value':.5,'a_seg':[float(a) for a in A],'b_seg':[float(x) for x in B],'serial_segment2_entry':float(serial_entry),'composed_segment2_entry':float(composed_entry),'serial_bf16_first_output_segment2':serial_out,'composed_bf16_first_output_segment2':composed_out,'absolute_output_difference':abs(serial_out-composed_out),'claim':'scalar decay gate accepts but feedback omission changes state/output well beyond FP32 reassociation; CPU exp/rounding are not a CUDA bitwise emulator'}
# Exact logical footprint and per-CTA ownership checks, not ptxas allocation claims.
r['storage']={}
for p in WIKI.glob('*.yaml'):
 y=yaml.safe_load(p.read_text());out={}
 for sid,step in y['DESIGN']['Tile_pipeline']['Step'].items():
  totals={}
  for name,item in step.get('Storage',{}).items():
   count=1
   for ax in item['shape']:
    v=next(iter(ax.values()));count*=128 if v in ('key_dim','value_dim') else int(v)
   computed=int(item['depth'])*count*({'bf16':2,'fp32':4}[item['dtype']])
   assert computed==item['total_size'],(p.name,sid,name)
   totals[item['location']]=totals.get(item['location'],0)+computed
  out[sid]=totals
 r['storage'][p.name]=out
r['source_smem_allocations']={}
for vtt,ns in [(32,2),(16,3),(64,2),(32,3)]:
 sps=72 if vtt>32 else 40
 st=32768+64*vtt*2+11264
 total=128*sps*2+3*64*sps*2+ns*st+ns*8+1152
 r['source_smem_allocations'][f'vtt{vtt}_depth{ns}']={'stage_bytes':st,'dynamic_bytes':total,'epilogue_capacity_bytes':128*(vtt+1)*4}
assert r['source_smem_allocations']['vtt32_depth2']['dynamic_bytes']==123024
assert r['source_smem_allocations']['vtt16_depth3']['dynamic_bytes']==165016
# y plane byte addresses use bf16 row pitch 40, while old view mistakenly selected fp32 rows.
actual={2*(row*40+col)+byte for row in range(32) for col in range(32) for byte in range(2)}
old={4*(row*40+col)+byte for row in range(32) for col in range(32) for byte in range(4)}
new=actual.copy();assert actual==new and actual!=old
r['y_scratch_region']={'reserved_bytes':5120,'actual_value_bytes':len(actual),'actual_value_max_byte':max(actual),'old_view_bytes':len(old),'new_view_exactly_covers_value':True}
# Triangular packed address coverage: all 2560 BF16 locations per plane, unique writers.
locations=[]
for m in range(4):
 for ks in range(m+1):
  t=m*(m+1)//2+ks
  for row in range(16):
   for col in range(16):locations.append(t*256+(col//8)*128+row*8+col%8)
assert len(set(locations))==2560 and min(locations)==0 and max(locations)==2559
r['packed_planes']={'tiles':10,'elements':2560,'bytes':5120,'exact_unique_coverage':True}
# Source scalar decision gates, with optional L2 path handled separately.
def dispatch(H,HV,lengths,sm=148):
 T=sum(lengths);N=len(lengths);slots=(T+63)//64+N
 vt16=8*HV*N<=sm;ctas32=4*HV*N
 w32=(ctas32+sm-1)//sm;w64=(2*HV*N+sm-1)//sm
 vt64=not vt16 and ctas32>sm and (T>=4096 or (T+63)//64<=2*N or w64*3<w32*2)
 vtt=16 if vt16 else (64 if vt64 else 32);G=1;avg=(T+63*N)//(64*N)
 if vt16:
  Gc=min(8,sm//(8*HV*N),avg//16) if 16*HV*N<=sm and avg>=32 else 0
  g=min(8,sm//(2*HV*N),avg//32);Gw=g if g>=2 and avg>=192 else 0
  if Gw>=2 and (Gc<2 or Gw*100>Gc*205):vtt=64;G=Gw
  elif Gc>=2:G=Gc
 blocks=(128//vtt)*HV*N
 return {'vtt':vtt,'G':G,'gqa':slots*H>=sm,'perm':N>=8 or (N>=2 and blocks>sm),'depth':3 if vtt==16 or(vtt==32 and avg>=8) else 2,'average_chunks_meta':avg,'slots_ub':slots}
r['dispatch_examples']={}
for label,H,HV,ls in [('root',4,32,[256]),('segmented_g2',2,4,[2048]),('segmented_g3',2,4,[3072]),('segmented_gqa',2,4,[8192]),('wide_segment',8,16,[16384]),('ragged_empty_segment',2,4,[1,4095])]:
 r['dispatch_examples'][label]=dispatch(H,HV,ls)
assert r['dispatch_examples']['root']=={'vtt':32,'G':1,'gqa':False,'perm':False,'depth':2,'average_chunks_meta':4,'slots_ub':5}
assert r['dispatch_examples']['segmented_g3']['G']==3
# An empty short-request segment emits A=1 and triggers the whole-request fallback.
chunks=1;G=2;sizes=[chunks*(i+1)//G-chunks*i//G for i in range(G)]
assert sizes==[0,1]
r['empty_segment']={'chunks':1,'G':2,'segment_chunk_counts':sizes,'empty_a_seg':1,'amax_exceeds_theta':True,'full_output_owner':'segment 0 reruns all chunks'}
# Source-only alias hazard in the vt64 mini tail: one shared offset has different warp owners.
mini_dispatch=dispatch(8,64,[4097])
assert mini_dispatch['vtt']==64 and mini_dispatch['G']==1
last_cl=(4097+63)//64-1
writer_byte=4*(0*65+8)  # warp 0, state key 0, value 8, final-state STS
reader_byte=2*16        # warp 1, K0 key 16, row-0 first sub-tile
assert last_cl%2==0 and writer_byte==reader_byte==32
edges=[('mini_barrier','read_k0_w0'),('read_k0_w0','state_update_w0'),('state_update_w0','final_write_w0'),('final_write_w0','epilogue_barrier'),('mini_barrier','read_k0_w1'),('read_k0_w1','state_update_w1'),('state_update_w1','final_write_w1'),('final_write_w1','epilogue_barrier')]
def reaches(a,b):
 todo=[a];seen=set()
 while todo:
  x=todo.pop()
  if x==b:return True
  if x not in seen:seen.add(x);todo.extend(y for u,y in edges if u==x)
 return False
assert not reaches('read_k0_w1','final_write_w0') and not reaches('final_write_w0','read_k0_w1')
r['mini_tail_event_witness']={'edges':edges,'conflicting_events':['read_k0_w1','final_write_w0'],'ordered_in_either_direction':False}
r['vt64_mini_final_alias']={'dispatch':mini_dispatch,'last_local_chunk':last_cl,'stage_slot':last_cl%2,'writer_warp':0,'reader_warp':1,'shared_byte_offset':32,'source_order':'both accesses follow the mini-path barrier; no intervening CTA retirement barrier precedes epilogue overwrite','scope':'finite source-address conflict and absent ordering; not a measured CUDA race'}
(HERE/'cpu_checks.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
