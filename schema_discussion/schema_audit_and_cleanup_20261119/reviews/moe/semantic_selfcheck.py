from pathlib import Path
import ast, hashlib, json, math, struct, subprocess
import yaml
ROOT=Path(__file__).resolve().parents[4]
FORMAL=ROOT/'schema_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe'
SOURCE=ROOT/'kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe'
OUT=Path(__file__).resolve().parent
COMMIT='24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2'
# Root is the repository, not a source import: no torch, CUDA or native entry executes.
assert (ROOT/'schema_template/check_yaml.py').is_file(), ROOT
plans={p.stem:yaml.safe_load(p.read_text()) for p in FORMAL.glob('*.yaml')}
assert sorted(plans)==['mega','root','split_k']
root=plans['root']
assert [plans[k]['META_DATA']['Version'] for k in ('root','split_k','mega')]==[5,2,3]
for name,p in plans.items():
 for area in ('Math_reshape','Work_schedule','Tile_pipeline'):
  assert isinstance(p['DESIGN'][area],dict) and 'from' not in p['DESIGN'][area]
source_names=['candidate/solution.py','candidate/nvfp4_moe.cu','candidate/nvfp4_fast.c','problem/input.py','problem/reference.py','problem/metadata.json','problem/shape_range.json','problem/shape_cases.json']
source_manifest={}
for name in source_names:
 p=SOURCE/name;raw=p.read_bytes();rel=p.relative_to(ROOT).as_posix()
 fixed=subprocess.run(['git','show',COMMIT+':'+rel],cwd=ROOT,capture_output=True,check=True).stdout
 assert raw==fixed,name
 source_manifest[name]=hashlib.sha256(raw).hexdigest()
# Caller arguments and the deliberate schema name for the output destination.
def forward_args(p):
 tree=ast.parse(p.read_text())
 model=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Model')
 f=next(n for n in model.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
 return [x.arg for x in f.args.args if x.arg!='self']
contract_args=list(root['CONTRACT']['Algorithm_requirement']['Input'])
contract_args=['output' if x=='output_buffer' else x for x in contract_args]
assert set(contract_args)==set(forward_args(SOURCE/'candidate/solution.py'))==set(forward_args(SOURCE/'problem/reference.py'))
inputs=root['CONTRACT']['Algorithm_requirement']['Input']
assert inputs['topk_ids']['dtype']=='tuple(int32, fp32)'
assert inputs['gemm1_clamp_limit']['nullable'] is True
assert 'nullable' not in inputs['output_buffer'] and 'default' not in inputs['output_buffer']
# Execute only the source's pure geometry selector and literal constants.
tree=ast.parse((SOURCE/'candidate/solution.py').read_text());nodes=[]
for n in tree.body:
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('CFGS','CFGS_FLOP_PER_BYTE') for t in n.targets):nodes.append(n)
 elif isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=='CFGS':nodes.append(n)
 elif isinstance(n,ast.FunctionDef) and n.name in ('_rung7_ok','_pick_cfg'):nodes.append(n)
ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),'pure_geometry_selector','exec'),ns)
geometries=[(2048,768,128),(4096,768,128),(4096,1536,128),(7168,256,256)]
root_geometry_count=0
for H,I,E in geometries:
 for T in range(1,8193):
  R=8*T
  assert ns['_pick_cfg'](H,I,E,R)==(99 if R<=1536 else 97)
  root_pred=R>1536 and math.ceil(R/128)*(H//128)*max(1,I//256)>=3072
  if root_pred:
   root_geometry_count+=1
   assert min(E,R)*(H//128)*max(1,I//256)>=3072
   assert R>1 and T>128 and R*H*2 < (1024<<20)
# Invert the prepared payload permutation and independently flatten the scale ABI.
row_cases=0;scale_cases=0
for H,I,E in geometries:
 for N,C,gated in ((2*I,H//16,True),(H,I//16,False)):
  seen=set()
  for logical in range(N):
   half=2*logical if gated and logical<N//2 else (2*(logical-N//2)+1 if gated else logical)
   stored=(half//32)*32+8*(half%4)+(half%32)//4
   inverse=(stored//32)*32+(stored%8)*4+(stored%32)//8
   if gated:inverse=inverse//2+(inverse%2)*(N//2)
   assert inverse==logical
   seen.add(stored);row_cases+=1
   for c in range(C):
    # Contract coordinate [row_tile,col_tile,row32,4*row_group+col4].
    off=(((stored//128)*(C//4)+c//4)*32+stored%32)*16+4*((stored%128)//32)+c%4
    cuda=((stored>>7)*(C>>2)+(c>>2))*512+(stored&31)*16+((stored>>5)&3)*4+(c&3)
    assert off==cuda and 0<=off<N*C
    scale_cases+=1
  assert seen==set(range(N))
# The shared scale image is exactly 2048 bytes; every 4-byte word has one writer.
offsets=[((((r&31)>>3)*4+j)*8+(r&7))*16+((r>>5)&3)*4 for r in range(128) for j in range(4)]
assert sorted(offsets)==list(range(0,2048,4))
# Evaluate every registered Storage region for every declared geometry.
size_checks=[]
for pname,p in plans.items():
 for H,I,E in geometries:
  vals={'hidden_dim':H,'intermediate_dim':I,'expert_count':E,'route_capacity':1024}
  def evaluate(x):
   if isinstance(x,(float,int)):return x
   return eval(x,{'__builtins__':{}},vals)
  for step,body in p['DESIGN']['Tile_pipeline']['Step'].items():
   for name,s in body.get('Storage',{}).items():
    elems=math.prod(evaluate(next(iter(axis.values()))) for axis in s['shape'])
    expected=evaluate(s['depth'])*elems*{'uint8':1,'fp32':4,'int32':4}[s['dtype']]
    assert evaluate(s['total_size'])==expected,(pname,step,name)
    size_checks.append((pname,step,name,H,int(expected)))
assert 3*(128*128+128*128)+2*2*2048+16+64+64==106640
assert 106640+8320==114960
assert 128+2*16*2==192 and 192<=256
# Clipped gather boxes include repeated first-row fetches in the last partial box.
gather=[]
for valid in range(1,129):
 nbox=max(1,min(32,(valid+3)//4));coords=[r if r<valid else 0 for r in range(4*nbox)]
 assert coords[:valid]==list(range(valid))
 assert len(coords)-valid in range(4)
 assert (128+4*nbox)*128==16384+512*nbox
 gather.append({'valid':valid,'boxes':nbox,'repeated_rows':len(coords)-valid,'tx_bytes':(128+4*nbox)*128})
# CPU scalar counterexamples establish that old formulas were not interchangeable.
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def bits(x):return struct.unpack('<I',struct.pack('<f',x))[0]
def frombits(x):return struct.unpack('<f',struct.pack('<I',x))[0]
def e4(code):
 e=(code>>3)&15;m=code&7
 return m*2**-9 if e==0 else (1+m/8)*2**(e-7)
levels=[e4(c) for c in range(127)]
def rn8(x):return min(range(127),key=lambda c:(abs(levels[c]-x),c%2))
scale_example=None
for c in range(126):
 peak=f32(((levels[c]+levels[c+1])/2)*6)
 for b in range(max(0,bits(peak)-3),bits(peak)+4):
  v=frombits(b);mul=f32(v*f32(1/6));div=f32(v/6)
  if rn8(mul)!=rn8(div):
   scale_example={'peak':v,'multiply_fp32':mul,'divide_fp32':div,'scale_codes':[rn8(mul),rn8(div)]};break
 if scale_example:break
assert scale_example is not None
fold_example=None
rec=f32(1/25);l2=f32(-1.44269504);twol2=f32(2*l2)
for n in range(1,100000):
 up=f32(n/4000)
 actual=f32(up*f32(rec*twol2));old=f32(f32(up*rec)*twol2)
 if actual!=old:
  fold_example={'beta':25,'up':up,'folded_prescale':actual,'unfolded_prescale':old};break
assert fold_example is not None
before=json.loads((OUT/'before.json').read_text())
for rel,h in before['manifest'].items():
 assert hashlib.sha256((OUT/'before'/rel).read_bytes()).hexdigest()==h
assert hashlib.sha256((FORMAL/'.extraction/receipt.json').read_bytes()).hexdigest()==before['manifest']['.extraction/receipt.json']
result={'status':'passed','scope':'CPU static/source/algebra checks; no CUDA/native/reference execution','source_commit':COMMIT,'source_manifest':source_manifest,'abi_argument_count':len(contract_args),'pure_source_geometry_selector_cases':32768,'root_geometry_cases_with_reachable_explicit_controls':root_geometry_count,'root_controls':{'NVFP4_MOE_SMALL_THRESH':'1','NVFP4_MOE_FIN16MB':'1024','NVFP4_MOE_PDL':'1','NVFP4_MOE_PDL_BIG_R':'-1'},'prepared_row_checks':row_cases,'scale_offset_checks':scale_cases,'storage_checks':len(size_checks),'shared_scale_words_checked':len(offsets),'clipped_gather_valid_counts_checked':len(gather),'gather_examples':[gather[i-1] for i in (1,3,4,5,127,128)],'source_rounding_counterexamples':{'reference_scale':scale_example,'root_type10_fold':fold_example},'before_bytes_and_receipt_preserved':True,'workspace_accounting':{'gemm1_dynamic_shared':106640,'gemm2_dynamic_shared':114960,'tmem_used_columns':192,'tmem_allocated_columns':256,'act_allocation_bytes':'2*R*I','act_payload_region_bytes':'[0,R*I/2)','act_scale_region_bytes':'[R*I/2,9*R*I/16)','act_unused_allocation_bytes':'23*R*I/16','gather_allocation_bytes':'9*R*H/16','gather_payload_region_bytes':'[0,R*H/2)','gather_scale_region_bytes':'[R*H/2,9*R*H/16)','mega_dynamic_shared_bytes':'(H+H/8+96)*4+64','mega_static_shared_bytes':108}}
(OUT/'semantic_selfcheck.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
