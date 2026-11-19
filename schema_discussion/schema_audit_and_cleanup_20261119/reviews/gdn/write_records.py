from pathlib import Path
import hashlib,json,datetime,yaml
H=Path('schema_discussion/schema_audit_and_cleanup_20261119/reviews/gdn');B=Path('schema_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16');S='kernel_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest={p.name:sha(p) for p in sorted(B.iterdir()) if p.suffix in ('.yaml','.yml') or p.name=='dispatch.md'}
versions={p.name:yaml.safe_load(p.read_text())['META_DATA']['Version'] for p in B.glob('*.yaml')}
before=json.loads((H/'before.json').read_text());checks=json.loads((H/'checks.json').read_text())
assert manifest==checks['final_manifest']
assert all(x['exit_code']==0 for x in checks['checks'])
assert sha(B/'.extraction/receipt.json')==before['files']['.extraction/receipt.json']
for name,h in before['files'].items():assert sha(H/'before'/name)==h
for check in checks['checks']:
 assert sha(Path(check['output_path']))==check['output_sha256']
 assert check['yaml_manifest']=={n:h for n,h in manifest.items() if n.endswith('.yaml')}
for node in ['root','segmented']:
 eff=json.loads((H/('effective_'+node+'.txt')).read_text())
 for n,h in eff['source_files'].items():assert h==manifest[n]
 assert all(v['provider']==node and v['chain']==[node] for v in eff['origins'].values())
source_evidence=[
 {'id':'S1','path':S+'candidate/solution.py','lines':'19-44, 53-69','supports':'construction/forward boundary, defaults and None-cu metadata synthesis'},
 {'id':'S2','path':S+'problem/reference.py','lines':'4-33, 36-46','supports':'FP32 block solve, nullable auxiliary metadata, two returned outputs and caller core prefix'},
 {'id':'S3','paths':[S+'problem/input.py',S+'problem/metadata.json',S+'problem/shape_range.json',S+'problem/shape_cases.json'],'supports':'shape/dtype domain, history prepare, standard chunk enumeration, writeback regions and tolerance record'},
 {'id':'S4','path':S+'candidate/gdn_kernel.cu','lines':'134-178, 205-328, 336-417, 434-566, 1329-1438, 1440-1687','supports':'prep storage, guarded metadata reads, zero-filled g/beta, scan order, gram narrowing and inverse hierarchy'},
 {'id':'S5','paths':[S+'candidate/gdn_kernel.cu',S+'candidate/gdn_common.cuh'],'lines':'kernel 1914-1942, 2074-2171, 2186-2310, 2320-2434, 2553-2795; common 45-72, 258','supports':'stage layout, BF16 state/operand planes, ring reuse, RHS/output alias and exact conversion helper'},
 {'id':'S6','path':S+'candidate/gdn_kernel.cu','lines':'1782-1804, 2201-2310, 2798-2877','supports':'propagation state, scalar segment composition, weak fallback, empty segment and final-state producer'},
 {'id':'S7','path':S+'candidate/gdn_kernel.cu','lines':'2892-2992, 3086-3146, 3257-3303','supports':'dynamic slabs, input/core contiguous handling, result backing, workspace and distinct normalization buffers'},
 {'id':'S8','path':S+'candidate/gdn_kernel.cu','lines':'3163-3251, 3320-3401, 3424-3478','supports':'vtile, segmentation arms, GQA, permutation, KQTMA, PRE0 and stage-depth dispatch'},
 {'id':'S9','paths':[S+'performance/CERTIFIED_BEST.json',S+'performance/ncu/result.json',S+'problem/metadata.json'],'supports':'finite stored candidate evidence, captured source/contract identity and separate original-library validation status; comparator implementation absent'},
 {'id':'S10','path':S+'candidate/gdn_kernel.cu','lines':'2498-2528, 2530-2550, 2835-2852','supports':'vt64 Lc=1 K0 reads occur after mini barrier; final stage-0 overwrite has no intervening cross-warp retirement barrier'}
]
findings=[
 {'id':'GDN-F01','classification':'schema_fixed','files':['root.yaml'],'summary':'Clarify two returned tensors, observable core writeback, disjoint fresh result views and reference block-solve association','evidence':['S1','S2','S7']},
 {'id':'GDN-F02','classification':'schema_fixed','files':['root.yaml','segmented.yaml','dispatch.md'],'summary':'Record gram BF16 rounding, exact inverse-level write domains, distinct 16/32 versus 64 products and accumulated approximation to the FP32 task','evidence':['S4','S5']},
 {'id':'GDN-F03','classification':'schema_fixed_source_limit_preserved','files':['root.yaml','segmented.yaml','dispatch.md'],'summary':'Remove unsupported finite-domain and bitwise-flat-tail guarantees; preserve actual zero fills, masks, scan order and intermediate overflow limits','evidence':['S3','S4','S5']},
 {'id':'GDN-F04','classification':'schema_fixed_source_limit_preserved','files':['segmented.yaml','dispatch.md'],'summary':'Separate propagation pc/pc_hi from full-rate sc/sc_hi; scalar a_seg omits feedback for G>=3 and theta alone has no full-domain error bound','evidence':['S6']},
 {'id':'GDN-F05','classification':'schema_fixed','files':['root.yaml','segmented.yaml','dispatch.md'],'summary':'Correct y_scratch byte-region units, add retained pred/qpred register sets, account replicated segment decay and final-stage alias including vt64 extension','evidence':['S4','S5','S6','S7']},
 {'id':'GDN-F06','classification':'schema_fixed','files':['root.yaml','segmented.yaml','dispatch.md'],'summary':'Resolve geometry for all segment workers, preserve B1/B3 and ring retirement order, correct production prep overlap and workspace/norm allocation identities','evidence':['S4','S5','S6','S7']},
 {'id':'GDN-F07','classification':'source_limit_documented','files':['root.yaml','segmented.yaml','dispatch.md'],'summary':'cu tensor with absent ci/co is legal in reference but rejected by candidate; view-compatible noncontiguous core can lose writeback after contiguous(); no Contract narrowing','evidence':['S1','S2','S7']},
 {'id':'GDN-F08','classification':'source_hazard_documented','files':['dispatch.md'],'summary':'vt64 mini final chunk on stage-0 can overwrite another warp K0 read; concrete legal geometry and byte-32 conflict, without a source retirement barrier','evidence':['S8','S10']},
 {'id':'GDN-F09','classification':'schema_fixed','files':['root.yaml','segmented.yaml','dispatch.md'],'summary':'Synchronize exact dispatch averages, local storage/precision differences, source evidence scope and same-stream cache lifetime limitation','evidence':['S7','S8','S9']}
]
limits=[
 'No GPU execution or end-to-end candidate numerical acceptance was performed; finite CPU checks and event graphs are explicitly limited.',
 'Hardware fields remain extraction placeholders; --skip-hardware excludes hardware rules and dependent relations.',
 'External comparator formula/basis and full per-output routing are unavailable; source-only thresholds remain unchanged.',
 'Finite Contract inputs do not guarantee finite internal arithmetic; scalar segment transfers for G>=3 have no full-domain error bound.',
 'Candidate rejects cu-tensor/None-ci-co combinations and can fail caller writeback for noncontiguous core; these remain source coverage gaps.',
 'Persistent workspace/norm caches lack cross-stream invocation isolation in the fixed source.',
 'The vt64 mini-tail final-stage reuse has an unsequenced cross-warp source address conflict; the two complete selected vt32/vt16 nodes do not use this local path.'
]
record={
 'format':'schema_wiki.manual_semantic_review.v1','operator_id':'gdn_prefill_bf16_state_bf16','completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'directory':str(B),'design_count':2,'manifest':manifest,'versions':versions,'changed_files':['root.yaml','segmented.yaml','dispatch.md'],
 'review_execution':{'agent':'/root/gdn_schema_review_xhigh','roles':['worker','reviewer'],'same_worker_and_reviewer':True,'delegated':False,'method':'direct fixed-source review, targeted edits, normalized artifact reread, CPU examples, finite event graphs and current complete-directory check','new_extract_calls':0,'gpu_runs':0,'kernel_changes':False,'framework_changes':False},
 'original_delivery':{'run_id':'68edadab-4371-402c-8acf-f73eb5ec9f92','receipt_path':str(B/'.extraction/receipt.json'),'receipt_sha256':sha(B/'.extraction/receipt.json'),'receipt_unchanged':True,'approval_scope':'original before bytes only; no rebinding to current edits'},
 'before_record':{'path':str(H/'before.json'),'sha256':sha(H/'before.json')},
 'report':{'path':str(H/'REVIEW.md'),'sha256':sha(H/'REVIEW.md')},
 'findings':findings,'source_commit':'24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2','source_evidence':source_evidence,
 'source_manifest':{'path':str(H/'source_manifest.json'),'sha256':sha(H/'source_manifest.json')},
 'checks':checks['checks'],'check_record':{'path':str(H/'checks.json'),'sha256':sha(H/'checks.json')},
 'finite_evidence':{n:{'path':str(H/n),'sha256':sha(H/n)} for n in ['cpu_checks.py','cpu_checks.json','abi_cpu_checks.py','abi_cpu_checks.json']},
 'self_review':['Reread actual normalized math, storage, action descriptions, pipe conditions and current dispatch against source.','Verified separate local regions, shared root Contract, current source_files and origins for both complete designs.','Corrected propagation-state naming and identified the vt64 mini-tail source retirement hazard during final reread.','All final recorded check commands exit zero; intermediate quoting/text-budget corrections did not alter checker rules.'],
 'remaining_limits':limits,
 'cleanup_handoff':['Preserve original .extraction/receipt.json identity and before bytes; subsequent cleanup may archive historical process fields and create truthful current metadata.','No global metadata or shared index was edited by this worker.','gdn-slot-padding-coverage concerns another postconv_mtp FP32-state operator and is not adopted as a finding for this prefill operator.'],
 'outcome':'complete_schema_review_with_documented_source_limits'
}
(H/'review.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
after={'directory':str(B),'design_count':2,'manifest':manifest,'versions':versions,'changed_files':record['changed_files'],'findings':findings,'source_evidence':source_evidence,'checks':checks['checks'],'remaining_limits':limits,'record_path':str(H/'review.json'),'record_sha256':sha(H/'review.json')}
(H/'after.json').write_text(json.dumps(after,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'after_path':str(H/'after.json'),'after_sha256':sha(H/'after.json'),'review_sha256':sha(H/'review.json'),'manifest':manifest,'versions':versions},indent=2))
