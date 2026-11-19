from pathlib import Path
base=Path('schema_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16')
def rep(s,a,b):
 if a not in s: raise ValueError(a[:120])
 return s.replace(a,b)
for name in ('root.yaml','segmented.yaml'):
 p=base/name;s=p.read_text();s=rep(s,'  Version: 2','  Version: 3')
 s=rep(s,'the row scale is x * rsqrt(ss + 1e-6) with the device rsqrt intrinsic (~2 ulp fp32); frozen inputs keep row sums finite','the row scale uses device rsqrtf; unknown: end-to-end error and intermediate finiteness over all finite Contract inputs')
 s=rep(s,'Pad rows i >= lc: g and beta act as 0 while loaded q/k/v bytes stay finite (foreign rows or hardware zero fill), so rhs/u/du rows >= lc are exactly 0, b is a flat tail past lc, and every formula below evaluates unconditionally on 64 rows without poisoning states or written outputs.','For i >= lc, g and beta are zero-filled; q/k/v loads retain following-request rows or tensor-boundary zero fill. All 64 rows are evaluated; only i < lc are stored. Zero beta contains padded rhs/u/du only when intermediate products remain finite; it does not suppress NaN from 0*inf.')
 s=rep(s,'Approx: every exp uses the device fast-exp intrinsic (__expf class); the frozen gate domain keeps |b| <~ 35 and all factors finite, ~1e-7 relative per factor','Approx: "every exp uses __expf; unknown: full-domain error and finiteness of anchors/rho products; sampled gates do not bound cumulative b"\n          Accumulate: b uses two 32-lane inclusive shuffle scans and one upper-warp correction; zero-filled tail sums can differ by FP32 association')
 s=rep(s,'gram(i,j) = sum_d k_n[tau(i),kh,d] * k_n[tau(j),kh,d]   (bf16 operands, fp32 accumulate)','gram(i,j) = bf16(dot32(k_n[tau(i),kh,:], k_n[tau(j),kh,:]))')
 s=rep(s,'qk(i,j) = sum_d q_n[tau(i),kh,d] * k_n[tau(j),kh,d]     (bf16 operands, fp32 accumulate)','qk(i,j) = dot32(q_n[tau(i),kh,:], k_n[tau(j),kh,:])')
 s=rep(s,'tinv = the unit lower triangular inverse of system, evaluated by the next step','tinv = the rounded hierarchical approximation of system^{-1} defined by the next step')
 s=rep(s,'gram and qk products strictly above the diagonal are never consumed and are not evaluated.','Only ten lower 16x16 tiles are evaluated; strict-upper elements inside diagonal tiles are computed then masked. Entire strict-upper tiles are skipped.')
 s=rep(s,"meaning: 'unit lower triangular inverse of system, bf16'","meaning: 'rounded unit lower triangular inverse approximation'")
 s=rep(s,'Gain: strictly-upper products are not evaluated where unconsumed (~17% of the gram/qk mma)','Gain: skip six strict-upper 16x16 tiles; the last two skips remove 17% of the prior twelve-tile work')
 s=rep(s,'tinv = system^{-1}; rows and columns partition into eight 8x8 diagonal blocks.','tinv approximates system^{-1}; rows and columns partition into eight 8x8 diagonal blocks.')
 s=rep(s,'for d = 1..7','for 1 <= d < 8-j')
 s=rep(s,'Merge levels 16, 32, 64:','Merge levels 16 and 32:')
 s=rep(s,'z = invd2 @ l21 (fp32 accumulate);  x = -bf16(z) @ invd1 (fp32 accumulate);','z = mm32(invd2, l21);  x = -mm32(bf16(z), invd1);')
 s=rep(s,'Level 64 materializes y = bf16(l21 @ invd1) as a [32][32] bf16 plane, then x = -invd2 @ y;','Level 64 materializes y = bf16(mm32(l21, invd1)) as a [32][32] bf16 plane, then x = -mm32(invd2, y);')
 s=rep(s,'pred = k_c @ sc_hi(c);  qpred = q_c @ sc_hi(c)   (fp32 accumulate)','pred = mm32(k_c, sc_hi(c));  qpred = mm32(q_c, sc_hi(c))')
 s=rep(s,'u = tinv @ rhs_h; u(i,.) equals the Contract delta(tau(i)) for i < lc;','u = mm32(tinv, rhs_h); u(i,.) approximates the Contract delta(c*64+i) for i < lc;')
 s=rep(s,'attn2 = qkd @ u_h;','attn2 = mm32(qkd, u_h);')
 s=rep(s,'ds = k_c^t @ du_h;  sc(c+1) = ai(63) * sc(c) + ds, equal to the Contract s at token cu_seqlens[r]+(c+1)*64-1','ds = mm32(k_c^t, du_h);  sc(c+1) = ai(63) * sc(c) + ds, approximating Contract s(min((c+1)*64,length_r)-1)')
 s=rep(s,'final_state[r,h,v,d] = sc(chunks(r))(d,v) in fp32, equal to the Contract s(length_r - 1), no further narrowing.','If requested, final_state[r,h,v,d] = sc(chunks(r))(d,v) in fp32, with the accumulated approximation error and no further narrowing.')
 s=rep(s,'Pad rows >= lc carry the containment convention in every published value: flat b tail, zero beta, finite ai/dl, foreign-finite or zero mirror bytes.','Pad rows publish zero beta and scanned b/ai/dl with the arithmetic limits above; mirror bytes include foreign rows or tensor-boundary zeros.')
 s=rep(s,'Gain: one contiguous bulk copy of an L2-resident image per chunk; the input tile is read once per (slot,kh) and shared by the group','Gain: one bulk image copy per chain chunk; one mirror writer per (slot,kh); this per-value-head prep still reloads q/k for each head')
 s=rep(s,'Cost: extra 32,768B write and read per (slot,kh), paying only while the footprint stays within L2; 11,264 slot bytes per (slot,h)','Cost: mirror storage 32,768B per (slot,kh), slot storage 11,264B per (slot,h); each chain value tile reloads its bundle')
 s=rep(s,'y_scratch: {meaning: fp32 scratch reused as the bf16 level-64 y plane, depth: 1, location: smem, shape: [{row: 32}, {column: 40}],\n            dtype: fp32, total_size: 5120, view: {y_plane: {region: {column: [0, 32]}}}}','y_scratch: {meaning: reserved scratch holding the bf16 level-64 y plane at 40-element pitch, depth: 1, location: smem, shape: [{row: 64}, {column: 40}],\n            dtype: bf16, total_size: 5120, view: {y_plane: {region: {row: [0, 32], column: [0, 32]}}}}')
 s=rep(s,'skeleton (unit diagonal, zeros)','skeleton (unit diagonal, zeros) before the per-head build')
 s=rep(s,'overlaps the serial scan with the in-flight operand boxes: preamble wall max(cp-wait+rho, box-wait+gram); bitwise-identical fp32 order','overlaps the scan with in-flight operand boxes; the production join completes both before gram: max(cp-wait+rho, box-wait)+gram')
 s=rep(s,'out/final_state stay fresh per call, carved after the prep launch','out/final_state occupy disjoint views of one fresh byte backing, carved after the prep launch')
 s=rep(s,'same-stream order is the only lifetime guarantee the never-returned scratch needs','same-stream reuse is ordered; the source adds no exclusion for concurrent calls on different streams')
 s=rep(s,"o staging, dead since the B1-joined reads","o staging, dead since the B1-joined reads") if 'o staging, dead since the B1-joined reads' in s else s
 s=rep(s,'rhh, dead since the B1-joined reads; bytes equal out','rhh, whose rhs readers retired at B3; bytes equal out')
 s=rep(s,'stage sreg through the free stage-0 region at odd pitch, join, then a pipelined transposed float4 sweep writes the final_state slice','stage sreg in stage_kq.final_stage at odd pitch, then join and transpose by float4 stores; stage-0 K/Q reads retired by B3 or an earlier loop-top join, with all issued stage writes completed')
 s=rep(s,'view: {kq_mirror(c): {slot_index: c%2}}','view: {kq_mirror(c): {slot_index: c%2}, final_stage: {slot_index: 0}}') if name=='root.yaml' else rep(s,'view: {kq_mirror(c): {slot_index: c%3}}','view: {kq_mirror(c): {slot_index: c%3}, final_stage: {slot_index: 0}}')
 if name=='root.yaml':
  s=rep(s,'The token recurrence becomes an equivalent 64-token chunked block solve:','The recurrence is algebraically chunked into a 64-token block solve, then approximated by explicit BF16 rounding:')
  s=rep(s,'out and final_state are newly allocated, out does not alias core_attn_out, core_attn_out is the only mutated object, and the storage of every input is preserved.','out and requested final_state are fresh result tensors; their element regions may share a new backing but do not overlap. out does not alias core_attn_out; only core_attn_out is mutated, and input backing allocations are preserved.')
  s=rep(s,'Gap: Constructor fault knob (default None) perturbs reference math when set; acceptance uses None.','Gap: Reference evaluates an FP32 block solve, so association differs from token-order recurrence. Constructor fault must be None to preserve the task math.')
  s=rep(s,'parity wait on the slot c%2 mbarrier, then CTA join; done publishes chunk c and retires the chunk c-1 reads of that slot','parity wait on slot c%2, then CTA join; done publishes c and retires prior-chunk reads before prefetch reuses the prior slot')
  # Document the sustained pred/qpred accumulator sets, per CTA, not per thread.
  needle='          shi: {meaning:'
  addition='          pred_acc: {meaning: even and odd pred accumulators until rhs consumes their sum, depth: 1, location: register, shape: [{parity: 2}, {row: 64}, {value: 32}],\n            dtype: fp32, total_size: 16384}\n          qpred_acc: {meaning: even and odd qpred accumulators retained through the output stage, depth: 1, location: register, shape: [{parity: 2}, {row: 64}, {value: 32}],\n            dtype: fp32, total_size: 16384}\n'
  s=rep(s,needle,addition+needle)
  s=rep(s,'planes, mirrors and normalized operands live in grow-only persistent buffers','planes/mirrors share a grow-only workspace; q_n/k_n use two separate grow-only normalized buffers')
 else:
  s=rep(s,'seg_count=2 bitwise, >=3 theta-gated with exact monolithic fallback','seg_count=2 preserves the same-VTT state path for finite arithmetic, >=3 approximates feedback; the scalar theta gate has no full-domain error bound')
  s=rep(s,'seg_count = 2 is the exact b_seg(0) roundtrip (bitwise); seg_count >= 3 drops cross-segment delta feedback (~2^-24 reassociation).','For finite arithmetic, seg_count = 2 only transfers b_seg(0). For seg_count >= 3, this replaces the true segment state-transition matrix by scalar a_seg and drops key/beta feedback, in addition to FP32 association error.')
  s=rep(s,'Approx: seg_count >= 3 drops cross-segment delta feedback; error declared bounded by the theta gate class, O(1) in the slow-decay regime the fallback excludes','Approx: "seg_count >= 3 replaces feedback transition by scalar decay; unknown: error bound over Contract inputs; amax <= theta alone is not a tolerance guarantee"')
  s=rep(s,'Cost: state path evaluated twice per chunk; transfers 0 -> seg_count*key_dim*value_dim*4 B per (r,h); seg_count >= 3 adds ~2^-24 and theta-bounded error','Cost: state path evaluated twice; per (r,h), transfers require seg_count*(key_dim*value_dim+value_dim/16)*4 B including one replicated a_seg per value tile; feedback approximation adds error')
  s=rep(s,'PASS_C composes the true entry','PASS_C composes an approximate entry')
  s=rep(s,'planes, mirrors, normalized operands and segment transfers live in one grow-only persistent workspace (transfers in the tail pad)','planes, mirrors and segment transfers share a grow-only workspace (transfers after the perm pad); normalized operands use two separate grow-only buffers')
  s=rep(s,'pass A\'s entry trigger is safe since all its CTAs are resident by then','pass C may start early but cannot read transfers until pass A completes')
  # Metadata is resolved by all segments, including segments that never read state0.
  for step in ('PASS_A','PASS_C'):
   start=s.index('      '+step+':',s.index('  Tile_pipeline:'));end=s.find('\n      PASS_',start+8)
   if end<0:end=s.index('\nREALIZATION:',start)
   block=s[start:end]
   block=rep(block,'                  barrier_init:','                  geometry: {execution_mode: serial, describe: all segments resolve n, start, chunks, seg_lo, nseg and slot0 from input cu/chunk_offsets; pass A initializes aprod to one}\n                  barrier_init:')
   block=rep(block,'segment 0 resolves n, seg_lo, nseg, slot0 from cu/chunk_offsets and loads the transposed initial_state slice into sreg; pre-wait input read','segment 0 loads the transposed initial_state slice into sreg using geometry; pre-wait input read')
   block=rep(block,'              - signal:launch_dependents','              - signal:launch_dependents\n              - compute:geometry')
   addition='          pred_acc: {meaning: even and odd pred accumulators until rhs consumes their sum, depth: 1, location: register, shape: [{parity: 2}, {row: 64}, {value: 16}],\n            dtype: fp32, total_size: 8192}\n'
   if step=='PASS_C':addition+='          qpred_acc: {meaning: even and odd qpred accumulators retained through the output stage, depth: 1, location: register, shape: [{parity: 2}, {row: 64}, {value: 16}],\n            dtype: fp32, total_size: 8192}\n'
   block=rep(block,'          shi: {meaning:',addition+'          shi: {meaning:')
   s=s[:start]+block+s[end:]
 p.write_text(s)
