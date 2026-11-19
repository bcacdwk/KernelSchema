from pathlib import Path
import re
B=Path('schema_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16')
changes={
 'rounded unit lower triangular inverse approximation':'rounded lower-triangular inverse approximation',
 'reserved scratch holding the bf16 level-64 y plane at 40-element pitch':'level-64 BF16 y plane; 40-element row pitch',
 'even and odd pred accumulators until rhs consumes their sum':'even/odd pred accumulators retained until rhs',
 'even and odd qpred accumulators retained through the output stage':'even/odd qpred accumulators retained until o_stage',
 'all threads build system in sys_plane with 4-column anchored decay factors and init the tinv_plane skeleton (unit diagonal, zeros) before the per-head build':'init tinv_plane identity/zeros once; build sys_plane with 4-column anchored decay factors',
 'eight key steps per warp: A = k_n then q_n rows of kq_mirror(c), B = this warp columns of sc_hi(c); even/odd fp32 sets in pred_acc/qpred_acc yield pred/qpred':'eight key steps: k_n/q_n from kq_mirror(c) times warp columns of shi.sc_hi(c), yielding even/odd FP32 pred_acc/qpred_acc',
 'eight key steps per warp: A = k_n rows of k_mirror(c), B = this warp columns of sc_hi(c), one x2 k-major load; even/odd fp32 sets in pred_acc yield pred':'eight key steps: k_n from k_mirror(c) times warp columns of shi.sc_hi(c), x2 k-major loads, yielding even/odd FP32 pred_acc',
 'stage sreg in stage_kq.final_stage at odd pitch, then join and transpose by float4 stores; stage-0 K/Q reads retired by B3 or an earlier loop-top join, with all issued stage writes completed':'stage sreg in stage_kq.final_stage at odd pitch, join, then float4 transpose; B3/prior loop join retired K/Q reads; all stage writes completed',
 'Reference evaluates an FP32 block solve, so association differs from token-order recurrence. Constructor fault must be None to preserve the task math.':'FP32 block solve changes association; constructor fault must be None for task math',
 'Monolithic vt32 execution: optional normalization prefix, one prep and one MODE0 chain grid. With runtime sm_count/l2_bytes and slots_ub':'Monolithic vt32: optional normalization, prep then MODE0. With sm_count/l2_bytes and slots_ub',
 'The recurrence is algebraically chunked into a 64-token block solve, then approximated by explicit BF16 rounding: intra-chunk feedback folds into a unit lower triangular system whose inverse yields the deltas, so chunks advance the fp32 register state by bf16 tensor-core products; derived mma operands narrow to single bf16 planes (~2^-9).':'A 64-token triangular block solve shortens the serial recurrence. Explicit BF16 narrowing of operands and intermediate inverses approximates the FP32 task; chunks advance authoritative FP32 state with BF16 tensor-core products.',
 'The chunked block solve is kept; the serial chain is cut into seg_count segments: a propagation-rate pass publishes per-segment affine transfers (end state b_seg, decay a_seg), composed full-rate reruns own all outputs; seg_count=2 preserves the same-VTT state path for finite arithmetic, >=3 approximates feedback; the scalar theta gate has no full-domain error bound.':'Split the chunk chain into segments. Pass A publishes end states and scalar decays; pass C combines them and reruns all outputs. G=2 preserves finite same-VTT state arithmetic; G>=3 approximates feedback. The theta gate has no full-domain error bound.',
 'per (r, h), fp32 unless a cast is written; k_c and q_c are the [64][key_dim] chunk-row matrices of k_n and q_n for kh;':'per (r,h), k_c/q_c are [64][key_dim] rows of k_n/q_n for kh;',
 'seg_count >= 3 replaces feedback transition by scalar decay; unknown: error bound over Contract inputs; amax <= theta alone is not a tolerance guarantee':'G>=3 replaces feedback by scalar decay; unknown: full-domain error bound; amax <= theta is not a tolerance guarantee',
 'fp32 descending-i accumulation with a running coefficient, empty products 1; seg_count = 2 is a single-term transfer with no omitted intervening segment; same-VTT finite state arithmetic is unchanged':'FP32 descending-i accumulation, running coefficient, empty products 1; G=2 transfers B0 with no intervening segment',
 'state path evaluated twice; per (r,h), transfers require seg_count*(key_dim*value_dim+value_dim/16)*4 B including one replicated a_seg per value tile; feedback approximation adds error':'twice the state work; transfers per (r,h): seg_count*(key_dim*value_dim+value_dim/16)*4 B, including replicated decays; feedback error',
 'planes, mirrors and segment transfers share a grow-only workspace (transfers after the perm pad); normalized operands use two separate grow-only buffers, reused under same-stream order, prep and pass A rewriting every consumed slot per call; out/final_state occupy disjoint views of one fresh byte backing, carved after the prep launch':'planes/mirrors/transfers share one grow-only workspace; q_n/k_n use two separate buffers; same-stream reuse rewrites consumed slots per call; out/final occupy disjoint views of fresh backing allocated after prep launch',
}
for p in B.glob('*.yaml'):
 s=p.read_text()
 for a,b in changes.items():s=s.replace(a,b)
 if p.name=='segmented.yaml':
  s=s.replace('sreg: {meaning: pass-local FP32 state; c indexes pc(seg_lo+c) in A or sc(seg_lo+c) in C','sreg: {meaning: FP32 state pc(seg_lo+c) for pass A',1)
  s=s.replace('sreg: {meaning: pass-local FP32 state; c indexes pc(seg_lo+c) in A or sc(seg_lo+c) in C','sreg: {meaning: FP32 state sc(seg_lo+c) for pass C',1)
  start=s.index('        Math: |',s.index('      - Name: Segment affine transfer and composition'));end=s.index('        Symbol:',start)
  s=s[:start]+'''        Math: |
          Per (r,h), seg_lo(sg)=floor(chunks(r)*sg/seg_count), seg_hi(sg)=seg_lo(sg+1), nseg(sg)=seg_hi(sg)-seg_lo(sg), possibly zero.
          Pass A uses the previous chunk state relation without qpred/attn2/o. Entry pc(seg_lo)=sc(0) for sg=0, else 0; b_seg(sg)=pc(seg_hi), a_seg(sg)=prod_{c=seg_lo..seg_hi-1} ai_c(63), empty product 1. a_seg has one copy per value tile.
          Pass C entry for sg>0: sc(seg_lo(sg))=sum_{i<sg} (prod_{i<j<sg} a_seg(j))*b_seg(i), descending i in FP32. G=seg_count=2 only transfers b_seg(0); G>=3 substitutes scalar decay for the true k/beta feedback transition.
          amax=max_sg a_seg(sg). If amax>theta, sg>0 exits and sg=0 reruns all chunks from sc(0); empty segments force this via a_seg=1.
          Pass C owns all out rows; sg=G-1 writes requested final_state, or sg=0 under fallback. Finite G=2/fallback state arithmetic follows the same-VTT monolithic path.
''' +s[end:]
 p.write_text(s)
