from pathlib import Path
import re
base=Path('schema_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe')
def math_block(s,name,body):
 pattern=r'(- Name: '+re.escape(name)+r'\n        Math: \|\n)(?:          .*\n)+'
 replacement=lambda m:m.group(1)+''.join('          '+line+'\n' for line in body.splitlines())
 s,n=re.subn(pattern,replacement,s,count=1)
 assert n==1,name
 assert len(body)<=1000,(name,len(body))
 return s
physical='          The source allocates 2*route_capacity*intermediate_dim bytes for act, using disjoint payload [0,route_capacity*intermediate_dim/2) and scale [route_capacity*intermediate_dim/2,9*route_capacity*intermediate_dim/16) regions; the remaining bytes are unused.\n'
for name in ('root.yaml','split_k.yaml'):
 p=base/name;s=p.read_text();s=s.replace(physical,'')
 if name=='root.yaml':
  domain='          The input construction uses activation_type 10 with gemm1_alpha = 4 and gemm1_beta = 25; type 3 uses alpha in [0.5,2] and beta in [0,1]. Each parameter vector is constant across experts in that construction. The two activation cases are the union of the supplied dynamic domain and fixed cases, whose activation declarations differ.\n'
  assert domain in s;s=s.replace(domain,'')
  marker='      - Name: Clamp and activation\n'
  assert marker in s
  s=s.replace(marker,'''      - Name: Generated activation parameters
        Math: |
          The supplied input construction uses activation_type 10 with gemm1_alpha = 4 and gemm1_beta = 25; type 3 uses alpha in [0.5,2] and beta in [0,1]. Each parameter vector is constant across experts. Both branches are retained because the dynamic domain specifies SiTU while nine fixed cases construct type 3.
'''+marker)
  s=s.replace("'Source-only: default atol=.01 rtol=.05 max_mismatch_fraction=0; mutated_inputs.output uses .01/.05, max_mismatch_fraction=.001, max_rel_l2=.01; comparison formulas and return/writeback mapping absent'", "'Source-only atol=.01 rtol=.05; default mismatch=0; output-write mismatch=.001 rel_l2=.01'")
  s=s.replace('Gap: Standard answer only inside the declared fixed domain; arguments marked inert are never read','Gap: Comparator formulas and return/writeback mapping absent; source activation declarations differ')
  s=s.replace('16 bytes per row for each of the first two stages', '16 bytes per row for the first two stages')
  s=math_block(s,'Route grouping and padding','''A route (t,s) is valid as in Contract; route_count counts them. counts[e] counts expert e and offs is its exclusive prefix, offs[expert_count] = route_count.
route_position maps valid routes onto [0,route_count), expert e owning [offs[e],offs[e]+counts[e]); order within an expert is an arbitrary permutation.
route_of[t,s] = route_position(t,s), or -1 when invalid; route_token[r], route_slot[r], route_expert[r] invert it.
tile_offsets is the exclusive prefix of ceil(counts[e]/bm), tile_count = tile_offsets[expert_count], tile_expert[j] owns j, tile_first[j] = (j-tile_offsets[tile_expert[j]])*bm.
Tile j starts at offs[tile_expert[j]]+tile_first[j]. Padding can overlap later experts' valid rows or the unwritten tail. Payload boxes zero-fill past route_capacity; scales repeat the first valid row. Padding accumulator lanes are discarded; neither encoding traps.''')
  s=math_block(s,'Clamp, activation and NVFP4 quantize','''For route r of tile j, e = tile_expert[j]; alpha, beta and optional limit are the Contract's per-expert parameters. Apply its clamp to fc1_up/fc1_gate, giving up_c/gate_c.
  type 3: act = (gate_c * sig(gate_c,fp32(alpha*L2E))) * (up_c + beta)
  type 10: act = ((alpha*tan_pre(gate_c,k_gu))*sig(gate_c,L2E))*(beta*tan_pre(up_c,k_uu))
  k_gu = fp32(rcp_rn(alpha)*fp32(2*L2E)); k_uu = fp32(rcp_rn(beta)*fp32(2*L2E))
  sig(x,k) = rcp_approx(1 + exp2_approx_ftz(fp32(x*k))); tan_pre(x,k) = fma32(sig(x,k),2,-1); L2E = fp32(-log2(e))
For each nvfp4_block-channel group b, peak = max abs(act) over its channels:
  act_scale[r,b] = fp8_e4m3fn(min(peak*fp32(1/e2m1_max),e4m3_max)), saturating
  act_code = e2m1_rne(act*rcp_rn(max(fp32(act_scale[r,b]),2^-126))), saturating
act_payload packs two codes per byte, even channel low; code times scale gives the design's act_q.''')
 else:
  s=math_block(s,'Route grouping and padding','''A route (t,s) is valid as in Contract; route_count counts them. counts[e] counts expert e and offs is its exclusive prefix, offs[expert_count] = route_count.
route_position maps valid routes onto [0,route_count), expert e owning [offs[e],offs[e]+counts[e]); order within an expert is an arbitrary permutation.
route_of[t,s] = route_position(t,s), or -1 when invalid; route_token[r], route_slot[r], route_expert[r] invert it.
tile_offsets is the exclusive prefix of ceil(counts[e]/bm); tile_count = tile_offsets[expert_count]. tile_expert[j] owns j, tile_first[j] = (j-tile_offsets[tile_expert[j]])*bm, tile_rows(j) = min(bm,counts[tile_expert[j]]-tile_first[j]).
Tile j starts at offs[tile_expert[j]]+tile_first[j]. Rows past tile_rows(j) are padding and never stored by this tile; rows r >= route_count belong to no valid route. FC1 and FC2 stage padding differently as defined below; neither encoding traps.''')
 p.write_text(s)
