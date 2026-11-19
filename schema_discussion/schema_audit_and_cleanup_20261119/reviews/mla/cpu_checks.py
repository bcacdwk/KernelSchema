"""Finite source-derived checks; no CUDA import, kernel execution or acceptance claim."""
from pathlib import Path
from fractions import Fraction
import hashlib
import json
import math
import struct
import subprocess
import yaml

BASE = Path('kernel_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8')
OUT = Path(__file__).parent
COMMIT = '24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2'

def f32(x):
    return struct.unpack('f', struct.pack('f', float(x)))[0]

def fma32(a, b, c):
    # Operands used here are exact FP32 dyadics of modest magnitude.
    return f32(Fraction(a) * Fraction(b) + Fraction(c))

def e4m3_positive(code):
    exponent, mantissa = code >> 3, code & 7
    return (mantissa * 2**-9 if exponent == 0
            else (1 + mantissa / 8) * 2**(exponent - 7))

def e4m3_rn(x):
    # Includes the finite terminal exponent, excludes the NaN code 0x7f.
    code = min(range(127), key=lambda k: (abs(e4m3_positive(k) - x), k & 1))
    return e4m3_positive(code)

source_identity = {}
for path in sorted(BASE.rglob('*')):
    if path.is_file():
        committed = subprocess.check_output(['git', 'show', f'{COMMIT}:{path}'])
        assert committed == path.read_bytes(), path
        source_identity[str(path.relative_to(BASE))] = hashlib.sha256(committed).hexdigest()

selection = json.loads((BASE / 'performance/SELECTION.json').read_text())
assert source_identity['candidate/solution.py'] == selection['candidate_files']['solution.py']
for name, digest in selection['contract_files'].items():
    assert source_identity[f'problem/{name}'] == digest

# Q/K/V regions, Meta arrays and reserved space for MODE0 KVST4.
meta_fields = [
    ('q_loaded', 16, 8), ('kv_loaded', 16, 32), ('qk_done', 16, 24),
    ('s_free', 16, 24), ('p_ready', 16, 24), ('pv_done', 16, 24),
    ('kv_free', 16, 32), ('o_rescaled', 16, 8), ('scale_ready', 16, 16),
    ('item_ready', 16, 16), ('pop_req', 16, 8), ('o_done', 16, 8),
    ('acc_scale', 4, 1024), ('fin_sum', 4, 512), ('fin_max', 4, 512),
    ('tmem_base', 4, 4), ('geo_valid', 4, 4), ('mi', 4, 80),
]
offset, offsets = 0, {}
for name, alignment, size in meta_fields:
    offset = (offset + alignment - 1) // alignment * alignment
    offsets[name] = offset
    offset += size
assert (offset + 15) // 16 * 16 == 2416
assert offsets['acc_scale'] == 280 and offsets['mi'] == 2336
registered_smem = 24576 + 98304 + 65536 + 1024 + 512 + 512 + 16640 + 80
assert registered_smem == 207184 and 209920 - registered_smem == 2736
assert 24576 + 98304 == 122880
assert 122880 + 65536 == 188416
assert 188416 + 4096 == 192512 and 192512 + 17408 == 209920
assert 3 * 128 * 128 * 4 + 128 * 128 * 4 == 262144
assert 128 * (64 - 32) * 4 == 128 * 128  # Packed P subregion.
assert 16384 + 4096 + 256 <= 128 * 248   # SOFTMAX registered words.
assert 16384 <= 128 * 160                # CORRECTION registered words.
schema_path = Path('schema_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/root.yaml')
schema = yaml.safe_load(schema_path.read_text())
storage = schema['DESIGN']['Tile_pipeline']['Step']['MAIN']['Storage']
assert sum(s['total_size'] for s in storage.values() if s['location'] == 'smem') == registered_smem
assert sum(s['total_size'] for s in storage.values() if s['location'] == 'tmem') == 262144
assert sum(s['total_size'] for s in storage.values() if s['location'] == 'register') == 148480
assert schema['CONTRACT']['Algorithm_requirement']['Input']['sage_attn_sfs']['shape'] == []
assert schema['CONTRACT']['Algorithm_requirement']['Input']['num_elts_per_sage_attn_blk']['shape'] == [{'element': 4}]

sw128 = [(row >> 3) * 1024 + (row & 7) * 128
         + ((((col >> 4) ^ (row & 7)) << 4) | (col & 15))
         for row in range(128) for col in range(128)]
assert sorted(sw128) == list(range(16384))
assert all(row * 130 + col < 32 * 130 for row in range(32) for col in range(128))

# Source geometry under an actual positive-attribute root predicate.
lengths, heads, nsm, l2 = [128, 257, 1024], 24, 148, 132644864
assert heads * sum((q + 127) // 128 for q in lengths) * 2 > nsm * 3
assert min(heads * len(lengths), nsm) * max(lengths) * 320 * 4 <= l2 * 3
geometry_checks = []
for causal in (False, True):
    items, first = [], 0
    for h in range(heads):
        first = 0
        for b, length in enumerate(lengths):
            for qt in reversed(range((length + 127) // 128)):
                qstart, qend = qt * 128, min((qt + 1) * 128, length)
                vend = qend if causal else length
                items.append((max(1, (vend + 127) // 128), h, b, first + qstart, qend - qstart))
            first += length
    items.sort(key=lambda item: -item[0])
    destinations = [(q + row, h) for _, h, _, q, count in items for row in range(count)]
    assert len(destinations) == len(set(destinations)) == sum(lengths) * heads
    assert set(destinations) == {(q, h) for q in range(sum(lengths)) for h in range(heads)}
    assert all(0 <= (q * heads + h) * 128 < sum(lengths) * heads * 128
               for q, h in destinations)
    geometry_checks.append({'causal': causal, 'items': len(items), 'written_rows': len(destinations)})

# Each real item requests one next item; bootstrap items plus refills cover
# all items exactly once and the final grid-width requests are sentinels.
queue_checks = []
for count in (223, 288, 3840):
    seen = list(range(nsm))
    claims = list(range(count))
    seen.extend(nsm + claim for claim in claims if nsm + claim < count)
    assert sorted(seen) == list(range(count))
    assert sum(nsm + claim >= count for claim in claims) == nsm
    assert claims[-1] == count - 1
    queue_checks.append({'items': count, 'claims': len(claims), 'sentinels': nsm})

# Tiny row_final precedes p_ready and the issuer's per-unit tail, so a
# non-monotone [1,5] key-tile sequence cannot overwrite either source early.
edges = [
    ('tiny_read_done', 'p_ready'), ('p_ready', 'tail_pv_issue'),
    ('tail_pv_issue', 'next_unit_qk0'), ('next_unit_qk0', 'next_unit_qk2'),
    ('tail_pv_issue', 'tail_pv_done'), ('tail_pv_done', 'kv_release'),
    ('kv_release', 'v_slot_refill'),
]
def reaches(start, end):
    pending, seen = [start], set()
    while pending:
        current = pending.pop()
        if current == end:
            return True
        if current not in seen:
            seen.add(current)
            pending.extend(b for a, b in edges if a == current)
    return False
assert reaches('tiny_read_done', 'next_unit_qk2')
assert reaches('tiny_read_done', 'v_slot_refill')

# Counterexample to an unconditional relative <=2^-4 E4M3 rounding claim.
small = 2**-11
assert struct.unpack('e', struct.pack('e', small))[0] == small
assert e4m3_rn(small) == 0 and abs(e4m3_rn(small) - small) / small == 1
# Counterexample to replacing the source offset FMA by multiply then add.
qk_scale = f32(0.125 / math.sqrt(192))
coefficient = f32(qk_scale * f32(math.log2(math.e)))
offset_example = None
for integer in range(1, 200000):
    safe = f32(integer / 16)
    fused = fma32(-safe, coefficient, 8.0)
    separate = f32(f32(-safe * coefficient) + 8.0)
    if fused != separate:
        offset_example = dict(safe=safe, c=coefficient, fused=fused, separate=separate)
        break
assert offset_example

fixed_cases = json.loads((BASE / 'problem/shape_cases.json').read_text())
fixed_selectors = []
for key, case in fixed_cases.items():
    kw = case['input_kwargs']
    items = kw['heads'] * sum((v + 127) // 128 for v in kw['query_lengths'])
    live = min(kw['kv_heads'] * len(kw['kv_lengths']), nsm) * max(kw['kv_lengths']) * 320
    assert min(kw['query_lengths']) >= 257 and not kw.get('sink', False)
    assert live * 4 <= l2 * 3
    fixed_selectors.append({'case': key, 'items': items, 'queued_at_assumed_148_sms': items * 2 > nsm * 3,
                            'live_kv_bytes_at_assumed_148_sms': live})
assert sum(v['queued_at_assumed_148_sms'] for v in fixed_selectors) == 9

result = {
    'status': 'passed', 'gpu_used': False,
    'source_commit': COMMIT, 'source_files_verified': len(source_identity),
    'source_identity': source_identity,
    'schema_sha256': hashlib.sha256(schema_path.read_bytes()).hexdigest(),
    'selection_hashes_match': True,
    'storage': {'meta4_size': 2416, 'meta4_offsets': offsets, 'registered_smem': registered_smem,
                'smem_allocation': 209920, 'unregistered_control_and_padding': 2736,
                'tmem_capacity_bytes': 262144, 'sw128_unique_bytes': len(sw128)},
    'root_geometry': geometry_checks, 'queue': queue_checks,
    'tiny_lifetime': {'key_tile_counts': [1, 5], 'edges': edges,
                      'score_overwrite_after_read': True, 'value_refill_after_read': True},
    'rounding': {'fp16_input': small, 'e4m3_result': e4m3_rn(small),
                 'relative_rounding_error': 1, 'offset_fma_example': offset_example},
    'fixed_selectors': fixed_selectors,
    'limits': ['Finite CPU arithmetic and source-level ordering only.',
               'No GPU execution, TMA descriptor execution, approximate-exp bound or comparator reconstruction.',
               'Register counts are declared array capacities, not compiled occupancy or spill measurements.'],
}
(OUT / 'cpu_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'source_identity'}, ensure_ascii=False, indent=2))
