# Prefill solution: upstream FA4 hd256 2CTA

This directory contains only the selected upstream kernel, a thin task-ABI
adapter, its dependency declaration, license, and provenance. It does not
contain a FlashAttention repository snapshot or an AKA-produced kernel.

- `kernel.py`: verbatim upstream `sm100_hd256_2cta_fmha_forward.py`.
- `solution.py`: maps the benchmark ABI to FA4 and explicitly installs the
  extracted class into FA4's compilation path.
- `requirements.txt`: runtime packages expected to provide the remaining FA4
  and CuTeDSL infrastructure.

The workload's FP8 Q/K/V, BF16 output, GQA ratio 16, head dimension 256,
causal masking, variable lengths, and SM103 target match this kernel family.
The initial implementation is intentionally not fully compatible: the task
uses page size 64, while this dedicated TMA paged-KV kernel currently requires
page size 128. It is therefore a valid open-source R0/bring-up point, not a
passing drop-in solution.
