# QSA BF16 FIT v3

Selected version: FIT v3, accepted commit fc30f74. The four files in
qwen38_core_qsa_cutedsl/ are byte-identical to the frozen FIT v3 source.
solution.py is the exact thin Model wrapper tested on Agate L20D.

Interface: Model.forward(q, k_cache, v_cache, logical_indices, block_table,
token_to_req, use_prefill_config, output_gate, out=None). The supplied out
is written in place and returned. Q/K/V are BF16.

Add this aka_solution directory to PYTHONPATH when importing solution.py
from an external runner. Multi-file submissions must include this whole directory.

Validation: dv_206e1c967c99, 2026-10-19, NVIDIA L20D / SM103.
4/4 scored shapes and 16/16 cases passed the original task tolerances
(atol=3.2e-5, rtol=0.02), including mutated out and unchanged inputs.
NCU kernel sum: 424.101 us versus Production 1811.925 us; geomean 4.2724x.
The two required kernels are QSAPreprocessUnion and QSAForward.

This frozen version is validated in eager mode. Its host tbase / device
ticket lifetime requires adaptation before CUDA Graph replay integration.
The original task's v7 AKA has been replaced by FIT v3 in this export.

Raw correctness, NCU CSV, per-kernel times and source hashes are under
../../evidence/qsa_prefill/.
