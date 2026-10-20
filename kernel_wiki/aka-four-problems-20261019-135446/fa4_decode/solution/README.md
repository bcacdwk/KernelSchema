# Decode solution: upstream FlashInfer CuTeDSL

This directory contains the upstream paged GQA decode kernel, its shared source
for provenance, a task adapter, dependency declaration, license, and provenance.
It no longer carries FlashInfer Python modules, XQA CUDA sources, TRT-LLM
launchers, or a FlashAttention repository snapshot.

- `gqa_decode.py`: shared decode kernel and math primitives.
- `gqa_decode_paged.py`: paged-KV decode kernel; its one import is redirected
  to the adjacent extracted `gqa_decode.py`.
- `solution.py`: a directly submit-ready, 103,589-byte Agate candidate. It
  inlines `GroupedQueryAttentionDecodePaged`, imports the shared decode base
  and planning/runtime wrapper from pinned FlashInfer v0.6.18, and exposes the
  benchmark's exact `class Model` ABI. It has no sibling-file imports.

This is the non-AKA FlashInfer v0.6.18 seed used by the Decode campaign. The
exact final `solution.py` passed Agate candidate check `cp_c55675921d5a` and
correctness job `ev_ae66c40d4d09` on L20D/SM103: all 30 validation shapes
passed at `atol=0.01, rtol=0.05` (max absolute error 0.03125, max relative-L2
0.027880733684855455). It supports the task's native FP8 query/KV inputs, head
dimension 256, GQA, page size 64, and MTP query rows.
