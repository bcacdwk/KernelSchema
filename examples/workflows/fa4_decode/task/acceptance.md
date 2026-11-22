# Fixed FA4 F1 acceptance

Shared Contract: the original four-node Schema root, unchanged. The evaluator
supports only `fa4_decode.f1.q4_direct.b1.q4.l64.p1`: B1, Q4, seq_lens=[64],
P1, valid block_tables=[0], N1/S1. This is a Schema representative, not one of
the 30 scored original shapes. All other shared-Contract inputs are uncovered.

Use the pinned original `_make_inputs` with all ten kwargs in f1-parameters.json,
seed 20261120 once immediately before input generation. Keep all 13 ABI arguments,
Q4 BF16 randn scaling (query 1.22, KV 0.96) before FP8 conversion, and the
413138944-byte zeroed workspace. out is originally torch.empty, then poisoned
in place outside measured time. Reference uses a separate output and workspace.

Reuse the pinned historical harness functions `checks`, `unchanged`, `poison`;
execute only those function AST nodes, never historical main. For return and the
original supplied out, require BF16 shape [4,16,256], FP32 finite actual/reference
at every element and abs(actual-reference) <= 0.06 + 0.04*abs(reference).
Bind original out object/storage/offset/shape/stride/dtype/device before any
candidate call. Return must alias that original view, which must remain intact
and completely overwritten. No output repair is permitted. query, kv_cache,
block_tables and seq_lens retain their original identity/view and exact bytes.
Only out and workspace_buffer may change. Check all of this after the cold call,
every warmup and every measured call, outside timing.

Cold import, construction and first complete forward+sync are separate facts.
Steady timing requires the pinned native launcher, same six tensor objects,
signature/thread/current device and no stream capture before and after each
call. Native launch probe 1e-2 is not numerical acceptance. Time complete
Model.forward plus CUDA completion using host monotonic nanoseconds. A lower
kernel-sum time is not a full-forward measurement.

Numerical, performance, runtime coverage and Schema fidelity are separate.
N1 avoids multi-round PageIndex reuse and global empty splits; empty parity,
TMA padding/page tail, TMEM and output lifetime remain fidelity obligations.
No static acyclic graph, CPU test or unqualified transport grants fidelity.
