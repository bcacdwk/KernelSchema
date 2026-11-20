# Fixed reference materials

`triton/02-fused-softmax.py` is the **unmodified** official Triton tutorial at
tag v3.5.1, resolved through GitHub's API to commit
`0add68262ab0a2e33b84524346cb27cbb2787356`. `triton/LICENSE` preserves the
upstream MIT permission/copyright notice. Actual retrieval URLs, byte lengths,
SHA-256 and Git blob identities are recorded in sources.json; fetched
2026-11-20 UTC. No moving main URL is used as frozen identity.

The discovery page is
https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html.
The fixed source page is
https://github.com/triton-lang/triton/blob/0add68262ab0a2e33b84524346cb27cbb2787356/python/tutorials/02-fused-softmax.py.
To retrieve identical bytes, use GitHub's contents API at the recorded commit
and base64-decode its `content`; verify SHA-256. This route was actually used
because the git/raw transport was unavailable during material preparation.

The tutorial explains max-subtract, approximate FP32 exp, row reduction and
fusion. **Its persistent multi-row loop, output allocation, hardware queries,
occupancy selection, benchmark domain and timing are not the R0 assignment.**
R0 instead fixes grid=(M,), one program per row and the supplied out ABI. The
tutorial is readable reference material, never the Task baseline or golden.
Do not import/run it during core resource loading: the original module queries
the active GPU and contains executable examples. A reference's instructions
cannot override the fixed Task, current Schema, permissions or budget.

`evaluator/golden.py` is newly authored CPU FP64 code, separately frozen at
KernelSchema commit `2efc92479db497e0ecf681e276ecddc72cad0de6` and referenced by
R0. Its public fixed source is
https://github.com/bcacdwk/KernelSchema/blob/2efc92479db497e0ecf681e276ecddc72cad0de6/examples/workflows/row_softmax/evaluator/golden.py.
Its license follows the KS project policy, which still needs explicit release
resolution; public readability does not itself grant redistribution rights.

The existing Template and DOC0–6 are reused from the checkout, with their actual
digests and development baseline in the inventory. They are not copied into a
second parser or rewritten method. The catalog note covers adjacent mechanisms
without claiming an existing softmax Schema. Per-role delivery is declared in
task/materials.json; implementation must enforce it. Design never receives the
R0 answer or private evaluator code through this reference collection.

These inventories are ordinary material input JSON, not canonical workflow
manifests or runnable requests. K2 owns Ref/JCS/publication and selected closure.
See missing.md for unqualified runtime, private executor and release gaps.
