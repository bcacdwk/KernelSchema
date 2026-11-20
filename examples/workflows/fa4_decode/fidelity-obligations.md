# FA4 F1 review boundary

Read complete original four-node raw Schema and dispatch with the existing
Template resolver. Selected obligations are root CONTRACT plus q4_direct local
DESIGN and REALIZATION. Scope covers only B1/Q4/L64/P1/N1/S1 at seed 20261120.
Root requires Q4/B15-or-16/long KV/S9 and is not a fallback. task_uncovered is
`fa4_decode.shared_contract.except_f1`.

Review the actual full Source with per-file, exact line evidence against the
selected math, FP8 hi/lo, FP32 denominator, ownership, pipeline, Require/Forbid
and lifetime obligations. N1 avoids the N>=5 PageIndex reuse and global empty
split cases. It does not prove empty parity initialization/readback, TMA padded
page/page-tail behavior, TMEM fences/return/deallocation or output staging life.

CPU source/AST/Template checks provide no fidelity pass. The existing row-softmax
AST reviewer and kernel.py-only line locator do not support this Source. Until a
complete multi-file reviewer is connected, fidelity stays unknown. Original
verification, explicit full-Source seed continuation and pure-Schema rebuild
are distinct requests. A rebuild coder must not receive optimized baseline,
historical campaigns, annotated source or old sessions as role materials.
