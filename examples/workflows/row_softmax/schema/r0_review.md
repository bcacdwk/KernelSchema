# R0 material review fixture

This hand-authored fixture is for fixed-R0 engineering and later finite fidelity
tests. It is not a universal checker, model review, GPU result or new R identity.
Its raw YAML `Version: 1` and the future Schema manifest's `design_revision: 0`
are different counters. K2 must bind exact raw/dispatch/Task bytes at publication.

| R0 obligation | Raw location | Material-level rationale / remaining evidence |
| --- | --- | --- |
| Preserve ABI, all four shapes, original out and x | CONTRACT.Algorithm_requirement | Complete public acceptance independently checks objects, storage/view, original regions, bytes and canaries; runtime still pending |
| Stable FP32 max-subtract/exp/sum/div | DESIGN.Math_reshape.Rewrite | All operations and permitted approximate exp are explicit; candidate numerical and precision evidence still required |
| One program/CTA per complete row | DESIGN.Work_schedule.Step.MAIN | row tile 1, worker_count=m, preassigned worker_id, grid=(m,), one launch; no unselected fallback |
| Program-local x and exponentials, no explicit global intermediates or cross-CTA communication | DESIGN.Math_reshape; DESIGN.Tile_pipeline | Consumers are inside one synchronous row computation; this does not prove no compiler spill |
| One owned Triton kernel, no external softmax | REALIZATION.Require/Forbid | Host wrapping and legal shape dispatch remain free; the baseline intentionally does not satisfy this obligation |
| Write all N outputs, x read-only | Contract and MAIN.compute:y | Both legal N are powers of two; no arbitrary-N claim |
| num_warps 4/8 is an implementation choice | REALIZATION.Require | Does not change path or row ownership; actual choice must be captured |

The input partitions are the four shapes, not the 36 test IDs. Each maps to
root with no variant. The explicit dispatch is part of the selected source
closure. Constructor output precomputation, test-order dependence or output
caching violates the Task, independently of R0 fidelity.

Finite reviewer method work belongs to K4. Its output must locate the **current**
Schema obligations and exact C, distinguish coverage from fidelity, and use
pass/fail/unknown/not_run with reasons. A synthetic multi-row-program C can fail
R0 yet pass a lawful multi-row Rd if all Rd obligations have evidence; new
uninterpretable obligations remain unknown. Removing max may pass numerical
checks on this domain and still fail R0. None of these examples is GPU evidence.

The device/architecture/runtime fields remain explicitly unselected. Full
Template structure can be checked, but a complete DOC2 hardware specification,
version-matched official device/stack material and execution qualification must
be bound before an executable Task/Schema is admitted. No missing environmental
fact is inferred from a syntactically valid field or a tutorial version.
