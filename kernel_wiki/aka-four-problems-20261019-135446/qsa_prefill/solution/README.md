# QSA solution: upstream vLLM Triton

`solution.py` is a dependency-free extraction of vLLM's core QSA attention
path at the pinned revision. It contains the split-K paged sparse GQA kernel,
the split merge kernel with fused sigmoid output gate, launch configuration,
and the benchmark `Model` adapter.

Only this directly matching implementation is retained. The older PAI-vLLM
variant and SGLang routing/indexer modules were removed because they do not
match the callable without additional stages or an unfused gate. No AKA or
Atrex-optimized kernel is included.

The exact final `solution.py` passed Agate correctness job `ev_a49db665e9d0`
on L20D/SM103 using all four validation shapes and four seeds per shape
(16/16 cases). The run used the task's production-fitted step-15 index law;
only its submission packaging was minified to satisfy the gateway input-source
size limit. At `atol=0.01, rtol=0.05`, the maximum absolute error was
1.52587890625e-05 and maximum relative-L2 was 0.0026641693115905074.
