# Native vector add

Write a KernelSchema for the native Atrex-Bench task below, targeting Triton
on NVIDIA L20D. The authoritative task files are `reference.py`, `input.py`
and `shapes.json` from AKA's `native-atrex-bench-vector-add` example.

The public interface is `Model(alpha=1.0).forward(x, residual)`. It returns
`x + alpha * residual`, preserving the PyTorch reference's operation ordering
and rounding. `x` and `residual` are equal-length, contiguous, one-dimensional
CUDA bfloat16 tensors; the output has the same shape and dtype. `alpha` is a
Python float stored by the module constructor.

| Native shape ID | Number of elements | alpha |
| --- | ---: | ---: |
| 0 | 1024 | 1.0 |
| 1 | 65536 | 0.5 |
| 2 | 1048576 | 2.0 |

Keep integer-string shape IDs and the original input generator. Express a
simple implementable design, including indexing, launch, tail handling,
memory accesses and the `Model` wrapper. Record unmeasured choices as such.
Writing does not establish correctness, latency or an optimization benefit;
AKA's normal evaluator measures the resulting implementation. For optimize,
use the supplied native results, including failures or unavailable timings,
without relabeling them as a successful candidate.
