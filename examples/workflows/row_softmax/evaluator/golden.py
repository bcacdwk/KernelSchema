"""CPU binary64 oracle for row_softmax_fp32_v1, independent of the baseline.

The trusted evaluator must copy x's FP32 bytes *before* calling the candidate,
then pass those saved values as CPU rows (for example, snapshot.tolist()).
Python float is IEEE binary64 on the supported Linux x86_64 platform. This
function never reads the candidate's post-call x, imports torch, or runs CUDA.
The return value is nested Python binary64 floats, not an FP32 reference Model.
"""

import math
from collections.abc import Sequence


def softmax_fp64(x_before: Sequence[Sequence[float]]) -> list[list[float]]:
    """Compute stable row softmax from an already captured FP32 input snapshot."""
    rows = [[float(value) for value in row] for row in x_before]
    if len(rows) not in (1, 32):
        raise ValueError("row_softmax_fp32_v1 requires M in {1, 32}")
    width = len(rows[0])
    if width not in (128, 1024) or any(len(row) != width for row in rows):
        raise ValueError("row_softmax_fp32_v1 requires rectangular N in {128, 1024}")
    if any(not math.isfinite(value) or not -20.0 <= value <= 20.0
           for row in rows for value in row):
        raise ValueError("x_before must contain finite FP32 values in [-20, 20]")
    result = []
    for row in rows:
        maximum = max(row)
        numerators = [math.exp(value - maximum) for value in row]
        denominator = math.fsum(numerators)
        result.append([value / denominator for value in numerators])
    return result
