# Complete forward timing recipe

For each of `(1,128)`, `(1,1024)`, `(32,128)`, `(32,1024)`, generate the seed 17
CPU FP32 input exactly as in inputs.json, then copy it to the fixed CUDA device.
Use independently preallocated out for candidate A and baseline B. Both calls
are exactly `model.forward(x=x, out=out)` in inference mode, eager, on the same
single call stream. Candidate and baseline see identical input bytes.

Record model construction, first invocation/compilation and **20 warmup calls
per model per shape** separately from steady-state measurements. Neither
construction nor compilation nor warmup contributes to the steady-state score.
Record the preparation sequence and any cold-start/probe work. Do not claim
these excluded costs vanished.

Collect **10 pairs of batches per shape**. Pair 0 executes AB, pair 1 BA,
alternating through pair 9 BA. Each individual batch does:

1. Synchronize the selected CUDA device.
2. Capture `start = time.perf_counter_ns()`.
3. Execute **100 complete forward calls**, with the same preallocated x/out.
4. Synchronize the device.
5. Capture `end = time.perf_counter_ns()` and save elapsed nanoseconds and
   `per_forward_ns = (end-start)/100`.

Preserve all 20 ordered batch observations per shape (10 for each model),
including pair index, AB/BA order and within-pair position, start/end/elapsed
nanoseconds and call count. Report each model's median, IQR, minimum, maximum
and actual sample count. The recipe uses the usual median (mean of middle two
when even); quartiles use linear interpolation at rank `(n-1)*p` for p=0.25
and 0.75. This explicit reporting convention fills V's unspecified quartile
estimator; it changes neither the samples nor the median-based goal.

The Task goal uses each model's **median** complete-forward time per shape:
`exp(mean(log(t_baseline/t_candidate))) >= 1`, equally weighting the four
shapes. Only a complete positive sample set with fixed code, environment and
timing identities may be met/unmet. Missing or unusable samples mean unknown.
Preserve noise/outliers and report observed timings; a ratio slightly over one
is not proof of a reproducible speedup. Native ABBA/promotion outcomes remain
separate from this Task score.

This measures amortized complete forward time including Python submission and
batch-end synchronization. Include every allocation, copy, kernel and required
synchronization performed inside forward. Keep input generation, poison,
comparison and caller-owned allocation outside both models' measured regions.
Do not substitute isolated-call latency, CUDA-event-only time, NCU kernel sums,
or a native benchmark's timing convention.

Execution requires a qualified fixed device/stack profile. The B01 files define
this recipe but contain no timing runtime, samples or measured performance.
