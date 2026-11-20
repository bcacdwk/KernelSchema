# Fixed F1 candidate interface

This is the finite KS evaluator interface, not baseline implementation source.
It is explicitly supplied to the coder and reviewer for original verification,
seed continuation and pure-Schema reconstruction. The fixed Schema, Task ABI,
acceptance and scope remain authoritative.

The complete `python_tree` must contain `solution.py`. Importing it must expose
`Model`, whose instances are callable through the complete forward ABI below
(a `torch.nn.Module` with `forward` supplies this call interface):

```text
query, kv_cache, workspace_buffer, block_tables, seq_lens, max_seq_len,
bmm1_scale, bmm2_scale, window_left, sinks, o_sf_scale, out, q_len_per_req
```

No argument may be removed, reordered, silently replaced or repaired by an
evaluation wrapper. Returned output must alias the original supplied `out`
view/storage. The fixed F1 is B1/Q4/L64/P1, with only page zero valid; all other
ABI constants and numerical/layout restrictions come from the root Contract.

The evaluator also needs the following narrow observation interface for the
Schema's native-cache `when`. The candidate may implement its code itself; no
optimized source is supplied merely to satisfy these names:

* Importing `solution.py` loads `qoder_greenfield_decode_attention.operator`.
  This module exports `_NATIVE_LAUNCHER`, initially `None` in a fresh process.
  The first complete forward establishes the actual launcher used in steady
  forward calls. It must not be replaced during the five warmups/40 samples.
* That actual launcher exposes `thread_id`, `signature` and `source_refs`.
  `source_refs` contains six callable weak references, in this exact order:
  query, kv_cache, workspace_buffer, block_tables, seq_lens, out. Each resolves
  to the original tensor object supplied to every measured call.
* The operator exports `_launch_signature` callable with all thirteen ABI
  arguments in the order above. Its result equals the actual launcher's cached
  signature for these tensor/scalar arguments. Source review must establish
  the signature and object references describe the native path actually used.
* The launcher belongs to the calling thread and current CUDA device; calls
  occur outside stream capture. The evaluator checks these conditions before
  and after each steady call, with a full CUDA synchronize in the timed forward.

The observation fields alone do not prove a native launch or Schema fidelity.
An implementation that fabricates these fields, delegates the computation or
cannot establish the selected CuTe/TMA/TMEM/parity obligations cannot obtain
fidelity from this interface. An absent interface yields failed/unknown checks;
it does not authorize the evaluator to invent a replacement execution path.

Use the original required SM103/CuTe4.6.1/torch2.9.0+cu130/cuda-python13.1.1
stack. The qualified channel permits read-only stage/root files, CUDA devices
and a separate writable compile/output scratch, with no network. It assumes
generated code does not attack the evaluator's Python address space. A new
filename layout, backend or native observation protocol needs an explicitly
supported adapter; no generic native-code sandbox or alternate stack is implied.
