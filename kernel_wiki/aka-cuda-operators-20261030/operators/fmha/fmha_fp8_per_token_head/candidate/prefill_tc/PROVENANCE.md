# Provenance — `candidate/prefill_tc/` (peer arm b, join 20261022)

These ten files are the **verbatim** prefill implementation of peer arm `b` of the
manual join `20261022` for `l20d-15 / fmha_fp8_per_token_head`, copied from the
read-only exchange drop `.join-exchange/20261022/b/candidate/`. Nothing here was
written by this session; it is adopted as a mechanism source and as a second
implementation to measure against, and every claim about it in this workspace is
backed by a measurement made here, not by the peer's numbers.

Peer provenance as recorded in `.join-exchange/20261022/b/implementation.json`:

- `source_commit` `5cf8370471f077c0f32520dfc7164834da742453`, tag
  `v4.12-full-suite-pass`, `candidate_tree`
  `1a9e9d1aa60061425a129da1c15b3c851d56275d`.
- Official fixed-suite receipt `results/20261022T025153Z-458998-l20d.json`, job
  `dv_38a3cfc03d08`, `ok: true`, gateway `succeeded`, 10/10 valid at two seeds,
  geomean **187.8657 us**, on NVIDIA L20D, torch 2.9.0+cu130, CUDA 13.0,
  `checker_sha256 9b6773...e85c7c` (the same frozen checker this workspace uses).

Per-case comparison against this workspace's promoted candidate (arm a, job
`dv_b68636085c6f`, geomean 177.9014 us) — **cross-job, so not a same-job speedup
claim**, which is exactly why the A/B in `results/prefill-tc-merge-20261022.md`
exists:

| case | a (this candidate) | b (peer) | b/a | shape |
|---|---|---|---|---|
| 0 | 2751.0 | **1643.0** | 0.60 | prefill, q=[1023,2049], d=128 |
| 1 | 216.8 | **134.6** | 0.62 | prefill, q=kv=2048, d=128 |
| 2 | **142.8** | 249.5 | 1.75 | decode |
| 3 | **110.4** | 162.5 | 1.47 | decode, d=256 |
| 4 | **121.4** | 318.4 | 2.62 | decode, B=32 |
| 5 | **26.9** | 36.3 | 1.35 | decode |
| 6 | **38.9** | 50.0 | 1.29 | decode |
| 7 | 645.8 | **385.5** | 0.60 | prefill, q=kv=8192, d=64 |
| 8 | **37.7** | 39.4 | 1.05 | decode |
| 9 | 1091.8 | **695.4** | 0.64 | prefill, q=1025, d=128 |

The two implementations are complementary: the peer is ~1.6x faster on all four
prefill cases and 1.3-2.6x slower on all six decode cases. Arm a won the join on the
whole-suite geomean, so the unified candidate is arm a's, and this directory exists
to supply only the prefill half.

## What the peer's prefill does that this candidate's does not

From `fmha_host.cu` and `fmha_tc.cuh` (its own comments, versioned v2.3b-v4.8.4):

- **tcgen05 (5th-gen tensor core) prefill for every head_dim** — `fa_prefill_tc<D,
  BNT, PIPE>` with M=128 query rows per CTA, 128 threads, TMEM holding S0/S1 and the
  f32 O accumulator, `tcgen05.mma` issued by an elected thread with mbarrier
  completion, `tcgen05.ld`/`tcgen05.st` for the softmax round trip through TMEM.
  `PIPE=true` only for D=64 (measured 1.14x there, 0.95x at D=128, so D=128/256 keep
  the sequential v2 loop).
- **Two work-distribution modes chosen on the host**: `fold` (v4.5 mirror pairing of
  two q tiles per CTA on the causal diagonal band, when the grid is between 1x and 2x
  the SM slot count) and `persist` (v4.8 dynamic work pull through an atomic counter
  with `grid = min(grid_orig, slots)` when the grid is at least 1.25x the slots). Its
  own note: "c0 2.8x / c7 2.6x waves won 15% / 35%", and a counter-example where
  persisting a 1.08-wave grid lost 4%.
- One translation unit per head_dim so ninja parallelises the builds and no single
  ptxas run is oversized.

This candidate has its own tcgen05 kernel (`candidate/fmha_tc.cu`, cfgs 16/17) which
was measured 1.35-1.76x **slower** than its mma.sync prefill and never promoted; the
peer's is the version that pays, so the interesting question is which of the three
mechanisms above carries the win.

## Integration boundary

Built as a **separate torch extension** (`fmha_pf_tc_<cap>`, see `solution.py`) so no
symbol, macro, `Params` layout or compile flag is shared with `fmha.cu`: the peer's
`--use_fast_math` is load-bearing for its numerics (its `exp2_precise` exists because
fast-math `ex2.approx` noise was amplified by output cancellation) and this
candidate does not use it. `FMHA_PF_TC=1` routes a shape to this module only when
`max_seqlen_q * g > 64`, which is the peer host's own prefill condition
(`decode = max_seqlen_q * P.g <= BM`, `BM = 64`), so its decode path — the half that
is 1.3-2.6x slower here — is never reachable. Its `fa_combine` and `fa_decode` are
compiled but dead. Files were unmodified when copied, and the hashes below are the peer's recorded ones,
re-verified byte for byte after copying (its `solution.py` was deliberately not copied;
this workspace keeps its own).

**One local modification since**, diagnostic only: `fmha.cuh`'s `e4m3x2_to_bf16x2`
gained an `#ifdef PF_NOCONV` branch replacing the four-instruction E4M3->F16->F32->BF16
chain with one integer multiply, so a timing A/B can bound what the conversion stage
costs this prefill path. That build produces wrong numbers by construction, is never
the default, and gets its own extension name (`fmha_pf_tc_<cap>_nc`); the hash below
describes the file as copied plus that guarded branch. `tc_convert` calls the helper
`2 * D / TPR` times per thread per KV tile — at D=128, BNT=64, 128 threads that is 32 K
pairs plus 32 V pairs, 256 calls and 1024 instructions per thread per tile — which is
why the bound is worth measuring before touching anything inside the kernel.

```
fmha.cuh        194a468563941ab6414b08710ace7b33c2658497c723a917a00d8cfd32f35a7f
fmha_d128.cu    4ed5f9a05db355d626a1001c0ff0b441ba8ab02e14bb3370f76a2324b0a16ef2
fmha_d256.cu    dc6feed1f15e7c0a88a1dd14331dbce6353d1b502773cbffc23775815b4e6251
fmha_d64.cu     e97471ff077c096190f270eafa0e460a01568a25421e26e9de7f88e9c66179af
fmha_host.cu    065aa0306f5f542ed23b3dfa34d63ad2405936d226d627cb3a97311144d5baf1
fmha_tc.cuh     f2d76de5d8fa8ea634213871159841ae9dee395620d48ffe4ef6f4a853ef7786
fmha_tc128.cu   91be50cff648a6207a03e291008a7243d60e4b3faf95158c0ea9c7790c36fd10
fmha_tc256.cu   3dadffb0706709a4acd4aa2f21bdf433100b1cd496980a106196fc34f69941e2
fmha_tc64.cu    4dd519c771604bd4d0446c328052f1dc1e495e96be791e1d5a4c78668a550452
fmha_tcr128.cu  ba121da557a641b9fa345865f33172d4570bcc52f48e2a6a02b947d7257c1070
```
