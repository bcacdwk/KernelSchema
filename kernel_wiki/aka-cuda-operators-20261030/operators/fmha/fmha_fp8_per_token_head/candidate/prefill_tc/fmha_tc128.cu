// TU: tcgen05 prefill instantiation for head_dim 128 (KV tile 64).
//
// PIPE=true at D=128 measured 0.95x in the peer's own v2.3b note ("TMEM contention /
// register pressure from batched corr"), which is why it shipped false while D=64 keeps
// PIPE=true at 1.14x. f16 QK has since removed three of the four instructions per K
// pair from the per-tile conversion - 96 of the roughly 840 instructions a thread issues
// per tile at D=128 - so the register pressure that argument rests on is lower than it
// was. PF_PIPE128 re-enables the deep pipeline to retest it under the new instruction
// mix. Off by default; own extension name, since it is a compile-time property of this
// TU and two arms of one A/B must not share a cached build.
#include "fmha_tc.cuh"

// PF_BNT128 doubles the KV tile. The peer measured BNT=32 at 0.911x on case 7 and
// concluded "doubled per-tile overhead beat the 4-CTA/16-warp pool", which points the
// other way: if halving the tile loses to per-tile overhead, doubling it may win. The
// budgets do fit at D=128 - TcSmem::bytes() is 166568 of the 232448 allowed, and TMEM
// NEED = 2*128 + 128 = 384 of 512 columns - and it compiles, but MEASURED 1.4074x
// SLOWER on case 0 (job dv_29fdfd3540df: 1572.9 -> 2213.7us, both arms reproducible to
// 0.14%). One resident CTA per SM instead of two, all 512 TMEM columns instead of 256,
// and 192 per-thread score registers instead of 96 cost far more than halving the
// per-tile fences, barriers and TMEM round trips saves. With the peer's BNT=32 at
// 0.911x, the tile is bracketed from both sides and 64 is a sharp optimum. Kept default
// off with the measurement recorded; see results/prefill-pipe128-20261023.md.
#ifdef PF_BNT128
DEFINE_TC_LAUNCHER(128, 128, false)
#elif defined(PF_PIPE128)
DEFINE_TC_LAUNCHER(128, 64, true)
#else
DEFINE_TC_LAUNCHER(128, 64, false)
#endif
