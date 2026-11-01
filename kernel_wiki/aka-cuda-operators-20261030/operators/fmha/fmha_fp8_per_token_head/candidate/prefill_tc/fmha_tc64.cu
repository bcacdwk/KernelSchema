// TU: tcgen05 prefill instantiation for head_dim 64 (KV tile 64, PIPE).
// v4.11 measured seq-loop D64 at 0.892x (c7 0.4246 vs 0.3788): PIPE stays.
#include "fmha_tc.cuh"

DEFINE_TC_LAUNCHER(64, 64, true)
