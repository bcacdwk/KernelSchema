// TU: tcgen05 prefill instantiation for head_dim 256 (KV tile 64).
#include "fmha_tc.cuh"

DEFINE_TC_LAUNCHER(256, 64, false)
