// TU: head_dim 128 kernel instantiations (prefill + decode, KV tile 64).
#include "fmha.cuh"

DEFINE_FMHA_LAUNCHERS(128, 64, 128)
