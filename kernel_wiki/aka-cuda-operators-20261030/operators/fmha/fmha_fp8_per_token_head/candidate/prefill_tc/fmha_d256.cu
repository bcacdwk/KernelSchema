// TU: head_dim 256 kernel instantiations (prefill + decode, KV tile 32).
#include "fmha.cuh"

DEFINE_FMHA_LAUNCHERS(256, 32, 256)
