// TU: head_dim 64 kernel instantiations (prefill + decode, KV tile 64).
#include "fmha.cuh"

DEFINE_FMHA_LAUNCHERS(64, 64, 128)
