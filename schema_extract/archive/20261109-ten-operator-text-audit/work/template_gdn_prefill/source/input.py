"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Exact upstream multidist_randu/gen_qkv excerpts; generator device set by caller.
import functools
import importlib.util
from pathlib import Path
import math
import torch

def multidist_randu(num_dists, dim, mean_mean=0.0, mean_std=1.0, lower=-1.0, upper=1.0):
    means = torch.distributions.Normal(mean_mean, mean_std).sample((num_dists,))
    data = torch.distributions.Uniform(means + lower, means + upper).sample((dim,))
    return data.T.contiguous()

def gen_qkv(
    seq_lens, num_q_heads, num_k_heads, num_v_heads, head_size, dtype=torch.float16
):
    # qkv_rng = functools.partial(multidist_randn, mean_std=0.1)
    qkv_rng = functools.partial(multidist_randu, mean_std=0.05, lower=-0.25, upper=0.25)

    total_seq_lens = sum(seq_lens)
    q = qkv_rng(total_seq_lens * num_q_heads, head_size)
    k = qkv_rng(total_seq_lens * num_k_heads, head_size)
    v = qkv_rng(total_seq_lens * num_v_heads, head_size)

    q = q.reshape(total_seq_lens, num_q_heads, head_size).to(dtype).contiguous()
    k = k.reshape(total_seq_lens, num_k_heads, head_size).to(dtype).contiguous()
    v = v.reshape(total_seq_lens, num_v_heads, head_size).to(dtype).contiguous()

    return q, k, v


def _make_inputs(sequence_lengths, initial_state_mode="zero", device="cuda"):
    """Generate the public numerical contract, outside the timed Model.forward."""
    if not isinstance(sequence_lengths, list) or not 1 <= len(sequence_lengths) <= 16:
        raise ValueError("sequence_lengths must contain 1..16 sequences")
    if any(type(n) is not int or n < 1 for n in sequence_lengths):
        raise ValueError("each sequence length must be a positive integer")
    tokens = sum(sequence_lengths)
    if tokens > 16384:
        raise ValueError("sum(sequence_lengths) must be <= 16384")
    if initial_state_mode not in {"zero", "continuation"}:
        raise ValueError("initial_state_mode must be zero or continuation")
    with torch.device(device):
        q, k, v = gen_qkv(sequence_lengths, 4, 4, 32, 128, torch.bfloat16)
        # Exactly the upstream prefill test: normalize the BF16 tensor itself.
        k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
        alpha = torch.rand(tokens, 32, dtype=torch.float32)
        beta = torch.rand(tokens, 32, dtype=torch.float32)
        initial_state = torch.zeros(len(sequence_lengths), 32, 128, 128, dtype=torch.float32)
        if initial_state_mode == "continuation":
            # A short, independently sampled GDN prefix produces a valid carried state.
            # All preparation, including this reference call, is outside timing.
            prefix_lengths = [32] * len(sequence_lengths)
            pq, pk, pv = gen_qkv(prefix_lengths, 4, 4, 32, 128, torch.bfloat16)
            pk = torch.nn.functional.normalize(pk, p=2.0, dim=-1)
            pa = torch.rand(sum(prefix_lengths), 32, dtype=torch.float32)
            pb = torch.rand(sum(prefix_lengths), 32, dtype=torch.float32)
            ref_path = Path(__file__).with_name("reference.py")
            spec = importlib.util.spec_from_file_location("_gdn_prefix_reference", ref_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _, state_kv = module.blockwise_delta_rule(
                pq.float(), pk.float(), pv.float(), prefix_lengths,
                alpha=pa, beta=pb, scale_factor=1.0 / math.sqrt(128),
                state_dtype=torch.float32,
            )
            initial_state = state_kv.transpose(-1, -2).contiguous()
        offsets = [0]
        for length in sequence_lengths:
            offsets.append(offsets[-1] + length)
        cu_seqlens = torch.tensor(offsets, dtype=torch.int64)
        # Official test pattern: preallocate and poison every output element.
        output = torch.full((tokens, 32, 128), float("nan"), dtype=torch.bfloat16)
        output_state = torch.full_like(initial_state, float("nan"))
    return dict(q=q, k=k, v=v, alpha=alpha, beta=beta,
                initial_state=initial_state, cu_seqlens=cu_seqlens,
                output=output, output_state=output_state,
                scale=1.0 / math.sqrt(128))
