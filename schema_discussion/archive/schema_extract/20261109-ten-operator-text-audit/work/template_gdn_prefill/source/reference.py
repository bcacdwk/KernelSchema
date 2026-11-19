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

# Exact function excerpts from the pinned upstream reference; see flashinfer_pin.json.
import torch
import torch.nn.functional as F

def exclusive_cumsum(a: list[int]):
    r = [0]
    for v in a:
        r.append(r[-1] + v)
    return r

def matmul(a: torch.Tensor, b: torch.Tensor):
    assert a.dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    assert b.dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    if (
        a.dtype == torch.float16
        or b.dtype == torch.float16
        or a.dtype == torch.bfloat16
        or b.dtype == torch.bfloat16
    ):
        a_f32 = a.to(torch.float32)
        b_f32 = b.to(torch.float32)
        c_f32 = a_f32 @ b_f32
        if a.dtype == torch.bfloat16:
            return c_f32
        else:
            return c_f32.to(torch.float16)
    else:
        return a @ b

def _linear_attention(
    q: torch.Tensor,  # [seq_len, num_heads, head_size]
    k: torch.Tensor,  # [seq_len, num_heads, head_size]
    v: torch.Tensor,  # [seq_len, num_heads, head_size]
    *,
    decay_factor: torch.Tensor | None = None,
    qk_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    # Compute Q @ K^T
    num_qo_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    assert num_qo_heads == num_kv_heads
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    # print(q.shape, k.shape, v.shape)
    scores = matmul(q, k.transpose(-2, -1))

    # Create causal mask
    seq_len = q.size(-2)
    mask = torch.tril(
        torch.ones(num_qo_heads, seq_len, seq_len, dtype=q.dtype, device=q.device)
    )
    if decay_factor is not None and (decay_factor != 1.0).any():
        _, sq, sk = mask.shape
        with torch.device(q.device):
            e = (
                torch.arange(sq).unsqueeze(1) - torch.arange(sk).unsqueeze(0)
            ).unsqueeze(0)
            M = torch.pow(decay_factor, e)
            M[mask == 0.0] = 0.0
    elif qk_weight is not None:
        M = qk_weight.clone()
        M[mask == 0.0] = 0.0
    else:
        M = mask

    # Apply mask (Q @ K^T \odot M)
    masked_scores = scores * M

    # Apply to values (Q @ K^T \odot M) V
    out = matmul(masked_scores, v)
    out = out.transpose(0, 1)

    return out

def identity_add_strict_lower_diagonal(m: torch.Tensor):
    SIZE = m.size(-1)
    assert m.size(-2) == SIZE
    with torch.device(m.device):
        m = m.clone()
        mask = torch.arange(SIZE).unsqueeze(1) <= torch.arange(SIZE)
        m[:, mask] = 0.0
        # m[mask.unsqueeze(0)] = 0.0
        m = m + torch.eye(SIZE).unsqueeze(0)
    return m

def to_logspace_Gamma_and_gamma(alpha_HS: torch.Tensor, epsilon=1e-10):
    g = torch.log(alpha_HS + epsilon)
    cu_g = torch.cumsum(g, dim=-1)
    cu_g_HSS = cu_g.unsqueeze(2) - cu_g.unsqueeze(1)
    cu_g_HS1 = cu_g.unsqueeze(2)
    return cu_g_HSS, cu_g_HS1

@torch.inference_mode
def blockwise_delta_rule(
    q: torch.Tensor,  # [total_seq_len, num_qo_heads, head_size]
    k: torch.Tensor,  # [total_seq_len, num_kv_heads, head_size]
    v: torch.Tensor,  # [total_seq_len, num_kv_heads, head_size]
    seq_lens: list[int],  # sequence length for each sequence
    alpha: torch.Tensor | None = None,  # [total_seq_len, num_qo_heads]
    beta: torch.Tensor | None = None,  # [total_seq_len, num_qo_heads]
    block_size: int = 32,
    scale_factor=1.0,
    state_dtype: torch.dtype = torch.float32,
    initial_state: torch.Tensor | None = None,
    # intermediate_outputs = None,  # debug output
) -> torch.Tensor:
    total_seqlen = q.size(0)
    num_q_heads = q.size(1)
    num_k_heads = k.size(1)
    num_v_heads = v.size(1)
    num_sab_heads = max(num_q_heads, num_v_heads)
    head_size = q.size(2)

    if alpha is None:
        alpha = torch.ones(
            total_seqlen, num_sab_heads, dtype=torch.float32, device=q.device
        )
    if beta is None:
        beta = torch.ones(
            total_seqlen, num_sab_heads, dtype=torch.float32, device=q.device
        )

    if num_q_heads > num_v_heads:  # GQA
        num_qkv_heads = num_q_heads
        k = k.repeat_interleave(num_q_heads // num_k_heads, dim=1)
        v = v.repeat_interleave(num_q_heads // num_v_heads, dim=1)
    else:  # GVA
        num_qkv_heads = num_v_heads
        q = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)
        k = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)

    kv = torch.zeros(
        (len(seq_lens), num_sab_heads, head_size, head_size),
        dtype=state_dtype,
        device=q.device,
    )
    if initial_state is not None:
        assert tuple(initial_state.shape) == tuple(kv.shape)
        kv.copy_(initial_state.to(state_dtype))
    output = torch.zeros_like(q)

    seq_offset = exclusive_cumsum(seq_lens)
    for seq_idx, seq_start in enumerate(seq_offset[:-1]):
        seq_end = seq_offset[seq_idx + 1]
        blk_offset = seq_start
        state_HKV = kv[seq_idx].clone()
        while blk_offset < seq_end:
            is_full_block = seq_end - blk_offset >= block_size
            valid_len = block_size if is_full_block else seq_end - blk_offset
            o_t = output[blk_offset : min(seq_end, blk_offset + block_size)]
            if is_full_block:
                q_SHQ = q[blk_offset : blk_offset + block_size]
                k_SHK = k[blk_offset : blk_offset + block_size]
                v_SHV = v[blk_offset : blk_offset + block_size]
                alpha_SH = alpha[blk_offset : blk_offset + block_size]
                beta_SH = beta[blk_offset : blk_offset + block_size]
            else:
                q_SHQ = torch.zeros(
                    (block_size, num_qkv_heads, head_size),
                    dtype=q.dtype,
                    device=q.device,
                )
                k_SHK = torch.zeros(
                    (block_size, num_qkv_heads, head_size),
                    dtype=k.dtype,
                    device=k.device,
                )
                v_SHV = torch.zeros(
                    (block_size, num_qkv_heads, head_size),
                    dtype=v.dtype,
                    device=v.device,
                )
                alpha_SH = torch.ones(
                    block_size, num_sab_heads, dtype=alpha.dtype, device=alpha.device
                )
                beta_SH = torch.zeros(
                    block_size, num_sab_heads, dtype=beta.dtype, device=beta.device
                )
                q_SHQ[:valid_len] = q[blk_offset:seq_end]
                k_SHK[:valid_len] = k[blk_offset:seq_end]
                v_SHV[:valid_len] = v[blk_offset:seq_end]
                alpha_SH[:valid_len] = alpha[blk_offset:seq_end]
                beta_SH[:valid_len] = beta[blk_offset:seq_end]

            alpha_HS = alpha_SH.transpose(0, 1)
            beta_HS1 = beta_SH.transpose(0, 1).unsqueeze(2)
            Gamma_HSS, gamma_HS1 = to_logspace_Gamma_and_gamma(alpha_HS)
            block_gamma = gamma_HS1[:, [valid_len - 1], :]

            q_HSQ = q_SHQ.transpose(0, 1)
            k_HSK = k_SHK.transpose(0, 1)
            v_HSV = v_SHV.transpose(0, 1)

            IKK = identity_add_strict_lower_diagonal(
                beta_HS1 * torch.exp(Gamma_HSS) * matmul(k_HSK, k_HSK.transpose(-2, -1))
            )  # NOTE: beta scale row-wise
            T = torch.inverse(IKK) * beta_HS1.transpose(
                1, 2
            )  # NOTE: beta scale col-wise
            T = T.to(q.dtype)
            # new_v_HSV = matmul(T, (v_HSV - matmul(torch.exp(gamma_HS1) * k_HSK, state_HKV)))
            u_HSV = matmul(T, v_HSV)
            w_HSK = matmul(T, torch.exp(gamma_HS1) * k_HSK)
            new_v_HSV = u_HSV - matmul(
                w_HSK.to(torch.float32), state_HKV.to(torch.float32)
            ).to(u_HSV.dtype)
            new_v_SHV = new_v_HSV.transpose(0, 1)

            # if intermediate_outputs is not None:
            #     intermediate_outputs["G"].append(Gamma_HSS.clone())
            #     intermediate_outputs["g"].append(gamma_HS1.clone())
            #     intermediate_outputs["IKK"].append(IKK.clone())
            #     intermediate_outputs["T"].append(T.clone())
            #     intermediate_outputs["u"].append(u_HSV.clone())
            #     intermediate_outputs["w"].append(w_HSK.clone())
            #     intermediate_outputs["new_v"].append(new_v_HSV.clone())

            o_inter = (
                matmul(
                    torch.exp(gamma_HS1) * q_HSQ.to(torch.float32),
                    state_HKV.to(torch.float32),
                )
                .transpose(0, 1)
                .to(q.dtype)
            )
            o_intra = _linear_attention(
                q_SHQ, k_SHK, new_v_SHV, qk_weight=torch.exp(Gamma_HSS)
            )

            if is_full_block:
                o_t[:] = scale_factor * (o_inter + o_intra)
            else:
                o_t[:] = scale_factor * (o_inter + o_intra)[: o_t.shape[0]]

            inc_HKV = matmul(
                (torch.exp(block_gamma - gamma_HS1) * k_HSK)
                .transpose(-2, -1)
                .to(torch.float32),
                new_v_HSV.to(torch.float32),
            )
            state_HKV = (
                torch.exp(block_gamma) * state_HKV.to(torch.float32) + inc_HKV
            ).to(state_dtype)

            blk_offset += block_size

        kv[seq_idx, :, :, :] = state_HKV

    return output, kv


class Model(torch.nn.Module):
    """GDN core; only packed ABI, dtype and [V,K]/[K,V] state adaptation."""

    @torch.inference_mode()
    def forward(self, q, k, v, alpha, beta, initial_state, cu_seqlens, scale, output, output_state):
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        result, state_kv = blockwise_delta_rule(
            q.float(), k.float(), v.float(), lengths,
            alpha=alpha, beta=beta, scale_factor=scale,
            state_dtype=torch.float32,
            initial_state=initial_state.transpose(-1, -2).contiguous(),
        )
        output.copy_(result.to(q.dtype))
        output_state.copy_(state_kv.transpose(-1, -2))
        return output, output_state
