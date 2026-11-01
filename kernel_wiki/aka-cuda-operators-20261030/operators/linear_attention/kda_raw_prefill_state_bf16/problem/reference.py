"""Same stored inputs, independent FP32 math; no native intermediate rounding."""
import torch

def recurrence(q, k, v, raw_g, raw_beta, A_log, g_bias, state, cu_seqlens, lower_bound, scale, checkpoint_every=0, checkpoints=None, checkpoint_cu_starts=None):
    qf, kf = (q[0].float(), k[0].float())
    qf = qf / torch.linalg.vector_norm(qf, dim=-1, keepdim=True).clamp_min(1e-12)
    kf = kf / torch.linalg.vector_norm(kf, dim=-1, keepdim=True).clamp_min(1e-12)
    vf = v[0].float()
    logs = lower_bound * torch.sigmoid(A_log.exp()[None, :, None] * (raw_g[0].float() + g_bias.float().reshape(1, q.shape[2], q.shape[3])))
    beta = raw_beta[0].float().sigmoid()
    output = torch.empty_like(v)
    final = torch.empty(state.shape, dtype=torch.float32, device=q.device)
    boundaries = cu_seqlens.tolist()
    for sequence, (first, last) in enumerate(zip(boundaries, boundaries[1:])):
        current = state[sequence].float().clone()
        for start in range(first, last, 8):
            if checkpoint_every and (start - first) % checkpoint_every == 0:
                index = int(checkpoint_cu_starts[sequence]) + (start - first) // checkpoint_every
                checkpoints[index].copy_(current)
            stop = min(last, start + 8)
            qs, ks, vs = (qf[start:stop], kf[start:stop], vf[start:stop])
            decay = logs[start:stop].cumsum(0)
            beta_h = beta[start:stop].transpose(0, 1)
            n = stop - start
            mask = torch.ones((n, n), dtype=torch.bool, device=q.device).tril()
            difference = decay[:, None] - decay[None, :]
            ratios = torch.exp(difference.masked_fill(~mask[:, :, None, None], -torch.inf))
            kk = torch.einsum('thk,jhk,tjhk->htj', ks, ks, ratios).tril(-1)
            prior_k = torch.einsum('hvk,thk->htv', current, ks * decay.exp())
            rhs = beta_h[..., None] * (vs.transpose(0, 1) - prior_k)
            matrix = torch.eye(n, dtype=torch.float32, device=q.device)[None] + beta_h[..., None] * kk
            residual = torch.linalg.solve_triangular(matrix, rhs, upper=False, unitriangular=True)
            qk = torch.einsum('thk,jhk,tjhk->htj', qs, ks, ratios)
            prior_q = torch.einsum('hvk,thk->htv', current, qs * decay.exp())
            result = (prior_q + torch.bmm(qk, residual)) * scale
            output[0, start:stop] = result.transpose(0, 1).to(q.dtype)
            end_keys = ks * torch.exp(decay[-1:] - decay)
            current = current * decay[-1].exp()[:, None, :] + torch.einsum('htv,thk->hvk', residual, end_keys)
        final[sequence] = current
    return (output, final)

class Model(torch.nn.Module):

    def forward(self, q, k, v, g, beta, A_log=None, dt_bias=None, scale=None, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=False, lower_bound=None, cu_seqlens=None, beta_is_logit=False, output=None, state_checkpoints=None, checkpoint_cu_starts=None, checkpoint_every_n_tokens=0):
        assert output_final_state and use_qk_l2norm_in_kernel and use_gate_in_kernel and beta_is_logit
        assert lower_bound == -5.0 and initial_state is not None
        computed, final = recurrence(q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens, lower_bound, scale or 128 ** (-0.5), checkpoint_every_n_tokens, state_checkpoints, checkpoint_cu_starts)
        if output is None:
            output = computed
        else:
            output.copy_(computed)
        initial_state.copy_(final)
        pair = (output, initial_state)
        return (*pair, state_checkpoints) if checkpoint_every_n_tokens else pair
