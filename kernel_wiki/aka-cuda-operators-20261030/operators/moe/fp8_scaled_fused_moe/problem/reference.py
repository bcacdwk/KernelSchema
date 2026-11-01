"""Independent FP32 reference over the actual stored production inputs.

**version**: explicit-deepseek-fp8-v1

**graph**: Decode actual FP8 payload/FP32 scales → FP32 GEMM1 → block128 FP8 QDQ → clamp/SwiGLU(alpha,beta) → block128 FP8 QDQ → FP32 GEMM2 → routing reduction → BF16 output.

**boundaries**: Both FC1 pre-activation output and post-activation FC2 input. Groups of128 consecutive channels per token/expert; up and gate widths are multiples of128.

**quantization**: sf=max(max(abs(group))/448, float32.min_normal); FP32 dequantization multiplier. payload=RN_E4M3FN_sat448(x/sf); dequant=payload*sf. Nearest ties-to-even. All-zero group dequantizes to zero.

**layout**: Logical scales are per row/group; production stores scales transposed [channel_group, padded_tokens]. Existing prepared input/weight layout and production buffers are unchanged.

**source**: Pinned FlashInfer d849eca: trtllm_fused_moe_kernel_launcher.cu allocates FP8+FP32 scales before/after activation; trtllm_fused_moe_dev_kernel.cu activation block128 scale and gatedSilu.

**arithmetic**: Independent FP32 between QDQ; TF32 disabled. No native quantizer, native internal tensor, pre-quantized hidden input or FP64 scoring truth.

Numerical comparison uses the supplied standard runner config. No native
kernel or hidden prequantization tensor supplies the scoring truth.
"""
import torch

def block128(value):
    blocks = value.float().reshape(*value.shape[:-1], -1, 128)
    scale = (blocks.abs().amax(-1) / 448.0).clamp_min(torch.finfo(torch.float32).tiny)
    code = (blocks / scale[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    return (code * scale[..., None]).reshape_as(value)

def expected(p):
    a = p['hidden_states'].float() * p['hidden_states_scale'].T.repeat_interleave(128, -1)
    w1 = p['gemm1_weights'].float() * p['gemm1_weights_scale'].repeat_interleave(128, -2).repeat_interleave(128, -1)
    w2 = p['gemm2_weights'].float() * p['gemm2_weights_scale'].repeat_interleave(128, -2).repeat_interleave(128, -1)
    ids, scores = p['topk_ids']
    y = torch.zeros_like(a)
    for e in range(w1.shape[0]):
        tok, slot = (ids == e).nonzero(as_tuple=True)
        for off in range(0, len(tok), 64):
            rows = tok[off:off + 64]
            slots = slot[off:off + 64]
            up, gate = block128(torch.einsum('th,ih->ti', a[rows], w1[e])).chunk(2, -1)
            limit = p['gemm1_clamp_limit']
            if limit is not None:
                up = up.clamp(-limit[e], limit[e])
                gate = gate.clamp(max=limit[e])
            alpha = 1.0 if p['gemm1_alpha'] is None else p['gemm1_alpha'][e]
            beta = 0.0 if p['gemm1_beta'] is None else p['gemm1_beta'][e]
            act = block128(gate * torch.sigmoid(alpha * gate) * (up + beta))
            y.index_add_(0, rows, torch.einsum('ti,hi->th', act, w2[e]).to(torch.bfloat16).float() * scores[rows, slots, None])
    return y.to(p['output'].dtype)

class Model(torch.nn.Module):

    def forward(self, topk_ids, routing_bias, hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale, num_experts, top_k, n_group, topk_group, intermediate_size, local_expert_offset, local_num_experts, routed_scaling_factor, routing_method_type=0, use_shuffled_weight=False, weight_layout=0, do_finalize=True, enable_pdl=None, gemm1_lora_delta=None, output=None, tune_max_num_tokens=8192, fp8_quantization_type=1, activation_type=3, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None, valid_hidden_size=None, valid_intermediate_size=None):
        torch.backends.cuda.matmul.allow_tf32 = False
        p = {'topk_ids': topk_ids, 'routing_bias': routing_bias, 'hidden_states': hidden_states, 'hidden_states_scale': hidden_states_scale, 'gemm1_weights': gemm1_weights, 'gemm1_weights_scale': gemm1_weights_scale, 'gemm2_weights': gemm2_weights, 'gemm2_weights_scale': gemm2_weights_scale, 'num_experts': num_experts, 'top_k': top_k, 'n_group': n_group, 'topk_group': topk_group, 'intermediate_size': intermediate_size, 'local_expert_offset': local_expert_offset, 'local_num_experts': local_num_experts, 'routed_scaling_factor': routed_scaling_factor, 'routing_method_type': routing_method_type, 'use_shuffled_weight': use_shuffled_weight, 'weight_layout': weight_layout, 'do_finalize': do_finalize, 'enable_pdl': enable_pdl, 'gemm1_lora_delta': gemm1_lora_delta, 'output': output, 'tune_max_num_tokens': tune_max_num_tokens, 'fp8_quantization_type': fp8_quantization_type, 'activation_type': activation_type, 'gemm1_alpha': gemm1_alpha, 'gemm1_beta': gemm1_beta, 'gemm1_clamp_limit': gemm1_clamp_limit, 'valid_hidden_size': valid_hidden_size, 'valid_intermediate_size': valid_intermediate_size}
        result = expected(p)
        output.copy_(result)
        return output
