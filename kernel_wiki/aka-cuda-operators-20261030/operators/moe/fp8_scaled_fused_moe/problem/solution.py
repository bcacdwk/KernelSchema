"""Original-library baseline for fp8_scaled_fused_moe.

Dependency: FlashInfer d849eca46875d3561fa48f6093936304580a1698.
Install the dependency before evaluation; no project native package or numerical patch.
"""
import torch
class Model(torch.nn.Module):
    def forward(self, topk_ids, routing_bias, hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale, num_experts, top_k, n_group, topk_group, intermediate_size, local_expert_offset, local_num_experts, routed_scaling_factor, routing_method_type=0, use_shuffled_weight=False, weight_layout=0, do_finalize=True, enable_pdl=None, gemm1_lora_delta=None, output=None, tune_max_num_tokens=8192, fp8_quantization_type=1, activation_type=3, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None, valid_hidden_size=None, valid_intermediate_size=None):
        from flashinfer.fused_moe import trtllm_fp8_block_scale_routed_moe
        return trtllm_fp8_block_scale_routed_moe(topk_ids=topk_ids,routing_bias=routing_bias,hidden_states=hidden_states,hidden_states_scale=hidden_states_scale,gemm1_weights=gemm1_weights,gemm1_weights_scale=gemm1_weights_scale,gemm2_weights=gemm2_weights,gemm2_weights_scale=gemm2_weights_scale,num_experts=num_experts,top_k=top_k,n_group=n_group,topk_group=topk_group,intermediate_size=intermediate_size,local_expert_offset=local_expert_offset,local_num_experts=local_num_experts,routed_scaling_factor=routed_scaling_factor,routing_method_type=routing_method_type,use_shuffled_weight=use_shuffled_weight,weight_layout=weight_layout,do_finalize=do_finalize,enable_pdl=enable_pdl,gemm1_lora_delta=gemm1_lora_delta,output=output,tune_max_num_tokens=tune_max_num_tokens,fp8_quantization_type=fp8_quantization_type,activation_type=activation_type,gemm1_alpha=gemm1_alpha,gemm1_beta=gemm1_beta,gemm1_clamp_limit=gemm1_clamp_limit,valid_hidden_size=valid_hidden_size,valid_intermediate_size=valid_intermediate_size)
