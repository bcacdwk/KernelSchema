import torch
from qwen38_core_qsa_cutedsl.operator import run

class Model(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, logical_indices, block_table,
                token_to_req, use_prefill_config, output_gate, out=None):
        return run(q, k_cache, v_cache, logical_indices, block_table,
                   token_to_req, use_prefill_config, output_gate, out)
