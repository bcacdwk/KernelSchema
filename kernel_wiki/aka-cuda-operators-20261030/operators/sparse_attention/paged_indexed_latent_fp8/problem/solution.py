"""Original-library baseline for paged_indexed_latent_fp8.

Dependency: FlashInfer d849eca46875d3561fa48f6093936304580a1698.
Install the dependency before evaluation; no project native package or numerical patch.
"""
import torch
from flashinfer.mla._core import TrtllmGenMlaDecodeRunner, get_trtllm_gen_fmha_module

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();get_trtllm_gen_fmha_module()
    def forward(self,query,kv_cache,block_tables,seq_lens,sparse_mla_top_k_lens,
                bmm1_scale,bmm2_scale,workspace_buffer,multi_ctas_kv_counter_buffer,out):
        d=query.shape[-1]
        runner=TrtllmGenMlaDecodeRunner(kv_cache=kv_cache,workspace_buffer=workspace_buffer,
            sm_count=torch.cuda.get_device_properties(query.device).multi_processor_count,
            qk_nope_head_dim=128 if d==576 else 256,kv_lora_rank=512,qk_rope_head_dim=d-512,
            max_seq_len=block_tables.shape[-1],sparse_mla_top_k=block_tables.shape[-1],bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,sinks=None,skip_softmax_threshold_scale_factor=None,enable_pdl=False,
            is_var_seq=True,uses_shared_paged_kv_idx=True,return_lse=False,lse=None,return_lse_base=None,use_fp16_softmax=None)
        return runner.forward([query,block_tables,seq_lens,out,sparse_mla_top_k_lens],
            multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer)
