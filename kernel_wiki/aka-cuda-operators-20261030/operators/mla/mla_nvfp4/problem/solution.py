"""Original-library baseline for mla_nvfp4.

Dependency: FlashInfer d849eca46875d3561fa48f6093936304580a1698.
Install the dependency before evaluation; no project native package or numerical patch.
"""
import torch

class Model(torch.nn.Module):
    def __init__(self, query, query_scale, kv_cache, kv_scale, block_tables, seq_lens, workspace_buffer, *, sm_scale, sinks=None, out=None, lse=None, return_lse=False, seq_lens_cpu=None, backend='cake'):
        super().__init__()
        from flashinfer.mla import prepare_nvfp4_batch_decode_with_kv_cache_mla
        arguments=locals().copy();arguments.pop("self");arguments.pop("__class__",None);arguments.pop("prepare_nvfp4_batch_decode_with_kv_cache_mla",None)
        self.runner=prepare_nvfp4_batch_decode_with_kv_cache_mla(**arguments)
    def forward(self):
        return self.runner()
