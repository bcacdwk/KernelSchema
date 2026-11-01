"""Original-library baseline for paged_latent_mxfp8_nvfp4.

Dependency: vLLM c723a831a81cb4ff89ea6d61b2d15109306ed1bc, vendored FlashMLA 0eee43b12f034b657133cf2afca6a72ebb6efccf.
Install the dependency before evaluation; no project native package or numerical patch.
"""
import torch
import vllm._flashmla_C  # Registers vLLM's original FlashMLA torch operators.


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.op = torch.ops._flashmla_C;self.implementation='FlashMLA native multi-query no-split'

    def forward(self,q,k_cache,indices,extra_k_cache,extra_indices,sm_scale,attn_sink,out):
        o,lse,_,_=self.op.sparse_decode_fwd(q,k_cache,indices,None,attn_sink,None,None,
            extra_k_cache,extra_indices,None,512,sm_scale,out)
        return o,lse
