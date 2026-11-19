"""Original-library baseline for fmha_mxfp8_qkv.

Dependency: nvidia-cudnn-frontend 1.30.0; original D256 exponent helpers, no project patch.
Install the dependency before evaluation; no project native package or numerical patch.
"""
from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

class Model(SdpaFwdDslSm100):
    # Constructor/check_support/compile/workspace query are native setup.
    def forward(self, q_tensor, k_tensor, v_tensor, o_tensor, lse_tensor=None, sinks=None, seq_q_lens=None, seq_kv_lens=None, scale_softmax=None, current_stream=None, sf_q=None, sf_k=None, sf_v=None, amax_o=None, descale_q=None, descale_k=None, descale_v=None, scale_o=None, workspace=None, block_table=None, block_table_v=None, gate=None, sf_o=None):
        return self.execute(q_tensor=q_tensor, k_tensor=k_tensor, v_tensor=v_tensor, o_tensor=o_tensor, lse_tensor=lse_tensor, sinks=sinks, seq_q_lens=seq_q_lens, seq_kv_lens=seq_kv_lens, scale_softmax=scale_softmax, current_stream=current_stream, sf_q=sf_q, sf_k=sf_k, sf_v=sf_v, amax_o=amax_o, descale_q=descale_q, descale_k=descale_k, descale_v=descale_v, scale_o=scale_o, workspace=workspace, block_table=block_table, block_table_v=block_table_v, gate=gate, sf_o=sf_o)
    __call__ = SdpaFwdDslSm100.execute
