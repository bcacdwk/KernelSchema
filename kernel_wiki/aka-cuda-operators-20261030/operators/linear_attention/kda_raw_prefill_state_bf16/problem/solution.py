"""Original-library baseline for kda_raw_prefill_state_bf16.

Dependency: FlashInfer d849eca46875d3561fa48f6093936304580a1698.
Install the dependency before evaluation; no project native package or numerical patch.
"""
import torch

class Model(torch.nn.Module):
    implementation = "FlashInfer d849eca46875 recurrent_kda backend=cute-dsl"
    backend = "cute-dsl"

    @staticmethod
    def supports(shape):
        return (shape.get('beta_layout', 'contiguous') == 'contiguous',
                'Pinned CuTe prefill requires contiguous beta; row-strided stays in question domain')

    def forward(self, q,k,v,g,beta,A_log=None,dt_bias=None,scale=None,
                initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=False,lower_bound=None,cu_seqlens=None,
                beta_is_logit=False,output=None,state_checkpoints=None,
                checkpoint_cu_starts=None,checkpoint_every_n_tokens=0):
        from flashinfer.kda import recurrent_kda
        return recurrent_kda(q,k,v,g,beta,A_log=A_log,dt_bias=dt_bias,scale=scale,
            initial_state=initial_state,output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,lower_bound=lower_bound,
            cu_seqlens=cu_seqlens,beta_is_logit=beta_is_logit,output=output,
            state_checkpoints=state_checkpoints,checkpoint_cu_starts=checkpoint_cu_starts,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,backend=self.backend)
