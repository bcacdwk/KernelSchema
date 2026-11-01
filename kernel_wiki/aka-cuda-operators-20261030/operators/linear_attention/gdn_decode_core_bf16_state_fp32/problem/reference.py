"""Same stored input precision, independent FP32 conv/delta/gated norm."""
import torch
from torch.nn import functional as F

class Model(torch.nn.Module):
    def __init__(self,fault=None):super().__init__();self.fault=fault
    def forward(self,qkvz,ba,conv_weight,conv_state,A_log,dt_bias,state,norm_weight,out,metadata):
        t=metadata['num_actual_tokens'];slots=metadata['non_spec_state_indices_tensor'];hv=state.shape[1];c=conv_weight.shape[0];h=(c-hv*128)//256
        for start in range(0,t,32):
            rows=torch.arange(start,min(start+32,t),device=state.device);ids=slots[rows].long();valid=ids>0
            out[rows[~valid]]=0;rows=rows[valid];ids=ids[valid]
            if not len(rows):continue
            tokens=qkvz[rows,:c].float();history=conv_state[ids,:,:3].float();s=state[ids].float()
            if self.fault=='ignore_initial':s.zero_()
            w=conv_weight.float().flip(-1) if self.fault=='reverse_conv' else conv_weight.float()
            stream=torch.cat((history,tokens[:,:,None]),-1);v=(stream*w[None]).sum(-1)
            if self.fault!='no_silu':v=F.silu(v)
            if self.fault=='bypass_conv':v=tokens
            conv_state[ids,:,:3]=stream[:,:,1:].to(conv_state.dtype)
            # The next production stage reads the materialized BF16 QKV tensor.
            qkvz[rows,:c]=v.to(qkvz.dtype)
            v=v.to(qkvz.dtype).float()
            q,k,value=v.split((h*128,h*128,hv*128),-1)
            q=q.reshape(-1,h,128);k=k.reshape_as(q);value=value.reshape(-1,hv,128)
            if self.fault!='no_qk_norm':
                q=q/(q.square().sum(-1,keepdim=True)+1e-6).sqrt();k=k/(k.square().sum(-1,keepdim=True)+1e-6).sqrt()
            head=torch.arange(hv,device=state.device)//(hv//h)
            if self.fault=='wrong_gva':head=torch.arange(hv,device=state.device)%h
            q=q[:,head]*128**-.5;k=k[:,head]
            b,a=ba[rows].float().chunk(2,-1)
            decay=(-A_log.float().exp()*F.softplus(a+dt_bias.float())).exp()
            beta=b.sigmoid()
            if self.fault=='ignore_decay':decay.fill_(1)
            if self.fault=='ignore_beta':beta.fill_(1)
            s=s*decay[:,:,None,None]
            residual=(value-(s*k[:,:,None,:]).sum(-1))*beta[:,:,None]
            s=s+residual[:,:,:,None]*k[:,:,None,:]
            raw=(s*q[:,:,None,:]).sum(-1)
            # Recurrent output is BF16 before the separate gated RMSNorm.
            raw=raw.to(out.dtype).float()
            gate=qkvz[rows,c:].float().reshape(-1,hv,128).sigmoid()
            if self.fault=='wrong_gate':gate=F.silu(qkvz[rows,c:].float().reshape(-1,hv,128))
            if self.fault!='no_norm':raw=raw/(raw.square().mean(-1,keepdim=True)+1e-6).sqrt()*norm_weight.float()
            out[rows]=(raw*gate).to(out.dtype);state[ids]=s.to(state.dtype)
        return None
