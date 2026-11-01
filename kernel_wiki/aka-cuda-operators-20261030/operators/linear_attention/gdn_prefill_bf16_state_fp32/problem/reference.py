"""Independent FP32 block solve of the delta recurrence; no native rounding."""
import torch

def recurrence(q,k,v,g,beta,state,cu,block=64,fault=None):
    q=q[0].float();k=k[0].float();v=v[0].float();g=g[0].float();beta=beta[0].float()
    ratio=v.shape[1]//q.shape[1];heads=torch.arange(v.shape[1],device=q.device)//ratio
    if fault=='wrong_gva':heads=torch.arange(v.shape[1],device=q.device)%q.shape[1]
    q=q[:,heads];k=k[:,heads]
    if fault=='renormalize':q=q/(q.square().sum(-1,keepdim=True)+1e-6).sqrt();k=k/(k.square().sum(-1,keepdim=True)+1e-6).sqrt()
    if fault=='ignore_decay':g=torch.zeros_like(g)
    if fault=='double_exp':g=g.exp()  # Supplying log g as an exp-decay is not equivalent.
    if fault=='ignore_beta':beta=torch.ones_like(beta)
    output=torch.empty_like(v);final=[];bounds=cu.tolist()
    for seq,(begin,end) in enumerate(zip(bounds,bounds[1:])):
        s=state[seq].float().transpose(-1,-2).clone()
        if fault=='ignore_initial':s.zero_()
        for start in range(begin,end,block):
            stop=min(end,start+block);length=stop-start
            qq=q[start:stop].transpose(0,1)*128**-.5;kk=k[start:stop].transpose(0,1);vv=v[start:stop].transpose(0,1)
            gg=g[start:stop].T.cumsum(-1);bb=beta[start:stop].T
            lower=torch.ones((length,length),device=q.device,dtype=torch.bool).tril()
            decay=(gg[:,:,None]-gg[:,None,:]).masked_fill(~lower,-torch.inf).exp()
            prefix=gg.exp();gram=kk@kk.transpose(-1,-2)
            system=(gram*decay*bb[:,:,None]).tril(-1)+torch.eye(length,device=q.device)[None]
            rhs=bb[:,:,None]*(vv-prefix[:,:,None]*(kk@s))
            delta=torch.linalg.solve_triangular(system,rhs,upper=False,unitriangular=True)
            oo=prefix[:,:,None]*(qq@s)+((qq@kk.transpose(-1,-2))*decay)@delta
            output[start:stop]=oo.transpose(0,1)
            s=prefix[:,-1,None,None]*s+kk.transpose(-1,-2)@(decay[:,-1,:,None]*delta)
        final.append(s.transpose(-1,-2))
    return output.unsqueeze(0).bfloat16(),torch.stack(final)

class Model(torch.nn.Module):
    def __init__(self,fault=None):super().__init__();self.fault=fault
    def forward(self,q,k,v,g,beta,initial_state,output_final_state,cu_seqlens=None,chunk_indices=None,chunk_offsets=None,use_qk_l2norm_in_kernel=True,core_attn_out=None):
        if use_qk_l2norm_in_kernel:
            q=q.float()/(q.float().square().sum(-1,keepdim=True)+1e-6).sqrt()
            k=k.float()/(k.float().square().sum(-1,keepdim=True)+1e-6).sqrt()
        if cu_seqlens is None:cu_seqlens=torch.tensor([0,q.shape[1]],device=q.device,dtype=torch.int32)
        out,final=recurrence(q,k,v,g,beta,initial_state,cu_seqlens,fault=self.fault)
        if core_attn_out is not None:core_attn_out.view(-1)[:out.numel()].copy_(out.reshape(-1))
        return out,final if output_final_state else None
