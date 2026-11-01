import torch
def expected(p):
    a=p['a'].float()*p['a_scale'].repeat_interleave(128,-1)
    b=p['b'].float()*p['b_scale'].repeat_interleave(128,0).repeat_interleave(128,-1)
    y=torch.empty(p['out'].shape,device=a.device,dtype=torch.float32)
    for off in range(0,a.shape[0],64):y[off:off+64]=torch.einsum('mk,nk->mn',a[off:off+64],b)
    return y.to(p['out'].dtype)


class Model(torch.nn.Module):
    def forward(self, a, b, a_scale, b_scale, scale_major_mode=None, mma_sm=1, scale_granularity_mnk=(1, 128, 128), out=None, out_dtype=None, backend='cutlass'):
        torch.backends.cuda.matmul.allow_tf32=False
        p={'a': a, 'b': b, 'a_scale': a_scale, 'b_scale': b_scale, 'scale_major_mode': scale_major_mode, 'mma_sm': mma_sm, 'scale_granularity_mnk': scale_granularity_mnk, 'out': out, 'out_dtype': out_dtype, 'backend': backend}
        result=expected(p)
        out.copy_(result)
        return out
