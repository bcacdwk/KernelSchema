import torch
def fp4(q,sf,swizzled=True):
    q=q.view(torch.uint8);m,k=q.shape[0],q.shape[1]*2
    if swizzled:
        sf=sf.view(torch.uint8).reshape((m+127)//128,(k+63)//64,32,4,4)
        sf=sf.permute(0,3,2,1,4).reshape((m+127)//128*128,(k+63)//64*4)[:m,:k//16]
        sf=sf.contiguous().view(torch.float8_e4m3fn)
    lut=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],device=q.device)
    return lut[torch.stack((q&15,q>>4),-1).long()].reshape(m,k)*sf.float().repeat_interleave(16,-1)
def expected(p):
    a=p['a'].float();b=fp4(p['b'],p['b_descale'],False)
    y=torch.empty(p['out'].shape,device=a.device,dtype=torch.float32)
    for off in range(0,a.shape[0],64):y[off:off+64]=torch.einsum('mk,nk->mn',a[off:off+64],b)
    if p['alpha'] is not None:y*=p['alpha']
    return y.to(p['out'].dtype)


class Model(torch.nn.Module):
    def forward(self, a, b, b_descale, alpha=None, *, backend, out_dtype=None, out=None, block_size=16, enable_pdl=True):
        torch.backends.cuda.matmul.allow_tf32=False
        p={'a': a, 'b': b, 'b_descale': b_descale, 'alpha': alpha, 'backend': backend, 'out_dtype': out_dtype, 'out': out, 'block_size': block_size, 'enable_pdl': enable_pdl}
        result=expected(p)
        out.copy_(result)
        return out
