"""Independent FP32 oracle; decodes only actual ABI operands, no solution imports."""
import torch

def expected(p):
    a, b = (activation_operand(p['a']), weight_operand(p['b']))
    y = torch.zeros(p['d'].shape, dtype=torch.float32, device=a.device)
    for e, count in enumerate(p['masked_m'].tolist()):
        for lo in range(0, count, 64):
            hi = min(count, lo + 64)
            y[e, lo:hi] = torch.einsum('mk,nk->mn', a[e, lo:hi], b[e])
    return y.to(p['d'].dtype)

class Model(torch.nn.Module):

    def forward(self, a, b, d, masked_m, expected_m, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='nk', disable_ue8m0_cast=False):
        torch.backends.cuda.matmul.allow_tf32 = False
        p = {'a': a, 'b': b, 'd': d, 'masked_m': masked_m, 'expected_m': expected_m, 'recipe': recipe, 'recipe_a': recipe_a, 'recipe_b': recipe_b, 'compiled_dims': compiled_dims, 'disable_ue8m0_cast': disable_ue8m0_cast}
        result = expected(p)
        for e, count in enumerate(masked_m.tolist()):
            d[e, :count].copy_(result[e, :count])
        return None

def activation_operand(x):
    q, sf = x
    value=q.float()
    return value*sf.float().repeat_interleave(32,-1)

def weight_operand(x):
    q, sf = x
    q=q.view(torch.uint8)
    lut=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],device=q.device)
    codes=torch.stack((q&15,q>>4),-1).long()
    value=lut[codes].reshape(*q.shape[:-1],q.shape[-1]*2)
    return value*sf.float().repeat_interleave(32,-1)
