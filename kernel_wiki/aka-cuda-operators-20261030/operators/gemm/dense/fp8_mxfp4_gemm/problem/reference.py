"""Independent FP32 oracle; decodes only actual ABI operands, no solution imports."""
import torch

def expected(p):
    a, b = (activation_operand(p['a']), weight_operand(p['b']))
    d = p['d']
    y = torch.empty(d.shape, dtype=torch.float32, device=d.device)
    for start in range(0, a.shape[0], 64):
        y[start:start + 64] = torch.einsum('mk,nk->mn', a[start:start + 64], b)
    if p['alpha'] is not None:
        y *= p['alpha']
    if p['c'] is not None:
        y += p['c'].float()
    return y.to(d.dtype)

class Model(torch.nn.Module):

    def forward(self, a, b, d, c=None, recipe=None, recipe_a=None, recipe_b=None, compiled_dims='nk', disable_ue8m0_cast=False, alpha=None):
        torch.backends.cuda.matmul.allow_tf32 = False
        p = {'a': a, 'b': b, 'd': d, 'c': c, 'recipe': recipe, 'recipe_a': recipe_a, 'recipe_b': recipe_b, 'compiled_dims': compiled_dims, 'disable_ue8m0_cast': disable_ue8m0_cast, 'alpha': alpha}
        result = expected(p)
        d.copy_(result)
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
