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

    def forward(self, a, b, d, c=None, compiled_dims='nk', alpha=None):
        torch.backends.cuda.matmul.allow_tf32 = False
        p = {'a': a, 'b': b, 'd': d, 'c': c, 'compiled_dims': compiled_dims, 'alpha': alpha}
        result = expected(p)
        d.copy_(result)
        return None

def activation_operand(x):
    return x.float()

def weight_operand(x):
    return x.float()
