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

    def forward(self, a, b, d, masked_m, expected_m, compiled_dims='nk'):
        torch.backends.cuda.matmul.allow_tf32 = False
        p = {'a': a, 'b': b, 'd': d, 'masked_m': masked_m, 'expected_m': expected_m, 'compiled_dims': compiled_dims}
        result = expected(p)
        for e, count in enumerate(masked_m.tolist()):
            d[e, :count].copy_(result[e, :count])
        return None

def activation_operand(x):
    return x.float()

def weight_operand(x):
    return x.float()
