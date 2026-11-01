"""Independent FP32 oracle; decodes only actual ABI operands, no solution imports."""
import torch

def expected(p):
    a, b = (activation_operand(p['a']), weight_operand(p['b']))
    y = torch.zeros(p['d'].shape, dtype=torch.float32, device=a.device)
    start = 0
    for e in range(b.shape[0]):
        if p['use_psum_layout']:
            end = int(p['grouped_layout'][e])
            rows = torch.arange(start, end, device=a.device)
            start = end
        else:
            rows = (p['grouped_layout'] == e).nonzero().flatten()
        for idx in rows.split(64):
            y[idx] = torch.einsum('mk,nk->mn', a[idx], b[e])
    return y.to(p['d'].dtype)

class Model(torch.nn.Module):

    def forward(self, a, b, d, grouped_layout, compiled_dims='nk', use_psum_layout=False, ensure_zero_padding=True, expected_m_for_psum_layout=None):
        torch.backends.cuda.matmul.allow_tf32 = False
        p = {'a': a, 'b': b, 'd': d, 'grouped_layout': grouped_layout, 'compiled_dims': compiled_dims, 'use_psum_layout': use_psum_layout, 'ensure_zero_padding': ensure_zero_padding, 'expected_m_for_psum_layout': expected_m_for_psum_layout}
        result = expected(p)
        d.copy_(result)
        return None

def activation_operand(x):
    return x.float()

def weight_operand(x):
    return x.float()
