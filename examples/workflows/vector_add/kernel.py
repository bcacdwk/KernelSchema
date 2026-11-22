import torch
import torch.nn as nn
import triton
import triton.language as tl

# Design-fixed tiling: each Triton program owns exactly one contiguous tile of
# 1024 elements of the single 1-D axis; grid = ceil_div(n, 1024); masked tail
# keeps the kernel valid for arbitrary n.
BLOCK_SIZE = 1024


@triton.jit
def _scaled_add_kernel(
    x_ptr,
    r_ptr,
    y_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Independent masked contiguous bf16 loads of both inputs.
    x = tl.load(x_ptr + offsets, mask=mask)
    r = tl.load(r_ptr + offsets, mask=mask)

    # Reference numerics: PyTorch promotes the Python scalar to the common
    # bf16 dtype, evaluates alpha * residual as one bf16 op (product rounded
    # to bf16), then x + product as a second bf16 op (sum rounded to bf16).
    # Wider fp32 arithmetic is used inside each step, but the two rounding
    # boundaries are preserved; no single fused multiply-add.
    alpha_bf = alpha.to(tl.bfloat16)
    prod = (alpha_bf.to(tl.float32) * r.to(tl.float32)).to(tl.bfloat16)
    y = (x.to(tl.float32) + prod.to(tl.float32)).to(tl.bfloat16)

    # Masked contiguous bf16 store; out-of-range lanes are never written.
    tl.store(y_ptr + offsets, y, mask=mask)


class Model(nn.Module):
    """y = x + alpha * residual via one fused Triton kernel (bf16 in/out)."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        if not residual.is_contiguous():
            residual = residual.contiguous()
        y = torch.empty_like(x)
        n = x.numel()
        if n == 0:
            return y
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        _scaled_add_kernel[grid](
            x, residual, y, n, self.alpha, BLOCK_SIZE=BLOCK_SIZE
        )
        return y
