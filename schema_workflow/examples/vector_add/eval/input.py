"""Fixed inputs; case coverage does not narrow the task's legal domain."""
import torch


def _make_inputs(n: int):
    if not 1 <= n <= 65536:
        raise ValueError("n must be in [1, 65536]")
    g = torch.Generator(device="cuda").manual_seed(20261122 + n)
    x = torch.rand(n, generator=g, device="cuda", dtype=torch.float32) * 2 - 1
    y = torch.rand(n, generator=g, device="cuda", dtype=torch.float32) * 2 - 1
    return {"x": x, "y": y}
