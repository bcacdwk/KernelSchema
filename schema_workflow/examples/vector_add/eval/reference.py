"""Fixed FP32 oracle, authored before any product coder invocation."""
import torch


class Model(torch.nn.Module):
    def forward(self, x, y):
        return x + y
