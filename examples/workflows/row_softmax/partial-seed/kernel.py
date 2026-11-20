"""Codex-authored manual partial seed; intentionally incomplete, never measured.

This seed preserves the ABI and makes the remaining fixed-Schema work explicit.
It is not the performance baseline and has never been GPU qualified.
"""
class Model:
    def forward(self, x, out):
        m, n = x.shape
        if m not in (1, 32) or n not in (128, 1024):
            raise ValueError("outside the fixed Task")
        raise NotImplementedError("Implement the selected row ownership and normalization in Triton")
