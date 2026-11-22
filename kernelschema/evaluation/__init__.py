"""Task evaluation without importing torch or initializing a device.

Concrete tensor code is loaded only inside the explicitly selected executor.
"""

from .task import RowSoftmaxTask, freeze_task, stage
from .local import LocalExecutor

__all__ = ["RowSoftmaxTask", "freeze_task", "stage", "LocalExecutor"]
