"""Current-Schema keep/revise backend for the unique kernelschema API/CLI."""
from .writing import WritingBackend


def workflow_backend():
    return WritingBackend("optimize")
