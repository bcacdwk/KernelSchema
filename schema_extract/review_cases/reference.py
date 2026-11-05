"""Synthetic scalar-copy semantics for document review, not a GPU benchmark."""


def reference(x, write_output=True):
    return x if write_output else None


def kernel(x, write_output=True):
    return reference(x, write_output)


def inputs():
    return [1.0]
