"""Failures that prevent checking, rather than invalidate a plan."""
from functools import wraps


class SchemaError(Exception):
    """Invalid YAML or task relations (exit 1)."""


class ToolError(Exception):
    """The check could not be completed (exit 2)."""


class CapacityError(ToolError):
    """An implementation limit was reached while reading or checking input."""


def bounded(function):
    """Keep Python recursion limits out of public input-validity diagnostics."""
    @wraps(function)
    def call(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except RecursionError as error:
            raise CapacityError(f"{function.__name__}: nesting or dependency depth exceeds the tool's capacity") from error
    return call
