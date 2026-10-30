"""Shared action parsing and integer substitution for finite expansion."""
from __future__ import annotations

from ..rules.pipeline import action_parts, parse_action, split_top
from .expressions import Domains, tree, variables
from .model import GraphError


def operands(action: str) -> tuple[str, list[tuple[str, list[str]]]]:
    if parse_action(action) is None:
        raise GraphError(f"invalid action: {action}")
    verb, objects = action_parts(action)
    result = []
    for value in objects:
        name, separator, args = value.partition("(")
        result.append((name, split_top(args[:-1], ",") if separator else []))
    return verb, result


def instantiate(action: str, domains: Domains, env: dict) -> str:
    verb, objects = operands(action)
    indices = iter(domains.integers([arg for _, args in objects for arg in args], env))
    return verb + ":" + "@".join(name + ("(" + ",".join(str(next(indices)) for _ in args) + ")"
                                        if args else "") for name, args in objects)


def action_variables(action: str, domains: Domains) -> set[str]:
    return set().union(*(variables(tree(a)) & domains.definitions.keys()
                         for _, args in operands(action)[1] for a in args))
