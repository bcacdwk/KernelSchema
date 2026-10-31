"""Shared collection of relation errors and small graph/range checks."""
from __future__ import annotations

from typing import Any

from ..diagnostics import Diagnostic

EXECUTION_MODES = ("serial", "async", "async_ordered_completion")


class CheckContext:
    def __init__(self) -> None:
        self.errors: list[Diagnostic] = []

    def require(self, condition: bool, path: str, message: str, *, missing: bool = False, hint: str = "") -> bool:
        if not condition:
            self.errors.append(Diagnostic(path, message, missing, hint))
        return condition

    def acyclic(self, graph: dict[Any, list[Any]], path: str,
                node_paths: dict[Any, str] | None = None) -> None:
        active, finished = {}, set()
        for root in graph:
            if root in finished:
                continue
            chain = [root]
            active[root] = 0
            stack = [(root, iter(graph.get(root, [])))]
            while stack:
                node, targets = stack[-1]
                try:
                    target = next(targets)
                except StopIteration:
                    stack.pop()
                    chain.pop()
                    active.pop(node)
                    finished.add(node)
                    continue
                if target in active:
                    cycle = chain[active[target]:] + [target]
                    self.require(False, node_paths[target] if node_paths else path,
                                 "cycle: " + " -> ".join(map(str, cycle)))
                elif target not in finished:
                    active[target] = len(chain)
                    chain.append(target)
                    stack.append((target, iter(graph.get(target, []))))

    def interval(self, value: list[Any], path: str) -> None:
        lo, hi = value
        if type(lo) is int and type(hi) is int:
            self.require(lo <= hi, path, "range endpoints are reversed")


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def indexed_values(value: Any, path: str):
    if isinstance(value, list):
        yield from ((item, f"{path}[{i}]") for i, item in enumerate(value))
    else:
        yield value, path
