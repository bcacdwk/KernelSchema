"""Shared collection of relation errors and small graph/range checks."""
from __future__ import annotations

from typing import Any

from ..diagnostics import Diagnostic


class CheckContext:
    def __init__(self) -> None:
        self.errors: list[Diagnostic] = []

    def require(self, condition: bool, path: str, message: str, *, missing: bool = False, hint: str = "") -> bool:
        if not condition:
            self.errors.append(Diagnostic(path, message, missing, hint))
        return condition

    def acyclic(self, graph: dict[Any, list[Any]], path: str,
                node_paths: dict[Any, str] | None = None) -> None:
        active, finished = set(), set()

        def visit(node: Any, chain: list[Any]) -> None:
            if node in active:
                self.require(False, node_paths[node] if node_paths else path,
                             "cycle: " + " -> ".join(map(str, chain + [node])))
                return
            if node in finished:
                return
            active.add(node)
            for target in graph.get(node, []):
                visit(target, chain + [node])
            active.remove(node)
            finished.add(node)

        for node in graph:
            visit(node, [])

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
