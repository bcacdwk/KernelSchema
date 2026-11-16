"""Shared error collection and conservative checks for declared relations."""
from __future__ import annotations

from typing import Any

from ..diagnostics import Diagnostic
from ..pipeline_syntax import literal_integer, parse_members

EXECUTION_MODES = ("serial", "async_unordered", "async_ordered")


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
                    location = node_paths.get(node, path) if node_paths else path
                    self.require(False, location, "cycle: " + " -> ".join(map(str, cycle)))
                elif target not in finished:
                    active[target] = len(chain)
                    chain.append(target)
                    stack.append((target, iter(graph.get(target, []))))

    def interval(self, value: list[Any], path: str) -> None:
        lo, hi = map(literal_integer, value)
        if lo is not None and hi is not None:
            self.require(lo <= hi, path, "range endpoints are reversed")


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def indexed_values(value: Any, path: str):
    if isinstance(value, list):
        yield from ((item, f"{path}[{i}]") for i, item in enumerate(value))
    else:
        yield value, path


def role_member_range(role_body: dict, mapping: dict) -> range | None:
    """Known Worker.unit members occupied by a role, never its warp count."""
    count = literal_integer(mapping["Worker"]["unit_count"])
    selected = range(count) if count is not None else None
    pieces = parse_members(role_body["Info"]["Member"])
    if pieces and pieces[0][0] == mapping["Worker"]["unit"]:
        _, lower, upper = pieces[0]
        lo = literal_integer(lower)
        hi = literal_integer(upper) if upper is not None else (lo + 1 if lo is not None else None)
        selected = range(lo, hi) if lo is not None and hi is not None else None
    return selected


def concrete_conflict(first: Any, second: Any, field: str | None = None) -> bool:
    """Only definite field/value disagreements; no symbolic equivalence claims."""
    if first == second:
        return False
    if first is None or second is None:
        return True
    if field == "by":
        return as_list(first) != as_list(second)
    if field in {"unit", "direction", "dtype", "execution_mode"}:
        return first != second
    if isinstance(first, dict) and isinstance(second, dict):
        return set(first) != set(second) or any(concrete_conflict(first[k], second[k], k) for k in first)
    if isinstance(first, list) and isinstance(second, list):
        return len(first) != len(second) or any(concrete_conflict(a, b, field) for a, b in zip(first, second))
    left, right = literal_integer(first), literal_integer(second)
    if left is not None and right is not None:
        return left != right
    if isinstance(first, str) or isinstance(second, str):
        return False
    return first != second
