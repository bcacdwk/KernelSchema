"""Step-local event identity, source provenance and cycle diagnostics."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from graphlib import CycleError, TopologicalSorter


class GraphError(ValueError):
    """A definite counterexample in the supplied finite pipeline sample."""


class ExpansionError(GraphError):
    """Missing values, unsupported evaluation or a tool limit: not completed."""


@dataclass
class Operation:
    id: str
    role: str
    member: int
    action: str
    execution_mode: str
    path: str
    context: dict[str, int]

    def event(self, time: str) -> str:
        if time not in {"issue", "done"}:
            raise ValueError("an action event time is issue or done")
        return self.id + "@" + time


@dataclass
class StepGraph:
    step: str
    operations: dict[str, Operation] = field(default_factory=dict)
    edges: dict[tuple[str, str, str], set[str]] = field(default_factory=dict)
    edge_paths: dict[tuple[str, str, str], set[str]] = field(default_factory=dict)
    boundaries: list[dict] = field(default_factory=list)
    events: dict[str, list[str]] = field(default_factory=dict)
    cycle: list[str] = field(default_factory=list)

    def add(self, source: str, target: str, kind: str, reason: str, path: str = "") -> None:
        key = (source, target, kind)
        self.edges.setdefault(key, set()).add(reason)
        if path:
            self.edge_paths.setdefault(key, set()).add(path)
        if len(self.edges) > 100000:
            raise ExpansionError("event edge limit exceeded (100000); use a smaller sample")

    def check(self) -> None:
        predecessors = {op.event(time): set() for op in self.operations.values() for time in ("issue", "done")}
        for source, target, _ in self.edges:
            if source not in predecessors or target not in predecessors:
                raise GraphError(f"edge names an absent event: {source} -> {target}")
            predecessors[target].add(source)
        self.cycle = []
        try:
            tuple(TopologicalSorter(predecessors).static_order())
        except CycleError as error:
            self.cycle = error.args[1]

    def report(self) -> dict:
        return {
            "step": self.step,
            "scope": "given sample, one representative worker, local step events; no cross-step liveness proof",
            "acyclic": not self.cycle,
            "cycle": list(self.cycle),
            "actions": [{**asdict(op), "issue": op.event("issue"), "done": op.event("done")}
                        for op in self.operations.values()],
            "events": {name: list(events) for name, events in self.events.items()},
            "edges": [{"from": source, "to": target, "kind": kind,
                       "reasons": sorted(reasons), "paths": sorted(self.edge_paths.get((source, target, kind), ()))}
                      for (source, target, kind), reasons in sorted(self.edges.items())],
            "boundary_skips": list(self.boundaries),
        }
