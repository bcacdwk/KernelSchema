"""Event identity, provenance and graph diagnostics."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from graphlib import CycleError, TopologicalSorter


class GraphError(ValueError):
    """The supplied finite pipeline cannot be expanded unambiguously."""


class ExpansionError(GraphError):
    """Missing sample values or a tool limit; not a rejection of schema syntax."""


@dataclass
class Operation:
    id: str
    role: str
    member: int
    action: str
    execution_mode: str
    path: str
    context: dict[str, int]

    def event(self, event: str) -> str:
        return self.id + "@" + event


@dataclass
class PhaseGraph:
    phase: str
    operations: dict[str, Operation] = field(default_factory=dict)
    edges: dict[tuple[str, str, str], set[str]] = field(default_factory=dict)
    boundaries: list[dict] = field(default_factory=list)
    milestones: dict[str, list[str]] = field(default_factory=dict)
    cycle: list[str] = field(default_factory=list)

    def add(self, source: str, target: str, kind: str, reason: str) -> None:
        self.edges.setdefault((source, target, kind), set()).add(reason)
        if len(self.edges) > 100000:
            raise ExpansionError("event edge limit exceeded (100000); use a smaller sample")

    def check(self) -> None:
        predecessors = {op.event(event): set() for op in self.operations.values()
                        for event in ("issue", "done")}
        for source, target, _ in self.edges:
            predecessors[target].add(source)
        try:
            tuple(TopologicalSorter(predecessors).static_order())
        except CycleError as error:
            self.cycle = error.args[1]

    def report(self) -> dict:
        return {
            "phase": self.phase, "acyclic": not self.cycle, "cycle": self.cycle,
            "operations": [asdict(op) for op in self.operations.values()],
            "edges": [{"from": a, "to": b, "kind": kind, "reasons": sorted(reasons)}
                      for (a, b, kind), reasons in sorted(self.edges.items())],
            "boundary_skips": self.boundaries, "milestones": self.milestones,
        }
