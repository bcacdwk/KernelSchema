"""Work scheduling: launch dependencies, shared workers and data handoffs."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from .common import CheckContext, as_list, indexed_values


def validate(check: CheckContext, plan: dict[str, Any], path: str) -> None:
    work = plan["design"]["work_schedule"]
    pipe = plan["design"]["tile_pipeline"]["phase"]
    launches = work["flow"]["launch"]
    owner = {p: i for i, launch in enumerate(launches) for p in launch["phase"]}
    graph = defaultdict(list)

    def event(ref: str, p: str, participants: set[str] | None = None) -> tuple[str, str] | None:
        phase, milestone = ref.split(".", 1)
        if not check.require(phase in pipe, p, f"unknown event phase {phase}"):
            return None
        if participants is not None:
            check.require(phase in participants, p, "event belongs to the wrong handoff participant")
        if milestone not in {"kernel_done", "launch_ready"}:
            identity = milestone.split("(", 1)[0]
            check.require(any(m.split("(", 1)[0] == identity for m in pipe[phase].get("milestone", {})),
                         p, f"undefined milestone {ref}")
        return phase, milestone

    for i, launch in enumerate(launches):
        p = f"{path}.design.work_schedule.flow.launch[{i}]"
        for ref, rp in indexed_values(launch.get("start_after", []), p + ".start_after"):
            result = event(ref, rp)
            if result:
                phase, milestone = result
                check.require(owner[phase] != i, rp, "a launch cannot wait for its own event before starting")
                if milestone == "kernel_done":
                    graph[i].append(owner[phase])
                if milestone == "launch_ready":
                    check.require("protocol" in launch, p + ".protocol", "launch_ready requires an explicit launch protocol")
        if len(launch["phase"]) > 1 and launch["order"] != "concurrent":
            first = work["phase"][launch["phase"][0]]
            resources = set()
            for pid in launch["phase"]:
                current = work["phase"][pid]
                for field in ("worker", "count", "placement"):
                    check.require(not concrete_conflict(first["mapping"].get(field), current["mapping"].get(field)),
                                 p, "sequential phases have incompatible worker mappings")
                for name in pipe[pid].get("resource", {}):
                    check.require(name not in resources, p, f"shared-worker resource {name} is defined more than once")
                    resources.add(name)
                if launch["order"] == "per_worktile":
                    pairs = [(first["decomposition"]["partition"], current["decomposition"]["partition"]),
                             (first["schedule"].get("worktile_order"), current["schedule"].get("worktile_order")),
                             (first["schedule"]["worker_assignment"].get("preassigned"),
                              current["schedule"]["worker_assignment"].get("preassigned"))]
                    for a, b in pairs:
                        check.require(not concrete_conflict(a, b), p, "per_worktile requires matching coordinates, order and preassignment")
    check.acyclic(graph, path + ".design.work_schedule.flow.launch")

    for name, handoff in work["flow"].get("handoff", {}).items():
        p = path + ".design.work_schedule.flow.handoff." + name
        producer, consumers = handoff["producer"], handoff["consumer"]
        for ref, rp in indexed_values(handoff.get("ready", []), p + ".ready"):
            event(ref, rp, {producer})
        for ref, rp in indexed_values(handoff.get("wait_at", []), p + ".wait_at"):
            event(ref, rp, set(consumers))
        if handoff["storage"] != "global":
            _, _, rp, _, rn = handoff["storage"].split(".")
            check.require(rp in pipe and rn in pipe.get(rp, {}).get("resource", {}), p + ".storage", "undefined resource")
            participants = [producer, *consumers, rp]
            if all(x in owner for x in participants):
                submit = owner[producer]
                check.require(all(owner[x] == submit for x in participants) and launches[submit].get("order") != "concurrent",
                             p + ".storage", "local storage requires the same worker instance within one launch")
    for pid, phase in work["phase"].items():
        p = path + ".design.work_schedule.phase." + pid
        for dim, part in phase["decomposition"]["partition"].items():
            if "range" in part:
                check.interval(part["range"], f"{p}.decomposition.partition.{dim}.range")
        for i, member in enumerate(phase["mapping"].get("member", [])):
            for dimension, region in member.items():
                check.interval(region, f"{p}.mapping.member[{i}].{dimension}")


def concrete_conflict(a: Any, b: Any, field: str | None = None) -> bool:
    """Detect definite disagreements, leaving symbolic equivalence to review."""
    if a == b:
        return False
    if a is None or b is None:
        return True
    if field in {"unit", "by", "direction"}:
        return a != b
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) != set(b) or any(concrete_conflict(a[k], b[k], k) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) != len(b) or any(concrete_conflict(x, y, field) for x, y in zip(a, b))
    if isinstance(a, str) or isinstance(b, str):
        return False
    return a != b
