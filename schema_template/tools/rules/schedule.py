"""Submission dependencies, shared-worker constraints and explicit handoffs."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import re
from typing import Any

from ..diagnostics import field_path
from ..pipeline_syntax import EventRef, Operand, literal_integer, parse_action, parse_event, parse_pipe
from .common import CheckContext, as_list, concrete_conflict, indexed_values, role_member_range


def consumer_waits(handoff: dict) -> dict[str, tuple[str, ...] | None]:
    """Per-consumer wait locations; None retains that consumer's default entry.

    One consumer's wait_at alias never removes another consumer's default wait.
    Raw references are retained here; validation expands each alias separately.
    """
    grouped = {consumer: [] for consumer in handoff["consumer"]}
    for reference in as_list(handoff.get("wait_at", [])):
        consumer = reference.split(".", 1)[0]
        if consumer in grouped:
            grouped[consumer].append(reference)
    return {consumer: tuple(references) if references else None for consumer, references in grouped.items()}


def _substitute(event: EventRef, formals: tuple[str, ...], arguments: tuple[str, ...], path: str) -> EventRef:
    replacements = dict(zip(formals, arguments))

    def expression(value):
        if value is None or not replacements:
            return value

        def replacement(match):
            value = replacements[match[0]]
            return value if re.fullmatch(r"[a-z_][a-z0-9_]*|[+-]?\d+", value) else "(" + value + ")"

        pattern = r"\b(?:" + "|".join(map(re.escape, replacements)) + r")\b"
        return re.sub(pattern, replacement, value)

    operands = tuple(Operand(operand.name, tuple(expression(arg) for arg in operand.args))
                     for operand in event.action.operands)
    action = replace(event.action, path=path, operands=operands)
    return replace(event, path=path, member=expression(event.member), action=action)


def _work_count(definition: dict) -> int | None:
    total = 1
    for part in definition["Decomposition"]["Partition"].values():
        size, tile = literal_integer(part["size"]), literal_integer(part["tile"])
        if size is None or tile is None:
            return None
        total *= (size + tile - 1) // tile
    return total


def _preassigned_range(definition: dict) -> range | None:
    pre = definition["Schedule"]["Worker_assignment"].get("Preassigned")
    if pre is None:
        return None
    start = literal_integer(pre["start"])
    if start is None:
        return None
    if "count" in pre:
        count = literal_integer(pre["count"])
        return range(start, start + count) if count is not None else None
    stride, total = literal_integer(pre["stride"]), _work_count(definition)
    return range(start, total, stride) if stride is not None and total is not None else None


def _preassignment_conflict(first, second):
    left, right = _preassigned_range(first), _preassigned_range(second)
    if left is not None and right is not None:
        return left != right
    a = first["Schedule"]["Worker_assignment"].get("Preassigned")
    b = second["Schedule"]["Worker_assignment"].get("Preassigned")
    # Different count/stride forms can describe the same sequence. Unknown
    # expressions, dynamic assignment and body-defined pairing remain for review.
    return a is not None and b is not None and set(a) == set(b) and concrete_conflict(a, b)


def _coordinates(definition):
    return {name: {"size": part["size"], "tile": part["tile"]}
            for name, part in definition["Decomposition"]["Partition"].items()}


def validate(check: CheckContext, plan: dict[str, Any], path: str) -> None:
    work = plan["DESIGN"]["Work_schedule"]
    pipe = plan["DESIGN"]["Tile_pipeline"]["Step"]
    launches = work["Flow"]["Launch"]
    owner = {step: i for i, launch in enumerate(launches) for step in launch["step"]}
    aliases, families, members = {}, {}, {}
    for step, definition in pipe.items():
        aliases[step], families[step], members[step] = {}, {}, {}
        for key, events in definition.get("Event", {}).items():
            location = field_path(f"{path}.DESIGN.Tile_pipeline.Step.{step}.Event", key)
            formal = parse_action("compute:" + key, path=location).operands[0]
            aliases[step][formal.name] = (formal.args, tuple(parse_event(event, path=f"{location}[{i}]")
                                                            for i, event in enumerate(events)))
        for role, body in definition["Role"].items():
            location = f"{path}.DESIGN.Tile_pipeline.Step.{step}.Role.{role}.Pipe"
            families[step][role] = {declaration.action.signature
                                   for declaration in parse_pipe(body["Pipe"], path=location).actions()}
            members[step][role] = role_member_range(body, work["Step"][step]["Mapping"])

    def event(reference, location, participants=None):
        step, name = reference.split(".", 1)
        if not check.require(step in pipe and step in owner, location,
                             f"event source step {step} is not a submitted step"):
            return None
        if participants is not None:
            check.require(step in participants, location, "event belongs to the wrong Handoff participant")
        call = parse_action("compute:" + name, path=location).operands[0]
        if call.name in {"kernel_done", "launch_ready"}:
            if not check.require(not call.args, location, "built-in launch events have no indices"):
                return None
            return step, call.name, None
        if not check.require(call.name in aliases[step], location, f"undefined Event alias {step}.{call.name}"):
            return None
        formals, definitions = aliases[step][call.name]
        if not check.require(len(formals) == len(call.args), location,
                             f"Event alias {step}.{call.name} expects {len(formals)} arguments, got {len(call.args)}"):
            return None
        expanded = tuple(_substitute(definition, formals, call.args, location) for definition in definitions)
        count = literal_integer(work["Step"][step]["Mapping"]["Worker"]["unit_count"])
        for occurrence in expanded:
            role = occurrence.role
            if not check.require(role in families[step], location, f"Event alias names an unknown role {role}"):
                continue
            check.require(occurrence.action.signature in families[step][role], location,
                          f"Event alias does not match a Pipe action signature in {step}.{role}")
            selection = members[step][role]
            if selection is not None:
                check.require(bool(selection), location, "Event alias selects a role with no worker members")
            member = literal_integer(occurrence.member)
            if member is not None:
                check.require(member >= 0 and (count is None or member < count), location,
                              "Event member is outside Worker.unit_count")
                if selection is not None:
                    check.require(member in selection, location,
                                  f"role {role} has no worker member {member}")
        return step, call.name, expanded

    graph = defaultdict(list)
    for i, launch in enumerate(launches):
        location = f"{path}.DESIGN.Work_schedule.Flow.Launch[{i}]"
        for reference, rp in indexed_values(launch.get("start_after", []), location + ".start_after"):
            resolved = event(reference, rp)
            if resolved is not None:
                source, _, _ = resolved
                if check.require(owner[source] != i, rp, "a launch cannot wait for an event from its own submission before starting"):
                    # All three event forms originate in this source submission;
                    # renaming the event cannot remove the launch prerequisite.
                    graph[i].append(owner[source])
        if len(launch["step"]) > 1 and launch["order"] != "concurrent":
            first = work["Step"][launch["step"][0]]
            storages = set()
            for step in launch["step"]:
                current = work["Step"][step]
                for field in ("Worker", "Worker_count", "Placement"):
                    a, b = first["Mapping"].get(field), current["Mapping"].get(field)
                    check.require(not concrete_conflict(a, b), location,
                                  "sequential steps have incompatible shared-worker mappings")
                for name in pipe[step].get("Storage", {}):
                    check.require(name not in storages, f"{path}.DESIGN.Tile_pipeline.Step.{step}.Storage.{name}",
                                  f"shared-worker Storage {name} must be declared once")
                    storages.add(name)
                if launch["order"] == "per_worktile":
                    check.require(not concrete_conflict(_coordinates(first), _coordinates(current)), location,
                                  "per_worktile requires matching work coordinates")
                    check.require(not concrete_conflict(first["Schedule"].get("Worktile_order"),
                                                        current["Schedule"].get("Worktile_order")), location,
                                  "per_worktile requires matching logical work order")
                    check.require(not _preassignment_conflict(first, current), location,
                                  "per_worktile requires matching preassigned work sequences")
    check.acyclic(graph, path + ".DESIGN.Work_schedule.Flow.Launch",
                  {i: f"{path}.DESIGN.Work_schedule.Flow.Launch[{i}].start_after" for i in graph})

    for name, handoff in work["Flow"].get("Handoff", {}).items():
        location = field_path(path + ".DESIGN.Work_schedule.Flow.Handoff", name)
        producer, consumers = handoff["producer"], handoff["consumer"]
        ready = []
        for reference, rp in indexed_values(handoff.get("ready", []), location + ".ready"):
            resolved = event(reference, rp, {producer})
            if resolved is not None:
                ready.append(resolved)
        for reference, rp in indexed_values(handoff.get("wait_at", []), location + ".wait_at"):
            resolved = event(reference, rp, set(consumers))
            if resolved is None:
                continue
            _, _, occurrences = resolved
            if not check.require(occurrences is not None, rp, "wait_at must name a local Event alias, not a built-in launch event"):
                continue
            for occurrence in occurrences:
                verb = occurrence.action.verb
                valid = ((occurrence.time == "done" and verb in {"await", "acquire"}) or
                         (occurrence.time == "issue" and verb in {"compute", "matmul", "load", "store"}))
                check.require(valid, rp, "wait_at targets a waiting action's done or a consuming action's issue")

        for consumer, waits in consumer_waits(handoff).items():
            submission = owner[consumer]
            launch = launches[submission]
            if waits is None and submission == owner[producer] and launch.get("order") in {"per_worktile", "per_worker", "all"}:
                nonempty = _work_count(work["Step"][consumer])
                if nonempty is not None and nonempty > 0:
                    check.require(launch["step"].index(producer) < launch["step"].index(consumer), location,
                                  f"default entry wait for consumer {consumer} cannot depend on a later sequential step")
            if submission == owner[producer] and any(kind == "kernel_done" for _, kind, _ in ready):
                nonempty = _work_count(work["Step"][consumer])
                if waits is not None or nonempty is not None and nonempty > 0:
                    check.require(False, location + ".ready",
                                  f"consumer {consumer} cannot wait for completion of its own kernel")

        if handoff["storage"] != "global":
            _, _, storage_step, _, storage_name = handoff["storage"].split(".")
            exists = storage_step in pipe and storage_name in pipe.get(storage_step, {}).get("Storage", {})
            if check.require(exists, location + ".storage", "undefined Tile_pipeline Storage"):
                participants = [producer, *consumers, storage_step]
                check.require(all(owner[step] == owner[producer] for step in participants), location + ".storage",
                              "cross-kernel handoff cannot preserve a worker-local Storage identity")
                # Concurrent workers keep distinct identities. Whether a declared
                # region is accessible across them requires the actual storage,
                # visibility and lifetime guarantees; concurrency alone is not a
                # blanket rejection or proof of accessibility.

    for step, definition in work["Step"].items():
        location = f"{path}.DESIGN.Work_schedule.Step.{step}"
        partitions = definition["Decomposition"]["Partition"]
        for name, part in partitions.items():
            if "range" in part:
                rp = field_path(location + ".Decomposition.Partition", name) + ".range"
                check.interval(part["range"], rp)
                size = literal_integer(part["size"])
                lo, hi = map(literal_integer, part["range"])
                if size is not None:
                    check.require((lo is None or size >= lo) and (hi is None or size <= hi), rp,
                                  "literal Partition.size lies outside its declared range")
        for i, member in enumerate(definition["Mapping"].get("Member", [])):
            for dimension, bounds in member.items():
                check.interval(bounds, field_path(f"{location}.Mapping.Member[{i}]", dimension))
        assignment = definition["Schedule"]["Worker_assignment"]
        pre = assignment.get("Preassigned")
        if pre is not None and "count" in pre:
            start, count, total = literal_integer(pre["start"]), literal_integer(pre["count"]), _work_count(definition)
            if start is not None and count is not None and total is not None and count > 0:
                check.require(start + count <= total, location + ".Schedule.Worker_assignment.Preassigned",
                              "the explicit preassigned sequence exceeds the worktile range")
        reduction = definition["Decomposition"]["Target"].get("Tile_reduction")
        if reduction is not None:
            dimensions = {name for definition in work["Step"].values()
                          for name in definition["Decomposition"]["Partition"]}
            check.require(reduction["dimension"] in dimensions, location + ".Decomposition.Target.Tile_reduction.dimension",
                          "Tile_reduction must name a dimension partitioned by a producing step")
