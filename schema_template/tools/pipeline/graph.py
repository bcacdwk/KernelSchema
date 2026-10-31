"""Expand roles, resolve event references and check a phase's finite graph.

The finite graph infers data only for mma operands, store, wait and
release. Custom verbs have explicit edges: their prose is never executed.
"""
from __future__ import annotations

import re
from collections import defaultdict

from ..pipeline_syntax import IndexScope, ScopedPipe, domain_choices, parse_action, parse_event
from ..rules.common import EXECUTION_MODES
from .actions import action_variables, instantiate, operands
from .expressions import Boundary, Domains, LIMIT, referenced_domains, tree
from .model import BoundEvent, ExpansionError, GraphError, Operation, PhaseGraph
from .windows import IndexWindows


def members(info: dict, worker: dict, domains: Domains) -> tuple[int, ...]:
    count = domains.integer(worker["unit_count"], {})
    if count < 1:
        raise GraphError("worker member count must be positive")
    if count > 64:
        raise ExpansionError(f"worker member count {count} exceeds the graph tool limit (64)")
    match = re.match(r"([^\[\]]+)\[(\d+)(?::(\d+))?\]", info["member"])
    if not match:
        raise GraphError(f"invalid member path: {info['member']}")
    if match[1] != worker["unit"]:
        return tuple(range(count))
    lo, hi = int(match[2]), int(match[3]) if match[3] else int(match[2]) + 1
    if not 0 <= lo < hi <= count:
        raise GraphError(f"role member selection exceeds worker count: {info['member']}")
    return tuple(range(lo, hi))


class Builder:
    def __init__(self, pid: str, phase: dict, worker: dict, parameters: dict, worktiles: int):
        self.graph = PhaseGraph(pid)
        self.phase = phase
        scope = IndexScope(phase.get("domain", {}))
        self.domains = Domains(scope, parameters, worktiles)
        self.pipes = {role: ScopedPipe.read(body["pipe"], scope,
                      f"design.tile_pipeline.phase.{pid}.role.{role}.pipe")
                      for role, body in phase["role"].items()}
        self.domain_cache = {id(scope): self.domains}
        self.members = {role: members(body["info"], worker, self.domains)
                        for role, body in phase["role"].items()}
        self.all_members = tuple(range(self.domains.integer(worker["unit_count"], {})))
        self.lookup = {}
        self.families = defaultdict(dict)
        self.windows = IndexWindows()
        self.steps = 0

    def domains_for(self, scope: IndexScope) -> Domains:
        if id(scope) not in self.domain_cache:
            self.domain_cache[id(scope)] = Domains(scope, self.domains.parameters, self.domains.worktiles)
        return self.domain_cache[id(scope)]

    def expand_pipe(self, block: ScopedPipe, env: dict):
        domains = self.domains_for(block.scope)
        for region, items in block.regions.items():
            contexts = [env]
            if region == "steady" and block.repeat:
                try:
                    contexts = [dict(env, **item) for item in domains.select(block.repeat, env)]
                except GraphError as error:
                    raise type(error)(f"{block.path}.repeat: {error}") from error
            for context in contexts:
                self.steps += 1
                if self.steps > 100000:
                    raise ExpansionError("pipe expansion step limit exceeded (100000); use a smaller sample")
                for item in items:
                    if isinstance(item, ScopedPipe):
                        yield from self.expand_pipe(item, context)
                    else:
                        try:
                            yield instantiate(item.text, domains, context), context, item.path
                        except GraphError as error:
                            raise type(error)(f"{item.path}: {error}") from error

    def add_operations(self) -> None:
        for role, body in self.phase["role"].items():
            pipe = self.pipes[role]
            for action in pipe.actions():
                self.families[role].setdefault(parse_action(action.text), []).append(action.scope)
                self.windows.register(role, action.text, self.domains_for(action.scope))
            info = body["info"]
            expanded = list(self.expand_pipe(pipe, {}))
            for member in self.members[role]:
                previous = ordered = None
                for action, context, location in expanded:
                    execution_mode = info["execution_mode"]
                    contract = info.get("verb", {}).get(action.split(":", 1)[0])
                    if isinstance(contract, dict):
                        execution_mode = contract["execution_mode"]
                    if execution_mode not in EXECUTION_MODES:
                        raise GraphError(f"{location}: unsupported execution_mode {execution_mode!r}; "
                                         f"expected one of {', '.join(EXECUTION_MODES)}")
                    identity = f"{self.graph.phase}/{role}[{member}]/{action}"
                    if identity in self.graph.operations:
                        raise GraphError(f"{location}: duplicate action {identity}")
                    op = Operation(identity, role, member, action, execution_mode, location, context)
                    self.graph.operations[identity] = op
                    self.lookup[(role, member, action)] = op
                    if len(self.graph.operations) > LIMIT:
                        raise ExpansionError(f"operation limit exceeded ({LIMIT}); use a smaller sample")
                    self.graph.add(op.event("issue"), op.event("done"), "execute", location)
                    if previous:
                        event = "done" if previous.execution_mode == "serial" else "issue"
                        self.graph.add(previous.event(event), op.event("issue"), "program", location)
                    if execution_mode == "async_ordered_completion":
                        if ordered:
                            self.graph.add(ordered.event("done"), op.event("done"), "completion", location)
                        ordered = op
                    previous = op

    def selected_members(self, current: str, member: int, target: str, selector: str | None):
        if selector == "*":
            return self.members[target]
        if selector is not None:
            value = int(selector)
            if value not in self.members[target]:
                raise GraphError(f"{target}[{value}] has no role instance")
            return (value,)
        if target == current or self.members[current] == self.members[target] == self.all_members:
            if member not in self.members[target]:
                raise GraphError(f"{target}[{member}] has no role instance")
            return (member,)
        return self.members[target]

    def parse_reference(self, reference: str, current: str | None, waiting=False):
        role, selector, action = parse_event(reference)
        if role is None:
            role = current
        if role is None:
            matches = [name for name, families in self.families.items() if parse_action(action) in families]
            if len(matches) != 1:
                raise GraphError(f"milestone needs an unambiguous role: {reference}")
            role = matches[0]
        if role not in self.members or parse_action(action) not in self.families[role]:
            raise GraphError(f"unknown action family: {role}/{action}")
        if waiting and role != current:
            raise GraphError(f"waiting action must belong to {current}: {reference}")
        match = re.search(r"@(issue|done)$", reference)
        return role, selector, action, match[1] if match else ("issue" if waiting else "done")

    def bind_reference(self, parsed, current: str, member: int, env: dict, domains: Domains) -> BoundEvent:
        role, selector, template, event = parsed
        selected = self.selected_members(current, member, role, selector)
        concrete = instantiate(template, domains, env)
        return BoundEvent(role, selected, template, concrete, event, not action_variables(template, domains))

    def resolve(self, reference: BoundEvent, env: dict, domains: Domains) -> list[str]:
        result = []
        for target in reference.members:
            op = self.lookup.get((reference.role, target, reference.action))
            if op is None:
                identity = f"{reference.role}[{target}]/{reference.action}"
                if self.windows.outside(reference.role, reference.template, reference.action, domains, env):
                    raise Boundary(f"{identity} outside its declared action index window")
                raise GraphError(f"dangling event: {identity}@{reference.event}")
            result.append(op.event(reference.event))
        return result

    def resolve_edge(self, left, right, role: str, member: int, env: dict, domains: Domains):
        """Validate both references before looking up the variable edge instance.

        Member selections, coordinates and fixed events must remain valid even
        when a variable shift makes this particular edge inactive.
        """
        endpoints, boundaries = [], []
        for parsed in (left, right):
            try:
                reference = self.bind_reference(parsed, role, member, env, domains)
                events = self.resolve(reference, env, domains) if reference.fixed else None
                endpoints.append((reference, events))
            except Boundary as error:
                boundaries.append(error)
        if boundaries:
            raise boundaries[0]
        return [events if events is not None else self.resolve(reference, env, domains)
                for reference, events in endpoints]

    def rule_domains(self, references) -> Domains:
        definitions = {}
        for reference in references:
            scopes = self.families[reference[0]][parse_action(reference[2])]
            pending = list(set().union(*(referenced_domains(tree(arg))
                           for _, args in operands(reference[2])[1] for arg in args)))
            visited = set()
            while pending:
                name = pending.pop()
                if name in visited:
                    continue
                visited.add(name)
                choices = domain_choices(scopes, name)
                if not choices:
                    continue
                if len(choices) > 1 or name in definitions and definitions[name] != choices[0]:
                    raise ExpansionError(f"ambiguous local edge index {name}; referenced action scopes disagree")
                definition = definitions[name] = choices[0]
                if isinstance(definition, dict):
                    pending.extend(definition["flatten"])
                elif isinstance(definition, list):
                    pending.extend(set().union(*(referenced_domains(tree(str(bound))) for bound in definition)))
        return Domains(definitions, self.domains.parameters, self.domains.worktiles)

    def explicit_edges(self) -> None:
        prepared = []
        for role, body in self.phase["role"].items():
            for kind, rules in body.get("waits_for", {}).items():
                for target, sources in rules.items():
                    left = self.parse_reference(target, role, waiting=True)
                    for source in sources:
                        right = self.parse_reference(source, role)
                        domains = self.rule_domains([left, right])
                        for ref in (left, right):
                            self.windows.register(ref[0], ref[2], domains, reference=True)
                        prepared.append((role, kind, target, source, left, right, domains))
        for role, kind, target, source, left, right, domains in prepared:
            names = action_variables(left[2], domains) | action_variables(right[2], domains)
            reason = f"{role}.waits_for.{kind}[{target}] <- {source}"
            for index, env in enumerate(domains.contexts(names)):
                if index >= LIMIT:
                    raise ExpansionError(f"rule expansion limit exceeded: {reason}")
                waiting_members = self.members[role]
                if left[1] not in (None, "*"):
                    waiting_members = self.selected_members(role, 0, role, left[1])
                for member in waiting_members:
                    self.steps += 1
                    if self.steps > 100000:
                        raise ExpansionError("combined expansion step limit exceeded (100000); use a smaller sample")
                    try:
                        local_left = (left[0], None, left[2], left[3])
                        targets, predecessors = self.resolve_edge(local_left, right, role, member, env, domains)
                    except Boundary as error:
                        self.graph.boundaries.append({"rule": reason, "context": env, "member": member,
                                                      "reason": str(error)})
                        continue
                    except GraphError as error:
                        raise type(error)(f"{reason}, context={env}: {error}") from error
                    for a in predecessors:
                        for b in targets:
                            self.graph.add(a, b, kind, reason)

    def default_edges(self) -> None:
        producers = defaultdict(list)
        for op in self.graph.operations.values():
            verb, objects = operands(op.action)
            if verb in {"load", "publish"}:
                producers[repr(objects[0])].append(op)
        for op in self.graph.operations.values():
            verb, objects = operands(op.action)
            if verb not in {"mma", "store", "wait", "release"}:
                continue
            for obj in objects:
                candidates = [p for p in producers[repr(obj)] if p.id != op.id]
                origins = {(p.role, p.action) for p in candidates}
                if len(origins) != 1:
                    continue
                source_role, _ = next(iter(origins))
                selected = self.selected_members(op.role, op.member, source_role, None)
                for source in candidates:
                    if source.member in selected:
                        self.graph.add(source.event("done"), op.event("issue"), "data",
                                       "default: unique load/publish of " + obj[0])

    def milestones(self) -> None:
        for name, references in self.phase.get("milestone", {}).items():
            for reference in references:
                parsed = self.parse_reference(reference, None)
                domains = self.rule_domains([parsed])
                names = action_variables(parsed[2], domains) | action_variables("wait:" + name, domains)
                for env in domains.contexts(names):
                    identity = instantiate("wait:" + name, domains, env).split(":", 1)[1]
                    result = self.graph.milestones.setdefault(identity, [])
                    for member in self.members[parsed[0]]:
                        reference = self.bind_reference(parsed, parsed[0], member, env, domains)
                        result.extend(self.resolve(reference, env, domains))
        self.graph.milestones = {name: sorted(set(events)) for name, events in self.graph.milestones.items()}

    def build(self) -> PhaseGraph:
        self.add_operations()
        self.explicit_edges()
        self.default_edges()
        self.milestones()
        self.graph.check()
        return self.graph


def build_phase(pid: str, phase: dict, worker: dict, parameters: dict, worktiles=2) -> PhaseGraph:
    try:
        return Builder(pid, phase, worker, parameters, worktiles).build()
    except GraphError as error:
        prefix = f"design.tile_pipeline.phase.{pid}"
        message = str(error) if str(error).startswith(prefix) else f"{prefix}: {error}"
        raise type(error)(message) from error
    except RecursionError as error:
        raise ExpansionError(f"design.tile_pipeline.phase.{pid}: nesting exceeds the graph tool's recursion limit") from error
