"""Ordered for/if expansion and explicit local event dependencies.

A graph is one representative worker in one Step. There are no inferred noun
or load/store data edges, implicit drains, or cross-step execution simulation.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from ..diagnostics import field_path
from ..pipeline_syntax import (
    ActionDeclaration, EventRef, ForLoop, IfBlock, IndexScope, PipelineSyntaxError,
    domain_choices, domain_queries, parse_action, parse_event, parse_members,
    parse_pipe, parse_waiting,
)
from ..rules.common import EXECUTION_MODES
from .actions import action_variables, arguments, instantiate
from .expressions import Boundary, Domains, EmptyCoordinate, LIMIT, merge_coordinates, referenced_domains, tree, variables
from .model import ExpansionError, GraphError, Operation, StepGraph
from .windows import IndexWindows


def members(info: dict, worker: dict, domains: Domains) -> tuple[int, ...]:
    count = domains.integer(worker["unit_count"], {})
    if count < 1:
        raise GraphError("Mapping.Worker.unit_count must be positive")
    if count > 64:
        raise ExpansionError("worker member count exceeds the graph tool limit (64)")
    pieces = parse_members(info["Member"])
    if pieces is None:
        raise GraphError("invalid role member path: " + info["Member"])
    selected = tuple(range(count))
    for i, (unit, lower, upper) in enumerate(pieces):
        lo = domains.integer(lower, {})
        hi = domains.integer(upper, {}) if upper is not None else lo + 1
        if not 0 <= lo < hi:
            raise GraphError("role member interval must be nonnegative and nonempty: " + info["Member"])
        if i == 0 and unit == worker["unit"]:
            if hi > count:
                raise GraphError("role member selection exceeds Mapping.Worker.unit_count: " + info["Member"])
            selected = tuple(range(lo, hi))
    return selected


class Builder:
    def __init__(self, step: str, definition: dict, worker: dict, parameters: dict, worktiles: int,
                 *, dynamic_worktiles=False):
        self.graph, self.definition = StepGraph(step), definition
        self.path = f"DESIGN.Tile_pipeline.Step.{step}"
        self.scope = IndexScope({name: entry["definition"] for name, entry in definition.get("Index", {}).items()})
        self.domains = Domains(self.scope, parameters, worktiles, dynamic_worktiles=dynamic_worktiles)
        self.domain_cache = {id(self.scope): self.domains}
        self.pipes = {role: parse_pipe(body["Pipe"], path=f"{self.path}.Role.{role}.Pipe", scope=self.scope)
                      for role, body in definition["Role"].items()}
        for pipe in self.pipes.values():
            for node in pipe.controls():
                if isinstance(node, ForLoop) and node.spec.bounds is not None:
                    outer = self.domains_for(node.scope)
                    inner = self.domains_for(node.body.scope)
                    inner.range_scopes = {**outer.range_scopes, node.spec.name: (outer, node.bound)}
        self.members = {role: members(body["Info"], worker, self.domains)
                        for role, body in definition["Role"].items()}
        self.lookup, self.families = {}, defaultdict(dict)
        self.windows, self.steps = IndexWindows(), 0
        self.present = defaultdict(set)

    def tick(self):
        self.steps += 1
        if self.steps > 100000:
            raise ExpansionError("pipe/rule expansion limit exceeded (100000); use a smaller sample")

    def domains_for(self, scope: IndexScope) -> Domains:
        if id(scope) not in self.domain_cache:
            self.domain_cache[id(scope)] = Domains(scope, self.domains.parameters, self.domains.worktiles,
                                                   dynamic_worktiles=self.domains.dynamic_worktiles)
        return self.domain_cache[id(scope)]

    def declarations(self):
        for role, pipe in self.pipes.items():
            using = self.definition["Role"][role]["Info"]["Using_verb"]
            for declaration in pipe.actions():
                action = declaration.action
                contract = using.get(action.verb, {}).get(action.noun)
                if contract is None:
                    raise GraphError(f"{declaration.path}: undeclared action family {action.verb}:{action.noun}")
                if contract["execution_mode"] not in EXECUTION_MODES:
                    raise GraphError(f"{declaration.path}: invalid noun execution_mode {contract['execution_mode']!r}")
                family = self.families[role].setdefault(action.family, [])
                if family and family[0].action.signature != action.signature:
                    raise GraphError(f"{declaration.path}: action family has inconsistent argument counts")
                family.append(declaration)
                self.windows.register(role, declaration, self.domains_for(declaration.scope))
            if self.domains.dynamic_worktiles:
                for node in pipe.walk():
                    if isinstance(node, ActionDeclaration):
                        expressions, local = [arg for _, arg in arguments(node.action)], self.domains_for(node.scope)
                    elif isinstance(node, IfBlock):
                        expressions, local = [node.condition], self.domains_for(node.body.scope)
                    else:
                        expressions, local = [node.spec.text], self.domains_for(node.scope)
                    for expression in expressions:
                        for _, name in domain_queries(expression):
                            if local.has_worktiles(name):
                                raise GraphError(f"{node.path}: count/last on a dynamically refilled worktile domain is undefined")

    def expand_pipe(self, pipe, env):
        domains = self.domains_for(pipe.scope)
        for item in pipe.items:
            self.tick()
            try:
                if isinstance(item, ActionDeclaration):
                    yield instantiate(item.action, domains, env), dict(env), item
                elif isinstance(item, IfBlock):
                    if domains.predicate(item.condition, env):
                        yield from self.expand_pipe(item.body, env)
                elif isinstance(item, ForLoop):
                    inner = self.domains_for(item.body.scope)
                    for values in inner.select(item.spec, env):
                        yield from self.expand_pipe(item.body, merge_coordinates(env, values))
            except GraphError as error:
                if str(error).startswith("DESIGN."):
                    raise
                raise type(error)(f"{item.path}: {error}") from error

    def add_operations(self):
        self.declarations()
        for role, body in self.definition["Role"].items():
            expanded = list(self.expand_pipe(self.pipes[role], {}))
            for member in self.members[role]:
                previous = ordered = None
                for action, context, declaration in expanded:
                    family = declaration.action.family
                    mode = body["Info"]["Using_verb"][family[0]][family[1]]["execution_mode"]
                    identity = f"{self.graph.step}/{role}[{member}]/{action}"
                    if identity in self.graph.operations:
                        first = self.graph.operations[identity]
                        raise GraphError(f"{declaration.path}: duplicate action instance {identity}; first declared at {first.path}")
                    op = Operation(identity, role, member, action, mode, declaration.path, context)
                    self.graph.operations[identity] = op
                    self.lookup[(role, member, action)] = op
                    self.present[role].add(family)
                    if len(self.graph.operations) > LIMIT:
                        raise ExpansionError(f"action limit exceeded ({LIMIT}); use a smaller sample")
                    self.graph.add(op.event("issue"), op.event("done"), "execute", "this action's issue precedes its done", op.path)
                    if previous is not None:
                        # Every issue is ordered. A serial predecessor also
                        # delays this issue until that predecessor alone is done.
                        self.graph.add(previous.event("issue"), op.event("issue"), "pipe", "Pipe issue order", op.path)
                        if previous.execution_mode == "serial":
                            self.graph.add(previous.event("done"), op.event("issue"), "pipe", "previous serial action completes before the next issue", op.path)
                    if mode == "async_ordered":
                        if ordered is not None:
                            self.graph.add(ordered.event("done"), op.event("done"), "ordered", "async_ordered cumulative completion across this role/member", op.path)
                        ordered = op
                    previous = op

    def reference(self, ref: EventRef, current: str | None, *, target=False) -> EventRef:
        role = ref.role or current
        if role not in self.members:
            raise GraphError(f"{ref.path}: unknown event role {role}")
        if target and role != current:
            raise GraphError(f"{ref.path}: Waiting target must belong to {current}")
        declarations = self.families[role].get(ref.action.family, [])
        if not declarations:
            raise GraphError(f"{ref.path}: unknown action family {role}/{ref.action.verb}:{ref.action.noun}")
        if ref.action.signature != declarations[0].action.signature:
            raise GraphError(f"{ref.path}: event argument counts do not match {role}/{ref.action.verb}:{ref.action.noun}")
        return replace(ref, role=role)

    def selected_members(self, ref: EventRef, member: int | None, env: dict, domains: Domains):
        if ref.member == "*":
            return self.members[ref.role]
        selected = domains.integer(ref.member, env) if ref.member is not None else member
        if selected is None:
            raise GraphError(f"{ref.path}: Event aliases must explicitly select a member")
        if selected not in self.members[ref.role]:
            raise GraphError(f"{ref.path}: {ref.role}[{selected}] has no role instance; omitted members mean the current worker member")
        return (selected,)

    def rule_domains(self, references, *, members_only=False) -> Domains:
        scopes = [declaration.scope for ref in references for declaration in self.families[ref.role][ref.action.family]]
        expressions = [] if members_only else [arg for ref in references for _, arg in arguments(ref.action)]
        expressions += [ref.member for ref in references if ref.member not in (None, "*")]
        pending = list(set().union(*(referenced_domains(tree(expression)) for expression in expressions)))
        definitions, visited, range_scopes = dict(self.scope.definitions), set(), {}
        while pending:
            name = pending.pop()
            if name in visited:
                continue
            visited.add(name)
            choices = domain_choices(scopes, name)
            if not choices:
                continue  # An ordinary parameter must come from the sample.
            if len(choices) != 1:
                raise ExpansionError(f"ambiguous Waiting index {name}; referenced declaration scopes disagree")
            definition = definitions[name] = choices[0]
            for scope in scopes:
                if name in self.domains_for(scope).range_scopes:
                    range_scopes[name] = self.domains_for(scope).range_scopes[name]
                    break
            if isinstance(definition, dict):
                pending.extend(definition["flatten"])
            elif isinstance(definition, list):
                pending.extend(set().union(*(referenced_domains(tree(bound)) for bound in definition)))
        domains = Domains(definitions, self.domains.parameters, self.domains.worktiles,
                          dynamic_worktiles=self.domains.dynamic_worktiles)
        domains.range_scopes = range_scopes
        return domains

    def precheck_members(self, refs, current, domains):
        """Member errors are independent of whether a target branch executes."""
        target = refs[0]
        if target.member not in (None, "*") and not variables(tree(target.member)) & domains.definitions.keys():
            current_members = self.selected_members(target, None, {}, domains)
        else:
            current_members = self.members[current]
        for ref in refs:
            if ref.member not in (None, "*") and variables(tree(ref.member)) & domains.definitions.keys():
                continue
            for member in current_members:
                self.selected_members(ref, member, {}, domains)

    def resolve(self, ref, concrete, selected, env, domains, *, target=False):
        events = []
        for member in selected:
            op = self.lookup.get((ref.role, member, concrete))
            if op is None:
                self.windows.missing(ref.role, ref.action, concrete, domains, env, target=target)
                return []
            events.append(op.event(ref.time))
        return events

    def explicit_edges(self):
        for role, body in self.definition["Role"].items():
            for kind, rules in body.get("Waiting", {}).items():
                for i, text in enumerate(rules):
                    path = f"{self.path}.Role.{role}.Waiting.{kind}[{i}]"
                    rule = parse_waiting(text, path=path)
                    refs = [self.reference(rule.target, role, target=True),
                            *(self.reference(ref, role) for ref in rule.predecessors)]
                    # Validate explicit/default member addressing even if this
                    # entire target family is absent in the selected sample.
                    inactive = refs[0].action.family not in self.present[role]
                    # An absent target has no Waiting occurrence. Keep member
                    # addressing checks, without trying to interpret unused
                    # mathematical action arguments in the finite evaluator.
                    domains = self.rule_domains(refs, members_only=inactive)
                    self.precheck_members(refs, role, domains)
                    if inactive:
                        continue
                    names = set().union(*(action_variables(ref.action, domains) for ref in refs))
                    for ref in refs:
                        if ref.member not in (None, "*"):
                            names.update(variables(tree(ref.member)) & domains.definitions.keys())
                    try:
                        for env in domains.contexts(names):
                            targets = self.members[role] if refs[0].member in (None, "*") else self.selected_members(refs[0], None, env, domains)
                            for member in targets:
                                self.tick()
                                selections = [self.selected_members(ref, member, env, domains) for ref in refs]
                                # The target's [*] means one constraint for each
                                # target member, with omissions resolved at that
                                # member; it is not a Cartesian product of roles.
                                selections[0] = (member,)
                                target_ref = refs[0]
                                target_boundary = None
                                try:
                                    concrete = instantiate(target_ref.action, domains, env)
                                    target_events = self.resolve(target_ref, concrete, selections[0], env, domains, target=True)
                                except EmptyCoordinate as error:
                                    if self.windows.empty_target(target_ref.role, target_ref.action, error):
                                        continue
                                    raise GraphError(str(error)) from error
                                except Boundary as error:
                                    self.windows.validate_fixed(target_ref.role, target_ref.action, domains, env)
                                    target_boundary, target_events = str(error), []
                                if not target_events and target_boundary is None:
                                    continue  # The target's own control excludes this occurrence.
                                if target_boundary is not None:
                                    # A shifted target cannot hide a bad fixed
                                    # counterpart, while a nonexistent target
                                    # never demands evaluation of variable work.
                                    for ref, selection in zip(refs[1:], selections[1:]):
                                        self.windows.validate_fixed(ref.role, ref.action, domains, env)
                                        if not action_variables(ref.action, domains):
                                            concrete = instantiate(ref.action, domains, env)
                                            self.resolve(ref, concrete, selection, env, domains)
                                    self.graph.boundaries.append({"path": path, "rule": text, "context": dict(env),
                                                                  "member": member, "reason": target_boundary})
                                    continue
                                for ref, selection in zip(refs[1:], selections[1:]):
                                    try:
                                        concrete = instantiate(ref.action, domains, env)
                                        predecessors = self.resolve(ref, concrete, selection, env, domains)
                                    except Boundary as error:
                                        self.windows.validate_fixed(ref.role, ref.action, domains, env)
                                        self.graph.boundaries.append({"path": path, "rule": text, "predecessor": ref.text,
                                                                      "context": dict(env), "member": member, "reason": str(error)})
                                        continue
                                    # Each predecessor is a constraint. A
                                    # boundary in one never removes the other
                                    # valid predecessors of this same target.
                                    for source in predecessors:
                                        for target in target_events:
                                            self.graph.add(source, target, kind, text, path)
                    except GraphError as error:
                        if str(error).startswith("DESIGN."):
                            raise
                        raise type(error)(f"{path}: {error}") from error

    def event_aliases(self, calls):
        aliases = {}
        for name, values in self.definition.get("Event", {}).items():
            path = field_path(self.path + ".Event", name)
            formal = parse_action("compute:" + name, path=path).operands[0]
            refs = [self.reference(parse_event(value, path=f"{path}[{i}]"), None) for i, value in enumerate(values)]
            if any(ref.member is None for ref in refs):
                raise GraphError(path + ": Event aliases must explicitly select ROLE and member")
            aliases[formal.name] = formal.args, refs
        # Event formals are bound by calls, never by a coincidentally named
        # loop or the first referenced action's declaration window. Unused
        # aliases need no finite values and do not create event occurrences.
        for name, path in calls:
            try:
                call = parse_action("compute:" + name, path=path).operands[0]
                if call.name in {"kernel_done", "launch_ready"}:
                    if call.args:
                        raise GraphError("built-in launch events have no indices")
                    continue
                if call.name not in aliases:
                    raise GraphError(f"undefined Event alias {call.name}")
                formals, refs = aliases[call.name]
                if len(formals) != len(call.args):
                    raise GraphError(f"Event alias {call.name} expects {len(formals)} arguments, got {len(call.args)}")
                values = self.domains.integers(call.args, {})
                env = dict(zip(formals, values))
                alias = call.name + ("(" + ",".join(map(str, values)) + ")" if values else "")
                events = []
                for ref in refs:
                    self.tick()
                    selection = self.selected_members(ref, None, env, self.domains)
                    concrete = instantiate(ref.action, self.domains, env)
                    # A called alias requires every listed event. A false
                    # control condition cannot remove this explicit reference.
                    for member in selection:
                        op = self.lookup.get((ref.role, member, concrete))
                        if op is None:
                            raise GraphError(f"Event {alias} names an absent event: {ref.role}[{member}]/{concrete}@{ref.time}")
                        events.append(op.event(ref.time))
                self.graph.events[alias] = sorted(set(events))
            except GraphError as error:
                if str(error).startswith("DESIGN."):
                    raise
                raise type(error)(f"{path}: {error}") from error

    def build(self, event_calls):
        self.add_operations()
        self.explicit_edges()
        self.event_aliases(event_calls)
        self.graph.check()
        return self.graph


def build_step(step: str, definition: dict, worker: dict, parameters: dict, worktiles=1,
               *, dynamic_worktiles=False, event_calls=()) -> StepGraph:
    """Expand one worker; event_calls contains (local alias call, source path).

    Only supplied calls are evaluated. Event declarations remain templates;
    callers must provide sample values for any free invocation arguments.
    """
    try:
        return Builder(step, definition, worker, parameters, worktiles,
                       dynamic_worktiles=dynamic_worktiles).build(event_calls)
    except PipelineSyntaxError as error:
        raise GraphError(str(error)) from error
    except EmptyCoordinate as error:
        message = str(error) if str(error).startswith("DESIGN.") else f"DESIGN.Tile_pipeline.Step.{step}: {error}"
        raise GraphError(message) from error
    except GraphError as error:
        prefix = f"DESIGN.Tile_pipeline.Step.{step}"
        message = str(error) if str(error).startswith("DESIGN.") else f"{prefix}: {error}"
        raise type(error)(message) from error
    except RecursionError as error:
        raise ExpansionError(f"DESIGN.Tile_pipeline.Step.{step}: nesting exceeds the graph tool's recursion limit") from error
