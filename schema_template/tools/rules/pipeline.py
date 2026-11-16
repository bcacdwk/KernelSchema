"""Conservative local action, index, member and storage relations.

The shared syntax records are authoritative. Mathematical expressions remain
open text: recognizable index uses are checked, but this pass never executes a
pipe, guesses a noun's inputs, or treats the finite evaluator as schema syntax.
"""
from __future__ import annotations

from math import prod
from typing import Any

from ..diagnostics import field_path
from ..pipeline_syntax import (
    ActionDeclaration, ForLoop, IfBlock, IndexScope, PipelineSyntaxError, Slice,
    domain_choices, domain_queries, expression_names, free_action_indices,
    literal_integer, parse_action, parse_event, parse_members, parse_pipe,
    parse_waiting,
)
from .common import CheckContext, indexed_values, role_member_range

# Only formats with an unambiguous byte-sized capacity unit are listed. An
# environment-defined or packed dtype stays valid without a guessed byte size.
DTYPE_BYTES = {
    "bool": 1, "int8": 1, "uint8": 1, "i8": 1, "u8": 1,
    "fp8_e4m3fn": 1, "fp8_e5m2": 1, "fp8_e4m3fnuz": 1, "fp8_e5m2fnuz": 1,
    "int16": 2, "uint16": 2, "i16": 2, "u16": 2,
    "fp16": 2, "float16": 2, "half": 2, "bf16": 2, "bfloat16": 2,
    "int32": 4, "uint32": 4, "i32": 4, "u32": 4, "fp32": 4, "float32": 4,
    "int64": 8, "uint64": 8, "i64": 8, "u64": 8, "fp64": 8, "float64": 8,
}


def _dependencies(definition: Any, known: dict) -> set[str]:
    if isinstance(definition, dict):
        return set(definition["flatten"])
    if isinstance(definition, list):
        return set().union(*(expression_names(str(value)) |
                             {name for _, name in domain_queries(str(value))}
                             for value in definition)) & known.keys()
    return set()


def _dynamic(name: str, definitions: dict, seen=frozenset()) -> bool:
    if name in seen or name not in definitions:
        return False
    definition = definitions[name]
    return definition == "worktile" or any(
        _dynamic(child, definitions, seen | {name})
        for child in _dependencies(definition, definitions))


def _empty(name: str, definitions: dict, seen=frozenset()) -> bool:
    if name in seen:
        return False
    definition = definitions.get(name)
    if isinstance(definition, list):
        lo, hi = map(literal_integer, definition)
        return lo is not None and hi is not None and lo == hi
    if isinstance(definition, dict):
        return any(_empty(child, definitions, seen | {name}) for child in definition["flatten"])
    return False


def _requirements(name: str, scope: IndexScope, seen=frozenset()) -> set[str]:
    """Coordinates needed to query a domain, excluding its flatten bindings."""
    if name in seen or name not in scope.definitions:
        return set()
    definition = scope.definitions[name]
    if isinstance(definition, list):
        result = set().union(*(expression_names(str(x)) for x in definition)) & scope.definitions.keys()
        for value in definition:
            for _, query in domain_queries(str(value)):
                result.update(_requirements(query, scope, seen | {name}))
        return result
    if isinstance(definition, dict):
        bound, needed = set(), set()
        for child in definition["flatten"]:
            needed.update(_requirements(child, scope, seen | {name}) - bound)
            bound.update(scope.coordinates(child))
        return needed
    return set()


def _expression(check, expression, scope, bound, all_indices, path, *, dynamic):
    if not isinstance(expression, str):
        return
    for name in expression_names(expression) & all_indices:
        check.require(name in bound, path, f"index {name} is not bound in this lexical scope")
    for operation, name in domain_queries(expression):
        if not check.require(name in scope.definitions, path, f"{operation} names an undefined index domain {name}"):
            continue
        if dynamic:
            check.require(not _dynamic(name, scope.definitions), path,
                          "count/last on a dynamically refilled worktile domain is undefined")
        if operation == "last":
            check.require(not _empty(name, scope.definitions), path, f"last({name}) needs a nonempty domain")
        for dependency in _requirements(name, scope):
            check.require(dependency in bound, path,
                          f"querying {name} requires bound outer index {dependency}")


def _domain(check, name, definition, scope, path, *, dynamic):
    if isinstance(definition, list):
        check.interval(definition, path)
    elif isinstance(definition, dict):
        restored = set()
        for i, child in enumerate(definition["flatten"]):
            cp = f"{path}.flatten[{i}]"
            if check.require(child in scope.definitions, cp, f"flatten names an undefined domain {child}"):
                coordinates = scope.coordinates(child)
                check.require(not restored & coordinates, cp,
                              "flatten components must restore distinct coordinates")
                restored.update(coordinates)
    # Domain declarations may depend on outer indices. Their binding is checked
    # at each loop/query use; here only queries and definite cycles are checked.
    if isinstance(definition, list):
        for value in definition:
            _expression(check, value, scope, set(scope.definitions), set(scope.definitions), path, dynamic=dynamic)


def _pipe(check, value, scope, path, all_indices, *, dynamic):
    try:
        root = parse_pipe(value, path=path, scope=scope)
    except PipelineSyntaxError as error:
        check.require(False, error.path, error.message)
        return None
    for node in root.walk():
        if isinstance(node, ActionDeclaration):
            for operand in node.action.operands:
                for arg in operand.args:
                    _expression(check, arg, node.scope, node.bound, all_indices, node.path, dynamic=dynamic)
        elif isinstance(node, IfBlock):
            _expression(check, node.condition, node.body.scope, node.body.bound, all_indices, node.path, dynamic=dynamic)
        elif isinstance(node, ForLoop):
            spec = node.spec
            if spec.bounds is not None:
                check.interval(list(spec.bounds), node.path)
                for value in spec.bounds:
                    _expression(check, value, node.scope, node.bound, all_indices, node.path, dynamic=dynamic)
            elif check.require(spec.name in node.scope.definitions, node.path,
                               f"for names an undefined domain {spec.name}"):
                for needed in _requirements(spec.name, node.scope):
                    check.require(needed in node.bound, node.path,
                                  f"iterating {spec.name} requires bound outer index {needed}")
            definition = node.scope.definitions.get(spec.name)
            rank = len(definition["flatten"]) if isinstance(definition, dict) else 1
            check.require(len(spec.selectors) <= rank, node.path, "for selection has too many coordinates")
            for selector in spec.selectors:
                if isinstance(selector, Slice):
                    if selector.lower is not None and selector.upper is not None:
                        check.interval([selector.lower, selector.upper], node.path)
                    values = (selector.lower, selector.upper)
                else:
                    values = (selector,)
                for value in values:
                    _expression(check, value, node.scope, node.bound, all_indices, node.path, dynamic=dynamic)
    return root


def _members(check, body, mapping, hardware, scope, all_indices, path, *, dynamic):
    pieces = parse_members(body["Info"]["Member"])
    if pieces is None:
        return
    count = literal_integer(mapping["Worker"]["unit_count"])
    for i, (unit, lower, upper) in enumerate(pieces):
        if hardware is not None:
            check.require(unit in hardware["Execution"], path, f"unregistered execution unit {unit}")
        for endpoint in (lower, upper):
            if endpoint is not None:
                check.require(not expression_names(endpoint) & all_indices, path,
                              "member composition must be fixed before the pipe, without loop indices")
                _expression(check, endpoint, scope, set(), all_indices, path, dynamic=dynamic)
        lo = literal_integer(lower)
        hi = literal_integer(upper) if upper is not None else (None if lo is None else lo + 1)
        if lo is not None and hi is not None:
            check.require(0 <= lo < hi, path, "member interval must be nonnegative and nonempty")
        if i == 0 and unit == mapping["Worker"]["unit"] and count is not None:
            check.require((lo is None or lo < count) and (hi is None or hi <= count), path,
                          "member selection exceeds Mapping.Worker.unit_count")


def _storage(check, entries, mapping, scope, all_indices, path, *, dynamic):
    count = literal_integer(mapping["Worker"]["unit_count"])
    for name, storage in entries.items():
        sp = field_path(path, name)
        depth, total = literal_integer(storage["depth"]), literal_integer(storage["total_size"])
        held = storage.get("member")
        if held is not None:
            for member, mp in indexed_values(held, sp + ".member"):
                if count is not None:
                    check.require(0 <= member < count, mp, "storage member exceeds Mapping.Worker.unit_count")
        holders = len(held) if isinstance(held, list) else (1 if held is not None else count)
        axes = {axis: size for item in storage["shape"] for axis, size in item.items()}
        sizes = [literal_integer(size) for size in axes.values()]
        unit = DTYPE_BYTES.get(storage["dtype"])
        if all(value is not None for value in [depth, total, holders, unit, *sizes]):
            required = depth * prod(sizes) * unit * holders
            check.require(total == required, sp + ".total_size",
                          f"total_size must equal depth × slot bytes × holding members ({required} bytes)")
        for field in ("depth", "total_size"):
            _expression(check, storage[field], scope, set(), all_indices, sp + "." + field, dynamic=dynamic)
        for axis, size in axes.items():
            _expression(check, size, scope, set(), all_indices, sp + ".shape", dynamic=dynamic)
        for view_name, view in storage.get("view", {}).items():
            vp = field_path(sp + ".view", view_name)
            formal = parse_action("compute:" + view_name, path=vp).operands[0]
            bound = set(formal.args)
            if "slot_index" in view:
                slot = literal_integer(view["slot_index"])
                if slot is not None and depth is not None:
                    check.require(0 <= slot < depth, vp + ".slot_index", "slot_index must be inside storage depth")
                _expression(check, view["slot_index"], scope, bound, all_indices, vp + ".slot_index", dynamic=dynamic)
            for axis, interval in view.get("region", {}).items():
                rp = field_path(vp + ".region", axis)
                check.require(axis in axes, rp, "region must name a parent Storage.shape axis")
                check.interval(interval, rp)
                lo, hi = map(literal_integer, interval)
                size = literal_integer(axes.get(axis))
                check.require(lo is None or lo >= 0, rp, "region starts before the parent slot")
                if size is not None and hi is not None:
                    check.require(hi <= size, rp, "region exceeds the parent slot")
                for endpoint in interval:
                    _expression(check, endpoint, scope, bound, all_indices, rp, dynamic=dynamic)


def validate(check: CheckContext, plan: dict[str, Any], hardware: dict[str, Any] | None, path: str) -> None:
    work = plan["DESIGN"]["Work_schedule"]["Step"]
    for step, body in plan["DESIGN"]["Tile_pipeline"]["Step"].items():
        if step not in work:  # The Step correspondence diagnostic owns this.
            continue
        p = f"{path}.DESIGN.Tile_pipeline.Step.{step}"
        definitions = {name: entry["definition"] for name, entry in body.get("Index", {}).items()}
        scope = IndexScope(definitions)
        mapping = work[step]["Mapping"]
        dynamic = work[step]["Schedule"]["Worker_assignment"]["Refill"] != "none"
        for name, definition in definitions.items():
            _domain(check, name, definition, scope, field_path(p + ".Index", name) + ".definition", dynamic=dynamic)
        check.acyclic({name: sorted(_dependencies(definition, definitions)) for name, definition in definitions.items()}, p + ".Index")

        # Local indices must also be recognized after a loop, where they are no
        # longer present in IndexScope; this catches leakage into siblings.
        all_indices = set(definitions)
        for role, role_body in body["Role"].items():
            try:
                syntax = parse_pipe(role_body["Pipe"], path=f"{p}.Role.{role}.Pipe", scope=scope)
                all_indices.update(node.spec.name for node in syntax.controls() if isinstance(node, ForLoop))
            except PipelineSyntaxError:
                pass
        families, selected = {}, {}
        for role, role_body in body["Role"].items():
            rp = f"{p}.Role.{role}"
            _members(check, role_body, mapping, hardware, scope, all_indices, rp + ".Info.Member", dynamic=dynamic)
            selected[role] = role_member_range(role_body, mapping)
            root = _pipe(check, role_body["Pipe"], scope, rp + ".Pipe", all_indices, dynamic=dynamic)
            families[role] = {}
            if root is None:
                continue
            declared = role_body["Info"]["Using_verb"]
            for declaration in root.actions():
                action = declaration.action
                check.require(action.noun in declared.get(action.verb, {}), declaration.path,
                              f"action needs Info.Using_verb.{action.verb}.{action.noun}")
                calls = families[role].setdefault(action.family, [])
                if calls:
                    check.require(action.signature == calls[0].action.signature, declaration.path,
                                  "one action family must have fixed operand argument counts in every branch")
                calls.append(declaration)
            for verb, nouns in declared.items():
                for noun in nouns:
                    check.require((verb, noun) in families[role], field_path(rp + ".Info.Using_verb." + verb, noun),
                                  "Using_verb declares an action family that is not used by this role's Pipe")

        def reference(ref, current=None, *, target=False, current_members=None):
            role = ref.role or current
            if not check.require(role in families, ref.path, f"unknown role {role}"):
                return []
            if target:
                check.require(role == current, ref.path, "Waiting target must belong to its current role")
            calls = families[role].get(ref.action.family, [])
            if check.require(bool(calls), ref.path, "event must name an action family declared in the referenced Pipe"):
                check.require(ref.action.signature == calls[0].action.signature, ref.path,
                              "event argument counts must match the referenced action family")
            member = literal_integer(ref.member)
            if member is not None and selected[role] is not None:
                check.require(member in selected[role], ref.path, f"role {role} has no worker member {member}")
            if ref.member is None and current_members is not None and selected[role] is not None:
                check.require(set(current_members) <= set(selected[role]), ref.path,
                              f"role {role} is absent on a current worker member; use explicit [n] or [*] for cross-member references")
            return calls

        for role, role_body in body["Role"].items():
            for kind, rules in role_body.get("Waiting", {}).items():
                for i, text in enumerate(rules):
                    rp = f"{p}.Role.{role}.Waiting.{kind}[{i}]"
                    rule = parse_waiting(text, path=rp)
                    current_members = selected[role]
                    literal = literal_integer(rule.target.member)
                    if literal is not None:
                        current_members = range(literal, literal + 1)
                    calls = list(reference(rule.target, role, target=True, current_members=current_members))
                    for ref in rule.predecessors:
                        calls += reference(ref, role, current_members=current_members)
                    scopes = [call.scope for call in calls]
                    merged = dict(definitions)
                    for ref in (rule.target, *rule.predecessors):
                        names = free_action_indices(ref.action) | {name for _, name in domain_queries(ref.text)}
                        if ref.member not in (None, "*"):
                            names |= expression_names(ref.member)
                        for name in names & all_indices:
                            choices = domain_choices(scopes, name)
                            if check.require(len(choices) == 1, rp,
                                             f"Waiting index {name} needs one applicable declared domain"):
                                merged[name] = choices[0]
                    rule_scope = IndexScope(merged)
                    for ref in (rule.target, *rule.predecessors):
                        for operand in ref.action.operands:
                            for arg in operand.args:
                                _expression(check, arg, rule_scope, set(merged), all_indices, rp, dynamic=dynamic)
                        if ref.member not in (None, "*"):
                            _expression(check, ref.member, rule_scope, set(merged), all_indices, rp, dynamic=dynamic)

        for name, references in body.get("Event", {}).items():
            ep = field_path(p + ".Event", name)
            formals = set(parse_action("compute:" + name, path=ep).operands[0].args)
            for i, value in enumerate(references):
                ref = parse_event(value, path=f"{ep}[{i}]")
                check.require(ref.role is not None and ref.member is not None, ref.path,
                              "Event aliases must explicitly name time, ROLE and member")
                reference(ref)
                for operand in ref.action.operands:
                    for arg in operand.args:
                        _expression(check, arg, scope, formals, all_indices, ref.path, dynamic=dynamic)
                if ref.member != "*":
                    _expression(check, ref.member, scope, formals, all_indices, ref.path, dynamic=dynamic)
        _storage(check, body.get("Storage", {}), mapping, scope, all_indices, p + ".Storage", dynamic=dynamic)
