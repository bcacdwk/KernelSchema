"""Local pipeline roles, resources, action references and index scope."""
from __future__ import annotations

import ast
import re
from typing import Any

from ..diagnostics import IDENTIFIER, field_path
from .common import CheckContext, indexed_values

GENERIC_VERBS = {"load", "publish", "store", "compute", "produce", "zero", "mma", "wait", "release"}
DOMAIN_FUNCTIONS = {"count", "last"}


def validate(check: CheckContext, plan: dict[str, Any], hardware: dict[str, Any] | None, path: str) -> None:
    work = plan["design"]["work_schedule"]["phase"]
    for pid, phase in plan["design"]["tile_pipeline"]["phase"].items():
        p = path + ".design.tile_pipeline.phase." + pid
        domains = phase.get("domain", {})
        graph = {name: v["flatten"] for name, v in domains.items() if isinstance(v, dict)}
        for name, domain in domains.items():
            _check_domain(check, domain, domains, p + ".domain." + name)
        check.acyclic(graph, p + ".domain")
        unit_count = work[pid]["mapping"]["worker"]["unit_count"]
        families, role_members = {}, {}
        for role, body in phase["role"].items():
            lp = p + ".role." + role
            pieces = re.findall(r"([^\[\]]+)\[(\d+)(?::(\d+))?\]", body["info"]["member"])
            unit, lo, hi = pieces[0]
            if unit == work[pid]["mapping"]["worker"]["unit"]:
                role_members[role] = range(int(lo), int(hi) if hi else int(lo) + 1)
            else:
                role_members[role] = range(unit_count) if type(unit_count) is int else None
            for i, (unit, lo, hi) in enumerate(pieces):
                unit = unit.removeprefix(".")
                if hardware is not None:
                    check.require(unit in hardware["execution"], lp + ".info.member", f"unregistered execution unit {unit}")
                check.require(not hi or int(hi) > int(lo), lp + ".info.member", "member interval must be nonempty")
                if i == 0 and unit == work[pid]["mapping"]["worker"]["unit"] and type(unit_count) is int:
                    check.require(int(hi) <= unit_count if hi else int(lo) < unit_count,
                                 lp + ".info.member", "member index exceeds worker.unit_count")
            actions = _check_pipe(check, body["pipe"], domains, set(), lp + ".pipe")
            families[role] = set()
            for action, ap in actions:
                parsed = parse_action(action)
                if check.require(parsed is not None, ap, "invalid action notation"):
                    verb, objects = parsed
                    check.require(verb in GENERIC_VERBS or verb in body["info"].get("verb", {}),
                                 ap, f"custom verb {verb} needs an info.verb contract")
                    families[role].add((verb, objects))
            if work[pid]["schedule"]["worker_assignment"]["refill"] != "none":
                for field in ("pipe", "waits_for"):
                    _check_dynamic_domain_queries(check, body.get(field, {}), domains, lp + "." + field)
        if work[pid]["schedule"]["worker_assignment"]["refill"] != "none":
            for field in ("domain", "resource"):
                _check_dynamic_domain_queries(check, phase.get(field, {}), domains, p + "." + field)

        def resolve(ref: str, rp: str, current: str | None, waiting: bool = False) -> None:
            target, member, action = parse_event(ref)
            if target is None:
                target = current
            parsed = parse_action(action)
            if not check.require(parsed is not None, rp, "invalid event/action notation; events are issue or done"):
                return
            if target is None:
                matches = [role for role, items in families.items() if parsed in items]
                check.require(len(matches) == 1, rp, "phase milestone needs an unambiguous role-qualified event")
                return
            if not check.require(target in families, rp, f"unknown role {target}"):
                return
            if waiting:
                check.require(target == current, rp, "the waiting event must belong to the current role")
            check.require(parsed in families[target], rp, "event does not name an action family in the referenced role")
            if member not in (None, "*") and type(unit_count) is int:
                if not check.require(int(member) < unit_count, rp, "event member exceeds worker.unit_count"):
                    return
            selected = role_members[target]
            if member not in (None, "*") and selected is not None:
                check.require(int(member) in selected, rp,
                              f"role {target} has no member {member}; check its info.member selection")

        for role, body in phase["role"].items():
            for kind, edges in body.get("waits_for", {}).items():
                for left, right in edges.items():
                    ep = field_path(f"{p}.role.{role}.waits_for.{kind}", left)
                    resolve(left, ep, role, True)
                    for i, ref in enumerate(right):
                        resolve(ref, f"{ep}[{i}]", role)
                        # Only reject literal same-instance dependencies. Index
                        # algebra and cross-member relationships remain for expansion.
                        ll, lm, la = parse_event(left)
                        rl, rm, ra = parse_event(ref)
                        le = "done" if left.endswith("@done") else "issue"
                        revent = "issue" if ref.endswith("@issue") else "done"
                        same = ((ll or role) == (rl or role) == role and lm == rm and la == ra
                                and parse_action(la) in families[role])
                        if lm not in (None, "*") and role_members[role] is not None:
                            same &= int(lm) in role_members[role]
                        if same and (le == "issue" or revent == "done"):
                            check.require(False, f"{ep}[{i}]",
                                          "an event cannot depend on itself or its own later completion; check the event suffix or index shift")
        for milestone, events in phase.get("milestone", {}).items():
            for i, ref in enumerate(events):
                resolve(ref, field_path(p + ".milestone", milestone) + f"[{i}]", None)
        for name, resource in phase.get("resource", {}).items():
            rp = p + ".resource." + name
            for member, mp in indexed_values(resource.get("member", []), rp + ".member"):
                if type(unit_count) is int:
                    check.require(member < unit_count, mp, "resource member exceeds worker.unit_count")
            capacity = resource["member_slot_shape"]
            axes = {k: v for axis in capacity for k, v in axis.items()}
            for view, value in resource.get("view", {}).items():
                vp = field_path(rp + ".view", view)
                check.require(parse_action("load:" + view) is not None, vp, "invalid view index notation")
                for axis, bounds in value.get("part", {}).items():
                    check.require(axis in axes, vp + ".part", "part must name a parent resource axis")
                    check.interval(bounds, vp + ".part." + axis)
                    if axis in axes and type(axes[axis]) is int and type(bounds[1]) is int:
                        check.require(bounds[1] <= axes[axis], vp + ".part." + axis, "view exceeds parent storage")


def _check_domain(check: CheckContext, domain: Any, known: dict[str, Any], path: str) -> None:
    if isinstance(domain, list):
        check.interval(domain, path)
    elif isinstance(domain, dict):
        for i, item in enumerate(domain["flatten"]):
            check.require(item in known, f"{path}.flatten[{i}]", f"flatten names an undefined domain {item}")


def _check_pipe(check: CheckContext, pipe: dict[str, Any], known: dict[str, Any], bound: set[str], path: str) -> list[tuple[str, str]]:
    scope = dict(known)
    repeat, index = pipe.get("repeat"), None
    if isinstance(repeat, dict):
        index, domain = next(iter(repeat.items()))
        _check_domain(check, domain, scope, path + ".repeat." + index)
        scope[index] = domain
    elif isinstance(repeat, str):
        index = repeat.split("[", 1)[0]
        check.require(index in scope, path + ".repeat", f"undefined repeat domain {index}")
    body_bound = set(bound)

    def bind(name: str, visited: set[str]) -> None:
        if name in visited:
            return
        visited.add(name)
        body_bound.add(name)
        domain = scope.get(name)
        if isinstance(domain, dict):
            for child in domain["flatten"]:
                bind(child, visited)

    if index:
        bind(index, set())
    actions = []
    for region in ("prologue", "steady", "epilogue"):
        current = body_bound if region == "steady" else bound
        for i, action in enumerate(pipe.get(region, [])):
            ap = f"{path}.{region}[{i}]"
            if isinstance(action, dict):
                actions.extend(_check_pipe(check, action, scope, current, ap))
            else:
                actions.append((action, ap))
                for free in free_action_indices(action):
                    if free in scope:
                        check.require(free in current, ap, f"index {free} is not bound in this pipe region")
    return actions


def _check_dynamic_domain_queries(check: CheckContext, value: Any, domains: dict[str, Any], path: str) -> None:
    if isinstance(value, str):
        for name, domain in domains.items():
            if domain == "worktile":
                check.require(re.search(r"\b(?:count|last)\(\s*" + re.escape(name) + r"\s*\)", value) is None,
                             path, "count/last on a dynamically refilled worktile domain is undefined")
    elif isinstance(value, dict):
        for k, v in value.items():
            _check_dynamic_domain_queries(check, k, domains, path)
            _check_dynamic_domain_queries(check, v, domains, field_path(path, k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _check_dynamic_domain_queries(check, v, domains, f"{path}[{i}]")


def split_top(text: str, delimiter: str) -> list[str] | None:
    pairs, stack, result, start = {"(": ")", "[": "]"}, [], [], 0
    for i, char in enumerate(text):
        if char in pairs:
            stack.append(pairs[char])
        elif char in ")]":
            if not stack or stack.pop() != char:
                return None
        elif char == delimiter and not stack:
            result.append(text[start:i])
            start = i + 1
    return None if stack else result + [text[start:]]


def action_parts(text: str) -> tuple[str, list[str]] | None:
    verb, colon, rest = text.partition(":")
    if not colon or not IDENTIFIER.fullmatch(verb):
        return None
    operands = split_top(rest, "@")
    if operands is None or len(operands) != (2 if verb == "mma" else 1):
        return None
    return verb, operands


def parse_action(text: str) -> tuple[str, tuple[tuple[str, int], ...]] | None:
    parts = action_parts(text)
    if parts is None:
        return None
    verb, operands = parts
    identities = []
    for operand in operands:
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(?:\((.+)\))?", operand)
        if not match:
            return None
        args = split_top(match[2], ",") if match[2] is not None else []
        if args is None or any(not a.strip() for a in args):
            return None
        identities.append((match[1], len(args)))
    return verb, tuple(identities)


def parse_event(text: str) -> tuple[str | None, str | None, str]:
    text = re.sub(r"@(issue|done)$", "", text)
    match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)(?:\[(\*|[0-9]+)\])?/", text)
    if match:
        return match[1], match[2], text[match.end():]
    return None, None, text


def free_action_indices(text: str) -> set[str]:
    """Check scopes when indices use familiar syntax; other math stays for review."""
    parts, names = action_parts(text), set()
    if parts is None:
        return names

    def walk(node: ast.AST) -> None:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in DOMAIN_FUNCTIONS:
            return
        if isinstance(node, ast.Call):
            for argument in node.args:
                walk(argument)
            return
        if isinstance(node, ast.Subscript):
            walk(node.slice)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        else:
            for child in ast.iter_child_nodes(node):
                walk(child)

    for operand in parts[1]:
        if "(" in operand and operand.endswith(")"):
            try:
                walk(ast.parse(operand.split("(", 1)[1][:-1], mode="eval"))
            except (SyntaxError, ValueError, RecursionError):
                pass  # General mathematical index interpretation is not a proof obligation here.
    return names
