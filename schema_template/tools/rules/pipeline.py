"""Local pipeline roles, resources, action references and index scope."""
from __future__ import annotations

import re
from typing import Any

from ..diagnostics import field_path
from ..pipeline_syntax import (IndexScope, ScopedPipe, common_domains,
                               free_action_indices, parse_action, parse_event, parse_members)
from .common import CheckContext, indexed_values
from .values import literal_integer

GENERIC_VERBS = {"load", "publish", "store", "compute", "produce", "zero", "mma", "wait", "release"}


def validate(check: CheckContext, plan: dict[str, Any], hardware: dict[str, Any] | None, path: str) -> None:
    work = plan["design"]["work_schedule"]["phase"]
    for pid, phase in plan["design"]["tile_pipeline"]["phase"].items():
        p = path + ".design.tile_pipeline.phase." + pid
        domains = phase.get("domain", {})
        graph = {name: v["flatten"] for name, v in domains.items() if isinstance(v, dict)}
        for name, domain in domains.items():
            _check_domain(check, domain, domains, p + ".domain." + name)
        check.acyclic(graph, p + ".domain")
        unit_count = literal_integer(work[pid]["mapping"]["worker"]["unit_count"])
        dynamic = work[pid]["schedule"]["worker_assignment"]["refill"] != "none"
        families, role_members = {}, {}
        for role, body in phase["role"].items():
            lp = p + ".role." + role
            pieces = parse_members(body["info"]["member"])
            role_members[role] = range(unit_count) if unit_count is not None else None
            for i, (unit, lower, upper) in enumerate(pieces):
                lo = literal_integer(lower)
                hi = literal_integer(upper) if upper is not None else (lo + 1 if lo is not None else None)
                if hardware is not None:
                    check.require(unit in hardware["execution"], lp + ".info.member", f"unregistered execution unit {unit}")
                if lo is not None and hi is not None:
                    check.require(hi > lo, lp + ".info.member", "member interval must be nonempty")
                if i == 0 and unit == work[pid]["mapping"]["worker"]["unit"]:
                    role_members[role] = range(lo, hi) if lo is not None and hi is not None else None
                    if unit_count is not None:
                        check.require((lo is None or lo < unit_count) and (hi is None or hi <= unit_count),
                                      lp + ".info.member", "member index exceeds worker.unit_count")
            actions = _check_pipe(check, body["pipe"], domains, set(), lp + ".pipe", dynamic=dynamic)
            families[role] = {}
            for action in actions:
                parsed = parse_action(action.text)
                if check.require(parsed is not None, action.path, "invalid action notation"):
                    verb = parsed[0]
                    check.require(verb in GENERIC_VERBS or verb in body["info"].get("verb", {}),
                                 action.path, f"custom verb {verb} needs an info.verb contract")
                    families[role].setdefault(parsed, []).append(action.scope)
        if dynamic:
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
                if not check.require(len(matches) == 1, rp, "phase milestone needs an unambiguous role-qualified event"):
                    return
                target = matches[0]
            if not check.require(target in families, rp, f"unknown role {target}"):
                return
            if waiting:
                check.require(target == current, rp, "the waiting event must belong to the current role")
            check.require(parsed in families[target], rp, "event does not name an action family in the referenced role")
            if dynamic and parsed in families[target]:
                _check_dynamic_domain_queries(check, action, common_domains(families[target][parsed]), rp)
            if member not in (None, "*") and unit_count is not None:
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
                if unit_count is not None:
                    check.require(member < unit_count, mp, "resource member exceeds worker.unit_count")
            capacity = resource["member_slot_shape"]
            axes = {k: v for axis in capacity for k, v in axis.items()}
            for view, value in resource.get("view", {}).items():
                vp = field_path(rp + ".view", view)
                check.require(parse_action("load:" + view) is not None, vp, "invalid view index notation")
                for axis, bounds in value.get("part", {}).items():
                    check.require(axis in axes, vp + ".part", "part must name a parent resource axis")
                    check.interval(bounds, vp + ".part." + axis)
                    size, end = literal_integer(axes.get(axis)), literal_integer(bounds[1])
                    if size is not None and end is not None:
                        check.require(end <= size, vp + ".part." + axis, "view exceeds parent storage")


def _check_domain(check: CheckContext, domain: Any, known: dict[str, Any], path: str) -> None:
    if isinstance(domain, list):
        check.interval(domain, path)
    elif isinstance(domain, dict):
        for i, item in enumerate(domain["flatten"]):
            check.require(item in known, f"{path}.flatten[{i}]", f"flatten names an undefined domain {item}")


def _check_pipe(check: CheckContext, pipe: dict[str, Any], known: dict[str, Any], bound: set[str],
                path: str, *, dynamic: bool):
    root = ScopedPipe.read(pipe, IndexScope(known), path, frozenset(bound))
    for block in root.blocks():
        repeat = block.body.get("repeat")
        scope = block.scope
        if isinstance(repeat, dict):
            name, domain = next(iter(repeat.items()))
            _check_domain(check, domain, scope.parent.definitions, block.path + ".repeat." + name)
        elif isinstance(repeat, str):
            index = repeat.split("[", 1)[0]
            check.require(index in scope.definitions, block.path + ".repeat", f"undefined repeat domain {index}")
        if dynamic:
            _check_dynamic_domain_queries(check, repeat, scope.definitions, block.path + ".repeat")
    actions = list(root.actions())
    for action in actions:
        if dynamic:
            _check_dynamic_domain_queries(check, action.text, action.scope.definitions, action.path)
        for free in free_action_indices(action.text):
            if free in action.scope.definitions:
                check.require(free in action.bound, action.path, f"index {free} is not bound in this pipe region")
    return actions


def _check_dynamic_domain_queries(check: CheckContext, value: Any, domains: dict[str, Any], path: str) -> None:
    if isinstance(value, str):
        for name, domain in domains.items():
            if domain == "worktile":
                check.require(re.search(r"\b(?:count|last)\s*\(\s*" + re.escape(name) + r"\s*\)", value) is None,
                             path, "count/last on a dynamically refilled worktile domain is undefined")
    elif isinstance(value, dict):
        for k, v in value.items():
            _check_dynamic_domain_queries(check, k, domains, path)
            _check_dynamic_domain_queries(check, v, domains, field_path(path, k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _check_dynamic_domain_queries(check, v, domains, f"{path}[{i}]")
