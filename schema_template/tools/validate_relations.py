"""Relations on concrete CUE data. No formula evaluation or event unfolding.

CUE owns field structure and types. These checks traverse named graphs and
parse the documented small notations, without inferring facts from prose.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
import re
from typing import Any

GENERIC_VERBS = {"load", "publish", "store", "compute", "produce", "zero", "mma", "wait", "release"}
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
INTEGER_FUNCTIONS = {"min", "max", "floor", "ceil", "ceil_div"}
DOMAIN_FUNCTIONS = {"count", "last"}


@dataclass(frozen=True)
class Diagnostic:
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


class Checker:
    def __init__(self) -> None:
        self.errors: list[Diagnostic] = []

    def require(self, condition: bool, path: str, message: str) -> bool:
        if not condition:
            self.errors.append(Diagnostic(path, message))
        return condition

    def acyclic(self, graph: dict[Any, list[Any]], path: str) -> None:
        active, finished = set(), set()

        def visit(node: Any, chain: list[Any]) -> None:
            if node in active:
                self.require(False, path, "cycle: " + " -> ".join(map(str, chain + [node])))
                return
            if node in finished:
                return
            active.add(node)
            for target in graph.get(node, []):
                visit(target, chain + [node])
            active.remove(node)
            finished.add(node)

        for node in graph:
            visit(node, [])

    def integer(self, value: Any, path: str, minimum: int | None = None, pipeline: bool = False) -> None:
        if type(value) is int:
            self.require(minimum is None or value >= minimum, path, f"integer must be >= {minimum}")
            return
        try:
            tree = ast.parse(value.strip(), mode="eval")
        except (SyntaxError, ValueError, RecursionError, AttributeError):
            self.require(False, path, "invalid integer-expression syntax")
            return
        functions = INTEGER_FUNCTIONS | (DOMAIN_FUNCTIONS if pipeline else set())

        def valid(node: ast.AST) -> bool:
            if isinstance(node, ast.Constant):
                return type(node.value) is int
            if isinstance(node, ast.Name):
                return bool(IDENTIFIER.fullmatch(node.id))
            if isinstance(node, ast.UnaryOp):
                return isinstance(node.op, (ast.UAdd, ast.USub)) and valid(node.operand)
            if isinstance(node, ast.BinOp):
                ops = (ast.Add, ast.Sub, ast.Mult, ast.Div) + ((ast.Mod,) if pipeline else ())
                return isinstance(node.op, ops) and valid(node.left) and valid(node.right)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                name, count = node.func.id, len(node.args)
                arity = count >= 1 if name in {"min", "max"} else count == (2 if name == "ceil_div" else 1)
                return name in functions and arity and not node.keywords and all(valid(a) for a in node.args)
            return False

        self.require(valid(tree.body), path, "only integer arithmetic, names and documented functions are allowed")
        literal, sign = tree.body, 1
        if isinstance(literal, ast.UnaryOp) and isinstance(literal.op, (ast.USub, ast.UAdd)):
            sign = -1 if isinstance(literal.op, ast.USub) else 1
            literal = literal.operand
        if isinstance(literal, ast.Constant) and type(literal.value) is int:
            self.require(minimum is None or sign * literal.value >= minimum, path, f"literal must be >= {minimum}")

    def interval(self, value: list[Any], path: str, minimum: int | None = None, pipeline: bool = False) -> None:
        lo, hi = value
        self.integer(lo, path + "[0]", minimum, pipeline)
        if hi is not None:
            self.integer(hi, path + "[1]", minimum, pipeline)
            if type(lo) is int and type(hi) is int:
                self.require(lo <= hi, path, "range endpoints are reversed")

    def shapes(self, plan: dict[str, Any], path: str) -> None:
        # Visit descriptor positions, not every mapping key spelled "shape":
        # that spelling is also a valid domain, milestone or object name.
        descriptors = []
        algorithm = plan.get("contract", {}).get("algorithm_requirement", {})
        for group in ("input", "output"):
            descriptors.extend((obj, f"{path}.contract.algorithm_requirement.{group}.{name}")
                               for name, obj in algorithm.get(group, {}).items())
        for i, step in enumerate(algorithm.get("compute", [])):
            descriptors.extend((obj, f"{path}.contract.algorithm_requirement.compute[{i}].symbol.{name}")
                               for name, obj in step.get("symbol", {}).items())
        for i, step in enumerate(plan["design"]["math_reshape"]["rewrite"]):
            descriptors.extend((obj, f"{path}.design.math_reshape.rewrite[{i}].symbol.{name}")
                               for name, obj in step.get("symbol", {}).items())
        for pid, phase in plan["design"]["work_schedule"]["phase"].items():
            dec = phase["decomposition"]
            blocks = {"target.single_tile": dec["target"]["single_tile"]}
            blocks.update({"worktile." + k: v for k, v in dec.get("worktile", {}).items()})
            for group, values in blocks.items():
                descriptors.extend((obj, f"{path}.design.work_schedule.phase.{pid}.decomposition.{group}.{name}")
                                   for name, obj in values.items())
        for pid, phase in plan["design"]["tile_pipeline"]["phase"].items():
            descriptors.extend((obj, f"{path}.design.tile_pipeline.phase.{pid}.resource.{name}")
                               for name, obj in phase.get("resource", {}).items())
        for descriptor, p in descriptors:
            shape = descriptor.get("shape")
            if isinstance(shape, list):
                axes = [next(iter(axis)) for axis in shape]
                self.require(len(axes) == len(set(axes)), p + ".shape", "axis names must be unambiguous within one shape")
                for i, axis in enumerate(shape):
                    self.integer(next(iter(axis.values())), f"{p}.shape[{i}]", 0, "tile_pipeline" in p)

    def contract(self, contract: dict[str, Any], path: str) -> None:
        objects, order = {}, {}
        names = set(contract.get("constant", {}))
        groups = [(contract[f], -1, f) for f in ("input", "output")]
        groups += [(s.get("symbol", {}), i, f"compute[{i}].symbol") for i, s in enumerate(contract["compute"])]
        for group, index, label in groups:
            for name, obj in group.items():
                self.require(name not in names, f"{path}.{label}.{name}", "duplicate Contract object name")
                names.add(name)
                objects[name], order[name] = obj, index
        for attr in ("shape", "dtype"):
            graph = {}
            for name, obj in objects.items():
                value = obj[attr]
                if isinstance(value, str) and value.startswith("ref:"):
                    target = value[4:].rsplit(".", 1)[0]
                    p = f"{path}.{name}.{attr}"
                    if self.require(target in objects, p, f"unknown Contract reference {target}"):
                        self.require(order[target] <= order[name], p, "reference precedes the defining compute step")
                        graph[name] = [target]
            self.acyclic(graph, path + "." + attr)
        for name, obj in contract["input"].items():
            for axis in obj["shape"]:
                size = next(iter(axis.values()))
                if isinstance(size, str) and IDENTIFIER.fullmatch(size) and size not in contract.get("constant", {}):
                    self.require(size not in names, path + ".input." + name + ".shape", "a runtime extent cannot also name a Contract object")
            bounds = obj.get("range")
            if isinstance(bounds, list) and all(type(x) in (int, float, Decimal) for x in bounds):
                self.require(bounds[0] <= bounds[1], path + ".input." + name + ".range", "range endpoints are reversed")

    def math(self, plan: dict[str, Any], contract: dict[str, Any], path: str) -> None:
        names = {}
        reserved = set(contract.get("constant", {})) | set(contract["input"]) | set(contract["output"])
        for i, step in enumerate(plan["design"]["math_reshape"]["rewrite"]):
            for name, obj in step.get("symbol", {}).items():
                p = f"{path}.design.math_reshape.rewrite[{i}].symbol.{name}"
                self.require(name not in names and name not in reserved, p, "symbol duplicates an existing definition")
                names[name] = obj
        for phase in plan["design"]["work_schedule"]["phase"].values():
            dec = phase["decomposition"]
            for block in [dec["target"]["single_tile"], *dec.get("worktile", {}).values()]:
                for name in block:
                    if name in names:
                        self.require("shape" in names[name] and "dtype" in names[name],
                                     path + ".design.math_reshape.symbol." + name,
                                     "a value consumed as a data block must publish shape and dtype")

    def schedule(self, plan: dict[str, Any], path: str) -> None:
        work = plan["design"]["work_schedule"]
        pipe = plan["design"]["tile_pipeline"]["phase"]
        launches = work["flow"]["launch"]
        owner = {p: i for i, launch in enumerate(launches) for p in launch["phase"]}
        graph = defaultdict(list)

        def event(ref: str, p: str, participants: set[str] | None = None) -> tuple[str, str] | None:
            phase, milestone = ref.split(".", 1)
            if not self.require(phase in pipe, p, f"unknown event phase {phase}"):
                return None
            if participants is not None:
                self.require(phase in participants, p, "event belongs to the wrong handoff participant")
            if milestone not in {"kernel_done", "launch_ready"}:
                identity = milestone.split("(", 1)[0]
                self.require(any(m.split("(", 1)[0] == identity for m in pipe[phase].get("milestone", {})),
                             p, f"undefined milestone {ref}")
            return phase, milestone

        for i, launch in enumerate(launches):
            p = f"{path}.design.work_schedule.flow.launch[{i}]"
            for ref in as_list(launch.get("start_after", [])):
                result = event(ref, p + ".start_after")
                if result:
                    phase, milestone = result
                    self.require(owner[phase] != i, p, "a launch cannot wait for its own event before starting")
                    if milestone == "kernel_done":
                        graph[i].append(owner[phase])
                    if milestone == "launch_ready":
                        self.require("protocol" in launch, p, "launch_ready requires an explicit launch protocol")
            if len(launch["phase"]) > 1 and launch["order"] != "concurrent":
                first = work["phase"][launch["phase"][0]]
                resources = set()
                for pid in launch["phase"]:
                    current = work["phase"][pid]
                    self.require(first["mapping"]["worker"]["unit"] == current["mapping"]["worker"]["unit"],
                                 p, "sequential phases share a worker unit")
                    for field in ("worker", "count", "placement"):
                        self.require(not concrete_conflict(first["mapping"].get(field), current["mapping"].get(field)),
                                     p, "sequential phases have incompatible worker mappings")
                    for name in pipe[pid].get("resource", {}):
                        self.require(name not in resources, p, f"shared-worker resource {name} is defined more than once")
                        resources.add(name)
                    if launch["order"] == "per_worktile":
                        pairs = [(first["decomposition"]["partition"], current["decomposition"]["partition"]),
                                 (first["schedule"].get("worktile_order"), current["schedule"].get("worktile_order")),
                                 (first["schedule"]["worker_assignment"].get("preassigned"),
                                  current["schedule"]["worker_assignment"].get("preassigned"))]
                        for a, b in pairs:
                            self.require(not concrete_conflict(a, b), p, "per_worktile requires matching coordinates, order and preassignment")
        self.acyclic(graph, path + ".design.work_schedule.flow.launch")

        for name, handoff in work["flow"].get("handoff", {}).items():
            p = path + ".design.work_schedule.flow.handoff." + name
            producer, consumers = handoff["producer"], handoff["consumer"]
            self.require(producer not in consumers, p, "handoff connects distinct phases")
            for ref in as_list(handoff.get("ready", [])):
                event(ref, p + ".ready", {producer})
            for ref in as_list(handoff.get("wait_at", [])):
                event(ref, p + ".wait_at", set(consumers))
            if handoff["storage"] != "global":
                _, _, rp, _, rn = handoff["storage"].split(".")
                self.require(rp in pipe and rn in pipe.get(rp, {}).get("resource", {}), p + ".storage", "undefined resource")
                participants = [producer, *consumers, rp]
                if all(x in owner for x in participants):
                    submit = owner[producer]
                    self.require(all(owner[x] == submit for x in participants) and launches[submit].get("order") != "concurrent",
                                 p + ".storage", "local storage requires the same worker instance within one launch")
        for pid, phase in work["phase"].items():
            p = path + ".design.work_schedule.phase." + pid
            dec, mapping, schedule = phase["decomposition"], phase["mapping"], phase["schedule"]
            for dim, part in dec["partition"].items():
                for field in ("size", "tile"):
                    self.integer(part[field], f"{p}.decomposition.partition.{dim}.{field}", 1)
                if "range" in part:
                    self.interval(part["range"], f"{p}.decomposition.partition.{dim}.range", 1)
            for name, block in dec.get("worktile", {}).get("invariant", {}).items():
                for i, offset in enumerate(block.get("offset", [])):
                    self.integer(offset, f"{p}.decomposition.worktile.invariant.{name}.offset[{i}]")
            self.integer(mapping["count"], p + ".mapping.count", 1)
            self.integer(mapping["worker"]["count"], p + ".mapping.worker.count", 1)
            if "placement" in mapping:
                self.integer(mapping["placement"]["count"], p + ".mapping.placement.count", 1)
            for member in mapping.get("member", []):
                for dimension, regions in member.items():
                    for region in regions if isinstance(regions[0], list) else [regions]:
                        self.interval(region, p + ".mapping.member." + dimension, 0)
            for entry in schedule.get("worktile_order", []):
                if "group" in entry:
                    self.integer(entry["group"], p + ".schedule.worktile_order.group", 1)
            assignment = schedule["worker_assignment"]
            if assignment["refill"] == "none":
                self.require("preassigned" in assignment, p + ".schedule.worker_assignment",
                             "without refill, declare the preassigned work sequence")
            for field, value in assignment.get("preassigned", {}).items():
                self.integer(value, p + ".schedule.worker_assignment.preassigned." + field, 1 if field == "step" else 0)

    def pipeline(self, plan: dict[str, Any], hardware: dict[str, Any], path: str) -> None:
        work = plan["design"]["work_schedule"]["phase"]
        for pid, phase in plan["design"]["tile_pipeline"]["phase"].items():
            p = path + ".design.tile_pipeline.phase." + pid
            domains = phase.get("domain", {})
            graph = {name: v["flatten"] for name, v in domains.items() if isinstance(v, dict)}
            for name, domain in domains.items():
                self.domain(domain, domains, p + ".domain." + name)
            self.acyclic(graph, p + ".domain")
            worker_count = work[pid]["mapping"]["worker"]["count"]
            families = {}
            for lane, body in phase["lane"].items():
                lp = p + ".lane." + lane
                pieces = re.findall(r"([^\[\]]+)\[(\d+)(?::(\d+))?\]", body["info"]["member"])
                for i, (unit, lo, hi) in enumerate(pieces):
                    unit = unit.removeprefix(".")
                    self.require(unit in hardware["execution"], lp + ".info.member", f"unregistered execution unit {unit}")
                    self.require(not hi or int(hi) > int(lo), lp + ".info.member", "member interval must be nonempty")
                    if i == 0 and unit == work[pid]["mapping"]["worker"]["unit"] and type(worker_count) is int:
                        self.require(int(hi) <= worker_count if hi else int(lo) < worker_count,
                                     lp + ".info.member", "member index exceeds worker count")
                actions = self.pipe(body["pipe"], domains, set(), lp + ".pipe")
                families[lane] = set()
                for action, ap in actions:
                    parsed = parse_action(action)
                    if self.require(parsed is not None, ap, "invalid action notation"):
                        verb, objects = parsed
                        self.require(verb in GENERIC_VERBS or verb in body["info"].get("verb", {}),
                                     ap, f"custom verb {verb} needs an info.verb contract")
                        families[lane].add((verb, objects))
                if work[pid]["schedule"]["worker_assignment"]["refill"] != "none":
                    for field in ("pipe", "edge"):
                        self.dynamic_domain_queries(body.get(field, {}), domains, lp + "." + field)
            if work[pid]["schedule"]["worker_assignment"]["refill"] != "none":
                for field in ("domain", "resource"):
                    self.dynamic_domain_queries(phase.get(field, {}), domains, p + "." + field)

            def resolve(ref: str, rp: str, current: str | None, waiting: bool = False) -> None:
                target, member, action = parse_event(ref)
                if target is None:
                    target = current
                parsed = parse_action(action)
                if not self.require(parsed is not None, rp, "invalid event/action notation; events are issue or done"):
                    return
                if target is None:
                    matches = [lane for lane, items in families.items() if parsed in items]
                    self.require(len(matches) == 1, rp, "phase milestone needs an unambiguous lane-qualified event")
                    return
                if not self.require(target in families, rp, f"unknown lane {target}"):
                    return
                if waiting:
                    self.require(target == current, rp, "the waiting event must belong to the current lane")
                self.require(parsed in families[target], rp, "event does not name an action family in the referenced lane")
                if member not in (None, "*") and type(worker_count) is int:
                    self.require(int(member) < worker_count, rp, "event member exceeds worker count")

            for lane, body in phase["lane"].items():
                for kind, edges in body.get("edge", {}).items():
                    for left, right in edges.items():
                        ep = f"{p}.lane.{lane}.edge.{kind}.{left}"
                        resolve(left, ep, lane, True)
                        for ref in right:
                            resolve(ref, ep, lane)
            for milestone, events in phase.get("milestone", {}).items():
                self.require(milestone.split("(", 1)[0] not in {"kernel_done", "launch_ready"},
                             p + ".milestone", "built-in launch events cannot be redefined")
                for ref in events:
                    resolve(ref, p + ".milestone." + milestone, None)
            for name, resource in phase.get("resource", {}).items():
                rp = p + ".resource." + name
                if "slot" in resource:
                    self.integer(resource["slot"], rp + ".slot", 1, True)
                for member in as_list(resource.get("member", [])):
                    if type(worker_count) is int:
                        self.require(member < worker_count, rp + ".member", "resource member exceeds worker count")
                axes = {k: v for axis in resource["shape"] for k, v in axis.items()}
                for view, value in resource.get("view", {}).items():
                    vp = rp + ".view." + view
                    self.require(parse_action("load:" + view) is not None, vp, "invalid view index notation")
                    if "slot" in value:
                        self.integer(value["slot"], vp + ".slot", 0, True)
                    for axis, bounds in value.get("part", {}).items():
                        self.require(axis in axes, vp + ".part", "part must name a parent resource axis")
                        self.interval(bounds, vp + ".part." + axis, 0, True)
                        if axis in axes and type(axes[axis]) is int and type(bounds[1]) is int:
                            self.require(bounds[1] <= axes[axis], vp + ".part." + axis, "view exceeds parent storage")

    def domain(self, domain: Any, known: dict[str, Any], path: str) -> None:
        if isinstance(domain, list):
            self.interval(domain, path, pipeline=True)
        elif isinstance(domain, dict):
            for item in domain["flatten"]:
                self.require(item in known, path, f"flatten names an undefined domain {item}")

    def pipe(self, pipe: dict[str, Any], known: dict[str, Any], bound: set[str], path: str) -> list[tuple[str, str]]:
        scope = dict(known)
        repeat, index = pipe.get("repeat"), None
        if isinstance(repeat, dict):
            index, domain = next(iter(repeat.items()))
            self.domain(domain, scope, path + ".repeat")
            scope[index] = domain
        elif isinstance(repeat, str):
            index = repeat.split("[", 1)[0]
            self.require(index in scope, path + ".repeat", f"undefined repeat domain {index}")
            self.require(valid_index(repeat), path + ".repeat", "invalid domain slice notation")
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
                    actions.extend(self.pipe(action, scope, current, ap))
                else:
                    actions.append((action, ap))
                    for free in free_action_indices(action):
                        if free in scope:
                            self.require(free in current, ap, f"index {free} is not bound in this pipe region")
        return actions

    def dynamic_domain_queries(self, value: Any, domains: dict[str, Any], path: str) -> None:
        if isinstance(value, str):
            for name, domain in domains.items():
                if domain == "worktile":
                    self.require(re.search(r"\b(?:count|last)\(\s*" + re.escape(name) + r"\s*\)", value) is None,
                                 path, "count/last on a dynamically refilled worktile domain is undefined")
        elif isinstance(value, dict):
            for k, v in value.items():
                self.dynamic_domain_queries(k, domains, path)
                self.dynamic_domain_queries(v, domains, path + "." + k)
        elif isinstance(value, list):
            for v in value:
                self.dynamic_domain_queries(v, domains, path)


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


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
        if match[2] is not None and not valid_index(match[2]):
            return None
        identities.append((match[1], len(args)))
    return verb, tuple(identities)


def parse_event(text: str) -> tuple[str | None, str | None, str]:
    text = re.sub(r"@(issue|done)$", "", text)
    match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)(?:\[(\*|[0-9]+)\])?/", text)
    if match:
        return match[1], match[2], text[match.end():]
    return None, None, text


def valid_index(text: str) -> bool:
    """Syntax only: mathematical function names remain upstream definitions."""
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except (SyntaxError, ValueError, RecursionError):
        return False

    def valid(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            return type(node.value) is int
        if isinstance(node, ast.Name):
            return bool(IDENTIFIER.fullmatch(node.id))
        if isinstance(node, ast.Tuple):
            return bool(node.elts) and all(valid(item) for item in node.elts)
        if isinstance(node, ast.UnaryOp):
            return isinstance(node.op, (ast.UAdd, ast.USub)) and valid(node.operand)
        if isinstance(node, ast.BinOp):
            return isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod)) and valid(node.left) and valid(node.right)
        if isinstance(node, ast.Call):
            return isinstance(node.func, ast.Name) and bool(node.args) and not node.keywords and all(valid(a) for a in node.args)
        if isinstance(node, ast.Subscript):
            return isinstance(node.value, ast.Name) and valid(node.slice)
        if isinstance(node, ast.Slice):
            return node.step is None and all(v is None or valid(v) for v in (node.lower, node.upper))
        return False

    return valid(tree.body)


def free_action_indices(text: str) -> set[str]:
    """Find variable uses, treating last(i) and the base of n[t,i] as domains."""
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


def validate_task(task: dict[str, Any]) -> list[Diagnostic]:
    """Call only after the concrete task has passed kernel.#Task in CUE."""
    check, parents = Checker(), {}
    for name, plan in task.items():
        if name != "root":
            parent = plan["meta_data"]["dispatch"].get("parent", "root")
            check.require(parent in task, name + ".meta_data.dispatch.parent", f"unknown parent {parent}")
            parents[name] = [parent]
    check.acyclic(parents, "meta_data.dispatch.parent")
    if len(task) == 1 and "dispatch" in task["root"]["meta_data"]:
        when = task["root"]["meta_data"]["dispatch"]["when"]
        check.require(" ".join(when.split()) == "Select for every input in the contract domain.",
                      "root.meta_data.dispatch.when", "a sole root covers the complete Contract domain")
    contract = task["root"]["contract"]["algorithm_requirement"]
    hardware = task["root"]["contract"]["hardware_specification"]
    check.contract(contract, "root.contract.algorithm_requirement")
    for name, plan in task.items():
        check.shapes(plan, name)
        check.math(plan, contract, name)
        check.schedule(plan, name)
        check.pipeline(plan, hardware, name)
    return check.errors
