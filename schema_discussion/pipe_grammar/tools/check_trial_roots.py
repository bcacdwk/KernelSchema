#!/usr/bin/env python3
"""Check the v0.2 control notation and compare finite root action traces.

This is an experiment checker, not the schema validator or a GPU correctness test.
Integer domains and action identities use the existing schema tools. Old blocks
and new controls are expanded by separate walkers; order and presence must agree.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from copy import deepcopy
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
TRIAL = REPO / "schema_discussion/pipe_grammar/trial_roots"
sys.path.insert(0, str(REPO))

from schema_template.tools.pipeline.actions import instantiate, operands
from schema_template.tools.pipeline.expressions import Domains
from schema_template.tools.pipeline_syntax import parse_action, split_top
from schema_template.tools.pipeline.graph import Builder
from schema_template.tools.pipeline.model import ExpansionError, GraphError

RULES_VERSION = "pipe-control-v0.2"


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def read_yaml(path):
    return yaml.load(path.read_text(), Loader=UniqueLoader)


def digest(value):
    return hashlib.sha256(value).hexdigest()


class SampleDomains(Domains):
    """The existing bounded evaluator, plus explicit finite sample bindings."""

    def __init__(self, definitions, case):
        self.case = case
        super().__init__(definitions, case.get("parameters", {}), case.get("worktiles", 1))

    def extend(self, name, bounds):
        return SampleDomains(self.scope.extend(name, bounds), self.case)

    def _evaluate(self, node, env):
        if isinstance(node, ast.Name) and node.id not in env:
            for outer, names in self.case.get("per_work", {}).items():
                if outer in env and node.id in names:
                    return names[node.id][env[outer]]
            alias = self.case.get("aliases", {}).get(node.id)
            if alias is not None:
                return self.evaluate(ast.parse(alias, mode="eval").body, env)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            table = self.case.get("functions", {}).get(node.func.id)
            if table is not None:
                args = [self.integer(ast.unparse(arg), env) for arg in node.args]
                return table[",".join(map(str, args))]
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return int(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return int(not self.evaluate(node.operand, env))
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return int(all(self.evaluate(part, env) for part in node.values))
            if isinstance(node.op, ast.Or):
                return int(any(self.evaluate(part, env) for part in node.values))
        if isinstance(node, ast.Compare):
            values = [self.evaluate(node.left, env)] + [self.evaluate(n, env) for n in node.comparators]
            for left, op, right in zip(values, node.ops, values[1:]):
                if isinstance(op, ast.Eq): ok = left == right
                elif isinstance(op, ast.NotEq): ok = left != right
                elif isinstance(op, ast.Lt): ok = left < right
                elif isinstance(op, ast.LtE): ok = left <= right
                elif isinstance(op, ast.Gt): ok = left > right
                elif isinstance(op, ast.GtE): ok = left >= right
                else: raise ValueError(f"unsupported comparison {ast.unparse(node)}")
                if not ok:
                    return 0
            return 1
        return super()._evaluate(node, env)


def control(key):
    if not isinstance(key, str):
        raise ValueError(f"control key must be a string: {key!r}")
    match = re.fullmatch(r"(for|if|while)\((.+)\)", key)
    if not match:
        raise ValueError(f"not a v0.2 control: {key}")
    kind, arg = match.groups()
    if not arg.strip() or split_top(arg, ",") is None:
        raise ValueError(f"unbalanced or empty control argument: {key}")
    return kind, arg.strip()


def while_parts(arg):
    parts = split_top(arg, ",")
    if parts is None or len(parts) != 2:
        raise ValueError("while requires an iteration name and a condition")
    index, condition = (p.strip() for p in parts)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", index) or not condition:
        raise ValueError(f"invalid while header: {arg}")
    ast.parse(condition, mode="eval")
    return index, condition


def binding(arg, domains):
    parts = split_top(arg, "=")
    if parts is None:
        raise ValueError(f"invalid domain: {arg}")
    if len(parts) == 1:
        node = ast.parse(arg, mode="eval").body
        if not (isinstance(node, ast.Name) or
                isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)):
            raise ValueError(f"for requires a named domain or a slice: {arg}")
        return arg, domains
    if len(parts) != 2 or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", parts[0].strip()):
        raise ValueError(f"invalid local domain: {arg}")
    name = parts[0].strip()
    bounds = split_top(parts[1], ":")
    if bounds is None or len(bounds) != 2 or any(not b.strip() for b in bounds):
        raise ValueError(f"local domain requires lower:upper: {arg}")
    for bound in bounds:
        ast.parse(bound.strip(), mode="eval")
    return name, domains.extend(name, [b.strip() for b in bounds])


def selector_index(selector):
    node = ast.parse(selector, mode="eval").body
    return node.id if isinstance(node, ast.Name) else node.value.id


def reject_while_domain_queries(expression, while_indices):
    for part in ast.walk(ast.parse(expression, mode="eval")):
        if (isinstance(part, ast.Call) and isinstance(part.func, ast.Name)
                and part.func.id in {"count", "last"} and len(part.args) == 1
                and isinstance(part.args[0], ast.Name) and part.args[0].id in while_indices):
            raise ValueError("while iteration indices have no declared finite count/last domain")


def syntax(items, path="pipe", while_indices=frozenset(), domains=None, bound=frozenset()):
    domains = SampleDomains({}, {}) if domains is None else domains
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path} must be a nonempty list")
    for number, item in enumerate(items):
        here = f"{path}[{number}]"
        if isinstance(item, str):
            if parse_action(item) is None:
                raise ValueError(f"invalid action at {here}: {item}")
            for _, arguments in operands(item)[1]:
                for argument in arguments:
                    reject_while_domain_queries(argument, while_indices)
        elif isinstance(item, dict) and len(item) == 1:
            key, body = next(iter(item.items()))
            kind, arg = control(key)
            inner_while = while_indices
            inner_domains, inner_bound = domains, bound
            if kind == "if":
                ast.parse(arg, mode="eval")
                reject_while_domain_queries(arg, while_indices)
            elif kind == "while":
                index, condition = while_parts(arg)
                if index in bound:
                    raise ValueError(f"{here}: while iteration {index} shadows an active index")
                inner_while = while_indices | {index}
                inner_bound = bound | {index}
                reject_while_domain_queries(condition, inner_while)
            else:
                selector, inner_domains = binding(arg, domains)
                index = selector_index(selector)
                if index in bound:
                    raise ValueError(f"{here}: for index {index} shadows an active index")
                inner_bound = bound | inner_domains.scope.coordinates(index)
                local_parts = split_top(arg, "=")
                expressions = inner_domains.definitions[index] if len(local_parts) == 2 else [arg]
                for expression in expressions:
                    reject_while_domain_queries(str(expression), while_indices)
            syntax(body, here + "." + key, inner_while, inner_domains, inner_bound)
        else:
            raise ValueError(f"{here} must be an action or a single-key control")


def control_counts(items):
    counts = {"for": 0, "if": 0, "while": 0}
    for item in items:
        if isinstance(item, dict):
            key, body = next(iter(item.items()))
            counts[control(key)[0]] += 1
            for kind, value in control_counts(body).items():
                counts[kind] += value
    return counts


def old_trace(block, domains, env):
    repeat = block.get("repeat")
    if isinstance(repeat, dict):
        name, bounds = next(iter(repeat.items()))
        domains = domains.extend(name, bounds)
        repeat = name
    for region in ("prologue", "steady", "epilogue"):
        contexts = [env]
        if region == "steady" and repeat is not None:
            contexts = [dict(env, **item) for item in domains.select(repeat, env)]
        for context in contexts:
            for item in block.get(region, []):
                if isinstance(item, dict):
                    yield from old_trace(item, domains, context)
                else:
                    yield instantiate(item, domains, context)


def new_trace(items, domains, env):
    for item in items:
        if isinstance(item, str):
            yield instantiate(item, domains, env)
            continue
        key, body = next(iter(item.items()))
        kind, arg = control(key)
        if kind == "if":
            if domains.integer(arg, env):
                yield from new_trace(body, domains, env)
        elif kind == "for":
            name, local = binding(arg, domains)
            index = selector_index(name)
            if index in env:
                raise ValueError(f"for index {index} shadows an active index")
            for context in local.select(name, env):
                conflicts = {key: (env[key], value) for key, value in context.items()
                             if key in env and env[key] != value}
                if conflicts:
                    raise ValueError(f"for({arg}) has a coordinate conflict with outer indices: {conflicts}")
                yield from new_trace(body, local, dict(env, **context))
        else:
            index, condition = while_parts(arg)
            if index in env:
                raise ValueError(f"while iteration {index} shadows an active index")
            budget = domains.case.get("while_budget", 1000)
            local = domains.extend(index, [0, budget + 1])
            iteration = 0
            while True:
                context = dict(env, **{index: iteration})
                if not local.integer(condition, context):
                    break
                if iteration >= budget:
                    raise ExpansionError(f"while({arg}) exhausted its finite-check budget {budget}; no termination claim")
                yield from new_trace(body, local, context)
                iteration += 1


def compare(old, new):
    if old == new:
        return None
    limit = min(len(old), len(new))
    index = next((i for i in range(limit) if old[i] != new[i]), limit)
    return {"index": index, "old": old[index:index+3], "new": new[index:index+3],
            "old_count": len(old), "new_count": len(new)}


def lower_controls(items, serial=None):
    """Private adapter to the existing finite graph model, not author syntax."""
    serial = [0] if serial is None else serial
    lowered = []
    for item in items:
        if isinstance(item, str):
            lowered.append(item)
            continue
        key, body = next(iter(item.items()))
        kind, arg = control(key)
        content = lower_controls(body, serial)
        if kind == "while":
            raise ExpansionError("Dynamic while graphs require a separate finite execution witness")
        if kind == "if":
            serial[0] += 1
            repeat = {f"_trial_condition_{serial[0]}": [0, f"not not ({arg})"]}
        else:
            name, local = binding(arg, SampleDomains({}, {}))
            repeat = {name: local.definitions[name]} if local.definitions else name
        lowered.append({"repeat": repeat, "steady": content})
    return lowered


class SampleBuilder(Builder):
    """Preserve the existing graph model; make v0.2 target guards authoritative."""

    def __init__(self, pid, phase, worker, case, *, control_targets=False):
        self.case = case
        self.control_targets = control_targets
        super().__init__(pid, phase, worker, case.get("parameters", {}), case.get("worktiles", 1))
        self.domains = SampleDomains(self.domains.scope, case)
        self.domain_cache = {id(self.domains.scope): self.domains}
        self.absent_targets = 0

    def domains_for(self, scope):
        if id(scope) not in self.domain_cache:
            self.domain_cache[id(scope)] = SampleDomains(scope, self.case)
        return self.domain_cache[id(scope)]

    def rule_domains(self, references, *, indices=None):
        raw = super().rule_domains(references, indices=indices)
        return SampleDomains(raw.scope, self.case)

    def resolve_edge(self, left, right, role, member, env, domains):
        if not self.control_targets:
            return super().resolve_edge(left, right, role, member, env, domains)
        # Both declarations/signatures were checked by explicit_edges before
        # this point. Enumerate targets from control declarations, not logs of
        # completed operations. Do not apply this suppression to predecessors.
        target = self.bind_reference(left, role, member, env, domains)
        operations = [self.lookup.get((target.role, selected, target.action))
                      for selected in target.members]
        if not any(operations):
            self.absent_targets += 1
            return [], []
        if not all(operations):
            raise GraphError("inconsistent member coverage of a dependency target")
        predecessor = self.bind_reference(right, role, member, env, domains)
        sources = self.resolve(predecessor, env, domains)
        return [operation.event(target.event) for operation in operations], sources


def phase_graph_comparison(pid, old, projected, worker, case):
    before = SampleBuilder(pid, old, worker, case).build()
    phase = deepcopy(old)
    phase["domain"] = projected.get("domain", {})
    for role, body in projected["role"].items():
        phase["role"][role]["pipe"] = {"steady": lower_controls(body["pipe"])}
        if "waits_for" in body:
            phase["role"][role]["waits_for"] = body["waits_for"]
    builder = SampleBuilder(pid, phase, worker, case, control_targets=True)
    after = builder.build()
    if set(before.operations) != set(after.operations):
        raise ValueError(f"{pid}: event-operation identity set changed")
    if set(before.edges) != set(after.edges):
        removed = sorted(set(before.edges) - set(after.edges))[:3]
        added = sorted(set(after.edges) - set(before.edges))[:3]
        raise ValueError(f"{pid}: finite dependency graph changed; removed={removed}, added={added}")
    if before.milestones != after.milestones:
        raise ValueError(f"{pid}: milestone instances changed")
    return {"operations": len(after.operations), "edges": len(after.edges),
            "absent_target_rule_instances": builder.absent_targets,
            "baseline_acyclic": not before.cycle,
            "edge_sha256": digest(json.dumps(sorted(after.edges)).encode())}


def check_controls():
    """Small semantic witnesses for while and asymmetric guarded dependencies."""
    checked = []
    def sequence_case(name, pipe, case, expected, definitions=None):
        domains = SampleDomains(definitions or {}, case)
        syntax(pipe, domains=domains)
        got = list(new_trace(pipe, domains, {}))
        if got != expected:
            raise ValueError(f"control witness {name}: {got} != {expected}")
        if len(set(got)) != len(got):
            raise ValueError(f"control witness {name}: duplicate event identity")
        checked.append(name)

    sequence_case("sibling_loops_may_reuse_an_index",
                  [{"for(i=0:2)": ["load:A(i)"]}, {"for(i=0:2)": ["load:B(i)"]}], {},
                  ["load:A(0)", "load:A(1)", "load:B(0)", "load:B(1)"])
    flat_definitions = {"t": "worktile", "i": [0, 2], "n": {"flatten": ["t", "i"]}}
    flat_case = {"worktiles": 2}
    sequence_case("flatten_may_recover_matching_outer_coordinates",
                  [{"for(t)": [{"for(n[t])": ["publish:X(t,n)"]}]}], flat_case,
                  ["publish:X(0,0)", "publish:X(0,1)", "publish:X(1,2)", "publish:X(1,3)"],
                  flat_definitions)
    for name, pipe in [
        ("reject_nested_for_rebinding_even_in_empty_domain",
         [{"for(i=0:0)": [{"for(i=0:2)": ["load:X(i)"]}]}]),
        ("reject_while_rebinding_an_outer_for_index",
         [{"for(i=0:1)": [{"while(i, False)": ["load:X(i)"]}]}]),
        ("reject_rebinding_a_flatten_coordinate",
         [{"for(t)": [{"for(n[t])": [{"for(i)": ["load:X(i)"]}]}]}]),
    ]:
        try:
            syntax(pipe, domains=SampleDomains(flat_definitions, flat_case))
        except ValueError as error:
            if "shadows an active index" not in str(error):
                raise
            checked.append(name)
        else:
            raise ValueError(f"index rebinding was accepted: {name}")
    try:
        list(new_trace([{"for(t)": [{"for(n)": ["publish:X(t,n)"]}]}],
                       SampleDomains(flat_definitions, flat_case), {}))
    except ValueError as error:
        if "coordinate conflict" not in str(error):
            raise
        checked.append("reject_conflicting_flatten_coordinates_instead_of_overwriting_or_filtering")
    else:
        raise ValueError("conflicting flatten coordinates were silently accepted")

    pipe = ["load:Initial", {"while(r, Continue(r))": ["compute:State(r+1)", "publish:Progress(r)"]}, "store:Final"]
    sequence_case("while_false_first_keeps_surrounding_actions", pipe,
                  {"functions": {"Continue": {"0": 0}}}, ["load:Initial", "store:Final"])
    sequence_case("while_feedback_three_rounds", pipe,
                  {"functions": {"Continue": {"0": 1, "1": 1, "2": 1, "3": 0}}, "while_budget": 3},
                  ["load:Initial", "compute:State(1)", "publish:Progress(0)", "compute:State(2)",
                   "publish:Progress(1)", "compute:State(3)", "publish:Progress(2)", "store:Final"])
    nested = [{"for(t=0:2)": ["load:State(t)", {"while(r, More(t,r))": ["publish:Step(t,r)"]}, "store:Final(t)"]}]
    sequence_case("while_index_resets_per_outer_iteration", nested,
                  {"functions": {"More": {"0,0": 0, "1,0": 1, "1,1": 1, "1,2": 0}}},
                  ["load:State(0)", "store:Final(0)", "load:State(1)", "publish:Step(1,0)",
                   "publish:Step(1,1)", "store:Final(1)"])
    try:
        list(new_trace([{"while(r, True)": ["publish:Step(r)"]}], SampleDomains({}, {"while_budget": 3}), {}))
    except ExpansionError as error:
        if "no termination claim" not in str(error):
            raise
        checked.append("while_budget_exhaustion_is_not_successful_exit")
    else:
        raise ValueError("nonterminating while was silently accepted")

    invalid = [[{"steady(i)": ["load:X(i)"]}], [{"prologue": ["load:X"]}],
               [{"for(i)": {"do": ["load:X(i)"]}}],
               [{"if(P)": ["load:X"], "else": ["load:Y"]}], [{"while(P)": ["load:X"]}],
               [{"while(r, r < 2)": ["publish:X(count(r))"]}]]
    for number, tree in enumerate(invalid):
        try:
            syntax(tree)
        except ValueError:
            checked.append(f"reject_noncanonical_syntax_{number}")
        else:
            raise ValueError(f"invalid control syntax accepted: {tree}")

    info = {"member": "warp[0:1]", "execution_mode": "serial"}
    def guarded_phase(consumer_guard=True, reference="Producer/publish:X(0)"):
        consumer = [{"if(P)": ["compute:Y(0)"]}] if consumer_guard else ["compute:Y(0)"]
        return {"role": {
            "Producer": {"info": info, "pipe": {"steady": lower_controls([{"if(P)": ["publish:X(0)"]}])}},
            "Consumer": {"info": info, "pipe": {"steady": lower_controls(consumer)},
                         "waits_for": {"data": {"compute:Y(0)": [reference]}}}}}
    worker = {"unit": "CTA", "unit_count": 1}
    for flag in [0, 1]:
        graph = SampleBuilder("guard", guarded_phase(), worker, {"parameters": {"P": flag}}, control_targets=True).build()
        if len(graph.operations) != 2 * flag:
            raise ValueError("guarded action presence is wrong")
        checked.append(f"target_guard_controls_dependency_presence_{flag}")
    for name, phase in [
        ("present_consumer_missing_producer_is_error", guarded_phase(False)),
        ("misspelled_producer_rejected_even_when_target_absent", guarded_phase(reference="Producer/publish:Typo(0)")),
        ("unknown_role_rejected_even_when_target_absent", guarded_phase(reference="Missing/publish:X(0)")),
        ("wrong_signature_rejected_even_when_target_absent", guarded_phase(reference="Producer/publish:X")),
    ]:
        try:
            SampleBuilder("guard", phase, worker, {"parameters": {"P": 0}}, control_targets=True).build()
        except GraphError:
            checked.append(name)
        else:
            raise ValueError(f"guard error was silently discarded: {name}")
    return {"passed": True, "witnesses": checked,
            "scope": "Control flow uses supplied versioned predicate observations; it does not execute action implementations or establish general termination."}


def check_operator(name, entry, cases):
    path = TRIAL / (name + ".yaml")
    report = TRIAL / (name + ".md")
    if not path.is_file() or not report.is_file():
        raise ValueError("missing trial YAML or report")
    draft = read_yaml(path)
    meta = draft["experiment"]
    if (meta["operator"], meta["rules"], meta["baseline"]) != (name, RULES_VERSION, entry["baseline"]):
        raise ValueError("experiment identity/rules/baseline mismatch")
    permitted_sources = {f["snapshot"] for f in entry["sources"]}
    if not meta.get("kernel_sources") or not set(meta["kernel_sources"]) <= permitted_sources:
        raise ValueError("kernel_sources must identify frozen source files in the manifest")
    baseline = read_yaml(REPO / entry["baseline"])
    old = baseline["design"]["tile_pipeline"]["phase"]
    new = draft["tile_pipeline"]["phase"]
    if set(old) != set(new):
        raise ValueError("phase coverage differs from root")
    result = {"roles": 0, "role_samples": 0, "cases": [], "dependency_overrides": [],
              "domain_changes": {}, "controls": {"for": 0, "if": 0, "while": 0},
              "graph_comparisons": 0}
    graph_phases = set()
    for phase in old:
        if set(old[phase]["role"]) != set(new[phase]["role"]):
            raise ValueError(f"role coverage differs in {phase}")
        previous = old[phase].get("domain", {})
        current = new[phase].get("domain", {})
        for key in previous.keys() & current.keys():
            if previous[key] != current[key]:
                raise ValueError(f"shared domain {phase}.{key} was redefined")
        if previous != current:
            result["domain_changes"][phase] = {"removed": sorted(previous.keys() - current.keys()),
                                                 "added": sorted(current.keys() - previous.keys())}
            graph_phases.add(phase)
        for role, body in new[phase]["role"].items():
            syntax(body["pipe"], phase + "." + role, domains=SampleDomains(current, {}))
            for kind, count in control_counts(body["pipe"]).items():
                result["controls"][kind] += count
            if set(body) - {"pipe", "waits_for"}:
                raise ValueError(f"unexpected projection fields for {phase}.{role}: {set(body)}")
            if "waits_for" in body:
                result["dependency_overrides"].append(phase + "." + role)
                graph_phases.add(phase)
            result["roles"] += 1
    for case in cases:
        summary = {"name": case["name"], "role_traces": {}, "changed_phase_graphs": {}}
        for phase in old:
            before = SampleDomains(old[phase].get("domain", {}), case)
            after = SampleDomains(new[phase].get("domain", {}), case)
            for role in old[phase]["role"]:
                a = list(old_trace(old[phase]["role"][role]["pipe"], before, {}))
                b = list(new_trace(new[phase]["role"][role]["pipe"], after, {}))
                difference = compare(a, b)
                if difference:
                    raise ValueError(f"{case['name']} {phase}.{role}: {json.dumps(difference)}")
                if len(set(b)) != len(b):
                    raise ValueError(f"{case['name']} {phase}.{role}: duplicate action identity (also in baseline)")
                summary["role_traces"][phase + "." + role] = {
                    "actions": len(b), "sha256": digest("\n".join(b).encode()),
                }
                result["role_samples"] += 1
            if phase in graph_phases:
                worker = baseline["design"]["work_schedule"]["phase"][phase]["mapping"]["worker"]
                summary["changed_phase_graphs"][phase] = phase_graph_comparison(phase, old[phase], new[phase], worker, case)
                result["graph_comparisons"] += 1
        result["cases"].append(summary)
    result["yaml_sha256"] = digest(path.read_bytes())
    result["report_sha256"] = digest(report.read_bytes())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", action="append", help="check just this completed operator")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--controls-only", action="store_true", help="check control semantics without waiting for operator files")
    args = parser.parse_args()
    manifest = json.loads((TRIAL / "manifest.json").read_text())
    casefile = json.loads((TRIAL / "cases.json").read_text())
    result = {"format": "kernelschema.pipe_trial.validation.v0.2", "operators": {}, "errors": [],
              "limits": "Finite action-instance order/presence comparison; changed domain/dependency phases also compare finite graphs using the existing model. Control witnesses supply predicate observations, not action implementations. No general termination, whole-dispatch, source-correctness, or GPU proof."}
    try:
        result["control_semantics"] = check_controls()
        print(f"PASS control semantics: {len(result['control_semantics']['witnesses'])} witnesses")
    except Exception as error:
        result["errors"].append("control semantics: " + str(error))
        print("FAIL " + result["errors"][-1])
    for relative, expected in manifest["protected_inputs"].items():
        if digest((REPO / relative).read_bytes()) != expected:
            result["errors"].append("protected input changed: " + relative)
    if digest((REPO / manifest["rules"]).read_bytes()) != manifest["rules_sha256"]:
        result["errors"].append("frozen trial rules changed")
    selected = [] if args.controls_only else args.operator or list(manifest["operators"])
    for name in selected:
        try:
            entry = manifest["operators"][name]
            for source in entry["sources"]:
                if digest((REPO / source["snapshot"]).read_bytes()) != source["sha256"]:
                    raise ValueError("frozen source changed: " + source["snapshot"])
            result["operators"][name] = check_operator(name, entry, casefile["operators"][name])
            checked = result["operators"][name]
            print(f"PASS {name}: {checked['roles']} roles, {len(checked['cases'])} cases, {checked['role_samples']} role traces, {checked['graph_comparisons']} changed-phase graphs")
        except Exception as error:
            result["errors"].append(name + ": " + str(error))
            print("FAIL " + result["errors"][-1])
    result["passed"] = not result["errors"]
    result["complete_ten_roots"] = set(result["operators"]) == set(manifest["operators"])
    result["checker_sha256"] = digest(Path(__file__).read_bytes())
    result["cases_sha256"] = digest((TRIAL / "cases.json").read_bytes())
    if args.write_report:
        (TRIAL / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for error in result["errors"]:
        if error.startswith(("protected input", "frozen trial")):
            print("FAIL " + error)
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
