"""Declared mathematical interfaces and references, without interpreting prose."""
from __future__ import annotations

import ast
from decimal import Decimal
from typing import Any

from ..diagnostics import IDENTIFIER, field_path
from ..pipeline_syntax import LOWER_NAME, literal_integer
from .common import CheckContext, concrete_conflict, indexed_values
from .values import is_repository, repository, semantic_reference

AREAS = ("CONTRACT.Algorithm_requirement", "CONTRACT.Hardware_specification",
         "DESIGN.Math_reshape", "DESIGN.Work_schedule", "DESIGN.Tile_pipeline")


def _constant_references(expression: str) -> set[str]:
    """Recognize simple value dependencies; leave unfamiliar notation to review.

    Pipeline index hints deliberately omit a subscript's object. Constants need
    that object, while local binders and external dotted names do not establish
    dependencies on similarly named Contract values.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, RecursionError):
        return set()
    names, pending = set(), [tree]
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            continue
        elif isinstance(node, ast.Call):
            pending.extend(node.args)
            pending.extend(keyword.value for keyword in node.keywords)
        elif isinstance(node, (ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp,
                               ast.Subscript, ast.Slice, ast.List, ast.Tuple,
                               ast.operator, ast.unaryop, ast.Load)):
            pending.extend(ast.iter_child_nodes(node))
        else:
            return set()
    return names


def contract_descriptors(algorithm: dict[str, Any], path: str) -> list:
    descriptors = []
    for group in ("Input", "Output"):
        descriptors.extend((obj, field_path(f"{path}.{group}", name))
                           for name, obj in algorithm.get(group, {}).items())
    for i, step in enumerate(algorithm.get("Compute", [])):
        descriptors.extend((obj, field_path(f"{path}.Compute[{i}].Symbol", name))
                           for name, obj in step.get("Symbol", {}).items())
    return descriptors


def _math_objects(plan):
    return {name: obj for rewrite in plan["DESIGN"]["Math_reshape"]["Rewrite"]
            for name, obj in rewrite.get("Symbol", {}).items()}


def _target(check, target, contract_objects, math_objects, path, *, contract_local=False, plan=None):
    """Resolve declared value objects; a semantic prose target remains unresolved."""
    if is_repository(target):
        repository(target, path, check.errors, fixed=True)
        return None
    math_names = ({rewrite["Name"] for rewrite in plan["DESIGN"]["Math_reshape"]["Rewrite"]}
                  if plan is not None else set())
    if "::" not in target:
        if not contract_local and target in math_names and target not in contract_objects and target not in math_objects:
            return None
        if IDENTIFIER.fullmatch(target):
            check.require(LOWER_NAME.fullmatch(target) is not None, path, "value reference names must be lowercase")
        objects = contract_objects if contract_local else {**contract_objects, **math_objects}
        if contract_local:
            check.require(target in objects, path, f"unknown Contract reference {target}")
        elif target not in objects:
            semantic_reference(target, path, check.errors, known=math_names)
        return objects.get(target)
    owner, name = (part.strip() for part in target.split("::", 1))
    area = next((area for area in AREAS if owner == area or owner.startswith(area + ".")), None)
    if not check.require(area is not None and bool(name), path, "reference needs a canonical upstream area and a target after ::"):
        return None
    if area == "CONTRACT.Algorithm_requirement" and owner == area:
        check.require(name in contract_objects, path, f"unknown shared Contract reference {name}")
        return contract_objects.get(name)
    if area == "DESIGN.Math_reshape" and owner == area:
        if name not in math_objects:
            semantic_reference(name, path, check.errors, known=math_names)
        return math_objects.get(name)
    if area == "CONTRACT.Hardware_specification":
        # Shared hardware may be explicitly skipped or owned by another node.
        # Published hardware paths are checked with that context in realization.
        return None
    if plan is not None and area in {"DESIGN.Work_schedule", "DESIGN.Tile_pipeline"}:
        if owner != area:
            name = owner[len(area) + 1:] + "." + name
        region = plan["DESIGN"][area.split(".")[-1]]
        name = name.removeprefix("Step.")
        step, separator, tail = name.partition(".")
        if step.isupper():
            if not check.require(step in region["Step"], path, f"unknown step {step}"):
                return None
            definition = region["Step"][step]
            if not separator:
                return definition
            if area == "DESIGN.Tile_pipeline" and tail.split(".", 1)[0] in definition["Role"]:
                tail = "Role." + tail
            return _published_path(check, definition, tail, path)
        if name.split(".", 1)[0] in region:
            return _published_path(check, region, name, path)
    # Work_schedule parameters and pipeline-local objects may be defined only in
    # Strategy/Describe. No pseudo symbol table is extracted from those strings.
    semantic_reference(name, path, check.errors, known=math_names)
    return None


def _property(value, attribute, contract_objects, math_objects, *, contract_local=False):
    """Follow known property references; return None when the value is not known."""
    seen = set()
    while isinstance(value, str) and value.startswith(("ref:", "bind:")):
        if value in seen:
            return None
        seen.add(value)
        target = value.split(":", 1)[1]
        suffix = "." + attribute
        if not target.endswith(suffix):
            return None
        target = target[:-len(suffix)]
        if "::" in target:
            owner, target = target.split("::", 1)
            if owner == "CONTRACT.Algorithm_requirement":
                objects = contract_objects
                contract_local = True
            elif owner == "DESIGN.Math_reshape":
                objects = math_objects
                contract_local = False
            else:
                return None
        else:
            objects = contract_objects if contract_local else {**contract_objects, **math_objects}
        obj = objects.get(target)
        if obj is None or attribute not in obj:
            return None
        value = obj[attribute]
    return None if isinstance(value, str) and value.startswith("unknown:") else value


def descriptor_properties(check: CheckContext, descriptors: list, contract_objects: dict[str, Any],
                          math_objects: dict[str, Any] | None = None, *, plan=None) -> None:
    math_objects = math_objects or {}
    locations = {id(descriptor): path for descriptor, path in descriptors}
    graph = {}
    for descriptor, path in descriptors:
        for attribute in ("shape", "dtype"):
            value = descriptor.get(attribute)
            if not isinstance(value, str) or not value.startswith(("ref:", "bind:")):
                continue
            target = value.split(":", 1)[1]
            if not target.endswith("." + attribute):
                continue  # Structure reports a property-name mismatch.
            target = target[:-(len(attribute) + 1)]
            obj = _target(check, target, contract_objects, math_objects, path + "." + attribute,
                          contract_local=".CONTRACT.Algorithm_requirement." in path, plan=plan)
            if obj is not None:
                check.require(isinstance(obj, dict) and attribute in obj, path + "." + attribute,
                              f"referenced object {target} has no {attribute} property")
                if id(obj) in locations:
                    graph[path + "." + attribute] = [locations[id(obj)] + "." + attribute]
    check.acyclic(graph, descriptors[0][1] if descriptors else "root", {node: node for node in graph})


def properties(check: CheckContext, plan: dict[str, Any], contract_objects: dict[str, Any], path: str) -> None:
    descriptors = contract_descriptors(plan.get("CONTRACT", {}).get("Algorithm_requirement", {}),
                                       path + ".CONTRACT.Algorithm_requirement")
    for i, rewrite in enumerate(plan["DESIGN"]["Math_reshape"]["Rewrite"]):
        descriptors.extend((obj, field_path(f"{path}.DESIGN.Math_reshape.Rewrite[{i}].Symbol", name))
                           for name, obj in rewrite.get("Symbol", {}).items())
    for step, definition in plan["DESIGN"]["Work_schedule"]["Step"].items():
        dec = definition["Decomposition"]
        blocks = {"Target.Single_tile": dec["Target"]["Single_tile"]}
        blocks.update({"Worktile." + key: value for key, value in dec.get("Worktile", {}).items()})
        for group, values in blocks.items():
            descriptors.extend((obj, field_path(f"{path}.DESIGN.Work_schedule.Step.{step}.Decomposition.{group}", name))
                               for name, obj in values.items())
    for step, definition in plan["DESIGN"]["Tile_pipeline"]["Step"].items():
        descriptors.extend((obj, field_path(f"{path}.DESIGN.Tile_pipeline.Step.{step}.Storage", name))
                           for name, obj in definition.get("Storage", {}).items())
    descriptor_properties(check, descriptors, contract_objects, _math_objects(plan), plan=plan)


def contract(check: CheckContext, algorithm: dict[str, Any], path: str) -> dict[str, Any]:
    objects, order, locations = {}, {}, {}
    constants = algorithm.get("Constant", {})
    names = set(constants)
    groups = [(algorithm[field], -1, field) for field in ("Input", "Output")]
    groups += [(step.get("Symbol", {}), i, f"Compute[{i}].Symbol")
               for i, step in enumerate(algorithm["Compute"])]
    for group, index, label in groups:
        for name, obj in group.items():
            location = field_path(f"{path}.{label}", name)
            check.require(name not in names, location, "duplicate Contract object name")
            names.add(name)
            objects[name], order[name], locations[name] = obj, index, location
            if label in ("Input", "Output") and obj["dtype"] == "python_none":
                check.require(obj["shape"] == [], location + ".shape",
                              "python_none is a singleton scalar; use shape: []")

    for attribute in ("shape", "dtype"):
        for name, obj in objects.items():
            value = obj[attribute]
            if isinstance(value, str) and value.startswith("ref:"):
                target = value[4:].rsplit(".", 1)[0]
                location = f"{locations[name]}.{attribute}"
                if check.require(target in objects, location, f"unknown Contract reference {target}"):
                    check.require(order[target] <= order[name], location,
                                  "reference precedes the defining Compute entry")

    extents = set()
    for name, obj in algorithm["Input"].items():
        location = field_path(path + ".Input", name)
        for i, axis in enumerate(obj["shape"]):
            key, size = next(iter(axis.items()))
            if isinstance(size, str) and IDENTIFIER.fullmatch(size) and size not in constants:
                check.require(LOWER_NAME.fullmatch(size) is not None, field_path(f"{location}.shape[{i}]", key),
                              "runtime extent names must be lowercase")
                check.require(size not in names, field_path(f"{location}.shape[{i}]", key),
                              "a runtime extent cannot also name a Contract object")
                extents.add(size)
        bounds = obj.get("range")
        if isinstance(bounds, list):
            endpoints = [integer if (integer := literal_integer(value)) is not None else value for value in bounds]
            if all(type(value) in (int, float, Decimal) for value in endpoints):
                lo, hi = endpoints
                check.require(lo <= hi, location + ".range", "range endpoints are reversed")
                default = obj.get("default")
                if type(default) in (int, float, Decimal):
                    check.require(lo <= default <= hi, location + ".default",
                                  "the literal default lies outside the declared input range")
        if obj.get("default") == "None":
            check.require(obj.get("nullable", False) or obj["dtype"] == "python_none", location + ".default",
                          "default None requires nullable: true or the python_none singleton type")
        if obj["dtype"] == "python_none" and "default" in obj:
            check.require(isinstance(obj["default"], str), location + ".default",
                          "python_none cannot have a numeric or boolean default")

    constant_graph = {}
    for name, value in constants.items():
        expression = value["value"]
        if not isinstance(expression, str):
            continue
        referenced = _constant_references(expression)
        location = field_path(path + ".Constant", name) + ".value"
        check.require(not referenced.intersection(set(objects) | extents), location,
                      "a Constant value cannot depend on runtime inputs, extents or computed objects")
        constant_graph[name] = sorted(referenced.intersection(constants))
    check.acyclic(constant_graph, path + ".Constant",
                  {name: field_path(path + ".Constant", name) + ".value" for name in constants})
    return {**constants, **objects}


def math(check: CheckContext, plan: dict[str, Any], algorithm: dict[str, Any], path: str) -> None:
    objects, locations, order = {}, {}, {}
    reserved = set(algorithm.get("Constant", {})) | set(algorithm["Input"]) | set(algorithm["Output"])
    for obj in algorithm["Input"].values():
        for axis in obj["shape"]:
            size = next(iter(axis.values()))
            if isinstance(size, str) and LOWER_NAME.fullmatch(size):
                reserved.add(size)
    for i, rewrite in enumerate(plan["DESIGN"]["Math_reshape"]["Rewrite"]):
        for name, obj in rewrite.get("Symbol", {}).items():
            location = field_path(f"{path}.DESIGN.Math_reshape.Rewrite[{i}].Symbol", name)
            check.require(name not in objects and name not in reserved, location,
                          "Symbol duplicates an existing mathematical definition")
            objects[name], locations[name], order[name] = obj, location, i
    contract_objects = {**algorithm.get("Constant", {}), **algorithm["Input"], **algorithm["Output"]}
    for step in algorithm["Compute"]:
        contract_objects.update(step.get("Symbol", {}))
    for attribute in ("shape", "dtype", "binding"):
        graph = {}
        for name, obj in objects.items():
            value = obj.get(attribute)
            if not isinstance(value, str) or not value.startswith(("ref:", "bind:")):
                continue
            target = value.split(":", 1)[1]
            if attribute != "binding":
                target = target[:-(len(attribute) + 1)]
            else:
                check.require(not target.endswith((".shape", ".dtype")), locations[name] + ".binding",
                              "binding references the whole object, not one property")
            source = _target(check, target, contract_objects, objects, locations[name] + "." + attribute, plan=plan)
            local = target.removeprefix("DESIGN.Math_reshape::")
            if local in objects:
                graph[name] = [local]
                if value.startswith("ref:"):
                    check.require(order[local] <= order[name], locations[name] + "." + attribute,
                                  "reference precedes the defining Rewrite entry")
            if attribute == "binding" and isinstance(source, dict):
                for prop in ("shape", "dtype"):
                    if prop not in obj or prop not in source:
                        continue
                    own = _property(obj[prop], prop, contract_objects, objects)
                    bound = _property(source[prop], prop, contract_objects, objects,
                                      contract_local=target.startswith("CONTRACT.Algorithm_requirement::"))
                    if own is not None and bound is not None:
                        check.require(not concrete_conflict(own, bound, prop), locations[name] + "." + prop,
                                      "declared property conflicts with the bound object's known property")
        if attribute == "binding":
            check.acyclic(graph, path + ".DESIGN.Math_reshape",
                          {name: location + ".binding" for name, location in locations.items()})

    for step, definition in plan["DESIGN"]["Work_schedule"]["Step"].items():
        dec = definition["Decomposition"]
        blocks = {"Target.Single_tile": dec["Target"]["Single_tile"]}
        blocks.update({"Worktile." + key: value for key, value in dec.get("Worktile", {}).items()})
        for group, entries in blocks.items():
            for name, block in entries.items():
                if name in objects:
                    for attribute in ("shape", "dtype"):
                        check.require(attribute in objects[name], locations[name] + "." + attribute,
                                      "a value consumed by another region must publish shape and dtype", missing=True)
                upstream = objects.get(name, contract_objects.get(name))
                if upstream is not None and "dtype" in upstream:
                    own = _property(block["dtype"], "dtype", contract_objects, objects)
                    expected = _property(upstream["dtype"], "dtype", contract_objects, objects,
                                         contract_local=name not in objects)
                    if own is not None and expected is not None:
                        check.require(not concrete_conflict(own, expected, "dtype"),
                                      field_path(f"{path}.DESIGN.Work_schedule.Step.{step}.Decomposition.{group}", name) + ".dtype",
                                      "data-block dtype conflicts with the declared upstream value")
    # Handoff values, storage Views and explicit mathematical references are
    # also published interfaces. Unknown prose-defined values are left to review.
    used = set(plan["DESIGN"]["Work_schedule"]["Flow"].get("Handoff", {}))
    for definition in plan["DESIGN"]["Tile_pipeline"]["Step"].values():
        for storage in definition.get("Storage", {}).values():
            used.update(value.split("(", 1)[0] for value in storage.get("view", {}))
    for name in used.intersection(objects):
        for attribute in ("shape", "dtype"):
            check.require(attribute in objects[name], locations[name] + "." + attribute,
                          "a value consumed by another region must publish shape and dtype", missing=True)

    # These weak references may name formula/strategy-defined conditions. Check
    # only their outer form after giving registered full names first priority.
    known = set(contract_objects) | set(objects)
    known.update(entry["Name"] for entry in algorithm["Compute"])
    known.update(entry["Name"] for entry in plan["DESIGN"]["Math_reshape"]["Rewrite"])

    def reference(value, location):
        if value in known:
            return
        target = value.split(":", 1)[1] if value.startswith(("ref:", "bind:")) else value
        if target in known:
            return
        _target(check, target, contract_objects, objects, location, plan=plan)

    work = plan["DESIGN"]["Work_schedule"]
    for step, definition in work["Step"].items():
        location = f"{path}.DESIGN.Work_schedule.Step.{step}"
        assignment = definition["Schedule"]["Worker_assignment"]
        for value, rp in indexed_values(assignment["Exit"], location + ".Schedule.Worker_assignment.Exit"):
            reference(value, rp)
        reduction = definition["Decomposition"]["Target"].get("Tile_reduction", {})
        for field in ("input", "output", "method"):
            if field in reduction:
                for value, rp in indexed_values(reduction[field], location + ".Decomposition.Target.Tile_reduction." + field):
                    reference(value, rp)
    for name, handoff in work["Flow"].get("Handoff", {}).items():
        if isinstance(handoff.get("initialize"), str):
            reference(handoff["initialize"], field_path(path + ".DESIGN.Work_schedule.Flow.Handoff", name) + ".initialize")


def _published_path(check, value, target, path):
    """Follow explicit mapping paths, treating an external dotted key as one key."""
    while target:
        if not isinstance(value, dict):
            check.require(False, path, "reference continues past an existing scalar or list")
            return None
        if target in value:
            return value[target]
        # A registered key may itself contain dots, spaces or parentheses.
        head = next((key for key in sorted(value, key=len, reverse=True) if target.startswith(key + ".")), None)
        if not check.require(head is not None, path, "unknown declared target " + target.split(".", 1)[0]):
            return None
        value = value[head]
        target = target[len(head) + 1:]
    return value


def realization(check: CheckContext, plan: dict, algorithm: dict, hardware: dict | None, path: str) -> None:
    """Validate published paths and evidence forms; never establish evidence truth."""
    body = plan.get("REALIZATION", {})
    if not body:
        return
    math_objects = _math_objects(plan)
    contract_objects = {**algorithm.get("Constant", {}), **algorithm["Input"], **algorithm["Output"]}
    for step in algorithm["Compute"]:
        contract_objects.update(step.get("Symbol", {}))
    contract_names = {step["Name"] for step in algorithm["Compute"]}
    math_names = {step["Name"] for step in plan["DESIGN"]["Math_reshape"]["Rewrite"]}
    work = plan["DESIGN"]["Work_schedule"]
    tile = plan["DESIGN"]["Tile_pipeline"]
    unresolved = set()

    def semantic_item(reference, location):
        if reference not in math_names and ("[" in reference or "]" in reference):
            check.require(False, location, "item references use object names and published paths, not list indices")
            return
        semantic_reference(reference, location, check.errors, known=math_names)

    def item(reference, location):
        # Registered names and mapping keys may themselves contain brackets.
        # Only reject index notation after those complete identities are tried.
        if "::" not in reference:
            candidates = []
            if reference in contract_objects or reference in contract_names:
                candidates.append("CONTRACT.Algorithm_requirement::" + reference)
            if reference in math_objects or reference in math_names:
                candidates.append("DESIGN.Math_reshape::" + reference)
            check.require(len(candidates) <= 1, location, "bare item is ambiguous; qualify its upstream owner")
            if not candidates:
                semantic_item(reference, location)
                unresolved.add(reference)
            return candidates[0] if len(candidates) == 1 else reference
        owner, target = (part.strip() for part in reference.split("::", 1))
        area = next((area for area in AREAS if owner == area or owner.startswith(area + ".")), None)
        if not check.require(area is not None and bool(target), location, "item needs a canonical upstream owner and an object after ::"):
            return reference
        if owner != area:
            target = owner[len(area) + 1:] + "." + target
        if area == "CONTRACT.Algorithm_requirement":
            if target in contract_names:
                return area + "::" + target
            group, separator, tail = target.partition(".")
            if separator and group in {"Constant", "Input", "Output"} and tail.split(".", 1)[0] in algorithm.get(group, {}):
                target = tail
            base = target.split(".", 1)[0]
            if base in contract_objects:
                suffix = target[len(base):].lstrip(".")
                if suffix:
                    _published_path(check, contract_objects[base], suffix, location)
            else:
                _published_path(check, algorithm, target, location)
        elif area == "CONTRACT.Hardware_specification":
            if hardware is not None:
                _published_path(check, hardware, target, location)
            else:
                unresolved.add(area + "::" + target)
        elif area == "DESIGN.Math_reshape":
            base, _, suffix = target.partition(".")
            if base in math_objects and suffix:
                _published_path(check, math_objects[base], suffix, location)
            if base not in math_objects and target not in math_names:
                semantic_item(target, location)
                unresolved.add(area + "::" + target)
            # A Rewrite name or a formula-defined operation is semantic, not
            # necessarily a registered Symbol; its correspondence needs review.
        else:
            region = work if area == "DESIGN.Work_schedule" else tile
            target = target.removeprefix("Step.")
            step, separator, rest = target.partition(".")
            if step.isupper():
                if check.require(step in region["Step"], location, f"unknown step {step}"):
                    definition = region["Step"][step]
                    if separator:
                        if area == "DESIGN.Tile_pipeline" and rest.split(".", 1)[0] in definition["Role"]:
                            rest = "Role." + rest
                        _published_path(check, definition, rest, location)
                        target = step + "." + rest
            elif target.split(".", 1)[0] in region:
                _published_path(check, region, target, location)
            else:
                semantic_item(target, location)
                unresolved.add(area + "::" + target)
        return area + "::" + target

    preferred = set()
    for kind, entries in body.items():
        seen = []
        for i, row in enumerate(entries):
            location = f"{path}.REALIZATION.{kind}[{i}]"
            targets = [item(reference, rp) for reference, rp in indexed_values(row["Item"], location + ".Item")]
            check.require(len(targets) == len(set(targets)), location + ".Item",
                          "item lists cannot repeat the same upstream object")
            signature = (tuple(sorted(targets)), row["Choice"])
            check.require(signature not in seen, location, "duplicate implementation choice for the same item")
            seen.append(signature)
            if kind == "Prefer":
                preferred.update(targets)
    for i, row in enumerate(body.get("Feedback", [])):
        location = f"{path}.REALIZATION.Feedback[{i}].Item"
        for reference, rp in indexed_values(row["Item"], location):
            target = item(reference, rp)
            # Formula/strategy-defined objects cannot be compared by spelling
            # alone. Only reject a mismatch when the involved targets resolve.
            check.require(target in preferred or bool(preferred) and
                          (target in unresolved or bool(preferred & unresolved)), rp,
                          "Feedback must refer to an object covered by Prefer")
