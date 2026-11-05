"""Contract and mathematical symbol names, property references and interfaces."""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from ..diagnostics import IDENTIFIER, field_path
from .common import CheckContext
from .values import literal_integer


def contract_descriptors(algorithm: dict[str, Any], path: str) -> list:
    descriptors = []
    for group in ("input", "output"):
        descriptors.extend((obj, f"{path}.{group}.{name}")
                           for name, obj in algorithm.get(group, {}).items())
    for i, step in enumerate(algorithm.get("compute", [])):
        descriptors.extend((obj, f"{path}.compute[{i}].symbol.{name}")
                           for name, obj in step.get("symbol", {}).items())
    return descriptors


def properties(check: CheckContext, plan: dict[str, Any], contract_objects: dict[str, Any], path: str) -> None:
    # Visit descriptor positions, not every mapping key spelled "shape":
    # that spelling is also a valid domain, milestone or object name.
    descriptors = contract_descriptors(plan.get("contract", {}).get("algorithm_requirement", {}),
                                       path + ".contract.algorithm_requirement")
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
        for name, resource in phase.get("resource", {}).items():
            descriptors.extend((view, field_path(f"{path}.design.tile_pipeline.phase.{pid}.resource.{name}.view", value))
                               for value, view in resource.get("view", {}).items())
    descriptor_properties(check, descriptors, contract_objects)


def descriptor_properties(check: CheckContext, descriptors: list, contract_objects: dict[str, Any]) -> None:
    for descriptor, p in descriptors:
        shape_field = "member_slot_shape" if "member_slot_shape" in descriptor else "shape"
        shape = descriptor.get(shape_field)
        if isinstance(shape, list):
            axes = [next(iter(axis)) for axis in shape]
            check.require(len(axes) == len(set(axes)), p + "." + shape_field, "axis names must be unambiguous within one shape")
        # Only explicit Contract property references have a fully declared
        # target table. Do not infer definitions from mathematical prose.
        for attribute in ("shape", "dtype"):
            value = descriptor.get(attribute)
            prefix = "ref:contract.algorithm_requirement::"
            if isinstance(value, str) and value.startswith(prefix):
                target = value[len(prefix):].rsplit(".", 1)[0]
                location = p + "." + attribute
                if check.require(target in contract_objects, location, f"unknown shared Contract reference {target}"):
                    check.require(attribute in contract_objects[target], location,
                                 f"Contract object {target} has no {attribute} property")


def contract(check: CheckContext, contract: dict[str, Any], path: str) -> dict[str, Any]:
    objects, order, locations = {}, {}, {}
    names = set(contract.get("constant", {}))
    groups = [(contract[f], -1, f) for f in ("input", "output")]
    groups += [(s.get("symbol", {}), i, f"compute[{i}].symbol") for i, s in enumerate(contract["compute"])]
    for group, index, label in groups:
        for name, obj in group.items():
            check.require(name not in names, f"{path}.{label}.{name}", "duplicate Contract object name")
            names.add(name)
            objects[name], order[name] = obj, index
            locations[name] = f"{path}.{label}.{name}"
            if label in ("input", "output") and obj["dtype"] == "python_none":
                check.require(obj["shape"] == [], locations[name] + ".shape",
                              "python_none is a singleton scalar; use shape: []")
    for attr in ("shape", "dtype"):
        graph = {}
        for name, obj in objects.items():
            value = obj[attr]
            if isinstance(value, str) and value.startswith("ref:"):
                target = value[4:].rsplit(".", 1)[0]
                p = f"{locations[name]}.{attr}"
                if check.require(target in objects, p, f"unknown Contract reference {target}"):
                    check.require(order[target] <= order[name], p, "reference precedes the defining compute step")
                    graph[name] = [target]
        check.acyclic(graph, path + "." + attr,
                     {name: f"{location}.{attr}" for name, location in locations.items()})
    for name, obj in contract["input"].items():
        for i, axis in enumerate(obj["shape"]):
            key, size = next(iter(axis.items()))
            if isinstance(size, str) and IDENTIFIER.fullmatch(size) and size not in contract.get("constant", {}):
                check.require(size not in names, f"{path}.input.{name}.shape[{i}].{key}", "a runtime extent cannot also name a Contract object")
        bounds = obj.get("range")
        if isinstance(bounds, list):
            endpoints = [number if (number := literal_integer(value)) is not None else value for value in bounds]
            if all(type(value) in (int, float, Decimal) for value in endpoints):
                check.require(endpoints[0] <= endpoints[1], path + ".input." + name + ".range", "range endpoints are reversed")
    return {**contract.get("constant", {}), **objects}


def math(check: CheckContext, plan: dict[str, Any], contract: dict[str, Any], path: str) -> None:
    names, locations = {}, {}
    reserved = set(contract.get("constant", {})) | set(contract["input"]) | set(contract["output"])
    for i, step in enumerate(plan["design"]["math_reshape"]["rewrite"]):
        for name, obj in step.get("symbol", {}).items():
            p = f"{path}.design.math_reshape.rewrite[{i}].symbol.{name}"
            check.require(name not in names and name not in reserved, p, "symbol duplicates an existing definition")
            names[name] = obj
            locations[name] = p
    for phase in plan["design"]["work_schedule"]["phase"].values():
        dec = phase["decomposition"]
        for block in [dec["target"]["single_tile"], *dec.get("worktile", {}).values()]:
            for name in block:
                if name in names:
                    check.require("shape" in names[name] and "dtype" in names[name],
                                 locations[name],
                                 "a value consumed as a data block must publish shape and dtype")
