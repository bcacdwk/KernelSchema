"""Explicit validation order: field shapes, declared interfaces, then named relations."""
from __future__ import annotations

from typing import Any

from ..diagnostics import Diagnostic
from . import declarations, pipeline, schedule, symbols
from .common import CheckContext
from .structure import algorithm as algorithm_structure, validate_task as validate_structure


def validate_algorithm(algorithm: dict[str, Any]) -> list[Diagnostic]:
    """Validate Contract shapes and relations without requiring any Design."""
    path, errors = "root.contract.algorithm_requirement", []
    algorithm_structure(algorithm, path, errors)
    if errors:
        return errors
    check = CheckContext()
    declarations.algorithm_outputs(check, algorithm)
    objects = symbols.contract(check, algorithm, path)
    symbols.descriptor_properties(check, symbols.contract_descriptors(algorithm, path), objects)
    return check.errors


def validate_relations(task: dict[str, Any], *, allow_missing_hardware=False) -> list[Diagnostic]:
    """Call after field shapes pass; check declared relationships, not prose meaning."""
    check, parents = CheckContext(), {}
    declarations.validate(check, task, allow_missing_hardware=allow_missing_hardware)
    if check.errors:
        return check.errors
    for name, plan in task.items():
        if name != "root":
            parent = plan["meta_data"]["dispatch"].get("parent", "root")
            check.require(parent in task, name + ".meta_data.dispatch.parent", f"unknown parent {parent}")
            parents[name] = [parent]
    check.acyclic(parents, "meta_data.dispatch.parent",
                 {name: name + ".meta_data.dispatch.parent" for name in parents})
    contract = task["root"]["contract"]["algorithm_requirement"]
    hardware = None if allow_missing_hardware else task["root"]["contract"]["hardware_specification"]
    contract_objects = symbols.contract(check, contract, "root.contract.algorithm_requirement")
    for name, plan in task.items():
        symbols.properties(check, plan, contract_objects, name)
        symbols.math(check, plan, contract, name)
        schedule.validate(check, plan, name)
        pipeline.validate(check, plan, hardware, name)
    return check.errors
