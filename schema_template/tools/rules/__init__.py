"""Read-only validation: structure, declared interfaces, then effective relations."""
from __future__ import annotations

from typing import Any

from ..diagnostics import Diagnostic
from ..errors import bounded
from .. import text_policy
from . import declarations, pipeline, schedule, symbols
from .common import CheckContext
from .structure import algorithm as algorithm_structure, validate_task as validate_structure


@bounded
def validate_algorithm(algorithm: dict[str, Any]) -> list[Diagnostic]:
    """Check one current Algorithm_requirement without any Design or file I/O."""
    path, errors = "root.CONTRACT.Algorithm_requirement", []
    algorithm_structure(algorithm, path, errors)
    if errors:
        return errors
    check = CheckContext()
    declarations.algorithm_outputs(check, algorithm, path)
    objects = symbols.contract(check, algorithm, path)
    symbols.descriptor_properties(check, symbols.contract_descriptors(algorithm, path), objects)
    return check.errors + text_policy.check_values(algorithm, path, scope="algorithm")


@bounded
def validate_relations(task: dict[str, Any], *, skip_hardware=False, standalone=False) -> list[Diagnostic]:
    """Call after structure passes and from is resolved in each consumer context."""
    check = CheckContext()
    declarations.validate(check, task, skip_hardware=skip_hardware, standalone=standalone)
    if check.errors:
        return check.errors
    root_name = next(iter(task)) if standalone else "root"
    root = task[root_name]
    algorithm = root["CONTRACT"]["Algorithm_requirement"]
    hardware = None if skip_hardware else root["CONTRACT"]["Hardware_specification"]
    contract_objects = symbols.contract(check, algorithm, root_name + ".CONTRACT.Algorithm_requirement")
    for name, plan in task.items():
        symbols.properties(check, plan, contract_objects, name)
        symbols.math(check, plan, algorithm, name)
        schedule.validate(check, plan, name)
        pipeline.validate(check, plan, hardware, name)
        symbols.realization(check, plan, algorithm, hardware, name)
    return check.errors
