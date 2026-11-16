"""Read-only, finite local event analysis of a checked effective plan.

configuration accepts explicit illustration values only. analyze performs Flow
alignment checks and returns complete StepGraph objects, or raises without
returning a partial result. PNG publication is a separate output.GraphOutput
operation; importing or analyzing never reads or writes a source file.
"""
from __future__ import annotations

from ..pipeline_syntax import PipelineSyntaxError
from ..rules.common import CheckContext
from ..rules import pipeline as pipeline_rules, schedule
from .graph import build_step
from .model import ExpansionError, GraphError, StepGraph

__all__ = ["configuration", "analyze", "build_step", "GraphError", "ExpansionError", "StepGraph"]


def configuration(parameters: dict[str, int], worktiles: int = 1) -> dict:
    if not isinstance(parameters, dict) or any(not isinstance(name, str) or not name.strip() or type(value) is not int
                                                for name, value in parameters.items()):
        raise ExpansionError("sample parameters must map nonempty expression names to integers")
    if type(worktiles) is not int or not 0 <= worktiles <= 64:
        raise ExpansionError("sample worktiles must be an integer in 0..64")
    normalized = {}
    for name, value in parameters.items():
        canonical = "".join(name.split())
        if canonical in normalized:
            raise ExpansionError(f"duplicate sample parameter {canonical!r} after removing whitespace")
        if abs(value) > 10**12:
            raise ExpansionError("sample parameter magnitude exceeds the tool limit (10^12)")
        normalized[canonical] = value
    return {"parameters": normalized, "worktiles": worktiles}


def analyze(plan: dict, config: dict) -> list[StepGraph]:
    """Analyze given sample values; no inference from names, images or dispatch.

    The input is an effective plan whose field structure has already been
    checked. Flow is validated statically; each returned graph checks local
    events in one representative worker, without simulating cross-step life.
    """
    config = configuration(config["parameters"], config["worktiles"])
    work = plan["DESIGN"]["Work_schedule"]
    steps = plan["DESIGN"]["Tile_pipeline"]["Step"]
    if steps.keys() != work["Step"].keys():
        raise GraphError("DESIGN.Tile_pipeline.Step: pipeline and work schedule Step IDs must agree")
    submitted = [step for launch in work["Flow"]["Launch"] for step in launch["step"]]
    if len(submitted) != len(set(submitted)) or set(submitted) != set(steps):
        raise GraphError("DESIGN.Work_schedule.Flow.Launch: each Step must be submitted exactly once")
    check = CheckContext()
    try:
        schedule.validate(check, plan, "")
        pipeline_rules.validate(check, plan, None, "")
    except PipelineSyntaxError as error:
        raise GraphError(str(error).lstrip(".")) from error
    if check.errors:
        error = check.errors[0]
        raise GraphError(f"{error.path.lstrip('.')}: {error.message}")
    graphs = []
    for step, definition in steps.items():
        scheduled = work["Step"][step]
        dynamic = scheduled["Schedule"]["Worker_assignment"]["Refill"] != "none"
        graphs.append(build_step(step, definition, scheduled["Mapping"]["Worker"], **config,
                                 dynamic_worktiles=dynamic))
    return graphs
