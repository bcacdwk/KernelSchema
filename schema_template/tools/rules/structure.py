"""Current field spelling and shapes from DOC0–6; relations stay in sibling modules."""
from __future__ import annotations

from functools import partial
import re

from ..pipeline_syntax import (
    VERBS, PipelineSyntaxError, parse_action, parse_event,
    _group_end, parse_members, parse_pipe, parse_waiting, split_top,
)
from .common import EXECUTION_MODES
from .hardware import hardware_rule
from .values import (
    attribute, binding, constant, enum, expression_text, hardware_identity, identifier, input_range, integer, known_text, literal,
    named, node_id, number, one_or_many, pair, predicate, record, reference,
    reference_list, reject, repository, scalar, sequence, shape, source, step_id, text,
)


def _syntax(parser):
    def check(value, path, errors):
        if not isinstance(value, str):
            text(value, path, errors)
            return
        try:
            parser(value, path=path)
        except PipelineSyntaxError as error:
            reject(errors, error.path, error.message)
    check.string_paths = ((),)
    return check


def _formal_name(value, path, errors):
    """view and Event keys name one object and optional distinct formal indices."""
    if not isinstance(value, str):
        text(value, path, errors)
        return
    try:
        operand = parse_action("compute:" + value, path=path).operands[0]
    except PipelineSyntaxError as error:
        reject(errors, error.path, error.message)
        return
    for argument in operand.args:
        identifier(argument, path, errors)
    if len(set(operand.args)) != len(operand.args):
        reject(errors, path, "formal indices must be distinct")


def identities(rule, label):
    entries = named(rule, key=step_id)

    def check(value, path, errors):
        entries(value, path, errors)
        if isinstance(value, dict) and len(value) == 1 and "MAIN" not in value:
            reject(errors, path, f"a single {label} must use MAIN")
    check.field_paths = entries.field_paths
    check.string_paths = entries.string_paths
    check.literal_types = entries.literal_types
    check.value_cases = entries.value_cases
    return check


# Contract.
nullable = predicate(lambda value: type(value) is bool, "a boolean")
nullable.literal_types = (((), bool),)
output = record({"meaning": text, "shape": shape, "dtype": attribute("dtype", context="io")},
                {"nullable": nullable})
# range/default keep their documented value types; text budgets are separate.
input_value = record(
    {"meaning": text, "shape": shape, "dtype": attribute("dtype", context="io")},
    {"nullable": nullable, "range": input_range, "default": scalar},
)
contract_symbol = record({
    "meaning": text,
    "shape": attribute("shape", context="contract_symbol"),
    "dtype": attribute("dtype", context="contract_symbol"),
})
compute = record({"Name": text, "Math": text}, {"Symbol": named(contract_symbol)})
golden_reference = record(
    {"Source": partial(repository, entry=True, fixed=True), "Tolerance": named(text)},
    {"Gap": text},
)
algorithm = record(
    {"Input": named(input_value), "Output": named(output),
     "Compute": sequence(compute), "Golden_reference": golden_reference},
    {"Constant": named(record({"value": constant, "meaning": text}))},
)
hardware = hardware_rule()

# Mathematical decisions and per-entry ledger.
def retention(value, path, errors):
    if not isinstance(value, str):
        text(value, path, errors)
        return
    match = re.match(r"(?:carry_forward|keep_value)\s+", value)
    if match is None:
        reject(errors, path, "expected carry_forward or keep_value followed by the logical instance")
        return
    instance = value[match.end():]
    if instance.startswith("invocation"):
        rest = instance[len("invocation"):].strip()
    elif instance.startswith("(") and (end := _group_end(instance, 0)) is not None:
        coordinates = split_top(instance[1:end], ",")
        for coordinate in coordinates or [""]:
            expression_text(coordinate, path, errors, integer=True)
        rest = instance[end + 1:].strip()
    else:
        reject(errors, path, "retention instance must be invocation or a parenthesized index group")
        return
    if rest:
        iteration = re.fullmatch(r";\s*([a-z_][a-z0-9_]*)\s*=\s*(.+?)\.\.(.+)", rest)
        if iteration is None or (parts := split_top(rest, ";")) is None or len(parts) != 2:
            reject(errors, path, "retention may only append an iteration range such as ; i=0..count-1")
            return
        identifier(iteration[1], path, errors)
        for endpoint in iteration.group(2, 3):
            integer()(endpoint, path, errors)


retention.string_paths = ((),)


math_symbol = record(
    {"meaning": text},
    {"shape": attribute("shape", context="math"), "dtype": attribute("dtype", context="math"),
     "binding": binding, "value": literal, "retention": retention},
)


def ledger_text(value, path, errors):
    if value is not None:
        known_text(value, path, errors)


ledger_text.string_paths = ((),)  # The source repairer always preserves null.


arithmetic = record({}, {"Approx": text, "Accumulate": text})
ledger = record({"Baseline": ledger_text}, {"Gain": ledger_text, "Cost": ledger_text})
rewrite = record({"Name": text, "Math": text},
                 {"Symbol": named(math_symbol), "Arithmetic": arithmetic, "Ledger": ledger})
math_reshape = record({"Thesis": text, "Rewrite": sequence(rewrite)})

# Work definitions and submission.
unit_count = record({"unit": hardware_identity, "unit_count": integer(1)})
data_block = record({"meaning": text, "shape": shape, "dtype": attribute("dtype")})
region = pair(integer(0))
direction = enum("ascending", "descending")
partition_dimension = record({"meaning": text, "size": integer(0), "tile": integer(1)},
                             {"range": pair(integer(0), nullable_upper=True)})
invariant_block = record(
    {"meaning": text, "shape": shape, "dtype": attribute("dtype")},
    {"offset": sequence(integer())},
)
iteration_block = record({
    "meaning": text, "shape": shape, "dtype": attribute("dtype"), "direction": direction,
})
worktile = record({}, {"Invariant": named(invariant_block), "Iteration": named(iteration_block)})
tile_reduction = record({
    "dimension": identifier, "input": reference_list, "output": reference_list, "method": reference,
})
target = record({"Single_tile": named(data_block)}, {"Tile_reduction": tile_reduction})
decomposition = record({"Partition": named(partition_dimension, empty=True), "Target": target},
                       {"Strategy": text, "Worktile": worktile})
mapping = record({"Worker": unit_count, "Worker_count": integer(1)},
                 {"Strategy": text, "Placement": unit_count, "Member": sequence(named(region))})
preassigned = record({"start": integer(0)}, {"stride": integer(1), "count": integer(0)})
no_refill = enum("none")


def refill(value, path, errors):
    (no_refill if isinstance(value, str) else named(text))(value, path, errors)


refill.string_paths = named(text).string_paths
refill.value_cases = no_refill.value_cases


def initialize(value, path, errors):
    (reference if isinstance(value, str) else number)(value, path, errors)


worker_assignment = record({"Refill": refill, "Exit": reference_list}, {"Preassigned": preassigned})
worktile_order = record({"by": one_or_many(identifier, unique=True), "direction": direction},
                       {"group": integer(1)})
schedule = record({"Worker_assignment": worker_assignment},
                  {"Strategy": text, "Worktile_order": sequence(worktile_order)})
work_step = record({
    "Describe": text, "Decomposition": decomposition, "Mapping": mapping, "Schedule": schedule,
})


def schedule_event(value, path, errors):
    if not isinstance(value, str):
        text(value, path, errors)
        return
    owner, separator, target = value.partition(".")
    if not separator or not target:
        reject(errors, path, "expected STEP.event or STEP.event(arguments)")
        return
    step_id(owner, path, errors)
    try:
        parse_action("compute:" + target, path=path)
    except PipelineSyntaxError as error:
        reject(errors, error.path, error.message)


schedule_event.string_paths = ((),)
events_or_one = one_or_many(schedule_event, unique=True)
launch = record(
    {"step": sequence(step_id, unique=True)},
    {"order": enum("per_worktile", "per_worker", "all", "concurrent"), "start_after": events_or_one},
)


def storage_reference(value, path, errors):
    if not isinstance(value, str):
        text(value, path, errors)
    elif value != "global" and not re.fullmatch(
            r"Tile_pipeline\.Step\.[A-Z_][A-Z0-9_]*\.Storage\.[a-z_][a-z0-9_]*", value):
        reject(errors, path, "expected global or Tile_pipeline.Step.<STEP>.Storage.<storage>")
    elif value != "global":
        parts = value.split(".")
        step_id(parts[2], path, errors)
        identifier(parts[4], path, errors)


storage_reference.string_paths = ((),)


handoff = record(
    {"producer": step_id, "consumer": sequence(step_id, unique=True), "storage": storage_reference},
    {"initialize": initialize, "ready": events_or_one, "wait_at": events_or_one},
)
flow = record({"Launch": sequence(launch)}, {"Handoff": named(handoff)})
work_schedule = record({"Thesis": text, "Step": identities(work_step, "step"), "Flow": flow})

# Worker-local pipeline.
flatten_domain = record({"flatten": sequence(identifier, unique=True)})
worktile_domain = enum("worktile")


def domain(value, path, errors):
    if isinstance(value, list):
        pair(integer())(value, path, errors)
    elif isinstance(value, dict):
        flatten_domain(value, path, errors)
    else:
        worktile_domain(value, path, errors)


domain.field_paths = flatten_domain.field_paths
domain.value_cases = worktile_domain.value_cases


def pipe(value, path, errors):
    try:
        parse_pipe(value, path=path)
    except PipelineSyntaxError as error:
        reject(errors, error.path, error.message)


def member(value, path, errors):
    if not isinstance(value, str):
        text(value, path, errors)
        return
    parts = parse_members(value)
    if parts is None:
        reject(errors, path, "expected unit[index] or unit[start:end] member path")
        return
    for unit, lower, upper in parts:
        hardware_identity(unit, path, errors)
        integer(0)(lower, path, errors)
        if upper is not None:
            integer(0)(upper, path, errors)


member.string_paths = ((),)


view = record({}, {"slot_index": integer(0), "region": named(region)}, empty=True)
storage = record(
    {"meaning": text, "depth": integer(1), "location": hardware_identity, "shape": shape,
     "dtype": attribute("dtype", context="storage"), "total_size": integer(0)},
    {"member": one_or_many(integer(0, expression=False), unique=True), "view": named(view, key=_formal_name)},
)
noun_detail = record({"execution_mode": enum(*EXECUTION_MODES), "describe": text})


def _noun_key(verb):
    def check(value, path, errors):
        if not isinstance(value, str):
            text(value, path, errors)
            return
        try:
            action = parse_action(verb + ":" + value, path=path)
        except PipelineSyntaxError as error:
            reject(errors, error.path, error.message)
            return
        if any(operand.args for operand in action.operands):
            reject(errors, path, "Using_verb declares noun bases; arguments belong at action use sites")
    return check


using_verb = record({}, {verb: named(noun_detail, key=_noun_key(verb)) for verb in sorted(VERBS)})
role_info = record({"Describe": text, "Member": member, "Using_verb": using_verb},
                   {"Register": integer(1, expression=False)})
waiting = record({}, {kind: sequence(_syntax(parse_waiting))
                      for kind in ("Prepared", "Released", "Allowed")})
role = record({"Info": role_info, "Pipe": pipe}, {"Waiting": waiting})


def alias_event(value, path, errors):
    if not isinstance(value, str):
        text(value, path, errors)
        return
    try:
        event = parse_event(value, path=path)
    except PipelineSyntaxError as error:
        reject(errors, error.path, error.message)
        return
    if event.role is None or event.member is None:
        reject(errors, path, "Event aliases must explicitly name the event time, ROLE and member selection")


alias_event.string_paths = ((),)


pipeline_step = record({"Role": identities(role, "role")},
                       {"Index": named(record({"meaning": text, "definition": domain})),
                        "Storage": named(storage),
                        "Event": named(sequence(alias_event), key=_formal_name)})
tile_pipeline = record({"Thesis": text, "Step": identities(pipeline_step, "step")})

# Implementation handoff.
choice = record({"Item": reference_list, "Choice": text, "Reason": text})
feedback = record({"Item": reference_list, "Choice": text, "Evidence": partial(repository, fixed=True)})
realization = record({}, {"Require": sequence(choice), "Prefer": sequence(choice),
                          "Forbid": sequence(choice), "Feedback": sequence(feedback)})


def design_region(local_rule, *, allow_from):
    def check(value, path, errors):
        if isinstance(value, dict) and "from" in value:
            record({"from": node_id})(value, path, errors)
            if not allow_from:
                reject(errors, path + ".from", "this region must be local")
        else:
            local_rule(value, path, errors)
    check.field_paths = tuple(
        (parent, fields + ("from",) if not parent else fields)
        for parent, fields in local_rule.field_paths
    )
    check.string_paths = local_rule.string_paths + (("from",),)
    check.literal_types = local_rule.literal_types
    check.value_cases = local_rule.value_cases
    return check


def _plan_rule(*, contract_owner, needs_dispatch, skip_hardware, allow_from):
    dispatch = record({"when": text, **({"id": node_id} if not contract_owner else {})},
                      {"id": node_id} if contract_owner else {})
    metadata = record(
        {"Name": text, "Describe": text, "Version": integer(1, expression=False),
         **({"Dispatch": dispatch} if needs_dispatch else {})},
        {"Source": source, **({"Dispatch": dispatch} if not needs_dispatch else {})},
    )
    regions = {"Math_reshape": math_reshape, "Work_schedule": work_schedule, "Tile_pipeline": tile_pipeline}
    design = record({region: design_region(rule, allow_from=allow_from and not contract_owner)
                     for region, rule in regions.items()})
    required = {"META_DATA": metadata, "DESIGN": design}
    if contract_owner:
        required["CONTRACT"] = (
            record({"Algorithm_requirement": algorithm},
                   {"Hardware_specification": lambda value, path, errors: None})
            if skip_hardware else
            record({"Algorithm_requirement": algorithm, "Hardware_specification": hardware})
        )
    return record(required, {"REALIZATION": realization})


# One current field table, derived from the validators above. Paths omit the
# task ID: None selects a named mapping key and int a list position. The union
# of local/from fields describes spelling only; design_region enforces shape.
_document_rule = _plan_rule(contract_owner=True, needs_dispatch=False,
                            skip_hardware=False, allow_from=True)
CANONICAL_FIELDS = _document_rule.field_paths
STRING_FIELDS = _document_rule.string_paths
# Only strict integer/boolean leaves authorize removal of scalar quotes.
LITERAL_TYPES = _document_rule.literal_types
# Only scalar values at these leaf paths are eligible for case normalization;
# union containers such as Refill mappings and Index ranges retain their shape.
CANONICAL_VALUES = _document_rule.value_cases


def validate_task(task, *, skip_hardware=False, allow_from=False, standalone=False):
    """Check raw/effective fields, or a single independent delivery document.

    skip_hardware ignores the entire Hardware_specification value without
    changing input data. standalone requires its own Contract and local regions;
    Dispatch.id remains the delivery identity, independent of its filename.
    """
    errors = []
    if not isinstance(task, dict) or not task:
        reject(errors, "root", "expected at least one task document")
        return errors
    if standalone and len(task) != 1:
        reject(errors, next(iter(task)), "standalone validation accepts one document")
        return errors
    if not standalone:
        if "root" not in task:
            reject(errors, next(iter(task)), "a task needs root with the shared Contract")
        if len(task) > 10:
            reject(errors, "root" if "root" in task else next(iter(task)),
                   "a task may contain at most 10 complete YAML designs, including root")
    for name, plan in task.items():
        if not standalone:
            node_id(name, name, errors)
        contract_owner = name == "root" or standalone
        needs_dispatch = not standalone and (name != "root" or len(task) > 1)
        _plan_rule(contract_owner=contract_owner, needs_dispatch=needs_dispatch,
                   skip_hardware=skip_hardware, allow_from=allow_from)(plan, name, errors)
    return errors
