"""Field shapes in schema_docs 0–6; named relations live in the other rule modules."""
from __future__ import annotations

from functools import partial
import re

from .values import (
    attribute, binding, constant, enum, identifier, input_range, integer,
    known_text, literal, named, node_id, number, one_or_many, pair, predicate,
    record, reference, reference_list, reject, repository, scalar, sequence,
    shape, source, text, wiki,
)

# Contract: schema_docs/1 and 2.
output = record({"meaning": text, "shape": shape, "dtype": attribute("dtype", context="io")})
input_value = record(
    {"meaning": text, "shape": shape, "dtype": attribute("dtype", context="io")},
    {"range": input_range, "default": scalar},
)
contract_symbol = record({
    "meaning": text,
    "shape": attribute("shape", context="contract_symbol"),
    "dtype": attribute("dtype", context="contract_symbol"),
})
compute = record({"name": text, "math": text}, {"symbol": named(contract_symbol)})
golden_reference = record(
    {"source": partial(repository, entry=True), "tolerance": named(text)},
    {"gap": text},
)
algorithm = record(
    {"input": named(input_value), "output": named(output),
     "compute": sequence(compute), "golden_reference": golden_reference},
    {"constant": named(record({"value": constant, "meaning": text}))},
)
stack = record({"language": known_text, "toolchain": known_text}, {"template": known_text})
programming = record({"language": wiki}, {"template": wiki, "virtual_isa": wiki, "machine_isa": wiki})
wiki_entries = record({
    "hardware": wiki, "microbench": wiki, "profile": wiki,
    "sota": record({"kernel": wiki, "schema": wiki}),
    "programming": programming,
})
hardware = record({
    "device": known_text, "arch": known_text,
    "execution": named(known_text, key=text), "feature": named(known_text, key=text),
    "stack": stack, "wiki": wiki_entries,
})

# Mathematical definitions and their ledger: schema_docs/3.
life = predicate(
    lambda value: isinstance(value, str) and re.fullmatch(r"(?:carried|stored)\s+\S.*", value),
    "carried or stored followed by the logical instance",
)
math_symbol = record(
    {"meaning": text},
    {"shape": attribute("shape", context="math"), "dtype": attribute("dtype", context="math"),
     "binding": binding, "value": literal, "life": life},
)
arithmetic = record({}, {"approx": text, "accumulate": text})
rewrite = record({"step": text, "name": text, "math": text},
                 {"symbol": named(math_symbol), "arithmetic": arithmetic})
ledger_entry = record({"item": text, "change": known_text, "by": sequence(text, unique=True)})
RESOURCES = ("matmul", "vector", "transcendental", "traffic", "footprint", "depth", "pass", "comm", "error")
ledger = record({"baseline": text}, {resource: sequence(ledger_entry) for resource in RESOURCES})
math_reshape = record({"thesis": text, "rewrite": sequence(rewrite)}, {"ledger": ledger})

# Work decomposition, assignment and handoff: schema_docs/4.
unit_count = record({"unit": text, "count": integer(1)})
data_block = record({"shape": shape, "dtype": attribute("dtype")})
region = pair(integer(0))
direction = enum("ascending", "descending")
partition_dimension = record({"size": integer(1), "tile": integer(1)},
                             {"range": pair(integer(1), nullable_upper=True)})
invariant_block = record({"shape": shape, "dtype": attribute("dtype")}, {"offset": sequence(integer())})
iteration_block = record({"shape": shape, "dtype": attribute("dtype"), "direction": direction})
worktile = record({}, {"invariant": named(invariant_block), "iteration": named(iteration_block)})
tile_reduction = record({
    "dimension": identifier, "input": reference_list, "output": reference_list, "method": reference,
})
target = record({"single_tile": named(data_block)}, {"tile_reduction": tile_reduction})
decomposition = record({"partition": named(partition_dimension, empty=True), "target": target},
                       {"strategy": text, "worktile": worktile})
mapping = record({"worker": unit_count, "count": integer(1)},
                 {"strategy": text, "placement": unit_count, "member": sequence(named(region))})
preassigned = record({"start": integer(0)}, {"step": integer(1), "count": integer(0)})


def refill(value, path, errors):
    (enum("none") if isinstance(value, str) else named(text))(value, path, errors)


def initialize(value, path, errors):
    (reference if isinstance(value, str) else number)(value, path, errors)


worker_assignment = record({"refill": refill, "exit": reference_list}, {"preassigned": preassigned})
worktile_order = record({"by": one_or_many(identifier, unique=True), "direction": direction},
                       {"group": integer(1)})
schedule = record({"worker_assignment": worker_assignment},
                  {"strategy": text, "worktile_order": sequence(worktile_order)})
work_phase = record({
    "describe": text, "decomposition": decomposition, "mapping": mapping, "schedule": schedule,
})
schedule_event = predicate(
    lambda value: isinstance(value, str) and re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*(?:\([^\r\n]+\))?", value),
    "phase.milestone event reference",
)
events_or_one = one_or_many(schedule_event)
launch = record(
    {"phase": sequence(identifier, unique=True)},
    {"order": enum("per_worktile", "per_worker", "all", "concurrent"),
     "start_after": events_or_one, "protocol": text},
)
storage = predicate(
    lambda value: isinstance(value, str) and (value == "global" or re.fullmatch(
        r"tile_pipeline\.phase\.[A-Za-z_][A-Za-z0-9_]*\.resource\.[A-Za-z_][A-Za-z0-9_]*", value)),
    "global or tile_pipeline.phase.<phase>.resource.<resource>",
)
handoff = record(
    {"producer": identifier, "consumer": sequence(identifier, unique=True), "storage": storage},
    {"initialize": initialize, "ready": events_or_one, "wait_at": events_or_one},
)
flow = record({"launch": sequence(launch)}, {"handoff": named(handoff)})
work_schedule = record({"thesis": text, "phase": named(work_phase), "flow": flow})

# Worker-local pipeline: schema_docs/5.
def domain(value, path, errors):
    if isinstance(value, list):
        pair(integer())(value, path, errors)
    elif isinstance(value, dict):
        record({"flatten": sequence(identifier, unique=True)})(value, path, errors)
    else:
        enum("worktile")(value, path, errors)


def repeat(value, path, errors):
    if isinstance(value, dict):
        named(domain)(value, path, errors)
        if len(value) != 1:
            reject(errors, path, "repeat defines one local domain")
    else:
        predicate(
            lambda item: isinstance(item, str) and re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*(?:\[[^\r\n]+\])?", item),
            "a domain name or domain slice",
        )(value, path, errors)


def pipe(value, path, errors):
    def action(item, location, diagnostics):
        (pipe if isinstance(item, dict) else text)(item, location, diagnostics)
    record({}, {"repeat": repeat, "prologue": sequence(action),
                "steady": sequence(action), "epilogue": sequence(action)})(value, path, errors)
    if isinstance(value, dict) and value and not any(value.get(r) for r in ("prologue", "steady", "epilogue")):
        reject(errors, path, "a lane needs actions; repeat alone does not describe any work")


mode = enum("sync", "async", "ordered")
member = predicate(
    lambda value: isinstance(value, str) and re.fullmatch(
        r"[^\[\]\r\n]+\[\d+(?::\d+)?\](?:\.[^\[\]\r\n]+\[\d+(?::\d+)?\])*", value),
    "unit[index] or unit[start:end] member path",
)
value_pattern = predicate(
    lambda value: isinstance(value, str) and re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*(?:\([^\r\n]+\))?", value),
    "a value name, optionally with indices",
)
view = record({}, {"slot": integer(0), "part": named(region), "dtype": attribute("dtype")}, empty=True)
resource = record(
    {"space": text, "shape": shape, "dtype": attribute("dtype")},
    {"slot": integer(1), "member": one_or_many(integer(0, expression=False), unique=True),
     "view": named(view, key=value_pattern)},
)
verb_detail = record({"meaning": text, "mode": mode})


def verb_contract(value, path, errors):
    (text if isinstance(value, str) else verb_detail)(value, path, errors)


lane_info = record({"member": member},
                   {"mode": mode, "register": integer(1, expression=False), "verb": named(verb_contract)})
edges = record({}, {kind: named(sequence(text), key=text) for kind in ("data", "reuse", "sync")})
lane = record({"info": lane_info, "pipe": pipe}, {"edge": edges})
pipeline_phase = record({"lane": named(lane)},
                        {"domain": named(domain), "resource": named(resource),
                         "milestone": named(sequence(text), key=value_pattern)})
tile_pipeline = record({"thesis": text, "phase": named(pipeline_phase)})

# Implementation handoff: schema_docs/6.
choice = record({"item": reference_list, "choice": text, "reason": text})
feedback = record({"choice": text, "evidence": repository})
preference = record({"item": reference_list, "choice": text, "reason": text}, {"feedback": feedback})
realization = record({}, {"require": sequence(choice), "prefer": sequence(preference), "forbid": sequence(choice)})


# Task and plan identity: schema_docs/0.
def validate_task(task):
    errors = []
    for name, plan in task.items():
        node_id(name, name, errors)
        dispatch = record({"when": text, **({"id": node_id} if name != "root" else {})},
                          {"parent": node_id, **({"id": node_id} if name == "root" else {})})
        metadata = record({"name": text, "describe": text, "version": integer(1, expression=False),
                           **({"dispatch": dispatch} if name != "root" else {})},
                          {"source": source, **({"dispatch": dispatch} if name == "root" else {})})
        required = {"meta_data": metadata, "design": record({"math_reshape": math_reshape,
                                                             "work_schedule": work_schedule, "tile_pipeline": tile_pipeline})}
        if name == "root":
            required["contract"] = record({"algorithm_requirement": algorithm, "hardware_specification": hardware})
        record(required, {"realization": realization})(plan, name, errors)
    return errors
