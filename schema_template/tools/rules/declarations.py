"""Task identities, declared interfaces and prerequisites for relation traversal."""
from __future__ import annotations

from ..diagnostics import field_path
from .common import CheckContext, as_list
from .values import literal_integer


def algorithm_outputs(check: CheckContext, algorithm: dict) -> None:
    """The shared output/criterion relation also applies to Contract-only drafts."""
    outputs, tolerance = algorithm["output"], algorithm["golden_reference"]["tolerance"]
    for name in outputs.keys() - tolerance.keys():
        check.require(False, "root.contract.algorithm_requirement.golden_reference.tolerance." + name,
                     "every Contract output needs a tolerance criterion", missing=True)
    for name in tolerance.keys() - outputs.keys():
        check.require(False, "root.contract.algorithm_requirement.golden_reference.tolerance." + name,
                     "tolerance names an undeclared output; correct the key, do not invent an output")


def validate(check: CheckContext, task: dict, *, allow_missing_hardware=False) -> None:
    """Conditions that keep identities, references and handoffs readable."""
    root = task["root"]
    algorithm_outputs(check, root["contract"]["algorithm_requirement"])
    hardware = None if allow_missing_hardware else root["contract"]["hardware_specification"]
    if hardware is not None:
        stack, programming = hardware["stack"], hardware["wiki"]["programming"]
        if ("template" in stack) != ("template" in programming):
            owner = "wiki.programming" if "template" in stack else "stack"
            check.require(False, "root.contract.hardware_specification." + owner + ".template",
                         "stack.template and wiki.programming.template are required together", missing=True)
    for name, plan in task.items():
        meta = plan["meta_data"]
        dispatch = meta.get("dispatch", {})
        for field in ("name", "describe"):
            check.require(meta[field] == root["meta_data"][field], f"{name}.meta_data.{field}",
                         f"all nodes must use the shared task {field} declared in root")
        if name == "root":
            check.require("parent" not in dispatch, name + ".meta_data.dispatch.parent", "root has no parent")
            if len(task) > 1:
                check.require("dispatch" in meta, name + ".meta_data.dispatch",
                             "dispatch is required on root when derived nodes exist", missing=True)
        check.require(dispatch.get("id", "root") == name, name + ".meta_data.dispatch.id",
                     "dispatch.id must match the YAML filename")
        math = plan["design"]["math_reshape"]
        steps = [step["step"] for step in math["rewrite"]]
        check.require(len(steps) == len(set(steps)), name + ".design.math_reshape.rewrite", "rewrite step IDs must be unique")
        ledger = math.get("ledger", {})
        if ledger:
            check.require(len(ledger) > 1, name + ".design.math_reshape.ledger", "ledger needs at least one actual resource change")
        for resource, rows in ledger.items():
            if resource != "baseline":
                for i, row in enumerate(rows):
                    for j, step in enumerate(row["by"]):
                        check.require(step in steps, f"{name}.design.math_reshape.ledger.{resource}[{i}].by[{j}]", "unknown rewrite step")
        for i, step in enumerate(math["rewrite"]):
            for symbol, value in step.get("symbol", {}).items():
                p = f"{name}.design.math_reshape.rewrite[{i}].symbol.{symbol}"
                check.require(not ("binding" in value and "value" in value), p, "binding and value cannot both define the same symbol")
                if "retention" in value:
                    for attr in ("shape", "dtype"):
                        check.require(attr in value, p + "." + attr,
                                     "carry_forward/keep_value values need shape and dtype for downstream use", missing=True)
        work = plan["design"]["work_schedule"]
        pipe = plan["design"]["tile_pipeline"]["phase"]
        wp = name + ".design.work_schedule"
        check.require(work["phase"].keys() == pipe.keys(), name + ".design.tile_pipeline.phase",
                     "tile_pipeline and work_schedule must describe the same phase IDs")
        submitted = set()
        for i, launch in enumerate(work["flow"]["launch"]):
            p = f"{wp}.flow.launch[{i}]"
            if len(launch["phase"]) > 1:
                check.require("order" in launch, p + ".order", "order is required when a launch contains multiple phases", missing=True)
            for j, pid in enumerate(launch["phase"]):
                check.require(pid in work["phase"], p + f".phase[{j}]", "launch names an undeclared phase")
                check.require(pid not in submitted, p + f".phase[{j}]", "each phase must be submitted exactly once")
                submitted.add(pid)
        for pid in work["phase"].keys() - submitted:
            check.require(False, wp + ".phase." + pid, "phase is not submitted by flow.launch")
        for value, handoff in work["flow"].get("handoff", {}).items():
            p = wp + ".flow.handoff." + value
            check.require(handoff["producer"] in work["phase"], p + ".producer", "unknown producer phase")
            for i, consumer in enumerate(handoff["consumer"]):
                check.require(consumer in work["phase"] and consumer != handoff["producer"], p + f".consumer[{i}]",
                             "handoff consumer must be another declared phase")
        for pid, phase in work["phase"].items():
            p = wp + ".phase." + pid
            dec, mapping, schedule = phase["decomposition"], phase["mapping"], phase["schedule"]
            for field in ("worker", "placement"):
                if field in mapping and hardware is not None:
                    check.require(mapping[field]["unit"] in hardware["execution"], p + ".mapping." + field + ".unit",
                                 f"{field} unit is not registered in shared hardware execution")
            member_count = literal_integer(mapping["worker"]["unit_count"])
            if "member" in mapping and member_count is not None:
                check.require(len(mapping["member"]) == member_count, p + ".mapping.member", "member entries must match worker.unit_count")
            for value, block in dec.get("worktile", {}).get("invariant", {}).items():
                if "offset" in block:
                    check.require(len(block["offset"]) == len(block["shape"]), p + ".decomposition.worktile.invariant." + value + ".offset", "offset length must match shape rank")
            if dec["partition"]:
                check.require("worktile_order" in schedule, p + ".schedule.worktile_order", "partitioned work requires worktile_order", missing=True)
            ordered = {dim for entry in schedule.get("worktile_order", []) for dim in as_list(entry["by"])}
            if "worktile_order" in schedule:
                check.require(ordered == dec["partition"].keys(), p + ".schedule.worktile_order", "worktile_order must cover the declared partition dimensions")
            assignment = schedule["worker_assignment"]
            ap = p + ".schedule.worker_assignment"
            if assignment["refill"] == "none":
                check.require("preassigned" in assignment, ap + ".preassigned", "refill is none: preassigned is required (start+step or start+count)", missing=True)
            if "preassigned" in assignment:
                pre = assignment["preassigned"]
                check.require(("step" in pre) != ("count" in pre), ap + ".preassigned", "use either start+step or start+count")
        for pid, phase in pipe.items():
            p = f"{name}.design.tile_pipeline.phase.{pid}"
            for milestone in phase.get("milestone", {}):
                check.require(milestone.split("(", 1)[0] not in {"kernel_done", "launch_ready"}, field_path(p + ".milestone", milestone),
                             "built-in launch events cannot also name a local milestone")
            for key, resource in phase.get("resource", {}).items():
                rp = p + ".resource." + key
                count = literal_integer(resource["slot_count"])
                if count is not None and count > 1:
                    check.require("view" in resource, rp + ".view", "multi-slot storage requires value-to-slot views", missing=True)
                for value, view in resource.get("view", {}).items():
                    vp = field_path(rp + ".view", value)
                    if count is not None and count > 1:
                        check.require("slot_index" in view, vp + ".slot_index", "multi-slot views require a slot_index", missing=True)
                    slot = literal_integer(view.get("slot_index"))
                    if count is not None and slot is not None:
                        check.require(slot < count, vp + ".slot_index", "view slot_index exceeds resource slot_count")
