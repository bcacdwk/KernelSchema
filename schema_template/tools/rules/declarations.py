"""Task identity, declared interfaces and conditions needed for relation traversal."""
from __future__ import annotations

from ..diagnostics import field_path
from .common import CheckContext, as_list
from .values import literal_integer


def algorithm_outputs(check: CheckContext, algorithm: dict,
                      path: str = "root.CONTRACT.Algorithm_requirement") -> None:
    """The output/criterion relation also applies to Contract-only documents."""
    outputs, tolerance = algorithm["Output"], algorithm["Golden_reference"]["Tolerance"]
    for name in outputs.keys() - tolerance.keys():
        check.require(False, field_path(path + ".Golden_reference.Tolerance", name),
                      "every Contract output needs a tolerance criterion", missing=True)
    for name in tolerance.keys() - outputs.keys():
        check.require(False, field_path(path + ".Golden_reference.Tolerance", name),
                      "Tolerance names an undeclared output; correct the key, do not invent an output")


def validate(check: CheckContext, task: dict, *, skip_hardware=False, standalone=False) -> None:
    """Check explicit identities and conditions; do not infer declarations from prose."""
    root_name = next(iter(task)) if standalone else "root"
    root = task[root_name]
    algorithm_outputs(check, root["CONTRACT"]["Algorithm_requirement"],
                      root_name + ".CONTRACT.Algorithm_requirement")
    hardware = None if skip_hardware else root["CONTRACT"]["Hardware_specification"]
    if hardware is not None:
        stack, programming = hardware["Stack"], hardware["Wiki"]["Programming"]
        if ("Template" in stack) != ("Template" in programming):
            owner = "Wiki.Programming" if "Template" in stack else "Stack"
            check.require(False, root_name + ".CONTRACT.Hardware_specification." + owner + ".Template",
                          "Stack.Template and Wiki.Programming.Template are required together", missing=True)
        reference = hardware["Wiki"]["Hardware"].strip("<>")
        check.require(not reference.split(":", 1)[-1].endswith("/"),
                      root_name + ".CONTRACT.Hardware_specification.Wiki.Hardware",
                      "Wiki.Hardware must identify a device file, not a directory")

    for name, plan in task.items():
        meta = plan["META_DATA"]
        dispatch = meta.get("Dispatch", {})
        for field in ("Name", "Describe"):
            check.require(meta[field] == root["META_DATA"][field], f"{name}.META_DATA.{field}",
                          f"all nodes must use the shared task {field} declared in root")
        if not standalone:
            check.require(dispatch.get("id", "root") == name, name + ".META_DATA.Dispatch.id",
                          "Dispatch.id must match the YAML filename", missing=name != "root" and "id" not in dispatch)

        for i, rewrite in enumerate(plan["DESIGN"]["Math_reshape"]["Rewrite"]):
            for symbol, value in rewrite.get("Symbol", {}).items():
                p = field_path(f"{name}.DESIGN.Math_reshape.Rewrite[{i}].Symbol", symbol)
                check.require(not ("binding" in value and "value" in value), p,
                              "binding and value cannot both define the same symbol")
                if "retention" in value:
                    for attr in ("shape", "dtype"):
                        check.require(attr in value, p + "." + attr,
                                      "carry_forward/keep_value values need shape and dtype for downstream use",
                                      missing=True)

        work = plan["DESIGN"]["Work_schedule"]
        pipe = plan["DESIGN"]["Tile_pipeline"]["Step"]
        wp = name + ".DESIGN.Work_schedule"
        check.require(work["Step"].keys() == pipe.keys(), name + ".DESIGN.Tile_pipeline.Step",
                      "Tile_pipeline and Work_schedule must describe the same step IDs")
        submitted = set()
        for i, launch in enumerate(work["Flow"]["Launch"]):
            p = f"{wp}.Flow.Launch[{i}]"
            if len(launch["step"]) > 1:
                check.require("order" in launch, p + ".order",
                              "order is required when a launch contains multiple steps", missing=True)
            for j, step in enumerate(launch["step"]):
                check.require(step in work["Step"], p + f".step[{j}]", "Launch names an undeclared step")
                check.require(step not in submitted, p + f".step[{j}]", "each step must be submitted exactly once")
                submitted.add(step)
        for step in work["Step"].keys() - submitted:
            check.require(False, wp + ".Step." + step, "step is not submitted by Flow.Launch")
        for value, handoff in work["Flow"].get("Handoff", {}).items():
            p = field_path(wp + ".Flow.Handoff", value)
            check.require(handoff["producer"] in work["Step"], p + ".producer", "unknown producer step")
            for i, consumer in enumerate(handoff["consumer"]):
                check.require(consumer in work["Step"] and consumer != handoff["producer"],
                              p + f".consumer[{i}]", "Handoff consumer must be another declared step")

        for step, definition in work["Step"].items():
            p = wp + ".Step." + step
            dec, mapping, schedule = definition["Decomposition"], definition["Mapping"], definition["Schedule"]
            for field in ("Worker", "Placement"):
                if field in mapping and hardware is not None:
                    check.require(mapping[field]["unit"] in hardware["Execution"], p + ".Mapping." + field + ".unit",
                                  f"{field} unit is not registered in shared Hardware_specification.Execution")
            count = literal_integer(mapping["Worker"]["unit_count"])
            if "Member" in mapping and count is not None:
                check.require(len(mapping["Member"]) == count, p + ".Mapping.Member",
                              "Member entries must match Worker.unit_count")
            for value, block in dec.get("Worktile", {}).get("Invariant", {}).items():
                if "offset" in block:
                    check.require(len(block["offset"]) == len(block["shape"]),
                                  field_path(p + ".Decomposition.Worktile.Invariant", value) + ".offset",
                                  "offset length must match shape rank")
            if dec["Partition"]:
                check.require("Worktile_order" in schedule, p + ".Schedule.Worktile_order",
                              "partitioned work requires Worktile_order", missing=True)
            ordered = {dim for entry in schedule.get("Worktile_order", []) for dim in as_list(entry["by"])}
            if "Worktile_order" in schedule:
                check.require(ordered == dec["Partition"].keys(), p + ".Schedule.Worktile_order",
                              "Worktile_order must cover exactly the declared Partition dimensions")
            assignment = schedule["Worker_assignment"]
            ap = p + ".Schedule.Worker_assignment"
            if assignment["Refill"] == "none":
                check.require("Preassigned" in assignment, ap + ".Preassigned",
                              "Refill is none: Preassigned is required (start+stride or start+count)", missing=True)
            if "Preassigned" in assignment:
                pre = assignment["Preassigned"]
                check.require(("stride" in pre) != ("count" in pre), ap + ".Preassigned",
                              "use exactly one of start+stride or start+count")

        for step, definition in pipe.items():
            p = f"{name}.DESIGN.Tile_pipeline.Step.{step}"
            event_bases = set()
            for event in definition.get("Event", {}):
                base = event.split("(", 1)[0].strip()
                ep = field_path(p + ".Event", event)
                check.require(base not in {"kernel_done", "launch_ready"}, ep,
                              "built-in launch event names cannot also name a local Event")
                check.require(base not in event_bases, ep, "an Event base has one declaration and one formal signature")
                event_bases.add(base)
            for key, storage in definition.get("Storage", {}).items():
                sp = field_path(p + ".Storage", key)
                depth = literal_integer(storage["depth"])
                if depth is not None and depth > 1:
                    check.require("view" in storage, sp + ".view",
                                  "multi-slot Storage requires value-to-slot views", missing=True)
                    for value, view in storage.get("view", {}).items():
                        check.require("slot_index" in view, field_path(sp + ".view", value) + ".slot_index",
                                      "multi-slot views require slot_index", missing=True)
