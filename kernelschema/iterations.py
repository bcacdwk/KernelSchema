"""A finite file-writing/native-implementation loop, shared by KS and AKA.

Rounds count Schema versions. Each version gets the full original AKA campaign;
this module neither evaluates candidates nor resumes interrupted stages.
"""
from __future__ import annotations

import json
from pathlib import Path

from .contracts import ProtocolError


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _schema(directory, result):
    relative = result.get("extensions", {}).get("information", {}).get("schema_directory")
    if not relative:
        return None
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory):
        raise ProtocolError("writing Schema directory must remain inside its stage")
    if not path.is_dir() or not (path / "root.yaml").is_file():
        return None
    return path


def iterate(*, task, out, problem, repo_root, platform, schema=None, rounds=2,
            runtime_profile=None, references=(), objective=None, environment=None,
            **native_options):
    """Design if needed, implement, then optimize between at most ``rounds`` versions.

    ``task`` is an explicit task-description file, even when starting from an
    existing Schema. Writing uses api.design/optimize; native options go unchanged
    to api.implement. No custom native owner account is required. Results and all
    declared feedback are retained in full, including failed implementations.
    Existing output directories are rejected, including interrupted runs: inspect
    and settle their original stages instead of replaying a potentially paid call.
    """
    from . import api

    if type(rounds) is not int or rounds < 1:
        raise ProtocolError("rounds must be a positive integer (Schema versions)")
    if native_options.get("prepare_only"):
        raise ProtocolError("iterate executes stages; use implement prepare_only for native preparation")
    if {"request", "continue_run", "workspace"} & native_options.keys():
        raise ProtocolError("iterate owns fresh workspaces; request/continue_run/workspace are not supported")
    task = Path(task).expanduser().resolve(strict=True)
    if not task.is_file():
        raise ProtocolError("task must be a UTF-8 task-description file")
    task.read_text(encoding="utf-8")
    problem = Path(problem).expanduser().resolve(strict=True)
    repo_root = Path(repo_root).expanduser().resolve(strict=True)
    if not problem.is_dir() or not repo_root.is_dir():
        raise ProtocolError("problem and repo_root must be existing native directories")
    if not isinstance(platform, str) or not platform.strip():
        raise ProtocolError("platform must be an explicit native hardware alias")
    current = Path(schema).expanduser().resolve(strict=True) if schema is not None else None
    root = Path(out).expanduser().absolute()
    try:
        root.mkdir(parents=True)
    except FileExistsError as error:
        raise ProtocolError("iteration output already exists; inspect the original stages and choose a new output, never replay them") from error
    root = root.resolve()
    summary = {"operation": "iterate", "status": "running", "stop_reason": "starting",
               "out": str(root), "task": str(task), "rounds": rounds,
               "schema": str(current) if current else None, "stages": []}

    def save():
        _json(root / "iterations.json", summary)

    def finish(status, reason):
        summary.update(status=status, stop_reason=reason)
        save()
        return summary

    def call(operation, number, **options):
        name = f"{operation}-{number:03d}"
        directory, result_path = root / name, root / (name + "-result.json")
        entry = {"operation": operation, "version": number, "directory": str(directory),
                 "result_file": str(result_path), "status": "running"}
        summary["stages"].append(entry)
        summary["stop_reason"] = name
        save()  # An interrupted call remains visible; this is not a resume token.
        try:
            location = {"workspace": directory} if operation == "implement" else {"out": directory}
            result = getattr(api, operation)(**location, **options)
            _json(result_path, result)
        except BaseException as error:
            entry.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                         error={"type": type(error).__name__, "message": str(error)})
            finish(entry["status"], name + " did not return a result; inspect the original stage, no automatic replay")
            raise
        entry["status"] = result["status"]
        save()
        return directory, result, entry

    writing = dict(task=task, runtime_profile=runtime_profile, references=tuple(references),
                   objective=objective, environment=environment)

    def written(directory, result):
        if result.get("usage", {}).get("unsettled"):
            finish("blocked", "writing usage is unsettled; inspect the original stage")
            return None
        if result["status"] != "completed":
            finish(result["status"], "writing stopped: " + result.get("stop_reason", result["status"]))
            return None
        try:
            selected = _schema(directory, result)
        except (OSError, ValueError, ProtocolError) as error:
            finish("failed", "writing supplied no usable Schema: " + str(error))
            return None
        if selected is None:
            finish("failed", "writing returned no Schema files; inspect the retained writing result")
        return selected

    save()
    if current is None:
        directory, result, _ = call("design", 1, **writing)
        current = written(directory, result)
        if current is None:
            return summary
        summary["schema"] = str(current)

    for number in range(1, rounds + 1):
        workspace, result, entry = call("implement", number, schema=current, problem=problem,
                                       repo_root=repo_root, platform=platform, **native_options)
        # Copy declared feedback before deciding whether another stage may run.
        # Missing/unreadable feedback is an error, never silently omitted input.
        feedback = [Path(entry["result_file"])]
        entry["feedback"] = []
        try:
            for index, filename in enumerate(result.get("feedback_files", [])):
                source = Path(filename).expanduser().resolve(strict=True)
                if not source.is_relative_to(workspace):
                    raise ProtocolError("native feedback must remain inside its original workspace")
                data = source.read_bytes()
                data.decode("utf-8")
                destination = root / f"feedback-{number:03d}" / f"{index:04d}-{source.name}"
                destination.parent.mkdir(exist_ok=True)
                destination.write_bytes(data)
                entry["feedback"].append({"source": str(source), "file": str(destination)})
                feedback.append(destination)
        except (OSError, ValueError, ProtocolError) as error:
            return finish("blocked", "native feedback could not be retained: " + str(error))
        save()
        if result["status"] not in {"completed", "failed"} or result.get("usage", {}).get("unsettled"):
            return finish("blocked", "native stage needs attention: " + result.get("stop_reason", result["status"]))
        if number == rounds:
            return finish(result["status"], "Schema version limit reached; see original native results")
        directory, optimized, _ = call("optimize", number + 1, schema=current, feedback=feedback, **writing)
        if (optimized["status"] == "completed" and optimized.get("decision") == "keep"
                and not optimized.get("usage", {}).get("unsettled")):
            return finish(result["status"], "optimize kept the current Schema")
        current = written(directory, optimized)
        if current is None:
            return summary
        summary["schema"] = str(current)

