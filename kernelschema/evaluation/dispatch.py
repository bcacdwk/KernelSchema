"""Finite implement Task routing; writing retains its row-softmax boundary."""
from .task import RowSoftmaxTask, stage as row_stage, verify_stage as row_verify
from .fa4 import FA4Task, TASK_ID, stage_candidate, verify_stage as fa4_verify
from ..contracts import UnsupportedError


def task_adapter(bundle, task):
    if bundle.document(task, "task")["task_id"] == TASK_ID:
        adapter, expected = FA4Task(bundle, task), "python_tree"
    else:
        adapter, expected = RowSoftmaxTask(bundle, task), "single_file"
    if bundle.document(adapter.task["baseline_source"], "source")["format"] != expected:
        raise UnsupportedError("fixed Task baseline Source format differs")
    return adapter


def stage(context, source, destination):
    if isinstance(task_adapter(context.bundle, context.request["task"]), FA4Task):
        return stage_candidate(context.bundle, {**{k: context.request[k] for k in ("task", "schema", "scope")},
                                               "source": source}, destination)
    return row_stage(context.bundle, context.request["task"], source, destination, scope=context.request["scope"])


def verify_stage(staged):
    if staged["manifest"].get("version", "").startswith("fa4-"):
        return fa4_verify(staged)
    return row_verify(staged)
