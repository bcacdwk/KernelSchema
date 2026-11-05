"""Read-only, content-bound evidence for retaining approvals across runtime fixes.

The caller obtains the independent Supervisor review. This module verifies its
record and exact scope; it cannot authenticate a person or establish independence
from names. A successful review is evidence, never execution authorization.
"""

from __future__ import annotations

from copy import deepcopy
import difflib
from pathlib import Path

from .common import ExtractError, digest, read_json, rules_root, sha256
from .inputs import SourceTree
from .snapshots import rule_files, verify_run_inputs, verify_snapshot


REPORT_FORMAT = "schema_extract.compatibility_report.v1"
REVIEW_FORMAT = "schema_extract.compatibility_review.v1"
PRESERVED = ("field_meanings", "acceptance", "source_and_task", "approval_meanings",
             "dependency_meanings", "receipt_meanings", "budget_policy")
COMPATIBLE = {"runtime_fix", "usage_documentation"}
REVIEW_START = {"contract": "P1", "acceptance": "P1", "source_or_task": "P1",
                "policy": "P1", "checker": "P1", "unknown": "P1",
                "paths": "P2", "design": "P3", "parent_selection": "P4"}


def _capture_target(repo: Path) -> tuple[dict, dict[str, bytes]]:
    # Enumerate twice as well as reading twice: additions/deletions during a read
    # must not produce a mixed manifest. Missing inputs can be reported, but can
    # never be used by the compatible branch or by the normal snapshot publisher.
    tree = SourceTree(repo)
    def capture():
        names = rule_files(repo, require_specs=False)
        missing = [name for name in names if not (repo / name).exists() and not (repo / name).is_symlink()]
        content = {name: tree.read(name) for name in names if name not in missing}
        return {"files": {name: sha256(data) for name, data in content.items()}, "missing_files": missing}, content
    first, content = capture()
    second, _ = capture()
    if first != second:
        raise ExtractError("target rules changed while reading the compatibility report")
    return {**first, "manifest_sha256": digest(first["files"])}, content


def _changes(before: dict[str, bytes], after: dict[str, bytes]) -> list[dict]:
    result = []
    for name in sorted(before.keys() | after.keys()):
        old, new = before.get(name), after.get(name)
        if old == new:
            continue
        item = {"path": name, "change": "added" if old is None else "deleted" if new is None else "modified",
                "before_sha256": sha256(old) if old is not None else None,
                "after_sha256": sha256(new) if new is not None else None}
        try:
            old_lines = (old or b"").decode("utf-8").splitlines(keepends=True)
            new_lines = (new or b"").decode("utf-8").splitlines(keepends=True)
            lines = difflib.unified_diff(old_lines, new_lines,
                fromfile="before/" + name if old is not None else "/dev/null",
                tofile="after/" + name if new is not None else "/dev/null")
            item["diff"] = "".join(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
                                   for line in lines)
        except UnicodeDecodeError:
            item["diff"] = None
        item.update(classification="unknown", reason="No independent semantic review is attached.",
                    evidence={"before": item["before_sha256"], "after": item["after_sha256"]})
        result.append(item)
    return result


def _report(state: dict, lock: dict, checkpoint: dict, target: dict, changes: list[dict], *, rules_lock=None) -> dict:
    rules_lock = rules_lock or lock
    basis = {"run_id": state["run_id"], "state_sha256": digest(state),
             "input_fingerprint": lock["fingerprint"], "inputs_lock_sha256": digest(lock),
             "source": deepcopy(lock["source"]), "config": deepcopy(lock["config"]),
             "check_policy": lock["check_policy"], "rules": deepcopy(rules_lock["rules"]),
             "checkpoint_id": checkpoint["id"], "checkpoint_sha256": digest(checkpoint),
             "artifact_files": deepcopy(checkpoint["files"]),
             "stage": state["stage"], "nodes": sorted(state["nodes"])}
    if state.get("semantic_revision"):
        basis.update(semantic_revision=state["semantic_revision"], rules_lock_sha256=digest(rules_lock))
    result = {"format": REPORT_FORMAT, "basis": basis, "target": target, "changes": changes}
    result["assessment"] = _assessment(result, ["unknown"] * len(changes), reviewed=False)
    return {**result, "id": digest(result)}


def _read_basis(store, run_id: str, checkpoint_id: str) -> tuple[dict, dict, dict]:
    state = store.load(run_id)
    lock = read_json(store.run_dir(run_id) / "inputs.lock.json")
    verify_run_inputs(state, lock)
    _, checkpoint = store.verify_checkpoint(state, checkpoint_id)
    return state, lock, checkpoint


def build_report(store, run_id: str, checkpoint_id: str, target_repo: Path | None = None) -> dict:
    """Read the selected checkpoint and the complete target rule selection; write nothing."""
    state, lock, checkpoint = _read_basis(store, run_id, checkpoint_id)
    target, content = _capture_target((target_repo or rules_root()).resolve())
    from .revisions import semantic_rules_lock
    semantic = semantic_rules_lock(store, state)
    old = {name: SourceTree(rules_root(semantic)).read(name) for name in semantic["rules"]["files"]}
    result = _report(state, lock, checkpoint, target, _changes(old, content), rules_lock=semantic)
    if _read_basis(store, run_id, checkpoint_id) != (state, lock, checkpoint):
        raise ExtractError("run inputs or checkpoint changed while reading the compatibility report")
    # The old-byte reads and report assembly also occurred after capture.
    current, _ = _capture_target((target_repo or rules_root()).resolve())
    if current != target:
        raise ExtractError("target rules changed while reading the compatibility report")
    return result


def _verify_id(report: dict) -> None:
    if (not isinstance(report, dict) or report.get("format") != REPORT_FORMAT or
            report.get("id") != digest({key: value for key, value in report.items() if key != "id"})):
        raise ExtractError("compatibility report digest or format mismatch")


def verify_report(store, report: dict, target_repo: Path | None = None, *, target_lock: dict | None = None) -> None:
    """Recheck the original state and actual target; use target_lock after freezing.

    A frozen target may have a different snapshot/source version identity, but
    must preserve source bytes, bindings, full original config and check policy.
    No run input lock, checkpoint or approval is rewritten by this function.
    """
    _verify_id(report)
    basis = report["basis"]
    if target_lock is None:
        current = build_report(store, basis["run_id"], basis["checkpoint_id"], target_repo)
    else:
        if target_repo is not None:
            raise ExtractError("choose either a working-tree target or a frozen target")
        state, lock, checkpoint = _read_basis(store, basis["run_id"], basis["checkpoint_id"])
        verify_snapshot(target_lock)
        if (target_lock["source"]["files"] != lock["source"]["files"] or
                target_lock["source"].get("bindings") != lock["source"].get("bindings") or
                target_lock["config"] != lock["config"] or target_lock["check_policy"] != lock["check_policy"]):
            raise ExtractError("compatible target must preserve source, bindings, original config and check policy")
        files = target_lock["rules"]["files"]
        target = {"files": files, "manifest_sha256": digest(files), "missing_files": []}
        from .revisions import semantic_rules_lock
        semantic = semantic_rules_lock(store, state)
        before = {name: SourceTree(rules_root(semantic)).read(name) for name in semantic["rules"]["files"]}
        after = {name: SourceTree(rules_root(target_lock)).read(name) for name in files}
        current = _report(state, lock, checkpoint, target, _changes(before, after), rules_lock=semantic)
        verify_snapshot(target_lock)
        if _read_basis(store, basis["run_id"], basis["checkpoint_id"]) != (state, lock, checkpoint):
            raise ExtractError("run inputs or checkpoint changed while verifying compatibility")
    if current != report:
        raise ExtractError("compatibility report no longer matches the run, checkpoint or complete target")


def _assessment(report: dict, kinds: list[str], *, reviewed: bool) -> dict:
    missing = bool(report["target"]["missing_files"])
    if not reviewed or "unknown" in kinds or missing:
        outcome, start = "insufficient_information", "P1"
    else:
        starts = [REVIEW_START[kind] for kind in kinds if kind in REVIEW_START]
        start = min(starts) if starts else None
        outcome = "full_re_review" if start == "P1" else "limited_re_review" if start else "compatible_runtime"
    compatible = outcome == "compatible_runtime"
    reason = ("Independent review records unchanged approval semantics for every difference."
              if compatible else "Semantic changes or missing evidence require independent review; uncertainty expands to P1.")
    return {"outcome": outcome,
            "artifacts": {"retained": True, "files_sha256": digest(report["basis"]["artifact_files"]),
                          "reason": "Selected immutable bytes remain available; retention alone grants no approval."},
            "approvals": {"compatible": compatible, "reason": reason},
            "executor": {"compatible": compatible, "registered": False,
                         "session_policy": "new_role_sessions" if compatible else "successor_re_review"},
            "semantic_review": {"suggested_start": start,
                                "stages": ["P" + str(number) for number in range(int(start[1:]), 7)] if start else [],
                                "nodes": report["basis"]["nodes"] if start else [],
                                "reason": reason, "automatic_scoped_migration": False},
            "execution_authorized": False}


def verify_review(report: dict, review: dict) -> dict:
    """Validate explicit Supervisor evidence; call verify_report before adopting it.

    Supervisor identity and record reference describe provenance. They are not
    authentication or a test that the reviewer was independent. The caller must
    obtain the actual independent review, then persist this exact record.
    """
    _verify_id(report)
    if (not isinstance(review, dict) or review.get("format") != REVIEW_FORMAT or
            review.get("report_id") != report["id"]):
        raise ExtractError("review is not bound to this complete compatibility report")
    supervisor = review.get("supervisor")
    if (not isinstance(supervisor, dict) or supervisor.get("role") != "Supervisor" or
            supervisor.get("independent_review") is not True or
            not all(isinstance(supervisor.get(key), str) and supervisor[key].strip() for key in ("identity", "record"))):
        raise ExtractError("compatibility requires an explicit independent Supervisor review record")
    if not isinstance(review.get("reason"), str) or not review["reason"].strip():
        raise ExtractError("compatibility review needs its overall reason")
    entries = review.get("changes")
    if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
        raise ExtractError("compatibility review needs every changed file")
    names = [item.get("path") for item in entries]
    if (not all(isinstance(name, str) for name in names) or len(set(names)) != len(names) or
            set(names) != {item["path"] for item in report["changes"]}):
        raise ExtractError("compatibility review has missing, duplicate or extra changes")
    indexed, kinds = {item["path"]: item for item in entries}, []
    for change in report["changes"]:
        item = indexed[change["path"]]
        if any(key not in item or item[key] != change[key] for key in ("before_sha256", "after_sha256")):
            raise ExtractError("review before/after digest mismatch: " + change["path"])
        kind = item.get("classification")
        if not isinstance(kind, str) or kind not in COMPATIBLE | REVIEW_START.keys():
            raise ExtractError("unrecognized compatibility classification: " + change["path"])
        if (not isinstance(item.get("reason"), str) or not item["reason"].strip() or
                not isinstance(item.get("evidence"), list) or not item["evidence"] or
                not all(isinstance(value, str) and value.strip() for value in item["evidence"])):
            raise ExtractError("each changed file needs a reason and review evidence: " + change["path"])
        if kind in COMPATIBLE and change["diff"] is None:
            raise ExtractError("unreadable changes require re-review: " + change["path"])
        kinds.append(kind)
    result = _assessment(report, kinds, reviewed=True)
    if review.get("decision") != result["outcome"]:
        raise ExtractError("review decision cannot override the changed files' semantic impact")
    if result["outcome"] == "compatible_runtime":
        preserved = review.get("preserves")
        if not isinstance(preserved, dict) or any(preserved.get(key) is not True for key in PRESERVED):
            raise ExtractError("runtime compatibility requires unchanged field, acceptance, source/task, approval, dependency, receipt and budget meanings")
    return {**result, "report_id": report["id"], "review_sha256": digest(review),
            "supervisor": deepcopy(supervisor), "changes": deepcopy(entries)}
