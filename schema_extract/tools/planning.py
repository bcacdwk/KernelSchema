"""Freeze the selected inputs and create or seed an extraction plan."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import time
from uuid import uuid4

from .codex_cli import write_json
from .records import Store, StateError, artifact_manifest, digest, file_digest, read_json, seal


DEFAULTS = {"model": "gpt-6-astra", "effort": "xhigh", "jobs": 2}

def selected_files(repo: Path, entry: dict, extras: list[str]) -> list[Path]:
    candidates = {repo / "EXTRACT_SCHEMA.md", repo / "README.md", repo / "schema_template/README.md",
                  repo / "schema_template/starter.yaml", repo / "schema_template/check_yaml.py"}
    candidates.update((repo / "schema_docs").glob("*.md"))
    candidates.update(p for p in (repo / "schema_template/tools").rglob("*.py") if "tests" not in p.parts)
    executor = repo / "schema_extract"
    candidates.update(executor / name for name in ("__init__.py", "__main__.py", "runner.py"))
    candidates.update(p for p in (executor / "tools").glob("*.py") if p.name not in {"supervisor.py", "scheduling.py"})
    operator = repo / "kernel_wiki" / entry["paths"]["readme"]
    candidates.update(operator.parent.rglob("*"))
    if annotated := entry["paths"].get("annotated_entrypoint"):
        candidates.add(repo / "kernel_wiki" / annotated)
    batch = repo / "kernel_wiki" / Path(entry["paths"]["readme"]).parts[0]
    for name in ("README.md", "EVIDENCE.md", "GUIDE.md", "CATALOG.md", "FACETS.md", "IMPORT.json"):
        if (repo / "kernel_wiki" / name).is_file():
            candidates.add(repo / "kernel_wiki" / name)
    for directory in operator.parent.parents:
        if directory.is_relative_to(batch / "operators") and (directory / "README.md").is_file():
            candidates.add(directory / "README.md")
    for name in ("README.md", "CONVENTIONS.md", "PROBLEM_NOTES.md"):
        if (batch / name).is_file():
            candidates.add(batch / name)
    for extra in extras:
        path = repo / extra
        if not path.exists():
            raise StateError(f"missing extra source: {extra}")
        candidates.update(path.rglob("*") if path.is_dir() else [path])
    files = []
    for path in sorted(candidates):
        if path.is_symlink() or not path.resolve().is_relative_to(repo.resolve()):
            raise StateError(f"source outside repository or symlink: {path}")
        if any(part in {".git", "__pycache__"} for part in path.parts) or path.name == ".DS_Store":
            continue
        if path.is_file():
            files.append(path)
    for name, relative in entry["paths"].items():
        if not (repo / "kernel_wiki" / relative).exists():
            raise StateError(f"catalog {name} path is missing: {relative}")
    return files


def freeze_inputs(store: Store, entry: dict, extras: list[str], config: dict, *, seed=None) -> dict:
    files = {p.relative_to(store.repo).as_posix(): file_digest(p)
             for p in selected_files(store.repo, entry, extras)}
    origin = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=store.repo, text=True).strip()
    # Source references retain the real repository commit when the selected
    # materials match it. Uncommitted runner/spec text is independently bound by
    # file digest; it does not rewrite clean kernel provenance.
    source_names = [name for name in files if name.startswith("kernel_wiki/") or any(
        name == extra.rstrip("/") or name.startswith(extra.rstrip("/") + "/") for extra in extras)]
    dirty_sources = []
    for name in source_names:
        result = subprocess.run(["git", "show", f"{origin}:{name}"], cwd=store.repo, capture_output=True)
        if result.returncode or hashlib.sha256(result.stdout).hexdigest() != files[name]:
            dirty_sources.append(name)
    identity = {"entry": entry, "files": files, "config": config, "origin_head": origin,
                "dirty_sources": dirty_sources, "check_policy": "skip-hardware"}
    if seed is not None:
        identity["seed"] = seed
    fingerprint = digest(identity)
    base = store.root / "inputs" / fingerprint
    lock = base / "inputs.lock.json"
    if lock.exists():
        return read_json(lock)
    temporary = base.with_name(".building-" + fingerprint + "-" + uuid4().hex[:8])
    snapshot = temporary / "repo"
    snapshot.mkdir(parents=True)
    for name, expected in files.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(store.repo / name, target)
        if file_digest(target) != expected:
            raise StateError(f"source changed during snapshot: {name}")
    commit, repository_id = origin, store.repo.name
    if dirty_sources:
        # Only selected dirty source material needs a private referenced commit;
        # never stage or commit the original workspace or unrelated user edits.
        env = os.environ.copy()
        env.update({"GIT_AUTHOR_NAME": "Schema Extract", "GIT_AUTHOR_EMAIL": "schema-extract@localhost",
                    "GIT_COMMITTER_NAME": "Schema Extract", "GIT_COMMITTER_EMAIL": "schema-extract@localhost",
                    "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z"})
        for args in (["init", "-q"], ["add", "--all"],
                     ["-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", "commit", "-qm", "Fixed selected extraction inputs"]):
            subprocess.run(["git", *args], cwd=snapshot, env=env, check=True, capture_output=True)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=snapshot, text=True).strip()
        repository_id = "KernelSchemaInputs_" + fingerprint[:12]
    baseline = {"fingerprint": fingerprint, "identity": identity, "files": files,
                "origin_repository": str(store.repo), "origin_head": origin,
                "snapshot": str(base / "repo"), "snapshot_commit": commit,
                "repository_id": repository_id, "source_ref_kind": "private_snapshot" if dirty_sources else "existing_commit",
                "created_at": time.time()}
    write_json(temporary / "inputs.lock.json", baseline)
    seal(temporary)
    base.parent.mkdir(parents=True, exist_ok=True)
    os.rename(temporary, base)
    return baseline


def new_task(store, baseline, output):
    entry = baseline["identity"]["entry"]
    key = entry["id"] + "-" + baseline["fingerprint"][:12]
    directory = store.task_dir(key)
    if (directory / "task.json").exists():
        state = store.load(key)
        if state["output"] != str(output):
            raise StateError("same inputs already have a different explicit output; use export --to")
        return state
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "inputs.lock.json", baseline)
    state = {"format": 1, "key": key, "operator_id": entry["id"],
             "fingerprint": baseline["fingerprint"], "output": str(output), "status": "queued",
             "phase": "worker", "stage": "contract", "node": "root", "review_scope": "contract",
             "checkpoint": None, "history": [], "attempts": [], "active_attempt": None,
             "active_process": None, "dispatches": [], "reviews": {}, "issues": [],
             "check": None, "eligible": False, "result": None, "reason": None,
             "usage": {"calls": 0, "reported_tokens": 0, "elapsed_seconds": 0, "unknown_usage_calls": 0},
             "observed_thread_usage": {}, "usage_observations": {},
             "retry_history": [], "created_at": time.time()}
    store.save(state)
    return state


def seed_inputs(store, old_run, operator):
    _, plan = store.load_plan(old_run)
    states = (store.load(key) for key in plan["tasks"])
    matches = [state for state in states if state["operator_id"] == operator]
    if len(matches) != 1:
        raise StateError(f"seed run must contain exactly one task for {operator}")
    old = matches[0]
    if old["status"] not in {"ended", "stopped"} or old.get("active_process"):
        raise StateError("seed task is not stopped; confirm its invocation termination first")
    for attempt in old["attempts"]:
        for call in attempt["calls"]:
            result_path = Path(call) / "result.json"
            recovery_path = Path(call) / "recovery_termination.json"
            confirmed = result_path.is_file() and read_json(result_path).get("termination", {}).get("confirmed")
            if not confirmed and not (recovery_path.is_file() and read_json(recovery_path).get("confirmed")):
                raise StateError("seed invocation termination is not confirmed")
    baseline = store.verify_baseline(old, verify_runtime=False)
    checkpoint = store.verify_checkpoint(old)
    if not checkpoint:
        raise StateError("seed task has no accepted draft checkpoint")
    binding = {"repository_id": baseline["repository_id"], "commit": baseline["snapshot_commit"],
               "paths": sorted(name for name in baseline["files"] if not name.startswith("schema_extract/"))}
    bindings = deepcopy(old.get("source_bindings", []))
    if binding not in bindings:
        bindings.append(binding)
    seed = {"run_id": old_run, "task_key": old["key"], "fingerprint": old["fingerprint"],
            "checkpoint": old["checkpoint"], "files": artifact_manifest(checkpoint / "artifacts"),
            "usage": deepcopy(old["usage"]), "source_bindings": bindings}
    return old, baseline, seed


def verify_seed_inputs(old_baseline, new_baseline):
    if old_baseline["identity"]["entry"] != new_baseline["identity"]["entry"]:
        raise StateError("seed operator selection/metadata changed; old draft cannot be imported")
    for name, expected in old_baseline["files"].items():
        if name.startswith("schema_extract/") and name.endswith(".py"):
            continue  # New execution code/protocol is explicit in the new plan.
        if new_baseline["files"].get(name) != expected:
            raise StateError(f"seed source/spec changed or removed: {name}")
    if old_baseline["identity"]["check_policy"] != new_baseline["identity"]["check_policy"]:
        raise StateError("seed checker policy changed")


def make_plan(store, ids, work_root, output_root, extras, config, run_id=None, *, seed_run=None):
    config = {key: config.get(key, value) for key, value in DEFAULTS.items()}
    if not ids or len(set(ids)) != len(ids):
        raise StateError("select one or more distinct operator IDs explicitly; no default all")
    if not 1 <= config["jobs"] <= 2 or config["effort"] != "xhigh":
        raise StateError("jobs must be 1 or 2 and effort must be xhigh")
    work_root, output_root = work_root.resolve(), output_root.resolve()
    if work_root.is_relative_to(store.repo) or store.root.is_relative_to(work_root):
        raise StateError("--work-root must be outside the repository and recorded snapshots")
    if output_root == store.repo or output_root.is_relative_to(store.root) or output_root.is_relative_to(work_root):
        raise StateError("output root must be separate from repository root, records and work")
    catalog = read_json(store.repo / "kernel_wiki/catalog.json")
    entries = {entry["id"]: entry for entry in catalog["operators"]}
    missing = set(ids) - entries.keys()
    if missing:
        raise StateError(f"unknown operator IDs: {sorted(missing)}")
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:6]
    directory = store.run_dir(run_id)
    if directory.exists():
        raise StateError("run ID already exists")
    tasks = []
    for operator in ids:
        old, old_baseline, seed = seed_inputs(store, seed_run, operator) if seed_run else (None, None, None)
        baseline = freeze_inputs(store, entries[operator], extras, config, seed=seed)
        if old is not None:
            verify_seed_inputs(old_baseline, baseline)
        state = new_task(store, baseline, output_root / operator)
        if old is not None:
            state = store.seed_checkpoint(state, old, seed)
        tasks.append(state["key"])
    directory.mkdir(parents=True)
    plan = {"format": 1, "run_id": run_id, "tasks": tasks, "config": config,
            "work_root": str(work_root), "output_root": str(output_root), "created_at": time.time()}
    if seed_run:
        plan["seed_run"] = seed_run
    write_json(directory / "plan.json", plan)
    return plan
