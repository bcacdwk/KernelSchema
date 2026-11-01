"""Immutable inputs/checkpoints and the single authoritative task.json record."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
from functools import lru_cache
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from uuid import uuid4

from .codex_cli import write_json


class StateError(RuntimeError):
    pass


def hardware_issue(issue: dict) -> bool:
    return ("hardware_specification" in issue.get("scope", "").lower()
            or issue.get("id", "").lower().startswith("hardware_"))


def issue_disposition(issue: dict) -> str:
    if hardware_issue(issue):
        return "ignored"
    if issue.get("disposition"):
        return issue["disposition"]
    return "fix" if issue.get("kind") in {"transcription", "extraction_error", "tooling", "simplification"} else "note"


def read_json(path: Path):
    return json.loads(path.read_text())


def digest(value) -> str:
    from decimal import Decimal
    def exact(item):
        if isinstance(item, Decimal):
            return ["decimal", str(item)]
        if isinstance(item, dict):
            return ["mapping", [[key, exact(item[key])] for key in sorted(item)]]
        if isinstance(item, list):
            return ["sequence", [exact(v) for v in item]]
        return [type(item).__name__, item]
    return hashlib.sha256(json.dumps(exact(value), ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_manifest(path: Path) -> dict[str, str]:
    result = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise StateError(f"symlink is not accepted: {item}")
        if item.is_file():
            result[item.relative_to(path).as_posix()] = file_digest(item)
        elif not item.is_dir():
            raise StateError(f"special file is not accepted: {item}")
    return result


def seal(path: Path):
    for item in path.rglob("*"):
        item.chmod(0o555 if item.is_dir() else 0o444)
    path.chmod(0o555)


def clone_artifacts(source: Path, target: Path):
    target.mkdir(parents=True, exist_ok=False)
    for name in tree_manifest(source):
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, dest)


def artifact_manifest(path: Path) -> dict[str, str]:
    manifest = tree_manifest(path)
    for name in manifest:
        if name != "dispatch.md" and not re.fullmatch(r"[a-z][a-z0-9_]*\.yaml", name):
            raise StateError(f"unexpected artifact (only node YAML and dispatch.md): {name}")
    if "root.yaml" not in manifest:
        raise StateError("worker did not save root.yaml")
    return manifest


@lru_cache(maxsize=8)
def parser_for(root: str):
    """Load the same fixed parser as the checker, under an isolated package name."""
    source = Path(root) / "schema_template/tools/__init__.py"
    name = "_schema_extract_checker_" + hashlib.sha256(root.encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, source, submodule_search_locations=[str(source.parent)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return importlib.import_module(name + ".yaml_source")


def schema_parts(path: Path, parser_root: str | None = None) -> dict:
    parser = parser_for(parser_root or str(Path(__file__).resolve().parents[1]))
    result = {}
    for item in path.glob("*.yaml"):
        try:
            data = parser.load(item, item.stem).value
        except (parser.ReadError, parser.DependencyError) as error:
            raise StateError(f"unparseable artifact {item.name}: {error}") from error
        if not isinstance(data, dict):
            raise StateError(f"expected YAML mapping in {item.name}")
        result[item.stem] = data
    return result


def semantic_digests(path: Path, parser_root: str | None = None) -> dict:
    parts = schema_parts(path, parser_root)
    root = parts.get("root", {})
    contract = deepcopy(root.get("contract", {}))
    if isinstance(contract, dict):
        contract.pop("hardware_specification", None)
    nodes = {}
    for name, data in parts.items():
        node = deepcopy({k: v for k, v in data.items() if k != "contract"})
        node.get("meta_data", {}).pop("version", None)
        nodes[name] = digest(node)
    return {"contract": digest(contract), "nodes": nodes,
            "layout": digest({name: data.get("meta_data", {}).get("dispatch") for name, data in parts.items()}),
            "conditions": {name: digest(data.get("meta_data", {}).get("dispatch")) for name, data in parts.items()}}


def check_versions(previous: Path | None, current: Path, parser_root: str | None = None):
    old = schema_parts(previous, parser_root) if previous else {}
    for node, value in schema_parts(current, parser_root).items():
        version = value.get("meta_data", {}).get("version")
        if type(version) is not int or version < 1:
            raise StateError(f"{node}: positive meta_data.version is required")
        if node in old:
            before = deepcopy(old[node])
            after = deepcopy(value)
            old_version = before.get("meta_data", {}).pop("version", 0)
            after.get("meta_data", {}).pop("version", None)
            if before != after and version <= old_version:
                raise StateError(f"{node}: data changed without incrementing version")
            if before == after and version != old_version:
                raise StateError(f"{node}: version changed without a data change")


def invalidate_reviews(state: dict, before: dict | None, after: dict) -> list[str]:
    reviews = state.setdefault("reviews", {})
    invalidated = []
    all_nodes = before is None or before["contract"] != after["contract"]
    if all_nodes:
        invalidated = list(reviews)
        reviews.clear()
    else:
        changed = {node for node in set(before["nodes"]) | set(after["nodes"])
                   if before["nodes"].get(node) != after["nodes"].get(node)}
        old_conditions = before.get("conditions", {})
        existing_layout_changed = any(after.get("conditions", {}).get(node) != condition
                                      for node, condition in old_conditions.items())
        for key in list(reviews):
            if key == "final" or key in changed or (("root" in changed or existing_layout_changed) and key != "contract"):
                invalidated.append(key)
                reviews.pop(key)
    state["eligible"] = False
    return invalidated


class Store:
    def __init__(self, repo: Path, root: Path):
        self.repo, self.root = repo.resolve(), root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def lock(self):
        with (self.root / ".runner.lock").open("a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise StateError("another runner is active; cancel uses a control request") from error
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def task_dir(self, key):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", key):
            raise StateError("invalid task key")
        return self.root / "tasks" / key

    def load(self, key):
        return read_json(self.task_dir(key) / "task.json")

    def save(self, state):
        state["updated_at"] = time.time()
        write_json(self.task_dir(state["key"]) / "task.json", state)

    def baseline(self, key):
        return read_json(self.task_dir(key) / "inputs.lock.json")

    def checkpoint(self, state):
        if not state.get("checkpoint"):
            return None
        return self.task_dir(state["key"]) / "checkpoints" / state["checkpoint"]

    def verify_baseline(self, state, *, verify_runtime=True):
        baseline = self.baseline(state["key"])
        source = Path(baseline["snapshot"])
        for name, expected in baseline["files"].items():
            if not (source / name).is_file() or file_digest(source / name) != expected:
                raise StateError(f"frozen source/spec changed: {name}")
        if digest(baseline["identity"]) != state["fingerprint"]:
            raise StateError("input fingerprint does not match task")
        if baseline["identity"]["files"] != baseline["files"]:
            raise StateError("source manifest does not match bound identity")
        # Calls must use the runner/prompt version named by this baseline too.
        # A code upgrade requires a new explicit plan, never silent continuation.
        runtime = Path(__file__).resolve().parent
        for name, expected in baseline["files"].items() if verify_runtime else ():
            if name.startswith("schema_extract/") and name.endswith(".py"):
                if file_digest(runtime / Path(name).name) != expected:
                    raise StateError(f"executor/protocol changed since plan: {name}; create a new plan")
        return baseline

    def verify_checkpoint(self, state):
        checkpoint = self.checkpoint(state)
        if checkpoint:
            receipt = read_json(checkpoint / "checkpoint.json")
            if receipt["fingerprint"] != state["fingerprint"]:
                raise StateError("checkpoint belongs to another input baseline")
            if artifact_manifest(checkpoint / "artifacts") != receipt["files"]:
                raise StateError("accepted checkpoint has been modified")
        return checkpoint

    def commit_checkpoint(self, state: dict, artifacts: Path, receipt: dict, check: dict) -> dict:
        baseline = self.verify_baseline(state)
        previous = self.verify_checkpoint(state)
        files = artifact_manifest(artifacts)
        check_versions(previous / "artifacts" if previous else None, artifacts, baseline["snapshot"])
        semantics = semantic_digests(artifacts, baseline["snapshot"])
        before = read_json(previous / "checkpoint.json")["semantics"] if previous else None
        changed = previous is None or files != read_json(previous / "checkpoint.json")["files"]
        next_state = deepcopy(state)
        invalidated = invalidate_reviews(next_state, before, semantics) if changed else []
        checkpoint_id = f"cp-{len(state.get('history', [])) + 1:04d}-{uuid4().hex[:10]}"
        parent = self.task_dir(state["key"]) / "checkpoints"
        parent.mkdir(exist_ok=True)
        temporary = parent / (".building-" + checkpoint_id)
        temporary.mkdir()
        clone_artifacts(artifacts, temporary / "artifacts")
        record = {"id": checkpoint_id, "fingerprint": state["fingerprint"], "files": files,
                  "semantics": semantics, "worker_receipt": receipt, "check": check,
                  "previous": state.get("checkpoint"), "invalidated_reviews": invalidated,
                  "created_at": time.time()}
        write_json(temporary / "checkpoint.json", record)
        seal(temporary)
        os.rename(temporary, parent / checkpoint_id)
        # The only acceptance boundary: a complete immutable checkpoint exists
        # before the authoritative task record can refer to it.
        next_state["checkpoint"] = checkpoint_id
        next_state["check"] = check
        next_state["history"].append(checkpoint_id)
        self.save(next_state)
        return next_state

    def new_attempt(self, state: dict, work_root: Path, role: str):
        self.verify_baseline(state)
        checkpoint = self.verify_checkpoint(state)
        attempt_id = f"a-{len(state['attempts']) + 1:04d}-{uuid4().hex[:8]}"
        work = work_root.resolve() / state["key"] / attempt_id
        if work.is_relative_to(self.repo) or self.root.is_relative_to(work):
            raise StateError("attempt must be outside repository and must not contain records/snapshots")
        if checkpoint:
            clone_artifacts(checkpoint / "artifacts", work)
        else:
            work.mkdir(parents=True, exist_ok=False)
        record = {"id": attempt_id, "role": role, "stage": state["stage"],
                  "node": state.get("node", "root"), "work": str(work),
                  "base_checkpoint": state.get("checkpoint"), "calls": [], "created_at": time.time()}
        state["attempts"].append(record)
        state["active_attempt"] = attempt_id
        state["status"] = "running"
        self.save(state)
        return record, work

    def seed_checkpoint(self, state, old_state, seed):
        """Import an already accepted draft unchanged; never fabricate a model receipt."""
        if state.get("checkpoint") or state["attempts"]:
            return state
        baseline = self.verify_baseline(state)
        old = self.verify_checkpoint(old_state)
        old_record = read_json(old / "checkpoint.json")
        checkpoint_id = "cp-0001-seed-" + uuid4().hex[:8]
        parent = self.task_dir(state["key"]) / "checkpoints"
        parent.mkdir(exist_ok=True)
        temporary = parent / (".building-" + checkpoint_id)
        temporary.mkdir()
        clone_artifacts(old / "artifacts", temporary / "artifacts")
        if artifact_manifest(temporary / "artifacts") != seed["files"]:
            raise StateError("seed checkpoint changed during copy")
        check = run_check(baseline, temporary / "artifacts", temporary)
        if check["exit_code"] == 2:
            raise StateError("cannot seed draft: checker did not complete")
        record = {"id": checkpoint_id, "fingerprint": state["fingerprint"], "kind": "imported_draft",
                  "files": artifact_manifest(temporary / "artifacts"),
                  "semantics": semantic_digests(temporary / "artifacts", baseline["snapshot"]),
                  "seed": seed, "check": check, "previous": None, "invalidated_reviews": [],
                  "imported_worker_receipt": old_record.get("worker_receipt", old_record.get("imported_worker_receipt")),
                  "created_at": time.time()}
        write_json(temporary / "checkpoint.json", record)
        seal(temporary)
        os.rename(temporary, parent / checkpoint_id)
        imported = deepcopy(state)
        imported.update(checkpoint=checkpoint_id, history=[checkpoint_id], check=check, seed=seed,
                        phase="reviewer", stage="contract", node="root", review_scope="contract",
                        reviews={}, eligible=False, status="queued", result=None, reason=None,
                        dispatches=deepcopy(old_state["dispatches"]), usage=deepcopy(old_state["usage"]),
                        issues=deepcopy(old_state["issues"]),
                        worker_status=(record["imported_worker_receipt"] or {}).get("status"))
        imported["source_bindings"] = seed["source_bindings"]
        imported["observed_thread_usage"] = deepcopy(old_state.get("observed_thread_usage", {}))
        imported["usage_observations"] = deepcopy(old_state.get("usage_observations", {}))
        for issue in imported["issues"]:
            issue.setdefault("ledger_task_key", old_state["key"])
            issue.setdefault("first_checkpoint", issue.get("checkpoint"))
            issue["disposition"] = issue_disposition(issue)
            if hardware_issue(issue):
                issue["state"] = "ignored"
        self.save(imported)
        return imported


def selected_files(repo: Path, entry: dict, extras: list[str]) -> list[Path]:
    candidates = {repo / "EXTRACT_SCHEMA.md", repo / "README.md", repo / "schema_template/README.md",
                  repo / "schema_template/starter.yaml", repo / "schema_template/check_yaml.py"}
    candidates.update((repo / "schema_docs").glob("*.md"))
    candidates.update(p for p in (repo / "schema_template/tools").rglob("*.py") if "tests" not in p.parts)
    candidates.update(p for p in (repo / "schema_extract").glob("*.py")
                      if not p.name.startswith("test_") and p.name not in {"probe.py", "supervisor.py"})
    operator = repo / "kernel_wiki" / entry["paths"]["readme"]
    candidates.update(operator.parent.rglob("*"))
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


def run_check(baseline: dict, artifacts: Path, records: Path) -> dict:
    command = [sys.executable, "-B", str(Path(baseline["snapshot"]) / "schema_template/check_yaml.py"),
               "--skip-hardware", str(artifacts / "root.yaml")]
    result = subprocess.run(command, text=True, capture_output=True)
    value = {"argv": command, "exit_code": result.returncode,
             "stdout": result.stdout, "stderr": result.stderr, "policy": "skip-hardware"}
    write_json(records / "check.json", value)
    return value
