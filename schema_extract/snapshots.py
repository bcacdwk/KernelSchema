"""Freeze source bytes separately from the current extraction rules/runtime."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from uuid import uuid4

from .common import ExtractError, digest, fsync_directory, read_json, seal, sha256, sync_tree, verify_manifest, write_json
from .inputs import SourceTree


PHASE_PAGES = {
    "P1": "01_contract.md", "P2": "02_dispatch_draft.md", "P3": "03_root.md",
    "P4": "04_dispatch_plan.md", "P5": "05_branch.md", "P6": "06_final.md",
}


def rule_files(repo: Path) -> list[str]:
    paths = {repo / "README.md", repo / "EXTRACT_SCHEMA.md", repo / "schema_template/README.md",
             repo / "schema_template/starter.yaml", repo / "schema_template/check_yaml.py",
             repo / "schema_extract/README.md", repo / "schema_extract/issues/README.md"}
    paths.update((repo / "schema_docs").glob("*.md"))
    paths.update(repo / "schema_extract/phases" / name for name in PHASE_PAGES.values())
    paths.update((repo / "schema_extract").glob("*.py"))
    paths.update(p for p in (repo / "schema_template/tools").rglob("*.py") if "tests" not in p.parts)
    if not (repo / "schema_docs/0.common.md").is_file() or not (repo / "schema_docs/1.contract.algorithm_requirement.md").is_file():
        raise ExtractError("missing current schema specifications")
    return sorted(p.relative_to(repo).as_posix() for p in paths)


def _head(tree: SourceTree) -> str | None:
    if tree.commit:
        return tree.commit
    try:
        return tree.git("rev-parse", "--verify", "HEAD").decode().strip()
    except ExtractError:
        return None


def _private_commit(directory: Path, files: dict[str, str]) -> str:
    """Create only a private bare source repository, never touch the user's index."""
    git_dir = directory.parent / "source.git"
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_AUTHOR_NAME": "Schema Extract", "GIT_AUTHOR_EMAIL": "schema-extract@localhost",
                "GIT_COMMITTER_NAME": "Schema Extract", "GIT_COMMITTER_EMAIL": "schema-extract@localhost",
                "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z"})
    def git(*args, data=None):
        result = subprocess.run(["git", "-c", "core.hooksPath=" + os.devnull,
                                 "-c", "commit.gpgsign=false", *args], env=env, input=data, capture_output=True)
        if result.returncode:
            raise ExtractError("private source Git snapshot failed: " + result.stderr.decode(errors="replace"))
        return result.stdout.decode().strip()
    git("-c", "init.defaultBranch=snapshot", "init", "--bare", "--quiet", str(git_dir))
    env["GIT_INDEX_FILE"] = str(directory.parent / ".source-index")
    base = ["--git-dir=" + str(git_dir), "--work-tree=" + str(directory)]
    # Bypass attributes, clean filters and line-ending conversion: these objects
    # must contain the bytes the worker actually reads, including untracked files.
    for name in files:
        blob = git(*base, "hash-object", "-w", "--stdin", "--no-filters", data=(directory / name).read_bytes())
        git(*base, "update-index", "--add", "--cacheinfo", "100644", blob, name)
    tree_id = git(*base, "write-tree")
    commit = git(*base, "commit-tree", tree_id, "-m", "Fixed extraction source bytes")
    git(*base, "update-ref", "refs/heads/snapshot", commit)
    Path(env["GIT_INDEX_FILE"]).unlink()
    fixed = SourceTree(git_dir, commit)
    for name, expected in files.items():
        if sha256(fixed.read(name)) != expected:
            raise ExtractError(f"private Git source differs from frozen bytes: {name}")
    return commit


def _version(tree: SourceTree, files: dict[str, str], frozen: Path, bindings=None, aliases=None) -> dict:
    head = _head(tree)
    dirty = []
    if not tree.commit:
        committed = SourceTree(tree.root, head) if head else None
        for name, expected in files.items():
            try:
                same = committed is not None and sha256(committed.read(name)) == expected
            except ExtractError:
                same = False
            if not same:
                dirty.append(name)
    identity = {"files": files, "origin_repository": str(tree.root), "origin_commit": head,
                "dirty_files": dirty}
    if bindings is not None:
        identity["bindings"] = bindings
    if aliases:
        for alias in aliases:
            if not alias["files"] or any(files.get(name) != value for name, value in alias["files"].items()):
                raise ExtractError("imported source alias is not bound to the new fixed source bytes")
        identity["aliases"] = aliases
    if bindings is not None and dirty:
        identity["ref"] = _private_commit(frozen, files)
        identity["ref_kind"] = "private_snapshot_commit"
        identity["repository_id"] = "SchemaExtractSource_" + digest(files)[:12]
    else:
        identity["ref"] = head if head and not dirty else "worktree-sha256-" + digest(identity)
        identity["ref_kind"] = "git_commit" if head and not dirty else "content_snapshot"
        identity["repository_id"] = tree.root.name
    identity["id"] = digest(identity)
    return identity


def freeze_inputs(state_root: Path, tree: SourceTree, bindings: dict, names: list[str],
                  config: dict, *, rules_repo: Path | None = None, source_aliases=None) -> dict:
    """Read actual worktree bytes; never substitute HEAD for uncommitted code."""
    rules_repo = (rules_repo or Path(__file__).resolve().parent.parent).resolve()
    state_root = state_root.resolve()
    snapshots = state_root / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    temporary = snapshots / (".building-" + uuid4().hex)
    temporary.mkdir()
    try:
        versions = {}
        for kind, origin, selected in (("source", tree, names),
                                       ("rules", SourceTree(rules_repo), rule_files(rules_repo))):
            hashes = {}
            for name in selected:
                data = origin.read(name)
                target = temporary / kind / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                hashes[name] = sha256(data)
            if not origin.commit:
                for name, expected in hashes.items():
                    if sha256(origin.read(name)) != expected:
                        raise ExtractError(f"{kind} changed during snapshot: {name}")
            versions[kind] = _version(origin, hashes, temporary / kind, bindings if kind == "source" else None,
                                      source_aliases if kind == "source" else None)
        fingerprint = digest({"source": versions["source"]["id"], "rules": versions["rules"]["id"],
                              "config": config, "policy": "skip-hardware"})
        destination = snapshots / fingerprint
        lock = {"format": "schema_extract.inputs.v2", "fingerprint": fingerprint,
                **versions, "config": config, "check_policy": "skip-hardware",
                "snapshot": str(destination)}
        if destination.exists():
            existing = read_json(destination / "inputs.lock.json")
            if existing != lock:
                raise ExtractError("snapshot identity collision")
            verify_snapshot(existing)
            return existing
        write_json(temporary / "inputs.lock.json", lock)
        seal(temporary)
        sync_tree(temporary)
        os.rename(temporary, destination)
        fsync_directory(snapshots)
        fsync_directory(state_root)
        return lock
    finally:
        if temporary.exists():
            for path in temporary.rglob("*"):
                path.chmod(0o755 if path.is_dir() else 0o644)
            temporary.chmod(0o755)
            shutil.rmtree(temporary)


def verify_snapshot(lock: dict) -> None:
    root = Path(lock["snapshot"])
    if read_json(root / "inputs.lock.json") != lock:
        raise ExtractError("input lock changed")
    for kind in ("source", "rules"):
        version = lock[kind]
        identity = {key: value for key, value in version.items() if key != "id"}
        if digest(identity) != version["id"]:
            raise ExtractError(f"{kind} version digest mismatch")
        verify_manifest(root / kind, version["files"])
        for alias in version.get("aliases", []):
            if any(version["files"].get(name) != value for name, value in alias["files"].items()):
                raise ExtractError("imported source alias no longer matches fixed bytes")
        if version["ref_kind"] == "private_snapshot_commit":
            fixed = SourceTree(root / "source.git", version["ref"])
            if sorted(fixed.files(".")) != sorted(version["files"]):
                raise ExtractError("private source Git tree differs from source manifest")
            for name, expected in version["files"].items():
                if sha256(fixed.read(name)) != expected:
                    raise ExtractError(f"private source Git snapshot changed: {name}")
    expected = digest({"source": lock["source"]["id"], "rules": lock["rules"]["id"],
                       "config": lock["config"], "policy": "skip-hardware"})
    if expected != lock["fingerprint"]:
        raise ExtractError("input fingerprint mismatch")


def verify_executor(lock: dict) -> None:
    """Direct API callers also must execute exactly the frozen Python sources."""
    current = Path(__file__).resolve().parent.parent
    expected = {name: value for name, value in lock["rules"]["files"].items()
                if name.startswith("schema_extract/") and name.endswith(".py")}
    actual = {path.relative_to(current).as_posix(): sha256(path.read_bytes())
              for path in (current / "schema_extract").glob("*.py")}
    if actual != expected:
        raise ExtractError("executor differs from inputs.lock; run python3 -B -m schema_extract from " +
                           str(Path(lock["snapshot"]) / "rules"))
