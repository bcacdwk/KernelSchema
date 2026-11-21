"""Verified bytes, semantic closure, immutable publication and local transactions.

Bundle readers never infer dependencies from prose or scan unrelated history.
All Ref targets use the stable artifacts/<128-bit-id>/ prefix. Source payloads
alone resolve relative to their manifest's sibling files/ directory.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import ctypes
import errno
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from uuid import uuid4

from .contracts import PROTOCOL, ProtocolError, UnsupportedError, Ref, digest, jcs, loads, ref, relative_path, sha256, validate


def fsync_directory(path: Path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_mkdir(path: Path):
    if path.parent != path:
        durable_mkdir(path.parent)
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise ProtocolError(f"not a real directory: {path}")
        return
    path.mkdir()
    fsync_directory(path.parent)
    fsync_directory(path)


def _rename_new(source: Path, target: Path):
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, "renameat2"):
        raise ProtocolError("durable publication requires Linux renameat2")
    fn = libc.renameat2
    fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    fn.restype = ctypes.c_int
    if fn(-100, os.fsencode(source), -100, os.fsencode(target), 1):
        number = ctypes.get_errno()
        if number in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(str(target))
        raise OSError(number, os.strerror(number), str(target))


def durable_write(path: Path, data: bytes, *, mode=0o644, replace=False):
    """flush/fsync payload, no-replace by default, then fsync its directory."""
    durable_mkdir(path.parent)
    fd, name = tempfile.mkstemp(prefix=".ks-write-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
        if replace:
            if path.is_symlink():
                raise ProtocolError(f"refuse symlink replacement: {path}")
            os.replace(temporary, path)
        else:
            _rename_new(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def read_regular(root: Path, path: str) -> tuple[bytes, int]:
    """Read through directory descriptors; no relative component follows links."""
    relative_path(path)
    descriptors = []
    try:
        current = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(current)
        parts = path.split("/")
        for part in parts[:-1]:
            current = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
            descriptors.append(current)
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=current)
        descriptors.append(fd)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ProtocolError(f"Ref is not a regular file: {path}")
        chunks = []
        while block := os.read(fd, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(fd)
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ProtocolError(f"file changed while reading: {path}")
        return b"".join(chunks), stat.S_IMODE(after.st_mode)
    except OSError as error:
        raise ProtocolError(f"cannot read regular bundle file {path}: {error}") from error
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


def artifact_path(path: str):
    relative_path(path)
    if not re.fullmatch(r"artifacts/[0-9a-f]{32}/.+", path):
        raise ProtocolError("importable Ref targets must use artifacts/<publication_id>/; request/result projections and .runtime are not targets")


@dataclass(frozen=True)
class FileBytes:
    data: bytes
    mode: int


class Bundle:
    def __init__(self, root: str | Path):
        self.root = Path(root).absolute()
        if self.root.is_symlink() or not self.root.is_dir():
            raise ProtocolError(f"bundle root must be a real directory: {self.root}")
        if (self.root / ".runtime/import-intent.json").exists():
            raise ProtocolError("interrupted import requires recover_import before bundle reads")

    def _file(self, path: str):
        return read_regular(self.root, path)

    def _names(self, directory: str):
        path = self.root / directory
        if path.is_symlink(): raise ProtocolError("source directory is a symlink")
        return {p.name for p in path.iterdir()}

    def _verified(self, reference: Ref) -> FileBytes:
        ref(reference)
        artifact_path(reference["path"])
        with _lock(self.root, shared=True):
            if (self.root / ".runtime/import-intent.json").exists():
                raise ProtocolError("interrupted import requires recover_import before reads")
            data, mode = self._file(reference["path"])
        if sha256(data) != reference["sha256"]:
            raise ProtocolError(f"Ref digest mismatch: {reference['path']}")
        return FileBytes(data, mode)

    def read(self, reference: Ref) -> bytes:
        return self._verified(reference).data

    def document(self, reference: Ref, kind: str | None = None) -> dict:
        data = self.read(reference)
        value = validate(loads(data))
        if data != jcs(value):
            raise ProtocolError(f"normative artifact is not canonical JCS: {reference['path']}")
        if kind and value.get("kind") != kind:
            raise ProtocolError(f"expected {kind} manifest at {reference['path']}")
        return value

    def source(self, reference: Ref) -> bytes:
        """Original single-file API. A tree must never silently lose its files."""
        manifest = self.document(reference, "source")
        if manifest["format"] != "single_file":
            raise UnsupportedError("source() is single_file only; use source_files() for python_tree")
        return self.source_files(reference)["kernel.py"].data

    def source_files(self, reference: Ref) -> dict[str, FileBytes]:
        """Verify the entire snapshot, including extra paths and original modes."""
        manifest = self.document(reference, "source")
        root = str(PurePosixPath(reference["path"]).parent / "files")
        files = {}
        directories = {"": set()}
        for entry in manifest["files"]:
            parts = entry["name"].split("/")
            for index, part in enumerate(parts):
                directory = "/".join(parts[:index])
                directories.setdefault(directory, set()).add(part)
            name = root + "/" + entry["name"]
            data, mode = self._file(name)
            if mode != int(entry["mode"], 8) or len(data) != entry["size"] or sha256(data) != entry["sha256"]:
                raise ProtocolError(f"source bytes/mode/size mismatch: {name}")
            source_bytes(data)
            files[entry["name"]] = FileBytes(data, mode)
        for directory, expected in directories.items():
            if self._names(root + ("/" + directory if directory else "")) != expected:
                raise ProtocolError("Source payload contains extra paths")
        return files

    def closure(self, roots: list[Ref], *, subjects: list[Ref] | None = None) -> dict[str, FileBytes]:
        """Selected roots plus explicitly supplied subject locators; no hash scan."""
        files: dict[str, FileBytes] = {}
        documents: dict[str, dict] = {}
        locators: dict[str, list[Ref]] = {}
        required: list[tuple[str, str]] = []
        queue = list(roots) + list(subjects or [])
        while queue:
            reference = queue.pop(0)
            ref(reference); artifact_path(reference["path"])
            if reference["path"] in files:
                if sha256(files[reference["path"]].data) != reference["sha256"]:
                    raise ProtocolError("same path has conflicting Ref digests")
                continue
            entry = self._verified(reference)
            data = entry.data
            files[reference["path"]] = entry
            locators.setdefault(reference["sha256"], []).append(reference)
            value = _metadata(data)
            if isinstance(value, dict) and "protocol" in value:
                value = loads(data)
                validate(value)
                if data != jcs(value):
                    raise ProtocolError("normative artifact is not JCS")
                documents[reference["sha256"]] = value
                queue.extend(dependencies(value))
                required.extend(subject_hashes(value))
                if value.get("kind") == "source":
                    for name, payload in self.source_files(reference).items():
                        path = str(PurePosixPath(reference["path"]).parent / "files" / name)
                        files[path] = payload
                        identity = sha256(payload.data)
                        locators.setdefault(identity, []).append({"path": path, "sha256": identity})
                elif value.get("kind") == "schema":
                    if value.get("extensions", {}).get("information", {}).get("writing_mode") == "write_only":
                        for raw in value["raw"]:
                            path = PurePosixPath(raw["file"]["path"])
                            if path.stem != raw["node"] or path.suffix not in (".yaml", ".yml"):
                                raise ProtocolError("Schema file must retain its safe node filename")
                            self.read(raw["file"]).decode("utf-8")
                    else:
                        from .template_api import resolve_bundle
                        resolve_bundle(self, value)  # Actual parser provenance, including root/from chains.
            elif isinstance(value, dict) and value.get("format") == "schema_extract.delivery.v2":
                queue.extend(_extract_dependencies(reference, value))
            # Plain text and unrecognized native reports are opaque bytes.
        for name, identity in required:
            if identity not in locators:
                raise ProtocolError(f"selected evidence has no locator for actual subject {name}={identity}; select its immutable subject/request snapshot")
            expected = {"task_sha256": "task", "schema_sha256": "schema", "source_sha256": "source", "rules_sha256": "rules", "scope_sha256": "scope", "evaluator_sha256": "evaluator"}.get(name)
            if expected and documents.get(identity, {}).get("kind") != expected:
                raise ProtocolError(f"{name} resolves to the wrong subject kind")
            if name == "request_sha256" and "request_id" not in documents.get(identity, {}):
                raise ProtocolError("request_sha256 must locate a canonical immutable request")
        _relationships(documents)
        return files


def source_bytes(data: bytes):
    if not isinstance(data, bytes) or data.startswith(b"\xef\xbb\xbf"):
        raise ProtocolError("source must be exact UTF-8 bytes without BOM")
    try:
        data.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ProtocolError("source must be strict UTF-8") from error


def source_manifest(data: bytes) -> dict:
    source_bytes(data)
    return {"protocol": PROTOCOL, "kind": "source", "format": "single_file", "encoding": "utf-8", "entrypoint": "kernel.py", "files": [{"name": "kernel.py", "mode": "0644", "size": len(data), "sha256": sha256(data)}]}


def source_files_manifest(files: dict[str, FileBytes]) -> dict:
    """Encode the finite python_tree format; original bytes remain separate."""
    for payload in files.values():
        source_bytes(payload.data)
    return validate({"protocol": PROTOCOL, "kind": "source", "format": "python_tree",
        "encoding": "utf-8", "entrypoint": "solution.py", "files": [
            {"name": name, "mode": f"{payload.mode:04o}", "size": len(payload.data), "sha256": sha256(payload.data)}
            for name, payload in sorted(files.items())]})


def install_source(bundle: Bundle, reference: Ref, destination: str | Path):
    """Install a whole Source into a new directory; never overlay a live draft."""
    files = bundle.source_files(reference)
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise ProtocolError("source installation requires a new directory; owner must authorize draft replacement separately")
    durable_mkdir(destination.parent)
    stage = Path(tempfile.mkdtemp(prefix=".ks-source-", dir=destination.parent))
    try:
        for name, payload in files.items():
            durable_write(stage / name, payload.data, mode=payload.mode)
        fsync_directory(stage)
        _rename_new(stage, destination)
        fsync_directory(destination.parent)
    finally:
        if stage.exists(): shutil.rmtree(stage)


def verify_source_directory(bundle: Bundle, reference: Ref, directory: str | Path):
    """Exact snapshot check for installed drafts; no hidden/extra files or links."""
    directory = Path(directory)
    expected = bundle.source_files(reference)
    actual = set()
    expected_dirs = {str(p) for name in expected for p in PurePosixPath(name).parents if str(p) != "."}
    if directory.is_symlink() or not directory.is_dir():
        raise ProtocolError("Source directory must be a real directory")
    for parent, dirs, names in os.walk(directory, followlinks=False):
        for name in dirs:
            child = Path(parent) / name
            if child.is_symlink() or child.relative_to(directory).as_posix() not in expected_dirs:
                raise ProtocolError("Source directory contains an extra directory or symlink")
        actual.update((Path(parent) / name).relative_to(directory).as_posix() for name in names)
    if actual != set(expected):
        raise ProtocolError("Source directory differs from complete snapshot")
    for name, payload in expected.items():
        if read_regular(directory, name) != (payload.data, payload.mode):
            raise ProtocolError("installed Source bytes/mode differ: " + name)


def replace_source(bundle: Bundle, reference: Ref, destination: str | Path, *, previous: Ref):
    """Caller-authorized atomic replacement, recoverable with the same two Refs.

    Linux exchange leaves either complete old or complete new directory visible.
    A retry after exchange only removes the verified old snapshot; it never
    overlays files or recreates a budget/operation. Concurrent readers must have
    quiesced before their owner calls this mutation.
    """
    destination = Path(destination).absolute()
    key = sha256(jcs({"path": str(destination), "new": reference, "old": previous}))[7:]
    pending = destination.parent / (".ks-replace-" + key)
    durable_mkdir(destination.parent)
    # Separate from Bundle's directory lock: the bundle can also be the draft's
    # parent, and nested shared reads must not deadlock this exclusive mutation.
    lock_fd = os.open(destination.parent / ".ks-source-replacement.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(lock_fd, "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            verify_source_directory(bundle, reference, destination)
            is_new = True
        except ProtocolError:
            is_new = False
        if is_new:
            if pending.exists() or pending.is_symlink():
                verify_source_directory(bundle, previous, pending)
                shutil.rmtree(pending)
                fsync_directory(destination.parent)
            return
        verify_source_directory(bundle, previous, destination)
        if pending.exists() or pending.is_symlink():
            verify_source_directory(bundle, reference, pending)
        else:
            install_source(bundle, reference, pending)
        fn = ctypes.CDLL(None, use_errno=True).renameat2
        if fn(-100, os.fsencode(pending), -100, os.fsencode(destination), 2):
            number = ctypes.get_errno()
            raise OSError(number, os.strerror(number))
        fsync_directory(destination.parent)
        verify_source_directory(bundle, previous, pending)
        shutil.rmtree(pending)
        fsync_directory(destination.parent)


def dependencies(v: dict) -> list[Ref]:
    """Only contract-declared dependency edges, never arbitrary dictionary walks."""
    out = []
    def add(x):
        if x is not None: out.append(x)
    def checks(x):
        for item in x.values(): add(item["receipt"])
    kind = v.get("kind")
    if "operation" in v and "run_id" not in v:
        for k in ("task", "rules", "materials", "scope", "schema", "seed"):
            add(v.get(k))
        add(v["backend"]["profile"]); add(v["budget"]["parent"])
        out += v.get("feedback", []) + v.get("evidence", [])
    elif "run_id" in v:
        add(v["usage"]["receipts"]); out += v["feedback"]
        if v["operation"] == "implement":
            for x in v["code"].values(): add(x)
            for a in v["assessments"]:
                checks({k: a[k] for k in ("coverage", "build", "correctness", "fidelity", "performance")})
        else:
            add(v["schema"]); add(v["scope"]); checks(v["writing_checks"])
            out += v.get("action_inputs", []) + v.get("action_evidence", [])
            info = v.get("extensions", {}).get("information", {})
            if info.get("writing_mode") == "write_only":
                for key in ("candidate_draft", "change_report", "rationale"):
                    add(info.get(key))
                out += info.get("retained_files", []) + info.get("call_artifacts", [])
    elif kind == "task":
        for k in ("contract", "acceptance", "evaluator", "environment", "workload", "baseline_source", "starter_source"): add(v[k])
    elif kind == "schema":
        out += [r["file"] for r in v["raw"]]
        for k in ("dispatch", "parent", "change_report"): add(v[k])
    elif kind == "rules":
        out += v["files"] + [x["source"] for x in v["tools"]]
    elif kind == "materials": out += [x["file"] for x in v["entries"]]
    elif kind == "backend_profile": add(v["environment"])
    elif kind == "evaluator":
        out += v["files"]
        for k in ("entrypoint", "inputs", "timing", "environment"): add(v[k])
    elif kind == "fidelity":
        add(v["method"]["spec"]); out += v["evidence"] + v["usage_receipts"]
    elif kind == "evaluation":
        checks({k: v[k] for k in ("build", "correctness", "performance")})
        add(v["samples"]); add(v["environment"]); out += v["raw_receipts"] + v["usage_receipts"]
    elif kind == "writing_review": checks(v["checks"]); out += v["evidence"] + v["usage_receipts"]
    elif kind == "change_report": out += v["raw"] + v["evidence"]
    elif kind == "feedback": out += v["evidence"]
    elif kind == "usage_index":
        for row in v["entries"]: add(row["receipt"])
    return out


def subject_hashes(v: dict) -> list[tuple[str, str]]:
    if v.get("kind") == "schema": return [("task_sha256", v["task_sha256"])]
    if "run_id" in v:
        return [("request_sha256", v["request_sha256"])] + [("source_sha256", a["source_sha256"]) for a in v.get("assessments", [])]
    return list(v.get("subject", {}).items())


def _relationships(documents: dict[str, dict]):
    def expected(reference, kind):
        if reference is not None and documents.get(reference["sha256"], {}).get("kind") != kind:
            raise ProtocolError(f"expected actual {kind} dependency")
    for v in documents.values():
        kind = v.get("kind")
        if kind == "task":
            expected(v["evaluator"], "evaluator")
            expected(v["baseline_source"], "source")
            expected(v["starter_source"], "source")
        if kind == "schema": expected(v["change_report"], "change_report")
        if "operation" in v and "run_id" not in v:
            for key in ("task", "rules", "materials", "scope", "schema"):
                if key in v: expected(v[key], key)
            expected(v.get("seed"), "source")
            expected(v["backend"]["profile"], "backend_profile")
            expected(v["budget"]["parent"], "budget_account")
            if "schema" in v and documents[v["schema"]["sha256"]]["task_sha256"] != v["task"]["sha256"]:
                raise ProtocolError("request Schema belongs to another Task")
        if "run_id" in v:
            expected(v["usage"]["receipts"], "usage_index")
            if v["operation"] == "implement":
                for reference in v["code"].values(): expected(reference, "source")
            else:
                expected(v["schema"], "schema"); expected(v["scope"], "scope")
        subject = v.get("subject", {})
        if "schema_sha256" in subject and documents[subject["schema_sha256"]]["task_sha256"] != subject["task_sha256"]:
            raise ProtocolError("evidence Schema belongs to another subject Task")
        if "request_sha256" in subject:
            request = documents[subject["request_sha256"]]
            for key in ("task", "rules", "scope", "schema"):
                field = key + "_sha256"
                if field not in subject or key not in request: continue
                if kind == "writing_review" and key in ("scope", "schema"): continue
                if subject[field] != request[key]["sha256"]:
                    raise ProtocolError(f"evidence {key} differs from its actual request subject")
        if kind == "fidelity":
            rules = documents[subject["rules_sha256"]]
            if v["method"]["spec"] not in rules["files"]:
                raise ProtocolError("fidelity method is not in the frozen Rules")
        if kind == "evaluation":
            task = documents[subject["task_sha256"]]
            if task["evaluator"]["sha256"] != subject["evaluator_sha256"]:
                raise ProtocolError("evaluation evaluator differs from subject Task evaluator")
        if v.get("kind") == "schema" and v["parent"]:
            parent = documents.get(v["parent"]["sha256"], {})
            if parent.get("kind") != "schema": raise ProtocolError("Schema.parent must refer to Schema")
            if v["change"] != "new_task" and parent["task_sha256"] != v["task_sha256"]:
                raise ProtocolError("same-Task Schema change has a different parent Task")
            expected_revision = parent["design_revision"] + (v["change"] == "new_revision")
            if v["change"] != "new_task" and v["design_revision"] != expected_revision:
                raise ProtocolError("incorrect design_revision relative to actual parent")
        if v.get("kind") == "evaluation":
            evaluator = documents.get(v["subject"]["evaluator_sha256"])
            if evaluator:
                for name in ("inputs", "timing"):
                    if evaluator[name]["sha256"] != v["subject"][name + "_sha256"]:
                        raise ProtocolError(f"evaluation {name} subject differs from evaluator")


def _extract_dependencies(reference, value):
    # The native delivery protocol declares checkpoint files relative to its
    # publication root, two directories above .extraction/receipt.json.
    path = PurePosixPath(reference["path"])
    if path.parts[-2:] != (".extraction", "receipt.json") or not isinstance(value.get("files"), dict):
        raise ProtocolError("invalid native extraction receipt location or files")
    out = []
    for name, identity in value["files"].items():
        relative_path(name)
        if not isinstance(identity, str): raise ProtocolError("native receipt digest must be a string")
        identity = identity if identity.startswith("sha256:") else "sha256:" + identity
        dependency = {"path": str(path.parent.parent / name), "sha256": identity}
        ref(dependency); artifact_path(dependency["path"]); out.append(dependency)
    return out


@contextmanager
def _lock(root: Path, *, shared=False):
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _target_preflight(root: Path, files: dict[str, FileBytes]):
    missing = {}
    for name, entry in files.items():
        path = root
        parts = name.split("/")
        for part in parts[:-1]:
            path /= part
            if path.is_symlink() or (path.exists() and not path.is_dir()):
                raise ProtocolError(f"destination path boundary conflict: {name}")
        target = root / name
        if target.exists() or target.is_symlink():
            actual, mode = read_regular(root, name)
            if actual != entry.data or mode != entry.mode:
                raise ProtocolError(f"immutable destination conflict (bytes/mode): {name}")
        else: missing[name] = entry
    return missing


def recover_import(destination: str | Path):
    """Roll back a crashed import before allowing bundle readers to proceed."""
    root = Path(destination).absolute()
    with _lock(root):
        _recover_import(root)


def _recover_import(root: Path):
    journal = root / ".runtime/import-intent.json"
    if not journal.exists(): return
    data, _ = read_regular(root, ".runtime/import-intent.json")
    record = loads(data)
    for name, expected in record["files"].items():
        artifact_path(name)
        path = root / name
        if path.exists() or path.is_symlink():
            content, mode = read_regular(root, name)
            if sha256(content) != expected["sha256"] or mode != expected["mode"]:
                raise ProtocolError("interrupted import target changed; preserve fence and inspect manually")
    for name in reversed(list(record["files"])):
        path = root / name
        if path.exists(): path.unlink(); fsync_directory(path.parent)
    for name in reversed(record["directories"]):
        relative_path(name)
        path = root / name
        if path.exists(): path.rmdir(); fsync_directory(path.parent)
    journal.unlink(); fsync_directory(journal.parent)
    if not record.get("runtime_existed", True) and not any(journal.parent.iterdir()):
        journal.parent.rmdir(); fsync_directory(root)


def import_bundle(source: Bundle | str | Path, destination: str | Path, roots: list[Ref], *, subjects: list[Ref] | None = None) -> list[Ref]:
    """Prevalidate a selected closure, stage all bytes, then merge transactionally.

    A durable rollback intent fences Bundle readers after an interrupted merge.
    Same bytes/mode are idempotent. Invalid input/conflicts perform no writes.
    This is a local Linux transaction, not a cross-machine storage service.
    """
    source = source if isinstance(source, Bundle) else Bundle(source)
    files = source.closure(roots, subjects=subjects)
    root = Path(destination).absolute()
    if root.is_symlink(): raise ProtocolError("destination cannot be a symlink")
    existed = root.exists()
    if existed:
        _target_preflight(root, files)  # Error path leaves target completely untouched.
    durable_mkdir(root.parent)
    if not existed:
        stage = Path(tempfile.mkdtemp(prefix=".ks-bundle-", dir=root.parent))
        try:
            for name, entry in files.items(): durable_write(stage / name, entry.data, mode=entry.mode)
            fsync_directory(stage)
            _rename_new(stage, root); fsync_directory(root.parent)
        finally:
            if stage.exists(): shutil.rmtree(stage)
        return roots
    with _lock(root):
        _recover_import(root)
        missing = _target_preflight(root, files)
        if not missing: return roots
        stage = Path(tempfile.mkdtemp(prefix=".ks-import-", dir=root.parent))
        journal = root / ".runtime/import-intent.json"
        runtime_existed = journal.parent.exists()
        try:
            for name, entry in missing.items(): durable_write(stage / name, entry.data, mode=entry.mode)
            directories = set()
            for name in missing:
                parent = PurePosixPath(name).parent
                while str(parent) != ".":
                    if not (root / parent).exists(): directories.add(str(parent))
                    parent = parent.parent
            ordered_dirs = sorted(directories, key=lambda x: (x.count("/"), x))
            intent = {"files": {n: {"sha256": sha256(e.data), "mode": e.mode} for n, e in missing.items()}, "directories": ordered_dirs, "runtime_existed": runtime_existed}
            durable_write(journal, jcs(intent))
            for name in ordered_dirs: durable_mkdir(root / name)
            for name in missing:
                _rename_new(stage / name, root / name); fsync_directory((root / name).parent)
            journal.unlink(); fsync_directory(journal.parent)
        except BaseException:
            if journal.exists(): _recover_import(root)
            raise
        finally:
            shutil.rmtree(stage)
            if not runtime_existed and journal.parent.exists() and not any(journal.parent.iterdir()):
                journal.parent.rmdir(); fsync_directory(root)
    return roots


class Publication:
    """One durable publication ID and immutable staging bytes; safe to reopen."""
    def __init__(self, root: str | Path, publication_id: str | None = None):
        self.root = Path(root).absolute()
        durable_mkdir(self.root)
        self.publication_id = publication_id or uuid4().hex
        if not re.fullmatch(r"[0-9a-f]{32}", self.publication_id): raise ProtocolError("publication_id must be 128-bit lowercase hex")
        self.prefix = f"artifacts/{self.publication_id}"
        self.intent = self.root / f".runtime/publications/{self.publication_id}.json"
        initial = {"publication_id": self.publication_id, "state": "intent"}
        if not self.intent.exists():
            durable_write(self.intent, jcs(initial))
            self.record = initial
        else:
            value = loads(read_regular(self.root, str(self.intent.relative_to(self.root)))[0])
            if value["publication_id"] != self.publication_id: raise ProtocolError("publication intent mismatch")
            self.record = value
        self.stage = self.root / f".runtime/staging/{self.publication_id}"
        durable_mkdir(self.stage)

    def put_bytes(self, name: str, data: bytes, *, mode=0o644) -> Ref:
        relative_path(name)
        path = f"{self.prefix}/{name}"
        target = self.stage / path
        if target.exists() or target.is_symlink():
            previous, old_mode = read_regular(self.stage, path)
            if previous != data or old_mode != mode: raise ProtocolError("publication path is immutable; allocate a new publication")
        else:
            if self.record["state"] == "published":
                raise ProtocolError("publication is frozen; allocate a new publication ID")
            durable_write(target, data, mode=mode)
        return {"path": path, "sha256": sha256(data)}

    def put_json(self, name: str, value: dict) -> Ref:
        validate(value)
        return self.put_bytes(name, jcs(value))

    def put_source(self, name: str, data: bytes) -> Ref:
        manifest = source_manifest(data)
        self.put_bytes(name + "/files/kernel.py", data)
        return self.put_json(name + "/manifest.json", manifest)

    def put_source_files(self, name: str, files: dict[str, FileBytes]) -> Ref:
        manifest = source_files_manifest(files)
        for path, payload in sorted(files.items()):
            self.put_bytes(name + "/files/" + path, payload.data, mode=payload.mode)
        return self.put_json(name + "/manifest.json", manifest)

    def commit(self, roots: list[Ref], *, subjects: list[Ref] | None = None):
        overlay = _OverlayBundle(self.stage, self.root)
        files = overlay.closure(roots, subjects=subjects)
        record = {"publication_id": self.publication_id, "state": "published", "roots": roots, "files": {p: sha256(e.data) for p, e in files.items()}}
        if self.record["state"] == "published" and record != self.record:
            raise ProtocolError("publication is frozen; different closure requires a new publication ID")
        import_bundle(overlay, self.root, roots, subjects=subjects)
        durable_write(self.intent, jcs(record), replace=True)
        self.record = record
        return roots


class _OverlayBundle(Bundle):
    """Read our new staged bytes plus already immutable published subjects."""
    def __init__(self, stage, published):
        super().__init__(stage)
        self.published = Bundle(published)

    def _file(self, path):
        candidate = self.root / path
        if candidate.exists() or candidate.is_symlink(): return super()._file(path)
        return self.published._file(path)

    def _names(self, directory):
        names = set()
        if (self.root / directory).exists(): names |= super()._names(directory)
        if (self.published.root / directory).exists(): names |= self.published._names(directory)
        return names


def _metadata(data):
    # Native JSON may contain floating statistics. Inspect its declared protocol
    # without reserializing it; strict workflow parsing happens only afterwards.
    try:
        return json.loads(data)
    except (ValueError, UnicodeError, RecursionError):
        return None
