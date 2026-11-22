"""Local file handling and bounded Claude writing into isolated candidates. No public API."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import time

CHECKOUT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "qwen3.8-max[1m]"
CONNECTION = {
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "ANTHROPIC_CUSTOM_HEADERS",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    "NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE", "SSL_CERT_DIR",
}
DOC_NAMES = (
    "0.common.md", "1.contract.algorithm_requirement.md", "2.contract.hardware_specification.md",
    "3.design.math_reshape.md", "4.design.work_schedule.md", "5.design.tile_pipeline.md", "6.realization.md",
)
PROTECTED = ("schema_extract", "schema_docs", "schema_template", "schema_wiki", "kernel_wiki")
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_PROMPT_BYTES = 16 * 1024 * 1024


class Failure(Exception):
    def __init__(self, reason, message, *, repairable=True):
        super().__init__(message)
        self.reason = reason
        self.repairable = repairable


class Interrupted(Exception):
    pass


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("non-finite JSON")))


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def no_links(path):
    path = Path(os.path.abspath(path))
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise Failure("needs_input", f"Symlinks are not accepted as materials: {path}")
    return path


def input_path(value):
    path = no_links(Path(value).expanduser())
    if not path.exists():
        raise Failure("needs_input", f"Input is missing: {path}")
    forbidden = [Path.home() / name for name in (".claude", ".codex", ".ssh", ".aws")]
    if os.environ.get("CLAUDE_CONFIG_DIR"):
        forbidden.append(Path(os.environ["CLAUDE_CONFIG_DIR"]).expanduser().resolve())
    if any(path == directory or directory in path.parents for directory in forbidden):
        raise Failure("needs_input", "Authentication directories are not writing materials")
    return path


def regular_files(directory):
    files = []
    for current, dirs, names in os.walk(directory, followlinks=False):
        for name in dirs + names:
            item = no_links(Path(current) / name)
            if item.is_file():
                files.append(item)
            elif not item.is_dir():
                raise Failure("needs_input", f"Material must be an ordinary file: {item}")
    return sorted(files)


def git_provenance(source, data):
    """Describe only a locally verified commit whose blob equals the fixed bytes."""
    root = next((parent for parent in source.parents if (parent / ".git").exists()), None)
    if root is None:
        return {"status": "not_in_repository"}
    environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8",
                   "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                   "GIT_OPTIONAL_LOCKS": "0", "GIT_NO_REPLACE_OBJECTS": "1",
                   "GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0"}

    def git(*arguments):
        try:
            result = subprocess.run(["git", "--literal-pathspecs", "-C", str(root), *arguments],
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                    env=environment, timeout=10, shell=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout if result.returncode == 0 else None

    if git("rev-parse", "--show-toplevel") != os.fsencode(root) + b"\n":
        return {"status": "unavailable"}
    revision = git("rev-parse", "--verify", "--end-of-options", "HEAD^{commit}")
    if revision is None or not re.fullmatch(rb"[0-9a-f]{40}(?:[0-9a-f]{24})?\n", revision):
        return {"status": "unavailable"}
    commit = revision.decode("ascii").strip()
    relative = source.relative_to(root).as_posix()
    facts = {"repository": root.name, "commit": commit, "path": relative}
    tree = git("ls-tree", "-l", "-z", "--full-tree", commit, "--", relative)
    if tree is None:
        return {**facts, "status": "unavailable"}
    entries = {name: metadata.split() for entry in tree.split(b"\0") if entry
               for metadata, _, name in (entry.partition(b"\t"),)}
    entry = entries.get(os.fsencode(relative))
    if entry is None:
        return {**facts, "status": "absent_from_commit"}
    if len(entry) != 4 or entry[0] not in {b"100644", b"100755"} or entry[1] != b"blob":
        return {**facts, "status": "not_a_regular_blob"}
    facts["blob"] = entry[2].decode("ascii")
    if entry[3] != str(len(data)).encode("ascii"):
        return {**facts, "status": "different_bytes"}
    committed = git("cat-file", "blob", facts["blob"])
    if committed is None:
        return {**facts, "status": "unavailable"}
    if committed != data:
        return {**facts, "status": "different_bytes"}
    if not re.fullmatch(r"[^@<>:\s]+", root.name):
        return {**facts, "status": "unusable_repository_identity"}
    return {**facts, "status": "verified", "locator": f"{root.name}@{commit}:{relative}"}


class Inputs:
    """One fixed byte snapshot; locators continue to name their original context."""
    def __init__(self, out):
        self.out = out
        self.root = out / "input"
        self.root.mkdir()
        self.records = {}
        self.total = 0
        self.bindings = {}

    def save(self):
        atomic_json(self.root / "manifest.json", {
            "files": list(self.records.values()), "bindings": self.bindings,
        })

    def file(self, source, destination=None):
        source = input_path(source)
        if not source.is_file():
            raise Failure("needs_input", f"Expected a file: {source}")
        before = source.stat()
        if str(source) not in self.records and before.st_size + self.total > MAX_INPUT_BYTES:
            raise Failure("needs_input", "Material exceeds the 64 MiB local snapshot limit; nothing was truncated")
        data = source.read_bytes()
        after = source.stat()
        if (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size) != (
                after.st_dev, after.st_ino, after.st_mtime_ns, after.st_size) or data != source.read_bytes():
            raise Failure("input_changed", f"Material changed while being fixed: {source}")
        digest = sha(data)
        old = self.records.get(str(source))
        if old and old["sha256"] != digest:
            raise Failure("input_changed", f"Material changed during input collection: {source}")
        if old and destination is None:
            return self.out / old["snapshot"]
        if destination is None:
            destination = self.root / "materials" / f"{len(self.records) + 1:04d}" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        if not old:
            self.total += len(data)
        self.records[str(source)] = {"source": str(source), "snapshot": str(destination.relative_to(self.out)),
                                     "sha256": digest, "bytes": len(data), "git": git_provenance(source, data)}
        self.save()
        return destination

    def tree(self, source, label):
        source = input_path(source)
        destination = self.root / label
        if source.is_file():
            return self.file(source, destination / source.name)
        files = regular_files(source)
        if not files:
            raise Failure("needs_input", f"Empty input directory: {source}")
        destination.mkdir(parents=True)
        before = {str(p.relative_to(source)): sha(p.read_bytes()) for p in files}
        for path in files:
            self.file(path, destination / path.relative_to(source))
        after = {str(p.relative_to(source)): sha(p.read_bytes()) for p in regular_files(source)}
        if before != after:
            raise Failure("input_changed", f"Directory changed while being fixed: {source}")
        return destination

    def closure(self, source, seen=None):
        """Fix explicit local links, resolving all feedback references at their source."""
        seen = set() if seen is None else seen
        source = input_path(source)
        if source in seen:
            return
        seen.add(source)
        if source.is_dir():
            # Directory material includes all helpers and build sources.
            for item in regular_files(source):
                self.closure(item, seen)
            return
        snapshot = self.file(source)
        try:
            content = snapshot.read_text(encoding="utf-8")
        except UnicodeError:
            return
        references = []
        if source.suffix == ".json":
            try:
                value = strict_json(content)
            except ValueError:
                value = None
            path_keys = {"path", "schema", "kernel", "feedback", "notes", "request", "response", "source_path",
                         "request_path", "response_path", "feedback_path", "notes_path", "kernel_path", "archive"}
            def walk(item, binding=False):
                if isinstance(item, dict):
                    for key, part in item.items():
                        if (key in path_keys and isinstance(part, str) and "\n" not in part
                                and (binding or (source.parent / part).exists() or "/" in part or re.search(r"\.(?:json|md|txt|ya?ml|py|cu|cpp)$", part))):
                            references.append((part, key != "path" or binding))
                        else:
                            walk(part, binding=key in {"schema", "kernel"})
                elif isinstance(item, list):
                    for part in item:
                        walk(part)
            walk(value)
        if source.suffix in {".md", ".txt"}:
            references.extend((match, True) for match in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content))
            references.extend((match, False) for match in re.findall(r"`([^`\n]+)`", content)
                              if "/" in match or re.search(r"\.(?:py|json|ya?ml|md|cu|cpp)$", match))
        for value, required in references:
            value = value.strip().strip("<>").split("#", 1)[0]
            if not value or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", value) or "@" in value or "*" in value:
                continue
            path = Path(value).expanduser()
            path = path if path.is_absolute() else source.parent / path
            if path.exists():
                self.closure(path, seen)
            elif required:
                raise Failure("needs_input", f"Missing local material referenced by {source}: {value}")

    def schema(self, source, label):
        from schema_template.tools import schema as template
        source = input_path(source)
        directory = source if source.is_dir() else source.parent
        files, documents = template.read_directory(directory)
        resolved = template.resolve_documents(files, documents)
        # Check identifiable repository locators before relocating their raw files.
        template.validate_local_repositories(files, documents)
        dispatch = directory / "dispatch.md"
        if not dispatch.is_file():
            raise Failure("needs_input", f"Schema is missing dispatch.md: {directory}")
        target = self.root / label
        target.mkdir(parents=True)
        for name, path in files.items():
            fixed = self.file(path, target / path.name)
            if sha(fixed.read_bytes()) != resolved.source_hashes[name]:
                raise Failure("input_changed", f"Schema changed during resolution: {path}")
        self.file(dispatch, target / "dispatch.md")
        if set(template.task_files(directory)) != set(files):
            raise Failure("input_changed", f"Schema nodes changed during resolution: {directory}")
        views = {name: {"effective_yaml": template.effective_yaml(resolved, name),
                        "source_manifest": resolved.source_manifest(name), "origins": resolved.origins_dict(name),
                        "effective_identity": resolved.effective_identity(name)} for name in files}
        self.bindings[label] = {"source": str(directory), "snapshot": str(target.relative_to(self.out)), "nodes": views}
        self.save()
        return target, resolved

    def text(self, paths):
        sections = []
        records = {self.out / item["snapshot"]: item for item in self.records.values()}
        for path in dict.fromkeys(paths):
            path = Path(path)
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeError as exc:
                raise Failure("needs_input", f"Required text material is not UTF-8: {path}") from exc
            record = records.get(path)
            provenance = "" if record is None else (
                "Original source (locator context only): " + record["source"] + "\n" +
                "Host Git byte verification: " + json.dumps(record["git"], ensure_ascii=False) + "\n")
            sections.append(f"\n{provenance}--- COMPLETE FILE {path} ---\n{content}\n--- END FILE ---\n")
        text = "".join(sections)
        if len(text.encode()) > MAX_PROMPT_BYTES:
            raise Failure("needs_input", "Required text exceeds the 16 MiB request limit; nothing was truncated")
        return text


def connection_environment(effort):
    # Do not relocate HOME or login state. Only connection fields bypass settings suppression.
    environment = {key: os.environ[key] for key in ("PATH", "HOME", "CLAUDE_CONFIG_DIR", "XDG_CONFIG_HOME",
                                                    "XDG_CACHE_HOME", "TMPDIR") if key in os.environ}
    environment.setdefault("HOME", str(Path.home()))
    environment.setdefault("PATH", "/usr/bin:/bin")
    source = Path(environment.get("CLAUDE_CONFIG_DIR", str(Path(environment["HOME"]) / ".claude"))).expanduser()
    settings = source / "settings.json"
    try:
        configured = strict_json(settings.read_bytes()).get("env", {}) if settings.exists() else {}
        if not isinstance(configured, dict):
            raise ValueError("env must be an object")
        for key in CONNECTION:
            value = os.environ.get(key, configured.get(key))
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError("connection values must be strings")
                environment[key] = value
    except (OSError, ValueError, AttributeError) as exc:
        raise Failure("needs_input", "Cannot read the existing Claude connection configuration") from exc
    environment.update(LANG="C.UTF-8", LC_ALL="C.UTF-8", CLAUDE_CODE_EFFORT_LEVEL=effort,
                       CLAUDE_CODE_SAFE_MODE="1", CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
                       DISABLE_AUTOUPDATER="1", CLAUDE_CODE_MAX_OUTPUT_TOKENS="128000")
    return environment


def command(args, output_schema, read_dirs, candidate=None):
    tools = ["Read", "Glob", "Grep"] + (["Write", "Edit"] if candidate else [])
    allowed = ["Read", "Glob", "Grep"]
    if candidate:
        # Native // rules are absolute; a single * matches one path component.
        literal = re.sub(r"([\\*?\[\]{}()!,])", r"\\\1", str(candidate))
        allowed.extend(f"{tool}(/{literal}/{name})" for tool in ("Write", "Edit")
                       for name in ("*.yaml", "*.yml", "dispatch.md"))
    settings = {"disableAllHooks": True, "permissions": {"allow": allowed}}
    denied = ["Bash", "Agent", "Task", "Skill", "WebFetch", "WebSearch"]
    if not candidate:
        denied += ["Edit", "Write"]
    result = ["claude", "-p", "--model", args.model, "--effort", args.effort,
              "--output-format", "stream-json", "--verbose", "--json-schema", json.dumps(output_schema),
              "--safe-mode", "--restricted", "--disable-slash-commands", "--no-chrome", "--setting-sources", "",
              "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}', "--settings", json.dumps(settings),
              "--permission-mode", "dontAsk", "--permission-prompts", "none", "--tools", ",".join(tools),
              "--disallowedTools", ",".join(denied), "--no-session-persistence"]
    for directory in sorted(set(map(str, read_dirs))):
        result.extend(("--add-dir", directory))
    return result


def parse_stream(stdout, stderr, returncode):
    draft = {"final_answer": None, "structured_output": None, "reads": [], "session_ids": [],
             "returncode": returncode, "diagnostics": [], "stderr": stderr, "writes": [],
             "public_status": {"assistant_messages": 0}, "errors": []}
    tools, terminal = {}, None
    final_text = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = strict_json(line)
        except ValueError:
            draft["diagnostics"].append("Non-JSON output from Claude")
            continue
        if not isinstance(event, dict):
            draft["diagnostics"].append("Unexpected Claude event")
            continue
        if event.get("session_id") and event["session_id"] not in draft["session_ids"]:
            draft["session_ids"].append(event["session_id"])
        if event.get("type") == "assistant":
            message = event.get("message", {})
            draft["public_status"]["assistant_messages"] += 1
            for key in ("model", "stop_reason", "usage"):
                if key in message:
                    draft["public_status"]["last_" + key] = message[key]
            if event.get("error"):
                draft["errors"].append(event["error"])
            chunks = message.get("content", [])
            final_text = [item["text"] for item in chunks if item.get("type") == "text"]
            for item in chunks:
                if item.get("type") != "tool_use":
                    continue
                name = item.get("name")
                if name not in {"Read", "Glob", "Grep", "Write", "Edit", "StructuredOutput"}:
                    draft["diagnostics"].append(f"Unexpected tool: {name}")
                if name in {"Read", "Glob", "Grep"}:
                    record = {"tool": name, "input": item.get("input", {}), "status": "requested"}
                    draft["reads"].append(record)
                    tools[item.get("id")] = record
                elif name in {"Write", "Edit"}:
                    record = {"tool": name, "input": {"file_path": item.get("input", {}).get("file_path")},
                              "status": "requested"}
                    draft["writes"].append(record)
                    tools[item.get("id")] = record
        if event.get("type") == "user":
            for item in event.get("message", {}).get("content", []):
                if item.get("type") == "tool_result" and item.get("tool_use_id") in tools:
                    record = tools[item["tool_use_id"]]
                    record["status"] = "error" if item.get("is_error") else "completed"
                    if item.get("is_error"):
                        content = item.get("content", "")
                        record["error"] = content if isinstance(content, str) else [
                            part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
        if event.get("type") == "error":
            draft["diagnostics"].append("Claude reported an error event")
            draft["errors"].append({key: event[key] for key in ("code", "message", "error") if key in event})
        if event.get("type") == "result":
            if terminal is not None:
                draft["diagnostics"].append("Multiple terminal results")
            terminal = event
    draft["final_answer"] = terminal.get("result") if terminal and "result" in terminal else "\n".join(final_text)
    if terminal:
        draft["structured_output"] = terminal.get("structured_output")
        draft["public_status"]["terminal"] = {key: terminal[key] for key in (
            "subtype", "is_error", "errors", "terminal_reason", "num_turns", "duration_ms", "duration_api_ms",
            "usage", "modelUsage", "total_cost_usd") if key in terminal}
    if (returncode != 0 or not terminal or terminal.get("subtype") != "success"
            or terminal.get("is_error") is not False or terminal.get("errors")
            or terminal.get("interrupted_by_shutdown") or terminal.get("deferred_tool_use")
            or terminal.get("queued_turn_count", 0) or terminal.get("terminal_reason") not in (None, "completed")):
        draft["diagnostics"].append("Claude did not finish a successful response")
    return draft


def stop_process(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        output = process.communicate(timeout=3)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output = process.communicate(timeout=3)
    # Even a exited group leader can leave descendants behind.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return output
    for _ in range(10):
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return output
        time.sleep(0.05)
    raise Interrupted("Cannot confirm that the previous Claude process group has exited")


def native_call(args, prompt, output_schema, out, draft_path, seed):
    environment = connection_environment(args.effort)
    def scrub(text):
        for key in CONNECTION:
            value = environment.get(key)
            if value and (key in {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_CUSTOM_HEADERS"} or len(value) >= 8):
                text = text.replace(value, "[connection value redacted]")
        return text
    process = None
    started = time.monotonic()
    candidate = out / seed["candidate_directory"] if seed.get("candidate_directory") else None
    try:
        process = subprocess.Popen(command(args, output_schema, [out / "input"], candidate), cwd=out,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding="utf-8", env=environment, shell=False, start_new_session=True)
        seed["pid"] = process.pid
        atomic_json(draft_path, seed)
        stdout, stderr = process.communicate(prompt, timeout=args.timeout_s)
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            pass
        else:
            stop_process(process)
        draft = parse_stream(stdout, stderr, process.returncode)
    except subprocess.TimeoutExpired:
        stdout, stderr = stop_process(process)
        draft = parse_stream(stdout, stderr, process.returncode)
        draft["diagnostics"].append(f"Claude exceeded {args.timeout_s:g} seconds; process group reaped")
    except KeyboardInterrupt as exc:
        if process is not None:
            stdout, stderr = stop_process(process)
            seed.update(parse_stream(stdout, stderr, process.returncode))
        seed["diagnostics"] = ["Writing interrupted; process group reaped"]
        atomic_json(draft_path, strict_json(scrub(json.dumps(seed, ensure_ascii=False))))
        raise Interrupted("Writing interrupted") from exc
    except OSError as exc:
        draft = {"final_answer": None, "structured_output": None, "reads": [], "diagnostics": [scrub(str(exc))]}
    draft = strict_json(scrub(json.dumps(draft, ensure_ascii=False)))
    seed.update(draft)
    seed["state"] = "finished"
    seed["elapsed_s"] = round(time.monotonic() - started, 3)
    atomic_json(draft_path, seed)
    return seed


WRITE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["files", "notes", "missing"],
                "properties": {"files": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
                               "notes": {"type": "string"},
                               "missing": {"type": "array", "items": {"type": "string"}}}}


def candidate_name(name):
    path = PurePosixPath(name)
    return (len(path.parts) == 1 and name == path.name and "\\" not in name
            and (path.suffix in {".yaml", ".yml"} or name == "dispatch.md"))


def candidate_files(directory):
    """Read only ordinary, direct, allowed candidate files after CLI exit."""
    if directory.is_symlink() or not directory.is_dir():
        raise Failure("write_failed", "Candidate directory was replaced", repairable=False)
    files, total = [], 0
    for path in sorted(directory.iterdir()):
        if not candidate_name(path.name) or path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise Failure("write_failed", f"Unexpected candidate entry: {path.name}", repairable=False)
        total += path.stat().st_size
        if total > MAX_INPUT_BYTES:
            raise Failure("write_failed", "Candidate exceeds 64 MiB; no file was truncated", repairable=False)
        files.append({"path": path.name, "content": path.read_bytes().decode("utf-8")})
    return files


def snapshot_hashes(directory):
    return {str(path.relative_to(directory)): sha(path.read_bytes()) for path in regular_files(directory)}


def accept_files(files, out, contract=None, source_context=None):
    from schema_template.tools import schema as template
    if not files:
        raise Failure("write_failed", "Response did not include complete Schema files")
    names = set()
    for item in files:
        path = PurePosixPath(item["path"])
        name = item["path"]
        if (path.is_absolute() or len(path.parts) != 1 or name != path.name or "\\" in name
                or name in names or (path.suffix not in {".yaml", ".yml"} and name != "dispatch.md")):
            raise Failure("write_failed", f"Expected distinct, direct YAML nodes and dispatch.md; rejected {name!r}")
        names.add(name)
    if "dispatch.md" not in names or not {"root.yaml", "root.yml"} & names:
        raise Failure("write_failed", "Complete delivery requires root YAML and dispatch.md")
    target = out / "schema"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir()
    for item in files:
        (target / item["path"]).write_text(item["content"], encoding="utf-8")
    if contract is not None:
        _, raw = template.read_directory(target)
        if not exact_value(raw["root"].value.get("CONTRACT"), contract):
            raise Failure("write_failed", "CONTRACT parsed values changed before normalization")
    plan = template.normalize_directory(target)
    for name, content in plan.candidates.items():
        path = plan.files[name]
        if sha(path.read_bytes()) != plan.source_hashes[name]:
            raise Failure("input_changed", "Candidate changed during normalization")
        path.write_text(content, encoding="utf-8")
    _, documents = template.read_directory(target)
    if contract is not None and not exact_value(documents["root"].value.get("CONTRACT"), contract):
        raise Failure("write_failed", "CONTRACT parsed values changed after normalization")
    notices = [f"{item.severity}: {item.path}: {item.message}" for item in plan.corrections]
    template.validate_directory(target, notices=notices)
    if source_context is not None:
        template.validate_local_repositories({name: source_context / file.name for name, file in plan.files.items()}, documents)
    return target, template.resolve_directory(target), notices


def exact_value(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact_value(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(exact_value(a, b) for a, b in zip(left, right))
    return left == right


class Run:
    def __init__(self, args, mode):
        self.args, self.mode, self.out = args, mode, args.out
        self.result = {"status": "interrupted", "stop_reason": "interrupted", "calls_used": 0,
                       "schema": None, "notes": "notes.md"}
        if mode == "optimize":
            self.result.update(decision=None, required_fix=None)
        self.notes = []
        self.save()
        self.inputs = Inputs(self.out)
        self.inline_reads = []

    def save(self):
        (self.out / "notes.md").write_text("\n\n".join(self.notes) + "\n", encoding="utf-8")
        atomic_json(self.out / "result.json", self.result)

    def finish(self, status, reason, note=None, **values):
        if note:
            self.notes.append(str(note))
        self.result.update(status=status, stop_reason=reason, **values)
        self.save()
        return {"complete": 0, "incomplete": 1, "interrupted": 2}[status]

    def preparation(self):
        task = self.inputs.file(self.args.task, self.inputs.root / "task" / self.args.task.name)
        self.inputs.closure(self.args.task)
        task_materials = [self.out / item["snapshot"] for item in self.inputs.records.values()]
        docs = [self.inputs.file(CHECKOUT / "schema_docs" / name, self.inputs.root / "schema_docs" / name)
                for name in DOC_NAMES]
        method = self.inputs.file(CHECKOUT / f"{self.mode.upper()}_SCHEMA.md", self.inputs.root / f"{self.mode.upper()}_SCHEMA.md")
        template_files = [self.inputs.file(CHECKOUT / "schema_template" / name,
                                          self.inputs.root / "schema_template" / name) for name in ("README.md", "starter.yaml")]
        # A small, complete existing format example; never the target Task's answer.
        example = CHECKOUT / "schema_template" / "examples" / "qsa_prefill"
        example_files = [self.inputs.file(example / name,
                        self.inputs.root / "schema_template" / "examples" / example.name / name)
                        for name in ("README.md", "root.yaml", "dispatch.md")]
        prompt_file = CHECKOUT / f"schema_{self.mode}" / "prompt" / f"{self.mode}.md"
        prompt = self.inputs.file(prompt_file, self.inputs.root / "instructions.md")
        self.inline_reads = [str(path) for path in (docs[0], method, prompt, *task_materials)]
        text = self.inputs.text([prompt, docs[0], method, *task_materials])
        text += "\nREADABLE DOC1–6 and Template (use Read for the relevant sections before writing):\n"
        text += "\n".join(map(str, [*docs[1:], *template_files]))
        text += "\nOPTIONAL complete Template format example (not current Task/device/stack evidence):\n"
        text += "\n".join(map(str, example_files))
        text += f"\nInput manifest (original paths, fixed bytes, verified Git facts, bindings): {self.inputs.root / 'manifest.json'}\n"
        text += ("Read/Glob/Grep may inspect this input snapshot and the current candidate directory. Original paths are locator context only. "
                 "For a relative link, identify its original target and use the matching manifest snapshot; "
                 "unlisted targets are not supplied. Do not search the original checkout or .git.\n")
        return text

    def repair_prompt(self, error):
        path = self.out / "drafts" / f"{self.result['calls_used']:02d}.json"
        draft = strict_json(path.read_bytes())
        draft.setdefault("host_diagnostics", []).append(str(error))
        atomic_json(path, draft)
        return ("\nHOST DIAGNOSTIC: " + str(error) +
                "\nPREVIOUS FINAL RESPONSE (may be null):\n" + json.dumps(draft.get("structured_output"), ensure_ascii=False) +
                "\nPrevious candidate files, if any, are copied into this call's candidate directory. "
                "Read and correct them there; finish the complete collection and return only its filenames and decision.")

    def call(self, prompt, output_schema, phase):
        if self.result["calls_used"] >= self.args.max_calls:
            raise Failure("write_failed", "The finite writing call allowance was exhausted")
        if len(prompt.encode()) > MAX_PROMPT_BYTES:
            raise Failure("needs_input", "Request exceeds 16 MiB; no input was truncated")
        self.result["calls_used"] += 1
        self.save()  # Count selection, malformed responses, timeouts, and failures before launch.
        drafts = self.out / "drafts"
        drafts.mkdir(exist_ok=True)
        draft_path = drafts / f"{self.result['calls_used']:02d}.json"
        seed = {"phase": phase, "state": "starting", "pid": None, "inline_reads": self.inline_reads}
        candidate = None
        self.last_candidate_written = False
        if phase != "select":
            candidate = drafts / f"{self.result['calls_used']:02d}"
            candidate.mkdir()
            seed["candidate_directory"] = str(candidate.relative_to(self.out))
            if phase in {"repair", "repair_write"} and self.result["calls_used"] > 1:
                previous = drafts / f"{self.result['calls_used'] - 1:02d}"
                if previous.is_dir():
                    for item in candidate_files(previous):
                        (candidate / item["path"]).write_text(item["content"], encoding="utf-8")
            prompt += (f"\nCANDIDATE DIRECTORY (the only writable location): {candidate}\n"
                       "Write only direct .yaml/.yml files and dispatch.md here; no subdirectories. "
                       "Choose meaningful node filenames. Write/Edit each file as decisions become concrete; do not hold the "
                       "whole collection for a final JSON message. Partial work is preserved but is not accepted. "
                       "Final files[] lists only the delivered filenames, never YAML content. "
                       "Files omitted from files[] are retained as drafts and are not delivered.\n")
        atomic_json(draft_path, seed)
        print(f"{self.mode}: {phase} call {self.result['calls_used']}/{self.args.max_calls}", file=sys.stderr)
        before = snapshot_hashes(self.inputs.root)
        candidate_before = snapshot_hashes(candidate) if candidate else {}
        draft = native_call(self.args, prompt, output_schema, self.out, draft_path, seed)
        if snapshot_hashes(self.inputs.root) != before:
            raise Failure("input_changed", "Writer changed fixed input bytes")
        for record in draft.get("writes", []):
            written = record.get("input", {}).get("file_path")
            if record.get("status") != "completed":
                continue
            written = (self.out / written).resolve() if isinstance(written, str) else None
            if candidate is None or written is None or written.parent != candidate or not candidate_name(written.name):
                raise Failure("write_failed", "Native writer modified a file outside its candidate scope", repairable=False)
        files = candidate_files(candidate) if candidate else []
        self.last_candidate_written = bool(files)
        draft["candidate_files"] = [{"path": item["path"], "bytes": len(item["content"].encode()),
                                     "sha256": sha(item["content"].encode())} for item in files]
        atomic_json(draft_path, draft)
        if draft.get("diagnostics"):
            repairable = bool(draft.get("structured_output") or
                              (candidate and snapshot_hashes(candidate) != candidate_before))
            detail = "; ".join(draft["diagnostics"])
            if not repairable:
                detail += "; no new candidate content or final decision was produced, so an identical automatic repair is not useful"
            raise Failure("write_failed", detail, repairable=repairable)
        from jsonschema import Draft202012Validator
        errors = list(Draft202012Validator(output_schema).iter_errors(draft.get("structured_output")))
        if errors:
            # Do not put an enormous/possibly sensitive rejected instance in diagnostics.
            raise Failure("write_failed", "Malformed response at " + ", ".join("/".join(map(str, error.path)) or "root"
                                                                                 for error in errors[:8]), repairable=bool(files or draft.get("structured_output")))
        self.last_reads = draft.get("reads", [])
        if phase != "select":
            read_paths = {str((self.out / record["input"].get("file_path", "")).resolve())
                          for record in draft.get("reads", [])
                          if record.get("tool") == "Read" and record.get("status") == "completed"}
            docs = {str(self.inputs.root / "schema_docs" / name) for name in DOC_NAMES[1:]}
            if not read_paths & docs:
                raise Failure("write_failed", "Writer must actually Read the relevant DOC1–6 material before deciding")
        answer = deepcopy(draft["structured_output"])
        if candidate is not None:
            delivered = answer["files"]
            available = {item["path"]: item for item in files}
            if any(name not in available for name in delivered):
                raise Failure("write_failed", "Final response names an absent or unauthorized candidate file")
            answer["files"] = [available[name] for name in delivered]
        return answer


def parser(mode):
    result = argparse.ArgumentParser(description=f"Standalone Schema {mode}; scoped candidate writing, host-validated outputs")
    result.add_argument("--task", required=True, type=Path)
    result.add_argument("--out", required=True, type=Path)
    result.add_argument("--model", default=DEFAULT_MODEL)
    result.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"), default="max")
    result.add_argument("--max-calls", type=int, default=3 if mode == "design" else 2)
    result.add_argument("--timeout-s", type=float, default=900)
    if mode == "design":
        result.add_argument("--reference", action="append", default=[], type=Path)
    else:
        result.add_argument("--schema", required=True, type=Path)
        result.add_argument("--kernel", type=Path)
        result.add_argument("--feedback", action="append", required=True, type=Path)
    return result


def cli(mode, execute, argv=None):
    arguments = parser(mode)
    args = arguments.parse_args(argv)
    if args.max_calls <= 0 or args.timeout_s <= 0 or not args.timeout_s < float("inf"):
        arguments.error("--max-calls and --timeout-s must be positive finite values")
    if not args.model.strip():
        arguments.error("--model cannot be empty")
    try:
        for key in ("task", "schema", "kernel"):
            if hasattr(args, key) and getattr(args, key) is not None:
                setattr(args, key, input_path(getattr(args, key)))
        if not args.task.is_file():
            raise Failure("needs_input", "--task must be a file")
        for key in ("feedback", "reference"):
            if hasattr(args, key):
                setattr(args, key, [input_path(path) for path in getattr(args, key)])
        args.out = no_links(args.out.expanduser())
        if args.out.exists():
            raise Failure("needs_input", "--out must be a new directory")
        sources = [args.task, *getattr(args, "feedback", []), *getattr(args, "reference", [])]
        for key in ("schema", "kernel"):
            if getattr(args, key, None):
                item = getattr(args, key)
                sources.append(item.parent if key == "schema" and item.is_file() else item)
        if any(args.out == path or args.out in path.parents or (path.is_dir() and path in args.out.parents)
               for path in sources):
            raise Failure("needs_input", "--out cannot contain or overlap an input")
        protected = [CHECKOUT / name for name in PROTECTED]
        if os.environ.get("CLAUDE_CONFIG_DIR"):
            protected.append(Path(os.environ["CLAUDE_CONFIG_DIR"]).expanduser().resolve())
        protected += [Path.home() / name for name in (".claude", ".codex", ".ssh", ".aws")]
        if any(args.out == path or path in args.out.parents or args.out in path.parents for path in protected):
            raise Failure("needs_input", "--out overlaps a protected source directory")
        args.out.mkdir(parents=True)
    except (Failure, OSError) as exc:
        arguments.error(str(exc))
    run = Run(args, mode)
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        code = execute(run)
    except (KeyboardInterrupt, Interrupted) as exc:
        code = run.finish("interrupted", "interrupted", str(exc) or "Writing interrupted")
    except Failure as exc:
        code = run.finish("incomplete", exc.reason, str(exc))
    except Exception as exc:
        # File/schema diagnostics are useful; no environment or stream dumps are attached.
        code = run.finish("incomplete", "write_failed", f"{type(exc).__name__}: {exc}")
    finally:
        signal.signal(signal.SIGTERM, previous)
    print(run.out / "result.json")
    return code


SELECT_SCHEMA = {"type": "object", "additionalProperties": False,
                 "required": ["references", "notes"], "properties": {
                     "references": {"type": "array", "maxItems": 3, "items": {
                         "type": "object", "additionalProperties": False, "required": ["id", "purpose"],
                         "properties": {"id": {"type": "string"}, "purpose": {"type": "string"}}}},
                     "notes": {"type": "string"}}}


def discover(run):
    candidates = []
    catalog_path = CHECKOUT / "kernel_wiki" / "catalog.json"
    catalog = {}
    if catalog_path.is_file():
        fixed = run.inputs.file(catalog_path, run.inputs.root / "catalog.json")
        catalog = strict_json(fixed.read_bytes())
    operators = {str(Path(item.get("paths", {}).get("readme", "")).parent): item
                 for item in catalog.get("operators", [])}
    wiki = CHECKOUT / "schema_wiki"
    directories = set()
    if wiki.is_dir():
        for directory, dirs, files in os.walk(wiki):
            dirs[:] = [name for name in dirs if not name.startswith(".")]
            if ("root.yaml" in files or "root.yml" in files) and "dispatch.md" in files:
                directories.add(Path(directory))
    for supplied in run.args.reference:
        directory = supplied if supplied.is_dir() else supplied.parent
        if ((directory / "root.yaml").is_file() or (directory / "root.yml").is_file()) and (directory / "dispatch.md").is_file():
            directories.add(directory)
        else:
            try:
                before = set(run.inputs.records)
                run.inputs.closure(supplied)
                added = [item["snapshot"] for source, item in run.inputs.records.items()
                         if source not in before or source == str(supplied)]
                run.inputs.bindings.setdefault("explicit_reference_files", []).extend(added)
            except Failure as exc:
                if exc.reason == "input_changed":
                    raise
                run.notes.append(f"Reference omitted: {supplied}: {exc}")
    for directory in sorted(directories):
        try:
            relative = str(directory.relative_to(wiki)) if directory.is_relative_to(wiki) else None
            item = operators.get(relative, {})
            index = run.inputs.file(directory / "dispatch.md")
            readme = CHECKOUT / "kernel_wiki" / item.get("paths", {}).get("readme", "")
            description = run.inputs.file(readme) if item and readme.is_file() else index
            candidates.append({"id": str(directory), "title": item.get("title", directory.name),
                               "summary": item.get("source_description", item.get("summary", "")),
                               "dispatch": str(index), "description": str(description)})
        except Failure as exc:
            if exc.reason == "input_changed":
                raise
            run.notes.append(f"Reference omitted: {directory}: {exc}")
    run.inputs.bindings["discovered_references"] = candidates
    run.inputs.save()
    return candidates


def execute(run):
    prompt = run.preparation()
    candidates = discover(run)
    selection = []
    if candidates and run.args.max_calls > 1:
        try:
            answer = run.call(prompt + "\nSELECTION ONLY. Read nearby descriptions/dispatches and choose at most three "
                              "complementary references. An empty selection is valid when none fits. "
                              "Consider mathematics, scheduling/pipeline, layout/precision and device conditions. "
                              "Return only reference ids and purposes, no Schema files.\n" +
                              json.dumps(candidates, ensure_ascii=False), SELECT_SCHEMA, "select")
            known = {candidate["id"] for candidate in candidates}
            if any(item["id"] not in known for item in answer["references"]):
                raise Failure("write_failed", "Selection named an undiscovered Schema reference")
            if len({item["id"] for item in answer["references"]}) != len(answer["references"]):
                raise Failure("write_failed", "Selection repeated a reference")
            read_paths = {str((run.out / record["input"].get("file_path", "")).resolve())
                          for record in run.last_reads if record.get("tool") == "Read" and record.get("status") == "completed"}
            for selected in answer["references"]:
                candidate = next(item for item in candidates if item["id"] == selected["id"])
                descriptions = {candidate.get("description"), candidate.get("dispatch")} - {None}
                if not read_paths & descriptions:
                    raise Failure("write_failed", "Selected reference description/dispatch was not actually read")
            selection = answer["references"]
            run.notes.append(answer["notes"])
        except Failure as exc:
            run.notes.append(f"Selection was unusable; continue from the explicit Task: {exc}")
    elif candidates:
        run.notes.append("Only one writing call is available; design directly from Task and supplied material")
    for index, item in enumerate(selection, 1):
        try:
            directory, resolved = run.inputs.schema(Path(item["id"]), f"reference_{index}")
            paths = sorted(directory.iterdir())
            run.inline_reads.extend(map(str, paths))
            prompt += "\nSELECTED REFERENCE (borrow only conditions that hold for this Task): " + item["purpose"]
            prompt += run.inputs.text(paths)
            prompt += json.dumps(run.inputs.bindings[f"reference_{index}"], ensure_ascii=False)
            candidate = next(candidate for candidate in candidates if candidate["id"] == item["id"])
            if candidate.get("description"):
                description = Path(candidate["description"])
                prompt += run.inputs.text([description])
                run.inline_reads.append(str(description))
        except Failure as exc:
            if exc.reason == "input_changed":
                raise
            run.notes.append(f"Selected reference omitted: {item['id']}: {exc}")
        except Exception as exc:
            run.notes.append(f"Selected reference cannot be resolved and was omitted: {item['id']}: {exc}")
    prompt += "\nAll additional fixed materials and original locator contexts:\n" + str(run.inputs.root / "manifest.json")
    for relative in dict.fromkeys(run.inputs.bindings.get("explicit_reference_files", [])):
        path = run.out / relative
        try:
            prompt += run.inputs.text([path])
            run.inline_reads.append(str(path))
        except Failure as exc:
            run.notes.append(f"Text reference omitted: {path}: {exc}")
    prompt += "\nWRITE: write the complete local YAML collection and dispatch.md into the candidate directory; return filenames. No applicable neighbor is required."
    diagnostic = ""
    while run.result["calls_used"] < run.args.max_calls:
        try:
            answer = run.call(prompt + diagnostic, WRITE_SCHEMA, "write" if not diagnostic else "repair")
            if answer["missing"]:
                return run.finish("incomplete", "needs_input", answer["notes"] + "\nMissing:\n" + "\n".join(answer["missing"]))
            directory, _, notices = accept_files(answer["files"], run.out, source_context=run.args.task.parent)
            run.notes.extend([answer["notes"], *notices])
            return run.finish("complete", "done", schema=str(directory.relative_to(run.out)))
        except Interrupted:
            raise
        except Exception as exc:
            if isinstance(exc, Failure) and (exc.reason in {"needs_input", "input_changed"} or not exc.repairable):
                raise
            diagnostic = run.repair_prompt(exc)
            run.notes.append(str(exc))
    return run.finish("incomplete", "write_failed", "No acceptable complete Schema was received within the call allowance")


def main(argv=None):
    return cli("design", execute, argv)
