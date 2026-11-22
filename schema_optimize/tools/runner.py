"""Local file handling and bounded, read-only Claude writing. No public API."""
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
    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


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
                                     "sha256": digest, "bytes": len(data)}
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
        for path in dict.fromkeys(paths):
            path = Path(path)
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeError as exc:
                raise Failure("needs_input", f"Required text material is not UTF-8: {path}") from exc
            sections.append(f"\n--- COMPLETE FILE {path} ---\n{content}\n--- END FILE ---\n")
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


def command(args, output_schema, read_dirs):
    result = ["claude", "-p", "--model", args.model, "--effort", args.effort,
              "--output-format", "stream-json", "--verbose", "--json-schema", json.dumps(output_schema),
              "--safe-mode", "--restricted", "--disable-slash-commands", "--no-chrome", "--setting-sources", "",
              "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}', "--settings", '{"disableAllHooks":true}',
              "--permission-mode", "dontAsk", "--permission-prompts", "none", "--tools", "Read,Glob,Grep",
              "--allowedTools", "Read,Glob,Grep", "--disallowedTools", "Bash,Edit,Write,Agent,Task,Skill,WebFetch,WebSearch",
              "--no-session-persistence"]
    for directory in sorted(set(map(str, read_dirs))):
        result.extend(("--add-dir", directory))
    return result


def parse_stream(stdout, stderr, returncode):
    draft = {"final_answer": None, "structured_output": None, "reads": [], "session_ids": [],
             "returncode": returncode, "diagnostics": [], "stderr": stderr}
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
            chunks = event.get("message", {}).get("content", [])
            final_text = [item["text"] for item in chunks if item.get("type") == "text"]
            for item in chunks:
                if item.get("type") != "tool_use":
                    continue
                name = item.get("name")
                if name not in {"Read", "Glob", "Grep", "StructuredOutput"}:
                    draft["diagnostics"].append(f"Unexpected tool: {name}")
                if name in {"Read", "Glob", "Grep"}:
                    record = {"tool": name, "input": item.get("input", {}), "status": "requested"}
                    draft["reads"].append(record)
                    tools[item.get("id")] = record
        if event.get("type") == "user":
            for item in event.get("message", {}).get("content", []):
                if item.get("type") == "tool_result" and item.get("tool_use_id") in tools:
                    tools[item["tool_use_id"]]["status"] = "error" if item.get("is_error") else "completed"
        if event.get("type") == "error":
            draft["diagnostics"].append("Claude reported an error event")
        if event.get("type") == "result":
            if terminal is not None:
                draft["diagnostics"].append("Multiple terminal results")
            terminal = event
    draft["final_answer"] = terminal.get("result") if terminal and "result" in terminal else "\n".join(final_text)
    if terminal:
        draft["structured_output"] = terminal.get("structured_output")
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
    try:
        process = subprocess.Popen(command(args, output_schema, [out / "input"]), cwd=out,
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
    atomic_json(draft_path, seed)
    return seed


FILE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["path", "content"],
               "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}
WRITE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["files", "notes", "missing"],
                "properties": {"files": {"type": "array", "items": FILE_SCHEMA}, "notes": {"type": "string"},
                               "missing": {"type": "array", "items": {"type": "string"}}}}


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
        docs = [self.inputs.file(CHECKOUT / "schema_docs" / name, self.inputs.root / "docs" / name)
                for name in DOC_NAMES]
        method = self.inputs.file(CHECKOUT / f"{self.mode.upper()}_SCHEMA.md", self.inputs.root / "method.md")
        template_files = [self.inputs.file(CHECKOUT / "schema_template" / name,
                                          self.inputs.root / "template" / name) for name in ("README.md", "starter.yaml")]
        prompt_file = CHECKOUT / f"schema_{self.mode}" / "prompt" / f"{self.mode}.md"
        prompt = self.inputs.file(prompt_file, self.inputs.root / "instructions.md")
        self.inline_reads = [str(path) for path in (docs[0], method, prompt, *task_materials)]
        text = self.inputs.text([prompt, docs[0], method, *task_materials])
        text += "\nREADABLE DOC1–6 and Template (use Read for the relevant sections before writing):\n"
        text += "\n".join(map(str, [*docs[1:], *template_files]))
        text += f"\nInput manifest (original paths, fixed bytes, bindings): {self.inputs.root / 'manifest.json'}\n"
        return text

    def repair_prompt(self, error):
        path = self.out / "drafts" / f"{self.result['calls_used']:02d}.json"
        draft = strict_json(path.read_bytes())
        draft.setdefault("host_diagnostics", []).append(str(error))
        atomic_json(path, draft)
        return ("\nHOST DIAGNOSTIC: " + str(error) +
                "\nPREVIOUS COMPLETE RESPONSE:\n" + json.dumps(draft.get("structured_output"), ensure_ascii=False) +
                "\nCorrect the specific issue and return the complete response again.")

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
        atomic_json(draft_path, seed)
        print(f"{self.mode}: {phase} call {self.result['calls_used']}/{self.args.max_calls}", file=sys.stderr)
        draft = native_call(self.args, prompt, output_schema, self.out, draft_path, seed)
        if draft.get("diagnostics"):
            raise Failure("write_failed", "; ".join(draft["diagnostics"]))
        from jsonschema import Draft202012Validator
        errors = list(Draft202012Validator(output_schema).iter_errors(draft.get("structured_output")))
        if errors:
            # Do not put an enormous/possibly sensitive rejected instance in diagnostics.
            raise Failure("write_failed", "Malformed response at " + ", ".join("/".join(map(str, error.path)) or "root"
                                                                                 for error in errors[:8]))
        self.last_reads = draft.get("reads", [])
        if phase != "select":
            read_paths = {str((self.out / record["input"].get("file_path", "")).resolve())
                          for record in draft.get("reads", [])
                          if record.get("tool") == "Read" and record.get("status") == "completed"}
            docs = {str(self.inputs.root / "docs" / name) for name in DOC_NAMES[1:]}
            if not read_paths & docs:
                raise Failure("write_failed", "Writer must actually Read the relevant DOC1–6 material before deciding")
        return draft["structured_output"]


def parser(mode):
    result = argparse.ArgumentParser(description=f"Standalone Schema {mode}; read-only model, host-written outputs")
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


OPTIMIZE_SCHEMA = deepcopy(WRITE_SCHEMA)
OPTIMIZE_SCHEMA["required"] += ["decision", "required_fix"]
OPTIMIZE_SCHEMA["properties"].update(
    decision={"enum": ["keep", "repair", "revise", None]}, required_fix={"type": ["boolean", "null"]})


def files_digest(files):
    return sha(json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode())


def tree_hashes(path):
    if path.is_file():
        return {path.name: sha(path.read_bytes())}
    return {str(item.relative_to(path)): sha(item.read_bytes()) for item in regular_files(path)}


def require_complete_kernel(path):
    if path is None or path.is_dir():
        return
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        raise Failure("needs_input", "An empty file is not a kernel; supply complete source or a real design_gap result")
    if path.suffix == ".py":
        import ast
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return  # Broken source is legitimate optimization evidence.
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    raise Failure("needs_input", "Kernel has relative imports; pass its complete source directory")
                modules = [node.module] if node.module else []
            elif isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            for module in modules:
                local = path.parent / module.split(".")[0]
                if local.with_suffix(".py").is_file() or local.is_dir():
                    raise Failure("needs_input", f"Kernel imports local {module}; pass its complete source directory")
    elif re.search(r'^\s*#\s*include\s*"', content, re.MULTILINE):
        raise Failure("needs_input", "Kernel has local includes; pass its complete source directory")


def feedback_evidence(run, schema_source, resolved, kernel_source):
    objects = []
    for path in run.args.feedback:
        run.inputs.closure(path)
    for record in list(run.inputs.records.values()):
        if Path(record["source"]).suffix != ".json":
            continue
        try:
            value = strict_json((run.out / record["snapshot"]).read_bytes())
        except ValueError:
            continue
        if isinstance(value, dict):
            objects.append((Path(record["source"]), value))
    schema_hashes = tree_hashes(schema_source)
    kernel_hashes = tree_hashes(kernel_source) if kernel_source else None
    for origin, value in objects:
        for key, expected in (("schema", schema_hashes), ("kernel", kernel_hashes)):
            binding = value.get(key)
            if not isinstance(binding, dict) or expected is None:
                continue
            stated = binding.get("files")
            if stated is not None and stated != expected:
                raise Failure("needs_input", f"Feedback {key} files do not match the supplied material: {origin}")
            if binding.get("sha256") is not None and binding["sha256"] != files_digest(expected):
                raise Failure("needs_input", f"Feedback {key} digest does not match the supplied material: {origin}")
            if key == "schema" and binding.get("effective_identity") is not None:
                actual = {name: resolved.effective_identity(name) for name in resolved.source_files}
                if binding["effective_identity"] != actual:
                    raise Failure("needs_input", f"Feedback Schema source identity does not match: {origin}")
    if kernel_source is None:
        valid = False
        by_path = {str(path): value for path, value in objects}
        for origin, value in objects:
            if value.get("stop_reason") != "design_gap" or value.get("kernel", "absent") is not None:
                continue
            link = value.get("feedback")
            if not isinstance(link, str):
                continue
            feedback_path = (origin.parent / link).resolve()
            diagnosis = by_path.get(str(feedback_path), {}).get("diagnosis", {})
            if diagnosis.get("kind") == "design_gap" and isinstance(diagnosis.get("detail"), str) and diagnosis["detail"].strip():
                valid = True
        if not valid:
            raise Failure("needs_input", "Without --kernel, feedback must include an Implement design_gap result with no kernel and its concrete diagnosis")


def substantively_changed(original, revised, original_dispatch, revised_dispatch):
    def content(resolved):
        result = {}
        for name in resolved.source_files:
            node = resolved.effective_document(name)
            meta = node["META_DATA"]
            result[name] = {"DESIGN": node["DESIGN"], "REALIZATION": node.get("REALIZATION"),
                            "Dispatch": {key: value for key, value in meta.get("Dispatch", {}).items() if key != "id"}}
        return result
    return not exact_value(content(original), content(revised))


def check_task_identity(original, revised):
    for name, node in revised.raw_task.items():
        old = original.raw_task.get(name, original.raw_task["root"])
        for field in ("Name", "Describe"):
            if not exact_value(old["META_DATA"].get(field), node["META_DATA"].get(field)):
                raise Failure("write_failed", f"Task identity changed at {name}.META_DATA.{field}")
        if name in original.raw_task:
            for field in set(old["META_DATA"]) | set(node["META_DATA"]):
                if field not in {"Version", "Dispatch"} and not exact_value(old["META_DATA"].get(field), node["META_DATA"].get(field)):
                    raise Failure("write_failed", f"Optimization changed fixed metadata at {name}.META_DATA.{field}")
            allowed = {"DESIGN", "REALIZATION", "META_DATA", "CONTRACT"}
            if any(not exact_value(old.get(key), node.get(key)) for key in set(old) | set(node) if key not in allowed):
                raise Failure("write_failed", f"Optimization changed fields outside Design/Realization at {name}")


def execute(run):
    prompt = run.preparation()
    source = run.args.schema if run.args.schema.is_dir() else run.args.schema.parent
    directory, original = run.inputs.schema(source, "schema")
    from schema_template.tools import schema as template
    notices = []
    try:
        template.validate_directory(directory, notices=notices)
    except Exception as exc:
        raise Failure("needs_input", f"The existing Schema is not a complete valid input: {exc}") from exc
    run.notes.extend(notices)
    require_complete_kernel(run.args.kernel)
    kernel = run.inputs.tree(run.args.kernel, "kernel") if run.args.kernel else None
    feedback_evidence(run, source, original, run.args.kernel)
    paths = [directory / file.name for file in original.source_files.values()] + [directory / "dispatch.md"]
    if kernel:
        paths += regular_files(kernel) if kernel.is_dir() else [kernel]
    excluded = {str(path) for path in paths} | set(run.inline_reads)
    excluded.update(str(run.inputs.root / "docs" / name) for name in DOC_NAMES)
    for record in run.inputs.records.values():
        path = run.out / record["snapshot"]
        if str(path) not in excluded and path.suffix not in {".png", ".jpg", ".jpeg", ".pdf", ".zip", ".gz", ".tar", ".whl"}:
            paths.append(path)
            excluded.add(str(path))
    run.inline_reads.extend(map(str, paths))
    prompt += run.inputs.text(paths)
    prompt += "\nFULL EFFECTIVE SCHEMA / FROM CHAINS / ALL SHARED USERS:\n" + json.dumps(run.inputs.bindings["schema"], ensure_ascii=False)
    prompt += "\nReturn one honest decision. Keep/repair have no files. Revise returns the complete Schema."
    if kernel is None:
        prompt += "\nThere is no kernel: this is a real Implement design_gap. Analyze that diagnosis; do not invent code or measurements."
    contract = original.raw_task["root"]["CONTRACT"]
    diagnostic = ""
    failed_revision = False
    while run.result["calls_used"] < run.args.max_calls:
        try:
            answer = run.call(prompt + diagnostic, OPTIMIZE_SCHEMA, "optimize" if not diagnostic else "repair_write")
            if answer["missing"]:
                return run.finish("incomplete", "needs_input", answer["notes"] + "\nMissing:\n" + "\n".join(answer["missing"]), decision=None)
            decision, required = answer["decision"], answer["required_fix"]
            if failed_revision and decision == "keep":
                raise Failure("write_failed", "A failed revision cannot be hidden by returning keep")
            if decision not in {"keep", "repair", "revise"} or type(required) is not bool or not answer["notes"].strip():
                raise Failure("write_failed", "A decision requires nonempty evidence notes and an explicit required_fix boolean")
            if decision == "keep" and required:
                raise Failure("write_failed", "keep must have required_fix=false")
            if decision == "repair" and not required:
                raise Failure("write_failed", "repair must have required_fix=true")
            if required and answer["notes"].strip():
                run.result["required_fix"] = True
            if decision in {"keep", "repair"}:
                if answer["files"]:
                    raise Failure("write_failed", "keep/repair must leave the input Schema unchanged and contain no files")
                return run.finish("complete", "done", answer["notes"], schema=str(directory.relative_to(run.out)), decision=decision, required_fix=required)
            failed_revision = True
            target, revised, notices = accept_files(answer["files"], run.out, contract, source_context=source)
            check_task_identity(original, revised)
            if not substantively_changed(original, revised, (directory / "dispatch.md").read_text(), (target / "dispatch.md").read_text()):
                raise Failure("write_failed", "revise has no substantive parsed Design/Realization/routing change")
            run.notes.extend(notices)
            return run.finish("complete", "done", answer["notes"], schema=str(target.relative_to(run.out)), decision="revise", required_fix=required)
        except Interrupted:
            raise
        except Exception as exc:
            if isinstance(exc, Failure) and exc.reason in {"needs_input", "input_changed"}:
                raise
            diagnostic = run.repair_prompt(exc) + "\nDo not relabel a failed revision as keep."
            run.notes.append(str(exc))
    return run.finish("incomplete", "write_failed", "No valid optimization decision was received within the call allowance", decision=None)


def main(argv=None):
    return cli("optimize", execute, argv)
