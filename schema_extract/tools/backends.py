"""The two supported CLIs: explicit commands and their unmodified event protocols."""

from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import tomllib
import stat
import shlex
from uuid import UUID

from .common import ExtractError, digest


DEFAULT_BACKEND = "claude"
DEFAULT_MODEL = "qwen3.8-max[1m]"
DEFAULT_EFFORT = "max"
TOKEN_FIELDS = ("input_tokens", "cached_input_tokens", "cache_creation_input_tokens", "output_tokens")
CODEX_TOKEN_FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens")
NETWORK_ENV = {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy"}
CLAUDE_ENV = {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "ANTHROPIC_CUSTOM_HEADERS"}


def exact_uuid(value: str) -> str:
    try:
        if str(UUID(value)) == value:
            return value
    except (ValueError, TypeError, AttributeError):
        pass
    raise ExtractError("session identity must be an exact UUID")


def strict_json(value):
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = item
        return result

    def invalid(_):
        raise ValueError("non-finite JSON value")

    return json.loads(value, object_pairs_hook=pairs, parse_constant=invalid)


def _number(value, *, integer=True):
    valid = type(value) is int if integer else type(value) in {int, float}
    return value if valid and (type(value) is int or math.isfinite(value)) and value >= 0 else None


def _tokens(raw, names):
    return {key: _number(raw.get(name)) if isinstance(raw, dict) else None for key, name in names.items()}


def _model_delta(raw):
    names = dict(zip(TOKEN_FIELDS, ("inputTokens", "cacheReadInputTokens", "cacheCreationInputTokens", "outputTokens")))
    if not isinstance(raw, dict) or any(not isinstance(item, dict) for item in raw.values()):
        return dict.fromkeys(TOKEN_FIELDS), None
    counters = [_tokens(item, names) for item in raw.values()]
    costs = [_number(item.get("costUSD"), integer=False) for item in raw.values()]
    return ({key: sum(item[key] for item in counters) if all(item[key] is not None for item in counters) else None
             for key in TOKEN_FIELDS}, sum(costs) if all(item is not None for item in costs) else None)


def _base_environment(home: Path):
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
            "HOME": str(home), "XDG_CACHE_HOME": str(home / ".cache"), "XDG_CONFIG_HOME": str(home / ".config"),
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
            **{key: value for key, value in os.environ.items() if key in NETWORK_ENV}}


@contextmanager
def _private_config(home, name):
    parent = os.open(home, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    descriptor = None
    try:
        try:
            os.mkdir(name, 0o700, dir_fd=parent)
        except FileExistsError:
            pass
        descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        yield descriptor
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def _private_write(directory, name, data, *, exclusive=False):
    flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | (os.O_EXCL if exclusive else 0)
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory)
    except FileExistsError:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ExtractError("private CLI config must be a regular file with one hard link")
        finally:
            os.close(descriptor)
        return
    with os.fdopen(descriptor, "wb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ExtractError("private CLI config must be a regular file with one hard link")
        stream.truncate()
        stream.write(data)


def _copy_auth(source: Path, directory: int, name: str):
    if source.exists():
        if source.is_symlink() or not source.is_file():
            raise ExtractError("CLI authentication source is not a regular file")
        _private_write(directory, name, source.read_bytes(), exclusive=True)


def _toml(value: dict, prefix=()) -> str:
    lines = []
    if prefix:
        lines.append("[" + ".".join(json.dumps(key) for key in prefix) + "]")
    for key, item in value.items():
        if not isinstance(item, dict):
            if item is None:
                raise ExtractError("unsupported null in provider configuration")
            lines.append(json.dumps(key) + " = " + json.dumps(item, ensure_ascii=False, allow_nan=False))
    for key, item in value.items():
        if isinstance(item, dict):
            lines.append(_toml(item, (*prefix, key)))
    return "\n".join(lines) + "\n"


class ClaudeBackend:
    name = "claude"

    def environment(self, home: Path, effort: str) -> tuple[dict, str]:
        source = Path(os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude"))).expanduser()
        settings = source / "settings.json"
        try:
            configured = strict_json(settings.read_bytes()).get("env", {}) if settings.exists() else {}
        except (OSError, ValueError, AttributeError) as exc:
            raise ExtractError("cannot read Claude authentication configuration") from exc
        if not isinstance(configured, dict):
            raise ExtractError("Claude settings env must be an object")
        environment = _base_environment(home)
        for key in CLAUDE_ENV | NETWORK_ENV:
            value = os.environ.get(key, configured.get(key))
            if value is not None:
                if not isinstance(value, str):
                    raise ExtractError("CLI connection environment must contain strings")
                environment[key] = value
        with _private_config(home, ".claude") as config:
            _copy_auth(source / ".credentials.json", config, ".credentials.json")
        environment.update(CLAUDE_CONFIG_DIR=str(home / ".claude"), CLAUDE_CODE_EFFORT_LEVEL=effort,
                           CLAUDE_CODE_SAFE_MODE="1", CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
                           DISABLE_AUTOUPDATER="1")
        # Model aliases, fixed thinking budgets, hooks and arbitrary inherited env never enter this environment.
        return environment, digest({key: environment[key] for key in CLAUDE_ENV | NETWORK_ENV if key in environment})

    def command(self, executable: str, *, model: str, effort: str, role: str,
                session_id: str | None, schema: dict, cwd: Path, workspace: Path, schema_path: Path,
                output_path: Path, writable_files=(), read_directories=(), check: Path | None = None) -> list[str]:
        if effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ExtractError("unsupported Claude effort")
        tools = ["Read", "Glob", "Grep"] + (["Edit", "Write"] if role == "worker" else [])
        allowed = ["Read", "Glob", "Grep"]
        if role == "worker":
            allowed += [f"{tool}(/{workspace / name})" for tool in ("Edit", "Write") for name in writable_files]
        if check:
            tools.append("Bash")
            allowed.append("Bash(" + shlex.quote(str(check)) + ")")
        command = [executable, "-p", "--model", model, "--effort", effort,
                   "--output-format", "stream-json", "--verbose", "--json-schema",
                   json.dumps(schema, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
                   "--safe-mode", "--restricted", "--disable-slash-commands", "--no-chrome",
                   "--setting-sources", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                   "--settings", '{"disableAllHooks":true}', "--permission-mode", "dontAsk",
                   "--permission-prompts", "none", "--tools", ",".join(tools),
                   "--allowedTools", ",".join(allowed), "--disallowedTools", "Agent,Task,Skill,WebFetch,WebSearch"]
        for directory in dict.fromkeys([str(workspace), *read_directories]):
            command += ["--add-dir", directory]
        if session_id is not None:
            command += ["--resume", exact_uuid(session_id)]
        return command


class CodexBackend:
    name = "codex"

    def environment(self, home: Path, effort: str) -> tuple[dict, str]:
        source = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).expanduser()
        path = source / "config.toml"
        try:
            configured = tomllib.loads(path.read_text()) if path.exists() else {}
        except (OSError, ValueError) as exc:
            raise ExtractError("cannot read Codex provider configuration") from exc
        preserved = {key: deepcopy(configured[key]) for key in
                     ("model_provider", "model_context_window", "model_auto_compact_token_limit",
                      "model_supports_reasoning_summaries") if key in configured}
        provider = configured.get("model_provider", "openai")
        provider_config = configured.get("model_providers", {}).get(provider)
        if provider_config is not None:
            preserved["model_providers"] = {provider: deepcopy(provider_config)}
        environment = _base_environment(home)
        names = {"OPENAI_API_KEY", "OPENAI_BASE_URL"}
        if isinstance(provider_config, dict):
            if provider_config.get("env_key"):
                names.add(provider_config["env_key"])
            names.update((provider_config.get("env_http_headers") or {}).values())
        environment.update({key: os.environ[key] for key in names if key in os.environ})
        environment["CODEX_HOME"] = str(home / ".codex")
        controlled = {**preserved, "approval_policy": "never", "web_search": "disabled", "mcp_servers": {}, "project_doc_max_bytes": 0,
                      "features": {key: False for key in ("hooks", "plugins", "plugin_discovery_prefetch", "remote_plugin",
                                   "multi_agent", "multi_agent_v2", "memories", "skill_search", "shell_snapshot")}}
        with _private_config(home, ".codex") as config:
            _copy_auth(source / "auth.json", config, "auth.json")
            _private_write(config, "config.toml", _toml(controlled).encode())
        return environment, digest({"provider": preserved, "connection_environment":
                                    {key: environment[key] for key in names | NETWORK_ENV if key in environment}})

    def command(self, executable: str, *, model: str, effort: str, role: str,
                session_id: str | None, schema: dict, cwd: Path, workspace: Path, schema_path: Path,
                output_path: Path, writable_files=(), read_directories=(), check: Path | None = None) -> list[str]:
        if effort not in {"none", "minimal", "low", "medium", "high", "xhigh"}:
            raise ExtractError("unsupported Codex effort; select an effort supported by this provider")
        command = [executable, "--no-daemon", "-a", "never", "-s", "workspace-write" if role == "worker" else "read-only", "-C", str(cwd),
                   "-c", "model_reasoning_effort=" + json.dumps(effort)]
        if role == "worker":
            command += ["--add-dir", str(workspace)]
        command += ["exec"]
        if session_id is not None:
            command += ["resume", exact_uuid(session_id)]
        return command + ["--skip-git-repo-check", "--json", "--model", model,
                          "--ignore-rules", "--output-schema", str(schema_path),
                          "--output-last-message", str(output_path), "-"]


def backend_for(name: str):
    if name == "claude":
        return ClaudeBackend()
    if name == "codex":
        return CodexBackend()
    raise ExtractError("backend must be explicitly registered as claude or codex")


class NativeEvents:
    def __init__(self, backend: str):
        backend_for(backend)
        self.backend = backend
        self.terminals, self.reports, self.errors, self.models, self.efforts = [], [], [], [], []
        self.init_count, self.last_type, self.response, self.cli_model = 0, None, None, None
        self.last_message = None

    def observe(self, event: dict, anchor: dict, trace: dict):
        kind = event["type"]
        self.last_type = kind
        session = event.get("session_id") if self.backend == "claude" else event.get("thread_id")
        if session is not None:
            session = exact_uuid(session)
            if session not in trace["session_ids"]:
                trace["session_ids"].append(session)
        if self.backend == "claude":
            self._claude(event, anchor)
        else:
            self._codex(event, anchor)

    def _claude(self, event, anchor):
        kind = event["type"]
        if kind == "system" and event.get("subtype") == "init":
            self.init_count += 1
            if not event.get("session_id"):
                self.errors.append("initial_session_missing")
            self.cli_model = event.get("model")
            if isinstance(event.get("effort"), str):
                self.efforts.append(event["effort"])
        if kind == "assistant":
            message = event.get("message", {})
            if isinstance(message.get("model"), str):
                self.models.append(message["model"])
            if event.get("error"):
                self.errors.append("assistant_api_error")
            for item in message.get("content", []):
                if isinstance(item, dict) and item.get("type") == "tool_use" and item.get("name") in {"Agent", "Task"}:
                    self.errors.append("forbidden_delegation")
        if kind == "error":
            self.errors.append("native_error_event")
        if kind == "result":
            self.terminals.append({**anchor, "type": kind, "subtype": event.get("subtype"),
                                   "is_error": event.get("is_error"), "session_id": event.get("session_id"),
                                   "result_index": event.get("result_index")})
            if (event.get("subtype") != "success" or event.get("is_error") is not False or event.get("errors") or
                    not event.get("session_id") or event.get("interrupted_by_shutdown") or
                    event.get("terminal_reason") not in {None, "completed"} or event.get("deferred_tool_use") or
                    event.get("queued_turn_count", 0) != 0):
                self.errors.append("unsuccessful_native_result")
            self.response = event.get("structured_output")
            raw = {key: deepcopy(event[key]) for key in ("usage", "modelUsage", "model_usage_since_last_result",
                   "total_cost_usd", "duration_ms", "duration_api_ms", "result_index") if key in event}
            tokens = _tokens(event.get("usage"), dict(zip(TOKEN_FIELDS,
                             ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens"))))
            self.reports.append({**anchor, "raw": raw, "tokens": tokens})

    def _codex(self, event, anchor):
        kind = event["type"]
        if kind == "thread.started":
            self.init_count += 1
            if not event.get("thread_id"):
                self.errors.append("initial_session_missing")
        if kind in {"error", "turn.failed"}:
            self.errors.append("native_error_event")
        if kind == "item.completed" and isinstance(event.get("item"), dict):
            item = event["item"]
            if item.get("type") == "agent_message":
                self.last_message = item.get("text")
            if item.get("type") == "collab_tool_call":
                self.errors.append("forbidden_delegation")
        if isinstance(event.get("model"), str):
            self.models.append(event["model"])
        if isinstance(event.get("reasoning_effort"), str):
            self.efforts.append(event["reasoning_effort"])
        if kind == "turn.completed":
            self.terminals.append({**anchor, "type": kind, "subtype": None})
            raw = event.get("usage")
            self.reports.append({**anchor, "raw": raw,
                                 "tokens": _tokens(raw, {field: field for field in CODEX_TOKEN_FIELDS})})

    def finish(self, trace):
        issues = []
        if trace["error"] or not trace["stable"]:
            issues.append("missing_or_changing_trace")
        if trace["truncated_line"]:
            issues.append("unterminated_final_line")
        if trace["parse_error_count"]:
            issues.append("malformed_events")
        native_issues = list(issues) + self.errors
        expected = "result" if self.backend == "claude" else "turn.completed"
        if len(self.terminals) != 1 or self.last_type != expected:
            native_issues.append("missing_or_nonfinal_unique_terminal")
        if self.init_count != 1 or len(trace["session_ids"]) != 1:
            native_issues.append("session_identity_conflict")
        if self.backend == "codex" and self.last_message is not None:
            try:
                self.response = strict_json(self.last_message)
            except (TypeError, ValueError):
                self.response = None
        usage = {"format": "schema_extract.usage.v2", "backend": self.backend, "reports": self.reports,
                 "issues": issues, "additive_tokens": dict.fromkeys(TOKEN_FIELDS), "reported_cost_usd": None,
                 "cost_source": None, "billing_cost_usd": None, "cost_is_billing": False,
                 "cached_input_is_input_subset": self.backend == "codex"}
        if self.backend == "codex":
            usage.update(source="events.jsonl:turn.completed.usage", counter_scope="provider_counter_semantics_unverified")
            for report in self.reports:
                tokens = report["tokens"]
                if any(value is None for value in tokens.values()):
                    issues.append("missing_or_invalid_token_fields")
                if (tokens["input_tokens"] is not None and tokens["cached_input_tokens"] is not None and
                        tokens["cached_input_tokens"] > tokens["input_tokens"]):
                    issues.append("cached_input_exceeds_input")
        else:
            usage.update(source="events.jsonl:result.usage", counter_scope="main_loop_invocation",
                         cumulative_source="events.jsonl:result.modelUsage", session_reported_cost_usd=None)
            if len(self.reports) == 1:
                report = self.reports[0]
                raw = report["raw"]
                usage["additive_tokens"] = deepcopy(report["tokens"])
                usage["session_reported_cost_usd"] = _number(raw.get("total_cost_usd"), integer=False)
                if "model_usage_since_last_result" in raw and raw.get("result_index") == 0:
                    counters, cost = _model_delta(raw["model_usage_since_last_result"])
                    usage.update(additive_tokens=counters, reported_cost_usd=cost,
                                 cost_source="result.model_usage_since_last_result.*.costUSD",
                                 source="events.jsonl:result.model_usage_since_last_result",
                                 counter_scope="query_pipeline_invocation_delta")
                else:
                    issues.append("query_pipeline_delta_unavailable")
                if any(value is None for value in usage["additive_tokens"].values()):
                    issues.append("missing_or_invalid_token_fields")
            else:
                issues.append("unique_invocation_usage_unavailable")
        if not self.reports or any(item["raw"] is None for item in self.reports):
            issues.append("missing_usage")
        if len(self.terminals) != 1 or trace["parse_error_count"] or trace["truncated_line"] or not trace["stable"]:
            usage["additive_tokens"] = dict.fromkeys(TOKEN_FIELDS)
            usage["reported_cost_usd"] = None
        if self.backend == "claude" and native_issues and usage["counter_scope"] == "main_loop_invocation":
            usage["additive_tokens"] = dict.fromkeys(TOKEN_FIELDS)
            issues.append("unsuccessful_main_loop_counter_unverified")
        usage["issues"] = sorted(set(issues))
        usage["status"] = "partial" if self.reports and issues else "reported" if self.reports else "unknown"
        models, efforts = sorted(set(self.models)), sorted(set(self.efforts))
        return {"native_status": {"success": not native_issues, "terminals": self.terminals,
                                   "issues": sorted(set(native_issues)), "init_count": self.init_count},
                "session_id": trace["session_ids"][0] if len(trace["session_ids"]) == 1 else None,
                "response": self.response, "reported_model": models[0] if len(models) == 1 else None,
                "reported_models": models, "reported_effort": efforts[0] if len(efforts) == 1 else None,
                "cli_reported_model": self.cli_model, "usage": usage}
