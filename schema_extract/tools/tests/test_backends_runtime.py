"""Direct native launch, role permissions, scoped sessions and owned descendants."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

from schema_extract.tools.agent import AgentCLI, read_result, retire_session, settle_session
from schema_extract.tools.backends import ClaudeBackend, CodexBackend
from schema_extract.tools.common import ExtractError, read_json, write_json
from schema_extract.tools.processes import process_alive, process_table


FAKE = r'''
import json, os, subprocess, sys, time
from pathlib import Path
from uuid import uuid4
if '--version' in sys.argv:
    print('extract-offline-cli 1')
    raise SystemExit()
p = json.load(sys.stdin)
is_claude = '--json-schema' in sys.argv
saved = Path.home() / 'session-fixture.json'
resumed = '--resume' in sys.argv if is_claude else 'resume' in sys.argv
if resumed:
    flag = '--resume' if is_claude else 'resume'
    previous = json.loads(saved.read_text())
    session = sys.argv[sys.argv.index(flag) + 1]
    assert session == previous['session_id'] and os.getcwd() == previous['cwd']
else:
    session = str(uuid4())
    previous = {'session_id': session, 'sticky': p['sticky'], 'cwd': os.getcwd()}
    saved.write_text(json.dumps(previous))
def emit(item):
    print(json.dumps(item), flush=True)
emit({'type':'system','subtype':'init','session_id':session,'model':'requested-alias'} if is_claude else
     {'type':'thread.started','thread_id':session})
if p.get('action') in ('daemon', 'child', 'orphan'):
    if os.fork() == 0:
        if p['action'] != 'child':
            os.setsid()
            if os.fork() != 0: os._exit(0)
        while True:
            (Path(os.environ['TMPDIR']) / 'heartbeat').write_text(str(time.time()))
            time.sleep(0.03)
    if p["action"] != "orphan":
        while True: time.sleep(1)
checks = {'source_read': Path(p['source']).read_text() == 'frozen'}
if p['role'] == 'worker':
    target = Path(p['workspace']) / 'root.yaml'
    temporary = target.with_suffix('.tmp')
    temporary.write_text('atomically changed')
    os.replace(temporary, target)
    checks['atomic_edit'] = target.read_text() == 'atomically changed'
if p.get('action') == 'cwd_write':
    Path('unexpected').write_text('bad')
helper = subprocess.run([p['check_command']],capture_output=True,text=True)
checks['helper'] = helper.returncode == 0 and helper.stdout.strip() == 'checked'
response = {'attempt_id':p['attempt_id'],'checks':checks,'sticky':previous['sticky'],'cwd':os.getcwd()}
if is_claude:
    emit({'type':'assistant','session_id':session,'message':{'model':'service-reported-model','content':[]}})
    delta = {'model':{'inputTokens':4,'cacheReadInputTokens':12,'cacheCreationInputTokens':8,'outputTokens':3,'costUSD':0.003}}
    emit({'type':'result','subtype':'success','is_error':False,'session_id':session,'structured_output':response,
          'usage':{'input_tokens':4,'cache_read_input_tokens':12,'cache_creation_input_tokens':8,'output_tokens':3},
          'modelUsage':{'model':{'inputTokens':100 if resumed else 4}},'total_cost_usd':2 if resumed else 0.003,
          'model_usage_since_last_result':delta,'result_index':0})
else:
    Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text(json.dumps(response))
    emit({'type':'item.completed','item':{'type':'agent_message','text':json.dumps(response)}})
    emit({'type':'turn.completed','usage':{'input_tokens':12,'cached_input_tokens':4,'output_tokens':3}})
'''


class BackendRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="extract-backends-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "frozen.txt").write_text("frozen")
        self.helper = self.source / "check.py"
        self.helper.write_text("#!" + sys.executable + "\nfrom pathlib import Path\nassert Path(" +
                               repr(str(self.source / "frozen.txt")) + ").read_text() == 'frozen'\nprint('checked')\n")
        self.helper.chmod(0o500)
        self.host = self.root / "runner-state.json"
        self.host.write_text("protected host")
        self.fake = self.root / "fake-cli"
        self.fake.write_text("#!" + sys.executable + "\n" + FAKE)
        self.fake.chmod(0o755)
        self.auth = self.root / "configuration"
        self.auth.mkdir()
        self.environment = patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": str(self.auth), "CODEX_HOME": str(self.auth)})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.run_id = str(uuid4())
        self.schema = {"type": "object", "additionalProperties": False,
                       "properties": {"attempt_id": {"type": "string"}, "checks": {"type": "object"},
                                      "sticky": {"type": "string"}, "cwd": {"type": "string"}},
                       "required": ["attempt_id", "checks", "sticky", "cwd"]}

    def attempt(self, *, backend="claude", role="worker", session_id=None, stage="P1", operation="normal"):
        identity = str(uuid4())
        directory = self.root / "attempts" / identity
        workspace = directory / "workspace"
        workspace.mkdir(parents=True)
        for name in ("root.yaml", "other.yaml"):
            (workspace / name).write_text("original")
        result = dict(backend=backend, model="qwen3.8-max[1m]" if backend == "claude" else "explicit-codex-model",
                      effort="max" if backend == "claude" else "xhigh", run_id=self.run_id, attempt_id=identity,
                      stage=stage, node="root", role=role, directory=str(directory), workspace=str(workspace),
                      session_directory=str(self.root / "sessions" / (backend + stage + role + operation)),
                      session_scope=dict(stage=stage, node="root", role=role, operation=operation), session_id=session_id,
                      writable_files=["root.yaml"] if role == "worker" else [],
                      read_directories=[str(self.source)], check_command=str(self.helper))
        write_json(directory / "attempt.json", result)
        return result

    def invoke(self, attempt, *, action="normal", cancel=None, on_process=lambda _: None, timeout=15):
        prompt = json.dumps({"attempt_id": attempt["attempt_id"], "host": str(self.host), "parent_pid": os.getpid(),
                             "sticky": attempt["attempt_id"], "action": action, "workspace": attempt["workspace"],
                             "role": attempt["role"], "source": str(self.source / "frozen.txt"), "check_command": attempt["check_command"]})
        return AgentCLI(backend=attempt["backend"], executable=str(self.fake), model=attempt["model"]).run(
            attempt=attempt, prompt=prompt, schema=self.schema, timeout=timeout, cancel_event=cancel, on_process=on_process)

    def test_real_cwd_atomic_workspace_and_exact_resume_for_both_backends_and_roles(self):
        for backend in ("claude", "codex"):
            for role in ("worker", "reviewer"):
                with self.subTest(backend=backend, role=role):
                    first = self.attempt(backend=backend, role=role)
                    result = self.invoke(first)
                    self.assertEqual(result["outcome"], "completed", result)
                    self.assertTrue(all(result["response"]["checks"].values()))
                    self.assertEqual(result["response"]["cwd"], str(Path(first["session_directory"]) / "cwd"))
                    self.assertEqual(result["cwd_before"], [])
                    self.assertEqual(result["cwd_after"], [])
                    original = (Path(first["workspace"]) / "root.yaml").read_bytes()
                    second = self.attempt(backend=backend, role=role, session_id=result["session_id"])
                    resumed = self.invoke(second)
                    self.assertEqual(resumed["outcome"], "completed", resumed)
                    self.assertEqual(resumed["session_id"], result["session_id"])
                    self.assertEqual(resumed["response"]["sticky"], first["attempt_id"])
                    self.assertNotEqual(first["workspace"], second["workspace"])
                    self.assertEqual((Path(first["workspace"]) / "root.yaml").read_bytes(), original)
                    if backend == "claude":
                        self.assertEqual(resumed["usage"]["reported_cost_usd"], 0.003)
                        self.assertEqual(resumed["usage"]["session_reported_cost_usd"], 2)
                    path = Path(second["directory"]) / "events.jsonl"
                    path.write_text(path.read_text().replace(result["session_id"], str(uuid4())))
                    for outcome in ("completed", "receipt_error"):
                        self.assertEqual(read_result(second, exit_code=0, termination={"confirmed": True}, outcome=outcome)["outcome"], "session_mismatch")

    def test_stable_cwd_rejects_untracked_mutable_material(self):
        attempt = self.attempt()
        result = self.invoke(attempt, action="cwd_write")
        self.assertEqual(result["outcome"], "cli_error")
        self.assertEqual(result["cwd_after"], ["unexpected"])
        self.assertEqual(result["session_status"], "retired")
        following = self.attempt()
        self.assertEqual(self.invoke(following)["outcome"], "cli_error")

    def test_codex_native_output_is_distinct_and_bound_to_last_message(self):
        attempt = self.attempt(backend="codex", role="reviewer")
        result = self.invoke(attempt)
        self.assertEqual(result["outcome"], "completed", result)
        self.assertEqual(result["backend"], "codex")
        self.assertNotIn("result", result["event_types"])
        self.assertIsNone(result["usage"]["additive_tokens"]["input_tokens"])
        write_json(Path(attempt["directory"]) / "cli-tmp/response.json", {"attempt_id": attempt["attempt_id"]})
        replay = read_result(attempt, exit_code=0, termination={"confirmed": True})
        self.assertEqual(replay["outcome"], "receipt_error")

    def test_native_success_schema_nonce_session_and_complete_trace_are_independent(self):
        attempt = self.attempt()
        result = self.invoke(attempt)
        self.assertEqual(result["outcome"], "completed")
        path = Path(attempt["directory"]) / "events.jsonl"
        original = [json.loads(line) for line in path.read_text().splitlines()]
        for mode in ("missing_structured", "bad_schema", "wrong_nonce", "different_session", "native_error", "no_terminal", "duplicate_terminal", "truncated"):
            with self.subTest(mode=mode):
                events = deepcopy(original)
                if mode == "missing_structured": events[-1].pop("structured_output")
                if mode == "bad_schema": events[-1]["structured_output"]["extra"] = 1
                if mode == "wrong_nonce": events[-1]["structured_output"]["attempt_id"] = str(uuid4())
                if mode == "different_session": events[-1]["session_id"] = str(uuid4())
                if mode == "native_error": events[-1]["is_error"] = True
                if mode == "no_terminal": events.pop()
                if mode == "duplicate_terminal": events.append(deepcopy(events[-1]))
                body = "\n".join(json.dumps(event) for event in events) + ("" if mode == "truncated" else "\n")
                path.write_text(body)
                for outcome in ("completed", "receipt_error"):
                    replay = read_result(attempt, exit_code=0, termination={"confirmed": True}, outcome=outcome)
                    expected = "receipt_error" if mode in {"missing_structured", "bad_schema", "wrong_nonce"} else "cli_error"
                    self.assertEqual(replay["outcome"], expected)
        path.write_text("\n".join(json.dumps(event) for event in original) + "\n")
        session_path = Path(attempt["session_directory"]) / "session.json"
        write_json(session_path, {**read_json(session_path), "status": "running"})
        recovered = read_result(attempt, exit_code=0, termination={"confirmed": True})
        self.assertEqual(settle_session(attempt, result=recovered)["status"], "ready")
        for outcome in ("completed", "receipt_error"):
            self.assertEqual(read_result(attempt, exit_code=1, termination={"confirmed": True}, outcome=outcome)["outcome"], "cli_error")
            self.assertEqual(read_result(attempt, exit_code=0, termination={"confirmed": False}, outcome=outcome)["outcome"], "termination_unconfirmed")
        for outcome in ("timeout", "cancelled", "termination_unconfirmed"):
            self.assertEqual(read_result(attempt, exit_code=0, termination={"confirmed": True}, outcome=outcome)["outcome"], outcome)

    def test_cancel_kills_detached_descendants_before_retiring_session(self):
        attempt = self.attempt()
        cancel = threading.Event()
        heartbeat = Path(attempt["directory"]) / "cli-tmp/heartbeat"
        def trigger():
            deadline = time.monotonic() + 8
            while not heartbeat.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            time.sleep(0.6)
            cancel.set()
        thread = threading.Thread(target=trigger)
        thread.start()
        result = self.invoke(attempt, action="daemon", cancel=cancel)
        thread.join(timeout=1)
        self.assertEqual(result["outcome"], "cancelled", result)
        self.assertTrue(result["termination"]["confirmed"])
        self.assertTrue(heartbeat.exists())
        stamp = heartbeat.stat().st_mtime_ns
        time.sleep(0.15)
        self.assertEqual(heartbeat.stat().st_mtime_ns, stamp)
        owned = read_json(Path(attempt["directory"]) / "process.json")
        table = process_table()
        self.assertTrue(all(process_alive(item, table) is False for item in [owned["identity"], *owned["descendants"]]))
        self.assertEqual(read_json(Path(attempt["session_directory"]) / "session.json")["status"], "retired")
        with self.assertRaisesRegex(ExtractError, "unconfirmed"):
            retire_session(attempt, termination={"confirmed": False}, reason="test")

    def test_launch_gate_and_symlink_inputs_fail_without_executing_cli(self):
        attempt = self.attempt()
        cancel = threading.Event()
        result = self.invoke(attempt, cancel=cancel, on_process=lambda _: cancel.set())
        self.assertEqual(result["outcome"], "cancelled")
        self.assertTrue(result["termination"]["confirmed"])
        self.assertEqual(read_json(Path(attempt["directory"]) / "launch.json")["status"], "not_started")
        self.assertFalse((Path(attempt["session_directory"]) / "home/session-fixture.json").exists())
        attempt = self.attempt()
        target = Path(attempt["workspace"]) / "root.yaml"
        target.unlink()
        target.symlink_to(self.host)
        result = self.invoke(attempt)
        self.assertEqual(result["outcome"], "cli_error")
        self.assertIsNone(result["exit_code"])
        self.assertEqual(self.host.read_text(), "protected host")

    def test_successful_cli_also_reaps_detached_descendants(self):
        attempt = self.attempt()
        result = self.invoke(attempt, action="orphan", timeout=5)
        self.assertEqual(result["outcome"], "completed", result)
        self.assertTrue(result["termination"]["confirmed"])
        self.assertEqual(read_json(Path(attempt["directory"]) / "cli-exit.json")["exit_code"], 0)
        owned = read_json(Path(attempt["directory"]) / "process.json")
        table = process_table()
        self.assertTrue(all(process_alive(item, table) is False for item in [owned["identity"], *owned["descendants"]]))

    def test_cancel_ordinary_child(self):
        attempt = self.attempt()
        result = self.invoke(attempt, action="child", timeout=0.8)
        self.assertEqual(result["outcome"], "timeout", result)
        self.assertTrue(result["termination"]["confirmed"])

    def test_timeout_and_unconfirmed_termination_keep_the_session_fence(self):
        attempt = self.attempt()
        result = self.invoke(attempt, action="daemon", timeout=0.8)
        self.assertEqual(result["outcome"], "timeout")
        self.assertTrue(result["termination"]["confirmed"])
        self.assertEqual(result["session_status"], "retired")
        attempt = self.attempt()
        with patch("schema_extract.tools.agent.terminate_owned", return_value={"confirmed": False, "actions": [], "survivors": []}):
            result = self.invoke(attempt)
        self.assertEqual(result["outcome"], "termination_unconfirmed")
        self.assertEqual(result["session_status"], "fenced")
        with self.assertRaisesRegex(ExtractError, "termination recovery"):
            self.invoke(self.attempt())

    def test_registered_scope_and_missing_backend_never_use_defaults(self):
        first = self.attempt()
        result = self.invoke(first)
        wrong = self.attempt(stage="P2", session_id=result["session_id"])
        wrong["session_directory"] = first["session_directory"]
        with self.assertRaisesRegex(ExtractError, "scope differs"):
            self.invoke(wrong)
        missing = self.attempt()
        missing.pop("backend")
        with self.assertRaises(ExtractError):
            AgentCLI(backend="claude", executable=str(self.fake), model=missing["model"]).run(
                attempt=missing, prompt="", schema=self.schema, timeout=5)

    def test_controlled_commands_preserve_model_and_do_not_load_customizations(self):
        cli = ClaudeBackend()
        argv = cli.command("claude", model="qwen3.8-max[1m]", effort="max", role="reviewer",
                           session_id=None, schema=self.schema, cwd=self.root, workspace=self.source,
                           schema_path=self.root / "schema.json", output_path=self.root / "response.json", check=self.helper)
        self.assertEqual(argv[argv.index("--model") + 1], "qwen3.8-max[1m]")
        self.assertIn("Bash(" + str(self.helper) + ")", argv[argv.index("--allowedTools") + 1])
        self.assertIn("--safe-mode", argv)
        self.assertNotIn("--fallback-model", argv)
        self.assertNotIn("--continue", argv)
        self.assertNotIn("Edit", argv[argv.index("--tools") + 1])
        for role in ("worker", "reviewer"):
            options = dict(model="fixture", effort="high", role=role, session_id=None, schema=self.schema,
                cwd=self.root, workspace=self.source, schema_path=self.root / "schema.json",
                output_path=self.root / "response.json", writable_files=["root.yaml"] if role == "worker" else [])
            native = CodexBackend().command("codex", **options)
            self.assertEqual(native[native.index("-s") + 1], "workspace-write" if role == "worker" else "read-only")
            self.assertEqual("--add-dir" in native, role == "worker")
            claude = cli.command("claude", **options)
            allowed = claude[claude.index("--allowedTools") + 1]
            self.assertEqual("Edit(" in allowed, role == "worker")
            if role == "worker":
                self.assertIn("Edit(/" + str(self.source / "root.yaml") + ")", allowed)
        (self.auth / "settings.json").write_text(json.dumps({"env": {"ANTHROPIC_AUTH_TOKEN": "fixture-secret",
            "ANTHROPIC_BASE_URL": "https://fixture.invalid", "CLAUDE_CODE_EFFORT_LEVEL": "low", "MAX_THINKING_TOKENS": "1"}}))
        home = self.root / "home"
        home.mkdir()
        env, binding = cli.environment(home, "max")
        self.assertEqual(env["CLAUDE_CODE_EFFORT_LEVEL"], "max")
        self.assertNotIn("MAX_THINKING_TOKENS", env)
        self.assertNotIn("fixture-secret", binding)
        (self.auth / "config.toml").write_text('model_provider="fixture"\n[model_providers.fixture]\nname="fixture"\nbase_url="https://fixture.invalid"\nwire_api="responses"\n[features]\nhooks=true\nmulti_agent=true\n')
        CodexBackend().environment(home, "xhigh")
        import tomllib
        saved = tomllib.loads((home / ".codex/config.toml").read_text())
        self.assertEqual(saved["model_provider"], "fixture")
        self.assertFalse(saved["features"]["hooks"])
        self.assertFalse(saved["features"]["multi_agent"])
        self.assertEqual(saved["project_doc_max_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
