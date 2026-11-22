"""Direct native launch, role permissions, scoped sessions and owned descendants."""

from copy import deepcopy
import json
import itertools
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
from schema_extract.tools.processes import (gate_reaped, launch_not_started, observe_owned,
                                           process_alive, process_identity, process_table, terminate_owned)
from schema_extract.tools.recovery import confirm_termination, never_launched, recover_result, termination_observed


FAKE = r'''
import json, os, subprocess, sys, time
from pathlib import Path
from uuid import uuid4
if '--version' in sys.argv:
    if (Path(__file__).parent / 'hang-version').exists():
        (Path(os.environ['TMPDIR']) / 'version-pid').write_text(str(os.getpid()))
        while True: time.sleep(1)
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
if p.get('action') == 'wait_for_release':
    signal = Path(os.environ['TMPDIR']) / 'release-model'
    while not signal.exists(): time.sleep(0.02)
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

    def invoke(self, attempt, *, action="normal", cancel=None, on_process=lambda _: None):
        prompt = json.dumps({"attempt_id": attempt["attempt_id"], "host": str(self.host), "parent_pid": os.getpid(),
                             "sticky": attempt["attempt_id"], "action": action, "workspace": attempt["workspace"],
                             "role": attempt["role"], "source": str(self.source / "frozen.txt"), "check_command": attempt["check_command"]})
        return AgentCLI(backend=attempt["backend"], executable=str(self.fake), model=attempt["model"]).run(
            attempt=attempt, prompt=prompt, schema=self.schema, cancel_event=cancel, on_process=on_process)

    def wait_closed(self, attempt, result, outcome):
        process = read_json(Path(attempt["directory"]) / "process.json")
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            termination = observe_owned(process["identity"], process["descendants"])
            if termination["confirmed"] and (gate_reaped(attempt) or launch_not_started(attempt)):
                break
            time.sleep(0.02)
        else:
            self.fail("synthetic invocation did not close")
        try:
            _, status = os.waitpid(process["identity"]["pid"], os.WNOHANG)
            exit_code = os.waitstatus_to_exitcode(status)
        except ChildProcessError:
            exit_code = (read_json(Path(attempt["directory"]) / "cli-exit.json")["exit_code"]
                         if gate_reaped(attempt) else result["exit_code"])
        settled = read_result(attempt, exit_code=exit_code, termination=termination, outcome=outcome)
        settled["session_status"] = settle_session(attempt, result=settled)["status"]
        return settled

    def cancel_on_file(self, path, cancel):
        def trigger():
            deadline = time.monotonic() + 8
            while not path.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            cancel.set()
        thread = threading.Thread(target=trigger)
        thread.start()
        self.addCleanup(thread.join, 9)
        return thread

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

    def test_session_preflight_failure_is_bound_unstarted_evidence(self):
        first = self.attempt()
        completed = self.invoke(first)
        session_path = Path(first["session_directory"]) / "session.json"
        original = session_path.read_bytes()
        attempt = self.attempt()  # Same scope, but missing the exact completed UUID.
        result = self.invoke(attempt)
        self.assertEqual(result["outcome"], "cli_error")
        self.assertIn("exact registered role session UUID", result["error"])
        self.assertEqual(result["session_status"], "not_started")
        self.assertIsNone(result["exit_code"])
        self.assertTrue(never_launched(attempt, result))
        self.assertTrue(termination_observed(attempt, result))
        termination = confirm_termination(attempt, result)
        self.assertTrue(termination["confirmed"])
        self.assertEqual(recover_result(attempt, result, termination)["session_status"], "not_started")
        self.assertEqual(session_path.read_bytes(), original)
        for name in ("launch.json", "process.json", "exit.json", "cli-exit.json", "events.jsonl"):
            self.assertFalse((Path(attempt["directory"]) / name).exists(), name)
        resumed = self.invoke(self.attempt(session_id=completed["session_id"]))
        self.assertEqual(resumed["outcome"], "completed")
        self.assertEqual(resumed["session_id"], completed["session_id"])
        unknown = self.attempt(stage="P2")
        self.assertFalse(confirm_termination(unknown, None, launch_state="dispatched")["confirmed"])
        self.assertFalse(termination_observed(unknown, None, launch_state="dispatched"))

    def test_codex_native_output_is_distinct_and_bound_to_last_message(self):
        attempt = self.attempt(backend="codex", role="reviewer")
        result = self.invoke(attempt)
        self.assertEqual(result["outcome"], "completed", result)
        self.assertEqual(result["backend"], "codex")
        self.assertNotIn("result", result["event_types"])
        self.assertIsNone(result["usage"]["additive_tokens"]["input_tokens"])
        write_json(Path(attempt["directory"]) / "cli-tmp/response.json", {"attempt_id": attempt["attempt_id"]})
        replay = read_result(attempt, exit_code=0, termination={"confirmed": True})
        self.assertEqual(replay["outcome"], "cli_error")
        self.assertEqual(replay["response_validation"]["failure_kind"], "native_output_mismatch")
        events_path = Path(attempt["directory"]) / "events.jsonl"
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        # Corrupt both channels differently: native disagreement takes priority
        # over the invalid proposal in the final-message event.
        for event in events:
            if event.get('type') == 'item.completed' and event.get('item', {}).get('type') == 'agent_message':
                event['item']['text'] = json.dumps({'attempt_id': attempt['attempt_id'], 'invalid': True})
        events_path.write_text('\n'.join(json.dumps(event) for event in events) + '\n')
        replay = read_result(attempt, exit_code=0, termination={"confirmed": True})
        self.assertFalse(replay['response_validation']['schema_valid'])
        self.assertEqual(replay['outcome'], 'cli_error')

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
        for outcome in ("cancelled", "termination_unconfirmed"):
            self.assertEqual(read_result(attempt, exit_code=0, termination={"confirmed": True}, outcome=outcome)["outcome"], outcome)

    def test_cancel_kills_detached_descendants_before_retiring_session(self):
        attempt = self.attempt()
        cancel = threading.Event()
        heartbeat = Path(attempt["directory"]) / "cli-tmp/heartbeat"
        self.cancel_on_file(heartbeat, cancel)
        result = self.invoke(attempt, action="daemon", cancel=cancel)
        self.assertIn(result["outcome"], {"cancelled", "termination_unconfirmed"}, result)
        self.assertTrue(all(item["signal"] == "SIGKILL" for item in result["termination"]["actions"]))
        if not result["termination"]["confirmed"]:
            self.assertEqual(result["session_status"], "fenced")
        result = self.wait_closed(attempt, result, "cancelled")
        self.assertEqual(result["outcome"], "cancelled", result)
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
        result = self.wait_closed(attempt, result, "cancelled")
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
        result = self.invoke(attempt, action="orphan")
        result = self.wait_closed(attempt, result, "completed")
        self.assertEqual(result["outcome"], "completed", result)
        self.assertTrue(result["termination"]["confirmed"])
        self.assertEqual(read_json(Path(attempt["directory"]) / "cli-exit.json")["exit_code"], 0)
        owned = read_json(Path(attempt["directory"]) / "process.json")
        table = process_table()
        self.assertTrue(all(process_alive(item, table) is False for item in [owned["identity"], *owned["descendants"]]))

    def test_cancel_ordinary_child(self):
        attempt = self.attempt()
        cancel = threading.Event()
        self.cancel_on_file(Path(attempt["directory"]) / "cli-tmp/heartbeat", cancel)
        result = self.invoke(attempt, action="child", cancel=cancel)
        result = self.wait_closed(attempt, result, "cancelled")
        self.assertEqual(result["outcome"], "cancelled", result)
        self.assertTrue(result["termination"]["confirmed"])

    def test_unconfirmed_termination_keeps_the_session_fence(self):
        attempt = self.attempt()
        with patch("schema_extract.tools.agent.observe_owned", return_value={"confirmed": False, "actions": [], "survivors": []}):
            result = self.invoke(attempt)
        self.assertEqual(result["outcome"], "termination_unconfirmed")
        self.assertEqual(result["session_status"], "fenced")
        session_path = Path(attempt["session_directory"]) / "session.json"
        original = session_path.read_bytes()
        following = self.attempt()
        rejected = self.invoke(following)
        self.assertTrue(never_launched(following, rejected))
        self.assertIn("termination recovery", rejected["error"])
        self.assertEqual(session_path.read_bytes(), original)

    def test_registered_scope_and_missing_backend_never_use_defaults(self):
        first = self.attempt()
        result = self.invoke(first)
        wrong = self.attempt(stage="P2", session_id=result["session_id"])
        wrong["session_directory"] = first["session_directory"]
        rejected = self.invoke(wrong)
        self.assertTrue(never_launched(wrong, rejected))
        self.assertIn("scope differs", rejected["error"])
        missing = self.attempt()
        missing.pop("backend")
        with self.assertRaises(ExtractError):
            AgentCLI(backend="claude", executable=str(self.fake), model=missing["model"]).run(
                attempt=missing, prompt="", schema=self.schema)

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

    def test_claude_default_output_is_fixed_and_environment_fingerprinted(self):
        (self.auth / "settings.json").write_text(json.dumps({"env": {
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "4096", "MAX_THINKING_TOKENS": "1"}}))
        home = self.root / "default-home"
        home.mkdir()
        backend = ClaudeBackend()
        with patch.dict(os.environ, {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "8192"}):
            environment, fingerprint = backend.environment(home, "max")
        self.assertEqual(environment["CLAUDE_CODE_MAX_OUTPUT_TOKENS"], "128000")
        self.assertEqual(environment["CLAUDE_CODE_EFFORT_LEVEL"], "max")
        self.assertNotIn("MAX_THINKING_TOKENS", environment)
        self.assertNotIn("CLAUDE_CODE_MAX_OUTPUT_TOKENS", CodexBackend().environment(home, "xhigh")[0])
        with patch.dict(os.environ, {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "16384"}):
            self.assertEqual(backend.environment(home, "max")[1], fingerprint)
        first = self.attempt()
        result = self.invoke(first)
        self.assertEqual(result["outcome"], "completed", result)
        self.assertNotIn("max_output_tokens", result)
        changed = self.attempt(session_id=result["session_id"])
        with patch("schema_extract.tools.backends.CLAUDE_MAX_OUTPUT_TOKENS", 65536):
            rejected = self.invoke(changed)
        self.assertTrue(never_launched(changed, rejected))
        self.assertIn("output configuration changed", rejected["error"])

    def test_large_elapsed_time_is_only_an_accounting_value(self):
        attempt = self.attempt()
        with patch("schema_extract.tools.agent.time.monotonic", side_effect=itertools.count(0, 1000000)):
            result = self.invoke(attempt)
        self.assertEqual(result["outcome"], "completed", result)
        self.assertGreater(result["elapsed_seconds"], 1000000)

    def test_version_mismatch_stops_before_model_execution(self):
        attempt = self.attempt()
        attempt["cli_version"] = "different-registered-version"
        write_json(Path(attempt["directory"]) / "attempt.json", attempt)
        result = self.invoke(attempt)
        self.assertIn("CLI version differs", result["error"])
        closed = self.wait_closed(attempt, result, "cli_error")
        self.assertTrue(closed["termination"]["confirmed"])
        self.assertTrue(launch_not_started(attempt))
        self.assertFalse((Path(attempt["session_directory"]) / "home/session-fixture.json").exists())

    def test_version_query_is_owned_cancellable_and_never_reaches_model(self):
        (self.root / "hang-version").touch()
        attempt, cancel = self.attempt(), threading.Event()
        self.cancel_on_file(Path(attempt["directory"]) / "cli-tmp/version-pid", cancel)
        result = self.invoke(attempt, cancel=cancel)
        self.assertIn(result["outcome"], {"cancelled", "termination_unconfirmed"})
        self.assertTrue(result["termination"]["actions"])
        closed = self.wait_closed(attempt, result, "cancelled")
        self.assertTrue(closed["termination"]["confirmed"])
        self.assertTrue(launch_not_started(attempt))
        self.assertFalse((Path(attempt["session_directory"]) / "home/session-fixture.json").exists())
        self.assertEqual((Path(attempt["directory"]) / "events.jsonl").read_bytes(), b"")

    def test_observation_failure_does_not_terminate_a_running_model(self):
        attempt = self.attempt()
        real_owned = __import__("schema_extract.tools.agent", fromlist=["owned_processes"]).owned_processes
        release = Path(attempt["directory"]) / "cli-tmp/release-model"
        calls = 0
        def failing_observation(identity, known):
            nonlocal calls
            calls += 1
            if calls > 1:
                release.touch()
                raise OSError("synthetic proc observation failure")
            return real_owned(identity, known)
        with patch("schema_extract.tools.agent.owned_processes", side_effect=failing_observation), \
                patch("schema_extract.tools.agent.terminate_owned", wraps=terminate_owned) as terminate:
            result = self.invoke(attempt, action="wait_for_release")
        self.assertEqual(result["outcome"], "completed", result)
        self.assertIn("synthetic proc observation failure", result["process_observation_error"])
        terminate.assert_not_called()

    def test_proc_identity_errors_are_unknown_and_reused_pid_is_not_signalled(self):
        identity = process_identity(os.getpid())
        self.assertIn("start_token", identity)
        self.assertNotIn("started", identity)
        with patch("schema_extract.tools.processes._linux_process", side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                process_table()
        stale = {**identity, "start_token": identity["start_token"] + "0"}
        with patch("schema_extract.tools.processes.signal_process") as deliver:
            self.assertEqual(terminate_owned(stale, include_anchor=True)["actions"], [])
        deliver.assert_not_called()


if __name__ == "__main__":
    unittest.main()
