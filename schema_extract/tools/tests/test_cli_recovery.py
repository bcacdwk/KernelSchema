"""Real CLI/store boundaries with tiny local inputs and an executable EX_TEMPFAIL stub."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import fcntl
import io
from uuid import uuid4
from unittest.mock import patch

from schema_extract.tools.common import ExtractError, read_json
from schema_extract.tools.monitor import reserve_launch
from schema_extract.tools.store import Store


REPO = Path(__file__).resolve().parents[3]
ENTRY = REPO / "schema_extract/extract_yaml.py"


class ControlCommitFactsTests(unittest.TestCase):
    """Real isolated ledger/state, with execution and process side effects replaced."""

    def setUp(self):
        from schema_extract.tools import cli, monitor
        from schema_extract.tools.common import write_json
        from schema_extract.tools.runtime import EXECUTOR_CAPABILITIES
        from schema_extract.tools.stages import new_scope
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = Store(self.root / "store")
        self.store.initialize()
        self.cli_module, self.monitor = cli, monitor
        self.run_id = str(uuid4())
        self.state = {"format": "schema_extract.run.v2", "run_id": self.run_id, "operator_id": "fixture",
            "input_fingerprint": "fixed", "executor": {"digest": "fixed", "capabilities": list(EXECUTOR_CAPABILITIES)},
            "output": str(self.root / "output"), "created_at": 1, "status": "queued", "published": False,
            "stage": "P1", "role": "worker", "node": "root", "mode": "normal", "checkpoint": None,
            "control": {"action": None, "requests": {}}, "active_attempts": {}, "unconfirmed_attempts": {},
            "unsettled_attempts": {}, "reopen_requests": {}, "selected_designs": None, "writing_plan": None,
            "reason": None, "exports": [], "plan_revision": 1, "attempts": [], "attempt_usage": {}, "call_additions": [],
            "stages": {stage: new_scope() for stage in ("P1", "P2", "P3", "P4", "P5", "P6")},
            "nodes": {"root": {"id": "root", "file": "root.yaml", "status": "draft", "generation": 0}},
            "config": {"backend": "claude", "agent_binary": sys.executable, "model": "fixture", "worker_effort": "high",
                       "reviewer_effort": "high", "max_calls": 6, "revision_limit": 2}}
        self.store.run_dir(self.run_id).mkdir(parents=True)
        self.store.save(self.state)
        write_json(self.store.run_dir(self.run_id) / "inputs.lock.json", {"fingerprint": "fixed"})
        with self.store.supervisor() as ledger:
            ledger["runs_indexed"] = True
            ledger["starts"][self.run_id] = {"run_id": self.run_id, "output": self.state["output"],
                                            "authorized": True, "created_at": 1}
        self.identity = {"pid": os.getpid(), "start_token": "linux:fixture:1", "stat": "S", "ppid": 1, "pgid": os.getpid()}
        for target, options in (
                ("schema_extract.tools.runtime.verify_execution", {}),
                ("schema_extract.tools.runtime.frozen_run_invocation", {}),
                ("schema_extract.tools.monitor.process_table", {"return_value": {os.getpid(): self.identity}}),
                ("schema_extract.tools.cli._frozen_entry", {}),
                ("schema_extract.tools.cli.Store", {"return_value": self.store})):
            mocker = patch(target, **options)
            mocker.start()
            self.addCleanup(mocker.stop)

    @contextmanager
    def locked(self):
        with (self.store.root / "supervisor.lock").open("a") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield

    def invoke(self, *arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr), patch.dict(os.environ, {}, clear=True):
            code = self.cli_module.main([*arguments, "--repo", str(self.root), "--state-root", str(self.store.root)])
        return code, json.loads(stderr.getvalue() or stdout.getvalue())

    def test_busy_before_registration_is_false_despite_unrelated_revision(self):
        before = self.store.read_supervisor()
        with self.locked(), patch.object(self.monitor.subprocess, "Popen") as popen:
            code, result = self.invoke("resume", self.run_id)
        self.assertEqual(code, 1)
        self.assertIs(result["committed"], False)
        self.assertGreater(before["revision"], 0)
        self.assertEqual(self.store.read_supervisor(), before)
        popen.assert_not_called()

    def test_background_busy_before_spawn_and_after_spawn_preserves_one_launch(self):
        from schema_extract.tools.common import SupervisorBusy
        for moment in ("before_spawn", "after_spawn"):
            with self.subTest(moment=moment):
                with self.store.supervisor() as ledger:
                    ledger["starts"][self.run_id].pop("launch", None)
                lock = None
                class Child:
                    pid = 87654321
                    def poll(self):
                        return None
                def spawn(*args, **kwargs):
                    nonlocal lock
                    if moment == "before_spawn":
                        raise SupervisorBusy("injected before spawn")
                    lock = (self.store.root / "supervisor.lock").open("a")
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return Child()
                try:
                    with patch.object(self.monitor.subprocess, "Popen", side_effect=spawn) as popen:
                        result = self.monitor.start_background(self.store, self.run_id, self.root)
                    self.assertIs(result["committed"], True)
                    original = self.store.read_supervisor()["starts"][self.run_id]["launch"]
                    self.assertEqual(result["id"], original["id"])
                    self.assertEqual(popen.call_count, 1)
                    if moment == "after_spawn":
                        self.assertEqual(result["delivery"]["pid"], Child.pid)
                        self.assertFalse(result["delivery"]["evidence_committed"])
                finally:
                    if lock:
                        lock.close()
                with patch.object(self.monitor.subprocess, "Popen") as repeated:
                    retry = self.monitor.start_background(self.store, self.run_id, self.root)
                repeated.assert_not_called()
                self.assertEqual(retry["id"], original["id"])
                self.assertTrue(retry["reused"])

    def test_unknown_commit_keeps_candidate_id_and_can_be_reconciled(self):
        from schema_extract.tools.common import write_json
        def uncertain(path, value):
            write_json(path, value)
            if path.name == "supervisor.json":
                raise OSError("lost commit acknowledgement")
        with patch("schema_extract.tools.store.write_json", side_effect=uncertain):
            code, result = self.invoke("resume", self.run_id, "--add-calls", "2")
        self.assertEqual(code, 2)
        self.assertIsNone(result["committed"])
        saved = self.store.read_supervisor()["starts"][self.run_id]["launch"]
        self.assertEqual(result["launch"]["id"], saved["id"])
        self.assertEqual(saved["add_calls"], 2)
        code, repeated = self.invoke("resume", self.run_id, "--add-calls", "2")
        self.assertEqual(repeated["launch"]["id"], saved["id"])
        self.assertEqual(self.store.load(self.run_id)["call_additions"], [])

    def test_resume_export_and_maintenance_keep_the_committed_identity(self):
        from schema_extract.tools.common import SupervisorBusy
        for command in ("resume", "export", "maintain"):
            with self.subTest(command=command):
                with self.store.supervisor() as ledger:
                    ledger["starts"][self.run_id].pop("launch", None)
                    ledger["controls"].clear()
                extra = []
                target = "run"
                if command == "export":
                    extra = ["--output", str(self.root / "export")]
                    target = "export_checkpoint"
                elif command == "maintain":
                    tasks = self.root / "tasks.json"
                    tasks.write_text(json.dumps([{"stage": "P1", "node": "root", "action": "review", "fields": [], "reason": "fixture"}]))
                    extra = ["--issue-id", "fixture", "--tasks", str(tasks)]
                    target = "monitor.start_background"
                with patch.object(self.store, "context"), patch("schema_extract.tools.cli." + target,
                        side_effect=SupervisorBusy("later transaction busy")) as side_effect:
                    code, result = self.invoke(command, self.run_id, *extra)
                self.assertEqual(code, 1)
                self.assertTrue(result["committed"])
                ledger = self.store.read_supervisor()
                if command == "maintain":
                    self.assertEqual(result["request"]["request"]["id"], ledger["maintenance_requests"][self.run_id]["id"])
                else:
                    self.assertEqual(result["launch"]["id"], ledger["starts"][self.run_id]["launch"]["id"])
                self.assertEqual(side_effect.call_count, 1)
                self.assertEqual(self.store.load(self.run_id)["call_additions"], [])

    def test_restart_request_and_inspect_reconcile_after_reservation_busy(self):
        evidence = {"run_id": self.run_id, "scope": "P1", "failure_id": "fixed-failure"}
        start = self.monitor.start_background
        def busy_start(*args, **kwargs):
            with self.locked():
                return start(*args, **kwargs)
        with patch.object(self.monitor, "restart_evidence", return_value=evidence), \
                patch.object(self.monitor, "start_background", side_effect=busy_start):
            code, result = self.invoke("resume", self.run_id, "--scope", "P1")
        self.assertEqual(code, 1)
        self.assertTrue(result["committed"])
        request = result["request"]
        self.assertIn(request["id"], self.store.read_supervisor()["retries"])
        with self.store.writer(self.run_id), patch.object(self.monitor, "restart_evidence", return_value=evidence):
            reused = self.monitor.request_restart(self.store, self.run_id, "P1", self.root)
        self.assertEqual(reused["id"], request["id"])
        before = (self.store.root / "supervisor.json").read_bytes()
        rows = self.monitor.inspect_runs(self.store, [self.run_id])
        self.assertEqual(rows[0]["requests"]["recovery"]["recent_by_scope"]["P1"]["id"], request["id"])
        self.assertEqual((self.store.root / "supervisor.json").read_bytes(), before)
        for action in ("pause", "cancel"):
            request = self.store.request_control(self.run_id, action)
            rows = self.monitor.inspect_runs(self.store, [self.run_id])
            self.assertIn(request["id"], [item["id"] for item in rows[0]["requests"]["control"]["pending"]])
            with patch.object(self.monitor, "restart_evidence", return_value=evidence):
                with self.assertRaises(ExtractError):
                    self.monitor.request_restart(self.store, self.run_id, "P1", self.root)

    def test_new_controls_and_replaced_launch_prevent_old_claim(self):
        for action in ("pause", "cancel", "hold", "replace"):
            with self.subTest(action=action):
                with self.store.supervisor() as ledger:
                    ledger["controls"].clear()
                    ledger["active_hold"] = None
                    ledger["starts"][self.run_id].pop("launch", None)
                state = self.store.load(self.run_id)
                state["control"]["requests"] = {item["id"]: item for item in self.store.control_requests(self.run_id)}
                self.store.save(state)
                launch = reserve_launch(self.store, state, mode="resume", background=True)
                if action in {"pause", "cancel"}:
                    self.store.request_control(self.run_id, action)
                elif action == "hold":
                    self.monitor.hold_all(self.store)
                else:
                    with self.store.supervisor() as ledger:
                        ledger["starts"][self.run_id]["launch"]["id"] = str(uuid4())
                with self.assertRaises(ExtractError):
                    self.monitor.claim_launch(self.store, self.run_id, launch["id"])

    def test_hold_release_survives_final_bookkeeping_busy(self):
        from schema_extract.tools.common import SupervisorBusy
        hold = self.monitor.hold_all(self.store)
        supervisor = self.store.supervisor
        count = 0
        @contextmanager
        def busy_final():
            nonlocal count
            count += 1
            if count == 2:
                with self.locked(), supervisor() as ledger:
                    yield ledger
            else:
                with supervisor() as ledger:
                    yield ledger
        with patch.object(self.store, "supervisor", busy_final), \
                patch.object(self.monitor, "start_background", side_effect=SupervisorBusy("launch busy")):
            code, result = self.invoke("resume", "--all", "--hold", hold["id"])
        self.assertEqual(code, 1)
        self.assertTrue(result["committed"])
        self.assertEqual(result["hold_id"], hold["id"])
        self.assertEqual(result["hold"]["members"][self.run_id]["release"]["status"], "busy")
        self.assertIsNone(self.store.read_supervisor()["active_hold"])

    def test_single_and_batch_extract_preserve_registration_before_launch_busy(self):
        from schema_extract.tools.common import SupervisorBusy, write_json
        created = []
        def create(repo, root, output, **options):
            state = deepcopy(self.state)
            state.update(run_id=options["run_id"], output=str(output), created_at=options["created_at"])
            self.store.run_dir(state["run_id"]).mkdir(parents=True, exist_ok=True)
            self.store.save(state)
            write_json(self.store.run_dir(state["run_id"]) / "inputs.lock.json", {"fingerprint": "fixed"})
            created.append(state["run_id"])
            return self.store, state, {}
        for batch in (False, True):
            with self.subTest(batch=batch):
                targets = [{"operator_id": name, "output": self.root / (name + str(batch))} for name in ("first", "second")]
                arguments = (["--ids", "first", "second", "--output-root", str(self.root / "batch")] if batch else
                             ["--id", "first", "--output", str(targets[0]["output"])])
                with patch("schema_extract.tools.runner.create_run", side_effect=create), \
                        patch.object(self.monitor, "catalog_targets", return_value=targets), \
                        patch.object(self.monitor, "start_background", side_effect=SupervisorBusy("before reservation")) as launch:
                    code, result = self.invoke("extract", *arguments, "--agent-binary", sys.executable, "--background")
                    repeated_code, repeated = self.invoke("extract", *arguments, "--agent-binary", sys.executable, "--background")
                self.assertEqual(code, 1)
                rows, repeated_rows = (result, repeated) if batch else ([result], [repeated])
                for row, retry in zip(rows, repeated_rows):
                    self.assertTrue(row["committed"])
                    self.assertEqual(row["run_id"], retry["run_id"])
                    saved = self.store.read_supervisor()["starts"][row["run_id"]]
                    self.assertEqual(row["registration"]["request_key"], saved["request_key"])
                    self.assertEqual(created.count(row["run_id"]), 1)
                    self.assertNotIn("launch", saved)
                self.assertEqual(launch.call_count, 4 if batch else 2)


class CLIRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "kernel.py").write_text("def kernel(x): return x + 1\ndef reference(x): return x + 1\ndef inputs(): return [1]\n")
        (self.source / "task.md").write_text("Scalar increment; integer input/output; exact comparison.\n")
        self.descriptor = self.root / "source.json"
        self.descriptor.write_text(json.dumps({"format": "schema_extract.source.v2", "id": "increment",
            "repository": str(self.source), "directory": ".", "entrypoint": "kernel.py::kernel",
            "input": "kernel.py::inputs", "reference": "kernel.py::reference", "io": ["task.md"],
            "acceptance": ["task.md"], "scope": ["task.md"]}))
        self.stub = self.root / "claude-stub"
        self.stub.write_text(f"#!{sys.executable}\nimport sys\nif '--version' in sys.argv:\n print('controlled-stub')\n sys.exit(0)\nsys.exit(75)\n")
        self.stub.chmod(0o755)
        self.store = Store(self.root / "runtime")
        self.output = self.root / "delivery"
        self.request = ["extract", "--source", str(self.descriptor), "--output", str(self.output),
                        "--agent-binary", str(self.stub), "--max-calls", "6"]
        result = self.cli(*self.request, "--queue")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.run_id = json.loads(result.stdout)["run_id"]

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(ENTRY), *args, "--repo", str(REPO),
                               "--state-root", str(self.store.root)], capture_output=True, text=True, timeout=30)

    def state(self):
        return self.store.load(self.run_id)

    def test_repeated_commands_report_saved_state_and_frozen_source(self):
        original = self.state()
        result = self.cli(*self.request)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(json.loads(result.stdout)["result"], "queued")
        state = self.state()
        state.update(status="blocked", reason={"code": "missing_input", "summary": "missing task fact"})
        self.store.save(state)
        (self.source / "kernel.py").write_text("changed source, intentionally not frozen\n")
        for flag in ([], ["--background"], ["--queue"]):
            result = self.cli(*self.request, *flag)
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual((report["run_id"], report["run_status"], report["published"]), (self.run_id, "blocked", False))
            self.assertEqual(report["frozen"]["input_fingerprint"], original["input_fingerprint"])
            self.assertEqual(report["frozen"]["config"], original["config"])
        result = self.cli(*self.request, "--worker-effort", "high")
        self.assertEqual(result.returncode, 2)
        self.assertIn("different request parameters", result.stderr)
        self.assertEqual(self.state()["attempts"], [])
        # Saved publication states exercise CLI reporting, not P6 validation.
        state = self.state()
        for published, expected in ((False, 1), (True, 0)):
            state.update(status="ready", published=published, reason=None,
                         publication={"status": "published" if published else "conflict"})
            self.store.save(state)
            result = self.cli(*self.request)
            self.assertEqual(result.returncode, expected, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["result"], "published" if published else "publication_pending")
            if not published:
                self.assertIn("export " + self.run_id, report["next_step"])

    def test_configuration_is_validated_before_registration_and_default_output_is_frozen(self):
        self.stub.write_text(f"#!{sys.executable}\nimport os, sys\n"
                            "if '--version' in sys.argv:\n print('controlled-stub')\n sys.exit(0)\n"
                            "print(os.environ.get('CLAUDE_CODE_MAX_OUTPUT_TOKENS', 'unset'), file=sys.stderr)\n"
                            "sys.exit(75)\n")
        request = list(self.request)
        output = self.root / "validated-delivery"
        request[request.index("--output") + 1] = str(output)
        for options, message in ((["--revision-limit", "0"], "revision_limit"),
                                 (["--max-calls", "3"], "max_calls must reserve"),
                                 (["--backend", "codex", "--model", "fixture", "--worker-effort", "max",
                                   "--reviewer-effort", "high"], "unsupported Codex effort")):
            rejected = self.cli(*request, *options, "--queue")
            self.assertEqual(rejected.returncode, 2, rejected.stderr)
            self.assertIn(message, rejected.stderr)
            self.assertFalse(output.exists())
            self.assertEqual(set(self.store.read_supervisor()["starts"]), {self.run_id})
            self.assertEqual(self.store.run_ids(), [self.run_id])
        queued = self.cli(*request, "--queue")
        self.assertEqual(queued.returncode, 0, queued.stderr)
        run_id = json.loads(queued.stdout)["run_id"]
        fixed = self.store.load(run_id)
        lock = read_json(self.store.run_dir(run_id) / "inputs.lock.json")
        self.assertEqual(lock["config"], fixed["config"])
        self.assertEqual(fixed["input_fingerprint"], self.state()["input_fingerprint"])
        changed = self.cli(*request, "--worker-effort", "high", "--queue")
        self.assertEqual(changed.returncode, 2, changed.stderr)
        self.assertIn("different request parameters", changed.stderr)
        with patch.dict(os.environ, {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "4096"}):
            resumed = self.cli("resume", run_id)
        self.assertEqual(resumed.returncode, 1, resumed.stderr)
        state = self.store.load(run_id)
        self.assertEqual(state["config"], fixed["config"])
        directory = self.store.run_dir(run_id) / "attempts" / state["attempts"][-1]
        self.assertEqual((directory / "stderr.txt").read_text().strip(), "128000")
        archive = self.root / "evidence.tar.gz"
        audit = self.cli("audit", run_id, "--verify", "--archive", str(archive), "--json")
        self.assertEqual(audit.returncode, 0, audit.stderr)
        report = json.loads(audit.stdout)
        self.assertEqual(report["verification"]["errors"], [])
        from schema_extract.tools.audit import _archive_manifest
        archived = {entry["name"] for entry in _archive_manifest(archive)["entries"]}
        prefix = directory.relative_to(self.store.root).as_posix() + "/cli-tmp/"
        self.assertTrue({prefix + name for name in ("version.json", "version.txt", "version-stderr.txt")} <= archived)

    def test_running_foreground_reuse_does_not_report_delivery_success(self):
        state = self.state()
        state["status"] = "running"
        self.store.save(state)
        reserve_launch(self.store, state)
        with self.store.writer(self.run_id):
            result = self.cli(*self.request)
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["result"], "already_running")
            self.assertFalse(report["published"])
            result = self.cli("resume", self.run_id, "--add-calls", "2")
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(json.loads(result.stdout)["result"], "already_running")
            self.assertEqual(self.state()["call_additions"], [])

    def test_resume_adds_calls_without_refunding_used_calls(self):
        result = self.cli("resume", self.run_id)
        self.assertEqual(result.returncode, 1, result.stderr)
        state = self.state()
        self.assertEqual((state["status"], state["stages"]["P1"]["calls"]), ("cli_error", 1))
        usage = deepcopy(state["attempt_usage"])
        result = self.cli("resume", self.run_id, "--add-calls", "2")
        self.assertEqual(result.returncode, 1, result.stderr)
        current = self.state()
        self.assertEqual((current["status"], current["stages"]["P1"]["calls"]), ("cli_error", 2))
        self.assertEqual(len(current["call_additions"]), 1)
        self.assertEqual(current["call_additions"][0]["calls"], 2)
        self.assertEqual(current["config"], state["config"])
        for key, value in usage.items():
            self.assertEqual(current["attempt_usage"][key], value)
        report = self.cli("audit", self.run_id, "--json")
        self.assertEqual(report.returncode, 0, report.stderr)
        self.assertEqual(len(json.loads(report.stdout)["call_additions"]), 1)

    def test_resume_confirms_cancel_before_later_pause_even_without_delivery_record(self):
        from schema_extract.tools.cli import _launch_context
        self.store.request_control(self.run_id, "cancel")
        self.store.request_control(self.run_id, "pause")
        order = []
        def deliver(store, request):
            order.append("delivery")
            return {"confirmed": True}
        def reserve(*args, **kwargs):
            order.append("reserve")
            return {"id": "new-launch", "status": "claimed"}
        with patch.dict(os.environ, {}, clear=True), \
                patch("schema_extract.tools.cli.monitor.deliver_cancel", return_value={"confirmed": False}), \
                patch("schema_extract.tools.cli.monitor.reserve_launch") as reservation:
            with self.assertRaisesRegex(ExtractError, "cancel termination is pending"):
                _launch_context(self.store, self.state())
            reservation.assert_not_called()
        with patch.dict(os.environ, {}, clear=True), \
                patch("schema_extract.tools.cli.monitor.deliver_cancel", side_effect=deliver), \
                patch("schema_extract.tools.cli.monitor.reserve_launch", side_effect=reserve):
            _launch_context(self.store, self.state())
        self.assertEqual(order, ["delivery", "reserve"])

    def test_rejection_records_the_registered_attempt_not_the_bad_nonce(self):
        self.stub.write_text(f"#!{sys.executable}\n" + '''import json, sys
from pathlib import Path
if '--version' in sys.argv:
    print('controlled-stub')
    sys.exit(0)
sys.stdin.read()
session = '00000000-0000-4000-8000-000000000001'
print(json.dumps({'type': 'system', 'subtype': 'init', 'session_id': session}))
print(json.dumps({'type': 'result', 'subtype': 'success', 'is_error': False, 'session_id': session,
                  'structured_output': {'format': 'schema_extract.proposal.v1', 'attempt_id': 'wrong-invocation'},
                  'usage': {'input_tokens': 1, 'output_tokens': 1}}))
''')
        result = self.cli("resume", self.run_id)
        self.assertEqual(result.returncode, 1, result.stderr)
        state = self.state()
        attempt_id = state["attempts"][-1]
        self.assertEqual(state["reason"]["code"], "proposal_rejected")
        self.assertEqual(state["reason"]["attempt_id"], attempt_id)
        rejected = read_json(self.store.run_dir(self.run_id) / "attempts" / attempt_id / "rejected.json")
        self.assertEqual(rejected["attempt_id"], attempt_id)



if __name__ == "__main__":
    unittest.main()
