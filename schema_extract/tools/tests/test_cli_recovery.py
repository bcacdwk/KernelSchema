"""Real CLI/store boundaries with tiny local inputs and an executable EX_TEMPFAIL stub."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from schema_extract.tools.common import ExtractError, read_json
from schema_extract.tools.monitor import reserve_launch
from schema_extract.tools.store import Store


REPO = Path(__file__).resolve().parents[3]
ENTRY = REPO / "schema_extract/extract_yaml.py"


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
