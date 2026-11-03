"""Real CLI/store boundaries with tiny local inputs and an executable EX_TEMPFAIL stub."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from schema_extract.tools.common import read_json
from schema_extract.tools.monitor import reserve_launch
from schema_extract.tools.stages import new_node
from schema_extract.tools.store import Store


REPO = Path(__file__).resolve().parents[3]
ENTRY = REPO / "schema_extract/extract.py"


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
        self.stub = self.root / "codex-stub"
        self.stub.write_text(f"#!{sys.executable}\nimport sys\nif '--version' in sys.argv:\n print('controlled-stub')\n sys.exit(0)\nsys.exit(75)\n")
        self.stub.chmod(0o755)
        self.store = Store(self.root / "runtime")
        self.output = self.root / "delivery"
        self.request = ["extract", "--source", str(self.descriptor), "--output", str(self.output),
                        "--codex-bin", str(self.stub), "--operator-seconds", "600", "--stage-seconds", "600",
                        "--call-seconds", "60", "--max-calls", "6"]
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
            self.assertIn("未重新读取工作树", report["input_policy"])
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
            result = self.cli("resume", self.run_id, "--add-seconds", "600")
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(json.loads(result.stdout)["result"], "already_running")
            self.assertEqual(self.state()["budget_extensions"], [])

    def test_resume_adds_finite_credit_after_expiry_without_refunding_calls(self):
        result = self.cli("resume", self.run_id)
        self.assertEqual(result.returncode, 1, result.stderr)
        state = self.state()
        self.assertEqual((state["status"], state["stages"]["P1"]["calls"]), ("cli_error", 1))
        usage = deepcopy(state["attempt_usage"])
        state.update(status="budget_exhausted", deadline=time.time() - 100, reason={"code": "budget_exhausted"})
        state["stages"]["P1"].update(status="budget_exhausted", deadline=time.time() - 100)
        self.store.save(state)
        self.assertEqual(self.cli("resume", self.run_id).returncode, 1)
        self.assertEqual(self.state()["attempt_usage"], usage)
        result = self.cli("resume", self.run_id, "--add-seconds", "600", "--add-calls", "2")
        self.assertEqual(result.returncode, 1, result.stderr)  # Stub stops; no schema was approved.
        current = self.state()
        self.assertEqual((current["status"], current["stages"]["P1"]["calls"]), ("cli_error", 2))
        self.assertGreater(current["deadline"], time.time() + 500)
        self.assertEqual(len(current["budget_extensions"]), 1)
        self.assertEqual(current["config"], state["config"])
        for key, value in usage.items():
            self.assertEqual(current["attempt_usage"][key], value)
        report = self.cli("audit", self.run_id, "--json")
        self.assertEqual(report.returncode, 0, report.stderr)
        self.assertEqual(len(json.loads(report.stdout)["time"]["budget_extensions"]), 1)

    def test_continuation_inherits_old_counts_and_only_adds_once(self):
        self.assertEqual(self.cli("resume", self.run_id, "--add-seconds", "600", "--add-calls", "2").returncode, 1)
        old = self.state()
        old["deadline"] = time.time() - 100
        old["stages"]["P1"]["deadline"] = time.time() - 100
        old["nodes"]["retired"] = new_node({"file": "retired.yaml", "parent": "root"})
        old["nodes"]["retired"]["status"] = "removed"
        old["nodes"]["retired"]["stage"].update(calls=5, revisions=5, adjudications=1, elapsed_seconds=42)
        self.store.save(old)
        before = (self.store.run_dir(self.run_id) / "state.json").read_bytes()
        command = ["continue", self.run_id, "--checkpoint", old["checkpoint"], "--add-seconds", "600", "--add-calls", "2"]
        result = self.cli(*command)
        self.assertEqual(result.returncode, 0, result.stderr)
        successor = json.loads(result.stdout)["run_id"]
        current = self.store.load(successor)
        self.assertEqual((current["status"], current["stage"], current["role"]), ("queued", "P1", "worker"))
        self.assertTrue(all(scope["approval"] is None for scope in current["stages"].values()))
        self.assertEqual(current["stages"]["P1"]["calls"], old["stages"]["P1"]["calls"])
        self.assertEqual(current["attempt_usage"], old["attempt_usage"])
        self.assertEqual(current["budget_extensions"][:-1], old["budget_extensions"])
        self.assertEqual(len(current["budget_extensions"]), 2)
        self.assertEqual(current["nodes"]["retired"]["stage"]["adjudications"], 1)
        self.assertEqual(current["nodes"]["retired"]["stage"]["calls"], 5)
        result = self.cli(*command)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["run_id"], successor)
        self.assertEqual(self.store.load(successor), current)
        self.assertEqual((self.store.run_dir(self.run_id) / "state.json").read_bytes(), before)
        self.assertEqual(self.cli("resume", self.run_id).returncode, 2)
        self.assertEqual(read_json(self.store.run_dir(successor) / "inputs.lock.json")["source"]["files"],
                         read_json(self.store.run_dir(self.run_id) / "inputs.lock.json")["source"]["files"])


if __name__ == "__main__":
    unittest.main()
