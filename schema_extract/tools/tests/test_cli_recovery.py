"""Real CLI/store boundaries with tiny local inputs and an executable EX_TEMPFAIL stub."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from schema_extract.tools.common import ExtractError, read_json, rules_root, source_root
from schema_extract.tools.compatibility import PRESERVED, build_report, verify_report, verify_review
from schema_extract.tools.inputs import SourceTree
from schema_extract.tools.monitor import reserve_launch
from schema_extract.tools.snapshots import freeze_inputs
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

    def test_compatibility_report_binds_complete_target_and_independent_review(self):
        old = self.state()
        lock = read_json(self.store.run_dir(self.run_id) / "inputs.lock.json")
        target = self.root / "target-rules"
        shutil.copytree(rules_root(lock), target)
        runtime = target / "schema_extract/tools/agent.py"
        runtime.chmod(0o644)
        runtime.write_text(runtime.read_text() + "\n# Isolated runtime maintenance fixture.\n")
        report = build_report(self.store, self.run_id, old["checkpoint"], target)
        self.assertEqual(report, build_report(self.store, self.run_id, old["checkpoint"], target))
        self.assertEqual(report["assessment"]["outcome"], "insufficient_information")
        change = report["changes"][0]
        self.assertEqual([item["path"] for item in report["changes"]], ["schema_extract/tools/agent.py"])
        review = {"format": "schema_extract.compatibility_review.v1", "report_id": report["id"],
                  "supervisor": {"role": "Supervisor", "identity": "synthetic-reviewer",
                                 "independent_review": True, "record": "isolated review fixture"},
                  "decision": "compatible_runtime", "reason": "Comment-only fixture retains all meanings.",
                  "preserves": dict.fromkeys(PRESERVED, True),
                  "changes": [{key: change[key] for key in ("path", "before_sha256", "after_sha256")} |
                              {"classification": "runtime_fix", "reason": "Only a Python comment was added.",
                               "evidence": ["Full unified diff has one comment addition."]}]}
        self.assertEqual(verify_review(report, review)["outcome"], "compatible_runtime")
        verify_report(self.store, report, target)
        frozen = freeze_inputs(self.root / "candidate-snapshots", SourceTree(source_root(lock)),
                               lock["source"]["bindings"], sorted(lock["source"]["files"]), old["config"],
                               rules_repo=target)
        verify_report(self.store, report, target_lock=frozen)
        command = self.cli("compatibility-report", self.run_id, "--checkpoint", old["checkpoint"],
                           "--target-repo", str(target))
        self.assertEqual(command.returncode, 0, command.stderr)
        self.assertEqual(json.loads(command.stdout)["report"], report)
        bad = deepcopy(review)
        bad["changes"][0]["after_sha256"] = "0" * 64
        with self.assertRaisesRegex(ExtractError, "digest mismatch"):
            verify_review(report, bad)
        bad = deepcopy(review)
        bad["changes"] = []
        with self.assertRaisesRegex(ExtractError, "missing, duplicate or extra"):
            verify_review(report, bad)
        bad = deepcopy(report)
        bad["target"]["files"][change["path"]] = "0" * 64
        with self.assertRaisesRegex(ExtractError, "digest or format"):
            verify_report(self.store, bad, target)
        # Simultaneous policy, normative and undeclared runtime changes cannot
        # reuse the report for the single comment change.
        readme = target / "schema_extract/README.md"
        readme.chmod(0o644)
        readme.write_text(readme.read_text() + "\nFixture policy changes acceptance requirements.\n")
        spec = target / "schema_docs/0.common.md"
        spec.chmod(0o644)
        spec.write_text(spec.read_text() + "\nFixture changes a field meaning.\n")
        extra = target / "schema_extract/tools/additional_fixture.py"
        extra.parent.chmod(0o755)
        extra.write_text("# Additional actual target file.\n")
        with self.assertRaisesRegex(ExtractError, "no longer matches"):
            verify_report(self.store, report, target)
        expanded = build_report(self.store, self.run_id, old["checkpoint"], target)
        bad = deepcopy(review)
        bad["report_id"] = expanded["id"]
        with self.assertRaisesRegex(ExtractError, "missing, duplicate or extra"):
            verify_review(expanded, bad)
        bad["changes"] = [{key: item[key] for key in ("path", "before_sha256", "after_sha256")} |
                          {"classification": "policy", "reason": "Changes acceptance policy.",
                           "evidence": ["Complete fixture diff."]} for item in expanded["changes"]]
        with self.assertRaisesRegex(ExtractError, "cannot override"):
            verify_review(expanded, bad)
        bad["decision"] = "full_re_review"
        self.assertEqual(verify_review(expanded, bad)["semantic_review"]["suggested_start"], "P1")
        self.assertEqual(self.state(), old)
        self.assertEqual(read_json(self.store.run_dir(self.run_id) / "inputs.lock.json"), lock)

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

    def test_rejection_records_the_registered_attempt_not_the_bad_nonce(self):
        self.stub.write_text(f"#!{sys.executable}\n" + '''import json, sys
from pathlib import Path
if '--version' in sys.argv:
    print('controlled-stub')
    sys.exit(0)
sys.stdin.read()
destination = Path(sys.argv[sys.argv.index('--output-last-message') + 1])
destination.write_text(json.dumps({'format': 'schema_extract.proposal.v1', 'attempt_id': 'wrong-invocation'}))
print(json.dumps({'type': 'thread.started', 'thread_id': '00000000-0000-4000-8000-000000000001'}))
print(json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 1, 'cached_input_tokens': 0, 'output_tokens': 1}}))
''')
        result = self.cli("resume", self.run_id)
        self.assertEqual(result.returncode, 1, result.stderr)
        state = self.state()
        attempt_id = state["attempts"][-1]
        self.assertEqual(state["reason"]["code"], "proposal_rejected")
        self.assertEqual(state["reason"]["attempt_id"], attempt_id)
        rejected = read_json(self.store.run_dir(self.run_id) / "attempts" / attempt_id / "rejected.json")
        self.assertEqual(rejected["attempt_id"], attempt_id)

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
        command = ["continue", self.run_id, "--full-restart", "--reason", "Explicit full re-extraction fixture",
                   "--checkpoint", old["checkpoint"], "--add-seconds", "600", "--add-calls", "2"]
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
