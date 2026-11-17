"""Real final publication with current stage gates and deterministic review input."""

from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from schema_extract.tools import cli
from schema_extract.tools.artifacts import check_complete, load_yaml, write_yaml
from schema_extract.tools.common import ExtractError, manifest, read_json, rules_root, write_json
from schema_extract.tools.publication import publish
from schema_extract.tools.runner import run
from schema_extract.tools.runtime import verify_execution
from schema_extract.tools.tests import test_stage_flow


class CurrentDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_stage_flow.StageFlowTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_real_complete_publication_conflicts_replay_p6_and_source_identity(self):
        f = self.fixture
        nodes = ("root", "hidden", "math", "schedule", "consumer")
        f.create(nodes, writing=[{"id": "hidden", "wait_for": []}, {"id": "math", "wait_for": ["hidden"]},
                                {"id": "schedule", "wait_for": ["root"]},
                                {"id": "consumer", "wait_for": ["math", "schedule"]}])
        output = Path(f.state["output"])
        output.mkdir()
        (output / ".gitkeep").touch()
        f.state = run(f.store, f.state["run_id"], transport=f.backend)
        self.assertEqual(f.state["status"], "ready", f.state.get("reason"))
        self.assertTrue(f.state["published"], f.state.get("publication"))
        raw, checkpoint = f.store.verify_checkpoint(f.state)
        delivered = manifest(output)
        self.assertEqual(set(delivered), {*checkpoint["files"], ".extraction/receipt.json"})
        self.assertFalse(any(name.endswith(".png") for name in delivered))
        proof = read_json(output / ".extraction/receipt.json")
        self.assertEqual(proof["files"], checkpoint["files"])
        self.assertEqual(proof["checked_nodes"], sorted(nodes))
        self.assertEqual(proof["approval"]["attempt_id"], f.state["stages"]["P6"]["approval"]["attempt_id"])
        self.assertEqual(set(proof["design_sources"]["consumer"]["source_manifest"]),
                         {"root.yaml", "math.yaml", "schedule.yaml", "consumer.yaml"})
        self.assertIn("hidden.yaml", proof["design_sources"]["consumer"]["resolution_manifest"])
        self.assertEqual(proof["design_sources"]["consumer"]["effective_identity"], f.store.source_identity(f.state, "consumer"))
        counts = {key: (item["calls"], item["elapsed_seconds"], item["adjudications"], item["signoffs"])
                  for key, item in f.state["stages"].items()}
        with f.store.writer(f.state["run_id"]):
            f.state = publish(f.store, f.state, output)
        self.assertEqual(manifest(output), delivered)
        self.assertEqual({key: (item["calls"], item["elapsed_seconds"], item["adjudications"], item["signoffs"])
                          for key, item in f.state["stages"].items()}, counts)
        # Existing user content wins; a conflict never mutates the approved raw.
        conflict = f.directory / "conflict"
        conflict.mkdir()
        (conflict / "keep.txt").write_text("user data")
        with f.store.writer(f.state["run_id"]):
            f.state = publish(f.store, f.state, conflict, export=True)
        self.assertEqual(f.state["exports"][-1]["status"], "conflict")
        self.assertEqual((conflict / "keep.txt").read_text(), "user data")
        self.assertEqual(manifest(raw), checkpoint["files"])
        # A crash after atomic rename is recovered by the complete output
        # manifest; the same export is recorded only once.
        replay = f.directory / "replay"
        with f.store.writer(f.state["run_id"]):
            with patch.object(f.store, "save", side_effect=OSError("simulated state-write interruption")):
                with self.assertRaises(OSError):
                    publish(f.store, f.state, replay, export=True)
            self.assertEqual(manifest(replay), delivered)
            f.state = f.store.load(f.state["run_id"])
            f.state = publish(f.store, f.state, replay, export=True)
        self.assertEqual(sum(item["target"] == str(replay) for item in f.state["exports"]), 1)
        self.assertEqual(f.state["exports"][-1]["status"], "published")
        stable = deepcopy(f.state)
        for case in ("unsettled", "pending_signoff", "wrong_p6", "unaccepted"):
            with self.subTest(rejected=case):
                bad = deepcopy(stable)
                if case == "unsettled":
                    bad["unsettled_attempts"][proof["approval"]["attempt_id"]] = "dispatched"
                elif case == "pending_signoff":
                    bad["stages"]["P6"].update(status="signoff", mode="signoff")
                elif case == "wrong_p6":
                    bad["stages"]["P6"]["approval"] = deepcopy(bad["stages"]["P4"]["approval"])
                else:
                    bad["accepted_receipts"].pop(proof["approval"]["attempt_id"])
                with self.assertRaises(ExtractError):
                    publish(f.store, bad, f.directory / ("refused-" + case), export=True)
                self.assertFalse((f.directory / ("refused-" + case)).exists())
        # Publication checks the full selected set, including extra valid
        # designs; no intermediate-snapshot exclusion is available.
        candidate = f.directory / "candidate"
        shutil.copytree(raw, candidate)
        for path in candidate.rglob("*"):
            path.chmod(0o644)
        candidate.chmod(0o755)
        extra = load_yaml(candidate, "schedule.yaml", parser_root=rules_root(f.lock))
        extra["META_DATA"]["Dispatch"]["id"] = "extra"
        write_yaml(candidate / "extra.yaml", extra, parser_root=rules_root(f.lock))
        checked = check_complete(candidate, rules_root(f.lock), selected_designs=f.state["selected_designs"])
        self.assertNotEqual(checked["exit_code"], 0)
        self.assertIn("selected designs", " ".join(checked["diagnostics"]))
        source = raw / "math.yaml"
        original = source.read_bytes()
        source.chmod(0o644)
        source.write_bytes(original + b"\n")
        try:
            with self.assertRaisesRegex(ExtractError, "fixed contents changed"):
                publish(f.store, f.state, f.directory / "drift", export=True)
        finally:
            source.write_bytes(original)
            source.chmod(0o444)
        self.assertEqual(manifest(output), delivered)
        # A pause persisted during staging must win before the rename.
        from schema_extract.tools import publication
        sync = publication.sync_tree
        def paused(path):
            sync(path)
            f.store.request_control(f.state["run_id"], "pause")
        refused = f.directory / "paused-export"
        with f.store.writer(f.state["run_id"]), patch.object(publication, "sync_tree", side_effect=paused):
            f.state = publish(f.store, f.state, refused, export=True)
        self.assertFalse(refused.exists())
        self.assertEqual(f.state["exports"][-1]["status"], "controlled")
        self.assertEqual(f.state["control"]["action"], "pause")

    def test_native_evidence_archive_uses_preserved_originals_and_is_read_only(self):
        import json
        import os
        import sys
        from schema_extract.tools.agent import AgentCLI
        from schema_extract.tools.audit import audit_run
        from schema_extract.tools.runner import DEFAULT_CONFIG
        from schema_extract.tools.snapshots import freeze_inputs
        from schema_extract.tools.store import Store
        f = self.fixture
        binary = f.directory / "native-fixture"
        binary.write_text("#!" + sys.executable + "\n" + """import json, sys
from uuid import uuid4
if '--version' in sys.argv:
    print('offline-proposal-fixture')
    raise SystemExit()
value = json.load(sys.stdin)
session = sys.argv[sys.argv.index('--resume') + 1] if '--resume' in sys.argv else str(uuid4())
print(json.dumps({'type':'system', 'subtype':'init', 'session_id':session}))
print(json.dumps({'type':'assistant', 'session_id':session, 'message':{'model':'fixture-service', 'content':[]}}))
print(json.dumps({'type':'result','subtype':'success','is_error':False,'session_id':session,
    'structured_output':value['fixture_response'], 'result_index':0,
    'usage':{'input_tokens':4,'cache_read_input_tokens':2,'cache_creation_input_tokens':0,'output_tokens':3},
    'model_usage_since_last_result':{'fixture':{'inputTokens':4,'cacheReadInputTokens':2,
       'cacheCreationInputTokens':0,'outputTokens':3,'costUSD':0.0}}}))
""")
        binary.chmod(0o755)
        config = {**DEFAULT_CONFIG, "agent_binary": str(binary), "adjudication_seconds": 30}
        f.store = Store(f.directory / "state")
        f.store.initialize()
        f.lock = freeze_inputs(f.store.root, f.source, f.bindings, ["kernel.py"], config)
        f.state = f.store.create(f.lock, f.directory / "output")
        author = test_stage_flow.FixtureBackend(f.lock, [{"id":"root", "instance":"kernel.py::copy(instance=root)"}], [])
        class NativeFixture:
            def run(self, *, attempt, prompt, schema, timeout, cancel_event=None):
                if attempt["role"] == "worker":
                    author.write(attempt)
                response = author.proposal(attempt)
                # Fixture semantics are predetermined. The process, exact
                # proposal schema, native stream and scoped cwd are real.
                return AgentCLI(backend="claude", executable=str(binary), model=config["model"]).run(
                    attempt=attempt, prompt=json.dumps({"frozen_prompt":prompt, "fixture_response":response}),
                    schema=schema, timeout=timeout, cancel_event=cancel_event)
        private = f.directory / "private-configuration"
        private.mkdir()
        with patch.dict(os.environ, {"CLAUDE_CONFIG_DIR":str(private)}):
            f.state = run(f.store, f.state["run_id"], transport=NativeFixture())
        self.assertTrue(f.state["published"], f.state.get("reason"))
        before = (f.store.run_dir(f.state["run_id"]) / "state.json").read_bytes()
        archive = f.directory / "evidence.tar"
        report = audit_run(f.store, f.state["run_id"], verify=True, archive=archive)
        self.assertEqual(report["verification"]["status"], "verified", report["verification"])
        self.assertEqual(report["archive"]["status"], "archived", report["archive"])
        self.assertEqual(report["totals"]["provider_usage"]["attempt_count"], len(f.state["attempts"]))
        self.assertEqual((f.store.run_dir(f.state["run_id"]) / "state.json").read_bytes(), before)
        again = audit_run(f.store, f.state["run_id"], verify=True, archive=archive)
        self.assertTrue(again["archive"]["reused"], again["archive"])
        attempt_id = f.state["attempts"][0]
        original = f.store.run_dir(f.state["run_id"]) / "attempts" / attempt_id / "original/root.yaml"
        outside = f.directory / "linked-original.yaml"
        outside.write_bytes(original.read_bytes())
        original.parent.chmod(0o755)
        original.unlink()
        os.link(outside, original)
        read_bytes = Path.read_bytes
        def guarded_read(path):
            if path == original:
                raise AssertionError("audit read linked model bytes before artifact preflight")
            return read_bytes(path)
        with patch.object(Path, "read_bytes", guarded_read):
            rejected = audit_run(f.store, f.state["run_id"], attempt_id=attempt_id, verify=True)
        self.assertEqual(rejected["verification"]["status"], "incomplete")
        self.assertTrue(any("without links" in item["reason"] for item in rejected["verification"]["errors"]))
        self.assertEqual((f.store.run_dir(f.state["run_id"]) / "state.json").read_bytes(), before)

    def test_one_call_audit_archives_setup_failure_and_preflights_original_names(self):
        import json
        import os
        import subprocess
        import sys
        from schema_extract.tools.agent import AgentCLI
        from schema_extract.tools.audit import audit_run
        from schema_extract.tools.recovery import account
        from schema_extract.tools.receipts import proposal_schema
        from schema_extract.tools.runner import DEFAULT_CONFIG, _receive
        from schema_extract.tools.snapshots import freeze_inputs
        from schema_extract.tools.store import Store
        f = self.fixture
        binary = f.directory / "offline-one-call"
        binary.write_text("#!" + sys.executable + "\n" + """import json, sys
from uuid import uuid4
if '--version' in sys.argv:
    print('offline-one-call')
    raise SystemExit()
session = str(uuid4())
print(json.dumps({'type':'system', 'subtype':'init', 'session_id':session}))
print(json.dumps({'type':'result','subtype':'success','is_error':False,'session_id':session,
                  'structured_output':json.load(sys.stdin)}))
""")
        binary.chmod(0o755)
        config = {**DEFAULT_CONFIG, "agent_binary": str(binary), "adjudication_seconds": 30, "retry_limit": 2}
        f.store = Store(f.directory / "state")
        f.store.initialize()
        f.lock = freeze_inputs(f.store.root, f.source, f.bindings, ["kernel.py"], config)
        author = test_stage_flow.FixtureBackend(f.lock, [{"id": "root", "instance": "kernel.py::copy(instance=root)"}], [])
        private = f.directory / "private-configuration"
        private.mkdir()
        for mode in ("setup_failure", "completed"):
            with self.subTest(mode=mode), patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": str(private)}):
                f.state = f.store.create(f.lock, f.directory / mode)
                attempt = f.prepare()
                if mode == "completed":
                    author.write(attempt)
                def invoke():
                    return AgentCLI(backend="claude", executable=str(binary), model=config["model"]).run(
                        attempt=attempt, prompt=json.dumps(author.proposal(attempt)),
                        schema=proposal_schema(attempt, source_files=f.lock["source"]["files"]), timeout=15)
                if mode == "setup_failure":
                    with patch("schema_extract.tools.agent.subprocess.check_output", side_effect=subprocess.CalledProcessError(75, [str(binary), "--version"])):
                        result = invoke()
                    self.assertEqual(result["outcome"], "cli_error")
                else:
                    result = invoke()
                    self.assertEqual(result["outcome"], "completed")
                account(f.state, attempt, result["elapsed_seconds"], result=result)
                f.state = _receive(f.store, f.state, f.lock, attempt, result, None)
                f.state["unsettled_attempts"].pop(attempt["attempt_id"], None)
                account(f.state, attempt, result["elapsed_seconds"], result=result, final=True)
                f.store.save(f.state)
                before = (f.store.run_dir(f.state["run_id"]) / "state.json").read_bytes()
                report = audit_run(f.store, f.state["run_id"], attempt_id=attempt["attempt_id"], verify=True,
                                   archive=f.directory / (mode + ".tar.gz"))
                self.assertEqual(report["verification"]["errors"], [])
                self.assertEqual(report["archive"]["status"], "archived", report["archive"])
                self.assertEqual((f.store.run_dir(f.state["run_id"]) / "state.json").read_bytes(), before)
                if mode == "setup_failure":
                    self.assertEqual(report["verification"]["status"], "partial")
                    self.assertFalse((Path(attempt["directory"]) / "events.jsonl").exists())
                    self.assertIsNone(report["attempts"][0]["usage"]["additive_tokens"]["input_tokens"])
                else:
                    original = Path(attempt["directory"]) / "original"
                    original.chmod(0o755)
                    (original / "unexpected.yaml").write_text("untrusted bytes must not be read")
                    read_bytes = Path.read_bytes
                    def guarded_read(path):
                        if path.is_relative_to(original):
                            raise AssertionError("audit read original bytes before complete inventory preflight")
                        return read_bytes(path)
                    with patch.object(Path, "read_bytes", guarded_read):
                        rejected = audit_run(f.store, f.state["run_id"], attempt_id=attempt["attempt_id"], verify=True,
                                             archive=f.directory / "rejected.tar.gz")
                    self.assertEqual(rejected["archive"]["status"], "not_written")
                    self.assertTrue(any("unexpected unexpected.yaml" in item["reason"] for item in rejected["verification"]["errors"]))

    def test_fixed_execution_and_maintenance_cli_do_not_guess_or_migrate(self):
        f = self.fixture
        f.create()
        f.state["status"] = "running"
        f.store.save(f.state)
        self.assertEqual(verify_execution(f.store, f.state), f.lock)
        for changed in ("backend", "source", "executor", "legacy"):
            bad = deepcopy(f.state)
            if changed == "backend":
                bad["config"].pop("backend")
            elif changed == "source":
                bad["source_identity"]["ref"] = "other"
            elif changed == "executor":
                bad["executor"]["digest"] = "0" * 64
            else:
                bad["semantic_revision"] = "old"
            with self.assertRaises(ExtractError):
                verify_execution(f.store, bad)
        tasks = f.directory / "tasks.json"
        write_json(tasks, [{"stage": "P1", "node": "root", "action": "review", "fields": [],
                            "reason": "Check this run's current Contract evidence"}])
        before = (f.store.run_dir(f.state["run_id"]) / "state.json").read_bytes()
        captured = StringIO()
        with patch.object(cli, "_frozen_entry"), patch.object(cli.monitor, "start_background", return_value={"status": "launching"}) as launch:
            with redirect_stdout(captured):
                code = cli.main(["maintain", f.state["run_id"], "--issue-id", "contract-check", "--tasks", str(tasks),
                                 "--repo", str(test_stage_flow.REPO), "--state-root", str(f.store.root)])
        self.assertEqual(code, 0, captured.getvalue())
        self.assertEqual(launch.call_args.kwargs["mode"], "maintenance")
        self.assertEqual((f.store.run_dir(f.state["run_id"]) / "state.json").read_bytes(), before)
        self.assertEqual(f.store.read_supervisor()["maintenance_requests"][f.state["run_id"]]["status"], "requested")


if __name__ == "__main__":
    unittest.main()
