"""Actual offline CLI exits and private sessions underpin current recovery."""

from copy import deepcopy
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from schema_extract.tools import monitor
from schema_extract.tools.audit import _Evidence, _verify_attempt, usage_summary, usage_totals
from schema_extract.tools.common import ExtractError, digest, manifest, read_json, sha256, write_json
from schema_extract.tools.recovery import (account, confirm_termination, handoff_session,
                                           recover_result, termination_observed, verified_result)
from schema_extract.tools.stages import new_scope
from schema_extract.tools.tests import test_backends_runtime


class CurrentRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_backends_runtime.BackendRuntimeTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def attempt(self, **options):
        value = self.fixture.attempt(**options)
        value.update(session_group=value["stage"] + ":root:normal", input_fingerprint="fixed")
        write_json(Path(value["directory"]) / "attempt.json", value)
        return value

    def records(self, directory):
        return {path.name: sha256(path.read_bytes()) for path in directory.iterdir()
                if path.is_file() and not path.is_symlink()}

    def state(self, attempt, session=None):
        return {"run_id": attempt["run_id"], "checkpoint": "safe-checkpoint", "stage": "P1",
                "input_fingerprint": "fixed", "sessions": {attempt["session_group"]: {"worker": session, "reviewer": None}},
                "stages": {"P1": {**new_scope(), "calls": 3, "adjudications": 1, "signoffs": 1}},
                "session_handoffs": [], "attempt_usage": {}, "active_attempts": {}, "unconfirmed_attempts": {},
                "unsettled_attempts": {}, "accepted_receipts": {}}

    def test_native_result_recovery_requires_real_exit_nonce_schema(self):
        for backend in ("claude", "codex"):
            with self.subTest(backend=backend):
                attempt = self.attempt(backend=backend)
                result = self.fixture.invoke(attempt)
                self.assertEqual(result["outcome"], "completed", result)
                directory = Path(attempt["directory"])
                if backend == "codex":
                    observation = read_json(directory / "usage-observation.json")
                    self.assertEqual(observation["session_scope"], attempt["session_scope"])
                    self.assertNotIn("native_scope_identity_conflict", observation["issues"])
                termination = confirm_termination(attempt, result)
                self.assertTrue(termination["confirmed"])
                before = self.records(directory)
                recovered = verified_result(attempt, result, termination)
                self.assertEqual(self.records(directory), before)
                self.assertEqual(recovered["outcome"], "completed")
                self.assertEqual(recover_result(attempt, result, termination)["session_status"], "ready")
                if backend == "claude":
                    from schema_extract.tools.audit import parse_events, result_usage
                    pending = {**result, "outcome": "termination_unconfirmed", "termination": {"confirmed": False}}
                    pending["usage"] = result_usage(parse_events(directory / "events.jsonl", backend=backend),
                                                    outcome=pending["outcome"], termination=pending["termination"])
                    write_json(directory / "result.json", pending)
                    notes = []
                    row = {"directory": str(directory), "origin_run_id": attempt["run_id"], "attempt_id": attempt["attempt_id"],
                           "usage": usage_summary(pending["usage"])}
                    files = _Evidence(self.fixture.root, False)
                    try:
                        _verify_attempt(object(), {"input_fingerprint": "fixed"}, row, files, notes)
                    finally:
                        files.close()
                    self.assertTrue(any(item.get("recovery_outcome") == "interrupted" for item in notes))
                    write_json(directory / "result.json", result)
                for name, mutate in (
                    ("exit.json", lambda value: value.update(exit_code=75)),
                    ("response.schema.json", lambda value: value.update(required=[])),
                ):
                    path, original = directory / name, (directory / name).read_bytes()
                    value = read_json(path)
                    mutate(value)
                    write_json(path, value)
                    try:
                        with self.assertRaises(ExtractError):
                            verified_result(attempt, result, termination)
                    finally:
                        path.write_bytes(original)
                truncated = directory / "events.jsonl"
                original = truncated.read_bytes()
                truncated.write_bytes(original.rstrip(b"\n"))
                with self.assertRaises(ExtractError):
                    verified_result(attempt, result, termination)
                truncated.write_bytes(original)
                # A complete stream cannot reverse a previously recorded timeout.
                interrupted = {**result, "outcome": "timeout"}
                self.assertEqual(recover_result(attempt, interrupted, termination)["outcome"], "timeout")
                self.assertEqual(read_json(Path(attempt["session_directory"]) / "session.json")["status"], "retired")

    def test_missing_result_retires_private_binding_only_after_confirmed_exit(self):
        attempt = self.attempt()
        result = self.fixture.invoke(attempt)
        self.assertEqual(result["outcome"], "completed")
        directory = Path(attempt["directory"])
        (directory / "result.json").unlink()
        # Emulate the interrupted Runner's uncommitted session settlement.
        session_path = Path(attempt["session_directory"]) / "session.json"
        session = read_json(session_path)
        session["status"] = "running"
        write_json(session_path, session)
        state = self.state(attempt, result["session_id"])
        before = deepcopy(state)
        with self.assertRaisesRegex(ExtractError, "termination"):
            handoff_session(state, attempt, None, "interrupted", termination={"confirmed": False})
        self.assertEqual(state, before)
        self.assertEqual(read_json(session_path)["status"], "running")
        termination = confirm_termination(attempt, None)
        self.assertTrue(termination["confirmed"])
        self.assertTrue(termination_observed(attempt, None))
        record = handoff_session(state, attempt, None, "interrupted", termination=termination)
        self.assertEqual(record["previous_session_id"], result["session_id"])
        self.assertEqual(read_json(session_path)["status"], "retired")
        self.assertIsNone(state["sessions"][attempt["session_group"]]["worker"])
        self.assertEqual(handoff_session(state, attempt, None, "repeat", termination=termination), record)
        for elapsed in (7, 7, 5, 9):
            account(state, attempt, elapsed, final=True)
        scope = state["stages"]["P1"]
        self.assertEqual((scope["elapsed_seconds"], scope["calls"], scope["adjudications"], scope["signoffs"]), (9, 3, 1, 1))
        self.assertIsNone(state["attempt_usage"][attempt["attempt_id"]]["usage"]["additive_tokens"]["input_tokens"])
        following = self.attempt()
        next_result = self.fixture.invoke(following)
        self.assertEqual(next_result["outcome"], "completed")
        self.assertNotEqual(next_result["session_id"], result["session_id"])

    def test_receipt_repair_resumes_only_while_native_execution_remains_verified(self):
        self.fixture.schema["required"].append("missing_proposal_field")
        first = self.attempt()
        result = self.fixture.invoke(first)
        self.assertEqual(result["outcome"], "receipt_error")
        self.assertTrue(result["native_status"]["success"])
        self.assertTrue(result["response_validation"]["nonce_valid"])
        termination = confirm_termination(first, result)
        self.assertEqual(recover_result(first, result, termination)["session_status"], "ready")
        following = self.attempt(session_id=result["session_id"])
        resumed = self.fixture.invoke(following)
        self.assertEqual((resumed["outcome"], resumed["session_id"]), ("receipt_error", result["session_id"]))
        self.assertEqual(resumed["response"]["sticky"], first["attempt_id"])
        directory = Path(following["directory"])
        termination = confirm_termination(following, resumed)
        recovered = recover_result(following, {**resumed, "outcome": "timeout"}, termination)
        self.assertEqual((recovered["outcome"], recovered["session_status"]), ("timeout", "retired"))
        session = read_json(Path(following["session_directory"]) / "session.json")
        self.assertEqual(session["status"], "retired")
        self.assertIsNone(session["session_id"])
        with self.assertRaisesRegex(ExtractError, "exact registered"):
            self.fixture.invoke(self.attempt(session_id=result["session_id"]))
    def test_lost_gate_without_closed_descendant_evidence_stays_fenced(self):
        from schema_extract.tools.agent import read_result
        attempt = self.attempt()
        result = self.fixture.invoke(attempt)
        directory = Path(attempt["directory"])
        self.assertTrue(confirm_termination(attempt, result)["confirmed"])
        (directory / "cli-exit.json").unlink()
        # Prior exit/result/termination claims cannot replace actual gate closure.
        self.assertFalse(termination_observed(attempt, result))
        replay = read_result(attempt, exit_code=0, termination={"confirmed": True})
        self.assertEqual(replay["outcome"], "termination_unconfirmed")
        with self.assertRaisesRegex(ExtractError, "descendant closure"):
            verified_result(attempt, result, {"confirmed": True})
        identity = read_json(directory / "process.json")["identity"]
        forced = {"confirmed": True, "survivors": [],
                  "actions": [{"pid": identity["pid"], "signal": "SIGKILL"}]}
        with patch("schema_extract.tools.recovery.terminate_owned", return_value=forced):
            confirmed = confirm_termination(attempt, result)
        self.assertFalse(confirmed["confirmed"])
        self.assertIn("descendant closure", confirmed["reason"])
        files = _Evidence(self.fixture.root, False)
        try:
            row = dict(directory=str(directory), origin_run_id=attempt["run_id"], attempt_id=attempt["attempt_id"],
                       usage=usage_summary(result["usage"]))
            with self.assertRaisesRegex(ExtractError, "descendant closure"):
                _verify_attempt(object(), {"input_fingerprint": "fixed"}, row, files, [])
        finally:
            files.close()
        (directory / "process.json").unlink()
        self.assertFalse(termination_observed(attempt, result))
        self.assertFalse(confirm_termination(attempt, result)["confirmed"])
        with self.assertRaisesRegex(ExtractError, "descendant closure"):
            verified_result(attempt, result, {"confirmed": True})

    def test_setup_failure_audit_preserves_missing_stream_and_rejects_forged_proof(self):
        from schema_extract.tools.audit import _archive, _archive_manifest
        import tempfile
        attempt = self.attempt()
        directory = Path(attempt["directory"])
        with patch("schema_extract.tools.agent.subprocess.check_output", side_effect=subprocess.CalledProcessError(75, ["fake-cli", "--version"])):
            result = self.fixture.invoke(attempt)
        termination = confirm_termination(attempt, result)
        self.assertEqual(verified_result(attempt, result, termination)["outcome"], "cli_error")
        before = self.records(directory)
        row = dict(directory=str(directory), origin_run_id=attempt["run_id"], attempt_id=attempt["attempt_id"],
                   usage=usage_summary(result["usage"]))
        notes, files = [], _Evidence(self.fixture.root, True)
        try:
            _verify_attempt(object(), {"input_fingerprint": "fixed"}, row, files, notes)
            with tempfile.TemporaryDirectory() as output:
                archive = Path(output) / "evidence.tar.gz"
                archived = _archive(archive, files, {"attempt_id": attempt["attempt_id"]},
                                    {"status": "partial", "unavailable": notes, "checkpoint_ids": []})
                self.assertEqual(archived["status"], "archived")
                entries = _archive_manifest(archive)["entries"]
                self.assertTrue({"attempt.json", "invocation.json", "result.json", "termination.json"} <=
                                {Path(entry["name"]).name for entry in entries})
                self.assertFalse(any(entry["name"].endswith("events.jsonl") for entry in entries))
        finally:
            files.close()
        self.assertTrue(any("no provider stream; usage unknown" in item["reason"] for item in notes))
        self.assertIsNone(row["usage"]["additive_tokens"]["input_tokens"])
        self.assertEqual(self.records(directory), before)
        invocation = read_json(directory / "invocation.json")
        for mode in ("launch", "event_link", "identity", "error", "termination", "survivor", "result_identity"):
            with self.subTest(mode=mode):
                if mode == "launch":
                    write_json(directory / "launch.json", {"status": "released"})
                elif mode == "event_link":
                    (directory / "events.jsonl").symlink_to(directory / "absent")
                elif mode == "identity":
                    write_json(directory / "invocation.json", {**invocation, "attempt_sha256": "other"})
                elif mode == "error":
                    write_json(directory / "invocation.json", {key: value for key, value in invocation.items() if key != "error"})
                elif mode == "survivor":
                    write_json(directory / "result.json", {**result, "termination": {**result["termination"], "survivors": [{"pid": 1}]}})
                elif mode == "result_identity":
                    write_json(directory / "result.json", {**result, "session_scope": {**result["session_scope"], "role": "reviewer"}})
                else:
                    write_json(directory / "result.json", {**result, "termination": {"confirmed": False, "reason": "not_started"}})
                files = _Evidence(self.fixture.root, False)
                try:
                    with self.assertRaises(ExtractError):
                        _verify_attempt(object(), {"input_fingerprint": "fixed"}, row, files, [])
                finally:
                    files.close()
                    (directory / "launch.json").unlink(missing_ok=True)
                    (directory / "events.jsonl").unlink(missing_ok=True)
                    write_json(directory / "invocation.json", invocation)
                    write_json(directory / "result.json", result)

    def test_setup_failure_keeps_the_previous_ready_session_and_usage_is_read_only(self):
        first = self.attempt()
        result = self.fixture.invoke(first)
        self.assertEqual(result["outcome"], "completed")
        following = self.attempt(session_id=result["session_id"])
        session_path = Path(first["session_directory"]) / "session.json"
        before = session_path.read_bytes()
        with patch("schema_extract.tools.agent.subprocess.check_output", side_effect=subprocess.CalledProcessError(75, ["fake-cli", "--version"])):
            failed = self.fixture.invoke(following)
        self.assertEqual(failed["outcome"], "cli_error")
        self.assertEqual(session_path.read_bytes(), before)
        state = self.state(following, result["session_id"])
        termination = confirm_termination(following, failed)
        self.assertTrue(termination["confirmed"])
        recovered = recover_result(following, failed, termination)
        self.assertEqual(recovered["session_status"], "not_started")
        self.assertEqual(handoff_session(state, following, failed, "setup failure", termination=termination)["status"], "not_started")
        self.assertEqual(state["sessions"][following["session_group"]]["worker"], result["session_id"])
        self.assertEqual(session_path.read_bytes(), before)
        account(state, following, 4, result=failed, final=True)
        totals = usage_totals([{**entry, "attempt_id": identity} for identity, entry in state["attempt_usage"].items()])
        self.assertEqual(totals["attempt_count"], 1)
        self.assertIsNone(totals["by_backend_and_scope"][0]["tokens"]["input_tokens"]["known_subtotal"])
        # The persisted stream parser is sufficient; observing Claude never
        # enters the Codex native-store observation helpers.
        with patch("schema_extract.tools.observation.begin_usage_observation", side_effect=AssertionError("Codex observation forbidden")):
            before = self.records(Path(first["directory"]))
            row = {"directory": first["directory"], "origin_run_id": first["run_id"], "attempt_id": first["attempt_id"],
                   "usage": usage_summary(result["usage"])}
            files = _Evidence(self.fixture.root, False)
            try:
                _verify_attempt(object(), {"input_fingerprint": "fixed"}, row, files, [])
            finally:
                files.close()
            self.assertEqual(self.records(Path(first["directory"])), before)
            class StoreView:
                def run_dir(self, run_id):
                    return self_root
            self_root = self.fixture.root
            observed = monitor._attempt_observation(StoreView(), {"run_id": first["run_id"]}, first["attempt_id"], {}, 1)
            self.assertEqual(observed["backend"], "claude")
            self.assertEqual(observed["usage"]["session_reported_cost_usd"], result["usage"]["session_reported_cost_usd"])


class CurrentScopeRecoveryTests(unittest.TestCase):
    def test_actual_transient_exit_recovers_only_its_scope_without_new_credit(self):
        import sys
        from schema_extract.tools.agent import AgentCLI
        from schema_extract.tools.runner import DEFAULT_CONFIG, _receive, _consume_scope_retries
        from schema_extract.tools.snapshots import freeze_inputs
        from schema_extract.tools.store import Store
        from schema_extract.tools.stages import effective_nodes
        from schema_extract.tools.prompts import build_prompt
        from schema_extract.tools.receipts import proposal_schema
        from schema_extract.tools.tests import test_stage_flow
        f = test_stage_flow.StageFlowTests(methodName="runTest")
        f.setUp()
        self.addCleanup(f.doCleanups)
        stub = f.directory / "temporary-cli"
        stub.write_text("#!" + sys.executable + "\nimport sys\nif '--version' in sys.argv:\n print('offline-exit-75')\n sys.exit(0)\nsys.exit(75)\n")
        stub.chmod(0o755)
        config = {**DEFAULT_CONFIG, "agent_binary": str(stub), "adjudication_seconds": 30, "retry_limit": 2}
        f.store = Store(f.directory / "state")
        f.store.initialize()
        f.lock = freeze_inputs(f.store.root, f.source, f.bindings, ["kernel.py"], config)
        f.state = f.store.create(f.lock, f.directory / "output")
        selected = [{"id": node, "instance": "kernel.py::copy(instance=" + node + ")"}
                    for node in ("root", "failed", "sibling")]
        f.backend = test_stage_flow.FixtureBackend(f.lock, selected,
                    [{"id": node, "wait_for": []} for node in ("failed", "sibling")])
        f.through_root()
        attempt = f.prepare(node="failed", role="worker")
        # The stage backend does run here, but it is an offline EX_TEMPFAIL
        # executable in the real invocation directory. No model request is made.
        f.state["unsettled_attempts"][attempt["attempt_id"]] = "dispatched"
        f.store.save(f.state)
        result = AgentCLI(backend="claude", executable=str(stub), model=config["model"]).run(
            attempt=attempt, prompt=build_prompt(f.state, f.lock, attempt),
            schema=proposal_schema(attempt, source_files=f.lock["source"]["files"]), timeout=15)
        self.assertEqual((result["outcome"], result["exit_code"]), ("cli_error", 75))
        account(f.state, attempt, result["elapsed_seconds"], result=result)
        f.state = _receive(f.store, f.state, f.lock, attempt, result, None)
        f.state["unsettled_attempts"].pop(attempt["attempt_id"])
        account(f.state, attempt, result["elapsed_seconds"], result=result, final=True)
        f.store.save(f.state)
        self.assertEqual(f.state["nodes"]["failed"]["status"], "cli_error", f.state["nodes"]["failed"].get("reason"))
        monitor.configure_auto_restart(f.store, [f.state["run_id"]], 1)
        with patch.object(monitor, "start_background", return_value={"status": "deferred"}):
            request = monitor.request_restart(f.store, f.state["run_id"], "P5:failed", test_stage_flow.REPO)
            self.assertEqual(monitor.request_restart(f.store, f.state["run_id"], "P5:failed", test_stage_flow.REPO)["id"], request["id"])
        scope_before = deepcopy(f.state["nodes"]["failed"]["stage"])
        f.step(node="sibling", role="worker")
        f.step(node="sibling", role="reviewer")
        evidence = monitor.restart_evidence(f.store, f.state, "P5:failed")
        self.assertEqual(evidence["failure_id"], request["failure_id"])
        self.assertIn("sibling", effective_nodes(f.state))
        # Make the already authorized retry due; time changes no budget or
        # consumed-call accounting. The Runner remains the sole state writer.
        with patch("schema_extract.tools.runner.time.time", return_value=request["next_due_at"] + 1):
            with f.store.writer(f.state["run_id"]):
                f.state, _ = _consume_scope_retries(f.store, f.state, f.lock, {})
        self.assertEqual(f.state["nodes"]["failed"]["status"], "draft")
        scope_after = f.state["nodes"]["failed"]["stage"]
        for key in ("calls", "elapsed_seconds", "cli_failures", "deadline", "adjudications", "signoffs"):
            self.assertEqual(scope_after[key], scope_before[key], key)
        ledger = f.store.read_supervisor()
        self.assertEqual(ledger["retries"][request["id"]]["status"], "consumed")
        self.assertEqual(ledger["auto_restart"]["used"][f.state["run_id"]], 1)
        f.store.request_control(f.state["run_id"], "pause")
        with self.assertRaisesRegex(ExtractError, "manual control"):
            monitor.check_restart_authority(f.store.read_supervisor(), request, f.state)
        self.assertIn("sibling", effective_nodes(f.state))


if __name__ == "__main__":
    unittest.main()
