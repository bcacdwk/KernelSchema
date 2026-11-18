"""Actual offline CLI exits and private sessions underpin current recovery."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import subprocess
import threading
import time
import unittest
from unittest.mock import patch

from schema_extract.tools import monitor
from schema_extract.tools.audit import _Evidence, _verify_attempt, usage_summary, usage_totals
from schema_extract.tools.common import ExtractError, digest, manifest, read_json, sha256, write_json
from schema_extract.tools.recovery import (account, confirm_termination, handoff_session,
                                           recover_result, termination_observed, verified_result, complete_pending_result)
from schema_extract.tools.stages import new_scope
from schema_extract.tools.tests import test_backends_runtime


class CurrentRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_backends_runtime.BackendRuntimeTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def attempt(self, **options):
        value = self.fixture.attempt(**options)
        value.update(session_group=value["stage"] + ":root:normal", input_fingerprint="fixed", created_at=time.time())
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
                        _verify_attempt({"input_fingerprint": "fixed"}, row, files, notes)
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
                # A complete stream cannot reverse a previously recorded cancellation.
                interrupted = {**result, "outcome": "cancelled"}
                self.assertEqual(recover_result(attempt, interrupted, termination)["outcome"], "cancelled")
                self.assertEqual(read_json(Path(attempt["session_directory"]) / "session.json")["status"], "retired")

    def test_pending_cancel_settles_from_real_gate_and_finalized_records_stay_fixed(self):
        attempt = self.attempt()
        cancel = threading.Event()
        self.fixture.cancel_on_file(Path(attempt["directory"]) / "cli-tmp/heartbeat", cancel)
        result = self.fixture.invoke(attempt, action="daemon", cancel=cancel)
        state = self.state(attempt)
        state["control"] = {"requests": {"cancel": {"action": "cancel", "created_at": time.time()}}}
        state["active_attempts"] = {"P1": attempt["attempt_id"]}
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            termination = confirm_termination(attempt, result)
            if termination["confirmed"]:
                break
            time.sleep(0.02)
        self.assertTrue(termination["confirmed"])
        directory = Path(attempt["directory"])
        pending = {**result, "exit_code": None, "outcome": "termination_unconfirmed", "termination": {"confirmed": False}}
        write_json(directory / "result.json", pending)
        exit_record = read_json(directory / "exit.json")
        write_json(directory / "exit.json", {**exit_record, "exit_code": None, "termination": {"confirmed": False}})
        account(state, attempt, pending["elapsed_seconds"], result=pending)
        interrupted_state = deepcopy(state)
        settled = complete_pending_result(state, attempt, pending, termination)
        self.assertEqual(complete_pending_result(interrupted_state, attempt, settled, termination), settled)
        self.assertEqual(interrupted_state["attempt_usage"], state["attempt_usage"])
        self.assertEqual(settled["outcome"], "cancelled")
        self.assertEqual(settled["exit_code"], read_json(directory / "cli-exit.json")["exit_code"])
        self.assertEqual(settled["session_status"], "retired")
        self.assertEqual(read_json(directory / "exit.json")["exit_source"], "cli-exit.json")
        state["active_attempts"].clear()
        account(state, attempt, settled["elapsed_seconds"], result=settled, final=True)
        before = self.records(directory)
        self.assertEqual(complete_pending_result(state, attempt, settled, termination), settled)
        self.assertEqual(self.records(directory), before)
        self.assertTrue(state["attempt_usage"][attempt["attempt_id"]]["settled"])
        self.assertEqual(state["stages"]["P1"]["calls"], 3)

    def test_monitor_needs_current_process_and_gate_proof_for_recorded_exit(self):
        attempt = self.attempt()
        result = self.fixture.invoke(attempt)
        directory = Path(attempt["directory"])
        store = SimpleNamespace(run_dir=lambda _: directory.parent.parent)
        view = {"run_id": attempt["run_id"], "unsettled_attempts": {attempt["attempt_id"]: "dispatched"}}
        def status():
            return monitor._attempt_observation(store, view, attempt["attempt_id"], {}, time.time())["status"]
        self.assertTrue(result["termination"]["confirmed"])
        self.assertEqual(status(), "terminated")
        path = directory / "process.json"
        process = read_json(path)
        invalid = deepcopy(process)
        invalid["identity"].pop("start_token")
        write_json(path, invalid)
        self.assertEqual(status(), "unknown")
        write_json(path, process)
        path = directory / "cli-exit.json"
        write_json(path, {**read_json(path), "descendants_reaped": False})
        self.assertEqual(status(), "unknown")

    def test_scope_retry_closes_provisional_result_and_accounts_it_once(self):
        from schema_extract.tools.runner import _settle_scope_retry
        attempt = self.attempt()
        result = self.fixture.invoke(attempt)
        directory = Path(attempt["directory"])
        pending = {**result, "elapsed_seconds": 0.01, "exit_code": None,
                   "outcome": "termination_unconfirmed", "termination": {"confirmed": False}}
        write_json(directory / "result.json", pending)
        state = self.state(attempt)
        state.update(status="cli_error", active_attempts={"P1": attempt["attempt_id"]},
                     unsettled_attempts={attempt["attempt_id"]: "dispatched"})
        store = SimpleNamespace(run_dir=lambda _: directory.parent.parent, save=lambda _: None)
        request = {"id": "scope-retry", "attempt_id": attempt["attempt_id"], "scope": "P1"}
        def interrupt(_store, current, _lock, _attempt, recovered, _termination, _reason, **_options):
            self.assertTrue(recovered["termination"]["confirmed"])
            current["active_attempts"].clear()
            return current
        with patch("schema_extract.tools.runner._interrupted", side_effect=interrupt):
            _settle_scope_retry(store, state, {}, request)
        saved = read_json(directory / "result.json")
        usage = deepcopy(state["attempt_usage"])
        entry = usage[attempt["attempt_id"]]
        self.assertTrue(entry["settled"])
        self.assertNotEqual(entry["outcome"], "termination_unconfirmed")
        self.assertEqual(entry["evidence"]["result.json"], digest(saved))
        self.assertEqual(entry["elapsed_seconds"], saved["elapsed_seconds"])
        account(state, attempt, saved["elapsed_seconds"], result=saved, final=True)
        self.assertEqual(state["attempt_usage"], usage)

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
        recovered = recover_result(following, {**resumed, "outcome": "cancelled"}, termination)
        self.assertEqual((recovered["outcome"], recovered["session_status"]), ("cancelled", "retired"))
        session = read_json(Path(following["session_directory"]) / "session.json")
        self.assertEqual(session["status"], "retired")
        self.assertIsNone(session["session_id"])
        failed = self.fixture.invoke(self.attempt(session_id=result["session_id"]))
        self.assertEqual((failed["outcome"], failed["termination"]["reason"]), ("cli_error", "not_started"))
        self.assertIn("exact registered", failed["error"])
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
                _verify_attempt({"input_fingerprint": "fixed"}, row, files, [])
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
            _verify_attempt({"input_fingerprint": "fixed"}, row, files, notes)
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
                        _verify_attempt({"input_fingerprint": "fixed"}, row, files, [])
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
                _verify_attempt({"input_fingerprint": "fixed"}, row, files, [])
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

    def test_unstarted_gate_preserves_previous_session_even_when_result_is_missing(self):
        first = self.attempt()
        initial = self.fixture.invoke(first)
        self.assertEqual(initial["outcome"], "completed")
        session_id = initial["session_id"]
        session_path = Path(first["session_directory"]) / "session.json"
        executable = self.fixture.fake.read_bytes()
        for interrupted in ("before_release", "exec_failed"):
            for saved_result in (True, False):
                with self.subTest(interrupted=interrupted, saved_result=saved_result):
                    attempt = self.attempt(session_id=session_id)
                    previous_session = read_json(session_path)
                    cancel = threading.Event()
                    def interrupt(_):
                        if interrupted == "before_release":
                            cancel.set()
                        else:
                            self.fixture.fake.unlink(missing_ok=True)
                    try:
                        result = self.fixture.invoke(attempt, cancel=cancel, on_process=interrupt)
                    finally:
                        self.fixture.fake.write_bytes(executable)
                        self.fixture.fake.chmod(0o755)
                    self.assertEqual(result["outcome"], "cancelled" if interrupted == "before_release" else "cli_error")
                    self.assertTrue(result["termination"]["confirmed"])
                    self.assertEqual(result["cwd_after"], [])
                    self.assertEqual(read_json(Path(attempt["directory"]) / "launch.json")["status"], "not_started")
                    self.assertEqual(read_json(session_path)["session_id"], session_id)
                    if not saved_result:
                        (Path(attempt["directory"]) / "result.json").unlink()
                        interrupted_session = (previous_session if interrupted == "before_release" else
                                               {**read_json(session_path), "status": "running"})
                        write_json(session_path, interrupted_session)
                        result = None
                    termination = confirm_termination(attempt, result)
                    self.assertTrue(termination["confirmed"])
                    if saved_result:
                        self.assertEqual(recover_result(attempt, result, termination)["session_status"], "ready")
                    state = self.state(attempt, session_id)
                    handoff = handoff_session(state, attempt, result, "unstarted invocation", termination=termination)
                    self.assertEqual(handoff["status"], "not_started")
                    self.assertEqual(state["sessions"][attempt["session_group"]]["worker"], session_id)
                    self.assertEqual(state["session_handoffs"], [])
                    session = read_json(session_path)
                    self.assertEqual((session["status"], session["session_id"]), ("ready", session_id))
                    self.assertEqual(session["retired_sessions"], [])
        resumed = self.fixture.invoke(self.attempt(session_id=session_id))
        self.assertEqual((resumed["outcome"], resumed["session_id"]), ("completed", session_id))
        self.assertEqual(resumed["response"]["sticky"], first["attempt_id"])


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
        config = {**DEFAULT_CONFIG, "agent_binary": str(stub), "retry_limit": 2}
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
            schema=proposal_schema(attempt, source_files=f.lock["source"]["files"]))
        self.assertEqual((result["outcome"], result["exit_code"]), ("cli_error", 75))
        account(f.state, attempt, result["elapsed_seconds"], result=result)
        f.state = _receive(f.store, f.state, f.lock, attempt, result, None)
        f.state["unsettled_attempts"].pop(attempt["attempt_id"])
        account(f.state, attempt, result["elapsed_seconds"], result=result, final=True)
        f.store.save(f.state)
        self.assertEqual(f.state["nodes"]["failed"]["status"], "cli_error", f.state["nodes"]["failed"].get("reason"))
        with patch.object(monitor, "start_background", return_value={"status": "deferred"}):
            request = monitor.request_restart(f.store, f.state["run_id"], "P5:failed", test_stage_flow.REPO)
            self.assertEqual(monitor.request_restart(f.store, f.state["run_id"], "P5:failed", test_stage_flow.REPO)["id"], request["id"])
        scope_before = deepcopy(f.state["nodes"]["failed"]["stage"])
        f.step(node="sibling", role="worker")
        f.step(node="sibling", role="reviewer")
        evidence = monitor.restart_evidence(f.store, f.state, "P5:failed")
        self.assertEqual(evidence["failure_id"], request["failure_id"])
        self.assertIn("sibling", effective_nodes(f.state))
        # Make the already authorized retry due; time changes no call limits or
        # consumed-call accounting. The Runner remains the sole state writer.
        with patch("schema_extract.tools.runner.time.time", return_value=request["next_due_at"] + 1):
            with f.store.writer(f.state["run_id"]):
                f.state, _ = _consume_scope_retries(f.store, f.state, f.lock, {})
        self.assertEqual(f.state["nodes"]["failed"]["status"], "draft")
        scope_after = f.state["nodes"]["failed"]["stage"]
        for key in ("calls", "elapsed_seconds", "cli_failures", "adjudications", "signoffs"):
            self.assertEqual(scope_after[key], scope_before[key], key)
        ledger = f.store.read_supervisor()
        self.assertEqual(ledger["retries"][request["id"]]["status"], "consumed")

        # A resumed call whose gate cannot exec has not received its recovery
        # context. Keep that same handoff available for the next actual call.
        resumed = f.prepare(node="failed", role="worker")
        self.assertEqual(resumed["recovery_handoff"]["attempt_id"], attempt["attempt_id"])
        executable = stub.read_bytes()
        try:
            result = AgentCLI(backend="claude", executable=str(stub), model=config["model"]).run(
                attempt=resumed, prompt=build_prompt(f.state, f.lock, resumed),
                schema=proposal_schema(resumed, source_files=f.lock["source"]["files"]),
                on_process=lambda _: stub.unlink(missing_ok=True))
        finally:
            stub.write_bytes(executable)
            stub.chmod(0o755)
        self.assertEqual((result["outcome"], result["exit_code"], result["session_status"]), ("cli_error", 125, "empty"))
        account(f.state, resumed, result["elapsed_seconds"], result=result)
        f.state = _receive(f.store, f.state, f.lock, resumed, result, None)
        f.state["unsettled_attempts"].pop(resumed["attempt_id"])
        account(f.state, resumed, result["elapsed_seconds"], result=result, final=True)
        f.store.save(f.state)
        self.assertEqual(f.state["nodes"]["failed"]["stage"]["calls"], scope_before["calls"] + 1)
        self.assertEqual(len(f.state["session_handoffs"]), 1)
        self.assertIsNone(f.state["session_handoffs"][0]["next_attempt_id"])
        following = f.prepare(node="failed", role="worker")
        self.assertEqual(following["recovery_handoff"]["attempt_id"], attempt["attempt_id"])
        f.store.request_control(f.state["run_id"], "pause")
        with self.assertRaisesRegex(ExtractError, "manual control"):
            monitor.check_restart_authority(f.store.read_supervisor(), request, f.state)
        self.assertIn("sibling", effective_nodes(f.state))


class AuditReceiptRecoveryTests(unittest.TestCase):
    def test_rejected_receipts_archive_without_claiming_artifact_validation(self):
        import json
        import os
        import sys
        from schema_extract.tools.agent import AgentCLI
        from schema_extract.tools.audit import audit_run
        from schema_extract.tools.receipts import proposal_schema
        from schema_extract.tools.runner import DEFAULT_CONFIG, _receive
        from schema_extract.tools.snapshots import freeze_inputs
        from schema_extract.tools.store import Store
        from schema_extract.tools.tests import test_stage_flow

        f = test_stage_flow.StageFlowTests(methodName="runTest")
        f.setUp()
        self.addCleanup(f.doCleanups)
        binary = f.directory / "offline-receipt"
        binary.write_text("#!" + sys.executable + "\n" + """import json, sys
from uuid import uuid4
if '--version' in sys.argv:
    print('offline-receipt')
    raise SystemExit()
session = str(uuid4())
print(json.dumps({'type': 'system', 'subtype': 'init', 'session_id': session}))
print(json.dumps({'type': 'result', 'subtype': 'success', 'is_error': False,
                  'session_id': session, 'structured_output': json.load(sys.stdin)}))
""")
        binary.chmod(0o755)
        config = {**DEFAULT_CONFIG, "agent_binary": str(binary), "retry_limit": 2}
        f.store = Store(f.directory / "runtime")
        f.store.initialize()
        f.lock = freeze_inputs(f.store.root, f.source, f.bindings, ["kernel.py"], config)
        author = test_stage_flow.FixtureBackend(f.lock, [{"id": "root", "instance": "kernel.py::copy(instance=root)"}], [])
        private = f.directory / "private-configuration"
        private.mkdir()
        for mode in ("rejected", "receipt_repair", "accepted_missing_validation"):
            with self.subTest(mode=mode), patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": str(private)}):
                f.state = f.store.create(f.lock, f.directory / mode)
                attempt = f.prepare()
                author.write(attempt)
                response = author.proposal(attempt)
                if mode == "rejected":
                    response["framework_candidates"] = [{"id": "invalid-evidence", "scope": "extract_tools",
                        "summary": "The candidate has an invalid evidence index", "evidence": [len(response["source_evidence"])]}]
                elif mode == "receipt_repair":
                    response["summary"] = ""
                result = AgentCLI(backend="claude", executable=str(binary), model=config["model"]).run(
                    attempt=attempt, prompt=json.dumps(response),
                    schema=proposal_schema(attempt, source_files=f.lock["source"]["files"]))
                self.assertEqual(result["outcome"], "completed")
                account(f.state, attempt, result["elapsed_seconds"], result=result)
                f.state = _receive(f.store, f.state, f.lock, attempt, result, None)
                f.state["unsettled_attempts"].pop(attempt["attempt_id"], None)
                account(f.state, attempt, result["elapsed_seconds"], result=result, final=True)
                f.store.save(f.state)
                validation = Path(attempt["directory"]) / "validated.json"
                if mode == "accepted_missing_validation":
                    self.assertIn(attempt["attempt_id"], f.state["accepted_receipts"])
                    validation.unlink()
                else:
                    self.assertNotIn(attempt["attempt_id"], f.state["accepted_receipts"])
                    self.assertFalse(validation.exists())
                before = (f.store.run_dir(f.state["run_id"]) / "state.json").read_bytes()
                report = audit_run(f.store, f.state["run_id"], attempt_id=attempt["attempt_id"], verify=True,
                                   archive=f.directory / (mode + ".tar.gz"))
                if mode == "accepted_missing_validation":
                    self.assertEqual(report["verification"]["status"], "incomplete")
                    self.assertEqual(report["archive"]["status"], "not_written")
                else:
                    self.assertEqual(report["verification"]["errors"], [])
                    self.assertEqual(report["verification"]["status"], "partial")
                    self.assertEqual(report["archive"]["status"], "archived", report["archive"])
                    self.assertTrue(any(item.get("reason") == "receipt_not_accepted; artifact validation not completed"
                                        for item in report["verification"]["unavailable"]))
                    self.assertIsNone(report["attempts"][0]["receipt_sha256"])
                self.assertEqual((f.store.run_dir(f.state["run_id"]) / "state.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
