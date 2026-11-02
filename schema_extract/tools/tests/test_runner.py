"""Staged extraction, author continuation, recovery and publishing."""

from copy import deepcopy
import io
import json
from pathlib import Path
import time
from unittest.mock import Mock, patch

from ...runner import main
from ..codex_cli import CodexCLI, write_json
from ..execution import Runner, bound_review, eligible, merge_issues, publish
from ..framework import candidate_report
from ..planning import seed_inputs
from ..prompts import prompt_for
from ..protocol import validate_receipt
from ..records import StateError, read_json, run_check, schema_parts
from .support import FIXTURE, PROPOSAL, ExtractionCase


class RunnerTests(ExtractionCase):
    def test_notes_and_skipped_hardware_allow_reviewed_design_to_finish_without_repeating(self):
        import yaml
        owner = self
        class GapCLI:
            calls = []
            def run(self, **kwargs):
                kwargs["records"].mkdir(parents=True)
                state = owner.store.load(owner.state["key"])
                role, work = kwargs["role"], kwargs["cwd"]
                self.calls.append((role, state["stage"], kwargs["thread_id"]))
                if role == "worker":
                    if state["stage"] == "contract":
                        owner.draft(work, gap=True)
                        root = schema_parts(work, owner.baseline["snapshot"])["root"]
                        root.pop("design")
                        root["contract"].pop("hardware_specification")
                    else:
                        root = schema_parts(work, owner.baseline["snapshot"])["root"]
                        root["design"] = schema_parts(FIXTURE.parent)["root"]["design"]
                        root["meta_data"]["version"] += 1
                    (work / "root.yaml").write_text(yaml.safe_dump(root, sort_keys=False))
                    response = owner.receipt(state, status="needs_input", gap=True)
                    response["issues"][0]["blocks"] = ["contract", "node", "index"]
                else:
                    if state["review_scope"] == "final":
                        owner.assertNotIn("FINAL INDEX REVIEW", kwargs["prompt"])
                        owner.assertIn("complete substantive node content", kwargs["prompt"])
                    response = owner.review_receipt(state, work, status="complete", gap=True)
                    response["unsupported_claims"] = ["Only supplied synthetic source facts are represented."]
                    response["issues"][0]["blocks"] = []
                response["framework_candidates"] = [PROPOSAL, {"id": "incomplete"}, PROPOSAL]
                if role == "reviewer" and state["review_scope"] == "final":
                    response["framework_candidates"].append({**PROPOSAL, "evidence": ["New final review evidence."]})
                return {"outcome": "completed", "response": response,
                        "thread_id": "00000000-0000-0000-0000-000000000001", "elapsed_seconds": 1, "usage": {"input_tokens": 10}}
        fake = GapCLI()
        runner = Runner(self.store, self.plan, fake)
        state = runner.work(self.state["key"])
        self.assertEqual(state["status"], "ended")
        self.assertEqual(state["result"], "ready")
        self.assertTrue(state["eligible"])
        runner.work(self.state["key"])
        self.assertEqual(fake.calls, [("worker", "contract", None), ("reviewer", "contract", None),
                                      ("worker", "node", "00000000-0000-0000-0000-000000000001"), ("reviewer", "node", None)])
        self.assertEqual(set(state["reviews"]), {"contract", "final"})
        self.assertEqual(state["reviews"]["contract"]["receipt"]["continuation"], "proceed")
        self.assertEqual(state["check"]["exit_code"], 0)
        self.assertEqual(state["check"]["policy"], "skip-hardware")
        self.assertIn("design", schema_parts(self.store.checkpoint(state) / "artifacts")["root"])
        target = Path(publish(self.store, state))
        self.assertNotIn(PROPOSAL["id"], (target / ".extraction/receipt.json").read_text())
        self.assertEqual([issue["id"] for issue in state["issues"]], ["missing_material"])
        groups = candidate_report(self.store, self.plan)
        self.assertEqual([group["id"] for group in groups], [PROPOSAL["id"]])
        self.assertIsNone(groups[0]["registered"])
        reports = groups[0]["reports"]
        original, followup = reports
        self.assertEqual(original["candidate"], PROPOSAL)
        self.assertEqual(len(original["sources"]), 4)
        self.assertEqual({s["role"] for s in original["sources"]}, {"worker", "reviewer"})
        self.assertEqual({s["checkpoint"] for s in original["sources"]}, set(state["history"]))
        self.assertEqual(followup["candidate"]["evidence"], ["New final review evidence."])
        self.assertEqual(followup["sources"], [original["sources"][-1]])

    def test_fundamental_contract_block_is_confirmed_by_independent_review(self):
        owner = self
        class BlockedCLI:
            roles = []
            def run(self, **kwargs):
                self.roles.append(kwargs["role"])
                kwargs["records"].mkdir(parents=True)
                state = owner.store.load(owner.state["key"])
                if kwargs["role"] == "worker":
                    owner.draft(kwargs["cwd"], gap=True)
                    response = owner.receipt(state, status="needs_input", gap=True)
                else:
                    response = owner.review_receipt(state, kwargs["cwd"], status="needs_input", gap=True)
                return {"outcome": "completed", "response": response, "thread_id": "00000000-0000-0000-0000-000000000001",
                        "elapsed_seconds": 1, "usage": {"input_tokens": 10}}
        fake = BlockedCLI()
        state = Runner(self.store, self.plan, fake).work(self.state["key"])
        self.assertEqual(fake.roles, ["worker", "reviewer"])
        self.assertEqual(state["result"], "needs_input")
        self.assertEqual(state["last_review"]["receipt"]["continuation"], "needs_input")
        self.assertNotIn("contract", state["reviews"])

    def test_cancelled_cli_draft_never_gets_checkpoint(self):
        owner = self
        class CancelCLI:
            def run(self, **kwargs):
                kwargs["records"].mkdir(parents=True)
                owner.draft(kwargs["cwd"])
                return {"outcome": "cancelled", "elapsed_seconds": 0.5, "usage": {}, "thread_id": None}
        state = Runner(self.store, self.plan, CancelCLI()).work(self.state["key"])
        self.assertEqual(state["status"], "stopped")
        self.assertIsNone(state["checkpoint"])
        self.assertEqual(len(state["attempts"]), 1)

    def test_pause_while_waiting_for_review_does_not_record_an_unstarted_call(self):
        self.state = self.checkpoint()
        self.state["phase"] = "reviewer"
        self.store.save(self.state)
        cli = Mock()
        runner = Runner(self.store, self.plan, cli)
        runner.control.parent.mkdir(parents=True)
        slot = Mock()
        slot.acquire.side_effect = lambda: write_json(runner.control, {"mode": "pause"})
        runner.review_slot = slot
        state = runner.work(self.state["key"])
        cli.run.assert_not_called()
        self.assertEqual(state["status"], "stopped")
        self.assertEqual(state["attempts"][-1]["calls"], [])
        self.assertEqual(state["usage"]["calls"], 0)
        self.assertEqual(state["checkpoint"], self.state["checkpoint"])
        write_json(runner.control.parent / "plan.json", self.plan)
        self.assertEqual(seed_inputs(self.store, "test", "synthetic")[0]["key"], state["key"])
        runner.recover(state)
        self.assertEqual(self.store.load(state["key"])["status"], "queued")

    def test_missing_cli_preflight_is_recorded_and_explicitly_recoverable(self):
        self.state = self.checkpoint()
        runner = Runner(self.store, self.plan, CodexCLI(str(self.base / "missing-codex")))
        state = runner.work(self.state["key"])
        self.assertEqual(state["result"], "engineering_failure")
        records = Path(state["attempts"][-1]["calls"][-1])
        result = read_json(records / "result.json")
        self.assertEqual(result["outcome"], "preflight_error")
        self.assertTrue(result["termination"]["confirmed"])
        self.assertEqual(result["termination"]["reason"], "process_not_started")
        self.assertFalse((records / "process.json").exists())
        self.assertEqual(state["checkpoint"], self.state["checkpoint"])
        directory = self.store.root / "runs/test"
        directory.mkdir(parents=True)
        write_json(directory / "plan.json", self.plan)
        self.assertEqual(seed_inputs(self.store, "test", "synthetic")[0]["key"], state["key"])
        runner.recover(state)
        self.assertEqual(self.store.load(state["key"])["status"], "queued")

    def test_cancel_at_final_boundary_keeps_the_checkpoint_and_stops(self):
        self.state = self.checkpoint()
        self.state.update(phase="reviewer", stage="node", review_scope="final")
        self.state["reviews"]["contract"] = bound_review(self.state, self.receipt(), self.store.checkpoint(self.state))
        self.store.save(self.state)
        runner = Runner(self.store, self.plan, Mock())
        def completed(**kwargs):
            kwargs["records"].mkdir(parents=True)
            runner.cancel_event.set()
            return {"outcome": "completed", "response": self.review_receipt(self.store.load(self.state["key"]), kwargs["cwd"]),
                    "thread_id": "00000000-0000-0000-0000-000000000001", "elapsed_seconds": 0.1, "usage": {}}
        runner.cli.run.side_effect = completed
        state = runner.work(self.state["key"])
        self.assertEqual((state["status"], state["reason"]), ("stopped", "cancel"))
        self.assertEqual(state["checkpoint"], self.state["checkpoint"])
        self.assertIn("final", state["reviews"])
        self.assertFalse(state["eligible"])

    def test_repeated_bad_receipts_cannot_replace_the_receipt_only_artifact_boundary(self):
        calls = []
        def malformed(**kwargs):
            calls.append(kwargs["records"])
            kwargs["records"].mkdir(parents=True)
            if len(calls) == 1:
                self.draft(kwargs["cwd"])
            else:
                self.assertEqual(len(calls), 2)
                (kwargs["cwd"] / "dispatch.md").write_text("unauthorized receipt-only file change\n")
            response = self.receipt(self.store.load(self.state["key"]))
            response["fingerprint"] = "incorrect"
            return {"outcome": "completed", "response": response, "thread_id": "00000000-0000-0000-0000-000000000001",
                    "elapsed_seconds": 0.1, "usage": {}}
        state = Runner(self.store, self.plan, Mock(run=malformed)).work(self.state["key"])
        self.assertEqual(state["result"], "protocol_failure")
        self.assertIn("receipt-only correction modified artifacts", state["last_error"])
        self.assertIsNone(state["checkpoint"])

    def test_bad_receipt_pause_then_recover_can_rebuild_unaccepted_artifacts(self):
        runner = Runner(self.store, self.plan, Mock())
        runner.control.parent.mkdir(parents=True)
        calls = []
        def respond(**kwargs):
            calls.append(kwargs["cwd"])
            kwargs["records"].mkdir(parents=True)
            state = self.store.load(self.state["key"])
            if kwargs["role"] == "worker":
                self.assertNotIn("receipt_correction", state)
                self.assertNotIn("only repair the receipt", kwargs["prompt"])
                self.assertFalse((kwargs["cwd"] / "root.yaml").exists())
                self.draft(kwargs["cwd"])
                response = self.receipt(state)
                if len(calls) == 1:
                    response["fingerprint"] = "incorrect"
                    write_json(runner.control, {"mode": "pause"})
            else:
                response = self.review_receipt(state, kwargs["cwd"], status="needs_input", gap=True)
            return {"outcome": "completed", "response": response, "thread_id": "00000000-0000-0000-0000-000000000001",
                    "elapsed_seconds": 0.1, "usage": {}}
        runner.cli.run.side_effect = respond
        paused = runner.work(self.state["key"])
        self.assertEqual((paused["status"], paused["reason"]), ("stopped", "pause"))
        self.assertIsNone(paused["checkpoint"])
        self.assertIn("receipt_correction", paused)
        runner.recover(paused)
        runner.control.unlink()
        state = runner.work(self.state["key"])
        self.assertEqual(state["result"], "needs_input")
        self.assertIsNotNone(state["checkpoint"])
        self.assertNotEqual(calls[0], calls[1])

    def test_reviewer_malformed_receipt_does_not_hide_writes_or_missing_input(self):
        self.state = self.checkpoint()
        self.state["phase"] = "reviewer"
        self.store.save(self.state)
        work = self.store.checkpoint(self.state) / "artifacts"
        receipt = self.review_receipt(self.state, work, status="needs_input")
        with self.assertRaisesRegex(StateError, "review needs_input requires an issue"):
            validate_receipt(receipt, self.state, self.baseline, work, "reviewer")
        def malformed(**kwargs):
            kwargs["records"].mkdir(parents=True)
            response = self.review_receipt(self.store.load(self.state["key"]), kwargs["cwd"])
            response["source_evidence"][0]["quote"] = "incorrect"
            (kwargs["cwd"] / "dispatch.md").write_text("unauthorized reviewer change\n")
            return {"outcome": "completed", "response": response, "thread_id": "00000000-0000-0000-0000-000000000001",
                    "elapsed_seconds": 0.1, "usage": {}}
        state = Runner(self.store, self.plan, Mock(run=malformed)).work(self.state["key"])
        self.assertEqual(state["result"], "protocol_failure")
        self.assertIn("read-only reviewer changed artifacts", state["last_error"])
        self.assertEqual(state["checkpoint"], self.state["checkpoint"])

    def test_failed_review_transition_discards_its_candidate(self):
        state = self.checkpoint()
        state["phase"] = "reviewer"
        self.store.save(state)
        def respond(**kwargs):
            kwargs["records"].mkdir(parents=True)
            response = self.review_receipt(state, kwargs["cwd"])
            response["framework_candidates"] = [PROPOSAL]
            return {"outcome": "completed", "response": response, "thread_id": "synthetic",
                    "elapsed_seconds": 0.1, "usage": {}}
        runner = Runner(self.store, self.plan, Mock(run=respond))
        with patch.object(runner, "after_review", side_effect=StateError("rejected transition")):
            result = runner.work(state["key"])
        self.assertEqual(result["result"], "protocol_failure")
        self.assertEqual(candidate_report(self.store, self.plan), [])

    def test_receipt_path_aliases_still_require_bound_sources_and_exact_quotes(self):
        self.state = self.checkpoint()
        self.state["phase"] = "reviewer"
        work = self.store.checkpoint(self.state) / "artifacts"
        receipt = self.review_receipt(self.state, work)
        receipt.pop("framework_candidates")  # Receipts from older protocols remain readable.
        source = receipt["source_evidence"][0]["path"]
        prefix = self.baseline["repository_id"] + "@" + self.baseline["snapshot_commit"] + ":"
        for alias in (prefix + source, str(Path(self.baseline["snapshot"]) / source)):
            raw = deepcopy(receipt)
            raw["source_evidence"][0]["path"] = alias
            raw["artifact_evidence"][0]["path"] = str(work / raw["artifact_evidence"][0]["path"])
            normalized = validate_receipt(raw, self.state, self.baseline, work, "reviewer")
            self.assertEqual(normalized, {**receipt, "framework_candidates": []})
            self.assertEqual(raw["source_evidence"][0]["path"], alias)
        for bad in ("other@commit:" + source, prefix + "../" + source,
                    str(self.repo / source), prefix + source + "::symbol"):
            raw = deepcopy(receipt)
            raw["source_evidence"][0]["path"] = bad
            with self.assertRaises(StateError):
                validate_receipt(raw, self.state, self.baseline, work, "reviewer")
        raw = deepcopy(receipt)
        raw["source_evidence"][0].update(path=prefix + source, quote="not present in the cited lines")
        raw["artifact_evidence"][0]["quote"] = "another incorrect quote"
        with self.assertRaisesRegex(StateError, "evidence quote does not match") as rejection:
            validate_receipt(raw, self.state, self.baseline, work, "reviewer")
        self.assertIn("source_evidence[0]", str(rejection.exception))
        self.assertIn("artifact_evidence[0]", str(rejection.exception))

    def test_pause_after_review_does_not_reuse_author_session_on_recovery(self):
        from uuid import uuid4
        calls = []
        runner = Runner(self.store, self.plan, Mock())
        runner.control.parent.mkdir(parents=True)
        def respond(**kwargs):
            calls.append((kwargs["role"], kwargs["cwd"], kwargs["thread_id"]))
            kwargs["records"].mkdir(parents=True)
            state = self.store.load(self.state["key"])
            if kwargs["role"] == "worker":
                if state["stage"] == "contract":
                    self.draft(kwargs["cwd"])
                response = self.receipt(state)
            else:
                response = self.review_receipt(state, kwargs["cwd"])
                if state["review_scope"] == "contract":
                    write_json(runner.control, {"mode": "pause"})
            result = {"outcome": "completed", "response": response, "thread_id": str(uuid4()),
                      "elapsed_seconds": 0.1, "usage": {}, "termination": {"confirmed": True}}
            write_json(kwargs["records"] / "result.json", result)
            return result
        runner.cli.run.side_effect = respond
        paused = runner.work(self.state["key"])
        self.assertEqual(paused["status"], "stopped")
        self.assertIn("contract", paused["reviews"])
        old_author = calls[0][1]
        (old_author / "dispatch.md").write_text("unaccepted later edit\n")
        runner.recover(paused)
        runner.control.unlink()
        completed = runner.work(self.state["key"])
        self.assertEqual(completed["result"], "ready")
        self.assertIsNone(calls[2][2])
        self.assertNotEqual(calls[2][1], old_author)
        self.assertNotIn("unaccepted", (calls[2][1] / "dispatch.md").read_text())
        self.assertTrue(all(thread is None for role, work, thread in calls if role == "reviewer"))

    def test_receipt_only_repair_releases_file_guard_before_worker_continues(self):
        from uuid import uuid4
        calls = []
        def respond(**kwargs):
            calls.append(kwargs["thread_id"])
            kwargs["records"].mkdir(parents=True)
            state = self.store.load(self.state["key"])
            if kwargs["role"] == "reviewer":
                response = self.review_receipt(state, kwargs["cwd"], status="needs_input", gap=True)
            else:
                if len(calls) == 1:
                    self.draft(kwargs["cwd"])
                if len(calls) == 3:
                    root = kwargs["cwd"] / "root.yaml"
                    root.write_text(root.read_text().replace("version: 1", "version: 2").replace(
                        "synthetic validation fixture", "updated synthetic validation fixture"))
                response = self.receipt(state, status="continue" if len(calls) < 3 else "complete")
                if len(calls) == 1:
                    response["fingerprint"] = "wrong binding"
                elif len(calls) == 2:
                    self.assertIn("only repair the receipt", kwargs["prompt"])
                else:
                    self.assertNotIn("only repair the receipt", kwargs["prompt"])
            return {"outcome": "completed", "response": response,
                    "thread_id": kwargs["thread_id"] or str(uuid4()), "elapsed_seconds": 0.1, "usage": {}}
        state = Runner(self.store, self.plan, Mock(run=respond)).work(self.state["key"])
        self.assertEqual(state["result"], "needs_input")
        self.assertEqual(len(state["history"]), 2)
        self.assertEqual(calls[1], calls[2])
        self.assertNotIn("receipt_repair_files", state["attempts"][0])

    def test_continuation_ignores_all_legacy_limits(self):
        owner = self
        self.plan["config"].update(continuations=0, task_calls=1, task_tokens=1,
                                   call_seconds=1, task_seconds=1, batch_seconds=1, revisions=0)
        class ContinuingCLI:
            calls = []
            def run(self, **kwargs):
                from uuid import uuid4
                self.calls.append((kwargs["cwd"], kwargs["thread_id"]))
                kwargs["records"].mkdir(parents=True)
                state = owner.store.load(owner.state["key"])
                number = len(self.calls)
                if kwargs["role"] == "reviewer":
                    return {"outcome": "completed", "response": owner.review_receipt(state, kwargs["cwd"], status="needs_input", gap=True),
                            "thread_id": str(uuid4()), "elapsed_seconds": 1, "usage": {"input_tokens": 10}}
                owner.draft(kwargs["cwd"], version=number)
                root = kwargs["cwd"] / "root.yaml"
                root.write_text(root.read_text().replace("synthetic validation fixture", f"synthetic validation fixture revision {number}"))
                return {"outcome": "completed", "response": owner.receipt(state, status="continue" if number < 4 else "needs_input", gap=number == 4),
                        "thread_id": kwargs["thread_id"] or str(uuid4()), "elapsed_seconds": 1, "usage": {"input_tokens": 10}}
        fake = ContinuingCLI()
        state = Runner(self.store, self.plan, fake).work(self.state["key"])
        self.assertEqual(state["result"], "needs_input")
        self.assertEqual(len(fake.calls), 5)
        self.assertIsNone(fake.calls[0][1])
        self.assertIsNotNone(fake.calls[1][1])
        self.assertEqual(fake.calls[1][1], fake.calls[2][1])
        self.assertEqual(fake.calls[3][1], fake.calls[1][1])
        self.assertEqual(fake.calls[0][0], fake.calls[3][0])
        self.assertEqual(state["usage"]["calls"], 5)
        self.assertEqual(state["usage"]["elapsed_seconds"], 5)

    def test_extraction_explicitly_skips_hardware(self):
        import yaml
        work = self.base / "missing-hardware"
        self.draft(work)
        root = schema_parts(work, self.baseline["snapshot"])["root"]
        root["contract"].pop("hardware_specification")
        (work / "root.yaml").write_text(yaml.safe_dump(root, sort_keys=False))
        check = run_check(self.baseline, work, self.base)
        self.assertEqual(check["exit_code"], 0)
        self.assertEqual(check["policy"], "skip-hardware")
        self.assertIn("--skip-hardware", check["argv"])
        self.assertIn("Hardware skipped explicitly", check["stderr"])
        state = self.store.commit_checkpoint(self.state, work, self.receipt(), check)
        self.assertFalse(eligible(self.store, state))

    def test_status_reports_live_activity_without_dumping_or_mutating_history(self):
        from contextlib import redirect_stdout
        directory = self.store.root / "runs/test"
        directory.mkdir(parents=True)
        write_json(directory / "plan.json", self.plan)
        records = self.store.task_dir(self.state["key"]) / "attempts/a-0001/call-01"
        records.mkdir(parents=True)
        events = records / "events.jsonl"
        events.write_text('{"type":"thread.started","thread_id":"live-thread"}\n' + 'x' * 100000)
        self.state.update(status="running", active_process={"records": str(records),
                          "identity": {"pid": 123}, "started_at": time.time() - 30})
        self.state["usage"]["calls"] = 1
        self.state["reviews"] = {"contract": {"receipt": "long historical receipt" * 1000}}
        self.store.save(self.state)
        task_path = self.store.task_dir(self.state["key"]) / "task.json"
        before = task_path.read_bytes()
        args = ["--repo", str(self.repo), "--state-root", str(self.store.root), "status", "test"]
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(args), 0)
        summary = json.loads(output.getvalue())[0]
        self.assertEqual(summary["active_call"]["thread_id"], "live-thread")
        self.assertGreaterEqual(summary["active_call"]["elapsed_seconds"], 30)
        self.assertEqual(summary["usage"]["reported_tokens"], 0)
        self.assertFalse(summary["usage"]["tokens_include_active_call"])
        self.assertLess(len(output.getvalue()), 2500)
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(args + ["--full"]), 0)
        self.assertEqual(json.loads(output.getvalue())[0], json.loads(before))
        self.assertEqual(task_path.read_bytes(), before)
        # A partial first event is normal while a call is starting.
        events.write_text('{"type":')
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)

    def test_two_node_flow_reaches_ready_with_independent_reviews_and_receipt_repair(self):
        import yaml
        owner = self
        dispatches = [{"id": "positive", "parent": "root", "reason": "Synthetic alternate scalar path."}]
        class TwoNodeCLI:
            calls = []
            malformed_once = False
            def run(self, **kwargs):
                from uuid import uuid4
                state = owner.store.load(owner.state["key"])
                work, records, role = kwargs["cwd"], kwargs["records"], kwargs["role"]
                records.mkdir(parents=True)
                self.calls.append((role, state["stage"], state["node"], kwargs["thread_id"]))
                if kwargs["thread_id"] and role == "reviewer":
                    owner.assertIn("only repair the receipt", kwargs["prompt"])
                    owner.assertIn("evidence quote does not match", kwargs["prompt"])
                    owner.assertNotIn("Read these stage-specific rules", kwargs["prompt"])
                if kwargs["thread_id"] and role == "worker":
                    owner.assertIn("Continue the same author session", kwargs["prompt"])
                    owner.assertNotIn("Selected operator (paths relative", kwargs["prompt"])
                if role == "reviewer" and state["review_scope"] == "final":
                    owner.assertIn("FINAL INDEX REVIEW", kwargs["prompt"])
                    owner.assertNotIn("/schema_docs/3.design.math_reshape.md", kwargs["prompt"])
                    invalidated = deepcopy(state)
                    invalidated["reviews"].pop("positive")
                    owner.assertNotIn("FINAL INDEX REVIEW", prompt_for(invalidated, owner.baseline, work, role))
                response = owner.receipt(state, role=role, status="complete")
                response["dispatches"] = dispatches
                if role == "worker":
                    if state["stage"] == "contract":
                        owner.draft(work)
                        text = (work / "root.yaml").read_text().replace("  version: 1", "  version: 1\n  dispatch: {when: Select exactly when X <= 0.}")
                        (work / "root.yaml").write_text(text)
                    elif state["node"] == "positive":
                        node = schema_parts(work, owner.baseline["snapshot"])["root"]
                        node.pop("contract")
                        node["meta_data"]["version"] = 1
                        node["meta_data"]["dispatch"] = {"id": "positive", "parent": "root", "when": "Select exactly when X > 0."}
                        (work / "positive.yaml").write_text(yaml.safe_dump(node, sort_keys=False))
                    elif state["stage"] == "node":
                        text = (work / "root.yaml").read_text().replace("version: 1", "version: 2").replace("Keep the Contract", "Preserve the Contract")
                        (work / "root.yaml").write_text(text)
                    else:
                        (work / "dispatch.md").write_text("[root](root.yaml)\n  - [positive](positive.yaml): alternate synthetic path.\n")
                else:
                    response["source_evidence"] = [{"path": "kernel_wiki/" + relative, "line_start": 1, "line_end": 1,
                                                    "quote": (Path(owner.baseline["snapshot"]) / "kernel_wiki" / relative).read_text().strip(),
                                                    "note": "Independently read synthetic source."}
                                                   for relative in owner.entry["paths"].values()]
                    names = [path.name for path in work.iterdir()]
                    response["artifact_evidence"] = [{"path": name, "line_start": 1, "line_end": 1,
                                                      "quote": (work / name).read_text().splitlines()[0],
                                                      "note": "Independently read complete synthetic artifact."} for name in names]
                    for evidence in response["source_evidence"]:
                        evidence["path"] = owner.baseline["repository_id"] + "@" + owner.baseline["snapshot_commit"] + ":" + evidence["path"]
                    for evidence in response["artifact_evidence"]:
                        evidence["path"] = str(work / evidence["path"])
                    if not self.malformed_once:
                        self.malformed_once = True
                        response["source_evidence"][0]["quote"] = "injected wrong quote"
                return {"outcome": "completed", "response": response,
                        "thread_id": kwargs["thread_id"] or str(uuid4()), "elapsed_seconds": 0.1,
                        "usage": {"input_tokens": 10, "output_tokens": 10}}
        fake = TwoNodeCLI()
        state = Runner(self.store, self.plan, fake).work(self.state["key"])
        self.assertEqual(state["result"], "ready", state.get("last_error"))
        self.assertTrue(eligible(self.store, state))
        self.assertEqual(set(state["reviews"]), {"contract", "root", "positive", "final"})
        self.assertEqual([stage for role, stage, node, thread in fake.calls if role == "worker"], ["contract", "node", "node", "index"])
        self.assertEqual(sum(thread is not None for role, stage, node, thread in fake.calls if role == "reviewer"), 1)
        workers = [thread for role, stage, node, thread in fake.calls if role == "worker"]
        self.assertIsNone(workers[0])
        self.assertTrue(workers[1])
        self.assertEqual(len(set(workers[1:])), 1)
        self.assertEqual(sum(a["role"] == "worker" for a in state["attempts"]), 1)

    def test_hardware_findings_are_ignored_without_claiming_resolution(self):
        receipt = self.receipt(gap=True)
        receipt["issues"][0].update(id="hardware_missing", scope="contract.hardware_specification", disposition="needs_input")
        merge_issues(self.state, receipt, reviewer=True)
        self.assertEqual(self.state["issues"][0]["state"], "ignored")
        self.assertEqual(self.state["issues"][0]["disposition"], "ignored")
        self.assertNotIn("closed_checkpoint", self.state["issues"][0])
