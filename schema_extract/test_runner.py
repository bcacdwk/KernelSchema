"""Policy and persistence regressions; model calls use synthetic CLI fixtures."""

from copy import deepcopy
from decimal import Decimal
import json
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from .codex_cli import CodexCLI, process_table, terminate_owned, write_json
from .framework import candidate_report, registry
from .protocol import RECEIPT, validate_receipt, prompt_for, check_feedback
from .records import (Store, StateError, artifact_manifest, digest, freeze_inputs,
                      file_digest, read_json, run_check, schema_parts, semantic_digests, tree_manifest)
from .runner import (DEFAULTS, Runner, bound_review, eligible, make_plan, new_task, publish)
from .runner import (SEED_POLICY_CHANGES, merge_issues, observe_usage, seed_inputs,
                     validate_source_bindings, verify_seed_inputs, main)


REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "schema_template/tools/tests/fixtures/minimal/root.yaml"
PROPOSAL = {"id": "selector-diagnostics", "category": "extract_tools", "problem": "Synthetic ambiguous diagnostic.",
            "evidence": ["schema_template/check_yaml.py:1 (synthetic example)"], "proposal": "Identify the selector.",
            "necessity": "Otherwise any selected task can be sent to fix the wrong node."}


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="schema-extract-test-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.repo = self.base / "repository"
        self.repo.mkdir()
        for name in ("schema_docs", "schema_extract"):
            shutil.copytree(REPO / name, self.repo / name, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(REPO / "schema_template/tools", self.repo / "schema_template/tools",
                        ignore=shutil.ignore_patterns("tests", "__pycache__"))
        for name in ("README.md", "EXTRACT_SCHEMA.md", "schema_template/README.md", "schema_template/starter.yaml", "schema_template/check_yaml.py"):
            destination = self.repo / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / name, destination)
        self.entry = {"id": "synthetic", "title": "Synthetic scalar copy", "paths": {}}
        operator = self.repo / "kernel_wiki/test/operators/synthetic"
        operator.mkdir(parents=True)
        for name in ("readme", "entrypoint", "input", "reference", "acceptance", "metadata"):
            relative = "test/operators/synthetic/" + name + ".txt"
            self.entry["paths"][name] = relative
            (self.repo / "kernel_wiki" / relative).write_text(f"{name}: synthetic scalar copy evidence\n")
        write_json(self.repo / "kernel_wiki/catalog.json", {"operators": [self.entry]})
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(["git", "add", "."], cwd=self.repo, check=True)
        subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
                        "-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "-qm", "fixture"], cwd=self.repo, check=True)
        self.store = Store(self.repo, self.base / "records")
        self.config = {**DEFAULTS, "jobs": 2}
        self.baseline = freeze_inputs(self.store, self.entry, [], self.config)
        self.state = new_task(self.store, self.baseline, self.base / "output/synthetic")
        self.plan = {"run_id": "test", "tasks": [self.state["key"]], "config": self.config,
                     "work_root": str(self.base / "work")}

    def draft(self, work, version=1, *, gap=False):
        work.mkdir(parents=True, exist_ok=True)
        prefix = self.baseline["repository_id"] + "@" + self.baseline["snapshot_commit"] + ":kernel_wiki/"
        text = FIXTURE.read_text().replace("  version: 1", f"  source: {prefix}{self.entry['paths']['entrypoint']}\n  version: {version}")
        text = text.replace("testing.invalid/kernel@fixture-v1:reference.py::copy", prefix + self.entry["paths"]["reference"] + "::copy")
        if not gap:
            import re
            text = re.sub(r'"<testing.invalid/[^\n]+?>"', '"' + prefix + self.entry["paths"]["metadata"] + '"', text)
        (work / "root.yaml").write_text(text)
        (work / "dispatch.md").write_text("[root](root.yaml): synthetic scalar copy.\n")

    def receipt(self, state=None, *, role="worker", status="complete", gap=False):
        state = state or self.state
        value = {key: [] if field["type"] == "array" else "" for key, field in RECEIPT["properties"].items()}
        value.update(task_key=state["key"], fingerprint=state["fingerprint"], base_checkpoint=state.get("checkpoint") or "none",
                     stage=state["stage"], node=state["node"], status=status, summary="Synthetic test receipt.", impact="none", correction="none",
                     continuation="not_assessed")
        if role == "reviewer" and state["review_scope"] == "contract":
            value["continuation"] = "needs_input" if status == "needs_input" else "proceed"
            value["continuation_reason"] = "Synthetic fundamental ambiguity remains." if status == "needs_input" else "Input, reference and entrypoint identify scalar-copy IO and computation; source-fact extraction can continue."
        if gap:
            value["issues"] = [{"id": "missing_material", "kind": "evidence_gap", "scope": "external numeric proof", "disposition": "note",
                                "detail": "Synthetic required material is absent.", "evidence": ["synthetic fixture"],
                                "restart_condition": "Supply the missing material and create a new plan.", "blocks": ["final"]}]
            if role == "reviewer" and status == "needs_input":
                value["issues"][0].update(disposition="needs_input", scope="fundamental task definition")
        return value

    def review_receipt(self, state, work, *, status="complete", gap=False):
        value = self.receipt(state, role="reviewer", status=status, gap=gap)
        baseline = self.store.baseline(state["key"])
        value["source_evidence"] = [{"path": "kernel_wiki/" + relative, "line_start": 1, "line_end": 1,
                                     "quote": (Path(baseline["snapshot"]) / "kernel_wiki" / relative).read_text().strip(),
                                     "note": "Independently read synthetic source."}
                                    for relative in self.entry["paths"].values()]
        value["artifact_evidence"] = [{"path": path.name, "line_start": 1, "line_end": 1,
                                       "quote": path.read_text().splitlines()[0],
                                       "note": "Independently read complete synthetic artifact."}
                                      for path in work.iterdir() if path.is_file()]
        return value

    def checkpoint(self, state=None, *, gap=False, candidates=()):
        state = state or self.state
        work = self.base / ("artifact-" + str(time.monotonic_ns()))
        self.draft(work, gap=gap)
        receipt = self.receipt(state)
        receipt["framework_candidates"] = list(candidates)
        return self.store.commit_checkpoint(state, work, receipt, {"exit_code": 0, "policy": "skip-hardware"})

    def test_checkpoint_commit_is_atomic_and_unaccepted_draft_is_not_recovered(self):
        self.state = self.checkpoint()
        accepted = self.state["checkpoint"]
        attempt, work = self.store.new_attempt(self.state, self.base / "work", "worker")
        (work / "root.yaml").write_text("half written: [")
        self.state["status"] = "stopped"
        self.store.save(self.state)
        _, fresh = self.store.new_attempt(self.state, self.base / "work", "worker")
        self.assertNotEqual(work, fresh)
        self.assertEqual(artifact_manifest(fresh), artifact_manifest(self.store.checkpoint(self.state) / "artifacts"))
        (fresh / "root.yaml").write_text((fresh / "root.yaml").read_text().replace("version: 1", "version: 2").replace("Keep the Contract", "Preserve the Contract"))
        with patch.object(self.store, "save", side_effect=OSError("injected crash before state commit")):
            with self.assertRaises(OSError):
                self.store.commit_checkpoint(self.state, fresh, self.receipt(self.state), {"exit_code": 0})
        self.assertEqual(self.store.load(self.state["key"])["checkpoint"], accepted)
        self.assertGreater(len(list((self.store.task_dir(self.state["key"]) / "checkpoints").iterdir())), 1)

    def test_contract_correction_invalidates_all_reviews_and_parser_is_exact(self):
        work = self.base / "exact"
        self.draft(work)
        text = (work / "root.yaml").read_text().replace("dtype: fp32", "dtype: off", 1)
        text = text.replace("Y = X.", "Y = X.") + "\nextra_for_parser: 1.000000000000000000000001\n"
        (work / "root.yaml").write_text(text)
        parsed = schema_parts(work, self.baseline["snapshot"])["root"]
        self.assertEqual(parsed["contract"]["algorithm_requirement"]["input"]["X"]["dtype"], "off")
        self.assertEqual(parsed["extra_for_parser"], Decimal("1.000000000000000000000001"))
        self.assertNotEqual(digest(parsed["extra_for_parser"]), digest(Decimal("1.000000000000000000000000")))
        self.state = self.store.commit_checkpoint(self.state, work, self.receipt(), {"exit_code": 1})
        self.state["reviews"] = {"contract": {"accepted": True}, "root": {"accepted": True}, "sibling": {"accepted": True}, "final": {"accepted": True}}
        self.store.save(self.state)
        text = text.replace("version: 1", "version: 2").replace("dtype: off", "dtype: fp32", 1)
        (work / "root.yaml").write_text(text)
        changed = self.store.commit_checkpoint(self.state, work, self.receipt(self.state), {"exit_code": 1})
        self.assertEqual(changed["reviews"], {})
        self.assertFalse(changed["eligible"])

    def test_duplicate_keys_are_rejected_without_accepting_checkpoint(self):
        work = self.base / "duplicate"
        self.draft(work)
        with (work / "root.yaml").open("a") as handle:
            handle.write("\ncontract: {}\n")
        with self.assertRaises(StateError):
            self.store.commit_checkpoint(self.state, work, self.receipt(), {"exit_code": 1})
        self.assertIsNone(self.store.load(self.state["key"])["checkpoint"])

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
        self.state["reviews"]["contract"] = bound_review(self.state, self.baseline, self.receipt(), self.store.checkpoint(self.state))
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

    def test_publish_requires_reviews_and_preserves_conflicting_directory(self):
        self.state = self.checkpoint()
        with self.assertRaises(StateError):
            publish(self.store, self.state)
        checkpoint = self.store.checkpoint(self.state)
        self.state["reviews"] = {scope: bound_review(self.state, self.baseline, self.receipt(status="complete"), checkpoint) for scope in ("contract", "final")}
        self.state.update(status="ended", result="ready", eligible=True)
        self.store.save(self.state)
        self.assertTrue(eligible(self.store, self.state))
        target = Path(publish(self.store, self.state))
        self.assertEqual(str(target), publish(self.store, self.state))
        (target / "root.yaml").write_text("user content\n")
        with self.assertRaises(StateError):
            publish(self.store, self.state)
        self.assertEqual((target / "root.yaml").read_text(), "user content\n")

    def test_inputs_bind_source_spec_and_clean_commit_without_touching_user_worktree(self):
        self.assertEqual(self.baseline["source_ref_kind"], "existing_commit")
        self.assertFalse((Path(self.baseline["snapshot"]) / ".git").exists())
        (self.repo / "unrelated.txt").write_text("unrelated user change")
        same = freeze_inputs(self.store, self.entry, [], self.config)
        self.assertEqual(same["fingerprint"], self.baseline["fingerprint"])
        source = self.repo / "kernel_wiki" / self.entry["paths"]["reference"]
        source.write_text("updated synthetic evidence\n")
        changed = freeze_inputs(self.store, self.entry, [], self.config)
        self.assertNotEqual(changed["fingerprint"], same["fingerprint"])
        self.assertEqual(changed["source_ref_kind"], "private_snapshot")
        status = subprocess.check_output(["git", "status", "--short"], cwd=self.repo, text=True)
        self.assertIn("unrelated.txt", status)
        frozen_spec = Path(self.baseline["snapshot"]) / "schema_docs/0.common.md"
        frozen_spec.chmod(0o644)
        frozen_spec.write_text("tampered frozen specification\n")
        with self.assertRaises(StateError):
            self.store.verify_baseline(self.state)

    def test_single_runner_lock_and_explicit_selection(self):
        with self.store.lock():
            with self.assertRaises(StateError):
                with self.store.lock():
                    pass
        with self.assertRaises(StateError):
            make_plan(self.store, [], self.base / "work", self.base / "out", [], self.config)

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

    def test_ordinary_task_issues_never_enter_framework_ledger(self):
        root = self.repo / "schema_extract/issues"
        original = tree_manifest(root)
        merge_issues(self.state, self.receipt(gap=True))
        self.store.save(self.state)
        self.assertEqual(candidate_report(self.store, self.plan), [])
        self.assertEqual(tree_manifest(root), original)
        self.assertFalse(any(path.startswith("schema_extract/issues/") for path in self.baseline["files"]))

    def stopped_seed_run(self):
        self.state = self.checkpoint(gap=True, candidates=[PROPOSAL])
        merge_issues(self.state, self.receipt(gap=True))
        self.state.update(status="ended", result="draft_with_gaps", reviews={"contract": {"legacy_review": True}})
        self.state["usage"] = {"calls": 1, "reported_tokens": 1234, "elapsed_seconds": 12.5, "unknown_usage_calls": 0}
        self.store.save(self.state)
        old_plan = {**self.plan, "run_id": "old"}
        directory = self.store.root / "runs/old"
        directory.mkdir(parents=True)
        write_json(directory / "plan.json", old_plan)
        return self.state

    def test_task_note_keeps_accepted_source_facts_after_summary_only_review(self):
        work = self.base / "origin"
        self.draft(work)
        first = self.receipt(gap=True)
        first["issues"][0]["detail"] = "Concrete original finding from the fixed source."
        first["issues"][0]["evidence"] = ["kernel_wiki/test/operators/synthetic/reference.txt:1"]
        state = self.store.commit_checkpoint(self.state, work, first, {"exit_code": 0, "policy": "skip-hardware"})
        merge_issues(state, first)
        final = deepcopy(first)
        final["issues"][0].update(detail="Retained unchanged.", evidence=["dispatch.md:1"])
        merge_issues(state, final, reviewer=True)
        self.store.save(state)
        task = self.store.task_dir(state["key"]) / "task.json"
        original_bytes = task.read_bytes()
        self.assertEqual(candidate_report(self.store, self.plan), [])
        self.assertEqual(state["issues"][0]["detail"], "Retained unchanged.")
        record = read_json(self.store.checkpoint(state) / "checkpoint.json")
        self.assertEqual(record["worker_receipt"]["issues"][0], first["issues"][0])
        self.assertEqual(task.read_bytes(), original_bytes)

    def test_replayed_candidate_respects_archive_and_excludes_orphan_checkpoint(self):
        state = self.checkpoint(candidates=[PROPOSAL])
        work = self.store.checkpoint(state) / "artifacts"
        receipt = self.receipt(state)
        receipt["framework_candidates"] = [{**PROPOSAL, "id": "unaccepted"}]
        with patch.object(self.store, "save", side_effect=OSError("interrupt")), self.assertRaises(OSError):
            self.store.commit_checkpoint(state, work, receipt, {"exit_code": 0})
        root = self.repo / "schema_extract/issues"
        pending = root / PROPOSAL["category"] / (PROPOSAL["id"] + ".json")
        write_json(pending, PROPOSAL)
        groups = candidate_report(self.store, self.plan)
        self.assertEqual([g["id"] for g in groups], [PROPOSAL["id"]])
        self.assertEqual(groups[0]["registered"]["status"], "pending")
        archive = pending.rename(root / "archive" / pending.name)
        for outcome in ("declined", "resolved", "partially_adopted"):
            decision = {"outcome": outcome, "reason": "Synthetic decision and its basis."}
            if outcome == "partially_adopted":
                write_json(archive, {**PROPOSAL, "decision": decision})
                with self.assertRaisesRegex(StateError, "partial decision"):
                    registry(self.repo)
                decision.update(adopted="Show current selector.", remaining="Defer selector history.")
            write_json(archive, {**PROPOSAL, "decision": decision})
            original = tree_manifest(root)
            replay = candidate_report(self.store, self.plan)
            self.assertEqual(replay[0]["registered"]["status"], outcome)
            self.assertEqual(candidate_report(self.store, self.plan), replay)
            self.assertFalse(pending.exists())
            self.assertEqual(original, tree_manifest(root))
        write_json(pending, PROPOSAL)
        with self.assertRaisesRegex(StateError, "duplicate"):
            registry(self.repo)

    def test_seed_preserves_draft_usage_and_issue_identity_but_restarts_independent_review(self):
        old = self.stopped_seed_run()
        old_record = read_json(self.store.checkpoint(old) / "checkpoint.json")
        old_task_bytes = (self.store.task_dir(old["key"]) / "task.json").read_bytes()
        (self.repo / "kernel_wiki/README.md").write_text("Synthetic explicit target background; no capacity guarantees.\n")
        (self.repo / "kernel_wiki/EVIDENCE.md").write_text("Synthetic evidence navigation and limits.\n")
        config = dict(self.config)
        plan = make_plan(self.store, ["synthetic"], self.base / "seed-work", self.base / "output", [], config,
                         "seeded", seed_run="old")
        state = self.store.load(plan["tasks"][0])
        baseline = self.store.verify_baseline(state)
        checkpoint = self.store.verify_checkpoint(state)
        imported = read_json(checkpoint / "checkpoint.json")
        self.assertEqual((state["phase"], state["stage"], state["review_scope"]), ("reviewer", "contract", "contract"))
        self.assertEqual(state["attempts"], [])
        self.assertEqual(state["reviews"], {})
        self.assertFalse(state["eligible"])
        self.assertEqual(state["usage"], old["usage"])
        self.assertEqual(imported["files"], old_record["files"])
        self.assertEqual(imported["imported_worker_receipt"], old_record["worker_receipt"])
        self.assertNotIn("worker_receipt", imported)
        self.assertEqual(state["seed"]["checkpoint"], old["checkpoint"])
        self.assertIn("kernel_wiki/README.md", baseline["files"])
        self.assertIn("kernel_wiki/EVIDENCE.md", baseline["files"])
        validate_source_bindings(baseline, checkpoint / "artifacts", state["source_bindings"])
        # Added evidence may not be falsely attributed to the older source ref.
        copy = self.base / "alias-check"
        shutil.copytree(checkpoint / "artifacts", copy)
        root = copy / "root.yaml"
        old_binding = state["source_bindings"][0]
        old_reference = old_binding["repository_id"] + "@" + old_binding["commit"] + ":kernel_wiki/README.md"
        text = root.read_text()
        source_line = next(line for line in text.splitlines() if line.startswith("  source:"))
        root.chmod(0o644)
        root.write_text(text.replace(source_line, "  source: " + old_reference))
        with self.assertRaises(StateError):
            validate_source_bindings(baseline, copy, state["source_bindings"])
        self.assertEqual((self.store.task_dir(old["key"]) / "task.json").read_bytes(), old_task_bytes)
        self.assertEqual(state["usage"]["elapsed_seconds"], 12.5)
        # A seed retains the local issue's identity, but importing its old
        # worker receipt is not a second framework proposal in this run.
        updated = self.receipt(state, gap=True)
        updated["issues"][0]["detail"] = "Independently confirmed narrower unsupported hardware claims."
        merge_issues(state, updated, reviewer=True)
        self.store.save(state)
        self.assertEqual([g["id"] for g in candidate_report(self.store, self.plan)], [PROPOSAL["id"]])
        self.assertEqual(candidate_report(self.store, plan), [])
        self.assertEqual(state["issues"][0]["ledger_task_key"], old["key"])

    def test_seed_rejects_changed_source_or_spec_and_keeps_old_task(self):
        old = self.stopped_seed_run()
        old_bytes = (self.store.task_dir(old["key"]) / "task.json").read_bytes()
        for label, path in (("source", self.repo / "kernel_wiki" / self.entry["paths"]["reference"]),
                            ("spec", self.repo / "schema_docs/0.common.md")):
            with self.subTest(label=label):
                original = path.read_text()
                try:
                    path.write_text(original + "\nChanged semantic material.\n")
                    with self.assertRaisesRegex(StateError, "seed source/spec changed"):
                        make_plan(self.store, ["synthetic"], self.base / "seed-work", self.base / "output", [],
                                  self.config, "rejected-" + label, seed_run="old")
                finally:
                    path.write_text(original)
                self.assertEqual((self.store.task_dir(old["key"]) / "task.json").read_bytes(), old_bytes)
                self.assertFalse((self.store.root / "runs" / ("rejected-" + label)).exists())

    def test_seed_policy_update_only_accepts_the_audited_hash_pairs(self):
        old, new = deepcopy(self.baseline), deepcopy(self.baseline)
        old["identity"]["check_policy"] = "strict-no-hardware-waiver"
        for name, (before, after) in SEED_POLICY_CHANGES.items():
            old["files"][name], new["files"][name] = before, after
        with self.assertRaisesRegex(StateError, "seed source/spec changed"):
            verify_seed_inputs(old, new)
        verify_seed_inputs(old, new, "skip-hardware")
        for name in ("kernel_wiki/" + self.entry["paths"]["reference"], "schema_docs/0.common.md",
                     "schema_template/starter.yaml", "schema_template/tools/schema.py"):
            with self.subTest(name=name):
                altered = deepcopy(new)
                altered["files"][name] = "unreviewed-change"
                with self.assertRaises(StateError):
                    verify_seed_inputs(old, altered, "skip-hardware")
        with self.assertRaisesRegex(StateError, "requires --seed-run"):
            make_plan(self.store, ["synthetic"], self.base / "work", self.base / "out", [], self.config,
                      seed_policy_update="skip-hardware")

    def test_policy_seed_only_replaces_hardware_and_rereviews_existing_design(self):
        strict = deepcopy(self.baseline)
        strict["identity"]["check_policy"] = "strict-no-hardware-waiver"
        strict["fingerprint"] = digest(strict["identity"])
        self.state = new_task(self.store, strict, self.base / "old-output/synthetic")
        self.plan["tasks"] = [self.state["key"]]
        old = self.stopped_seed_run()
        old_files = tree_manifest(self.store.task_dir(old["key"]))
        old_artifacts = self.store.checkpoint(old) / "artifacts"
        previous = schema_parts(old_artifacts, strict["snapshot"])["root"]
        pairs = {}
        for name in SEED_POLICY_CHANGES:
            path = self.repo / name
            before = file_digest(path)
            path.write_text(path.read_text() + "\n# Synthetic audited policy edit.\n")
            pairs[name] = (before, file_digest(path))
        with patch("schema_extract.runner.SEED_POLICY_CHANGES", pairs):
            plan = make_plan(self.store, ["synthetic"], self.base / "seed-work", self.base / "output", [],
                             self.config, "policy-seeded", seed_run="old", seed_policy_update="skip-hardware")
        state = self.store.load(plan["tasks"][0])
        baseline = self.store.baseline(state["key"])
        checkpoint = self.store.checkpoint(state)
        record = read_json(checkpoint / "checkpoint.json")
        imported = schema_parts(checkpoint / "artifacts", baseline["snapshot"])["root"]
        expected = deepcopy(previous)
        expected["contract"]["hardware_specification"] = schema_parts(self.repo / "schema_template")["starter"]["contract"]["hardware_specification"]
        expected["meta_data"]["version"] += 1
        self.assertEqual(imported, expected)
        self.assertEqual(record["policy_edit"]["before"], artifact_manifest(old_artifacts)["root.yaml"])
        self.assertEqual(record["imported_worker_receipt"], read_json(self.store.checkpoint(old) / "checkpoint.json")["worker_receipt"])
        self.assertEqual(state["usage"], old["usage"])
        self.assertEqual(state["reviews"], {})
        self.assertEqual(state["check"]["exit_code"], 0)
        self.assertEqual(state["issues"][0]["ledger_task_key"], old["key"])
        self.assertEqual(tree_manifest(self.store.task_dir(old["key"])), old_files)
        roles = []
        def review(**kwargs):
            current = self.store.load(state["key"])
            roles.append((kwargs["role"], current["review_scope"]))
            self.assertEqual(kwargs["role"], "reviewer")
            kwargs["records"].mkdir(parents=True)
            return {"outcome": "completed", "response": self.review_receipt(current, kwargs["cwd"]),
                    "thread_id": "00000000-0000-0000-0000-000000000001", "elapsed_seconds": 0.1, "usage": {}}
        complete = Runner(self.store, plan, Mock(run=review)).work(state["key"])
        self.assertEqual(complete["result"], "ready", complete.get("last_error"))
        self.assertEqual(roles, [("reviewer", "contract"), ("reviewer", "final")])
        self.assertEqual(complete["checkpoint"], state["checkpoint"])
        self.assertEqual(tree_manifest(self.store.task_dir(old["key"])), old_files)

    def test_seed_accepts_explicit_recovery_confirmation_but_never_missing_evidence(self):
        old = self.stopped_seed_run()
        records = self.store.task_dir(old["key"]) / "attempts/interrupted/call-01"
        records.mkdir(parents=True)
        old["attempts"] = [{"id": "interrupted", "calls": [str(records)]}]
        self.store.save(old)
        with self.assertRaisesRegex(StateError, "termination is not confirmed"):
            seed_inputs(self.store, "old", "synthetic")
        write_json(records / "recovery_termination.json", {"confirmed": True, "actions": [], "survivors": []})
        self.assertEqual(seed_inputs(self.store, "old", "synthetic")[0]["key"], old["key"])

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


class CheckFeedbackTests(unittest.TestCase):
    def test_diagnostic_locations_shorten_without_hiding_actual_values_or_errors(self):
        raw = {"exit_code": 1, "policy": "skip-hardware", "argv": ["python", "/long/work/root.yaml"],
               "stderr": '/long/work/root.yaml:3:2\n  contract.field: wrong value\n'
                         '  Actual: "/long/work/keep-this-literal"\n  See: /fixed/repo/schema_docs/1.contract.algorithm_requirement.md\n\n'
                         '/long/work/derived.yaml:7:4\n  design.field: missing name\n  Shared root: /long/work/root.yaml\n'}
        before = deepcopy(raw)
        compact = check_feedback(raw, {"snapshot": "/fixed/repo"})
        self.assertEqual(raw, before)
        self.assertEqual(compact["exit_code"], 1)
        self.assertEqual(compact["argv"], raw["argv"])
        for expected in ('root.yaml:3:2', 'derived.yaml:7:4', 'contract.field: wrong value',
                         'design.field: missing name', 'Actual: "/long/work/keep-this-literal"',
                         'See: schema_docs/1.contract.algorithm_requirement.md', 'Shared root: root.yaml'):
            self.assertIn(expected, compact["stderr"])
        self.assertNotIn("/fixed/repo/schema_docs", compact["stderr"])
        passed = {"exit_code": 0, "stderr": "Hardware skipped explicitly.\n"}
        self.assertEqual(check_feedback(passed, {"snapshot": "/fixed/repo"}), passed)


class UsageObservationTests(unittest.TestCase):
    def test_real_cumulative_pairs_and_duplicate_observation(self):
        for first, second, expected in (
            ((1736471, 23617), (2026963, 26840), 293715),
            ((64427, 225), (132440, 449), 68237),
        ):
            with self.subTest(first=first):
                state = {"usage": {"reported_tokens": 0, "unknown_usage_calls": 0}}
                one = {"thread_id": "thread", "usage": {"input_tokens": first[0], "output_tokens": first[1]}}
                two = {"thread_id": "thread", "resume_thread_id": "thread",
                       "usage": {"input_tokens": second[0], "output_tokens": second[1], "cached_input_tokens": 99, "reasoning_output_tokens": 11}}
                raw = deepcopy(two)
                observe_usage(state, one, "first")
                observed = observe_usage(state, two, "second")
                self.assertEqual(observed["delta_tokens"], expected)
                self.assertEqual(state["usage"]["reported_tokens"], sum(second))
                self.assertEqual(observe_usage(state, two, "second"), observed)
                self.assertEqual(state["usage"]["reported_tokens"], sum(second))
                self.assertEqual(two, raw)

    def test_missing_and_decreased_usage_are_observational_unknowns(self):
        state = {"usage": {"reported_tokens": 0, "unknown_usage_calls": 0}}
        observe_usage(state, {"thread_id": "t", "usage": {"input_tokens": 100, "output_tokens": 5}}, "first")
        missing = observe_usage(state, {"thread_id": "t", "usage": {}}, "missing")
        decreased = observe_usage(state, {"thread_id": "t", "usage": {"input_tokens": 90, "output_tokens": 2}}, "decreased")
        unknown = observe_usage(state, {"thread_id": "other", "resume_thread_id": "other",
                                       "usage": {"input_tokens": 80, "output_tokens": 1}}, "unknown-resume")
        self.assertIsNone(missing["delta_tokens"])
        self.assertIsNone(decreased["delta_tokens"])
        self.assertIsNone(unknown["delta_tokens"])
        self.assertEqual(state["usage"]["reported_tokens"], 105)
        self.assertEqual(state["usage"]["unknown_usage_calls"], 3)
        self.assertNotIn("status", state)


class TransportTests(unittest.TestCase):
    def test_spawn_failure_records_proof_that_no_agent_process_started(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cli = CodexCLI()
            with patch.object(cli, "version", return_value="synthetic"), patch("schema_extract.codex_cli.subprocess.Popen", side_effect=OSError("synthetic spawn failure")):
                result = cli.run(cwd=directory, records=directory / "call", prompt="test", schema={})
            self.assertEqual(result["outcome"], "spawn_error")
            self.assertIn("synthetic spawn failure", result["error"])
            self.assertEqual(result, read_json(directory / "call/result.json"))
            self.assertTrue(result["termination"]["confirmed"])
            self.assertEqual(result["termination"]["reason"], "process_not_started")
            self.assertFalse((directory / "call/process.json").exists())

    def test_version_timeout_is_a_preflight_error_not_an_agent_timeout(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cli = CodexCLI()
            with patch.object(cli, "version", side_effect=subprocess.TimeoutExpired("codex --version", 15)), patch("schema_extract.codex_cli.subprocess.Popen") as spawn:
                result = cli.run(cwd=directory, records=directory / "call", prompt="test", schema={})
            spawn.assert_not_called()
            self.assertEqual(result["outcome"], "preflight_error")
            self.assertTrue(result["termination"]["confirmed"])

    def test_unconfirmed_termination_takes_priority_over_missing_resume_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cli = CodexCLI()
            process = Mock(pid=12345)
            process.poll.return_value = process.wait.return_value = 0
            identity = {"pid": process.pid, "started": "synthetic"}
            with patch.object(cli, "version", return_value="synthetic"), \
                    patch("schema_extract.codex_cli.subprocess.Popen", return_value=process), \
                    patch("schema_extract.codex_cli.process_table", return_value={process.pid: identity}), \
                    patch("schema_extract.codex_cli.terminate_owned", return_value={"confirmed": False, "actions": [], "survivors": [identity]}):
                result = cli.run(cwd=directory, records=directory / "call", prompt="test", schema={},
                                 thread_id="00000000-0000-0000-0000-000000000001")
            self.assertEqual(result["outcome"], "termination_unconfirmed")
            self.assertFalse(result["termination"]["confirmed"])

    def test_process_tree_termination_and_safe_cli_arguments(self):
        process = subprocess.Popen(["python3", "-c", "import subprocess,time; subprocess.Popen(['python3','-c','import time;time.sleep(60)']); time.sleep(60)"], start_new_session=True)
        try:
            identity = process_table()[process.pid]
            time.sleep(0.2)
            result = terminate_owned(identity)
            self.assertTrue(result["confirmed"])
            process.wait(timeout=5)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        command = CodexCLI().command(Path("/tmp/task"), Path("/tmp/schema"), Path("/tmp/output"), "reviewer", None)
        self.assertIn("read-only", command)
        self.assertIn('model_reasoning_effort="xhigh"', command)
        self.assertNotIn("--ignore-rules", command)
        self.assertNotIn("danger-full-access", command)
        supervisor = CodexCLI().command(Path("/tmp/task"), Path("/tmp/schema"), Path("/tmp/output"), "supervisor", None)
        self.assertIn("read-only", supervisor)
        with self.assertRaises(ValueError):
            CodexCLI().command(Path("/tmp/task"), Path("/tmp/schema"), Path("/tmp/output"), "worker", "--last")


if __name__ == "__main__":
    unittest.main()
