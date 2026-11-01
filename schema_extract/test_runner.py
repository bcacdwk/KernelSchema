"""Policy/invariant tests, updated without execution in the current change."""

from copy import deepcopy
from decimal import Decimal
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from .codex_cli import CodexCLI, process_table, terminate_owned, write_json
from .protocol import RECEIPT
from .records import (Store, StateError, artifact_manifest, digest, freeze_inputs,
                      read_json, run_check, schema_parts, semantic_digests)
from .runner import (DEFAULTS, Runner, bound_review, eligible, make_plan, new_task, publish)
from .runner import merge_issues, observe_usage, sync_issues, validate_source_bindings


REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "schema_template/tools/tests/fixtures/minimal/root.yaml"


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

    def checkpoint(self, state=None, *, gap=False):
        state = state or self.state
        work = self.base / ("artifact-" + str(time.monotonic_ns()))
        self.draft(work, gap=gap)
        return self.store.commit_checkpoint(state, work, self.receipt(state), {"exit_code": 0, "policy": "skip-hardware"})

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
                    response = owner.review_receipt(state, work, status="complete", gap=True)
                    response["unsupported_claims"] = ["No extra numerical experiment was conducted."]
                    response["issues"][0]["blocks"] = []
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
                                      ("worker", "node", None), ("reviewer", "node", None)])
        self.assertEqual(set(state["reviews"]), {"contract", "final"})
        self.assertEqual(state["reviews"]["contract"]["receipt"]["continuation"], "proceed")
        self.assertEqual(state["check"]["exit_code"], 0)
        self.assertEqual(state["check"]["policy"], "skip-hardware")
        self.assertIn("design", schema_parts(self.store.checkpoint(state) / "artifacts")["root"])
        self.assertTrue(Path(publish(self.store, state)).is_dir())

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

    def test_issue_projection_is_a_patch_and_never_edits_repository(self):
        original = (self.repo / "schema_extract/ISSUES.md").read_text()
        merge_issues(self.state, self.receipt(gap=True))
        self.store.save(self.state)
        patch_path = Path(sync_issues(self.store, self.plan))
        self.assertTrue(patch_path.is_relative_to(self.store.root))
        self.assertIn("Synthetic required material is absent", patch_path.read_text())
        self.assertEqual((self.repo / "schema_extract/ISSUES.md").read_text(), original)

    def stopped_seed_run(self):
        self.state = self.checkpoint(gap=True)
        merge_issues(self.state, self.receipt(gap=True))
        self.state.update(status="ended", result="draft_with_gaps", reviews={"contract": {"legacy_review": True}})
        self.state["usage"] = {"calls": 1, "reported_tokens": 1234, "elapsed_seconds": 12.5, "unknown_usage_calls": 0}
        self.store.save(self.state)
        old_plan = {**self.plan, "run_id": "old"}
        directory = self.store.root / "runs/old"
        directory.mkdir(parents=True)
        write_json(directory / "plan.json", old_plan)
        return self.state

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
        # Known IDs keep their original ledger identity after a fresh baseline.
        old_marker = f"<!-- extraction:{old['key']}:missing_material -->"
        ledger = self.repo / "schema_extract/ISSUES.md"
        ledger.write_text(ledger.read_text() + "\n" + old_marker + "\nold issue body\n<!-- /extraction -->\n")
        updated = self.receipt(state, gap=True)
        updated["issues"][0]["detail"] = "Independently confirmed narrower unsupported hardware claims."
        merge_issues(state, updated, reviewer=True)
        self.store.save(state)
        patch_text = Path(sync_issues(self.store, plan)).read_text()
        self.assertIn("+" + old_marker, patch_text)
        self.assertNotIn(f"<!-- extraction:{state['key']}:missing_material -->", patch_text)
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
        self.assertEqual(sum(thread is not None for role, stage, node, thread in fake.calls), 1)
        self.assertTrue(all(thread is None for role, stage, node, thread in fake.calls if role == "worker"))

    def test_hardware_findings_are_ignored_without_claiming_resolution(self):
        receipt = self.receipt(gap=True)
        receipt["issues"][0].update(id="hardware_missing", scope="contract.hardware_specification", disposition="needs_input")
        merge_issues(self.state, receipt, reviewer=True)
        self.assertEqual(self.state["issues"][0]["state"], "ignored")
        self.assertEqual(self.state["issues"][0]["disposition"], "ignored")
        self.assertNotIn("closed_checkpoint", self.state["issues"][0])


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
