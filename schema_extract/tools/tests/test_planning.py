"""Plan, input identity, seed and checkpoint regressions."""

from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

from ..codex_cli import write_json
from ..execution import bound_review, eligible, merge_issues, publish, validate_source_bindings
from ..framework import candidate_report
from ..planning import freeze_inputs, make_plan, new_task, seed_inputs, verify_seed_inputs
from ..prompts import prompt_for
from ..records import StateError, artifact_manifest, digest, read_json, schema_parts
from .support import PROPOSAL, ExtractionCase


class PlanningTests(ExtractionCase):
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

    def test_publish_requires_reviews_and_preserves_conflicting_directory(self):
        self.state = self.checkpoint()
        with self.assertRaises(StateError):
            publish(self.store, self.state)
        checkpoint = self.store.checkpoint(self.state)
        self.state["reviews"] = {scope: bound_review(self.state, self.receipt(status="complete"), checkpoint) for scope in ("contract", "final")}
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

    def test_only_selected_catalog_annotation_is_frozen(self):
        relative = "annotations/synthetic/solution.py"
        annotated = self.repo / "kernel_wiki" / relative
        annotated.parent.mkdir(parents=True)
        annotated.write_text("# Corrected source commentary.\n")
        annotated.with_name("unselected.py").write_text("# Another listing.\n")
        entry = {**self.entry, "paths": {**self.entry["paths"], "annotated_entrypoint": relative}}
        baseline = freeze_inputs(self.store, entry, [], self.config)
        source = "kernel_wiki/" + relative
        self.assertEqual((Path(baseline["snapshot"]) / source).read_bytes(), annotated.read_bytes())
        self.assertNotIn(source.replace("solution.py", "unselected.py"), baseline["files"])
        state = new_task(self.store, baseline, self.base / "annotated-output")
        self.assertIn("read paths.annotated_entrypoint", prompt_for(state, baseline, self.base / "work", "worker"))

    def test_single_runner_lock_and_explicit_selection(self):
        with self.store.lock():
            with self.assertRaises(StateError):
                with self.store.lock():
                    pass
        with self.assertRaises(StateError):
            make_plan(self.store, [], self.base / "work", self.base / "out", [], self.config)

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

    def test_seed_rejects_a_different_checker_policy(self):
        old = deepcopy(self.baseline)
        old["identity"]["check_policy"] = "another-policy"
        with self.assertRaisesRegex(StateError, "seed checker policy changed"):
            verify_seed_inputs(old, self.baseline)

    def test_frozen_entrypoint_and_runtime_binding(self):
        files = {p for p in self.baseline["files"] if p.startswith("schema_extract/")}
        self.assertIn("schema_extract/runner.py", files)
        self.assertIn("schema_extract/tools/reading.py", files)
        self.assertFalse(any("/tests/" in p or p.endswith(("supervisor.py", "scheduling.py")) for p in files))
        for args in (["-m", "schema_extract"], ["schema_extract/runner.py"]):
            result = subprocess.run([sys.executable, "-B", *args, "--help"], cwd=self.baseline["snapshot"],
                                    capture_output=True, text=True, check=True)
            self.assertIn("discover", result.stdout)
        prompt = self.repo / "schema_extract/tools/prompts.py"
        prompt.write_text(prompt.read_text() + "\n# Changed frozen runtime.\n")
        baseline = freeze_inputs(self.store, self.entry, [], self.config)
        state = new_task(self.store, baseline, self.base / "fresh-output")
        with self.assertRaisesRegex(StateError, "executor/protocol changed.*tools/prompts.py"):
            self.store.verify_baseline(state)
