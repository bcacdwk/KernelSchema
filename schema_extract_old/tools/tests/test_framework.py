"""Framework proposals, archive decisions and accepted evidence."""

from copy import deepcopy
from unittest.mock import patch

from ..codex_cli import write_json
from ..execution import merge_issues
from ..framework import candidate_report, registry
from ..records import StateError, read_json, tree_manifest
from .support import PROPOSAL, ExtractionCase


class FrameworkTests(ExtractionCase):
    def test_ordinary_task_issues_never_enter_framework_ledger(self):
        root = self.repo / "schema_extract/issues"
        original = tree_manifest(root)
        merge_issues(self.state, self.receipt(gap=True))
        self.store.save(self.state)
        self.assertEqual(candidate_report(self.store, self.plan), [])
        self.assertEqual(tree_manifest(root), original)
        self.assertFalse(any(path.startswith("schema_extract/issues/") for path in self.baseline["files"]))

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
        archive = pending.rename(root / "archive" / PROPOSAL["category"] / pending.name)
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

    def test_archive_requires_the_matching_category_directory(self):
        root = self.repo / "schema_extract/issues/archive"
        record = {**PROPOSAL, "decision": {"outcome": "resolved", "reason": "Test decision."}}
        for directory in (root, root / "kernel_wiki"):
            with self.subTest(directory=directory):
                path = directory / (PROPOSAL["id"] + ".json")
                write_json(path, record)
                with self.assertRaisesRegex(StateError, "expected issue under"):
                    registry(self.repo)
                path.unlink()
