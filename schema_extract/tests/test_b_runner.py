from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

from schema_extract.artifacts import load_root, validate_artifacts
from schema_extract.common import ExtractError, read_json, sha256
from schema_extract.inputs import resolve_request
from schema_extract.reading import encoded, read_range
from schema_extract.runner import accept_result
from schema_extract.snapshots import verify_snapshot
from support import Fixture, REPO


class RunnerTests(unittest.TestCase):
    def fixture(self, scenario=None):
        fixture = Fixture(scenario)
        self.addCleanup(fixture.cleanup)
        return fixture

    def test_two_independent_approvals_and_exact_p2_resumes(self):
        fixture = self.fixture()
        state = fixture.run(worker_effort="high", reviewer_effort="xhigh")
        self.assertEqual((state["stage"], state["status"]), ("P6", "ready"), state["reason"])
        self.assertTrue(state["published"])
        self.assertTrue((fixture.base / "output/root.yaml").is_file())
        calls = fixture.calls()[:4]
        self.assertEqual([(item["stage"], item["role"]) for item in calls],
                         [("P1", "worker"), ("P1", "reviewer"), ("P2", "worker"), ("P2", "reviewer")])
        self.assertNotEqual(calls[0]["session_id"], calls[1]["session_id"])
        self.assertEqual(calls[2]["resumed"], calls[0]["session_id"])
        self.assertEqual(calls[3]["resumed"], calls[1]["session_id"])
        for item in calls:
            self.assertIn("01_contract.md" if item["stage"] == "P1" else "02_dispatch_draft.md", item["prompt"])
            self.assertIn("# P1：共享 Contract" if item["stage"] == "P1" else "# P2：dispatch 草案", item["prompt"])
            self.assertNotIn("--ignore-user-config", item["argv"])
            self.assertIn('model_reasoning_effort="high"' if item["role"] == "worker" else 'model_reasoning_effort="xhigh"', item["argv"])
            self.assertLess(item["argv"].index("-C"), item["argv"].index("exec"))
        self.assertNotEqual(state["stages"]["P1"]["approval"]["attempt_id"], state["stages"]["P2"]["approval"]["attempt_id"])
        self.assertTrue((fixture.store.run_dir(state["run_id"]) / "stages/P2/handoff.md").is_file())
        artifacts, _ = fixture.store.verify_checkpoint(state, state["stages"]["P2"]["approval"]["checkpoint"])
        self.assertIsNone(load_root(artifacts)["design"]["math_reshape"]["thesis"])
        self.assertEqual(state["config"]["revision_limit"], 5)

    def test_continue_saves_a_draft_without_approval_or_revision(self):
        fixture = self.fixture({"P1.worker.1": "continue"})
        state = fixture.run()
        self.assertEqual(state["status"], "ready", state["reason"])
        self.assertEqual(state["stages"]["P1"]["continues"], 1)
        self.assertEqual(state["stages"]["P1"]["revisions"], 0)
        calls = fixture.calls()
        self.assertEqual(calls[0]["role"], calls[1]["role"])
        self.assertEqual(calls[1]["resumed"], calls[0]["session_id"])
        self.assertNotEqual(calls[0]["identity"]["base_checkpoint"], calls[1]["identity"]["base_checkpoint"])
        cp = fixture.store.verify_checkpoint(state, calls[1]["identity"]["base_checkpoint"])[0]
        self.assertIsNone(load_root(cp)["design"]["math_reshape"]["thesis"])

    def test_bad_evidence_identity_role_and_artifact_digest_do_not_advance(self):
        for action in ("bad_fingerprint", "role_forge", "bad_artifact_sha"):
            with self.subTest(action=action):
                fixture = self.fixture({"P1.worker.1": action})
                fixture.create()
                initial = fixture.state["checkpoint"]
                from schema_extract.runner import run
                state = run(fixture.store, fixture.state["run_id"])
                self.assertEqual((state["stage"], state["role"], state["status"]), ("P1", "worker", "invalid_receipt"))
                self.assertEqual(state["checkpoint"], initial)
                self.assertFalse(state["accepted_receipts"])

        fixture = self.fixture({"P1.worker.1": "bad_quote"})
        state = fixture.run()
        self.assertEqual(state["status"], "ready", state.get("reason"))
        self.assertEqual(state["stages"]["P1"]["receipt_repairs"], 1)
        first = fixture.calls()[0]["identity"]["attempt_id"]
        self.assertNotIn(first, state["accepted_receipts"])

    def test_stage_field_and_file_scopes(self):
        cases = [("P1.worker.1", "extra_file"), ("P1.worker.1", "design_write"),
                 ("P1.reviewer.1", "reviewer_write"), ("P2.worker.1", "protected_contract")]
        for call, action in cases:
            with self.subTest(call=call, action=action):
                fixture = self.fixture({call: action})
                state = fixture.run()
                self.assertEqual(state["status"], "invalid_receipt", state["reason"])
                self.assertEqual(state["stage"], call.split(".")[0])
                self.assertIsNone(state["stages"][state["stage"]]["approval"])

    def test_duplicate_receipt_is_never_committed_twice(self):
        fixture = self.fixture()
        state = fixture.run()
        attempt_id = state["attempts"][-1]
        directory = fixture.store.run_dir(state["run_id"]) / "attempts" / attempt_id
        before = deepcopy(state)
        with self.assertRaisesRegex(ExtractError, "duplicate"):
            accept_result(fixture.store, state, fixture.lock, read_json(directory / "attempt.json"), read_json(directory / "result.json"))
        self.assertEqual(state, before)
        self.assertEqual(fixture.store.load(state["run_id"]), before)

    def test_replayed_earlier_receipt_cannot_approve_p2(self):
        fixture = self.fixture({"P2.worker.1": "replay"})
        state = fixture.run()
        self.assertEqual((state["stage"], state["status"]), ("P2", "invalid_receipt"))
        self.assertEqual(state["stages"]["P1"]["status"], "approved")
        self.assertIsNone(state["stages"]["P2"]["approval"])

    def test_cli_extract_and_status(self):
        fixture = self.fixture()
        command = [sys.executable, "-B", "-m", "schema_extract", "extract", "--source", str(fixture.source),
                   "--output", str(fixture.base / "delivery"), "--state-root", str(fixture.base / "v2"),
                   "--codex-bin", str(fixture.fake)]
        result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        state = json.loads(result.stdout)
        result = subprocess.run([sys.executable, "-B", "-m", "schema_extract", "status", state["run_id"],
                                 "--state-root", str(fixture.base / "v2")], cwd=REPO, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["approved_stages"], ["P1", "P2", "P3", "P4", "P6"])

    def test_frozen_rules_include_current_uncommitted_bytes_and_six_pages(self):
        fixture = self.fixture()
        _, _, lock = fixture.create()
        verify_snapshot(lock)
        for path in ("schema_extract/runner.py", "schema_extract/README.md", "schema_template/check_yaml.py", "README.md"):
            self.assertEqual(lock["rules"]["files"][path], sha256((REPO / path).read_bytes()))
        self.assertEqual(len([name for name in lock["rules"]["files"] if name.startswith("schema_extract/phases/")]), 6)
        self.assertEqual(lock["source"]["ref_kind"], "private_snapshot_commit")
        git = subprocess.run(["git", "-C", str(Path(lock["snapshot"]) / "source.git"), "show", lock["source"]["ref"] + ":kernel.py"], capture_output=True)
        self.assertEqual(git.returncode, 0)
        self.assertEqual(git.stdout, (fixture.source / "kernel.py").read_bytes())


class InputAndRangeTests(unittest.TestCase):
    def test_catalog_resolves_selected_implementation_not_production(self):
        tree, binding, files = resolve_request(REPO, operator_id="bf16_gemm")
        self.assertTrue(binding["entrypoint"]["path"].endswith("candidate/solution.py"))
        self.assertEqual(binding["entrypoint"]["symbol"], "Model.forward")
        self.assertTrue(binding["reference"]["path"].endswith("problem/reference.py"))
        self.assertIn(binding["entrypoint"]["path"], files)
        self.assertIn(b"def forward", tree.read(binding["entrypoint"]["path"]))

    def test_historical_fa4_entry_and_current_rules_are_frozen_separately(self):
        fixture = Fixture()
        self.addCleanup(fixture.cleanup)
        commit = "01fff90d714abc6e84904ea9c44523ba66d34a0e"
        spec = {"format": "schema_extract.source.v2", "id": "fa4_prefill", "repository": str(REPO),
                "revision": commit, "directory": "kernel_wiki/aka-four-problems-20261019/fa4_prefill",
                "entrypoint": "aka_solution/solution.py::Model.forward",
                "host": "aka_solution/flash_attn/cute/interface.py::_flash_attn_fwd",
                "io": ["metadata.json"], "input": "input.py::_make_inputs",
                "reference": "reference.py::Model.forward",
                "acceptance": ["metadata.json", "aka_solution/AGATE_VALIDATION.json"],
                "scope": ["shapes.json", "shape_train.json", "shape_valid.json"]}
        descriptor = fixture.base / "historical.json"
        descriptor.write_text(json.dumps(spec))
        tree, binding, names = resolve_request(REPO, source=str(descriptor))
        self.assertEqual(tree.commit, commit)
        self.assertTrue(binding["entrypoint"]["path"].endswith("fa4_prefill/aka_solution/solution.py"))
        self.assertIn(b"from flash_attn.cute.interface import _flash_attn_fwd", tree.read(binding["entrypoint"]["path"]))
        self.assertIn(binding["host"]["path"], names)
        from schema_extract.runner import create_run
        _, _, lock = create_run(REPO, fixture.base / "history-state", fixture.base / "history-output",
                                source=str(descriptor), config={"codex_binary": str(fixture.fake)})
        self.assertEqual(lock["source"]["ref"], commit)
        self.assertEqual(lock["source"]["dirty_files"], [])
        self.assertEqual(lock["rules"]["files"]["schema_extract/runner.py"], sha256((REPO / "schema_extract/runner.py").read_bytes()))
        verify_snapshot(lock)

    def test_missing_roles_are_reported_concretely(self):
        fixture = Fixture()
        self.addCleanup(fixture.cleanup)
        spec = read_json(fixture.descriptor)
        spec.pop("reference")
        fixture.descriptor.write_text(json.dumps(spec))
        with self.assertRaisesRegex(ExtractError, "reference"):
            resolve_request(REPO, source=str(fixture.source))

    def test_range_pagination_is_bounded_lossless_and_rejects_stale_cursor(self):
        fixture = Fixture()
        self.addCleanup(fixture.cleanup)
        text = '中文和"引号" ' * 1100 + "\nsecond line\n"
        (fixture.source / "long.txt").write_text(text)
        fragments, cursor = [], None
        while True:
            page = read_range(fixture.source, "long.txt", 1, 2, max_bytes=1024, cursor=cursor)
            self.assertLessEqual(len(encoded(page)), 1024)
            fragments += [entry["text"] for entry in page["fragments"]]
            cursor = page["next_cursor"]
            if cursor is None:
                break
        self.assertEqual("".join(fragments), text)
        first = read_range(fixture.source, "long.txt", 1, 2, max_bytes=1024)
        (fixture.source / "long.txt").write_text(text + "changed\n")
        with self.assertRaisesRegex(ValueError, "stale"):
            read_range(fixture.source, "long.txt", 1, 2, max_bytes=1024, cursor=first["next_cursor"])

    def test_p2_protection_uses_exact_decimal_checker_semantics(self):
        fixture = Fixture()
        self.addCleanup(fixture.cleanup)
        before, after = fixture.base / "before", fixture.base / "after"
        before.mkdir()
        after.mkdir()
        starter = REPO / "schema_template/starter.yaml"
        content = starter.read_text().replace("tolerance: {Output: null}", "tolerance: {Output: 0.100000000000000005}")
        (before / "root.yaml").write_text(content)
        (after / "root.yaml").write_text(content.replace("0.100000000000000005", "0.100000000000000006").replace("version: 1", "version: 2"))
        with self.assertRaisesRegex(ExtractError, "P2 changed Contract"):
            validate_artifacts("P2", "worker", before, after, starter, decision="continue")
