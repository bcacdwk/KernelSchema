"""C review provenance and tree boundaries retained under dependency scheduling."""

from copy import deepcopy
from pathlib import Path
import unittest

from schema_extract.agent import CodexCLI
from schema_extract.common import ExtractError
from schema_extract.runner import run
from test_c_independent import IndependentFixture


class TreeBoundaryTests(unittest.TestCase):
    def fixture(self, **kwargs):
        fixture = IndependentFixture(**kwargs)
        self.addCleanup(fixture.cleanup)
        return fixture

    def test_legal_tree_can_exceed_old_total_32_call_guard(self):
        fixture = self.fixture(pairs=tuple((f"branch_{i}", "root") for i in range(12)))
        state = fixture.run()
        self.assertEqual(state["status"], "ready", state["reason"])
        self.assertGreater(len(state["attempts"]), 32)
        self.assertEqual(len(state["nodes"]), 13)
        for node_id, node in state["nodes"].items():
            if node_id != "root":
                self.assertEqual(node["stage"]["calls"], 2)
                self.assertEqual(node["stage"]["revisions"], 0)

    def test_blocked_branch_waits_with_descendants_while_sibling_finishes(self):
        fixture = self.fixture(pairs=(("blocked_path", "root"), ("child", "blocked_path"), ("sibling", "root")),
            actions={"P5.blocked_path.reviewer.normal.1": "blocked"})
        state = fixture.run()
        self.assertEqual(state["status"], "blocked", state["reason"])
        self.assertEqual(state["nodes"]["sibling"]["status"], "approved")
        self.assertEqual(state["nodes"]["child"]["status"], "pending")
        self.assertFalse(fixture.calls(stage="P5", node="child"))
        self.assertFalse(fixture.calls(stage="P6"))
        merged, _ = fixture.store.verify_checkpoint(state, state["merged_checkpoint"])
        self.assertTrue((merged / "sibling.yaml").exists())
        self.assertFalse((merged / "blocked_path.yaml").exists())

    def test_unconfirmed_node_call_keeps_its_fence_while_sibling_finishes(self):
        fixture = self.fixture(pairs=(("first", "root"), ("sibling", "root")))
        fixture.create()
        cli = CodexCLI(str(fixture.fake))

        class UnconfirmedNodeCLI:
            def run(self, *, attempt, **kwargs):
                result = cli.run(attempt=attempt, **kwargs)
                if attempt["stage"] == "P5" and attempt["node"] == "first":
                    result["termination"]["confirmed"] = False
                return result

        state = run(fixture.store, fixture.state["run_id"], transport=UnconfirmedNodeCLI())
        self.assertEqual(state["status"], "termination_unconfirmed")
        self.assertEqual(set(state["active_attempts"]), {"P5:first"})
        self.assertIn(state["active_attempts"]["P5:first"], state["unconfirmed_attempts"])
        self.assertEqual(len(fixture.calls(stage="P5", node="first")), 1)
        self.assertEqual(state["nodes"]["sibling"]["status"], "approved")
        self.assertEqual(len(fixture.calls(stage="P5", node="sibling")), 2)
        self.assertFalse(fixture.calls(stage="P6"))
        attempt_id = state["active_attempts"]["P5:first"]
        directory = fixture.store.run_dir(state["run_id"]) / "attempts" / attempt_id
        process_path = directory / "process.json"
        original_process = process_path.read_bytes()
        self.assertTrue((directory / "launch.json").is_file())
        process_path.unlink()
        still_fenced = run(fixture.store, state["run_id"], transport=cli)
        self.assertEqual(still_fenced["status"], "termination_unconfirmed")
        self.assertEqual(still_fenced["active_attempts"], {"P5:first": attempt_id})
        self.assertEqual(len(fixture.calls(stage="P5", node="first")), 1)
        self.assertEqual(len(fixture.calls(stage="P5", node="sibling")), 2)
        process_path.write_bytes(original_process)
        recovered = run(fixture.store, state["run_id"], transport=cli)
        self.assertEqual(recovered["status"], "ready", recovered.get("reason"))
        self.assertTrue(recovered["published"])
        self.assertEqual(len(fixture.calls(stage="P5", node="first", role="worker")), 1)
        self.assertEqual(len(fixture.calls(stage="P5", node="first", role="reviewer")), 1)
        self.assertEqual(len(fixture.calls(stage="P5", node="sibling")), 2)
        self.assertFalse(recovered["active_attempts"])

    def test_final_start_requires_node_receipt_and_checkpoint_binding(self):
        fixture = self.fixture(pairs=(("child", "root"),))
        state = fixture.run()
        self.assertEqual(state["status"], "ready", state["reason"])
        # Reenter the decision boundary with a corrupted saved approval; no model
        # may turn a completion marker into a replacement for its actual receipt.
        changed = deepcopy(state)
        changed.update(status="running", role="reviewer")
        child = changed["nodes"]["child"]
        changed["accepted_receipts"].pop(child["approval"]["attempt_id"])
        fixture.store.save(changed)
        prior_calls = len(fixture.calls())
        result = run(fixture.store, state["run_id"])
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["reason"]["code"], "invalid_final_prerequisites")
        self.assertEqual(len(fixture.calls()), prior_calls)
        self.assertEqual(result["checkpoint"], state["checkpoint"])


if __name__ == "__main__":
    unittest.main()
