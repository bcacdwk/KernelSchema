"""A concurrent branch's diagnostic must not point at its sibling's attempt."""

import unittest
from unittest.mock import patch

from schema_extract.tools.monitor import classify


class MonitorEvidenceTests(unittest.TestCase):
    def run_view(self):
        return {"run_id": "operator", "state_directory": "/runtime/runs/operator", "stage": "P5",
                "status": "running", "runner_observation": {"lock": "held", "alive": True},
                "last_attempt_id": "healthy-sibling", "node_details": {
                    "first": {"reason": {"code": "proposal_rejected", "summary": "First branch failed.",
                                          "attempt_id": "failed-first"}},
                    "second": {"reason": {"code": "proposal_rejected", "summary": "Second branch failed.",
                                           "attempt_id": "failed-second"}}}}

    def test_parallel_failures_retain_their_own_attempts(self):
        observations = {item["scope"]: item for item in classify(self.run_view())}
        self.assertEqual(set(observations), {"P5:first", "P5:second"})
        for name in ("first", "second"):
            self.assertEqual(observations["P5:" + name]["engineering_handoff"]["read_only_evidence"],
                             ["/runtime/runs/operator/state.json", "/runtime/runs/operator/attempts/failed-" + name])

    def test_branch_without_identity_does_not_guess_the_latest_attempt(self):
        view = self.run_view()
        for detail in view["node_details"].values():
            detail["reason"].pop("attempt_id")
        for item in classify(view):
            self.assertEqual(item["engineering_handoff"]["read_only_evidence"], ["/runtime/runs/operator/state.json"])

    def test_sequential_failure_uses_an_available_last_attempt(self):
        view = self.run_view()
        view.update(stage="P4", status="invalid_receipt", node_details={},
                    reason={"code": "proposal_rejected", "summary": "Root failed."})
        for latest in ("failed-root", None):
            view["last_attempt_id"] = latest
            expected = ["/runtime/runs/operator/state.json"]
            if latest:
                expected.append("/runtime/runs/operator/attempts/" + latest)
            self.assertEqual(classify(view)[0]["engineering_handoff"]["read_only_evidence"], expected)


class MonitorLaunchTests(unittest.TestCase):
    def test_exited_background_before_claim_does_not_reuse_living_launcher(self):
        from schema_extract.tools import monitor
        from schema_extract.tools.tests import test_stage_flow

        f = test_stage_flow.StageFlowTests(methodName="runTest")
        f.setUp()
        self.addCleanup(f.doCleanups)
        f.create()
        f.state["status"] = "queued"
        f.store.save(f.state)
        popen, children = monitor.subprocess.Popen, []

        def failed_start(command, **options):
            if "_background" not in command:
                return popen(command, **options)
            child = popen(["/usr/bin/false"], **options)
            children.append(child)
            child.wait(timeout=5)
            return child

        with patch.object(monitor.subprocess, "Popen", side_effect=failed_start):
            launch = monitor.start_background(f.store, f.state["run_id"], test_stage_flow.REPO)
        self.assertEqual(children[0].returncode, 1)
        self.assertEqual((launch["status"], launch.get("exit_code")), ("launch_failed", 1))
        observed = monitor.inspect_runs(f.store, [f.state["run_id"]])[0]
        self.assertTrue(any(item["scope"] == "startup" and item["classification"] == "execution_failure"
                            for item in observed["anomalies"]))
        state = f.store.load(f.state["run_id"])
        retry = monitor.reserve_launch(f.store, state, mode="queued", background=True)
        self.assertFalse(retry.get("reused", False))
        self.assertNotEqual(retry["id"], launch["id"])
        self.assertEqual((state["status"], state["stages"]["P1"]["calls"], state["attempts"]), ("queued", 0, []))


if __name__ == "__main__":
    unittest.main()
