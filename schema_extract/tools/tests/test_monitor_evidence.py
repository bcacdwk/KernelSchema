"""A concurrent branch's diagnostic must not point at its sibling's attempt."""

import unittest

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

    def test_legacy_branch_without_identity_does_not_guess_the_latest_attempt(self):
        view = self.run_view()
        for detail in view["node_details"].values():
            detail["reason"].pop("attempt_id")
        for item in classify(view):
            self.assertEqual(item["engineering_handoff"]["read_only_evidence"], ["/runtime/runs/operator/state.json"])

    def test_sequential_legacy_failure_uses_an_available_last_attempt(self):
        view = self.run_view()
        view.update(stage="P4", status="invalid_receipt", node_details={},
                    reason={"code": "proposal_rejected", "summary": "Root failed."})
        for latest in ("failed-root", None):
            view["last_attempt_id"] = latest
            expected = ["/runtime/runs/operator/state.json"]
            if latest:
                expected.append("/runtime/runs/operator/attempts/" + latest)
            self.assertEqual(classify(view)[0]["engineering_handoff"]["read_only_evidence"], expected)


if __name__ == "__main__":
    unittest.main()
