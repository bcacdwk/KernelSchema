"""Review examples exercise semantics outside structural checking, without an LLM."""

from pathlib import Path
import tempfile
import unittest

from schema_extract.tools.artifacts import check_complete, write_yaml
from schema_extract.tools.review_eval import cases_from_suite, grade_response
from schema_extract.tools.stages import DOCUMENT_CONCLUSIONS


REPO = Path(__file__).resolve().parents[3]


class ReviewEvaluationTests(unittest.TestCase):
    def test_every_case_is_structurally_valid_and_suite_has_positive_controls(self):
        suite, cases = cases_from_suite(REPO / "schema_extract/review_cases/suite.json")
        self.assertEqual(sum(case["expected"]["decision"] == "approve" for case in suite["cases"]), 4)
        self.assertEqual(sum(case["expected"]["decision"] == "revise" for case in suite["cases"]), 6)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for identity, value in cases.items():
                with self.subTest(case=identity):
                    write_yaml(directory / "root.yaml", value)
                    result = check_complete(directory, REPO)
                    self.assertEqual(result["exit_code"], 0, result["stderr"])

    def test_grading_rejects_blank_coverage_wrong_locations_and_reject_everything(self):
        conclusions = [{"subject": key, "text": "Current document assessed."} for key in DOCUMENT_CONCLUSIONS]
        good = {"decision": "approve", "conclusions": conclusions, "issues": []}
        expected_good = {"decision": "approve", "criteria": [], "path": ""}
        expected_bad = {"decision": "revise", "criteria": ["document_consistency"], "path": "design.work_schedule"}
        self.assertTrue(grade_response(good, expected_good)["passed"])
        self.assertFalse(grade_response(good, expected_bad)["passed"])
        self.assertFalse(grade_response({**good, "conclusions": []}, expected_good)["passed"])
        bad = {**good, "decision": "revise", "issues": [
            {"criterion": "document_consistency", "path": "design.work_schedule.flow", "detail": "A submitted phase is cancelled by prose."}]}
        self.assertTrue(grade_response(bad, expected_bad)["passed"])
        self.assertFalse(grade_response(bad, expected_good)["passed"])
        bad["issues"][0]["path"] = "meta_data.describe"
        self.assertFalse(grade_response(bad, expected_bad)["passed"])
        bad["issues"][0]["path"] = ""
        self.assertFalse(grade_response(bad, expected_bad)["passed"])


if __name__ == "__main__":
    unittest.main()
