"""Prompt integration checks for the supported reader and verified scope."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from .. import reading
from ..prompts import GPU_POLICY, check_feedback, prompt_for, reading_instructions


class ProtocolTests(unittest.TestCase):
    def test_reader_example_is_executable_and_shell_quotes_both_roots(self):
        with tempfile.TemporaryDirectory(prefix="extract-prompt-") as temporary:
            root = Path(temporary).resolve()
            snapshot = root / "source $(touch INJECTED) 'with spaces'"
            workspace = root / "artifacts 'with spaces'"
            (snapshot / "schema_extract/tools").mkdir(parents=True)
            workspace.mkdir()
            shutil.copyfile(Path(reading.__file__), snapshot / "schema_extract/tools/reading.py")
            content = "真实来源\\\"\r\n" * 100
            (snapshot / "EXTRACT_SCHEMA.md").write_bytes(content.encode())
            instructions = reading_instructions({"snapshot": str(snapshot)}, workspace)
            line = next(line for line in instructions.splitlines() if line.startswith('store("schema_read",'))
            command, _ = json.JSONDecoder().raw_decode(line.split("command: ", 1)[1])
            result = subprocess.run(command, shell=True, executable="/bin/sh", cwd=workspace,
                                    capture_output=True, text=True, check=True)
            page = json.loads(result.stdout)
            self.assertLessEqual(reading.output_size(page), reading.MAX_PAGE_BYTES)
            self.assertEqual(str(snapshot / "EXTRACT_SCHEMA.md"), page["files"][0]["path"])
            self.assertEqual("真实来源\\\"\r\n", page["files"][0]["fragments"][0]["text"])
            self.assertEqual([], list(workspace.iterdir()))
            self.assertFalse((snapshot / "INJECTED").exists())
            self.assertIn("session_id", instructions)
            # Use the argument replacement from the prompt, not a separately
            # hardcoded quoting rule. Both the directory and the query contain
            # shell syntax; they must reach the reader as literal data.
            prefix = next(line.removeprefix("Command prefix: ") for line in instructions.splitlines()
                          if line.startswith("Command prefix: "))
            replacement, _ = json.JSONDecoder().raw_decode(instructions.split('JSON.stringify(requests).replaceAll("\'", ', 1)[1])
            query = [{"path": str(snapshot / "EXTRACT_SCHEMA.md"), "find": ["'$(touch INJECTED)`printf changed`"]}]
            argument = "'" + json.dumps(query, ensure_ascii=False).replace("'", replacement) + "'"
            result = subprocess.run(prefix + " --requests " + argument, shell=True, executable="/bin/sh",
                                    cwd=workspace, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout), reading.read_page(query, roots=[snapshot, workspace]))
            self.assertEqual([], list(workspace.iterdir()))

    def test_reuse_requires_current_context_and_the_correct_stage(self):
        state = {"key": "sample-key", "fingerprint": "fixed", "checkpoint": "cp-1",
                 "stage": "index", "node": "root", "review_scope": "final", "issues": [],
                 "dispatches": [{"id": "child", "parent": "root", "reason": "alternate"}],
                 "reviews": {"contract": {"complete": True}, "root": {"complete": True},
                             "child": {"complete": True}}}
        baseline = {"snapshot": "/fixed", "repository_id": "repo", "snapshot_commit": "commit",
                    "identity": {"entry": {"id": "sample", "paths": {}}}}
        context = {"kind": "index", "fingerprint": "fixed", "checkpoint": "cp-1"}
        full = prompt_for(state, baseline, Path("/work"), "reviewer")
        self.assertNotIn("FINAL INDEX REVIEW", full)
        focused = prompt_for(state, baseline, Path("/work"), "reviewer", review_context=context)
        self.assertIn("FINAL INDEX REVIEW", focused)
        self.assertNotIn("/schema_docs/3.design.math_reshape.md", focused)
        self.assertNotIn("/EXTRACT_SCHEMA.md", focused)
        self.assertNotIn("/schema_template/README.md", focused)
        self.assertIn("选择条件与索引", focused)
        self.assertIn(GPU_POLICY, focused)
        for changed in (dict(context, checkpoint="cp-old"), dict(context, fingerprint="old")):
            self.assertNotIn("FINAL INDEX REVIEW", prompt_for(state, baseline, Path("/work"), "reviewer", review_context=changed))
        single = dict(state, dispatches=[])
        self.assertNotIn("FINAL INDEX REVIEW", prompt_for(single, baseline, Path("/work"), "reviewer", review_context=context))
        contract = deepcopy(state)
        contract.update(stage="contract", review_scope="contract")
        self.assertNotIn("Verified index context", prompt_for(contract, baseline, Path("/work"), "worker", review_context=context))
        self.assertNotIn("Verified index context", prompt_for(contract, baseline, Path("/work"), "reviewer", review_context=context))


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


if __name__ == "__main__":
    unittest.main()
