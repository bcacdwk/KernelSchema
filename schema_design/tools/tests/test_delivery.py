"""Candidate custody and native protocol boundaries; no real model claims."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from schema_design.tools import runner as design
from schema_design.tools.tests.test_design import schema_files


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="schema-delivery-cpu-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.task = self.base / "task.md"
        self.task.write_text("A protocol fixture, not a hardware claim.")

    def test_partial_timeout_is_copied_then_repaired_without_overwriting_raw_draft(self):
        out = self.base / "run"
        files = schema_files()
        root = files[0]
        seen = []
        def native(args, prompt, schema, directory, draft_path, seed):
            candidate = directory / seed["candidate_directory"]
            seen.append(candidate)
            if len(seen) == 1:
                (candidate / root["path"]).write_text(root["content"])
                answer, diagnostics = None, ["timeout; process group reaped"]
            else:
                self.assertEqual((candidate / root["path"]).read_text(), root["content"])
                self.assertIn("Previous candidate files", prompt)
                self.assertNotIn(root["content"], prompt)
                for item in files:
                    (candidate / item["path"]).write_text(item["content"])
                answer, diagnostics = {"files": [item["path"] for item in files], "notes": "complete", "missing": []}, []
            return {**seed, "structured_output": answer, "diagnostics": diagnostics,
                    "reads": [{"tool": "Read", "status": "completed", "input": {
                        "file_path": str(directory / "input/schema_docs/3.design.math_reshape.md")}}]}
        with patch.object(design, "native_call", native), patch.object(design, "discover", return_value=[]), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            code = design.main(["--task", str(self.task), "--out", str(out), "--max-calls", "2"])
        self.assertEqual(code, 0, (out / "notes.md").read_text())
        self.assertEqual((seen[0] / root["path"]).read_text(), root["content"])
        self.assertFalse((seen[0] / "dispatch.md").exists())
        self.assertEqual(json.loads((out / "result.json").read_text())["calls_used"], 2)
        final = json.loads((out / "drafts/02.json").read_text())
        self.assertEqual(final["structured_output"]["files"], [item["path"] for item in files])
        self.assertTrue(final["candidate_files"])

    def test_fixed_input_mutation_is_rejected_before_delivery(self):
        out = self.base / "mutated"
        def native(args, prompt, schema, directory, draft_path, seed):
            (directory / "input/task/task.md").write_text("tampered")
            return {**seed, "structured_output": {"files": [], "notes": "ignored", "missing": []}, "diagnostics": []}
        with patch.object(design, "native_call", native), patch.object(design, "discover", return_value=[]), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            code = design.main(["--task", str(self.task), "--out", str(out), "--max-calls", "2"])
        result = json.loads((out / "result.json").read_text())
        self.assertEqual((code, result["calls_used"], result["stop_reason"], result["schema"]), (1, 1, "input_changed", None))

    def test_candidate_links_and_unlisted_files_are_rejected_in_both_modules(self):
        for mode in ("design", "optimize"):
            runner = importlib.import_module(f"schema_{mode}.tools.runner")
            candidate = self.base / mode
            candidate.mkdir()
            (candidate / "root.yaml").symlink_to(self.task)
            with self.assertRaises(runner.Failure):
                runner.candidate_files(candidate)
            (candidate / "root.yaml").unlink()
            (candidate / "unexpected.txt").write_text("not a candidate")
            with self.assertRaises(runner.Failure):
                runner.candidate_files(candidate)
            (candidate / "unexpected.txt").unlink()
            (candidate / "nested").mkdir()
            (candidate / "nested/root.yaml").write_text("nested: rejected")
            with self.assertRaises(runner.Failure):
                runner.candidate_files(candidate)

    def test_permission_rules_scope_candidates_and_selection_has_no_write_tools(self):
        for mode in ("design", "optimize"):
            runner = importlib.import_module(f"schema_{mode}.tools.runner")
            args = SimpleNamespace(model=runner.DEFAULT_MODEL, effort="max")
            allowed = self.base / "candidate"
            command = runner.command(args, runner.WRITE_SCHEMA, [self.base / "input"], allowed)
            settings = json.loads(command[command.index("--settings") + 1])
            self.assertEqual(settings["permissions"]["allow"], ["Read", "Glob", "Grep"] + [
                f"{tool}(/{allowed}/{name})" for tool in ("Write", "Edit") for name in ("*.yaml", "*.yml", "dispatch.md")])
            self.assertNotIn("Bash", command[command.index("--tools") + 1].split(","))
            selection = runner.command(args, runner.WRITE_SCHEMA, [self.base / "input"])
            self.assertEqual(selection[selection.index("--tools") + 1], "Read,Glob,Grep")
            self.assertIn("Write", selection[selection.index("--disallowedTools") + 1].split(","))

    def test_public_errors_usage_and_write_paths_exclude_reasoning_and_file_payload(self):
        events = [
            {"type": "assistant", "message": {"model": "fixture", "stop_reason": "tool_use", "usage": {"output_tokens": 10}, "content": [
                {"type": "thinking", "thinking": "hidden-reasoning"},
                {"type": "tool_use", "id": "w", "name": "Write", "input": {"file_path": "/draft/root.yaml", "content": "yaml-payload"}}]}},
            {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "w", "is_error": True, "content": "permission denied"}]}},
            {"type": "error", "code": "provider-error", "message": "public provider message"},
            {"type": "result", "subtype": "error_during_execution", "is_error": True, "errors": ["public error"], "num_turns": 1, "usage": {"output_tokens": 10}},
        ]
        for mode in ("design", "optimize"):
            runner = importlib.import_module(f"schema_{mode}.tools.runner")
            draft = runner.parse_stream("\n".join(map(json.dumps, events)), "", 1)
            self.assertEqual(draft["writes"][0]["error"], "permission denied")
            self.assertEqual(draft["public_status"]["terminal"]["usage"]["output_tokens"], 10)
            self.assertIn("public provider message", json.dumps(draft))
            self.assertNotIn("hidden-reasoning", json.dumps(draft))
            self.assertNotIn("yaml-payload", json.dumps(draft))
