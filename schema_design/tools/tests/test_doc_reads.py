"""Public tool-result evidence at both local writers' actual acceptance gate."""
from contextlib import redirect_stderr
from io import StringIO
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


DOC = "5.design.tile_pipeline.md"
BODY = "meaning: private-doc-body-marker"
ANSWER = {"files": [], "notes": "Protocol fixture", "missing": []}


def stream(tool, arguments, content, *, error=False, returned=True):
    events = [{"type": "assistant", "message": {"content": [
        {"type": "thinking", "thinking": "private-thinking-marker"},
        {"type": "tool_use", "id": "read-doc", "name": tool, "input": arguments}]}}]
    if returned:
        events.append({"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "read-doc", "is_error": error, "content": content}]}})
    events.append({"type": "result", "subtype": "success", "is_error": False,
                   "result": "Protocol fixture", "structured_output": ANSWER})
    return "\n".join(map(json.dumps, events))


class DocReadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="doc-evidence-cpu-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.count = 0

    def run_fixture(self, runner):
        self.count += 1
        out = self.base / str(self.count)
        out.mkdir()
        run = runner.Run(SimpleNamespace(out=out, max_calls=2), "design")
        docs = out / "input/schema_docs"
        docs.mkdir()
        (docs / DOC).write_text(BODY)
        return run

    def call(self, runner, run, output, phase="write"):
        def native(args, prompt, schema, out, draft_path, seed):
            return {**seed, **runner.parse_stream(output, "", 0)}
        with patch.object(runner, "native_call", native), redirect_stderr(StringIO()):
            return run.call("Protocol fixture", runner.WRITE_SCHEMA, phase)

    def test_completed_read_keeps_existing_exact_path_rule(self):
        for mode in ("design", "optimize"):
            runner = importlib.import_module(f"schema_{mode}.tools.runner")
            for path, error, returned, accepted in (
                (f"input/schema_docs/{DOC}", False, True, True),
                (f"input/schema_docs/{DOC}", True, True, False),
                (f"input/schema_docs/{DOC}", False, False, False),
                (f"input/other/{DOC}", False, True, False),
            ):
                with self.subTest(mode=mode, path=path, error=error, returned=returned):
                    run = self.run_fixture(runner)
                    output = stream("Read", {"file_path": path}, "permission denied" if error else BODY,
                                    error=error, returned=returned)
                    if accepted:
                        self.assertEqual(self.call(runner, run, output), ANSWER)
                    else:
                        with self.assertRaises(runner.Failure):
                            self.call(runner, run, output)

    def test_grep_requires_returned_nonempty_body_and_persists_only_paths_and_counts(self):
        for mode in ("design", "optimize"):
            runner = importlib.import_module(f"schema_{mode}.tools.runner")
            cases = [
                ("numbered", "content", f"12:{BODY}\n13:second body", False, True, True),
                ("text blocks", "content", [{"type": "text", "text": f"12:{BODY}"}], False, True, True),
                ("numeric body", "content", "12:34:body", False, True, True),
                ("locator in body", "content", f"12:See /elsewhere/{DOC}:34:details", False, True, True),
                ("no matches", "content", "No matches found", False, True, False),
                ("empty", "content", "", False, True, False),
                ("whitespace body", "content", "12:  \t", False, True, False),
                ("context only", "content", f"12-{BODY}", False, True, False),
                ("unnumbered", "content", BODY, False, True, False),
                ("list mode", "files_with_matches", f"12:{BODY}", False, True, False),
                ("default list mode", None, DOC, False, True, False),
                ("count mode", "count", f"{DOC}:12", False, True, False),
                ("bare count", "content", "12", False, True, False),
                ("bare filename", "content", f"input/schema_docs/{DOC}", False, True, False),
                ("permission denied", "content", "permission denied", True, True, False),
                ("error with partial body", "content", f"12:{BODY}", True, True, False),
                ("request without result", "content", None, False, False, False),
            ]
            for label, output_mode, content, error, returned, accepted in cases:
                with self.subTest(mode=mode, case=label):
                    run = self.run_fixture(runner)
                    arguments = {"path": f"input/schema_docs/{DOC}", "pattern": "meaning|describe"}
                    if output_mode is not None:
                        arguments["output_mode"] = output_mode
                    output = stream("Grep", arguments, content, error=error, returned=returned)
                    if accepted:
                        self.assertEqual(self.call(runner, run, output), ANSWER)
                        saved = (run.out / "drafts/01.json").read_text()
                        self.assertNotIn(BODY, saved)
                        self.assertNotIn("private-thinking-marker", saved)
                        hits = json.loads(saved)["reads"][0]["content_hits"]
                        self.assertEqual(hits, [{"path": arguments["path"], "lines": 2 if label == "numbered" else 1}])
                    else:
                        with self.assertRaises(runner.Failure):
                            self.call(runner, run, output)

    def test_directory_grep_binds_actual_hit_to_doc_and_requested_scope(self):
        for mode in ("design", "optimize"):
            runner = importlib.import_module(f"schema_{mode}.tools.runner")
            for scope, returned_path, accepted in (
                ("input/schema_docs", f"input/schema_docs/{DOC}", True),
                ("input/schema_docs", DOC, True),
                (".", f"input/schema_docs/{DOC}", True),
                (None, f"input/schema_docs/{DOC}", True),
                ("input/schema_docs", None, False),
                ("input/schema_docs", "input/schema_docs/0.common.md", False),
                ("input/schema_docs", f"../other/{DOC}", False),
                ("input/task", f"input/schema_docs/{DOC}", False),
                (f"input/schema_docs/{DOC}", "input/schema_docs/1.contract.algorithm_requirement.md", False),
            ):
                with self.subTest(mode=mode, scope=scope, returned_path=returned_path):
                    run = self.run_fixture(runner)
                    arguments = {"pattern": "meaning", "output_mode": "content"}
                    if scope is not None:
                        arguments["path"] = scope
                    content = (f"{returned_path}:" if returned_path else "") + f"12:{BODY}"
                    output = stream("Grep", arguments, content)
                    if accepted:
                        self.assertEqual(self.call(runner, run, output), ANSWER)
                    else:
                        with self.assertRaises(runner.Failure):
                            self.call(runner, run, output)
            with self.subTest(mode=mode, scope="absolute"):
                run = self.run_fixture(runner)
                doc = run.out / "input/schema_docs" / DOC
                self.assertEqual(self.call(runner, run, stream("Grep", {
                    "path": str(doc.parent), "pattern": "meaning", "output_mode": "content"},
                    f"{doc}:12:{BODY}")), ANSWER)

    def test_repair_does_not_inherit_previous_call_read(self):
        for mode in ("design", "optimize"):
            with self.subTest(mode=mode):
                runner = importlib.import_module(f"schema_{mode}.tools.runner")
                run = self.run_fixture(runner)
                self.call(runner, run, stream("Read", {"file_path": f"input/schema_docs/{DOC}"}, BODY))
                with self.assertRaisesRegex(runner.Failure, "in this call"):
                    self.call(runner, run, stream("Grep", {"path": f"input/schema_docs/{DOC}",
                        "pattern": "meaning", "output_mode": "content"}, "No matches found"), phase="repair")
                self.assertEqual(run.result["calls_used"], 2)


if __name__ == "__main__":
    unittest.main()
