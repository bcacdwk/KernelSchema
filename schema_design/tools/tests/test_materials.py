"""Real local Git and snapshot boundaries; no model, network, or GPU calls."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class WritingMaterialTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="writing-materials-cpu-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "source-repository"
        self.repo.mkdir()
        self.git("init", "-q")
        (self.repo / "eval").mkdir()
        self.reference = self.repo / "eval" / "reference [cpu].py"
        self.reference.write_text("def add(x, y): return x + y\n")
        self.task = self.repo / "task.md"
        self.task.write_text("Use [reference](eval/reference [cpu].py) and [input](eval/input.py).\n")
        (self.repo / "eval/input.py").write_text("def inputs(): return 1, 2\n")
        self.git("add", "--", ".")
        self.git("-c", "user.name=CPU fixture", "-c", "user.email=fixture@testing.invalid", "commit", "-qm", "fixture")
        self.commit = self.git("rev-parse", "HEAD").strip()
        self.git("config", "remote.origin.url", "https://user:fake-remote-secret@testing.invalid/source")

    def git(self, *arguments):
        result = subprocess.run(["git", "-c", "init.templateDir=", "-c", "core.hooksPath=/dev/null",
                                 "-C", str(self.repo), *arguments], check=True, capture_output=True, text=True,
                                env={"PATH": os.environ["PATH"], "GIT_CONFIG_NOSYSTEM": "1",
                                     "GIT_CONFIG_GLOBAL": os.devnull, "GIT_TERMINAL_PROMPT": "0"})
        return result.stdout

    def run_materials(self, mode):
        runner = importlib.import_module(f"schema_{mode}.tools.runner")
        out = self.base / mode
        out.mkdir()
        run = runner.Run(SimpleNamespace(task=self.task, out=out), mode)
        return runner, run, run.preparation()

    def test_exact_commit_bytes_reach_manifest_and_request_without_remote_credentials(self):
        for mode in ("design", "optimize"):
            with self.subTest(mode=mode):
                # Caller Git overrides must not redirect verification to an unrelated repository.
                with patch.dict(os.environ, {"GIT_DIR": str(self.base / "wrong.git")}):
                    _, run, prompt = self.run_materials(mode)
                record = run.inputs.records[str(self.reference)]
                facts = record["git"]
                self.assertEqual(facts["status"], "verified")
                self.assertEqual(facts["commit"], self.commit)
                self.assertEqual(facts["path"], "eval/reference [cpu].py")
                self.assertEqual(facts["locator"], f"source-repository@{self.commit}:eval/reference [cpu].py")
                self.assertEqual(self.git("cat-file", "blob", facts["blob"]), self.reference.read_text())
                self.assertIn(facts["locator"], prompt)
                saved = json.loads((run.inputs.root / "manifest.json").read_text())
                self.assertIn(record, saved["files"])
                for path in run.out.rglob("*"):
                    if path.is_file():
                        self.assertNotIn(b"fake-remote-secret", path.read_bytes())
                self.assertNotIn("fake-remote-secret", prompt)
                self.assertFalse(any(".git" in Path(item["snapshot"]).parts for item in saved["files"]))

    def test_modified_and_uncommitted_bytes_are_not_given_committed_locators(self):
        self.reference.write_text("def add(x, y): return x - y\n")  # Same length, different bytes.
        extra = self.repo / "new.txt"
        extra.write_text("An uncommitted material\n")
        self.task.write_text(self.task.read_text() + "Read [new](new.txt).\n")
        before = self.git("status", "--porcelain")
        for mode in ("design", "optimize"):
            with self.subTest(mode=mode):
                _, run, prompt = self.run_materials(mode)
                modified = run.inputs.records[str(self.reference)]["git"]
                uncommitted = run.inputs.records[str(extra)]["git"]
                self.assertEqual(modified["status"], "different_bytes")
                self.assertEqual(uncommitted["status"], "absent_from_commit")
                self.assertNotIn("locator", modified)
                self.assertNotIn("locator", uncommitted)
                self.assertIn("return x - y", prompt)
                self.assertIn("An uncommitted material", prompt)
        self.assertEqual(self.git("status", "--porcelain"), before)
        self.assertEqual(self.git("rev-parse", "HEAD").strip(), self.commit)

    def test_doc_links_and_optional_example_use_only_fixed_readable_materials(self):
        for mode in ("design", "optimize"):
            with self.subTest(mode=mode):
                runner, run, prompt = self.run_materials(mode)
                template = run.inputs.root / "schema_template/README.md"
                algorithm = (template.parent / "../schema_docs/1.contract.algorithm_requirement.md").resolve()
                self.assertEqual(algorithm.read_bytes(), (runner.CHECKOUT / "schema_docs" / algorithm.name).read_bytes())
                method = run.inputs.root / f"{mode.upper()}_SCHEMA.md"
                self.assertTrue((method.parent / "schema_template/starter.yaml").is_file())
                example_source = runner.CHECKOUT / "schema_template/examples/qsa_prefill"
                for name in ("README.md", "root.yaml", "dispatch.md"):
                    record = run.inputs.records[str(example_source / name)]
                    fixed = run.out / record["snapshot"]
                    self.assertEqual(fixed.read_bytes(), (example_source / name).read_bytes())
                    self.assertIn(str(fixed), prompt)
                    self.assertNotIn(fixed.read_text(), prompt)
                self.assertFalse(any("schema_workflow/examples/vector_add/schema" in source
                                     for source in run.inputs.records))
                command = runner.command(SimpleNamespace(model=runner.DEFAULT_MODEL, effort="max"),
                                         runner.WRITE_SCHEMA, [run.inputs.root])
                readable = [command[index + 1] for index, value in enumerate(command) if value == "--add-dir"]
                self.assertEqual(readable, [str(run.inputs.root)])
                self.assertEqual(command[command.index("--tools") + 1], "Read,Glob,Grep")

    def test_unavailable_git_preserves_material_without_claiming_a_revision(self):
        for mode in ("design", "optimize"):
            with self.subTest(mode=mode):
                runner = importlib.import_module(f"schema_{mode}.tools.runner")
                out = self.base / mode
                out.mkdir()
                inputs = runner.Inputs(out)
                with patch.object(runner.subprocess, "run", side_effect=FileNotFoundError("git")):
                    fixed = inputs.file(self.reference)
                self.assertEqual(fixed.read_bytes(), self.reference.read_bytes())
                self.assertEqual(inputs.records[str(self.reference)]["git"], {"status": "unavailable"})
                self.assertIn("return x + y", inputs.text([fixed]))


if __name__ == "__main__":
    unittest.main()
