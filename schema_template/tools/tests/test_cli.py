"""Cli behavior through the same task checker used by authors."""
from __future__ import annotations

from contextlib import redirect_stderr
from contextlib import redirect_stdout
from copy import deepcopy
from unittest.mock import patch
import io
import shutil
import stat
import subprocess
import sys
import check_yaml
from .. import schema
from .support import (TaskCase, LANE, PHASE, TEMPLATE_ROOT, add_child, drop, put)


class CliTests(TaskCase):
    def test_copied_starter_and_unfilled_holes_fail(self):
        directory = self.directory / "new"
        directory.mkdir()
        target = directory / "root.yaml"
        shutil.copyfile(TEMPLATE_ROOT / "starter.yaml", target)
        self.assertEqual([target], list(directory.iterdir()))
        self.assertEqual("root.yaml", target.name)
        self.assertTrue(target.stat().st_mode & stat.S_IWUSR)
        before = target.read_bytes()
        with self.assertRaises(schema.SchemaError):
            schema.validate_file(target)
        self.assertEqual(before, target.read_bytes())
        target.write_text(self.source)
        self.assertEqual(self.base, schema.validate_directory(directory))

    def test_cli_checks_the_whole_task_from_any_node(self):
        data = deepcopy(self.base)
        add_child(data)
        add_child(data, "tiny", "small")
        self.write_task(data)
        (self.directory / "notes.md").write_text("Task notes are not plan input.\n")
        for entry in ("root.yaml", "small.yaml", "tiny.yaml"):
            with self.subTest(entry=entry):
                output, errors = io.StringIO(), io.StringIO()
                before = self.snapshot()
                with redirect_stdout(output), redirect_stderr(errors):
                    status = check_yaml.main([str(self.directory / entry)])
                self.assertEqual(0, status, errors.getvalue())
                self.assertIn("3 YAML node(s)", output.getvalue())
                self.assertEqual(before, self.snapshot())
        put(data, ("tiny", "meta_data", "dispatch", "parent"), "missing")
        self.write_task(data)
        for entry in ("root.yaml", "small.yaml"):
            errors = io.StringIO()
            before = self.snapshot()
            with redirect_stderr(errors):
                self.assertEqual(1, check_yaml.main([str(self.directory / entry)]))
            self.assertIn("tiny.yaml", errors.getvalue())
            self.assertIn("meta_data.dispatch.parent", errors.getvalue())
            self.assertEqual(before, self.snapshot())
        data = deepcopy(self.base)
        add_child(data)
        # A structural/identity error in a sibling must also fail.
        put(data, ("small", "meta_data", "name"), "import failed")
        self.write_task(data)
        for entry in ("root.yaml", "small.yaml"):
            errors = io.StringIO()
            before = self.snapshot()
            with redirect_stderr(errors):
                self.assertEqual(1, check_yaml.main([str(self.directory / entry)]))
            self.assertIn("small.yaml", errors.getvalue())
            self.assertIn("meta_data.name", errors.getvalue())
            self.assertEqual(before, self.snapshot())

    def test_cli_external_cwd_spaces_yml_and_invalid_arguments(self):
        task = self.directory / "a task with spaces"
        task.mkdir()
        (task / "root.yml").write_text(self.source)
        command = [sys.executable, str(TEMPLATE_ROOT / "check_yaml.py")]
        before = (task / "root.yml").read_bytes()
        result = subprocess.run(command + ["a task with spaces/root.yml"], cwd=self.directory,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn(str(task), result.stdout)
        self.assertEqual(before, (task / "root.yml").read_bytes())
        for arguments in ([], [str(task)], [str(task / "absent.yaml")],
                          [str(task / "root.yml"), str(task / "root.yml")]):
            with self.subTest(arguments=arguments):
                result = subprocess.run(command + arguments, cwd=self.directory,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(2, result.returncode, result.stderr)
                self.assertNotIn("Valid:", result.stdout)
        (task / "root.yml").rename(task / "small.yml")
        result = subprocess.run(command + [str(task / "small.yml")], cwd=self.directory,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(1, result.returncode)
        self.assertIn("missing root", result.stderr)

    def test_author_diagnostics_keep_positions_and_explain_conditions(self):
        data = deepcopy(self.base)
        put(data, LANE + ("info", "mode"), "parallel")
        self.write_task(data)
        source = (self.directory / "root.yaml").read_text()
        with self.assertRaises(schema.SchemaError) as caught:
            schema.validate_file(self.directory / "root.yaml")
        message = str(caught.exception)
        self.assertIn("root.yaml:1:" + str(source.index('"parallel"') + 1), message)
        self.assertIn("expected one of 'sync', 'async', 'ordered'", message)
        self.assertIn('Actual: "parallel"', message)
        drop(data, LANE + ("info", "mode"))
        drop(data, PHASE + ("schedule", "worker_assignment", "preassigned"))
        self.write_task(data)
        with self.assertRaises(schema.SchemaError) as caught:
            schema.validate_file(self.directory / "root.yaml")
        message = str(caught.exception)
        self.assertIn("refill is none", message)
        self.assertIn("start+step or start+count", message)
        self.assertNotRegex(message, r"root\.yaml:\d+")

    def test_relation_diagnostics_keep_indices_and_shared_contract_source(self):
        data = deepcopy(self.base)
        put(data, PHASE + ("mapping", "worker", "count"), 2)
        put(data, PHASE + ("mapping", "member"), [{"x": [0, 2]}, {"x": [4, 2]}])
        self.assert_task_invalid(data, "mapping.member[1].x", "Actual: [4,2]")
        data = deepcopy(self.base)
        put(data, PHASE + ("decomposition", "partition"), {"x": {"size": 8, "tile": 1}})
        put(data, PHASE + ("schedule", "worktile_order"), [{"by": "x", "direction": "ascending", "group": "0"}])
        self.assert_task_invalid(data, "worktile_order[0].group")
        data = deepcopy(self.base)
        add_child(data)
        put(data, ("small",) + PHASE[1:] + ("decomposition", "target", "single_tile", "Y", "dtype"),
            "ref:contract.algorithm_requirement::Missing.dtype")
        self.write_task(data)
        with self.assertRaises(schema.SchemaError) as caught:
            schema.validate_file(self.directory / "root.yaml")
        self.assertIn("small.yaml", str(caught.exception))
        self.assertIn("Shared root: " + str(self.directory / "root.yaml"), str(caught.exception))

    def test_missing_parser_reports_one_setup_step_without_writing(self):
        import builtins
        original = builtins.__import__
        def without_yaml(name, *args, **kwargs):
            if name == "yaml":
                raise ModuleNotFoundError("yaml is unavailable", name="yaml")
            return original(name, *args, **kwargs)
        output, errors = io.StringIO(), io.StringIO()
        before = self.snapshot()
        with patch("builtins.__import__", side_effect=without_yaml), redirect_stdout(output), redirect_stderr(errors):
            self.assertEqual(2, check_yaml.main([str(self.directory / "root.yaml")]))
        self.assertIn("pip install -r", errors.getvalue())
        self.assertIn("requirements.txt", errors.getvalue())
        self.assertNotIn("Valid:", output.getvalue())
        self.assertEqual(before, self.snapshot())
