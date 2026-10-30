"""Missing hardware is explicit, local to active schema_wiki, and never global."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from unittest.mock import patch

import check_yaml

from .. import check_policy, schema
from ..pipeline_graph import main as graph_main
from ..diagnostics import encode_value
from .support import ALGO, HARD, PHASE, ROLE, TaskCase, drop, put


class HardwarePolicyTests(TaskCase):
    def setUp(self):
        super().setUp()
        self.wiki = self.directory / "schema_wiki"
        self.active = self.wiki / "batch" / "task"
        self.active.mkdir(parents=True)
        self.policy = self.wiki / "check_policy.json"
        self.policy.write_text('{"allow_missing_hardware":true}\n')
        self.policy_root = patch.object(check_policy, "WIKI_ROOT", self.wiki)
        self.policy_root.start()
        self.addCleanup(self.policy_root.stop)

    def write_at(self, data, directory=None):
        directory = directory or self.active
        directory.mkdir(parents=True, exist_ok=True)
        file = directory / "root.yaml"
        file.write_text(encode_value(data["root"]))
        return file

    def test_missing_null_and_partial_hardware_are_read_only(self):
        partial = {"device": None, "arch": "", "execution": {"warp": None, "CTA": "<hardware@pending:CTA>"},
                   "feature": {}, "stack": {"language": "unknown: not selected"},
                   "wiki": {"hardware": None, "programming": "<wiki@pending:programming>"}}
        for value in (None, {}, "", partial):
            with self.subTest(value=value):
                data = deepcopy(self.base)
                put(data, HARD, value)
                file = self.write_at(data)
                before = (file.read_bytes(), file.stat().st_mtime_ns, self.policy.read_bytes())
                notices = []
                self.assertEqual(schema.validate_directory(self.active, notices=notices), data)
                self.assertEqual(len(notices), 1)
                self.assertIn("Hardware exemption", notices[0])
                self.assertEqual(before, (file.read_bytes(), file.stat().st_mtime_ns, self.policy.read_bytes()))
        data = deepcopy(self.base)
        drop(data, HARD)
        self.write_at(data)
        self.assertEqual(schema.validate_directory(self.active), data)

    def test_exemption_does_not_escape_active_wiki(self):
        data = deepcopy(self.base)
        drop(data, HARD)
        for directory in (self.directory / "outside", self.wiki / "archived" / "task"):
            with self.subTest(directory=directory):
                file = self.write_at(data, directory)
                (directory / "check_policy.json").write_text('{"allow_missing_hardware":true}')
                with self.assertRaisesRegex(schema.SchemaError, "hardware_specification"):
                    schema.validate_file(file)
        self.write_at(data)
        self.policy.write_text('{"allow_missing_hardware":false}')
        with self.assertRaises(schema.SchemaError):
            schema.validate_directory(self.active)
        self.policy.unlink()
        with self.assertRaises(schema.SchemaError):
            schema.validate_directory(self.active)

    def test_only_hardware_completeness_is_exempt(self):
        def missing(data):
            drop(data, HARD)
        for change, message in (
            (lambda d: drop(d, ALGO + ("compute",)), "compute"),
            (lambda d: put(d, PHASE + ("mapping", "worker", "unit_count"), 0), "literal"),
            (lambda d: put(d, ROLE + ("info", "member"), "warp[0:0]"), "nonempty"),
            (lambda d: put(d, ROLE + ("info", "mode"), "sync"), "field not allowed"),
        ):
            with self.subTest(message=message):
                data = deepcopy(self.base)
                missing(data)
                change(data)
                self.write_at(data)
                with self.assertRaisesRegex(schema.SchemaError, message):
                    schema.validate_directory(self.active)
        # Supplied hardware still has the current field names and value types.
        for hardware in ({"unexpected": "x"}, {"device": 123}, {"execution": []}):
            data = deepcopy(self.base)
            put(data, HARD, hardware)
            self.write_at(data)
            with self.assertRaises(schema.SchemaError):
                schema.validate_directory(self.active)

    def test_policy_errors_are_tool_errors_and_cli_reports_exemption(self):
        data = deepcopy(self.base)
        drop(data, HARD)
        file = self.write_at(data)
        for contents in ('{', '{"allow_missing_hardware":"true"}', '{"ignore_all":true}',
                         '{"allow_missing_hardware":true,"allow_missing_hardware":false}'):
            with self.subTest(contents=contents):
                self.policy.write_text(contents)
                with self.assertRaises(schema.ToolError):
                    schema.validate_file(file)
                output, errors = io.StringIO(), io.StringIO()
                with redirect_stdout(output), redirect_stderr(errors):
                    self.assertEqual(check_yaml.main([str(file)]), 2)
                self.assertNotIn("Traceback", errors.getvalue())
        self.policy.write_text('{"allow_missing_hardware":true}')
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            self.assertEqual(check_yaml.main([str(file)]), 0)
        self.assertIn("Valid:", output.getvalue())
        self.assertIn("Hardware exemption", errors.getvalue())

    def test_graph_report_keeps_the_hardware_exemption_visible(self):
        data = deepcopy(self.base)
        drop(data, HARD)
        file = self.write_at(data)
        output, errors = io.StringIO(), io.StringIO()
        figures = self.active / "figures"
        with redirect_stdout(output), redirect_stderr(errors):
            self.assertEqual(graph_main([str(file), "--format", "dot", "--out", str(figures)]), 0)
        report = json.loads((figures / "root.json").read_text())
        self.assertTrue(any("Hardware exemption" in note for note in report["notes"]))
        self.assertIn("Hardware exemption", errors.getvalue())
