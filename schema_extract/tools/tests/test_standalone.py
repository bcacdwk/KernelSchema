"""A standalone reading view expands Contract without importing a parent's Design."""

from copy import deepcopy
from decimal import Decimal
import importlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from schema_extract.tools.artifacts import load_yaml, parser_for, write_yaml
from schema_extract.tools.common import ExtractError, manifest, read_json
from schema_extract.tools.standalone import create_view


REPO = Path(__file__).resolve().parents[3]


class StandaloneTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.task = self.root / "task"
        self.task.mkdir()
        fixture = load_yaml(REPO / "schema_template/tools/tests/fixtures/minimal", "root.yaml")
        fixture["meta_data"]["dispatch"] = {"when": "Select exactly when X >= 0."}
        fixture["contract"]["algorithm_requirement"]["constant"] = {
            "Offset": {"value": Decimal("0.10000000000000000000000000001"), "meaning": "Fixed synthetic shift."}}
        fixture["contract"]["algorithm_requirement"]["compute"][0]["math"] = "Y = X + Offset."
        fixture["design"]["math_reshape"]["rewrite"][0]["math"] = "Y = X + Offset."
        write_yaml(self.task / "root.yaml", fixture)
        branch = deepcopy(fixture)
        branch.pop("contract")
        branch["meta_data"]["dispatch"] = {"id": "negative", "parent": "root", "when": "Select exactly when X < 0."}
        write_yaml(self.task / "negative.yaml", branch)
        self.original = manifest(self.task)
        parser = parser_for(str(REPO))
        self.schema = importlib.import_module(parser.__package__ + ".schema")

    def test_export_is_complete_exact_independent_and_idempotent(self):
        output = self.root / "view"
        first = create_view(self.task / "negative.yaml", output, REPO)
        second = create_view(self.task / "negative.yaml", output, REPO)
        self.assertEqual((first["result"], second["result"]), ("created", "reused"))
        _, view = self.schema.validate_standalone_file(output / "negative.yaml", skip_hardware=True)
        expected = load_yaml(self.task, "negative.yaml")
        expected["contract"] = load_yaml(self.task, "root.yaml")["contract"]
        self.assertEqual(view["negative"], expected)
        self.assertIsInstance(view["negative"]["contract"]["algorithm_requirement"]["constant"]["Offset"]["value"], Decimal)
        self.assertEqual(manifest(self.task), self.original)
        self.assertEqual(read_json(output / ".source.json")["source_files"], self.original)
        # The file remains checkable after both authoring files have disappeared.
        (self.task / "root.yaml").unlink()
        (self.task / "negative.yaml").unlink()
        self.schema.validate_standalone_file(output / "negative.yaml", skip_hardware=True)
        with self.assertRaises(self.schema.SchemaError):
            self.schema.validate_file(output / "negative.yaml", skip_hardware=True)

    def test_standalone_mode_does_not_hide_missing_contract_or_bad_structure(self):
        output = self.root / "view"
        create_view(self.task / "negative.yaml", output, REPO)
        path = output / "negative.yaml"
        original = load_yaml(output, path.name)
        variants = []
        missing = deepcopy(original)
        missing.pop("contract")
        variants.append(missing)
        unexpected = deepcopy(original)
        unexpected["meta_data"]["dispatch"]["new_rule"] = True
        variants.append(unexpected)
        dangling = deepcopy(original)
        dangling["design"]["work_schedule"]["flow"]["launch"][0]["start_after"] = "missing.kernel_done"
        variants.append(dangling)
        recursive = deepcopy(original)
        recursive["meta_data"]["dispatch"]["parent"] = "negative"
        variants.append(recursive)
        for data in variants:
            write_yaml(path, data)
            with self.assertRaises(self.schema.SchemaError):
                self.schema.validate_standalone_file(path, skip_hardware=True)

    def test_destination_conflicts_and_source_tree_are_preserved(self):
        existing = self.root / "existing"
        existing.mkdir()
        (existing / "keep.txt").write_text("user content")
        with self.assertRaises(ExtractError):
            create_view(self.task / "negative.yaml", existing, REPO)
        self.assertEqual((existing / "keep.txt").read_text(), "user content")
        with self.assertRaisesRegex(ExtractError, "outside the source"):
            create_view(self.task / "negative.yaml", self.task / "view", REPO)
        self.assertEqual(manifest(self.task), self.original)

    def test_public_commands_export_and_check_one_file(self):
        output = self.root / "view"
        command = [sys.executable, "-B", str(REPO / "schema_extract/extract.py"), "standalone",
                   str(self.task / "negative.yaml"), "--output", str(output), "--repo", str(REPO)]
        made = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(made.returncode, 0, made.stderr)
        checked = subprocess.run([sys.executable, "-B", str(REPO / "schema_template/check_yaml.py"),
                                  "--standalone", "--skip-hardware", str(output / "negative.yaml")],
                                 capture_output=True, text=True, timeout=30)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertIn("standalone view checked", checked.stdout)


if __name__ == "__main__":
    unittest.main()
