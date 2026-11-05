"""Standalone exports use validated effective Design and complete raw provenance."""

from copy import deepcopy
from decimal import Decimal
import importlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from schema_extract.tools.artifacts import load_yaml, parser_for, write_yaml
from schema_extract.tools.common import ExtractError, manifest, read_json
from schema_extract.tools import standalone
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
        receipt = read_json(output / ".source.json")
        resolved = self.schema.resolve_directory(self.task)
        self.assertEqual(receipt["format"], "schema_extract.standalone.v2")
        self.assertEqual(receipt["source_files"], self.original)
        self.assertEqual(receipt["origins"], resolved.origins_dict("negative"))
        self.assertEqual(receipt["effective_identity"], resolved.effective_identity("negative"))
        # The file remains checkable after both authoring files have disappeared.
        (self.task / "root.yaml").unlink()
        (self.task / "negative.yaml").unlink()
        self.schema.validate_standalone_file(output / "negative.yaml", skip_hardware=True)
        with self.assertRaises(self.schema.SchemaError):
            self.schema.validate_file(output / "negative.yaml", skip_hardware=True)

    def derived_chain(self):
        """Two inherited regions cross an intermediate node; schedule stays local."""
        branch = load_yaml(self.task, "negative.yaml")
        provider = deepcopy(branch)
        provider["meta_data"]["dispatch"] = {"id": "provider", "parent": "root",
                                                 "when": "Select exactly when X < -20."}
        provider["design"]["math_reshape"]["thesis"] = "Share the exact offset computation."
        provider["realization"] = {"require": [{"item": "design.math_reshape::Y",
                                                  "choice": "Preserve the offset exactly.",
                                                  "reason": "Provider implementation constraint."}]}
        write_yaml(self.task / "provider.yml", provider)
        middle = deepcopy(provider)
        middle["meta_data"]["dispatch"] = {"id": "middle", "parent": "provider",
                                               "when": "Select exactly when -20 <= X < -10."}
        middle["design"] = {region: {"from_parent": "provider"} for region in provider["design"]}
        middle.pop("realization")
        write_yaml(self.task / "middle.yaml", middle)
        branch["meta_data"]["dispatch"]["parent"] = "middle"
        branch["meta_data"]["dispatch"]["when"] = "Select exactly when -10 <= X < 0."
        branch["design"]["math_reshape"] = {"from_parent": "middle"}
        branch["design"]["tile_pipeline"] = {"from_parent": "middle"}
        branch["design"]["work_schedule"]["thesis"] = "Use the current child's local work schedule."
        write_yaml(self.task / "negative.yaml", branch)
        return self.schema.resolve_directory(self.task)

    def test_transitive_export_is_local_equivalent_and_independent(self):
        resolved = self.derived_chain()
        original = manifest(self.task)
        output = self.root / "derived-view"
        made = create_view(self.task / "negative.yaml", output, REPO)
        self.assertEqual(made["source_files"], original)
        self.assertEqual(made["source"], "negative.yaml")
        self.assertEqual(made["root"], "root.yaml")
        self.assertEqual(made["origins"]["math_reshape"], {
            "provider": "provider", "chain": ["negative", "middle", "provider"]})
        self.assertEqual(made["origins"]["work_schedule"], {
            "provider": "negative", "chain": ["negative"]})
        self.assertEqual(made["effective_identity"], resolved.effective_identity("negative"))
        _, checked = self.schema.validate_standalone_file(output / "negative.yaml", skip_hardware=True)
        self.assertEqual(checked["negative"], resolved.effective_document("negative"))
        self.assertNotIn("realization", checked["negative"])
        for region in checked["negative"]["design"].values():
            self.assertNotIn("from_parent", region)
        self.assertEqual(manifest(self.task), original)

        inline_task = self.root / "inline-task"
        inline_task.mkdir()
        for node, data in resolved.effective_task.items():
            write_yaml(inline_task / resolved.source_files[node].name, data)
        inline_output = self.root / "inline-view"
        inline = create_view(inline_task / "negative.yaml", inline_output, REPO)
        self.assertEqual(load_yaml(output, "negative.yaml"), load_yaml(inline_output, "negative.yaml"))
        self.assertEqual(set(inline["source_files"]), {"root.yaml", "negative.yaml"})
        self.assertNotEqual(inline["effective_identity"], made["effective_identity"])

        self.task.rename(self.root / "authoring-moved")
        self.schema.validate_standalone_file(output / "negative.yaml", skip_hardware=True)

    def test_every_dependency_is_rechecked_before_generation_and_publication(self):
        self.derived_chain()
        for source_name in ("root.yaml", "provider.yml", "middle.yaml", "negative.yaml"):
            for boundary in ("validated", "generated", "synced"):
                with self.subTest(source=source_name, boundary=boundary):
                    source = self.task / source_name
                    original = source.read_bytes()
                    output = self.root / (source_name + "-" + boundary)

                    def drift(function):
                        def changed(*args, **kwargs):
                            result = function(*args, **kwargs)
                            source.write_bytes(original + b"\n")
                            return result
                        return changed

                    if boundary == "validated":
                        target, method = self.schema, "validate_resolved_file"
                    elif boundary == "generated":
                        target, method = self.schema, "validate_standalone_file"
                    else:
                        target, method = standalone, "sync_tree"
                    try:
                        with patch.object(target, method, side_effect=drift(getattr(target, method))):
                            with self.assertRaisesRegex(ExtractError, "source changed.*" + source_name):
                                create_view(self.task / "negative.yaml", output, REPO)
                        self.assertFalse(output.exists())
                        self.assertEqual(list(self.root.glob(".standalone-*")), [])
                    finally:
                        source.write_bytes(original)

    def test_provider_change_invalidates_owned_output_even_with_identical_effective_content(self):
        self.derived_chain()
        output = self.root / "view"
        first = create_view(self.task / "negative.yaml", output, REPO)
        before = manifest(output)
        provider = self.task / "provider.yml"
        provider.write_bytes(provider.read_bytes() + b"\n")
        with self.assertRaisesRegex(ExtractError, "different content"):
            create_view(self.task / "negative.yaml", output, REPO)
        self.assertEqual(manifest(output), before)
        second = create_view(self.task / "negative.yaml", self.root / "updated-view", REPO)
        self.assertNotEqual(first["effective_identity"], second["effective_identity"])
        self.assertEqual(first["view"], second["view"])

    def test_reuse_rechecks_sources_after_comparing_owned_output(self):
        self.derived_chain()
        output = self.root / "view"
        create_view(self.task / "negative.yaml", output, REPO)
        original = manifest(output)
        provider = self.task / "provider.yml"
        actual_read = standalone.read_json

        def changed(path):
            result = actual_read(path)
            provider.write_bytes(provider.read_bytes() + b"\n")
            return result

        with patch.object(standalone, "read_json", side_effect=changed):
            with self.assertRaisesRegex(ExtractError, "source changed.*provider.yml"):
                create_view(self.task / "negative.yaml", output, REPO)
        self.assertEqual(manifest(output), original)

    def test_resolvable_unfinished_or_invalid_task_cannot_be_exported(self):
        self.derived_chain()
        child = load_yaml(self.task, "negative.yaml")
        child["meta_data"]["dispatch"]["when"] = ""
        write_yaml(self.task / "negative.yaml", child)
        self.schema.resolve_directory(self.task)
        output = self.root / "unfinished"
        with self.assertRaises(ExtractError):
            create_view(self.task / "negative.yaml", output, REPO)
        self.assertFalse(output.exists())

    def test_old_rules_use_the_selected_frozen_exporter(self):
        frozen = self.root / "old-rules"
        executor = frozen / "schema_extract"
        (executor / "tools").mkdir(parents=True)
        (executor / "tools/standalone.py").write_text("# Frozen exporter entry lives in this package.\n")
        # A minimal historical entry reports its own result. No current resolver
        # may be called to reinterpret the selected old rules.
        (executor / "extract.py").write_text(
            "import json, pathlib, sys\n"
            "assert sys.argv[1] == 'standalone'\n"
            "assert pathlib.Path(sys.argv[-1]).resolve() == pathlib.Path.cwd()\n"
            "print(json.dumps({'format': 'schema_extract.standalone.v1', "
            "'result': 'created', 'source': sys.argv[2], 'output': sys.argv[4]}))\n")
        output = self.root / "old-view"
        with patch.object(standalone, "schema_for", return_value=object()) as selected:
            result = create_view(self.task / "negative.yaml", output, frozen)
        selected.assert_called_once_with(frozen)
        self.assertEqual(result, {"format": "schema_extract.standalone.v1", "result": "created",
                                  "source": str(self.task / "negative.yaml"), "output": str(output)})
        self.assertEqual(manifest(self.task), self.original)

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
        inherited = deepcopy(original)
        inherited["design"]["math_reshape"] = {"from_parent": "root"}
        variants.append(inherited)
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
