"""Current standalone views retain provenance and reject complete-source drift."""

from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from schema_extract.tools import standalone
from schema_extract.tools.artifacts import DESIGN_REGIONS, load_yaml, schema_for, write_yaml
from schema_extract.tools.common import ExtractError, digest, manifest, read_json
from schema_extract.tools.snapshots import rule_files
from schema_extract.tools.standalone import create_view


REPO = Path(__file__).resolve().parents[3]


class StandaloneTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.task = self.root / "raw"
        self.task.mkdir()
        root = load_yaml(REPO / "schema_template/tools/tests/fixtures/minimal", "root.yaml")
        root["META_DATA"]["Dispatch"] = {"when": "All scalar inputs"}
        root["CONTRACT"]["Algorithm_requirement"]["Constant"] = {
            "offset": {"value": Decimal("0.10000000000000000000000000001"), "meaning": "Synthetic shift"}}
        root["CONTRACT"]["Algorithm_requirement"]["Compute"][0]["Math"] = "y = x + offset"
        root["DESIGN"]["Math_reshape"]["Rewrite"][0]["Math"] = "y = x + offset"
        self.documents = {"root": root}
        for node in ("hidden", "provider", "middle", "child"):
            value = deepcopy(root)
            value.pop("CONTRACT")
            value["META_DATA"]["Dispatch"] = {"id": node, "when": "All scalar inputs"}
            self.documents[node] = value
        self.documents["provider"]["DESIGN"]["Work_schedule"] = {"from": "hidden"}
        self.documents["provider"]["REALIZATION"] = {"Require": [
            {"Item": "x.dtype", "Choice": "Keep fp32", "Reason": "Provider precision"}]}
        self.documents["middle"]["DESIGN"] = {region: {"from": "provider"} for region in DESIGN_REGIONS}
        self.documents["child"]["DESIGN"]["Math_reshape"] = {"from": "middle"}
        self.documents["child"]["DESIGN"]["Tile_pipeline"] = {"from": "provider"}
        self.documents["child"]["META_DATA"]["Version"] = 3
        self.documents["child"]["REALIZATION"] = {"Prefer": [
            {"Item": "x.dtype", "Choice": "Keep fp32", "Reason": "Consumer precision"}]}
        for node, value in self.documents.items():
            write_yaml(self.task / (node + (".yml" if node == "provider" else ".yaml")), value)
        rows = ["# 标量复制", "", "## 范围", "", "临时合成检查材料", "", "## 完整设计", "",
                "| 设计 | 具体机制与保留理由 | 关系 |", "| --- | --- | --- |"]
        rows.extend(f"| [{path.stem}]({path.name}) | 临时合成设计 | 实际整区引用 |" for path in sorted(self.task.iterdir()))
        (self.task / "dispatch.md").write_text("\n".join(rows) + "\n")
        self.original = manifest(self.task)
        self.schema = schema_for(REPO, cache_identity=digest(standalone._template_files(REPO)))

    def test_export_expands_actual_regions_and_preserves_node_context(self):
        output = self.root / "view"
        first = create_view(self.task / "child.yaml", output, REPO)
        self.assertEqual(create_view(self.task / "child.yaml", output, REPO)["result"], "reused")
        _, resolved = self.schema.validate_resolved_file(self.task / "child.yaml", skip_hardware=True)
        _, checked = self.schema.validate_standalone_file(output / "child.yaml", skip_hardware=True)
        self.assertEqual(checked["child"], resolved.effective_document("child"))
        self.assertEqual(checked["child"]["META_DATA"], self.documents["child"]["META_DATA"])
        self.assertEqual(checked["child"]["REALIZATION"], self.documents["child"]["REALIZATION"])
        self.assertIsInstance(checked["child"]["CONTRACT"]["Algorithm_requirement"]["Constant"]["offset"]["value"], Decimal)
        self.assertEqual(set(first["source_files"]), {"root.yaml", "provider.yml", "middle.yaml", "child.yaml"})
        self.assertEqual(first["validated_directory_files"], self.original)
        self.assertEqual(first["origins"]["Math_reshape"], {"provider": "provider", "chain": ["child", "middle", "provider"]})
        self.assertEqual(first["effective_identity"], resolved.effective_identity("child"))
        self.assertEqual(manifest(self.task), self.original)
        self.task.rename(self.root / "moved-raw")
        self.schema.validate_standalone_file(output / "child.yaml", skip_hardware=True)

    def test_complete_source_rechecked_at_validation_generation_and_commit(self):
        for name in ("root.yaml", "provider.yml", "middle.yaml", "child.yaml", "hidden.yaml", "dispatch.md"):
            for boundary in ("validated", "generated", "synced"):
                with self.subTest(source=name, boundary=boundary):
                    source, output = self.task / name, self.root / (name + boundary)
                    original = source.read_bytes()
                    def drift(function):
                        def changed(*args, **kwargs):
                            result = function(*args, **kwargs)
                            source.write_bytes(original + b"\n")
                            return result
                        return changed
                    target, method = (self.schema, "validate_resolved_file") if boundary == "validated" else (
                        (self.schema, "validate_standalone_file") if boundary == "generated" else (standalone, "sync_tree"))
                    try:
                        with patch.object(target, method, side_effect=drift(getattr(target, method))):
                            with self.assertRaisesRegex(ExtractError, "source changed.*" + name):
                                create_view(self.task / "child.yaml", output, REPO)
                        self.assertFalse(output.exists())
                        self.assertEqual(list(self.root.glob(".standalone-*")), [])
                    finally:
                        source.write_bytes(original)

    def test_complete_directory_invalidities_cannot_hide_behind_one_valid_node(self):
        mutations = ("extra", "contract", "dispatch", "cycle")
        for case in mutations:
            with self.subTest(case=case):
                if case == "extra":
                    path = self.task / "draft.yaml"
                    original = None
                    path.write_text("unfinished: [")
                elif case == "dispatch":
                    path = self.task / "dispatch.md"
                    original = path.read_bytes()
                    path.write_text("# 目录\n没有完整设计登记\n")
                else:
                    path = self.task / "hidden.yaml"
                    original = path.read_bytes()
                    value = deepcopy(self.documents["hidden"])
                    if case == "contract":
                        value["CONTRACT"] = deepcopy(self.documents["root"]["CONTRACT"])
                    else:
                        value["DESIGN"]["Tile_pipeline"] = {"from": "provider"}
                    write_yaml(path, value)
                try:
                    with self.assertRaises(ExtractError):
                        create_view(self.task / "child.yaml", self.root / case, REPO)
                finally:
                    path.unlink() if original is None else path.write_bytes(original)

    def test_repeated_views_use_the_current_template_validator(self):
        rules = self.root / "tests/current-rules"
        shutil.copytree(REPO / "schema_template/tools", rules / "schema_template/tools",
                        ignore=shutil.ignore_patterns("tests", "__pycache__"))
        first = create_view(self.task / "child.yaml", self.root / "before-update", rules)
        self.assertIn("schema_template/tools/schema.py", first["template"]["files"])
        self.assertIn("schema_template/tools/schema.py", rule_files(rules, require_specs=False))
        validator = rules / "schema_template/tools/schema.py"
        validator.write_text(validator.read_text() + "\n"
            "def validate_standalone_file(*args, **kwargs):\n"
            "    raise ToolError('Updated Template rejects this view')\n")
        output = self.root / "after-update"
        with self.assertRaisesRegex(ExtractError, "Updated Template rejects"):
            create_view(self.task / "child.yaml", output, rules)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".standalone-*")), [])

    def test_source_drift_conflict_reuse_and_containment_preserve_existing_bytes(self):
        output = self.root / "view"
        made = create_view(self.task / "child.yaml", output, REPO)
        before = manifest(output)
        provider = self.task / "provider.yml"
        provider.write_bytes(provider.read_bytes() + b"\n")
        with self.assertRaisesRegex(ExtractError, "different content"):
            create_view(self.task / "child.yaml", output, REPO)
        changed = create_view(self.task / "child.yaml", self.root / "updated", REPO)
        self.assertNotEqual(changed["effective_identity"], made["effective_identity"])
        self.assertEqual(changed["view"], made["view"])
        self.assertEqual(manifest(output), before)
        with self.assertRaisesRegex(ExtractError, "outside the source"):
            create_view(self.task / "child.yaml", self.task / "view", REPO)
        actual_read = standalone.read_json
        def drift(path):
            value = actual_read(path)
            provider.write_bytes(provider.read_bytes() + b"\n")
            return value
        with patch.object(standalone, "read_json", side_effect=drift):
            with self.assertRaisesRegex(ExtractError, "source changed"):
                create_view(self.task / "child.yaml", self.root / "updated", REPO)

    def test_current_cli_and_result_validation(self):
        output = self.root / "cli-view"
        made = subprocess.run([sys.executable, "-B", str(REPO / "schema_extract/extract_yaml.py"), "standalone",
                               str(self.task / "child.yaml"), "--output", str(output), "--repo", str(REPO)],
                              capture_output=True, text=True, timeout=30)
        self.assertEqual(made.returncode, 0, made.stderr)
        checked = subprocess.run([sys.executable, "-B", str(REPO / "schema_template/check_yaml.py"), "--standalone",
                                  "--read-only", "--skip-hardware", str(output / "child.yaml")],
                                 capture_output=True, text=True, timeout=30)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        value = load_yaml(output, "child.yaml")
        value["DESIGN"]["Math_reshape"] = {"from": "provider"}
        write_yaml(output / "child.yaml", value)
        with self.assertRaises(self.schema.SchemaError):
            self.schema.validate_standalone_file(output / "child.yaml", skip_hardware=True)
        self.assertEqual(read_json(output / ".source.json")["validated_directory_files"], self.original)


if __name__ == "__main__":
    unittest.main()
