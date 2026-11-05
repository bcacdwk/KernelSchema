"""Catalog reading aids are frozen without changing authoritative source roles."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from schema_extract.tools.common import ExtractError, sha256, source_root
from schema_extract.tools.inputs import resolve_request
from schema_extract.tools.snapshots import freeze_inputs, verify_snapshot


class CatalogInputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "repository"
        self.wiki = self.repo / "kernel_wiki"
        directory = "fixtures/operators/increment"
        self.entry = {
            "id": "increment",
            "interface": {"forward_target": "Model.forward"},
            "paths": {
                "readme": directory + "/README.md",
                "entrypoint": directory + "/candidate/solution.py",
                "input": directory + "/problem/input.py",
                "reference": directory + "/problem/reference.py",
                **{key: directory + "/problem/task.json"
                   for key in ("metadata", "acceptance", "shape_range", "shape_cases")},
            },
        }
        self.program = "# Original explanation.\nclass Model:\n    def forward(self, x): return x + 1\n"
        for key, name in self.entry["paths"].items():
            path = self.wiki / name
            path.parent.mkdir(parents=True, exist_ok=True)
            text = (self.program if key in {"entrypoint", "reference"} else
                    "def _make_inputs(): return [1]\n" if key == "input" else
                    "{}\n" if path.suffix == ".json" else "Scalar increment.\n")
            path.write_text(text)
        self.annotation = "annotations/increment/solution.py"
        path = self.wiki / self.annotation
        path.parent.mkdir(parents=True)
        path.write_text(self.program.replace("Original explanation.", "Clarified explanation."))

    def resolve(self):
        (self.wiki / "catalog.json").write_text(json.dumps({"operators": [self.entry]}))
        return resolve_request(self.repo, operator_id=self.entry["id"])

    def test_optional_annotation_is_collected_without_rebinding_source_roles(self):
        _, original_bindings, original_files = self.resolve()
        self.assertNotIn("kernel_wiki/" + self.annotation, original_files)
        self.entry["paths"]["annotated_entrypoint"] = self.annotation
        _, bindings, names = self.resolve()
        self.assertEqual(bindings, original_bindings)
        self.assertEqual(set(names) - set(original_files), {"kernel_wiki/" + self.annotation})
        self.assertEqual(bindings["entrypoint"]["path"], "kernel_wiki/" + self.entry["paths"]["entrypoint"])
        self.assertEqual(bindings["host"], bindings["entrypoint"])

    def test_declared_annotation_must_exist_as_a_regular_in_repository_file(self):
        (self.wiki / "annotations/link.py").symlink_to(self.wiki / self.annotation)
        (self.wiki / "annotations/linked_directory").symlink_to(self.wiki / "annotations/increment",
                                                                 target_is_directory=True)
        for name in ("annotations/missing.py", "annotations/increment", "annotations/link.py",
                     "annotations/linked_directory/solution.py"):
            with self.subTest(path=name), self.assertRaisesRegex(ExtractError, "missing annotated_entrypoint input"):
                self.entry["paths"]["annotated_entrypoint"] = name
                self.resolve()

    def test_annotation_path_uses_the_existing_relative_path_boundary(self):
        for name in (None, "", "../outside.py", "/outside.py", "annotations/../../outside.py",
                     "annotations\\solution.py"):
            with self.subTest(path=name), self.assertRaisesRegex(ExtractError, "invalid relative path"):
                self.entry["paths"]["annotated_entrypoint"] = name
                self.resolve()

    def test_source_snapshot_contains_original_and_annotation_as_separate_files(self):
        self.entry["paths"]["annotated_entrypoint"] = self.annotation
        tree, bindings, names = self.resolve()
        rules = self.root / "rules"
        rules.mkdir()
        (rules / "rules.md").write_text("Rules are independent of the source collection under test.\n")
        # Exercise real source sealing and manifests with a tiny unrelated rule set.
        with patch("schema_extract.tools.snapshots.rule_files", return_value=["rules.md"]):
            lock = freeze_inputs(self.root / "runtime", tree, bindings, names, {}, rules_repo=rules)
        fixed = source_root(lock)
        original = "kernel_wiki/" + self.entry["paths"]["entrypoint"]
        annotation = "kernel_wiki/" + self.annotation
        for name in (original, annotation):
            self.assertEqual(lock["source"]["files"][name], sha256((self.repo / name).read_bytes()))
            self.assertEqual((fixed / name).read_bytes(), (self.repo / name).read_bytes())
        self.assertEqual(lock["source"]["bindings"]["entrypoint"]["path"], original)
        self.assertEqual(lock["source"]["bindings"]["host"]["path"], original)
        old_bytes = (fixed / annotation).read_bytes()
        (self.repo / annotation).write_text("Later annotation changes do not edit frozen inputs.\n")
        self.assertEqual((fixed / annotation).read_bytes(), old_bytes)
        verify_snapshot(lock)


if __name__ == "__main__":
    unittest.main()
