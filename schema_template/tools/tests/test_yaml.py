"""Yaml behavior through the same task checker used by authors."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import hashlib
from .. import schema, yaml_source
from ..diagnostics import encode_value
from .support import (TaskCase, ALGO, ROLE, MATH, add_child, at, drop, put)


class YamlTests(TaskCase):
    def test_memory_and_file_parser_share_exact_scalars_marks_and_hashes(self):
        root = self.directory / "root.yaml"
        text = ('base: &base {flag: on, date: 2026-11-02, count: 080}\r\n'
                'meta_data: {<<: *base, explicit: !!str 1e3, exact: 1.000000000000000000000001}\r\n')
        root.write_bytes(text.encode("utf-8"))
        before = self.snapshot()
        file_document = yaml_source.load(root, "root")
        memory_document = yaml_source.loads(text, "root", path=root)
        self.assertEqual(file_document, memory_document)
        values = memory_document.value["meta_data"]
        self.assertEqual("on", values["flag"])
        self.assertEqual("2026-11-02", values["date"])
        self.assertEqual(80, values["count"])
        self.assertEqual("1e3", values["explicit"])
        self.assertEqual(Decimal("1.000000000000000000000001"), values["exact"])
        self.assertEqual(before, self.snapshot())

    def test_memory_parser_keeps_errors_and_location_without_creating_files(self):
        before = self.snapshot()
        for text, message in (("key: 1\nkey: 2\n", "duplicate mapping key"),
                              ("key: &loop {again: *loop}", "recursive aliases"),
                              ("key: 1\n---\nkey: 2", "exactly one YAML mapping"),
                              ("key: " + chr(0xD800), "valid UTF-8")):
            with self.subTest(message=message), self.assertRaisesRegex(yaml_source.ReadError, "projection.yaml.*" + message):
                yaml_source.loads(text, "root", path="projection.yaml")
        self.assertEqual(before, self.snapshot())

    def test_source_hashes_describe_the_exact_bytes_parsed(self):
        root = self.directory / "root.yaml"
        original = self.source.replace("\n", "\r\n").encode("utf-8")
        root.write_bytes(original)
        hashes = {"retired_node": "a previous check's hash"}
        _, task = schema.validate_file(root, source_hashes=hashes)
        self.assertEqual(task, self.base)
        self.assertEqual(hashes, {"root": hashlib.sha256(original).hexdigest()})
        self.assertEqual(root.read_bytes(), original)

    def test_validation_never_changes_sources_or_permissions(self):
        (self.directory / "dispatch.md").write_text("Synthetic task navigation.\n")
        root = self.directory / "root.yaml"
        for permissions in (0o644, 0o444):
            root.chmod(permissions)
            before = self.snapshot()
            self.assertEqual(self.base, schema.validate_directory(self.directory))
            self.assertEqual(before, self.snapshot())
        root.chmod(0o644)
        data = deepcopy(self.base)
        put(data, ROLE + ("waits_for",), {"data": {"produce:Y": ["Missing/produce:Y"]}})
        self.assert_task_invalid(data, "root.yaml", "design.tile_pipeline.phase.copy.role.Main.waits_for.data", "unknown role")

    def test_compact_expanded_and_multiline_yaml(self):
        forms = [
            'X: {meaning: Input scalar., shape: [token: T, d: 128], dtype: fp32}',
            'X: {meaning: Input scalar., shape: [{token: T}, {d: 128}], dtype: fp32}',
            'X:\n        meaning: Input scalar.\n        shape:\n          - token: T\n          - d: 128\n        dtype: fp32',
        ]
        root = self.directory / "root.yaml"
        results = []
        for form in forms:
            with self.subTest(form=form):
                text = self.source.replace('X: {meaning: Input scalar., shape: [], dtype: fp32}', form)
                root.write_text(text)
                results.append(schema.validate_directory(self.directory))
                self.assertEqual("Y = X.\n", at(results[-1], ALGO)["compute"][0]["math"])
                for invalid in (text.replace("d: 128", "d: -1"), text.replace("dtype:", "datatype:", 1)):
                    root.write_text(invalid)
                    with self.assertRaisesRegex(schema.SchemaError, "root.yaml"):
                        schema.validate_directory(self.directory)
        self.assertTrue(all(result == results[0] for result in results))

    def test_yaml_anchors_are_local_value_reuse(self):
        root = self.directory / "root.yaml"
        source = self.source.replace("shape: [], dtype: fp32", "shape: &scalar [], dtype: fp32", 1)
        source = source.replace("Y: {meaning: Output scalar., shape: []", "Y: {meaning: Output scalar., shape: *scalar")
        root.write_text(source)
        self.assertEqual(self.base, schema.validate_directory(self.directory))
        (self.directory / "small.yaml").write_text("design: *scalar\n")
        with self.assertRaisesRegex(schema.SchemaError, "small.yaml"):
            schema.validate_directory(self.directory)

    def test_yml_extension_and_duplicate_file_identity(self):
        root = self.directory / "root.yaml"
        root.rename(self.directory / "root.yml")
        self.assertEqual(self.base, schema.validate_directory(self.directory))
        root.write_text(self.source)
        with self.assertRaisesRegex(schema.SchemaError, "duplicate node ID root"):
            schema.validate_directory(self.directory)
        root.unlink()
        data = deepcopy(self.base)
        add_child(data)
        (self.directory / "small.yaml").write_text(encode_value(data["small"]))
        (self.directory / "small.yml").write_text(encode_value(data["small"]))
        with self.assertRaisesRegex(schema.SchemaError, "duplicate node ID small"):
            schema.validate_directory(self.directory)

    def test_one_mapping_document_per_file(self):
        for text in ("", "null\n", "42\n", "[]\n", self.source + "---\n" + self.source,
                     "---\n" + self.source + "---\n"):
            with self.subTest(text=text[:30]):
                (self.directory / "root.yaml").write_text(text)
                with self.assertRaisesRegex(schema.SchemaError, "root.yaml.*exactly one YAML mapping"):
                    schema.validate_directory(self.directory)
        (self.directory / "root.yaml").write_text("---\n" + self.source + "...\n")
        self.assertEqual(self.base, schema.validate_directory(self.directory))
        data = deepcopy(self.base)
        add_child(data)
        self.write_task(data)
        child = self.directory / "small.yaml"
        child.write_text(child.read_text() + "\n---\n" + child.read_text())
        with self.assertRaisesRegex(schema.SchemaError, "small.yaml.*exactly one YAML mapping"):
            schema.validate_directory(self.directory)

    def test_syntax_and_structure_errors_locate_yaml(self):
        root = self.directory / "root.yaml"
        root.write_text("meta_data: [\n")
        with self.assertRaisesRegex(schema.SchemaError, r"root.yaml:"):
            schema.validate_directory(self.directory)
        data = deepcopy(self.base)
        put(data, ("root", "extra"), 1)
        self.assert_task_invalid(data, "root.yaml", "extra", "field not allowed")
        data = deepcopy(self.base)
        drop(data, ("root", "meta_data", "name"))
        self.assert_task_invalid(data, "root.yaml", "meta_data.name", "required")

    def test_root_is_required_and_all_yaml_files_are_checked(self):
        root = self.directory / "root.yaml"
        for path in (root, self.directory / "absent"):
            with self.assertRaisesRegex(schema.SchemaError, "task directory"):
                schema.validate_directory(path)
        root.unlink()
        with self.assertRaisesRegex(schema.SchemaError, "missing root.yaml"):
            schema.validate_directory(self.directory)
        (self.directory / "small.yaml").write_text(self.source)
        with self.assertRaisesRegex(schema.SchemaError, "missing root.yaml"):
            schema.validate_directory(self.directory)
        root.write_text(self.source)
        with self.assertRaisesRegex(schema.SchemaError, "small.yaml"):
            schema.validate_directory(self.directory)

    def test_filename_defines_node_identity(self):
        data = deepcopy(self.base)
        add_child(data)
        self.write_task(data)
        (self.directory / "small.yaml").rename(self.directory / "other.yaml")
        with self.assertRaisesRegex(schema.SchemaError, "other.yaml"):
            schema.validate_directory(self.directory)
        (self.directory / "other.yaml").unlink()
        for name in ("Bad", "bad-name", "1bad", "root.meta_data"):
            (self.directory / (name + ".yaml")).write_text(encode_value(data["small"]))
            with self.subTest(name=name), self.assertRaises(schema.SchemaError) as caught:
                schema.validate_file(self.directory / "root.yaml")
            self.assertIn(name + ".yaml", str(caught.exception))
            (self.directory / (name + ".yaml")).unlink()

    def test_exact_decimal_numbers_and_range_comparison(self):
        data = deepcopy(self.base)
        exact = Decimal("1.000000000000000000000001")
        lower = Decimal("1.000000000000000000000000")
        put(data, ALGO + ("input", "X", "range"), [lower, exact])
        put(data, MATH + ("rewrite", 0, "symbol"), {"coef": {"meaning": "Exact fixture coefficient.", "value": exact}})
        self.write_task(data)
        result = schema.validate_directory(self.directory)
        self.assertEqual(exact, at(result, MATH)["rewrite"][0]["symbol"]["coef"]["value"])
        put(data, ALGO + ("input", "X", "range"), [exact, lower])
        self.assert_task_invalid(data, "root.yaml", "input.X.range", "reversed")

    def test_file_symlink_preserves_task_directory_and_node_identity(self):
        actual = self.directory / "outside"
        actual.mkdir()
        (actual / "different_name.yaml").write_text(self.source)
        root = self.directory / "root.yaml"
        root.unlink()
        root.symlink_to(actual / "different_name.yaml")
        directory, task = schema.validate_file(root)
        self.assertEqual(self.directory, directory)
        self.assertEqual(self.base, task)
        self.assertTrue(root.is_symlink())

    def test_yaml_scalars_aliases_and_diagnostics_keep_their_meaning(self):
        root = self.directory / "root.yaml"
        root.write_text("first: &first {x: 1}\nsecond: &second {x: 2, y: 3}\nmerged: {<<: [*first, *second], y: 4}\n")
        before = root.read_bytes()
        self.assertEqual({"x": 1, "y": 4}, yaml_source.load(root, "root").value["merged"])
        self.assertEqual(before, root.read_bytes())
        for scalar in ("on", "off", "yes", "2026-10-27"):
            root.write_text(self.source.replace("version: 1", "version: 1\n  source: " + scalar))
            self.assertEqual(scalar, schema.validate_directory(self.directory)["root"]["meta_data"]["source"])
        root.write_text(self.source.replace("version: 1", "version: 1\n  version: 1"))
        with self.assertRaisesRegex(schema.SchemaError, "duplicate mapping key"):
            schema.validate_directory(self.directory)
        root.write_text("meta_data: &loop {name: *loop}\n")
        with self.assertRaisesRegex(schema.SchemaError, "recursive aliases"):
            schema.validate_directory(self.directory)
        root.write_text("!!python/object/apply:os.system [echo forbidden]\n")
        with self.assertRaises(schema.SchemaError):
            schema.validate_directory(self.directory)
        for content in (b"name: \x00\n", b"name: \xff\n"):
            root.write_bytes(content)
            with self.assertRaises(schema.SchemaError):
                schema.validate_directory(self.directory)
        # Decimal exponent syntax must never pass through binary floating point.
        data = deepcopy(self.base)
        put(data, ALGO + ("constant",), {"Tiny": {"meaning": "Exact coefficient.", "value": Decimal("1e-30")}})
        self.write_task(data)
        self.assertEqual(Decimal("1e-30"), schema.validate_directory(self.directory)["root"]["contract"]["algorithm_requirement"]["constant"]["Tiny"]["value"])
        data = deepcopy(self.base)
        put(data, ROLE + ("waits_for",), {"data": {"produce:Y": ["Missing/produce:Y"]}})
        self.write_task(data)
        with self.assertRaises(schema.SchemaError) as caught:
            schema.validate_file(root)
        source = root.read_text()
        self.assertIn("root.yaml:1:" + str(source.index('"Missing/produce:Y"') + 1), str(caught.exception))
        self.assertIn('waits_for.data["produce:Y"][0]', str(caught.exception))
