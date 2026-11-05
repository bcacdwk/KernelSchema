"""Nullable I/O through YAML and Contract gates; no branch/default interpreter."""
from __future__ import annotations

from copy import deepcopy

from .. import yaml_source
from ..rules import validate_algorithm
from .support import ALGO, TEMPLATE_ROOT, TaskCase, at, drop, put


class NullableTests(TaskCase):
    def test_nullable_accepts_only_booleans_on_input_and_output(self):
        for group, name in (("input", "X"), ("output", "Y")):
            path = ALGO + (group, name, "nullable")
            for value in (True, False):
                with self.subTest(group=group, value=value):
                    self.assert_valid(lambda d: put(d, path, value))
            for value in (0, 1, "true", "false", None, [], {}):
                with self.subTest(group=group, value=value):
                    self.assert_invalid(lambda d: put(d, path, value), "expected a boolean")

    def test_nullable_does_not_change_defaults_or_tensor_shapes(self):
        inputs = yaml_source.loads("""
Required: {meaning: Explicit tensor or None., shape: [element: 4], dtype: fp32, nullable: true}
DefaultNone: {meaning: Absent by default., shape: [element: 4], dtype: fp32, nullable: true, default: "None"}
DefaultZero: {meaning: Zero tensor by default., shape: [element: 4], dtype: fp32, nullable: true, default: 0}
Empty: {meaning: An empty tensor., shape: [element: 0], dtype: fp32, nullable: false}
ScalarZero: {meaning: A scalar zero by default., shape: [], dtype: fp32, default: 0}
""", "nullable_values").value
        task = self.assert_valid(lambda d: at(d, ALGO)["input"].update(inputs))
        actual = at(task, ALGO)["input"]
        self.assertNotIn("default", actual["Required"])
        self.assertEqual(actual["DefaultNone"]["default"], "None")
        self.assertEqual(actual["DefaultZero"]["default"], 0)
        self.assertEqual(actual["Empty"]["shape"], [{"element": 0}])
        self.assertEqual(actual["ScalarZero"]["shape"], [])
        self.assertNotIn("nullable", actual["ScalarZero"])
        # YAML null remains invalid even on an explicitly nullable input.
        self.assert_invalid(lambda d: put(d, ALGO + ("input", "Bad"),
                                        dict(inputs["DefaultNone"], default=None)), "expected a known scalar")

    def test_nullable_keeps_payload_fields_required_and_is_limited_to_io(self):
        for group, name in (("input", "X"), ("output", "Y")):
            for field in ("meaning", "shape", "dtype"):
                with self.subTest(group=group, missing=field):
                    self.assert_invalid(lambda d: (
                        put(d, ALGO + (group, name, "nullable"), True),
                        drop(d, ALGO + (group, name, field))), "field is required")
        self.assert_invalid(lambda d: put(d, ALGO + ("compute", 0, "symbol"), {
            "A": {"meaning": "A mathematical value.", "shape": [], "dtype": "fp32", "nullable": True}
        }), "field not allowed")

    def test_python_none_singleton_requires_scalar_shape_in_both_gates(self):
        for group, name in (("input", "X"), ("output", "Y")):
            path = ALGO + (group, name)
            self.assert_valid(lambda d: at(d, path).update(dtype="python_none"))
            for shape in ([{"element": 0}], [{"element": 4}]):
                with self.subTest(group=group, shape=shape):
                    task = deepcopy(self.base)
                    at(task, path).update(dtype="python_none", shape=shape)
                    errors = validate_algorithm(at(task, ALGO))
                    self.assertTrue(any("python_none is a singleton scalar" in str(error) for error in errors))
                    self.assert_task_invalid(task, "python_none is a singleton scalar")

    def test_real_io_excerpts_use_nullable_without_a_union_dtype(self):
        excerpts = yaml_source.load(TEMPLATE_ROOT / "tools/tests/fixtures/nullable_io.yaml", "nullable_io").value
        for family, excerpt in excerpts.items():
            with self.subTest(family=family):
                algorithm = deepcopy(at(self.base, ALGO))
                algorithm.update(deepcopy(excerpt))
                # The synthetic enclosing reference tests structure only.
                algorithm["golden_reference"]["tolerance"] = {name: "exact" for name in excerpt["output"]}
                self.assertEqual(validate_algorithm(algorithm), [])
