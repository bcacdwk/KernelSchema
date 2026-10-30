"""Shared synthetic tasks and assertions; no production validation rules here."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import stat
import tempfile
import unittest

from .. import schema
from ..diagnostics import encode_value

TEMPLATE_ROOT = Path(__file__).resolve().parents[2]
ALGO = ("root", "contract", "algorithm_requirement")
HARD = ("root", "contract", "hardware_specification")
MATH = ("root", "design", "math_reshape")
WORK = ("root", "design", "work_schedule")
PHASE = WORK + ("phase", "copy")
PIPE = ("root", "design", "tile_pipeline", "phase", "copy")
ROLE = PIPE + ("role", "Main")


def at(data, path):
    for key in path:
        data = data[key]
    return data


def put(data, path, value):
    at(data, path[:-1])[path[-1]] = value


def drop(data, path):
    del at(data, path[:-1])[path[-1]]


def add_phase(data, *, order=None):
    at(data, WORK)["phase"]["next"] = deepcopy(at(data, PHASE))
    at(data, ("root", "design", "tile_pipeline", "phase"))["next"] = deepcopy(at(data, PIPE))
    at(data, WORK)["flow"]["launch"] = (
        [{"phase": ["copy", "next"], "order": order}] if order else
        [{"phase": ["copy"]}, {"phase": ["next"], "start_after": "copy.kernel_done"}]
    )


def add_child(data, name="small", parent=None):
    root = data["root"]
    root["meta_data"]["dispatch"] = {"when": "Select exactly when the input is nonnegative."}
    child = deepcopy(root)
    del child["contract"]
    child["meta_data"]["dispatch"] = {"id": name, "when": "Select exactly when the input is negative."}
    if parent:
        child["meta_data"]["dispatch"]["parent"] = parent
    data[name] = child


class TaskCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = schema.validate_directory(TEMPLATE_ROOT / "tools/tests/fixtures/minimal")
        cls.source = (TEMPLATE_ROOT / "tools/tests/fixtures/minimal/root.yaml").read_text()

    def setUp(self):
        # Tasks may live outside the repository.
        self.temp = tempfile.TemporaryDirectory(prefix="schema-yaml-test-")
        self.directory = Path(self.temp.name).resolve()
        self.addCleanup(self.temp.cleanup)
        (self.directory / "root.yaml").write_text(self.source)

    def write_task(self, data):
        # JSON flow syntax is also YAML; mutations exercise the directory entry.
        # The checked-in fixture and presentation tests use block/compact YAML.
        for file in self.directory.iterdir():
            if file.suffix in {".yaml", ".yml"}:
                file.unlink()
        for node, plan in data.items():
            (self.directory / (node + ".yaml")).write_text(encode_value(plan))

    def snapshot(self):
        return {p.name: (p.read_bytes(), stat.S_IMODE(p.stat().st_mode), p.stat().st_mtime_ns)
                for p in self.directory.iterdir()}

    def assert_task_invalid(self, data, *messages):
        self.write_task(data)
        before = self.snapshot()
        with self.assertRaises(schema.SchemaError) as caught:
            schema.validate_directory(self.directory)
        for message in messages:
            self.assertIn(message, str(caught.exception))
        self.assertEqual(before, self.snapshot())

    def assert_valid(self, transform=lambda d: None):
        data = deepcopy(self.base)
        transform(data)
        self.write_task(data)
        self.assertEqual(data, schema.validate_directory(self.directory))
        return data

    def assert_invalid(self, transform, message=None):
        data = deepcopy(self.base)
        transform(data)
        self.assert_task_invalid(data, *([message] if message else []))
