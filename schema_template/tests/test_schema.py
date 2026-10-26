"""Acceptance tests for CUE constraints and direct YAML task validation."""
from __future__ import annotations

from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal
import io
from pathlib import Path
import stat
import sys
import tempfile
import unittest

TEMPLATE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEMPLATE_ROOT / "tools"))
import schema
import json_value

ALGO = ("root", "contract", "algorithm_requirement")
HARD = ("root", "contract", "hardware_specification")
MATH = ("root", "design", "math_reshape")
WORK = ("root", "design", "work_schedule")
PHASE = WORK + ("phase", "copy")
PIPE = ("root", "design", "tile_pipeline", "phase", "copy")
LANE = PIPE + ("lane", "Main")


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


class SchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = schema.validate_directory(TEMPLATE_ROOT / "tests/fixtures/minimal")
        cls.source = (TEMPLATE_ROOT / "tests/fixtures/minimal/root.yaml").read_text()

    def setUp(self):
        # Tasks need not be in the repository's CUE module.
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
            (self.directory / (node + ".yaml")).write_text(json_value.dumps(plan))

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

    def test_local_constraints_are_enforced_by_cue(self):
        assignment = {"preassigned": {"start": 0, "count": 1}, "refill": "none", "exit": "assigned_done"}
        handoff = {"producer": "copy", "consumer": ["next"], "storage": "global"}
        lane = deepcopy(at(self.base, LANE))
        cases = [
            ("#Schedule", {"worker_assignment": assignment},
             {"worker_assignment": {"refill": "none", "exit": "assigned_done"}}),
            ("#Handoff", handoff, dict(handoff, consumer=["copy", "next"])),
            ("#MemberRegion", [0, 4], [-1, 4]),
            ("#MemberRegion", [[0, 4], [4, 8]], [[0, 4], [-1, 8]]),
            ("#View", {"part": {"x": [0, 4]}}, {"part": {"x": [-1, 4]}}),
        ]
        for name in ("kernel_done", "launch_ready", "kernel_done(i)", "launch_ready(i)"):
            cases.append(("#PipelinePhase", {"lane": {"Main": lane}, "milestone": {"copied": ["Main/produce:Y"]}},
                          {"lane": {"Main": lane}, "milestone": {name: ["Main/produce:Y"]}}))
        for definition, valid, invalid in cases:
            with self.subTest(definition=definition, invalid=invalid):
                args = ("vet", "-c", "./schema_template:kernel", "-d", definition, "yaml:", "-")
                schema.cue(*args, data=json_value.dumps(valid))
                with self.assertRaises(schema.SchemaError):
                    schema.cue(*args, data=json_value.dumps(invalid))

    def test_signed_domains_and_textual_integer_bounds(self):
        self.assert_valid(lambda d: put(d, LANE + ("pipe",), {
            "repeat": {"i": [-1, 2]}, "steady": ["produce:Y(i)"]}))
        for path, value in [
            (PHASE + ("mapping", "member"), [{"x": ["-1", 4]}]),
            (PIPE + ("resource",), {"R": {"space": "SMEM", "shape": [{"x": 4}], "dtype": "fp32",
                                          "view": {"Y": {"part": {"x": ["-1", 4]}}}}}),
        ]:
            with self.subTest(path=path):
                self.assert_invalid(lambda d: put(d, path, value), "literal")

    def test_closed_structures_required_fields_and_types(self):
        cases = [
            (("root", "extra"), 1), (("root", "meta_data", "version"), 0),
            (("root", "meta_data", "version"), "1"), (("root", "meta_data", "name"), " "),
            (ALGO + ("input",), {}), (ALGO + ("output",), {}), (ALGO + ("compute",), []),
            (ALGO + ("input", "X", "shape"), [{"x": 1, "y": 2}]),
            (ALGO + ("input", "X", "shape"), [{"x": -1}]),
            (ALGO + ("input", "X", "shape"), [1]),
            (ALGO + ("input", "X", "shape"), [{"1x": 1}]),
            (ALGO + ("input", "X", "dtype"), "<type@pending:x>"),
            (ALGO + ("input", "X", "dtype"), "not_applicable"),
            (ALGO + ("input", "X", "default"), None),
            (ALGO + ("input", "X", "default"), "unknown: missing default"),
            (ALGO + ("output", "Y", "default"), 0),
            (ALGO + ("compute", 0, "life"), "stored invocation"),
            (ALGO + ("golden_reference", "source"), "test@v1:reference.py"),
            (ALGO + ("golden_reference", "tolerance"), {"Typo": "exact"}),
            (ALGO + ("golden_reference", "tolerance"), {}),
            (HARD + ("execution",), {}), (HARD + ("feature", "extra"), {"nested": "no"}),
            (HARD + ("execution", "CTA"), "<hardware@pending:CTA>"),
            (HARD + ("stack", "toolchain"), "Compiler; <version@pending:toolchain>"),
            (HARD + ("wiki", "profile"), ["test@v1:one", "test@v1:two"]),
            (HARD + ("wiki", "profile"), "test@pending:profile"),
            (HARD + ("wiki", "new_category"), "test@v1:extra"),
            (MATH + ("rewrite", 0, "step"), 1),
            (MATH + ("rewrite", 0, "arithmetic"), {}),
            (MATH + ("ledger",), {"baseline": "A baseline without deltas."}),
            (MATH + ("ledger",), {"baseline": "Direct", "new_resource": [{"item": "x", "change": "+1", "by": ["copy"]}]}),
            (PHASE + ("decomposition", "partition"), {"x": {"size": 0, "tile": 1}}),
            (PHASE + ("mapping", "worker", "unit"), "InventedUnit"),
            (PHASE + ("mapping", "placement"), {"unit": "InventedUnit", "count": 1}),
            (PHASE + ("mapping", "count"), -1),
            (PHASE + ("schedule", "worker_assignment", "preassigned"), {"start": 0, "step": 1, "count": 1}),
            (PHASE + ("schedule", "worker_assignment", "refill"), {}),
            (PHASE + ("schedule", "worker_assignment", "exit"), []),
            (LANE + ("info", "mode"), "parallel"),
            (LANE + ("info", "register"), 0),
            (LANE + ("pipe",), {}),
            (LANE + ("pipe", "extra"), []),
            (LANE + ("pipe", "steady"), []),
            (LANE + ("edge",), {}),
            (LANE + ("edge",), {"data": {"produce:Y": []}}),
            (("root", "realization"), {}),
            (("root", "realization"), {"thesis": "Not a realization field."}),
            (("root", "realization"), {"prefer": []}),
        ]
        for path, value in cases:
            with self.subTest(path=path, value=value):
                self.assert_invalid(lambda d: put(d, path, value))
        for path in [("root", "contract"), ALGO + ("input", "X", "meaning"),
                     ALGO + ("output", "Y", "dtype"), MATH + ("thesis",),
                     PHASE + ("mapping",), LANE + ("pipe",)]:
            with self.subTest(missing=path):
                self.assert_invalid(lambda d: drop(d, path))

    def test_open_vocabularies_and_meaningful_empty_values(self):
        def update(d):
            at(d, HARD)["execution"]["AI Core"] = "A composite unit."
            at(d, HARD)["feature"].update({"AI Core": "Same spelling in an independent vocabulary.",
                                          "L1/SMEM": "A storage mechanism.", "tcgen05.mma": "A matrix mechanism."})
            at(d, ALGO)["input"]["X"]["dtype"] = "fp6_e3m2"
            at(d, ALGO)["input"]["X"]["default"] = 0
            at(d, PIPE)["resource"] = {
                "Scalar": {"space": "register", "shape": [], "dtype": "fp32", "view": {"X": {}, "Y": {}}}
            }
        self.assert_valid(update)

    def test_hardware_template_pair(self):
        self.assert_invalid(lambda d: put(d, HARD + ("stack", "template"), "Library 1"))
        self.assert_invalid(lambda d: put(d, HARD + ("wiki", "programming", "template"), "repo@v1:docs/"))
        self.assert_valid(lambda d: (put(d, HARD + ("stack", "template"), "Library 1"),
                                     put(d, HARD + ("wiki", "programming", "template"), "repo@v1:docs/")))

    def test_contract_names_refs_and_shape_rules(self):
        symbol = {"meaning": "A scalar.", "shape": "ref:X.shape", "dtype": "ref:X.dtype"}
        self.assert_valid(lambda d: put(d, ALGO + ("compute", 0, "symbol"), {"A": symbol}))
        self.assert_invalid(lambda d: put(d, ALGO + ("compute", 0, "symbol"), {"X": symbol}), "duplicate")
        self.assert_invalid(lambda d: put(d, ALGO + ("compute", 0, "symbol"),
                                         {"A": dict(symbol, shape="ref:Missing.shape")}), "unknown Contract")
        self.assert_invalid(lambda d: put(d, ALGO + ("compute", 0, "symbol"),
                                         {"A": dict(symbol, shape="ref:B.shape"), "B": dict(symbol, shape="ref:A.shape")}), "cycle")
        self.assert_invalid(lambda d: put(d, ALGO + ("compute",), [
            {"name": "First", "math": "A = B.", "symbol": {"A": dict(symbol, dtype="ref:B.dtype")}},
            {"name": "Second", "math": "B = X.", "symbol": {"B": symbol}}]), "precedes")
        self.assert_valid(lambda d: put(d, ALGO + ("compute", 0, "symbol"),
                                       {"f": {"meaning": "A function.", "shape": "not_applicable", "dtype": "not_applicable"}}))
        self.assert_invalid(lambda d: put(d, ALGO + ("input", "X", "shape"), [{"x": 2}, {"x": 3}]), "axis names")
        self.assert_invalid(lambda d: put(d, ALGO + ("input", "X", "range"), [5, 2]), "reversed")

    def test_math_symbols_and_ledger(self):
        row = {"item": "Traffic", "change": "-4 bytes per invocation", "by": ["copy"]}
        self.assert_valid(lambda d: put(d, MATH + ("ledger",), {"baseline": "Direct copy.", "traffic": [row]}))
        self.assert_invalid(lambda d: put(d, MATH + ("ledger",),
                                         {"baseline": "Direct", "traffic": [dict(row, by=["missing"])]}))
        self.assert_invalid(lambda d: at(d, MATH)["rewrite"].append(deepcopy(at(d, MATH)["rewrite"][0])))
        self.assert_invalid(lambda d: put(d, MATH + ("rewrite", 0, "symbol"),
                                         {"A": {"meaning": "A parameter.", "binding": "bind:design.work_schedule::A", "value": 1}}))
        self.assert_invalid(lambda d: put(d, MATH + ("rewrite", 0, "symbol"),
                                         {"A": {"meaning": "A state.", "life": "carried invocation"}}))
        self.assert_invalid(lambda d: put(d, MATH + ("rewrite", 0, "symbol"),
                                         {"A": {"meaning": "Not a tensor.", "shape": "not_applicable"}}))
        self.assert_invalid(lambda d: put(d, MATH + ("rewrite", 0, "symbol"),
                                         {"A": {"meaning": "Known value required.", "value": "unknown: value"}}))
        self.assert_valid(lambda d: put(d, MATH + ("rewrite", 0, "symbol"), {
            "A": {"meaning": "State with declared missing evidence.", "life": "stored invocation",
                  "shape": "unknown: shape not yet established", "dtype": "unknown: format not yet established"},
            "Coeff": {"meaning": "Fixed coefficients.", "value": [[1, 2], [3, 4]]},
            "Size": {"meaning": "Downstream parameter.", "binding": "bind:design.work_schedule::Size"},
        }))

    def test_work_dimensions_and_assignment(self):
        def partition(d):
            put(d, PHASE + ("decomposition", "partition"), {"x": {"size": "N", "tile": 1, "range": [1, None]}})
            put(d, PHASE + ("schedule", "worktile_order"), [{"by": "x", "direction": "ascending"}])
        self.assert_valid(partition)
        self.assert_valid(lambda d: (partition(d), put(d, PHASE + ("schedule", "worktile_order"),
            [{"by": ["x"], "group": 8, "direction": "ascending"}, {"by": "x", "direction": "descending"}])))
        self.assert_invalid(lambda d: (partition(d), drop(d, PHASE + ("schedule", "worktile_order"))))
        self.assert_invalid(lambda d: (partition(d), put(d, PHASE + ("schedule", "worktile_order", 0, "by"), "missing")))
        self.assert_invalid(lambda d: (partition(d), put(d, PHASE + ("decomposition", "partition", "x", "range"), [5, 2])), "reversed")
        self.assert_invalid(lambda d: put(d, PHASE + ("mapping", "count"), "__import__('os')"))
        self.assert_invalid(lambda d: put(d, PHASE + ("mapping", "count"), "not an expression"), "integer")
        self.assert_invalid(lambda d: put(d, PHASE + ("mapping", "count"), "0"), "literal")
        self.assert_invalid(lambda d: drop(d, PHASE + ("schedule", "worker_assignment", "preassigned")), "preassigned")
        self.assert_valid(lambda d: (drop(d, PHASE + ("schedule", "worker_assignment", "preassigned")),
                                    put(d, PHASE + ("schedule", "worker_assignment", "refill"), {"counter": "Fetch until exhausted."})))
        self.assert_invalid(lambda d: put(d, PHASE + ("decomposition", "worktile"),
            {"invariant": {"X": {"shape": [], "dtype": "fp32", "offset": [0]}}}))
        self.assert_invalid(lambda d: put(d, PHASE + ("decomposition", "worktile"),
            {"iteration": {"X": {"shape": [], "dtype": "fp32"}}}))

    def test_phase_launch_topology_and_handoff(self):
        for order in ["per_worktile", "per_worker", "all", "concurrent", None]:
            with self.subTest(order=order):
                self.assert_valid(lambda d: add_phase(d, order=order))
        self.assert_invalid(lambda d: put(d, WORK + ("flow", "launch"), [{"phase": ["missing"]}]))
        self.assert_invalid(lambda d: put(d, WORK + ("flow", "launch"), [{"phase": ["copy"]}, {"phase": ["copy"]}]))
        self.assert_invalid(lambda d: (add_phase(d, order="all"), drop(d, WORK + ("flow", "launch", 0, "order"))))
        self.assert_invalid(lambda d: put(d, WORK + ("flow", "launch", 0, "start_after"), "copy.kernel_done"), "own event")
        self.assert_invalid(lambda d: (add_phase(d), put(d, WORK + ("flow", "launch", 0, "start_after"), "next.kernel_done")), "cycle")
        self.assert_invalid(lambda d: (add_phase(d), put(d, WORK + ("flow", "launch", 1, "start_after"), "copy.launch_ready")), "protocol")
        self.assert_valid(lambda d: (add_phase(d), put(d, WORK + ("flow", "launch", 1, "start_after"), "copy.launch_ready"),
                                     put(d, WORK + ("flow", "launch", 1, "protocol"), "pdl")))
        handoff = {"producer": "copy", "consumer": ["next"], "storage": "global", "initialize": 0}
        self.assert_valid(lambda d: (add_phase(d), put(d, WORK + ("flow", "handoff"), {"Y": handoff})))
        self.assert_invalid(lambda d: (add_phase(d), put(d, WORK + ("flow", "handoff"),
            {"Y": dict(handoff, storage="tile_pipeline.phase.copy.resource.Missing")})), "resource")
        self.assert_invalid(lambda d: (add_phase(d), put(d, WORK + ("flow", "handoff"),
            {"Y": dict(handoff, ready="copy.missing")})), "milestone")

    def test_recursive_pipe_and_index_scope(self):
        def nested(d):
            block = {"steady": ["produce:Y(" + ",".join(f"k{i}" for i in range(12)) + ")"]}
            for i in reversed(range(12)):
                block = {"repeat": {f"k{i}": [0, 2]}, "steady": [block]}
            put(d, LANE + ("pipe",), block)
        self.assert_valid(nested)
        def corrupt(d):
            nested(d)
            block = at(d, LANE)["pipe"]
            for _ in range(10):
                block = block["steady"][0]
            block["typo"] = "must fail even at depth 10"
        self.assert_invalid(corrupt)
        self.assert_invalid(lambda d: put(d, LANE + ("pipe",),
                                         {"repeat": {"i": [0, 8]}, "prologue": ["load:X(i)"], "steady": ["produce:Y(i)"]}), "not bound")
        self.assert_invalid(lambda d: put(d, LANE + ("pipe",),
                                         {"repeat": "missing", "steady": ["produce:Y(i)"]}), "undefined repeat")
        self.assert_invalid(lambda d: put(d, PIPE + ("domain",),
                                         {"a": {"flatten": ["b"]}, "b": {"flatten": ["a"]}}), "cycle")
        self.assert_invalid(lambda d: put(d, LANE + ("pipe",), {"repeat": {"i": [0, 8]}}))
        self.assert_valid(lambda d: (put(d, PIPE + ("domain",), {"t": "worktile", "i": [0, 8], "n": {"flatten": ["t", "i"]}}),
                                     put(d, LANE + ("pipe",), {"repeat": "t", "steady": [
                                         {"repeat": "n[t]", "steady": ["produce:Y(n,t,i)"], "epilogue": ["store:Y(n[t,last(i)])"]}]})))

    def test_action_events_and_custom_verbs(self):
        self.assert_invalid(lambda d: put(d, LANE + ("pipe", "steady"), ["magical:Y"]), "custom verb")
        self.assert_valid(lambda d: (put(d, LANE + ("pipe", "steady"), ["magical:Y"]),
                                     put(d, LANE + ("info", "verb"), {"magical": {"meaning": "Produce Y.", "mode": "sync"}})))
        self.assert_invalid(lambda d: put(d, LANE + ("edge",), {"data": {"produce:Y": ["Missing/produce:Y"]}}), "unknown lane")
        self.assert_invalid(lambda d: put(d, LANE + ("edge",), {"data": {"produce:Y": ["produce:Missing"]}}), "action family")
        self.assert_invalid(lambda d: put(d, LANE + ("edge",), {"data": {"produce:Y": ["produce:Y@read"]}}), "notation")
        self.assert_valid(lambda d: (put(d, LANE + ("pipe", "steady"), ["load:X", "mma:X@X", "produce:Y"]),
                                     put(d, LANE + ("edge",), {"data": {"produce:Y@issue": ["mma:X@X@done"]}})))
        self.assert_invalid(lambda d: put(d, LANE + ("info", "member"), "warp[0:0]"), "nonempty")
        self.assert_invalid(lambda d: put(d, LANE + ("info", "member"), "phantom[0:1]"), "unregistered")

    def test_resource_bounds_and_view_exceptions(self):
        resource = {"space": "SMEM", "shape": [{"x": 8}], "dtype": "fp32", "slot": 2, "view": {"Y": {"slot": 1}}}
        self.assert_valid(lambda d: put(d, PIPE + ("resource",), {"R": resource}))
        for replacement in [
            dict(resource, slot=0), dict(resource, view={"Y": {}}),
            dict(resource, view={"Y": {"slot": 2}}),
            dict(resource, view={"Y": {"slot": 0, "part": {"x": [0, 9]}}}),
            dict(resource, view={"Y": {"slot": 0, "part": {"other": [0, 2]}}}),
            dict(resource, member=[0, 0]), dict(resource, member=1),
        ]:
            with self.subTest(resource=replacement):
                self.assert_invalid(lambda d: put(d, PIPE + ("resource",), {"R": replacement}))

    def test_local_handoff_and_shared_worker_resource_identity(self):
        resource = {"space": "SMEM", "shape": [], "dtype": "fp32"}
        handoff = {"producer": "copy", "consumer": ["next"], "storage": "tile_pipeline.phase.copy.resource.Y"}
        def setup(d, order):
            add_phase(d, order=order)
            put(d, PIPE + ("resource",), {"Y": resource})
            put(d, WORK + ("flow", "handoff"), {"Y": handoff})
        self.assert_valid(lambda d: setup(d, "per_worktile"))
        self.assert_invalid(lambda d: setup(d, None), "same worker")
        self.assert_invalid(lambda d: setup(d, "concurrent"), "same worker")
        self.assert_invalid(lambda d: (setup(d, "per_worker"),
            put(d, ("root", "design", "tile_pipeline", "phase", "next", "resource"), {"Y": resource})), "more than once")

    def test_dynamic_domains_and_malformed_index_syntax(self):
        def dynamic(d):
            put(d, PIPE + ("domain",), {"t": "worktile"})
            put(d, PHASE + ("schedule", "worker_assignment", "refill"), {"counter": "Take the next work until exhausted."})
            put(d, LANE + ("pipe",), {"repeat": "t", "steady": ["produce:Y(t)"]})
        self.assert_valid(dynamic)
        self.assert_invalid(lambda d: (dynamic(d), put(d, LANE + ("pipe", "epilogue"), ["store:Y(last(t))"])), "count/last")
        # Contract prose is not parsed as an executed expression.
        self.assert_valid(lambda d: (dynamic(d), put(d, LANE + ("info", "verb"),
            {"produce": "Produce the result; do not query count(t) or last(t)."})))
        self.assert_invalid(lambda d: put(d, LANE + ("pipe", "steady"), ["produce:Y(i;broken)"]), "notation")
        self.assert_invalid(lambda d: (put(d, PIPE + ("domain",), {"i": [0, 8]}),
            put(d, LANE + ("pipe",), {"repeat": "i[0:8:2]", "steady": ["produce:Y(i)"]})), "slice notation")

    def test_instance_names_can_match_field_names(self):
        self.assert_valid(lambda d: (
            put(d, PIPE + ("domain",), {"shape": [0, 8]}),
            put(d, LANE + ("pipe",), {"repeat": "shape", "steady": ["produce:Y(shape)"]}),
            put(d, PIPE + ("milestone",), {"shape": ["Main/produce:Y(0)"]})))

    def test_per_worktile_order_must_agree(self):
        def mismatch(d):
            put(d, PHASE + ("decomposition", "partition"), {"x": {"size": 8, "tile": 1}})
            put(d, PHASE + ("schedule", "worktile_order"), [{"by": "x", "direction": "ascending"}])
            add_phase(d, order="per_worktile")
            put(d, WORK + ("phase", "next", "schedule", "worktile_order", 0, "direction"), "descending")
        self.assert_invalid(mismatch, "matching coordinates")

    def test_math_interface_cannot_omit_numeric_properties(self):
        self.assert_invalid(lambda d: (put(d, MATH + ("rewrite", 0, "symbol"), {"A": {"meaning": "Intermediate."}}),
            put(d, PHASE + ("decomposition", "target", "single_tile"), {"A": {"shape": [], "dtype": "fp32"}})), "publish shape")

    def test_realization_strengths(self):
        entry = {"item": "Y", "choice": "A concrete implementation.", "reason": "A specific cost."}
        self.assert_valid(lambda d: put(d, ("root", "realization"), {"require": [entry], "forbid": [
            dict(entry, choice="A different excluded path.")], "prefer": [
                dict(entry, feedback={"choice": "An alternate implementation.", "evidence": "testing.invalid/evidence@v1:result.md"})]}))
        for group in ("require", "forbid"):
            self.assert_invalid(lambda d: put(d, ("root", "realization"), {
                group: [dict(entry, feedback={"choice": "Alternative", "evidence": "repo@v1:file"})]}))
        self.assert_invalid(lambda d: put(d, ("root", "realization"), {"prefer": [dict(entry, item=["Y", "Y"])]}))
        self.assert_invalid(lambda d: put(d, ("root", "realization"), {"prefer": [
            dict(entry, feedback={"choice": "Alternate", "evidence": "<repo@pending:record>"})]}))

    def test_dispatch_identity_contract_and_tree(self):
        self.assert_valid(add_child)
        self.assert_invalid(lambda d: (add_child(d), put(d, ("small", "contract"), d["root"]["contract"])))
        self.assert_invalid(lambda d: (add_child(d), put(d, ("small", "meta_data", "name"), "different")))
        self.assert_invalid(lambda d: (add_child(d), put(d, ("small", "meta_data", "dispatch", "id"), "wrong")))
        self.assert_invalid(lambda d: (add_child(d), put(d, ("small", "meta_data", "dispatch", "parent"), "absent")), "parent")
        self.assert_invalid(lambda d: add_child(d, parent="small"), "cycle")
        self.assert_invalid(lambda d: (add_child(d, parent="other"), add_child(d, "other", "small")), "cycle")
        self.assert_invalid(lambda d: put(d, ("root", "meta_data", "dispatch"), {"parent": "small", "when": "All"}))
        self.assert_valid(lambda d: put(d, ("root", "meta_data", "dispatch"),
                                       {"when": "Select for every input in the contract domain."}))

    def test_validation_never_changes_sources_or_permissions(self):
        (self.directory / "dispatch.md").write_text("Synthetic task navigation.\n")
        root = self.directory / "root.yaml"
        for mode in (0o644, 0o444):
            root.chmod(mode)
            before = self.snapshot()
            self.assertEqual(self.base, schema.validate_directory(self.directory))
            self.assertEqual(before, self.snapshot())
        root.chmod(0o644)
        data = deepcopy(self.base)
        put(data, LANE + ("edge",), {"data": {"produce:Y": ["Missing/produce:Y"]}})
        self.assert_task_invalid(data, "root.yaml", "design.tile_pipeline.phase.copy.lane.Main.edge.data", "unknown lane")

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
        (self.directory / "small.yaml").write_text(json_value.dumps(data["small"]))
        (self.directory / "small.yml").write_text(json_value.dumps(data["small"]))
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

    def test_derived_identity_fields_are_checked_not_filled(self):
        for field in (("name",), ("describe",), ("dispatch", "id"), ("dispatch", "when")):
            data = deepcopy(self.base)
            add_child(data)
            drop(data, ("small", "meta_data") + field)
            with self.subTest(missing=field):
                self.assert_task_invalid(data, "small.yaml", "required")
        for path, value in [
            (("small", "meta_data", "name"), "other"),
            (("small", "meta_data", "describe"), "Different task."),
            (("small", "meta_data", "dispatch", "id"), "wrong"),
            (("root", "meta_data", "dispatch", "id"), "small"),
            (("small", "contract"), self.base["root"]["contract"]),
            (("small", "design"), {}),
        ]:
            data = deepcopy(self.base)
            add_child(data)
            put(data, path, value)
            with self.subTest(path=path):
                self.assert_task_invalid(data, path[0] + ".yaml")
        data = deepcopy(self.base)
        add_child(data)
        drop(data, ("root", "meta_data", "dispatch"))
        self.assert_task_invalid(data, "root.yaml", "dispatch", "required")

    def test_derived_tree_and_shared_contract(self):
        data = deepcopy(self.base)
        add_child(data)
        add_child(data, "tiny", "small")
        self.write_task(data)
        before = self.snapshot()
        self.assertEqual(data, schema.validate_directory(self.directory))
        self.assertEqual(before, self.snapshot())
        for node in ("small", "tiny"):
            self.assertNotIn("contract", data[node])
        for path, value, message in [
            (("small", "meta_data", "dispatch", "parent"), "absent", "unknown parent"),
            (("small", "meta_data", "dispatch", "parent"), "tiny", "cycle"),
            (("small", "design", "work_schedule", "phase", "copy", "mapping", "worker", "unit"), "Missing", "worker unit"),
            (("small", "design", "tile_pipeline", "phase", "copy", "lane", "Main", "info", "member"), "Missing[0:1]", "unregistered"),
            (("small", "design", "math_reshape", "rewrite", 0, "symbol"), {"X": {"meaning": "Repeated Contract input."}}, "duplicates"),
        ]:
            changed = deepcopy(data)
            put(changed, path, value)
            with self.subTest(path=path):
                self.assert_task_invalid(changed, "small.yaml", message)

    def test_explicit_contract_property_references_use_root(self):
        data = deepcopy(self.base)
        put(data, ALGO + ("constant",), {"Scale": {"meaning": "Test constant.", "value": 1}})
        put(data, ALGO + ("compute", 0, "symbol"), {
            "Original": {"meaning": "Contract intermediate.", "shape": "ref:X.shape", "dtype": "ref:X.dtype"}})
        add_child(data)
        shape_ref = "ref:contract.algorithm_requirement::Original.shape"
        dtype_ref = "ref:contract.algorithm_requirement::X.dtype"
        symbol_path = ("small", "design", "math_reshape", "rewrite", 0, "symbol", "A")
        target_path = ("small",) + PHASE[1:] + ("decomposition", "target", "single_tile", "Y", "dtype")
        resource_path = ("small",) + PIPE[1:] + ("resource", "Storage")
        put(data, ("small",) + MATH[1:] + ("rewrite", 0, "symbol"), {
            "A": {"meaning": "Test intermediate.", "shape": shape_ref, "dtype": dtype_ref}})
        put(data, target_path, dtype_ref)
        put(data, ("small",) + PIPE[1:] + ("resource",), {
            "Storage": {"space": "register", "shape": [], "dtype": dtype_ref, "view": {"Y": {"dtype": dtype_ref}}}})
        self.write_task(data)
        self.assertEqual(data, schema.validate_directory(self.directory))
        paths = [symbol_path + ("shape",), symbol_path + ("dtype",), target_path,
                 resource_path + ("dtype",), resource_path + ("view", "Y", "dtype")]
        for path in paths:
            changed = deepcopy(data)
            put(changed, path, "ref:contract.algorithm_requirement::Missing." + path[-1])
            with self.subTest(path=path):
                self.assert_task_invalid(changed, "small.yaml", "unknown shared Contract reference Missing")
        changed = deepcopy(data)
        put(changed, target_path, "ref:contract.algorithm_requirement::Scale.dtype")
        self.assert_task_invalid(changed, "small.yaml", "Scale has no dtype")
        # Removing the shared declaration invalidates the child's reference;
        # neither its own math symbols nor its parent's Design supply Contract.
        changed = deepcopy(data)
        drop(changed, ALGO + ("compute", 0, "symbol"))
        self.assert_task_invalid(changed, "small.yaml", "unknown shared Contract reference Original")

    def test_filename_defines_node_identity(self):
        data = deepcopy(self.base)
        add_child(data)
        self.write_task(data)
        (self.directory / "small.yaml").rename(self.directory / "other.yaml")
        with self.assertRaisesRegex(schema.SchemaError, "other.yaml"):
            schema.validate_directory(self.directory)
        (self.directory / "other.yaml").unlink()
        for name in ("Bad", "bad-name", "1bad"):
            (self.directory / (name + ".yaml")).write_text(json_value.dumps(data["small"]))
            with self.subTest(name=name), self.assertRaises(schema.SchemaError):
                schema.validate_directory(self.directory)
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

    def test_init_editable_yaml_and_unfilled_holes_fail(self):
        directory = self.directory / "new"
        target = schema.init(directory)
        self.assertEqual([target], list(directory.iterdir()))
        self.assertEqual("root.yaml", target.name)
        self.assertTrue(target.stat().st_mode & stat.S_IWUSR)
        with self.assertRaises(schema.SchemaError):
            schema.validate_directory(directory)
        before = target.read_bytes()
        with self.assertRaises(schema.SchemaError):
            schema.init(directory)
        self.assertEqual(before, target.read_bytes())
        target.write_text(self.source)
        self.assertEqual(self.base, schema.validate_directory(directory))

    def test_cli_validates_every_requested_task(self):
        (self.directory / "notes.md").write_text("Task notes are not plan input.\n")
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            status = schema.main(["validate", str(self.directory / "missing"), str(self.directory)])
        self.assertEqual(1, status)
        self.assertIn("missing", errors.getvalue())
        self.assertIn("Valid:", output.getvalue())
        self.assertEqual(self.base, schema.validate_directory(self.directory))


if __name__ == "__main__":
    unittest.main()
