"""Executable acceptance tests for the authoring contract and generation path."""
from __future__ import annotations

from copy import deepcopy
from contextlib import redirect_stdout
from decimal import Decimal
import io
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest

TEMPLATE_ROOT = Path(__file__).resolve().parents[1]
ROOT = TEMPLATE_ROOT.parent
sys.path.insert(0, str(TEMPLATE_ROOT / "tools"))
import schema
import compact_yaml
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
        cls.base = schema.compile_task(TEMPLATE_ROOT / "tests/fixtures/minimal")

    def assert_valid(self, transform=lambda d: None):
        data = deepcopy(self.base)
        transform(data)
        schema.validate_data(data)
        return data

    def assert_invalid(self, transform, message=None):
        data = deepcopy(self.base)
        transform(data)
        with self.assertRaises(schema.SchemaError) as caught:
            schema.validate_data(data)
        if message:
            self.assertIn(message, str(caught.exception))

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


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = schema.compile_task(TEMPLATE_ROOT / "tests/fixtures/minimal")

    def setUp(self):
        TEMPLATE_ROOT.joinpath(".tools").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=TEMPLATE_ROOT / ".tools", prefix="schema-test-")
        self.directory = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.write_source(deepcopy(self.base))

    def write_source(self, data):
        # JSON syntax is valid CUE. The generator must enforce #Task even when
        # the author forgot the import; these remain .cue source packages.
        (self.directory / "task.cue").write_text("package plan\nplans: " + json_value.dumps(data) + "\n")

    def test_build_check_readonly_and_reproducibility(self):
        schema.build(self.directory)
        before = {p.name: p.read_bytes() for p in self.directory.glob("*.yaml")}
        self.assertEqual(["root.yaml"], sorted(before))
        self.assertFalse((self.directory / "root.yaml").stat().st_mode & stat.S_IWUSR)
        schema.check_generated(self.directory)
        schema.build(self.directory)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.directory.glob("*.yaml")})

    def test_compact_yaml_round_trip_and_full_cue_validation(self):
        data = deepcopy(self.base)
        tensor = {"meaning": "Fixture tensor.", "shape": [{"row": "Rows"}, {"d": 128}], "dtype": "fp32"}
        at(data, ALGO)["input"]["X"] = tensor
        at(data, ALGO)["output"]["Y"] = deepcopy(tensor)
        at(data, ALGO)["compute"][0]["math"] = "For each row and d:\n  Y[row,d] = X[row,d].\n"
        text = compact_yaml.dumps(data["root"])
        self.assertIn('X: {meaning: "Fixture tensor.", shape: [row: Rows, d: 128], dtype: fp32}', text)
        self.assertIn("math: |\n", text)
        schema.cue("vet", "-c", "./schema_template:kernel", "-d", "#RootPlan", "yaml:", "-", data=text)
        decoded = json_value.loads(schema.cue("export", "yaml:", "-", "--out", "json", data=text))
        self.assertEqual(data["root"], decoded)
        schema.validate_data({"root": decoded})

    def test_yaml_presentation_does_not_relax_constraints(self):
        forms = [
            "meaning: Query\nshape:\n  - token: T\n  - d: 128\ndtype: bf16\n",
            "{meaning: Query, shape: [token: T, d: 128], dtype: bf16}\n",
            "{meaning: Query, shape: [{token: T}, {d: 128}], dtype: bf16}\n",
        ]
        decoded = []
        for text in forms:
            schema.cue("vet", "-c", "./schema_template:kernel", "-d", "#Output", "yaml:", "-", data=text)
            decoded.append(json_value.loads(schema.cue("export", "yaml:", "-", "--out", "json", data=text)))
            with self.assertRaises(schema.SchemaError):
                schema.cue("vet", "-c", "./schema_template:kernel", "-d", "#Output", "yaml:", "-",
                           data=text.replace("d: 128", "d: -1"))
            with self.assertRaises(schema.SchemaError):
                schema.cue("vet", "-c", "./schema_template:kernel", "-d", "#Output", "yaml:", "-",
                           data=text.replace("dtype:", "datatype:"))
        self.assertTrue(all(value == decoded[0] for value in decoded))

    def test_compact_yaml_preserves_scalars_and_multiline_whitespace(self):
        strings = ["", "true", "yes", "null", "1", "2026-10-26", "x: y", "[x,y]",
                   "a\nb", "a\nb\n", "a\n\n", "\n", "\n\n", "\n  indented\nout\n",
                   "  leading\nnormal", "a\tb\n", "a\r\nb", "a\u0085b", "a\u2028b", "a\u2029b",
                   "\x00", "Aᵀ ⊙ B", 'quote " and slash \\']
        value = {
            "strings": strings,
            "by_name": {f"case_{i}": v for i, v in enumerate(strings)},
            "numbers": [False, True, None, 0, -3, Decimal("1.000000000000000000000001"), Decimal("1E-10")],
            "shapes": [{"token": "T"}, {"d": 128}],
            "nested": [[0, 2], [2, 4]],
            "empty": {},
            "a: key": "value",
        }
        text = compact_yaml.dumps(value)
        decoded = json_value.loads(schema.cue("export", "yaml:", "-", "--out", "json", data=text))
        self.assertEqual(json_value.dumps(value, sort_keys=True), json_value.dumps(decoded, sort_keys=True))

    def test_edit_detection_and_explicit_repair(self):
        schema.build(self.directory)
        output = self.directory / "root.yaml"
        output.chmod(0o644)
        output.write_text(output.read_text() + "\n# Hand edit\n")
        with self.assertRaises(schema.SchemaError):
            schema.check_generated(self.directory)
        with self.assertRaises(schema.SchemaError):
            schema.build(self.directory)
        schema.build(self.directory, repair=True)
        schema.check_generated(self.directory)

    def test_failed_generation_preserves_outputs_and_no_import_bypass(self):
        schema.build(self.directory)
        before = (self.directory / "root.yaml").read_bytes()
        data = deepcopy(self.base)
        data["root"]["not_a_field"] = 1
        self.write_source(data)
        with self.assertRaises(schema.SchemaError):
            schema.build(self.directory)
        self.assertEqual(before, (self.directory / "root.yaml").read_bytes())

    def test_derived_export_omits_contract_and_removes_owned_stale_file(self):
        data = deepcopy(self.base)
        add_child(data)
        self.write_source(data)
        schema.build(self.directory)
        self.assertNotIn("\ncontract:", (self.directory / "small.yaml").read_text())
        next_data = deepcopy(self.base)
        next_data["root"]["meta_data"]["version"] = 2
        self.write_source(next_data)
        schema.build(self.directory)
        self.assertFalse((self.directory / "small.yaml").exists())
        schema.check_generated(self.directory)

    def test_handwritten_yaml_is_never_adopted_or_overwritten(self):
        legacy = self.directory / "root.yaml"
        legacy.write_text("handwritten: true\n")
        with self.assertRaises(schema.SchemaError):
            schema.build(self.directory)
        self.assertEqual("handwritten: true\n", legacy.read_text())

    def test_extra_public_field_and_unfilled_starter_fail(self):
        with (self.directory / "task.cue").open("a") as output:
            output.write("surprise: 1\n")
        with self.assertRaises(schema.SchemaError):
            schema.compile_task(self.directory)
        (self.directory / "task.cue").unlink()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(0, schema.main(["init", str(self.directory)]))
        self.assertTrue((self.directory / "task.cue").read_text().startswith("package plan\n"))
        with self.assertRaises(schema.SchemaError):
            schema.compile_task(self.directory)
        self.assertFalse(list(self.directory.glob("*.yaml")))

    def test_plan_revision_must_advance_but_comments_do_not_change_plan(self):
        schema.build(self.directory)
        with (self.directory / "task.cue").open("a") as output:
            output.write("// Source-only annotation.\n")
        schema.build(self.directory)
        data = deepcopy(self.base)
        at(data, MATH)["thesis"] = "Changed design rationale."
        self.write_source(data)
        with self.assertRaisesRegex(schema.SchemaError, "increment meta_data.version"):
            schema.build(self.directory)
        data["root"]["meta_data"]["version"] = 2
        self.write_source(data)
        schema.build(self.directory)
        schema.check_generated(self.directory)

    def test_decimal_values_do_not_round_through_python_float(self):
        data = deepcopy(self.base)
        exact = Decimal("1.000000000000000000000001")
        at(data, MATH)["rewrite"][0]["symbol"] = {
            "coef": {"meaning": "Exact fixture coefficient.", "value": exact}
        }
        self.write_source(data)
        compiled = schema.compile_task(self.directory)
        self.assertEqual(exact, at(compiled, MATH)["rewrite"][0]["symbol"]["coef"]["value"])
        schema.build(self.directory)
        self.assertIn(str(exact), (self.directory / "root.yaml").read_text())
        schema.check_generated(self.directory)

    def test_extra_yaml_and_removed_sources_fail_check(self):
        schema.build(self.directory)
        (self.directory / "extra.yaml").write_text("extra: true\n")
        with self.assertRaises(schema.SchemaError):
            schema.check_generated(self.directory)
        (self.directory / "extra.yaml").unlink()
        (self.directory / "task.cue").unlink()
        with self.assertRaises(schema.SchemaError):
            schema.check_generated(self.directory)

    def test_multiple_cue_files_compose_one_validated_task(self):
        data = deepcopy(self.base)
        add_child(data)
        (self.directory / "task.cue").write_text('package plan\nimport kernel "kernelschema.local/schema_template:kernel"\nplans: kernel.#Task\n')
        for name, plan in data.items():
            (self.directory / (name + ".cue")).write_text("package plan\nplans: " + json_value.dumps({name: plan}) + "\n")
        schema.build(self.directory)
        schema.check_generated(self.directory)
        self.assertEqual({"root.yaml", "small.yaml"}, {p.name for p in self.directory.glob("*.yaml")})


if __name__ == "__main__":
    unittest.main()
