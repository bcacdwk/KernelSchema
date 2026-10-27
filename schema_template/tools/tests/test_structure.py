"""Structure behavior through the same task checker used by authors."""
from __future__ import annotations

from copy import deepcopy
from .support import (TaskCase, ALGO, HARD, LANE, MATH, PHASE, PIPE, WORK, at, drop, put)


class StructureTests(TaskCase):
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
        self.assert_invalid(lambda d: put(d, PHASE + ("mapping", "count"), "0"), "literal")
        self.assert_invalid(lambda d: drop(d, PHASE + ("schedule", "worker_assignment", "preassigned")), "preassigned")
        self.assert_valid(lambda d: (drop(d, PHASE + ("schedule", "worker_assignment", "preassigned")),
                                    put(d, PHASE + ("schedule", "worker_assignment", "refill"), {"counter": "Fetch until exhausted."})))
        self.assert_invalid(lambda d: put(d, PHASE + ("decomposition", "worktile"),
            {"invariant": {"X": {"shape": [], "dtype": "fp32", "offset": [0]}}}))
        self.assert_invalid(lambda d: put(d, PHASE + ("decomposition", "worktile"),
            {"iteration": {"X": {"shape": [], "dtype": "fp32"}}}))

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

    def test_readable_design_notation_is_not_an_executable_language(self):
        def update(d):
            at(d, ("root", "meta_data"))["dispatch"] = {"when": "Use this plan for all valid inputs."}
            at(d, HARD)["stack"]["toolchain"] = "Compiler 1.0; documentation is tracked at <docs@pending:manual>."
            at(d, ALGO)["input"]["X"]["dtype"] = "vendor::float32"
            at(d, ALGO)["golden_reference"]["source"] = "testing.invalid@v1:reference.cc::ns::operator()"
            at(d, PHASE)["mapping"]["count"] = "min(ceil(N / B), available_workers(N))"
            at(d, PHASE)["decomposition"]["target"]["single_tile"]["Y"]["shape"] = [{"x": "ceil(N ÷ B)"}]
            at(d, LANE)["pipe"] = {"repeat": {"i": [0, 8]}, "prologue": ["produce:Y"]}
        self.assert_valid(update)
        # A repeated domain alone still says nothing about the lane's work.
        self.assert_invalid(lambda d: put(d, LANE + ("pipe",), {"repeat": {"i": [0, 8]}}), "actions")
        # Keep the documented single-interval member shape instead of inventing
        # a second region language solely because an old checker accepted it.
        self.assert_invalid(lambda d: put(d, PHASE + ("mapping", "member"), [{"x": [[0, 2], [4, 6]]}]))
        def numbered_phase(d):
            at(d, WORK)["phase"]["copy2"] = at(d, WORK)["phase"].pop("copy")
            at(d, WORK)["flow"]["launch"] = [{"phase": ["copy2"]}]
            phases = at(d, ("root", "design", "tile_pipeline", "phase"))
            phases["copy2"] = phases.pop("copy")
        self.assert_valid(numbered_phase)
