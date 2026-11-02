"""Structure behavior through the same task checker used by authors."""
from __future__ import annotations

from copy import deepcopy
from .support import (TaskCase, ALGO, HARD, ROLE, MATH, PHASE, PIPE, WORK, at, drop, put)


class StructureTests(TaskCase):
    def test_current_fields_preserve_validation_and_source_data(self):
        def resource(data):
            put(data, PIPE + ("resource",), {
                "R": {"slot_count": 1, "space": "SMEM", "member_slot_shape": [{"x": 8}], "dtype": "fp32"}})
        self.assert_valid(resource)
        for value in (0, None):
            self.assert_invalid(lambda d: put(d, PHASE + ("mapping", "worker_count"), value))
        self.assert_invalid(lambda d: drop(d, PHASE + ("mapping", "worker_count")), "worker_count")
        for key in ("slot_count", "member_slot_shape"):
            self.assert_invalid(lambda d: (resource(d), drop(d, PIPE + ("resource", "R", key))), key)
        self.assert_invalid(lambda d: (resource(d), put(d, PIPE + ("resource", "R", "member_slot_shape"), [{"x": -1}])))
        self.assert_invalid(lambda d: drop(d, ROLE + ("info", "execution_mode")), "execution_mode")

    def test_legacy_fields_are_rejected_without_aliases_or_conversion(self):
        def complete(data):
            put(data, PIPE + ("resource",), {"R": {"space": "SMEM", "slot_count": 1,
                "member_slot_shape": [{"x": 8}], "dtype": "fp32", "view": {"Y": {"slot_index": 0}}}})
            put(data, ROLE + ("waits_for",), {"data": {"produce:Y@done": ["produce:Y@issue"]}})
            put(data, MATH + ("rewrite", 0, "symbol"), {"Saved": {"meaning": "A saved value.",
                "shape": [], "dtype": "fp32", "retention": "keep_value invocation"}})
        locations = [
            (PHASE + ("mapping",), "worker_count", "count"),
            (PHASE + ("mapping", "worker"), "unit_count", "count"),
            (PIPE, "role", "lane"),
            (ROLE, "waits_for", "edge"),
            (ROLE + ("info",), "execution_mode", "mode"),
            (PIPE + ("resource", "R"), "member_slot_shape", "shape"),
            (PIPE + ("resource", "R"), "slot_count", "slot"),
            (PIPE + ("resource", "R", "view", "Y"), "slot_index", "slot"),
            (MATH + ("rewrite", 0, "symbol", "Saved"), "retention", "life"),
        ]
        for path, current, old in locations:
            for keep_current in (False, True):
                with self.subTest(path=path, old=old, keep_current=keep_current):
                    def legacy(data):
                        complete(data)
                        item = at(data, path)
                        item[old] = item[current]
                        if not keep_current:
                            del item[current]
                    self.assert_invalid(legacy, "field not allowed")
        for value in ("sync", "ordered", "ordered_async"):
            self.assert_invalid(lambda d: put(d, ROLE + ("info", "execution_mode"), value), "expected one of")
        for value in ("stored invocation", "carried invocation", "save_once invocation"):
            self.assert_invalid(lambda d: (complete(d), put(d, MATH + ("rewrite", 0, "symbol", "Saved", "retention"), value)), "carry_forward or keep_value")
        # Old words remain legal instance names, not alternate fixed fields.
        self.assert_valid(lambda d: put(d, MATH + ("rewrite", 0, "symbol"), {"life": {"meaning": "A named parameter.", "value": 1}}))

    def test_signed_domains_and_textual_integer_bounds(self):
        self.assert_valid(lambda d: put(d, ROLE + ("pipe",), {
            "repeat": {"i": [-1, 2]}, "steady": ["produce:Y(i)"]}))
        for path, value in [
            (PHASE + ("mapping", "member"), [{"x": ["-1", 4]}]),
            (PIPE + ("resource",), {"R": {"slot_count": 1, "space": "SMEM", "member_slot_shape": [{"x": 4}], "dtype": "fp32",
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
            (ALGO + ("compute", 0, "retention"), "keep_value invocation"),
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
            (PHASE + ("mapping", "placement"), {"unit": "InventedUnit", "unit_count": 1}),
            (PHASE + ("mapping", "worker_count"), -1),
            (PHASE + ("schedule", "worker_assignment", "preassigned"), {"start": 0, "step": 1, "count": 1}),
            (PHASE + ("schedule", "worker_assignment", "refill"), {}),
            (PHASE + ("schedule", "worker_assignment", "exit"), []),
            (ROLE + ("info", "execution_mode"), "parallel"),
            (ROLE + ("info", "register"), 0),
            (ROLE + ("pipe",), {}),
            (ROLE + ("pipe", "extra"), []),
            (ROLE + ("pipe", "steady"), []),
            (ROLE + ("waits_for",), {}),
            (ROLE + ("waits_for",), {"data": {"produce:Y": []}}),
            (("root", "realization"), {}),
            (("root", "realization"), {"thesis": "Not a realization field."}),
            (("root", "realization"), {"prefer": []}),
        ]
        for path, value in cases:
            with self.subTest(path=path, value=value):
                self.assert_invalid(lambda d: put(d, path, value))
        for path in [("root", "contract"), ALGO + ("input", "X", "meaning"),
                     ALGO + ("output", "Y", "dtype"), MATH + ("thesis",),
                     PHASE + ("mapping",), ROLE + ("pipe",)]:
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
                "Scalar": {"slot_count": 1, "space": "register", "member_slot_shape": [], "dtype": "fp32", "view": {"X": {}, "Y": {}}}
            }
        self.assert_valid(update)

    def test_hardware_template_pair(self):
        self.assert_invalid(lambda d: put(d, HARD + ("stack", "template"), "Library 1"))
        self.assert_invalid(lambda d: put(d, HARD + ("wiki", "programming", "template"), "repo@v1:docs/"))
        self.assert_valid(lambda d: (put(d, HARD + ("stack", "template"), "Library 1"),
                                     put(d, HARD + ("wiki", "programming", "template"), "repo@v1:docs/")))

    def test_required_non_numeric_sentinel_is_not_missing_information(self):
        sentinel = {"meaning": "Required Python None sentinel; optional scaling is disabled.",
                    "shape": [], "dtype": "python_none", "range": "Python None only."}
        path = ALGO + ("input", "scaling")
        self.assert_valid(lambda d: put(d, path, sentinel))
        # A known singleton type does not authorize absent data or an unknown type.
        for update in ({"dtype": None}, {"dtype": "<schema-abi@pending:None>"}, {"default": None}):
            with self.subTest(update=update):
                self.assert_invalid(lambda d: put(d, path, dict(sentinel, **update)))

    def test_math_symbols_and_ledger(self):
        row = {"item": "Traffic", "change": "-4 bytes per invocation", "by": ["copy"]}
        self.assert_valid(lambda d: put(d, MATH + ("ledger",), {"baseline": "Direct copy.", "traffic": [row]}))
        self.assert_invalid(lambda d: put(d, MATH + ("ledger",),
                                         {"baseline": "Direct", "traffic": [dict(row, by=["missing"])]}))
        self.assert_invalid(lambda d: at(d, MATH)["rewrite"].append(deepcopy(at(d, MATH)["rewrite"][0])))
        self.assert_invalid(lambda d: put(d, MATH + ("rewrite", 0, "symbol"),
                                         {"A": {"meaning": "A parameter.", "binding": "bind:design.work_schedule::A", "value": 1}}))
        self.assert_invalid(lambda d: put(d, MATH + ("rewrite", 0, "symbol"),
                                         {"A": {"meaning": "A state.", "retention": "carry_forward invocation"}}))
        self.assert_invalid(lambda d: put(d, MATH + ("rewrite", 0, "symbol"),
                                         {"A": {"meaning": "Not a tensor.", "shape": "not_applicable"}}))
        self.assert_invalid(lambda d: put(d, MATH + ("rewrite", 0, "symbol"),
                                         {"A": {"meaning": "Known value required.", "value": "unknown: value"}}))
        self.assert_valid(lambda d: put(d, MATH + ("rewrite", 0, "symbol"), {
            "A": {"meaning": "State with declared missing evidence.", "retention": "keep_value invocation",
                  "shape": "unknown: shape not yet established", "dtype": "unknown: format not yet established"},
            "Coeff": {"meaning": "Fixed coefficients.", "value": [[1, 2], [3, 4]]},
            "Size": {"meaning": "Downstream parameter.", "binding": "bind:design.work_schedule::Size"},
        }))

    def test_work_dimensions_and_assignment(self):
        def partition(d, size="N", tile=1, bounds=(1, None)):
            put(d, PHASE + ("decomposition", "partition"), {"x": {"size": size, "tile": tile, "range": list(bounds)}})
            put(d, PHASE + ("schedule", "worktile_order"), [{"by": "x", "direction": "ascending"}])
        self.assert_valid(partition)
        self.assert_valid(lambda d: partition(d, bounds=(0, None)))
        self.assert_valid(lambda d: partition(d, size=0, bounds=(0, 0)))
        for fields in ({"size": -1}, {"bounds": (-1, None)}, {"bounds": (0, -1)}, {"tile": 0}):
            with self.subTest(fields=fields):
                self.assert_invalid(lambda d: partition(d, **fields), "literal must be >=")
        self.assert_valid(lambda d: (partition(d), put(d, PHASE + ("schedule", "worktile_order"),
            [{"by": ["x"], "group": 8, "direction": "ascending"}, {"by": "x", "direction": "descending"}])))
        self.assert_invalid(lambda d: (partition(d), drop(d, PHASE + ("schedule", "worktile_order"))))
        self.assert_invalid(lambda d: (partition(d), put(d, PHASE + ("schedule", "worktile_order", 0, "by"), "missing")))
        self.assert_invalid(lambda d: (partition(d), put(d, PHASE + ("decomposition", "partition", "x", "range"), [5, 2])), "reversed")
        self.assert_invalid(lambda d: put(d, PHASE + ("mapping", "worker_count"), "0"), "literal")
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
            at(d, PHASE)["mapping"]["worker_count"] = "min(ceil(N / B), available_workers(N))"
            at(d, PHASE)["decomposition"]["target"]["single_tile"]["Y"]["shape"] = [{"x": "ceil(N ÷ B)"}]
            at(d, ROLE)["pipe"] = {"repeat": {"i": [0, 8]}, "prologue": ["produce:Y"]}
        self.assert_valid(update)
        # A repeated domain alone still says nothing about the role's work.
        self.assert_invalid(lambda d: put(d, ROLE + ("pipe",), {"repeat": {"i": [0, 8]}}), "actions")
        # Keep the documented single-interval member shape instead of inventing
        # a second region language solely because an old checker accepted it.
        self.assert_invalid(lambda d: put(d, PHASE + ("mapping", "member"), [{"x": [[0, 2], [4, 6]]}]))
        def numbered_phase(d):
            at(d, WORK)["phase"]["copy2"] = at(d, WORK)["phase"].pop("copy")
            at(d, WORK)["flow"]["launch"] = [{"phase": ["copy2"]}]
            phases = at(d, ("root", "design", "tile_pipeline", "phase"))
            phases["copy2"] = phases.pop("copy")
        self.assert_valid(numbered_phase)
