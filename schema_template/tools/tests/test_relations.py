"""Relations behavior through the same task checker used by authors."""
from __future__ import annotations

from copy import deepcopy
from .. import schema
from ..rules import validate_algorithm
from .support import (TaskCase, ALGO, ROLE, MATH, PHASE, PIPE, WORK, add_child, add_phase, at, drop, put)


class RelationsTests(TaskCase):
    def test_contract_only_gate_reuses_shapes_interfaces_and_property_relations(self):
        original = at(self.base, ALGO)
        self.assertFalse(validate_algorithm(deepcopy(original)))
        cases = [
            (lambda a: a.pop("output"), "output"),
            (lambda a: a["golden_reference"].update(tolerance={"Z": "exact"}), "every Contract output"),
            (lambda a: a["golden_reference"]["tolerance"].update(Z="exact"), "undeclared output"),
            (lambda a: a["compute"][0].update(symbol={"X": {"meaning": "Duplicate", "shape": [], "dtype": "fp32"}}), "duplicate Contract"),
            (lambda a: a["compute"][0].update(symbol={"A": {"meaning": "Intermediate", "shape": [], "dtype": "ref:Missing.dtype"}}), "unknown Contract"),
            (lambda a: a["compute"][0].update(symbol={"A": {"meaning": "Intermediate", "shape": [], "dtype": "ref:A.dtype"}}), "cycle"),
            (lambda a: a["input"]["X"].update(shape=[{"D": 2}, {"D": 3}]), "axis names"),
            (lambda a: a["input"]["X"].update(range=[2, 1]), "range endpoints are reversed"),
            (lambda a: a["input"]["X"].update(shape=[{"D": "Y"}]), "runtime extent"),
        ]
        for change, message in cases:
            with self.subTest(message=message):
                algorithm = deepcopy(original)
                change(algorithm)
                self.assertTrue(any(message in str(error) for error in validate_algorithm(algorithm)))
        unknown = deepcopy(original)
        unknown["golden_reference"]["tolerance"]["Y"] = "Exact; external arithmetic and nonfinite handling are not specified."
        self.assertFalse(validate_algorithm(unknown))

    def test_integer_literal_spellings_have_identical_resource_and_member_checks(self):
        for literal in (int, str):
            with self.subTest(spelling=literal.__name__):
                resource = {"space": "SMEM", "member_slot_shape": [{"x": literal(8)}],
                            "dtype": "fp32", "slot_count": literal(2)}
                self.assert_invalid(lambda d: put(d, PIPE + ("resource",), {"R": resource}), "multi-slot")
                resource["view"] = {"Y": {"slot_index": literal(1), "part": {"x": [literal(0), literal(8)]}}}
                self.assert_valid(lambda d: put(d, PIPE + ("resource",), {"R": resource}))
                resource["view"]["Y"]["slot_index"] = literal(2)
                self.assert_invalid(lambda d: put(d, PIPE + ("resource",), {"R": resource}), "slot_index exceeds")
                resource["view"]["Y"]["slot_index"] = literal(1)
                resource["view"]["Y"]["part"]["x"] = [literal(0), literal(9)]
                self.assert_invalid(lambda d: put(d, PIPE + ("resource",), {"R": resource}), "exceeds parent storage")
                resource["view"]["Y"]["part"]["x"] = [literal(4), literal(2)]
                self.assert_invalid(lambda d: put(d, PIPE + ("resource",), {"R": resource}), "reversed")
                self.assert_invalid(lambda d: (
                    put(d, PHASE + ("mapping", "worker", "unit_count"), literal(2)),
                    put(d, ROLE + ("info", "member"), "CTA[2].warp[0:1]")), "member index exceeds")
                self.assert_invalid(lambda d: (
                    put(d, PHASE + ("mapping", "worker", "unit_count"), literal(2)),
                    put(d, PHASE + ("mapping", "member"), [{"x": [0, 8]}])), "member entries")
                self.assert_invalid(lambda d: put(d, ALGO + ("input", "X", "range"), [literal(4), 2]), "reversed")
        # General expressions still belong to design review, not this literal check.
        self.assert_valid(lambda d: put(d, PIPE + ("resource",), {"R": {
            "space": "SMEM", "member_slot_shape": [{"x": "size(N)"}], "dtype": "fp32", "slot_count": "stages(N)"}}))

    def test_shared_workers_compare_integer_literals_without_evaluating_expressions(self):
        def setup(data, value):
            add_phase(data, order="all")
            put(data, WORK + ("phase", "next", "mapping", "worker_count"), value)
        self.assert_valid(lambda d: setup(d, "1"))
        self.assert_invalid(lambda d: setup(d, "2"), "incompatible worker mappings")
        self.assert_valid(lambda d: setup(d, "available_workers(N)"))

    def test_worker_counts_and_resource_capacity_are_checked(self):
        def shared(data):
            add_phase(data, order="all")
        self.assert_valid(shared)
        self.assert_invalid(lambda d: (shared(d), put(d, WORK + ("phase", "next", "mapping", "worker_count"), 2)),
                            "incompatible worker mappings")
        resource = {"slot_count": 1, "space": "SMEM", "member_slot_shape": [{"x": 8}], "dtype": "fp32"}
        self.assert_invalid(lambda d: put(d, PIPE + ("resource",), {"R": dict(resource,
            view={"Y": {"part": {"x": [0, 9]}}})}), "exceeds parent storage")
        self.assert_invalid(lambda d: put(d, PIPE + ("resource",), {"R": dict(resource,
            member_slot_shape=[{"x": 8}, {"x": 2}])}), "member_slot_shape")

    def test_explicit_event_member_must_exist_in_its_lane(self):
        def setup(data, count, member):
            put(data, PHASE + ("mapping", "worker", "unit_count"), count)
            put(data, ROLE + ("info", "member"), "CTA[0].warp[0:1]")
            put(data, ROLE + ("waits_for",), {"data": {"produce:Y@done": [f"Main[{member}]/produce:Y@issue"]}})
        for count in (2, "MemberCount"):
            self.assert_invalid(lambda d: setup(d, count, 1), "role Main has no member 1")
        self.assert_valid(lambda d: setup(d, 2, 0))

    def test_obvious_self_waits_and_valid_same_action_references(self):
        for target, source in (("produce:Y", "produce:Y"), ("produce:Y@issue", "Main/produce:Y@issue"),
                               ("produce:Y@done", "produce:Y@done")):
            self.assert_invalid(lambda d: put(d, ROLE + ("waits_for",), {"data": {target: [source]}}), "cannot depend on itself")
        self.assert_valid(lambda d: put(d, ROLE + ("waits_for",), {"data": {"produce:Y@done": ["produce:Y@issue"]}}))
        self.assert_valid(lambda d: (
            put(d, PHASE + ("mapping", "worker", "unit_count"), 2),
            put(d, ROLE + ("waits_for",), {"sync": {"Main[1]/produce:Y": ["Main[0]/produce:Y"]}})))

    def test_local_conditions_prevent_ambiguous_references(self):
        self.assert_invalid(lambda d: (add_phase(d), put(d, WORK + ("flow", "handoff"), {
            "Y": {"producer": "copy", "consumer": ["copy", "next"], "storage": "global"}})), "another declared phase")
        self.assert_valid(lambda d: put(d, PHASE + ("mapping", "member"), [{"x": [0, 4]}]))
        self.assert_invalid(lambda d: put(d, PHASE + ("mapping", "member"), [{"x": [-1, 4]}]), "literal")
        for name in ("kernel_done", "launch_ready", "kernel_done(i)", "launch_ready(i)"):
            self.assert_invalid(lambda d: put(d, PIPE + ("milestone",), {name: ["Main/produce:Y"]}), "built-in")

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
            put(d, ROLE + ("pipe",), block)
        self.assert_valid(nested)
        def corrupt(d):
            nested(d)
            block = at(d, ROLE)["pipe"]
            for _ in range(10):
                block = block["steady"][0]
            block["typo"] = "must fail even at depth 10"
        self.assert_invalid(corrupt)
        self.assert_invalid(lambda d: put(d, ROLE + ("pipe",),
                                         {"repeat": {"i": [0, 8]}, "prologue": ["load:X(i)"], "steady": ["produce:Y(i)"]}), "not bound")
        self.assert_invalid(lambda d: put(d, ROLE + ("pipe",),
                                         {"repeat": "missing", "steady": ["produce:Y(i)"]}), "undefined repeat")
        self.assert_invalid(lambda d: put(d, PIPE + ("domain",),
                                         {"a": {"flatten": ["b"]}, "b": {"flatten": ["a"]}}), "cycle")
        self.assert_invalid(lambda d: put(d, ROLE + ("pipe",), {"repeat": {"i": [0, 8]}}))
        self.assert_valid(lambda d: (put(d, PIPE + ("domain",), {"t": "worktile", "i": [0, 8], "n": {"flatten": ["t", "i"]}}),
                                     put(d, ROLE + ("pipe",), {"repeat": "t", "steady": [
                                         {"repeat": "n[t]", "steady": ["produce:Y(n,t,i)"], "epilogue": ["store:Y(n[t,last(i)])"]}]})))

    def test_action_events_and_custom_verbs(self):
        self.assert_invalid(lambda d: put(d, ROLE + ("pipe", "steady"), ["magical:Y"]), "custom verb")
        self.assert_valid(lambda d: (put(d, ROLE + ("pipe", "steady"), ["magical:Y"]),
                                     put(d, ROLE + ("info", "verb"), {"magical": {"meaning": "Produce Y.", "execution_mode": "serial"}})))
        self.assert_invalid(lambda d: put(d, ROLE + ("waits_for",), {"data": {"produce:Y": ["Missing/produce:Y"]}}), "unknown role")
        self.assert_invalid(lambda d: put(d, ROLE + ("waits_for",), {"data": {"produce:Y": ["produce:Missing"]}}), "action family")
        self.assert_invalid(lambda d: put(d, ROLE + ("waits_for",), {"data": {"produce:Y": ["produce:Y@read"]}}), "notation")
        self.assert_valid(lambda d: (put(d, ROLE + ("pipe", "steady"), ["load:X", "mma:X@X", "produce:Y"]),
                                     put(d, ROLE + ("waits_for",), {"data": {"produce:Y@issue": ["mma:X@X@done"]}})))
        self.assert_invalid(lambda d: put(d, ROLE + ("info", "member"), "warp[0:0]"), "nonempty")
        self.assert_invalid(lambda d: put(d, ROLE + ("info", "member"), "phantom[0:1]"), "unregistered")

    def test_resource_bounds_and_view_exceptions(self):
        resource = {"space": "SMEM", "member_slot_shape": [{"x": 8}], "dtype": "fp32", "slot_count": 2, "view": {"Y": {"slot_index": 1}}}
        self.assert_valid(lambda d: put(d, PIPE + ("resource",), {"R": resource}))
        for replacement in [
            dict(resource, slot_count=0), dict(resource, view={"Y": {}}),
            dict(resource, view={"Y": {"slot_index": 2}}),
            dict(resource, view={"Y": {"slot_index": 0, "part": {"x": [0, 9]}}}),
            dict(resource, view={"Y": {"slot_index": 0, "part": {"other": [0, 2]}}}),
            dict(resource, member=[0, 0]), dict(resource, member=1),
        ]:
            with self.subTest(resource=replacement):
                self.assert_invalid(lambda d: put(d, PIPE + ("resource",), {"R": replacement}))

    def test_local_handoff_and_shared_worker_resource_identity(self):
        resource = {"slot_count": 1, "space": "SMEM", "member_slot_shape": [], "dtype": "fp32"}
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
            put(d, ROLE + ("pipe",), {"repeat": "t", "steady": ["produce:Y(t)"]})
        self.assert_valid(dynamic)
        self.assert_invalid(lambda d: (dynamic(d), put(d, ROLE + ("pipe", "epilogue"), ["store:Y(last(t))"])), "count/last")
        # Contract prose is not parsed as an executed expression.
        self.assert_valid(lambda d: (dynamic(d), put(d, ROLE + ("info", "verb"),
            {"produce": "Produce the result; do not query count(t) or last(t)."})))
        self.assert_invalid(lambda d: put(d, ROLE + ("pipe", "steady"), ["produce:Y(i"]), "notation")
        self.assert_invalid(lambda d: (put(d, PIPE + ("domain",), {"i": [0, 8]}),
            put(d, ROLE + ("pipe",), {"repeat": "i[0:8", "steady": ["produce:Y(i)"]})))

    def test_instance_names_can_match_field_names(self):
        self.assert_valid(lambda d: (
            put(d, PIPE + ("domain",), {"shape": [0, 8]}),
            put(d, ROLE + ("pipe",), {"repeat": "shape", "steady": ["produce:Y(shape)"]}),
            put(d, PIPE + ("milestone",), {"shape": ["Main/produce:Y(0)"]})))

    def test_domain_queries_follow_nested_scope_and_restore_the_outer_domain(self):
        for query in ("last(t)", "count(t)", "count ( t )"):
            with self.subTest(query=query):
                def nested(d):
                    put(d, PIPE + ("domain",), {"t": "worktile"})
                    put(d, PHASE + ("schedule", "worker_assignment", "refill"), {"counter": "Take work until exhausted."})
                    put(d, ROLE + ("pipe",), {
                        "repeat": "t",
                        "prologue": [{"repeat": {"t": [0, 2]}, "steady": ["zero:Init(t)"],
                                      "epilogue": [f"release:Init({query})"]}],
                        "steady": ["produce:Y(t)"],
                    })
                self.assert_valid(nested)
                self.assert_invalid(lambda d: (nested(d), put(d, ROLE + ("pipe", "epilogue"),
                                                               [f"store:Y({query})"])), "count/last")
                self.assert_invalid(lambda d: (nested(d), put(d, ROLE + ("pipe", "steady"),
                    [{"repeat": {"i": [0, query]}, "steady": ["produce:Y(t,i)"]}])), "count/last")

    def test_local_worktile_queries_obey_the_assignment_policy(self):
        for domain, refill, valid in (("worktile", "none", True), ([0, 2], {"counter": "Take work."}, True),
                                       ("worktile", {"counter": "Take work."}, False)):
            for query in ("last(t)", "count(t)"):
                with self.subTest(domain=domain, refill=refill, query=query):
                    def task(d):
                        put(d, PHASE + ("schedule", "worker_assignment", "refill"), refill)
                        put(d, ROLE + ("pipe",), {"repeat": {"t": domain}, "steady": ["produce:Y(t)"],
                                                  "epilogue": [f"store:Y({query})"]})
                    if valid:
                        self.assert_valid(task)
                    else:
                        self.assert_invalid(task, "count/last")

    def test_event_queries_use_the_referenced_action_scope(self):
        for domain, valid in (([0, 2], True), ("worktile", False)):
            for event_field in ("waits_for", "milestone"):
                with self.subTest(domain=domain, event_field=event_field):
                    def task(d):
                        put(d, PHASE + ("schedule", "worker_assignment", "refill"), {"counter": "Take work."})
                        put(d, PIPE + ("domain",), {"t": "worktile"})
                        put(d, ROLE + ("pipe",), {"repeat": {"t": domain}, "steady": ["produce:Y(t)"],
                                                  "epilogue": ["store:Result"]})
                        if event_field == "waits_for":
                            put(d, ROLE + ("waits_for",), {"data": {"store:Result": ["produce:Y(last(t))"]}})
                        else:
                            put(d, PIPE + ("milestone",), {"ready": ["produce:Y(last(t))"]})
                    if valid:
                        self.assert_valid(task)
                    else:
                        self.assert_invalid(task, "count/last")

    def test_per_worktile_order_accepts_scalar_and_singleton_keys(self):
        for first, second in [("x", ["x"]), (["x"], "x")]:
            with self.subTest(first=first, second=second):
                def equivalent(d):
                    put(d, PHASE + ("decomposition", "partition"), {"x": {"size": 8, "tile": 1}})
                    put(d, PHASE + ("schedule", "worktile_order"), [{"by": first, "direction": "ascending"}])
                    add_phase(d, order="per_worktile")
                    put(d, WORK + ("phase", "next", "schedule", "worktile_order", 0, "by"), second)
                self.assert_valid(equivalent)

    def test_per_worktile_order_preserves_dimension_order(self):
        def mismatch(d):
            put(d, PHASE + ("decomposition", "partition"),
                {"x": {"size": 8, "tile": 1}, "y": {"size": 4, "tile": 1}})
            put(d, PHASE + ("schedule", "worktile_order"), [{"by": ["x", "y"], "direction": "ascending"}])
            add_phase(d, order="per_worktile")
            put(d, WORK + ("phase", "next", "schedule", "worktile_order", 0, "by"), ["y", "x"])
        self.assert_invalid(mismatch, "matching coordinates")

    def test_per_worktile_order_must_agree(self):
        def mismatch(d):
            put(d, PHASE + ("decomposition", "partition"), {"x": {"size": 8, "tile": 1}})
            put(d, PHASE + ("schedule", "worktile_order"), [{"by": "x", "direction": "ascending"}])
            add_phase(d, order="per_worktile")
            put(d, WORK + ("phase", "next", "schedule", "worktile_order", 0, "direction"), "descending")
        self.assert_invalid(mismatch, "matching coordinates")

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
            (("small", "design", "tile_pipeline", "phase", "copy", "role", "Main", "info", "member"), "Missing[0:1]", "unregistered"),
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
            "Storage": {"slot_count": 1, "space": "register", "member_slot_shape": [], "dtype": dtype_ref, "view": {"Y": {"dtype": dtype_ref}}}})
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
