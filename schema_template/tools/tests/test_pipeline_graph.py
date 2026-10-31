"""Finite event semantics, negative examples and the data-driven trial corpus."""
from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from graphlib import CycleError, TopologicalSorter
import subprocess
import sys
import unittest

from ..pipeline import GraphError, build_phase
from ..pipeline.model import ExpansionError
from ..pipeline.expressions import Domains
from ..pipeline.render import display_members, dot_text
from ..pipeline_graph import configuration, read_sample
from .. import schema
from .support import ROLE, PIPE, PHASE, TaskCase, TEMPLATE_ROOT, add_child, put


def role(sequence, execution_mode="serial", member="warp[0:1]", **extra):
    return {"info": {"member": member, "execution_mode": execution_mode}, "pipe": {"steady": sequence}, **extra}


def build(roles, domains=None, count=1, parameters=None, worktiles=2):
    return build_phase("main", {"domain": domains or {}, "role": roles},
                       {"unit": "CTA", "unit_count": count}, parameters or {}, worktiles)


def edge(graph, source, target, kind):
    return ("main/" + source, "main/" + target, kind) in graph.edges


class EventGraphTests(unittest.TestCase):
    def test_graph_builder_rejects_unknown_execution_modes(self):
        for value in ("sync", "ordered", "ordered_async", "parallel"):
            with self.subTest(execution_mode=value), self.assertRaisesRegex(GraphError, "unsupported execution_mode"):
                build({"Main": role(["compute:X"], value)})

    def test_modes_and_synchronous_exception(self):
        for execution_mode, event in (("serial", "done"), ("async", "issue"), ("async_ordered_completion", "issue")):
            with self.subTest(execution_mode=execution_mode):
                body = role(["compute:A", "compute:B"], execution_mode)
                graph = build({"Main": body})
                self.assertTrue(edge(graph, f"Main[0]/compute:A@{event}", "Main[0]/compute:B@issue", "program"))
                self.assertEqual(edge(graph, "Main[0]/compute:A@done", "Main[0]/compute:B@done", "completion"), execution_mode == "async_ordered_completion")
        body = role(["compute:A", "release:X", "compute:B"], "async_ordered_completion")
        body["info"]["verb"] = {"release": {"execution_mode": "serial", "meaning": "Read complete."}}
        graph = build({"Main": body})
        self.assertTrue(edge(graph, "Main[0]/release:X@done", "Main[0]/compute:B@issue", "program"))
        self.assertTrue(edge(graph, "Main[0]/compute:A@done", "Main[0]/compute:B@done", "completion"))
        self.assertFalse(edge(graph, "Main[0]/compute:A@done", "Main[0]/release:X@issue", "program"))

    def test_nested_flatten_and_slices(self):
        body = role([])
        body["pipe"] = {"repeat": "t", "steady": [
            {"repeat": "n[t,1:]", "steady": ["compute:Y(n,t,i)"]}, "store:Z(t)"]}
        graph = build({"Main": body}, {"t": "worktile", "i": [0, 3], "n": {"flatten": ["t", "i"]}})
        self.assertEqual([op.action for op in graph.operations.values()],
                         ["compute:Y(1,0,1)", "compute:Y(2,0,2)", "store:Z(0)",
                          "compute:Y(4,1,1)", "compute:Y(5,1,2)", "store:Z(1)"])

    def test_empty_steady_keeps_prefix_and_tail(self):
        body = role([])
        body["pipe"] = {"repeat": {"i": [0, 0]}, "prologue": ["zero:X"],
                        "steady": ["compute:Y(i)"], "epilogue": ["store:Z"]}
        self.assertEqual([o.action for o in build({"Main": body}).operations.values()], ["zero:X", "store:Z"])

    def test_pipe_never_drops_shifted_operations(self):
        body = role([])
        body["pipe"] = {"repeat": "i", "steady": ["compute:Y(i+1)"]}
        graph = build({"Main": body}, {"i": [0, 2]})
        self.assertEqual([o.action for o in graph.operations.values()], ["compute:Y(1)", "compute:Y(2)"])

    def test_duplicate_identity_is_lane_and_member_local(self):
        self.assertEqual(len(build({"A": role(["load:X"]), "B": role(["load:X"])}, count=2).operations), 4)
        with self.assertRaisesRegex(GraphError, "duplicate action"):
            build({"A": role([{ "repeat": {"i": [0, 2]}, "steady": ["load:X"]}])})

    def test_ring_edges_skip_only_outside_declared_window(self):
        load = role([{ "repeat": "k", "steady": ["load:X(k)"]}], "async",
                    waits_for={"reuse": {"load:X(k+2)": ["Compute/compute:Y(k)"]}})
        compute = role([{ "repeat": "k", "steady": ["compute:Y(k)"]}],
                       waits_for={"data": {"compute:Y(k)": ["Load/load:X(k)"]}})
        graph = build({"Load": load, "Compute": compute}, {"k": [0, 4]})
        self.assertFalse(graph.cycle)
        self.assertEqual(len(graph.boundaries), 2)
        self.assertTrue(edge(graph, "Compute[0]/compute:Y(0)@done", "Load[0]/load:X(2)@issue", "reuse"))
        compute["pipe"]["steady"] = [
            {"repeat": "k[:1]", "steady": ["compute:Y(k)"]},
            {"repeat": "k[2:]", "steady": ["compute:Y(k)"]}]
        with self.assertRaisesRegex(GraphError, "dangling event"):
            build({"Load": load, "Compute": compute}, {"k": [0, 4]})

    def test_unrolled_and_repeated_actions_share_the_declared_window(self):
        sequences = [
            ["compute:X(0)", "compute:X(1)"],
            [{"repeat": "i", "steady": ["compute:X(i)"]}],
            ["compute:X(0)", {"repeat": "i[1:]", "steady": ["compute:X(i)"]}],
        ]
        graphs = [build({"Main": role(sequence, waits_for={
            "data": {"compute:X(i+1)": ["compute:X(i)"]}})}, {"i": [0, 2]}) for sequence in sequences]
        for graph in graphs:
            self.assertFalse(graph.cycle)
            self.assertEqual(len(graph.boundaries), 1)
            self.assertEqual(set(graph.edges), set(graphs[0].edges))

    def test_unrolled_actions_do_not_shrink_domains_or_hide_holes(self):
        for sequence, stop, missing in ((["compute:X(0)", "compute:X(1)"], 4, "X(2)"),
                                        (["compute:X(0)", "compute:X(2)"], 3, "X(1)")):
            with self.subTest(sequence=sequence), self.assertRaises(GraphError) as caught:
                build({"Main": role(sequence, waits_for={
                    "data": {"compute:X(i+1)": ["compute:X(i)"]}})}, {"i": [0, stop]})
            self.assertNotIsInstance(caught.exception, ExpansionError)
            self.assertIn("dangling event", str(caught.exception))
            self.assertIn(missing, str(caught.exception))

    def test_shifted_unrolled_window_needs_an_identifiable_domain(self):
        rules = {"data": {"compute:X(i+1)": ["compute:X(i-1)"]}}
        repeated = role([{"repeat": "i", "steady": ["compute:X(i+1)"]}], waits_for=rules)
        self.assertFalse(build({"Main": repeated}, {"i": [0, 2]}).cycle)
        with self.assertRaisesRegex(ExpansionError, "cannot determine declared index window"):
            build({"Main": role(["compute:X(1)", "compute:X(2)"], waits_for=rules)}, {"i": [0, 2]})

    def test_dependent_domains_keep_outer_coordinates_and_ignore_variable_spelling(self):
        graphs = []
        for index in ("i", "z"):
            body = role([{"repeat": "t", "steady": [
                {"repeat": index, "steady": [f"compute:X(t,{index})"]}]}], waits_for={
                "data": {f"compute:X(t,{index}+1)": [f"compute:X(t,{index})"]}})
            graph = build({"Main": body}, {"t": "worktile", index: [0, "t+1"]}, worktiles=3)
            self.assertFalse(graph.cycle)
            self.assertEqual(len(graph.operations), 6)
            self.assertEqual(len(graph.boundaries), 3)
            self.assertEqual(sum(kind == "data" for _, _, kind in graph.edges), 3)
            graphs.append(graph)
        self.assertEqual(set(graphs[0].edges), set(graphs[1].edges))

    def test_shifting_outer_coordinate_rechecks_the_dependent_inner_domain(self):
        body = role([{"repeat": "t", "steady": [{"repeat": "i", "steady": ["compute:X(t,i)"]}]}],
                    waits_for={"data": {"compute:X(t,i)": ["compute:X(t-1,i)"]}})
        graph = build({"Main": body}, {"t": "worktile", "i": [0, "t+1"]}, worktiles=3)
        self.assertFalse(graph.cycle)
        self.assertEqual(len(graph.boundaries), 3)
        self.assertEqual(sum(kind == "data" for _, _, kind in graph.edges), 3)

    def test_offset_outer_coordinate_matches_direct_coordinate_notation(self):
        shifted = role([{"repeat": "t", "steady": [
            {"repeat": "i", "steady": ["compute:X(t+2,i)"]}]}],
            waits_for={"data": {"compute:X(t+2,i)": ["compute:X(t+1,i)"]}})
        direct = role([{"repeat": "u", "steady": [
            {"repeat": "i", "steady": ["compute:X(u,i)"]}]}],
            waits_for={"data": {"compute:X(u,i)": ["compute:X(u-1,i)"]}})
        a = build({"Main": shifted}, {"t": "worktile", "i": [0, "t+1"]})
        b = build({"Main": direct}, {"u": [2, 4], "i": [0, "u-1"]})
        self.assertEqual(set(a.edges), set(b.edges))
        self.assertEqual(len(a.boundaries), 2)
        self.assertFalse(a.cycle)

    def test_unrelated_local_loops_do_not_change_an_action_domain(self):
        body = role([
            {"repeat": {"k": [0, 2]}, "steady": ["compute:X(k)"]},
            {"repeat": {"k": [0, 3]}, "steady": ["compute:Y(k)"]},
        ], waits_for={"data": {"compute:X(k+1)": ["compute:X(k)"],
                               "compute:Y(k+1)": ["compute:Y(k)"]}})
        graph = build({"Main": body})
        self.assertFalse(graph.cycle)
        self.assertEqual(len(graph.boundaries), 2)
        self.assertEqual(sum(kind == "data" for _, _, kind in graph.edges), 3)

    def test_local_domain_queries_and_flatten_references_keep_their_definitions(self):
        body = role([{"repeat": {"k": [0, 2]}, "steady": ["compute:X(k)"]}, "compute:X(2)"],
                    waits_for={"data": {"compute:X(count(k))": ["compute:X(last(k))"]}})
        graph = build({"Main": body})
        self.assertTrue(edge(graph, "Main[0]/compute:X(1)@done", "Main[0]/compute:X(2)@issue", "data"))

        body = role([{"repeat": {"t": [0, 1]}, "steady": [
            {"repeat": {"i": [0, "count(t)"]}, "steady": ["compute:X(i)"]}]}],
            waits_for={"data": {"compute:X(i+1)": ["compute:X(i)"]}})
        graph = build({"Main": body})
        self.assertEqual(len(graph.operations), 1)
        self.assertEqual(len(graph.boundaries), 1)

        body = role([{"repeat": {"n": {"flatten": ["t", "i"]}}, "steady": ["compute:X(n)"]}],
                    waits_for={"data": {"compute:X(n[t,i+1])": ["compute:X(n[t,i])"]}})
        graph = build({"Main": body}, {"t": "worktile", "i": [0, 2]})
        self.assertEqual(len(graph.boundaries), 2)
        self.assertEqual(sum(kind == "data" for _, _, kind in graph.edges), 2)

    def test_variable_length_flatten_distinguishes_local_and_cross_work_shifts(self):
        domains = {"t": "worktile", "i": [0, "t+1"], "n": {"flatten": ["t", "i"]}}
        for target, source, dependencies, skips in (
            ("compute:X(n+1)", "compute:X(n)", 2, 1),
            ("compute:X(n[t,i+1])", "compute:X(n[t,i])", 1, 2),
        ):
            with self.subTest(target=target):
                body = role([{"repeat": "n", "steady": ["compute:X(n)"]}],
                            waits_for={"data": {target: [source]}})
                graph = build({"Main": body}, domains)
                self.assertFalse(graph.cycle)
                self.assertEqual(len(graph.boundaries), skips)
                self.assertEqual(sum(kind == "data" for _, _, kind in graph.edges), dependencies)

    def test_inactive_unsupported_math_does_not_block_the_sample(self):
        body = role([])
        body["pipe"] = {"repeat": {"i": [0, 0]}, "steady": ["compute:X(i ÷ 2)"],
                        "epilogue": ["produce:Final"]}
        graph = build({"Main": body})
        self.assertEqual([op.action for op in graph.operations.values()], ["produce:Final"])

    def test_fixed_missing_reference_is_not_a_boundary(self):
        body = role([{ "repeat": "k", "steady": ["load:X(k)"]}],
                    waits_for={"data": {"load:X(0)": ["load:X(99)"]}})
        with self.assertRaisesRegex(GraphError, "dangling event"):
            build({"Main": body}, {"k": [0, 2]})
        body["waits_for"]["data"] = {"load:X(k+99)": ["load:TYPO(k)"]}
        with self.assertRaisesRegex(GraphError, "unknown action family"):
            build({"Main": body}, {"k": [0, 2]})

    def test_shifted_boundary_does_not_hide_a_broken_fixed_reference(self):
        for target, source in (("compute:X(i+99)", "compute:X(99)"),
                               ("compute:X(99)", "compute:X(i+99)")):
            body = role([{"repeat": "i", "steady": ["compute:X(i)"]}],
                        waits_for={"data": {target: [source]}})
            with self.subTest(target=target), self.assertRaisesRegex(GraphError, "dangling event") as caught:
                build({"Main": body}, {"i": [0, 2]})
            self.assertNotIsInstance(caught.exception, ExpansionError)
        body["waits_for"] = {"data": {"compute:X(i+99)": ["compute:X(0)"]}}
        self.assertEqual(len(build({"Main": body}, {"i": [0, 2]}).boundaries), 2)

    def test_unshifted_reference_never_silently_shrinks_its_window(self):
        source = role([{ "repeat": "j", "steady": ["load:X(j)"]}])
        target = role([{ "repeat": "i", "steady": ["compute:Y(i)"]}],
                      waits_for={"data": {"compute:Y(i)": ["Load/load:X(i)"]}})
        with self.assertRaisesRegex(GraphError, "dangling event"):
            build({"Load": source, "Main": target}, {"i": [0, 4], "j": [0, 2]})

    def test_boundary_checks_both_endpoints_with_mixed_flatten_coordinates(self):
        domains = {"t": [0, 2], "i": [0, 2], "n": {"flatten": ["t", "i"]}}
        for left, right in (("n+99", "n[99,i]"), ("n[99,i]", "n+99")):
            body = role([{"repeat": "n", "steady": ["compute:X(n)"]}], "async",
                        waits_for={"data": {f"compute:X({left})": [f"compute:X({right})"]}})
            with self.subTest(left=left), self.assertRaisesRegex(GraphError, "t=99 outside") as caught:
                build({"Main": body}, domains)
            self.assertNotIsInstance(caught.exception, ExpansionError)
        for left, right, skips in (("n[t,i+1]", "n[t,i]", 2), ("n+99", "n[0,i]", 4),
                                    ("n[0,i]", "n+99", 4)):
            body = role([{"repeat": "n", "steady": ["compute:X(n)"]}], "async",
                        waits_for={"data": {f"compute:X({left})": [f"compute:X({right})"]}})
            graph = build({"Main": body}, domains)
            self.assertFalse(graph.cycle)
            self.assertEqual(len(graph.boundaries), skips)

    def test_boundary_checks_members_and_expression_support_before_action_lookup(self):
        for reference, error_type, message in (
            ("Main[1]/compute:X(i)", GraphError, "no role instance"),
            ("Main/compute:X(f(i))", ExpansionError, "unsupported expression"),
        ):
            body = role([{"repeat": "i[:1]", "steady": ["compute:X(i)"]}],
                        waits_for={"data": {"compute:X(i+99)": [reference]}})
            with self.subTest(reference=reference), self.assertRaisesRegex(error_type, message):
                build({"Main": body}, {"i": [0, 2]})
        body = role([{"repeat": "i", "steady": ["compute:X(i)"]},
                     {"repeat": "i[:1]", "steady": ["release:Y(i)"]}], "async",
                    waits_for={"data": {"compute:X(i+2)": ["release:Y(i)"]}})
        graph = build({"Main": body}, {"i": [0, 3]})
        self.assertEqual(len(graph.boundaries), 2)
        self.assertTrue(edge(graph, "Main[0]/release:Y(0)@done", "Main[0]/compute:X(2)@issue", "data"))

    def test_independent_coordinates_and_operands_cannot_hide_fixed_errors(self):
        domains = {"t": [0, 2], "i": [0, 2], "n": {"flatten": ["t", "i"]}}
        for action, reference in (
            ("compute:X(n)", "compute:X(n[t+99,99])"),
            ("compute:X(n)", "compute:X(n[t+99,0]+n[99,i])"),
            ("compute:X(n)", "compute:X(max(n[t+99,0],n[99,i]))"),
            ("compute:X(n,0)", "compute:X(n[t+99,0],n[99,i])"),
            ("mma:X(n)@Y(n)", "mma:X(n[t+99,0])@Y(n[99,i])"),
        ):
            body = role([{"repeat": "n", "steady": [action]}], "async",
                        waits_for={"data": {action: [reference]}})
            with self.subTest(reference=reference), self.assertRaisesRegex(GraphError, "99 outside") as caught:
                build({"Main": body}, domains)
            self.assertNotIsInstance(caught.exception, ExpansionError)
        domains["i"] = [0, "2-t"]
        body = role([{"repeat": "n", "steady": ["compute:X(n)"]}], "async",
                    waits_for={"data": {"compute:X(n[t+1,0])": ["compute:X(n[t,i])"]}})
        graph = build({"Main": body}, domains)
        self.assertFalse(graph.cycle)
        self.assertEqual(len(graph.boundaries), 1)
        self.assertEqual(sum(kind == "data" for _, _, kind in graph.edges), 2)

    def test_nested_flatten_restores_and_invalidates_its_coordinates(self):
        domains = {"t": [0, 2], "i": [0, 1], "p": {"flatten": ["t", "i"]},
                   "j": [0, "2-t"], "n": {"flatten": ["p", "j"]}}
        for target, source, skips in (("n", "n[p+99,99]", 3), ("n[p+1,0]", "n", 1)):
            body = role([{"repeat": "n", "steady": ["compute:X(n)"]}], "async",
                        waits_for={"data": {f"compute:X({target})": [f"compute:X({source})"]}})
            graph = build({"Main": body}, domains)
            self.assertFalse(graph.cycle)
            self.assertEqual(len(graph.boundaries), skips)
        domains["j"] = [0, 2]
        body["waits_for"] = {"data": {"compute:X(n)": ["compute:X(n[p+99,99])"]}}
        with self.assertRaisesRegex(GraphError, "j=99 outside"):
            build({"Main": body}, domains)

    def test_each_predecessor_has_its_own_boundary(self):
        body = role([{ "repeat": "i", "steady": ["compute:Y(i)"]}],
                    waits_for={"data": {"compute:Y(i)": ["Load/load:X(i)", "compute:Y(i-1)"]}})
        graph = build({"Load": role([{ "repeat": "i", "steady": ["load:X(i)"]}]), "Main": body}, {"i": [0, 3]})
        self.assertTrue(edge(graph, "Load[0]/load:X(0)@done", "Main[0]/compute:Y(0)@issue", "data"))

    def test_full_local_flatten_coordinate_is_strict(self):
        body = role(["compute:Y(n[0,0])"], waits_for={"data": {"compute:Y(n[0,0])": ["compute:Y(n[0,9])"]}})
        with self.assertRaisesRegex(GraphError, "outside its declared domain"):
            build({"Main": body}, {"t": "worktile", "i": [0, 2], "n": {"flatten": ["t", "i"]}})

    def test_local_edge_domain_and_ambiguous_shadowing(self):
        body = role([{ "repeat": {"k": [0, 3]}, "steady": ["compute:Y(k)"]}],
                    waits_for={"data": {"compute:Y(k+1)": ["compute:Y(k)"]}})
        self.assertFalse(build({"Main": body}).cycle)
        body["pipe"]["steady"].append({"repeat": {"k": [3, 4]}, "steady": ["compute:Y(k)"]})
        with self.assertRaisesRegex(ExpansionError, "ambiguous local edge index"):
            build({"Main": body})

    def test_custom_output_is_not_an_implicit_consumer(self):
        body = role(["prefetch:X", "publish:X"], "async",
                    waits_for={"data": {"publish:X": ["prefetch:X"]}})
        graph = build({"Main": body})
        self.assertFalse(graph.cycle)
        self.assertFalse(edge(graph, "Main[0]/publish:X@done", "Main[0]/prefetch:X@issue", "data"))
        graph = build({"Load": role(["load:X"]), "Compute": role(["mma:X@Y"])})
        self.assertTrue(edge(graph, "Load[0]/load:X@done", "Compute[0]/mma:X@Y@issue", "data"))

    def test_same_member_all_members_and_explicit_member(self):
        graph = build({"Load": role(["load:X"]), "Compute": role(["mma:X@Y"])}, count=2)
        self.assertTrue(edge(graph, "Load[0]/load:X@done", "Compute[0]/mma:X@Y@issue", "data"))
        self.assertFalse(edge(graph, "Load[1]/load:X@done", "Compute[0]/mma:X@Y@issue", "data"))
        graph = build({"Load": role(["load:X"]), "Compute": role(["mma:X@Y"], member="CTA[0].warp[1:2]")}, count=2)
        self.assertTrue(edge(graph, "Load[1]/load:X@done", "Compute[0]/mma:X@Y@issue", "data"))
        consumer = role(["compute:Y"], waits_for={"data": {"compute:Y": ["Load[*]/load:X"]}})
        graph = build({"Load": role(["load:X"]), "Compute": consumer}, count=2)
        self.assertTrue(edge(graph, "Load[1]/load:X@done", "Compute[0]/compute:Y@issue", "data"))
        consumer["waits_for"]["data"]["compute:Y"] = ["Load[1]/load:X"]
        with self.assertRaisesRegex(GraphError, "no role instance"):
            build({"Load": role(["load:X"], member="CTA[0].warp[0:1]"), "Compute": consumer}, count=2)

    def test_cycles_include_issue_done_and_source_identity(self):
        graph = build({"A": role(["compute:X"], waits_for={"sync": {"compute:X": ["B/compute:Y"]}}),
                       "B": role(["compute:Y"], waits_for={"sync": {"compute:Y": ["A/compute:X"]}})})
        self.assertTrue(graph.cycle)
        self.assertEqual(graph.cycle[0], graph.cycle[-1])
        self.assertIn("main/A[0]/compute:X@issue", graph.cycle)

    def test_waiting_member_does_not_collect_other_members_predecessors(self):
        consumer = role(["compute:Y"], waits_for={"data": {"Compute[0]/compute:Y": ["Load/load:X"]}})
        graph = build({"Load": role(["load:X"]), "Compute": consumer}, count=2)
        self.assertTrue(edge(graph, "Load[0]/load:X@done", "Compute[0]/compute:Y@issue", "data"))
        self.assertFalse(edge(graph, "Load[1]/load:X@done", "Compute[0]/compute:Y@issue", "data"))

    def test_indexed_milestones_remain_separate(self):
        phase = {"domain": {"i": [0, 2]}, "role": {"Main": role([{ "repeat": "i", "steady": ["publish:X(i)"]}])},
                 "milestone": {"ready(i)": ["Main/publish:X(i)"]}}
        graph = build_phase("main", phase, {"unit": "CTA", "unit_count": 1}, {})
        self.assertEqual(graph.milestones, {"ready(0)": ["main/Main[0]/publish:X(0)@done"],
                                           "ready(1)": ["main/Main[0]/publish:X(1)@done"]})

    def test_done_can_wait_for_a_later_issue(self):
        body = role(["load:X(0)", "load:X(1)", "release:Tail"], "async",
                    waits_for={"sync": {"load:X(0)@done": ["load:X(1)@issue"]}})
        graph = build({"Main": body})
        self.assertFalse(graph.cycle)
        self.assertTrue(edge(graph, "Main[0]/load:X(1)@issue", "Main[0]/load:X(0)@done", "sync"))

    def test_member_folding_preserves_asymmetry(self):
        graph = build({"Main": role(["compute:X"])}, count=2)
        self.assertEqual(display_members(graph), {"Main"})
        body = role(["compute:X"], waits_for={"sync": {"Main[1]/compute:X": ["Main[0]/compute:X"]}})
        graph = build({"Main": body}, count=2)
        self.assertFalse(display_members(graph))

    def test_sync_event_display_preserves_interleaving_and_cycles(self):
        def displayed_graph(graph):
            dot = dot_text("Event boundary review", [graph],
                           {"main": {"parameters": {}, "worktiles": 1}})
            predecessors = {}
            for line in dot.splitlines():
                match = re.match(r'("(?:[^"\\]|\\.)*") -> ("(?:[^"\\]|\\.)*") ', line)
                if match and "style=invis" not in line:
                    source, target = (json.loads(value) for value in match.groups())
                    predecessors.setdefault(target, set()).add(source)
            return predecessors

        # B may run after the synchronous wait starts, and finish before it
        # returns. Contracting A's two events must not turn this into a cycle.
        for count in (1, 2):
            graph = build({
                "A": role(["wait:Gate"], waits_for={"sync": {"wait:Gate@done": ["B/compute:Ready"]}}),
                "B": role(["compute:Ready"], member="warp[1:2]",
                          waits_for={"sync": {"compute:Ready": ["A/wait:Gate@issue"]}}),
            }, count=count)
            self.assertFalse(graph.cycle)
            tuple(TopologicalSorter(displayed_graph(graph)).static_order())

        # A real issue/done cycle must remain visible as well.
        graph = build({"A": role(["wait:Gate"], waits_for={"sync": {"wait:Gate": ["wait:Gate@done"]}})})
        self.assertTrue(graph.cycle)
        with self.assertRaises(CycleError):
            tuple(TopologicalSorter(displayed_graph(graph)).static_order())

        # Ordinary synchronous sequences still use one display node per action.
        graph = build({"A": role(["compute:X", "compute:Y"])})
        shown = displayed_graph(graph)
        self.assertTrue(shown)
        self.assertTrue(all(event.endswith("@done") for event in shown))

    def test_exact_arithmetic_and_no_code_execution(self):
        domains = Domains({}, {"N": 6}, 2)
        self.assertEqual(domains.integer("ceil(N/4) + floor(N/4)", {}), 3)
        self.assertEqual(Domains({}, {"0": 99, "i": 99}, 2).integer("i+0", {"i": 2}), 2)
        with self.assertRaisesRegex(GraphError, "noninteger"):
            domains.integer("N/4", {})
        for expression in ("__import__('os').getcwd()", "2**100000", "x.attr", "1/0"):
            with self.subTest(expression=expression), self.assertRaises(GraphError):
                domains.integer(expression, {})
        with self.assertRaisesRegex(GraphError, "oversized"):
            Domains({"i": [0, 20001]}, {}, 2).entries("i", {})
        with self.assertRaises(ExpansionError):
            build({"Main": role(["produce:Y"])}, count=65)

    def test_deep_domain_chains_report_a_tool_limit_without_recursion_crashes(self):
        domains = {"d0": [0, 1]}
        for i in range(1, 1100):
            domains[f"d{i}"] = [0, f"min(1,d{i-1}+1)"]
        body = role(["compute:X(0)"], waits_for={"data": {"compute:X(d1099+1)": ["compute:X(d1099)"]}})
        with self.assertRaisesRegex(ExpansionError, "dependency depth"):
            build({"Main": body}, domains)

    def test_supplied_math_expression_does_not_require_an_index_domain(self):
        domains = Domains({"i": [0, "count(S)+1"]}, {"count(S)": 2}, 1)
        self.assertEqual(list(domains.contexts({"i"})), [{"i": 0}, {"i": 1}, {"i": 2}])

    def test_all_trial_examples_and_representative_scales(self):
        total = 0
        for directory in sorted((TEMPLATE_ROOT / "examples").iterdir()):
            if not directory.is_dir() or not (directory / "root.yaml").is_file():
                continue
            sample = read_sample(directory / "pipeline_samples.json")
            for name, plan in schema.validate_directory(directory).items():
                config = configuration(sample, name, {}, None)
                for pid, phase in plan["design"]["tile_pipeline"]["phase"].items():
                    worker = plan["design"]["work_schedule"]["phase"][pid]["mapping"]["worker"]
                    with self.subTest(task=directory.name, node=name, phase=pid):
                        graph = build_phase(pid, phase, worker, **config)
                        self.assertFalse(graph.cycle, graph.cycle)
                        self.assertTrue(graph.operations)
                total += 1
        self.assertEqual(total, 21)

    def test_qsa_empty_and_nonempty_workers_keep_valid_tail_handling(self):
        directory = TEMPLATE_ROOT / "examples/qsa_prefill"
        plan = schema.validate_directory(directory)["root"]
        for worktiles in (0, 1, 2):
            with self.subTest(worktiles=worktiles):
                graph = build_phase("attention", plan["design"]["tile_pipeline"]["phase"]["attention"],
                                    plan["design"]["work_schedule"]["phase"]["attention"]["mapping"]["worker"],
                                    {"N(b)": 32}, worktiles)
                self.assertFalse(graph.cycle, graph.cycle)
                actions = [op.action for op in graph.operations.values()]
                self.assertIn("drain:KeyRing", actions)
                self.assertIn("drain:ValueRing", actions)
                self.assertEqual(sum(a.startswith("load:Kblk(") for a in actions), 32 * worktiles)
                self.assertEqual(sum(a.startswith("load:Vblk(") for a in actions), 32 * worktiles)

    def test_decode_empty_and_short_splits_only_wait_for_existing_setup(self):
        directory = TEMPLATE_ROOT / "examples/fa4_decode"
        sample = read_sample(directory / "pipeline_samples.json")
        for name, plan in schema.validate_directory(directory).items():
            for pid, phase in plan["design"]["tile_pipeline"]["phase"].items():
                if "active" not in phase.get("domain", {}):
                    continue
                worker = plan["design"]["work_schedule"]["phase"][pid]["mapping"]["worker"]
                for count in (0, 1, 2, 6):
                    with self.subTest(node=name, phase=pid, iterations=count):
                        config = configuration(sample, name, {"N_r": count}, None)
                        members = Domains({}, config["parameters"], config["worktiles"]).integer(worker["unit_count"], {})
                        graph = build_phase(pid, phase, worker, **config)
                        self.assertFalse(graph.cycle, graph.cycle)
                        actions = [op.action for op in graph.operations.values()]
                        self.assertEqual(actions.count("zero:Y(0)"), members * int(count > 0))
                        self.assertEqual(sum(a.startswith("score:ScoreBlock(") for a in actions), members * count)
                        if name == "root":
                            for action in ("initialize:Statistic", "publish:FinalMaximum", "publish:FinalSum", "store:Mpartial", "store:Lpartial"):
                                self.assertIn(action, actions)


class GraphCliTests(TaskCase):
    def run_graph(self, *options, env=None):
        return subprocess.run([sys.executable, "-B", str(TEMPLATE_ROOT / "tools/pipeline_graph.py"),
                               str(self.directory / "root.yaml"), "--out", str(self.directory / "figures"), *options],
                              cwd="/tmp", capture_output=True, text=True, env=env)

    def test_missing_sample_is_not_reported_as_invalid_yaml(self):
        task = deepcopy(self.base)
        put(task, PIPE + ("domain",), {"i": [0, "N"]})
        put(task, ROLE + ("pipe",), {"repeat": "i", "steady": ["produce:Y(i)"]})
        self.write_task(task)
        result = self.run_graph("--format", "dot")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("Graph not completed", result.stderr)
        self.assertIn("phase.copy.role.Main.pipe.repeat", result.stderr)
        self.assertIn("--set", result.stderr)
        result = self.run_graph("--format", "dot", "--set", "N=3")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_mixed_fixed_coordinate_failure_has_context_and_exit_one(self):
        task = deepcopy(self.base)
        put(task, PIPE + ("domain",), {"t": [0, 2], "i": [0, 2], "n": {"flatten": ["t", "i"]}})
        put(task, ROLE + ("pipe",), {"repeat": "n", "steady": ["produce:Y(n)"]})
        put(task, ROLE + ("waits_for",), {"data": {"produce:Y(n+99)": ["produce:Y(n[99,i])"]}})
        self.write_task(task)
        schema.validate_directory(self.directory)
        result = self.run_graph("--format", "dot")
        self.assertEqual(result.returncode, 1, result.stderr)
        for text in ("Main.waits_for.data", "context=", "t=99 outside"):
            self.assertIn(text, result.stderr)
        self.assertNotIn("Acyclic", result.stdout)

    def test_local_domain_scope_is_preserved_through_the_cli(self):
        task = deepcopy(self.base)
        put(task, PHASE + ("schedule", "worker_assignment", "refill"), {"counter": "Take work until exhausted."})
        put(task, PIPE + ("domain",), {"t": "worktile"})
        put(task, ROLE + ("pipe",), {
            "repeat": "t", "steady": ["produce:Y(t)"],
            "prologue": [{"repeat": {"t": [0, 2]}, "steady": ["zero:Init(t)"],
                          "epilogue": ["release:Init(last(t))"]}],
        })
        self.write_task(task)
        before = (self.directory / "root.yaml").read_bytes()
        result = self.run_graph("--format", "dot", "--worktiles", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((self.directory / "figures/root.json").read_text())
        self.assertEqual([op["action"] for op in report["phases"][0]["operations"]],
                         ["zero:Init(0)", "zero:Init(1)", "release:Init(1)", "produce:Y(0)", "produce:Y(1)"])
        self.assertEqual(before, (self.directory / "root.yaml").read_bytes())
        put(task, PIPE + ("domain",), {"i": [0, 2]})
        put(task, ROLE + ("pipe",), {"repeat": {"t": "worktile"}, "steady": ["produce:Y(t)"],
                                  "epilogue": ["store:Y(last(t))"]})
        self.write_task(task)
        result = self.run_graph("--format", "dot", "--worktiles", "2")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("count/last", result.stderr)
        self.assertFalse((self.directory / "figures/root.json").exists())

    def test_unsupported_indexed_math_is_not_reported_as_invalid_yaml(self):
        for expression in ("offset[i]", "offset[0]", "f(i)"):
            with self.subTest(expression=expression):
                task = deepcopy(self.base)
                put(task, PIPE + ("domain",), {"i": [0, 2]})
                put(task, ROLE + ("pipe",), {"repeat": "i", "steady": [f"produce:Y({expression})"]})
                self.write_task(task)
                schema.validate_directory(self.directory)
                result = self.run_graph("--format", "dot")
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("Graph not completed", result.stderr)
                self.assertIn("phase.copy.role.Main.pipe", result.stderr)
                self.assertIn("unsupported", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_unidentifiable_window_is_reported_as_incomplete_expansion(self):
        task = deepcopy(self.base)
        put(task, PIPE + ("domain",), {"i": [0, 2]})
        put(task, ROLE + ("pipe",), {"steady": ["produce:Y(1)", "produce:Y(2)"]})
        put(task, ROLE + ("waits_for",), {"data": {"produce:Y(i+1)": ["produce:Y(i-1)"]}})
        self.write_task(task)
        schema.validate_directory(self.directory)
        result = self.run_graph("--format", "dot")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("cannot determine declared index window", result.stderr)
        self.assertIn("Graph not completed", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_duplicate_sample_key_and_bad_options_have_clear_diagnostics(self):
        path = self.directory / "pipeline_samples.json"
        path.write_text('{"parameters":{"N":3,"N":9}}')
        result = self.run_graph("--format", "dot")
        self.assertEqual(result.returncode, 2)
        self.assertIn("duplicate sample key 'N'", result.stderr)
        self.assertIn(str(path), result.stderr)
        path.unlink()
        for options in (("--set", "=4"), ("--worktiles", "-1")):
            result = self.run_graph("--format", "dot", *options)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_dot_only_refresh_removes_stale_pictures(self):
        output = self.directory / "figures"
        result = self.run_graph("--format", "dot")
        self.assertEqual(result.returncode, 0, result.stderr)
        for suffix in (".svg", ".view.html"):
            (output / ("root" + suffix)).write_text("old picture")
        result = self.run_graph("--format", "dot")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((output / "root.dot").is_file())
        self.assertFalse((output / "root.svg").exists())
        self.assertFalse((output / "root.view.html").exists())

    def test_failed_refresh_invalidates_results_and_gallery(self):
        output = self.directory / "figures"
        sample = self.directory / "pipeline_samples.json"
        for failure, code in (("schema", 1), ("yaml", 1), ("sample", 2), ("sample_node", 2),
                              ("option", 2), ("parameter", 2), ("event", 1)):
            with self.subTest(failure=failure):
                self.write_task(deepcopy(self.base))
                sample.unlink(missing_ok=True)
                result = self.run_graph("--format", "dot", "--all")
                self.assertEqual(result.returncode, 0, result.stderr)
                for suffix in (".svg", ".view.html"):
                    (output / ("root" + suffix)).write_text("previous successful rendering")
                (output / "notes.txt").write_text("keep unrelated files")
                options = []
                if failure == "yaml":
                    (self.directory / "root.yaml").write_text("meta_data: [")
                elif failure == "sample":
                    sample.write_text('{"parameters":')
                elif failure == "sample_node":
                    sample.write_text('{"nodes":{"missing":{}}}')
                elif failure == "option":
                    options = ["--set", "N=not-an-integer"]
                else:
                    task = deepcopy(self.base)
                    if failure == "schema":
                        put(task, ROLE + ("info", "execution_mode"), "sync")
                    elif failure == "parameter":
                        put(task, ROLE + ("pipe",), {"repeat": {"i": [0, "N"]}, "steady": ["produce:Y(i)"]})
                    else:
                        put(task, ROLE + ("pipe",), {"steady": ["produce:Y(0)"]})
                        put(task, ROLE + ("waits_for",), {"data": {"produce:Y(0)": ["produce:Y(99)"]}})
                    self.write_task(task)
                source = self.directory / "root.yaml"
                before = (source.read_bytes(), source.stat().st_mtime_ns, source.stat().st_mode)
                result = self.run_graph("--format", "dot", *options)
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertEqual({p.name for p in output.iterdir()}, {"notes.txt"})
                self.assertEqual(before, (source.read_bytes(), source.stat().st_mtime_ns, source.stat().st_mode))

    def test_all_refresh_retires_removed_nodes_without_deleting_unrelated_reports(self):
        task = deepcopy(self.base)
        add_child(task)
        self.write_task(task)
        result = self.run_graph("--format", "dot", "--all")
        self.assertEqual(result.returncode, 0, result.stderr)
        output = self.directory / "figures"
        foreign = json.loads((output / "root.json").read_text())
        foreign["source"] = str(self.directory.parent / "another_task/root.yaml")
        (output / "foreign.json").write_text(json.dumps(foreign))
        (output / "unrelated.json").write_text('{"notes":"not a pipeline report"}')
        self.write_task(deepcopy(self.base))
        result = self.run_graph("--format", "dot", "--all")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual({p.name for p in output.iterdir()},
                         {"root.dot", "root.json", "index.html", "foreign.json", "unrelated.json"})
        self.assertNotIn("small.dot", (output / "index.html").read_text())

    def test_failed_whole_task_check_clears_every_requested_node(self):
        task = deepcopy(self.base)
        add_child(task)
        self.write_task(task)
        result = self.run_graph("--format", "dot", "--all")
        self.assertEqual(result.returncode, 0, result.stderr)
        put(task, ("small", "design", "tile_pipeline", "phase", "copy", "role", "Main", "info", "execution_mode"), "sync")
        self.write_task(task)
        result = self.run_graph("--format", "dot", "--all")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(list((self.directory / "figures").iterdir()), [])

    def test_default_output_directory_has_the_same_failure_lifecycle(self):
        command = [sys.executable, "-B", str(TEMPLATE_ROOT / "tools/pipeline_graph.py"),
                   str(self.directory / "root.yaml"), "--format", "dot"]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        (self.directory / "pipeline_samples.json").write_text("{")
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(list((self.directory / "pipeline_graphs").iterdir()), [])

    def test_missing_graphviz_keeps_checked_graph_data(self):
        output = self.directory / "figures"
        result = self.run_graph("--format", "dot")
        self.assertEqual(result.returncode, 0, result.stderr)
        for suffix in (".svg", ".view.html"):
            (output / ("root" + suffix)).write_text("previous successful rendering")
        result = self.run_graph(env={**os.environ, "PATH": ""})
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("Graphviz", result.stderr)
        self.assertTrue((self.directory / "figures/root.dot").is_file())
        self.assertTrue((self.directory / "figures/root.json").is_file())
        self.assertFalse((self.directory / "figures/root.svg").exists())
        self.assertFalse((self.directory / "figures/root.view.html").exists())
        self.assertEqual({p.name for p in output.iterdir()}, {"root.dot", "root.json"})

    def test_output_conflicts_preserve_unrelated_files_even_for_missing_inputs(self):
        output = self.directory / "figures"
        output.mkdir()
        (output / "root.json").write_text('{"notes":"unrelated report"}')
        (output / "index.html").write_text("independent index")
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        for exists in (True, False):
            if not exists:
                (self.directory / "root.yaml").unlink()
            result = self.run_graph("--format", "dot", "--all")
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("output conflicts", result.stderr)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_cycle_diagnostic_includes_the_rules_that_close_it(self):
        task = deepcopy(self.base)
        put(task, ROLE + ("waits_for",), {"data": {"produce:Y": ["Peer/produce:Y"]}})
        put(task, PIPE + ("role", "Peer"), {"info": {"execution_mode": "serial", "member": "warp[1:2]"}, "pipe": {"steady": ["produce:Y"]},
                                           "waits_for": {"data": {"produce:Y": ["Main/produce:Y"]}}})
        self.write_task(task)
        result = self.run_graph("--format", "dot")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("DEPENDENCY CYCLE", result.stdout)
        self.assertIn("Main.waits_for.data", result.stderr)
        self.assertIn("Peer.waits_for.data", result.stderr)

    def test_external_task_read_only_and_deterministic_output(self):
        command = [sys.executable, "-B", str(TEMPLATE_ROOT / "tools/pipeline_graph.py"),
                   str(self.directory / "root.yaml"), "--format", "dot", "--out", str(self.directory / "figures")]
        before = (self.directory / "root.yaml").stat().st_mtime_ns
        result = subprocess.run(command, cwd="/tmp", capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        files = self.directory / "figures"
        report = json.loads((files / "root.json").read_text())
        self.assertTrue(report["acyclic"])
        first = (files / "root.dot").read_bytes()
        result = subprocess.run(command, cwd="/tmp", capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(first, (files / "root.dot").read_bytes())
        self.assertEqual(before, (self.directory / "root.yaml").stat().st_mtime_ns)

    def test_argument_failure(self):
        result = subprocess.run([sys.executable, "-B", str(TEMPLATE_ROOT / "tools/pipeline_graph.py")], capture_output=True)
        self.assertEqual(result.returncode, 2)
