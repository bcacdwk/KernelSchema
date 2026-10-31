"""Shared scopes and declared coordinate windows through checking and expansion."""
from copy import deepcopy
import re
import subprocess
import sys

from .. import schema
from ..pipeline import build_phase, GraphError
from ..pipeline.model import ExpansionError
from ..pipeline.coordinates import recover
from ..pipeline.expressions import Domains, tree
from .support import TaskCase, PIPE, ROLE, PHASE, TEMPLATE_ROOT, put


class DomainTests(TaskCase):
    def task(self, domains, pipe, waits):
        task = deepcopy(self.base)
        if domains:
            put(task, PIPE + ("domain",), domains)
        put(task, ROLE + ("pipe",), pipe)
        put(task, ROLE + ("info", "execution_mode"), "async")
        put(task, ROLE + ("waits_for",), {"data": waits})
        return task

    def graph(self, task):
        self.write_task(task)
        _, plans = schema.validate_file(self.directory / "root.yaml")
        plan = plans["root"]
        return build_phase("copy", plan["design"]["tile_pipeline"]["phase"]["copy"],
                           plan["design"]["work_schedule"]["phase"]["copy"]["mapping"]["worker"], {})

    def run_graph(self, task):
        self.write_task(task)
        return subprocess.run([sys.executable, "-B", str(TEMPLATE_ROOT / "tools/pipeline_graph.py"),
                               str(self.directory / "root.yaml"), "--check-only"], capture_output=True, text=True)

    def test_local_shadow_and_unrelated_phase_domain_have_identical_edges(self):
        pipe = {"repeat": {"i": [0, 2]}, "steady": ["produce:Y(i)"]}
        waits = {"produce:Y(i)": ["produce:Y(i-1)"]}
        graphs = [self.graph(self.task(domains, pipe, waits)) for domains in ({}, {"i": [0, 3]})]
        self.assertEqual(set(graphs[0].edges), set(graphs[1].edges))
        self.assertEqual(len(graphs[1].operations), 2)
        self.assertEqual(len(graphs[1].boundaries), 1)
        result = self.run_graph(self.task({"i": [0, 3]}, pipe, waits))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unindexed_peer_does_not_contribute_an_unused_phase_domain(self):
        task = self.task({"i": [0, 9]}, {"repeat": {"i": [0, 2]}, "steady": ["produce:Y(i)"]},
                         {"produce:Y(i)": ["Peer/compute:Init"]})
        put(task, PIPE + ("role", "Peer"), {"info": {"member": "warp[1:2]", "execution_mode": "serial"},
                                            "pipe": {"steady": ["compute:Init"]}})
        self.assertEqual(len(self.graph(task).operations), 3)

    def test_disagreeing_referenced_scopes_are_incomplete_not_invalid(self):
        task = self.task({}, {"repeat": {"i": [0, 2]}, "steady": ["produce:Y(i)"]},
                         {"produce:Y(i)": ["Peer/publish:X(i)"]})
        put(task, PIPE + ("role", "Peer"), {"info": {"member": "warp[1:2]", "execution_mode": "serial"},
            "pipe": {"repeat": {"i": [0, 3]}, "steady": ["publish:X(i)"]}})
        result = self.run_graph(task)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("scopes disagree", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_indexed_milestone_cannot_silently_drop_a_required_event(self):
        task = deepcopy(self.base)
        put(task, ROLE + ("pipe",), {"repeat": {"i": [0, 2]}, "steady": ["produce:Y(i)"]})
        put(task, PIPE + ("role", "Peer"), {"info": {"member": "warp[1:2]", "execution_mode": "serial"},
            "pipe": {"repeat": {"i": [0, 3]}, "steady": ["publish:X(i)"]}})
        put(task, PIPE + ("milestone",), {"ready(i)": ["Main/produce:Y(i)", "Peer/publish:X(i)"]})
        with self.assertRaisesRegex(ExpansionError, "ambiguous milestone index i"):
            self.graph(task)
        result = self.run_graph(task)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("scopes disagree", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

        # An unindexed aggregate can collect independently scoped event families.
        put(task, PIPE + ("milestone",), {"ready": ["Main/produce:Y(i)", "Peer/publish:X(i)"]})
        self.assertEqual(len(self.graph(task).milestones["ready"]), 5)
        put(task, PIPE + ("milestone",), {"ready(i)": ["Main/produce:Y(i)", "Peer/publish:X(i)"]})
        for stop in (2, "2", "+2", "02"):
            with self.subTest(stop=stop):
                put(task, PIPE + ("role", "Peer", "pipe", "repeat", "i"), [0, stop])
                graph = self.graph(task)
                self.assertEqual(set(graph.milestones), {"ready(0)", "ready(1)"})
                self.assertTrue(all(len(events) == 2 for events in graph.milestones.values()))
                put(task, ROLE + ("waits_for",), {"data": {"produce:Y(i)": ["Peer/publish:X(i)"]}})
                self.assertFalse(self.graph(task).cycle)

    def test_milestone_checks_flatten_coordinate_definitions(self):
        task = deepcopy(self.base)
        put(task, PIPE + ("domain",), {"t": [0, 1], "i": [0, 2], "n": {"flatten": ["t", "i"]}})
        for name, value, stop in (("Main", "Y", 2), ("Peer", "X", 3)):
            put(task, PIPE + ("role", name), {
                "info": {"member": "warp[0:1]" if name == "Main" else "warp[1:2]", "execution_mode": "serial"},
                "pipe": {"repeat": {"i": [0, stop]}, "prologue": [
                    {"repeat": "n", "steady": [f"publish:{value}(n)"]}]}})
        put(task, PIPE + ("milestone",), {"ready(n)": ["Main/publish:Y(n)", "Peer/publish:X(n)"]})
        with self.assertRaisesRegex(ExpansionError, "ambiguous milestone index i"):
            self.graph(task)

    def test_affine_reparameterization_preserves_dependencies_and_boundaries(self):
        domains = {"t": [0, 3], "i": [0, "t+1"]}
        variants = [("t", "i", "t+1", "i", lambda a, b: (a, b)),
                    ("t+i", "i", "t+i+1", "i", lambda a, b: (a-b, b)),
                    ("2*t+i", "i", "2*t+i+2", "i", lambda a, b: ((a-b)//2, b)),
                    ("t+i", "t-i", "t+i+1", "t-i+1", lambda a, b: ((a+b)//2, (a-b)//2))]
        expected = None
        for a, b, successor_a, successor_b, inverse in variants:
            with self.subTest(arguments=(a, b)):
                action = f"compute:X({a},{b})"
                task = self.task(domains, {"repeat": "t", "steady": [
                    {"repeat": "i", "steady": [action]}]},
                    {f"compute:X({successor_a},{successor_b})": [action]})
                graph = self.graph(task)
                def normalize(event):
                    return re.sub(r"X\((-?\d+),(-?\d+)\)",
                                  lambda m: "X(%d,%d)" % inverse(int(m[1]), int(m[2])), event)
                edges = {(normalize(a), normalize(b), kind) for a, b, kind in graph.edges}
                expected = edges if expected is None else expected
                self.assertEqual(edges, expected)
                self.assertEqual(len(graph.operations), 6)
                self.assertEqual(len(graph.boundaries), 3)
                self.assertFalse(graph.cycle)
        result = self.run_graph(task)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_affine_window_never_shrinks_to_the_generated_subset(self):
        task = self.task({"t": [0, 3], "i": [0, "t+1"]}, {"repeat": "t", "steady": [
            {"repeat": "i[:1]", "steady": ["compute:X(t+i,i)"]}]},
            {"compute:X(t+i+1,i)": ["compute:X(t+i,i)"]})
        with self.assertRaisesRegex(GraphError, "dangling event") as caught:
            self.graph(task)
        self.assertNotIsInstance(caught.exception, ExpansionError)

    def test_omitted_outer_coordinates_use_the_complete_declared_projection(self):
        task = self.task({"s": [0, 4]}, {"repeat": "s", "steady": [
            {"repeat": {"p": ["s", "s+1"]}, "steady": ["load:X(p)"]}, "compute:Y(s)"]},
            {"load:X(s+1)": ["compute:Y(s)"]})
        graph = self.graph(task)
        self.assertEqual(len(graph.boundaries), 1)
        self.assertEqual(sum(kind == "data" for _, _, kind in graph.edges), 3)
        task["root"]["design"]["tile_pipeline"]["phase"]["copy"]["role"]["Main"]["pipe"]["steady"][0] = {
            "repeat": {"p": ["s", "s+1"]}, "steady": [{"repeat": "p[:0]", "steady": ["load:X(p)"]}]}
        with self.assertRaisesRegex(GraphError, "dangling event") as caught:
            self.graph(task)
        self.assertNotIsInstance(caught.exception, ExpansionError)

    def test_empty_projected_domain_is_a_boundary_without_generated_actions(self):
        task = self.task({"s": [0, 0], "j": [0, 2]}, {"steady": [
            {"repeat": "s", "steady": [{"repeat": {"p": ["s", "s+1"]}, "steady": ["load:X(p)"]}]},
            {"repeat": "j", "steady": ["compute:Y(j)"]}]}, {"load:X(j+1)": ["compute:Y(j)"]})
        graph = self.graph(task)
        self.assertEqual(len(graph.operations), 2)
        self.assertEqual(len(graph.boundaries), 2)

    def test_unknown_inverse_cannot_be_reported_as_a_hole_or_boundary(self):
        task = self.task({"t": [0, 3], "i": [0, "t+1"]}, {"repeat": "t", "steady": [
            {"repeat": "i", "steady": ["compute:X(t*t+i,i)"]}]},
            {"compute:X((t+1)*(t+1)+i,i)": ["compute:X(t*t+i,i)"]})
        result = self.run_graph(task)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("cannot uniquely recover", result.stderr)
        self.assertNotIn("dangling event", result.stderr)

    def test_unknown_candidate_is_not_excluded_by_another_declaration(self):
        task = self.task({"t": [0, 3], "i": [1, "t+2"]}, {"repeat": "t", "steady": [
            {"repeat": "i", "steady": ["compute:X(100+t+i,i)"]},
            {"repeat": "i[:2]", "steady": ["compute:X(t*i,i)"]}]},
            {"compute:X(t*i,i)": ["compute:X(100+t+i,i)"]})
        result = self.run_graph(task)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertNotIn("Acyclic sample", result.stdout)

    def test_source_outer_coordinate_is_not_evidence_for_the_target(self):
        domains = Domains({"t": [0, 3], "i": [0, "t+1"]}, {}, 2)
        with self.assertRaisesRegex(ExpansionError, "cannot uniquely recover"):
            recover([(tree("i"), 1)], domains, {"t": 0, "i": 0}, {"t"})

    def test_inconsistent_operand_pair_is_a_missing_action(self):
        task = self.task({"k": [0, 3]}, {"repeat": "k", "steady": ["mma:X(k)@Y(k)"]},
                         {"mma:X(k+1)@Y(k)": ["mma:X(k)@Y(k)"]})
        result = self.run_graph(task)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("dangling event", result.stderr)

    def test_current_role_qualification_keeps_the_current_member(self):
        task = self.task({}, {"steady": ["compute:X", "wait:Y"]}, {"wait:Y": ["Main/compute:X"]})
        put(task, PHASE + ("mapping", "worker", "unit_count"), 3)
        put(task, ROLE + ("info", "member"), "CTA[0:2].warp[0:1]")
        edges = {(a, b) for a, b, kind in self.graph(task).edges if kind == "data"}
        self.assertEqual(edges, {(f"copy/Main[{i}]/compute:X@done", f"copy/Main[{i}]/wait:Y@issue") for i in (0, 1)})
        put(task, ROLE + ("waits_for",), {"data": {"wait:Y": ["Main[*]/compute:X"]}})
        self.assertEqual(sum(kind == "data" for _, _, kind in self.graph(task).edges), 4)

    def test_ordered_completion_chain_excludes_async_overrides(self):
        task = deepcopy(self.base)
        put(task, ROLE + ("info", "execution_mode"), "async_ordered_completion")
        put(task, ROLE + ("info", "verb"), {"independent": {"execution_mode": "async", "meaning": "Independent operation."}})
        put(task, ROLE + ("pipe",), {"steady": ["compute:A", "independent:B", "compute:C"]})
        edges = {(a, b) for a, b, kind in self.graph(task).edges if kind == "completion"}
        self.assertEqual(edges, {("copy/Main[0]/compute:A@done", "copy/Main[0]/compute:C@done")})
