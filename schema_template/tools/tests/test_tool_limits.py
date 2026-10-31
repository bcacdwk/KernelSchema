"""Capacity and actual invalidity remain distinct at both public entry points."""
from copy import deepcopy
import subprocess
import sys

from ..diagnostics import encode_value
from .support import TaskCase, MATH, PIPE, ROLE, TEMPLATE_ROOT, put


class ToolLimitTests(TaskCase):
    def command(self, graph):
        script = "tools/pipeline_graph.py" if graph else "check_yaml.py"
        command = [sys.executable, "-B", str(TEMPLATE_ROOT / script), str(self.directory / "root.yaml")]
        if graph:
            command += ["--check-only"]
        return subprocess.run(command, capture_output=True, text=True)

    def test_deep_flat_domain_chain_is_valid_but_exceeds_expansion_capacity(self):
        task = deepcopy(self.base)
        domains = {f"d{i}": {"flatten": [f"d{i+1}"]} for i in range(1100)}
        domains["d1100"] = [0, 1]
        put(task, PIPE + ("domain",), domains)
        put(task, ROLE + ("pipe",), {"repeat": "d0", "steady": ["produce:Y(d0)"]})
        self.write_task(task)
        result = self.command(False)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.command(True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("Graph not completed", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        domains["d1100"] = {"flatten": ["d0"]}
        self.write_task(task)
        for graph in (False, True):
            result = self.command(graph)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("cycle", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_parser_depth_is_a_tool_limit_not_invalid_yaml(self):
        task = deepcopy(self.base)
        put(task, MATH + ("rewrite", 0, "symbol"), {"Constants": {"meaning": "Nested literal array.", "value": "DEPTH_VALUE"}})
        source = encode_value(task["root"]).replace('"DEPTH_VALUE"', "[" * 1500 + "0" + "]" * 1500)
        path = self.directory / "root.yaml"
        path.write_text(source)
        before = (path.read_bytes(), path.stat().st_mtime_ns)
        for graph in (False, True):
            result = self.command(graph)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("capacity", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns))
