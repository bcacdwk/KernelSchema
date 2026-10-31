"""Capacity and actual invalidity remain distinct at both public entry points."""
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import subprocess
import sys
from unittest.mock import patch

import check_yaml
from .. import yaml_source
from ..pipeline_graph import main as graph_main
from ..diagnostics import encode_value
from .support import TaskCase, MATH, PIPE, ROLE, TEMPLATE_ROOT, put


class ToolLimitTests(TaskCase):
    def test_alias_expansion_has_a_total_budget_at_both_entries(self):
        task = deepcopy(self.base)
        put(task, MATH + ("rewrite", 0, "symbol"), {
            "Constants": {"meaning": "Repeated literal arrays.", "value": "ALIAS_ARRAY"}})
        aliases = ["&a0 [0]"] + [f"&a{i} [*a{i-1}, *a{i-1}]" for i in range(1, 10)]
        source = encode_value(task["root"]).replace('"ALIAS_ARRAY"', "[" + ", ".join(aliases) + "]")
        path = self.directory / "root.yaml"
        path.write_text(source)
        before = self.snapshot()
        for entry, arguments in ((check_yaml.main, [str(path)]), (graph_main, [str(path), "--check-only"])):
            output, errors = StringIO(), StringIO()
            with patch.object(yaml_source, "MAX_EXPANDED_NODES", 2000), redirect_stdout(output), redirect_stderr(errors):
                result = entry(arguments)
            self.assertEqual(result, 2, errors.getvalue())
            self.assertIn("capacity", errors.getvalue())
            self.assertNotIn("Traceback", errors.getvalue())
        self.assertEqual(before, self.snapshot())
        # The same finite aliases remain valid when the tool can expand them.
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(check_yaml.main([str(path)]), 0)

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
