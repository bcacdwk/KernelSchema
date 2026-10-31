"""PNG-only publication, provenance, failure cleanup and input read-only behavior."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
from io import StringIO
import os
from unittest.mock import patch

from ..pipeline import output
from ..pipeline.render import RenderError
from ..pipeline_graph import main
from .support import TaskCase, PIPE, ROLE, add_child, put, tiny_png


class OutputTests(TaskCase):
    def run_graph(self, *options, node="root", renderer=None):
        stdout, stderr = StringIO(), StringIO()
        render = renderer or (lambda dot: tiny_png())
        with patch.object(output, "graphviz", side_effect=render), redirect_stdout(stdout), redirect_stderr(stderr):
            code = main([str(self.directory / (node + ".yaml")), *options])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_each_node_writes_only_its_sibling_png(self):
        task = deepcopy(self.base)
        add_child(task, "index")
        self.write_task(task)
        before = {p.name: (p.read_bytes(), p.stat().st_mode, p.stat().st_mtime_ns)
                  for p in self.directory.glob("*.yaml")}
        for options in (("--all",), (), ("--all",)):
            code, _, error = self.run_graph(*options, node="index")
            self.assertEqual(code, 0, error)
            self.assertEqual({p.name for p in self.directory.iterdir()},
                             {"root.yaml", "index.yaml", "root.png", "index.png"})
            for node in ("root", "index"):
                info = output.metadata(self.directory / (node + ".png"))
                self.assertEqual(info["source"], node + ".yaml")
                self.assertEqual(info["source_sha256"], hashlib.sha256(before[node + ".yaml"][0]).hexdigest())
                self.assertTrue(info["acyclic"])
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mode, p.stat().st_mtime_ns)
                                  for p in self.directory.glob("*.yaml")})

    def test_png_is_deterministic_and_ownership_survives_a_source_revision(self):
        code, _, error = self.run_graph()
        self.assertEqual(code, 0, error)
        image = self.directory / "root.png"
        first = image.read_bytes()
        self.assertEqual(self.run_graph()[0], 0)
        self.assertEqual(first, image.read_bytes())
        task = deepcopy(self.base)
        task["root"]["meta_data"]["version"] += 1
        self.write_task(task)
        self.assertEqual(self.run_graph()[0], 0)
        self.assertNotEqual(first, image.read_bytes())
        info = output.metadata(image)
        self.assertEqual(info["source"], "root.yaml")
        self.assertNotIn(str(self.directory), str(info))

    def test_failed_refresh_removes_the_stale_png(self):
        for failure, code in (("schema", 1), ("yaml", 1), ("option", 2), ("parameter", 2), ("event", 1)):
            with self.subTest(failure=failure):
                self.write_task(deepcopy(self.base))
                self.assertEqual(self.run_graph()[0], 0)
                options = []
                if failure == "yaml":
                    (self.directory / "root.yaml").write_text("meta_data: [")
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
                status, _, error = self.run_graph(*options)
                self.assertEqual(status, code, error)
                self.assertNotIn("Traceback", error)
                self.assertEqual({p.name for p in self.directory.iterdir()}, {"root.yaml"})
                self.assertEqual(before, (source.read_bytes(), source.stat().st_mtime_ns, source.stat().st_mode))

    def test_all_refresh_retires_removed_nodes_and_preserves_unrelated_files(self):
        task = deepcopy(self.base)
        add_child(task)
        self.write_task(task)
        self.assertEqual(self.run_graph("--all")[0], 0)
        unrelated = self.directory / "photo.png"
        unrelated.write_bytes(tiny_png())
        (self.directory / "notes.md").write_text("Independent notes.")
        self.write_task(deepcopy(self.base))
        self.assertEqual(self.run_graph("--all")[0], 0)
        self.assertEqual({p.name for p in self.directory.iterdir()}, {"root.yaml", "root.png", "photo.png", "notes.md"})
        self.assertEqual(unrelated.read_bytes(), tiny_png())

    def test_failed_whole_task_check_clears_all_selected_pngs(self):
        task = deepcopy(self.base)
        add_child(task)
        self.write_task(task)
        self.assertEqual(self.run_graph("--all")[0], 0)
        put(task, ("small", "design", "tile_pipeline", "phase", "copy", "role", "Main", "info", "execution_mode"), "sync")
        self.write_task(task)
        code, _, error = self.run_graph("--all")
        self.assertEqual(code, 1, error)
        self.assertEqual({p.name for p in self.directory.iterdir()}, {"root.yaml", "small.yaml"})

    def test_check_only_never_changes_existing_images(self):
        self.assertEqual(self.run_graph()[0], 0)
        task = deepcopy(self.base)
        put(task, ROLE + ("info", "execution_mode"), "sync")
        self.write_task(task)
        before = self.snapshot()
        code, _, error = self.run_graph("--check-only")
        self.assertEqual(code, 1, error)
        self.assertEqual(before, self.snapshot())

    def test_render_failure_or_invalid_png_never_leaves_successful_output(self):
        def unavailable(dot):
            raise RenderError("Graphviz unavailable.")
        for render in (unavailable, lambda dot: b"not a PNG"):
            self.assertEqual(self.run_graph()[0], 0)
            code, _, error = self.run_graph(renderer=render)
            self.assertEqual(code, 2, error)
            self.assertFalse((self.directory / "root.png").exists())
            self.assertEqual({p.name for p in self.directory.iterdir()}, {"root.yaml"})

    def test_missing_graphviz_is_optional_for_check_only(self):
        with patch.dict(os.environ, {"PATH": ""}):
            stdout, stderr = StringIO(), StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                self.assertEqual(main([str(self.directory / "root.yaml")]), 2)
                self.assertEqual(main([str(self.directory / "root.yaml"), "--check-only"]), 0)
            self.assertIn("Graphviz", stderr.getvalue())
            self.assertEqual({p.name for p in self.directory.iterdir()}, {"root.yaml"})

    def test_output_conflicts_and_wrong_input_types_preserve_existing_files(self):
        image = self.directory / "root.png"
        image.write_bytes(tiny_png())
        before = self.snapshot()
        for options in ((), ("--all",)):
            code, _, error = self.run_graph(*options)
            self.assertEqual(code, 2, error)
            self.assertIn("output conflicts", error)
            self.assertEqual(before, self.snapshot())
        with redirect_stderr(StringIO()):
            self.assertEqual(main([str(image)]), 2)
        self.assertEqual(before, self.snapshot())

    def test_cycles_stay_visible_in_the_png_and_diagnostics(self):
        task = deepcopy(self.base)
        put(task, ROLE + ("waits_for",), {"data": {"produce:Y": ["Peer/produce:Y"]}})
        put(task, PIPE + ("role", "Peer"), {"info": {"execution_mode": "serial", "member": "warp[1:2]"},
            "pipe": {"steady": ["produce:Y"]}, "waits_for": {"data": {"produce:Y": ["Main/produce:Y"]}}})
        self.write_task(task)
        rendered = []
        def render(dot):
            rendered.append(dot)
            return tiny_png()
        code, stdout, stderr = self.run_graph(renderer=render)
        self.assertEqual(code, 1)
        self.assertIn("DEPENDENCY CYCLE", stdout)
        self.assertIn("DEPENDENCY CYCLE", rendered[0])
        self.assertIn("Main.waits_for.data", stderr)
        self.assertFalse(output.metadata(self.directory / "root.png")["acyclic"])
