"""Artifact identity is independent of node names and HTML presentation."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
from unittest.mock import patch

from ..pipeline import output
from ..pipeline_graph import main
from .support import TaskCase, add_child


class OutputIdentityTests(TaskCase):
    def run_graph(self, *options, node="root"):
        stdout, stderr = StringIO(), StringIO()
        def render_svg(dot, svg):
            svg.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>')
        with patch.object(output, "graphviz", side_effect=render_svg), redirect_stdout(stdout), redirect_stderr(stderr):
            code = main([str(self.directory / (node + ".yaml")), *options])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_index_is_a_normal_node_and_never_the_gallery(self):
        task = deepcopy(self.base)
        add_child(task, "index")
        self.write_task(task)
        for options in (("--all",), (), ("--all",)):
            code, _, error = self.run_graph(*options, node="index")
            self.assertEqual(code, 0, error)
            directory = self.directory / "pipeline_graphs"
            view = (directory / "index.view.html").read_text()
            self.assertIn('id="diagram"', view)
            self.assertIn('data="index.svg"', view)
            if options:
                gallery = (directory / "index.html").read_text()
                self.assertIn('href="index.view.html"', gallery)
                self.assertNotIn('href="index.html"', gallery)
                self.assertNotIn('id="diagram"', gallery)
            else:
                self.assertFalse((directory / "index.html").exists())

    def test_presentation_changes_and_missing_images_do_not_change_ownership(self):
        code, _, error = self.run_graph("--all")
        self.assertEqual(code, 0, error)
        directory = self.directory / "pipeline_graphs"
        (directory / "root.svg").unlink()
        render = output.gallery
        with patch.object(output, "gallery", side_effect=lambda *args: render(*args).replace("1100px", "1120px") + "\n"):
            code, _, error = self.run_graph("--all")
        self.assertEqual(code, 0, error)
        self.assertIn("1120px", (directory / "index.html").read_text())
        code, _, error = self.run_graph("--all")
        self.assertEqual(code, 0, error)

    def test_gallery_identity_handles_escaped_paths_and_rejects_foreign_metadata(self):
        self.directory = self.directory / 'task & "quoted"'
        self.directory.mkdir()
        self.write_task(deepcopy(self.base))
        for _ in range(2):
            code, _, error = self.run_graph("--all", "--format", "dot")
            self.assertEqual(code, 0, error)
        directory = self.directory / "pipeline_graphs"
        index = directory / "index.html"
        index.write_text(output.gallery("Other task", [], str(self.directory.parent / "foreign")))
        before = {p.name: p.read_bytes() for p in directory.iterdir()}
        code, _, error = self.run_graph("--all")
        self.assertEqual(code, 2, error)
        self.assertIn("output conflicts", error)
        self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir()})
