"""PNG ownership and source checks retain previous output on every failed refresh."""
from copy import deepcopy
from unittest.mock import patch

from .. import schema
from ..pipeline import analyze, configuration, output
from ..pipeline.render import RenderError
from .support import TaskCase, add_child, tiny_png


class OutputTests(TaskCase):
    def prepared(self):
        data = deepcopy(self.base)
        add_child(data, source='root')
        self.write_task(data, canonical=True)
        _, resolved = schema.validate_resolved_file(self.directory / 'branch.yaml')
        graph = analyze(resolved.effective_task['branch'], configuration({}))[0]
        destination = self.directory / 'branch.png'
        destination.write_bytes(output.stamp(tiny_png(), {'kind': 'pipeline_graph', 'source': 'branch.yaml'}))
        return resolved, graph, destination

    def publish(self, resolved, graph):
        return output.GraphOutput().write(self.directory / 'branch.yaml', [graph], {'MAIN': configuration({})}, [],
                                          source_sha256=resolved.source_hashes['branch'],
                                          source_manifest=resolved.source_manifest('branch'),
                                          effective_identity=resolved.effective_identity('branch'),
                                          origins=resolved.origins_dict('branch'))

    def test_provider_drift_before_or_during_render_keeps_previous_image(self):
        for during in (False, True):
            with self.subTest(during=during):
                resolved, graph, destination = self.prepared()
                previous = destination.read_bytes()
                provider = self.directory / 'root.yaml'

                def drift(_):
                    provider.write_bytes(provider.read_bytes() + b'\n')
                    return tiny_png()

                if not during:
                    drift('')
                with patch.object(output, 'graphviz', side_effect=drift if during else None,
                                  return_value=tiny_png()) as renderer:
                    with self.assertRaisesRegex(RenderError, 'source changed'):
                        self.publish(resolved, graph)
                    if not during:
                        renderer.assert_not_called()
                self.assertEqual(destination.read_bytes(), previous)
                self.assertFalse(any(p.name.startswith('.pipeline-') for p in self.directory.iterdir()))

    def test_renderer_failure_and_invalid_png_leave_previous_output(self):
        resolved, graph, destination = self.prepared()
        previous = destination.read_bytes()
        for failure in (RenderError('no dot'), b'not a PNG'):
            with self.subTest(failure=failure):
                options = {'side_effect': failure} if isinstance(failure, Exception) else {'return_value': failure}
                with patch.object(output, 'graphviz', **options), self.assertRaisesRegex(RenderError, 'PNG not updated'):
                    self.publish(resolved, graph)
                self.assertEqual(destination.read_bytes(), previous)

    def test_preflight_and_plain_all_never_clean_up_other_images(self):
        resolved, graph, destination = self.prepared()
        orphan = self.directory / 'removed.png'
        orphan.write_bytes(output.stamp(tiny_png(), {'kind': 'pipeline_graph', 'source': 'removed.yaml'}))
        before = self.snapshot()
        output.GraphOutput().preflight(resolved.source_files.values())
        self.assertEqual(before, self.snapshot())
        self.assertEqual(self.cli('--all')[0], 0)
        self.assertEqual(before, self.snapshot())
        destination.unlink()
        destination.symlink_to(orphan)
        with self.assertRaisesRegex(RenderError, 'not owned'):
            output.GraphOutput().preflight([self.directory / 'branch.yaml'])
        self.assertTrue(destination.is_symlink())
