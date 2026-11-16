"""One CLI: checked formatting, read-only requests and safe PNG publication."""
from copy import deepcopy
import hashlib
import json
from unittest.mock import patch

from .. import normalization, schema, yaml_source
from ..diagnostics import Diagnostic
from ..pipeline import output
from .support import ALGO, PIPE, ROLE, TaskCase, add_child, at, tiny_png


class CliTests(TaskCase):
    def test_default_normalizes_the_task_and_preserves_permissions_versions_and_mtime(self):
        data = deepcopy(self.base)
        add_child(data, source='root')
        self.write_task(data)
        child = self.directory / 'branch.yaml'
        child.chmod(0o640)
        before = self.snapshot()
        code, out, error = self.cli(node='branch.yaml')
        self.assertEqual(code, 0, error)
        self.assertIn('Normalization completed: 2 YAML; 2 changed', out)
        self.assertNotEqual(before['root.yaml'][0], self.snapshot()['root.yaml'][0])
        self.assertEqual(child.stat().st_mode & 0o777, 0o640)
        self.assertEqual(yaml_source.load(child, 'branch').value, data['branch'])
        before = self.snapshot()
        self.assertEqual(self.cli(node='branch.yaml')[0], 0)
        self.assertEqual(before, self.snapshot())

    def test_read_only_previews_the_whole_task_and_all_only_changes_graph_scope(self):
        data = deepcopy(self.base)
        add_child(data, source='root')
        self.write_task(data)
        before = self.snapshot()
        code, out, error = self.cli('--read-only', '--all')
        self.assertEqual(code, 1)
        self.assertIn('Normalization preview: 2 YAML; 2 would change', out)
        self.assertIn('warning: planned', error)
        self.assertNotIn('warning: applied', error)
        self.assertIn('Schema checked:', out)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(self.cli('--all')[0], 0)
        before = self.snapshot()
        self.assertEqual(self.cli('--read-only', '--all')[0], 0)
        self.assertEqual(before, self.snapshot())

    def test_storage_layout_is_the_same_candidate_from_child_preview_and_default_cli(self):
        data = deepcopy(self.base)
        at(data, PIPE)['Storage'] = {'buffer': {'meaning': 'Two scalar slots', 'depth': 2,
                                              'location': 'register', 'shape': [], 'dtype': 'fp32',
                                              'total_size': 8, 'view': {'x(i)': {'slot_index': 'i%2'}}}}
        add_child(data, source='root')
        self.write_task(data, canonical=True)
        path = self.directory / 'root.yaml'
        document = yaml_source.load(path, 'root')
        entry = next(use.node for use in document.occurrences
                     if not use.key and use.parts == PIPE[1:] + ('Storage', 'buffer'))
        expected = document.source
        begin, end = entry.start_mark.index, entry.end_mark.index
        self.assertEqual(entry.end_mark.line - entry.start_mark.line + 1, 2)
        source = expected[:begin] + expected[begin:end].replace('\n            ', ' ') + expected[end:]
        path.write_text(source, encoding='utf-8')
        before = self.snapshot()
        plan = schema.normalize_directory(self.directory)
        self.assertEqual(plan.candidates, {'root': expected})
        code, out, error = self.cli('--read-only', node='branch.yaml')
        self.assertEqual(code, 1, error)
        self.assertIn('Normalization preview: 2 YAML; 1 would change', out)
        self.assertIn('normalization preview; not written', error)
        self.assertIn('layout', error)
        self.assertEqual(before, self.snapshot())
        code, _, error = self.cli(node='branch.yaml')
        self.assertEqual(code, 0, error)
        self.assertEqual(path.read_text(encoding='utf-8'), expected)
        self.assertEqual(yaml_source.load(path, 'root').value, data['root'])
        self.assertEqual(before['branch.yaml'], self.snapshot()['branch.yaml'])
        self.assertEqual(yaml_source.load(path, 'root').value['META_DATA']['Version'], 1)
        after = self.snapshot()
        self.assertEqual(self.cli('--read-only', node='branch.yaml')[0], 0)
        self.assertEqual(self.cli(node='branch.yaml')[0], 0)
        self.assertEqual(after, self.snapshot())

    def test_safe_normalization_precedes_unknown_fields_and_budget_diagnostics(self):
        data = deepcopy(self.base)
        data['root']['DESIGN']['unexpected'] = {}
        self.write_task(data)
        before = self.snapshot()
        code, _, error = self.cli()
        self.assertEqual(code, 1)
        self.assertIn('unexpected', error)
        self.assertIn('field not allowed', error)
        self.assertNotEqual(before['root.yaml'][0], self.snapshot()['root.yaml'][0])
        self.assertEqual(yaml_source.load(self.directory / 'root.yaml', 'root').value, data['root'])
        data = deepcopy(self.base)
        data['root']['META_DATA']['Describe'] = 'x' * 151
        self.write_task(data)
        before = (self.directory / 'root.yaml').read_bytes()
        code, out, error = self.cli()
        self.assertEqual(code, 1)
        self.assertIn('151 Unicode characters', error)
        self.assertIn('Normalized:', out)
        self.assertNotEqual(before, (self.directory / 'root.yaml').read_bytes())
        self.assertEqual(yaml_source.load(self.directory / 'root.yaml', 'root').value, data['root'])

    def test_sentence_period_warns_once_and_default_cleanup_precedes_the_budget_check(self):
        data = deepcopy(self.base)
        data['root']['META_DATA']['Describe'] = '界' * 150 + '。'
        self.write_task(data)
        before = self.snapshot()
        code, out, error = self.cli('--read-only')
        self.assertEqual(code, 1)
        self.assertIn('Normalization preview:', out)
        self.assertIn('warning: planned sentence punctuation', error)
        self.assertNotIn('151 Unicode characters', error)
        self.assertNotIn('warning: applied', error)
        self.assertEqual(before, self.snapshot())
        code, _, error = self.cli()
        self.assertEqual(code, 0, error)
        self.assertEqual(error.count('warning: applied sentence punctuation'), 1)
        self.assertNotIn('warning: ordinary explanatory text', error)
        after = yaml_source.load(self.directory / 'root.yaml', 'root').value
        self.assertEqual(after['META_DATA']['Describe'], '界' * 150)
        self.assertEqual(after['META_DATA']['Version'], data['root']['META_DATA']['Version'])
        before = self.snapshot()
        code, _, error = self.cli()
        self.assertEqual(code, 0, error)
        self.assertNotIn('warning:', error)
        self.assertEqual(before, self.snapshot())

    def test_whole_task_and_dispatch_errors_are_checked_from_a_child(self):
        data = deepcopy(self.base)
        add_child(data, source='root')
        self.write_task(data, canonical=True)
        at(data, ALGO)['Golden_reference']['Tolerance'] = {'missing': 'exact'}
        self.write_task(data, canonical=True)
        (self.directory / 'dispatch.md').write_text('## 范围\nOnly a scope\n', encoding='utf-8')
        code, _, error = self.cli('--read-only', '--graph', 'check', node='branch.yaml')
        self.assertEqual(code, 1)
        self.assertIn('root.yaml', error)
        self.assertIn('required section', error)
        self.assertIn('Graph not run', error)

    def test_argument_conflicts_and_bad_samples_do_not_write(self):
        before = self.snapshot()
        for args in [('--read-only', '--graph', 'png'), ('--standalone', '--all'),
                     ('--show-effective', '--graph', 'check'), ('--set', 'n=2'),
                     ('--graph', 'check', '--set', 'n=x'),
                     ('--graph', 'check', '--set', 'n=2', '--set', 'n=3'),
                     ('--graph', 'check', '--worktiles', '-1')]:
            with self.subTest(args=args):
                self.assertEqual(self.cli(*args)[0], 2)
                self.assertEqual(before, self.snapshot())

    def test_standalone_and_draft_view_are_read_only_special_requests(self):
        (self.directory / 'broken.yaml').write_text('x: [', encoding='utf-8')
        (self.directory / 'dispatch.md').unlink()
        before = self.snapshot()
        self.assertEqual(self.cli('--standalone')[0], 0)
        self.assertEqual(before, self.snapshot())
        (self.directory / 'broken.yaml').unlink()
        data = deepcopy(self.base)
        data['root']['DESIGN']['Math_reshape'] = {'Thesis': None}
        self.write_task(data)
        (self.directory / 'dispatch.md').unlink()
        before = self.snapshot()
        code, out, error = self.cli('--show-effective')
        self.assertEqual(code, 0, error)
        self.assertEqual(json.loads(out)['validation'], 'not_run')
        self.assertNotIn('Schema checked:', out)
        self.assertEqual(before, self.snapshot())

    def test_graph_check_does_not_require_graphviz_or_write_images(self):
        self.canonicalize()
        before = self.snapshot()
        with patch('tools.pipeline.render.shutil.which', return_value=None):
            code, out, error = self.cli('--read-only', '--graph', 'check')
            self.assertEqual(code, 0, error)
            self.assertIn('Graph checked: root/MAIN', out)
            self.assertIn('worktiles=1', out)
            self.assertEqual(before, self.snapshot())
            self.assertEqual(self.cli('--graph', 'png')[0], 2)
        self.assertEqual(before, self.snapshot())

    def test_all_only_expands_graph_selection_after_complete_task_normalization(self):
        data = deepcopy(self.base)
        add_child(data, source='root')
        self.write_task(data)
        code, out, error = self.cli('--graph', 'check', node='branch.yaml')
        self.assertEqual(code, 0, error)
        self.assertIn('Normalization completed: 2 YAML; 2 changed', out)
        self.assertIn('Graph checked: branch/MAIN', out)
        self.assertNotIn('Graph checked: root/', out)
        before = self.snapshot()
        code, out, error = self.cli('--all', '--graph', 'check', node='branch.yaml')
        self.assertEqual(code, 0, error)
        self.assertIn('Graph checked: branch/MAIN', out)
        self.assertIn('Graph checked: root/MAIN', out)
        self.assertEqual(before, self.snapshot())

    def test_local_repository_gate_is_outside_the_snapshot_api_and_blocks_graphs(self):
        data = deepcopy(self.base)
        data['root']['META_DATA']['Source'] = 'project@abc123:reference.py'
        self.write_task(data, canonical=True)
        files, documents = schema.read_directory(self.directory)
        with patch.object(schema.values, 'local_repository_errors', side_effect=AssertionError('unexpected file I/O')):
            schema.validate_documents(files, documents)
        before = self.snapshot()
        errors = [Diagnostic('root.META_DATA.Source', 'repository path is missing at the referenced local revision')]
        with patch.object(schema.values, 'local_repository_errors', return_value=errors) as checker:
            code, _, error = self.cli('--read-only', '--graph', 'check')
        self.assertEqual(code, 1, error)
        self.assertIn('repository path is missing', error)
        self.assertIn('Graph not run', error)
        self.assertEqual(before, self.snapshot())
        checker.assert_called_once()
        final_documents = checker.call_args.args[1]
        self.assertEqual(final_documents['root'].source_sha256, documents['root'].source_sha256)

    def test_png_uses_final_formatted_source_identity(self):
        self.write_task(deepcopy(self.base))
        with patch.object(output, 'graphviz', return_value=tiny_png()):
            code, out, error = self.cli('--graph', 'png')
        self.assertEqual(code, 0, error)
        self.assertIn('PNG updated:', out)
        source = self.directory / 'root.yaml'
        information = output.metadata(source.with_suffix('.png'))
        self.assertEqual(information['source_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(information['source_manifest'], {'root.yaml': information['source_sha256']})
        self.assertEqual(information['samples'], {'MAIN': {'parameters': {}, 'worktiles': 1}})
        self.assertEqual(information['effective_identity'], schema.resolve_directory(self.directory).effective_identity('root'))

    def test_missing_one_sample_prevents_all_selected_png_publication(self):
        data = deepcopy(self.base)
        child = add_child(data)
        child['DESIGN']['Tile_pipeline']['Step']['MAIN']['Index'] = {'i': {'meaning': 'Sample domain', 'definition': [0, 'n']}}
        child['DESIGN']['Tile_pipeline']['Step']['MAIN']['Role']['MAIN']['Pipe'] = [{'for(i)': ['compute:y(i)']}]
        self.write_task(data, canonical=True)
        for name in data:
            path = self.directory / (name + '.png')
            path.write_bytes(output.stamp(tiny_png(), {'kind': 'pipeline_graph', 'source': name + '.yaml'}))
        before = self.snapshot()
        with patch.object(output, 'graphviz', return_value=tiny_png()) as renderer:
            code, _, error = self.cli('--all', '--graph', 'png')
        self.assertEqual(code, 2, error)
        self.assertIn('did not all complete', error)
        renderer.assert_not_called()
        self.assertEqual(before, self.snapshot())

    def test_complete_cycle_can_publish_a_red_diagnostic_graph_with_status_one(self):
        data = deepcopy(self.base)
        at(data, PIPE)['Index'] = {'i': {'meaning': 'Two actions', 'definition': [0, 2]}}
        main = at(data, ROLE)
        main['Pipe'] = [{'for(i)': ['compute:y(i)']}]
        main['Waiting'] = {'Allowed': ['{done compute:y(i)} ← [{issue compute:y(i+1)}]']}
        self.write_task(data, canonical=True)
        with patch.object(output, 'graphviz', return_value=tiny_png()) as renderer:
            code, out, error = self.cli('--graph', 'png')
        self.assertEqual(code, 1, error)
        self.assertIn('cycle found', out)
        self.assertIn('root.yaml:', error)
        self.assertIn('DESIGN.Tile_pipeline.Step.MAIN.Role.MAIN.Pipe', error)
        self.assertIn('PNG updated:', out)
        self.assertFalse(output.metadata(self.directory / 'root.png')['acyclic'])
        self.assertIn('#c83232', renderer.call_args.args[0])

    def test_source_drift_and_file_symlinks_are_not_silently_replaced(self):
        real_normalize = normalization.normalize_documents
        path = self.directory / 'root.yaml'
        altered = self.source + '\n'

        def drift(*arguments, **kwargs):
            candidate = real_normalize(*arguments, **kwargs)
            path.write_text(altered, encoding='utf-8')
            return candidate

        with patch.object(schema, 'normalize_documents', side_effect=drift):
            code, _, error = self.cli()
        self.assertEqual(code, 2, error)
        self.assertIn('source changed', error)
        self.assertEqual(path.read_text(encoding='utf-8'), altered)
        target = self.directory / 'target.txt'
        target.write_text(encode_source := json.dumps(self.base['root']), encoding='utf-8')
        path.unlink()
        path.symlink_to(target)
        code, _, error = self.cli()
        self.assertEqual(code, 2, error)
        self.assertIn('symlink was preserved', error)
        self.assertTrue(path.is_symlink())
        self.assertEqual(target.read_text(encoding='utf-8'), encode_source)

    def test_foreign_png_preflight_preserves_files_and_does_not_format(self):
        path = self.directory / 'root.png'
        path.write_bytes(tiny_png())
        before = self.snapshot()
        code, _, error = self.cli('--graph', 'png')
        self.assertEqual(code, 2)
        self.assertIn('not owned', error)
        self.assertEqual(before, self.snapshot())
