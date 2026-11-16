"""Whole-region sources, consumer bindings and independent delivery views."""
from copy import deepcopy
import hashlib

from .. import schema
from ..diagnostics import Diagnostic, encode_value, format_diagnostics
from .support import MATH, TaskCase, add_child, at


class ResolutionTests(TaskCase):
    def task(self):
        data = deepcopy(self.base)
        add_child(data, 'unused')
        provider = add_child(data, 'provider')
        provider['DESIGN']['Work_schedule'] = {'from': 'unused'}
        middle = add_child(data, 'middle', source='provider')
        child = add_child(data, 'child', source='middle')
        child['DESIGN']['Work_schedule'] = deepcopy(data['root']['DESIGN']['Work_schedule'])
        child['DESIGN']['Tile_pipeline'] = {'from': 'provider'}
        return data

    def test_three_regions_keep_actual_chains_and_local_context(self):
        data = self.task()
        self.write_task(data)
        (self.directory / 'provider.yaml').rename(self.directory / 'provider.yml')
        self.write_index()
        _, resolved = schema.validate_resolved_file(self.directory / 'child.yaml')
        self.assertEqual(data, resolved.raw_task)
        self.assertEqual(resolved.origins['child']['Math_reshape'], schema.RegionOrigin('provider', ('child', 'middle', 'provider')))
        self.assertEqual(resolved.origins['child']['Work_schedule'].chain, ('child',))
        self.assertEqual(resolved.origins['child']['Tile_pipeline'].chain, ('child', 'provider'))
        self.assertEqual(set(resolved.source_manifest('child')), {'root.yaml', 'provider.yml', 'middle.yaml', 'child.yaml'})
        self.assertNotIn('CONTRACT', resolved.effective_task['child'])
        self.assertEqual(resolved.effective_task['child']['META_DATA'], data['child']['META_DATA'])
        self.assertNotIn('unused', resolved.dependency_nodes['child'])

    def test_binding_uses_consumer_definition_and_diagnostics_use_provider_marks(self):
        data = self.task()
        provider = data['provider']
        provider['DESIGN']['Math_reshape']['Rewrite'][0]['Symbol'] = {
            'current': {'meaning': 'Current worktile input', 'shape': [], 'dtype': 'fp32',
                        'binding': 'bind:DESIGN.Work_schedule::MAIN.Decomposition.Worktile.Invariant.x'}}
        self.write_task(data)
        schema.validate_directory(self.directory)
        # This binding comes from provider, but must see the child's worktile.
        data['child']['DESIGN']['Work_schedule']['Step']['MAIN']['Decomposition']['Worktile']['Invariant']['x']['shape'] = [{'element': 2}]
        self.write_task(data)
        resolved = schema.resolve_directory(self.directory)
        with self.assertRaises(schema.SchemaError) as caught:
            schema.validate_directory(self.directory)
        message = str(caught.exception)
        self.assertIn('Consumer: child.DESIGN.Math_reshape', message)
        self.assertIn('Source chain: child -> middle -> provider', message)
        self.assertIn('provider.yaml:', message)
        self.assertIn('child.yaml:', message)
        self.assertIn('Referenced at:', message)
        diagnostic = format_diagnostics([Diagnostic('child.DESIGN.Math_reshape.absent', 'required', missing=True)],
                                        resolved.source_files, resolved.documents, resolved=resolved)
        self.assertNotIn('Actual:', diagnostic)
        self.assertIn('provider.yaml:', diagnostic)

    def test_raw_effective_and_export_copies_are_independent(self):
        self.write_task(self.task())
        resolved = schema.resolve_directory(self.directory)
        raw = deepcopy(resolved.raw_task)
        provider = deepcopy(resolved.effective_task['provider'])
        resolved.effective_task['child']['DESIGN']['Math_reshape']['Rewrite'].clear()
        view = resolved.effective_document('middle')
        view['CONTRACT'].clear()
        view['DESIGN']['Tile_pipeline'].clear()
        self.assertEqual(resolved.raw_task, raw)
        self.assertEqual(resolved.effective_task['provider'], provider)
        self.assertEqual(resolved.documents['provider'].value, raw['provider'])

    def test_identity_tracks_actual_sources_and_excludes_other_provider_regions(self):
        self.write_task(self.task())
        first = schema.resolve_directory(self.directory)
        source = self.directory / 'unused.yaml'
        source.write_bytes(source.read_bytes() + b'\n')
        second = schema.resolve_directory(self.directory)
        self.assertEqual(first.effective_identity('child'), second.effective_identity('child'))
        self.assertNotEqual(first.effective_identity('provider'), second.effective_identity('provider'))
        source = self.directory / 'middle.yaml'
        source.write_bytes(source.read_bytes() + b'\n')
        third = schema.resolve_directory(self.directory)
        self.assertNotEqual(second.effective_identity('child'), third.effective_identity('child'))
        self.assertEqual(third.source_hashes['middle'], hashlib.sha256(source.read_bytes()).hexdigest())

    def test_cross_region_cycles_and_root_references_are_rejected(self):
        data = deepcopy(self.base)
        a = add_child(data, 'a')
        b = add_child(data, 'b')
        a['DESIGN']['Math_reshape'] = {'from': 'b'}
        b['DESIGN']['Tile_pipeline'] = {'from': 'a'}
        self.assert_invalid(data, 'cross-region source cycle')
        data = deepcopy(self.base)
        add_child(data, 'a')
        data['root']['DESIGN']['Math_reshape'] = {'from': 'a'}
        self.assert_invalid(data, 'must be local')
        data = deepcopy(self.base)
        add_child(data, 'a', source='root')['DESIGN']['Math_reshape']['Thesis'] = 'No partial overrides'
        self.assert_invalid(data, 'field not allowed')

    def test_task_count_includes_root_and_reference_nodes(self):
        data = deepcopy(self.base)
        for number in range(9):
            add_child(data, 'node_' + str(number), source='root')
        self.write_task(data)
        self.assertEqual(len(schema.validate_directory(self.directory)), 10)
        add_child(data, 'node_9', source='root')
        self.assert_invalid(data, 'at most 10')

    def test_raw_text_budget_is_not_repeated_at_from_consumers(self):
        data = deepcopy(self.base)
        add_child(data, 'branch', source='root')
        at(data, MATH)['Thesis'] = 'x' * 301
        message = self.assert_invalid(data, 'Thesis', '300')
        self.assertEqual(message.count('DESIGN.Math_reshape.Thesis:'), 1)
        self.assertNotIn('Consumer:', message)

    def test_standalone_keeps_identity_and_ignores_siblings_and_dispatch(self):
        self.write_task(self.task())
        _, resolved = schema.validate_resolved_file(self.directory / 'child.yaml')
        content = schema.effective_yaml(resolved, 'child')
        isolated = self.directory / 'delivery'
        isolated.mkdir()
        path = isolated / 'view.yaml'
        path.write_text(content, encoding='utf-8')
        (isolated / 'broken.yaml').write_text('broken: [', encoding='utf-8')
        _, task = schema.validate_standalone_file(path)
        self.assertEqual(set(task), {'child'})
        self.assertEqual(task['child']['META_DATA']['Dispatch']['id'], 'child')
        original = task['child']['DESIGN']['Math_reshape']['Thesis']
        task['child']['DESIGN']['Math_reshape']['Thesis'] = 'x' * 301
        path.write_text(encode_value(task['child']), encoding='utf-8')
        with self.assertRaisesRegex(schema.SchemaError, '300 characters'):
            schema.validate_standalone_file(path)
        task['child']['DESIGN']['Math_reshape']['Thesis'] = original
        task['child']['DESIGN']['Math_reshape'] = {'from': 'provider'}
        path.write_text(encode_value(task['child']), encoding='utf-8')
        with self.assertRaisesRegex(schema.SchemaError, 'must be local'):
            schema.validate_standalone_file(path)
