"""Current structure, interfaces and static relationships through read-only APIs."""
from copy import deepcopy
from decimal import Decimal
import subprocess
from unittest.mock import patch

from .. import rules, schema, text_policy, yaml_source
from ..diagnostics import encode_value
from ..rules import structure, values
from ..rules.schedule import consumer_waits
from .support import ALGO, HARD, MATH, PIPE, ROLE, STEP, TEMPLATE_ROOT, WORK, TaskCase, add_step, at, put


class SchemaTests(TaskCase):
    def test_fixture_and_document_api_are_read_only(self):
        before = self.snapshot()
        self.assertEqual(self.base, schema.validate_directory(self.directory))
        files, documents = schema.read_directory(self.directory)
        with patch('pathlib.Path.read_bytes', side_effect=AssertionError('snapshot API must not reread files')):
            resolved = schema.validate_documents(files, documents)
        self.assertEqual(resolved.raw_task, self.base)
        self.assertEqual(before, self.snapshot())

    def test_structure_gate_prevents_invalid_downstream_traversal(self):
        data = deepcopy(self.base)
        put(data, ROLE + ('Info',), None)
        with patch.object(rules, 'validate_relations', side_effect=AssertionError('relations ran before the gate')):
            self.assert_invalid(data, 'Info', 'expected')
        data = deepcopy(self.base)
        at(data, ALGO)['Inputz'] = at(data, ALGO).pop('Input')
        self.assert_invalid(data, 'Inputz', 'field not allowed')

    def test_nullable_and_exact_contract_values(self):
        algorithm = deepcopy(at(self.base, ALGO))
        algorithm['Input']['optional'] = {'meaning': 'Optional singleton', 'shape': [], 'dtype': 'python_none',
                                          'nullable': True, 'default': 'None'}
        algorithm['Input']['x']['range'] = [Decimal('1.0000000000000000001'), Decimal('1.0000000000000000002')]
        self.assertFalse([d for d in rules.validate_algorithm(algorithm) if d.severity == 'error'])
        algorithm['Input']['optional']['shape'] = [{'element': 1}]
        self.assertTrue(any('singleton scalar' in d.message for d in rules.validate_algorithm(algorithm)))
        algorithm['Input']['optional']['shape'] = []
        algorithm['Input']['optional']['nullable'] = 'true'
        self.assertTrue(any(d.path.endswith('.nullable') for d in rules.validate_algorithm(algorithm)))

    def test_nullable_contract_fixtures_use_explicit_python_none_values(self):
        examples = yaml_source.load(TEMPLATE_ROOT / 'tools/tests/fixtures/nullable_io.yaml', 'algorithms').value
        for name, algorithm in examples.items():
            with self.subTest(name=name):
                self.assertFalse([d for d in rules.validate_algorithm(algorithm) if d.severity == 'error'])

    def test_contract_only_checks_criteria_properties_and_text_budget(self):
        algorithm = deepcopy(at(self.base, ALGO))
        algorithm['Compute'][0]['Symbol'] = {
            'z': {'meaning': 'Copied type', 'shape': 'ref:x.shape', 'dtype': 'ref:x.dtype'}}
        self.assertFalse(rules.validate_algorithm(algorithm))
        algorithm['Compute'][0]['Symbol']['z']['dtype'] = 'ref:absent.dtype'
        self.assertTrue(any('unknown' in d.message for d in rules.validate_algorithm(algorithm)))
        algorithm['Compute'][0].pop('Symbol')
        algorithm['Input']['x']['range'] = '1+' * 75 + '1'
        algorithm['Golden_reference']['Tolerance'] = {'absent': 'exact'}
        diagnostics = rules.validate_algorithm(algorithm)
        self.assertTrue(any(d.path.endswith('.range') for d in diagnostics))
        self.assertTrue(any('Tolerance' in d.path for d in diagnostics))

    def test_nullable_ledger_and_open_expressions_remain_valid(self):
        data = deepcopy(self.base)
        at(data, MATH)['Rewrite'][0]['Ledger'] = {'Baseline': None, 'Gain': None, 'Cost': None}
        at(data, ALGO)['Input']['x'].update(range='finite or -inf', default='1/n')
        at(data, MATH)['Rewrite'][0]['Math'] = 'y = external_library.copy(x)\n'
        self.write_task(data)
        self.assertEqual(data, schema.validate_directory(self.directory))
        at(data, MATH)['Rewrite'][0]['Ledger'].pop('Baseline')
        self.assert_invalid(data, 'Baseline', 'required')

    def test_constant_dependencies_distinguish_values_from_local_or_external_names(self):
        algorithm = deepcopy(at(self.base, ALGO))
        algorithm['Constant'] = {'fixed': {'meaning': 'Fixed expression', 'value': '1 + x[0]'}}
        self.assertTrue(any('cannot depend on runtime' in d.message for d in rules.validate_algorithm(algorithm)))
        for expression in ('sum(x for x in [1, 2, 3])', 'x.fixed_value'):
            with self.subTest(expression=expression):
                algorithm['Constant']['fixed']['value'] = expression
                self.assertFalse(rules.validate_algorithm(algorithm))
        algorithm['Constant'] = {'left': {'meaning': 'First constant', 'value': 'right + 1'},
                                 'right': {'meaning': 'Second constant', 'value': 'left - 1'}}
        self.assertTrue(any('cycle' in d.message for d in rules.validate_algorithm(algorithm)))

    def test_feedback_matches_known_paths_and_leaves_semantic_targets_to_review(self):
        data = deepcopy(self.base)
        data['root']['REALIZATION'] = {
            'Prefer': [{'Item': 'DESIGN.Tile_pipeline::MAIN.MAIN',
                        'Choice': 'One scalar store', 'Reason': 'Final write'}],
            'Feedback': [{'Item': 'DESIGN.Tile_pipeline.Step.MAIN.Role::MAIN',
                          'Choice': 'Another scalar store in the same role',
                          'Evidence': 'experiments@abc123:report.md'}]}
        self.write_task(data)
        self.assertEqual(data, schema.validate_directory(self.directory))
        data['root']['REALIZATION']['Feedback'][0]['Item'] = 'DESIGN.Tile_pipeline::MAIN'
        self.assert_invalid(data, 'covered by Prefer')
        for preferred, feedback in (
                ('CONTRACT.Algorithm_requirement.Input::x', 'CONTRACT.Algorithm_requirement::x'),
                ('x.dtype', 'CONTRACT.Algorithm_requirement::x.dtype'),
                ('DESIGN.Math_reshape::formula_value', 'formula_value')):
            with self.subTest(preferred=preferred, feedback=feedback):
                data['root']['REALIZATION']['Prefer'][0]['Item'] = preferred
                data['root']['REALIZATION']['Feedback'][0]['Item'] = feedback
                at(data, MATH)['Rewrite'][0]['Math'] = 'formula_value = x; y = formula_value\n'
                self.write_task(data)
                self.assertEqual(data, schema.validate_directory(self.directory))
        data['root']['REALIZATION']['Prefer'][0]['Item'] = 'x.dtype'
        data['root']['REALIZATION']['Feedback'][0]['Item'] = 'x.shape'
        self.assert_invalid(data, 'covered by Prefer')
        data['root']['REALIZATION'].pop('Prefer')
        self.assert_invalid(data, 'covered by Prefer')

    def test_item_resolves_known_bare_properties_and_full_names_first(self):
        data = deepcopy(self.base)
        at(data, MATH)['Rewrite'][0]['Symbol'] = {
            'copy': {'meaning': 'Copied value', 'shape': [], 'dtype': 'fp32'}}
        row = {'Item': ['x.dtype', 'DESIGN.Math_reshape::copy.dtype'],
               'Choice': 'Preserve the value types', 'Reason': 'Required precision'}
        data['root']['REALIZATION'] = {'Require': [row]}
        self.write_task(data)
        self.assertEqual(data, schema.validate_directory(self.directory))
        for name, owner in (('x', 'CONTRACT.Algorithm_requirement'), ('copy', 'DESIGN.Math_reshape')):
            with self.subTest(name=name):
                row['Item'] = [name + '.dtype', owner + '::' + name + '.dtype']
                self.assert_invalid(data, 'cannot repeat the same upstream object')
                for target in (name + '.no_such_property', owner + '::' + name + '.no_such_property'):
                    row['Item'] = target
                    self.assert_invalid(data, 'unknown declared target no_such_property')
        at(data, MATH)['Rewrite'][0]['Name'] = 'x.dtype'
        row['Item'] = ['x.dtype', 'CONTRACT.Algorithm_requirement::x.dtype']
        self.write_task(data)
        self.assertEqual(data, schema.validate_directory(self.directory))
        at(data, MATH)['Rewrite'][0]['Name'] = 'Copy'
        at(data, ALGO)['Compute'][0]['Name'] = 'x.dtype'
        row['Item'] = ['x.dtype', 'CONTRACT.Algorithm_requirement.Input::x.dtype']
        self.write_task(data)
        self.assertEqual(data, schema.validate_directory(self.directory))

    def test_operation_names_must_be_unique_only_when_referenced(self):
        for area, field, owner in ((ALGO, 'Compute', 'CONTRACT.Algorithm_requirement'),
                                   (MATH, 'Rewrite', 'DESIGN.Math_reshape')):
            with self.subTest(owner=owner):
                data = deepcopy(self.base)
                entries = [{'Name': 'Cast', 'Math': 'a = fp32(x)'},
                           {'Name': 'Cast', 'Math': 'y = fp32(a)'}]
                at(data, area)[field] = entries
                self.write_task(data)
                self.assertEqual(data, schema.validate_directory(self.directory))
                row = {'Item': 'Cast', 'Choice': 'Use a scalar conversion', 'Reason': 'Preserve rounding'}
                data['root']['REALIZATION'] = {'Require': [row]}
                for target in ('Cast', owner + '::Cast'):
                    row['Item'] = target
                    self.assert_invalid(data, 'Item', 'ambiguous')
                del data['root']['REALIZATION']
                assignment = at(data, STEP)['Schedule']['Worker_assignment']
                previous = assignment['Exit']
                assignment['Exit'] = owner + '::Cast'
                self.assert_invalid(data, 'Exit', 'ambiguous')
                assignment['Exit'] = previous
                if area == MATH:
                    entries[0]['Symbol'] = {'cast_alias': {'meaning': 'Referenced conversion',
                                                          'binding': 'ref:' + owner + '::Cast'}}
                    self.assert_invalid(data, 'binding', 'ambiguous')
                entries[1]['Name'] = 'Final_cast'
                row['Item'] = [owner + '::Cast', owner + '::Final_cast']
                data['root']['REALIZATION'] = {'Require': [row]}
                self.write_task(data)
                self.assertEqual(data, schema.validate_directory(self.directory))

    def test_skip_hardware_preserves_the_raw_payload(self):
        data = deepcopy(self.base)
        put(data, HARD, {'unfinished': 'x' * 2000, 'device': 'd' * 100})
        self.write_task(data)
        before = self.snapshot()
        notices = []
        self.assertEqual(data, schema.validate_directory(self.directory, skip_hardware=True, notices=notices))
        self.assertIn('Hardware skipped', notices[0])
        self.assertEqual(before, self.snapshot())
        with self.assertRaises(schema.SchemaError):
            schema.validate_directory(self.directory)
        put(data, ROLE + ('Info', 'Member'), 'cta[1]')
        self.write_task(data)
        with self.assertRaises(schema.SchemaError):
            schema.validate_directory(self.directory, skip_hardware=True)

    def test_preassigned_alternatives_and_launch_submission(self):
        data = deepcopy(self.base)
        pre = at(data, STEP)['Schedule']['Worker_assignment']['Preassigned']
        pre['stride'] = 1
        self.assert_invalid(data, 'exactly one of start+stride or start+count')
        pre.pop('count')
        self.write_task(data)
        self.assertEqual(data, schema.validate_directory(self.directory))
        at(data, WORK)['Flow']['Launch'].append({'step': ['MAIN']})
        self.assert_invalid(data, 'submitted exactly once')

    def test_start_after_event_ownership_and_launch_cycles(self):
        data = deepcopy(self.base)
        add_step(data)
        self.write_task(data)
        self.assertEqual(data, schema.validate_directory(self.directory))
        at(data, WORK)['Flow']['Launch'][0]['start_after'] = 'MAIN.launch_ready'
        self.assert_invalid(data, 'own submission')
        at(data, WORK)['Flow']['Launch'][0]['start_after'] = 'NEXT.launch_ready'
        self.assert_invalid(data, 'cycle')

    def test_handoff_keeps_unmentioned_consumers_at_the_default_entry(self):
        handoff = {'producer': 'A', 'consumer': ['B', 'C'], 'storage': 'global', 'wait_at': 'B.input_ready'}
        self.assertEqual(consumer_waits(handoff), {'B': ('B.input_ready',), 'C': None})
        data = deepcopy(self.base)
        add_step(data)
        at(data, WORK)['Flow']['Handoff'] = {'y': {'producer': 'MAIN', 'consumer': ['NEXT'], 'storage': 'global',
                                                  'ready': 'NEXT.kernel_done'}}
        self.assert_invalid(data, 'wrong Handoff participant')

    def test_known_storage_size_and_member_bounds(self):
        data = deepcopy(self.base)
        at(data, PIPE)['Storage'] = {'buffer': {'meaning': 'Four scalar slots', 'depth': 1, 'location': 'register',
                                               'shape': [{'element': 4}], 'dtype': 'fp32', 'total_size': 16}}
        self.write_task(data)
        self.assertEqual(data, schema.validate_directory(self.directory))
        at(data, PIPE)['Storage']['buffer']['total_size'] = 20
        self.assert_invalid(data, 'total_size')
        at(data, PIPE)['Storage']['buffer']['total_size'] = 16
        at(data, ROLE)['Info']['Member'] = 'cta[0:2]'
        self.assert_invalid(data, 'unit_count')

    def test_attribute_references_bind_to_the_declared_object(self):
        data = deepcopy(self.base)
        at(data, MATH)['Rewrite'][0]['Symbol'] = {
            'copy': {'meaning': 'Alias of the input', 'shape': 'ref:CONTRACT.Algorithm_requirement::x.shape',
                     'dtype': 'ref:CONTRACT.Algorithm_requirement::x.dtype'}}
        self.write_task(data)
        self.assertEqual(data, schema.validate_directory(self.directory))
        symbols = at(data, MATH)['Rewrite'][0]['Symbol']
        symbols['copy']['dtype'] = 'ref:CONTRACT.Algorithm_requirement::absent.dtype'
        errors = self.assert_invalid(data, 'unknown')
        self.assertEqual(errors.count('unknown shared Contract reference absent'), 1)

    def test_name_and_dtype_budgets_follow_the_value_branch(self):
        errors = []
        values.identifier('n' * 50, 'name', errors)
        values.attribute('dtype')('d' * 100, 'dtype', errors)
        values.identity('Hardware unit (' + 'x' * 70 + ')', 'hardware', errors)
        self.assertFalse(errors)
        values.identifier('n' * 51, 'name', errors)
        values.attribute('dtype')('d' * 101, 'dtype', errors)
        self.assertEqual([d.path for d in errors], ['name', 'dtype'])
        self.assertIn('51 Unicode characters', errors[0].message)
        self.assertIn('100 characters', errors[1].message)
        data = deepcopy(self.base)
        at(data, MATH)['Rewrite'][0]['Symbol'] = {'precision': {'meaning': 'Unresolved precision',
                                                              'dtype': 'unknown:' + 'a' * 142}}
        self.assertFalse(rules.validate_structure(data))
        self.assertFalse(text_policy.check_values(data['root']))
        at(data, MATH)['Rewrite'][0]['Symbol']['precision']['dtype'] += 'a'
        self.assertTrue(any('150 characters' in d.message for d in text_policy.check_values(data['root'])))

    def test_numeric_forms_keep_unicode_values_and_contextual_budgets(self):
        for check in (values.integer(), values.constant, values.scalar, values.input_range):
            for invalid in (' ', '，', '...', '++'):
                with self.subTest(check=check.__name__, value=invalid):
                    errors = []
                    check(invalid, 'value', errors)
                    self.assertTrue(errors)
        errors = []
        for value in ('∞', '-∞', '+∞', '∅', 'None', '⌊n/2⌋'):
            values.scalar(value, 'value', errors)
        values.input_range(['-∞', '+∞'], 'range', errors)
        values.integer()('⌈n/2⌉', 'integer', errors)
        values.constant('1' * 100, 'constant', errors)
        values.literal([0, ['1' * 100, '∅']], 'array', errors)
        self.assertFalse(errors)
        for value in ('∞', '-∞', '+inf'):
            values.integer()(value, 'integer', errors)
        values.constant('1' * 101, 'constant', errors)
        values.literal([['1' * 101]], 'array', errors)
        self.assertTrue(any(d.path == 'array[0][0]' and '100 characters' in d.message for d in errors))
        self.assertTrue(any('infinity' in d.message for d in errors))
        algorithm = deepcopy(at(self.base, ALGO))
        algorithm['Input']['x'].update(range=['1' * 150, '2' * 150], default='1+' * 74 + '1')
        self.assertFalse(rules.validate_algorithm(algorithm))
        errors = []
        values.constant([1, 2], 'constant', errors)
        structure.initialize([1, 2], 'initialize', errors)
        self.assertEqual([d.path for d in errors], ['constant', 'initialize'])

    def test_retention_and_weak_references_preserve_real_names(self):
        for value in ('carry_forward (b,h,q); i=-1..iteration_count-1', 'keep_value invocation',
                      'keep_value (b,h)', 'carry_forward (b); i=0..⌈n/2⌉'):
            errors = []
            structure.retention(value, 'retention', errors)
            self.assertFalse(errors, errors)
        for value in ('keep_value state', 'keep_value invocation in shared memory',
                      'carry_forward (...)', 'carry_forward (b,h); finished',
                      'carry_forward (b,h); i=0..n; stored in shared memory'):
            errors = []
            structure.retention(value, 'retention', errors)
            self.assertTrue(errors, value)
        data = deepcopy(self.base)
        at(data, MATH)['Rewrite'][0]['Name'] = 'Copy (scalar)'
        data['root']['REALIZATION'] = {'Prefer': [
            {'Item': 'DESIGN.Math_reshape::Copy (scalar)', 'Choice': 'One scalar store', 'Reason': 'Known rewrite'},
            {'Item': 'DESIGN.Math_reshape::f(\ni)', 'Choice': 'Use the formula', 'Reason': 'Formula-defined target'}]}
        at(data, HARD)['Feature']['l1/smem (shared)'] = 'Shared capacity'
        data['root']['REALIZATION']['Prefer'].append({
            'Item': 'CONTRACT.Hardware_specification::Feature.l1/smem (shared)',
            'Choice': 'Use shared capacity', 'Reason': 'Known full hardware name'})
        data['root']['REALIZATION']['Prefer'].append({
            'Item': 'CONTRACT.Algorithm_requirement::Copy',
            'Choice': 'Preserve the specified copy', 'Reason': 'Known Contract computation'})
        at(data, MATH)['Rewrite'][0]['Symbol'] = {'alias': {'meaning': 'Known rewrite reference',
                                                        'binding': 'ref:DESIGN.Math_reshape::Copy (scalar)'}}
        self.write_task(data)
        self.assertEqual(data, schema.validate_directory(self.directory))
        data['root']['REALIZATION']['Prefer'][0]['Item'] = 'DESIGN.Math_reshape::formula value with explanation'
        self.assert_invalid(data, 'put explanations in a descriptive field')
        errors = []
        values.semantic_reference('x[:]', 'reference', errors)
        self.assertFalse(errors)

    def test_semantic_text_budgets_keep_two_lines_and_external_locator_branches(self):
        errors = []
        target = 'bind:DESIGN.Math_reshape::f(\n' + 'i,' * 125 + 'i)'
        values.binding(target, 'binding', errors)
        values.semantic_reference(target, 'binding', errors)
        values.binding('ref:repo@v1:src/' + 'x' * 600 + '.py::entry', 'external', errors)
        values.source('https://example.invalid/' + 'x' * 600, 'source', errors)
        values.source('a' * 149 + '\n' + 'b' * 150, 'summary', errors)
        self.assertFalse(errors)
        values.source('a' * 301, 'summary', errors)
        values.binding('ref:' + 'a' * 297, 'binding', errors)
        self.assertEqual([d.path for d in errors], ['summary', 'binding'])
        data = deepcopy(self.base['root'])
        assignment = data['DESIGN']['Work_schedule']['Step']['MAIN']['Schedule']['Worker_assignment']
        assignment['Exit'] = ['a' * 149 + '\n' + 'b' * 150, 'c' * 300]
        data['REALIZATION'] = {'Require': [{'Item': 'a' * 149 + '\n' + 'b' * 150,
                                          'Choice': 'Use the target', 'Reason': 'Preserve the condition'}]}
        self.assertFalse(text_policy.check_values(data))
        assignment['Exit'][1] += 'c'
        self.assertTrue(any(d.path.endswith('.Exit[1]') and '300 characters' in d.message
                            for d in text_policy.check_values(data)))

    def test_item_prefers_registered_bracket_names_over_index_notation(self):
        data = deepcopy(self.base)
        at(data, MATH)['Rewrite'][0]['Name'] = 'Copy [scalar]'
        at(data, HARD)['Feature']['l1/smem [shared]'] = 'Shared capacity'
        references = ('Copy [scalar]', 'DESIGN.Math_reshape::Copy [scalar]',
                      'CONTRACT.Hardware_specification::Feature.l1/smem [shared]')
        data['root']['REALIZATION'] = {'Prefer': [
            {'Item': target, 'Choice': 'Use declared target ' + str(i), 'Reason': 'Preserve declared behavior'}
            for i, target in enumerate(references)]}
        self.write_task(data)
        self.assertEqual(data, schema.validate_directory(self.directory))
        data['root']['REALIZATION']['Prefer'][0]['Item'] = 'DESIGN.Math_reshape::formula[0]'
        self.assert_invalid(data, 'not list indices')

    def test_repository_paths_keep_internal_spaces_but_reject_ambiguous_roots(self):
        for value in ('repo@v1:', 'repo@v1:/absolute/file', 'repo@v1:../file',
                      'repo@v1: file', 'repo@v1:file ', 'repo@v1:file ::entry',
                      'repo@v1:file::', 'repo@v1:a other@v2:b'):
            with self.subTest(value=value):
                errors = []
                values.repository(value, 'source', errors)
                self.assertTrue(errors)
        errors = []
        values.repository('repo@v1:dir with spaces/(model)/file.py::Kernel.run', 'source', errors, entry=True)
        values.repository('repo@v1:known directory/', 'source', errors)
        self.assertFalse(errors)

    def test_local_repository_uses_available_commit_not_the_checkout(self):
        def git(*arguments):
            return subprocess.run(['git', '-C', str(self.directory), *arguments], check=True,
                                  capture_output=True, text=True).stdout.strip()
        git('init', '-q')
        original = self.directory / 'source (original).py'
        original.write_text('def entry():\n    return 1\n', encoding='utf-8')
        materials = self.directory / 'device materials'
        materials.mkdir()
        manual = materials / 'device.md'
        manual.write_text('Synthetic device material\n', encoding='utf-8')
        git('add', original.name, materials.name)
        git('-c', 'user.name=Schema test', '-c', 'user.email=schema@example.invalid',
            'commit', '-qm', 'Local reference fixture')
        commit = git('rev-parse', 'HEAD')
        original.unlink()
        data = deepcopy(self.base['root'])
        data['META_DATA']['Source'] = f'{self.directory.name}@{commit}:{original.name}::entry'
        files = {'root': self.directory / 'root.yaml'}

        def diagnostics(**options):
            document = yaml_source.loads(encode_value(data), 'root')
            return values.local_repository_errors(files, {'root': document}, **options)

        self.assertFalse(diagnostics())
        data['META_DATA']['Source'] = f'{self.directory.name}@{commit}:absent.py::entry'
        self.assertEqual([d.path for d in diagnostics()], ['root.META_DATA.Source'])
        data['META_DATA']['Source'] = f'{self.directory.name}@unavailable-ref:absent.py::entry'
        self.assertFalse(diagnostics())
        data['CONTRACT']['Hardware_specification']['Wiki']['Profile'] = f'{self.directory.name}@{commit}:absent.md'
        self.assertTrue(diagnostics())
        self.assertFalse(diagnostics(skip_hardware=True))
        data['CONTRACT']['Hardware_specification']['Wiki']['Profile'] = f'<{self.directory.name}@{commit}:absent.md>'
        self.assertFalse(diagnostics())
        wiki = data['CONTRACT']['Hardware_specification']['Wiki']
        wiki['Programming']['Language'] = f'{self.directory.name}@{commit}:{materials.name}'
        wiki['Hardware'] = f'{self.directory.name}@{commit}:{materials.name}'
        errors = diagnostics()
        self.assertEqual([d.path for d in errors], ['root.CONTRACT.Hardware_specification.Wiki.Hardware'])
        self.assertIn('not a directory', errors[0].message)
        self.assertFalse(diagnostics(skip_hardware=True))
        wiki['Hardware'] = f'<{self.directory.name}@pending:{materials.name}/device.md>'
        self.assertFalse(diagnostics())
        manual.unlink()
        wiki['Hardware'] = f'{self.directory.name}@{commit}:{materials.name}/device.md'
        self.assertFalse(diagnostics())
        git('update-index', '--add', '--cacheinfo', f'160000,{commit},device_submodule')
        git('-c', 'user.name=Schema test', '-c', 'user.email=schema@example.invalid',
            'commit', '-qm', 'Device submodule fixture')
        wiki['Hardware'] = f'{self.directory.name}@{git("rev-parse", "HEAD")}:device_submodule'
        errors = diagnostics()
        self.assertEqual([d.path for d in errors], ['root.CONTRACT.Hardware_specification.Wiki.Hardware'])
        self.assertIn('not a directory', errors[0].message)
