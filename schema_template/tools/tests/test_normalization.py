"""Whole-task normalization, source recovery and publication boundaries."""
from copy import deepcopy
from decimal import Decimal
import hashlib
from pathlib import Path
from unittest.mock import patch

from .. import normalization, schema, text_policy, yaml_format, yaml_source
from .support import ALGO, MATH, PIPE, ROLE, STEP, WORK, TaskCase, add_child, at


class NormalizationTests(TaskCase):
    def mixed_task(self):
        data = deepcopy(self.base)
        algorithm = at(data, ALGO)
        algorithm['Input']['X'] = algorithm['Input'].pop('x')
        algorithm['Output']['Y'] = algorithm['Output'].pop('y')
        algorithm['Golden_reference']['Tolerance'] = {'Y': 'exact'}
        algorithm['Compute'][0]['Math'] = 'Y = X\n'
        at(data, MATH)['Rewrite'][0]['Math'] = 'Y = X\n'
        work = at(data, STEP)
        invariant = work['Decomposition']['Worktile']['Invariant']
        invariant['X'] = invariant.pop('x')
        target = work['Decomposition']['Target']['Single_tile']
        target['Y'] = target.pop('y')
        pipe = at(data, PIPE)
        role = at(data, ROLE)
        compute = role['Info']['Using_verb']['compute']
        compute['Y'] = compute.pop('y')
        role['Pipe'] = ['compute:Y']
        role['Waiting'] = {'Allowed': ['{done compute:Y} ← [{issue compute:Y}]']}
        pipe['Event'] = {'Finished': ['{done worker[0]::compute:Y}']}
        pipe['Role']['worker'] = pipe['Role'].pop('MAIN')
        at(data, WORK)['Step']['copy'] = at(data, WORK)['Step'].pop('MAIN')
        at(data, WORK)['Flow']['Launch'] = [{'step': ['copy']}]
        data['root']['DESIGN']['Tile_pipeline']['Step']['copy'] = data['root']['DESIGN']['Tile_pipeline']['Step'].pop('MAIN')
        data['root']['META_DATA']['describe'] = data['root']['META_DATA'].pop('Describe')
        child = add_child(data, source='root')
        child['META_DATA']['Dispatch']['when'] = 'X > 0'
        child['REALIZATION'] = {'Require': [{'Item': 'DESIGN.Tile_pipeline::copy.worker',
                                            'Choice': 'Keep the scalar result', 'Reason': 'Preserve the selected behavior'}]}
        return data

    def test_default_updates_provider_and_from_consumer_names_references_and_main(self):
        self.write_task(self.mixed_task())
        code, out, error = self.cli(node='branch.yaml')
        self.assertEqual(code, 0, error)
        self.assertIn('Normalization completed: 2 YAML; 2 changed', out)
        self.assertIn('warning: applied field spelling', error)
        self.assertIn('warning: applied declarations and references', error)
        self.assertIn("'X > 0' → 'x > 0'", error)
        self.assertIn("'copy' → 'MAIN'", error)
        result = schema.validate_directory(self.directory)
        self.assertEqual(set(at(result, ALGO)['Input']), {'x'})
        self.assertEqual(at(result, MATH)['Rewrite'][0]['Math'], 'y = x\n')
        self.assertEqual(at(result, WORK)['Flow']['Launch'], [{'step': ['MAIN']}])
        self.assertEqual(at(result, PIPE)['Event'], {'finished': ['{done MAIN[0]::compute:y}']})
        self.assertEqual(at(result, ROLE)['Waiting']['Allowed'], ['{done compute:y} ← [{issue compute:y}]'])
        self.assertEqual(result['branch']['REALIZATION']['Require'][0]['Item'], 'DESIGN.Tile_pipeline::MAIN.MAIN')
        child = yaml_source.load(self.directory / 'branch.yaml', 'branch').value
        self.assertEqual(child['DESIGN'], {region: {'from': 'root'} for region in ('Math_reshape', 'Work_schedule', 'Tile_pipeline')})
        self.assertEqual(child['META_DATA']['Dispatch'], {'id': 'branch', 'when': 'x > 0'})
        before = self.snapshot()
        code, _, error = self.cli()
        self.assertEqual(code, 0, error)
        self.assertNotIn('warning:', error)
        self.assertEqual(before, self.snapshot())

    def test_explicit_plan_is_read_only_keeps_original_hashes_and_validates_without_io(self):
        self.write_task(self.mixed_task())
        before = self.snapshot()
        plan = schema.normalize_directory(self.directory)
        self.assertEqual(before, self.snapshot())
        for name, path in plan.files.items():
            self.assertEqual(plan.source_hashes[name], hashlib.sha256(before[path.name][0]).hexdigest())
            self.assertEqual(plan.documents[name].source_sha256, hashlib.sha256(plan.candidates[name].encode()).hexdigest())
            self.assertNotEqual(plan.source_hashes[name], plan.documents[name].source_sha256)
        original = {name: deepcopy(document.value) for name, document in plan.documents.items()}
        with patch('pathlib.Path.read_bytes', side_effect=AssertionError('snapshot API performed file I/O')):
            resolved = schema.validate_documents(plan.files, plan.documents)
            again = normalization.normalize_documents(plan.files, plan.documents)
        self.assertEqual(resolved.raw_task, original)
        self.assertFalse(again.candidates)
        self.assertFalse(again.corrections)
        self.assertEqual(original, {name: document.value for name, document in plan.documents.items()})
        self.assertEqual(before, self.snapshot())

    def test_final_field_spellings_and_lowercase_attributes_share_source_recovery(self):
        source = ('DESIGN:\n  Math_reshape:\n    Thesis: Retain the calculation\n    Rewrite:\n'
                  '      - Arithmetic: {approx: true, accumulate: 001}\n'
                  '        Ledger: {baseline: null, gain: 0x10, cost: false}\n'
                  '  Work_schedule:\n    Step:\n      MAIN:\n'
                  '        Decomposition: {Target: {Tile_reduction: {Dimension: row, Input: x, Output: y, Method: combine}}}\n'
                  '        Schedule: {Worker_assignment: {Preassigned: {Start: 0, Stride: 1, Count: 2}}}\n'
                  '    Flow: {Launch: [{Step: [main], Order: ALL, Start_after: [main.KERNEL_DONE]}], '
                  'Handoff: {x: {Producer: MAIN, Consumer: [MAIN], Storage: global, '
                  'Initialize: 0, Ready: MAIN.kernel_done, Wait_at: MAIN.launch_ready}}}\n'
                  '  Tile_pipeline:\n    thesis: 0x10\n    Step:\n      MAIN:\n'
                  '        Index: {i: {Meaning: Iteration index, Definition: [0, 2]}}\n'
                  '        Role: {MAIN: {Info: {Describe: Role purpose, Member: "warp[0:1]", '
                  'Using_verb: {compute: {x: {Execution_mode: serial, Describe: Keep instruction ex2.approx}}}}}}\n')
        document = yaml_source.loads(source, 'root')
        child = yaml_source.loads('DESIGN: {Math_reshape: {From: root}, Work_schedule: {From: root}, '
                                  'Tile_pipeline: {From: root}}\n', 'branch')
        files = {name: self.directory / (name + '.yaml') for name in ('root', 'branch')}
        plan = normalization.normalize_documents(files, {'root': document, 'branch': child})
        value = plan.documents['root'].value['DESIGN']
        rewrite = value['Math_reshape']['Rewrite'][0]
        self.assertEqual(rewrite['Arithmetic'], {'Approx': 'true', 'Accumulate': '001'})
        self.assertEqual(rewrite['Ledger'], {'Baseline': None, 'Gain': '0x10', 'Cost': 'false'})
        self.assertEqual(value['Tile_pipeline']['Thesis'], '0x10')
        work = value['Work_schedule']['Step']['MAIN']
        self.assertEqual(set(work['Decomposition']['Target']['Tile_reduction']), {'dimension', 'input', 'output', 'method'})
        self.assertEqual(work['Schedule']['Worker_assignment']['Preassigned'], {'start': 0, 'stride': 1, 'count': 2})
        self.assertEqual(value['Work_schedule']['Flow']['Launch'],
                         [{'step': ['MAIN'], 'order': 'all', 'start_after': ['MAIN.kernel_done']}])
        self.assertEqual(set(value['Work_schedule']['Flow']['Handoff']['x']),
                         {'producer', 'consumer', 'storage', 'initialize', 'ready', 'wait_at'})
        pipeline = value['Tile_pipeline']['Step']['MAIN']
        self.assertEqual(pipeline['Index']['i'], {'meaning': 'Iteration index', 'definition': [0, 2]})
        self.assertEqual(pipeline['Role']['MAIN']['Info']['Using_verb']['compute']['x'],
                         {'execution_mode': 'serial', 'describe': 'Keep instruction ex2.approx'})
        self.assertEqual(plan.documents['branch'].value['DESIGN'], {name: {'from': 'root'} for name in value})
        field_warnings = {item.path for item in plan.corrections if item.message.startswith('field:')}
        for suffix in ('Arithmetic.Approx', 'Arithmetic.Accumulate', 'Ledger.Baseline', 'Ledger.Gain', 'Ledger.Cost'):
            self.assertIn('root.DESIGN.Math_reshape.Rewrite[0].' + suffix, field_warnings)
        self.assertIn('root.DESIGN.Tile_pipeline.Thesis', field_warnings)
        for field in ('step', 'order', 'start_after'):
            self.assertIn('root.DESIGN.Work_schedule.Flow.Launch[0].' + field, field_warnings)
        self.assertFalse(normalization.normalize_documents(plan.files, plan.documents).candidates)
        self.assertIn('thesis', document.value['DESIGN']['Tile_pipeline'])
        budgeted = deepcopy(plan.documents['root'].value)
        budgeted['DESIGN']['Tile_pipeline']['Thesis'] = 'x' * 301
        budgeted['DESIGN']['Math_reshape']['Rewrite'][0]['Arithmetic']['Approx'] = 'x' * 151
        budgeted['DESIGN']['Math_reshape']['Rewrite'][0]['Ledger']['Gain'] = 'x' * 151
        self.assertEqual({item.path for item in text_policy.check_values(budgeted)}, {
            'root.DESIGN.Tile_pipeline.Thesis', 'root.DESIGN.Math_reshape.Rewrite[0].Arithmetic.Approx',
            'root.DESIGN.Math_reshape.Rewrite[0].Ledger.Gain'})

    def test_read_only_checks_candidates_and_marks_their_locations(self):
        data = self.mixed_task()
        data['root']['META_DATA']['describe'] = '界' * 151
        self.write_task(data)
        before = self.snapshot()
        code, out, error = self.cli('--read-only', node='branch.yaml')
        self.assertEqual(code, 1)
        self.assertIn('151 Unicode characters', error)
        self.assertIn('normalization preview; not written', error)
        self.assertIn('Normalization preview: 2 YAML; 2 would change', out)
        self.assertNotIn('warning: applied', error)
        self.assertEqual(before, self.snapshot())
        code, _, error = self.cli(node='branch.yaml')
        self.assertEqual(code, 1)
        document = yaml_source.load(self.directory / 'root.yaml', 'root')
        line, column = document.positions['root.META_DATA.Describe']
        self.assertIn(f'root.yaml:{line}:{column}', error)
        self.assertNotIn('normalization preview', error)
        self.assertEqual(len(document.value['META_DATA']['Describe']), 151)

    def test_common_missing_quotes_and_scalar_lexemes_share_one_checked_recovery(self):
        source = self.source.replace('Name: scalar_copy', 'Name: 0x10')
        source = source.replace('Describe: Copy one scalar in a synthetic validation task', 'Describe: Copy: one scalar')
        source = source.replace('      MAIN:\n        Role:',
                                '      MAIN:\n        Event:\n          finished:\n            - {done MAIN[0]::compute:y}\n        Role:')
        source = source.replace('            Pipe:\n',
                                '            Waiting:\n              Allowed:\n                - {done compute:y} ← [{issue compute:y}]\n            Pipe:\n')
        path = self.directory / 'root.yaml'
        path.write_bytes(source.replace('\n', '\r\n').encode())
        before = self.snapshot()
        plan = schema.normalize_directory(self.directory)
        self.assertEqual(plan.source_hashes['root'], hashlib.sha256(before['root.yaml'][0]).hexdigest())
        self.assertEqual(before, self.snapshot())
        code, _, error = self.cli()
        self.assertEqual(code, 0, error)
        self.assertIn('warning: applied YAML scalar quoting', error)
        value = yaml_source.load(path, 'root').value
        self.assertEqual(value['META_DATA']['Name'], '0x10')
        self.assertEqual(value['META_DATA']['Describe'], 'Copy: one scalar')
        self.assertEqual(value['META_DATA']['Version'], 1)
        self.assertEqual(value['DESIGN']['Tile_pipeline']['Step']['MAIN']['Event']['finished'], ['{done MAIN[0]::compute:y}'])
        self.assertEqual(value['DESIGN']['Tile_pipeline']['Step']['MAIN']['Role']['MAIN']['Waiting']['Allowed'],
                         ['{done compute:y} ← [{issue compute:y}]'])

    def test_quoted_integer_and_boolean_fields_normalize_in_preview_and_default_cli(self):
        for member in ('0', ['0']):
            with self.subTest(member=member):
                data = deepcopy(self.base)
                data['root']['META_DATA']['Version'] = '1'
                at(data, ALGO)['Input']['x']['nullable'] = 'true'
                at(data, ALGO)['Output']['y']['nullable'] = 'FALSE'
                at(data, ROLE)['Info']['Register'] = '0x20'
                at(data, PIPE)['Storage'] = {'buffer': {'meaning': 'One scalar slot', 'depth': 1,
                                                       'location': 'register', 'shape': [], 'dtype': 'fp32',
                                                       'total_size': 4, 'member': member}}
                self.write_task(data)
                before = self.snapshot()
                with self.assertRaisesRegex(schema.SchemaError, 'expected an integer'):
                    schema.validate_directory(self.directory)
                code, out, error = self.cli('--read-only')
                self.assertEqual(code, 1, error)
                self.assertIn('Schema checked:', out)
                self.assertIn('warning: planned YAML scalar quoting', error)
                self.assertIn('normalization preview; not written', error)
                self.assertNotIn('warning: applied', error)
                self.assertEqual(before, self.snapshot())
                code, _, error = self.cli()
                self.assertEqual(code, 0, error)
                self.assertIn('warning: applied YAML scalar quoting', error)
                result = schema.validate_directory(self.directory)
                self.assertIs(type(result['root']['META_DATA']['Version']), int)
                self.assertEqual(result['root']['META_DATA']['Version'], 1)
                self.assertIs(at(result, ALGO)['Input']['x']['nullable'], True)
                self.assertIs(at(result, ALGO)['Output']['y']['nullable'], False)
                self.assertEqual(at(result, ROLE)['Info']['Register'], 32)
                self.assertEqual(at(result, PIPE)['Storage']['buffer']['member'], [0] if isinstance(member, list) else 0)
                after = self.snapshot()
                code, _, error = self.cli()
                self.assertEqual(code, 0, error)
                self.assertNotIn('warning:', error)
                self.assertEqual(after, self.snapshot())

    def test_quoted_type_recovery_preserves_source_hashes_aliases_and_merge_uses(self):
        source = ('META_DATA: {version: &number "01"}\n'
                  'CONTRACT:\n  Algorithm_requirement:\n'
                  "    Input: {x: &input {nullable: &flag 'TRUE'}, z: {<<: *input}}\n"
                  '    Output: {y: {nullable: *flag}}\n'
                  'DESIGN:\n  Tile_pipeline:\n    Step:\n      MAIN:\n'
                  '        Storage: {buffer: {member: [*number]}}\n'
                  '        Role: {MAIN: {Info: {Register: *number}}}\n').replace('\n', '\r\n')
        document = yaml_source.loads(source, 'root')
        original = deepcopy(document.value)
        formatted = yaml_source.loads(yaml_format.format_document(document), 'root')
        self.assertIs(type(formatted.value['META_DATA']['version']), str)
        number = document.anchors['number']
        uses = tuple(context for context in yaml_source.scalar_contexts(document) if context.node is number)
        with self.assertRaisesRegex(schema.ToolError, 'only edit string scalars'):
            yaml_format.format_document(document, edits=(yaml_format.ScalarEdit(number, 1, uses),))
        with patch('pathlib.Path.read_bytes', side_effect=AssertionError('snapshot recovery performed file I/O')):
            plan = normalization.normalize_documents({'root': self.directory / 'root.yaml'}, {'root': document})
        candidate = plan.documents['root']
        self.assertEqual(document.value, original)
        self.assertEqual(document.source, source)
        self.assertEqual(plan.source_hashes['root'], hashlib.sha256(source.encode()).hexdigest())
        self.assertEqual(candidate.source_sha256, hashlib.sha256(plan.candidates['root'].encode()).hexdigest())
        self.assertNotEqual(plan.source_hashes['root'], candidate.source_sha256)
        self.assertEqual(candidate.value['META_DATA']['Version'], 1)
        self.assertEqual(candidate.anchors['number'].value, '01')
        self.assertEqual(candidate.anchors['number'].tag, 'tag:yaml.org,2002:int')
        self.assertEqual(candidate.anchors['flag'].tag, 'tag:yaml.org,2002:bool')
        self.assertIs(candidate.value['CONTRACT']['Algorithm_requirement']['Input']['z']['nullable'], True)
        self.assertEqual(set(candidate.anchors), set(document.anchors))
        self.assertEqual([use.alias for use in candidate.occurrences if use.alias],
                         [use.alias for use in document.occurrences if use.alias])
        paths = {item.path for item in plan.corrections if item.message.startswith('quote: ')}
        self.assertEqual(paths, {'root.META_DATA.Version', 'root.CONTRACT.Algorithm_requirement.Input.x.nullable',
                                 'root.CONTRACT.Algorithm_requirement.Input.z.nullable',
                                 'root.CONTRACT.Algorithm_requirement.Output.y.nullable',
                                 'root.DESIGN.Tile_pipeline.Step.MAIN.Storage.buffer.member[0]',
                                 'root.DESIGN.Tile_pipeline.Step.MAIN.Role.MAIN.Info.Register'})
        self.assertTrue(all(path in candidate.positions for path in paths))

    def test_type_quote_recovery_preserves_tags_null_unions_and_nonliteral_text(self):
        documents = []
        for value in ('null', '"null"', '!!str "1"', '"1.0"', '"1e0"', '"true"', '"0x_"',
                      '" 1"', '"1 "', '"\\x31"', '''"'1'"'''):
            documents.append('META_DATA: {Version: ' + value + '}\n')
        for value in ('null', '"null"', '!!str "true"', '"yes"', '"no"', '"on"', '"1"', '"0"',
                      '" true"', '"false "', '"tr\\u0075e"'):
            documents.append('CONTRACT: {Algorithm_requirement: {Input: {x: {nullable: ' + value + '}}}}\n')
        documents.extend(('META_DATA:\n  Version: |-\n    1\n', 'META_DATA:\n  Version: "1\\\n    "\n',
                          'CONTRACT: {Algorithm_requirement: {Constant: {n: {value: "1"}}, '
                          'Input: {x: {default: "true", shape: [{n: "1"}], range: ["1", "2"]}}}}\n'
                          'DESIGN: {Work_schedule: {Step: {MAIN: {Mapping: {Worker_count: "1"}}}}}\n'))
        for source in documents:
            with self.subTest(source=source):
                original = yaml_source.loads(source, 'root')
                loaded = normalization.loads_for_normalization(source, 'root')
                self.assertEqual(loaded.document.value, original.value)
                self.assertEqual(loaded.document.source, source)
                self.assertFalse(loaded.corrections)

    def test_type_quote_recovery_requires_every_alias_and_merge_context_to_agree(self):
        for source in (
                'META_DATA: {Version: &value "1", Name: *value}\n',
                'META_DATA: {Version: &value "1"}\n'
                'CONTRACT: {Algorithm_requirement: {Input: {x: {nullable: *value}}}}\n',
                'CONTRACT: {Algorithm_requirement: {Input: {x: {nullable: &value "true", default: *value}}}}\n',
                'META_DATA: {Version: &value "1"}\n'
                'DESIGN: {Work_schedule: {Step: {MAIN: {Mapping: {Worker_count: *value}}}}}\n',
                'META_DATA: {Version: &value "1"}\nunrelated: {*value: Keep the key}\n',
                'CONTRACT: {Algorithm_requirement: {Input: {x: &shared {nullable: "true"}}, '
                'Output: {y: {<<: *shared}}}}\nunrelated: {<<: *shared, nullable: false}\n'):
            with self.subTest(source=source):
                loaded = normalization.loads_for_normalization(source, 'root')
                self.assertEqual(loaded.document.source, source)
                self.assertEqual(loaded.document.value, yaml_source.loads(source, 'root').value)
                self.assertFalse(loaded.corrections)

    def test_recovered_literal_types_still_obey_value_validation(self):
        for original, expected, message in (('0', 0, 'literal must be >= 1'), ('1.0', '1.0', 'expected an integer')):
            with self.subTest(original=original):
                data = deepcopy(self.base)
                data['root']['META_DATA']['Version'] = original
                self.write_task(data)
                code, _, error = self.cli()
                self.assertEqual(code, 1, error)
                self.assertIn(message, error)
                actual = yaml_source.load(self.directory / 'root.yaml', 'root').value['META_DATA']['Version']
                self.assertIs(type(actual), type(expected))
                self.assertEqual(actual, expected)
                if type(expected) is int:
                    self.assertIn('warning: applied YAML scalar quoting', error)
                    self.assertIn("YAML integer literal '0'", error)
                else:
                    self.assertNotIn('YAML integer literal', error)

    def test_collisions_and_unprovable_damage_leave_every_source_untouched(self):
        collision = self.mixed_task()
        collision['root']['META_DATA']['Describe'] = 'A second distinct field'
        names = self.mixed_task()
        at(names, ALGO)['Input']['x'] = deepcopy(at(names, ALGO)['Input']['X'])
        coordinates = self.mixed_task()
        at(coordinates, MATH)['Rewrite'][0]['Math'] = 'z[T,t] = T + t\nY = X\n'
        for data in (collision, names, coordinates):
            with self.subTest(data=data['root']['META_DATA']):
                self.write_task(data)
                before = self.snapshot()
                code, _, error = self.cli(node='branch.yaml')
                self.assertEqual(code, 1, error)
                self.assertRegex(error, 'collision|collid|ambiguous|would merge')
                self.assertNotIn('warning: applied', error)
                self.assertEqual(before, self.snapshot())
        path = self.directory / 'root.yaml'
        path.write_text('META_DATA:\n  Describe: "unterminated\n', encoding='utf-8')
        before = self.snapshot()
        self.assertEqual(self.cli()[0], 1)
        self.assertEqual(before, self.snapshot())

    def test_null_and_unknown_old_words_remain_actionable_errors(self):
        data = deepcopy(self.base)
        data['root']['META_DATA']['Describe'] = None
        data['root']['CONTRACT']['Hardware_specification']['arch'] = 'test'
        self.write_task(data)
        code, _, error = self.cli()
        self.assertEqual(code, 1)
        self.assertIn('Describe', error)
        self.assertIn('arch', error)
        self.assertIn('field not allowed', error)
        value = yaml_source.load(self.directory / 'root.yaml', 'root').value
        self.assertIsNone(value['META_DATA']['Describe'])
        self.assertEqual(value['CONTRACT']['Hardware_specification']['arch'], 'test')

    def test_file_ids_and_from_source_values_are_not_renamed(self):
        data = deepcopy(self.base)
        data['root']['META_DATA']['Dispatch'] = {'id': 'ROOT'}
        self.write_task(data)
        code, _, error = self.cli()
        self.assertEqual(code, 1)
        self.assertIn('Dispatch.id', error)
        self.assertEqual(yaml_source.load(self.directory / 'root.yaml', 'root').value['META_DATA']['Dispatch']['id'], 'ROOT')
        self.assertFalse((self.directory / 'ROOT.yaml').exists())
        data = deepcopy(self.base)
        add_child(data, source='ROOT')
        self.write_task(data)
        before = self.snapshot()
        code, _, error = self.cli()
        self.assertEqual(code, 1)
        self.assertIn("unknown source 'ROOT'", error)
        self.assertEqual(before, self.snapshot())

    def test_math_formals_cover_operator_continuations_without_leaking(self):
        for area, collection in ((ALGO, 'Compute'), (MATH, 'Rewrite')):
            for body in ('T +\n  T', 'T\n  + T', 'T +\n  T +\n  T'):
                with self.subTest(area=area, body=body):
                    data = deepcopy(self.base)
                    formula = (f'F(T) = {body}\nG(U) = U + T\n  + U\n'
                               'y = F(x) + G(x) + sum(Q for Q in x)\nz = T + U + Q\n')
                    at(data, area)[collection][0]['Math'] = formula
                    self.write_task(data)
                    before = self.snapshot()
                    plan = schema.normalize_directory(self.directory)
                    expected = (f'f(t) = {body.lower()}\ng(u) = u + T\n  + u\n'
                                'y = f(x) + g(x) + sum(q for q in x)\nz = T + U + Q\n')
                    self.assertEqual(at({'root': plan.documents['root'].value}, area)[collection][0]['Math'], expected)
                    schema.validate_documents(plan.files, plan.documents)
                    self.assertEqual(before, self.snapshot())

    def test_math_formals_and_independent_quantifiers_keep_their_own_scopes(self):
        data = deepcopy(self.base)
        at(data, MATH)['Rewrite'][0]['Math'] = (
            'F(t) = T +\n  sum(Q for Q in x) + T\n'
            'T for T in x\ny = F(x) + T + Q\n')
        self.write_task(data)
        plan = schema.normalize_directory(self.directory)
        self.assertEqual(plan.documents['root'].value['DESIGN']['Math_reshape']['Rewrite'][0]['Math'],
                         'f(t) = T +\n  sum(q for q in x) + T\n'
                         't for t in x\ny = f(x) + T + Q\n')
        schema.validate_documents(plan.files, plan.documents)

    def test_unregistered_math_temporaries_keep_distinct_case_and_arithmetic(self):
        for assignment in ('=', ':='):
            with self.subTest(assignment=assignment):
                data = deepcopy(self.base)
                algorithm = at(data, ALGO)
                algorithm['Input']['v'] = algorithm['Input'].pop('x')
                algorithm['Compute'][0]['Math'] = 'y = 2*v + 1\n'
                formula = f'V {assignment} v + 1\ny = V + v\n'
                at(data, MATH)['Rewrite'][0]['Math'] = formula
                self.write_task(data)
                before = self.snapshot()
                plan = schema.normalize_directory(self.directory)
                actual = plan.documents['root'].value['DESIGN']['Math_reshape']['Rewrite'][0]['Math']
                self.assertEqual(actual, formula)
                values = {'v': 1}
                exec(actual.replace(':=', '='), {}, values)
                self.assertEqual(values['y'], 3)
                self.assertEqual(before, self.snapshot())
                self.assertFalse(normalization.normalize_documents(plan.files, plan.documents).candidates)

    def test_registered_rename_rejects_capture_of_unregistered_math_before_writes(self):
        for formula in ('v = V + 1\ny = V + v\n', 'v := V + 1\ny = V + v\n',
                        'y = (lambda v: V + v)(1)\n'):
            with self.subTest(formula=formula):
                data = deepcopy(self.base)
                algorithm = at(data, ALGO)
                algorithm['Input']['V'] = algorithm['Input'].pop('x')
                algorithm['Compute'][0]['Math'] = 'y = 2*V + 1\n'
                at(data, MATH)['Rewrite'][0]['Math'] = formula
                add_child(data, source='root')
                self.write_task(data)
                before = self.snapshot()
                with self.assertRaisesRegex(schema.SchemaError, 'capture unbound name'):
                    schema.normalize_directory(self.directory)
                self.assertEqual(before, self.snapshot())

    def test_registered_objects_with_distinct_case_never_become_related_aliases(self):
        data = deepcopy(self.base)
        at(data, ALGO)['Input']['V'] = at(data, ALGO)['Input'].pop('x')
        at(data, ALGO)['Compute'][0]['Math'] = 'y = V\n'
        at(data, MATH)['Rewrite'][0]['Math'] = 'y = V\n'
        at(data, STEP)['Decomposition']['Worktile']['Invariant']['v'] = {'definition': 'V'}
        self.write_task(data)
        before = self.snapshot()
        with self.assertRaisesRegex(schema.SchemaError, 'would share an identity'):
            schema.normalize_directory(self.directory)
        self.assertEqual(before, self.snapshot())

    def test_one_provider_cannot_be_renamed_under_different_consumer_bindings(self):
        data = deepcopy(self.base)
        at(data, STEP)['Decomposition']['Worktile']['Invariant']['x']['shape'] = [{'i': 'V'}]
        child = add_child(data, source='root')
        child['DESIGN']['Math_reshape'] = deepcopy(at(data, MATH))
        child['DESIGN']['Math_reshape']['Rewrite'][0]['Symbol'] = {
            'V': {'meaning': 'Consumer extent', 'shape': [], 'dtype': 'int32'}}
        self.write_task(data)
        before = self.snapshot()
        with self.assertRaisesRegex(schema.SchemaError, 'incompatible normalization bindings'):
            schema.normalize_directory(self.directory)
        self.assertEqual(before, self.snapshot())

    def test_local_renames_cannot_capture_outer_names_or_publish_sibling_changes(self):
        for kind in ('formal', 'quantifier', 'open_quantifier', 'pipe', 'event', 'waiting', 'view'):
            with self.subTest(kind=kind):
                data = deepcopy(self.base)
                algorithm = at(data, ALGO)
                algorithm['Input']['t'] = algorithm['Input'].pop('x')
                algorithm['Compute'][0]['Math'] = 'y = t\n'
                invariant = at(data, STEP)['Decomposition']['Worktile']['Invariant']
                invariant['t'] = invariant.pop('x')
                rewrite = at(data, MATH)['Rewrite'][0]
                rewrite['Math'] = 'y = t\n'
                if kind == 'formal':
                    rewrite['Math'] = 'f(T) = T + t\ny = f(0)\n'
                elif kind == 'quantifier':
                    rewrite['Math'] = 'y = sum(T + t for T in [0])\n'
                elif kind == 'open_quantifier':
                    rewrite['Math'] = 'y = sum(\n  T + t\n  for T in [0]\n) ⊙ t\n'
                elif kind == 'pipe':
                    at(data, ROLE)['Pipe'] = [{'for(T=0:2)': ['compute:y(T + t)']}]
                elif kind == 'event':
                    at(data, ROLE)['Pipe'] = ['compute:y(0)']
                    at(data, PIPE)['Event'] = {'finished(T)': ['{done MAIN[0]::compute:y(T + t)}']}
                elif kind == 'waiting':
                    at(data, ROLE)['Pipe'] = [{'for(T=0:2)': ['compute:y(T)']}]
                    at(data, ROLE)['Waiting'] = {'Allowed': [
                        '{done MAIN[t]::compute:y(T)} ← [{issue MAIN[t]::compute:y(T)}]']}
                else:
                    at(data, PIPE)['Storage'] = {'buffer': {
                        'meaning': 'Slots', 'depth': 16, 'location': 'register', 'shape': [],
                        'dtype': 'fp32', 'total_size': 64, 'view': {'y(T)': {'slot_index': 'T + t'}}}}
                child = add_child(data, source='root')
                child['META_DATA']['describe'] = child['META_DATA'].pop('Describe') + '.'
                self.write_task(data)
                before = self.snapshot()
                with self.assertRaisesRegex(schema.SchemaError, 'capture outer name'):
                    schema.normalize_directory(self.directory)
                for args in ((), ('--read-only', '--graph', 'check')):
                    code, out, error = self.cli(*args)
                    self.assertEqual(code, 1, error)
                    self.assertIn('capture outer name', error)
                    self.assertNotIn('Schema checked:', out)
                    self.assertNotIn('warning:', error)
                    self.assertEqual(before, self.snapshot())

    def test_safe_local_renames_preserve_exact_shadowing_and_outer_scopes(self):
        data = deepcopy(self.base)
        algorithm = at(data, ALGO)
        algorithm['Input']['t'] = algorithm['Input'].pop('x')
        algorithm['Compute'][0]['Math'] = 'y = t\n'
        invariant = at(data, STEP)['Decomposition']['Worktile']['Invariant']
        invariant['t'] = invariant.pop('x')
        at(data, MATH)['Rewrite'][0]['Math'] = (
            'F(T) = T + 1\nSame(t) = t + 1\nSelf(self) = self + 1\n'
            'y = F(0) + Same(t) + t\ny = α + sum(T for T in t) + t\n'
            'y = sum(T for T in t) ⊙ t\ny = sum(\n  T\n  for T in t\n) ⊙ t\n')
        at(data, ROLE)['Pipe'] = [{'for(T=0:t)': ['compute:y(T)']}, {'for(t=2:4)': ['compute:y(t)']}]
        at(data, PIPE)['Event'] = {'finished(T)': ['{done MAIN[0]::compute:y(T)}']}
        at(data, PIPE)['Storage'] = {'buffer': {
            'meaning': 'Slots', 'depth': 4, 'location': 'register', 'shape': [], 'dtype': 'fp32',
            'total_size': 16, 'view': {'y(T)': {'slot_index': 'T'}}}}
        self.write_task(data)
        code, _, error = self.cli()
        self.assertEqual(code, 0, error)
        normalized = schema.validate_directory(self.directory)
        self.assertEqual(at(normalized, MATH)['Rewrite'][0]['Math'],
                         'f(t) = t + 1\nsame(t) = t + 1\nself(self) = self + 1\n'
                         'y = f(0) + same(t) + t\ny = α + sum(t for t in t) + t\n'
                         'y = sum(t for t in t) ⊙ t\ny = sum(\n  t\n  for t in t\n) ⊙ t\n')
        self.assertEqual(at(normalized, ROLE)['Pipe'],
                         [{'for(t=0:t)': ['compute:y(t)']}, {'for(t=2:4)': ['compute:y(t)']}])
        self.assertEqual(at(normalized, PIPE)['Event'], {'finished(t)': ['{done MAIN[0]::compute:y(t)}']})
        self.assertEqual(at(normalized, PIPE)['Storage']['buffer']['view'], {'y(t)': {'slot_index': 't'}})
        before = self.snapshot()
        code, _, error = self.cli('--read-only')
        self.assertEqual(code, 0, error)
        self.assertNotIn('warning:', error)
        self.assertEqual(before, self.snapshot())

    def test_open_quantifiers_preserve_text_and_require_evidence_for_renames(self):
        data = deepcopy(self.base)
        at(data, ALGO)['Constant'] = {'N': {'meaning': 'Fixed scale', 'value': 2}}
        rewrite = at(data, MATH)['Rewrite'][0]
        formula = 'y = sum(N * X ⊙ x for x in [0])\ny = f("sum(T for T in t)") ⊙ x\n'
        rewrite['Math'] = formula
        self.write_task(data)
        plan = schema.normalize_directory(self.directory)
        self.assertEqual(plan.documents['root'].value['DESIGN']['Math_reshape']['Rewrite'][0]['Math'],
                         formula.replace('N *', 'n *'))
        schema.validate_documents(plan.files, plan.documents)
        self.assertFalse(normalization.normalize_documents(plan.files, plan.documents).candidates)
        rewrite['Math'] = 'y = sum(X ⊙ x for X in [0])\n'
        add_child(data, source='root')
        self.write_task(data)
        before = self.snapshot()
        for args in ((), ('--read-only',)):
            code, out, error = self.cli(*args)
            self.assertEqual(code, 1, error)
            self.assertIn('cannot uniquely bind an open mathematical quantifier', error)
            self.assertNotIn('Schema checked:', out)
            self.assertNotIn('warning: applied', error)
            self.assertEqual(before, self.snapshot())

    def test_property_references_resolve_their_owner_once_with_from_and_aliases(self):
        source = self.source.replace('Input: {x:', 'Input: {X:').replace('Invariant: {x:', 'Invariant: {X:')
        source = source.replace('y = x', 'y = X').replace('  Version: 1\n',
                                  '  Version: 1\n  Dispatch: {when: All scalar inputs}\n')
        source = source.replace('    Rewrite:\n      - Name: Copy\n',
                                '    Rewrite:\n      - Name: Copy\n        Symbol:\n'
                                '          z: {meaning: Copied view, shape: &shape_ref ref:X.SHAPE, dtype: fp32}\n'
                                '          z2: {meaning: Second view, shape: *shape_ref, dtype: fp32}\n')
        data = {'root': yaml_source.loads(source, 'root').value}
        child = add_child(data, source='root')
        child['REALIZATION'] = {'Require': [{'Item': 'X.SHAPE', 'Choice': 'Keep the scalar shape',
                                            'Reason': 'Preserve the input interface'}]}
        self.write_task(data)
        (self.directory / 'root.yaml').write_text(source, encoding='utf-8')
        before = self.snapshot()
        plan = schema.normalize_directory(self.directory)
        self.assertEqual(before, self.snapshot())
        symbols = plan.documents['root'].value['DESIGN']['Math_reshape']['Rewrite'][0]['Symbol']
        self.assertEqual([symbol['shape'] for symbol in symbols.values()], ['ref:x.shape', 'ref:x.shape'])
        self.assertEqual([use.alias for use in plan.documents['root'].occurrences if use.alias], ['shape_ref'])
        self.assertEqual(plan.documents['branch'].value['REALIZATION']['Require'][0]['Item'], 'x.shape')
        schema.validate_documents(plan.files, plan.documents)
        self.assertFalse(normalization.normalize_documents(plan.files, plan.documents).candidates)
        code, _, error = self.cli()
        self.assertEqual(code, 0, error)
        self.assertIn('warning: applied declarations and references', error)
        before = self.snapshot()
        for args in ((), ('--read-only',)):
            code, _, error = self.cli(*args)
            self.assertEqual(code, 0, error)
            self.assertNotIn('warning:', error)
            self.assertEqual(before, self.snapshot())

    def test_ambiguous_multiline_formals_cannot_publish_partial_renames(self):
        for body in ('T\n  T', 'T +\n  T\n  T', '(T +\n  T)', 'T +\ny = T',
                     'T\n  Q for Q in x\n  T'):
            with self.subTest(body=body):
                data = deepcopy(self.base)
                at(data, MATH)['Rewrite'][0]['Math'] = f'F(T) = {body}\ny = F(x)\n'
                add_child(data, source='root')
                self.write_task(data)
                before = self.snapshot()
                code, _, error = self.cli()
                self.assertEqual(code, 1, error)
                self.assertIn('multiline formal definition', error)
                self.assertIn('.DESIGN.Math_reshape.Rewrite[0].Math', error)
                self.assertNotIn('warning: applied', error)
                self.assertEqual(before, self.snapshot())

    def test_bound_attribute_roots_follow_declarations_and_formals(self):
        data = deepcopy(self.base)
        algorithm = at(data, ALGO)
        algorithm['Input']['X'] = algorithm['Input'].pop('x')
        invariant = at(data, STEP)['Decomposition']['Worktile']['Invariant']
        invariant['X'] = invariant.pop('x')
        algorithm['Compute'][0]['Math'] = 'y = X.item() + Torch.Op(X)\n'
        at(data, MATH)['Rewrite'][0]['Math'] = (
            'F(T) = T . Item() + X.Value + X.item(). X + Torch . X(T, "X.item() T")\n'
            'y = F(X) + Torch.Op(X)\n')
        self.write_task(data)
        code, _, error = self.cli()
        self.assertEqual(code, 0, error)
        self.assertIn('warning: applied declarations and references', error)
        value = schema.validate_directory(self.directory)
        self.assertEqual(set(at(value, ALGO)['Input']), {'x'})
        self.assertEqual(at(value, ALGO)['Compute'][0]['Math'], 'y = x.item() + Torch.Op(x)\n')
        self.assertEqual(at(value, MATH)['Rewrite'][0]['Math'],
                         'f(t) = t . Item() + x.Value + x.item(). X + Torch . X(t, "X.item() T")\n'
                         'y = f(x) + Torch.Op(x)\n')

    def test_skip_hardware_preserves_its_unfinished_values_through_cli_normalization(self):
        data = deepcopy(self.base)
        hardware = {'Device': True, 'Architecture': 17, 'Execution': {'CTA': 'Known block'}}
        data['root']['CONTRACT']['Hardware_specification'] = hardware
        self.write_task(data)
        code, _, error = self.cli('--skip-hardware')
        self.assertEqual(code, 0, error)
        self.assertIn('Hardware skipped explicitly', error)
        after = yaml_source.load(self.directory / 'root.yaml', 'root').value
        self.assertEqual(after['CONTRACT']['Hardware_specification'], hardware)

    def test_shared_aliases_and_precise_values_survive_the_complete_plan(self):
        source = self.source.replace('Describe: Copy one scalar in a synthetic validation task',
                                     'describe: &copy Copy a scalar.')
        source = source.replace('Describe: Produce the scalar y', 'Describe: *copy')
        source = source.replace('  Algorithm_requirement:\n',
                                '  Algorithm_requirement:\n    Constant: {epsilon: {meaning: Exact constant, value: 1.00000000000000000001}}\n')
        document = yaml_source.loads(source, 'root')
        original = deepcopy(document.value)
        plan = normalization.normalize_documents({'root': self.directory / 'root.yaml'}, {'root': document})
        candidate = plan.documents['root']
        self.assertEqual(document.value, original)
        self.assertEqual(document.source, source)
        self.assertEqual(set(candidate.anchors), {'copy'})
        self.assertEqual([use.alias for use in candidate.occurrences if use.alias], ['copy'])
        self.assertEqual(candidate.value['META_DATA']['Describe'], 'Copy a scalar')
        self.assertEqual(candidate.value['DESIGN']['Work_schedule']['Step']['MAIN']['Describe'], 'Copy a scalar')
        self.assertEqual(candidate.value['CONTRACT']['Algorithm_requirement']['Constant']['epsilon']['value'],
                         Decimal('1.00000000000000000001'))
        self.assertIn('1.00000000000000000001', candidate.source)
        schema.validate_documents(plan.files, plan.documents)
        mixed = normalization.loads_for_normalization(
            'META_DATA: {Name: &number 001}\n'
            'CONTRACT: {Algorithm_requirement: {Constant: {fixed: {value: *number, meaning: Exact integer}}}}\n',
            'root')
        self.assertIs(type(mixed.document.value['META_DATA']['Name']), int)
        self.assertEqual(mixed.document.value['CONTRACT']['Algorithm_requirement']['Constant']['fixed']['value'], 1)
        self.assertFalse(mixed.corrections)
        source = self.source.replace(
            'Input: {x: {meaning: Input scalar, shape: [], dtype: fp32}}',
            'Input: {x: {meaning: Input scalar, &property SHAPE: [], dtype: fp32}}\n'
            '    Constant: {*property: {value: 1, meaning: Dimension}}')
        document = yaml_source.loads(source, 'root')
        plan = normalization.normalize_documents({'root': self.directory / 'root.yaml'}, {'root': document})
        candidate = plan.documents['root']
        algorithm = candidate.value['CONTRACT']['Algorithm_requirement']
        self.assertIn('shape', algorithm['Input']['x'])
        self.assertEqual(algorithm['Constant']['shape']['value'], 1)
        self.assertEqual([use.alias for use in candidate.occurrences if use.alias], ['property'])
        self.assertIn('SHAPE', document.value['CONTRACT']['Algorithm_requirement']['Input']['x'])
        self.assertFalse(normalization.normalize_documents(plan.files, plan.documents).candidates)

    def test_write_failure_restores_the_task_including_original_mtimes(self):
        data = deepcopy(self.base)
        add_child(data, source='root')
        self.write_task(data)
        before = self.snapshot()
        replace = Path.replace
        publications = 0

        def fail_second(path, destination):
            nonlocal publications
            if path.name.startswith('.schema-normalize-'):
                publications += 1
                if publications == 2:
                    raise OSError('synthetic replacement failure')
            return replace(path, destination)

        with patch('pathlib.Path.replace', new=fail_second):
            code, _, error = self.cli()
        self.assertEqual(code, 2, error)
        self.assertIn('synthetic replacement failure', error)
        self.assertNotIn('warning: applied', error)
        self.assertEqual(before, self.snapshot())
        self.assertFalse(any(path.name.startswith('.schema-normalize-') for path in self.directory.iterdir()))

    def test_post_write_read_failures_restore_original_unrepaired_bytes(self):
        for failure in (schema.ToolError, schema.SchemaError):
            with self.subTest(failure=failure.__name__):
                data = deepcopy(self.base)
                add_child(data, source='root')
                self.write_task(data)
                source = self.source.replace('  Version: 1\n', '  Version: "1"\n  Dispatch: {when: All scalar inputs}\n')
                source = source.replace('Name: scalar_copy', 'Name: 0x10')
                source = source.replace('Describe: Copy one scalar in a synthetic validation task',
                                        'Describe: Copy: one scalar')
                (self.directory / 'root.yaml').write_bytes(source.replace('\n', '\r\n').encode())
                before = self.snapshot()
                with patch.object(schema, 'read_directory', side_effect=failure('synthetic post-write read failure')):
                    code, out, error = self.cli()
                self.assertEqual(code, 2, error)
                self.assertIn('synthetic post-write read failure', error)
                self.assertIn('restored', error)
                self.assertNotIn('warning: applied', error)
                self.assertNotIn('Normalization completed', out)
                self.assertEqual(before, self.snapshot())

    def test_post_write_document_hash_mismatch_restores_every_written_file(self):
        data = deepcopy(self.base)
        add_child(data, source='root')
        self.write_task(data)
        before = self.snapshot()
        read = schema.read_directory

        def mismatched(directory):
            files, documents = read(directory)
            documents['root'].source_sha256 = 'synthetic mismatched hash'
            return files, documents

        with patch.object(schema, 'read_directory', side_effect=mismatched):
            code, out, error = self.cli()
        self.assertEqual(code, 2, error)
        self.assertIn('task source changed while rereading', error)
        self.assertIn('restored', error)
        self.assertNotIn('warning: applied', error)
        self.assertNotIn('Normalization completed', out)
        self.assertEqual(before, self.snapshot())

    def test_post_write_recovery_preserves_external_edits_before_or_after_read(self):
        read = schema.read_directory
        for moment in ('before_read', 'after_read'):
            with self.subTest(moment=moment):
                data = deepcopy(self.base)
                add_child(data, source='root')
                self.write_task(data)
                before = self.snapshot()
                external = None

                def drift(directory):
                    nonlocal external
                    documents = read(directory) if moment == 'after_read' else None
                    path = self.directory / 'root.yaml'
                    path.write_bytes(b'META_DATA: [external unfinished edit\n')
                    external = self.snapshot()['root.yaml']
                    return documents if documents is not None else read(directory)

                with patch.object(schema, 'read_directory', side_effect=drift):
                    code, out, error = self.cli()
                self.assertEqual(code, 2, error)
                self.assertIn('could not be restored safely', error)
                self.assertIn(str(self.directory / 'root.yaml'), error)
                self.assertNotIn('warning: applied', error)
                self.assertNotIn('Normalization completed', out)
                before['root.yaml'] = external
                self.assertEqual(before, self.snapshot())
