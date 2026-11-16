"""Exact source parsing, reversible formatting, and path-specific text budgets."""
from copy import deepcopy
from decimal import Decimal
import hashlib
from unittest.mock import patch

from .. import schema, text_policy, yaml_format, yaml_source
from ..diagnostics import encode_value
from ..errors import CapacityError
from .support import ALGO, PIPE, STEP, TaskCase, at


class YamlTests(TaskCase):
    def test_memory_and_file_readers_keep_exact_values_marks_and_hashes(self):
        source = ('base: &base {flag: on, date: 2026-11-02, count: 080}\r\n'
                  'copied: {<<: *base, explicit: !!str 1e3, exact: 1.000000000000000000000001}\r\n')
        path = self.directory / 'root.yaml'
        path.write_bytes(source.encode('utf-8'))
        document = yaml_source.load(path, 'root')
        memory = yaml_source.loads(source, 'root', path=path)
        self.assertEqual(document.value, memory.value)
        self.assertEqual(document.positions, memory.positions)
        self.assertEqual(document.source_sha256, hashlib.sha256(source.encode('utf-8')).hexdigest())
        values = document.value['copied']
        self.assertEqual((values['flag'], values['date'], values['count'], values['explicit']), ('on', '2026-11-02', 80, '1e3'))
        self.assertEqual(values['exact'], Decimal('1.000000000000000000000001'))
        self.assertEqual(document.source, source)
        self.assertEqual(set(document.anchors), {'base'})

    def test_invalid_or_recursive_yaml_is_distinct_from_capacity(self):
        for source, message in [('x: 1\nx: 2\n', 'duplicate mapping key'),
                                ('x: &loop {again: *loop}', 'recursive aliases'),
                                ('base: &base {x: 1}\nmerged: {<<: !bad [*base]}', 'tags'),
                                ('base: &base {x: 1}\nmerged: {<<: !!map [*base]}', 'tags'),
                                ('x: 1\n---\nx: 2', 'exactly one YAML mapping'),
                                ('x: !!python/object/apply:os.system [touch forbidden]', 'ordinary'),
                                ('x: ' + chr(0xD800), 'UTF-8')]:
            with self.subTest(source=source), self.assertRaisesRegex(yaml_source.ReadError, message):
                yaml_source.loads(source, 'root')
        with patch.object(yaml_source, 'MAX_EXPANDED_NODES', 5), self.assertRaises(CapacityError):
            yaml_source.loads('x: [1, 2, 3, 4, 5, 6]', 'root')

    def test_format_keeps_decimal_alias_merge_and_scalar_types(self):
        source = ('first: &first {a: 1, text: "true", exact: 0.12345678901234567890123456789}\n'
                  'second: &second {a: 2, extra: false}\n'
                  'merged: {<<: [*first, *second], text: "null"}\n'
                  'reused: *first\n'
                  'multiline: |+\n  alpha\n\n  beta\n\n')
        document = yaml_source.loads(source, 'root')
        before = deepcopy(document.value)
        content = yaml_format.format_document(document)
        after = yaml_source.loads(content, 'root')
        self.assertEqual(after.value, before)
        self.assertEqual(document.value, before)
        self.assertEqual(set(after.anchors), {'first', 'second'})
        self.assertEqual([use.alias for use in after.occurrences if use.alias], ['first', 'second', 'first'])
        self.assertIn('<<:', content)
        self.assertEqual(content, yaml_format.format_document(after))
        self.assertIsInstance(after.value['first']['exact'], Decimal)
        self.assertIsInstance(after.value['first']['text'], str)
        self.assertIs(after.value['second']['extra'], False)

    def test_current_layout_compacts_entries_and_keeps_structures_and_math(self):
        document = yaml_source.loads(encode_value(self.base['root']), 'root')
        content = yaml_format.format_document(document)
        self.assertIn('CONTRACT:\n', content)
        self.assertIn('DESIGN:\n', content)
        self.assertIn('Pipe:\n', content)
        self.assertRegex(content, r'(?m)^\s*Input:.*x:')
        self.assertRegex(content, r'(?m)^\s*Output:.*y:')
        self.assertEqual(yaml_source.loads(content, 'root').value, self.base['root'])
        self.assertEqual(yaml_source.loads(content, 'root').value['DESIGN']['Math_reshape']['Rewrite'][0]['Math'], 'y = x\n')
        self.assertEqual(content, yaml_format.format_document(yaml_source.loads(content, 'root')))

    def test_storage_entries_use_two_lines_with_name_meaning_and_all_views(self):
        data = deepcopy(self.base)
        entries = at(data, PIPE)['Storage'] = {
            'scalar': {'total_size': 4, 'dtype': 'fp32', 'shape': [], 'location': 'register',
                       'depth': 1, 'meaning': 'One scalar'},
            'ring': {'view': {'x(i)': {'slot_index': 'i%2'}}, 'total_size': 8, 'dtype': 'fp32',
                     'shape': [], 'location': 'register', 'depth': 2, 'meaning': 'Two scalar slots'},
            'shared': {'view': {'x': {}, 'y(i)': {'region': {'column': [0, 32]}, 'slot_index': 'i%2'}},
                       'member': [0], 'total_size': 'size', 'dtype': 'fp32',
                       'shape': [{'row': 'rows'}, {'column': 'cols'}], 'location': 'smem',
                       'depth': 2, 'meaning': 'Shared value slots'},
        }
        document = yaml_source.loads(encode_value(data['root']), 'root')
        content = yaml_format.format_document(document)
        after = yaml_source.loads(content, 'root')
        self.assertEqual(after.value, document.value)
        self.assertEqual(document.value, data['root'])
        self.assertIn('Storage:\n', content)
        self.assertIn('scalar: {meaning: One scalar, depth: 1, location: register, shape: [],\n'
                      '            dtype: fp32, total_size: 4}', content)
        for name in entries:
            use = next(use for use in after.occurrences if not use.key and use.parts == PIPE[1:] + ('Storage', name))
            self.assertEqual(use.node.end_mark.line - use.node.start_mark.line + 1, 2)
            lines = content.splitlines()[use.line - 1:use.line + 1]
            self.assertTrue(lines[0].startswith('          ' + name + ': {meaning:'))
            self.assertTrue(lines[1].startswith('            '))
            self.assertTrue(lines[1].lstrip().startswith('dtype:'))
            self.assertTrue(lines[1].endswith('}'))
            if 'view' in entries[name]:
                self.assertIn('view:', lines[1])
        self.assertEqual(content, yaml_format.format_document(after))
        for name, value in entries.items():
            entries[name] = dict(reversed(tuple(value.items())))
        self.assertEqual(content, yaml_format.format_document(yaml_source.loads(encode_value(data['root']), 'root')))

    def test_storage_quotes_long_text_without_wrapping_or_shortening(self):
        meaning = 'Buffer: keep every value, bracket [x], and "quoted word"; ' * 8
        for name in ('buffer', 'buffer_' + 'x' * 160):
            with self.subTest(name=name):
                source = encode_value({'DESIGN': {'Tile_pipeline': {'Step': {'MAIN': {'Storage': {
                    name: {'meaning': meaning, 'depth': 1, 'location': 'register', 'shape': [],
                           'dtype': 'fp32', 'total_size': 4, 'view': {'x': {}}}}}}}}})
                document = yaml_source.loads(source, 'root')
                warnings = []
                content = yaml_format.format_document(document, corrections=warnings)
                after = yaml_source.loads(content, 'root')
                use = next(use for use in after.occurrences if not use.key and use.parts == PIPE[1:] + ('Storage', name))
                lines = content.splitlines()[use.line - 1:use.line + 1]
                self.assertEqual(use.node.end_mark.line - use.node.start_mark.line + 1, 2)
                self.assertIn(name + ": {meaning: '", lines[0])
                self.assertTrue(lines[1].lstrip().startswith('dtype: fp32,'))
                self.assertEqual(after.value, document.value)
                self.assertEqual(after.value['DESIGN']['Tile_pipeline']['Step']['MAIN']['Storage'][name]['meaning'], meaning)
                self.assertEqual(content, yaml_format.format_document(after))
                self.assertTrue(any(item.message.startswith('layout:') for item in warnings))

    def test_storage_fixed_order_is_not_disabled_by_independent_anchors(self):
        source = ('unrelated: &other ignored\n'
                  'DESIGN:\n  Tile_pipeline:\n    Step:\n      MAIN:\n        Storage:\n'
                  '          buffer:\n            view: {x(i): {slot_index: &slot i%2}}\n'
                  '            dtype: fp32\n            meaning: Two scalar slots\n'
                  '            depth: &depth 2\n            location: register\n'
                  '            shape: []\n            total_size: *depth\n'
                  'copy: [*other, *slot]\n')
        document = yaml_source.loads(source, 'root')
        content = yaml_format.format_document(document)
        after = yaml_source.loads(content, 'root')
        self.assertIn('buffer: {meaning: Two scalar slots, depth: &depth 2, location: register, shape: [],\n'
                      '            dtype: fp32, total_size: *depth, view: {x(i): {slot_index: &slot i%2}}}', content)
        self.assertEqual(after.value, document.value)
        self.assertEqual(sorted((use.path, use.key, use.alias) for use in after.occurrences if use.alias),
                         sorted((use.path, use.key, use.alias) for use in document.occurrences if use.alias))
        self.assertEqual(document.source, source)
        self.assertEqual(content, yaml_format.format_document(after))

    def test_storage_views_with_single_logical_line_block_scalars_use_two_lines(self):
        for style in ('>-', '|-'):
            with self.subTest(style=style):
                source = ('DESIGN:\n  Tile_pipeline:\n    Step:\n      MAIN:\n        Storage:\n'
                          '          buffer:\n            meaning: Two scalar slots\n'
                          '            depth: 2\n            location: register\n            shape: []\n'
                          '            dtype: fp32\n            total_size: 8\n            view:\n'
                          '              x(i):\n                slot_index: ' + style + '\n                  i%2\n')
                document = yaml_source.loads(source, 'root')
                content = yaml_format.format_document(document)
                after = yaml_source.loads(content, 'root')
                self.assertIn('buffer: {meaning: Two scalar slots, depth: 2, location: register, shape: [],\n'
                              '            dtype: fp32, total_size: 8, view: {x(i): {slot_index: i%2}}}', content)
                self.assertEqual(after.value, document.value)
                self.assertEqual(document.source, source)
                self.assertEqual(content, yaml_format.format_document(after))

    def test_storage_layout_preserves_tagged_aliases_merges_and_decimal_lexemes(self):
        source = ('exact: &exact !!float 1.2300000000000000000001\ncopy: *exact\n'
                  'defaults: &defaults !!map {depth: 02, location: register, shape: [], dtype: !!str fp32, total_size: 08}\n'
                  'DESIGN:\n  Tile_pipeline:\n    Step:\n      MAIN:\n        Storage:\n'
                  '          ring: &ring {meaning: Two scalar slots, <<: *defaults, '
                  'view: &views {x(i): {slot_index: i%2}}}\n'
                  '          alias: *ring\n'
                  '          merged: {meaning: Shared scalar slots, <<: *defaults, '
                  'view: {<<: *views, y(i): {slot_index: i%2}}}\n')
        document = yaml_source.loads(source, 'root')
        original = deepcopy(document.value)
        content = yaml_format.format_document(document)
        after = yaml_source.loads(content, 'root')
        self.assertEqual(document.value, original)
        self.assertEqual(document.source, source)
        self.assertEqual(after.value, original)
        self.assertEqual(after.anchors.keys(), document.anchors.keys())
        self.assertEqual(sorted((use.path, use.key, use.alias) for use in after.occurrences if use.alias),
                         sorted((use.path, use.key, use.alias) for use in document.occurrences if use.alias))
        self.assertEqual(after.anchors['exact'].value, '1.2300000000000000000001')
        self.assertEqual(after.anchors['exact'].explicit_tag, 'tag:yaml.org,2002:float')
        self.assertIsInstance(after.value['exact'], Decimal)
        self.assertIn('depth: 02', content)
        self.assertIn('total_size: 08', content)
        self.assertIn('dtype: !!str fp32', content)
        self.assertIn('alias: *ring', content)
        for name in ('ring', 'merged'):
            use = next(use for use in after.occurrences if not use.key and use.parts == PIPE[1:] + ('Storage', name))
            self.assertEqual(use.node.end_mark.line - use.node.start_mark.line + 1, 2)
        self.assertEqual(content, yaml_format.format_document(after))

    def test_storage_keeps_real_block_text_and_unlisted_structures_expanded(self):
        source = ('DESIGN:\n  Tile_pipeline:\n    Step:\n      MAIN:\n        Storage:\n'
                  '          buffer:\n            meaning: |\n              First line\n              Second line\n'
                  '            depth: 1\n            location: register\n            shape: []\n'
                  '            dtype: fp32\n            total_size: 4\n            view: {x: {}}\n'
                  '          extra: {meaning: Unlisted structure, view: {x: {unlisted: {deep: {value: 1}}}}}\n'
                  'other: {buffer: {meaning: Other context, depth: 1, location: register, shape: [], dtype: fp32, total_size: 4}}\n')
        document = yaml_source.loads(source, 'root')
        content = yaml_format.format_document(document)
        after = yaml_source.loads(content, 'root')
        self.assertEqual(after.value, document.value)
        self.assertIn('buffer:\n            meaning: |\n              First line\n              Second line\n', content)
        self.assertRegex(content, r'unlisted:\n\s+deep:\n\s+value: 1')
        self.assertIn('other:\n  buffer:\n    meaning: Other context\n', content)
        self.assertEqual(content, yaml_format.format_document(after))
        source = ('DESIGN:\n  Tile_pipeline:\n    Step:\n      MAIN:\n        Storage:\n'
                  '          buffer:\n            view: {x: {slot_index: &slot 0}}\n'
                  '            meaning: One scalar\n            depth: 1\n            location: register\n'
                  '            shape: []\n            dtype: fp32\n            total_size: 4\n            member: [*slot]\n')
        document = yaml_source.loads(source, 'root')
        content = yaml_format.format_document(document)
        after = yaml_source.loads(content, 'root')
        self.assertIn('buffer:\n            view:', content)
        self.assertLess(content.index('&slot'), content.index('*slot'))
        self.assertEqual(after.value, document.value)
        self.assertEqual([use.alias for use in after.occurrences if use.alias], ['slot'])
        self.assertEqual(content, yaml_format.format_document(after))

    def test_schedule_records_keep_order_types_and_flow_quotes(self):
        data = deepcopy(self.base)
        step = at(data, STEP)
        step['Mapping'] = {'Strategy': 'Pair workers within one cluster',
                           'Worker': {'unit_count': 2, 'unit': 'cta'},
                           'Placement': {'unit_count': 1, 'unit': 'cluster'},
                           'Worker_count': '2*sm_count',
                           'Member': [{'row': [0, 16]}, {'row': [16, 32]}]}
        order = {'direction': 'ascending', 'group': 4, 'by': ['row', 'column']}
        refill = 'Fetch current item, then advance by one'
        for multiple in (False, True):
            with self.subTest(multiple=multiple):
                assignment = {'Preassigned': {'count' if multiple else 'stride': 2, 'start': 0},
                              'Refill': {'atomic_counter': refill}, 'Exit': ['assigned_done', 'counter_empty']}
                step['Schedule'] = {'Worktile_order': [order], 'Worker_assignment': assignment}
                if multiple:
                    step['Schedule']['Worktile_order'].append({'by': 'row', 'direction': 'descending'})
                    assignment['Refill']['queue'] = 'Claim remaining queue entries'
                document = yaml_source.loads(encode_value(data['root']), 'root')
                content = yaml_format.format_document(document)
                after = yaml_source.loads(content, 'root')
                self.assertEqual(after.value, data['root'])
                self.assertEqual(content, yaml_format.format_document(after))
                self.assertIn('Mapping:\n', content)
                self.assertIn('Worker: {unit: cta, unit_count: 2}', content)
                self.assertIn('Placement: {unit: cluster, unit_count: 1}', content)
                self.assertIn('Worker_count: 2*sm_count', content)
                self.assertIn('Member: [{row: [0, 16]}, {row: [16, 32]}]', content)
                self.assertIn('Preassigned: {start: 0, ' + ('count' if multiple else 'stride') + ': 2}', content)
                self.assertIn('Exit: [assigned_done, counter_empty]', content)
                if multiple:
                    self.assertRegex(content, r'Worktile_order:\n\s+- \{by: \[row, column\], group: 4, direction: ascending\}\n'
                                              r'\s+- \{by: row, direction: descending\}')
                    self.assertRegex(content, r'Refill:\n\s+atomic_counter: Fetch current item, then advance by one\n'
                                              r'\s+queue: Claim remaining queue entries')
                else:
                    self.assertIn('Worktile_order: [{by: [row, column], group: 4, direction: ascending}]', content)
                    self.assertIn("Refill: {atomic_counter: '" + refill + "'}", content)

    def test_folded_schedule_scalars_preserve_decoded_newlines(self):
        source = ('DESIGN:\n  Work_schedule:\n    Step:\n      MAIN:\n        Mapping:\n'
                  '          Worker:\n            unit: >-\n              cta\n            unit_count: 2\n'
                  '        Schedule:\n'
                  '          Worker_assignment:\n            Refill:\n              counter: >-\n'
                  '                Fetch one item,\n                then advance\n')
        for multiple in (False, True):
            with self.subTest(multiple=multiple):
                tail = '              queue: |\n                Claim one item\n' if multiple else ''
                tail += '            Exit:\n              - |-\n                assigned_done\n'
                document = yaml_source.loads(source + tail, 'root')
                content = yaml_format.format_document(document)
                after = yaml_source.loads(content, 'root')
                self.assertEqual(after.value, document.value)
                self.assertEqual(content, yaml_format.format_document(after))
                self.assertIn('Worker: {unit: cta, unit_count: 2}', content)
                self.assertIn('Exit: [assigned_done]', content)
                if multiple:
                    self.assertRegex(content, r'Refill:\n\s+counter: Fetch one item, then advance\n'
                                              r'\s+queue: "Claim one item\\n"')
                else:
                    self.assertIn("Refill: {counter: 'Fetch one item, then advance'}", content)

    def test_exit_layout_preserves_short_lists_long_lists_and_two_line_scalars(self):
        two_lines = 'a' * 149 + '\n' + 'b' * 150
        cases = [(['assigned_done', 'counter_empty', '界' * 50], True),
                 (['a', 'b', 'c', 'd'], False), (['界' * 51], False),
                 (['ready\nfinished'], False), (['ready\vfinished'], False), (two_lines, False)]
        for value, inline in cases:
            with self.subTest(value=value):
                data = deepcopy(self.base)
                at(data, STEP)['Schedule']['Worker_assignment']['Exit'] = value
                document = yaml_source.loads(encode_value(data['root']), 'root')
                content = yaml_format.format_document(document)
                after = yaml_source.loads(content, 'root')
                self.assertEqual(after.value, document.value)
                self.assertEqual(content, yaml_format.format_document(after))
                self.assertEqual('Exit: [' in content, inline)
                if isinstance(value, list) and not inline:
                    self.assertRegex(content, r'Exit:\n\s+- ')
                elif isinstance(value, str):
                    self.assertIn('Exit: |-\n', content)

    def test_quotes_are_context_sensitive_for_keys_and_values(self):
        document = yaml_source.loads('META_DATA: {Name: "plain", Describe: "true"}\n'
                                     'outer:\n  "safe": "a,b"\n'
                                     'flow: {"comma,key": "a,b"}\n', 'root')
        warnings = [d for d in text_policy.check_document(document) if d.severity == 'warning']
        paths = {d.path for d in warnings}
        self.assertIn('root.META_DATA.Name', paths)
        self.assertIn('root.outer.safe', paths)
        self.assertNotIn('root.META_DATA.Describe', paths)
        self.assertNotIn('root.flow["comma,key"]', paths)
        content = yaml_format.format_document(document)
        self.assertEqual(yaml_source.loads(content, 'root').value, document.value)
        self.assertNotIn('"plain"', content)
        self.assertIn('true', content)

    def test_strict_reader_reports_missing_quotes_and_cli_recovers_string_lexemes(self):
        with self.assertRaisesRegex(yaml_source.ReadError, 'quotes'):
            yaml_source.loads('META_DATA:\n  Describe: first: second\n', 'root', path='root.yaml')
        data = deepcopy(self.base)
        data['root']['META_DATA']['Describe'] = True
        self.write_task(data)
        code, _, error = self.cli()
        self.assertEqual(code, 0, error)
        self.assertIn('warning: applied YAML scalar quoting', error)
        self.assertEqual(yaml_source.load(self.directory / 'root.yaml', 'root').value['META_DATA']['Describe'], 'true')

    def test_real_comments_are_preserved_by_refusing_to_format(self):
        document = yaml_source.loads('META_DATA:\n  Describe: hello # explanation\n', 'root')
        self.assertEqual(len(document.comments), 1)
        with self.assertRaisesRegex(schema.SchemaError, 'comments'):
            yaml_format.format_document(document)
        literal = yaml_source.loads('META_DATA: {Describe: "a # b"}\nmath: |\n  x # literal\n', 'root')
        self.assertFalse(literal.comments)
        self.assertEqual(yaml_source.loads(yaml_format.format_document(literal), 'root').value, literal.value)

    def test_unicode_and_logical_lines_use_decoded_text_without_trimming(self):
        value = {'META_DATA': {'Describe': '界' * 149 + '😀'}}
        self.assertFalse(text_policy.check_values(value))
        value['META_DATA']['Describe'] += 'a'
        diagnostics = text_policy.check_values(value)
        self.assertIn('151 Unicode characters', diagnostics[0].message)
        value['META_DATA']['Describe'] = 'a\n\nb\n'
        self.assertIn('3 logical lines', text_policy.check_values(value)[0].message)
        value['META_DATA']['Describe'] = 'a\n'
        self.assertFalse(text_policy.check_values(value))
        value['META_DATA']['Describe'] = ' ' * 150 + 'a'
        self.assertTrue(text_policy.check_values(value))

    def test_range_endpoints_items_and_feedback_have_specific_budgets(self):
        algorithm = deepcopy(at(self.base, ALGO))
        algorithm['Input']['x']['range'] = ['a' * 150, 'b' * 150]
        algorithm['Input']['x']['default'] = None
        self.assertFalse(text_policy.check_values(algorithm, 'root.CONTRACT.Algorithm_requirement', scope='algorithm'))
        algorithm['Input']['x']['range'][1] += 'b'
        diagnostics = text_policy.check_values(algorithm, 'root.CONTRACT.Algorithm_requirement', scope='algorithm')
        self.assertEqual([d.path for d in diagnostics], ['root.CONTRACT.Algorithm_requirement.Input.x.range[1]'])
        value = {'REALIZATION': {'Require': [{'Item': ['x' * 300, 'y' * 300], 'Choice': 'z' * 300}],
                                 'Feedback': [{'Item': 'x' * 300, 'Choice': 'z' * 151}]},
                 'unrelated': {'name': 'z' * 1000}}
        self.assertEqual([d.path for d in text_policy.check_values(value)], ['root.REALIZATION.Feedback[0].Choice'])

    def test_warning_does_not_fail_read_only_apis_and_hardware_budget_can_be_skipped(self):
        data = deepcopy(self.base)
        data['root']['META_DATA']['Describe'] = 'Copy a synthetic scalar.'
        self.write_task(data)
        before = self.snapshot()
        notices = []
        schema.validate_directory(self.directory, notices=notices)
        self.assertTrue(any('warning:' in text for text in notices))
        self.assertEqual(before, self.snapshot())
        hardware = {'CONTRACT': {'Hardware_specification': {'Device': '界' * 51}}}
        self.assertTrue(text_policy.check_values(hardware))
        self.assertFalse(text_policy.check_values(hardware, skip_hardware=True))

    def test_sentence_period_formatting_preserves_all_other_text_and_source_document(self):
        for text in ('Copy a scalar.', '计算输入结果。', 'Copy a scalar.\n',
                     '  Copy a scalar.  \n\n', 'Use 3.14 as the threshold.'):
            with self.subTest(text=text):
                source = encode_value({'META_DATA': {'Name': 'Keep the name.', 'Describe': text},
                                       'unrelated': 'Keep this text.'})
                document = yaml_source.loads(source, 'root')
                original = deepcopy(document.value)
                nodes = [(use.node, use.node.value) for use in document.occurrences]
                warnings = []
                candidate = yaml_format.format_document(document, corrections=warnings)
                after = yaml_source.loads(candidate, 'root')
                stop = len(text.rstrip()) - 1
                expected = deepcopy(original)
                expected['META_DATA']['Describe'] = text[:stop] + text[stop + 1:]
                self.assertEqual(after.value, expected)
                self.assertEqual(document.value, original)
                self.assertEqual(document.source, source)
                self.assertTrue(all(node.value == value for node, value in nodes))
                self.assertEqual([item.path for item in warnings if item.message.startswith('period: ')],
                                 ['root.META_DATA.Describe'])
                self.assertTrue(all(item.severity == 'warning' for item in warnings))
                self.assertEqual(candidate, yaml_format.format_document(after))
        real_render = yaml_format._render

        def altered_name(document, changes=None):
            return real_render(document, changes).replace('Keep the name.', 'Keep the name')

        with patch.object(yaml_format, '_render', side_effect=altered_name):
            with self.assertRaisesRegex(schema.ToolError, 'would change a scalar'):
                yaml_format.format_document(document)

    def test_uncertain_periods_and_quoted_or_syntax_text_are_preserved(self):
        for text in ('3.14', '3.', 'matrix.shape.', 'foo.py.', '/tmp/result.', 'Read /tmp/result.',
                     'https://example.test/path.', 'Use foo.bar.', 'Use the approx.', 'Use std.',
                     'Support MPI.', 'Finished.', 'Read this etc.', "'Copy a scalar.'",
                     '“计算输入结果。”', 'Use `x`.', 'x = y.', '引用 ref: foo.', 'Copy values [1].',
                     '.', '。', '...', '普通说明。。', 'Copy a scalar..', '数据.名称。', '矩阵A⊗B。'):
            with self.subTest(text=text):
                document = yaml_source.loads(encode_value({'META_DATA': {'Describe': text}}), 'root')
                warnings = []
                candidate = yaml_format.format_document(document, corrections=warnings)
                self.assertEqual(yaml_source.loads(candidate, 'root').value, document.value)
                self.assertFalse([item for item in warnings if item.message.startswith('period: ')])

    def test_shared_sentence_period_requires_every_scalar_or_collection_use_to_be_prose(self):
        safe_sources = (
            'META_DATA: {Describe: &copy Copy a scalar.}\nDESIGN: {Work_schedule: {Thesis: *copy}}\n',
            'META_DATA: &record {Describe: Copy a scalar.}\n'
            'DESIGN: {Work_schedule: {Step: {MAIN: {<<: *record}}}}\n',
        )
        unsafe_sources = (
            'META_DATA: {Name: &copy Copy a scalar., Describe: *copy}\n',
            'META_DATA: {Describe: &copy Copy a scalar.}\n'
            'DESIGN: {Math_reshape: {Rewrite: [{Math: *copy}]}}\n',
            'META_DATA: {Describe: &copy Copy a scalar.}\nDESIGN: {Math_reshape: {from: *copy}}\n',
            'META_DATA: &record {Describe: Copy a scalar.}\nother: {<<: *record}\n',
        )
        for source in safe_sources + unsafe_sources:
            with self.subTest(source=source):
                document = yaml_source.loads(source, 'root')
                warnings = []
                candidate = yaml_format.format_document(document, corrections=warnings)
                after = yaml_source.loads(candidate, 'root')
                self.assertEqual(set(after.anchors), set(document.anchors))
                self.assertEqual([use.alias for use in after.occurrences if use.alias],
                                 [use.alias for use in document.occurrences if use.alias])
                if source in safe_sources:
                    self.assertEqual(after.value['META_DATA']['Describe'], 'Copy a scalar')
                    self.assertEqual(len([item for item in warnings if item.message.startswith('period: ')]), 2)
                else:
                    self.assertEqual(after.value, document.value)
                    self.assertFalse([item for item in warnings if item.message.startswith('period: ')])
                    self.assertTrue(any('Period retained' in item.hint for item in text_policy.check_document(document)))
                self.assertEqual(candidate, yaml_format.format_document(after))

    def test_skipped_hardware_prevents_direct_and_shared_period_changes(self):
        source = ('META_DATA: {Describe: &copy Copy a scalar.}\n'
                  'CONTRACT: {Hardware_specification: {Feature: {copy: *copy, other: Ordinary hardware prose.}}}\n')
        document = yaml_source.loads(source, 'root')
        candidate = yaml_format.format_document(document, skip_hardware=True)
        self.assertEqual(yaml_source.loads(candidate, 'root').value, document.value)
        diagnostics = text_policy.check_document(document, skip_hardware=True)
        self.assertEqual([item.path for item in diagnostics], ['root.META_DATA.Describe'])
        self.assertIn('excluded hardware', diagnostics[0].hint)
        after = yaml_source.loads(yaml_format.format_document(document), 'root')
        self.assertEqual(after.value['META_DATA']['Describe'], 'Copy a scalar')
        self.assertEqual(after.value['CONTRACT']['Hardware_specification']['Feature']['other'], 'Ordinary hardware prose')
