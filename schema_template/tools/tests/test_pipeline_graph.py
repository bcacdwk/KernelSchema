"""Finite event semantics: ordered control, windows, members and four times."""
import unittest
from copy import deepcopy

from ..pipeline import ExpansionError, GraphError, analyze, build_step, configuration
from ..pipeline_syntax import PipelineSyntaxError, parse_action, parse_event, parse_for, parse_pipe, parse_waiting
from .support import PIPE, WORK, TaskCase, add_step, at, role


class GraphTests(unittest.TestCase):
    def build(self, roles, *, index=None, parameters=None, worktiles=1, members=1, dynamic=False):
        definition = {'Role': roles}
        if index:
            definition['Index'] = {name: {'meaning': 'Synthetic index', 'definition': value} for name, value in index.items()}
        return build_step('MAIN', definition, {'unit': 'cta', 'unit_count': members},
                          parameters=parameters or {}, worktiles=worktiles, dynamic_worktiles=dynamic)

    def test_sibling_loops_and_independent_conditions_keep_list_order(self):
        main = role([{'for(i=0:2)': ['load:x(i)']}, {'for(j=0:2)': ['compute:y(j)']},
                     {'if(flag == 1)': ['signal:a']}, {'if(flag == 0)': ['signal:b']}],
                    {'load': {'x': 'async_unordered'}, 'compute': {'y': 'serial'},
                     'signal': {'a': 'serial', 'b': 'serial'}})
        graph = self.build({'MAIN': main}, parameters={'flag': 0})
        self.assertEqual([op.action for op in graph.operations.values()],
                         ['load:x(0)', 'load:x(1)', 'compute:y(0)', 'compute:y(1)', 'signal:b'])
        with self.assertRaises(ExpansionError):
            self.build({'MAIN': main})
        main['Info']['Using_verb']['signal'].pop('a')
        with self.assertRaisesRegex(GraphError, 'undeclared action family'):
            self.build({'MAIN': main}, parameters={'flag': 0})

    def test_empty_work_keeps_actions_outside_the_loop(self):
        main = role([{'for(t)': ['compute:y(t)']}, 'signal:end'],
                    {'compute': {'y': 'serial'}, 'signal': {'end': 'serial'}})
        graph = self.build({'MAIN': main}, index={'t': 'worktile'}, worktiles=0)
        self.assertEqual([op.action for op in graph.operations.values()], ['signal:end'])

    def test_all_four_explicit_event_edges(self):
        for target_time in ('issue', 'done'):
            for predecessor_time in ('issue', 'done'):
                with self.subTest(target=target_time, source=predecessor_time):
                    a = role(['compute:a'], {'compute': {'a': 'async_unordered'}})
                    b = role(['compute:b'], {'compute': {'b': 'async_unordered'}},
                             waiting={'Prepared': [f'{{{target_time} compute:b}} ← [{{{predecessor_time} A::compute:a}}]']})
                    graph = self.build({'A': a, 'B': b})
                    edge = (f'MAIN/A[0]/compute:a@{predecessor_time}', f'MAIN/B[0]/compute:b@{target_time}', 'Prepared')
                    self.assertIn(edge, graph.edges)
                    self.assertFalse(graph.cycle)
                    report = graph.report()
                    self.assertTrue(any(item['paths'] for item in report['edges'] if item['kind'] == 'Prepared'))

    def test_same_noun_does_not_create_data_dependencies(self):
        a = role(['load:x'], {'load': {'x': 'async_unordered'}})
        b = role(['compute:x'], {'compute': {'x': 'serial'}})
        graph = self.build({'A': a, 'B': b})
        self.assertEqual(len(graph.edges), 2)  # Each action's own issue -> done.
        self.assertFalse(any('/A[' in source and '/B[' in target for source, target, _ in graph.edges))

    def test_ordered_completion_crosses_nouns_verbs_and_worktiles_without_async_drain(self):
        main = role([{'for(t)': ['load:a(t)', 'compute:b(t)', 'signal:c(t)', 'store:d(t)']}],
                    {'load': {'a': 'async_ordered'}, 'compute': {'b': 'serial'},
                     'signal': {'c': 'async_unordered'}, 'store': {'d': 'async_ordered'}})
        graph = self.build({'MAIN': main}, index={'t': 'worktile'}, worktiles=2)
        ordered = [(source, target) for source, target, kind in graph.edges if kind == 'ordered']
        self.assertEqual(ordered, [
            ('MAIN/MAIN[0]/load:a(0)@done', 'MAIN/MAIN[0]/store:d(0)@done'),
            ('MAIN/MAIN[0]/store:d(0)@done', 'MAIN/MAIN[0]/load:a(1)@done'),
            ('MAIN/MAIN[0]/load:a(1)@done', 'MAIN/MAIN[0]/store:d(1)@done')])
        self.assertNotIn(('MAIN/MAIN[0]/load:a(0)@done', 'MAIN/MAIN[0]/compute:b(0)@issue', 'pipe'), graph.edges)

    def test_completion_can_wait_for_a_later_issue_and_cycles_remain_complete_graphs(self):
        main = role([{'for(i)': ['load:x(i)']}], {'load': {'x': 'async_unordered'}},
                    waiting={'Allowed': ['{done load:x(i)} ← [{issue load:x(i+1)}]']})
        graph = self.build({'MAIN': main}, index={'i': [0, 2]})
        self.assertFalse(graph.cycle)
        self.assertEqual(len(graph.boundaries), 1)
        main['Info']['Using_verb']['load']['x']['execution_mode'] = 'serial'
        graph = self.build({'MAIN': main}, index={'i': [0, 2]})
        self.assertTrue(graph.cycle)
        self.assertEqual(len(graph.operations), 2)
        self.assertFalse(graph.report()['acyclic'])

    def test_shifted_window_boundary_is_distinct_from_fixed_bad_reference(self):
        main = role([{'for(i)': ['compute:y(i)']}], {'compute': {'y': 'async_unordered'}},
                    waiting={'Prepared': ['{issue compute:y(i)} ← [{done compute:y(i-1)}]']})
        graph = self.build({'MAIN': main}, index={'i': [0, 2]})
        self.assertEqual(len(graph.boundaries), 1)
        main['Waiting']['Prepared'] = ['{issue compute:y(i)} ← [{done compute:y(99)}]']
        with self.assertRaises(GraphError) as caught:
            self.build({'MAIN': main}, index={'i': [0, 2]})
        self.assertNotIsInstance(caught.exception, ExpansionError)

    def test_absent_target_has_no_wait_but_present_target_needs_its_in_window_source(self):
        a = role([{'for(i)': [{'if(i == 1)': ['load:x(i)']}]}], {'load': {'x': 'async_unordered'}})
        b = role([{'for(i)': [{'if(flag == 1)': ['compute:y(i)']}]}], {'compute': {'y': 'async_unordered'}},
                 waiting={'Prepared': ['{issue compute:y(i)} ← [{done A::load:x(i)}]']})
        graph = self.build({'A': a, 'B': b}, index={'i': [0, 2]}, parameters={'flag': 0})
        self.assertEqual(len(graph.operations), 1)
        with self.assertRaises(GraphError) as caught:
            self.build({'A': a, 'B': b}, index={'i': [0, 2]}, parameters={'flag': 1})
        self.assertNotIsInstance(caught.exception, ExpansionError)

    def test_omitted_members_are_local_and_warps_do_not_multiply_workers(self):
        a = role(['load:x'], {'load': {'x': 'async_unordered'}}, member='cta[0]')
        b = role(['compute:y'], {'compute': {'y': 'serial'}}, member='cta[1]',
                 waiting={'Prepared': ['{issue compute:y} ← [{done A::load:x}]']})
        with self.assertRaisesRegex(GraphError, 'no role instance'):
            self.build({'A': a, 'B': b}, members=2)
        b['Waiting']['Prepared'] = ['{issue compute:y} ← [{done A[0]::load:x}]']
        self.assertFalse(self.build({'A': a, 'B': b}, members=2).cycle)
        a['Info']['Member'] = 'warp[0:4]'
        self.assertEqual(len(self.build({'MAIN': a}).operations), 1)

    def test_absent_target_does_not_evaluate_unused_open_math_arguments(self):
        a = role(['load:x(0)'], {'load': {'x': 'serial'}})
        b = role([{'if(flag == 1)': ['compute:y(0)']}], {'compute': {'y': 'serial'}},
                 waiting={'Prepared': ['{issue compute:y(0)} ← [{done A::load:x(⌊limit⌋)}]']})
        graph = self.build({'A': a, 'B': b}, parameters={'flag': 0})
        self.assertEqual(len(graph.operations), 1)
        with self.assertRaises(ExpansionError):
            self.build({'A': a, 'B': b}, parameters={'flag': 1})
        b['Waiting']['Prepared'] = ['{issue compute:y(0)} ← [{done A[1]::load:x(⌊limit⌋)}]']
        with self.assertRaisesRegex(GraphError, 'no role instance'):
            self.build({'A': a, 'B': b}, parameters={'flag': 0})

    def test_fixed_first_target_has_no_wait_in_an_empty_work_domain(self):
        a = role([{'for(t)': ['release:x(t)']}], {'release': {'x': 'async_unordered'}})
        b = role([{'for(t)': [{'for(g[t])': ['compute:y(g)']}]}],
                 {'compute': {'y': 'async_unordered'}},
                 waiting={'Released': ['{issue compute:y(g[t,0])} ← [{done A::release:x(t-1)}]']})
        index = {'t': 'worktile', 'i': [0, 'n(t)'], 'g': {'flatten': ['t', 'i']}}
        graph = self.build({'A': a, 'B': b}, index=index,
                           parameters={'n(0)': 0, 'n(1)': 2}, worktiles=2)
        edges = [(source, target) for source, target, kind in graph.edges if kind == 'Released']
        self.assertEqual(edges, [('MAIN/A[0]/release:x(0)@done', 'MAIN/B[0]/compute:y(0)@issue')])
        self.assertFalse(graph.boundaries)
        b['Waiting']['Released'] = ['{issue compute:y(g[t,99])} ← [{done A::release:x(t-1)}]']
        with self.assertRaisesRegex(GraphError, 'i=99 outside'):
            self.build({'A': a, 'B': b}, index=index,
                       parameters={'n(0)': 0, 'n(1)': 2}, worktiles=2)

    def test_scope_shadowing_and_flatten_coordinate_conflicts(self):
        with self.assertRaises(PipelineSyntaxError):
            parse_pipe([{'for(i=0:2)': [{'for(i=0:1)': ['compute:y(i)']}]}])
        main = role([{'for(t)': [{'for(n)': ['compute:y(n)']}]}], {'compute': {'y': 'serial'}})
        with self.assertRaises(GraphError):
            self.build({'MAIN': main}, index={'t': [0, 2], 'i': [0, 2], 'n': {'flatten': ['t', 'i']}})

    def test_local_range_endpoints_keep_outer_scope_in_expansion_and_waiting(self):
        for endpoint, index, parameters in (('count(i)', {'i': [0, 3]}, {}), ('i', None, {'i': 3})):
            with self.subTest(endpoint=endpoint):
                main = role([{f'for(i=0:{endpoint})': ['compute:y(i)']}],
                            {'compute': {'y': 'async_unordered'}},
                            waiting={'Prepared': ['{issue compute:y(i)} ← [{done compute:y(i-1)}]']})
                graph = self.build({'MAIN': main}, index=index, parameters=parameters)
                self.assertEqual([op.action for op in graph.operations.values()],
                                 ['compute:y(0)', 'compute:y(1)', 'compute:y(2)'])
                self.assertEqual(sum(kind == 'Prepared' for _, _, kind in graph.edges), 2)
                self.assertEqual(len(graph.boundaries), 1)
                main['Waiting']['Prepared'] = ['{issue compute:y(i)} ← [{done compute:y(99)}]']
                with self.assertRaises(GraphError) as caught:
                    self.build({'MAIN': main}, index=index, parameters=parameters)
                self.assertNotIsInstance(caught.exception, ExpansionError)
        main = role([{'for(i)': ['compute:y(i)']}], {'compute': {'y': 'serial'}})
        with self.assertRaisesRegex(GraphError, 'recursive index domain'):
            self.build({'MAIN': main}, index={'i': [0, 'count(i)']})

    def test_dynamic_domain_queries_cannot_be_hidden_by_empty_sample(self):
        main = role([{'for(t)': ['compute:y(count(t))']}], {'compute': {'y': 'serial'}})
        with self.assertRaisesRegex(GraphError, 'dynamically refilled'):
            self.build({'MAIN': main}, index={'t': 'worktile'}, worktiles=0, dynamic=True)
        with self.assertRaises(ExpansionError):
            configuration({'n': 10**13})
        with self.assertRaises(ExpansionError):
            configuration({}, 65)

    def test_composite_matmul_keeps_operand_signatures(self):
        action = parse_action('matmul:a(i)@b(i,j)')
        self.assertEqual(action.noun, 'a@b')
        self.assertEqual([operand.args for operand in action.operands], [('i',), ('i', 'j')])
        with self.assertRaises(PipelineSyntaxError):
            parse_action('load:a(i)@b(i)')

    def test_parsed_names_and_integer_forms_do_not_limit_whole_events(self):
        noun = 'n' * 50
        action = parse_action(f'matmul:{noun}(i)@{noun}(i,j)')
        self.assertEqual(len(action.operands), 2)
        for value in ('compute:' + 'n' * 51, 'compute:x(，)', 'compute:x(++)',
                      'compute:x(∞)', 'compute:x(' + 'p' * 51 + ')',
                      'compute:x(f(' + 'p' * 51 + '=1))'):
            with self.subTest(value=value), self.assertRaises(PipelineSyntaxError):
                parse_action(value)
        parse_for('for(i[:])')
        parse_for('for(i=0:⌊n/2⌋)')
        parse_event('{done MAIN[*]::compute:x(i)}')
        for value in ('for(i=0:...)', 'for(i[+:])', 'for(' + 'i' * 51 + '=0:2)'):
            with self.subTest(value=value), self.assertRaises(PipelineSyntaxError):
                parse_for(value)
        event = '{done MAIN[*]::compute:' + noun + '(' + ','.join(['i'] * 70) + ')}'
        rule = parse_waiting(event + ' ← [' + ', '.join([event] * 6) + ']')
        self.assertEqual(len(rule.predecessors), 6)


class EventCallTests(TaskCase):
    def plan(self, formal='k', *, reverse=False):
        data = deepcopy(self.base)
        add_step(data)
        main = at(data, PIPE)
        main['Role'] = {'MAIN': role(
            ['signal:entry', {'for(i=0:2)': ['compute:x(i+1)']}, {'for(j=0:3)': ['store:y(j)']}],
            {'signal': {'entry': 'serial'}, 'compute': {'x': 'async_unordered'},
             'store': {'y': 'async_unordered'}})}
        references = ['{done MAIN[0]::signal:entry}',
                      f'{{done MAIN[0]::compute:x({formal})}}', f'{{done MAIN[0]::store:y({formal})}}']
        main['Event'] = {f'ready({formal})': list(reversed(references)) if reverse else references,
                         'unused(q)': ['{done MAIN[0]::compute:x(⌊q⌋)}']}
        at(data, WORK)['Flow']['Launch'][1]['start_after'] = 'MAIN.ready(2)'
        return data['root']

    def test_calls_bind_formals_without_capturing_loop_domains_or_list_order(self):
        expected = {'ready(2)': ['MAIN/MAIN[0]/compute:x(2)@done',
                                 'MAIN/MAIN[0]/signal:entry@done', 'MAIN/MAIN[0]/store:y(2)@done']}
        for formal in ('i', 'k'):
            for reverse in (False, True):
                with self.subTest(formal=formal, reverse=reverse):
                    graph = analyze(self.plan(formal, reverse=reverse), configuration({}))[0]
                    self.assertEqual(graph.events, expected)

    def test_all_flow_event_call_sites_are_evaluated_without_cross_step_edges(self):
        plan = self.plan()
        work, pipe = plan['DESIGN']['Work_schedule'], plan['DESIGN']['Tile_pipeline']['Step']
        work['Flow']['Handoff'] = {'y': {'producer': 'MAIN', 'consumer': ['NEXT'], 'storage': 'global',
                                               'ready': ['MAIN.ready(1)'], 'wait_at': ['NEXT.input_wait(0)']}}
        pipe['NEXT']['Role'] = {'MAIN': role(['await:input(0)'], {'await': {'input': 'serial'}})}
        pipe['NEXT']['Event'] = {'input_wait(v)': ['{done MAIN[0]::await:input(v)}']}
        graphs = analyze(plan, configuration({}))
        self.assertEqual(set(graphs[0].events), {'ready(1)', 'ready(2)'})
        self.assertEqual(graphs[1].events, {'input_wait(0)': ['NEXT/MAIN[0]/await:input(0)@done']})
        for graph in graphs:
            self.assertTrue(all(source.startswith(graph.step + '/') and target.startswith(graph.step + '/')
                                for source, target, _ in graph.edges))

    def test_only_used_aliases_need_values_and_missing_called_events_are_errors(self):
        plan = self.plan()
        launch = plan['DESIGN']['Work_schedule']['Flow']['Launch'][1]
        launch['start_after'] = 'MAIN.ready(selection)'
        with self.assertRaises(ExpansionError):
            analyze(plan, configuration({}))
        self.assertEqual(set(analyze(plan, configuration({'selection': 2}))[0].events), {'ready(2)'})
        launch['start_after'] = 'MAIN.unused(2)'
        with self.assertRaises(ExpansionError):
            analyze(plan, configuration({}))
        launch['start_after'] = 'MAIN.ready(0)'
        with self.assertRaises(GraphError) as caught:
            analyze(plan, configuration({}))
        self.assertNotIsInstance(caught.exception, ExpansionError)
        self.assertIn('Flow.Launch[1].start_after', str(caught.exception))
        self.assertIn('compute:x(0)', str(caught.exception))
