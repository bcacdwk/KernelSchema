"""Independent D acceptance using executable CLI processes and fixed artifacts.

The CLI deliberately synchronizes three real workers and three real reviewers.
It logs process IDs and monotonic intervals outside the Runner's state, so a
queue length or an in-memory transport cannot satisfy the concurrency checks.
Only the scalar-copy source/authoring fixture is shared with independent C.
"""

from copy import deepcopy
import json
from pathlib import Path
import threading
import time
import unittest

import yaml

from schema_extract.artifacts import validate_artifacts
from schema_extract.common import ExtractError, manifest, read_json
from schema_extract.runner import accept_result, run, summary
from schema_extract.tests.test_c_independent import CONTROLLED_CLI, IndependentFixture, plan_for


CLI_CONTROL = r'''
controls = scenario.get('controls', {}).get(key + '.' + str(number),
           scenario.get('controls', {}).get(key + '.*', {}))
markers = base / 'markers'
markers.mkdir(exist_ok=True)
marker_name = key + '.' + str(number)

def mark(event, **details):
    path = markers / (marker_name + '.' + event + '.json')
    temporary = path.with_suffix('.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps({'event': event, 'stage': stage, 'node': node,
                         'role': role, 'mode': mode, 'number': number,
                         'pid': os.getpid(), 'time_ns': time.monotonic_ns(), **details}))
    temporary.replace(path)

def wait_for(names):
    until = time.monotonic() + 14
    while not all((markers / (name + '.json')).exists() for name in names):
        if time.monotonic() > until:
            mark('barrier_timeout', missing=[name for name in names
                 if not (markers / (name + '.json')).exists()])
            raise SystemExit('the expected independent CLI calls never overlapped')
        time.sleep(0.01)

mark('begin', before_hashes=record['before_hashes'])
poisoned = None
if controls.get('poison') and stage == 'P5' and role == 'worker':
    poisoned = node + '.yaml'
    (workspace / poisoned).write_text('unfinished_private_draft: [\n')
    mark('poison')
wait_for(controls.get('wait_for', []))
time.sleep(controls.get('delay', 0))
if poisoned:
    (workspace / poisoned).write_text(before_text[poisoned])
'''


def independent_cli():
    """Extend only the model fixture, never import scheduling logic in the CLI."""
    script = CONTROLLED_CLI.replace('import hashlib\n', 'import hashlib\nimport fcntl\nimport os\nimport time\n', 1)
    script = script.replace(
        "previous = [json.loads(line) for line in log_path.read_text().splitlines()] if log_path.exists() else []",
        "with log_path.open('a+') as stream:\n"
        "    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)\n"
        "    stream.seek(0)\n"
        "    previous = [json.loads(line) for line in stream.read().splitlines()]",
        1,
    )
    script = script.replace(
        "with log_path.open('a') as stream:\n    stream.write(json.dumps(record) + '\\n')",
        "record['pid'] = os.getpid()\n"
        "with log_path.open('a') as stream:\n"
        "    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)\n"
        "    stream.write(json.dumps(record) + '\\n')\n" + CLI_CONTROL,
        1,
    )
    script = script.replace(
        "plan = scenario['plan']",
        "plan = scenario.get('plans', {}).get(str(number), scenario['plan']) if stage == 'P4' else scenario['plan']",
        1,
    )
    script = script.replace("if role == 'worker':\n", "if role == 'worker' and not action.startswith('reopen:'):\n", 1)
    script = script.replace(
        "        if number > 1:\n            current['meta_data']['version'] += 1",
        "        if not context['first_draft']:\n"
        "            current['meta_data']['version'] += 1",
        1,
    )
    script = script.replace(
        "    if stage not in ('P4', 'P6'):\n",
        "    if stage in ('P2', 'P3') and controls.get('root_when'):\n"
        "        current['meta_data'].setdefault('dispatch', {})['when'] = controls['root_when']\n"
        "    if stage == 'P2' and controls.get('omit_root_dispatch'):\n"
        "        current['meta_data'].pop('dispatch', None)\n"
        "    if stage not in ('P4', 'P6'):\n",
        1,
    )
    script = script.replace(
        "output.write_text(json.dumps(receipt))",
        "if action.startswith('reopen:'):\n"
        "    evidence = dict(source_evidence[1])\n"
        "    line = controls.get('reopen_line', 2)\n"
        "    evidence.update(line_start=line, line_end=line, quote=(Path(context['source_root']) / evidence['path']).read_text().splitlines()[line - 1])\n"
        "    receipt['source_evidence'].append(evidence)\n"
        "    new_index = len(receipt['source_evidence']) - 1\n"
        "    receipt['decision'] = 'reopen'\n"
        "    receipt['reopen'] = {'issue_id': action.split(':', 2)[2], 'target': action.split(':', 2)[1],\n"
        "        'reason': 'The fixed host evidence requires revisiting the recorded upstream choice.', 'evidence': [new_index]}\n"
        "    receipt['issues'] = [{'kind': 'plan', 'scope': filename,\n"
        "        'detail': 'The newly cited host branch conflicts with the recorded upstream choice.',\n"
        "        'recovery_condition': 'Review this precise host branch and approve the affected upstream choice.',\n"
        "        'evidence': [new_index]}]\n"
        "if controls.get('reverse_evidence'):\n"
        "    last = len(receipt['source_evidence']) - 1\n"
        "    receipt['source_evidence'].reverse()\n"
        "    for collection in ('conclusions', 'issues', 'framework_candidates', 'plan'):\n"
        "        for item in receipt.get(collection, []):\n"
        "            item['evidence'] = [last - index for index in item['evidence']]\n"
        "    if receipt.get('reopen'):\n"
        "        receipt['reopen']['evidence'] = [last - index for index in receipt['reopen']['evidence']]\n"
        "wire_schema = json.loads(Path(sys.argv[sys.argv.index('--output-schema') + 1]).read_text())\n"
        "if 'reopen' in wire_schema['properties'] and 'reopen' not in receipt:\n"
        "    receipt['reopen'] = None\n"
        "output.write_text(json.dumps(receipt))\n"
        "mark('end', decision=receipt['decision'])",
        1,
    )
    compile(script, '<independent-D-cli>', 'exec')
    return script


class IndependentDFixture(IndependentFixture):
    def __init__(self, *, controls=None, plans=None, **kwargs):
        super().__init__(**kwargs)
        self.fake.write_text(independent_cli(), encoding='utf-8')
        self.scenario['controls'] = controls or {}
        self.scenario['plans'] = plans or {}
        (self.base / 'scenario.json').write_text(json.dumps(self.scenario), encoding='utf-8')

    def create(self, **config):
        state = super().create(call_seconds=config.pop('call_seconds', 25), **config)
        self.initial_state = deepcopy(state)
        return state

    def markers(self, event=None):
        directory = self.base / 'markers'
        values = [read_json(path) for path in directory.glob('*.json')] if directory.exists() else []
        return [item for item in values if event is None or item['event'] == event]

    def marker(self, stage, node, role, number=1, event='begin', mode='normal'):
        path = self.base / 'markers' / f'{stage}.{node}.{role}.{mode}.{number}.{event}.json'
        return read_json(path)


class IndependentDTests(unittest.TestCase):
    def fixture(self, **kwargs):
        fixture = IndependentDFixture(**kwargs)
        self.addCleanup(fixture.cleanup)
        return fixture

    def assert_ready(self, fixture, state):
        self.assertEqual(state['status'], 'ready', state.get('reason'))
        planned = {entry['id'] for entry in state['plan']}
        approved = {node_id for node_id, node in state['nodes'].items()
                    if node['status'] == 'approved' and node.get('merged') and node.get('approval')}
        self.assertEqual(approved, planned)
        artifacts, info = fixture.store.verify_checkpoint(state)
        self.assertEqual(info['files'], manifest(artifacts))
        self.assertEqual(set(info['files']), {'dispatch.md', *(entry['file'] for entry in state['plan'])})
        for node_id in planned:
            node = state['nodes'][node_id]
            fixture.store.verify_approval(state, node['approval'])
            self.assertEqual(info['files'][node['file']], node['approved_sha256'])
        self.assertEqual(len(fixture.calls(stage='P6', role='reviewer')), 1)
        self.assertFalse(fixture.calls(stage='P6', role='worker'))
        for call in fixture.calls():
            attempt_id = call['context']['identity']['attempt_id']
            directory = fixture.store.run_dir(state['run_id']) / 'attempts' / attempt_id
            self.assert_strict_wire_schema(read_json(directory / 'response.schema.json'))
        return artifacts

    def assert_strict_wire_schema(self, value):
        if isinstance(value, dict):
            if 'properties' in value:
                self.assertEqual(set(value.get('required', [])), set(value['properties']),
                                 'the actual CLI output schema must require every object field; absence uses null')
                self.assertIs(value.get('additionalProperties'), False)
            for item in value.values():
                self.assert_strict_wire_schema(item)
        elif isinstance(value, list):
            for item in value:
                self.assert_strict_wire_schema(item)

    def test_three_real_workers_and_reviewers_overlap_and_unlock_a_child_early(self):
        siblings = ('alpha', 'beta', 'gamma')
        controls = {}
        for node, worker_delay, review_delay in zip(siblings, (0.7, 0.35, 0), (3.2, 0.6, 0)):
            controls[f'P5.{node}.worker.normal.1'] = {
                'poison': True,
                'wait_for': [f'P5.{peer}.worker.normal.1.poison' for peer in siblings],
                'delay': worker_delay,
            }
            controls[f'P5.{node}.reviewer.normal.1'] = {
                'wait_for': [f'P5.{peer}.reviewer.normal.1.begin' for peer in siblings],
                'delay': review_delay,
            }
        fixture = self.fixture(pairs=tuple((node, 'root') for node in siblings) + (('leaf', 'gamma'),),
                               controls=controls)
        fixture.create()
        result, errors, observed = [], [], []

        def execute():
            try:
                result.append(run(fixture.store, fixture.state['run_id']))
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=execute, name='independent-D-runner')
        thread.start()
        while thread.is_alive():
            state = fixture.store.load(fixture.state['run_id'])
            observed.append(summary(fixture.store, state))
            thread.join(0.08)
        if errors:
            raise errors[0]
        state = result[0]
        artifacts = self.assert_ready(fixture, state)

        for role in ('worker', 'reviewer'):
            starts = [fixture.marker('P5', node, role) for node in siblings]
            ends = [fixture.marker('P5', node, role, event='end') for node in siblings]
            self.assertEqual(len({item['pid'] for item in starts}), 3)
            self.assertLess(max(item['time_ns'] for item in starts), min(item['time_ns'] for item in ends),
                            'three actual CLI lifetimes must have a common overlap')
            self.assertEqual([item['node'] for item in sorted(ends, key=lambda item: item['time_ns'])],
                             ['gamma', 'beta', 'alpha'])
        leaf_started = fixture.marker('P5', 'leaf', 'worker')['time_ns']
        self.assertLess(leaf_started, fixture.marker('P5', 'alpha', 'reviewer', event='end')['time_ns'],
                        'a completed parent must unlock its child before a slow sibling finishes')
        self.assertGreater(leaf_started, fixture.marker('P5', 'gamma', 'reviewer', event='end')['time_ns'])
        for call in fixture.calls(stage='P5'):
            expected = {'root.yaml', call['node'] + '.yaml'}
            if call['node'] == 'leaf':
                expected.add('gamma.yaml')
            self.assertEqual(set(call['before_hashes']), expected,
                             'siblings and their malformed private drafts must not enter another workspace')
            self.assertIn('model_reasoning_effort="xhigh"', call['argv'])
            self.assertIn('exec', call['argv'])
            self.assertFalse(any('model_provider' in item for item in call['argv']))
            node = call['node']
            self.assertIn('node ' + node, yaml.safe_load((artifacts / (node + '.yaml')).read_text())['design']['math_reshape']['thesis'])
        node_calls = fixture.calls(stage='P5')
        self.assertEqual(len({call['session_id'] for call in node_calls}), len(node_calls))
        self.assertTrue(observed, 'status must remain readable during actual parallel calls')
        self.assertTrue(any(set(siblings) <= set(item.get('nodes_by_status', {}).get('in_flight', []))
                            for item in observed), 'status must show every actually in-flight sibling')
        self.assertTrue(any('leaf' in item.get('nodes_by_status', {}).get('waiting_parent', [])
                            for item in observed), 'status must identify the child waiting on its parent approval')

        before = deepcopy(fixture.store.load(state['run_id']))
        checkpoints_before = {path.name for path in (fixture.store.run_dir(state['run_id']) / 'checkpoints').iterdir()}
        approved_attempt = state['nodes']['gamma']['approval']['attempt_id']
        directory = fixture.store.run_dir(state['run_id']) / 'attempts' / approved_attempt
        try:
            replayed = accept_result(fixture.store, before, fixture.lock,
                                     read_json(directory / 'attempt.json'), read_json(directory / 'result.json'))
        except ExtractError as exc:
            self.assertIn('duplicate', str(exc).lower())
        else:
            self.assertEqual(replayed, before)
        self.assertEqual(fixture.store.load(state['run_id']), before)
        self.assertEqual({path.name for path in (fixture.store.run_dir(state['run_id']) / 'checkpoints').iterdir()},
                         checkpoints_before)
        self.assertEqual(run(fixture.store, state['run_id']), before)
        self.assertEqual(len(fixture.calls(stage='P6')), 1)

    def test_blocked_branch_leaves_only_its_descendants_waiting(self):
        fixture = self.fixture(pairs=(('blocked_path', 'root'), ('waiting_leaf', 'blocked_path'),
                                     ('healthy', 'root'), ('healthy_leaf', 'healthy')),
                               actions={'P5.blocked_path.reviewer.normal.1': 'blocked'},
                               controls={'P5.healthy.worker.normal.1': {
                                   'wait_for': ['P5.blocked_path.reviewer.normal.1.end'], 'delay': 0.2}})
        state = fixture.run()
        self.assertEqual(state['status'], 'blocked', state.get('reason'))
        self.assertEqual(state['nodes']['blocked_path']['status'], 'blocked')
        self.assertNotEqual(state['nodes']['waiting_leaf']['status'], 'approved')
        for node in ('healthy', 'healthy_leaf'):
            self.assertEqual(state['nodes'][node]['status'], 'approved')
            self.assertTrue(state['nodes'][node]['merged'])
            fixture.store.verify_approval(state, state['nodes'][node]['approval'])
        self.assertFalse(fixture.calls(node='waiting_leaf'))
        self.assertFalse(fixture.calls(stage='P6'))
        self.assertLess(fixture.marker('P5', 'blocked_path', 'reviewer', event='end')['time_ns'],
                        fixture.marker('P5', 'healthy', 'reviewer')['time_ns'])
        merged, info = fixture.store.verify_checkpoint(state, state['merged_checkpoint'])
        self.assertEqual(set(info['files']), {'root.yaml', 'dispatch.md', 'healthy.yaml', 'healthy_leaf.yaml'})
        self.assertEqual(info['files'], manifest(merged))
        classified = summary(fixture.store, state)['nodes_by_status']
        self.assertEqual(set(classified['approved']), {'root', 'healthy', 'healthy_leaf'})
        self.assertEqual(classified['blocked'], ['blocked_path'])
        self.assertEqual(classified['waiting_parent'], ['waiting_leaf'])
        self.assertFalse(classified['in_flight'])

    def test_changed_parent_fences_late_child_and_retains_an_unrelated_approval(self):
        pairs = (('alpha', 'root'), ('safe', 'root'), ('requester', 'root'), ('leaf', 'alpha'))
        revised = plan_for(pairs)
        revised[1]['selection'] = 'The newly checked host branch requires a different alpha selection.'
        fixture = self.fixture(
            pairs=pairs, plans={'2': revised},
            actions={'P5.alpha.reviewer.normal.1': 'revise',
                     'P5.requester.worker.normal.1': 'reopen:P4:alpha-selection'},
            controls={
                'P5.requester.worker.normal.1': {'wait_for': ['P5.leaf.worker.normal.1.begin']},
                'P5.leaf.worker.normal.1': {'wait_for': ['P5.alpha.worker.normal.2.end'], 'delay': 0.2},
            },
        )
        state = fixture.run(revision_limit=1)
        artifacts = self.assert_ready(fixture, state)
        self.assertEqual(state['plan_revision'], 2)
        self.assertEqual(state['plan'], revised)
        self.assertEqual(len(fixture.calls(stage='P4', role='reviewer')), 2)
        self.assertEqual(len(fixture.calls(stage='P5', node='safe', role='worker')), 1)
        self.assertEqual(len(fixture.calls(stage='P5', node='safe', role='reviewer')), 1)
        old_safe = fixture.calls(stage='P5', node='safe', role='reviewer')[0]
        self.assertEqual(state['nodes']['safe']['approval']['attempt_id'], old_safe['context']['identity']['attempt_id'])
        self.assertEqual(state['nodes']['safe']['approval']['plan_revision'], 1,
                         'a retained approval must keep its true original receipt identity')
        fixture.store.verify_approval(state, state['nodes']['safe']['approval'])

        old_leaf, new_leaf = fixture.calls(stage='P5', node='leaf', role='worker')
        old_parent = old_leaf['before']['alpha.yaml']['design']['math_reshape']['thesis']
        new_parent = new_leaf['before']['alpha.yaml']['design']['math_reshape']['thesis']
        self.assertNotEqual(old_parent, new_parent)
        self.assertLess(fixture.marker('P5', 'alpha', 'worker', number=2, event='end')['time_ns'],
                        fixture.marker('P5', 'leaf', 'worker', event='end')['time_ns'],
                        'the old child response must actually arrive after the new parent was authored')
        final_leaf = yaml.safe_load((artifacts / 'leaf.yaml').read_text())
        self.assertIn('node leaf, revision 2', final_leaf['design']['math_reshape']['thesis'])
        self.assertEqual(len(fixture.calls(stage='P5', node='leaf', role='reviewer')), 1,
                         'the stale first child draft must not be independently approved')
        self.assertNotEqual(state['nodes']['leaf']['approval']['attempt_id'], old_leaf['context']['identity']['attempt_id'])
        first_budget = fixture.calls(stage='P5', node='alpha', role='worker')[0]['context']['budget']
        reopened_budget = fixture.calls(stage='P5', node='alpha', role='worker')[1]['context']['budget']
        self.assertEqual(reopened_budget['deadline'], first_budget['deadline'])
        self.assertEqual(reopened_budget['revisions'], 1)
        self.assertGreater(reopened_budget['calls'], first_budget['calls'])
        self.assertEqual(state['nodes']['alpha']['stage']['adjudications'], 1)
        self.assertEqual(state['deadline'], fixture.initial_state['deadline'])

    def test_removing_and_readding_a_node_preserves_its_budget_and_used_adjudication(self):
        first_pairs = (('victim', 'root'), ('safe', 'root'), ('first', 'root'), ('leaf', 'victim'))
        second_pairs = (('safe', 'root'), ('first', 'root'), ('second', 'root'))
        third_pairs = (*second_pairs, ('victim', 'root'), ('leaf', 'victim'))
        fixture = self.fixture(
            pairs=first_pairs, plans={'2': plan_for(second_pairs), '3': plan_for(third_pairs)},
            actions={
                'P5.victim.reviewer.normal.1': 'revise',
                'P5.victim.reviewer.normal.2': 'revise',
                'P5.first.worker.normal.1': 'reopen:P4:remove-victim',
                'P5.second.worker.normal.1': 'reopen:P4:restore-victim',
            },
            controls={
                'P5.first.worker.normal.1': {'wait_for': ['P5.leaf.worker.normal.1.begin']},
                'P5.leaf.worker.normal.1': {'wait_for': ['P4.root.reviewer.normal.2.end'], 'delay': 0.1},
                'P5.second.worker.normal.1': {'wait_for': ['P5.first.reviewer.normal.1.end'],
                                            'delay': 0.1, 'reopen_line': 3},
            },
        )
        state = fixture.run(revision_limit=1)
        self.assertEqual(state['status'], 'blocked', state.get('reason'))
        self.assertEqual(state['plan_revision'], 3)
        self.assertEqual(state['nodes']['victim']['status'], 'blocked')
        workers = fixture.calls(stage='P5', node='victim', role='worker')
        self.assertEqual(len(workers), 2)
        first_budget, restored_budget = (call['context']['budget'] for call in workers)
        self.assertEqual(first_budget['deadline'], restored_budget['deadline'])
        self.assertEqual(restored_budget['revisions'], 1)
        self.assertGreaterEqual(restored_budget['calls'], 4)
        victim = state['nodes']['victim']['stage']
        self.assertEqual(victim['revisions'], 2)
        self.assertEqual(victim['adjudications'], 1)
        self.assertEqual(len(fixture.calls(stage='P5', node='victim', mode='adjudicate')), 1,
                         'removing and readding a node must never grant a second adjudication')
        finished_seconds = 0.0
        for call in fixture.calls(stage='P5', node='victim'):
            attempt_id = call['context']['identity']['attempt_id']
            directory = fixture.store.run_dir(state['run_id']) / 'attempts' / attempt_id
            finished_seconds += read_json(directory / 'finished.json')['elapsed_seconds']
        self.assertGreaterEqual(victim['elapsed_seconds'] + 1e-7, finished_seconds)
        for node in ('safe', 'first', 'second'):
            self.assertEqual(state['nodes'][node]['status'], 'approved', node)
            fixture.store.verify_approval(state, state['nodes'][node]['approval'])
        self.assertEqual(state['nodes']['safe']['approval']['plan_revision'], 1)
        self.assertEqual(state['stages']['P4']['calls'], 6)
        self.assertEqual(state['deadline'], fixture.initial_state['deadline'])
        self.assertFalse(fixture.calls(stage='P6'))

    def test_p3_can_reopen_p2_before_any_branch_plan_exists(self):
        when = 'Select the reconfirmed root path for every finite scalar in the Contract domain.'
        fixture = self.fixture(
            actions={'P3.root.worker.normal.1': 'reopen:P2:representative-root-choice'},
            controls={'P2.root.worker.normal.2': {'root_when': when}},
        )
        state = fixture.run()
        artifacts = self.assert_ready(fixture, state)
        self.assertEqual(state['plan_revision'], 2)
        self.assertEqual(len(fixture.calls(stage='P1', role='reviewer')), 1)
        self.assertEqual(len(fixture.calls(stage='P2', role='reviewer')), 2)
        self.assertEqual(len(fixture.calls(stage='P3', role='worker')), 2)
        self.assertEqual(len(fixture.calls(stage='P4', role='reviewer')), 1)
        self.assertEqual(state['stages']['P1']['approval']['plan_revision'], 1)
        fixture.store.verify_approval(state, state['stages']['P1']['approval'])
        p2_workers = fixture.calls(stage='P2', role='worker')
        self.assertEqual(p2_workers[0]['context']['budget']['deadline'], p2_workers[1]['context']['budget']['deadline'])
        self.assertEqual(state['stages']['P2']['calls'], 4)
        self.assertEqual(yaml.safe_load((artifacts / 'root.yaml').read_text())['meta_data']['dispatch']['when'], when)

    def test_root_when_reopening_rechecks_p3_and_fences_the_old_branch(self):
        for initial_dispatch in (True, False):
            with self.subTest(initial_dispatch=initial_dispatch):
                self.root_when_reopening_case(initial_dispatch)

    def root_when_reopening_case(self, initial_dispatch):
        when = 'Select root exactly for strictly positive finite scalar X after source correction.'
        fixture = self.fixture(
            pairs=(('requester', 'root'), ('slow', 'root')),
            actions={'P5.requester.worker.normal.1': 'reopen:P3:root-condition'},
            controls={
                'P5.requester.worker.normal.1': {'wait_for': ['P5.slow.worker.normal.1.begin']},
                'P5.slow.worker.normal.1': {'wait_for': ['P3.root.worker.normal.2.end'], 'delay': 0.1},
                'P3.root.worker.normal.2': {'root_when': when},
                'P4.root.reviewer.normal.*': {'reverse_evidence': True},
                'P2.root.worker.normal.1': {'omit_root_dispatch': not initial_dispatch},
            },
        )
        state = fixture.run()
        artifacts = self.assert_ready(fixture, state)
        self.assertEqual(len(fixture.calls(stage='P3', role='reviewer')), 2)
        self.assertEqual(len(fixture.calls(stage='P4', role='reviewer')), 2)
        self.assertEqual(len(fixture.calls(stage='P5', node='slow', role='reviewer')), 1)
        old, new = fixture.calls(stage='P5', node='slow', role='worker')
        old_metadata = old['before']['root.yaml']['meta_data']
        self.assertEqual('dispatch' in old_metadata, initial_dispatch)
        self.assertNotEqual(old_metadata.get('dispatch', {}).get('when'), when)
        self.assertEqual(new['before']['root.yaml']['meta_data']['dispatch']['when'], when)
        self.assertEqual(yaml.safe_load((artifacts / 'root.yaml').read_text())['meta_data']['dispatch']['when'], when)
        self.assertIn('node slow, revision 2', yaml.safe_load((artifacts / 'slow.yaml').read_text())['design']['math_reshape']['thesis'])
        self.assertEqual(state['stages']['P3']['calls'], 4)
        self.assertEqual(state['deadline'], fixture.initial_state['deadline'])
        p4_calls = fixture.calls(stage='P4')
        receipts = [read_json(fixture.store.run_dir(state['run_id']) / 'attempts' /
                              call['context']['identity']['attempt_id'] / 'receipt.json') for call in p4_calls]
        self.assertNotEqual(receipts[0]['plan'][0]['evidence'], receipts[1]['plan'][0]['evidence'],
                            'independent reviewer indices must actually differ from the worker indices')
        for call in fixture.calls(stage='P5'):
            context = call['context']
            plan_receipt = read_json(Path(context['plan_receipt']))
            entry = next(item for item in plan_receipt['plan'] if item['id'] == call['node'])
            expected = [plan_receipt['source_evidence'][index] for index in entry['evidence']]
            self.assertEqual(context['node_evidence'], expected,
                             'P5 must receive the actual approved source anchors, not unexplained receipt-local indices')
            self.assertTrue(expected)
            self.assertTrue(any(anchor['category'] == 'host' for anchor in expected))
        if not initial_dispatch:
            self.assert_root_when_scope(fixture, artifacts)

    def assert_root_when_scope(self, fixture, artifacts):
        """Exercise the public gate for the precise permission being changed."""
        baseline = yaml.safe_load((artifacts / 'root.yaml').read_text())
        before = fixture.base / 'root-scope-before'
        after = fixture.base / 'root-scope-after'
        before.mkdir()
        after.mkdir()
        starter = Path(fixture.lock['snapshot']) / 'rules/schema_template/starter.yaml'

        def gate(old, new, *, authorized=True):
            new = deepcopy(new)
            new['meta_data']['version'] = old['meta_data']['version'] + 1
            (before / 'root.yaml').write_text(yaml.safe_dump(old, sort_keys=False))
            (after / 'root.yaml').write_text(yaml.safe_dump(new, sort_keys=False))
            return validate_artifacts('P3', 'worker', before, after, starter,
                                      decision='complete', root_revision=authorized)

        omitted = deepcopy(baseline)
        omitted['meta_data'].pop('dispatch', None)
        added = deepcopy(omitted)
        added['meta_data']['dispatch'] = {'when': 'Select all valid scalar inputs.'}
        identified = deepcopy(added)
        identified['meta_data']['dispatch'].update(id='root', parent='')
        without_when = deepcopy(identified)
        without_when['meta_data']['dispatch'].pop('when')
        for old, new in ((omitted, added), (added, omitted), (identified, without_when)):
            with self.subTest(scope='authorized when change'):
                gate(old, new)
                with self.assertRaisesRegex(ExtractError, 'P3 changed'):
                    gate(old, new, authorized=False)

        for field in ('id', 'parent', 'source', 'contract'):
            with self.subTest(protected=field):
                changed = deepcopy(identified)
                changed['meta_data']['dispatch'].pop('when')
                if field in ('id', 'parent'):
                    changed['meta_data']['dispatch'][field] = 'another_node'
                elif field == 'source':
                    changed['meta_data']['source'] += '-different-source'
                else:
                    changed['contract']['algorithm_requirement']['compute'][0]['math'] = 'Y = 42.'
                with self.assertRaisesRegex(ExtractError, 'P3 changed'):
                    gate(identified, changed)


if __name__ == '__main__':
    unittest.main()
