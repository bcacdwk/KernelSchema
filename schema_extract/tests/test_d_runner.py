"""Runner integration boundaries beyond the independent D scheduling evidence."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

import yaml

from schema_extract.common import ExtractError, read_json
from schema_extract.artifacts import validate_artifacts, write_yaml
from schema_extract.runner import accept_result, run, summary
from test_c_independent import CONTROLLED_CLI, IndependentFixture


CONTROL = r'''
control = scenario.get('control', {}).get(key + '.' + str(number), {})
request = control.get('request')
marks = base / 'marks'
marks.mkdir(exist_ok=True)
mark_name = key + '.' + str(number)

def mark(event):
    (marks / (mark_name + '.' + event + '.json')).write_text(json.dumps(
        {'pid': os.getpid(), 'time': time.monotonic(), 'identity': identity}))

mark('begin')
until = time.monotonic() + 15
while not all((marks / (name + '.json')).is_file() for name in control.get('wait_for', [])):
    if time.monotonic() > until:
        raise SystemExit('controlled CLI prerequisite did not start or complete')
    time.sleep(0.01)
time.sleep(control.get('delay', 0))
'''


REOPEN_OUTPUT = r'''
if request:
    anchor = dict(source_evidence[1])
    line = request.get('line', 2)
    quote = (Path(context['source_root']) / anchor['path']).read_text().splitlines()[line - 1]
    if request.get('short_quote'):
        quote = quote.strip()
    anchor.update(line_start=line, line_end=line, quote=quote, note=request.get('note', 'New fixed source lines.'))
    receipt['source_evidence'].append(anchor)
    index = len(receipt['source_evidence']) - 1
    receipt['decision'] = 'reopen'
    receipt['reopen'] = {'issue_id': request['issue_id'], 'target': request['target'],
        'reason': 'The fixed source identifies an upstream decision to review.', 'evidence': [index]}
    receipt['issues'] = [{'kind': request.get('kind', 'external' if request['target'] == 'external' else 'source'),
        'scope': request.get('scope', filename), 'detail': 'Resolve the recorded input requirement or source transcription.',
        'recovery_condition': 'Task owner must confirm the changed external requirement.' if request['target'] == 'external'
                              else 'Recheck the fixed source lines in the named upstream stage.', 'evidence': [index]}]
    if request.get('kind') == 'transcription':
        path = workspace / 'root.yaml'
        lines = path.read_text().splitlines()
        line = next(i for i, text in enumerate(lines) if 'meaning:' in text)
        receipt['artifact_evidence'].append({'path': 'root.yaml', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'line_start': line + 1, 'line_end': line + 1, 'quote': lines[line], 'note': 'The current field omits the fixed source meaning.'})
output.write_text(json.dumps(receipt))
mark('end')
'''


class DFixture(IndependentFixture):
    def __init__(self, *, control=None, **kwargs):
        super().__init__(**kwargs)
        script = CONTROLLED_CLI.replace('import hashlib\n', 'import hashlib\nimport fcntl\nimport os\nimport time\n', 1)
        script = script.replace(
            "previous = [json.loads(line) for line in log_path.read_text().splitlines()] if log_path.exists() else []",
            "with log_path.open('a+') as stream:\n"
            "    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)\n"
            "    stream.seek(0)\n"
            "    previous = [json.loads(line) for line in stream.read().splitlines()]", 1)
        script = script.replace("with log_path.open('a') as stream:\n    stream.write(json.dumps(record) + '\\n')",
                                "with log_path.open('a') as stream:\n"
                                "    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)\n"
                                "    stream.write(json.dumps(record) + '\\n')\n" + CONTROL, 1)
        script = script.replace("if role == 'worker':\n", "if role == 'worker' and not request:\n", 1)
        script = script.replace("        if number > 1:\n            current['meta_data']['version'] += 1",
                                "        if not context['first_draft']:\n            current['meta_data']['version'] += 1", 1)
        script = script.replace("    if stage not in ('P4', 'P6'):\n",
                                "    if stage == 'P1' and control.get('contract_correction'):\n"
                                "        current['contract']['algorithm_requirement']['output']['Y']['meaning'] = "
                                "'Output scalar Y, copied exactly from the fixed input X.'\n"
                                "    if stage not in ('P4', 'P6'):\n", 1)
        script = script.replace("output.write_text(json.dumps(receipt))", REOPEN_OUTPUT, 1)
        compile(script, '<D-boundary-cli>', 'exec')
        self.fake.write_text(script, encoding='utf-8')
        self.scenario['control'] = control or {}
        (self.base / 'scenario.json').write_text(json.dumps(self.scenario), encoding='utf-8')

    def create(self, **config):
        state = super().create(call_seconds=config.pop('call_seconds', 25), **config)
        self.initial = deepcopy(state)
        return state

    def mark(self, key, event):
        return read_json(self.base / 'marks' / (key + '.' + event + '.json'))


class DRunnerTests(unittest.TestCase):
    def fixture(self, **kwargs):
        fixture = DFixture(**kwargs)
        self.addCleanup(fixture.cleanup)
        return fixture

    def test_authorized_root_when_can_be_added_and_removed_without_changing_other_fields(self):
        starter = Path(__file__).resolve().parents[2] / 'schema_template/starter.yaml'
        original = yaml.safe_load(starter.read_text())
        original['meta_data']['name'] = 'Example'
        added = deepcopy(original)
        added['meta_data'].update(version=2, dispatch={'when': 'Select exactly when N > 1.'})
        removed = deepcopy(added)
        removed['meta_data'].pop('dispatch')
        removed['meta_data']['version'] = 3
        with TemporaryDirectory(prefix='ks-d-when-scope-') as temporary:
            before, after = Path(temporary) / 'before', Path(temporary) / 'after'
            before.mkdir()
            after.mkdir()
            def gate(old, new, authorized=True):
                write_yaml(before / 'root.yaml', old)
                write_yaml(after / 'root.yaml', new)
                return validate_artifacts('P3', 'worker', before, after, starter,
                                          decision='continue', root_revision=authorized)
            self.assertEqual(gate(original, added)['changed_files'], ['root.yaml'])
            self.assertEqual(gate(added, removed)['changed_files'], ['root.yaml'])
            with self.assertRaisesRegex(ExtractError, 'P3 changed'):
                gate(original, added, authorized=False)
            for field in ('id', 'parent', 'source', 'contract'):
                with self.subTest(field=field):
                    changed = deepcopy(added)
                    if field in {'id', 'parent'}:
                        changed['meta_data']['dispatch'][field] = 'root' if field == 'id' else ''
                    elif field == 'source':
                        changed['meta_data']['source'] = 'another-source'
                    else:
                        changed['contract']['algorithm_requirement']['compute'][0]['math'] = 'Y = 42.'
                    with self.assertRaisesRegex(ExtractError, 'P3 changed'):
                        gate(original, changed)

    def test_old_call_survives_an_unrelated_plan_reopen(self):
        fixture = self.fixture(pairs=(('requester', 'root'), ('kept', 'root')), control={
            'P5.requester.worker.normal.1': {'request': {'target': 'P4', 'issue_id': 'check-unchanged-tree'},
                                             'wait_for': ['P5.kept.worker.normal.1.begin']},
            'P5.kept.worker.normal.1': {'delay': 0.35},
            'P4.root.worker.normal.2': {'wait_for': ['P5.kept.worker.normal.1.end'], 'delay': 0.35},
        })
        state = fixture.run()
        self.assertEqual(state['status'], 'ready', state['reason'])
        old = fixture.calls(stage='P5', node='kept', role='worker')[0]['context']['identity']
        self.assertEqual(old['plan_revision'], 1)
        self.assertEqual(state['plan_revision'], 2)
        self.assertIn(old['attempt_id'], state['accepted_receipts'])
        self.assertNotIn(old['attempt_id'], state['stale_results'])
        self.assertEqual(state['nodes']['kept']['generation'], 0)
        self.assertEqual(len(fixture.calls(stage='P5', node='kept', role='worker')), 1)
        self.assertLess(fixture.mark('P5.kept.worker.normal.1', 'end')['time'],
                        fixture.mark('P4.root.worker.normal.2', 'end')['time'])
        self.assertEqual(state['p6_starts'], 1)

    def test_rephrasing_the_same_fixed_evidence_cannot_reopen_twice(self):
        request = {'target': 'P4', 'issue_id': 'same-host-issue', 'line': 2}
        fixture = self.fixture(pairs=(('requester', 'root'), ('safe', 'root')), control={
            'P5.requester.worker.normal.1': {'request': request},
            'P5.requester.worker.normal.2': {'request': {**request, 'short_quote': True, 'note': 'Reworded note.'}},
        })
        state = fixture.run()
        self.assertEqual(state['status'], 'blocked', state['reason'])
        self.assertEqual(state['nodes']['requester']['reason']['code'], 'reopen_already_used')
        self.assertEqual(len(state['reopen_requests']), 1)
        self.assertEqual(state['plan_revision'], 2)
        self.assertEqual(state['stages']['P4']['calls'], 4)
        self.assertEqual(state['nodes']['requester']['stage']['calls'], 2)
        self.assertEqual(state['nodes']['safe']['status'], 'approved')
        self.assertEqual(state['deadline'], fixture.initial['deadline'])
        p4 = fixture.calls(stage='P4', role='worker')
        self.assertEqual(p4[0]['context']['budget']['deadline'], p4[1]['context']['budget']['deadline'])
        self.assertFalse(fixture.calls(stage='P6'))

    def test_a_known_source_anchor_is_not_new_reopening_evidence(self):
        fixture = self.fixture(pairs=(('requester', 'root'), ('safe', 'root')), control={
            'P5.requester.worker.normal.1': {'request': {'target': 'P4', 'issue_id': 'old-evidence', 'line': 1}},
        })
        state = fixture.run()
        self.assertEqual(state['status'], 'blocked', state['reason'])
        self.assertEqual(state['nodes']['requester']['reason']['code'], 'reopen_requires_new_evidence')
        self.assertEqual(state['plan_revision'], 1)
        self.assertFalse(state['reopen_requests'])
        self.assertEqual(state['nodes']['safe']['status'], 'approved')
        self.assertEqual(len(fixture.calls(stage='P4')), 2)

    def test_external_requirement_change_blocks_with_confirmation_condition(self):
        fixture = self.fixture(pairs=(('requester', 'root'), ('safe', 'root')), control={
            'P5.requester.worker.normal.1': {'request': {'target': 'external', 'issue_id': 'changed-output-requirement'}},
        })
        state = fixture.run()
        self.assertEqual(state['status'], 'blocked', state['reason'])
        reason = state['nodes']['requester']['reason']
        self.assertEqual(reason['code'], 'external_confirmation')
        self.assertIn('confirm', reason['issues'][0]['recovery_condition'])
        self.assertEqual(state['nodes']['safe']['status'], 'approved')
        self.assertEqual(len(fixture.calls(stage='P1')), 2)
        self.assertEqual(state['plan_revision'], 1)
        self.assertFalse(state['reopen_requests'])
        merged, _ = fixture.store.verify_checkpoint(state, state['merged_checkpoint'])
        p3, _ = fixture.store.verify_checkpoint(state, state['stages']['P3']['approval']['checkpoint'])
        self.assertEqual((merged / 'root.yaml').read_bytes(), (p3 / 'root.yaml').read_bytes())
        self.assertFalse(fixture.calls(stage='P6'))

    def test_contract_reopen_reviews_all_downstream_stages_without_resetting_budgets(self):
        fixture = self.fixture(pairs=(('requester', 'root'), ('safe', 'root')), control={
            'P5.requester.worker.normal.1': {'request': {'target': 'P1', 'issue_id': 'source-transcription', 'line': 1,
                                                         'kind': 'transcription', 'scope': 'root.yaml:contract.algorithm_requirement'},
                                             'wait_for': ['P5.safe.reviewer.normal.1.end'], 'delay': 0.7},
            'P1.root.worker.normal.2': {'contract_correction': True},
        })
        state = fixture.run()
        self.assertEqual(state['status'], 'ready', state['reason'])
        self.assertEqual(state['plan_revision'], 2)
        for stage in ('P1', 'P2', 'P3', 'P4'):
            self.assertEqual(len(fixture.calls(stage=stage)), 4)
            self.assertEqual(state['stages'][stage]['calls'], 4)
            first, second = fixture.calls(stage=stage, role='worker')
            self.assertEqual(first['context']['budget']['deadline'], second['context']['budget']['deadline'])
        self.assertEqual(len(fixture.calls(stage='P5', node='safe', role='worker')), 2)
        self.assertGreater(state['nodes']['safe']['generation'], 0)
        artifacts, _ = fixture.store.verify_checkpoint(state)
        root = yaml.safe_load((artifacts / 'root.yaml').read_text())
        self.assertIn('copied exactly', root['contract']['algorithm_requirement']['output']['Y']['meaning'])
        self.assertEqual(state['p6_starts'], 1)
        self.assertEqual(state['deadline'], fixture.initial['deadline'])

    def test_serial_commits_use_latest_authority_and_replays_cannot_replace_it(self):
        fixture = self.fixture(pairs=(('a', 'root'), ('b', 'root'), ('c', 'root'), ('leaf', 'c')))
        fixture.create()
        writers, old_state = set(), []
        save = fixture.store.save
        def record_writer(state):
            writers.add(threading.get_ident())
            save(state)
        def deliver(store, state, lock, attempt, result):
            if attempt['stage'] == 'P5':
                if not old_state:
                    old_state.append(deepcopy(state))
                state = old_state[0]
            return accept_result(store, state, lock, attempt, result)
        with patch.object(fixture.store, 'save', side_effect=record_writer), \
                patch('schema_extract.runner.accept_result', side_effect=deliver):
            state = run(fixture.store, fixture.state['run_id'])
        self.assertEqual(state['status'], 'ready', state['reason'])
        self.assertEqual(writers, {threading.get_ident()})
        self.assertEqual(summary(fixture.store, state)['completed_node_count'], 5)
        original = deepcopy(state)
        worker = fixture.calls(stage='P5', node='a', role='worker')[0]['context']['identity']['attempt_id']
        directory = fixture.store.run_dir(state['run_id']) / 'attempts' / worker
        with self.assertRaisesRegex(ExtractError, 'duplicate'):
            accept_result(fixture.store, old_state[0], fixture.lock,
                          read_json(directory / 'attempt.json'), read_json(directory / 'result.json'))
        self.assertEqual(fixture.store.load(state['run_id']), original)
        # Another node's later merge contains earlier files, but its approval
        # cannot replace the missing independent review for one of those files.
        forged = deepcopy(state)
        forged['nodes']['a']['approval'] = forged['nodes']['b']['approval']
        with self.assertRaisesRegex(ExtractError, 'another node'):
            fixture.store.verify_node(forged, 'a')


if __name__ == '__main__':
    unittest.main()
