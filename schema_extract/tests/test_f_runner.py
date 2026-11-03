"""Real-run regressions using independent CLI output and the frozen checker."""

from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from schema_extract.common import manifest, read_json
from schema_extract.receipts import QuoteTextMismatch, _anchor
from schema_extract.runner import run
from test_c_independent import IndependentFixture


class RealRunBoundaryTests(unittest.TestCase):
    def fixture(self, *, unfinished=False, manifest_cases=False, **kwargs):
        fixture = IndependentFixture(**kwargs)
        self.addCleanup(fixture.cleanup)
        if unfinished:
            # The controlled process emits a parseable but incomplete Design
            # with a blocked request, without importing Runner implementation.
            script = fixture.fake.read_text()
            script = script.replace("if action == 'bad_design':",
                                    "if action in ('bad_design', 'blocked_draft'):")
            script = script.replace("if action in ('continue', 'revise', 'blocked'):",
                                    "if action == 'blocked_draft':\n    decision = 'blocked'\n"
                                    "elif action in ('continue', 'revise', 'blocked'):")
            fixture.fake.write_text(script)
        if manifest_cases:
            script = fixture.fake.read_text().replace(
                "if role == 'worker':\n    current =",
                "if role == 'worker' and action not in ('receipt_only', 'omit_readonly_again', 'quote_again'):\n    current =")
            marker = "output = Path(sys.argv[sys.argv.index('--output-last-message') + 1])"
            assert marker in script
            script = script.replace(marker, r'''
if action.startswith('omit_readonly'):
    omitted = next(item['path'] for item in receipt['artifacts']
                   if item['path'] not in context['writable_files'])
    receipt['artifacts'] = [item for item in receipt['artifacts'] if item['path'] != omitted]
    if action == 'omit_readonly_bad_hash':
        receipt['artifacts'][0]['sha256'] = '0' * 64
    elif action == 'omit_readonly_duplicate':
        receipt['artifacts'].append(dict(receipt['artifacts'][0]))
    elif action in ('omit_readonly_extra', 'omit_readonly_escape'):
        receipt['artifacts'].append({'path': 'unexpected.yaml' if action == 'omit_readonly_extra' else '../root.yaml',
                                     'sha256': receipt['artifacts'][0]['sha256']})
    elif action == 'omit_readonly_bad_source':
        receipt['source_evidence'][0]['sha256'] = '0' * 64
elif action == 'omit_writable':
    receipt['artifacts'] = [item for item in receipt['artifacts'] if item['path'] not in context['writable_files']]
elif action == 'omit_review_evidence':
    receipt['artifact_evidence'] = [item for item in receipt['artifact_evidence'] if item['path'] != 'dispatch.md']
elif action == 'short_review_digest':
    receipt['artifact_evidence'][0]['sha256'] = receipt['artifact_evidence'][0]['sha256'][:-2]
elif action == 'swapped_review_digest':
    evidence = receipt['artifact_evidence'][0]
    evidence['sha256'] = next(item['sha256'] for item in receipt['artifacts'] if item['path'] != evidence['path'])
elif action.startswith('quote_'):
    if action != 'quote_artifact':
        receipt['source_evidence'][0]['quote'] += ' +p'
    if action in ('quote_artifact', 'quote_both'):
        receipt['artifact_evidence'][0]['quote'] += ' +p'
    if action == 'quote_then_bad_hash':
        receipt['source_evidence'][1]['sha256'] = '0' * 64
    elif action == 'quote_then_bad_path':
        receipt['source_evidence'][1]['path'] = '../outside.py'
    elif action == 'quote_then_bad_range':
        receipt['source_evidence'][1]['line_start'] = 0
    elif action == 'quote_then_empty':
        receipt['source_evidence'][1]['quote'] = ''
    elif action == 'quote_then_overlong':
        receipt['source_evidence'][1]['quote'] = 'x' * 321
''' + '\n' + marker)
            fixture.fake.write_text(script)
        return fixture

    def assert_ready(self, fixture, state):
        self.assertEqual(state['status'], 'ready', state.get('reason'))
        self.assertTrue(state['published'])
        self.assertEqual(state['active_attempts'], {})
        for stage in ('P1', 'P2', 'P3', 'P4', 'P6'):
            fixture.store.verify_approval(state, state['stages'][stage]['approval'])
        receipt = read_json(Path(state['output']) / '.extraction/receipt.json')
        self.assertEqual(receipt['checkpoint'], state['merged_checkpoint'])

    def receipt(self, call):
        return read_json(Path(call['context']['workspace']).parent / 'receipt.json')

    def test_worker_blocked_is_independently_reviewed_in_every_stage(self):
        actions = {f'{stage}.root.worker.normal.1': 'blocked' for stage in ('P1', 'P2', 'P3', 'P4', 'P6')}
        actions.update({'P5.branch.worker.normal.1': 'blocked', 'P6.root.reviewer.normal.1': 'revise'})
        fixture = self.fixture(actions=actions, pairs=(('branch', 'root'), ('leaf', 'branch')),
                               final_edit_node='branch')
        state = fixture.run()
        self.assert_ready(fixture, state)
        for stage, node in [(stage, 'root') for stage in ('P1', 'P2', 'P3', 'P4', 'P6')] + [('P5', 'branch')]:
            worker = fixture.calls(stage=stage, node=node, role='worker')[0]
            reviewer = fixture.calls(stage=stage, node=node, role='reviewer')[-1]
            request = self.receipt(worker)
            self.assertEqual(request['decision'], 'blocked')
            self.assertIn(request['issues'][0], reviewer['context']['feedback'])
            self.assertNotEqual(worker['session_id'], reviewer['session_id'])
            self.assertEqual(self.receipt(reviewer)['decision'], 'approve')
            expected_evidence = ({'root.yaml'} if stage == 'P1' else
                                 {'root.yaml', 'dispatch.md'} if stage in {'P2', 'P3', 'P4'} else
                                 {'root.yaml', 'branch.yaml'} if stage == 'P5' else
                                 {'root.yaml', 'branch.yaml', 'leaf.yaml', 'dispatch.md'})
            self.assertEqual(set(reviewer['context']['required_artifact_evidence']), expected_evidence)
            for item in request['artifacts']:
                self.assertEqual(reviewer['before_hashes'][item['path']], item['sha256'])
        for stage in ('P1', 'P2', 'P3', 'P4'):
            self.assertEqual(state['stages'][stage]['revisions'], 0)
        self.assertEqual(state['nodes']['branch']['stage']['revisions'], 0)
        self.assertEqual(state['stages']['P6']['revisions'], 1)
        p4_review = fixture.calls(stage='P4', role='reviewer')[0]
        self.assertEqual(p4_review['context']['plan'], fixture.scenario['plan'])
        final_review = fixture.calls(stage='P6', role='reviewer')[-1]
        self.assertEqual(set(final_review['context']['pending_rechecks']), {'branch', 'leaf'})
        leaf_review = fixture.calls(stage='P5', node='leaf', role='reviewer')[0]
        self.assertEqual(set(leaf_review['context']['required_artifact_evidence']), {'root.yaml', 'branch.yaml', 'leaf.yaml'})

    def test_reviewer_confirms_real_blocking_without_releasing_descendants(self):
        for stage, node in (('P1', 'root'), ('P5', 'branch')):
            with self.subTest(stage=stage):
                fixture = self.fixture(
                    pairs=(('branch', 'root'), ('leaf', 'branch'), ('sibling', 'root')),
                    actions={f'{stage}.{node}.worker.normal.1': 'blocked',
                             f'{stage}.{node}.reviewer.normal.1': 'blocked'})
                state = fixture.run()
                self.assertEqual(state['status'], 'blocked', state.get('reason'))
                self.assertFalse(state['published'])
                reviews = fixture.calls(stage=stage, node=node, role='reviewer')
                self.assertEqual(len(reviews), 1)
                self.assertEqual(self.receipt(reviews[0])['decision'], 'blocked')
                self.assertFalse(fixture.calls(stage='P6'))
                if stage == 'P1':
                    self.assertFalse(fixture.calls(stage='P2'))
                    self.assertIsNone(state['stages']['P1']['approval'])
                else:
                    self.assertEqual(state['nodes']['sibling']['status'], 'approved')
                    self.assertEqual(state['nodes']['branch']['status'], 'blocked')
                    self.assertFalse(fixture.calls(stage='P5', node='leaf'))

    def test_incomplete_blocked_designs_return_to_author_as_counted_revisions(self):
        fixture = self.fixture(unfinished=True, pairs=(('branch', 'root'),), actions={
            'P3.root.worker.normal.1': 'blocked_draft', 'P3.root.reviewer.normal.1': 'revise',
            'P5.branch.worker.normal.1': 'blocked_draft', 'P5.branch.reviewer.normal.1': 'revise'})
        state = fixture.run()
        self.assert_ready(fixture, state)
        for stage, node in (('P3', 'root'), ('P5', 'branch')):
            scope = state['stages'][stage] if stage == 'P3' else state['nodes'][node]['stage']
            self.assertEqual((scope['check_failures'], scope['revisions'], scope['adjudications']), (1, 1, 0))
            workers = fixture.calls(stage=stage, node=node, role='worker')
            reviews = fixture.calls(stage=stage, node=node, role='reviewer')
            self.assertEqual((len(workers), len(reviews)), (2, 2))
            self.assertEqual(workers[1]['resumed'], workers[0]['session_id'])
            self.assertEqual(reviews[1]['resumed'], reviews[0]['session_id'])
            first_check = read_json(Path(reviews[0]['context']['workspace']).parent / 'check.json')
            self.assertEqual(first_check['exit_code'], 1)
            self.assertEqual(workers[1]['context']['feedback'][0]['check'], first_check)
            self.assertEqual(workers[1]['context']['budget']['deadline'], workers[0]['context']['budget']['deadline'])
            approval_check = read_json(Path(reviews[1]['context']['workspace']).parent / 'check.json')
            self.assertEqual(approval_check['exit_code'], 0)

    def test_draft_reviews_keep_approval_tool_error_and_single_adjudication_gates(self):
        for outcome in ('invalid_approval', 'adjudication', 'tool_error'):
            with self.subTest(outcome=outcome):
                actions = {'P3.root.worker.normal.1': 'blocked_draft'}
                if outcome != 'invalid_approval':
                    actions['P3.root.reviewer.normal.1'] = 'revise'
                fixture = self.fixture(unfinished=True, actions=actions)
                fixture.create(revision_limit=1)
                original_run = subprocess.run

                def checker_error(command, *args, **kwargs):
                    if isinstance(command, list) and any(str(part).endswith('schema_template/check_yaml.py') for part in command):
                        command = [*command, '--invalid-checker-argument']
                    return original_run(command, *args, **kwargs)

                if outcome == 'tool_error':
                    with patch('schema_extract.artifacts.subprocess.run', side_effect=checker_error):
                        state = run(fixture.store, fixture.state['run_id'])
                else:
                    state = run(fixture.store, fixture.state['run_id'])
                scope = state['stages']['P3']
                reviews = fixture.calls(stage='P3', role='reviewer', mode='normal')
                if outcome == 'invalid_approval':
                    self.assert_ready(fixture, state)
                    self.assertEqual(scope['check_failures'], 1)
                    self.assertEqual(scope['revisions'], 0)
                    rejected_approval = self.receipt(reviews[0])
                    self.assertEqual(rejected_approval['decision'], 'approve')
                    self.assertNotEqual(scope['approval']['attempt_id'], rejected_approval['identity']['attempt_id'])
                    self.assertEqual(len(fixture.calls(stage='P3', role='worker')), 2)
                elif outcome == 'adjudication':
                    self.assertEqual(state['status'], 'blocked')
                    self.assertEqual((scope['revisions'], scope['adjudications'], scope['check_failures']), (1, 1, 2))
                    self.assertEqual(len(fixture.calls(stage='P3', mode='adjudicate')), 1)
                    self.assertEqual(scope['adjudication']['decision'], 'blocked')
                    self.assertFalse(fixture.calls(stage='P4'))
                    calls = len(fixture.calls())
                    resumed = run(fixture.store, state['run_id'], resume=True)
                    self.assertEqual(resumed['status'], 'blocked')
                    self.assertEqual(len(fixture.calls()), calls)
                else:
                    self.assertEqual(state['status'], 'tool_error')
                    self.assertEqual((scope['tool_failures'], scope['revisions'], scope['check_failures']), (1, 0, 0))
                    self.assertEqual(len(reviews), 1)
                    self.assertFalse(fixture.calls(stage='P4'))

    def test_initial_root_context_and_version_gate_share_the_checkpoint_baseline(self):
        fixture = self.fixture(pairs=(('branch', 'root'),), actions={'P1.root.worker.normal.1': 'continue'})
        state = fixture.run()
        self.assert_ready(fixture, state)
        workers = fixture.calls(stage='P1', role='worker')
        self.assertEqual([call['context']['first_draft'] for call in workers], [True, False])
        self.assertEqual(workers[1]['before']['root.yaml']['meta_data']['version'], 1)
        self.assertTrue(fixture.calls(stage='P5', role='worker')[0]['context']['first_draft'])
        self.assertFalse(fixture.calls(stage='P5', role='reviewer')[0]['context']['first_draft'])

        invalid = self.fixture(actions={'P1.root.worker.normal.1': 'blocked'})
        script = invalid.fake.read_text().replace("= 1 if first else", "= 2 if first else")
        invalid.fake.write_text(script)
        rejected = invalid.run()
        self.assertEqual(rejected['status'], 'invalid_receipt')
        self.assertIn('new node starts at version 1', rejected['reason']['summary'])
        self.assertTrue(invalid.calls(stage='P1', role='worker')[0]['context']['first_draft'])
        self.assertFalse(invalid.calls(role='reviewer'))
        self.assertFalse(rejected['accepted_receipts'])

    def test_missing_readonly_manifest_members_repair_in_place_with_exact_sessions(self):
        fixture = self.fixture(manifest_cases=True, actions={
            'P3.root.worker.normal.1': 'omit_readonly', 'P3.root.worker.normal.2': 'receipt_only',
            'P3.root.reviewer.normal.1': 'omit_readonly', 'P3.root.reviewer.normal.2': 'receipt_only',
            'P4.root.worker.normal.1': 'omit_readonly', 'P4.root.worker.normal.2': 'receipt_only',
            'P6.root.reviewer.normal.1': 'omit_readonly', 'P6.root.reviewer.normal.2': 'receipt_only'})
        state = fixture.run()
        self.assert_ready(fixture, state)
        for stage, role in (('P3', 'worker'), ('P3', 'reviewer'), ('P4', 'worker'), ('P6', 'reviewer')):
            calls = fixture.calls(stage=stage, role=role)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[1]['resumed'], calls[0]['session_id'])
            original = Path(calls[0]['context']['workspace'])
            corrected = Path(calls[1]['context']['workspace'])
            self.assertEqual(manifest(original), calls[1]['before_hashes'])
            self.assertEqual(manifest(original), manifest(corrected))
            self.assertEqual(self.receipt(calls[1])['changed_files'], [])
            self.assertEqual({item['path']: item['sha256'] for item in self.receipt(calls[1])['artifacts']}, manifest(corrected))
            self.assertEqual(calls[1]['context']['budget']['deadline'], calls[0]['context']['budget']['deadline'])
            first_id = calls[0]['context']['identity']['attempt_id']
            self.assertNotIn(first_id, state['accepted_receipts'])
            self.assertTrue((original.parent / 'repair.json').is_file())
            properties = read_json(original.parent / 'response.schema.json')['properties']
            for name in ('artifacts', 'artifact_evidence'):
                item = properties[name]['items']['properties']
                if role == 'reviewer':
                    fixed = calls[0]['context']['identity']['dependency_digests']
                    self.assertEqual(set(item['path']['enum']), set(fixed))
                    self.assertEqual(set(item['sha256']['enum']), set(fixed.values()))
                else:
                    self.assertNotIn('enum', item['sha256'])
        self.assertEqual(state['stages']['P3']['receipt_repairs'], 2)
        self.assertEqual(state['stages']['P4']['receipt_repairs'], 1)
        self.assertEqual(state['stages']['P6']['receipt_repairs'], 1)
        self.assertTrue(all(scope['revisions'] == 0 for scope in state['stages'].values()))

    def test_manifest_omission_cannot_hide_wrong_values_paths_or_source_evidence(self):
        for action in ('omit_writable', 'omit_readonly_bad_hash', 'omit_readonly_duplicate',
                       'omit_readonly_extra', 'omit_readonly_escape', 'omit_readonly_bad_source',
                       'omit_review_evidence', 'short_review_digest', 'swapped_review_digest'):
            with self.subTest(action=action):
                role = 'reviewer' if action in {'omit_review_evidence', 'short_review_digest', 'swapped_review_digest'} else 'worker'
                fixture = self.fixture(manifest_cases=True, actions={f'P3.root.{role}.normal.1': action})
                state = fixture.run()
                self.assertEqual(state['status'], 'invalid_receipt', state.get('reason'))
                self.assertEqual(state['stages']['P3']['receipt_repairs'], 0)
                self.assertIsNone(state['stages']['P3']['approval'])
                self.assertEqual(len(fixture.calls(stage='P3', role='reviewer')), int(role == 'reviewer'))
                self.assertFalse(fixture.calls(stage='P4'))
                self.assertFalse(state['published'])

    def test_manifest_repairs_keep_retry_budget_and_do_not_repeat_once_only_roles(self):
        fixture = self.fixture(manifest_cases=True, actions={
            'P3.root.worker.normal.1': 'omit_readonly',
            'P3.root.worker.normal.*': 'omit_readonly_again'})
        state = fixture.run()
        self.assertEqual(state['status'], 'invalid_receipt')
        self.assertEqual(state['reason']['code'], 'receipt_repair_budget_exhausted')
        self.assertEqual(state['stages']['P3']['receipt_repairs'], 3)
        calls = fixture.calls(stage='P3', role='worker')
        self.assertEqual(len(calls), 3)
        self.assertEqual({call['session_id'] for call in calls}, {calls[0]['session_id']})
        self.assertEqual({call['context']['budget']['deadline'] for call in calls}, {calls[0]['context']['budget']['deadline']})
        self.assertTrue(all(manifest(Path(call['context']['workspace'])) == manifest(Path(calls[0]['context']['workspace'])) for call in calls))
        self.assertFalse(fixture.calls(stage='P3', role='reviewer'))

        for mode in ('adjudicate', 'signoff'):
            with self.subTest(mode=mode):
                actions = ({'P3.root.reviewer.normal.1': 'revise',
                            'P3.root.reviewer.adjudicate.1': 'omit_readonly'} if mode == 'adjudicate' else
                           {'P6.root.reviewer.normal.1': 'revise',
                            'P6.root.reviewer.adjudicate.1': 'patch',
                            'P6.root.reviewer.signoff.1': 'omit_readonly'})
                fixture = self.fixture(manifest_cases=True, actions=actions)
                state = fixture.run(revision_limit=1)
                self.assertEqual(state['status'], 'blocked', state.get('reason'))
                self.assertEqual(len(fixture.calls(mode=mode)), 1)
                stage = 'P3' if mode == 'adjudicate' else 'P6'
                self.assertEqual(state['stages'][stage]['receipt_repairs'], 0)
                self.assertFalse(state['published'])

    def test_quote_mismatch_is_rejected_then_exact_resubmission_can_advance(self):
        fixture = self.fixture(manifest_cases=True, actions={
            'P3.root.worker.normal.1': 'quote_source', 'P3.root.worker.normal.2': 'receipt_only',
            'P3.root.reviewer.normal.1': 'quote_both', 'P3.root.reviewer.normal.2': 'receipt_only'})
        # Reproduce the observed missing "+p" against an actual fixed file.
        probe = fixture.base / 'quote-probe'
        probe.mkdir()
        text = 'state has strides (HV*Dv*Dk+p,Dv*Dk,Dk,1), where p is any integer in [0,65536].'
        (probe / 'layout.txt').write_text(text + '\n')
        hashes = manifest(probe)
        anchor = {'path': 'layout.txt', 'sha256': hashes['layout.txt'], 'line_start': 1,
                  'line_end': 1, 'quote': text.replace('+p', ''), 'note': 'Fixed stride formula.'}
        with self.assertRaises(QuoteTextMismatch):
            _anchor(anchor, probe, hashes)
        _anchor({**anchor, 'quote': text}, probe, hashes)

        state = fixture.run()
        self.assert_ready(fixture, state)
        scope = state['stages']['P3']
        self.assertEqual((scope['receipt_repairs'], scope['revisions'], scope['calls']), (2, 0, 4))
        for role in ('worker', 'reviewer'):
            calls = fixture.calls(stage='P3', role=role)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[1]['resumed'], calls[0]['session_id'])
            first = Path(calls[0]['context']['workspace'])
            second = Path(calls[1]['context']['workspace'])
            self.assertEqual(manifest(first), manifest(second))
            self.assertEqual(calls[1]['before_hashes'], manifest(first))
            self.assertEqual(calls[0]['context']['budget']['deadline'], calls[1]['context']['budget']['deadline'])
            first_id = calls[0]['context']['identity']['attempt_id']
            self.assertNotIn(first_id, state['accepted_receipts'])
            self.assertFalse((first.parent / 'receipt.json').exists())
            correction = read_json(first.parent / 'repair.json')['summary']
            self.assertIn('source_evidence[0].quote', correction)
            self.assertIn('重新核对', correction)
            if role == 'reviewer':
                self.assertIn('artifact_evidence[0].quote', correction)
            self.assertEqual(self.receipt(calls[1])['changed_files'], [])
            self.assertEqual(read_json(second.parent / 'check.json')['exit_code'], 0)
        approval = state['stages']['P3']['approval']['attempt_id']
        self.assertEqual(approval, fixture.calls(stage='P3', role='reviewer')[-1]['context']['identity']['attempt_id'])

    def test_quote_repair_does_not_hide_hard_errors_or_reset_limits(self):
        for action in ('quote_then_bad_hash', 'quote_then_bad_path', 'quote_then_bad_range',
                       'quote_then_empty', 'quote_then_overlong'):
            with self.subTest(action=action):
                fixture = self.fixture(manifest_cases=True, actions={'P3.root.worker.normal.1': action})
                state = fixture.run()
                self.assertEqual(state['status'], 'invalid_receipt')
                self.assertEqual(state['stages']['P3']['receipt_repairs'], 0)
                self.assertFalse(fixture.calls(stage='P3', role='reviewer'))
                self.assertFalse(state['published'])

        fixture = self.fixture(manifest_cases=True, actions={
            'P3.root.worker.normal.1': 'quote_source', 'P3.root.worker.normal.*': 'quote_again'})
        state = fixture.run()
        self.assertEqual(state['status'], 'invalid_receipt')
        self.assertEqual(state['reason']['code'], 'receipt_repair_budget_exhausted')
        calls = fixture.calls(stage='P3', role='worker')
        self.assertEqual(len(calls), 3)
        self.assertEqual(state['stages']['P3']['receipt_repairs'], 3)
        self.assertEqual({call['session_id'] for call in calls}, {calls[0]['session_id']})
        self.assertEqual({call['context']['budget']['deadline'] for call in calls}, {calls[0]['context']['budget']['deadline']})
        self.assertTrue(all(manifest(Path(call['context']['workspace'])) == manifest(Path(calls[0]['context']['workspace'])) for call in calls))
        self.assertFalse(fixture.calls(stage='P3', role='reviewer'))

        for mode in ('adjudicate', 'signoff'):
            with self.subTest(mode=mode):
                actions = ({'P3.root.reviewer.normal.1': 'revise',
                            'P3.root.reviewer.adjudicate.1': 'quote_artifact'} if mode == 'adjudicate' else
                           {'P6.root.reviewer.normal.1': 'revise',
                            'P6.root.reviewer.adjudicate.1': 'patch',
                            'P6.root.reviewer.signoff.1': 'quote_artifact'})
                fixture = self.fixture(manifest_cases=True, actions=actions)
                state = fixture.run(revision_limit=1)
                self.assertEqual(state['status'], 'blocked')
                self.assertEqual(len(fixture.calls(mode=mode)), 1)
                stage = 'P3' if mode == 'adjudicate' else 'P6'
                self.assertEqual(state['stages'][stage]['receipt_repairs'], 0)
                self.assertFalse(state['published'])


if __name__ == '__main__':
    unittest.main()
