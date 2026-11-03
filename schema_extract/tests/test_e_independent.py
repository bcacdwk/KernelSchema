"""Independent E recovery checks against real, separately owned CLI processes.

The executable model fixture is shared with independent D.  These tests kill
the Runner process, so neither a Python finally block nor its in-memory event
queue can perform recovery on its behalf.  Assertions use the persisted
artifacts, process state, role-call log, and budgets from a new Runner process.
"""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import unittest

from schema_extract.common import ExtractError, digest, manifest, read_json, write_json
from schema_extract.processes import process_table, terminate_owned
from schema_extract.runner import accept_result, summary
from schema_extract.tests.test_d_independent import IndependentDFixture


REPO = Path(__file__).resolve().parents[2]


RUNNER_DRIVER = r'''
import json
import os
from pathlib import Path
import sys
from schema_extract.common import write_json
from schema_extract.runner import run
from schema_extract.store import Store

state_root, run_id, resume, fault, marker = sys.argv[1:]
store = Store(Path(state_root))
if fault == 'after_steady_merge':
    real_save = Store.save
    def save_then_exit(self, state):
        real_save(self, state)
        node = state['nodes'].get('steady', {})
        if node.get('merged') and node.get('approval') and not Path(marker).exists():
            write_json(Path(marker), {'checkpoint': state['merged_checkpoint'],
                                     'approval': node['approval']})
            os._exit(86)
    Store.save = save_then_exit
state = run(store, run_id, **({'resume': True} if resume == 'yes' else {}))
print(json.dumps(state), flush=True)
'''


class IndependentEFixture(IndependentDFixture):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.runners = []
        self.original_process_records = []

    def runner(self, *, resume=False, fault=''):
        log = self.base / ('runner-' + str(len(self.runners) + 1) + '.log')
        with log.open('wb') as output:
            process = subprocess.Popen(
                [sys.executable, '-B', '-c', RUNNER_DRIVER, str(self.store.root),
                 self.state['run_id'], 'yes' if resume else 'no', fault,
                 str(self.base / 'fault.json')],
                cwd=Path(self.lock['snapshot']) / 'rules', stdout=output, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
            )
        process.log = log
        self.runners.append(process)
        return process

    def wait_until(self, process, condition, *, timeout=45, description='expected persisted boundary'):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = condition()
            if value:
                return value
            if process.poll() is not None:
                raise AssertionError(f'Runner exited before {description}:\n{process.log.read_text()}')
            time.sleep(0.025)
        raise AssertionError(f'Runner did not reach {description}:\n{process.log.read_text()}')

    def wait_marker(self, process, stage, node, role, number=1, *, event='begin', mode='normal'):
        path = self.base / 'markers' / f'{stage}.{node}.{role}.{mode}.{number}.{event}.json'
        return self.wait_until(process, lambda: read_json(path) if path.exists() else None,
                               description=path.name)

    def finish(self, process, *, timeout=60):
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise AssertionError(f'Runner did not finish:\n{process.log.read_text()}') from exc
        if process.returncode:
            raise AssertionError(f'Runner failed ({process.returncode}):\n{process.log.read_text()}')
        self.state = self.store.load(self.state['run_id'])
        return self.state

    def remember_process(self, attempt_id):
        directory = self.store.run_dir(self.state['run_id']) / 'attempts' / attempt_id
        record = read_json(directory / 'process.json')
        self.original_process_records.append(deepcopy(record))
        return record

    def cleanup(self):
        for runner in self.runners:
            if runner.poll() is None:
                runner.kill()
            runner.wait(timeout=5)
        records = list(self.original_process_records)
        if self.store is not None:
            for path in self.store.run_dir(self.state['run_id']).glob('attempts/*/process.json'):
                try:
                    records.append(read_json(path))
                except ExtractError:
                    pass
        for record in records:
            terminate_owned(record['identity'], record.get('descendants', []), grace=0.1)
        super().cleanup()


def live_pid(pid):
    """Observe the OS separately from the Runner's saved termination claim."""
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'stat='], capture_output=True, text=True, timeout=5)
    return result.returncode == 0 and bool(result.stdout.strip()) and not result.stdout.strip().startswith('Z')


class IndependentETests(unittest.TestCase):
    def fixture(self, **kwargs):
        fixture = IndependentEFixture(**kwargs)
        self.addCleanup(fixture.cleanup)
        return fixture

    def assert_ready(self, fixture, state):
        self.assertEqual(state['status'], 'ready', state.get('reason'))
        planned = {entry['id'] for entry in state['plan']}
        self.assertEqual(set(summary(fixture.store, state)['approved_nodes']), planned)
        self.assertEqual(state['active_attempts'], {})
        artifacts, checkpoint = fixture.store.verify_checkpoint(state)
        self.assertEqual(checkpoint['files'], manifest(artifacts))
        self.assertEqual(state['checkpoint'], state['merged_checkpoint'])
        for node_id in planned:
            fixture.store.verify_node(state, node_id)
        self.assertEqual(state['p6_starts'], 1)
        self.assertTrue(state['published'])
        output = Path(state['output'])
        published = manifest(output)
        self.assertEqual(set(published), {*checkpoint['files'], '.extraction/receipt.json'})
        self.assertEqual({name: published[name] for name in checkpoint['files']}, checkpoint['files'])
        proof = read_json(output / '.extraction/receipt.json')
        self.assertEqual(proof['files'], checkpoint['files'])
        self.assertEqual(proof['checkpoint'], checkpoint['id'])
        self.assertEqual(proof['input_fingerprint'], state['input_fingerprint'])
        self.assertEqual(proof['approval']['receipt_sha256'], state['stages']['P6']['approval']['receipt_sha256'])
        self.assertEqual(state['publication']['proof'], proof)
        self.assertEqual(state['publication']['checkpoint'], state['checkpoint'])
        return artifacts

    def assert_budgets_preserved(self, before, after):
        for key in ('created_at', 'deadline', 'config'):
            self.assertEqual(before[key], after[key], key)
        for stage, old in before['stages'].items():
            new = after['stages'][stage]
            if old['started_at'] is not None:
                for key in ('started_at', 'deadline'):
                    self.assertEqual(old[key], new[key], (stage, key))
            for key in ('calls', 'revisions', 'continues', 'adjudications', 'signoffs', 'elapsed_seconds'):
                self.assertGreaterEqual(new[key], old[key], (stage, key))
        for node_id, old_node in before['nodes'].items():
            if 'stage' not in old_node:
                continue
            old, new = old_node['stage'], after['nodes'][node_id]['stage']
            if old['started_at'] is not None:
                for key in ('started_at', 'deadline'):
                    self.assertEqual(old[key], new[key], (node_id, key))
            for key in ('calls', 'revisions', 'continues', 'adjudications', 'signoffs', 'elapsed_seconds'):
                self.assertGreaterEqual(new[key], old[key], (node_id, key))

    def test_completed_deferred_review_survives_runner_sigkill_during_plan_reopen(self):
        fixture = self.fixture(
            pairs=(('reopener', 'root'), ('steady', 'root'), ('leaf', 'steady')),
            actions={'P5.reopener.worker.normal.1': 'reopen:P4:e-independent-deferred'},
            controls={
                'P5.reopener.worker.normal.1': {'wait_for': ['P5.steady.reviewer.normal.1.begin']},
                'P5.steady.reviewer.normal.1': {'wait_for': ['P4.root.worker.normal.2.begin']},
                'P4.root.worker.normal.2': {'delay': 120},
            },
        )
        fixture.create()
        first = fixture.runner()
        fixture.wait_marker(first, 'P4', 'root', 'worker', 2)
        fixture.wait_marker(first, 'P5', 'steady', 'reviewer', event='end')
        before = fixture.store.load(fixture.state['run_id'])
        self.assertIsNotNone(before['reopening'])
        review_id = before['active_attempts']['P5:steady']
        directory = fixture.store.run_dir(before['run_id']) / 'attempts' / review_id
        review_result = fixture.wait_until(
            first, lambda: read_json(directory / 'result.json') if (directory / 'result.json').exists() else None,
            description='completed deferred reviewer result on disk',
        )
        self.assertEqual(review_result['outcome'], 'completed')
        self.assertTrue(review_result['termination']['confirmed'])
        old_call = fixture.remember_process(before['active_attempts']['P4'])
        self.assertTrue(live_pid(old_call['identity']['pid']))
        first.kill()
        first.wait(timeout=5)
        before = fixture.store.load(before['run_id'])
        original_receipt_hash = digest(review_result['response'])
        self.assertNotIn(review_id, before['accepted_receipts'])

        state = fixture.finish(fixture.runner(resume=True))
        artifacts = self.assert_ready(fixture, state)
        self.assertFalse(live_pid(old_call['identity']['pid']))
        self.assertEqual(len(fixture.calls(stage='P5', node='steady', role='worker')), 1)
        self.assertEqual(len(fixture.calls(stage='P5', node='steady', role='reviewer')), 1)
        self.assertEqual(state['accepted_receipts'][review_id], original_receipt_hash)
        self.assertEqual(state['nodes']['steady']['approval']['attempt_id'], review_id)
        self.assertEqual(manifest(artifacts)['steady.yaml'],
                         fixture.calls(stage='P5', node='steady', role='reviewer')[0]['before_hashes']['steady.yaml'])
        self.assertEqual(len(fixture.calls(stage='P5', node='leaf', role='worker')), 1)
        self.assert_budgets_preserved(before, state)
        stable = deepcopy(state)
        stable_calls = deepcopy(fixture.calls())
        state = fixture.finish(fixture.runner(resume=True))
        self.assertEqual(fixture.calls(), stable_calls)
        self.assertEqual(state['accepted_receipts'], stable['accepted_receipts'])
        self.assertEqual(state['nodes']['steady']['stage'], stable['nodes']['steady']['stage'])

    def test_unprovable_live_process_fences_only_its_node_while_a_sibling_finishes(self):
        fixture = self.fixture(
            pairs=(('fenced', 'root'), ('steady', 'root'), ('blocked_child', 'fenced'), ('leaf', 'steady')),
            controls={
                'P5.fenced.worker.normal.1': {'delay': 120},
                'P5.steady.worker.normal.1': {'wait_for': ['P5.fenced.worker.normal.1.begin'], 'delay': 120},
            },
        )
        fixture.create()
        first = fixture.runner()
        fixture.wait_marker(first, 'P5', 'fenced', 'worker')
        fixture.wait_marker(first, 'P5', 'steady', 'worker')
        before = fixture.store.load(fixture.state['run_id'])
        fenced_id, steady_id = before['active_attempts']['P5:fenced'], before['active_attempts']['P5:steady']
        fenced_process, steady_process = fixture.remember_process(fenced_id), fixture.remember_process(steady_id)
        self.assertTrue(live_pid(fenced_process['identity']['pid']))
        self.assertTrue(live_pid(steady_process['identity']['pid']))
        first.kill()
        first.wait(timeout=5)

        # The PID/group really remains alive, but the saved start identity can
        # no longer prove its ownership.  This is disk damage, not a mocked
        # termination result.  Preserve the true identity solely for cleanup.
        unprovable = deepcopy(fenced_process)
        unprovable['identity']['started'] = 'lost-start-identity'
        unprovable['descendants'] = []
        attempt_dir = fixture.store.run_dir(before['run_id']) / 'attempts' / fenced_id
        write_json(attempt_dir / 'process.json', unprovable)
        state = fixture.finish(fixture.runner(resume=True))
        self.assertEqual(state['status'], 'termination_unconfirmed', state.get('reason'))
        self.assertEqual(state['active_attempts'], {'P5:fenced': fenced_id})
        self.assertIn(fenced_id, state['unconfirmed_attempts'])
        self.assertTrue(live_pid(fenced_process['identity']['pid']))
        self.assertFalse(live_pid(steady_process['identity']['pid']))
        self.assertEqual(len(fixture.calls(stage='P5', node='fenced', role='worker')), 1)
        self.assertEqual(len(fixture.calls(stage='P5', node='steady', role='worker')), 2)
        self.assertFalse(fixture.calls(stage='P5', node='blocked_child'))
        self.assertFalse(fixture.calls(stage='P6'))
        self.assertEqual(set(summary(fixture.store, state)['approved_nodes']), {'root', 'steady', 'leaf'})
        self.assertEqual(summary(fixture.store, state)['completed_node_count'], 3)
        fixture.store.verify_node(state, 'steady')
        fixture.store.verify_node(state, 'leaf')
        self.assert_budgets_preserved(before, state)

    def test_merge_durable_before_process_exit_recovers_accounting_without_reapproval(self):
        fixture = self.fixture(pairs=(('steady', 'root'), ('leaf', 'steady')))
        fixture.create()
        first = fixture.runner(fault='after_steady_merge')
        first.wait(timeout=60)
        self.assertEqual(first.returncode, 86, first.log.read_text())
        before = fixture.store.load(fixture.state['run_id'])
        boundary = read_json(fixture.base / 'fault.json')
        self.assertEqual(before['merged_checkpoint'], boundary['checkpoint'])
        self.assertEqual(before['nodes']['steady']['approval'], boundary['approval'])
        self.assertFalse(fixture.calls(stage='P5', node='leaf'))
        review_id = boundary['approval']['attempt_id']
        directory = fixture.store.run_dir(before['run_id']) / 'attempts' / review_id
        attempt, original_result = read_json(directory / 'attempt.json'), read_json(directory / 'result.json')
        approved_receipts = deepcopy(before['accepted_receipts'])

        state = fixture.finish(fixture.runner(resume=True))
        self.assert_ready(fixture, state)
        self.assertEqual(len(fixture.calls(stage='P5', node='steady', role='worker')), 1)
        self.assertEqual(len(fixture.calls(stage='P5', node='steady', role='reviewer')), 1)
        self.assertEqual(state['nodes']['steady']['stage']['calls'], 2)
        self.assertGreaterEqual(state['nodes']['steady']['stage']['elapsed_seconds'], original_result['elapsed_seconds'])
        self.assertEqual(state['nodes']['steady']['approval'], before['nodes']['steady']['approval'])
        self.assertTrue(approved_receipts.items() <= state['accepted_receipts'].items())
        self.assert_budgets_preserved(before, state)
        stable = deepcopy(state)
        accepted_directory = manifest(fixture.store.run_dir(state['run_id']))
        with self.assertRaisesRegex(ExtractError, 'duplicate'):
            accept_result(fixture.store, state, fixture.lock, attempt, original_result)
        self.assertEqual(fixture.store.load(state['run_id']), stable)
        self.assertEqual(manifest(fixture.store.run_dir(state['run_id'])), accepted_directory)
        state = fixture.finish(fixture.runner(resume=True))
        self.assertEqual(state['nodes']['steady']['stage'], stable['nodes']['steady']['stage'])
        self.assertEqual(state['accepted_receipts'], stable['accepted_receipts'])

    def test_status_verifies_the_frozen_entry_before_executing_it(self):
        fixture = self.fixture()
        fixture.create()
        entry = Path(fixture.lock['snapshot']) / 'rules/schema_extract/__main__.py'
        marker = fixture.base / 'unverified-entry-executed'
        entry.chmod(0o644)
        entry.write_text('from pathlib import Path\nPath(' + repr(str(marker)) +
                         ').write_text("unverified code ran")\n', encoding='utf-8')
        before = manifest(fixture.store.run_dir(fixture.state['run_id']))
        result = subprocess.run(
            [sys.executable, '-B', '-m', 'schema_extract', 'status', fixture.state['run_id'],
             '--repo', str(REPO), '--state-root', str(fixture.store.root), '--full'],
            cwd=REPO, capture_output=True, text=True, timeout=15,
            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
        )
        self.assertFalse(marker.exists(), 'the unverified frozen entry executed before its digest was checked')
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertRegex(result.stderr, 'changed|mismatch|fingerprint')
        self.assertEqual(manifest(fixture.store.run_dir(fixture.state['run_id'])), before)

    def test_interrupted_final_signoff_cannot_be_granted_again_on_resume(self):
        fixture = self.fixture(
            actions={'P6.root.reviewer.normal.1': 'revise', 'P6.root.reviewer.adjudicate.1': 'patch'},
            controls={'P6.root.reviewer.signoff.1': {'delay': 120}},
        )
        fixture.create(revision_limit=1)
        first = fixture.runner()
        fixture.wait_marker(first, 'P6', 'root', 'reviewer', mode='signoff')
        before = fixture.store.load(fixture.state['run_id'])
        original_process = fixture.remember_process(before['active_attempts']['P6'])
        self.assertEqual(before['stages']['P6']['adjudications'], 1)
        self.assertEqual(before['stages']['P6']['signoffs'], 1)
        self.assertEqual(before['stages']['P6']['adjudication']['decision'], 'approve')
        first.kill()
        first.wait(timeout=5)

        state = fixture.finish(fixture.runner(resume=True))
        self.assertEqual(state['status'], 'blocked', state.get('reason'))
        self.assertEqual(state['stages']['P6']['adjudication'], before['stages']['P6']['adjudication'])
        self.assertEqual(state['stages']['P6']['adjudications'], 1)
        self.assertEqual(state['stages']['P6']['signoffs'], 1)
        self.assertEqual(state['stages']['P6']['calls'], before['stages']['P6']['calls'])
        self.assertEqual(state['stages']['P6']['signoff']['decision'], 'blocked')
        self.assertFalse(live_pid(original_process['identity']['pid']))
        self.assertFalse(state['published'])
        self.assertFalse(Path(state['output']).exists())
        self.assert_budgets_preserved(before, state)
        calls = deepcopy(fixture.calls())
        state = fixture.finish(fixture.runner(resume=True))
        self.assertEqual(state['status'], 'blocked')
        self.assertEqual(fixture.calls(), calls)
        self.assertEqual(len(fixture.calls(stage='P6', mode='adjudicate')), 1)
        self.assertEqual(len(fixture.calls(stage='P6', mode='signoff')), 1)

    def test_inspection_reuses_archived_decisions_without_writing_run_or_ledger(self):
        from schema_extract.maintenance import candidate_report, inspect_runs, registry

        fixture = self.fixture()
        proposal = {'id': 'reader-cursor-round-trips', 'scope': 'extract_tools',
                    'summary': 'The previously reported source-pagination behavior recurred.', 'evidence': [0]}
        script = fixture.fake.read_text()
        fixture.fake.write_text(script.replace(
            'output.write_text(json.dumps(receipt))',
            "if role == 'reviewer' and stage in ('P1', 'P2'):\n"
            "    receipt['framework_candidates'] = [" + repr(proposal) + "]\n"
            'output.write_text(json.dumps(receipt))',
            1,
        ))
        fixture.create()
        state = fixture.finish(fixture.runner())
        self.assert_ready(fixture, state)
        old_decision = read_json(REPO / 'schema_extract_old/issues/archive/extract_tools/reader-cursor-round-trips.json')['decision']
        before_store = manifest(fixture.store.root)
        before_ledger = manifest(REPO / 'schema_extract_old/issues')
        before_calls = deepcopy(fixture.calls())

        records = registry(REPO)
        reports = candidate_report(fixture.store, state, REPO)
        inspected = inspect_runs(fixture.store, REPO)
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]['id'], proposal['id'])
        self.assertEqual(reports[0]['registered']['decision'], old_decision)
        self.assertEqual(records['records'][proposal['id']]['decision'], old_decision)
        self.assertEqual(len(reports[0]['reports']), 2)
        self.assertEqual(len(inspected), 1)
        self.assertEqual(inspected[0]['run']['run_id'], state['run_id'])
        self.assertEqual(inspected[0]['run']['completed_node_count'], 1)
        self.assertEqual(inspected[0]['executor'], str(Path(fixture.lock['snapshot']) / 'rules'))
        self.assertEqual(manifest(fixture.store.root), before_store)
        self.assertEqual(manifest(REPO / 'schema_extract_old/issues'), before_ledger)
        self.assertEqual(fixture.calls(), before_calls)

    def test_pause_request_cannot_adopt_an_outside_workspace_mutation(self):
        fixture = self.fixture(controls={'P1.root.worker.normal.1': {'delay': 5}})
        fixture.create()
        first = fixture.runner()
        fixture.wait_marker(first, 'P1', 'root', 'worker')
        state = fixture.store.load(fixture.state['run_id'])
        old_process = fixture.remember_process(state['active_attempts']['P1'])
        illegal = fixture.store.run_dir(state['run_id']) / 'outside-worker-workspace.json'
        write_json(illegal, {'unexpected': 'An invocation cannot add authoritative files here.'})
        fixture.store.request_control(state['run_id'], 'pause')
        first.wait(timeout=15)
        self.assertNotEqual(first.returncode, 0, 'the control request silently adopted an unauthorized file')
        self.assertIn('agent modified Runner', first.log.read_text())
        after = fixture.store.load(state['run_id'])
        self.assertEqual(after['accepted_receipts'], {})
        self.assertIsNone(after['stages']['P1']['approval'])
        self.assertFalse(fixture.calls(stage='P1', role='reviewer'))
        self.assertFalse(live_pid(old_process['identity']['pid']))

    def test_inspector_queue_request_cannot_resume_a_subsequent_user_pause(self):
        from schema_extract.runner import create_run

        fixture = self.fixture(controls={'P1.root.worker.normal.1': {'delay': 2}})
        fixture.store, fixture.state, fixture.lock = create_run(
            REPO, fixture.base / 'state', fixture.base / 'output', source=str(fixture.source),
            config={'codex_binary': str(fixture.fake), 'call_seconds': 25, 'max_calls': 80}, queued=True,
        )
        observed = fixture.base / 'inspector-observed-queued'
        release = fixture.base / 'release-inspector'
        driver = r'''
from pathlib import Path
import sys
import time
from schema_extract import cli
state_root, run_id, repo, observed, release = sys.argv[1:]
redirect = cli._frozen_entry
def delayed_redirect(lock, command):
    Path(observed).touch()
    until = time.monotonic() + 30
    while not Path(release).exists():
        if time.monotonic() > until:
            raise SystemExit('independent test did not release the inspected queue request')
        time.sleep(0.01)
    return redirect(lock, command)
cli._frozen_entry = delayed_redirect
raise SystemExit(cli.main(['inspect', '--start', run_id, '--repo', repo, '--state-root', state_root]))
'''
        log = fixture.base / 'inspector-race.log'
        with log.open('wb') as output:
            inspector = subprocess.Popen(
                [sys.executable, '-B', '-c', driver, str(fixture.store.root), fixture.state['run_id'],
                 str(REPO), str(observed), str(release)],
                cwd=REPO, stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
            )
        inspector.log = log
        fixture.runners.append(inspector)
        fixture.wait_until(inspector, observed.exists, description='inspector observing queued work')

        owner = fixture.runner(resume=True)
        fixture.wait_marker(owner, 'P1', 'root', 'worker')
        fixture.store.request_control(fixture.state['run_id'], 'pause')
        paused = fixture.finish(owner)
        self.assertEqual(paused['status'], 'paused')
        self.assertEqual(paused['control']['resume_status'], 'running')
        before_calls = deepcopy(fixture.calls())
        before_state = deepcopy(paused)
        release.touch()
        inspector.wait(timeout=30)
        self.assertEqual(fixture.calls(), before_calls,
                         'an old queued-work inspection revived the subsequently paused operator')
        self.assertEqual(fixture.store.load(paused['run_id']), before_state)

    def test_legacy_import_refuses_live_recorded_processes_and_new_source_bytes(self):
        from schema_extract.legacy import import_legacy

        fixture = self.fixture()
        # Reuse only the native-v1 data builder.  Its real archived package and
        # original Runner/Store produce the stopped task and immutable records;
        # this test supplies independent OS and source-integrity counterexamples.
        builder = r'''
import json
from pathlib import Path
import sys
from test_e_runner import LegacyFixture
legacy = LegacyFixture(Path(sys.argv[1]))
print(json.dumps({'base': str(legacy.base), 'task': str(legacy.task), 'state': legacy.state,
                  'lock': legacy.lock, 'descriptor': str(legacy.descriptor), 'spec': legacy.spec}))
'''
        built = subprocess.run(
            [sys.executable, '-B', '-c', builder, str(fixture.base)], cwd=REPO,
            env={**os.environ, 'PYTHONPATH': os.pathsep.join((str(REPO / 'schema_extract/tests'), str(REPO))),
                 'PYTHONDONTWRITEBYTECODE': '1'}, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        legacy = json.loads(built.stdout.splitlines()[-1])
        old_base, task = Path(legacy['base']), Path(legacy['task'])

        def cleanup_legacy():
            for path in old_base.rglob('*'):
                if path.is_dir():
                    path.chmod(0o755)
            shutil.rmtree(old_base)

        self.addCleanup(cleanup_legacy)
        target = fixture.base / 'must-not-create-v2'
        kwargs = {'source_task': task, 'checkpoint_id': legacy['state']['checkpoint'],
                  'source': legacy['descriptor'], 'config': {'codex_binary': str(fixture.fake)}}
        records = Path(legacy['state']['attempts'][-1]['calls'][-1])
        sleeper = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(120)'],
                                   start_new_session=True, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            write_json(records / 'process.json', {'identity': process_table()[sleeper.pid], 'descendants': []})
            old_manifest = manifest(task.parent.parent)
            with self.assertRaisesRegex(ExtractError, 'live|ownership'):
                import_legacy(REPO, target, fixture.base / 'delivery', **kwargs)
            self.assertTrue(live_pid(sleeper.pid), 'read-only legacy inspection must not terminate an old process itself')
            self.assertFalse(target.exists())
            self.assertEqual(manifest(task.parent.parent), old_manifest)
        finally:
            sleeper.kill()
            sleeper.wait(timeout=5)

        changed_source = fixture.base / 'changed-source'
        shutil.copytree(Path(legacy['lock']['snapshot']) / 'kernel_wiki', changed_source / 'kernel_wiki')
        changed_input = changed_source / legacy['spec']['directory'] / 'input.txt'
        changed_input.chmod(0o644)
        changed_input.write_text(changed_input.read_text() + 'This is a new, unaccepted input requirement.\n')
        descriptor = fixture.base / 'changed-legacy-source.json'
        write_json(descriptor, {**legacy['spec'], 'repository': str(changed_source)})
        old_manifest = manifest(task.parent.parent)
        with self.assertRaisesRegex(ExtractError, 'legacy frozen source bytes'):
            import_legacy(REPO, target, fixture.base / 'delivery', **{**kwargs, 'source': str(descriptor)})
        self.assertFalse(target.exists())
        self.assertEqual(manifest(task.parent.parent), old_manifest)


if __name__ == '__main__':
    unittest.main()
