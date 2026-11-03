"""Supervisor controls, crash windows, atomic delivery and genuine v1 import."""

from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import unittest

from schema_extract.common import ExtractError, manifest, read_json, write_json
from schema_extract.legacy import import_legacy
from schema_extract.processes import terminate_owned
from schema_extract.runner import export_checkpoint, run
from test_c_independent import REPO
from test_d_runner import DFixture


DRIVER = r'''
import json
import os
from pathlib import Path
import sys
from schema_extract.common import write_json
from schema_extract.runner import run
from schema_extract.store import Store
root, run_id, resume, fault, marker = sys.argv[1:]
if fault == 'checkpoint':
    original = Store.checkpoint
    def checkpoint(self, state, artifacts, **kwargs):
        checkpoint_id = original(self, state, artifacts, **kwargs)
        attempt_id = kwargs.get('attempt_id')
        if attempt_id:
            attempt = json.loads((self.run_dir(run_id) / 'attempts' / attempt_id / 'attempt.json').read_text())
            if attempt['stage'] == 'P5' and attempt['node'] == 'left' and attempt['role'] == 'reviewer':
                write_json(Path(marker), {'checkpoint': checkpoint_id, 'attempt_id': attempt_id})
                os._exit(86)
        return checkpoint_id
    Store.checkpoint = checkpoint
if fault == 'publication':
    import schema_extract.publication as publication
    original = publication.write_json
    def write(path, value):
        if path.parent.name == 'publications' and not path.name.endswith('.intent.json'):
            write_json(Path(marker), {'publication': value, 'target_exists': Path(value['target']).is_dir()})
            os._exit(87)
        return original(path, value)
    publication.write_json = write
if fault == 'cli_result':
    original = Store.save
    def save(self, state):
        original(self, state)
        for attempt_id in state['active_attempts'].values():
            path = self.run_dir(run_id) / 'attempts' / attempt_id / 'result.json'
            if attempt_id in state['attempt_usage'] and path.is_file() and json.loads(path.read_text())['outcome'] == 'cli_error':
                write_json(Path(marker), {'attempt_id': attempt_id})
                os._exit(88)
    Store.save = save
state = run(Store(Path(root)), run_id, resume=resume == 'yes')
print(json.dumps(state), flush=True)
'''


class EFixture(DFixture):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.runners = []

    def runner(self, *, resume=False, fault=''):
        rules = Path(self.lock['snapshot']) / 'rules'
        log = self.base / ('runner-' + str(len(self.runners)) + '.log')
        with log.open('wb') as stream:
            process = subprocess.Popen([sys.executable, '-B', '-c', DRIVER, str(self.store.root),
                self.state['run_id'], 'yes' if resume else 'no', fault, str(self.base / 'fault.json')],
                cwd=rules, env={**os.environ, 'PYTHONPATH': str(rules), 'PYTHONDONTWRITEBYTECODE': '1'},
                stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        process.log = log
        self.runners.append(process)
        return process

    def wait_for(self, process, predicate, *, timeout=40):
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            found = predicate()
            if found:
                return found
            if process.poll() is not None:
                raise AssertionError(process.log.read_text())
            time.sleep(0.02)
        raise AssertionError('boundary not reached: ' + process.log.read_text())

    def started(self, process, key):
        path = self.base / 'marks' / (key + '.begin.json')
        return self.wait_for(process, lambda: read_json(path) if path.is_file() else None)

    def finish(self, process, *, code=0):
        found = process.wait(timeout=60)
        if found != code:
            raise AssertionError(f'Runner exit {found}, expected {code}: {process.log.read_text()}')
        self.state = self.store.load(self.state['run_id'])
        return self.state

    def cleanup(self):
        for process in self.runners:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
        if self.store is not None:
            for path in self.store.root.glob('runs/*/attempts/*/process.json'):
                value = read_json(path)
                terminate_owned(value['identity'], value.get('descendants', []), grace=0.1)
        super().cleanup()


LEGACY_DRIVER = r'''
import json
from pathlib import Path
from uuid import uuid4
from schema_extract.tools.tests.support import ExtractionCase, FIXTURE
from schema_extract.tools.codex_cli import write_json
from schema_extract.tools.execution import Runner
from schema_extract.tools.records import schema_parts
import yaml
case = ExtractionCase()
case.setUp()
case.temporary._finalizer.detach()
directory = case.store.run_dir(case.plan['run_id'])
directory.mkdir(parents=True)
write_json(directory / 'plan.json', case.plan)
class ControlledLegacyCLI:
    def __init__(self):
        self.calls = 0
    def run(self, **kwargs):
        self.calls += 1
        state = case.store.load(case.state['key'])
        records, work, role = kwargs['records'], kwargs['cwd'], kwargs['role']
        records.mkdir(parents=True)
        if role == 'worker':
            if state['stage'] == 'contract':
                case.draft(work)
                root = yaml.safe_load((work / 'root.yaml').read_text())
                root.pop('design')
                root['contract'].pop('hardware_specification')
            else:
                root = yaml.safe_load((work / 'root.yaml').read_text())
                root['design'] = yaml.safe_load(FIXTURE.read_text())['design']
                root['meta_data']['version'] += 1
            (work / 'root.yaml').write_text(yaml.safe_dump(root, sort_keys=False))
            response = case.receipt(state)
        else:
            response = case.review_receipt(state, work)
        response['issues'] = [{'id': 'legacy-source-note', 'kind': 'evidence_gap', 'scope': 'source documentation',
            'disposition': 'note', 'detail': 'Synthetic note retained through import.', 'evidence': ['synthetic source'],
            'restart_condition': 'Read the same fixed input.', 'blocks': []}]
        value = {'outcome': 'completed', 'response': response,
            'thread_id': kwargs['thread_id'] or str(uuid4()), 'elapsed_seconds': 0.25,
            'usage': {'input_tokens': self.calls * 10, 'output_tokens': self.calls},
            'termination': {'confirmed': True, 'actions': [], 'survivors': [], 'reason': 'controlled in-process transport; no subprocess'}}
        write_json(records / 'result.json', value)
        return value
with case.store.lock():
    state = Runner(case.store, case.plan, ControlledLegacyCLI()).work(case.state['key'])
assert state['result'] == 'ready', (state['result'], state.get('last_error'))
print(json.dumps({'base': str(case.base), 'task': str(case.store.task_dir(state['key'])),
                  'state': state, 'lock': case.baseline, 'plan': case.plan}), flush=True)
'''


class LegacyFixture:
    """Freeze real 69e3027 v1 sources, and use its native Runner and Store."""
    def __init__(self, base: Path):
        self.code = base / 'legacy-code'
        self.code.mkdir()
        archive = subprocess.check_output(['git', 'archive', '69e3027', 'schema_extract', 'schema_docs',
                                           'schema_template', 'README.md', 'EXTRACT_SCHEMA.md'], cwd=REPO)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(self.code, filter='data')
        result = subprocess.run([sys.executable, '-B', '-c', LEGACY_DRIVER], cwd=self.code,
            env={**os.environ, 'PYTHONPATH': str(self.code), 'PYTHONDONTWRITEBYTECODE': '1'},
            capture_output=True, text=True, timeout=40)
        if result.returncode:
            raise AssertionError(result.stderr + result.stdout)
        value = json.loads(result.stdout.splitlines()[-1])
        self.base, self.task = Path(value['base']), Path(value['task'])
        self.state, self.lock, self.plan = value['state'], value['lock'], value['plan']
        self.descriptor = base / 'legacy-source.json'
        self.spec = {'format': 'schema_extract.source.v2', 'id': 'synthetic', 'repository': self.lock['snapshot'],
            'directory': 'kernel_wiki/test/operators/synthetic', 'entrypoint': 'entrypoint.txt',
            'host': 'entrypoint.txt', 'io': ['metadata.txt'], 'input': 'input.txt', 'reference': 'reference.txt::copy',
            'acceptance': ['acceptance.txt'], 'scope': ['metadata.txt']}
        write_json(self.descriptor, self.spec)

    def cleanup(self):
        for path in self.base.rglob('*'):
            if path.is_dir():
                path.chmod(0o755)
        shutil.rmtree(self.base)


class ERunnerTests(unittest.TestCase):
    def fixture(self, **kwargs):
        fixture = EFixture(**kwargs)
        self.addCleanup(fixture.cleanup)
        return fixture

    def test_pause_finishes_the_current_call_and_resume_keeps_normal_sessions(self):
        fixture = self.fixture(control={'P1.root.worker.normal.1': {'delay': 0.8}})
        fixture.create()
        deadline = fixture.state['deadline']
        process = fixture.runner()
        fixture.started(process, 'P1.root.worker.normal.1')
        fixture.store.request_control(fixture.state['run_id'], 'pause')
        paused = fixture.finish(process)
        self.assertEqual(paused['status'], 'paused')
        self.assertFalse(paused['active_attempts'])
        self.assertEqual(len(fixture.calls()), 1)
        self.assertEqual(paused['role'], 'reviewer')
        first_id = fixture.calls()[0]['session_id']
        fixture.store.request_control(paused['run_id'], 'cancel')
        cancelled = fixture.finish(fixture.runner())
        self.assertEqual(cancelled['status'], 'cancelled')
        self.assertEqual(len(fixture.calls()), 1)
        state = fixture.finish(fixture.runner(resume=True))
        self.assertTrue(state['published'])
        self.assertEqual(state['deadline'], deadline)
        self.assertFalse(state['session_handoffs'])
        self.assertEqual(fixture.calls(stage='P2', role='worker')[0]['resumed'], first_id)

    def test_cancel_terminates_all_parallel_calls_and_resumes_with_new_contexts(self):
        names = ('left', 'middle', 'right')
        fixture = self.fixture(pairs=[(name, 'root') for name in names],
            control={f'P5.{name}.worker.normal.1': {'delay': 20} for name in names})
        fixture.create()
        process = fixture.runner()
        for name in names:
            fixture.started(process, f'P5.{name}.worker.normal.1')
        started = time.monotonic()
        fixture.store.request_control(fixture.state['run_id'], 'cancel')
        stopped = fixture.finish(process)
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(stopped['status'], 'cancelled')
        self.assertFalse(stopped['active_attempts'])
        self.assertEqual(len(stopped['session_handoffs']), 3)
        for name in names:
            scope = stopped['nodes'][name]['stage']
            self.assertEqual((scope['calls'], scope['revisions'], scope['continues']), (1, 0, 0))
            old = fixture.calls(stage='P5', node=name, role='worker')[0]
            result = read_json(fixture.store.run_dir(stopped['run_id']) / 'attempts' / old['context']['identity']['attempt_id'] / 'result.json')
            self.assertTrue(result['termination']['confirmed'])
        state = fixture.finish(fixture.runner(resume=True))
        self.assertTrue(state['published'])
        self.assertEqual(state['deadline'], stopped['deadline'])
        for name in names:
            before, after = fixture.calls(stage='P5', node=name, role='worker')
            self.assertIsNone(after['resumed'])
            self.assertNotEqual(before['session_id'], after['session_id'])
        self.assertEqual(state['p6_starts'], 1)

    def test_killed_runner_confirms_old_model_exit_before_reissuing(self):
        for fault in ('live_call', 'cli_result'):
            with self.subTest(fault=fault):
                fixture = self.fixture(control={'P1.root.worker.normal.1': {'delay': 20}} if fault == 'live_call' else {},
                    actions={'P1.root.worker.normal.1': 'cli_failure'} if fault == 'cli_result' else {})
                fixture.create()
                process = fixture.runner(fault='' if fault == 'live_call' else fault)
                if fault == 'live_call':
                    fixture.started(process, 'P1.root.worker.normal.1')
                    fixture.wait_for(process, lambda: next(fixture.store.root.glob('runs/*/attempts/*/process.json'), None))
                    process.kill()
                fixture.finish(process, code=-signal.SIGKILL if fault == 'live_call' else 88)
                state = fixture.finish(fixture.runner(resume=True))
                self.assertTrue(state['published'])
                old, new = fixture.calls(stage='P1', role='worker')
                self.assertIsNone(new['resumed'])
                self.assertNotEqual(old['session_id'], new['session_id'])
                handoff = state['session_handoffs'][0]
                self.assertEqual(handoff['next_session_id'], new['session_id'])
                self.assertEqual(state['stages']['P1']['calls'], 3)
                self.assertEqual(state['stages']['P1']['cli_failures'], int(fault == 'cli_result'))
                self.assertEqual(fixture.calls(stage='P2', role='worker')[0]['resumed'], new['session_id'])
                self.assertEqual(len(state['accepted_receipts']), len(fixture.calls()) - 1)

    def test_checkpoint_durable_before_pointer_replays_once(self):
        fixture = self.fixture(pairs=[('left', 'root')])
        fixture.create()
        fixture.finish(fixture.runner(fault='checkpoint'), code=86)
        fault = read_json(fixture.base / 'fault.json')
        before = fixture.state
        self.assertNotIn(fault['attempt_id'], before['accepted_receipts'])
        fixture.store.verify_checkpoint(before, fault['checkpoint'])
        calls = fixture.calls(stage='P5', node='left', role='reviewer')
        self.assertEqual(len(calls), 1)
        state = fixture.finish(fixture.runner(resume=True))
        self.assertTrue(state['published'])
        self.assertEqual(state['nodes']['left']['approval']['checkpoint'], fault['checkpoint'])
        self.assertEqual(len(fixture.calls(stage='P5', node='left', role='reviewer')), 1)
        self.assertEqual(state['nodes']['left']['stage']['calls'], 2)
        self.assertEqual(state['p6_starts'], 1)

    def test_publication_after_rename_before_receipt_is_idempotent(self):
        fixture = self.fixture()
        fixture.create()
        fixture.finish(fixture.runner(fault='publication'), code=87)
        fault = read_json(fixture.base / 'fault.json')
        self.assertTrue(fault['target_exists'])
        self.assertFalse(fixture.state['published'])
        before, calls = manifest(fixture.base / 'output'), deepcopy(fixture.calls())
        state = fixture.finish(fixture.runner(resume=True))
        self.assertTrue(state['published'])
        self.assertEqual(manifest(fixture.base / 'output'), before)
        self.assertEqual(fixture.calls(), calls)
        state_again = fixture.finish(fixture.runner(resume=True))
        self.assertEqual(state_again, state)

    def test_conflicting_target_and_explicit_export_keep_both_results(self):
        fixture = self.fixture()
        target = fixture.base / 'output'
        target.mkdir()
        (target / 'existing.txt').write_text('Keep this result.\n')
        before = manifest(target)
        state = fixture.run()
        self.assertEqual(state['status'], 'ready')
        self.assertFalse(state['published'])
        self.assertEqual(state['publication']['status'], 'conflict')
        self.assertEqual(manifest(target), before)
        alternative = fixture.base / 'alternate'
        first = export_checkpoint(fixture.store, state['run_id'], alternative)
        second = export_checkpoint(fixture.store, state['run_id'], alternative)
        self.assertEqual(first, second)
        self.assertEqual(first['status'], 'published')
        self.assertEqual(len(fixture.store.load(state['run_id'])['exports']), 1)
        empty = fixture.base / 'empty-existing'
        empty.mkdir()
        self.assertEqual(export_checkpoint(fixture.store, state['run_id'], empty)['status'], 'conflict')
        self.assertEqual(list(empty.iterdir()), [])
        self.assertEqual(manifest(target), before)

    def test_real_cli_queue_inspection_and_start_use_the_frozen_executor(self):
        fixture = self.fixture()
        command = [sys.executable, '-B', '-m', 'schema_extract']
        result = subprocess.run([*command, 'extract', '--source', str(fixture.source), '--output', str(fixture.base / 'output'),
            '--state-root', str(fixture.base / 'state'), '--codex-bin', str(fixture.fake), '--queue'],
            cwd=REPO, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        queued = json.loads(result.stdout)
        self.assertEqual(queued['status'], 'queued')
        inspected = subprocess.run([*command, 'inspect', '--state-root', str(fixture.base / 'state')],
            cwd=REPO, capture_output=True, text=True, timeout=30)
        self.assertEqual(inspected.returncode, 0, inspected.stderr)
        self.assertEqual(json.loads(inspected.stdout)[0]['run']['run_id'], queued['run_id'])
        self.assertFalse(fixture.calls())
        started = subprocess.run([*command, 'inspect', '--start', queued['run_id'], '--state-root', str(fixture.base / 'state')],
            cwd=REPO, capture_output=True, text=True, timeout=50)
        self.assertEqual(started.returncode, 0, started.stderr)
        self.assertTrue(json.loads(started.stdout)['published'])

    def test_genuine_v1_frozen_status_and_import_preserve_identity_usage_and_sources(self):
        fixture = self.fixture()
        legacy = LegacyFixture(fixture.base)
        self.addCleanup(legacy.cleanup)
        old_manifest = manifest(legacy.base / 'records')
        store, state, lock = import_legacy(REPO, fixture.base / 'imported-v2', fixture.base / 'imported-output',
            source_task=legacy.task, checkpoint_id=legacy.state['checkpoint'], source=str(legacy.descriptor),
            config={'codex_binary': str(fixture.fake), 'call_seconds': 20})
        self.assertEqual(manifest(legacy.base / 'records'), old_manifest)
        self.assertEqual(state['stage'], 'P1')
        self.assertFalse(state['accepted_receipts'])
        self.assertTrue(all(value['approval'] is None for value in state['stages'].values()))
        imported = state['imported_draft']
        self.assertEqual(imported['checkpoint'], legacy.state['checkpoint'])
        self.assertEqual(imported['usage'], legacy.state['usage'])
        self.assertGreater(imported['usage']['calls'], 0)
        self.assertEqual(imported['prior_issues'], legacy.state['issues'])
        self.assertEqual(manifest(Path(imported['artifacts'])), imported['files'])
        self.assertEqual(imported['frozen_status']['exit_code'], 0)
        self.assertEqual(lock['source']['aliases'][0]['ref'], legacy.lock['snapshot_commit'])
        state = run(store, state['run_id'])
        self.assertTrue(state['published'], state.get('reason'))
        self.assertEqual(state['stages']['P1']['calls'], 2)
        self.assertEqual(state['imported_draft']['usage'], legacy.state['usage'])
        with self.assertRaisesRegex(ExtractError, 'legacy runtime'):
            export_checkpoint(store, state['run_id'], legacy.base / 'records/new-delivery')
        self.assertFalse((legacy.base / 'records/new-delivery').exists())
        self.assertEqual(manifest(legacy.base / 'records'), old_manifest)


if __name__ == '__main__':
    unittest.main()
