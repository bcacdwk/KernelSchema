"""Targeted evidence only; runtime mutations occur solely through ordinary resume."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = Path('/home/shaohanyong.shy/KernelSchema')
ROOT = REPO / '.schema_extract/v2'
RID = 'd138394a-20a7-4397-896e-ebac43aa10bb'
RUN = ROOT / 'runs' / RID
OUT = Path(__file__).resolve().parent
PYTHON = '/opt/venv/python3.13.12/bin/python3'
CLI = [PYTHON, '-B', str(REPO / 'schema_extract/extract_yaml.py')]
LOCATION = ['--repo', str(REPO), '--state-root', str(ROOT)]


def read(path):
    return json.loads(path.read_text())


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(prefix):
    result = subprocess.run(CLI + ['inspect', '--run', RID, '--json'] + LOCATION,
                            capture_output=True, text=True, check=True)
    (OUT / f'{prefix}_inspect.json').write_text(result.stdout)
    (OUT / f'{prefix}_inspect.stderr').write_text(result.stderr)
    return json.loads(result.stdout)[0]


def no_fence(observation):
    return (all(not observation[k] for k in ['active_attempts', 'unconfirmed_attempts',
            'unsettled_attempts', 'global_hold', 'maintenance_request'])
            and not any(observation['control'].values())
            and observation['runner_observation']['lock'] == 'free'
            and observation['runner_observation']['alive'] is False
            and observation['launch']['status'] == 'finished')


def feedback_excerpt(path, name):
    lines = path.read_text().splitlines()
    indices = set()
    for index, line in enumerate(lines):
        if '1200 characters' in line:
            indices.update(range(max(0, index - 9), min(len(lines), index + 3)))
    (OUT / name).write_text('\n'.join(f'{i+1}: {lines[i]}' for i in sorted(indices)) + '\n')


def before():
    assert not (OUT / 'before_state.json').exists(), 'Never overwrite baseline'
    observation = inspect('before')
    for fn in ['state.json', 'inputs.lock.json']:
        shutil.copyfile(RUN / fn, OUT / ('before_' + fn))
    state = read(OUT / 'before_state.json')
    aid = state['attempts'][-1]
    attempt = RUN / 'attempts' / aid
    meta, result, response, receipt = [read(attempt / fn) for fn in
                                    ['attempt.json', 'result.json', 'response.json', 'receipt.json']]
    session = read(Path(result['session_directory']) / 'session.json')
    write('before_session.json', session)
    for fn in ['attempt.json', 'result.json', 'response.json', 'receipt.json',
               'repair.json', 'exit.json', 'cli-exit.json', 'finished.json', 'invocation.json']:
        shutil.copyfile(attempt / fn, OUT / ('failed_' + fn))
    history, settlements = [], []
    for old_id in state['attempts']:
        path = RUN / 'attempts' / old_id
        old_meta, old_exit = read(path / 'attempt.json'), read(path / 'exit.json')
        old_cli = read(path / 'cli-exit.json')
        settlements.append({'attempt_id': old_id, 'settled': state['attempt_usage'][old_id]['settled'],
                            'termination_confirmed': old_exit['termination']['confirmed'],
                            'survivors': old_exit['termination']['survivors'],
                            'exit_code': old_exit['exit_code'], 'cli_exit_code': old_cli['exit_code'],
                            'descendants_reaped': old_cli['descendants_reaped']})
        if old_meta['stage'] != 'P4':
            continue
        old_response = read(path / 'response.json')
        handoff = old_response['handoff']
        repair = read(path / 'repair.json') if (path / 'repair.json').exists() else {}
        history.append({'attempt_id': old_id, 'role': old_meta['role'],
                        'decision': old_response['decision'], 'handoff_characters': len(handoff),
                        'handoff_nonempty_lines': sum(bool(line.strip()) for line in handoff.splitlines()),
                        'repair_code': repair.get('code'), 'repair_summary': repair.get('summary'),
                        'prompt_has_length_diagnostic': '1200 characters' in (path / 'prompt.txt').read_text()})
    write('p4_history.json', history)
    write('historical_settlements.json', settlements)
    previous_id = [row['attempt_id'] for row in history[:-1] if row['role'] == 'reviewer'][-1]
    previous = read(RUN / 'attempts' / previous_id / 'response.json')
    write('previous_reviewer_response.json', previous)
    (OUT / 'handoff_diff.txt').write_text('\n'.join(difflib.unified_diff(
        previous['handoff'].splitlines(), response['handoff'].splitlines(),
        fromfile=previous_id, tofile=aid, lineterm='')) + '\n')
    feedback_excerpt(attempt / 'prompt.txt', 'failed_prompt_feedback_excerpt.txt')
    gate = Path(meta['materials']['executor_root']) / 'schema_extract/tools/receipts.py'
    lines = gate.read_text().splitlines()
    index = next(i for i, line in enumerate(lines) if 'approval needs a short handoff' in line)
    (OUT / 'frozen_handoff_gate.txt').write_text(str(gate) + '\nsha256=' + digest(gate) + '\n' +
        '\n'.join(f'{i+1}: {lines[i]}' for i in range(index-3, index+2)) + '\n')
    paths = {RUN / 'inputs.lock.json', gate}
    for checkpoint in {state['checkpoint'], state['merged_checkpoint']}:
        paths.update(p for p in (RUN / 'checkpoints' / checkpoint).rglob('*') if p.is_file())
    for old_id in state['attempts']:
        path = RUN / 'attempts' / old_id
        paths.update(p for p in path.iterdir() if p.is_file() and p.suffix == '.json')
    for folder in ['original', 'workspace']:
        paths.update(p for p in (attempt / folder).rglob('*') if p.is_file())
    write('preserved_files_before.json', {str(p): digest(p) for p in sorted(paths)})
    validation = result['response_validation']
    checks = {
        'expected_stopped_p4_reviewer': (state['status'], state['stage'], state['role'], state['mode']) ==
            ('invalid_receipt', 'P4', 'reviewer', 'normal') and aid == 'dbca9743-0c86-4698-9dcb-01916ac01e02'
            and state['checkpoint'] == '2485ba254ac242ce90a9714c9c1a8cdaf15faea77fb15d02d2ea67679031caa0',
        'budget_available_without_addition': state['stages']['P4']['calls'] == 6
            and state['config']['max_calls'] == 32 and not state['call_additions'],
        'no_attempt_control_or_recovery_fence': no_fence(observation)
            and not state['reopening'] and not state['reopen_requests'] and not state['pending_rechecks'],
        'all_historical_calls_settled': set(state['attempt_usage']) == set(state['attempts'])
            and all(row['settled'] and row['termination_confirmed'] and not row['survivors']
                    and row['descendants_reaped'] and row['exit_code'] == row['cli_exit_code']
                    for row in settlements),
        'native_parse_and_identity_valid': result['exit_code'] == 0 and result['native_status']['success']
            and not result['parse_error_lines'] and validation['schema_valid'] and validation['nonce_valid']
            and validation['native_output_matches']
            and result['session_id'] == result['resume_session_id'] == state['sessions']['P4']['reviewer']
            and session['status'] == 'ready' and not session['retired_sessions'],
        'real_handoff_violation_without_approval': response['handoff'] == receipt['handoff'] == result['response']['handoff']
            and len(response['handoff']) > 1200 and response['decision'] == 'approve'
            and aid not in state['accepted_receipts'] and state['stages']['P4']['approval'] is None,
        'first_handoff_length_diagnostic_in_scope': all(not row['prompt_has_length_diagnostic'] for row in history)
            and all('1200 characters' not in (row['repair_summary'] or '') for row in history[:-1])
            and '1200 characters' in history[-1]['repair_summary'],
    }
    diagnosis = {'run_id': RID, 'observed_at': time.time(), 'attempt_id': aid,
        'previous_reviewer_attempt': previous_id, 'previous_decision': previous['decision'],
        'previous_characters': len(previous['handoff']), 'characters': len(response['handoff']),
        'nonempty_lines': sum(bool(line.strip()) for line in response['handoff'].splitlines()),
        'reason': state['reason'], 'p4': state['stages']['P4'], 'config': state['config'],
        'historical_attempts': len(state['attempts']), 'preconditions': checks,
        'basis': 'First handoff length diagnostic in this P4 scope, not a claim of repeated-failure convergence. Prior two receipt repairs were source_evidence index violations; normal resume delivers the real new length diagnostic to the original reviewer.',
        'authorization': 'Supervisor relayed user authorization for the twelve-operator extraction: on a new handoff diagnostic, with sufficient unchanged call budget and all calls settled/no barriers, submit one ordinary resume and verify; preserve original run/source/rules/drafts/approvals/sessions/counters.'}
    write('diagnosis.json', diagnosis)
    assert all(checks.values()), checks
    print(json.dumps({'calls': 6, 'limit': 32, 'characters': diagnosis['characters'],
                      'nonempty_lines': diagnosis['nonempty_lines'], 'preconditions': checks}, ensure_ascii=False))


def launch():
    assert not (OUT / 'resume_launch.json').exists(), 'Only one launch through this helper'
    assert all(read(OUT / 'diagnosis.json')['preconditions'].values())
    observation = inspect('prelaunch')
    assert no_fence(observation), 'Control or live-call barrier appeared'
    assert digest(RUN / 'state.json') == digest(OUT / 'before_state.json'), 'Run changed; re-evaluate'
    argv = CLI + ['resume', RID] + LOCATION
    with (OUT / 'resume.stdout').open('w') as stdout, (OUT / 'resume.stderr').open('w') as stderr:
        child = subprocess.Popen(argv, cwd=REPO, stdout=stdout, stderr=stderr, start_new_session=True)
    record = {'argv': argv, 'pid': child.pid, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'authorized_scope': 'One ordinary P4 reviewer resume for first handoff length diagnostic; no add-calls; preserve original identities and counters'}
    write('resume_launch.json', record)
    print(json.dumps(record, ensure_ascii=False))


def after():
    observation = inspect('after')
    shutil.copyfile(RUN / 'state.json', OUT / 'after_state.json')
    state, old = read(OUT / 'after_state.json'), read(OUT / 'before_state.json')
    added = [aid for aid in state['attempts'] if aid not in old['attempts']]
    assert len(added) == 1, added
    aid = added[0]
    attempt = RUN / 'attempts' / aid
    meta, invocation = read(attempt / 'attempt.json'), read(attempt / 'invocation.json')
    for fn in ['attempt.json', 'invocation.json', 'process.json', 'launch.json']:
        if (attempt / fn).exists():
            shutil.copyfile(attempt / fn, OUT / ('resumed_' + fn))
    native_init = []
    if (attempt / 'events.jsonl').exists():
        for line in (attempt / 'events.jsonl').read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get('type') == 'system' and event.get('subtype') == 'init':
                native_init.append(event)
    write('resumed_native_init.json', native_init)
    prompt = (attempt / 'prompt.txt').read_text()
    feedback_excerpt(attempt / 'prompt.txt', 'resumed_prompt_feedback_excerpt.txt')
    files = read(OUT / 'preserved_files_before.json')
    changed = [path for path, sha in files.items() if not Path(path).exists() or digest(Path(path)) != sha]
    identities = ['run_id', 'operator_id', 'input_fingerprint', 'snapshot', 'source_identity',
                  'executor', 'config', 'output', 'call_additions']
    drafts = ['checkpoint', 'merged_checkpoint', 'selected_designs', 'selection_signature',
              'nodes', 'writing_plan', 'writing_signature', 'plan_draft', 'plan_history', 'plan_revision']
    counts = ['receipt_repairs', 'invalid_receipts', 'revisions', 'adjudications', 'signoffs']
    session_id = old['sessions']['P4']['reviewer']
    checks = {
        'original_identities_and_budget_preserved': all(state[k] == old[k] for k in identities),
        'drafts_and_approvals_preserved': all(state[k] == old[k] for k in drafts)
            and all(state['stages'][stage]['approval'] == old['stages'][stage]['approval'] for stage in ['P1', 'P2', 'P3', 'P4'])
            and state['accepted_receipts'] == old['accepted_receipts'],
        'historical_attempts_and_accounting_preserved': state['attempts'][:len(old['attempts'])] == old['attempts']
            and all(state['attempt_usage'][key] == val for key, val in old['attempt_usage'].items()) and not changed,
        'one_new_call_and_no_counter_reset': state['stages']['P4']['calls'] == old['stages']['P4']['calls'] + 1
            and all(state['stages']['P4'][key] == old['stages']['P4'][key] for key in counts),
        'exact_reviewer_session': invocation['resume_session_id'] == session_id
            and state['sessions'] == old['sessions']
            and bool(native_init) and all(event['session_id'] == session_id for event in native_init)
            and meta['stage'] == 'P4' and meta['role'] == 'reviewer',
        'latest_real_failure_delivered': 'approval needs a short handoff (<=1200 characters, <=20 nonempty lines)' in prompt
            and old['attempts'][-1] in prompt and invocation['prompt_sha256'] == digest(attempt / 'prompt.txt'),
        'new_call_running_with_fence': aid in observation['active_attempts'].values()
            and observation['runner_observation']['alive']
            and observation['runner_observation']['lock'] == 'held' and not observation['unconfirmed_attempts'],
    }
    report = {'run_id': RID, 'observed_at': time.time(), 'new_attempt': aid,
              'before_calls': old['stages']['P4']['calls'], 'after_calls': state['stages']['P4']['calls'],
              'session_id': session_id, 'checks': checks, 'changed_preserved_files': changed,
              'preserved_file_count': len(files), 'status': state['status'], 'stage': state['stage'],
              'p4_approval': state['stages']['P4']['approval'], 'native_result_exists': (attempt / 'result.json').exists()}
    write('recovery_verification.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    {'before': before, 'launch': launch, 'after': after}[sys.argv[1]]()
