"""Read-only, targeted recovery evidence collector; writes only beside this file."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path('/home/shaohanyong.shy/KernelSchema')
ROOT = REPO / '.schema_extract/v2'
OUT = Path(__file__).resolve().parent
RUNS = {
    'mla_nvfp4': 'af01989c-c712-4fcf-8ecc-3a88f10541eb',
    'fmha_fp8_per_token_head': '1a493588-28a0-4097-849f-85cac55e5b16',
}
PYTHON = '/opt/venv/python3.13.12/bin/python3'

def read(path):
    return json.loads(path.read_text())

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def inspect(rid, out, prefix):
    argv = [PYTHON, '-B', str(REPO / 'schema_extract/extract_yaml.py'),
            'inspect', '--run', rid, '--json', '--repo', str(REPO),
            '--state-root', str(ROOT)]
    result = subprocess.run(argv, capture_output=True, text=True, check=True)
    (out / f'{prefix}_inspect.json').write_text(result.stdout)
    (out / f'{prefix}_inspect.stderr').write_text(result.stderr)
    return json.loads(result.stdout)[0]

def before(name):
    rid = RUNS[name]
    run = ROOT / 'runs' / rid
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'before_state.json').exists(), 'Do not overwrite baseline'
    observation = inspect(rid, out, 'before')
    shutil.copyfile(run / 'state.json', out / 'before_state.json')
    shutil.copyfile(run / 'inputs.lock.json', out / 'before_inputs.lock.json')
    state = read(out / 'before_state.json')
    aid = state['attempts'][-1]
    attempt = run / 'attempts' / aid
    meta, result, response, receipt = [read(attempt / fn) for fn in
                                     ['attempt.json', 'result.json', 'response.json', 'receipt.json']]
    session = read(Path(result['session_directory']) / 'session.json')
    write(out / 'before_session.json', session)
    for fn in ['attempt.json', 'result.json', 'response.json', 'receipt.json',
               'repair.json', 'exit.json', 'cli-exit.json', 'finished.json', 'invocation.json']:
        shutil.copyfile(attempt / fn, out / ('failed_' + fn))
    history = []
    for old_id in state['attempts']:
        path = run / 'attempts' / old_id
        old_meta = read(path / 'attempt.json')
        old_exit = read(path / 'exit.json')
        assert old_exit['termination']['confirmed'] and not old_exit['termination']['survivors']
        assert state['attempt_usage'][old_id]['settled']
        if old_meta['stage'] != 'P3':
            continue
        old_response = read(path / 'response.json')
        old_handoff = old_response['handoff']
        old_repair = read(path / 'repair.json') if (path / 'repair.json').exists() else {}
        history.append({'attempt_id': old_id, 'role': old_meta['role'],
                        'decision': old_response['decision'],
                        'handoff_characters': len(old_handoff),
                        'handoff_nonempty_lines': sum(bool(line.strip()) for line in old_handoff.splitlines()),
                        'repair': old_repair.get('summary'), 'repair_code': old_repair.get('code'),
                        'prompt_has_length_diagnostic': '1200 characters' in (path / 'prompt.txt').read_text()})
    write(out / 'p3_history.json', history)
    previous_id = [row['attempt_id'] for row in history[:-1] if row['role'] == 'reviewer'][-1]
    previous = read(run / 'attempts' / previous_id / 'response.json')
    write(out / 'previous_reviewer_response.json', previous)
    (out / 'handoff_diff.txt').write_text('\n'.join(difflib.unified_diff(
        previous['handoff'].splitlines(), response['handoff'].splitlines(),
        fromfile=previous_id, tofile=aid, lineterm='')) + '\n')
    prompt = (attempt / 'prompt.txt').read_text()
    lines = prompt.splitlines()
    excerpts = set()
    for index, line in enumerate(lines):
        if '1200 characters' in line:
            excerpts.update(range(max(0, index - 9), min(len(lines), index + 3)))
    (out / 'failed_prompt_feedback_excerpt.txt').write_text(
        '\n'.join(f'{i+1}: {lines[i]}' for i in sorted(excerpts)) + '\n')
    gate = Path(meta['materials']['executor_root']) / 'schema_extract/tools/receipts.py'
    gate_lines = gate.read_text().splitlines()
    (out / 'frozen_handoff_gate.txt').write_text(str(gate) + '\nsha256=' + digest(gate) + '\n' +
        '\n'.join(f'{i+1}: {gate_lines[i]}' for i in range(460, 464)) + '\n')
    # Preserve identity lock, actual draft checkpoints, historical receipts/accounting evidence.
    paths = {run / 'inputs.lock.json', gate}
    for checkpoint in {state['checkpoint'], state['merged_checkpoint']}:
        paths.update(p for p in (run / 'checkpoints' / checkpoint).rglob('*') if p.is_file())
    for old_id in state['attempts']:
        path = run / 'attempts' / old_id
        paths.update(p for p in path.iterdir() if p.is_file() and p.suffix == '.json')
    for folder in ['original', 'workspace']:
        paths.update(p for p in (attempt / folder).rglob('*') if p.is_file())
    write(out / 'preserved_files_before.json', {str(p): digest(p) for p in sorted(paths)})
    validation = result['response_validation']
    checks = {
        'stopped_p3_reviewer': (state['status'], state['stage'], state['role'], state['mode']) ==
                               ('invalid_receipt', 'P3', 'reviewer', 'normal'),
        'budget_available_without_addition': state['stages']['P3']['calls'] < state['config']['max_calls'] == 32
                                             and not state['call_additions'],
        'no_attempt_or_control_fence': all(not observation[k] for k in
            ['active_attempts', 'unconfirmed_attempts', 'unsettled_attempts', 'global_hold', 'maintenance_request'])
            and not any(observation['control'].values()),
        'runner_terminated': observation['runner_observation']['lock'] == 'free'
                             and observation['runner_observation']['alive'] is False
                             and observation['launch']['status'] == 'finished',
        'all_historical_calls_settled': set(state['attempt_usage']) == set(state['attempts'])
                                      and all(u['settled'] for u in state['attempt_usage'].values()),
        'native_parse_and_identity_valid': result['exit_code'] == 0 and result['native_status']['success']
            and not result['parse_error_lines'] and validation['schema_valid']
            and validation['nonce_valid'] and validation['native_output_matches']
            and result['session_id'] == result['resume_session_id'] == state['sessions']['P3']['reviewer']
            and session['status'] == 'ready' and not session['retired_sessions'],
        'real_handoff_violation_and_feedback': response['handoff'] == receipt['handoff'] == result['response']['handoff']
            and len(response['handoff']) > 1200 and '1200 characters' in prompt
            and previous_id in prompt and aid not in state['accepted_receipts']
            and state['stages']['P3']['approval'] is None,
    }
    diagnosis = {'run_id': rid, 'observed_at': time.time(), 'attempt_id': aid,
                 'previous_reviewer_attempt': previous_id,
                 'previous_characters': len(previous['handoff']),
                 'characters': len(response['handoff']),
                 'nonempty_lines': sum(bool(line.strip()) for line in response['handoff'].splitlines()),
                 'reduction_fraction': 1 - len(response['handoff']) / len(previous['handoff']),
                 'reason': state['reason'], 'p3': state['stages']['P3'],
                 'config': state['config'], 'historical_attempts': len(state['attempts']),
                 'preconditions': checks,
                 'authorization': 'Supervisor explicitly approved one ordinary resume per run after reviewing the 3241→1418 and 2357→1396 reductions; preserve counters and all original identities.'}
    write(out / 'diagnosis.json', diagnosis)
    assert all(checks.values()), checks
    print(json.dumps({'name': name, 'calls': state['stages']['P3']['calls'],
                      'characters': diagnosis['characters'], 'preconditions': checks}, ensure_ascii=False))

def after(name):
    rid = RUNS[name]
    run = ROOT / 'runs' / rid
    out = OUT / name
    observation = inspect(rid, out, 'after')
    shutil.copyfile(run / 'state.json', out / 'after_state.json')
    state, old = read(out / 'after_state.json'), read(out / 'before_state.json')
    added = [aid for aid in state['attempts'] if aid not in old['attempts']]
    assert len(added) == 1, added
    aid = added[0]
    attempt = run / 'attempts' / aid
    meta, invocation = read(attempt / 'attempt.json'), read(attempt / 'invocation.json')
    for fn in ['attempt.json', 'invocation.json', 'process.json', 'launch.json']:
        if (attempt / fn).exists():
            shutil.copyfile(attempt / fn, out / ('resumed_' + fn))
    native_init = []
    if (attempt / 'events.jsonl').exists():
        for line in (attempt / 'events.jsonl').read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get('type') == 'system' and event.get('subtype') == 'init':
                native_init.append(event)
    write(out / 'resumed_native_init.json', native_init)
    prompt = (attempt / 'prompt.txt').read_text()
    lines = prompt.splitlines()
    excerpts = set()
    for index, line in enumerate(lines):
        if '1200 characters' in line:
            excerpts.update(range(max(0, index - 9), min(len(lines), index + 3)))
    (out / 'resumed_prompt_feedback_excerpt.txt').write_text(
        '\n'.join(f'{i+1}: {lines[i]}' for i in sorted(excerpts)) + '\n')
    files = read(out / 'preserved_files_before.json')
    changed = [path for path, sha in files.items() if not Path(path).exists() or digest(Path(path)) != sha]
    identities = ['run_id', 'input_fingerprint', 'snapshot', 'source_identity', 'executor', 'config', 'output', 'call_additions']
    drafts = ['checkpoint', 'merged_checkpoint', 'selected_designs', 'selection_signature',
              'nodes', 'writing_plan', 'writing_signature', 'plan_draft', 'plan_history', 'plan_revision']
    counts = ['receipt_repairs', 'invalid_receipts', 'revisions', 'adjudications', 'signoffs']
    session_id = old['sessions']['P3']['reviewer']
    checks = {
        'original_identities_and_budget_preserved': all(state[k] == old[k] for k in identities),
        'drafts_and_approvals_preserved': all(state[k] == old[k] for k in drafts)
            and all(state['stages'][stage]['approval'] == old['stages'][stage]['approval'] for stage in ['P1', 'P2', 'P3'])
            and state['accepted_receipts'] == old['accepted_receipts'],
        'historical_attempts_and_accounting_preserved': state['attempts'][:len(old['attempts'])] == old['attempts']
            and all(state['attempt_usage'][key] == val for key, val in old['attempt_usage'].items()) and not changed,
        'one_new_call_and_no_counter_reset': state['stages']['P3']['calls'] == old['stages']['P3']['calls'] + 1
            and all(state['stages']['P3'][key] == old['stages']['P3'][key] for key in counts),
        'exact_reviewer_session': invocation['resume_session_id'] == session_id
            and state['sessions']['P3']['reviewer'] == session_id
            and bool(native_init) and all(event['session_id'] == session_id for event in native_init)
            and meta['stage'] == 'P3' and meta['role'] == 'reviewer',
        'latest_real_failure_delivered': '1200 characters' in prompt and old['attempts'][-1] in prompt
            and invocation['prompt_sha256'] == digest(attempt / 'prompt.txt'),
        'new_call_running_with_fence': aid in observation['active_attempts'].values()
            and observation['runner_observation']['alive']
            and observation['runner_observation']['lock'] == 'held'
            and not observation['unconfirmed_attempts'],
    }
    report = {'run_id': rid, 'observed_at': time.time(), 'new_attempt': aid,
              'before_calls': old['stages']['P3']['calls'], 'after_calls': state['stages']['P3']['calls'],
              'session_id': session_id, 'checks': checks, 'changed_preserved_files': changed,
              'preserved_file_count': len(files), 'status': state['status'], 'stage': state['stage'],
              'p3_approval': state['stages']['P3']['approval'], 'native_result_exists': (attempt / 'result.json').exists()}
    write(out / 'recovery_verification.json', report)
    print(json.dumps(report, ensure_ascii=False))

if __name__ == '__main__':
    {'before': before, 'after': after}[sys.argv[1]](sys.argv[2])
