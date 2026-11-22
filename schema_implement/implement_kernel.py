#!/usr/bin/env python3
"""Implement a fixed Schema and evaluate it through the official Agate SDK."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parent
_PACKAGE = '_ks_implement_tools'
if _PACKAGE not in sys.modules:
    spec = importlib.util.spec_from_file_location(_PACKAGE, ROOT / 'tools/__init__.py',
                                                 submodule_search_locations=[str(ROOT / 'tools')])
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PACKAGE] = module
    spec.loader.exec_module(module)
coder = __import__(_PACKAGE + '.coder', fromlist=['coder'])
agate = __import__(_PACKAGE + '.agate', fromlist=['agate'])
Problem = coder.Problem


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', type=Path)
    p.add_argument('--eval-task', type=Path)
    p.add_argument('--schema', type=Path)
    p.add_argument('--kernel', type=Path)
    p.add_argument('--feedback', type=Path, action='append', default=[])
    p.add_argument('--gpu')
    p.add_argument('--out', type=Path)
    p.add_argument('--resume', type=Path)
    p.add_argument('--model')
    p.add_argument('--effort', choices=['low','medium','high','xhigh','max'])
    p.add_argument('--max-calls', type=int)
    p.add_argument('--timeout-s', type=float)
    return p


def validate_args(p, args):
    if args.resume:
        if any(getattr(args,k) is not None for k in ('task','eval_task','schema','kernel','gpu','out','model','effort','max_calls','timeout_s')):
            p.error('--resume only accepts additional --feedback; original inputs/model/limits are fixed')
        if not (args.resume / 'state.json').is_file():
            p.error('--resume requires an existing Implement state.json')
        return args.resume.resolve()
    if not args.task or not args.out:
        p.error('new runs require --task and --out')
    if args.max_calls is not None and args.max_calls <= 0:
        p.error('--max-calls must be positive')
    if args.timeout_s is not None and (not 0 < args.timeout_s < float('inf')):
        p.error('--timeout-s must be finite and positive')
    out = args.out.absolute()
    if out.exists():
        p.error('--out must be new')
    protected = [coder.CHECKOUT / n for n in (*coder.PROTECTED, 'schema_design','schema_optimize','schema_implement','schema_workflow')]
    out_resolved = out.resolve()
    if any(out_resolved == x.resolve() or out_resolved.is_relative_to(x.resolve()) for x in protected):
        p.error('--out is in a protected/product directory')
    for path in [args.task,args.eval_task,args.schema,args.kernel,*args.feedback]:
        if path:
            source = path.resolve()
            if source == out_resolved or source.is_relative_to(out_resolved) or (source.is_dir() and out_resolved.is_relative_to(source)):
                p.error('--out cannot contain or overwrite inputs, or lie within an input directory')
    return out_resolved


def rel(run, path):
    return os.path.relpath(path, run) if path else None


def result(run, state, status, reason, attempt=None):
    value = {'status':status, 'stop_reason':reason, 'calls_used':state['calls_used'],
             'schema':'inputs/schema' if state.get('schema') else None,
             'kernel':rel(run, attempt / 'kernel') if attempt and state.get('candidate') and (attempt / 'kernel').is_dir() else None,
             'feedback':rel(run, attempt / 'feedback.json') if attempt and (attempt / 'feedback.json').exists() else None}
    coder.save(run / 'result.json', value)
    state['result'] = value
    coder.save(run / 'state.json', state)
    return value


def feedback(run, state, attempt, kind, detail, *, rows=None, environment=None, reply=None):
    path = attempt / 'feedback.json'
    if path.exists():
        return coder.read_json(path)
    candidate = state.get('candidate')
    schema = state.get('schema')
    value = {'entry_point':candidate['entry_point'] if candidate else None,
             'kernel':({'path':'kernel','sha256':candidate['sha256'],'files':candidate['files']} if candidate else None),
             'schema':({'path':os.path.relpath(run / 'inputs/schema',attempt), 'sha256':schema['sha256'],
                        'files':schema['files'], 'effective_identity':state['effective_identity']} if schema else None),
             'implementation_notes':(attempt / 'notes.md').read_text() if (attempt / 'notes.md').exists() else '',
             'gaps':(reply or state.get('reply') or {}).get('gaps', []),
             'cases':rows or [], 'environment':environment or {},
             'request':'request.json' if (attempt / 'request.json').exists() else None,
             'response':'response.json' if (attempt / 'response.json').exists() else None,
             'job':None, 'diagnosis':{'kind':kind, 'detail':detail}}
    action = state.get('action') or {}
    if action.get('job_id') or action.get('reserved_job_id'):
        observed_status = coder.read_json(attempt / 'response.json').get('status') if (attempt / 'response.json').exists() else None
        value['job'] = {'id':action.get('job_id') or action['reserved_job_id'],
                        'status':observed_status, 'phase':action['phase']}
    coder.save(path, value)
    state['manifest'][rel(run,path)] = {'files':coder.inventory(path)}
    for name in ('notes.md','request.json','response.json'):
        if (attempt / name).is_file():
            state['manifest'][rel(run,attempt / name)] = {'files':coder.inventory(attempt / name)}
    return value


def next_attempt(run, state):
    n = state.get('attempts', 0) + 1
    path = run / 'attempts' / f'{n:02d}'
    path.mkdir(parents=True)
    state.update(attempts=n, current=rel(run,path), candidate=None, reply=None)
    coder.save(run / 'state.json',state)
    return path


def add_feedback(run, state, sources):
    changed = False
    for source in sources:
        identity = coder.digest({'origin':str(source.absolute()),'files':coder.inventory(source)})
        if identity in state['feedback_ids']:
            continue
        target = run / 'inputs/feedback' / f'{len(state["feedback_ids"]):03d}'
        captured = coder.freeze_feedback(source, target)
        for p, info in captured.items():
            state['manifest'][rel(run,p)] = info
        state['feedback_ids'].append(identity)
        state['feedback_paths'].append(rel(run,target))
        changed = True
    coder.save(run / 'state.json',state)
    return changed


def initialize(run, args, state):
    def capture(source, dest):
        info = coder.snapshot(source, run / dest)
        state['manifest'][dest] = info
        return info
    if not args.schema or not args.eval_task or not args.gpu:
        raise Problem('needs_input', 'Implement requires fixed --schema, --eval-task and --gpu')
    capture(args.task, 'inputs/task')
    state['task'] = 'inputs/task/' + args.task.name
    # Task links are resolved at their actual origin, with original bytes and paths.
    linked = coder.freeze_feedback(args.task, run / 'inputs/task_materials')
    for p, info in linked.items():
        state['manifest'][rel(run,p)] = info
    try:
        info, views, notices = coder.schema_view(args.schema, run / 'inputs/schema')
    except coder.template().SchemaError as e:
        raise Problem('design_gap', 'Template rejected the fixed Schema: ' + str(e)) from e
    state['schema'] = info
    state['manifest']['inputs/schema'] = info
    state['effective_identity'] = {node:entry['effective_identity'] for node,entry in views.items()}
    coder.save(run / 'inputs/effective.json', {'nodes':views,'warnings':notices})
    state['manifest']['inputs/effective.json'] = {'files':coder.inventory(run / 'inputs/effective.json')}
    capture(args.eval_task, 'inputs/eval')
    config, reference = agate.load_eval(run / 'inputs/eval')
    contract = coder.template().resolve_directory(run / 'inputs/schema').effective_document('root')['CONTRACT']
    hardware = contract['Hardware_specification']
    device = hardware['Device']
    if args.gpu.casefold() not in device.casefold():
        raise Problem('needs_input', 'Requested GPU conflicts with or is not bound by Schema Hardware.Device: ' + device)
    language = hardware['Stack']['Language'].casefold()
    if not any(x in language for x in ('triton','python','pytorch')):
        raise Problem('needs_input', 'Schema implementation stack cannot be expressed by this Python Model/Triton adapter')
    if args.kernel:
        capture(args.kernel, 'inputs/seed')
        state['seed'] = 'inputs/seed'
    for name in ['IMPLEMENT_SCHEMA.md', 'schema_docs/0.common.md', 'schema_docs/1.contract.algorithm_requirement.md',
                 'schema_docs/2.contract.hardware_specification.md', 'schema_docs/3.design.math_reshape.md',
                 'schema_docs/4.design.work_schedule.md','schema_docs/5.design.tile_pipeline.md',
                 'schema_docs/6.realization.md','schema_template/README.md','schema_template/starter.yaml']:
        capture(coder.CHECKOUT / name, 'inputs/rules/' + name.replace('/','__'))
    add_feedback(run,state,args.feedback)
    state['initialized'] = True
    coder.save(run / 'state.json',state)


def prompt(run, state, attempt):
    rules = run / 'inputs/rules'
    common = (rules / 'schema_docs__0.common.md/0.common.md').read_text()
    method = (rules / 'IMPLEMENT_SCHEMA.md/IMPLEMENT_SCHEMA.md').read_text()
    seed = state.get('seed')
    if state.get('previous_kernel'):
        seed = state['previous_kernel']
    return '\n\n'.join([
        (ROOT / 'prompt/implement.md').read_text(), 'DOC0:\n' + common, 'Implement method:\n' + method,
        'Fixed task:\n' + (run / state['task']).read_text(),
        'Read the complete material tree below. Paths are absolute; original locations and hashes are in state.json.',
        'RAW schema: ' + str(run / 'inputs/schema'), 'Effective schemas, complete from chains: ' + str(run / 'inputs/effective.json'),
        'Fixed eval (reference, input, all cases, tolerance/timing): ' + str(run / 'inputs/eval'),
        'DOC1–6, Template README/starter: ' + str(rules),
        'Task external bindings: ' + str(run / 'inputs/task_materials'),
        'Complete previous source/seed: ' + (str(run / seed) if seed else 'none'),
        'External repair feedback (read raw and all referenced artifacts): ' + json.dumps([str(run / p) for p in state['feedback_paths']]),
        'Most recent actual feedback: ' + (str(run / state['last_feedback']) if state.get('last_feedback') else 'none'),
        'Host diagnostic from last attempt: ' + state.get('diagnostic','none'),
        'Requested GPU: ' + state['config']['gpu'],
    ])


def execute(run, state, *, new_feedback=False, client_factory=None, writer=None):
    persist = lambda: coder.save(run / 'state.json',state)
    client_factory = client_factory or agate.connect
    writer = writer or coder.invoke
    coder.check_manifest(run, state['manifest'])
    action = state.get('action')
    if action and action['kind'] == 'coder':
        if coder.active_process(action):
            raise Problem('interrupted','Prior coder process may still be active; locate the recorded PID/session')
        state['diagnostic'] = 'Previous coder exited without a committed candidate; counted call is retained'
        state['action'] = None
        persist()
    if not state.get('action') and state.get('result',{}).get('status') != 'interrupted' and state.get('result') and not new_feedback:
        return state['result']
    if (not state.get('action') and state.get('result',{}).get('stop_reason') in {'evaluation_failed','input_changed','needs_input'}):
        # Added prose cannot turn infrastructure/material failure into a code repair.
        return state['result']
    config, reference = agate.load_eval(run / 'inputs/eval')
    client, sdk_version = client_factory(state['config']['timeout_s'])
    identity = getattr(client, 'connection_identity', None)
    if identity is not None:
        if state.get('connection_identity') not in (None,identity):
            raise Problem('interrupted','Agate connection changed; restore the original connection before locating its job')
        state['connection_identity'] = identity
    state['sdk_version'] = sdk_version
    persist()
    while True:
        coder.check_manifest(run, state['manifest'])
        if state.get('action') and state['action']['kind'] == 'agate':
            attempt = run / state['current']
            if coder.inventory(attempt / 'kernel') != state['candidate']['files']:
                raise Problem('input_changed','Fixed candidate bytes differ before Agate recovery')
            if state['action']['phase'] == 'terminal':
                response = coder.read_json(attempt / 'response.json')
            else:
                response = agate.evaluate(run,attempt,state,persist,client)
            rows, environment, reason, detail = agate.interpret(response,reference,config)
            gaps = state['reply']['gaps']
            if gaps and reason == 'done':
                reason, detail = gaps[0]['kind'], gaps[0]['detail']
            feedback(run,state,attempt,reason,detail,rows=rows,environment=environment)
            state.update(last_feedback=rel(run,attempt / 'feedback.json'), previous_kernel=rel(run,attempt / 'kernel'), action=None)
            persist()
            pending_feedback = len(state['feedback_ids']) > state['consumed_feedback']
            if reason == 'done' and not pending_feedback:
                return result(run,state,'complete','done',attempt)
            if reason not in {'implementation_failed','done'}:
                return result(run,state,'incomplete',reason,attempt)
            state['diagnostic'] = detail
        if state['calls_used'] >= state['config']['max_calls']:
            attempt = run / state['current'] if state.get('current') else next_attempt(run,state)
            if (attempt / 'feedback.json').exists() and coder.read_json(attempt / 'feedback.json')['diagnosis']['kind'] == 'done':
                attempt = next_attempt(run,state)
                state['diagnostic'] = 'No coder allowance remains for the newly supplied repair feedback'
            feedback(run,state,attempt,'implementation_failed',state.get('diagnostic','Coder allowance exhausted'))
            reason = (coder.read_json(attempt / 'feedback.json')['diagnosis']['kind'])
            return result(run,state,'incomplete',reason if reason != 'done' else 'implementation_failed',attempt)
        attempt = next_attempt(run,state)
        # Consume newly appended feedback only when the counted coder action begins.
        state['consumed_feedback'] = len(state['feedback_ids'])
        persist()
        try:
            reply = writer(prompt(run,state,attempt),run,attempt,state,persist,[run / 'inputs',run / 'attempts'])
            state['reply'] = reply
            (attempt / 'notes.md').write_text(reply['notes'],encoding='utf-8')
            candidate = coder.materialize(reply,attempt / 'kernel')
            state['candidate'] = candidate
            if candidate:
                state['manifest'][rel(run,attempt / 'kernel')] = {'files':candidate['files']}
            persist()
            if any(g['kind'] in {'design_gap','needs_input'} for g in reply['gaps']):
                gap = next(g for g in reply['gaps'] if g['kind'] in {'design_gap','needs_input'})
                feedback(run,state,attempt,gap['kind'],gap['detail'],reply=reply)
                return result(run,state,'incomplete',gap['kind'],attempt)
            if not candidate:
                raise Problem('implementation_failed','Coder supplied no complete kernel and no concrete design/input diagnosis')
            if reply['gaps']:
                raise Problem('implementation_failed','Known unresolved implementation gap: ' + reply['gaps'][0]['detail'])
            key = 'ks-implement-' + uuid.uuid4().hex
            request = agate.build_request(attempt / 'kernel',candidate['entry_point'],config,reference,
                                         state['config']['gpu'],key,'ks-' + run.name + '-' + attempt.name)
            coder.save(attempt / 'request.json',request)
            state['action'] = {'kind':'agate','phase':'prepared','key':key,'request_sha256':coder.digest(request)}
            persist()
        except Problem as e:
            if e.kind == 'interrupted':
                raise
            feedback(run,state,attempt,e.kind,e.detail,reply=state.get('reply'))
            state.update(last_feedback=rel(run,attempt / 'feedback.json'),diagnostic=e.detail)
            if state.get('candidate'):
                state['previous_kernel'] = rel(run,attempt / 'kernel')
            persist()
            if e.kind not in {'write_failed','implementation_failed'}:
                return result(run,state,'incomplete',e.kind,attempt)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    run = validate_args(p,args)
    if not args.resume:
        run.mkdir(parents=True)
    directory_fd = os.open(run, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            fcntl.flock(directory_fd,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Implement directory is already active',file=sys.stderr)
            return 2
        if args.resume:
            state = coder.read_json(run / 'state.json')
        else:
            state = {'config':{'model':args.model or 'qwen3.8-max[1m]','effort':args.effort or 'max',
                      'max_calls':args.max_calls or 6,'timeout_s':args.timeout_s or 900,'gpu':args.gpu},
                     'calls_used':0,'manifest':{},'feedback_ids':[],'feedback_paths':[],
                     'consumed_feedback':0,'action':None,'initialized':False}
            coder.save(run / 'state.json',state)
        try:
            if not args.resume:
                initialize(run,args,state)
            elif not state.get('initialized'):
                raise Problem('needs_input','Original run initialization failed; its fixed inputs cannot be replaced on resume')
            new_feedback = add_feedback(run,state,args.feedback) if args.resume else bool(args.feedback)
            new_feedback = new_feedback or len(state['feedback_ids']) > state['consumed_feedback']
            value = execute(run,state,new_feedback=new_feedback)
        except KeyboardInterrupt:
            value = result(run,state,'interrupted','interrupted',run / state['current'] if state.get('current') else None)
        except Problem as e:
            state['diagnostic'] = e.detail
            print(e.detail,file=sys.stderr)
            attempt = run / state['current'] if state.get('current') else next_attempt(run,state)
            if e.kind != 'interrupted':
                if (attempt / 'feedback.json').exists():
                    attempt = next_attempt(run,state)
                feedback(run,state,attempt,e.kind,e.detail)
            value = result(run,state,'interrupted' if e.kind == 'interrupted' else 'incomplete',e.kind,attempt)
        except (FileNotFoundError, json.JSONDecodeError, UnicodeError) as e:
            state['diagnostic'] = 'Required fixed material is missing or malformed: ' + type(e).__name__
            attempt = next_attempt(run,state)
            feedback(run,state,attempt,'needs_input',state['diagnostic'])
            value = result(run,state,'incomplete','needs_input',attempt)
        except Exception as e:
            # No raw exception repr: SDK/CLI configuration can contain connection data.
            state['diagnostic'] = 'Host failure: ' + type(e).__name__
            print(state['diagnostic'],file=sys.stderr)
            attempt = run / state['current'] if state.get('current') else next_attempt(run,state)
            pending = bool(state.get('action'))
            if not pending:
                feedback(run,state,attempt,'evaluation_failed',state['diagnostic'])
            value = result(run,state,'interrupted' if pending else 'incomplete',
                           'interrupted' if pending else 'evaluation_failed',attempt)
        print(str((run / 'result.json').absolute()))
        return {'complete':0,'incomplete':1,'interrupted':2}[value['status']]
    finally:
        os.close(directory_fd)


if __name__ == '__main__':
    raise SystemExit(main())
