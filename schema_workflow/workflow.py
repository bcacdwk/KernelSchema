#!/usr/bin/env python3
"""Run the standalone Design, Implement and Optimize CLIs with finite iteration."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

CHECKOUT = Path(__file__).resolve().parents[1]
SCRIPTS = {name: CHECKOUT / f'schema_{name}' / filename for name, filename in
           [('design', 'design_schema.py'), ('implement', 'implement_kernel.py'), ('optimize', 'optimize_schema.py')]}
CODES = {'complete': 0, 'incomplete': 1, 'interrupted': 2}
PROTECTED = ('schema_extract', 'schema_docs', 'schema_template', 'schema_wiki', 'kernel_wiki',
             'schema_design', 'schema_optimize', 'schema_implement', 'schema_workflow')


class Problem(Exception):
    def __init__(self, reason, detail):
        self.reason, self.detail = reason, detail
        super().__init__(detail)


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    """Hash ordinary input bytes without interpreting Schema or evaluator data."""
    path = Path(path)
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise Problem('input_changed', f'Symlink input is not supported: {path}')
    if not path.exists():
        raise Problem('needs_input', f'Missing input: {path}')
    entries = [path] if path.is_file() else sorted(path.rglob('*'))
    values = {}
    for item in entries:
        if item.is_symlink():
            raise Problem('input_changed', f'Symlink input is not supported: {item}')
        if item.is_file():
            values[item.name if path.is_file() else item.relative_to(path).as_posix()] = hashlib.sha256(item.read_bytes()).hexdigest()
        elif not item.is_dir():
            raise Problem('needs_input', f'Not an ordinary input: {item}')
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def relative(run, path):
    return os.path.relpath(path, run) if path else None


def artifact(result_path, value, key):
    name = value.get(key)
    if name is None:
        return None
    if not isinstance(name, str) or Path(name).is_absolute():
        raise Problem('write_failed', f'Invalid {key} path in {result_path}')
    original = result_path.parent / name
    path = original.resolve()
    # Business CLIs own their fixed snapshots and outputs.
    if not path.is_relative_to(result_path.parent.resolve()) or not path.exists():
        raise Problem('write_failed', f'Missing or escaping {key} in {result_path}')
    digest(original)
    return str(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('task', 'eval-task', 'schema', 'kernel', 'out', 'resume'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--reference', action='append', type=Path, default=[],
                   help='existing reference file or Schema directory for the Task → Design start; repeatable')
    p.add_argument('--feedback', action='append', type=Path, default=[])
    p.add_argument('--gpu')
    p.add_argument('--model')
    p.add_argument('--effort', choices=['low', 'medium', 'high', 'xhigh', 'max'])
    p.add_argument('--max-versions', type=int)
    p.add_argument('--max-calls-per-schema', type=int)
    p.add_argument('--timeout-s', type=float)
    return p


def input_paths(args):
    for path in [args.task, args.eval_task, args.schema, args.kernel, *args.feedback, *args.reference]:
        if path:
            # Design treats a supplied file beside root/dispatch as a complete Schema reference.
            schema_reference = (path in args.reference and (path.parent / 'dispatch.md').is_file()
                                and any((path.parent / name).is_file() for name in ('root.yaml', 'root.yml')))
            yield path.parent if path.is_file() and (path == args.schema or schema_reference) else path


def arguments(p, args):
    if args.resume:
        if any(getattr(args, k) is not None for k in ('task', 'eval_task', 'schema', 'kernel', 'out', 'gpu',
                'model', 'effort', 'max_versions', 'max_calls_per_schema', 'timeout_s')) or args.feedback or args.reference:
            p.error('--resume uses only the original inputs, model, limits and directory')
        if not (args.resume / 'state.json').is_file():
            p.error('--resume requires a Workflow state.json')
        return args.resume.resolve()
    if not args.task or not args.out:
        p.error('new runs require --task and --out')
    if args.reference and args.schema:
        p.error('--reference is only used by the Task → Design start without --schema')
    if args.kernel and (not args.schema or not args.feedback):
        p.error('--kernel requires --schema and --feedback')
    if args.feedback and not args.kernel:
        p.error('the external-evidence start requires --schema, --kernel and --feedback')
    if any(x is not None and x <= 0 for x in (args.max_versions, args.max_calls_per_schema)):
        p.error('call/version limits must be positive integers')
    if args.timeout_s is not None and not 0 < args.timeout_s < float('inf'):
        p.error('--timeout-s must be positive and finite')
    if args.model is not None and not args.model.strip():
        p.error('--model cannot be empty')
    out = args.out.expanduser().absolute()
    if any(x.is_symlink() for x in (out, *out.parents)):
        p.error('--out cannot use symlinks')
    out = out.resolve()
    if out.exists():
        p.error('--out must be new')
    protected = [CHECKOUT / n for n in PROTECTED]
    protected += [Path.home() / n for n in ('.claude', '.codex', '.ssh', '.aws')]
    if os.environ.get('CLAUDE_CONFIG_DIR'):
        protected.append(Path(os.environ['CLAUDE_CONFIG_DIR']).expanduser().resolve())
    if any(out == x or out.is_relative_to(x) or x.is_relative_to(out) for x in protected):
        p.error('--out overlaps a protected directory')
    for name in ('task', 'eval_task', 'schema', 'kernel'):
        path = getattr(args, name)
        if path:
            setattr(args, name, path.expanduser().absolute())
    args.feedback = [x.expanduser().absolute() for x in args.feedback]
    args.reference = [x.expanduser().absolute() for x in args.reference]
    for path in input_paths(args):
        source = path.resolve()
        if source == out or source.is_relative_to(out) or (source.is_dir() and out.is_relative_to(source)):
            p.error('--out cannot overlap an input')
    return out


def process_identity(pid):
    try:
        value = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return value[19] if value[0] != 'Z' else None
    except (FileNotFoundError, ProcessLookupError):
        return None


def live(action):
    pid = action.get('pid')
    if not pid:
        return False
    current = process_identity(pid)
    return current is not None and current == action.get('process_start')


class Workflow:
    def __init__(self, run, state):
        self.run, self.state = run, state

    def persist(self):
        save(self.run / 'state.json', self.state)

    def finish(self, status, reason, detail=None):
        if detail:
            self.state['diagnostic'] = detail
            print(detail, file=sys.stderr)
        value = {'status': status, 'stop_reason': reason, 'last_usable': self.state['last_usable'],
                 'latest_attempt': self.state['latest_attempt']}
        self.state['result'] = value
        self.persist()
        save(self.run / 'result.json', value)
        return value

    def limit(self):
        version = self.state['versions'][-1]
        okay = (self.state['last_usable'] is not None and self.state.get('verified_decision')
                and not self.state['required_fix'] and version.get('implementation', {}).get('status') == 'complete')
        return self.finish('complete' if okay else 'incomplete', 'limit')

    def version(self, schema, seed=None, feedback=()):
        self.state['versions'].append({'schema': schema, 'seed': seed, 'feedback': list(feedback),
                                       'implement': None, 'implementation': None, 'optimize': []})

    def plan(self, module, out, args, **context):
        self.state['action'] = {'module': module, 'out': relative(self.run, out), 'args': args,
                                'pid': None, 'process_start': None,
                                'prior_result': digest(out / 'result.json') if (out / 'result.json').exists() else None,
                                **context}
        self.persist()

    def common(self):
        config = self.state['config']
        return ['--task', config['task'], '--model', config['model'], '--effort', config['effort'],
                '--timeout-s', str(config['timeout_s'])]

    def implement(self, extra=()):
        v = self.state['versions'][-1]
        out = self.run / f'v{len(self.state["versions"]):03d}/implement'
        if v['implement']:
            args = ['--resume', str(out)]
            feedback = list(extra)
        else:
            args = self.common() + ['--schema', v['schema'], '--out', str(out),
                    '--max-calls', str(self.state['config']['max_calls_per_schema'])]
            for option in ('eval_task', 'gpu'):
                if self.state['config'][option]:
                    args += ['--' + option.replace('_', '-'), self.state['config'][option]]
            if v['seed']:
                args += ['--kernel', v['seed']]
            feedback = list(dict.fromkeys([*v['feedback'], *extra]))
        for path in feedback:
            args += ['--feedback', path]
        self.plan('implement', out, args, feedback=feedback)

    def optimize(self, external=False):
        v = self.state['versions'][-1]
        remaining = 2 - sum(item['calls_used'] for item in v['optimize'])
        if remaining <= 0:
            return False
        out = self.run / f'v{len(self.state["versions"]):03d}/optimize/{len(v["optimize"]) + 1:02d}'
        imp = v['implementation']
        feedback = v['feedback'] if external else [str(self.run / v['implement'])]
        kernel = v['seed'] if external else artifact(self.run / v['implement'], imp, 'kernel')
        args = self.common() + ['--schema', v['schema'], '--out', str(out), '--max-calls', str(1 if external else remaining)]
        if kernel:
            args += ['--kernel', kernel]
        for path in feedback:
            args += ['--feedback', path]
        verified = not external and imp['status'] == 'complete'
        self.plan('optimize', out, args, external=external, verified=verified,
                  observed_feedback=artifact(self.run / v['implement'], imp, 'feedback') if verified else None)
        return True

    def child(self):
        self.check_inputs()
        action = self.state['action']
        module = action['module']
        out = self.run / action['out']
        path = out / 'result.json'
        if live(action):
            raise Problem('interrupted', 'The recorded child is still active; resume after it exits')
        if action.get('ack'):
            value = self.validate_child(path, action['ack']['returncode'], action['ack']['stdout'])
            if value['status'] != 'interrupted':
                return value
        elif out.exists() and path.is_file() and digest(path) != action.get('prior_result'):
            value = self.validate_child(path)
            if value['status'] != 'interrupted':
                return value
        if out.exists():
            if module != 'implement' or not (out / 'state.json').is_file():
                raise Problem('interrupted', 'Writing output has no trusted terminal result; locate its recorded draft/process before resuming')
            # The Implement CLI owns all original-job and coder recovery.
            args = ['--resume', str(out)]
            for feedback in action.get('feedback', []):
                args += ['--feedback', feedback]
        else:
            args = action['args']
        action.pop('ack', None)
        self.persist()
        print(f'workflow: {module} {action["out"]}', file=sys.stderr)
        process = subprocess.Popen([sys.executable, str(SCRIPTS[module]), *args], cwd=self.run,
                                   stdout=subprocess.PIPE, text=True, shell=False, start_new_session=True)
        action.update(pid=process.pid, process_start=process_identity(process.pid))
        self.persist()
        try:
            stdout, _ = process.communicate()
        except KeyboardInterrupt:
            # Ask the same child to checkpoint. Never replace a possibly live child.
            try:
                os.kill(process.pid, signal.SIGINT)
                process.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
            raise Problem('interrupted', 'Workflow interrupted; the original child/action is retained')
        action['ack'] = {'returncode': process.returncode, 'stdout': stdout}
        self.persist()
        return self.validate_child(path, process.returncode, stdout)

    def validate_child(self, path, code=None, stdout=None):
        if path.exists():
            self.state['latest_attempt'] = relative(self.run, path)
        if stdout is not None and stdout.splitlines() != [str(path.absolute())]:
            raise Problem('interrupted', 'Child stdout did not identify its one expected result path')
        try:
            value = read(path)
        except (OSError, ValueError) as exc:
            raise Problem('interrupted', f'No readable child result at {path}') from exc
        if value.get('status') not in CODES or (code is not None and code != CODES[value['status']]):
            raise Problem('interrupted', 'Child exit code and result status disagree')
        if type(value.get('calls_used')) is not int or value['calls_used'] < 0:
            raise Problem('interrupted', 'Child call usage is unknown; no replacement run is allowed')
        module = self.state['action']['module']
        maximum = (self.state['config']['max_calls_per_schema'] if module == 'implement' else
                   int(self.state['action']['args'][self.state['action']['args'].index('--max-calls') + 1]))
        if value['calls_used'] > maximum:
            raise Problem('interrupted', 'Child exceeded its fixed call allowance')
        if module == 'implement':
            previous = self.state['versions'][-1].get('implementation')
            if previous and value['calls_used'] < previous['calls_used']:
                raise Problem('interrupted', 'Implement call usage decreased on resume')
        for key in ('schema', 'kernel', 'feedback', 'notes'):
            if key in value:
                artifact(path, value, key)
        if value['status'] == 'complete':
            required = ['schema', 'kernel', 'feedback'] if module == 'implement' else ['schema', 'notes']
            if not all(value.get(key) for key in required):
                raise Problem('write_failed', 'Complete child result lacks its required artifacts')
            if module == 'optimize':
                decision, fix = value.get('decision'), value.get('required_fix')
                if decision not in ('keep', 'repair', 'revise') or type(fix) is not bool or (decision == 'keep' and fix) or (decision == 'repair' and not fix):
                    raise Problem('write_failed', 'Invalid Optimize decision/required_fix')
        return value

    def consume(self, value):
        action = self.state['action']
        module = action['module']
        path = self.run / action['out'] / 'result.json'
        self.state['latest_attempt'] = relative(self.run, path)
        if value['status'] == 'interrupted':
            return self.finish('interrupted', 'interrupted')
        self.state['action'] = None
        for key in ('schema', 'kernel', 'feedback', 'notes'):
            if value.get(key):
                fixed = artifact(path, value, key)
                self.state.setdefault('artifacts', {})[fixed] = digest(Path(fixed))
        if module != 'implement':
            self.state.setdefault('artifacts', {})[str(path)] = digest(path)
        if module == 'design':
            self.state['design'] = relative(self.run, path)
            if value['status'] != 'complete':
                return self.finish('incomplete', value['stop_reason'])
            self.version(artifact(path, value, 'schema'))
            self.implement()
            return None
        v = self.state['versions'][-1]
        if module == 'implement':
            v['implement'], v['implementation'] = relative(self.run, path), value
            if value['status'] == 'complete':
                self.state['last_usable'] = {key: relative(self.run, artifact(path, value, key))
                                              for key in ('schema', 'kernel', 'feedback')}
                self.state['required_fix'] = False
            elif value['stop_reason'] == 'design_gap':
                self.state['last_usable'], self.state['required_fix'] = None, True
            elif value['stop_reason'] != 'implementation_failed' or not value.get('kernel'):
                return self.finish('incomplete', value['stop_reason'])
            if self.optimize():
                return None
            if value['status'] != 'complete' and value['stop_reason'] != 'design_gap':
                return self.finish('incomplete', value['stop_reason'])
            return self.limit()
        v['optimize'].append({'result': relative(self.run, path), 'calls_used': value['calls_used'],
                              'feedback': action.get('observed_feedback')})
        if value.get('required_fix') is True:
            self.state['required_fix'], self.state['last_usable'] = True, None
        if value['status'] != 'complete':
            return self.finish('incomplete', value['stop_reason'])
        if action['verified']:
            self.state['verified_decision'] = relative(self.run, path)
        decision = value['decision']
        if decision == 'keep':
            if v['implementation']:
                if v['implementation']['status'] == 'complete':
                    return self.finish('complete', 'done')
                return self.finish('incomplete', v['implementation']['stop_reason'])
            self.implement([str(path)])
        elif decision == 'repair':
            if v['implementation'] and v['implementation']['calls_used'] >= self.state['config']['max_calls_per_schema']:
                return self.finish('incomplete', 'limit')
            self.implement([str(path)])
        elif len(self.state['versions']) >= self.state['config']['max_versions']:
            return self.limit()
        else:
            seed = (artifact(self.run / v['implement'], v['implementation'], 'kernel')
                    if v['implementation'] else v['seed']) or v['seed']
            self.version(artifact(path, value, 'schema'), seed, [str(path)])
            self.implement()
        return None

    def check_inputs(self):
        for path, expected in {**self.state['inputs'], **self.state.get('artifacts', {})}.items():
            try:
                unchanged = digest(Path(path)) == expected
            except (Problem, OSError):
                unchanged = False
            if not unchanged:
                self.state['last_usable'] = None
                raise Problem('input_changed', f'Original input changed: {path}')

    def execute(self):
        self.check_inputs()
        if self.state.get('result') and self.state['result']['status'] != 'interrupted' and not self.state['action']:
            return self.state['result']
        if not self.state['action']:
            if self.state['versions']:
                if self.state['config']['kernel']:
                    self.optimize(external=True)
                else:
                    self.implement()
            else:
                args = self.common() + ['--out', str(self.run / 'design'), '--max-calls', '3']
                for path in self.state['config'].get('reference', []):
                    args += ['--reference', path]
                self.plan('design', self.run / 'design', args)
        while True:
            result = self.consume(self.child())
            if result:
                return result


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    run = arguments(p, args)
    if not args.resume:
        run.mkdir(parents=True)
    descriptor = os.open(run, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Workflow directory is already active', file=sys.stderr)
            return 2
        if args.resume:
            state = read(run / 'state.json')
        else:
            config = {key: str(getattr(args, key)) if getattr(args, key) is not None else None
                      for key in ('task', 'eval_task', 'schema', 'kernel', 'gpu')}
            config.update(model=args.model or 'qwen3.8-max[1m]', effort=args.effort or 'max',
                          max_versions=args.max_versions or 3, max_calls_per_schema=args.max_calls_per_schema or 6,
                          timeout_s=args.timeout_s or 900, reference=list(map(str, args.reference)))
            state = {'config': config, 'inputs': {}, 'versions': [], 'action': None, 'design': None,
                     'last_usable': None, 'latest_attempt': None, 'required_fix': False, 'verified_decision': None}
        workflow = Workflow(run, state)
        try:
            if not args.resume:
                for path in input_paths(args):
                    state['inputs'][str(path)] = digest(path)
                if args.schema:
                    workflow.version(str(args.schema), str(args.kernel) if args.kernel else None, map(str, args.feedback))
                workflow.persist()
            value = workflow.execute()
        except KeyboardInterrupt:
            value = workflow.finish('interrupted', 'interrupted', 'Workflow interrupted; resume the recorded action')
        except Problem as exc:
            value = workflow.finish('interrupted' if exc.reason == 'interrupted' else 'incomplete', exc.reason, exc.detail)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            value = workflow.finish('interrupted' if state.get('action') else 'incomplete',
                                    'interrupted' if state.get('action') else 'write_failed',
                                    'Workflow could not reconcile its fixed state: ' + type(exc).__name__)
        print(str(run / 'result.json'))
        return CODES[value['status']]
    finally:
        os.close(descriptor)


if __name__ == '__main__':
    raise SystemExit(main())
