"""Implement's read-only writer and immutable material handling (no other runtime)."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import tempfile
import uuid

CHECKOUT = Path(__file__).resolve().parents[2]
CONNECTION_KEYS = {
    'ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL',
    'ANTHROPIC_CUSTOM_HEADERS', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY',
    'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy', 'NODE_EXTRA_CA_CERTS',
    'SSL_CERT_FILE', 'SSL_CERT_DIR',
}
PROTECTED = ('schema_extract', 'schema_docs', 'schema_template', 'schema_wiki', 'kernel_wiki', '.git')


class Problem(Exception):
    def __init__(self, kind, detail):
        super().__init__(detail)
        self.kind, self.detail = kind, detail


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode())


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def regular(path):
    path = Path(path)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise Problem('needs_input', f'Symlinks are not accepted as input: {path}')
    if not path.is_file():
        raise Problem('needs_input', f'Missing regular file: {path}')
    return path.read_bytes()


def inventory(path):
    path = Path(path)
    if path.is_symlink():
        raise Problem('input_changed', f'Symlink source: {path}')
    if path.is_file():
        return {path.name: sha(regular(path))}
    if not path.is_dir():
        raise Problem('needs_input', f'Missing source: {path}')
    files = {}
    for p in sorted(path.rglob('*')):
        if p.is_symlink():
            raise Problem('needs_input', f'Symlink in source: {p}')
        if p.is_file():
            files[p.relative_to(path).as_posix()] = sha(regular(p))
        elif not p.is_dir():
            raise Problem('needs_input', f'Non-regular source: {p}')
    return files


def snapshot(source, dest):
    """Copy original bytes once, compare both sides after copying, keep provenance."""
    source, dest = Path(source).absolute(), Path(dest)
    credential_roots = [Path.home() / '.claude', Path.home() / '.atrex', Path.home() / '.codex',
                        Path(os.environ.get('CLAUDE_CONFIG_DIR', str(Path.home() / '.claude')))]
    actual = source.resolve()
    if any(actual == p.resolve() or actual.is_relative_to(p.resolve()) or
           (source.is_dir() and p.resolve().is_relative_to(actual)) for p in credential_roots):
        raise Problem('needs_input', 'A material path would expose a CLI/authentication directory')
    before = inventory(source)
    if not before:
        raise Problem('needs_input', f'Empty input: {source}')
    dest.mkdir(parents=True, exist_ok=False)
    root = source if source.is_dir() else source.parent
    for name in before:
        p = dest / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(regular(root / name))
    if inventory(source) != before or inventory(dest) != before:
        raise Problem('input_changed', f'Input changed during snapshot: {source}')
    return {'origin': str(source), 'files': before, 'sha256': digest(before)}


def check_manifest(run, manifest):
    for rel, info in manifest.items():
        if inventory(run / rel) != info['files']:
            raise Problem('input_changed', f'Fixed input differs: {rel}')


def safe_relative(name):
    if not isinstance(name, str) or not name or '\\' in name or '\x00' in name:
        raise Problem('write_failed', 'Expected a normalized relative source path')
    p = PurePosixPath(name)
    if p.is_absolute() or str(p) != name or any(x in ('..', '.', '') for x in p.parts):
        raise Problem('write_failed', f'Unsafe source path: {name}')
    return p


def materialize(reply, dest):
    files = reply['files']
    if not files:
        return None
    names = [str(safe_relative(f['path'])) for f in files]
    if len(set(names)) != len(names):
        raise Problem('write_failed', 'Duplicate source path in model response')
    for name in names:
        if any(name.startswith(other + '/') for other in names):
            raise Problem('write_failed', 'Source file also used as a directory')
    entry = str(safe_relative(reply['entry_point']))
    if entry not in names or not entry.endswith('.py'):
        raise Problem('write_failed', 'Missing Python Model entry point in complete file set')
    for item in files:
        if item['path'].endswith('.py'):
            try:
                ast.parse(item['content'], filename=item['path'])
            except SyntaxError as e:
                raise Problem('implementation_failed', f'{item["path"]}:{e.lineno}: {e.msg}') from e
    tree = ast.parse(next(f['content'] for f in files if f['path'] == entry))
    if not any(isinstance(n, ast.ClassDef) and n.name == 'Model' for n in tree.body):
        raise Problem('implementation_failed', 'Entry must export top-level class Model')
    dest.mkdir()
    for item in files:
        p = dest / item['path']
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(item['content'], encoding='utf-8')
    return {'entry_point': entry, 'files': inventory(dest), 'sha256': digest(inventory(dest))}


def template():
    name = '_ks_implement_template'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, CHECKOUT / 'schema_template/tools/__init__.py',
                        submodule_search_locations=[str(CHECKOUT / 'schema_template/tools')])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return __import__(name + '.schema', fromlist=['schema'])


def schema_view(source, target):
    api = template()
    source = Path(source)
    directory = source if source.is_dir() else source.parent
    notices = []
    # Validate original repository locators before copying into a detached snapshot.
    api.validate_directory(directory, notices=notices)
    resolved = api.resolve_directory(directory)
    info = snapshot(directory, target)
    copied = api.resolve_directory(target)
    if copied.source_hashes != resolved.source_hashes:
        raise Problem('input_changed', 'Schema changed while fixing effective sources')
    views = {node: {'effective_document': resolved.effective_document(node),
                    'effective_yaml': api.effective_yaml(resolved, node),
                    'source_manifest': resolved.source_manifest(node),
                    'origins': resolved.origins_dict(node),
                    'effective_identity': resolved.effective_identity(node)}
             for node in resolved.raw_task}
    # Decimal values are represented by the exact YAML text, not coerced JSON floats.
    for v in views.values():
        del v['effective_document']
    return info, views, notices


def freeze_feedback(source, dest):
    """Retain raw feedback and resolve referenced files from their original context."""
    manifest, seen = {}, {}
    def capture(path):
        path = Path(path).absolute()
        key = str(path)
        if key in seen:
            return seen[key]
        target = dest / f'{len(seen):03d}'
        seen[key] = str(target)
        info = snapshot(path, target)
        manifest[str(target)] = info
        if path.is_file():
            try:
                value = json.loads(regular(path))
            except (ValueError, UnicodeError):
                value = None
            def visit(v, key=''):
                if isinstance(v, dict):
                    for k, x in v.items():
                        visit(x, k)
                elif isinstance(v, list):
                    for x in v:
                        visit(x, key)
                elif isinstance(v, str) and key in {'path','schema','kernel','feedback','notes','request','response'}:
                    p = Path(v)
                    p = p if p.is_absolute() else path.parent / p
                    if p.exists():
                        capture(p)
            visit(value)
            if path.suffix == '.md':
                for link in re.findall(r'\]\(([^)]+)\)', regular(path).decode()):
                    p = path.parent / link.split('#')[0].strip('<>')
                    if p.exists() and p != path.parent:
                        capture(p)
        return str(target)
    capture(source)
    return manifest


def cli_environment(effort):
    env = os.environ.copy()
    config = Path(env.get('CLAUDE_CONFIG_DIR', str(Path.home() / '.claude'))) / 'settings.json'
    try:
        additions = read_json(config).get('env', {})
    except FileNotFoundError:
        additions = {}
    for key in CONNECTION_KEYS:
        if key not in env and isinstance(additions.get(key), str):
            env[key] = additions[key]
    env.update({'CLAUDE_CODE_EFFORT_LEVEL': effort, 'CLAUDE_CODE_MAX_OUTPUT_TOKENS': '128000',
                'DISABLE_AUTOUPDATER': '1', 'DISABLE_TELEMETRY': '1',
                'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1'})
    return env


def redact(text, env=None):
    if env:
        for key in CONNECTION_KEYS:
            if env.get(key):
                text = text.replace(env[key], '[connection redacted]')
    return re.sub(r'https?://[^\s"\']+\?[^\s"\']+', '[signed URL redacted]', text)


REPLY_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['files', 'entry_point', 'notes', 'gaps'],
    'properties': {
        'files': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False,
                  'required': ['path','content'], 'properties': {'path': {'type':'string'}, 'content': {'type':'string'}}}},
        'entry_point': {'type': ['string','null']}, 'notes': {'type':'string'},
        'gaps': {'type':'array', 'items': {'type':'object', 'additionalProperties':False,
                 'required':['kind','detail'], 'properties': {
                     'kind': {'enum':['needs_input','design_gap','implementation_failed']},
                     'detail': {'type':'string', 'minLength':1}}}},
    },
}


def process_identity(pid):
    try:
        parts = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if parts[0] == 'Z':
            return None
        return parts[19]
    except (OSError, IndexError):
        return None


def active_process(action):
    pid = action.get('pid')
    if not pid:
        return action.get('phase') == 'starting'
    current = process_identity(pid)
    return current is not None and (action.get('process_start') is None or current == action['process_start'])


def parse_stream(stdout):
    texts, reads, final, session = [], [], None, None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get('session_id'):
            session = event['session_id']
        if event.get('type') == 'assistant':
            for part in event.get('message', {}).get('content', []):
                if part.get('type') == 'text':
                    texts.append(part.get('text', ''))
                if part.get('type') == 'tool_use' and part.get('name') in {'Read','Glob','Grep'}:
                    reads.append({'tool': part['name'], 'input': part.get('input', {})})
        if event.get('type') == 'result':
            final = event
    answer = (final or {}).get('structured_output')
    raw = (final or {}).get('result') or '\n'.join(texts)
    if answer is None and raw:
        match = re.fullmatch(r'\s*```(?:json)?\s*([\s\S]*?)\s*```\s*', raw)
        try:
            answer = json.loads(match[1] if match else raw)
        except ValueError:
            pass
    return answer, {'final_answer': raw, 'structured_output': answer, 'reads': reads,
                    'session_id': session, 'result_type': (final or {}).get('subtype'),
                    'is_error': (final or {}).get('is_error')}


def invoke(prompt, run, attempt, state, persist, material_dirs):
    """One counted writer call; never gives the model a file-writing tool."""
    import jsonschema
    state['calls_used'] += 1
    state['action'] = {'kind':'coder', 'phase':'starting', 'attempt':str(attempt.relative_to(run)),
                       'session_id':str(uuid.uuid4())}
    persist()
    try:
        env = cli_environment(state['config']['effort'])
    except (OSError, ValueError) as e:
        state['action'] = None
        persist()
        raise Problem('write_failed', 'Cannot read existing CLI connection settings: ' + type(e).__name__) from e
    argv = ['claude', '-p', '--model', state['config']['model'], '--effort', state['config']['effort'],
            '--output-format','stream-json','--verbose','--no-session-persistence','--json-schema',json.dumps(REPLY_SCHEMA),
            '--safe-mode','--restricted','--disable-slash-commands','--no-chrome',
            '--setting-sources','','--strict-mcp-config','--mcp-config','{"mcpServers":{}}',
            '--settings','{"disableAllHooks":true}', '--permission-mode','dontAsk',
            '--permission-prompts','none', '--tools','Read,Glob,Grep','--allowedTools','Read,Glob,Grep',
            '--disallowedTools','Bash,Edit,Write,Agent,Task,Skill,WebFetch,WebSearch',
            '--session-id',state['action']['session_id']]
    for directory in sorted(set(map(str, material_dirs))):
        argv += ['--add-dir', directory]
    try:
        process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, text=True, cwd=run, env=env, shell=False, start_new_session=True)
    except OSError as e:
        state['action'] = None
        persist()
        raise Problem('write_failed', f'Cannot start claude: {type(e).__name__}') from e
    state['action'].update(phase='running', pid=process.pid, process_start=process_identity(process.pid))
    persist()
    interrupted = False
    try:
        out, err = process.communicate(prompt, timeout=state['config']['timeout_s'])
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        interrupted = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            out, err = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            out, err = process.communicate(timeout=5)
    answer, draft = parse_stream(redact(out, env))
    draft.update(exit_code=process.returncode, stderr=redact(err, env), stopped=interrupted)
    save(attempt / 'draft.json', draft)
    state['action'] = None
    persist()
    if interrupted:
        raise Problem('interrupted', 'Coder timed out or was interrupted; process group reaped')
    if process.returncode or draft['is_error']:
        raise Problem('write_failed', f'Coder failed; see {attempt / "draft.json"}')
    try:
        jsonschema.validate(answer, REPLY_SCHEMA)
    except jsonschema.ValidationError as e:
        raise Problem('write_failed', 'Malformed coder response: ' + e.message) from e
    return answer
