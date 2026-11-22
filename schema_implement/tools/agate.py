"""Direct official Agate SDK adapter; imports/configuration are deferred."""
from __future__ import annotations

import ast
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import time
import zipfile
from urllib.parse import urlsplit

from .coder import Problem, digest, inventory, read_json, regular, save, sha


def load_eval(directory):
    directory = Path(directory)
    for name, symbol in [('reference.py','Model'), ('input.py','_make_inputs')]:
        try:
            tree = ast.parse(regular(directory / name).decode('utf-8'))
        except (SyntaxError, UnicodeError) as e:
            raise Problem('needs_input', f'Invalid fixed eval source {name}') from e
        if not any(getattr(n, 'name', None) == symbol for n in tree.body):
            raise Problem('needs_input', f'{name} must export {symbol}')
    for name in ('shapes.json','eval.json'):
        regular(directory / name)
    shapes = read_json(directory / 'shapes.json')
    if not isinstance(shapes, dict) or not shapes:
        raise Problem('needs_input', 'shapes.json must contain nonempty fixed cases')
    for key, case in shapes.items():
        if not isinstance(case, dict) or not all(isinstance(case.get(k), dict) for k in ('init_kwargs','input_kwargs')):
            raise Problem('needs_input', f'Case {key} needs init_kwargs/input_kwargs objects')
    config = read_json(directory / 'eval.json')
    supported = {'operator','languages','requirements','harness','atrex_bench_version','mode',
                 'options','runner_overrides','deps_mode','lock_clocks'}
    if not isinstance(config, dict) or set(config) - supported:
        raise Problem('needs_input', 'eval.json has unsupported fields/comparator/build requirements')
    if config.get('harness') != 'atrex' or config.get('mode', 'full') != 'full':
        raise Problem('needs_input', 'This entry requires Atrex full correctness and timing')
    languages = config.get('languages')
    if not isinstance(languages, list) or not languages or set(languages) - {'triton','python','pytorch'}:
        raise Problem('needs_input', 'Supported candidate path is Python Model/Triton')
    options = config.get('options', {})
    for field in ('atol','rtol'):
        val = options.get(field)
        if isinstance(val, bool) or not isinstance(val, (float,int)) or not math.isfinite(val) or val < 0:
            raise Problem('needs_input', f'eval.json options.{field} must fix the actual tolerance')
    for field in ('num_correctness_cases', 'bench_iters', 'timeout_s'):
        if type(options.get(field)) is not int or options[field] <= 0:
            raise Problem('needs_input', f'eval.json must fix positive options.{field}')
    if options.get('shape_ids') and set(map(str, options['shape_ids'])) != set(shapes):
        raise Problem('needs_input', 'Shape selection must include every fixed case')
    overrides = config.get('runner_overrides', {})
    if any(k in overrides for k in ('input','output','reference_dir','checkpoint_dir','baseline_input')):
        raise Problem('needs_input', 'Runner paths cannot replace the fixed eval bundle')
    if overrides.get('validation_mode', 'full') != 'full':
        raise Problem('needs_input', 'Runner cannot disable full correctness or timing')
    for key in ('atol','rtol','num_correctness_cases','bench_iters'):
        if key in overrides and overrides[key] != options[key]:
            raise Problem('needs_input', f'Conflicting options and runner_overrides.{key}')
    known = {'reference.py','input.py','shapes.json','eval.json','metadata.json','roofline.json','README.md'}
    extra = set(inventory(directory)) - known
    if extra:
        raise Problem('needs_input', 'SDK reference bundle cannot express additional comparator/build files: ' + ', '.join(sorted(extra)))
    reference = {'operator': config.get('operator', directory.name),
                 'reference_py': regular(directory / 'reference.py').decode(),
                 'input_py': regular(directory / 'input.py').decode(), 'shapes': shapes}
    for name in ('metadata','roofline'):
        if (directory / (name + '.json')).exists():
            reference[name] = read_json(directory / (name + '.json'))
    return config, reference


def connect(timeout):
    try:
        from atrex_gateway_client import Client
        from atrex_gateway_client.auth import AkSkAuth
        from atrex_gateway_client.profiles import CONFIG_PATH, load_config, resolve_connection
    except ImportError as e:
        raise Problem('needs_input', 'Install the official atrex-gateway-client implement dependency') from e
    path = Path(os.environ.get('KS_AGATE_CONFIG', str(CONFIG_PATH)))
    config = load_config(path)
    if 'AGATE_AK' in os.environ or 'AGATE_SK' in os.environ:
        ak, sk = os.environ.get('AGATE_AK'), os.environ.get('AGATE_SK')
    else:
        ak, sk = config.get('ak'), config.get('sk')
    if not ak or not sk:
        raise Problem('needs_input', 'Agate needs an existing complete AGATE_AK/AGATE_SK or config ak/sk pair')
    connection = resolve_connection(config, profile=os.environ.get('AGATE_PROFILE'),
                                    environment_url=os.environ.get('AGATE_URL'))
    parsed = urlsplit(connection.url)
    if parsed.scheme not in {'http','https'} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise Problem('needs_input','Agate connection must be an HTTP(S) endpoint without embedded credentials or query')
    client = Client(connection.url, auth=AkSkAuth(ak, sk), timeout=min(30, timeout),
                    environment_id=connection.environment_id, route_label=connection.route_label)
    client.connection_identity = digest({'url':connection.url, 'environment_id':connection.environment_id,
                                         'route_label':connection.route_label})
    return client, version('atrex-gateway-client')


def build_request(kernel, entry, config, reference, gpu, key, name):
    from atrex_gateway_client import build_eval_request_from_content
    files = inventory(kernel)
    candidate = regular(kernel / entry).decode() if len(files) == 1 else {'archive':'candidate.zip','entry_point':entry}
    return build_eval_request_from_content(candidate, reference, gpu, name=name,
        spec_fields={'languages':config['languages']}, options=config['options'],
        requirements=config.get('requirements', []), harness=config['harness'],
        atrex_bench_version=config.get('atrex_bench_version'), mode='full',
        runner_overrides=config.get('runner_overrides'), deps_mode=config.get('deps_mode'),
        lock_clocks=config.get('lock_clocks'), idempotency_key=key)


def safe_response(value):
    """Preserve service observations, omit temporary credentials and signed URLs."""
    if isinstance(value, dict):
        return {k:safe_response(v) for k,v in value.items()
                if k.lower() not in {'put_url','download_url','signed_url','authorization','access-key','token'}}
    if isinstance(value, list):
        return [safe_response(v) for v in value]
    if isinstance(value, str) and value.startswith(('https://','http://')) and '?' in value:
        return value.split('?',1)[0] + '?[redacted]'
    return value


def service_error(error):
    # SDK messages can contain endpoints; only safe structured attribution is durable.
    return {'type':type(error).__name__, 'status':getattr(error,'status',None),
            'reason':getattr(error,'reason',None), 'error_class':getattr(error,'error_class',None)}


def evaluate(run, attempt, state, persist, client):
    """Advance one fixed request. Known jobs are only polled, never resubmitted."""
    action = state['action']
    request_path = attempt / 'request.json'
    request = read_json(request_path)
    if digest(request) != action['request_sha256']:
        raise Problem('input_changed', 'Fixed Agate request was modified')
    if isinstance(request['candidate'], dict) and action['phase'] == 'prepared':
        archive = attempt / 'candidate.zip'
        if not archive.exists():
            with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
                for name in inventory(attempt / 'kernel'):
                    z.writestr(zipfile.ZipInfo(name, (1980,1,1,0,0,0)), regular(attempt / 'kernel' / name))
        action['phase'] = 'reserving_upload'
        persist()
        try:
            batch = client.prepare_uploads(state['config']['gpu'],
                    [{'path':'candidate.zip','bytes':archive.stat().st_size,'sha256':sha(regular(archive))}], kind='eval')
            uploads = batch.get('uploads', [])
            if not batch.get('job_id') or len(uploads) != 1 or uploads[0].get('path') != 'candidate.zip':
                raise Problem('interrupted', 'Invalid upload reservation; original outcome needs locating')
            action.update(phase='uploading', reserved_job_id=batch['job_id'],
                          uploads=[{'path':u['path'],'upload_ref':u['upload_ref']} for u in uploads])
            persist()
            for retry in range(2):
                try:
                    client.upload_file(uploads[0]['put_url'], str(archive))
                    break
                except Exception as upload_error:
                    if retry or getattr(upload_error, 'status', None) in {400,401,403,404,422}:
                        raise
                    # Same reserved object and bytes; never reserve another upload.
            request['oss_files'] = action['uploads']
            save(request_path, request)
            action.update(phase='uploaded', request_sha256=digest(request))
            persist()
        except Problem:
            raise
        except Exception as e:
            action['diagnostic'] = service_error(e)
            persist()
            raise Problem('interrupted', 'Original source upload is unresolved; reservation retained') from e
    if action['phase'] in {'reserving_upload','uploading'}:
        raise Problem('interrupted', 'Original upload was interrupted; SDK has no safe signed-URL refresh operation. Reservation retained; no new reservation or job created')
    if action['phase'] in {'prepared','uploaded','submitting'}:
        # Official unified eval API documents deduplication for unchanged request+key.
        # A lost acknowledgement reuses exactly those bytes; no fresh key is allocated.
        recovering_submission = action['phase'] == 'submitting'
        action['phase'] = 'submitting'
        persist()
        try:
            accepted = client.submit_job('eval', request)
        except Exception as e:
            action['diagnostic'] = service_error(e)
            if not recovering_submission and getattr(e, 'status', None) in {400,401,403,422}:
                response = {'status':'rejected','error':action['diagnostic']}
                save(attempt / 'response.json',response)
                action['phase'] = 'terminal'
                persist()
                return response
            persist()
            raise Problem('interrupted', 'Agate submission acknowledgement missing; resume the unchanged request/key') from e
        job_id = accepted.get('job_id')
        if not isinstance(job_id, str) or not job_id:
            raise Problem('interrupted', 'Agate acceptance omitted job_id; retain original key')
        if action.get('reserved_job_id') and job_id != action['reserved_job_id']:
            action['received_job_id'] = job_id
            persist()
            raise Problem('interrupted', 'Accepted job differs from upload reservation; identities retained')
        action.update(job_id=job_id, phase='waiting', acceptance=safe_response(accepted))
        persist()
    deadline = time.monotonic() + state['config']['timeout_s']
    while time.monotonic() < deadline:
        try:
            response = client.get_job(action['job_id'], wait=True,
                        timeout=min(30, max(0.001, deadline-time.monotonic())), include_spec=True)
        except Exception as e:
            action['diagnostic'] = service_error(e)
            persist()
            raise Problem('interrupted', 'Agate query failed; retain and query the original job') from e
        if response.get('job_id') != action['job_id']:
            raise Problem('interrupted', 'Query returned a different job identity')
        save(attempt / 'response.json', safe_response(response))
        if response.get('execution_quarantined') or response.get('cleanup_pending'):
            raise Problem('interrupted', 'Original execution or cleanup is unresolved')
        if response.get('status') in {'succeeded','failed','cancelled'}:
            action['phase'] = 'terminal'
            persist()
            return response
    raise Problem('interrupted', 'Agate wait timed out; original job remains queryable')


def interpret(response, reference, config):
    """Read the Bench verdict structure. A succeeded job alone cannot pass."""
    result = response.get('result') or {}
    # Multi-device SDK wrappers remain explicit; do not silently choose one result.
    if isinstance(result.get('sdk_results'), list):
        if len(result['sdk_results']) != 1:
            return [], {}, 'evaluation_failed', 'Expected one complete Atrex result'
        result = result['sdk_results'][0]
    passed = result.get('passed') or {}
    correct = (result.get('correctness') or {}).get('shapes') or {}
    performance = (result.get('performance') or {}).get('shapes') or {}
    rows, missing, code_failure = [], [], False
    for case_id in reference['shapes']:
        verdicts = {stage:(passed.get(stage) or {}).get(case_id) for stage in ('compile','correctness','performance')}
        cases = (correct.get(case_id) or {}).get('cases') or []
        observed = []
        comparator_ran = len(cases) >= config['options']['num_correctness_cases']
        for case in cases:
            outputs = case.get('outputs') or []
            comparator_ran = comparator_ran and bool(outputs)
            observed += outputs + (case.get('mutated_inputs') or []) + (case.get('unexpected_mutations') or [])
            if case.get('error'):
                code_failure = True
        output_passed = bool(observed) and all(o.get('passed') is True and not o.get('error') for o in observed)
        stage_passed = all(isinstance(v, dict) and v.get('status') == 'passed' for v in verdicts.values())
        for stage in ('compile','correctness'):
            if isinstance(verdicts[stage],dict) and verdicts[stage].get('status') == 'failed':
                code_failure = True
        if comparator_ran and not output_passed:
            code_failure = True
        perf = performance.get(case_id) or {}
        stats = perf.get('timing_stats') or {}
        latency = stats.get('median_ms')
        timing_ok = type(latency) in (int,float) and math.isfinite(latency) and latency > 0 and stats.get('runs',0) > 0
        okay = stage_passed and comparator_ran and output_passed and timing_ok and not perf.get('error')
        if not okay:
            missing.append(case_id)
        rows.append({'case_id':case_id, 'passed':okay, 'latency':latency if timing_ok else None,
                     'unit':'ms' if timing_ok else None, 'statistic':'median' if timing_ok else None,
                     'timing_stats':stats, 'benchmark_mode':perf.get('benchmark_mode'),
                     'comparison':{'executed':comparator_ran, 'verdicts':verdicts, 'observations':cases}})
    environment = result.get('environment') or {}
    error = response.get('error') or result.get('error') or {}
    actual_config = result.get('runner_config') or {}
    for key in ('atol','rtol','num_correctness_cases'):
        if key in actual_config and actual_config[key] != config['options'][key]:
            return rows, environment, 'evaluation_failed', 'Observed runner setting differs from fixed eval options: ' + key
    if isinstance(error, dict) and error.get('error_class') == 'code':
        code_failure = True
    if isinstance(error, dict) and error.get('error_class') in {'infra','timeout','validation','auth'}:
        return rows, environment, 'evaluation_failed', 'Agate service failure: ' + json.dumps(safe_response(error), ensure_ascii=False)
    if response.get('status') != 'succeeded':
        return rows, environment, 'implementation_failed' if code_failure else 'evaluation_failed', 'Agate terminal failure: ' + json.dumps(safe_response(error), ensure_ascii=False)
    if missing:
        return rows, environment, 'implementation_failed' if code_failure else 'evaluation_failed', 'Fixed cases failed or lack comparison/timing evidence: ' + ', '.join(missing)
    if error:
        return rows, environment, 'evaluation_failed', 'Service reported an error alongside results'
    return rows, environment, 'done', 'All fixed cases passed with measured timing'
