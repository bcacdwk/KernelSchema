"""Independent eager ABI checks and paired NCU sampling of frozen source files."""
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback

import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str))


def environment():
    p = torch.cuda.get_device_properties(0)
    versions = {}
    for package in ('torch', 'triton', 'nvidia-cutlass-dsl', 'cuda-python', 'flashinfer-python'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    smi = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu', '--format=csv'], capture_output=True, text=True)
    return dict(name=p.name, sm_count=p.multi_processor_count, capability=torch.cuda.get_device_capability(), versions=versions, nvidia_smi=smi.stdout, clock_lock_requested=False)


def checks(actual, expected, tol):
    if not isinstance(actual, torch.Tensor) or actual.shape != expected.shape or actual.dtype != expected.dtype:
        return dict(passed=False, error='output type, shape or dtype mismatch')
    a, b = actual.float(), expected.float()
    diff = (a-b).abs()
    limit = tol['atol'] + tol['rtol'] * b.abs()
    ok = torch.isfinite(a) & torch.isfinite(b) & (diff <= limit)
    return dict(passed=bool(ok.all()), max_abs=float(diff.max()), max_tolerance_ratio=float((diff/limit).max()), failing_elements=int((~ok).sum()), elements=a.numel())


def unchanged(inputs, before):
    # Compare logical values byte for byte; snapshots need not preserve backing
    # allocation holes. The actual candidate still receives the original views.
    return {key: bool(torch.equal(value.contiguous().view(torch.uint8), inputs[key].contiguous().view(torch.uint8))) for key, value in before.items()}


def poison(inputs, mutated):
    for key in mutated:
        inputs[key].fill_(float('nan'))


def flatten_outputs(value):
    return list(value) if isinstance(value, (tuple, list)) else [value]


def correctness(models, reference, make_inputs, shape, op, sid, case, contract):
    seed = int.from_bytes(hashlib.sha256(f'wiki-recheck-20261021:{op}:{sid}:{case}'.encode()).digest()[:4], 'little')
    torch.manual_seed(seed)
    random.seed(seed)
    inputs = make_inputs(**shape['input_kwargs'])
    mutated = contract['mutates_inputs']
    scratch = contract.get('scratch_inputs', [])
    before = {k:v.clone() for k,v in inputs.items() if isinstance(v,torch.Tensor) and k not in mutated + scratch}
    poison(inputs, mutated)
    expected = [v.clone() for v in flatten_outputs(reference(**inputs))]
    expected_mutated = {k:inputs[k].clone() for k in mutated}
    results = {}
    for side, model in models.items():
        poison(inputs, mutated)
        result = flatten_outputs(model(**inputs))
        torch.cuda.synchronize()
        outputs = {}
        for i, (actual, ref) in enumerate(zip(result, expected)):
            key = f'output[{i}]' if len(expected) > 1 else 'output'
            outputs[key] = checks(actual, ref, contract['correctness_tolerances'][key])
        for key in mutated:
            outputs[f'mutated_inputs.{key}'] = checks(inputs[key], expected_mutated[key], contract['correctness_tolerances'][f'mutated_inputs.{key}'])
        untouched = unchanged(inputs, before)
        aliases = len(result) == len(mutated) and all(result[i].data_ptr() == inputs[k].data_ptr() for i,k in enumerate(mutated))
        passed = len(result) == len(expected) and all(x['passed'] for x in outputs.values()) and all(untouched.values()) and aliases
        results[side] = dict(passed=passed, outputs=outputs, unchanged_inputs=untouched, returned_supplied_buffers=aliases)
    del inputs, before, expected, expected_mutated
    return dict(seed=seed, case=case, sides=results)


def eager_timing(model, inputs, n=40):
    times = []
    for _ in range(n):
        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        model(**inputs)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end)*1000)
    return dict(samples_us=times, median_us=statistics.median(times), mean_us=statistics.mean(times), cv=statistics.stdev(times)/statistics.mean(times), min_us=min(times), max_us=max(times))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--op', required=True)
    parser.add_argument('--phase', choices=['validate','profile'], required=True)
    parser.add_argument('--shapes', default='')
    parser.add_argument('--samples', type=int, default=4)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    sys.path.insert(0,str(root/'aka_solution'))
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    make_inputs=load('problem_input',root/'input.py')._make_inputs
    reference_module=load('problem_reference',root/'reference.py')
    aka_module=load('frozen_aka',root/'aka_solution/solution.py')
    production_module=load('frozen_production',root/'production.py')
    shapes=json.loads((root/'shapes.json').read_text())
    if args.shapes:
        shapes={k:shapes[k] for k in args.shapes.split(',')}
    contract=json.loads((root/'metadata.json').read_text())['benchmark_contract']
    report=dict(op=args.op,phase=args.phase,environment=environment(),contract=contract,records=[],profile_sequence=[],source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.py')})
    output=f'{args.phase}.json'
    save(output,report)
    for sid,shape in shapes.items():
        start=time.monotonic()
        try:
            models={'production':production_module.Model(**(shape.get('init_kwargs') or {})), 'aka':aka_module.Model(**(shape.get('init_kwargs') or {}))}
            ref=reference_module.Model(**(shape.get('init_kwargs') or {}))
            record=dict(shape_id=sid,cases=[])
            if args.phase=='validate':
                for case in range(contract['correctness_min_cases']):
                    result=correctness(models,ref,make_inputs,shape,args.op,sid,case,contract)
                    record['cases'].append(result)
                    print('CASE',sid,case,{k:v['passed'] for k,v in result['sides'].items()},flush=True)
                record['passed']=all(x['passed'] for c in record['cases'] for x in c['sides'].values())
            torch.manual_seed(20261021+int(sid))
            inputs=make_inputs(**shape['input_kwargs'])
            for model in models.values():
                for _ in range(10):
                    model(**inputs)
            torch.cuda.synchronize()
            if args.phase=='validate':
                # ABBA order; each block has 20 independent eager event samples.
                timing={side:[] for side in models}
                for side in ['production','aka','aka','production']:
                    timing[side].extend(eager_timing(models[side],inputs,20)['samples_us'])
                record['eager']={side:dict(samples_us=ts,median_us=statistics.median(ts),cv=statistics.stdev(ts)/statistics.mean(ts)) for side,ts in timing.items()}
                # Check another poisoned call after repeated timing, so a stateful
                # implementation cannot pass by returning stale timed outputs.
                poison(inputs,contract['mutates_inputs'])
                expected=[x.clone() for x in flatten_outputs(ref(**inputs))]
                record['post_timing']={}
                for side,model in models.items():
                    poison(inputs,contract['mutates_inputs'])
                    actual=flatten_outputs(model(**inputs))
                    cs=[checks(a,b,contract['correctness_tolerances'][f'output[{i}]' if len(expected)>1 else 'output']) for i,(a,b) in enumerate(zip(actual,expected))]
                    record['post_timing'][side]=cs
                    record['passed'] &= all(c['passed'] for c in cs)
                del expected,actual
            else:
                # --profile-from-start off captures only these calls. Four
                # alternating samples per side; all required GPU kernels count.
                for sample in range(args.samples):
                    order=['production','aka'] if sample%2==0 else ['aka','production']
                    for side in order:
                        label=f'shape_{sid}__{side}__sample_{sample}'
                        torch.cuda.nvtx.range_push(label)
                        torch.cuda.cudart().cudaProfilerStart()
                        models[side](**inputs)
                        torch.cuda.synchronize()
                        torch.cuda.cudart().cudaProfilerStop()
                        torch.cuda.nvtx.range_pop()
                        report['profile_sequence'].append(dict(shape_id=sid,side=side,sample=sample,label=label))
                        save(output,report)
            record['seconds']=time.monotonic()-start
            report['records'].append(record)
            report['environment_end']=environment()
            save(output,report)
            print('SHAPE_DONE',sid,'seconds',round(record['seconds'],1),flush=True)
            del inputs,models,ref
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            report['error']=traceback.format_exc()
            report['failed_shape']=sid
            save(output,report)
            raise
    print('COMPLETE',args.op,args.phase,flush=True)


if __name__=='__main__':
    with torch.inference_mode():
        main()
