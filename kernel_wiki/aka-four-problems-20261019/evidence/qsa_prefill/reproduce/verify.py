import dataclasses
import gc
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path('checker').resolve()), str(Path('candidate').resolve())]
import torch
from atrex_bench.eval.correctness import check_correctness


def main():
    versions = {}
    for name in ('torch', 'triton', 'nvidia-cutlass-dsl', 'cuda-python', 'atrex-bench'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    environment = {'gpu': torch.cuda.get_device_name(0),
                   'capability': torch.cuda.get_device_capability(0),
                   'cuda': torch.version.cuda, 'versions': versions}
    Path('environment.json').write_text(json.dumps(environment, indent=2))
    print('ENVIRONMENT', json.dumps(environment), flush=True)
    for item in json.loads(Path('source_manifest.json').read_text())['files']:
        assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
    results = {}
    failed = False
    for shape_id in json.loads(Path('reference/shapes.json').read_text()):
        results[shape_id] = {}
        for label, candidate in (('aka', 'candidate/solution.py'), ('production', 'production.py')):
            print('VERIFY_START', shape_id, label, flush=True)
            result = check_correctness(
                Path('reference/reference.py').resolve(), Path(candidate).resolve(),
                shape_id=shape_id, atol=3.2e-5, rtol=0.02,
                num_correctness_cases=4, device='cuda', candidate_timeout_s=240,
            )
            results[shape_id][label] = dataclasses.asdict(result)
            Path('correctness.json').write_text(json.dumps(results, indent=2))
            print('VERIFY_DONE', shape_id, label, result.status, result.reason, flush=True)
            if result.status != 'passed':
                print(json.dumps(dataclasses.asdict(result)), flush=True)
                failed = True
            gc.collect()
            torch.cuda.empty_cache()
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
