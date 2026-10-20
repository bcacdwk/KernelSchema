import argparse
import gc
import importlib.util
import json
import statistics
import sys
from pathlib import Path

sys.path[:0] = [str(Path('checker').resolve()), str(Path('candidate').resolve())]
import torch
from atrex_bench.eval._runtime import deterministic_input_seed, seed_all_input_rngs


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save(name, value):
    Path(name).write_text(json.dumps(value, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('eager', 'ncu'), required=True)
    args = parser.parse_args()
    shapes = json.loads(Path('reference/shapes.json').read_text())
    generator = load('task_input', 'reference/input.py')
    models = {'production': load('production', 'production.py').Model().cuda().eval(),
              'aka': load('candidate', 'candidate/solution.py').Model().cuda().eval()}
    timings, launches, abi = {}, [], {}
    with torch.inference_mode():
        for shape_id, shape in shapes.items():
            seed = deterministic_input_seed('performance', shape_id, 0)
            seed_all_input_rngs(seed)
            base = generator._make_inputs(**shape['input_kwargs'])
            inputs = {label: dict(base, out=torch.empty_like(base['out'])) for label in models}
            abi[shape_id] = {'seed': seed, 'tensors': {
                key: {'shape': list(t.shape), 'stride': list(t.stride()),
                      'storage_offset': t.storage_offset(), 'dtype': str(t.dtype)}
                for key, t in base.items() if isinstance(t, torch.Tensor)},
                'kv_shared_storage': base['k_cache'].untyped_storage()._cdata == base['v_cache'].untyped_storage()._cdata}
            assert abi[shape_id]['kv_shared_storage']
            assert list(base['k_cache'].stride()) == shape['input_kwargs']['cache_stride']
            save('input_abi_' + args.phase + '.json', abi)
            for _ in range(5):
                for label, model in models.items():
                    result = model(**inputs[label])
                    assert result.data_ptr() == inputs[label]['out'].data_ptr()
            torch.cuda.synchronize()
            torch.testing.assert_close(inputs['aka']['out'], inputs['production']['out'], atol=3.2e-5, rtol=0.02)
            if args.phase == 'eager':
                samples = {label: [] for label in models}
                for repeat in range(10):
                    for label in ('production', 'aka', 'aka', 'production'):
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        models[label](**inputs[label])
                        end.record()
                        end.synchronize()
                        samples[label].append(start.elapsed_time(end) * 1000)
                timings[shape_id] = {label: {'median_us': statistics.median(values), 'samples_us': values}
                                     for label, values in samples.items()}
                save('timing.json', timings)
                print('EAGER_TIMING', shape_id, {k: v['median_us'] for k, v in timings[shape_id].items()}, flush=True)
            else:
                for repeat in range(2):
                    for position, label in enumerate(('production', 'aka', 'aka', 'production')):
                        record = {'shape_id': shape_id, 'repeat': repeat, 'position': position,
                                  'side': label, 'expected_kernels': 2 if label == 'aka' else 1}
                        print('NCU_LAUNCH', json.dumps(record), flush=True)
                        torch.cuda.synchronize()
                        torch.cuda.cudart().cudaProfilerStart()
                        models[label](**inputs[label])
                        torch.cuda.synchronize()
                        torch.cuda.cudart().cudaProfilerStop()
                        launches.append(record)
                        save('launches.json', launches)
                # This also catches profiler replay interfering with FIT's persistent ticket state.
                torch.testing.assert_close(inputs['aka']['out'], inputs['production']['out'], atol=3.2e-5, rtol=0.02)
                print('NCU_DONE', shape_id, flush=True)
            del base, inputs, result
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
