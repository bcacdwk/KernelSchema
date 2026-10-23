#!/bin/bash
set -u
date -u
timeout 60s python3 -m pip install --no-deps --no-index --target runtime_deps flashinfer_python-0.6.18-py3-none-any.whl > setup.log 2>&1
echo "install_status=$?"
date -u
tail -12 setup.log
PYTHONPATH="$PWD/runtime_deps" timeout 90s python3 -B -u - <<'PY' > import.log 2>&1
import time
print('BEGIN',time.time(),flush=True)
import torch
print('TORCH',torch.__version__,time.time(),flush=True)
import flashinfer
print('FLASHINFER',flashinfer.__version__,flashinfer.__file__,time.time(),flush=True)
from flashinfer.prefill import trtllm_batch_context_with_kv_cache
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
print('READY',time.time(),flush=True)
PY
echo "import_status=$?"
date -u
cat import.log
tar -czf results.tar.gz setup.log import.log
