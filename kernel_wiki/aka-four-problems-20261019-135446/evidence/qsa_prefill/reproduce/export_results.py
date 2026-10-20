"""Return bounded text results via the supported Agate dev stdout channel."""
import base64
import gzip
import hashlib
import json
from pathlib import Path

files = {}
for name in ('environment.json', 'correctness.json', 'timing.json', 'launches.json',
             'input_abi_eager.json', 'input_abi_ncu.json', 'metrics.csv', 'ncu.log',
             'source_manifest.json'):
    data = Path(name).read_bytes()
    files[name] = {'text': data.decode('utf-8'), 'sha256': hashlib.sha256(data).hexdigest()}
payload = gzip.compress(json.dumps(files, separators=(',', ':')).encode(), mtime=0)
print('RESULT_BUNDLE_V1=' + base64.b64encode(payload).decode(), flush=True)
