import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'MANIFEST.json').read_text())
expected = {x['path'] for x in manifest['files']}
actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
          and p.name != 'MANIFEST.json' and '__pycache__' not in p.parts}
assert expected == actual, (expected - actual, actual - expected)
for item in manifest['files']:
    data = (root / item['path']).read_bytes()
    assert len(data) == item['bytes'], item['path']
    assert hashlib.sha256(data).hexdigest() == item['sha256'], item['path']
for name, problem in manifest['problems'].items():
    shapes = json.loads((root / name / 'shape_valid.json').read_text())
    assert len(shapes) == problem['scored_shapes'], name
    assert (root / name / 'shapes.json').read_bytes() == (root / name / 'shape_valid.json').read_bytes()
print(f"Verified {len(expected)} files and {len(manifest['problems'])} problems")
