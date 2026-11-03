from pathlib import Path
import json
import shutil
import tempfile

from schema_extract.runner import create_run, run


REPO = Path(__file__).resolve().parents[2]


class Fixture:
    def __init__(self, scenario=None):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.source = self.base / "source"
        self.source.mkdir()
        contents = {"kernel.py": "def kernel(x):\n    return x + 1\n",
                    "reference.py": "def reference(x):\n    return x + 1\n",
                    "input.py": "def make_inputs():\n    return [0.0, 1.0]\n",
                    "io.md": "Input X: N float32 values; output Y: N float32 values.\n",
                    "acceptance.md": "Compare every Y to reference with absolute error <= 0.001.\n",
                    "scope.md": "N is any positive integer.\n"}
        for name, content in contents.items():
            (self.source / name).write_text(content)
        spec = {"format": "schema_extract.source.v2", "id": "increment", "repository": ".",
                "entrypoint": "kernel.py::kernel", "host": "kernel.py::kernel", "io": ["io.md"],
                "input": "input.py::make_inputs", "reference": "reference.py::reference",
                "acceptance": ["acceptance.md"], "scope": ["scope.md"]}
        self.descriptor = self.source / "schema-inputs.json"
        self.descriptor.write_text(json.dumps(spec))
        self.fake = self.base / "fake_codex"
        shutil.copyfile(Path(__file__).with_name("fake_codex.py"), self.fake)
        self.fake.chmod(0o755)
        (self.base / "scenario.json").write_text(json.dumps(scenario or {}))
        self.store = self.state = self.lock = None

    def create(self, **config):
        self.store, self.state, self.lock = create_run(
            REPO, self.base / "v2", self.base / "output", source=str(self.source),
            config={"codex_binary": str(self.fake), "call_seconds": 10, **config})
        return self.store, self.state, self.lock

    def run(self, **config):
        self.create(**config)
        self.state = run(self.store, self.state["run_id"])
        return self.state

    def calls(self):
        return [json.loads(line) for line in (self.base / "calls.jsonl").read_text().splitlines()]

    def cleanup(self):
        for path in self.base.rglob("*"):
            if path.is_dir():
                path.chmod(0o755)
        self.temporary.cleanup()
