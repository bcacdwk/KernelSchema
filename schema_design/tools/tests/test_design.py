"""CPU boundaries with self-contained task material and no real model calls."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from schema_design.tools import runner as r


def schema_files(chain=False):
    import yaml
    root = {
        "META_DATA": {"Name": "pair_add", "Describe": "Add two scalar inputs in a CPU protocol fixture", "Version": 1},
        "CONTRACT": {
            "Algorithm_requirement": {
                "Input": {name: {"meaning": name + " scalar", "shape": [], "dtype": "fp32"} for name in ("x", "y")},
                "Output": {"z": {"meaning": "Sum scalar", "shape": [], "dtype": "fp32"}},
                "Compute": [{"Name": "Add", "Math": "z = x + y"}],
                "Golden_reference": {"Source": "testing.invalid/pair@v1:reference.py::add", "Tolerance": {"z": "exact"}},
            },
            "Hardware_specification": {
                "Device": "Protocol fixture device", "Architecture": "fixture",
                "Execution": {"cta": "A work group", "warp": "One group within a CTA"},
                "Feature": {"register": "Private scalar storage"},
                "Stack": {"Language": "Fixture language 1", "Toolchain": "Fixture compiler 1"},
                "Wiki": {"Hardware": "<testing.invalid/hardware@pending:device.md>",
                         "Programming": {"Language": "<testing.invalid/language@pending:guide/>"},
                         "Microbench": "<testing.invalid/microbench@pending:bench/>",
                         "Sota": {"Kernel": "<testing.invalid/kernel@pending:code/>",
                                  "Schema": "<testing.invalid/schema@pending:plans/>"},
                         "Profile": "<testing.invalid/profile@pending:results/>"},
            },
        },
        "DESIGN": {
            "Math_reshape": {"Thesis": "Add the two scalars directly", "Rewrite": [{"Name": "Add", "Math": "z = x + y"}]},
            "Work_schedule": {"Thesis": "Assign the sum to one worker", "Step": {"MAIN": {
                "Describe": "Produce z", "Decomposition": {"Partition": {},
                    "Worktile": {"Invariant": {name: {"meaning": name + " scalar", "shape": [], "dtype": "fp32"} for name in ("x", "y")}},
                    "Target": {"Single_tile": {"z": {"meaning": "Sum scalar", "shape": [], "dtype": "fp32"}}}},
                "Mapping": {"Worker": {"unit": "cta", "unit_count": 1}, "Worker_count": 1},
                "Schedule": {"Worker_assignment": {"Preassigned": {"start": 0, "count": 1}, "Refill": "none", "Exit": "assigned_done"}},
            }}, "Flow": {"Launch": [{"step": ["MAIN"]}]}},
            "Tile_pipeline": {"Thesis": "One serial role adds the scalars", "Step": {"MAIN": {"Role": {"MAIN": {
                "Info": {"Describe": "Add x and y", "Member": "warp[0:1]", "Using_verb": {
                    "compute": {"z": {"execution_mode": "serial", "describe": "Read x and y; done makes z visible"}}}},
                "Pipe": ["compute:z"],
            }}}}},
        },
    }
    nodes = {"root": root}
    if chain:
        root["META_DATA"]["Dispatch"] = {"when": "All legal scalar pairs"}
        for name, provider in (("middle", "root"), ("leaf", "middle")):
            node = deepcopy(root)
            node.pop("CONTRACT")
            node["META_DATA"]["Dispatch"] = {"id": name, "when": "All legal scalar pairs"}
            node["DESIGN"] = {region: {"from": provider} for region in root["DESIGN"]}
            nodes[name] = node
    files = [{"path": name + ".yaml", "content": yaml.safe_dump(node, sort_keys=False)} for name, node in nodes.items()]
    dispatch = "# Pair addition\n\n## 范围\n\n合成协议任务，所有标量对；重叠时选择 root\n\n## 完整设计\n\n"
    dispatch += "| 设计 | 具体机制与保留理由 | 关系 |\n| --- | --- | --- |\n"
    dispatch += "".join(f"| [{name}]({name}.yaml) | 标量加法 | 完整共享设计 |\n" for name in nodes)
    return files + [{"path": "dispatch.md", "content": dispatch}]


class DesignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="design-cpu-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.task = self.base / "task.md"
        self.task.write_text("Add two FP32 scalars x and y, producing z on the fixture device.")
        self.prompts = []

    def fake(self, answers, reads=True):
        iterator = iter(answers)
        def call(args, prompt, schema, out, path, seed):
            self.prompts.append(prompt)
            answer = deepcopy(next(iterator))
            if "files" in answer:
                files = answer["files"]
                candidate = out / seed["candidate_directory"]
                for item in files:
                    if r.candidate_name(item["path"]):
                        (candidate / item["path"]).write_text(item["content"])
                answer = {**answer, "files": [item["path"] for item in files]}
            value = {**seed, "state": "finished", "final_answer": json.dumps(answer), "structured_output": answer,
                     "reads": [{"tool": "Read", "input": {"file_path": str(out / "input/schema_docs/3.design.math_reshape.md")},
                                "status": "completed"}] if reads else [], "diagnostics": []}
            if seed["phase"] == "select" and hasattr(self, "selection_description"):
                value["reads"].append({"tool": "Read", "input": {"file_path": self.selection_description}, "status": "completed"})
            r.atomic_json(path, value)
            return value
        return call

    def run_design(self, answers, count=1, reads=True):
        out = self.base / f"run{len(self.prompts)}"
        stdout, stderr = StringIO(), StringIO()
        with patch.object(r, "native_call", self.fake(answers, reads)), patch.object(r, "discover", return_value=[]), redirect_stdout(stdout), redirect_stderr(stderr):
            code = r.main(["--task", str(self.task), "--out", str(out), "--max-calls", str(count)])
        self.assertEqual(stdout.getvalue(), str(out / "result.json") + "\n")
        return code, out, json.loads((out / "result.json").read_text())

    def test_text_task_without_neighbors_writes_valid_schema(self):
        code, out, result = self.run_design([{"files": schema_files(), "notes": "Direct addition; no measured claim", "missing": []}])
        self.assertEqual(code, 0, (out / "notes.md").read_text())
        self.assertEqual(result["calls_used"], 1)
        from schema_template.tools.schema import validate_directory
        validate_directory(out / result["schema"])
        draft = json.loads((out / "drafts/01.json").read_text())
        self.assertTrue(draft["reads"])
        self.assertIn("0.common.md", " ".join(draft["inline_reads"]))
        self.assertIn("Add two FP32 scalars", self.prompts[0])

    def test_bad_draft_preserved_then_specific_repair_with_previous_answer(self):
        invalid = {"files": [{"path": "../escape.yaml", "content": "bad"}], "notes": "draft", "missing": []}
        valid = {"files": schema_files(), "notes": "corrected", "missing": []}
        code, out, result = self.run_design([invalid, valid], count=2)
        self.assertEqual(code, 0, (out / "notes.md").read_text())
        self.assertEqual(result["calls_used"], 2)
        self.assertFalse((self.base / "escape.yaml").exists())
        self.assertEqual(json.loads((out / "drafts/01.json").read_text())["structured_output"]["files"], ["../escape.yaml"])
        self.assertIn("../escape.yaml", self.prompts[1])
        self.assertIn("HOST DIAGNOSTIC", self.prompts[1])

    def test_no_actual_doc_read_cannot_be_accepted(self):
        code, out, result = self.run_design([{"files": schema_files(), "notes": "unread", "missing": []}], reads=False)
        self.assertEqual((code, result["schema"], result["stop_reason"]), (1, None, "write_failed"))
        self.assertIn("Read", (out / "notes.md").read_text())

    def test_missing_material_preserves_response_without_inventing_schema(self):
        code, out, result = self.run_design([{"files": [], "notes": "Need dtype", "missing": ["Task input dtype"]}])
        self.assertEqual((code, result["stop_reason"], result["schema"]), (1, "needs_input", None))
        self.assertTrue((out / "drafts/01.json").exists())

    def test_complete_intermediate_from_and_all_users_remain_raw(self):
        source = self.base / "schema"
        source.mkdir()
        for item in schema_files(chain=True):
            (source / item["path"]).write_text(item["content"])
        out = self.base / "snapshot"
        out.mkdir()
        inputs = r.Inputs(out)
        fixed, resolved = inputs.schema(source / "leaf.yaml", "schema")
        self.assertEqual(set(resolved.source_files), {"root", "middle", "leaf"})
        self.assertEqual(resolved.origins_dict("leaf")["Math_reshape"]["chain"], ["leaf", "middle", "root"])
        self.assertIn("from: middle", (fixed / "leaf.yaml").read_text())
        self.assertIn("CONTRACT", __import__("yaml").safe_load(inputs.bindings["schema"]["nodes"]["leaf"]["effective_yaml"]))
        self.assertEqual(set(resolved.source_manifest("leaf")), {"root.yaml", "middle.yaml", "leaf.yaml"})

    def test_discovery_finds_schema_outside_catalog_ignores_placeholders(self):
        checkout = self.base / "checkout"
        (checkout / "schema_wiki/placeholder").mkdir(parents=True)
        source = checkout / "schema_wiki/extra"
        source.mkdir()
        for item in schema_files():
            (source / item["path"]).write_text(item["content"])
        out = self.base / "discovery"
        out.mkdir()
        run = SimpleNamespace(inputs=r.Inputs(out), args=SimpleNamespace(reference=[]), notes=[])
        with patch.object(r, "CHECKOUT", checkout):
            candidates = r.discover(run)
        self.assertEqual([item["id"] for item in candidates], [str(source)])

    def test_snapshot_change_and_symlink_rejected(self):
        out = self.base / "snapshot"
        out.mkdir()
        inputs = r.Inputs(out)
        inputs.file(self.task)
        self.task.write_text("changed")
        with self.assertRaises(r.Failure) as error:
            inputs.file(self.task)
        self.assertEqual(error.exception.reason, "input_changed")
        link = self.base / "link.md"
        link.symlink_to(self.task)
        with self.assertRaises(r.Failure):
            inputs.file(link)

    def test_selection_call_count_and_full_reference_pass_to_writer(self):
        source = self.base / "reference"
        source.mkdir()
        for item in schema_files(chain=True):
            (source / item["path"]).write_text(item["content"])
        selected = {"references": [{"id": str(source), "purpose": "Compare scalar scheduling"}], "notes": "Compatible arithmetic"}
        written = {"files": schema_files(), "notes": "Borrowed scheduling with local math", "missing": []}
        out = self.base / "selected"
        def discovered(run):
            self.selection_description = str(run.inputs.file(source / "dispatch.md"))
            return [{"id": str(source), "description": self.selection_description, "dispatch": self.selection_description}]
        with patch.object(r, "native_call", self.fake([selected, written])), patch.object(r, "discover", side_effect=discovered), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            code = r.main(["--task", str(self.task), "--out", str(out), "--max-calls", "3"])
        self.assertEqual(code, 0, (out / "notes.md").read_text())
        self.assertEqual(json.loads((out / "result.json").read_text())["calls_used"], 2)
        self.assertIn("from: middle", self.prompts[1])
        self.assertIn('"chain": ["leaf", "middle", "root"]', self.prompts[1])

    def test_native_child_receives_connection_but_artifacts_do_not(self):
        config = self.base / "auth"
        config.mkdir()
        (config / "settings.json").write_text(json.dumps({"env": {"ANTHROPIC_AUTH_TOKEN": "test-connection-secret",
                                                                 "ANTHROPIC_BASE_URL": "https://private-fixture.invalid", "NO_PROXY": "*"}}))
        binaries = self.base / "bin"
        binaries.mkdir()
        script = binaries / "claude"
        answer = {"files": schema_files(), "notes": "Native fixture answer preserves the * operator", "missing": []}
        script.write_text(f'''#!{sys.executable}
import json, os, sys
from pathlib import Path
prompt = sys.stdin.read()
assert os.environ["ANTHROPIC_AUTH_TOKEN"] == "test-connection-secret"
assert os.environ["ANTHROPIC_BASE_URL"] == "https://private-fixture.invalid"
assert json.loads(Path("result.json").read_text())["calls_used"] == 1
assert "--safe-mode" in sys.argv and "--no-session-persistence" in sys.argv
doc = Path.cwd() / "input/schema_docs/3.design.math_reshape.md"
doc.read_text()
print(json.dumps({{"type":"assistant","message":{{"content":[{{"type":"tool_use","id":"doc","name":"Read","input":{{"file_path":str(doc)}}}}]}}}}))
print(json.dumps({{"type":"user","message":{{"content":[{{"type":"tool_result","tool_use_id":"doc","content":"read"}}]}}}}))
answer = json.loads({json.dumps(answer)!r})
candidate = Path.cwd() / "drafts/01"
settings = json.loads(sys.argv[sys.argv.index("--settings") + 1])
assert "Write(/" + str(candidate) + "/*.yaml)" in settings["permissions"]["allow"]
for item in answer["files"]:
    (candidate / item["path"]).write_text(item["content"])
answer["files"] = [item["path"] for item in answer["files"]]
print(json.dumps({{"type":"result","subtype":"success","is_error":False,"result":"genuine fixture final","structured_output":answer}}))
''')
        script.chmod(0o755)
        out = self.base / "native"
        env = {"HOME": str(self.base), "CLAUDE_CONFIG_DIR": str(config), "PATH": str(binaries) + os.pathsep + os.environ["PATH"]}
        process = subprocess.run([sys.executable, str(r.CHECKOUT / "schema_design/design_schema.py"), "--task", str(self.task),
                                  "--out", str(out), "--max-calls", "1"], cwd=self.base, env=env, text=True, capture_output=True)
        self.assertEqual(process.returncode, 0, process.stderr + (out / "notes.md").read_text())
        self.assertEqual(process.stdout, str(out / "result.json") + "\n")
        draft = json.loads((out / "drafts/01.json").read_text())
        self.assertEqual(draft["final_answer"], "genuine fixture final")
        self.assertIn("* operator", draft["structured_output"]["notes"])
        for path in out.rglob("*"):
            if path.is_file():
                self.assertNotIn(b"test-connection-secret", path.read_bytes())
                self.assertNotIn(b"private-fixture.invalid", path.read_bytes())

    def test_timeout_is_counted_and_process_reaped_before_retry(self):
        binaries = self.base / "bin"
        binaries.mkdir()
        script = binaries / "claude"
        script.write_text(f"#!{sys.executable}\nimport sys,time\nsys.stdin.read()\ntime.sleep(10)\n")
        script.chmod(0o755)
        out = self.base / "timeout"
        env = {"HOME": str(self.base), "PATH": str(binaries) + os.pathsep + os.environ["PATH"]}
        with patch.dict(os.environ, env, clear=True), patch.object(r, "discover", return_value=[]), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            code = r.main(["--task", str(self.task), "--out", str(out), "--max-calls", "2", "--timeout-s", "0.1"])
        result = json.loads((out / "result.json").read_text())
        self.assertEqual((code, result["calls_used"], result["stop_reason"]), (1, 1, "write_failed"))
        for path in (out / "drafts").glob("*.json"):
            draft = json.loads(path.read_text())
            self.assertTrue(any("process group reaped" in item for item in draft["diagnostics"]))
            with self.assertRaises(ProcessLookupError):
                os.kill(draft["pid"], 0)

    def test_connection_settings_preserved_only_in_child_environment(self):
        config = self.base / "auth"
        config.mkdir()
        (config / "settings.json").write_text(json.dumps({"env": {"ANTHROPIC_AUTH_TOKEN": "fake-secret-token",
            "ANTHROPIC_BASE_URL": "https://fixture.invalid", "HTTPS_PROXY": "https://proxy.invalid",
            "SSL_CERT_FILE": "/fixture/cert.pem", "UNTRUSTED_HOOK": "do-not-pass"}, "hooks": {"anything": "do-not-run"}}))
        environment = {"HOME": str(self.base), "PATH": os.environ["PATH"], "CLAUDE_CONFIG_DIR": str(config),
                       "ANTHROPIC_BASE_URL": "https://process.invalid"}
        with patch.dict(os.environ, environment, clear=True):
            child = r.connection_environment("max")
        self.assertEqual(child["HOME"], str(self.base))
        self.assertEqual(child["CLAUDE_CONFIG_DIR"], str(config))
        self.assertEqual(child["ANTHROPIC_AUTH_TOKEN"], "fake-secret-token")
        self.assertEqual(child["ANTHROPIC_BASE_URL"], "https://process.invalid")
        self.assertEqual(child["SSL_CERT_FILE"], "/fixture/cert.pem")
        self.assertNotIn("UNTRUSTED_HOOK", child)
        argv = r.command(SimpleNamespace(model=r.DEFAULT_MODEL, effort="max"), r.WRITE_SCHEMA, [self.base / "input"])
        self.assertEqual(argv[argv.index("--setting-sources") + 1], "")
        self.assertTrue(json.loads(argv[argv.index("--settings") + 1])["disableAllHooks"])
        self.assertNotIn("fake-secret-token", json.dumps(argv))

    def test_native_stream_records_read_and_final_but_not_thinking(self):
        events = [
            {"type": "assistant", "session_id": "fixture-session", "message": {"content": [
                {"type": "thinking", "thinking": "private-reasoning"},
                {"type": "tool_use", "id": "r1", "name": "Read", "input": {"file_path": "/input/doc.md"}}]}},
            {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "r1", "content": "doc"}]}},
            {"type": "result", "subtype": "success", "is_error": False, "result": "real final",
             "structured_output": {"files": [], "notes": "true", "missing": []}},
        ]
        draft = r.parse_stream("\n".join(map(json.dumps, events)), "", 0)
        self.assertEqual(draft["final_answer"], "real final")
        self.assertEqual(draft["reads"][0]["status"], "completed")
        self.assertNotIn("private-reasoning", json.dumps(draft))

    def test_help_from_other_cwd_never_reads_invalid_auth(self):
        environment = {**os.environ, "CLAUDE_CONFIG_DIR": "/nonexistent/forbidden-config"}
        process = subprocess.run([sys.executable, str(r.CHECKOUT / "schema_design/design_schema.py"), "--help"],
                                 cwd=self.base, env=environment, text=True, capture_output=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("--reference", process.stdout)
        self.assertFalse((self.base / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
