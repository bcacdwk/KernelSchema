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

from schema_optimize.tools import runner as r


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


class OptimizeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="optimize-cpu-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.task = self.base / "task.md"
        self.task.write_text("Add x and y producing z; all scalar pairs are legal.")
        self.schema = self.base / "original"
        self.schema.mkdir()
        for item in schema_files(chain=True):
            (self.schema / item["path"]).write_text(item["content"])
        # Input bytes are canonical and valid, so a byte-equivalent revise is truly a no-op.
        from schema_template.tools import schema as template
        plan = template.normalize_directory(self.schema)
        for name, content in plan.candidates.items():
            plan.files[name].write_text(content)
        template.validate_directory(self.schema)
        self.original = r.tree_hashes(self.schema)
        self.kernel = self.base / "kernel"
        self.kernel.mkdir()
        (self.kernel / "model.py").write_text("from helper import add\nclass Model:\n    def forward(self, x, y): return add(x,y)\n")
        (self.kernel / "helper.py").write_text("def add(x,y): return x+y\n")
        self.feedback = self.base / "feedback.json"
        self.feedback.write_text(json.dumps({"cases": [{"case_id": "0", "passed": True, "latency": 2.5,
                                                       "unit": "us", "statistic": "median"}],
            "kernel": {"path": "kernel", "files": r.tree_hashes(self.kernel), "sha256": r.files_digest(r.tree_hashes(self.kernel))},
            "schema": {"path": "original", "files": self.original, "sha256": r.files_digest(self.original)},
            "implementation_notes": "The helper implements the scalar sum", "environment": {"device": "observed fixture"}}))
        self.prompts = []
        self.run_count = 0

    def fake(self, answers):
        iterator = iter(answers)
        def call(args, prompt, schema, out, path, seed):
            self.prompts.append(prompt)
            answer = next(iterator)
            value = {**seed, "state": "finished", "final_answer": json.dumps(answer), "structured_output": answer,
                     "reads": [{"tool": "Read", "input": {"file_path": str(out / "input/schema_docs/6.realization.md")},
                                "status": "completed"}], "diagnostics": []}
            r.atomic_json(path, value)
            return value
        return call

    def run_optimize(self, answers, kernel=True, feedback=None, count=1):
        self.run_count += 1
        out = self.base / f"run{self.run_count}"
        argv = ["--task", str(self.task), "--schema", str(self.schema / "leaf.yaml"), "--feedback", str(feedback or self.feedback),
                "--out", str(out), "--max-calls", str(count)]
        if kernel:
            argv += ["--kernel", str(self.kernel if kernel is True else kernel)]
        stdout, stderr = StringIO(), StringIO()
        with patch.object(r, "native_call", self.fake(answers)) as native, redirect_stdout(stdout), redirect_stderr(stderr):
            code = r.main(argv)
        self.assertEqual(stdout.getvalue(), str(out / "result.json") + "\n")
        self.assertEqual(r.tree_hashes(self.schema), self.original)
        return code, out, json.loads((out / "result.json").read_text())

    def answer(self, decision, required=False, files=None):
        return {"decision": decision, "required_fix": required, "files": files or [], "missing": [],
                "notes": "helper.py and feedback.json show the concrete state; no invented speedup"}

    def original_files(self):
        return [{"path": path.name, "content": path.read_text()} for path in sorted(self.schema.iterdir())]

    def revised_files(self):
        files = self.original_files()
        root = next(item for item in files if item["path"] == "root.yaml")
        root["content"] = root["content"].replace("Version: 1", "Version: 2")
        root["content"] = root["content"].replace("Member: warp[0:1]", "Member: cta[0:1]")
        self.assertIn("Member: cta[0:1]", root["content"])
        return files

    def test_keep_reads_complete_kernel_feedback_and_full_from_chain(self):
        code, out, result = self.run_optimize([self.answer("keep")])
        self.assertEqual(code, 0, (out / "notes.md").read_text())
        self.assertEqual((result["decision"], result["required_fix"]), ("keep", False))
        self.assertEqual(r.tree_hashes(out / result["schema"]), self.original)
        self.assertIn("def add(x,y): return x+y", self.prompts[0])
        self.assertIn('"latency": 2.5', self.prompts[0])
        self.assertIn('"chain": ["leaf", "middle", "root"]', self.prompts[0])
        self.assertIn('"source_manifest"', self.prompts[0])

    def test_repair_preserves_schema_and_requires_fix(self):
        code, out, result = self.run_optimize([self.answer("repair", True)])
        self.assertEqual((code, result["decision"], result["required_fix"]), (0, "repair", True))
        self.assertFalse((out / "schema").exists())
        self.assertEqual(r.tree_hashes(out / result["schema"]), self.original)

    def test_contract_drift_rejected_before_normalization_and_draft_retained(self):
        files = self.original_files()
        root = next(item for item in files if item["path"] == "root.yaml")
        root["content"] = root["content"].replace("z: exact", "z: 0.5")
        self.assertIn("z: 0.5", root["content"])
        code, out, result = self.run_optimize([self.answer("revise", True, files)])
        self.assertEqual((code, result["decision"], result["schema"]), (1, None, None))
        self.assertIs(result["required_fix"], True)
        self.assertIn("before normalization", (out / "notes.md").read_text())
        self.assertIn("z: 0.5", (out / "drafts/01.json").read_text())

    def test_contract_drift_after_normalization_is_rejected(self):
        from schema_template.tools import schema as template
        original_normalize = template.normalize_directory
        def change_plan(directory):
            from dataclasses import replace
            plan = original_normalize(directory)
            candidates = dict(plan.candidates)
            candidates["root"] = (directory / "root.yaml").read_text().replace("z: exact", "z: 0.5")
            return replace(plan, candidates=candidates)
        with patch.object(template, "normalize_directory", side_effect=change_plan):
            code, out, result = self.run_optimize([self.answer("revise", False, self.original_files())])
        self.assertEqual((code, result["decision"]), (1, None))
        self.assertIn("after normalization", (out / "notes.md").read_text())

    def test_noop_reorder_or_version_bump_is_not_revision(self):
        files = self.original_files()
        for item in files:
            item["content"] = item["content"].replace("Version: 1", "Version: 2")
        code, out, result = self.run_optimize([self.answer("revise", False, files)])
        self.assertEqual((code, result["decision"]), (1, None))
        self.assertIn("no substantive", (out / "notes.md").read_text())

    def test_revise_accepts_changed_design_without_contract_drift(self):
        files = self.revised_files()
        code, out, result = self.run_optimize([self.answer("revise", False, files)])
        self.assertEqual(code, 0, (out / "notes.md").read_text())
        self.assertEqual((result["decision"], result["required_fix"]), ("revise", False))
        from schema_template.tools.schema import resolve_directory
        updated = resolve_directory(out / result["schema"])
        self.assertEqual(updated.effective_document("leaf")["DESIGN"]["Tile_pipeline"]["Step"]["MAIN"]["Role"]["MAIN"]["Info"]["Member"], "cta[0:1]")

    def test_bad_response_does_not_infer_safe_required_fix(self):
        code, out, result = self.run_optimize([{"unexpected": "response"}])
        self.assertEqual((code, result["decision"], result["required_fix"]), (1, None, None))

    def test_failed_revision_cannot_turn_into_keep(self):
        code, out, result = self.run_optimize([self.answer("revise", False, self.original_files()), self.answer("keep")], count=2)
        self.assertEqual((code, result["decision"], result["calls_used"]), (1, None, 2))
        self.assertIn("cannot be hidden", (out / "notes.md").read_text())

    def test_wrong_feedback_binding_fails_before_model(self):
        value = json.loads(self.feedback.read_text())
        value["kernel"]["files"]["helper.py"] = "wrong"
        self.feedback.write_text(json.dumps(value))
        code, out, result = self.run_optimize([])
        self.assertEqual((code, result["calls_used"], result["stop_reason"]), (1, 0, "needs_input"))

    def test_invalid_existing_schema_cannot_be_reported_as_keep(self):
        path = self.schema / "root.yaml"
        path.write_text(path.read_text().replace("    Flow:", "    Not_a_flow:"))
        self.original = r.tree_hashes(self.schema)
        code, out, result = self.run_optimize([])
        self.assertEqual((code, result["calls_used"], result["decision"], result["stop_reason"]), (1, 0, None, "needs_input"))

    def test_relative_feedback_closure_keeps_original_measurement_bytes(self):
        evidence = self.base / "measurements"
        evidence.mkdir()
        response = evidence / "response.json"
        response.write_text('{"cases":[{"passed":true,"latency":1.25}],"job":"original"}\n')
        value = json.loads(self.feedback.read_text())
        value["response"] = "measurements/response.json"
        self.feedback.write_text(json.dumps(value))
        code, out, result = self.run_optimize([self.answer("keep")])
        self.assertEqual(code, 0, (out / "notes.md").read_text())
        manifest = json.loads((out / "input/manifest.json").read_text())
        record = next(item for item in manifest["files"] if item["source"] == str(response))
        self.assertEqual((out / record["snapshot"]).read_bytes(), response.read_bytes())
        self.assertIn('"latency":1.25', self.prompts[0])

    def test_no_kernel_only_accepts_real_design_gap_result(self):
        self.feedback.write_text(json.dumps({"diagnosis": {"kind": "design_gap", "detail": "Work schedule omits a required branch"}, "kernel": None}))
        result_path = self.base / "implement-result.json"
        result_path.write_text(json.dumps({"status": "incomplete", "stop_reason": "design_gap", "kernel": None,
                                           "schema": "original", "feedback": "feedback.json", "calls_used": 1}))
        code, out, result = self.run_optimize([self.answer("revise", True, self.revised_files())], kernel=False, feedback=result_path)
        self.assertEqual(code, 0, (out / "notes.md").read_text())
        self.assertIn("There is no kernel", self.prompts[0])
        self.assertFalse((out / "input/kernel").exists())
        code, out, result = self.run_optimize([], kernel=False, feedback=self.feedback)
        self.assertEqual((code, result["stop_reason"], result["calls_used"]), (1, "needs_input", 0))

    def test_single_file_with_real_helper_is_incomplete_not_silently_truncated(self):
        code, out, result = self.run_optimize([], kernel=self.kernel / "model.py")
        self.assertEqual((code, result["stop_reason"], result["calls_used"]), (1, "needs_input", 0))
        self.assertIn("complete source directory", (out / "notes.md").read_text())

    def test_help_from_other_cwd(self):
        process = subprocess.run([sys.executable, str(r.CHECKOUT / "schema_optimize/optimize_schema.py"), "--help"],
                                 cwd=self.base, text=True, capture_output=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("--feedback", process.stdout)

    def test_native_cli_from_another_cwd_reads_complete_inputs(self):
        binaries = self.base / "bin"
        binaries.mkdir()
        script = binaries / "claude"
        answer = self.answer("keep")
        script.write_text(f'''#!{sys.executable}
import json,sys
from pathlib import Path
prompt=sys.stdin.read()
assert "def add(x,y): return x+y" in prompt
assert '"latency": 2.5' in prompt
assert "middle" in prompt and "effective_identity" in prompt
doc=Path.cwd()/"input/schema_docs/6.realization.md"
doc.read_text()
print(json.dumps({{"type":"assistant","message":{{"content":[{{"type":"tool_use","id":"read","name":"Read","input":{{"file_path":str(doc)}}}}]}}}}))
print(json.dumps({{"type":"user","message":{{"content":[{{"type":"tool_result","tool_use_id":"read","content":"read"}}]}}}}))
print(json.dumps({{"type":"result","subtype":"success","is_error":False,"result":"fixture final","structured_output":json.loads({json.dumps(answer)!r})}}))
''')
        script.chmod(0o755)
        out = self.base / "native"
        environment = {"HOME": str(self.base), "PATH": str(binaries) + os.pathsep + os.environ["PATH"]}
        process = subprocess.run([sys.executable, str(r.CHECKOUT / "schema_optimize/optimize_schema.py"), "--task", str(self.task),
                                  "--schema", str(self.schema), "--kernel", str(self.kernel), "--feedback", str(self.feedback),
                                  "--out", str(out), "--max-calls", "1"], cwd=self.base, env=environment, text=True, capture_output=True)
        self.assertEqual(process.returncode, 0, process.stderr + (out / "notes.md").read_text())
        self.assertEqual(process.stdout, str(out / "result.json") + "\n")
        self.assertEqual(json.loads((out / "result.json").read_text())["decision"], "keep")
        self.assertEqual(r.tree_hashes(self.schema), self.original)


if __name__ == "__main__":
    unittest.main()
