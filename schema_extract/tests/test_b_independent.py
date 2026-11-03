"""Independent milestone B regressions at source and trust boundaries."""

from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

import yaml

from schema_extract.agent import CodexCLI
from schema_extract.common import ExtractError, manifest, read_json, write_json
from schema_extract.inputs import resolve_request
from schema_extract.receipts import call_identity
from schema_extract.runner import accept_result, create_run, run
from schema_extract.snapshots import verify_snapshot


def make_source(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "operator.py").write_text(
        "from device import inner_kernel\n\n"
        "def kernel(x):\n    return inner_kernel(x)\n\n"
        "def reference(x):\n    return pow(x, 2)\n",
        encoding="utf-8",
    )
    (root / "device.py").write_text(
        "def inner_kernel(x):\n    return x * x\n", encoding="utf-8"
    )
    (root / "input.py").write_text(
        "def make_input():\n    return 3\n", encoding="utf-8"
    )
    for name in ("io.json", "acceptance.json", "scope.json"):
        (root / name).write_text("{}\n", encoding="utf-8")
    descriptor = {
        "format": "schema_extract.source.v2",
        "id": "square",
        "entrypoint": "operator.py::kernel",
        "reference": "operator.py::reference",
        "input": "input.py::make_input",
        "io": "io.json",
        "acceptance": "acceptance.json",
        "scope": "scope.json",
    }
    write_json(root / "schema-inputs.json", descriptor)
    return descriptor


def continue_result(attempt, session_id):
    files = manifest(Path(attempt["workspace"]))
    receipt = {
        "format": "schema_extract.receipt.v2",
        "identity": call_identity(attempt),
        "decision": "continue", "summary": "Saved an unfinished Contract draft.",
        "changed_files": [],
        "artifacts": [{"path": name, "sha256": value} for name, value in files.items()],
        "source_evidence": [], "artifact_evidence": [], "conclusions": [],
        "checks": [], "checks_note": "Design is unfinished; no full YAML check was run.",
        "issues": [], "framework_candidates": [], "handoff": "",
    }
    return {
        "outcome": "completed", "termination": {"confirmed": True},
        "session_id": session_id, "response": receipt,
    }


def complete_p1_result(attempt, lock, *, implementation=None, reference=None):
    source = lock["source"]
    prefix = f"{source['repository_id']}@{source['ref']}:"
    workspace = Path(attempt["workspace"])
    before = manifest(workspace)
    root_path = workspace / "root.yaml"
    root = yaml.safe_load(root_path.read_text(encoding="utf-8"))
    root["meta_data"].update({
        "name": "Square", "describe": "Square a scalar input.",
        "source": implementation or prefix + "operator.py::kernel",
    })
    root["contract"]["algorithm_requirement"] = {
        "input": {"X": {"meaning": "Input scalar", "shape": [], "dtype": "float32"}},
        "output": {"Y": {"meaning": "Squared scalar", "shape": [], "dtype": "float32"}},
        "compute": [{"name": "square", "math": "Y = X * X"}],
        "golden_reference": {
            "source": reference or prefix + "operator.py::reference",
            "tolerance": {"Y": {"atol": 0.001}},
        },
    }
    root_path.write_text(yaml.safe_dump(root, sort_keys=False), encoding="utf-8")
    result = continue_result(attempt, str(uuid4()))
    receipt = result["response"]
    receipt["decision"] = "complete"
    receipt["changed_files"] = [name for name, value in manifest(workspace).items() if before.get(name) != value]
    for category in ("implementation", "io", "input", "reference", "acceptance", "scope"):
        binding = source["bindings"]["entrypoint" if category == "implementation" else category]
        locator = binding[0] if isinstance(binding, list) else binding
        path = locator["path"]
        lines = (Path(lock["snapshot"]) / "source" / path).read_text(encoding="utf-8").splitlines()
        index = next(i for i, line in enumerate(lines) if line.strip())
        receipt["source_evidence"].append({
            "category": category, "source_id": source["id"], "path": path,
            "sha256": source["files"][path], "line_start": index + 1, "line_end": index + 1,
            "quote": lines[index], "note": "Source for this controlled Contract.",
        })
    receipt["conclusions"] = [
        {"subject": "contract", "text": "The reference defines the square output.", "evidence": [3]},
        {"subject": "coverage", "text": "The selected input scope is recorded.", "evidence": [5]},
    ]
    return result


class SourceBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.descriptor = make_source(self.root)

    def resolve(self):
        (self.root / "schema-inputs.json").write_text(
            json.dumps(self.descriptor), encoding="utf-8"
        )
        return resolve_request(self.root, source=str(self.root))

    def test_same_file_with_distinct_reference_entrypoint_is_accepted(self):
        _, bindings, names = self.resolve()
        self.assertEqual(bindings["entrypoint"], {"path": "operator.py", "symbol": "kernel"})
        self.assertEqual(bindings["reference"], {"path": "operator.py", "symbol": "reference"})
        self.assertIn("operator.py", names)

    def test_same_implementation_entrypoint_cannot_masquerade_as_reference(self):
        self.descriptor["reference"] = self.descriptor["entrypoint"]
        with self.assertRaisesRegex(ExtractError, "reference"):
            self.resolve()


class InvocationLifecycleTests(unittest.TestCase):
    def test_post_spawn_record_failure_terminates_child_before_confirming(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            workspace = directory / "workspace"
            workspace.mkdir()
            executable = directory / "sleeping_codex"
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import sys, time\n"
                "if '--version' in sys.argv:\n"
                "    print('controlled CLI')\n"
                "    raise SystemExit(0)\n"
                "time.sleep(30)\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            original_popen, children = subprocess.Popen, []

            def capture_process(*args, **kwargs):
                child = original_popen(*args, **kwargs)
                command = args[0] if args else kwargs["args"]
                if command[0] == str(executable) and "--version" not in command:
                    children.append(child)
                return child

            def fail_to_record(_):
                raise OSError("controlled failure to persist process identity")

            attempt = {
                "directory": str(directory), "workspace": str(workspace),
                "role": "worker", "session_id": None,
            }
            try:
                with patch("schema_extract.agent.subprocess.Popen", side_effect=capture_process):
                    result = CodexCLI(str(executable)).run(
                        attempt=attempt, prompt="test", schema={}, effort="xhigh",
                        timeout=5, on_process=fail_to_record,
                    )
                self.assertEqual(len(children), 1)
                self.assertEqual(result["outcome"], "cli_error")
                self.assertIsNotNone(
                    children[0].poll(),
                    "a process-record failure left an invocation running after return",
                )
                self.assertTrue(result["termination"]["confirmed"])
                self.assertNotEqual(result["termination"].get("reason"), "not_started")
            finally:
                for child in children:
                    if child.poll() is None:
                        child.kill()
                    child.wait(timeout=5)


class RunnerTrustTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        source = self.root / "source"
        make_source(source)
        self.store, self.initial, self.lock = create_run(
            Path(__file__).resolve().parents[2], self.root / "state", self.root / "output",
            source=str(source), config={"codex_binary": sys.executable},
        )

    def test_unconfirmed_process_identity_survives_rejected_out_of_scope_write(self):
        calls = []

        class UnconfirmedWriter:
            def run(self, *, attempt, **kwargs):
                calls.append(attempt)
                state_path = Path(attempt["directory"]).parent.parent / "state.json"
                unauthorized = read_json(state_path)
                unauthorized["stage"] = "P3"
                write_json(state_path, unauthorized)
                return {"outcome": "termination_unconfirmed", "termination": {"confirmed": False}}

        result = run(self.store, self.initial["run_id"], transport=UnconfirmedWriter())
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["stage"], "P1")
        self.assertEqual(result["checkpoint"], self.initial["checkpoint"])
        self.assertEqual(result["accepted_receipts"], {})
        self.assertEqual(
            result["active_attempts"]["P1"], calls[0]["attempt_id"],
            "rejecting a proposal must retain any invocation whose termination is unconfirmed",
        )
        self.assertEqual(self.store.load(self.initial["run_id"]), result)

    def test_wrong_resume_session_cannot_accept_a_second_draft(self):
        calls = []
        sessions = [str(uuid4()), str(uuid4())]

        class WrongResume:
            def run(self, *, attempt, **kwargs):
                calls.append(attempt)
                return continue_result(attempt, sessions[len(calls) - 1])

        result = run(self.store, self.initial["run_id"], transport=WrongResume())
        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[0]["session_id"])
        self.assertEqual(calls[1]["session_id"], sessions[0])
        self.assertEqual(result["status"], "invalid_receipt")
        self.assertEqual(list(result["accepted_receipts"]), [calls[0]["attempt_id"]])
        self.assertEqual(result["checkpoint"], calls[1]["base_checkpoint"])
        self.assertEqual(result["sessions"]["P1_P2"]["worker"], sessions[0])

    def test_fixed_source_modified_during_call_cannot_produce_an_accepted_draft(self):
        source_file = Path(self.lock["snapshot"]) / "source/operator.py"

        class SourceWriter:
            def run(self, *, attempt, **kwargs):
                source_file.chmod(0o644)
                source_file.write_text("def kernel(x):\n    return 0\n", encoding="utf-8")
                return continue_result(attempt, str(uuid4()))

        result = run(self.store, self.initial["run_id"], transport=SourceWriter())
        self.assertEqual(result["status"], "invalid_receipt")
        self.assertEqual(result["stage"], "P1")
        self.assertEqual(result["checkpoint"], self.initial["checkpoint"])
        self.assertEqual(result["accepted_receipts"], {})
        self.assertIn("fixed contents changed", result["reason"]["summary"])

    def test_replayed_result_with_saved_old_state_cannot_overwrite_newer_authority(self):
        session_id = str(uuid4())
        with self.store.writer(self.initial["run_id"]):
            first_attempt = self.store.prepare_attempt(self.initial, None)
            old_state = deepcopy(self.initial)
            first_result = continue_result(first_attempt, session_id)
            first_state = accept_result(self.store, old_state, self.lock, first_attempt, first_result)
            second_attempt = self.store.prepare_attempt(first_state, session_id)
            second_state = accept_result(
                self.store, first_state, self.lock, second_attempt,
                continue_result(second_attempt, session_id),
            )
            with self.assertRaises(ExtractError):
                accept_result(self.store, old_state, self.lock, first_attempt, first_result)
        self.assertEqual(self.store.load(self.initial["run_id"]), second_state)
        self.assertEqual(len(second_state["accepted_receipts"]), 2)

    def test_correct_receipt_evidence_cannot_hide_false_yaml_sources(self):
        source = self.lock["source"]
        prefix = f"{source['repository_id']}@{source['ref']}:"
        bad_ref = f"{source['repository_id']}@{'0' * 40}:"
        cases = (
            {"implementation": f"wrong_repository@{source['ref']}:operator.py::kernel"},
            {"implementation": bad_ref + "operator.py::kernel"},
            {"implementation": prefix + "missing.py::kernel"},
            {"implementation": prefix + "operator.py::missing_symbol"},
            {"reference": bad_ref + "operator.py::reference"},
            {"reference": prefix + "operator.py::kernel"},
        )
        for index, overrides in enumerate(cases):
            with self.subTest(overrides=overrides):
                state = self.store.create(self.lock, self.root / f"bad-source-{index}")
                with self.store.writer(state["run_id"]):
                    attempt = self.store.prepare_attempt(state, None)
                    before = deepcopy(state)
                    result = complete_p1_result(attempt, self.lock, **overrides)
                    with self.assertRaises(ExtractError):
                        accept_result(self.store, state, self.lock, attempt, result)
                self.assertEqual(self.store.load(state["run_id"]), before)
                self.assertEqual(state["accepted_receipts"], {})

    def test_actual_inner_implementation_source_can_pass_the_p1_worker_gate(self):
        source = self.lock["source"]
        inner_source = f"{source['repository_id']}@{source['ref']}:device.py::inner_kernel"
        with self.store.writer(self.initial["run_id"]):
            attempt = self.store.prepare_attempt(self.initial, None)
            updated = accept_result(
                self.store, self.initial, self.lock, attempt,
                complete_p1_result(attempt, self.lock, implementation=inner_source),
            )
        self.assertEqual((updated["stage"], updated["role"]), ("P1", "reviewer"))
        self.assertEqual(len(updated["accepted_receipts"]), 1)
        self.assertIsNone(updated["stages"]["P1"]["approval"])


class SnapshotIdentityTests(unittest.TestCase):
    def test_git_attributes_do_not_change_frozen_source_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "source"
            make_source(source)
            source_file = source / "operator.py"
            original = source_file.read_bytes().replace(b"\n", b"\r\n")
            source_file.write_bytes(original)
            (source / ".gitattributes").write_text("*.py text eol=lf\n", encoding="utf-8")
            _, _, lock = create_run(
                Path(__file__).resolve().parents[2], directory / "state", directory / "output",
                source=str(source), config={"codex_binary": sys.executable},
            )
            snapshot = Path(lock["snapshot"])
            committed = subprocess.check_output([
                "git", "-C", str(snapshot / "source.git"), "show",
                lock["source"]["ref"] + ":operator.py",
            ])
            self.assertEqual(committed, original)
            self.assertEqual((snapshot / "source/operator.py").read_bytes(), original)
            self.assertEqual(manifest(snapshot / "source"), lock["source"]["files"])
            self.assertFalse(any("source.git" in Path(name).parts for name in lock["source"]["files"]))
            source_file.write_text("The original worktree changed after freezing.\n", encoding="utf-8")
            verify_snapshot(lock)
            self.assertEqual((snapshot / "source/operator.py").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
