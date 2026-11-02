"""CLI process lifecycle and usage accounting."""

from copy import deepcopy
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from ..codex_cli import CodexCLI, process_table, terminate_owned
from ..execution import observe_usage
from ..records import read_json


class UsageObservationTests(unittest.TestCase):
    def test_real_cumulative_pairs_and_duplicate_observation(self):
        for first, second, expected in (
            ((1736471, 23617), (2026963, 26840), 293715),
            ((64427, 225), (132440, 449), 68237),
        ):
            with self.subTest(first=first):
                state = {"usage": {"reported_tokens": 0, "unknown_usage_calls": 0}}
                one = {"thread_id": "thread", "usage": {"input_tokens": first[0], "output_tokens": first[1]}}
                two = {"thread_id": "thread", "resume_thread_id": "thread",
                       "usage": {"input_tokens": second[0], "output_tokens": second[1], "cached_input_tokens": 99, "reasoning_output_tokens": 11}}
                raw = deepcopy(two)
                observe_usage(state, one, "first")
                observed = observe_usage(state, two, "second")
                self.assertEqual(observed["delta_tokens"], expected)
                self.assertEqual(state["usage"]["reported_tokens"], sum(second))
                self.assertEqual(observe_usage(state, two, "second"), observed)
                self.assertEqual(state["usage"]["reported_tokens"], sum(second))
                self.assertEqual(two, raw)

    def test_missing_and_decreased_usage_are_observational_unknowns(self):
        state = {"usage": {"reported_tokens": 0, "unknown_usage_calls": 0}}
        observe_usage(state, {"thread_id": "t", "usage": {"input_tokens": 100, "output_tokens": 5}}, "first")
        missing = observe_usage(state, {"thread_id": "t", "usage": {}}, "missing")
        decreased = observe_usage(state, {"thread_id": "t", "usage": {"input_tokens": 90, "output_tokens": 2}}, "decreased")
        unknown = observe_usage(state, {"thread_id": "other", "resume_thread_id": "other",
                                       "usage": {"input_tokens": 80, "output_tokens": 1}}, "unknown-resume")
        self.assertIsNone(missing["delta_tokens"])
        self.assertIsNone(decreased["delta_tokens"])
        self.assertIsNone(unknown["delta_tokens"])
        self.assertEqual(state["usage"]["reported_tokens"], 105)
        self.assertEqual(state["usage"]["unknown_usage_calls"], 3)
        self.assertNotIn("status", state)


class TransportTests(unittest.TestCase):
    def test_spawn_failure_records_proof_that_no_agent_process_started(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cli = CodexCLI()
            with patch.object(cli, "version", return_value="synthetic"), patch("schema_extract.tools.codex_cli.subprocess.Popen", side_effect=OSError("synthetic spawn failure")):
                result = cli.run(cwd=directory, records=directory / "call", prompt="test", schema={})
            self.assertEqual(result["outcome"], "spawn_error")
            self.assertIn("synthetic spawn failure", result["error"])
            self.assertEqual(result, read_json(directory / "call/result.json"))
            self.assertTrue(result["termination"]["confirmed"])
            self.assertEqual(result["termination"]["reason"], "process_not_started")
            self.assertFalse((directory / "call/process.json").exists())

    def test_version_timeout_is_a_preflight_error_not_an_agent_timeout(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cli = CodexCLI()
            with patch.object(cli, "version", side_effect=subprocess.TimeoutExpired("codex --version", 15)), patch("schema_extract.tools.codex_cli.subprocess.Popen") as spawn:
                result = cli.run(cwd=directory, records=directory / "call", prompt="test", schema={})
            spawn.assert_not_called()
            self.assertEqual(result["outcome"], "preflight_error")
            self.assertTrue(result["termination"]["confirmed"])

    def test_unconfirmed_termination_takes_priority_over_missing_resume_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cli = CodexCLI()
            process = Mock(pid=12345)
            process.poll.return_value = process.wait.return_value = 0
            identity = {"pid": process.pid, "started": "synthetic"}
            with patch.object(cli, "version", return_value="synthetic"), \
                    patch("schema_extract.tools.codex_cli.subprocess.Popen", return_value=process), \
                    patch("schema_extract.tools.codex_cli.process_table", return_value={process.pid: identity}), \
                    patch("schema_extract.tools.codex_cli.terminate_owned", return_value={"confirmed": False, "actions": [], "survivors": [identity]}):
                result = cli.run(cwd=directory, records=directory / "call", prompt="test", schema={},
                                 thread_id="00000000-0000-0000-0000-000000000001")
            self.assertEqual(result["outcome"], "termination_unconfirmed")
            self.assertFalse(result["termination"]["confirmed"])

    def test_process_tree_termination_and_safe_cli_arguments(self):
        process = subprocess.Popen(["python3", "-c", "import subprocess,time; subprocess.Popen(['python3','-c','import time;time.sleep(60)']); time.sleep(60)"], start_new_session=True)
        try:
            identity = process_table()[process.pid]
            time.sleep(0.2)
            result = terminate_owned(identity)
            self.assertTrue(result["confirmed"])
            process.wait(timeout=5)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        command = CodexCLI().command(Path("/tmp/task"), Path("/tmp/schema"), Path("/tmp/output"), "reviewer", None)
        self.assertIn("read-only", command)
        self.assertIn('model_reasoning_effort="xhigh"', command)
        self.assertNotIn("--ignore-rules", command)
        self.assertNotIn("danger-full-access", command)
        supervisor = CodexCLI().command(Path("/tmp/task"), Path("/tmp/schema"), Path("/tmp/output"), "supervisor", None)
        self.assertIn("read-only", supervisor)
        with self.assertRaises(ValueError):
            CodexCLI().command(Path("/tmp/task"), Path("/tmp/schema"), Path("/tmp/output"), "worker", "--last")
