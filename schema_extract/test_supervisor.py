"""Supervisor boundaries, with all Codex/process/scheduler operations mocked."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .codex_cli import write_json
from .framework import candidate_report
from .records import StateError, Store, read_json
from . import supervisor


THREAD = "11111111-1111-4111-8111-111111111111"


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="schema-supervisor-test-")
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        self.store = Store(base / "repo", base / "records")
        self.store.repo.mkdir()
        self.directory = self.store.root / "runs" / "selected"
        self.directory.mkdir(parents=True)
        self.plan = {"run_id": "selected", "tasks": ["selected-task"],
                     "work_root": str(base / "work"), "config": {"model": "synthetic"}}
        write_json(self.directory / "plan.json", self.plan)
        self.task = {"key": "selected-task", "operator_id": "selected-operator", "status": "queued",
                     "phase": "worker", "stage": "contract", "node": "root", "attempts": [],
                     "active_attempt": None, "active_process": None, "usage": {}, "issues": []}
        self.store.task_dir(self.task["key"]).mkdir(parents=True)
        self.store.save(self.task)
        process_table = patch.object(supervisor, "process_table", return_value={})
        process_table.start()
        self.addCleanup(process_table.stop)
        resolve = patch.object(supervisor, "executable_path", return_value="/synthetic/bin/codex")
        resolve.start()
        self.addCleanup(resolve.stop)

    def result(self, action="observe"):
        return {"outcome": "completed", "thread_id": THREAD, "termination": {"confirmed": True},
                "response": {"run_id": "selected", "action": action, "summary": "Observed current state.",
                             "observations": []}}

    def test_idle_and_paused_work_never_call_model_or_runner(self):
        for status, control in (("ended", None), ("stopped", None), ("queued", "pause"), ("queued", "cancel")):
            with self.subTest(status=status, control=control):
                self.task["status"] = status
                self.store.save(self.task)
                if control:
                    write_json(self.directory / "control.json", {"mode": control})
                with patch.object(supervisor, "CodexCLI") as cli, patch.object(supervisor, "start_runner") as launch:
                    self.assertEqual(supervisor.tick(self.store, "selected")["outcome"], "idle")
                cli.assert_not_called()
                launch.assert_not_called()

    def test_completed_tick_resumes_its_exact_thread(self):
        with patch.object(supervisor, "CodexCLI") as cli:
            cli.return_value.run.return_value = self.result()
            supervisor.tick(self.store, "selected")
            supervisor.tick(self.store, "selected")
        calls = cli.return_value.run.call_args_list
        self.assertIsNone(calls[0].kwargs["thread_id"])
        self.assertEqual(calls[1].kwargs["thread_id"], THREAD)
        self.assertEqual(calls[1].kwargs["role"], "supervisor")
        self.assertEqual(calls[0].kwargs["cwd"], self.directory / "supervisor" / "workspace")
        self.assertEqual(calls[0].kwargs["cwd"], calls[1].kwargs["cwd"])
        self.assertNotEqual(calls[0].kwargs["records"], calls[1].kwargs["records"])
        self.assertNotIn("timeout", calls[1].kwargs)

    def test_candidate_survives_later_tick_and_is_independent_of_dispatch(self):
        (self.store.repo / "schema_extract/issues").mkdir(parents=True)
        task = self.store.task_dir(self.task["key"]) / "task.json"
        before = task.read_bytes()
        proposal = {"id": "missing-baseline", "category": "extract_tools", "problem": "Unknown baseline in snapshot.",
                    "proposal": "Report the baseline.", "necessity": "Otherwise each tick repeats the investigation.",
                    "evidence": ["selected-task: synthetic summary shows unknown baseline."]}
        first = self.result()
        first["response"]["framework_candidates"] = [proposal]
        invalid = self.result("start_queued")
        invalid["response"]["framework_candidates"] = [{"id": "not-enough"}]
        results = iter([first, invalid, self.result()])
        def respond(**kwargs):
            result = next(results)
            kwargs["records"].mkdir()
            write_json(kwargs["records"] / "result.json", result)
            return result
        with patch.object(supervisor, "CodexCLI") as cli, patch.object(supervisor, "start_runner", return_value={}) as launch:
            cli.return_value.run.side_effect = respond
            for _ in range(3):
                self.assertEqual(supervisor.tick(self.store, "selected")["outcome"], "observed")
            launch.assert_called_once()
            self.assertIn("framework_candidates", cli.return_value.run.call_args.kwargs["prompt"])
        self.assertEqual(task.read_bytes(), before)
        groups = candidate_report(self.store, self.plan)
        self.assertEqual([g["id"] for g in groups], [proposal["id"]])
        source, = groups[0]["reports"][0]["sources"]
        self.assertEqual(source["role"], "supervisor")
        self.assertEqual(read_json(self.store.root / source["record"])["response"]["framework_candidates"], [proposal])
        self.assertIn("selected-task", read_json(self.store.root / source["summary"])["snapshot"]["tasks"])

    def test_changed_terminal_problems_are_observed_once_without_dispatch(self):
        with patch.object(supervisor, "CodexCLI") as cli, patch.object(supervisor, "start_runner") as launch:
            cli.return_value.run.return_value = self.result("needs_attention")
            for status, reason in (("stopped", "runner_failure"), ("ended", "protocol_failure"), ("ended", "needs_input")):
                self.task.update(status=status, reason=reason, last_error=reason)
                self.store.save(self.task)
                before = deepcopy(self.store.load(self.task["key"]))
                self.assertEqual(supervisor.tick(self.store, "selected")["outcome"], "observed")
                self.assertEqual(supervisor.tick(self.store, "selected")["outcome"], "idle")
                self.assertEqual(self.store.load(self.task["key"]), before)
        self.assertEqual(cli.return_value.run.call_count, 3)
        launch.assert_not_called()

    def test_failed_terminal_diagnosis_is_retried_before_idle(self):
        self.task.update(status="stopped", reason="runner_failure", last_error="synthetic failure")
        self.store.save(self.task)
        failed = self.result()
        failed["outcome"] = "cli_error"
        with patch.object(supervisor, "CodexCLI") as cli:
            cli.return_value.run.side_effect = [failed, self.result("needs_attention")]
            self.assertEqual(supervisor.tick(self.store, "selected")["outcome"], "model_error")
            self.assertEqual(supervisor.tick(self.store, "selected")["outcome"], "observed")
            self.assertEqual(supervisor.tick(self.store, "selected")["outcome"], "idle")
        self.assertEqual(cli.return_value.run.call_count, 2)

    def test_explicit_pause_defers_a_new_terminal_problem(self):
        self.task.update(status="ended", reason="needs_input")
        self.store.save(self.task)
        write_json(self.directory / "control.json", {"mode": "pause"})
        with patch.object(supervisor, "CodexCLI") as cli:
            cli.return_value.run.return_value = self.result("needs_attention")
            self.assertEqual(supervisor.tick(self.store, "selected")["outcome"], "idle")
            cli.assert_not_called()
            (self.directory / "control.json").unlink()
            self.assertEqual(supervisor.tick(self.store, "selected")["outcome"], "observed")
            self.assertEqual(supervisor.tick(self.store, "selected")["outcome"], "idle")
            write_json(self.directory / "control.json", {"mode": "pause"})
            self.assertEqual(supervisor.tick(self.store, "selected")["outcome"], "idle")
            (self.directory / "control.json").unlink()
            self.assertEqual(supervisor.tick(self.store, "selected")["outcome"], "idle")
        self.assertEqual(cli.return_value.run.call_count, 1)

    def test_pause_arriving_during_model_call_prevents_dispatch(self):
        def decide(**kwargs):
            write_json(self.directory / "control.json", {"mode": "pause"})
            self.assertFalse(kwargs["cancel"]())
            return self.result("start_queued")

        with patch.object(supervisor, "CodexCLI") as cli, patch.object(supervisor.subprocess, "Popen") as spawn:
            cli.return_value.run.side_effect = decide
            result = supervisor.tick(self.store, "selected")
        self.assertEqual(result["runner"]["outcome"], "control_requested")
        spawn.assert_not_called()
        self.assertEqual(self.store.load(self.task["key"])["status"], "queued")

    def test_cancel_arriving_during_model_call_stops_the_inspection(self):
        before = deepcopy(self.store.load(self.task["key"]))

        def inspect(**kwargs):
            self.assertIn("cancel", kwargs)
            self.assertFalse(kwargs["cancel"]())
            write_json(self.directory / "control.json", {"mode": "cancel"})
            self.assertTrue(kwargs["cancel"]())
            return {**self.result("start_queued"), "outcome": "cancelled", "response": None}

        with patch.object(supervisor, "CodexCLI") as cli, patch.object(supervisor, "start_runner") as launch:
            cli.return_value.run.side_effect = inspect
            self.assertEqual(supervisor.tick(self.store, "selected")["outcome"], "cancelled")
            self.assertEqual(supervisor.tick(self.store, "selected")["outcome"], "idle")
        self.assertEqual(cli.return_value.run.call_count, 1)
        launch.assert_not_called()
        self.assertNotIn("pending", read_json(self.directory / "supervisor" / "state.json"))
        self.assertEqual(self.store.load(self.task["key"]), before)

    def test_two_busy_workers_leave_an_independent_read_only_inspection_slot(self):
        self.task["status"] = "running"
        self.store.save(self.task)
        other = {**self.task, "key": "other-task", "operator_id": "other-operator"}
        self.store.task_dir(other["key"]).mkdir()
        self.store.save(other)
        self.plan["tasks"].append(other["key"])
        write_json(self.directory / "plan.json", self.plan)
        before = [deepcopy(self.store.load(key)) for key in self.plan["tasks"]]
        with self.store.lock(), patch.object(supervisor, "CodexCLI") as cli:
            cli.return_value.run.return_value = self.result()
            self.assertEqual(supervisor.tick(self.store, "selected")["outcome"], "observed")
        cli.return_value.run.assert_called_once()
        self.assertEqual(cli.return_value.run.call_args.kwargs["role"], "supervisor")
        self.assertNotIn("timeout", cli.return_value.run.call_args.kwargs)
        self.assertEqual([self.store.load(key) for key in self.plan["tasks"]], before)

    def test_failed_or_out_of_scope_response_never_dispatches(self):
        failed = self.result("start_queued")
        failed["outcome"] = "cli_error"
        outside = self.result("start_queued")
        outside["response"]["observations"] = [{"task_key": "other-task", "finding": "bad scope", "suggestion": "start"}]
        for response, outcome in ((failed, "model_error"), (outside, "error")):
            with self.subTest(outcome=outcome):
                with patch.object(supervisor, "CodexCLI") as cli, patch.object(supervisor, "start_runner") as launch:
                    cli.return_value.run.return_value = response
                    self.assertEqual(supervisor.tick(self.store, "selected")["outcome"], outcome)
                launch.assert_not_called()

    def test_queued_dispatch_uses_existing_runner_and_only_records_a_request(self):
        (self.directory / "supervisor").mkdir()
        before = deepcopy(self.store.load(self.task["key"]))
        with patch.object(supervisor.subprocess, "Popen") as spawn:
            spawn.return_value.pid = 12345
            result = supervisor.start_runner(self.store, "selected", "/synthetic/bin/codex")
        args, kwargs = spawn.call_args
        self.assertEqual(args[0][-2:], ["run", "selected"])
        self.assertEqual(args[0][1:3], ["-m", "schema_extract"])
        self.assertIn("/synthetic/bin/codex", args[0])
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(kwargs["stdin"], supervisor.subprocess.DEVNULL)
        self.assertEqual(kwargs["stderr"], supervisor.subprocess.STDOUT)
        self.assertEqual(result["outcome"], "requested")
        self.assertEqual(self.store.load(self.task["key"]), before)
        spawn.return_value.wait.assert_not_called()

    def test_long_call_observes_activity_without_reading_artifacts(self):
        records = self.store.task_dir(self.task["key"]) / "attempts" / "a-1" / "call-01"
        records.mkdir(parents=True)
        work = Path(self.plan["work_root"]) / self.task["key"] / "a-1"
        work.mkdir(parents=True)
        (work / "root.yaml").write_text("private source/design content must not enter the summary\n")
        (records / "events.jsonl").write_text(json.dumps({"type": "item.completed", "item": {
            "type": "command_execution", "exit_code": 0, "aggregated_output": "private tool output"}}) + "\n")
        self.task.update(status="running", active_attempt="a-1", attempts=[{
            "id": "a-1", "work": str(work), "calls": [str(records)]}], active_process={
            "records": str(records), "started_at": 1, "identity": {"pid": 123, "started": "known"}})
        self.store.save(self.task)
        before = deepcopy(self.store.load(self.task["key"]))
        first = supervisor.snapshot(self.store, self.plan, now=7201)
        later = supervisor.snapshot(self.store, self.plan, now=8101)
        self.assertTrue(first["tasks"][self.task["key"]]["investigate_activity"])
        self.assertNotIn("private", json.dumps(first))
        self.assertEqual(supervisor.changes(first, later)[self.task["key"]]["changed"], [])
        with patch.object(supervisor, "CodexCLI") as cli:
            cli.return_value.run.return_value = self.result("needs_attention")
            supervisor.tick(self.store, "selected")
        self.assertEqual(self.store.load(self.task["key"]), before)

    def test_live_interrupted_supervisor_is_not_replaced(self):
        directory = self.directory / "supervisor"
        records = directory / "tick-old" / "codex"
        records.mkdir(parents=True)
        write_json(records / "process.json", {"identity": {"pid": 123, "started": "known"}, "descendants": []})
        write_json(directory / "state.json", {"run_id": "selected", "thread_id": THREAD,
            "pending": {"records": str(records), "thread_id": THREAD}})
        table = {123: {"pid": 123, "started": "known", "stat": "S", "pgid": 123}}
        with patch.object(supervisor, "process_table", return_value=table), patch.object(supervisor, "CodexCLI") as cli:
            self.assertEqual(supervisor.tick(self.store, "selected")["outcome"], "previous_tick_active")
        cli.assert_not_called()
        self.assertEqual(read_json(directory / "state.json")["thread_id"], THREAD)

    def test_schedule_is_periodic_without_immediate_or_persistent_execution(self):
        config = supervisor.schedule_config(self.store, "selected", "/synthetic/bin/codex")
        plist = supervisor.launchd_plist(config)
        self.assertEqual(plist["StartInterval"], 900)
        self.assertFalse(plist["RunAtLoad"])
        self.assertFalse(plist["KeepAlive"])
        self.assertEqual(set(plist["EnvironmentVariables"]), {"PATH"})
        self.assertTrue(Path(plist["ProgramArguments"][0]).is_absolute())
        self.assertIn("/synthetic/bin/codex", plist["ProgramArguments"])
        self.assertNotEqual(supervisor.schedule_label(self.store, "selected"), supervisor.schedule_label(self.store, "other"))

    def test_schedule_changes_wait_for_active_tick_and_exclude_new_ticks(self):
        directory = self.directory / "supervisor"
        directory.mkdir()
        with patch.object(supervisor.sys, "platform", "darwin"), \
                patch.object(Path, "home", return_value=self.store.repo / "home"), \
                patch.object(supervisor, "launchd_state", return_value={"loaded": True}) as query, \
                patch.object(supervisor.subprocess, "run") as launchctl:
            config = supervisor.schedule_config(self.store, "selected", "/synthetic/bin/codex")
            config["outcome"] = "enabled"
            write_json(directory / "schedule.json", config)
            plist = Path(config["plist"])
            plist.parent.mkdir(parents=True)
            plist.write_bytes(supervisor.plistlib.dumps(supervisor.launchd_plist(config)))
            original = plist.read_bytes()
            launchctl.return_value = supervisor.subprocess.CompletedProcess([], 0, stderr="")
            with supervisor.try_lock(directory / ".tick.lock") as active:
                self.assertIsNotNone(active)
                for enable in (False, True):
                    with self.subTest(active=True, enable=enable):
                        with self.assertRaisesRegex(StateError, "tick is active"):
                            supervisor.schedule(self.store, "selected", "/synthetic/bin/codex", enable=enable)
            query.assert_not_called()
            launchctl.assert_not_called()
            self.assertEqual(read_json(directory / "schedule.json"), config)
            self.assertEqual(plist.read_bytes(), original)

            def update(command, **kwargs):
                self.assertEqual(supervisor.tick(self.store, "selected")["outcome"], "tick_already_active")
                return supervisor.subprocess.CompletedProcess(command, 0, stderr="")

            launchctl.side_effect = update
            for enable, outcome in ((True, "enabled"), (False, "disabled")):
                with self.subTest(active=False, enable=enable):
                    result = supervisor.schedule(self.store, "selected", "/synthetic/bin/codex", enable=enable)
                    self.assertEqual(result["outcome"], outcome)
            self.assertEqual([call.args[0][1] for call in launchctl.call_args_list], ["bootout", "bootstrap", "bootout"])
            self.assertFalse(plist.exists())
            self.assertFalse((self.directory / "control.json").exists())
            with supervisor.try_lock(directory / ".tick.lock") as released:
                self.assertIsNotNone(released)

    def test_unselected_decision_is_rejected(self):
        value = self.result()["response"]
        value["run_id"] = "other"
        with self.assertRaises(StateError):
            supervisor.validate_decision(value, self.plan)


if __name__ == "__main__":
    unittest.main()
