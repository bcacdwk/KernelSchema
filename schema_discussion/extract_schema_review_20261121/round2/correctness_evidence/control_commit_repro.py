"""Fault injection against real temporary Store transactions; never spawn a child.

Static source/process checks are fixtures. CLI -> request_restart ->
reserve_launch -> post-Popen ledger writes are current production code.
"""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from schema_extract.tools import cli, monitor
from schema_extract.tools.common import SupervisorBusy
from schema_extract.tools.store import Store


def scenario(busy_at):
    with tempfile.TemporaryDirectory() as temporary:
        store = Store(Path(temporary) / "store")
        store.initialize()
        run_id = str(uuid4())
        store.run_dir(run_id).mkdir(parents=True)
        state = {"run_id": run_id, "status": "invalid_receipt", "created_at": 1,
                 "input_fingerprint": "fixed-input", "executor": {"digest": "fixed-executor", "capabilities": ["scoped_recovery.v2"]},
                 "output": str(Path(temporary) / "output"), "control": {"action": None, "requests": {}}}
        evidence = {"run_id": run_id, "scope": "P3", "attempt_id": str(uuid4()), "failure_id": "fixed-failure",
                    "target": {"input_fingerprint": "fixed-input", "executor_digest": "fixed-executor"}}
        launcher = {"pid": os.getpid(), "start_token": "linux:fixture:1", "stat": "S", "ppid": 1, "pgid": os.getpid()}
        child = {"pid": 87654321, "start_token": "linux:fixture:2", "stat": "S", "ppid": os.getpid(), "pgid": 87654321}
        table = {item["pid"]: item for item in (launcher, child)}
        entries = 0
        popens = 0
        actual_supervisor = store.supervisor

        @contextmanager
        def injected():
            nonlocal entries
            entries += 1
            if entries == busy_at:
                raise SupervisorBusy("injected contention at transaction " + str(entries))
            with actual_supervisor() as ledger:
                yield ledger

        class FakeChild:
            pid = child["pid"]
            def poll(self):
                return None

        def fake_popen(*args, **kwargs):
            nonlocal popens
            popens += 1
            return FakeChild()

        patches = [patch.object(store, "load", side_effect=lambda _: deepcopy(state)),
                   patch.object(store, "writer_status", return_value="free"),
                   patch.object(store, "supervisor", side_effect=injected),
                   patch.object(cli, "Store", return_value=store), patch.object(cli, "_frozen_entry"),
                   patch.object(monitor, "restart_evidence", return_value=evidence),
                   patch.object(monitor, "process_table", return_value=table),
                   patch.object(monitor, "_processes", return_value=(table, None)),
                   patch.object(monitor.subprocess, "Popen", side_effect=fake_popen),
                   patch("schema_extract.tools.runtime.verify_execution"),
                   patch("schema_extract.tools.runtime.frozen_run_invocation")]
        for item in patches:
            item.start()
        try:
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = cli.main(["resume", run_id, "--scope", "P3", "--repo", str(ROOT), "--state-root", str(store.root)])
            first = store.read_supervisor()
            registered = list(first["retries"])
            launch = first["starts"].get(run_id, {}).get("launch")
            report = {"busy_transaction": busy_at, "exit_code": exit_code, "stderr": json.loads(stderr.getvalue()),
                      "durable_retry_count": len(registered), "durable_retry_statuses": [value["status"] for value in first["retries"].values()],
                      "launch_reserved": launch is not None, "launch_links_request": bool(launch and launch.get("retry_id") in registered),
                      "fake_popen_calls_at_failure": popens, "real_child_processes": 0}
            if busy_at == 3:
                stdout, stderr = io.StringIO(), io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    replay_code = cli.main(["resume", run_id, "--scope", "P3", "--repo", str(ROOT), "--state-root", str(store.root)])
                after = store.read_supervisor()
                replay = json.loads(stdout.getvalue())
                report["same_fixture_cli_replay"] = {"exit_code": replay_code, "retry_ids_unchanged": list(after["retries"]) == registered,
                    "launch_id_unchanged": after["starts"][run_id]["launch"]["id"] == launch["id"],
                    "reused_launch": replay.get("launch", {}).get("reused"), "total_fake_popen_calls": popens}
            return report
        finally:
            for item in reversed(patches):
                item.stop()


print(json.dumps([scenario(number) for number in (1, 2, 3)], ensure_ascii=False, indent=2))
