"""Concurrent executor publication retains immutable, stable input identity."""

from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
import shutil
import threading
import unittest
from unittest.mock import patch

from schema_extract.tools import snapshots
from schema_extract.tools.cli import _monitor_wait
from schema_extract.tools.common import ExtractError, read_json, rules_root, write_json
from schema_extract.tools.tests import test_stage_flow


class ExecutorSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.f = test_stage_flow.StageFlowTests(methodName="runTest")
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.f.create()
        self.tree = self.f.directory / "executor"
        shutil.copytree(rules_root(self.f.lock), self.tree)
        for path in (self.tree, *self.tree.rglob("*")):
            path.chmod(0o755 if path.is_dir() else 0o644)
        path = self.tree / "schema_extract/tools/executor_repairs.py"
        path.write_text(path.read_text() + "\n# Isolated snapshot test.\n")

    def freeze(self):
        return snapshots.freeze_executor_patch(self.f.store.root, self.f.lock, self.tree)

    def test_concurrent_identical_publishers_reuse_only_sealed_complete_winner(self):
        barrier = threading.Barrier(2)
        rename = snapshots.os.rename
        def simultaneous(source, destination):
            barrier.wait(timeout=10)
            return rename(source, destination)
        with patch.object(snapshots.os, "rename", side_effect=simultaneous), ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda _: self.freeze(), range(2)))
        self.assertEqual(results[0], results[1])
        root = Path(results[0]["snapshot"])
        target = root / "rules/schema_extract/tools/executor_repairs.py"
        body = target.read_bytes()
        target.chmod(0o644)
        with self.assertRaisesRegex(ExtractError, "not immutable"):
            self.freeze()
        target.chmod(0o444)
        target.parent.chmod(0o755)
        target.unlink()
        with self.assertRaises(ExtractError):
            self.freeze()
        target.write_bytes(body)
        target.chmod(0o444)
        target.rename(target.with_suffix(".saved"))
        target.symlink_to(target.with_suffix(".saved"))
        with self.assertRaises(ExtractError):
            self.freeze()
        target.unlink()
        target.with_suffix(".saved").rename(target)
        target.parent.chmod(0o555)
        lock_path = root / "executor.lock.json"
        root.chmod(0o755)
        lock_path.chmod(0o644)
        write_json(lock_path, {**results[0], "fingerprint": "0" * 64})
        with self.assertRaisesRegex(ExtractError, "identity collision"):
            self.freeze()

    def test_target_changes_during_read_are_rejected(self):
        target = self.tree / "schema_extract/tools/executor_repairs.py"
        read = Path.read_bytes
        observed = False
        def drift(path):
            nonlocal observed
            data = read(path)
            if path == target and not observed:
                observed = True
                target.write_bytes(data + b"\n# drift\n")
            return data
        with patch.object(Path, "read_bytes", drift):
            with self.assertRaisesRegex(ExtractError, "changed during snapshot"):
                self.freeze()
        self.assertTrue(observed)

    def test_monitor_wait_is_display_only_and_throttled(self):
        waiting = _monitor_wait()
        with patch("sys.stderr", new_callable=io.StringIO) as output:
            for elapsed in (0, 0.99, 1, 2, 30.9, 31, 31.1, 61):
                waiting("scan_claim", elapsed)
            self.assertEqual(len(output.getvalue().splitlines()), 3)
            self.assertIn("同一提醒 ID", output.getvalue())


if __name__ == "__main__":
    unittest.main()
