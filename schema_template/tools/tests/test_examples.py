"""Optional maintained examples use the public single-file entry point."""
from contextlib import redirect_stderr, redirect_stdout
import io
import stat
import unittest

import check_yaml
from .support import TEMPLATE_ROOT


def snapshot(directory):
    return {path.name: (path.read_bytes(), stat.S_IMODE(path.stat().st_mode), path.stat().st_mtime_ns)
            for path in directory.iterdir() if path.is_file()}


class ExampleTests(unittest.TestCase):
    def test_examples_pass_without_changes(self):
        examples = TEMPLATE_ROOT / 'examples'
        if not examples.is_dir():
            self.skipTest('No optional examples are maintained.')
        for task in sorted(path for path in examples.iterdir() if path.is_dir()):
            with self.subTest(task=task.name):
                entry = task / 'root.yaml'
                if not entry.is_file():
                    entry = task / 'root.yml'
                before = snapshot(task)
                output, errors = io.StringIO(), io.StringIO()
                with redirect_stdout(output), redirect_stderr(errors):
                    status = check_yaml.main([str(entry)])
                self.assertEqual(0, status, errors.getvalue())
                self.assertEqual(before, snapshot(task))
