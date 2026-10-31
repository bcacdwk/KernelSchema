"""Optional maintained examples use the public single-file entry point."""
from contextlib import redirect_stderr, redirect_stdout
import io
import hashlib
import stat
import unittest

import check_yaml
from ..pipeline.output import metadata
from .support import TEMPLATE_ROOT


def snapshot(directory):
    return {path.name: (path.read_bytes(), stat.S_IMODE(path.stat().st_mode), path.stat().st_mtime_ns)
            for path in directory.iterdir() if path.is_file()}


class ExampleTests(unittest.TestCase):
    def test_operator_directories_keep_dispatch_yaml_and_current_pngs(self):
        examples = TEMPLATE_ROOT / 'examples'
        for task in sorted(path for path in examples.iterdir() if path.is_dir()):
            with self.subTest(task=task.name):
                sources = sorted(path for path in task.iterdir() if path.suffix in {'.yaml', '.yml'})
                expected = {'dispatch.md'} | {path.name for path in sources} | {path.with_suffix('.png').name for path in sources}
                self.assertEqual({path.name for path in task.iterdir()}, expected)
                for source in sources:
                    image = source.with_suffix('.png')
                    info = metadata(image)
                    self.assertEqual(info.get('kind'), 'pipeline_graph', str(image))
                    self.assertEqual(info.get('source'), source.name, str(image))
                    self.assertEqual(info.get('source_sha256'), hashlib.sha256(source.read_bytes()).hexdigest(), str(image))

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
