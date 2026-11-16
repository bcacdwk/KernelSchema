"""Synthetic current-format tasks; fixtures never read or update examples."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
import stat
import struct
import tempfile
import unittest
import zlib

import check_yaml
from .. import schema, yaml_source
from ..diagnostics import encode_value

TEMPLATE_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = TEMPLATE_ROOT / 'tools/tests/fixtures/minimal'
ALGO = ('root', 'CONTRACT', 'Algorithm_requirement')
HARD = ('root', 'CONTRACT', 'Hardware_specification')
MATH = ('root', 'DESIGN', 'Math_reshape')
WORK = ('root', 'DESIGN', 'Work_schedule')
STEP = WORK + ('Step', 'MAIN')
PIPE = ('root', 'DESIGN', 'Tile_pipeline', 'Step', 'MAIN')
ROLE = PIPE + ('Role', 'MAIN')


def at(data, path):
    for key in path:
        data = data[key]
    return data


def put(data, path, value):
    at(data, path[:-1])[path[-1]] = value


def add_child(data, name='branch', *, source=None):
    root = data['root']
    root['META_DATA']['Dispatch'] = {'when': 'All scalar inputs'}
    child = deepcopy(root)
    del child['CONTRACT']
    child['META_DATA']['Dispatch'] = {'id': name, 'when': 'All scalar inputs'}
    if source:
        child['DESIGN'] = {region: {'from': source} for region in ('Math_reshape', 'Work_schedule', 'Tile_pipeline')}
    data[name] = child
    return child


def add_step(data, name='NEXT', *, order=None):
    work = at(data, WORK)
    work['Step'][name] = deepcopy(at(data, STEP))
    at(data, ('root', 'DESIGN', 'Tile_pipeline', 'Step'))[name] = deepcopy(at(data, PIPE))
    work['Flow']['Launch'] = ([{'step': ['MAIN', name], 'order': order}] if order else
                              [{'step': ['MAIN']}, {'step': [name], 'start_after': 'MAIN.kernel_done'}])


def tiny_png():
    def chunk(kind, payload):
        return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(b'\0\xff\xff\xff')) + chunk(b'IEND', b''))


def role(pipe, families, *, waiting=None, member='cta[0:1]'):
    result = {'Info': {'Describe': 'Synthetic role', 'Member': member,
                       'Using_verb': {verb: {noun: {'execution_mode': mode, 'describe': 'Synthetic action guarantee'}
                                            for noun, mode in nouns.items()} for verb, nouns in families.items()}},
              'Pipe': pipe}
    if waiting:
        result['Waiting'] = waiting
    return result


class TaskCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (FIXTURE / 'root.yaml').read_text(encoding='utf-8')
        cls.base = {'root': yaml_source.loads(cls.source, 'root').value}

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='schema-test-')
        self.directory = Path(self.temp.name).resolve()
        self.addCleanup(self.temp.cleanup)
        (self.directory / 'root.yaml').write_text(self.source, encoding='utf-8')
        (self.directory / 'dispatch.md').write_text((FIXTURE / 'dispatch.md').read_text(encoding='utf-8'), encoding='utf-8')

    def write_index(self):
        names = sorted(path.name for path in self.directory.iterdir() if path.suffix in {'.yaml', '.yml'})
        content = '# 合成任务\n\n## 范围\n\n工具回归的合成标量任务\n\n## 完整设计\n\n'
        content += '| 设计 | 具体机制与保留理由 | 关系 |\n| --- | --- | --- |\n'
        content += ''.join(f'| [{Path(name).stem}]({name}) | 合成检查节点 | 当前完整基线 |\n' for name in names)
        (self.directory / 'dispatch.md').write_text(content, encoding='utf-8')

    def write_task(self, data, *, canonical=False):
        from ..yaml_format import format_document
        for path in self.directory.iterdir():
            if path.suffix in {'.yaml', '.yml'}:
                path.unlink()
        for name, value in data.items():
            content = encode_value(value)
            if canonical:
                content = format_document(yaml_source.loads(content, name))
            (self.directory / (name + '.yaml')).write_text(content, encoding='utf-8')
        self.write_index()

    def canonicalize(self):
        from ..yaml_format import format_document
        for path in self.directory.iterdir():
            if path.suffix in {'.yaml', '.yml'}:
                path.write_text(format_document(yaml_source.load(path, path.stem)), encoding='utf-8')

    def snapshot(self):
        return {p.name: (p.read_bytes(), stat.S_IMODE(p.stat().st_mode), p.stat().st_mtime_ns)
                for p in self.directory.iterdir() if p.is_file()}

    def cli(self, *args, node='root.yaml'):
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try:
                code = check_yaml.main([str(self.directory / node), *args])
            except SystemExit as error:
                code = error.code
        return code, out.getvalue(), err.getvalue()

    def assert_invalid(self, data, *messages):
        self.write_task(data)
        before = self.snapshot()
        with self.assertRaises(schema.SchemaError) as caught:
            schema.validate_directory(self.directory)
        for message in messages:
            self.assertIn(message, str(caught.exception))
        self.assertEqual(before, self.snapshot())
        return str(caught.exception)
