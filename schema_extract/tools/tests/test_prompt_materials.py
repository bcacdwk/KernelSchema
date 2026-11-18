"""Current invocation materials, staged checks and real source evidence."""

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from schema_extract.tools.agent import AgentCLI
from schema_extract.tools.common import manifest, read_json, rules_root, source_root, write_json
from schema_extract.tools.prompts import build_prompt
from schema_extract.tools.reading import stage_check
from schema_extract.tools.tests import test_stage_flow


FAKE_CHECKER = r'''
import json, subprocess, sys
from pathlib import Path
from uuid import uuid4
if '--version' in sys.argv:
    print('extract-stage-helper-fixture 1')
    raise SystemExit()
prompt = sys.stdin.read()
context = json.loads(prompt.split("<invocation>\n", 1)[1].split("\n</invocation>", 1)[0])
import shlex
helper = shlex.split(context["check_command"])
schema = json.loads(sys.argv[sys.argv.index('--json-schema') + 1])
session = str(uuid4())
print(json.dumps({'type':'system','subtype':'init','session_id':session}), flush=True)
check = subprocess.run(helper, capture_output=True, text=True)
extra = subprocess.run([*helper, '--write'], capture_output=True, text=True)
response = {'attempt_id':schema['properties']['attempt_id']['enum'][0], 'exit_code':check.returncode,
            'output':check.stdout, 'stderr':check.stderr, 'extra_rejected':extra.returncode != 0,
            'host_visible':Path(schema['description']).exists()}
print(json.dumps({'type':'result','subtype':'success','is_error':False,'session_id':session,
                  'structured_output':response}), flush=True)
'''


class PromptMaterialsTests(unittest.TestCase):
    def setUp(self):
        self.flow = test_stage_flow.StageFlowTests()
        self.flow.setUp()
        self.addCleanup(self.flow.doCleanups)

    def check(self, attempt):
        config = read_json(Path(attempt['directory']) / 'materials/check.json')
        return stage_check(config, workspace=Path(attempt['workspace']), rules=rules_root(self.flow.lock),
                           baseline=Path(attempt['directory']) / 'materials/baseline', source=source_root(self.flow.lock))

    def test_p1_preview_is_read_only_and_reports_pending_normalization(self):
        self.flow.create()
        attempt = self.flow.prepare()
        self.flow.backend.write(attempt)
        before = manifest(Path(attempt['workspace']))
        completed = subprocess.run([attempt['check_command']], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(manifest(Path(attempt['workspace'])), before)
        self.assertTrue(result['preview_passed'], result)
        self.assertTrue(result['pending_corrections'])
        self.assertFalse(result['raw_passed'])
        self.assertEqual(result['exit_code'], 0)
        self.assertTrue(result['corrections'])
        self.assertEqual(result['source_validation']['status'], 'verified')

        root = Path(attempt['workspace']) / 'root.yaml'
        text = root.read_text()
        describe = self.flow.backend.base['META_DATA']['Describe'] + '.'
        self.assertIn(describe, text)
        root.write_text(text.replace(describe, 'x' * 151, 1))
        rejected = self.check(attempt)
        self.assertEqual(rejected['exit_code'], 1, rejected)
        self.assertFalse(rejected['preview_passed'])
        self.assertIn('META_DATA.Describe: text has 151', '; '.join(rejected['check']['diagnostics']))

    def test_p2_and_p4_do_not_require_planned_files_and_scopes_still_apply(self):
        self.flow.create(('root', 'planned'))
        self.flow.step(stage='P1', role='worker')
        self.flow.step(stage='P1', role='reviewer')
        attempt = self.flow.prepare(stage='P2', role='worker')
        self.flow.backend.write(attempt)
        checked = self.check(attempt)
        self.assertTrue(checked['preview_passed'], checked)
        self.assertEqual(checked['check']['selection'], 'awaiting_proposal')
        self.assertFalse(checked['raw_passed'])
        self.flow.accept(attempt)
        self.flow.step(stage='P2', role='reviewer')
        self.flow.step(stage='P3', role='worker')
        self.flow.step(stage='P3', role='reviewer')
        attempt = self.flow.prepare(stage='P4', role='worker')
        self.flow.backend.write(attempt)
        checked = self.check(attempt)
        self.assertEqual(checked['exit_code'], 0, checked)
        root = Path(attempt['workspace']) / 'root.yaml'
        root.write_text(root.read_text().replace('Version: 3', 'Version: 4'))
        rejected = self.check(attempt)
        self.assertEqual(rejected['exit_code'], 1, rejected)
        self.assertTrue(rejected['diagnostics'])

    def test_p6_receives_worker_facts_after_final_content_and_only_real_documents_are_provided(self):
        self.flow.create()
        self.flow.through_root()
        attempt = self.flow.prepare(stage='P6', role='reviewer')
        text = build_prompt(self.flow.state, self.flow.lock, attempt)
        context = json.loads(text.split('<invocation>\n', 1)[1].split('\n</invocation>', 1)[0])
        self.assertEqual(context['workspace'], attempt['workspace'])
        self.assertEqual(context['workflow_page'], str(rules_root(self.flow.lock) / 'EXTRACT_SCHEMA.md'))
        self.assertNotIn('/source', context['implementation_source'])
        self.assertEqual(context['reading_order'][-1], 'process_records')
        self.assertTrue(any(item['stage'] == 'P2' and item.get('role') == 'worker' for item in context['process_records']))
        self.assertTrue(any(item['stage'] == 'P4' and item.get('role') == 'worker' for item in context['process_records']))
        self.assertFalse(set(context['read_specs']) & set(context['provided_documents']))
        for path in context['provided_documents']:
            self.assertIn('<document path=' + json.dumps(path) + '>', text)
        material = attempt['materials']
        for record in material['records']:
            relative = Path(record['path']).relative_to(material['directory']).as_posix()
            self.assertEqual(manifest(Path(material['directory']))[relative], material['files'][relative]['sha256'])
        exposed = set(attempt['read_directories'])
        available_rules = {Path(item['path']).relative_to(rules_root(self.flow.lock)).as_posix()
                           for item in material['rules']}
        self.assertTrue({'README.md', 'EXTRACT_SCHEMA.md', 'schema_extract/README.md',
                         'schema_extract/REVIEW.md',
                         'schema_extract/tools/README.md', 'schema_extract/issues/README.md',
                         'schema_template/README.md', 'schema_template/tools/README.md'} <= available_rules)
        self.assertFalse(any('/archive/' in name or name.startswith(('schema_wiki/', 'kernel_wiki/'))
                             for name in available_rules))
        self.assertNotIn(str(self.flow.store.run_dir(attempt['run_id'])), exposed)
        self.assertEqual(self.check(attempt)['exit_code'], 0)
        root = Path(attempt['workspace']) / 'root.yaml'
        self.assertIn('META_DATA', root.read_text())
        root.write_text(root.read_text().replace('META_DATA', 'meta_data', 1))
        rejected = self.check(attempt)
        self.assertEqual(rejected['exit_code'], 1, rejected)
        self.assertFalse(rejected['raw_passed'])
        self.assertNotIn(str(self.flow.store.root), exposed)

    def test_real_helper_reads_frozen_source_and_rejects_extra_arguments(self):
        self.flow.create()
        self.flow.step(stage='P1', role='worker')
        attempt = self.flow.prepare(stage='P1', role='reviewer')
        fake = self.flow.directory / 'helper-cli'
        fake.write_text('#!' + sys.executable + '\n' + FAKE_CHECKER)
        fake.chmod(0o755)
        attempt['executable'] = str(fake)
        write_json(Path(attempt['directory']) / 'attempt.json', attempt)
        schema = {'type':'object', 'additionalProperties':False,
                  'description':str(self.flow.store.run_dir(attempt['run_id']) / 'state.json'),
                  'properties':{'attempt_id':{'type':'string','enum':[attempt['attempt_id']]},
                                'exit_code':{'type':'integer'}, 'output':{'type':'string'},
                                'stderr':{'type':'string'}, 'extra_rejected':{'type':'boolean'},
                                'host_visible':{'type':'boolean'}},
                  'required':['attempt_id','exit_code','output','stderr','extra_rejected','host_visible']}
        configuration = self.flow.directory / 'claude-config'
        configuration.mkdir()
        before = manifest(Path(attempt['workspace']))
        with patch.dict(os.environ, {'CLAUDE_CONFIG_DIR':str(configuration)}):
            result = AgentCLI(backend='claude', executable=str(fake), model=attempt['model']).run(
                attempt=attempt, prompt=build_prompt(self.flow.state, self.flow.lock, attempt), schema=schema)
        self.assertEqual(result['outcome'], 'completed', result)
        response = result['response']
        self.assertEqual(response['exit_code'], 0, response)
        checked = json.loads(response['output'])
        self.assertTrue(checked['raw_passed'], checked)
        self.assertEqual(checked['source_validation']['status'], 'verified')
        self.assertTrue(response['extra_rejected'])
        self.assertTrue(response['host_visible'])
        self.assertEqual(manifest(Path(attempt['workspace'])), before)
        root = Path(attempt['workspace']) / 'root.yaml'
        self.assertIn('META_DATA', root.read_text())
        root.write_text(root.read_text().replace('META_DATA', 'meta_data', 1))
        rejected = self.check(attempt)
        self.assertEqual(rejected['exit_code'], 1, rejected)
        self.assertFalse(rejected['raw_passed'])


if __name__ == '__main__':
    unittest.main()
