"""Current document fixtures and the shared registered reviewer protocol."""

from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from schema_extract.tools.common import ExtractError, digest, read_json
from schema_extract.tools.review_eval import cases_from_suite, evaluate, grade_response
from schema_extract.tools.stages import DOCUMENT_CONCLUSIONS


REPO = Path(__file__).resolve().parents[3]
FIXTURE = Path(__file__).resolve().parent / 'fixtures/review_eval'


class ReviewEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='review-eval-')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.suite = self.directory / 'suite.json'
        for name in ('base.yaml', 'reference.py', 'source.md', 'suite.json'):
            shutil.copyfile(FIXTURE / name, self.directory / name)

    def response(self, attempt_id, *, decision='approve'):
        return {'attempt_id':attempt_id, 'decision':decision,
                'conclusions':[{'subject':key,'text':'Current document assessed'} for key in DOCUMENT_CONCLUSIONS],
                'issues':[] if decision=='approve' else [
                    {'criterion':'document_consistency','path':'DESIGN.Work_schedule.Thesis','detail':'The explanation changes the copy result'}]}

    def test_grading_requires_current_nonce_coverage_and_exact_field_boundary(self):
        attempt = str(uuid4())
        _, cases = cases_from_suite(self.suite)
        self.assertIn('META_DATA', cases['scalar_copy'])
        self.assertNotEqual(cases['scalar_copy']['DESIGN']['Work_schedule']['Thesis'], cases['contradictory_schedule']['DESIGN']['Work_schedule']['Thesis'])
        good = {'decision':'approve','criteria':[],'path':''}
        bad = {'decision':'revise','criteria':['document_consistency'],'path':'DESIGN.Work_schedule.Thesis'}
        response = self.response(attempt)
        self.assertTrue(grade_response(response,good,attempt_id=attempt)['passed'])
        self.assertFalse(grade_response(response,bad,attempt_id=attempt)['passed'])
        self.assertFalse(grade_response(response,good,attempt_id=str(uuid4()))['passed'])
        self.assertFalse(grade_response({**response,'conclusions':[]},good,attempt_id=attempt)['passed'])
        response = self.response(attempt,decision='revise')
        self.assertTrue(grade_response(response,bad,attempt_id=attempt)['passed'])
        response['issues'][0]['path'] = 'META_DATA.Describe'
        self.assertFalse(grade_response(response,bad,attempt_id=attempt)['passed'])

    def test_evaluation_uses_registered_agent_backend_and_independent_read_only_scopes(self):
        calls, constructors = [], []
        test = self
        class RecordedAgent:
            def __init__(self, **kwargs):
                constructors.append(kwargs)
            def run(self, *, attempt, prompt, schema, timeout):
                calls.append(deepcopy(attempt))
                test.assertEqual(schema['properties']['attempt_id']['enum'],[attempt['attempt_id']])
                test.assertIn(attempt['workspace'], prompt)
                test.assertIn('provided_documents',prompt)
                decision = 'approve' if attempt['node']=='scalar_copy' else 'revise'
                return {'outcome':'completed','backend':attempt['backend'],
                        'response':test.response(attempt['attempt_id'],decision=decision),
                        'termination':{'confirmed':True}, 'session_id':str(uuid4()),
                        'session_status':'ready','requested_model':attempt['model'],'requested_effort':attempt['effort']}
        for backend in ('claude','codex'):
            options = {'backend':backend,'agent_binary':'/usr/bin/true'}
            if backend=='codex':
                options.update(model='explicit-codex-model',effort='high')
            with patch('schema_extract.tools.review_eval.AgentCLI', RecordedAgent):
                report=evaluate(self.suite,self.directory/('output-'+backend),REPO,**options)
            self.assertTrue(report['passed'],report)
            self.assertEqual(report['config']['model'],'qwen3.8-max[1m]' if backend=='claude' else 'explicit-codex-model')
            self.assertEqual(report['config']['effort'],'max' if backend=='claude' else 'high')
        self.assertEqual([item['backend'] for item in constructors],['claude','codex'])
        self.assertEqual(len({item['session_directory'] for item in calls}),4)
        self.assertTrue(all(item['writable_files']==[] and item['session_id'] is None for item in calls))
        for attempt in calls:
            self.assertEqual(read_json(Path(attempt['directory'])/'attempt.json'),attempt)
            self.assertEqual(attempt['session_scope']['role'],'reviewer')

    def test_codex_does_not_receive_implicit_qwen_or_max(self):
        with self.assertRaisesRegex(ExtractError,'explicit model and effort'):
            evaluate(self.suite,self.directory/'missing',REPO,backend='codex',agent_binary='/usr/bin/true')
        self.assertFalse((self.directory/'missing').exists())


if __name__ == '__main__':
    unittest.main()
