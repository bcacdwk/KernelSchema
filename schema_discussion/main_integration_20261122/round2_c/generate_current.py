"""Generate nine current manifests only after D has supplied a frozen review SHA."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
SUBJECT_PATH = 'schema_discussion/main_integration_20261122/W2_SUBJECT.json'
SUBJECT_SHA = '4936e08c5611c703497c6e53a244ff4e3cf746f993ebff62dce57c60752a2242'
REVIEW_PATH = 'schema_discussion/main_integration_20261122/W2_REVIEW.json'
INDEX_PATH = 'schema_discussion/extract_schema_review_20261121/CURRENT_SCHEMA_INDEX.json'

def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()

def reference(path):
    return {'path': path, 'sha256': sha(path)}

def dump(path, data):
    (ROOT / path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-sha', required=True,
                        help='Immutable SHA supplied by D together with W2_REVIEW_PASS')
    args = parser.parse_args()
    assert sha(SUBJECT_PATH) == SUBJECT_SHA
    assert sha(REVIEW_PATH) == args.review_sha
    review_text = (ROOT / REVIEW_PATH).read_text()
    json.loads(review_text)
    assert SUBJECT_SHA in review_text
    subject = json.loads((ROOT / SUBJECT_PATH).read_text())
    for record in subject['rules_and_checker'] + subject['evidence']:
        assert sha(record['path']) == record['sha256'], record['path']
    operators = []
    for directory in subject['directories']:
        files, dependencies, fixed_sources = {}, {}, {}
        for name, record in directory['files'].items():
            assert sha(record['path']) == record['sha256']
            files[name] = {key: record[key] for key in ('sha256','version','source','source_file_sha256') if key in record}
            if name.endswith('.yaml'):
                dependencies[name] = dict(record['dependency_closure'], explicit_from=record['explicit_from'])
                for key in ('fixed_source','golden_reference_source'):
                    if key in record:
                        source = record[key]
                        assert sha(source['path']) == source['sha256']
                        fixed_sources[source['locator']] = source
        historical = {key:directory['historical_receipt'][key] for key in ('path','sha256')}
        assert sha(historical['path']) == historical['sha256']
        historical['coverage'] = 'Only the original delivery bytes named by this receipt; not approval of current amendments'
        check = directory['structure_check']
        assert check['exit_code'] == 0 and all(check['raw_bytes_equal_preview'].values())
        assert len(check['raw_bytes_equal_preview']) == sum(name.endswith('.yaml') for name in files)
        assert sha(check['output']['path']) == check['output']['sha256']
        current = {
            'format':'schema_wiki.current_schema_manifest.v1','path_base':'repository_root',
            'operator_id':directory['operator_id'],'directory':directory['directory'],
            'files':files,'dependencies':dependencies,'fixed_sources':fixed_sources,
            'acceptance':{
                'basis':'independent_finite_source_review_and_read_only_structure_check',
                'runner_p6_receipt':False,'semantic_evidence':reference(REVIEW_PATH),
                'review_subject':reference(SUBJECT_PATH),
                'reviewed_yaml_bytes_unchanged':True,'raw_bytes_equal_checked_preview':True,
                'structure_check':{key:check[key] for key in ('command','cwd','exit_code','read_only','skip_hardware','output','dispatch_check')},
                'rules_and_checker':dict(reference(SUBJECT_PATH),json_pointer='/rules_and_checker'),
                'scope_limits':'Finite review of the nine approved YAML and two dispatch amendments and their real inheriting consumers. Existing source/comparator/caller limitations and risks remain. Hardware was skipped; no GPU, all-domain numerical/performance proof, full 33-design audit, or new Runner P6 approval.',
            },
            'historical_receipt':historical,
        }
        path = directory['directory'] + '/.extraction/current.json'
        assert not (ROOT/path).exists(), path
        dump(path,current)
        operators.append({
            'operator_id':current['operator_id'],'directory':current['directory'],
            'design_count':sum(name.endswith('.yaml') for name in current['files']),
            'files':current['files'],'current_metadata':reference(path),
            'semantic_evidence':current['acceptance']['semantic_evidence'],
            'review_subject':current['acceptance']['review_subject'],
            'historical_receipt':current['historical_receipt'],
        })
    assert len(operators) == 9 and sum(op['design_count'] for op in operators) == 19
    index = {
        'format':'schema_wiki.current_schema_index.v1','path_base':'repository_root',
        'operator_count':len(operators),'design_count':sum(op['design_count'] for op in operators),
        'acceptance_basis':'independent_finite_source_review_and_read_only_structure_check',
        'runner_p6_for_current_bytes':False,'hardware_validation':False,'gpu_validation':False,
        'operators':operators,
        'projection_only':'All entries project the nine current manifests; this index grants no additional approval.',
    }
    assert not (ROOT/INDEX_PATH).exists()
    dump(INDEX_PATH,index)
    dump('schema_discussion/main_integration_20261122/round2_c/current_generation.json', {
        'format':'main_integration.w2_current_generation.v1','subject':reference(SUBJECT_PATH),
        'independent_review':reference(REVIEW_PATH),'review_gate':'D supplied W2_REVIEW_PASS with this immutable SHA before generation',
        'generated_currents':[op['current_metadata'] for op in operators],
        'generated_index':reference(INDEX_PATH),'runner_p6_receipt':False,
        'validation':'Subject, rules, evidence, actual directory bytes, fixed sources and original receipts matched all frozen digests immediately before generation',
    })
    print(json.dumps({'currents':len(operators),'yaml':19,'dispatch':9,'index':reference(INDEX_PATH)},ensure_ascii=False))
