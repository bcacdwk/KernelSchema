"""Finite current/index check and native receipt rejection using new temp Publications."""
from pathlib import Path
import hashlib,json,tempfile
from importlib.util import module_from_spec,spec_from_file_location
from kernelschema.artifacts import Publication,Bundle
from kernelschema.contracts import ProtocolError
from kernelschema.extract_bridge import consume_raw
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'schema_discussion/main_integration_20261122'
def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def ref(path):return {'path':path,'sha256':sha(path)}
subject_path='schema_discussion/main_integration_20261122/W2_SUBJECT.json'
review_path='schema_discussion/main_integration_20261122/W2_REVIEW.json'
s=json.loads((ROOT/subject_path).read_text());review=json.loads((ROOT/review_path).read_text())
assert sha(subject_path)=='4936e08c5611c703497c6e53a244ff4e3cf746f993ebff62dce57c60752a2242'
assert sha(review_path)=='af96e504979f3a98cd975f9f82922ad088821ad4a0c13ed0e217ea8d45679255'
assert review['status']=='W2_REVIEW_PASS' and review['subject']==ref(subject_path)
for x in review['evidence']+s['evidence']+s['rules_and_checker']:assert sha(x['path'])==x['sha256'],x['path']
index_path='schema_discussion/extract_schema_review_20261121/CURRENT_SCHEMA_INDEX.json'
index=json.loads((ROOT/index_path).read_text());assert index['operator_count']==9 and index['design_count']==19 and index['runner_p6_for_current_bytes'] is False
rows=[]
with tempfile.TemporaryDirectory(prefix='kernelschema-d-current-') as tmp:
 spec=spec_from_file_location('finite_current_fixture',ROOT/'tests/workflows/fixtures/v1/row_softmax/build.py');module=module_from_spec(spec);spec.loader.exec_module(module)
 fixture=module.build_bundle(Path(tmp)/'bundle')
 for directory,entry in zip(s['directories'],index['operators']):
  path=directory['directory']+'/.extraction/current.json';current=json.loads((ROOT/path).read_text());assert entry['current_metadata']==ref(path)
  assert current['format']=='schema_wiki.current_schema_manifest.v1' and current['directory']==directory['directory']
  assert set(current['files'])==set(directory['files']) and entry['files']==current['files']
  for name,identity in current['files'].items():
   original=directory['files'][name];assert identity=={k:original[k] for k in ('sha256','version','source','source_file_sha256') if k in original}
   assert sha(original['path'])==identity['sha256']
   if name.endswith('.yaml'):
    assert current['dependencies'][name]==dict(original['dependency_closure'],explicit_from=original['explicit_from'])
  assert set(current['dependencies'])=={x for x in current['files'] if x.endswith('.yaml')}
  for locator,source in current['fixed_sources'].items():assert locator==source['locator'] and sha(source['path'])==source['sha256']
  acceptance=current['acceptance'];assert acceptance['runner_p6_receipt'] is False and acceptance['semantic_evidence']==ref(review_path) and acceptance['review_subject']==ref(subject_path)
  assert acceptance['raw_bytes_equal_checked_preview'] is True and acceptance['structure_check']['exit_code']==0 and acceptance['structure_check']['skip_hardware'] is True
  check=acceptance['structure_check']['output'];assert sha(check['path'])==check['sha256']
  receipt=current['historical_receipt'];assert sha(receipt['path'])==receipt['sha256']==directory['historical_receipt']['sha256']
  assert entry['historical_receipt']==receipt and entry['semantic_evidence']==ref(review_path) and entry['review_subject']==ref(subject_path)
  folder=ROOT/directory['directory']; before={name:sha(directory['directory']+'/'+name) for name in directory['files']}
  rejected=Publication(Path(tmp)/directory['operator_id']/'reject');task=fixture['task']
  try: consume_raw(rejected,folder,task=task,origin='extract',receipt=ROOT/receipt['path'])
  except ProtocolError as error: assert 'native receipt dependency mismatch' in str(error);reason=str(error)
  else: raise AssertionError('old receipt incorrectly accepted new bytes')
  accepted=Publication(fixture['root']);task=fixture['task']
  imported=consume_raw(accepted,folder,task=task,origin='curated',receipt=None);accepted.commit([imported['schema']],subjects=[task]);assert imported['receipt'] is None
  copied=Bundle(accepted.root).document(imported['schema']);assert len(copied['raw'])==entry['design_count']
  wrong=Publication(Path(tmp)/directory['operator_id']/'wrong-current');task=fixture['task']
  try:consume_raw(wrong,folder,task=task,origin='curated',receipt=ROOT/path)
  except ProtocolError as error:assert 'expected original schema_extract.delivery.v2 receipt' in str(error)
  else:raise AssertionError('current incorrectly accepted as P6 receipt')
  assert before=={name:sha(directory['directory']+'/'+name) for name in directory['files']}
  rows.append({'current':ref(path),'operator_id':directory['operator_id'],'design_count':entry['design_count'],'complete_inventory_dependencies_sources_evidence':'passed','old_receipt_rejected':reason,'curated_without_receipt':'accepted','current_as_receipt':'rejected','raw_source_unchanged':True})
result={'format':'kernelschema.integration.current_verification.v1','status':'passed','subject':ref(subject_path),'review':ref(review_path),'index':ref(index_path),'currents':rows,'counts':{'currents':9,'yaml':19,'dispatch':9,'old_receipt_rejections':9,'curated_acceptances':9,'current_as_receipt_rejections':9},'limits':['Read-only current/source verification; all Publications in new temporary directories.','Current is ordinary provenance; no new P6 or API implement gate.','No model/GPU/Extract/monitor calls.']}
(BASE/'round2_d/current_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result['counts']))
