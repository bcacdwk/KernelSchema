"""Poll only this verification's jobs and collect their declared artifacts."""
import json
import os
from pathlib import Path
import subprocess
import tarfile
import urllib.request

root=Path(__file__).resolve().parent
os.environ['AGATE_NO_UPDATE_CHECK']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['TMPDIR']=str(root/'tmp')

def artifact_urls(obj):
    if isinstance(obj,dict):
        for key,value in obj.items():
            if key in ('url','download_url') and isinstance(value,str) and value.startswith('https://'):
                yield value
            else:
                yield from artifact_urls(value)
    elif isinstance(obj,list):
        for value in obj:
            yield from artifact_urls(value)

for submission in sorted((root/'results').glob('*.submission.json')):
    name=submission.name.removesuffix('.submission.json')
    job=json.loads(submission.read_text())['job_id']
    output=root/'results'/f'{name}.job.json'
    previous=json.loads(output.read_text()) if output.exists() else {}
    if previous.get('status') in ('succeeded','failed','cancelled','timed_out'):
        data=previous
    else:
        result=subprocess.run(['agate','get',job],capture_output=True,text=True)
        data=json.loads(result.stdout)
        output.write_text(result.stdout)
    print(name,job,data['status'],data.get('execution_phase'),round((data.get('phase_elapsed_ms') or 0)/1000),flush=True)
    if data['status'] in ('succeeded','failed','cancelled','timed_out'):
        target=root/'results'/name
        target.mkdir(exist_ok=True)
        if name!='gdn_agate_eval' and not (target/'results.tar.gz').exists():
            for url in artifact_urls(data.get('result')):
                tar_path=target/'results.tar.gz'
                urllib.request.urlretrieve(url,tar_path)
                with tarfile.open(tar_path) as archive:
                    archive.extractall(target,filter='data')
                print('collected',name,flush=True)
                break
        result=data.get('result') or {}
        print('result keys:',list(result),'error:',data.get('error'),flush=True)
        for key in (() if previous.get('status')==data['status'] else ('stdout','stderr','output')):
            if key in result:
                print(key,str(result[key])[-2200:],flush=True)
