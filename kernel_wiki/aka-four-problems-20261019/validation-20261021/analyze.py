"""Join ordered NCU samples to frozen wiki evidence, without dropping kernels."""
import csv
import io
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
WIKI=ROOT.parent

def gm(values):
    values=list(values)
    return math.exp(statistics.mean(map(math.log,values)))

def csv_rows(path):
    return list(csv.DictReader(path.open()))

def history(op):
    p=WIKI/'evidence'/op
    if op=='fa4_prefill':
        return {r['shape']:dict(production_us=float(r['production_sum_us']),aka_us=float(r['candidate_sum_us']),production_count=int(r['production_launches']),aka_count=int(r['candidate_launches'])) for r in csv_rows(p/'per_shape.csv')}
    if op=='fa4_decode':
        return {r['shape']:dict(production_us=float(r['production_kernel_sum_us']),aka_us=float(r['qoder_kernel_sum_us']),production_count=int(r['production_launches']),aka_count=int(r['qoder_launches'])) for r in csv_rows(p/'per_shape_kernel_sum.csv')}
    if op=='qsa_prefill':
        return {r['shape_id']:dict(production_us=float(r['production_kernel_sum_us']),aka_us=float(r['aka_kernel_sum_us']),production_count=1,aka_count=2) for r in csv_rows(p/'per_shape.csv')}
    d=json.loads((p/'gdn_summary.json').read_text())['per_shape']
    return {k:dict(production_us=v['prod_kernel_us_sum'],aka_us=v['aka_kernel_us_sum'],production_count=v['prod_kernel_count'],aka_count=v['aka_kernel_count']) for k,v in d.items()}

def ncu_kernels(path):
    text=path.read_text()
    begin=text.index('"ID","Process ID"')
    rows=list(csv.DictReader(io.StringIO(text[begin:])))
    kernels={}
    for row in rows:
        if not row.get('ID','').isdigit():
            continue
        kid=int(row['ID'])
        kernel=kernels.setdefault(kid,dict(id=kid,name=row['Kernel Name']))
        value=float(row['Metric Value'].replace(',',''))
        if row['Metric Name']=='gpu__time_duration.sum':
            kernel['us']=value*{'ns':0.001,'us':1,'ms':1000,'s':1e6}[row['Metric Unit']]
        elif row['Metric Name']=='gpc__cycles_elapsed.avg.per_second':
            kernel['ghz']=value/1e9
    return [kernels[k] for k in sorted(kernels)]

def analyze(op, result_name=None, validation_name=None):
    result_name=result_name or op
    validation_name=validation_name or result_name
    folder=ROOT/'results'/result_name
    hist=history(op)
    val=json.loads((ROOT/'results'/validation_name/'validate.json').read_text())
    prof=json.loads((folder/'profile.json').read_text())
    kernels=ncu_kernels(folder/'ncu.csv')
    expected_count=sum(hist[s['shape_id']][s['side']+'_count'] for s in prof['profile_sequence'])
    if len(kernels)!=expected_count:
        raise ValueError(f'{op}: expected {expected_count} launches; observed {len(kernels)}. Inspect all launches before attributing.')
    offset=0
    samples={}
    launches=[]
    for step in prof['profile_sequence']:
        sid,side=step['shape_id'],step['side']
        count=hist[sid][side+'_count']
        group=kernels[offset:offset+count]
        offset+=count
        names=' '.join(k['name'] for k in group)
        if op.startswith('fa4_'):
            assert ('fmhaSm' in names)==(side=='production'), (op,step,names)
        elif op=='qsa_prefill':
            assert ('qsa_sparse_paged' in names)==(side=='production'), (op,step,names)
        samples.setdefault(sid,{}).setdefault(side,[]).append(sum(k['us'] for k in group))
        for k in group:
            launches.append(dict(**step,**k))
    assert offset==len(kernels)
    records={r['shape_id']:r for r in val['records']}
    per_shape=[]
    for sid,side_samples in samples.items():
        r=dict(shape_id=sid,correctness_passed=records[sid]['passed'])
        for side in ['production','aka']:
            times=side_samples[side]
            r[side+'_us']=statistics.median(times)
            r[side+'_samples_us']=times
            r[side+'_cv']=statistics.stdev(times)/statistics.mean(times)
            r['wiki_'+side+'_us']=hist[sid][side+'_us']
            r[side+'_eager_us']=records[sid]['eager'][side]['median_us']
        r['speedup']=r['production_us']/r['aka_us']
        r['wiki_speedup']=r['wiki_production_us']/r['wiki_aka_us']
        r['aka_delta_pct']=(r['aka_us']/r['wiki_aka_us']-1)*100
        r['speedup_delta_pct']=(r['speedup']/r['wiki_speedup']-1)*100
        per_shape.append(r)
    ids=json.loads((ROOT/'bundles'/op/'experiment_shape_ids.json').read_text())
    def aggregate(rows):
        out=dict(shapes=len(rows),passed=sum(r['correctness_passed'] for r in rows),aka_faster_shapes=sum(r['speedup']>1 for r in rows))
        for key in ['production_us','aka_us','wiki_production_us','wiki_aka_us','speedup','wiki_speedup','production_eager_us','aka_eager_us']:
            out[key]=gm(r[key] for r in rows)
        out['aka_delta_pct']=(out['aka_us']/out['wiki_aka_us']-1)*100
        out['speedup_delta_pct']=(out['speedup']/out['wiki_speedup']-1)*100
        out['max_ncu_cv']=max(r[side+'_cv'] for r in rows for side in ('aka','production'))
        return out
    by_side={}
    for side in ('aka','production'):
        by_side[side]=dict(cases=sum(len(r['cases']) for r in val['records']),passed_cases=sum(c['sides'][side]['passed'] for r in val['records'] for c in r['cases']),max_abs=max(v['max_abs'] for r in val['records'] for c in r['cases'] for v in c['sides'][side]['outputs'].values()),max_tolerance_ratio=max(v['max_tolerance_ratio'] for r in val['records'] for c in r['cases'] for v in c['sides'][side]['outputs'].values()),ncu_median_gpc_ghz=statistics.median(k['ghz'] for k in launches if k['side']==side))
    summary=dict(op=op,job_id=json.loads((ROOT/'results'/f'{result_name}.submission.json').read_text())['job_id'],validation_job_id=json.loads((ROOT/'results'/f'{validation_name}.submission.json').read_text())['job_id'],wiki_set=aggregate(per_shape),current_experiment_set=aggregate([r for r in per_shape if r['shape_id'] in ids]),correctness=by_side,environment=val['environment'],profile_environment=prof['environment'],per_shape=per_shape)
    (folder/'summary.json').write_text(json.dumps(summary,indent=2))
    with (folder/'per_shape.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_shape[0]));writer.writeheader();writer.writerows(per_shape)
    with (folder/'kernel_launches.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(launches[0]));writer.writeheader();writer.writerows(launches)
    return summary

if __name__=='__main__':
    all_results={}
    for op in ('fa4_prefill','fa4_decode','qsa_prefill','gdn_prefill'):
        if not (ROOT/'results'/op/'ncu.csv').exists():
            continue
        try:
            result=analyze(op)
            all_results[op]=result
            print(op,json.dumps(result['wiki_set']))
        except Exception as e:
            print(op,'ANALYSIS_ERROR',repr(e))
    (ROOT/'summary.json').write_text(json.dumps(all_results,indent=2))
    for op,result_name,validation_name in [('gdn_prefill','gdn_clock_base','gdn_prefill'),('fa4_prefill','fa4_prefill_v0618',None),('fa4_decode','fa4_decode_v0618',None)]:
        if (ROOT/'results'/result_name/'ncu.csv').exists():
            result=analyze(op,result_name,validation_name)
            print(result_name,json.dumps(result['wiki_set']))
