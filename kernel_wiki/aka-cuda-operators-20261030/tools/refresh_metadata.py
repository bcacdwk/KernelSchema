#!/usr/bin/env python3
"""Refresh derived documentation/metadata; never rebind changed candidate bytes."""
import ast
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from library_metadata import build_summary, extract_interface, read, sha

ROOT = Path(__file__).resolve().parents[1]
STATUSES = {'passed_sampled': '采样验收通过', 'certified': '原包认证通过',
            'accepted_by_user_with_performance_waiver': '人工接受性能例外'}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def dependencies(folder):
    imports = set()
    for path in (folder / 'candidate').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imports.update(x.name.split('.')[0] for x in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                imports.add(node.module.split('.')[0])
    # Locally compiled extension module names are not pip dependencies.
    return {'python': sorted(imports & {'torch', 'tvm_ffi', 'filelock'}),
            'toolchain': ['CUDA Toolkit', 'C++ compiler'],
            'authority': 'candidate/solution.py and its local build modules'}


def render_readme(folder, entry, summary):
    registry = Path(os.path.relpath(ROOT / 'MANIFEST.json', folder)).as_posix()
    convention = Path(os.path.relpath(ROOT / 'CONVENTIONS.md', folder)).as_posix()
    notes = Path(os.path.relpath(ROOT / 'PROBLEM_NOTES.md', folder)).as_posix()
    initializer = ('[prepare_model](problem/input.py) 钩子' if entry['interface']['prepare_model']
                   else '`Model(**init_kwargs)`')
    metric = 'NCU kernel-sum' if entry['performance']['metric'] == 'ncu_kernel_sum' else '完整 forward CUDA event'
    problem_status = (f'[problem 待最终回执核验]({notes})'
                      if entry['problem_requires_latest_receipt_confirmation'] else 'problem 已核对')
    acceptance_record = Path(entry['performance']['record']).relative_to(entry['directory']).as_posix()
    primary = next(r for r in summary['measurements'] if r['id'] == summary['primary_measurement'])
    record = 'performance/' + primary['source_pointer'].split('#', 1)[0]
    supplement = []
    for name in ['PRECISION.md', 'VARIANTS.md', 'USER_ACCEPTANCE.md']:
        if (folder / name).is_file():
            supplement.append(f'[{name}]({name})')
    if (folder / 'LICENSES').is_dir():
        supplement.append('[来源与许可](LICENSES/)')
    else:
        supplement.append('版权／来源声明见源码；原包未附独立 LICENSES')
    docs = ' · '.join(supplement)
    dep = '、'.join(entry['dependencies']['python']) + '；CUDA/C++ 工具链，版本与 flags 见源码构建入口'
    note = ''
    if entry['operator'] == 'paged_latent_mxfp8_nvfp4':
        note = '| 源码约束 | `kernel_mma.cuh` 在两个 namespace 内重复包含，保留其无 include guard 的组织。 |\n'
    return f'''# {entry['operator']}

{entry['display_name']}。

| 项目 | 说明 |
|---|---|
| 来源 | **{entry['solution_label']}** |
| 入口 | [candidate/solution.py](candidate/solution.py):`Model.forward`；参数及别名见[接口清单]({registry})。 |
| 初始化 | {initializer}；整个 [candidate/](candidate/) 一起复用。 |
| 题目 | [合同](problem/metadata.json) · [输入生成](problem/input.py) · [正确性参考](problem/reference.py)；`problem/solution.py` 是生产对照。 |
| 范围 | [固定用例](problem/shape_cases.json) · [支持范围](problem/shape_range.json)；容差、输出和状态约束以合同为准。 |
| 依赖 | {dep}。 |
| 性能 | [SUMMARY.json](performance/SUMMARY.json)，{metric}（µs）；[原始测量]({record})。 |
| 状态 | [{STATUSES[summary['acceptance_status']]}]({acceptance_record})；{problem_status}。 |
| 补充 | {docs}。 |
{note}
[统一规范]({convention})
'''


def main():
    manifest = read(ROOT / 'MANIFEST.json')
    for entry in manifest['operators']:
        folder = ROOT / entry['directory']
        for directory, key in [('candidate', 'candidate_files'), ('problem', 'problem_files')]:
            actual = {p.relative_to(folder / directory).as_posix(): sha(p)
                      for p in (folder / directory).rglob('*') if p.is_file()}
            if actual != entry[key]:
                raise ValueError(f'{entry["operator"]}: {directory} changed; obtain matching verification first')
        entry['interface'] = extract_interface(folder)
        entry['dependencies'] = dependencies(folder)
        summary = build_summary(folder, entry)
        write(folder / 'performance/SUMMARY.json', summary)
        primary = next(r for r in summary['measurements'] if r['id'] == summary['primary_measurement'])
        entry['performance'] = {
            'summary': entry['directory'] + '/performance/SUMMARY.json',
            'metric': primary['metric'], 'status': summary['acceptance_status'],
            'record': entry['directory'] + '/performance/' +
                      ('CERTIFIED_BEST.json' if entry['solution_origin'] == 'AKA' else 'ACCEPTANCE.json'),
        }
        (folder / 'README.md').write_text(render_readme(folder, entry, summary))
    manifest['schema'] = 'aka.cuda_operator_library.v3'
    manifest['conventions'] = 'CONVENTIONS.md'
    manifest['performance_summary_schema'] = 'aka.performance_summary.v1'
    write(ROOT / 'MANIFEST.json', manifest)
    homepage = '''# CUDA 算子题库：44 题

六类算子，20 个 AKA 解、24 个参考实现。每题的 candidate、problem、performance 放在一起。

| 分类 | 题数 |
|---|---:|
'''
    for category in manifest['categories']:
        homepage += f"| [{category['title']}]({category['directory']}/README.md) | {category['operator_count']} |\n"
    homepage += '''
入口统一为 `candidate/solution.py:Model.forward`；具体参数、初始化钩子见 [MANIFEST.json](MANIFEST.json) 的 interface。problem/solution.py 是生产对照，problem/reference.py 是正确性参考。

每题先读 README，再按需查看 problem 合同和 performance/SUMMARY.json。摘要保留计时方法、对照身份、缺测和人工例外；原始性能证据保持原样。

[统一规范](CONVENTIONS.md) · [problem 对应证据](PROBLEM_AUDIT.json) · [三题待核验备注](PROBLEM_NOTES.md)

检查：`python3 tools/verify.py`；元数据转换测试：`python3 tools/test_metadata.py`。本次规范化未重跑 GPU 评测；通用评测执行器由运行环境提供。
'''
    (ROOT / 'README.md').write_text(homepage)
    checksum = ROOT / 'CHECKSUMS.sha256'
    checksum.write_text(''.join(f'{sha(p)}  {p.relative_to(ROOT).as_posix()}\n'
                               for p in sorted(ROOT.rglob('*')) if p.is_file() and p != checksum))
    print('Refreshed 44 interfaces, 44 performance summaries, 44 concise READMEs and checksums.')


if __name__ == '__main__':
    main()
