#!/usr/bin/env python3
"""Join the two focused reviews with explicit supervisor routing; verify evidence links.

This does not reclassify text groups or modify the original census/annotations.
Run from any directory. Outputs are restricted to this gap_review directory.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
REPO = BASE.parents[2]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


# These 50 findings were screened by the specialists but outside their detailed
# remit. The supervisor reviewed their original claims/alternatives against DOC0–6.
MANUAL = {}


def route(op, numbers, disposition, mechanism, reason):
    for number in numbers.split():
        fid = f'{op}-F{int(number):02d}'
        assert fid not in MANUAL, fid
        MANUAL[fid] = {
            'disposition': disposition, 'mechanism': mechanism, 'reason': reason,
        }


route('template_gdn_prefill', '1 2', 'no_gap', 'summary_or_formal_definition',
      '标题/释义和正式数值定义分别处理；未找到需要新关系的反例，不能整字段按长度删除。')
route('template_gdn_prefill', '4', 'no_gap', 'task_environment_definition',
      '输入生成/初态可用现行关系表达；CUDA布局与计时范围是需短写一次的环境事实，无新增固定字段的证据。')
route('template_gdn_prefill', '6', 'existing_structure', 'parameter_expression',
      'P已有确定数量公式，可直接使用worker_count数值表达式；不是缺参数字段。')
route('template_gdn_prefill', '16', 'no_gap', 'hardware_definition',
      '设备能力及实现栈需要定义；wiki pending须保留，不把资料不足当流水线关系缺失。')
route('template_fa4_prefill', '3 4', 'no_gap', 'summary_or_redundancy',
      '摘要与已重复的range可短写；不产生新的执行关系或对象语义。')
route('template_fa4_prefill', '5', 'representation_cost', 'cross_node_predicate_reuse',
      '11个when需独立成立，共同派生式无跨节点引用；可表达但复制，不能直接继承父when。')
route('template_fa4_prefill', '6', 'no_gap', 'hardware_definition',
      '硬件组成/机制属于既有开放定义；补资料或短定义，不据此扩动作字段。')
route('template_fa4_prefill', '8', 'existing_structure', 'whole_region_reuse',
      '七个已核对的完整math父子对可按DOC0使用from_parent；局部不同区域不在此结论内。')
route('template_fa4_prefill', '9', 'no_gap', 'formal_schedule_algorithm',
      'LPT、非单调折叠与prefix解析须有确定算法/映射定义；现有数学及strategy允许短关系，不因无专属排序字段扩语言。')
route('template_fa4_decode', '1 2', 'no_gap', 'summary_or_formal_definition',
      '短摘要、when谓词与已定义参数应按用途保留或精简；没有新的关系表达缺口。')
route('template_fa4_decode', '12', 'no_gap', 'hardware_definition',
      '容量、设备/库身份为已有定义与绑定；pending资料不当作已经落实的共享语义。')
route('template_fa4_decode', '15', 'existing_structure', 'whole_region_reuse',
      'q1_global与直接父的完整math相同且binding在子环境解析，可用现行整区引用。')
route('template_qsa_prefill', '2 4', 'no_gap', 'summary_or_formal_definition',
      '区分摘要复述与公式/符号/成本关系；没有新增固定字段需求。')
route('template_qsa_prefill', '3', 'no_gap', 'hardware_definition',
      '硬件能力和可见性是规范允许的环境定义；本路径无用的cluster背景可删。')
route('template_qsa_prefill', '13', 'source_or_uncertain', 'source_to_domain_mismatch',
      'out=None的来源域需确认；nullable已经存在，缺填或域未核实不是缺语言。')
route('extract_gdn', '2 4', 'no_gap', 'summary_or_formal_definition',
      '公式求值与正式引用有语义，释义摘要可短写；不将字符串统计当新字段证据。')
route('extract_gdn', '15', 'source_or_uncertain', 'source_and_launch_coverage',
      '非连续复制提交及copyback存在来源/覆盖问题；已有launch能记录实际计算，不能凭猜测增加前缀。')
route('extract_gdn', '18', 'representation_cost', 'partial_definition_reuse',
      '跨文件只有整区from_parent且本树三区有真实差异；局部定义复制是成本，非核心关系不可表达。')
route('extract_gemm', '1', 'shared_semantic_definition', 'external_api_identity',
      '公式与归约已有载体；对象身份/tuple次序需一次短外部定义，与其他算子ABI机制合并，不据此增加专属字段。')
route('extract_gemm', '5', 'existing_structure', 'work_distribution',
      '分块数、静态步长与归属已有partition/assignment；ragged映射的算法定义另保留。')
route('extract_gemm', '14', 'no_gap', 'summary_or_redundancy',
      '短标题、reason与dispatch导航不独占执行关系；必填项不能任意删掉。')
route('extract_gemm', '16', 'source_or_uncertain', 'permitted_hardware_placeholder',
      '提取允许硬件占位；这是结论边界，不是违规或需要新增字段。')
route('extract_mla', '1 2 7', 'no_gap', 'summary_or_formal_definition',
      '标题、形式类型/域与定量ledger分别处理；不将所有字符串视为可删解释。')
route('extract_mla', '4', 'source_or_uncertain', 'source_limit',
      '比较器、实现拒绝和源缺陷需短说明，新增字段无法补来源事实。')
route('extract_mla', '16', 'existing_structure', 'whole_region_binding',
      'bootstrap的数学provider可在本地调度/流水线重新绑定；无需新增继承或复制定义。')
route('extract_sparse', '1 2', 'no_gap', 'summary_or_formal_definition',
      '公式、谓词、成本及正式ID与摘要分开；未构成字段缺失证据。')
route('extract_sparse', '3', 'existing_structure', 'work_distribution',
      '七个strategy已有工作排列结构完全覆盖；其余独占索引/参数继续保留。')
route('extract_sparse', '4', 'representation_cost', 'partial_definition_reuse',
      '文档内alias可复用，跨文件整区引用不足以复用局部packed phase；这是复制成本。')
route('extract_sparse', '18', 'source_or_uncertain', 'permitted_hardware_placeholder',
      'DOC2允许当前提取的硬件占位；限制硬件推论，不判提取违规。')
route('extract_fmha', '1 2 7', 'no_gap', 'summary_or_formal_definition',
      '公式/接口、正式ledger关系及短摘要是不同用途；字段面积不能证明缺语言。')
route('extract_fmha', '5', 'existing_structure', 'artifact_omission',
      '五份when为空是草稿缺项；已有when足够，必须补真实域而非新语言。')
route('extract_fmha', '11', 'source_or_uncertain', 'permitted_hardware_placeholder',
      '冻结DOC2明确允许--skip-hardware；保留限制但不据此新增字段或要求本轮补硬件。')
route('extract_fmha', '22', 'existing_structure', 'work_distribution',
      '自然worktile排列和单CTA归属已表达物理grid翻译；冗余说明可短写。')
route('extract_moe', '1 2 3', 'no_gap', 'summary_or_formal_definition',
      '同名describe符合任务身份规则；公式/谓词是形式内容，摘要可压缩但不能按重复次数整删。')
route('extract_moe', '14 19 20 21', 'existing_structure', 'existing_schedule_and_value_relations',
      '已有partition/assignment/launch、math和bounded pipe提供对应定义；保留独有服务绑定，删除复述不需新字段。')


def main():
    originals = read(BASE / 'data/findings_catalog.json')
    action_doc = read(HERE / 'action_findings.json')
    boundary_doc = read(HERE / 'boundary_findings.json')
    action = {x['finding_id']: x for x in action_doc['findings']}
    boundary = {x['finding_id']: x for x in boundary_doc['findings']}
    ids = {x['id'] for x in originals}
    assert len(ids) == len(originals) == 182
    assert len(action) == len(action_doc['findings']) == 182 and set(action) == ids
    assert len(boundary) == len(boundary_doc['findings']) and set(boundary) <= ids
    uncovered = {i for i in ids if i not in boundary and not action[i]['in_action_scope']}
    assert set(MANUAL) == uncovered, (sorted(set(MANUAL) - uncovered), sorted(uncovered - set(MANUAL)))
    results = []
    for original in originals:
        fid = original['id']
        reviews = []
        if action[fid]['in_action_scope']:
            a = action[fid]
            reviews.append({'file': 'action_findings.json', 'disposition': a['disposition'],
                            'scope': a['review_scope']})
        if fid in boundary:
            b = boundary[fid]
            reviews.append({'file': 'boundary_findings.json', 'disposition': b['decision'],
                            'scope': '调用/外部身份/状态相关子事实'})
            disposition = b['decision']
            mechanism = b['mechanism_groups']
            reason = b['supporting_reason']
            fact = b.get('minimum_fact', b['minimum_fact_from_audit'])
            deletion = b.get('deletion_counterexample', b['deletion_counterexample_from_audit'])
            primary = 'boundary_findings.json'
        elif action[fid]['in_action_scope']:
            a = action[fid]
            disposition, mechanism, reason = a['disposition'], [a['mechanism']], a['reason']
            fact, deletion = a['core_fact'], a['deletion_counterexample']
            primary = 'action_findings.json'
        else:
            m = MANUAL[fid]
            disposition, mechanism, reason = m['disposition'], [m['mechanism']], m['reason']
            fact, deletion = original['core_fact'], original['deletion_counterexample']
            primary = 'build_triage.py::MANUAL'
            reviews.append({'file': primary, 'disposition': disposition,
                            'scope': '原审计事实/结构替代对照DOC0–6；未重做全量文本标注'})
        # The boundary review confirms missing current endpoints, not that a new
        # fixed field is necessary: a shared boundary convention remains possible.
        if disposition == 'confirmed_structural_gap':
            disposition = 'uncovered_boundary_relation'
        results.append({
            'id': fid, 'operator': original['operator'],
            'previous_kind': original['kind'], 'previous_gap_kind': original['gap_kind'],
            'in_previous_candidate_pool': original['gap_kind'] != 'none',
            'disposition': disposition, 'mechanisms': mechanism, 'reason': reason,
            'minimum_fact_for_reviewed_scope': fact,
            'deletion_counterexample_for_reviewed_scope': deletion,
            'new_fixed_field_required': 'not_proven',
            'review_source': primary, 'reviews': reviews,
            'original_record': f'../data/findings_catalog.json::{fid}',
            'original_annotation': f'../annotations/{original["operator"]}.json',
            'qualification': '主判为议题分流，不表示finding所涉整个字段同类、可整删或全部为独占核心。',
        })
    candidates = [r for r in results if r['in_previous_candidate_pool']]
    six = [r for r in results if r['previous_gap_kind'] == 'missing_field_or_event']
    assert len(candidates) == 79 and len(six) == 6
    assert {r['id'] for r in results if r['disposition'] == 'uncovered_boundary_relation'} == {
        'extract_mla-F18', 'extract_sparse-F05'}
    summary = {
        'findings_screened': len(results), 'operators': len({r['operator'] for r in results}),
        'previous_candidate_pool': len(candidates),
        'previous_gap_kind_counts': dict(Counter(r['previous_gap_kind'] for r in results)),
        'routing_counts_all_findings': dict(Counter(r['disposition'] for r in results)),
        'routing_counts_previous_candidate_pool': dict(Counter(r['disposition'] for r in candidates)),
        'six_original_missing_candidates': {r['id']: r['disposition'] for r in six},
        'specialist_action_in_scope': sum(x['in_action_scope'] for x in action.values()),
        'specialist_boundary_in_scope': len(boundary), 'supervisor_only_records': len(MANUAL),
        'new_fixed_field_necessity_proven': 0,
        'qualifications': [
            '两条端点缺失不等于两项必须新增字段；统一边界约定尚未被排除。',
            '0表示尚未证明新增字段不可避免，不表示当前结构语义完备。',
            '各finding可覆盖同一文本组；分流数量不能换算成缺口字符量或占比。',
            '本轮没有重新将5875文本组划分成用户四类；原全量标注保持原样。',
        ],
    }
    write('triage.json', {
        'version': 1, 'date': '2026-11-10', 'summary': summary,
        'disposition_semantics': {
            'uncovered_boundary_relation': '现行显式关系没有对应端点；补共享约定或新增端点仍待完整样例比较。',
            'shared_semantic_definition': '必要语义宜短写/共用；尚未证明必须新增固定字段。',
            'existing_structure': '对应子事实已有结构/合法关系，或只是现稿缺填；完整替代仍需语义核对。',
            'representation_cost': '可用既有表达承载，但展开/跨节点复用成本高；不是表达不可能。',
            'scope_decision': '需先明确IR包含的外部状态/预算/host范围；不能把实现全量抄入。',
            'source_or_uncertain': '来源限制、已证源缺陷或依据未闭合；也包括合法硬件占位的结论限制。',
            'no_gap': '该子事实不支持扩语言；不授予删除整个文本组的权限。',
        },
        'supervisor_corrections': [
            'retention以逻辑实例定义，现有MoE已使用keep_value device_module；真正未定的是物理owner及跨请求关系。',
            '属性ref不共享存储；整个对象binding可以引用外部对象，但不会自动建立owner/容量/事件关系。',
            'MLA current metadata已由Contract/math规定；F18的独占反例收窄为外部上传完成到整个main grid启动。',
            '已证端点不存在与已证必须加字段分开；尚无反例排除正确的统一边界约定。',
            'template_gdn_prefill F11通知流身份没有恢复为缺事件。',
        ],
        'review_agents': [
            {'name': '/root/gap_action_review', 'effort': 'high', 'fresh': True},
            {'name': '/root/gap_boundary_review', 'effort': 'high', 'fresh': True},
        ],
        'findings': results,
    })

    errors, inputs, current_rules, links = [], 0, 0, 0
    manifest = read(BASE / 'data/manifest.json')
    for op in manifest['operators'].values():
        records = [(n['snapshot'], n['source_file'], n['sha256'])
                   for n in op['nodes'] if n['material_present']]
        records.append((op['dispatch_snapshot'], op['dispatch_source'], op['dispatch_sha256']))
        for snapshot, original, expected in records:
            inputs += 1
            for path in (REPO / snapshot, REPO / original):
                if not path.is_file() or digest(path) != expected:
                    errors.append(f'input changed/missing: {path}')
    for r in manifest['rules']:
        if digest(REPO / r['snapshot']) != r['sha256']:
            errors.append('frozen rule changed: ' + r['snapshot'])
        if digest(REPO / r['path']) != r['sha256']:
            errors.append('current rule differs: ' + r['path'])
        current_rules += 1
    docs = [BASE / 'SCHEMA_LANGUAGE_GAPS.md', HERE / 'action_review.md', HERE / 'boundary_review.md',
            BASE / 'README.md', BASE.parent / 'README.md']
    for doc in docs:
        for raw in re.findall(r'\]\(([^)]+)\)', doc.read_text()):
            raw = unquote(raw.strip('<>'))
            if '://' in raw or raw.startswith(('mailto:', '#')):
                continue
            raw = raw.split('#', 1)[0]
            match = re.search(r':(\d+)(?:[-–](\d+))?$', raw)
            line = int(match.group(2) or match.group(1)) if match else None
            target = raw[:match.start()] if match else raw
            path = Path(target) if Path(target).is_absolute() else doc.parent / target
            links += 1
            if path.resolve() == (HERE / 'verification.json').resolve():
                continue
            if not path.exists():
                errors.append(f'broken link {doc.name}: {raw}')
            elif line is not None and len(path.read_text().splitlines()) < line:
                errors.append(f'line beyond file {doc.name}: {raw}')
    result = {
        'ok': not errors, 'checked_at': datetime.now(timezone.utc).isoformat(),
        'findings': summary, 'frozen_and_original_inputs_verified': inputs,
        'frozen_and_current_rules_verified': current_rules,
        'markdown_links_checked': links, 'errors': errors,
        'semantic_scope': '关系缺口与文档议题复核；不是完整删减稿的等价性证明，也不是GPU验收。',
        'input_hashes': {p.name: digest(p) for p in [BASE / 'data/findings_catalog.json',
                        HERE / 'action_findings.json', HERE / 'boundary_findings.json']},
        'output_hashes': {p.name: digest(p) for p in [BASE / 'SCHEMA_LANGUAGE_GAPS.md',
                         HERE / 'action_review.md', HERE / 'boundary_review.md', HERE / 'triage.json']},
    }
    write('verification.json', result)
    print(json.dumps({'ok': not errors, 'summary': summary, 'links': links, 'errors': errors},
                     ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
