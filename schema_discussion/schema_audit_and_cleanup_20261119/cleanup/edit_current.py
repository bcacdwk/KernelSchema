"""Remove construction narrative, retaining current source facts and limitations."""
from pathlib import Path
import json
import hashlib

ROOT = Path(__file__).resolve().parents[3]
CLEAN = Path('schema_discussion/schema_audit_and_cleanup_20261119/cleanup')
OPS = Path('schema_wiki/aka-cuda-operators-20261030/operators')
changes = []


def edit(path, replacements):
    p = ROOT / path
    before = p.read_text()
    value = before
    for old, new in replacements:
        assert old in value, (str(path), old)
        value = value.replace(old, new)
    if value != before:
        p.write_text(value)
        changes.append({'path': str(path), 'before_sha256': hashlib.sha256(before.encode()).hexdigest(),
                        'after_sha256': hashlib.sha256(value.encode()).hexdigest(),
                        'replacements': [{'from': a, 'to': b} for a, b in replacements]})


edit(OPS / 'fmha/fmha_mxfp8_qkv/dispatch.md', [
    ('；含 root 共三份，低于十份上限，root 唯一，没有未计划的草稿、备份 YAML，也没有改名、换扩展名或另藏目录继续交付的额外完整方案。', '。'),
    ('；本次任务要求交付的完整设计是 P2/P4 已定的这三份，已由原提取选定；当前字节另经本轮用户授权的同一 worker/reviewer 人工审修，未产生新 Runner P6 批准；', '；'),
    ('**已知缺口与边界（继承，非阻塞）。**', '**已知缺口与边界。**'),
    ('均为来源报告，未在本轮复现，不证明完整动态域通过验收', '均为来源报告，缺少可复现的完整实验材料，不证明完整动态域通过验收'),
    ('不再宣称 NO_KTINY 的全域 ~1.1 ulp 上界', '没有可由该推导成立的 NO_KTINY 全域 ~1.1 ulp 上界'),
    ('仍是 P1 已批准的占位状态', '仍是未绑定的占位状态'),
    (' 本轮审修记录见 [.extraction/schema-audit-20261119.json](.extraction/schema-audit-20261119.json)；原 [receipt](.extraction/receipt.json) 只证明此前交付字节，原 `invalid_receipt / P6 / published:false` 与全部历史 audit marker 保持原意。', ''),
    ('类型判断（P2 交办）：', '类型依据：'),
    ('本轮未取得该穷举 harness 或复现数据', '固定材料未提供该穷举 harness 或复现数据'),
    ('；缺陷已在 P2 的既有 issue 记录中定位', ''),
])
edit(OPS / 'sparse_attention/dual_cache_latent_bf16/dispatch.md', [
    ('2026-11-19 人工审修及前后字节证据见 [.extraction/schema-audit-20261119.json](.extraction/schema-audit-20261119.json)。原 P6 receipt 只覆盖审修前字节；本轮由同一 xhigh 承担 worker 与 reviewer，未新增 P6。', ''),
])
edit(OPS / 'moe/nvfp4_fused_moe/dispatch.md', [
    ('不在提取范围内', '不属于本目录的实现范围'),
    ('本次不交付完整 schema', '本目录不收录其完整 schema'),
    ('P2 交 P3 定向核实的调用边界已全部核实并落在 root.yaml：', 'root.yaml 的调用与可见性边界：'),
    ('沿用 P1 的两项来源缺口：', '两项来源缺口：'),
    ('本次未独立验证', '现有材料未提供独立验证'),
])
edit(OPS / 'gemm/grouped/fp8_scaled_grouped_gemm/dispatch.md', [
    ('所以本次实际可进入的实现只有', '所以该环境中实际可进入的实现只有'),
    ('因此本次不交付完整 YAML', '因此本目录不收录其完整 YAML'),
    ('**本次收录范围。** 三者分别说明：共同 Contract 的有效域是 P1 已冻结的输入域', '**收录范围。** 共同 Contract 的有效域是固定输入域'),
])
edit(OPS / 'mla/mla_expanded_fp8/dispatch.md', [('硬件区保留来源提取的占位。', '硬件区仍是未绑定的占位。')])
edit(OPS / 'linear_attention/gdn_prefill_bf16_state_bf16/dispatch.md', [
    ('等硬件事实由后续硬件绑定阶段登记，本目录分派条件只引用', '等硬件事实尚未绑定；本目录分派条件只引用'),
])
for root in sorted((ROOT / OPS).rglob('root.yaml')):
    path = root.parent.relative_to(ROOT) / 'dispatch.md'
    heading = (ROOT / path).read_text().splitlines()[0]
    edit(path, [(heading + '\n', heading + '\n\n文件版本、来源与检查信息见[当前元数据](.extraction/current.json)。\n')])

edit(Path('README.md'), [('旧规范仅在追溯时查阅 `schema_docs/archived`；维护规范或检查器时，再读', '维护规范或检查器时，再读')])
edit(Path('schema_extract/README.md'), [('；历史记录从 [archive](archive/README.md) 查阅。', '。')])
p = ROOT / 'schema_extract/issues/README.md'
s = p.read_text()
start = s.index('六项真实提取后的')
end = s.index('批次收口时，')
edit(Path('schema_extract/issues/README.md'), [(s[start:end], ''),
    ('批次收口时，Supervisor 逐个读取本批各 run 的 `candidates`；', '批次验收时，Supervisor 逐个读取各 run 的 `candidates`；')])

def replace_document(path, text):
    edit(Path(path), [((ROOT / path).read_text(), text)])

replace_document('schema_extract/issues/archive/README.md', '''# 框架问题决定台账

维护新问题和更新既有决定时，先读[问题维护入口](../README.md)。本目录按稳定 ID 保存正式决定；`partially_adopted` 的 `remaining` 与各项来源限制继续有效。

| 类别 | 内容 |
| --- | --- |
| [extract_workflow](extract_workflow/) | 提取流程、审阅、连续修订与 Design 引用 |
| [extract_tools](extract_tools/) | 调用、恢复、记账与巡检工具 |
| [schema_docs](schema_docs/) | 规范表达与 Contract 边界 |
| [schema_template](schema_template/) | 模板、结构检查和流水线图 |
| [kernel_wiki](kernel_wiki/) | 按证据记录的源实现或资料问题 |

`issues` 与 `candidates` 命令从这些 JSON 读取决定。沿用原 ID，按问题维护规则更新 `decision.outcome`、理由、版本和证据；归档整理不改变决定。

JSON 中的版本、行号与旧路径保留其证据时点含义。历史调查与旧路径的实际位置见[档案导航](../../../schema_discussion/archive/README.md)及[位置映射](../../../schema_discussion/schema_audit_and_cleanup_20261119/cleanup/LOCATIONS.json)。
''')
names = {
    'fp8_scaled_grouped_gemm': 'FP8 Grouped GEMM', 'nvfp4_fused_moe': 'NVFP4 MoE',
    'fmha_mxfp8_qkv': 'FMHA MXFP8', 'mla_expanded_fp8': 'MLA FP8',
    'dual_cache_latent_bf16': 'Sparse Attention', 'gdn_prefill_bf16_state_bf16': 'GDN Prefill',
}
rows = []
for root in sorted((ROOT / OPS).rglob('root.yaml')):
    directory = root.parent.relative_to(ROOT / 'schema_wiki')
    rows.append(f'| {names[root.parent.name]} | {len(list(root.parent.glob("*.yaml")))} | [dispatch]({directory}/dispatch.md) | [文件与验收元数据]({directory}/.extraction/current.json) |')
replace_document('schema_wiki/README.md', '''# Schema Wiki

当前收录六个算子的 15 份完整设计。先读各算子的 `dispatch.md`，再按选择条件阅读对应 YAML；共享 Contract 位于各目录的 `root.yaml`。

| 算子 | 设计数 | 阅读入口 | 当前版本与检查 |
| --- | ---: | --- | --- |
''' + '\n'.join(rows) + '''

[当前总索引](../schema_discussion/schema_audit_and_cleanup_20261119/CURRENT_SCHEMA_INDEX.json)汇总文件摘要、版本、固定来源与语义审阅证据。各 `.extraction/current.json` 是当前成品清单和检查记录，不是 Runner 的阶段回执。验收覆盖来源语义审阅和只读结构检查；硬件字段仍未完整绑定，GPU 与完整动态域数值正确性没有由这些检查证明。外部比较器缺口、源码限制和调用前提以各 YAML、dispatch 为准。

## Catalog 输出目录

输出目录按 [kernel_wiki/catalog.json](../kernel_wiki/catalog.json) 的 44 条 `paths.readme` 父路径镜像；六项已有成品，其余保留输出占位。

| 分类 | 输出目录 | 算子数 |
| --- | --- | ---: |
| GEMM | [gemm](aka-cuda-operators-20261030/operators/gemm/)（dense / grouped / grouped_masked） | 15 |
| MoE | [moe](aka-cuda-operators-20261030/operators/moe/) | 5 |
| FMHA | [fmha](aka-cuda-operators-20261030/operators/fmha/) | 8 |
| MLA | [mla](aka-cuda-operators-20261030/operators/mla/) | 4 |
| 稀疏注意力 | [sparse_attention](aka-cuda-operators-20261030/operators/sparse_attention/) | 4 |
| 线性注意力 | [linear_attention](aka-cuda-operators-20261030/operators/linear_attention/) | 8 |

算子说明、源码与证据从 [Kernel Wiki](../kernel_wiki/CATALOG.md)进入。提取操作见 [Supervisor 手册](../schema_extract/README.md#输入与输出)和[流程总览](../EXTRACT_SCHEMA.md)。旧结果、交付回执与调查材料见[历史档案](../schema_discussion/archive/README.md)。
''')
(ROOT / CLEAN / 'content_changes.json').write_text(json.dumps(changes, ensure_ascii=False, indent=2) + '\n')
print(f'Edited {len(set(c["path"] for c in changes))} existing documents; YAML unchanged.')
