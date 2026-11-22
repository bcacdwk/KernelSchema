# B40：一次真实 write-only optimize

2026-11-21 UTC。唯一公开 `api.optimize` 调用已完成：**completed / revise**，原生 CLI succeeded、exit0、未超时，diagnostics=[]。写出了一个可读的 root YAML，供下一次原标准 AKA implement 使用。本轮只完成 Schema 写作，未评估实现正确性、性能或优化效果。

下一次 AKA 应直接读取这个原模型文件：

[/tmp/schema-b40-write-only-optimize-20261121/optimize/artifacts/6dcd8f97b77f4bbd8ed96a01758feff6/raw/root.yaml](/tmp/schema-b40-write-only-optimize-20261121/optimize/artifacts/6dcd8f97b77f4bbd8ed96a01758feff6/raw/root.yaml)

- **9,106 bytes**，SHA256 `58a83e9be44e4d1d5f656454fbec1de65ec948f49c79aad06f9daeb6b6d30e1c`。
- 基本 YAML mapping 可读；逐字等于本次 planner `content.raw[0].text` 和 `retained_files` 原稿。没有宿主手改 root、截短文本或生成替代 Schema。
- 只有 root，未新增 dispatch 分支。单文件或其实际 raw 目录均只包含此次可读 root；建议本次 handoff 使用上面的精确单文件。
- [机器交接摘要](/tmp/schema-b40-write-only-optimize-20261121/root-handoff.json)、[完整终态摘要](/tmp/schema-b40-write-only-optimize-20261121/optimize-summary.json)。

## 输入与固定源码

实际运行源码从公共 KS worktree `/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1` 的固定提交 `907c99aa1589bddf73e26079f2304b83d68a8e51` 通过 `git archive` 提取，运行于 [source](/tmp/schema-b40-write-only-optimize-20261121/source)。未热安装活动树，未改原 main/extract。139 个文件的 [source-manifest.json](/tmp/schema-b40-write-only-optimize-20261121/source-manifest.json) 与收尾逐项核对一致；archive SHA256 `4b7489f47485c0e838f21ec40b8b49540da6d95fdd62515931646afe775d172a`。

使用现有 `/home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k1-evaluation/bin/python`。原 B33 profile 逐字复制，canonical SHA256 `3e1766d7fad58da4e76b77c28a296b0adcb225c04e7e56704dac45a848efc530`；CLI version/hash 与已有认证的只读预检通过，未新造认证或读取/打印凭证。

任务保留 `Model(alpha).forward(x, residual) = x + alpha * residual`，两个同长度、连续、1D CUDA bfloat16 输入，bfloat16 输出，标准 L20D/Triton。数字 shape IDs 为 `0: N1024, alpha1`、`1: N65536, alpha0.5`、`2: N1048576, alpha2`。task/reference/input/shapes 均来自 B33 已固定材料。

当前 Schema 只采用 B33 的原 [root.yaml](/tmp/schema-b33-write-only-real-20261121/retained-native-draft/raw/root.yaml)：8,076 bytes / SHA256 `22022d1880e1336e9efb059a1bb285e619646bba84ca7da07e71fda21321c88c`。没有传入原误标为 YAML 的 Markdown dispatch。

真实反馈为 B36 [stage-summary.json](/tmp/schema-b36-standard-resume-20261121/stage-summary.json)，以及它明确声明的五个文件：`v0.json`、`v1.json`、`attempt.json`、`public-candidate.json`、`schema-native-result.json`。全文逐字复制到 [materials/feedback](/tmp/schema-b40-write-only-optimize-20261121/materials/feedback)，源路径、长度、SHA 均在 [input-manifest.json](/tmp/schema-b40-write-only-optimize-20261121/input-manifest.json)。

反馈没有被改成成功：只有原 PyTorch V0 通过，B36 模型超时，candidate_commit=null、changed_paths=[]，candidate full Evaluate0、ABBA0；v1 的 PASS/时延是 incumbent carry-forward。plugin_changed 是工具准备阶段的真实失败，插件解释器兼容修复由另一个任务处理。没有为此次写作等待新 kernel 成功或补造性能结果。

[objective.md](/tmp/schema-b40-write-only-optimize-20261121/materials/objective.md) 是明确标出的当前写作目标：先产出可立即实现的简单 root，补足索引、launch、访存、精度和 Model wrapper 责任，避免无关资料探索，后续有真实实现后再扩 dispatch/调参。它是目标，不是人工 Schema 或效果证据。

实际 native 输入为 **75,487 bytes、22 materials entries**，核对完整原 root 正文已送达；见 [input-size.json](/tmp/schema-b40-write-only-optimize-20261121/input-size.json)。所有十二个源输入与其副本收尾 SHA 一致，见 [original-inputs-unchanged.json](/tmp/schema-b40-write-only-optimize-20261121/original-inputs-unchanged.json)。没有修改旧 B33/B36 原件或旧 B14 账户。

## 唯一调用与产物

先通过 [preflight.json](/tmp/schema-b40-write-only-optimize-20261121/preflight.json) 的零调用准备。预检在 `api._invoke` 边界保存输入，并显式禁止 `ClaudeRuntime.admit/invoke/settle` 与 owner 创建；没有将预检当作模型运行。

[run_writing.py](/tmp/schema-b40-write-only-optimize-20261121/run_writing.py) 的 `run` 只调用一次公开 `api.optimize`，无 retry、reviewer 或隐式 implement，并有不可覆盖的单次启动标记。真实命令及摘要在 [run-command.json](/tmp/schema-b40-write-only-optimize-20261121/run-command.json)。

- run ID：`b447ee67420f4073a74f2326914a3590`。
- request ID：`optimize-7f3f961c000a4e3ca1fe85ccd537589f`。
- session：`ad2e7026-482c-411e-998a-8d39d046d0ac`。
- 新独立写作 owner，parent=null；planner1/reviewer0/coder0/extract0/model1/GPU0/active610000ms/tokens=null。
- 原 runtime profile：`qwen3.8-max[1m]`、xhigh、600000ms、USD2.00 原生停止阈值；该美元阈值不是硬账单上限。

原始结果和文本全部保留：

| 内容 | 精确产物 |
| --- | --- |
| API 原结果 | [result.json](/tmp/schema-b40-write-only-optimize-20261121/optimize/result.json)；[API return](/tmp/schema-b40-write-only-optimize-20261121/optimize-return.json) 与其对象一致 |
| 原生最终回复文本 | [native-result-text.json](/tmp/schema-b40-write-only-optimize-20261121/native-result-text.json) |
| 实际 content、原回复字符串、raw 原稿 | [draft.json](/tmp/schema-b40-write-only-optimize-20261121/optimize/artifacts/83eaf41ad4824fe387741a5e09db15f7/draft.json)、[original/0.txt](/tmp/schema-b40-write-only-optimize-20261121/optimize/artifacts/83eaf41ad4824fe387741a5e09db15f7/original/0.txt) |
| 模型说明 | [rationale.md](/tmp/schema-b40-write-only-optimize-20261121/optimize/artifacts/47e724d4a10747ad8a62ecd636aa8b4c/rationale.md)、[change_report.json](/tmp/schema-b40-write-only-optimize-20261121/optimize/artifacts/47e724d4a10747ad8a62ecd636aa8b4c/change_report.json) |
| 原生回执与完整 stream | [claude-receipt.json](/tmp/schema-b40-write-only-optimize-20261121/optimize/artifacts/1f34433edc704a09ab7b63ac3f758b27/claude-receipt.json)、[claude-events.jsonl](/tmp/schema-b40-write-only-optimize-20261121/optimize/artifacts/1f34433edc704a09ab7b63ac3f758b27/claude-events.jsonl) |

版本2保留了原单 kernel、每 program 1024 元素、masked 连续访存的方案，补明 Model wrapper 及乘积先舍入至 bfloat16、再相加的边界，并说明每一步内部可用更宽运算。它把现有失败反馈与未测 launch 参数写清楚，没有声称已改善性能。root 长度由 8,076 增至 9,106 bytes；不把“更短”或“已减少实现耗时”冒称本轮结果。

`structure=pass` 仅表示基本 YAML 可读；`publication=pass` 表示原稿/说明/记录写出；permissions、semantics 均为 `not_run`。没有独立 reviewer、Template 效果门或 GPU 验收。下一步实现仍由原标准 AKA 的真实 candidate、正确性和性能结果决定。

## 实际用量与终态

[native-terminal-summary.json](/tmp/schema-b40-write-only-optimize-20261121/native-terminal-summary.json) 和 [native-receipt-summary.json](/tmp/schema-b40-write-only-optimize-20261121/native-receipt-summary.json) 只摘取终态/用量字段，不包含模型思维。

| 项目 | 本轮实际 |
| --- | ---: |
| host model invocation / native rounds | 1 / 1 |
| planner / reviewer / GPU jobs | 1 / 0 / 0 |
| input tokens | 6 |
| cache creation / cache read | 22,691 / 0 |
| output tokens | 17,606 |
| total tokens | **40,303** |
| native reported USD | **0.5819987499999999**（约 0.58199875） |
| native duration | 387,922ms |
| runtime active | 390,056ms |
| API request cumulative active | 393,087ms |

原生 `success / is_error=false / end_turn`，receipt `succeeded / accounting_complete=true / terminated=true / errors=[]`，API unsettled=[]。进程已退出，模型 slot 已释放。physical_retries=null，供应商内部重试数 unknown；原 terminal 的 costBasis 为 unknown，费用按 native 实际报告值保留。没有第二次 host 调用。旧 B14/B35/B36 的历史费用与 unknown 保持原记录，不拿本次独立写作账户改账或补余额。

## 最简复用方式

[api-sequence.md](/tmp/schema-b40-write-only-optimize-20261121/api-sequence.md) 给出了可以填写具体路径后使用的四步 Python 调用。核心关系如下：

```python
d = api.design(task=task, out=d_out, references=refs, runtime_profile=profile)
s1 = d_out / d["extensions"]["information"]["schema_directory"]
i1 = api.implement(schema=s1, workspace=i1_out, **native_aka_options)
# 将 i1 原样保存为 i1_json；使用原结果明确声明的 feedback_files。
o = api.optimize(task=task, schema=s1,
                 feedback=[i1_json, *i1.get("feedback_files", [])],
                 out=o_out, references=refs, runtime_profile=profile)
if o["decision"] == "revise":
    s2 = o_out / o["extensions"]["information"]["schema_directory"]
    i2 = api.implement(schema=s2, workspace=i2_out, **native_aka_options)
```

`native_aka_options` 用原 standard implement 的 problem、repo_root、platform、framework、max_iters 和所选原生预算参数；Python 环境须能 import 对应固定 AKA 源码。这里没有新增调度平台或默认额外 OwnerAccount。keep 时结束；缺少可读 Schema 时保留写作诊断，不用虚构路径继续。原生失败和 unknown 可直接成为 optimize 输入；AKA campaign 的 `completed` 本身不代表交付了新 kernel。

本轮未执行这套四步示例，只运行其中一次 optimize。预检辅助脚本最初误用 workflow JSON loader 解析含浮点 alpha 的 native shapes，已改用标准 JSON parser；未触发模型。收尾首次整份回执打印带出了内嵌 base64 originals 并被工具截断，未解码或阅读模型思维，随后改为字段白名单和仅提取 result event 的终态摘要。其余检查仅为 YAML 可读性、原文/输入/固定源码一致性及示例语法，不是语义或效果验收。没有新产品源码改动、提交、推送、派生 agent、home 递归搜索或访问禁止仓库。
