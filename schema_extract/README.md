# Schema Extract

本地执行器从显式选择的 kernel 材料调用已登录的 Codex CLI，形成可追溯的 schema。程序、CLI 探针及两算子旧政策试点已经实现；**本轮新政策只做静态审查，未运行单测、探针、模型、提取、真实产物检查或调度启用。** 历史依据见 [ISSUES.md](ISSUES.md)。

## 当前政策

- **硬件完全不在提取范围。** `contract.hardware_specification` 只保留 starter 原样占位；撰写、搜索、检查、审阅、收尾都跳过。extract 固定调用现有检查器的 `--skip-hardware`，复用缺硬件的关联处理。普通检查默认不变。旧硬件问题标为 `ignored`，不影响 ready/export，不代表资料补齐或硬件已验证。
- **没有 agent 硬性额度。** 不按时长、token、调用次数、修订轮次、无进展次数或续接次数停止。历史 plan 中的旧 limit 字段不再生效。只保留明确 pause/cancel、真实执行错误处理和进程退出／明确取消后的清理。超过 60 分钟仅供 Supervisor 调查活动与产出，不能自动终止。
- **Reviewer 是只读合作者。** 重点源码忠实性、实质遗漏、值得简化的冗余；具体可修问题合并交 worker 小修，由 reviewer 自行判断什么值得改。不凑问题数量、不因格式偏好或额外硬件／实验／误差证明扩大工作。已知原实现缺陷与外部证明缺失如实备注，不自动否决提取。只有基础任务定义确实不清楚或必要源码缺失才 `needs_input`。
- **真实要求保持不变。** 不为迁就实现缩小 Contract 或伪造 fallback；来源明确的转录可自动纠正并重新查看受影响节点。数学、调度与流水线仍按规范逐段读写回查。

## 流程和记录

工作阶段为 `contract → root node → 其余 node → index`，Contract 使用单独会话协作审阅；单节点合并 root 与最终审阅。节点内部依次写数学、调度、流水线和适用 realization，需要继续时保存检查点并续接，没有“两次续接”限制。具体修订不设轮数上限。

固定输入接入现有 `kernel_wiki/catalog.json`，只选择指定 operator；公共说明仅供当前算子导航，不读取整库上下文。干净来源使用实际仓库 commit，选定的未提交来源才生成独立私有快照。提示词、规范、检查器和执行器内容摘要一同冻结，worker 只写其独立暂存目录，reviewer／supervisor 使用只读沙箱。提取与复核显式 `gpt-6-astra / xhigh`，总提取调用并发最多 2，reviewer 最多 1。

独立 Supervisor 每轮可另有 1 个只读巡检调用，不占转换名额，也不因两个 worker 忙而跳过调查；不为巡检设置 agent 时长额度。

```text
schema_extract/
  runner.py             计划、调度、恢复、导出
  codex_cli.py          无 agent 超时的 CLI 调用、事件与取消确认
  records.py            固定输入、原子 task.json、不可变检查点
  protocol.py           提取与合作审阅提示／回执
  supervisor.py         独立的巡检与可选 15 分钟调度入口
.schema_extract/        忽略的运行记录
  inputs/<fingerprint>/repo/
  runs/<run-id>/plan.json
  tasks/<key>/task.json
  tasks/<key>/attempts/...
  tasks/<key>/checkpoints/...
<work-root>/<key>/<attempt-id>/   仓库外的单次可写目录
```

每个 `task.json` 是唯一状态权威。先保存完整不可变检查点，再原子更新任务指针；中断半成品不被接受。恢复先确认旧调用停止，再从已接受检查点建立新目录／会话。相同已结束任务不会因再次 `run` 自动重做。

回执意见分 `fix / note / needs_input / ignored`。ready/export 需要非硬件检查通过、当前产物已完成相应协作审阅、待修错误已处理；诚实备注可随交付保留。完成提取不等于完成 GPU 数值、性能或硬件验证。正式输出只创建新目录；相同内容重复导出幂等，目标内容不同则保留原样。

## 命令

需要 Python 3.10+、现有检查器的 PyYAML 和已登录本地 `codex`。以下仅为后续新上下文的操作说明，本轮没有执行：

```sh
python3 -m schema_extract discover
python3 -m schema_extract plan --run-id NEW_RUN --id OPERATOR \
  --work-root /absolute/path/outside-repository \
  --output-root schema_wiki/NEW_COLLECTION --jobs 2
python3 -m schema_extract run NEW_RUN
python3 -m schema_extract status NEW_RUN
python3 -m schema_extract pause NEW_RUN
python3 -m schema_extract cancel NEW_RUN
python3 -m schema_extract resume NEW_RUN --id OPERATOR
python3 -m schema_extract retry NEW_RUN --id OPERATOR --reason '说明重新尝试原因'
python3 -m schema_extract export NEW_RUN --id OPERATOR
```

没有默认 all，不自动导出。全局 `--repo / --state-root / --codex` 放在子命令前；`--codex` 可明确本地可执行文件。pause 等待当前调用边界，cancel 请求终止所属进程并确认；状态以 task.json 为准。`run` 只派发 queued，不自动恢复 stopped/ended 或用户的暂停／取消。

`plan --seed-run OLD_RUN` 可显式复用兼容的已接受草稿：逐文件核对原语义来源、规范、检查器与检查政策；保留原字节、回执、来源、用量和问题身份，重新开始 Contract 协作审阅。**新硬件政策改变了检查器／规范，旧政策 plan 默认不能直接 seed 或静默续跑。** 旧草稿和证据保留，不提供通用迁移入口；后续新上下文应先审计兼容边界（SE-012），不能把过去命令当成已验证的新政策运行办法。

结束时生成 `.schema_extract/runs/<id>/issues.patch`，由 Supervisor 审阅后使用内置 `apply_patch` 更新台账。程序不直接改仓库 ISSUES；已知 issue 沿用原 ID／`ledger_task_key`，不因新 fingerprint 复制问题。

## 观察与 Supervisor

CLI 原始 `usage` 保留。已实证本机 0.163.0 的同 thread 续接报告累计用量，新调用按相邻累计差额观察；cached/reasoning 子项不再加一次。缺失、未知基线或下降记为观测异常，不产生负增量，也不据此阻断流程。历史状态／回执不改写，旧累计数字保留其历史口径。

Supervisor 独立入口为 `python3 -m schema_extract.supervisor tick|enable|disable|status <run-id>`；全局路径参数同样在子命令前。默认不启用。macOS `enable` 为选定 run 安装 15 分钟 LaunchAgent，使用绝对 Python/Codex/仓库路径，非常驻循环；本轮没有运行 enable、tick 或模型。每次只读 compact 状态及活动摘要，给出建议；固定程序只可启动该 run 已 queued 的工作，不恢复用户暂停／取消，不自动发布。会话按明确 thread ID 续接，`tick --new-session` 可显式换上下文。真实定时启停和端到端链路留待新上下文验证。

## 已有试点与验证边界

旧政策下，两题都只在运行区保存，未导出、未运行 GPU：

| 算子 | 已完成 | 尚未完成 |
| --- | --- | --- |
| `gdn_postconv_mtp_bf16_state_fp32` | root v12，Contract 与 final 均独立审阅为带备注草稿 | 新政策运行未验证 |
| `fp8_scaled_gemm` | root v10、小 M v8，Contract 和两个节点均独立审阅 | index / final 因旧 8M 预算未派发；本轮不继续 |

第二轮 FP8 曾遇旧 900 秒超时并从检查点恢复。该自动超时与预算逻辑现已移除。旧 GDN 累计计量存在续接重复相加，实证差分为 5,874,314；FP8 已报告 8,777,991，另一次超时用量未知。历史报告保持原样，见 `.schema_extract/runs/pilot_xhigh_20261101_r2/supervisor-verification.json`。

本轮只允许静态阅读、`ast.parse` 和 `git diff --check`。必要测试代码已随政策更新，但未执行；新政策的检查、工作流、取消与 Supervisor 实际行为尚待运行验证。
