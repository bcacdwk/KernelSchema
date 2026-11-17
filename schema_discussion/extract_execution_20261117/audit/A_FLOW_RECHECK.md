# 独立审阅 A：四项修复复验

结论：**A1–A4 四项均已闭合，签收本审阅范围。** 原审阅者独立核对了修复前后生产代码、实际调用链、新增断言，并重放原反例及最相关的完整后续路径；没有发现需要继续修复的同根问题。初审发现及触发条件保留在 [A_FLOW.md](A_FLOW.md)。

实施报告只用于确定修复范围。本次另建一致副本 `/tmp/extract-flow-recheck-d0qe7e4i`，从最终工作树复制 169 个仓库文件，记录 `recheck-manifest.json`；检查结束后再次逐字节核对，差异为零。副本包含 B 的最终执行器/审计修复。产品文件只读；本审阅仅写报告及 `/tmp` 复现、日志。

## A1：读前许可集合 — 关闭

核对 `artifacts.py:224` 的 `artifact_inventory()` 与后续散列分离，正常调用的可信名单来自已验证的 base checkpoint，只有 P2 worker 可新增 `dispatch.md`。`receipts._artifact_baseline()` 为 select 取得已登记候选自己的名单。`runner.accept_result()`、失败/中断草稿的 `_normalize()`、checkpoint 创建/复核、只读 helper 和 `audit._verify()` 均接入这些名单；审计没有拿 validation 自报的原稿集合扩大权限。

原 `repro_preflight.py` 独立重放结果：

```text
final_rejection: ExtractError artifact inventory differs from the prepared scope: unexpected extra.yaml
unauthorized_extra_reads_before_rejection: 0
unauthorized_extra_captured_in_original: False
```

另执行实际 normal / failed / interrupted / checkpoint 入口的拒读观察器用例：即使 `extra` 是已选但尚未准备的节点，其 YAML 仍在任何候选内容读取前被拒绝。公开 `audit_run(..., verify=True, archive=...)` 的单调用用例也通过：合法原稿可以归档，加入 `unexpected.yaml` 后先报名单错误，全部原稿拒读观察器未触发，错误归档未写出。这里只使用一条原生离线 fake CLI 调用，不是模型或阶段全链。

只读现存内容仍需读取、散列及语义权限核对；新增名单检查没有放开字段修改。正常原子 rename 后已消失的临时文件不成为额外产物。

## A2：跨代际及移除实例的事实材料 — 关闭

核对 `prompts.py:119` 的完整回执复制及身份验证、`:168` 的知识传递条件和共同提示。历史 handoff/issues 的阅读资格不再由当前 generation 决定；指针保留原 receipt digest、固定 source_id、实例、代际及 `current_instance/changed_instance/removed` 关系。长事实内容留在只读 materials，P6 仍先独立读当前成果，再查过程记录。

原 `repro_handoff.py` 中，仅更正共享名称后的独立结果为：

```text
new_p5_reviewer_can_find_worker_fact: True
after_return_worker_receipt_provided: True
new_p6_can_find_worker_fact: True
original_receipt_remains_accepted: True
original_receipt_still_has_fact: True
```

另执行移除实例的实际回转用例：P2 删除该代表、P4 完成安排后，P6 仍拿到标记 `removed` 且与原回执逐项一致的事实材料；该节点同时不在当前 `source_identities` 和 `dependency_digests` 中。没有恢复旧批准、把历史定义重新放回当前 workspace 或新增知识台账。源码中对改变实例的标记分支也已核对。

## A3：P1 Contract 与待修 Design 的阶段边界 — 关闭

核对 `artifacts.py:762`、`:906`：P1 普通 Contract 语义/文案修改不再将所有待修 Design 加入本阶段解析前提。真正由 Template 产生的 `identifier+reference` 关联修正仍扩大到消费者集合，并保持整组权限门禁；此处没有移除 P3/P5/P6 的真实来源闭合检查。

原低层复现的两条路径现在均成功：

```text
metadata landed: landed
contract landed: landed
```

另执行扩展后的真实状态/事务用例，实际完成 P2 退休 provider 后的 P1 Contract 修订和独立批准、P3/P4 后续审阅与安排、consumer 的 P5 显式本地化和独立批准。该修订包含需 Template 清理的末句号。核对期间旧 provider 不在当前来源集合，consumer 修稿前仍保留原 `from`，历史 checkpoint 摘要未变。关联名称规范化的反例也通过：仍将全部消费者纳入事务，涉及只读 provider 改动时整组拒绝。

## A4：P1 Source 更正后的独立 P2 复审 — 关闭

核对 `store.py:687` 的阶段 projection 失效、原批准真实性核验与 validation 记录，以及 `stages.py:411` 的当前 `plan_draft` 准备。改变 Source 时仅打开确实受影响的 P2 职责；Name/Describe 的原复现仍直接进入 P3，说明未将无关元数据修改一律退回 P2。

独立改写原 A4 复现的最后一段，沿修后正确路径实际调用独立 reviewer，而非继续硬编码旧 P3 路径。脚本 `repro_p1_source_projection_rechecked.py` 断言旧回执可追溯、generation 正常增长、所有既有 calls/deadline/revisions/adjudications/signoffs 未重置，随后得到：

```text
current_path: [('P2', 'reviewer'), ('P3', 'reviewer'), ('P6', 'reviewer')]
final_status: ready
old_approval_immutable_and_budgets_preserved: True
selection_and_unaffected_p4_approval_preserved: True
```

P2 使用新回执，调用数只增加这一次必要独立复审；原选择实例和未受影响的 P4 批准保持。这是确定性 backend 配合真实 Runner/Store/Template 的离线完成路径，未调用模型或发布真实成果。

## 独立执行记录与签收边界

副本中的 `recheck-a1.log`、`recheck-a2.log`、`recheck-a3.log`、`recheck-a4.log` 保存上述四条独立复现输出。统一使用已有 `/opt/venv/python3.13.12/bin/python -B`，PATH 的 `claude/codex` 均为固定 exit 97 的阻断脚本。

另外五项最相关回归全部通过，无 skip：

- normal / failed / interrupted / checkpoint 的额外 YAML 读前拒绝。
- 事实跨元数据回转与移除实例后仍可读。
- 退休 provider 历史保留、P1 Contract 修订与 P5 显式修稿完整路径。
- 真正关联规范化不能改写只读 provider。
- 单调用公开审计：合法归档、未启动证据及额外原稿名字读前拒绝。

前四项为 `recheck-regression.log`，4 项通过，82.289 秒；审计项为 `recheck-audit.log`，1 项通过，4.037 秒。审计的离线 namespace 按既有外部系统 bootstrap 准备真实 proc，并在任何仓库 Python 运行前通过 `setpriv` 永久降回原 UID/GID、设 NNP；产品没有增加 sudo 或隔离绕过。`git diff --check` 通过。

本次没有重复 125 项整套，没有任何模型、GPU、queue 或真实 P1–P6 全流程试跑，没有工作区提交。原始四项问题及其相邻不变量已在本职责范围签收；不据此宣称真实模型语义全链验收完成。两个已停止真实试跑仍是零阶段批准、未发布；既有真实 CLI / Reviewer 正反例的覆盖边界保持原记录，已取消全链不列作待补跑。

复验完成，停止写入；本审阅范围没有遗留需实施者处理的问题。
