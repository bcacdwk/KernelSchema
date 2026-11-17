# 独立审阅 A：阶段、候选事务、来源与批准

> 本页保留初审记录。最终复验已关闭 A1–A4，结论与独立证据见 [A_FLOW_RECHECK.md](A_FLOW_RECHECK.md)。

结论：**需要修复后复验，当前不签收最终完成。** 本次找到四项可复现的 P2 级代码问题；没有要求恢复旧父树、旧 run 迁移、别名或兼容壳。它们均在当前已授权流程中发生。

审阅者为全新 XHIGH，未参与实施。审阅基线为 `27f36e5` 到当前工作树，新增模块、prompt、入口和测试也纳入范围。实际重点阅读了 `runner/stages/store/artifacts/receipts/maintenance/budget/prompts/reading` 的调用链、当前阶段提示、规范、保留测试变化，以及冻结执行、后端结果、恢复、交付和审计接口。实施报告用于定位证据，不作为代码正确性的证明。

## A1 — P2：额外 YAML 在许可清单拒绝前已被读取和归档

位置：`schema_extract/tools/artifacts.py:224–241`、`schema_extract/tools/runner.py:581–590`；实际阶段集合检查迟至 `artifacts.py:888`。

`artifact_manifest()` 检查普通单链接文件后，将所有以 `.yaml` 结尾的名字视为可读取条目。它不掌握本次许可集合，立即调用 `manifest()` 读取并散列文件。正常 `accept_result()` 随后构造回执、复制 `original/`，才进入 `_normalize()` 的 `_file_scope()`。因此扩展名合法、但本次根本不允许存在的 `extra.yaml` 被先读后拒。

最小复现：一致副本中的 `repro_preflight.py` 创建正常 P1 worker proposal，在其工作目录增加普通 `extra.yaml`，观察实际 `accept_result()`。没有改变生产代码或替换门禁。

```text
final_rejection: CandidateRejected write outside stage scope or deleted artifact: extra.yaml, root.yaml
unauthorized_extra_reads_before_rejection: 4
unauthorized_extra_captured_in_original: True
```

影响：不会产生阶段批准，但“不许可条目在任何产物 read/hash/copy 前拒绝”的承诺不成立；不许可内容已进入原稿证据目录，并产生不必要的读取、内存和磁盘开销。`test_atomic_replacement_is_accepted_but_every_residual_entry_is_rejected` 的读取观察器仅覆盖 temporary/directory/symlink/hardlink，`yaml` 分支目前只断言最终被拒，因此未暴露顺序错误。

最小修法：将无内容读取的目录形状/名字预检与散列分开；调用方将准确的现有文件集合及本阶段许可新增集合传到入口，在首次原稿散列、回执取证、原稿复制之前完成集合核验。统一用于正常采纳、恢复/中断草稿、规范化、裁决候选和审计原稿；已清理的原生原子写临时文件继续允许。不要把“任何 `.yaml`”当作许可集合。补验需要断言额外 `.yaml` 未被读取、未被复制，而不只是最后抛异常。

## A2 — P2：代际过滤使未落实的 P5 知识从新 reviewer 材料中消失

位置：`schema_extract/tools/prompts.py:147–165`，尤其 `generation == current_scope.generation`；同 scope 的非框架记录又只提供最近一份。

P5 按现行规则把普通参数/局部记录放入 worker `handoff/issues`，交 P6 集中落实。若随后仅有 P1 `Name/Describe` 更正，节点正文重新独立审阅会增加 generation。P6 只复制当前 generation 的 P5 receipts；新的 P5 reviewer 只拿最近同 scope 回执，较早 worker 记录也未提供。只要后来 reviewer 没有重复转抄这项事实，原始合法记录虽仍在 accepted receipts 中，却已不在任何后继 reviewer 的隔离材料中。

最小复现：`repro_handoff.py` 正常批准 root/record，在 record worker 的 handoff 放入普通目录待补事实；P6 返回 P1 仅改名称描述，再完成 P1、P3 和 record 的独立复审。检查新的 P5 和 P6 全部 `/materials` 文件与最终渲染 prompt。

```text
before_return_worker_receipt_provided: True
new_p5_reviewer_can_find_worker_fact: False
worker_original_generation: 0
current_node_generation: 1
after_return_worker_receipt_provided: False
new_p6_can_find_worker_fact: False
original_receipt_remains_accepted: True
original_receipt_still_has_fact: True
```

影响：当前源码、实例及待补事实未改变，却因批准代际更新丢失 P6 反查知识去向的必要输入。现有“P6 能看到 worker facts”用例只覆盖没有这种回转的路径。新 reviewer 无法访问整个 run，不能靠“原始回执仍在磁盘”补足可读材料。

最小修法：继续复用现有 accepted receipts/materials，不另建知识台账。把当前批准适用性与历史事实的阅读资格分开；为仍相关的 P2/P4/P5 记录保留原来源、实例、代际及回执身份，作为过程线索提供给对应复审及 P6。对已改变或移除的实例明确标记历史关系，让 reviewer 判断知识去向；不要把历史记录当成当前批准，也不要要求每次 reviewer 手工重复转抄全部旧事实来维持传递。补验上述仅元数据变化场景，并保持 P6 先独立读成果、再读过程记录的顺序。

## A3 — P2：P1 合法 Contract 修订被等待修复的退休来源引用卡住

位置：`schema_extract/tools/artifacts.py:749–772`、`:896–901`；实际准备路径为 `Store.prepare_shared_metadata()`。

`_normalization_consumers()` 对任意 Contract 变化无条件返回目录全部节点，随后 `_normalization_snapshot()` 会解析它们所有 Design 区域。P2 已删除 provider 时，consumer 的历史 raw 被合法保留、等待 P5 显式本地化；这时当前目录内不应恢复 provider。若在此期间按有据问题回 P1 修共享 Contract，解析 consumer 的旧 `from` 会报 missing provider，P1 不能完成。P5 又必须等 P1–P4 批准，形成循环阻塞。

低层复现：`repro_p1_retired_contract.py` 在相同的当前 root/consumer 目录中分别修 `Describe` 和 `CONTRACT.Algorithm_requirement.Input.x.meaning`。修改后者仅澄清一句输入含义，没有改变符号、Design 或实际来源资格。

```text
metadata landed: landed
contract rejected: missing material required to resolve from: 'provider'
```

实际阶段链复现：`repro_p1_contract_flow.py` 正常建立 provider/consumer 和实际 `from`，P6 返回 P2 删除冗余 provider，P4 返回 P1 修上述输入说明，再调用正常 worker 结果采纳。

```text
rejected: CandidateRejected missing material required to resolve from: 'provider'
stage: P1
consumer_role: worker
ready_p5_calls: []
```

影响：现行 P1 明确不以 Design 完成为准入或批准前提，这里却要求待修 Design 来源先闭合。即使 Contract 修订不需要任何引用同步，也无法进入独立 reviewer。仅 Name/Describe 的最新补缺没有覆盖这个相邻出口。

最小修法：P1 候选规范化/检查遵守 P1 完成范围，将待修或未完成的非 root Design 保留为原文草稿及后续复核对象。区分真正需要同步的规范化关联改动，与“共同 Contract 改变后需要复核”的影响集合；后者不能直接变成 P1 必须解析完全部 Design 的前置条件。仍需同步的改动保持整组授权与原子事务门禁；P3/P5/P6 使用前照常要求当前真实来源闭合。不得通过恢复退休 provider 的当前资格、重解释历史链、绕过未来 P5 或放开 P1 worker 写其他 YAML 来解决。补验应完成 P1 独立审阅、恢复受影响正文修复，并保持旧 provider 只作历史材料。

## A4 — P2：P1 更正 Source 后未重开已失效的 P2 职责投影

位置：`schema_extract/tools/store.py:685–725`、`schema_extract/tools/runner.py:705–713`；相关现有门禁为 `artifacts.py:723–726` 的 P2 projection 和 `stages.py:402–413` 的进度恢复。

P1 允许修改 `META_DATA.Source`。P2 的批准职责投影也明确绑定 root Source。当前上游合入后，`refresh_approvals()` 只核 P3/P5 的节点使用上下文，没有核 P2 的当前职责投影。`resume_progress()` 只看 P2 的状态和粗粒度身份，因此把它继续视为 approved，直接进入 P3；真正的 projection 比较直到下一次 prepare 才失败。

最小复现：`repro_p1_source_projection.py` 从正常单 root 的 P6 返回 P1，在同一冻结文件中将合法 locator `kernel.py::copy` 更正为合法的 `kernel.py`，不更换 source tree/ref/实例；P1 worker 和独立 reviewer 均通过，然后尝试准备下一次 P3 review。

```text
stage_after_p1_approval: P3
p2_status: approved
next_stage_failed: ExtractError approved stage subject changed: P2
```

影响：旧批准没有被误放行，但正常流程停在没有正确回转安排的中间状态；后阶段无法准备，普通恢复仍遇到同一旧投影。现有权限和批准模型已表明这应由当前 P2 的独立复核解决。

最小修法：上游合法合入后，按实际受影响的阶段职责投影核当前适用性，保留原始批准/回执；P2 Source 投影变化时打开对应 P2 当前独立复审，并提供当前选择及所需 plan_draft，之后继续必要的 P3/P4/P5/P6。不重置预算和 once/generation 记录，不默认重选所有代表。保留 Name/Describe 等不改变 P2 投影时继续沿用 P2 批准的已有行为。补验要证明能准备并完成正确的下一个独立阶段，而不只是证明旧投影会被拒绝。

## 已核实的有效边界

- P2 `{id, instance}` 与 P4 `{id, wait_for}` 分离；含 root 至多十项、P4 精确覆盖非 root、自依赖/环拒绝，普通 reviewer 不能悄改 worker 决定。机制是否独特由独立语义审阅承担，确定性 fixture 不证明真实代表价值。
- P5 开工同时核写作等待与真实已形成来源；实际同名区域 manifest 与解析加载闭包分开。无关素材不自动变成永久批准依赖，实际采用来源要绑定调用时提供的独立批准与当前有效身份。
- 原始回执/checkpoint 与当前 applicability 分开保存；来源 identity 漂移保留原批准需比较原/当前 raw、origins、Contract、区域及完整使用上下文，并核 accepted validation。没有直接将新身份写进旧批准。
- worker 规范化、只读 reviewer、Runner 裁决 patch 与独立 signoff 的职责清楚；select 在调用前完成候选版本/材料/规范化准备，批准后不修改所选 raw。全部阶段保留 signoff 额度，代际回转没有重置已用一次决定。
- P1 Name/Describe 的已审与未合入稿同步、恢复旧节点时的当前 P1 元数据初始化有独立原/新字节、Version 和批准绑定；正文仍须当前 P3/P5 审阅。A3 是该机制之外的 Contract 扩展缺口。
- 当前 prompt 明确 phase/role/node/operation、实际材料、决定、有效全文、来源/consumer 责任、框架候选闭环，直接给出 `/run/extract/check` 及 `/rules/EXTRACT_SCHEMA.md`。A2 是材料选择的遗漏，不是要求放弃当前成果先行的阅读方法。
- 已删除的迁移模块、parent/from_parent 主链、source aliases 和旧入口没有重新进入当前执行路径。活动 DOC2/Template README 变化为链接和状态说明，未发现借本轮重开 Template 字段规则。

## 验证记录与结论范围

自建一致副本：`/tmp/extract-audit-flow-_alrnpdm`。`audit-manifest.json` 记录 169 个仓库文件；收束时重新逐字节比对当前最终工作树，**差异为零**，当前生产 Python 没有漏出清单的新文件。所有复现脚本在此副本顶层，使用临时状态及确定性 `FixtureBackend`，调用真实 Runner/Store/Template 门禁；状态清理后没有留下运行任务。

执行方式均为：

```sh
cd /tmp/extract-audit-flow-_alrnpdm
PATH=/tmp/extract-audit-flow-_alrnpdm/audit-bin:$PATH \
  /opt/venv/python3.13.12/bin/python -B repro_preflight.py
```

其余复现仅替换脚本名：`repro_handoff.py`、`repro_p1_retired_contract.py`、`repro_p1_contract_flow.py`、`repro_p1_source_projection.py`。PATH 中 `claude/codex` 均为拒绝脚本；这些路径没有启动 AgentCLI 或真实模型。最初用系统 Python 的一轮因缺 PyYAML 停在 fixture 初始化，随后统一用已有 venv，没有安装或修改依赖。

另独立运行四项既有短回归：有限选择/写作等待、跨区解析闭包、来源 identity 等价、全阶段签收预留与 P2 patch/plan 签收。**4 项通过，9.125 秒，无 skip**，完整输出位于副本 `audit-regression.log`。没有重复 125 项整套或运行真实 P1–P6 模型链。`git diff --check` 通过；对保留测试的检查没有发现新增 skip，A1/A2 明确列出当前断言未覆盖的触发条件。

R4 报告中的 125 项离线 suite、25 项补验、6 项 Template API 和真实 CLI/Reviewer 正反例属于实施侧已留证据，不能替代本报告的独立发现。真实完整链依用户要求已停止，两个 run 均没有阶段批准和发布；最终记录为 51 次进程登记观察、36 个不同进程身份、live=0/unknown=0。本审阅没有追加模型调用、全流程试跑、GPU、产品/测试/规则修改或工作区提交。

本机真实私有 procfs 的系统 bootstrap 前置条件已在最终活动文档说明：系统工具准备私有 mount 环境，永久降回原 UID/GID 后再运行仓库 Python；产品不含 sudo。该已明确的环境边界不构成要求增加自动提权或恢复假 proc 的理由。

四项修复完成后，请原审阅者对对应反例、必要正例及最终变更复验。当前仅签收上述已核实边界，不签收整套实现完成，也不签收未执行的真实模型全链。
