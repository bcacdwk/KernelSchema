# B05 / K5 shared design/optimize writing backend

2026-11-20。源码工作树为 `KernelSchema/.worktrees/schema-workflows-v1`，分支
`feat/schema-workflows-v1`；开工固定输入为
`14a72466a995131965139294ddfc2aa70005ab34`。本批只做 K5.interface/cpu，
没有新增真实模型、GPU 或 Agate 调用，没有访问、读写或运行 EXP，没有派生代理，
没有 Git 提交/推送/配置动作。Python 环境为现有
`.worktrees/.tooling/k2-cpu/bin/python`，实际 Python 3.13.12；没有下载解释器或大栈。

## 交付和接口

本代理独占修改/新增九个文件：

- `kernelschema/writing.py`：共享有限写作内核、唯一 owner 接入、持久化阶段、发布与恢复。
- `kernelschema/_writing_policy.py`：原 Template 规范化和有限 Task/权限/语义方法。
- `kernelschema/design.py`、`optimize.py`：原唯一 API/CLI 的实际 factory。
- `kernelschema/WRITING.md`：完整接口、内层协议和 K6/A6 消费约定。
- `schema_design/README.md`、`schema_optimize/README.md`：薄入口说明。
- `tests/workflows/test_writing.py` 和 `fixtures/writing/fake_claude.py`：专属 CPU 测试及真实 subprocess 的 synthetic CLI。

实际工厂为 `kernelschema.design.workflow_backend()` 和
`kernelschema.optimize.workflow_backend()`，backend=`claude`。公共业务函数统一
从 `kernelschema.api` 导入，包根同名空间继续留给模块。没有另一套 CLI、账本、
顶层调度或自动调用 implement 的隐藏路径。

```python
from kernelschema import writing
from kernelschema.api import design, optimize
from kernelschema.artifacts import Publication, durable_write
from kernelschema.budget import use_authority
from kernelschema.contracts import jcs

p = Publication(bundle_root)
rules = writing.freeze_rules(p)
profile = p.put_json("profile.json", writing.backend_profile(fixed_claude_profile, environment_ref))
q = writing.request(
    operation="design", request_id="design-1", task=task_ref,
    rules=rules, materials=materials_ref, scope=scope_ref,
    budget={"parent": parent_ref, "limits": child_limits}, profile=profile,
)
q_ref = p.put_json("request.json", q)
p.commit([q_ref])
durable_write(bundle_root / "request.json", jcs(q))
with use_authority(original_parent_authority_path):
    rd = design(request=bundle_root / "request.json", out=new_run)
# 相同原 run 的恢复，不换输入、模型或余额：
rd = design(continue_run=new_run)
```

`writing.request(operation="optimize",...)` 另传 `schema=current_schema_ref`、
`feedback=[...]`、`evidence=[...]`，后两者可空。`writing.backend_profile` 将 K3
profile 收窄为实际支持的两种写作操作，保留原 requested model、parameters.claude。
`writing.template_contract(bundle, task_ref)` 提供无模型调用的公开 Task Contract
预览。完整示例和格式见源码 `kernelschema/WRITING.md`。

## 资料、Task 和草稿

Materials 强制唯一保留 ID `workflow_objective`，至少 planner/reviewer 可见；
它是固定 Task/coverage/硬判据以内的当轮指令。改目标改变 Materials→Q，
不偷改 Task/scope 或 Task 的 performance_goal。选择记录必须包含目标，并只
列真实 planner-granted ID；所有参考均通过显式 role Materials 投递。没有
扫描 author cache、全部历史或任意仓库目录。Task 的公共 Contract、acceptance、
workload 和 environment 均向实际角色投递；私有 evaluator 和 baseline 不进角色输入。

Design 的 locked Contract 由 K1 公开 Task 生成；生产代码不打开手写 R0 的 Design
或私有 evaluator。Golden_reference 使用已知 K1 原件的真实固定 locator：
`KernelSchema@2efc92479db497e0ecf681e276ecddc72cad0de6:examples/workflows/row_softmax/evaluator/golden.py::softmax_fp64`，
不把 bundle digest 伪装成仓库版本。Optimize 读取当前显式完整 Schema 并锁定
其原 Contract，独立审阅继续与原 Task/acceptance/environment 比较。

K3 外层输出仍为 `ks.claude.output.v1`；planner 的 `kind=writing_draft`，其
content 必须严格匹配 `ks.writing.draft.v1`：

```json
{
  "format": "ks.writing.draft.v1",
  "decision": "keep",
  "reason": "No evidenced change or follow-up is needed.",
  "selection": [{"id":"workflow_objective","reason":"Fixed current goal."}],
  "raw": [],
  "dispatch": null,
  "bindings": null,
  "next_action": "none",
  "action_reason": null,
  "action_inputs": [],
  "action_evidence": []
}
```

初始 design 用 `decision=revise`；完整候选以
`raw=[{"node":"root","text":"<完整 YAML>"},...]`、inline dispatch Markdown
及完整 Scope bindings 交付。严格检查重复/非法 node ID、unknown 字段、部分候选、
缺 root 和非法行动 ID；模型字符串不能成为宿主路径。Keep 使用上述空/null 形式
并复用准确原 Schema/scope。已证明只改合法表示的候选使用 unchanged、计数不变；
同 Task 的实质 Design/约束/when/dispatch 决定变化为 parent+1 的 new_revision。
初始设计为 new_task、revision=0。不能凭 planner 自称等价绕过影响检查。

行动 catalog 只包含原请求 `feedback:0`、`evidence:0` 等显式 Ref，加上存在时的
`task/starter_source`。非 none 行动必须有理由、非空显式证据和准确一个非空 Source
输入。源码证据通过 RoleDelivery(kind=source) 投递真实 kernel.py 字节；普通证据
投递实际正文，不只交文件路径。有据的同 R implement 或 verify 可以正常返回；
没有行动的 keep 正常结束。

## 发布、独立审阅和结果

真实流程为固定目标/role 资料 → 原 planner 操作/草稿 → 原 Template normalize
和结构/关联检查 → 全 Task coverage/权限及有限 scope/设计影响检查 → 冻结
precheck/change_report → 冻结 raw/scope/Schema → 独立 reviewer → writing_review
→ 结果。Template parser/normalizer 没有被复制；Rules 冻结原 docs、Template 全部
传递 Python 源码以及本方法/checker/policy，运行和恢复核对实际版本。

写作调用需要的唯一公共接口缺口已由主代理在 `evaluation/task.py` 实现并负责：
`RowSoftmaxTask.validate_scope_documents(schema_doc,scope_doc,*,resolved=None,require_hardware_binding=False)`
及 `require_evaluation_scope_document(scope_doc)`。旧 Ref 接口委托同一逻辑。
K5 直接传原 Template 的真实 ResolvedTask，在 Schema/change_report 尚未发布时
完成 scope 检查，不先伪发布一个 Schema，也不另写 when/variant 解释器。
本代理没有修改公共 evaluation/core/runtime/budget 源码。

Change_report 只绑定输入 Q/Task、归一化候选原字节和实际差异/影响事实；Schema
引用 change_report；writing_review 绑定最终 Schema/scope 和原 Q；Schema 不反向
引用 review。每段有独立且预先固定的 publication ID，没有互指 hash。Keep 的
reviewer 读取准确旧 Schema 原字节，不以归一化预览冒充输出。YAML Decimal 用
明确 `yaml_decimal` 引文编码交付义务，不转成 binary float，也不改写原 YAML。

Reviewer content 严格为 `ks.writing.review.v1`：`format/subject/items`，每项
`id/status/reason/location/quote`。必须覆盖全部实际义务 ID、准确位置/引文及原因；
主体错配、重复/漏项、无位置或未知结果为 unknown。它独立判断完整公共 Task、
root Contract、当前 scope、所选数学/调度/流水/约束、目标、设计影响和行动证据。
未选设计保留并结构检查，但不强制实现。结构 pass 不签语义，quoted obligation
本身不等于证明，CPU 结果不签 GPU/模型/性能资格。

四项写作门均 pass 才 completed。失败/unknown 候选保持不可变且不 completed，
修正另建原请求/发布。无合法 Schema 的失败保存 native receipt 和 retained draft，
不伪造成功 writing_review。仅格式合法的 REALIZATION.Feedback locator 不证明
实测；不能认证新的测量来源时 permissions=unknown，普通反馈继续留在外部。

## 单账户和恢复

K5 只使用 K4 的 OwnerAccount/use_authority。顶层只创建原 owner；同父跨 Q/R/out
绑定同一 authority。Planner 准入前 reserve 包含 reviewer 最低收尾，实际 invoke
前通过同一 authorize 检查 pause/fence、余额和期限；原 UTC/monotonic 截止字段
原样交给 K3，不能借 input preparation 或队列延长期限。Finite request/parent/native
token 预算照实拒绝。

原 operation、reservation、K3 intent/native locator/终态、draft、precheck、各 publication
ID、冻结候选和 review 均在首 result 前持久化。K3 独占原 runtime terminal，写作
关系放本 workflow 的持久 state，不覆盖 K3 的 immutable terminal。原操作未知则
blocked/settle_only，继续查原操作，不新提同意图。原累计 receipt 只吸收一次。

恢复核对 owner、operation、reservation、原 native 意图以及冻结 draft/precheck/raw、
从原草稿重新推导的 scope/义务和 review 输出主体。改变私有 candidate state 不能
绕过已冻结原件。在 planner 已收费、Schema 已发布、review 已返回、首 result 前或
顶层 result 投影丢失时，重新创建 backend 实例仍从原 run 找回，无第二次真实费用。
终态 replay 和 keep/none 不新开调用。

## CPU 证据

所有行为测试使用实际 API/CLI、真实文件/Publication/Bundle、原 Template、K4
账户和 K3 real subprocess fake CLI。Draft/review/measurement 字样明确 synthetic；
测试不声称真正数学正确性、模型推理、GPU 或收益。每个成功写作 fake run 有
planner/reviewer 各一次原 native 收据，共 52 synthetic tokens。

已执行的轮次，不把增量记录写成一次总 pass：

| 轮次 | 范围 | 实际结果 |
| --- | --- | --- |
| 初始实际链 | design / optimize keep / revise；API、闭包和终态恢复 | 3 passed / 26.27s |
| 主功能轮 | 两入口七个首 result 前断点、目标/权限/未知、同 R 行动、未发布 scope 前检 | 33 passed / 223.91s |
| 原调用/账户 | unknown settle-only、native 失败、review floor、同父新 Q、actual factory/CLI | 9 passed, 30 deselected / 69.20s |
| 输出精确性 | unchanged、YAML Decimal、keep 原字节、precheck 及发布后恢复 | 6 passed, 36 deselected / 43.27s |
| 全 Task 投递 | environment/workload、新绑定 Task 拒绝、首费用后恢复与 CLI | 5 passed, 39 deselected / 33.46s |
| 新实例/Feedback | 新 backend 恢复、R0 Design 不投递、deadline 字段、伪实测 locator | 4 passed, 41 deselected / 31.67s |
| 最终 profile helper | 两个真实 factory、公共 backend_profile 构造及 CLI/投影恢复 | 2 passed, 43 deselected / 17.23s |
| 恢复 state 完整性 | 修改 private candidate 拒绝、两入口冻结后恢复、初始设计与 unchanged | 5 passed, 41 deselected / 38.47s |

当前专属文件共 46 个测试；上述功能轮与受影响增量轮覆盖它们，不声称一次 46-pass。主功能测试命令为：

```sh
/home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k2-cpu/bin/python -B \
  -m pytest tests/workflows/test_writing.py -q -x
```

增量轮使用表内对应测试函数的 `-k` 选择；最后一轮实际选择为
`resume_rejects or first_result_loss and schema_frozen or design_actual or same_design_representation`。

未反复跑全库。`git diff --check` 通过，另对九个专属新增文件做逐行空白检查和
Python AST parse，避免未跟踪文件不在 diff 范围内造成虚假检查。没有修改共享
fixture/conftest/pyproject、原 Schema/extract 产物或计划/AGENTS。

## K6 / A6 后续消费和限制

Completed optimize 的 `action_inputs` 以前两项输出 Schema/Scope 开头，随后是
明确选择的原 Source/输入；`action_evidence` 是原授权 Ref。driver 必须先结算并
关闭旧事务/scope，在同一 owner 余额/fence 内构造并准入新的固定 implement 请求。
Verify 对应 verify_only 和准确非空 Source。`extensions.information.admitted_request`
始终为 null：发布新 R 不代表已准入新 R，没有移用旧 C 的 qualified/fidelity。
新请求需要 K4/A5 对实际 Task/R/rules/scope/C 再判。无行动 keep 正常结束，不强迫 revise。

本批保留的明确限制：

1. 当前有限 Task/scope 方法支持 K1 固定 row-softmax；design Contract 生成器要求
   固定公开未绑定 environment。若 Task 真正改为具体设备/栈，应另提供相应明确
   Contract adapter；当前选择是在收费前拒绝，绝不捏造设备/版本或静默沿用旧骨架。
   这不阻止当前未绑定 Task 的 CPU 工程验证。
2. 新 REALIZATION.Feedback 的不可变 locator 仍需要可认证的原测量来源；本有限方法
   不把仅字符串/模板结构当实测，不能认证时 unknown。外部反馈与有证据行动正常工作。
3. 未新增真实 design/optimize MODEL 调用。完整已结算反馈驱动的真实 optimize、
   真实数学/资源判断与 K4/A5 GPU 后续行动留给 K5.real/K6/A6；原 K3 公开资格不代签
   本写作实现，未打开私有 452 份 capture。
4. 本 backend 给明确合法行动和固定主体，不执行外壳的下一请求；K6/A6 的消费/准入/
   实际执行仍是其 owner 工作，不能用本 CPU 接口声称动态产品门完成。

范围内代码和测试交付完成后由主代理固定提交；本代理不提交或推送。
