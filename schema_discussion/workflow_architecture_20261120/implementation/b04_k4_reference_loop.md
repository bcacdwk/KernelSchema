# B04 / K4 fixed reference loop、有限 fidelity 与单 owner

2026-11-20。唯一源码工作树 `KernelSchema/.worktrees/schema-workflows-v1`，
分支 `feat/schema-workflows-v1`；开工输入 `5d497dbd50c85df82e415aea35e5b331aaa47a46`。
本代理未提交、推送或修改 Git 配置，未派生代理，未访问、修改或运行 EXP。

本批是 **K4.interface / CPU 工程交付**。没有模型、GPU、Agate 作业或真实
kernel/fidelity 资格调用。测试中的 pass 为明示 synthetic 运输/状态事实，
不用于签 Task、CUDA 或实际设计正确性。K1.real 仍缺配置与真实远端资格；
K3 原有真实 CLI 运输记录也不代签 K4.real。

## 实际代码与唯一入口

- `kernelschema/implement/{__init__.py,reference.py}`：实际工厂为
  `kernelschema.implement.reference.workflow_backend()`，backend=`reference`；
  直接接现有 `kernelschema implement --request/--continue` 与
  `kernelschema.api.implement`，没有另一套 run.py/CLI。
- `kernelschema/budget.py`：唯一私有 authority、请求贡献、单活动时钟、
  预约/真实 launch 准入、未知 fence、累计收据去重及公共 usage。
- `kernelschema/fidelity.py`、`implement/fidelity-method.v1.md`：读取当前完整
  Schema/root Contract 与所选 scope；生成具体义务；独立只读 reviewer +
  有限机器事实；覆盖与设计符合独立发布。
- `kernelschema/implement/INTERFACE.md`：K5/A5 可消费的稳定账户 API、
  外进程绑定/有限并发组、reference/evaluation channel 与恢复边界。
- `examples/workflows/row_softmax/partial-seed/{kernel.py,ORIGIN.md}`：Codex 在
  本工程实现中手写的非空 ABI/shape fixture，明示真实作者、非独立 workflow
  模型运行产物、未测量；独立于 Task 的数值 reference 与
  性能 baseline。
- 专属 tests：`test_reference.py`、`test_budget.py`、`test_fidelity.py` 和
  `fixtures/reference_support.py`。未改共同 fixture/conftest。

固定请求内执行 coder → 完整不可变 C → 显式 full-Task staging → 原评测
终态 → 独立只读 reviewer → 外部反馈 → 同 R 继续。后续 coder 实际收到
准确最新 C、反馈正文和相应报告正文；不只收到不可读取的报告路径。
无 planner；普通编译/数值/设计符合缺陷、检查 unknown 都是同 R 工作。
能力/材料缺失与不可核定的原操作分开报告，后者总是 blocked/settle_only。

`seed=null` 只使用 Task.starter；显式损坏 seed 拒绝，绝不退 baseline。
seed intent/ready 与每操作 plan/reservation/admitted/input 边界持久化；
已准入不重装 seed，改变的预安装目录也不覆盖。损坏源码输出保留最近完整
稳定 C 和原始诊断/收费；完整但编译不通过的源码有自己的 C 与检查结果。
`verify_only` 非空准确 seed，零 coder、零改码 episode、零晋升，必要评价/
审阅照常计费。

每 C 独立 assessment；latest/qualified 分列，不向新 C 复制旧 pass。
五项硬检查和规定测量齐全即可 completed，软目标 unmet 不阻止完成。
参考循环默认在首个 qualified C 停止。latest=C2/qualified=C1 的专项测试
是明示的公共投影 synthetic 测试，不声称本默认循环已执行额外优化 episode。
完成/预算终止后的结果存为可重建投影，continue 不新增费用，首个 result
文件丢失也无需重做原调用。

## K5 / A5 共用账户接口

`OwnerAccount.create(path, owner=..., limits=...)` 仅用于显式新顶层 owner。
冻结 `owner.descriptor` 为子请求的 budget.parent；`OwnerAccount(path)`
只打开已有私有 authority。`use_authority(path)` 或外进程环境
`KS_BUDGET_AUTHORITY` 把子请求接到同一个文件；子请求不复制余额，
不猜路径，不因新 Q/R/out 创建另一份 parent。

`for_context(context, create=...)` 固定原 owner、descriptor、run/Q、原根
和请求贡献。continue 丢失 authority、绑定或原请求贡献均拒绝，不能重发
额度。显式同 Q 新 run 仍产生新真实费用；公共 request_cumulative 只投影
本 run 的贡献，父历史不复制进来。

主要 API（完整签名见源码/INTERFACE）：

1. `register`，`begin/end`，`pause`：一个 owner 时钟；相同当前请求的
   父等待/子工作不重复相加。pause 后在途仍计，证明全部静止才停。
   即使外部 caller 未显式 begin，reserve 也自动开始活动段，所有原操作
   静止才自动停；显式 begin/end 则把 host 准备与操作间工作也纳入。
2. `reserve(...amount=cost(...), floor=cost(...), native_limits=...)`：
   parent/request/native 取紧，先留最低评测/审阅收尾 calls/jobs/active。
3. `authorize(reservation)`：原预约、暂停/fence、余额和截止逐次核验，
   在实际外部 exec 前记 may-have-launched。只有证明未启动可
   `release_unstarted`；reference 对过期未启动 intent 明确结束 partial/none，
   不热改原输入/期限，也不重置余额。
4. `reserve_group`：同 owner/同请求的有限已知并发组一次原子准入，
   供 native V0 + prefetch reviewer 使用；各成员独立 receipt，活动时间
   共用同一窗口。`claim_group(group_id, 完整原 reservations)` 只在全部
   未启动时允许一次外进程交接；半启动、unknown 或已转交组不接受替代
   进程再启动。`mark_unknown` 也 fence 同组尚未启动成员。
5. `absorb(...terminated=..., lower_bounds=...)`：producer/op/receipt/sequence
   关联，累计只吸收增量；同序号冲突或回退拒绝，旧原件重放零费。
   统计 null 与终止独立；受限余额仍不能确定时保留 fence。
6. `operations/fences/usage`：原 operation 与公共索引/累计投影，无第二
   账本。原收据、已知下界与实际超额均保留，不退款已发生消耗。

真实同机时钟使用 monotonic 纳秒，纳秒以十进制字符串保存，不丢弃频繁
短事务的小数毫秒。reservation 带 deadline_utc_ms，并带
deadline_monotonic_ns / clock_boot_id；重启后保守计入不能证明静止的间隔，
重启且墙钟回拨时不制造免费活动时间。K3 的 launch-time 截止收紧补丁由
主代理安排 K3 owner 完成；本代理只生产/消费这些原授权字段。

首期支持串行请求 + 明确有限 native 并发组，不做任意预算树、通用调度器
或多请求并行。外部 adapter 仍必须在它的真实 launch/提交点使用同一
authorization，不能把私有 authority 当成可随意重建的账户描述。

## 有限方法及真实执行边界

`fidelity.freeze_method(Publication)` 返回 Rules 的 file/tool 条目：
固定方法原字节、checker.py 实际源码与版本，均随公开选定闭包可定位。
当前数学/精度、work ownership/schedule、pipeline/lifetime、Require/Forbid、
必要 dispatch 来自实际选中 effective Schema；非选节点不被强制实现。
每项都有 Schema 定位、准确代码行/原文、判定及理由；漏项/伪位置/无法解释
的适用义务为 unknown。机器项只作极窄的直接来源事实，不能靠 AST 或
coder 自评签 pass；未使用的外部 helper 也不被误判成实际委托归一化。

R0/Rd 专项使用**同一原 Task、同一 C**，Rd 有真实 parent/new_revision 和
change_report 闭包，合法多行 ownership 与 R0 单行 ownership 判定不同。
外部 torch.softmax 或省略 max 对 R0 的反例是 fidelity failure，不编造
[-20,20] 有限域上的数值溢出。真实运行/数值/性能仍由完整 Task evaluator
独立负责。

默认真实 factory 没有伪通过 executor。使用固定公开 K3 profile 的真实
CLI 入口，在未选已资格 channel 时保留 seed 并输出 blocked/new_request，
calls/jobs 都是零。可显式选择固定源码摘要的已安装 evaluation factory，
其 narrow prepare/submit/settle/validate 接缝接同一完整 stage/owner；
它必须证明实际 profile/Task/权限/终态条件，不能以静态 hardware 文档或
LocalExecutor CPU logic 代替 CUDA。

本批没有配置真实 GPU channel，也没有提供把 unqualified Agate 自动升级
成 pass 的 adapter。K1 的现有 Local/Agate executor 保持原有工程与资格
边界；后续已资格 channel 可按 INTERFACE 直接接入，缺项没有硬编码进
Task 的静态 scope 判定。

## 验证与发现

环境为已有 `.worktrees/.tooling/k2-cpu/bin/python`，Python 3.13.12。
未下载解释器、依赖或 GPU 环境；3.11 仍留后置资格。

```sh
python -B -m pytest tests/workflows/test_budget.py tests/workflows/test_fidelity.py tests/workflows/test_reference.py -q
```

完整功能轮 **49 passed in 123.99s**。随后新增 parent/native 有限 token
准入反例并复核真实 API/K3 fake-process 路径，**4 passed, 30 deselected,
13.73s**；两种受限 token 都在收费前拒绝。最后对外部 caller 自动活动段、
整个 budget 组、真实 API/K3 fake-process、同父同 Q 新 run 和 token 准入
定点复核，**18 passed, 29 deselected in 21.91s**。当前本功能共 52 个
测试，在上述完整轮与受影响增量轮中覆盖；不把分轮记录写成一次 52-pass。
不反复运行整个仓库；仅本功能适用组与新增反例。`git diff --check` 通过；
另对全部 12 份本代理新文件检查空白，避免未跟踪文件被 git diff 跳过，通过。

覆盖 V02–V08 适用工程轨迹：真正 API/CLI；固定完整 Task staging 与 source
subject；同 R 准确续写；verify_only；软目标 unmet；独立 coverage/fidelity；
源码/BOM/seed 改动负例；预算最低收尾/parent/native/未知统计/receipt 重放
冲突；父子时钟、暂停/崩溃区间；有限并发组原子性与外进程；同 Q 新 run
真费用、同父不重置、缺 authority/贡献不重建；原 evaluator/model unknown
禁重提；14 个首 result 前断点；无首 result 也能重收原操作；完整终态但
无报告为 unknown。实际 K3 subprocess fake CLI 与同一 owner 集成，两个
fake host invokes 的 52 synthetic tokens 原收据可核；不是模型调用。

强化真实 parent/new_revision 闭包时发现 core `_relationships` 的
`def expected(...)` 被局部 revision 整数同名覆盖，导致多个 Schema 遍历
TypeError。已向主代理给出精确两行补丁；主代理将整数改名
`expected_revision`，并独立运行本强回归 **1 passed in 4.38s**。本代理没有
修改/提交 core；强回归留在 K4 test_fidelity.py，不另造镜像测试。

上述测试使用专属注入 runtime/executor，却走真实 artifacts、RunContext、
固定 source staging、owner 持久化与 result 验证。默认 factory 不引用测试
fixture。临时测试原 run/原件留在 pytest 临时根，测试可重建；它们不是
K6 公共 GPU evidence 或本批真实资格 capture。

## 修改边界

本代理仅写开头列出的 K4 源码、方法、partial seed、专属测试与本报告。
未改 runtime/evaluation/core 公共源码、pyproject、原提取/计划/AGENTS、
共同 fixtures/conftest 或私有 qualification 档案。工作树同期出现的 core
两行修复和 runtime 截止补丁由主代理及对应 owner 管理，不包含在本代理
独占交付中。无 Git 提交/推送/配置动作，无新增模型/GPU 调用。
