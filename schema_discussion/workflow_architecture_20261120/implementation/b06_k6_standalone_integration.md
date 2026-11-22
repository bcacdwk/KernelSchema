# B06 / K6 standalone integration

2026-11-20。源码工作树 `/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1`，开工固定 HEAD `878b1d29bd24cd6231376ddb1d4cf0bade3bbf33`，本代理未提交/推送/配置 Git，未访问/修改/运行 EXP，未修改 AKA，未派生代理。

本批完成可执行独立 driver、K4 Agate channel/资格准备、K5 typed target Contract 与行动策略的工程整合。实际业务 MODEL/GPU 门仍未通过：新增真实模型 invokes=0、GPU jobs=0、已发生新增模型/GPU费用=0，无本批真实在途操作。此前 K3 的真实 CLI 运输/权限资格不代签本批业务写作，既往费用不算本批新增。

## 交付与重大决定

| 文件/职责 | 改变及原因 |
| --- | --- |
| `examples/workflows/row_softmax/driver.py`、`run_demo.py` | 实际调用同一 `kernelschema.api.design/implement/optimize`，一个持久 OwnerAccount，design/已有 raw 消费→固定 implement→至多一次 optimize→执行合法后续固定请求。支持 fixed R0、明确 partial seed、verify_only、任意 cwd、原 run 恢复与选定闭包导出。 |
| `kernelschema/evaluation/channel.py` | 仓库自带 `workflow_channel`；提供 selection/profile 固定、实际 qualification 读取、admission/locator/prepare/submit/settle/validate。直接复用 K1 AgateDevExecutor；原 authorizer/OwnerAccount 在真实提交点决定准入。不是用户还需自行编写的 seam。 |
| 同文件 qualification | 最终选择 **6-job 有界 suite**：probe、完整 baseline 正例和计时、漏写 out / 改 x / 超差 / 同 out 对象换 storage 四个固定坏 Source。只有原同通道完整报告实际拒绝相应硬项才放行。probe+baseline 不足以签 V01。原账户总8 jobs意味着完整资格后至多余2个业务 jobs；不足就 partial，不能换 run 重置。 |
| `kernelschema/evaluation/environment.py` | 有限 `ks.row-softmax.target.v1`。固定公开 Task 算法/判据派生 Contract，硬件来自显式目标材料；原文、真实摘要、immutable repository locator 与许可自含于 Task.environment。目标声明与实际 CUDA capture 分开，全部指定 runtime 字段必须实际匹配；未知格式收费前拒绝。 |
| `_writing_policy.py`、`writing.py` | verify 仍要求恰好一个准确非空 Source；implement 可明确选 Source，或选 `task/default_start`（输出 Task Ref），driver 据此构造 seed=null，使用 Task.starter/合法 empty_start。不是自动选旧 C。行动仍须理由、完整 Schema/Scope/原 Task 与授权输入；共同 granted 的 `material:<id>`（含 objective）可作依据，不强造测量。新 REALIZATION.Feedback 仍保持原实测来源限制。 |
| `evaluation/agate.py` | 提取同一 `interpret_terminal` 给实时和原资格记录复用，无第二个 evaluator 判据；把已有 timing.soft_goal 准确投影到 report.extensions.information.performance_goal。最终 POST 重新收紧原 UTC/monotonic 截止，重复原 submit 只查询。 |
| `evaluation/_remote.py` | 入队/准备后在实际远端 worker 启动前重核原 UTC 截止，并收紧 subprocess timeout。过期不会启动替代 job。跨主机 UTC 时钟一致性仍是实际 profile 需验证的条件，本批无远端实证。 |
| `WRITING.md`、reference `INTERFACE.md`、evaluator `EXECUTION.md` | 更新真实已实现接口、行动语义、6-job 准备和仍未通过的实际资格。 |
| `docs/workflows/standalone.md`、`target-environment.md`、三个 schema_* README、根 README | 提供可复制命令、五类请求、target 输入、行动执行/恢复/闭包边界；只宣称已取得的工程证据，不宣称真实完整实现已完成。 |
| `test_standalone.py`、`test_agate_channel.py`、`test_writing.py` | 真实 driver/API/文件/账户/Template 配 synthetic 运输的行为验证；现有陌生 environment 拒绝测试改为明确 typed-adapter 诊断。 |

20 个本批源码/说明/测试文件的实际 SHA-256 在同名审计目录 `source-snapshot.json`。所有改动保持原 Task contract/acceptance/input/workload/timing、六个 Task evaluator 算法模块、现有 Schema/YAML、原 extract、六个算子和 FA4 源材料不变。仅 K1 `_remote.py` 执行胶水因截止约束改动；其真实新字节随 freeze_task 产生新的 evaluator/Task 身份。没有改核心协议/账本/Template parser。Qualification 入口在任何费用前强制显式 typed target，不能先付 probe 再发现目标未固定。

手工 partial seed 的实际来源仍是 B04 Codex 手写材料，**不是人类工程师或一次真实模型业务运行产物**；原 ORIGIN.md 随选定导出保留，没有伪造来源。

## 请求、行动与持久状态

`prepare` 冻结 `design.request.json`、`fixed.request.json`、`partial.request.json`、`verify.request.json`、`optimize.request.json` 五类请求。每份是实际可核 JCS/Ref 闭包，含同一父 account descriptor，没有占位哈希；原 Task、完整 evaluator、Sources、Rules/原 Template/方法正文及必要宿主胶水源码均可定位。参考只来自固定 material manifest，不扫描历史或任意目录。设计角色不读默认 R0 Design。

运行中的 `step-N.request.json` 从准确当前结果/反馈/报告构造。初始 design 的输出真正交固定 implement；optimize 的非 none 行动真正调用下一 implement/verify_only，不把未执行请求当闭环。保持 keep/none 正常结束。verify 起点只做 verify，不强加 optimize。

原实现终态/费用先结算，写作发布后只保存新 R/Scope；下一请求另固定、再准入。不同 R 的 old result 保留原主体，`current_qualified` 不贴给新 R。人工 Source 与旧 Source 都只有显式选择后才作 seed。Scope 仍 full_task_only，任何 partial scope 由原 Task adapter 在 staging/调用前拒绝。

`.runtime/demo-state.json` 只记录原 run 与选定 Ref，不是新账本。恢复包括 request-ready/首 result 前、child result 已发布但 driver 尚未选择、写作发布、新 R 未准入、原模型或 GPU unknown、child result 投影丢失。继续同一 child API/run；unknown 只结算原 operation，不能重投/换通道。账户或必需原状态丢失时不重建余额。Qualification 也沿用原 owner/request/job，保留失败/unknown 和原原始记录。

## CPU 与工程证据

主要环境：已有 `.worktrees/.tooling/k2-cpu/bin/python`，Python 3.13.12。没有下载解释器、系统改动或安装大栈。适用增量测试如下，重复覆盖不计成新增能力：

| 轮次 | 实际结果 |
| --- | --- |
| 独立 driver 主功能：完整两轮、初始四模式、同 R Source/verify、预算、unknown、四恢复边界、typed target、selected export | **14 passed / 329.52s** |
| 最终两轮不同 C1/C2、删除 result.json 后恢复、typed target 增量 | **3 passed, 11 deselected / 95.61s** |
| 既有 K5 actual fake-CLI subprocess：初始 design、同 R 两行动、陌生 target 拒绝 | **4 passed, 42 deselected / 25.16s** |
| 最终 Agate channel + 原 K1 wire 组，含6-job正反、unknown/lost ack、driver资格入口、宿主/远端截止 | **11 passed, 1 skipped / 41.63s** |
| 上一行的 SDK skip，在已有 k1-evaluation 官方 client 环境定点补跑 | **1 passed, 4 deselected / 0.25s** |
| 最终新增未绑定 target 资格准备拒绝：零收费/零POST | **1 passed, 7 deselected / 4.20s** |

SDK补跑使用已有 `atrex-gateway-client 0.14.37`，`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`；只核真实 SDK Connection/AuthProvider/request 构造配假 HTTP opener，无真实网络提交或鉴权。该环境结果不称实际服务资格。所有 synthetic report/receipt/input 测试均非真实模型/CUDA/收益证据。

测试命令：

```sh
cd /home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1
/home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k2-cpu/bin/python -B -m pytest tests/workflows/test_standalone.py -q -x
/home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k2-cpu/bin/python -B -m pytest tests/workflows/test_standalone.py -q -x -k 'two_rounds or child_result_before_driver or typed_target'
/home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k2-cpu/bin/python -B -m pytest tests/workflows/test_writing.py -q -x -k 'newly_bound_task or keep_explicit_same_r_action or design_actual'
/home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k2-cpu/bin/python -B -m pytest tests/workflows/test_agate_channel.py tests/workflows/test_evaluation_agate.py -q -x
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k1-evaluation/bin/python -B -m pytest tests/workflows/test_evaluation_agate.py -q -x -k official_auth_connection_raw_http_no_retry
```

早期一次 writing 旧诊断正则期待“explicit Contract adapter”，新代码已正确拒绝陌生格式但诊断已变，导致 3 passed/1 failed；改为新 typed-format 诊断后上表4项通过。早期 channel 测试试图在仅2-job child里再申请第3job，被正确预算拒绝；后改为实际6-job suite/独立明确测试请求额度并在最终组通过。未削弱预算或验收来消除失败。`git diff --check` 和11个Python改动文件 AST/逐行空白检查通过。没有无差别跑全库。

## 可直接重跑的无 GPU smoke

从 `/tmp` 实际执行成功：

```sh
/home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k2-cpu/bin/python -B /home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/examples/workflows/row_softmax/run_demo.py prepare --out /tmp/ks-b06-final-smoke --start fixed
/home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k2-cpu/bin/python -B /home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/examples/workflows/row_softmax/run_demo.py run --demo /tmp/ks-b06-final-smoke
/home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k2-cpu/bin/python -B /home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/examples/workflows/row_softmax/run_demo.py export --demo /tmp/ks-b06-final-smoke --out /tmp/ks-b06-selected-export
```

现有路径已存在，重跑准备需选新的空输出路径；恢复原路径用 `continue --demo ...`。这不是给真实预算换目录重试的许可。

实际 smoke 返回 status=blocked、recovery=new_request、reason=`no configured qualified full-Task evaluation channel`；原 starter 保留，calls各角色0、evaluator_jobs=0、tokens=0、active_ms=0、current_qualified=null、无 fence。准确结果/主体摘要在 `b06_k6_standalone_integration/engineering-smoke.json`。它是实际本地 API/CLI 行为记录，不是模型/GPU执行证据。选定导出成功，不包含 owner/CLI认证/452份私有 runtime 档案。

## 真实使用缺项与后续门

| 层 | 当前结论 |
| --- | --- |
| 工程/接口 | driver、三处胶水、实际后续行动、恢复/账户/选定闭包已实现并有CPU行为证据；待下一位独立审阅。 |
| CPU | 上表通过；源checkout/Python3.13；3.11和完整wheel资源发行后置，未伪称通过。 |
| MODEL | 本批0实际invoke、无业务design/optimize/implement写作记录。K3现有qwen3.8-max[1m] no-tools资格仅作已存在运行时入口。缺明确可来源核查的真实目标环境/资料，不为展示模型调用而编造目标。 |
| GPU | 本批0 jobs。会话既有官方SDK env GET401缺鉴权；已知标准config/AGATE环境未找到，未再次广搜凭证、history、其他仓库或EXP。并非断言用户服务没有GPU。 |
| 实际资格 | 鉴权/dev ACL、服务never-replay、目标GPU/栈、既有bwrap/namespace/readonly、完整stdout/原终态cleanup、跨主机UTC时钟、六作业正反suite均未在真实Agate运行。 |
| 费用 | 本批实际模型/GPU费用增量0；此前K3真实CLI reported USD0.232672750000000010属于历史，非核实账单且未加到本批。无新真实unknown/超额。 |

已经有具体入口，无需用户补写 channel/Contract/driver。后续先准备明确合法目标材料和现有官方 config（`KS_AGATE_CONFIG`），`prepare --environment ... --remote ... --start design`，然后在**同一个 demo owner**运行 `qualify`（需要时反复同原run结算）及 `run/continue`。默认父上限仍10 invokes/8 jobs/60分钟，6-job资格费用占该额度。认证或材料仍缺则保留阻塞；不得改成普通eval默认allclose、换题/通道抹去失败、复制旧pass给新R或称全部真实目标完成。

完整使用说明在工作树 `docs/workflows/standalone.md`；target CLI直接按所选原文件生成真实哈希，格式在 `target-environment.md`。公开说明、可导入选定包与私有原运行状态分开；原字节证据有必需依赖时不得用脱敏新字节代替。没有递归发布任何私有CLI资格档案/认证文件。
