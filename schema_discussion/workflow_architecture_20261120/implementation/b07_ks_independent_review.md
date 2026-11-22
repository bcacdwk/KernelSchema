# B07 / KS 独立工程审查与定点修复

2026-11-20。审查输入为 `feat/schema-workflows-v1` 的
`067b37d5bbaa31cad56680266ade018c17e97e8f`，唯一源码工作树为
`KernelSchema/.worktrees/schema-workflows-v1`。本代理先读取 AGENTS、现行
P/W/C/V、progress 和 B04/B05/B06/B07 交接，未派生代理，未访问、修改或运行
EXP，未修改 AKA，未提交、推送或配置 Git，未修改原 Schema/extract/六算子/
FA4 源材料或私有资格归档。

本报告是独立 CPU 工程审查，不是 K1.real/K4.real/K5.real/K6.demo 或 KS-P1
签收。本批新增真实模型调用 **0**、GPU/Agate 作业 **0**、新增真实模型/GPU
费用 **0**，没有本批真实在途操作。所有测试传输均明确标为 synthetic；
LocalCPU 的实际数值/只读执行也不代签 CUDA 或设计符合。

## 已复现并修复

| 严重性 | 原缺陷、证据与修复 |
| --- | --- |
| 高 | **D3 交接越界。** `Demo._run` 原来只拦 blocked，对已结算 implement/partial 仍进入 optimize。新增 fail/unknown reviewer 轨迹均复现 `optimize_count=1`；修为所有非 completed 结果保留原主体/状态并终止链。实现合格后仍能为既定目标执行一次 optimize，keep/none 仍停止；普通失败不再自动制造设计阻碍。 |
| 高 | **资格记录未把实际上传内容绑定到所称 evaluator。** 原 `_read_record` 只核 request 自摘要与报告/intent，未核 request.files 的真正 worker/kernel/命令。新增同一原终态下改上传 worker/kernel/command/replay_policy、同步改 request 自摘要的负例。`agate.validate_prepared_request` 现由实际 submit 和原资格记录读取共同调用，核完整文件集合/原字节摘要、规范 C、stage/settings、frozen inputs/environment、固定命令、GPU/profile、never-replay/key。未替换比较器或再造 evaluator。 |
| 中 | **Agate 非 JSON/截断/非对象返回打断恢复。** 原 ACK 为 HTML、query 为不完整 JSON、frame 为数组时可能抛 JSONDecodeError/TypeError/AttributeError/EOFError。现在缺可靠 ACK/job ID 或查询正文不完整时保留原 unknown/settle_only；有已确认原终态但帧坏时结算原 1 job、无 report。原字节保留，404/丢 ACK 不重 POST。 |
| 高 | **合法 baseline Source 被权限边界误拒。** A5 的真实 V0/预取 reviewer 整合触发：Task.baseline_source 与 evaluator/baseline.py 同字节，K3 原规则拒绝明确获授的合法 C。修复仅允许实际授予当前角色、规范 Source 身份等于 Task.baseline_source 且完整原字节相同的 Source payload；支持合法 import 后 Ref 路径变化。普通 Materials/text、其它 evaluator 包成 Source 仍拒绝，design 不默认获得 baseline。 |
| 中 | **资格与业务的状态竞争。** 原 `prepare_agate` 不取 driver 锁，也不刷新旧 Demo 实例；已开启业务后旧实例可继续资格准备。现 run/qualify 共用原 driver flock 和刷新状态；公开 `qualify(context)` 另锁原 child run。旧实例在业务已准入后拒绝；两个并发资格调用者仍只有同一组 6 jobs/原账户。 |
| 中 | **授权比较的 Python bool/int 混同。** `authorize` 和 `claim_group` 原用 dict equality，`calls=True` 可与冻结 `calls=1` 相等。已改比较 JCS 原字节，与 `check_active` 一致；失败不会 launch，原合法授权仍可用。未改变 authority 格式、VERSION、余额或账户语义。 |
| 低 | **公开接口说明过时且依赖未交付的私有规划路径。** `kernelschema/INTERFACE.md` 原说业务入口尚未实现，并把未在开发分支交付的 private planning 文件列为权威合同。现明确已实现 backend/driver，并补 `docs/workflows/protocol.md` 最小公开入口，直接指向实际 contracts/API 和已有接口。没有另造 codec、字段格式或复制整份私有讨论；发行 license/notices 仍待交付。 |

A5 提供的额外 V0/reviewer 反例已用完整实际 row-softmax Task/规范 Source
独立重现并修复。最后还补了同类 Agate 恢复检查：query 也复核原上传 request，
避免提交后文件变化影响结算；probe 若缺实际 environment/inputs/bwrap capture，
结算原 1 job 后停止，不进入下一阶段预留额度再抛 KeyError。公开函数签名未变。

## 检查事实与验证

使用现有 Python 3.13.12 专用 `k2-cpu` / `k1-evaluation`，未下载解释器或大栈。
K1 环境固定 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，官方客户端 0.14.37 已存在。

| 检查轮 | 实际结果 |
| --- | --- |
| 修复前定点反例 | **5 failed / 1 passed / 69 deselected，67.99s**：D3 两项、授权 bool/int、两个非对象帧均复现；原截断帧控制正例通过。不是一次“全通过”。 |
| 第一组修复后新增/受影响项 | **16 passed / 70 deselected，49.80s**：D3、授权、坏 ACK/query/frame、上传材料绑定、旧 driver 实例和并发资格。 |
| 全 CPU 预跑的环境故障 | **14 failed / 85 passed，199.42s，主动中止**。K1 专用 env 当时尚未 editable 安装 KS；任意 cwd `import kernelschema` 和 K3 supervisor `-m` 启动失败，不能把这些算功能通过或依赖 skip。之后仅在该专用 env 用本地 `pip install --no-index --no-deps --no-build-isolation -e .` 安装当前源码。无联网、无模型/服务调用。 |
| 唯一完整 CPU 集成轮 | **320 passed，1090.67s，0 skipped**；包含官方 SDK 运输构造和实际 LocalCPU 完整 evaluator/只读/原作业恢复，不代表真实模型/GPU。此轮在最后 baseline/probe/query 增量修复之前，保留其准确事实。 |
| A5 baseline Source 边界修复前 | **1 failed / 67 deselected，6.30s**，完整 Task 的真实 baseline C 在 reviewer/coder 明确 Source 投递处被 private-evaluator 规则拒绝。 |
| 最后两个 Agate 恢复反例修复前 | **2 failed / 26 deselected，6.19s**：缺 probe environment 引发 KeyError；原 query 不拒绝提交后改动的 prepared upload。 |
| 最终受影响路径增量 | **9 passed / 131 deselected，42.10s**：完整 baseline C 的两角色投递/合法重定位/真实 K3 fake-CLI reviewer，private text/其它 Source 反例、design 默认边界；原 6-job 正例/上传绑定/超时/截断、新 probe/query 恢复以及 group claim 的 JCS 反例。没有再跑整库。 |
| 基于公开源码的新目录安装与命令 | **通过**；详见下节。 |

第一组与完整轮命令：

```sh
cd /home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k1-evaluation/bin/python -B -m pytest \
  tests/workflows/test_budget.py tests/workflows/test_agate_channel.py \
  tests/workflows/test_evaluation_agate.py tests/workflows/test_standalone.py -q \
  -k 'authorize_rejects_python_equal or malformed or incomplete_fixed_work or qualification_binds or stale_driver or concurrent_qualification' --tb=short
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k1-evaluation/bin/python -B -m pytest tests/workflows -q --disable-warnings --tb=short
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k1-evaluation/bin/python -B -m pytest \
  tests/workflows/test_claude_runtime.py tests/workflows/test_budget.py \
  tests/workflows/test_agate_channel.py tests/workflows/test_evaluation_agate.py -q \
  -k 'explicit_task_baseline or material_grants_and_private or unstarted_group_can or incomplete_probe_capture or original_query_rejects or qualification_same_k1 or qualification_binds or missing_qualification or truncation' --tb=short
```

本次完整轮之后的新修复仅补受影响路径，不反复全跑。尚未验证 Python 3.11，
不会把 >=3.11 metadata 当作实际兼容资格。

## 新目录与可运行命令

从 `git ls-files` 明确选出 **165 个** KS/Template/docs/row-softmax/pyproject
已跟踪文件，按当前字节复制到
`/tmp/ks-b07-source-smoke-il2fg2bs/checkout`。没有复制 `.git`、private planning、
runtime qualification 归档或作者 HOME。该检查只验证支持范围内的源码/资源
子集，不把它称作完整仓库发行或 wheel 资格。

在现有 k2 专用 env 离线 editable 安装此副本，实际从 `/tmp` 执行 import
与 reference/design/optimize factory capability 查询；import guard 明确拒绝
AKA/torch/triton/cuda。模块实际来自临时副本，规则资源也定位该副本。随后运行
prepare/run/continue/export，最后在 finally 中把 k2 editable 恢复到原工作树。
原始命令/输出记录在该临时根 `commands.json`、`summary.json`。
这个安装检查发生在最终 baseline/probe/query 增量及新增协议文档之前；后续
没有改变安装/资源机制，最终增量单独由上表验证，不把该副本冒称最终全部字节。

真实本地 smoke 的结果：5 份请求可核；status=blocked、recovery=new_request、
reason=`no configured qualified full-Task evaluation channel`；所有角色 calls、
evaluator_jobs、tokens、active_ms 均 **0**；unsettled=[]；continue 与原结果完全
一致；选定 export 成功且不含 owner authority。未把未配置静默替成 fake。

可复制命令（输出需新空目录，原运行恢复用 continue）：

```sh
KS=/absolute/path/to/KernelSchema
python -m pip install --no-deps --no-build-isolation -e "$KS"
python "$KS/examples/workflows/row_softmax/run_demo.py" prepare --out /tmp/softmax-demo --start fixed
python "$KS/examples/workflows/row_softmax/run_demo.py" run --demo /tmp/softmax-demo
python "$KS/examples/workflows/row_softmax/run_demo.py" continue --demo /tmp/softmax-demo
python "$KS/examples/workflows/row_softmax/run_demo.py" export --demo /tmp/softmax-demo --out /tmp/softmax-selected
```

上述缺配置 smoke 是当前实际支持行为。具备明确 target 原材料、既有官方
认证、remote GPU/python/bwrap 后，公开文档已经提供 `target`、
`prepare --environment ... --remote ... --start design`、同 owner `qualify` 及
`run/continue` 入口；不需要另写 adapter。此处没有执行这些收费步骤。

## 保持的边界与未签收项

- D1 保持：核心、reference/design/optimize 不依赖 AKA；只有显式 implement
  backend=aka 才 lazy import 外部 adapter。没有修改 AKA switch 或另一顶层循环。
- 明确支持 Linux/source checkout/editable、单 row-softmax Task、full_task_only、
  single_file/kernel.py、显式 Source/default_start/verify_only、一个 owner、
  固定 backend 和至多一次 optimize。新 R 的实际资格仍必须重新评估；慢但五项
  合格可以 completed，soft_goal=unmet 不阻止完成。
- 普通失败/unknown 不升级 planner。本有限 driver 尚无独立认证设计阻碍的
  专用输入接缝，故只采用“当前实现合格”的 D3 路径；不猜测错误属于设计失败。
- 新目录 smoke 不要求私有规划文件。最小公开协议入口已补齐，实际本地链接
  检查通过。**发行阻塞仍有**：KS 项目根 license/THIRD_PARTY_NOTICES 与所选资料的分发归口。
  `task/materials.json` 多条仍明确 `KS project license unresolved`。本代理不代
  权利人选 license，不把私有计划/日志整体发布为解决方案。
- MODEL/GPU/真实业务门仍未通过。现有 K3 四次运输资格不能签业务 design/
  optimize/implement；本会话未定位 Agate 认证和实际 target 原材料，不等于
  服务不存在。真实 auth/dev ACL、never-replay、CUDA/bwrap/readonly、完整输出/
  cleanup、跨主机时钟及六作业正反 suite 仍必须取得真实记录。
- Agate 原记录校验是原字节/主体/传输配置的一致性检查，不是服务端签名或
  独立计费证明。原完整 capture 与可信来源仍是实际资格前提，不能提交手编的
  “pass JSON”代替服务执行。
- **中 / 未修 bootstrap 恢复边界：** core `api.prepare_run` 已发布 run anchor、
  backend 尚未建立 `.runtime/state/budget.json` 时中断，`continue` 会报
  `owner binding missing; continue cannot allocate a balance`。实际零调用复现目录
  `/tmp/ks-b07-init-gap-9xpwhbcx` 中父账户 operations={}、anchor 存在而 binding
  不存在。此复现不意味着真实缺失记录时可以一般性断言零费：同样的文件缺失
  也可能是原状态丢失。已按主代理决定收窄 standalone 恢复表，保留目录，
  不凭缺 binding 重建账户、不自动换 run 重试。已有费用/operation 持久后的
  恢复保证与这个初始化窗口分开；未来若要自动恢复，先需可信初始化 handshake。
- GPU 实测、Python 3.11、完整 wheel、K8 的 FA4 Task/原栈/多文件 Source 等
  均未被此批扩大或宣称完成。未读取/复制私有 452 capture。

## 下游交接与实际改动

A6 必须消费 D3 gate：实现合格或有独立可核对设计阻碍才能写作；ordinary
partial 不能靠新 Q/R 进入 planner。非 none 的行动仍由原外壳固定新请求、
同 owner 准入；default_start 是明确 Task marker，verify 必须非空准确 C。

K4/K5/A5 账户 API、authority VERSION 和公共请求/结果 codec **没有迁移**。
`authorize/claim_group` 现在拒绝 JCS 不同的伪造预约；原 int 类型预约行为不变。
Agate 资格入口仍是原 `qualify(context, remote=..., transport=...)`，只是原 run
互斥与上传内容绑定加强。K8 应复用这些预算/恢复边界，不复制余额或把完整
row-softmax evaluator 换成默认 allclose。

A5 的 V0 reviewer 与显式 baseline seed 现可通过 K3 原入口投递；无新参数。
baseline 的数值通过仍不能当 R0 fidelity pass，原 native baseline 与当前实现
资格仍分列。运行时新增的只是严格角色/身份/字节一致的 Source 例外，绝非放开
原 Materials/evaluator 信息边界。

所有修改都会按既有设计改变相应实现摘要（budget/runtime/channel/driver 与
包含它们的 Rules/profile）。已冻结真实运行必须保留原固定源码做原操作恢复，
不能热改 run/profile/authority 以绕过身份检查。此批没有既有真实业务作业要迁移。
新准备请求使用最终当前字节；后续 root 固定提交和 AKA gitlink 由主代理负责。

最终 `git diff --check`、全部 15 个改动/新增文件的逐行空白检查、Python AST
检查、所有改动 Markdown 的本地链接检查均通过。实际文件及 SHA-256 如下；
不含本审计报告，且所有改动仍未提交：

| 文件 | SHA-256 |
| --- | --- |
| docs/workflows/protocol.md | c8622bf543aa7631b3e34a4955b07a8cad91b30e042bc52aa38bcba9bc3bfcd6 |
| docs/workflows/standalone.md | 90af7b311dc5e6024012140f1b0a0eca2f8a793f1660fe68cff156b6b787b5ea |
| examples/workflows/row_softmax/driver.py | ed92f5baffd630eb088f05050f298dbd2f147161e9cd1fad587e0521bda7a110 |
| kernelschema/INTERFACE.md | f350e1d455e8e3e2181c6f885b007df672b224e654c784059b13a9c21e02f95e |
| kernelschema/budget.py | e1fe196b566ad9cd1deae1e37df76bcee926b1c6e777b52748173563e0c0e497 |
| kernelschema/evaluation/agate.py | 5312c25cb71c81ed0c089b8ae041b96d6f3d955cb07f1d8a7c5d6fdbfe8563e1 |
| kernelschema/evaluation/channel.py | 317f07944febab72d000fc59e5bf797edab734ac8e25a869db6c5e6f0b4beef0 |
| kernelschema/implement/INTERFACE.md | c7e85f915e92392d52bf96aa66f999f4518d5a5609c6cdef533af5942a847e97 |
| kernelschema/runtime/INTERFACE.md | e50bd16fb7f3dd8f9385b45974c8a3bd9d37b76a090ed89a0be8f1770340d4f3 |
| kernelschema/runtime/claude.py | c5f23d789b70d3f1bf8d42210d17ab1c9980dd966cbe0b7c4acb22593eb70ad9 |
| tests/workflows/test_agate_channel.py | 45bcb8411679b8b75bcce22bdca3c1a96a4e4af9d9bf12f6b3cce70fa4ec8550 |
| tests/workflows/test_budget.py | df39ba4d4744b17a54236d7f10dae98b332b8bba1b979e74749009f73de644d3 |
| tests/workflows/test_claude_runtime.py | 7a02c3f188fead22b95d673f2d16d92f3a2e30eeb2f0f6934b9b71c392c081a4 |
| tests/workflows/test_evaluation_agate.py | 4bef44e6d9a1f0bd517d239380744b44c934599ae327225ac76be1b9e64aae52 |
| tests/workflows/test_standalone.py | ffd46f8dcae0070a3910e2c1091a5141f88bbf34fc1d41745d37cf3d0fa8ae1c |
