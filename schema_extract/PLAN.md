# Schema Extract 第二轮审计与实施计划

审计基线为 2026-11-03 的 `2.0.4`，本地 HEAD 与远端 main 均为 `f984cc80a63e71a09f446627143e343f30940536`。原 A–F 计划可用 `git show f984cc8:schema_extract/PLAN.md` 查看。下文 R1–R5 是本轮开发里程碑，算子提取仍然按 P1–P6 运行。代码位置按审计提交给出，工具归目录后仍可追溯。

## 验收到什么程度

| 范围 | 本轮核对 | 判断 |
| --- | --- | --- |
| 单条命令与真实产物 | 已发布 GDN root 和 dispatch 的 SHA-256 都与交付凭据一致，重新执行 `check_yaml.py --skip-hardware` 退出 0。比较基准 `relative_to=candidate`、host 的对齐拒绝和 BF16 中间舍入均可在固定材料中定位。 | 单 root 的真实闭环成立。7 个会话、9 次调用来自远端验收记录，本机没有该 v2 run 的原始会话日志。其 P5 为 skipped。 |
| 并行与阶段边界 | 受控 CLI 测试实际启动独立进程，覆盖兄弟同时写、同时审、孙节点先于另一兄弟启动，以及计划修订后的迟到结果。P5 私有目录只包含 root、祖先和目标文件。 | 无固定 jobs 或 reviewer 槽的调度已实现。仍缺真实 Codex 从源码派生多个节点并发布的证据。 |
| 全量回归与输入 | 本机 macOS、Python 3.12.13，82 项提取回归全部通过（740.957 秒），147 项检查器测试全部通过（6.357 秒）。当前 catalog 的 44 个 ID 都能解析绑定。基线 147 份 Markdown 中的 1,505 条本地链接（含标题）和 14 个本仓历史 Git 目标也重新核对通过。 | 普通路径和大量恢复边界已有回归，可作为修改基线。下面的反例是额外运行的探针。 |
| 旧任务 | 本机五个真实 v1 checkpoint 仍在。在仓库外临时运行区执行 `import-legacy --queue`，四个成功排队，一个失败。 | 已验证的范围是读取和导入草稿。导入没有继承旧批准，也没有启动新提取。 |

反例使用仓库外的临时目录，主要沿用 `schema_extract/tests/test_c_independent.py` 的 `IndependentFixture`，运行真实 Runner、受控 CLI 和冻结检查器。测试替身的批准特意模拟 reviewer 漏判，用来区分程序保证的事实与依赖人工判断的事实。此次没有发起新的真实模型提取。

值得保留的设计有三处。其一，源码与规则分别冻结，草稿 checkpoint、独立批准和合入是三个不同事件，子节点只从已审父节点出发。其二，Runner 单写者提交本节点差异，能让未受影响的兄弟继续，也能拒绝过期结果。其三，原 reviewer 有一次受限裁决，最终产物由裁决者修改时才另开一次独立签收。P1/P2 续用原角色、其他阶段与分支换上下文、终审先审后修、硬件占位和离线检查均继续作为验收条件。[真实验收记录](README.md#验收记录) · [并行实现与进程测试](https://github.com/bcacdwk/KernelSchema/blob/f984cc80a63e71a09f446627143e343f30940536/schema_extract/tests/test_d_independent.py#L197-L244) · [发布凭据](../schema_wiki/extract-v2-acceptance/gdn_postconv_mtp_bf16_state_fp32/.extraction/receipt.json) · [GDN 的实际判据](../schema_wiki/extract-v2-acceptance/gdn_postconv_mtp_bf16_state_fp32/root.yaml#L158-L178)

## R1 · 让每次调度都有确定的出口

**已复现。** 在一次已结束的 agent 调用中，仅让写 `event.json` 抛出 `OSError`。工作线程退出，内存队列没有收到完成事件。超过设定的算子 deadline 后 Runner 仍活着，P1 仍在途，提交 cancel 后也没有结束。另一个反例是 P4 的节点顺序为 root、c、b，c 的父是 b，b 的父不存在。当前校验遍历到缺失祖先时抛 `KeyError`，state 保持 running，resume 重放同一结果再次抛错。[完成事件与等待循环](https://github.com/bcacdwk/KernelSchema/blob/f984cc80a63e71a09f446627143e343f30940536/schema_extract/runner.py#L789-L855) · [计划校验](https://github.com/bcacdwk/KernelSchema/blob/f984cc80a63e71a09f446627143e343f30940536/schema_extract/artifacts.py#L148-L177)

**改动与验收。**

- 分开处理“通知调度器调用已结束”和“持久化审计记录”。注入事件文件写入失败、线程异常或取消，主循环必须在有限时间内返回可定位错误。能确认终止才释放对应尝试，磁盘持续不可写时也要返回错误。保留成功结果的幂等重放与原预算。
- 先检查全部直接父节点存在，再检查环和可达性。打乱合法树的输入顺序应照常通过。用同一个坏计划的两种节点顺序走 CLI 和 resume，均给出具体节点与缺失父节点，不能留下反复重放的崩溃。修订或停止沿用现有协议。

## R2 · 让错误停在有权限修它的阶段

| 已复现的行为 | 要求及验收 |
| --- | --- |
| 删除共享 Contract 的 `algorithm_requirement.output` 后，受控 worker/reviewer 让 P1、P2 均通过。P3 worker 无权改 Contract，连续四次遇到同一个问题后耗尽测试预算。另一个用例让固定验收材料写明误差相对 candidate，YAML 却相对 reference，P1 的 acceptance 锚点只引用 JSON 第一行的 `{`，最终仍能 ready 并发布。44 个 catalog 算子中，22 个绑定的验收文件显式写了 candidate。例如 candidate=1.061、reference=1 时，两种判据的通过结果相反。实际 GDN 产物的比较基准是正确的。 | P1 的 complete/approve 复用冻结检查器已有的算法区字段及关联规则，允许尚未填写的 Design。对于来源明确给出 `relative_to` 的输出，要求审阅依据落到该输出的声明行，并核对 YAML 中的实际判据。已有旧版回归可复用。缺失必填项、引用文件开头代替声明行的反例在 P1 得到修正或阻塞。保留未知比较细节的写法，无需实现通用自然语言公式证明器。 |
| 多节点清单在 P4 获批，但已审 root 没有 `dispatch.when`，P5 才报 root 缺少 dispatch，子节点耗尽预算。另一次完整受控运行中，计划和 `leaf.yaml` 的父节点为 `a`，索引却写成 `root`，仍然发布成功。现行索引校验只比对 YAML 链接集合。 | P4 按计划核对已有 root 的身份和选择条件，缺失时定位上游修复，再开放 P5。树、链接与父子差异由已审结构化清单生成或做确定性核对，精确条件仍在 YAML。单 root 索引留选择理由、链接和实际覆盖缺口。保留 GDN 对齐缺口，避免再复制流水线正文与通用规范。不要直接改已签收的发布物来刷新凭据。 |
| 派生节点重新序列化整个父 YAML。根中合法文本 `math: "1e3"` 通过检查，复制后变成 `Decimal('1E+3')`，派生检查退出 1。`"0o12"` 同样可能变成整数。FA4 示例的 Design 加 Realization 在相同语义下由 204 行展开成 531 行。 | 优先只更改复制文件的元数据和 Contract，保留可读的原 Design。若需要重写 YAML，用冻结解析器读回核对类型和值，覆盖数值形文本、精确 Decimal、alias 和多行文本。在 P5 初始化、旧稿导入及裁决选稿路径共用正确的写法。用上例证明根和派生公式都保持字符串并通过检查。 |

现行 [P1 回执验证](https://github.com/bcacdwk/KernelSchema/blob/f984cc80a63e71a09f446627143e343f30940536/schema_extract/receipts.py#L225-L243) 能定位来源文件。P1 还可复用 [Contract 结构规则](../schema_template/tools/rules/structure.py#L16-L36) 和 [旧版比较基准反例](../schema_extract_old/issues/archive/extract_workflow/contract-comparator-evidence.json)。P4 与 P5 的反例分别落在[索引门槛](https://github.com/bcacdwk/KernelSchema/blob/f984cc80a63e71a09f446627143e343f30940536/schema_extract/artifacts.py#L166-L196)与[父文件复制](https://github.com/bcacdwk/KernelSchema/blob/f984cc80a63e71a09f446627143e343f30940536/schema_extract/store.py#L338-L370)。旧版也有[以冻结解析器核对回写类型的做法](../schema_extract_old/tools/review_context.py#L64-L81)。

## R3 · 整理工具目录并守住冻结入口

当前 `schema_extract/` 根有 19 个 Python 文件，共 4,068 行生产代码，`runner.py` 占 978 行。测试和夹具另有 3,606 行。参照 [`schema_template/check_yaml.py`](../schema_template/check_yaml.py)，新版根目录只留一个 Python 入口 `extract.py`，移除根 `__init__.py` 和 `__main__.py`。`extract.py` 直接调用 `tools/` 中的 CLI，版本号也在工具层单点定义。README、PLAN、阶段页、tests 和 issues 各保留现有用途。

目标命令为 `python3 schema_extract/extract.py extract --id <算子> --output <目录>`。`status`、`resume` 等现有子命令也通过同一脚本调用。

下表的四组职责用于决定哪些脚本应合并，不要求再建四层包。

| 职责 | 应当放在一起的逻辑 |
| --- | --- |
| 命令与运行 | CLI、只读巡检、就绪节点派发及单写者提交。让调度循环只管理调用生命周期和提交时机。 |
| 阶段与产物 | 阶段定义、允许读写的范围、prompt、回执和产物验证。阶段页保有写审方法，代码保有可机械判断的规则，消除两处各自维护的阶段权限表。 |
| 证据与持久化 | 输入绑定、文件读取、快照、checkpoint、发布和旧版导入。统一源码和规则根目录的定位。 |
| Agent 与恢复 | CLI 会话和进程、事件交付、中断对账与恢复。保留精确 UUID、进程归属和终止确认。 |

**迁移时一起验三件事。**

1. 冻结清单包含 `extract.py` 及 `tools/` 中的运行代码。同步调整执行器校验、reader 路径、巡检和入口跳转。Runner 按已验证的快照清单选择入口。新快照直接运行其中的 `schema_extract/extract.py`，2.0.4 快照仍从它自己的 `rules/` 目录运行 `python3 -B -m schema_extract`。从仓库外以脚本绝对路径验证 `--help` 和 `--version`。显式指定仓库与运行区，验证新旧任务的 `status/resume`、冻结 reader 和检查器，并覆盖暂停、取消、发布与巡检。根目录的 `*.py` 必须恰好只有 `extract.py`。手册和测试中的新版示例切到脚本，历史验收记录保留当时的命令。[冻结清单与校验](https://github.com/bcacdwk/KernelSchema/blob/f984cc80a63e71a09f446627143e343f30940536/schema_extract/snapshots.py#L21-L31) · [冻结入口](https://github.com/bcacdwk/KernelSchema/blob/f984cc80a63e71a09f446627143e343f30940536/schema_extract/cli.py#L65-L74)
2. 最早的真实 `bf16_grouped_gemm_masked` v1 快照不接受 `status --full`，普通 `status RUN_ID` 退出 0，且返回与原 task.json 完全相同的整份记录。按冻结版本支持的参数取回状态，继续逐项核对身份、来源、checkpoint 与终止证明。两代真实接口都要通过。没有原 Mac 状态的环境可使用历史 `2574b7d` 的普通 status 与 `69e3027` 的 --full 行为建回归，不能用“命令失败即读当前状态”绕过验证。[旧稿导入入口](https://github.com/bcacdwk/KernelSchema/blob/f984cc80a63e71a09f446627143e343f30940536/schema_extract/legacy.py#L44-L70)
3. 两个输出不同、输入完全相同的排队请求同时发布同一快照时，本机稳定复现一个成功，另一个在 `freeze_inputs` 的 rename 处收到 `PermissionError`。增加竞争用例。先成功者发布，后到者验证同一不可变快照后复用，两个 run 保持独立。其他权限或 I/O 错误仍如实报告。[快照发布](https://github.com/bcacdwk/KernelSchema/blob/f984cc80a63e71a09f446627143e343f30940536/schema_extract/snapshots.py#L135-L156)

P4 阶段页引用的可选 FA4 索引也未进入冻结规则清单。让需要的示例在实际冻结环境中可读，或在阶段页内放最小的格式示例。验收阶段页的导航时要从冻结树出发。

## R4 · 用同一负载验证瘦身收益

在仓库外用现有 `IndependentFixture`、平级节点 `b_0…b_n` 和受控 CLI 运行，统计 `_protected` 的调用次数及实际交给 SHA-256 的字节。下表为单次本机样本，MB 为十进制，prompt 字节是 UTF-8 文件大小。

| 派生数 | Agent 调用 | 全运行目录扫描 | 守护扫描累计哈希 | 最大单条 prompt |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 9 | 37 | 8.2 MB | 21.1 kB |
| 4 | 17 | 69 | 31.5 MB | 30.8 kB |
| 10 | 29 | 117 | 95.7 MB | 48.0 kB |

- 保留越权写入检测，减少反复读取已经封存的历史日志与 checkpoint。在同机用基线提交及修订版的相同 0/4/10 负载报告前后字节、扫描次数和耗时。以 10 分支重复哈希量至少减半为本轮优化目标，结果未达到时交代瓶颈。用同长度内容替换、保留 mtime、增加文件、符号链接和并发控制请求验证检测仍有效。[现行完整性保护](https://github.com/bcacdwk/KernelSchema/blob/f984cc80a63e71a09f446627143e343f30940536/schema_extract/runner.py#L78-L119)
- P5 现在会收到全计划、各阶段批准以及所有已审节点记录。将普通分支的定位信息收束到本节点、实际祖先、P4 交接、相关依据和当前反馈。终审按需读取整树，裁决按需读取候选。对固定依赖链增加无关兄弟时，P5 prompt 不应随兄弟审批正文增长。减少模型抄录程序已知字段的工作，同时维持当前回执与产物的身份绑定、源码引文核验和独立审阅。[现行 prompt](https://github.com/bcacdwk/KernelSchema/blob/f984cc80a63e71a09f446627143e343f30940536/schema_extract/prompts.py#L42-L67)
- Supervisor 手册保留入口、参数、状态解释、停止与恢复。重复的机器协议说明就近放在工具维护说明或阶段页，只留一个定义。整理测试夹具时保留独立黑盒 CLI 的性质。报告生产代码和测试代码的前后体量，以重复职责是否消除作为取舍依据。

## R5 · 用一次真实分支提取收尾

建议在新输出目录无 seed 提取 `bf16_grouped_gemm_masked`。当前源包含按条件选择 `_f2 / _f1 / _f0` 的 host 路由，既有 v1 树可供完成后对照。节点身份和个数仍由本次固定源码及 reviewer 决定。完整 FA4 重提继续单列为长程验收。[候选 host 路由](../kernel_wiki/aka-cuda-operators-20261030/operators/gemm/grouped_masked/bf16_grouped_gemm_masked/candidate/solution.py#L2468-L2495)

1. 使用实际 Codex，经单条命令完成包含 P5 的算子。给出源与执行器身份、计划树、父 checkpoint、节点会话与调用起止、验证结果、最终 manifest 及发布位置。核对每个分支从已审父 YAML 出发、兄弟实际重叠、终审在全部节点合入后开始。若算子经源码核定只有一条路径，用确有分支的算子完成这项验收。
2. 重跑 `python3 -B -m unittest discover -s schema_extract/tests -p 'test_*.py'` 和 `python3 -B -m unittest discover -s schema_template/tools/tests -t schema_template`。保留 2.0.4 的冻结入口、旧证据和 GDN 原交付，新的结果独立发布。继续使用 `--skip-hardware`。
3. 报告各修复的最小反例及其现在的结果、真实性能对比、真实模型验收与受控测试各自覆盖什么。阻塞时保留原因、最后有效 checkpoint 和已完成的分支。

每个里程碑先复现、再修最小范围、最后独立验收。R1–R5 都维持原定的会话边界、就绪即派发、第五次退修的一次裁决以及按内容合入。
