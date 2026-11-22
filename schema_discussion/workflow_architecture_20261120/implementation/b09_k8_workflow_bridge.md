# B09 / K8：FA4 F1 固定 implement/verify 接线与角色投递瘦身

2026-11-20。唯一源码工作树 `/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1`，分支 `feat/schema-workflows-v1`，起始 HEAD `80e006ecd444ee6d885b53ffe9b33028ea1e505c`。本批不提交、推送或配置 Git；原 main 唯一写入为本报告。已读 AGENTS、公开 protocol/INTERFACE/runtime/implement/WRITING、B07/B08 与 C/V 的 F1/V15。未派生代理，未访问、修改或运行 EXP，未修改 AKA。原四节点 YAML、原 kernel/reference/input/shapes、K7 原材料和旧失败 capture 均保持原字节。

**结论：本批把 KS reference 的固定 F1 Source→coder→完整 C→候选/基线 evaluator→独立 fidelity→result/continue 链实际接入代码。K8.real 仍未签。** 默认零预算 prepare 是材料入口；另有明确 runtime/profile/预算请求路径及收费资格入口，缺资格时在业务调用前 blocked。本批新增真实模型调用、GPU 作业和网络提交均为 **0**。后述所有执行器、模型和资格轨迹均明确为 synthetic CPU 测试。

## 实际接口与边界

| 位置（相对源码工作树） | 实际行为 |
| --- | --- |
| `kernelschema/evaluation/dispatch.py` | 有限分派固定 row-softmax 与 FA4 F1；对应 Task 格式检查、stage 和 stage 验证。不是通用 Task 注册/表达式平台。 |
| `kernelschema/implement/reference.py` | 现有 `api.implement` / CLI 使用完整 `source_files` 验证、seed 原子安装、typed candidate 完整冻结、逐 C stage、实际 evaluator/reviewer 和原 owner 恢复。 |
| `kernelschema/runtime/claude.py` | `python_tree` 全文件角色投递、`candidate_files` 严格输出；固定格式进入 assignment/intent/reopen；仍无模型工具或宿主路径写权。 |
| `kernelschema/fidelity.py` 与方法文档 | 当前完整 Schema/Scope/root Contract+direct Design，全部实际 Python 文件；精确 file/line/quote 核验和独立 reviewer；保留有限 AST 范围与未知。 |
| `kernelschema/evaluation/fa4.py` | `stage_original` 仍只准原 C；新增 `stage_candidate` 冻结候选、固定 baseline 和当前受信 launcher；共享 `execution_materials` 校验真实 evaluator/原 baseline。 |
| `kernelschema/evaluation/fa4_channel.py`、`_fa4_remote.py` | FA4 原栈 channel、六原作业资格、两个 fresh process、完整输出/文件/Source/environment/readonly 原谓词和报告。 |
| `kernelschema/evaluation/agate.py` | 仅以有限 `fa4_probe` / `fa4_execute` 分派接入原 prepare/submit/query/settle/report；原 key、ACK 丢失、404、terminal/cleanup 和收费事实继续由原实现处理。 |
| `examples/workflows/fa4_decode/prepare.py` | 原零预算材料准备保留；新增显式 Claude profile、FA4 evaluation selection、预算、三种 source intent 及外部 seed bundle/ref 参数；准备本身零收费。 |
| `examples/workflows/fa4_decode/candidate-interface.md` | 明确 coder/reviewer Materials，给出 Model/13ABI 及固定 operator/native 观察字段契约；纯 Schema 重建不再猜隐藏模块/字段，未夹带优化实现。 |

Task 仍是共享 Contract 的有限 F1 点 `B1/Q4/L64/P1/N1/S1`，只绑定固定 `q4_direct`。结果/复核保留 `fa4_decode.shared_contract.except_f1`。root 不是 direct/Q1 fallback。改动或未知 Schema/variant 保守 unknown；不把 row-softmax 的 full_task_only 或简化 when 套给 F1。FA4 design/optimize 本批不支持；原 writing/driver 继续支持 row-softmax。

typed coder 输出仍在原 `ks.claude.output.v1` envelope 内，固定树请求必须用：

```json
{"kind":"candidate_files","content":{"format":"python_tree","entrypoint":"solution.py","files":[{"path":"solution.py","mode":"0644","content":"<complete text>"}]}}
```

上例仅列 kind/content；真实 envelope 仍必须包含 protocol/role/assignment_sha256/status/reason。文件完整、name 排序、唯一；path/mode/UTF-8/count/size 只调用现有 core `source_files_manifest` 校验。不存在第二份 Source parser/codec。绝对路径、`..`、超层目录、可执行 mode、重复、未知字段和单字符串树输出拒绝。原 single_file/candidate_source 接口保留；业务实际 capability 只对 reference/K3 扩展，不代宣称 AKA 支持。

Source 是完整快照，候选 stage 新建而非 overlay；未列旧文件消失。每文件 byte/mode/size/hash 和 C Ref 通过 stage 上传、输出和恢复核验。原 seed 安装只在未准入阶段幂等核验；损坏不会被旧 seed 覆盖。verify_only 只接受明确非空 C，零 coder/改码/promotion；必要 evaluator 和 reviewer 计入同一 owner。

original_verify、显式 seed_continuation 和 schema_rebuild 分开。重建 coder 无 Source/starter 输入；K3 不因 Task.baseline_source 元数据而展开原优化源码，并拒绝 schema_rebuild 中的显式 baseline Source 投递。下一轮只交当前最新 C，反馈不把 C 回退到 baseline/native best。显式获得 Source 的 baseline 例外仍受角色、C 身份和实际文件字节边界限制。

## 候选受信执行、资格与未签范围

`FA4Channel` 需要显式选择的原栈/原资格记录；没记录时返回 unknown admission，reference 产出 blocked，在 coder/evaluator 前停止。`qualify(context, remote=...)` 是独立明确调用，不是 prepare/admission 暗藏收费。六个原操作为 probe、完整 baseline、漏 out、改 query、容差、原 out storage 替换。它们使用原 `OwnerAccount`，保留 operation/key/reservation/locator/原终态，并在原 404 后查询同一作业；没有第二份余额或免费 helper。

资格返回原记录供一个后续固定 profile 使用；request/Q/profile 不就地热改。资格恢复核对原实现摘要和全部原记录。synthetic 即使完整跑完六个逻辑 jobs，也明确不能资格。原源码、输入、比较器、stage/settings、命令、never replay/key、bwrap、实际环境和 cleanup 均须一致；字符串 `pass` 或内部自摘要一致不是独立服务认证。

每个 FA4 execute 逻辑 job 在两个独立 fresh process 中分别执行固定 baseline 和候选，共享原判据，分别使用 compile/output scratch。预装且固定摘要的 bwrap 将 root/stage 只读、丢 capability、隔离网络，允许 CUDA device 和该 scratch。每一个 staged 文件在候选 import 前实际尝试 write-open，只有 denial 才记录权限证据。完整 framed stdout 和实际原 job settled/quiescent 条件由原 Agate 验证。

**明确的信任限制：** 此路径要求生成代码不攻击同进程 Python evaluator。bwrap 限制文件/网络，不把恶意 native 代码与同进程 tensor 比较器隔离。它不是对敌意 CUDA/native code 的通用安全沙箱。用户/实际资格需要接受并实测这一有限边界；本批 CPU 不证明目标机 namespace、CUDA、readonly、完整 stdout 或清理能力。

原比较器和 input 不变：13 ABI、seed20261120、query BF16 randn×1.22→FP8 后 KV×0.96→FP8、workspace413138944、有效 page0。返回值与调用前原 out 各自做 FP32 finite、原 shape/dtype、`0.06+0.04*abs(reference)` 逐元素判据；alias/original storage/view、readonly 输入对象/字节和 NaN poison 每次检查。原 native 1e-2 probe 不替代 Task 容差。

冷 import/constructor/首完整 forward+sync 单独记录，之后五 warmup、40 全 forward+sync 样本，每次之前/之后核相同 native launcher、六对象/signature/thread/device/noncapture。baseline 和候选的原生成输入摘要、实际环境必须相同。任何历史 kernel-sum、旧30shape 或1.1126x 均不签当前收益。本批没有“原码更快/更慢”的真实数据。

真实目标仍固定 CuTe DSL4.6.1、SM103、torch2.9.0+cu130/CUDA13.0、cuda-python13.1.1。入口另记录实际 Python executable/hash、nvidia-smi/driver、TVM-FFI 版本/module hash 和固定 `/usr/local/cuda/bin/nvcc --version`。它不安装环境；本会话没有这些实际资格，也未继续找认证/history/EXP 或用 H20/Triton 替代。原 K7 许可/notice、匿名固定提交取得和硬件文档缺项仍存在。

fidelity 读取真实所有 Python 文件，不拿 `kernel.py` 冒充树。每个接受的位置必须属于当前 C 的文件并逐行 quote 相同；报告留对应定位引用。完整 current Schema/Scope、root Contract 与 direct Design 都进入义务。N1 不签空 parity、TMA padding/page-tail、TMEM fences/deallocation 或 output staging；reviewer 不能建立时仍 unknown。row-softmax 原 R0/Rd 同 C 和外部 softmax/省 max 反例继续有独立测试。

## 实际请求命令

默认材料准备（CPU、零预算）仍可执行。明确 reference 请求另用已有配置和预算文件：

```sh
python -B -m examples.workflows.fa4_decode.prepare --out /tmp/fa4-reference \
  --claude-profile /selected/claude-profile.json --limits /selected/limits.json \
  --evaluation /selected/fa4-evaluation.json --intent original_verify
python -m kernelschema implement --request /tmp/fa4-reference/request.json --out /tmp/fa4-run
python -m kernelschema implement --continue /tmp/fa4-run
```

`--intent schema_rebuild` 无 seed；`--intent seed_continuation --seed-bundle <bundle> --seed-ref <ref.json>` 显式交付所选完整 Source。配置来自 `ClaudeProfile.as_dict()` 和 `fa4_channel.selection(remote=..., qualification=...)`，无 profile 检查探测或隐式模型调用。真实 `api.implement(request=..., out=...)` / `continue_run=...` 与 CLI 同入口。

显式资格 API（**本批未真实执行**）：先用相同 owner 创建/打开准备 context，调用 `kernelschema.evaluation.fa4_channel.qualify(context, remote={"channel":"agate-dev-bwrap-v1","gpu":"L20D","python":"/selected/original/python","bwrap":"/selected/bwrap"})`。首次 probe 返回实际 bwrap/environment，最终原记录用于新 selection。未知作业只重开原 context 后调用相同 qualify，不换 run/key/余额。需要足够的六 job 原预算与原环境，零预算 prepare 不能替代。

资格后必须复用原 prepared 的 case Refs，由 `freeze_request` 固定新的 profile/请求，parent Ref 指向原 OwnerAccount descriptor，并在 `use_authority(owner.path)` 下准入新调用。重新 `freeze_case` 会产生新的 Task 身份，不能借旧记录资格。README 已给同一 bundle/case/owner 的具体代码接线；初始总预算须覆盖资格和业务，剩余额度不足仍由原 owner 拒绝。

## 新增协调项：真实输入按权限瘦身

主代理报告的第一次 AKA 真实 coder 使用旧完整请求，1,012,929 bytes 输入、120s 超时，没有完整 tokens/费用终态。本批不重写其 capture/配置，不把失败算零费或 pass，也没有新增真实调用。仅只读打开已指定的失败请求/角色输入，选定闭包复制到新 `/tmp/ks-b09-role-projection-u4ms0t47/` 后生成当前 K3 projection。

不修改 Rules wire grammar、完整依赖闭包、原 tool version/source 或 host 校验。仅省去 Rules 中明确声明的工具入口正文，以及已声明当前 Template 工具的、路径/版本/实际字节与固定安装匹配的29份实现。未知 `.py` 仍是规则正文；显式 Materials/RoleDelivery 授权工具代码仍可读。每项先独立私有 evaluator/角色/Source/UTF-8 校验，再按实际 byte+hash 去重。

每个原 ID/category/ref/authority/size/hash（Source 另有 path/mode）保留；重复正文 `text_alias_of` 指向同 payload 的唯一完整正文。`runtime.material_text(payload,id)` 做直接 alias/hash 验证，无文件访问。single_file `current_source.text` 优先保留，objective 在去重前绑定。新增 system prompt 说明 alias 和 host implementation；这是当前 CPU 增量，旧真实 CLI 捕获未测试该指令。

| 实际文本分类 | 旧 bytes | 新正文 bytes |
| --- | ---: | ---: |
| assignment | 14,474 | 18,153 |
| Rules | 687,103 | 277,142 |
| role Materials | 276,673 | 18,566 |
| 当前 Source | 377 | 377 |
| 明确额外交付 | 358 | 358 |
| **完整编码输入** | **1,012,929** | **354,400** |

29 host 正文省去409,961 bytes；13 alias 共享258,107 bytes；新增加 Task workload/environment 的3679 bytes。原67项全部保留（另增两项），所有旧非host正文按原 ID 解析后逐字相同；没有截断 Contract/Schema/方法/语义文档。只读断言旧输入字节不变。

原输入 SHA-256 `d03d9235e0e96fd1e50021578fe543f11fe5ce1bbcfdefc470d524310123183e`；新 CPU projection SHA-256 `989818c494f6b5f541e5e41c5cc898ff6081f3f9ec6cd3d9c977127466d62649`。新 `summary.json` 保留统计，`new-role-input.json` 保留原文投递；这些是本地权限投递事实，不是新的真实模型证据。

## 同批独立审阅反例与写作 handoff 小补充

AKA 审阅者提供的旧固定 KS 反例确认 `evaluator.inputs` 不在原私有集合，18,977 bytes 可被 coder Materials 直接获得。现由 `materials.private_evaluator_hashes` 统一定义 **files + entrypoint + inputs**，K2 与 K3 共同使用；timing/environment 仍可明确公开投递。直接 Ref、同 bytes 重定位、Rules/RoleDelivery 的实际正文先检查，再进行 alias/host 省略。baseline 例外只准实际已交付的 row-softmax 公开 baseline Source；inputs/entrypoint 优先拒绝，伪造 Task.baseline_source 包装其它私有实现不能获得授权。K3 原身份另外固定共享 `materials.py` 的实际字节摘要。

对真实失败请求的复制闭包重现了原 `inputs.json` Ref（SHA-256 `cbdfb55ad958788552a4b3927accc5928e89d42480e111b0694299bd325a9f7b`）：直接/alias × validate_request/prepare_role_input 四条均拒绝；记录为上述投递目录的 `privacy-regression.json`。旧输入再次逐字核对未变，最终权限修补后重新生成的投递仍与354,400 bytes/上述新 SHA 完全相同。没有读取或修改费用/认证历史。

A6 指出的 writer floor 缺口也在 KS 唯一写入处修补。新增固定 `profile.parameters.writing_finish_floor`，默认 `budget.cost()` 全零。必须为完整 flat cost 向量，所有分量非负 int、拒 bool/null/缺项/未知项，model_calls 等于四角色 calls 之和。planner 的 floor 是此向量加原 reviewer 最小 call/time；reviewer 的 floor 保留此向量。原 reserve 同时检查 request/parent/native，外壳自己的 writing limits 也必须包含该容量。没有改 authority/version/账本，没有向 writing state 塞占位操作。它不代准入下一请求；写作结束后外壳仍在同一 owner 检查/准入后续固定工作。

另保守修正：F1 在缺资格 blocked 或尚无 fidelity 时也列真实 task_uncovered；native 条件未成立但已获得的数值调用未失败，只能是 correctness unknown，不能伪造数值 fail。补充 F1 义务的 schema_locations 指向原真实 Math_reshape/Tile_pipeline/Dispatch.when 节点，不制造虚构 YAML 字段。

## CPU 验证记录

本节保留中间失败事实，不把后来修复覆盖成最初全通过。

- 初期 FA4 Source/case/fidelity 组：25 passed，54.58s。
- 第一轮新增 API bridge：4 passed / 2 failed，171.89s；两个失败为新测试仍按旧位置参数调用 keyword-only prepare_run，已改测试并定点3 passed/4 deselected，48.25s（包含新增实际 fake CLI tree 输出/reopen）。四个已通过项包括原 C verify、两轮 C/文件删除和无首 result 恢复。
- 第一轮 FA4 channel：5 passed / 9 failed，18.07s；负例意外原地改了 module fixture 的 stage manifest，后续被正确身份保护拒绝。改为深拷贝后14 passed，17.49s。
- 初始157项回归：154 passed / 3 failed，283.86s。一个是测试运行时实现文件变化触发原 identity 保护；一个是 seed 损坏诊断文字兼容（已修）；一个是 `/tmp/schema-b05-a3-venv` 的 arbitrary-cwd import 实际指向 AKA optional/KernelSchema，造成旧 checker 摘要不匹配。未改该共享 env 或 AKA；后续使用已有 `.worktrees/.tooling/k1-evaluation`，任意 cwd import 指向本 KS 工作树。
- role projection + FA4 channel 增量：19 passed，57.69s；包括实际六原 job/owner/404恢复、synthetic 不资格、两个 fresh readonly stage 的 CPU subprocess stub、原上传/帧反例、未知 `.py`/明确工具代码/私有资料先拒绝与 alias 闭包。
- 源码稳定后的正确环境轮：core/runtime/reference/fidelity/contracts/artifacts/API/role projection **160 passed，282.15s**；FA4 source/case/bridge/channel **43 passed，303.40s**；writing + 原 Agate 兼容 **74 passed，411.60s**。三组共277 passed、0 skipped，期间没有修改源码。这不是全库320项轮跑。
- 独立反例/保守表达修复后定点：**28 passed，122 deselected，259.98s**；包括 inputs/aliases、假 baseline、原合法公开 baseline、role/Source恢复、F1 rebuild/verify/blocked未覆盖域、原 Schema 位置、原 channel 六作业与数值 unknown、默认reference等。
- writing finish-floor/default/same-parent 定点：初轮9 passed/1 failed（53.60s），失败在新测试的 `OwnerAccount.register` 漏必需 root 参数；补参后的第一次重跑又被 JCS 正确拒绝 Path 对象（10.02s）。修测试为明确字符串 root 后，该同 parent/reviewer floor/下一 coder 预留实际路径 **1 passed，51 deselected，10.47s**。生产 floor 代码未因测试调用错误而改账本/接口。
- 树 structured 输出非法 Unicode surrogate 补充拒绝后：typed output + 实际 fake CLI tree/reopen **2 passed，6 deselected，24.16s**。
- 最后补入公开候选观察接口后：准备/来源角色边界、纯Schema两轮完整C与删除、真实channel缺资格blocked **4 passed，13 deselected，116.78s**。接口只给13ABI/模块和观察字段，不给原优化代码；字段自报不是原语、生命周期或fidelity证明。

稳定轮及定点命令均使用已存在的专用解释器，没有下载或安装：

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k1-evaluation/bin/python -B -m pytest -q \
  tests/workflows/test_claude_runtime.py tests/workflows/test_reference.py tests/workflows/test_fidelity.py \
  tests/workflows/test_contracts.py tests/workflows/test_artifacts.py tests/workflows/test_api.py tests/workflows/test_role_materials.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k1-evaluation/bin/python -B -m pytest -q \
  tests/workflows/test_fa4_source.py tests/workflows/test_fa4_case.py tests/workflows/test_fa4_bridge.py tests/workflows/test_fa4_channel.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k1-evaluation/bin/python -B -m pytest -q \
  tests/workflows/test_writing.py tests/workflows/test_evaluation_agate.py tests/workflows/test_agate_channel.py
```

最后仅按 `-k 'private or baseline or role or rebuild or original_verify or production_channel or method_locations or complete_measurement or six_job or paired or machine or default_factory or first_ready'` 以及 writing `-k 'finish_floor or same_parent or complete_public_task or design_actual_api'`、typed/fake CLI 相关项补查，未重跑全套。

最终实际工厂 smoke：`/tmp/ks-b09-fa4-final-smoke-hvccism8/summary.json`。使用正式 prepare、已存在公开 Claude profile、reference + FA4Channel 工厂和**非零** reviewer/evaluator 预算，qualification=null；没有注入 fake channel。实际 API 返回 blocked/new_request，明确缺原栈 probe/baseline/四反例资格；完整原 C 为 `sha256:c8395325f884d636f9c2e0dd3b622680848e270c40cb55a8b9fb178190116611`，task_uncovered 为 REST，全部 calls/jobs/tokens/active_ms=0、unsettled=[]，没有导入 torch/CUDA/原Source模块，也没有选定或调用 transport。

continue 保留相同 subjects/status/counters 和 empty usage-index 摘要；当前 reference 对这种非终态 blocked 重发一个新的空 usage-index Ref 路径，不能宣称 result 字典逐字段相同。这是在初次 smoke 的过强 equality 断言失败后按实际状态检查确认的事实，初次目录 `/tmp/ks-b09-fa4-reference-smoke-pji313_6/` 保留。没有因此修改账本/费用或强行把 blocked 当 completed。最后源码 AST、本批 Markdown 本地链接、`git diff --check` 均通过，HEAD 仍为起始80e006e；所有改动未提交，留主代理固定。

## AKA 最小后续兼容清单与停止边界

本批不修改 AKA。它需要自己的固定 Task dispatch、typed 全文件 K3/native session 输入输出、完整 source install/freeze/delete 与 C→实际 job stage 文件/报告映射，采用原 OwnerAccount 的费用/settlement并固定 identity。明确 old C verify 零 coder/改码/promotion，seed续改只交所选 C，重建不读 baseline/history。可明确采用新的 KS FA4 channel，或为自己的等价有限权限原栈取得独立资格。新的 alias 输入正文可用 `material_text` 解析；不得将 metadata-only host 条目当作空的语义规则，也不得忽略 alias 引用。

A6 可在固定 writing profile 中设置 `writing_finish_floor` 并让请求/parent/native limits 包含该容量；不要复制 KS 私有集合或插入占位 budget operation。它需要消费本批公开的候选观察接口，不能让重建 coder 猜私有字段或用 readiness 自报代替独立审查。

尚未支持：任意源树/语言、FA4新设计或未知variant、F2/F3/全shared Task、FA4设计优化泛化、敌意native安全沙箱、远端迁移私有运行、丢失原owner/anchor后的重建余额，以及实际原栈/模型/性能/fidelity/发行许可签收。没有接入的 AKA 代码与没有取得的真实资格分别列出，不能统称环境未验而遮盖接口缺口。
