# B08 / K8：FA4 F1 KS 侧部分工程交付

日期：2026-11-20。KS 工作树 `/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1`，分支 `feat/schema-workflows-v1`，开始及结束 HEAD 均为 `b1049e19403238125d4210dd275a8f3e2874ba47`。已读 AGENTS、公开协议、core/runtime/writing/implement/evaluation 接口、C、V 的 FA4/V15、K7 报告和 progress。未派生代理。

**签收：K8 KS 侧部分 engineering 完成；FA4 的 `api.implement` 调用链尚未接通；K8.real 未验证。** 本批交了可调用的准备、完整 Source codec/安装、原码 evaluator 和观察身份映射，不是第二轮材料预审。没有扩大任一业务 backend 的支持表，没有用 unknown 代 pass。

## 实际文件与 API

下列路径均相对于上述 KS 工作树；原 main 本批唯一写入为本报告。

| 文件 | 实际实现 |
| --- | --- |
| `kernelschema/contracts.py` | 增量有限 `python_tree` Source 语法；原 `single_file` 语法保持。 |
| `kernelschema/artifacts.py` | `source_files_manifest`、`Publication.put_source_files`、`Bundle.source_files`；闭包/导入跟随全部文件；`install_source` 兼容两格式；`verify_source_directory`、`replace_source(..., previous=...)`。 |
| `kernelschema/materials.py` | 固定请求验证完整 Source，记录实际所需格式；profile 与实际 backend 必须同时支持所需格式。 |
| `kernelschema/api.py` | 仅将原 intent 的 Source 完整性验证改为 `source_files`，没有改预算/恢复状态机。 |
| `kernelschema/evaluation/fa4.py` | `originals`、`freeze_case`、`FA4Task`、`freeze_request`、`stage_original`、`freeze_observation`。使用同一 core codec 和原 Template resolver。 |
| `examples/workflows/fa4_decode/prepare.py` | 独立 CPU 准备 CLI：实际固定 Git blobs → 正式协议对象 → 完整 stage。 |
| `examples/workflows/fa4_decode/evaluator/entry.py` | 固定原码的独立原栈入口，复用真实 reference/helper，完整副作用和冷/稳态计时逻辑。CUDA 只在显式 execute 中导入。 |
| `examples/workflows/fa4_decode/task/{acceptance.md,workload.json,timing.json,environment.json}` | 固定判据、shared Contract 域与 F1 分区、40 次 full-forward 计时、原栈要求与真实缺项。 |
| `examples/workflows/fa4_decode/{README.md,fidelity-obligations.md}` | 可执行命令、精确格式/能力限制、来源分界和后续接线清单。 |
| `tests/workflows/test_fa4_{source,case}.py` | 20 个专属 CPU 测试，包括实际 Git/Source/Task/stage 与明确 synthetic 的 helper/中断反例。 |
| `docs/workflows/{protocol.md,cases/fa4_decode-admission.md}`、`kernelschema/INTERFACE.md` | 公开 Source 规范及 case/API 更新；协议单元变更已由 supervisor 明确授权。 |

原四节点 YAML、原 kernel/reference/input/shapes、原提取结果和四份 K7 材料记录未修改。没有复制整份 AKA 或大源树；准备产物只冻结所选 27 个原件及当前小型 case/evaluator 文件。

## 已闭合的 KS 工程边界

### 1. 正式 F1 Task、完整原件、来源绑定

`kernel_source_commit=3d64ad35900a0e6a66d6f93da34108fbceba0daf`；`schema_commit=484902855fe7be82a74fe7c394503d1d7d222dbb`。准备器逐件读取固定 Git 对象，核 Git blob、SHA-256、size、mode。保留四节点 root/q4_direct/q4_cluster/q1_global 和实际 dispatch，没有回退源码提交中的旧七节点。

Task 保留原 root 共享 Contract，baseline 是完整原 v55 Source，starter 为 null。Scope 只选 `fa4_decode.f1.q4_direct.b1.q4.l64.p1`；`task_uncovered=[fa4_decode.shared_contract.except_f1]`。F1 是原 Schema 代表点，不属于原 30 个 scored shapes。K7 inventory/preflight 以无角色授权 Materials 条目绑定请求；角色只获目标/许可缺项，不通过 Materials 投递原优化源码或历史 campaign。

固定 B1/Q4/seq_lens[64]/P1/有效 block_tables[0]/N1/S1，保留 tablewidth16434、workspace413138944、16Q heads/1KV head/D256/page64、两个 scale、两个 None 和全部 13 ABI 参数。原 helper 的 seed20261120、Q4 query BF16 randn *1.22 → FP8，随后 KV BF16 randn *0.96 → FP8，初零 workspace 和 empty out 均保留。CPU recorder 只记录原 helper 调用，仍记录 413138944 字节请求，没有拿缩小 CPU shape 替代真实 F1。

`FA4Task` 通过真实 Template resolver 读取 root Contract + direct Design。仅认识所固定四节点字节及 F1 绑定；root 绑定 fail，改动/未知 Schema 或 variant unknown，非 F1 scope 早拒。纯 host AST 验证 `_select_policy(1,4,64,1)==(1,"none")`、tile `(16,2)`。这是材料 predicate，不是 native runtime when 或完整 fidelity。

### 2. 真实 comparator/source 复用与原码入口

stage 实际包含原 `input.py`、`reference.py`、历史 `harness.py` 原字节。evaluator 从已核哈希的 harness 提取并执行 `checks/unchanged/poison` 三个原函数 AST，保留原来源行号；不执行其历史 main，也不加载 Production/旧 shapes/旧 timing。没有只写路径而未接入 comparator。

原 reference 使用独立 readonly tensor 副本、out 和 workspace，不能覆盖候选证据。比较器在 FP32 对返回值及调用前原 out 分别要求 BF16 `[4,16,256]`、actual/reference 全元素 finite、逐元素 `abs(a-r)<=0.06+0.04*abs(r)`。额外冻结原 out 的 object/storage/data pointer/offset/shape/stride/dtype/device，检查返回 alias 与原 view 不变；每次调用前同一 out 填 NaN，不能补写输出。query/KV/block_tables/seq_lens 的原 view 和逐字节内容均检查；workspace 允许变化。

完整检查发生在 cold call、5 次 warmup 和 40 次 measurement 的每一次调用后；检查和 poison 均在计时外。记录全部 readonly 输入和初始完整 workspace 的摘要，后者分块读取哈希，未缩小 scratch。原 empty out 不当作定义好的初值做数值比较，而在同一存储上 poison。

原码 entry 在任何原 CUDA 模块导入前要求 torch2.9.0+cu130/CUDA build13.0/CuTe DSL4.6.1/cuda-python13.1.1/实际 SM103。记录实际 Python binary、GPU name/UUID/driver、TVM-FFI 等事实；没有 fallback 或安装器。冷阶段记录 source import/input generation/constructor/首个完整 forward+sync，首调包含 normal fmha、native FFI probe/同步/可能 restore；稳态前后逐调用核 launcher、同六对象/signature/thread/current device、非 capture，且 launcher 不能失效重建后继续冒称同一稳态。

计时为 `time.perf_counter_ns` 包围完整 `Model.forward(**same_inputs)` + CUDA synchronize。没有以 kernel-sum、H20/Triton、旧 1.1126x 或 native `1e-2` 探测代替 Task 比较/全调用性能。此入口当前只准入固定完整原 Source；改动或缺文件候选在原码 stage 准入前即拒绝，不能冒称候选 sandbox。

### 3. 有限 Source codec、原子安装、删除/恢复

新增 `format=python_tree, encoding=utf-8, entrypoint=solution.py`。完整 name 排序清单，1–32 个普通 `0644` 文件，总量≤4 MiB；路径只支持标识符命名的 root Python 模块或一层 package Python 模块，以及 root `.json/.md/.txt` 或 `LICENSE`。拒绝 BOM/无效 UTF-8、符号链接、native binary、深层树、执行模式、缺入口和任何未列文件。不是任意源树/任意安装平台。

F1 Source 完整保留原 `aka_solution` 的 12 文件，含六个 Python runtime 文件和原 notices/provenance。`gqa_decode.py` 没有因 direct 不启动 combine 而删掉。原 paths/modes/bytes/每文件 hash 保留，manifest 不夹带 Q/父 C/历史状态。`Bundle.source(tree)` 明确拒绝，避免老调用者只取入口而漏包；老 `source(single_file)` 和 `put_source(bytes)` 行为保持。

`install_source` 向新目录 fsync+no-replace rename；已有目录拒绝。`replace_source` 要求 caller 提供准确 previous/new Ref、先让使用该目录的 reader quiesce；独立文件锁下验证旧/新完整快照，再以 Linux renameat2 exchange 原子换目录，fsync 后删除验证过的旧快照。交换后中断可用相同 Ref 重入完成清理，没有 overlay 或第二份预算/运行账本。已实测两文件→一文件，旧文件消失；交换后 synthetic 中断重入也通过。

### 4. 请求与观察身份、未知资格

`freeze_request` 分离三种意图和各自目标 Materials：original_verify 固定原 C + verify_only；seed_continuation 必须显式完整 Source；schema_rebuild 无 seed/starter，且目标明示不得交付优化源码/历史。后两者只是正常 K2 请求构造能力，当前 backend 不支持其真实执行。

默认 prepare 创建零 calls/jobs/active_ms 的原码请求，以及 capabilities 全空的 `aka/fa4-preparation-only-v1` profile。该 profile 如实表示“尚未接通”，不假造 provider/model/GPU 支持。`require_capabilities` 按请求实际 Source 格式对 profile 与实际 backend 双重检查；single_file-only backend 对 python_tree 早拒。

`freeze_observation` 核原 stage 字节/mode/size、Task/R/Scope/C/evaluator/input/timing raw subject，原样冻结 stdout JSON、实际 environment 及 normal evaluation Ref。观察导入不是认证通道：即使 synthetic raw 写 passed=True，也只产生 unknown build/correctness/performance，fidelity/performance_goal 仍 unknown；真正 owner/native receipts 由未来实际 adapter 传入。K8 不创建替代余额，也不把历史原码账目当本请求费用。

## 实际命令与产物

CPU 准备（已执行成功，不导入原 CUDA 模块）：

```sh
cd /home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1
/tmp/schema-b05-a3-venv/bin/python -B -m examples.workflows.fa4_decode.prepare --out /tmp/ks-b08-fa4-final
```

产物入口 `/tmp/ks-b08-fa4-final/prepared.json`；正式请求 projection `/tmp/ks-b08-fa4-final/request.json`，stage `/tmp/ks-b08-fa4-final/stage`，共 19 文件。所有正式 Ref 位于 prepared.json。精确抽查身份：

| 对象 | SHA-256 |
| --- | --- |
| 原完整 Source C | `sha256:c8395325f884d636f9c2e0dd3b622680848e270c40cb55a8b9fb178190116611` |
| Task | `sha256:7640996fa914cf3c6a09e02de29c2eaabcca0a48b00f9df065f1000326c20eab` |
| 四节点 Schema | `sha256:ccea84ed37d61a1dd4be949665ecee625b5d06ccdad30f352569ed035864e6c0` |
| 默认不可执行请求 Q | `sha256:0057b3ff65984fe1bf567155f995719d7f0db22ea9f8bb7b89ffc434e35fc2ba` |
| 完整 stage | `sha256:808b9d63f85f51b460e82ba6395431f0c057cfe12e569cdd87a0cd30f520094a` |

原栈手工观察命令（**未执行 GPU**；这里只验证了 `--help` 可调用）：

```sh
/path/to/original/python -I -B /tmp/ks-b08-fa4-final/stage/evaluator/entry.py \
  --stage /tmp/ks-b08-fa4-final/stage \
  --stage-sha256 sha256:808b9d63f85f51b460e82ba6395431f0c057cfe12e569cdd87a0cd30f520094a \
  --output /tmp/fa4-f1-observation.json
```

stage 可由选中 Task/Schema/Scope 的实际 closure 导入后再次构造，不需要原 Git checkout。对象输出路径是本次本地准备事实，不是远端工作或资格收据。

## CPU 检测结果

使用现有 `/tmp/schema-b05-a3-venv/bin/python`：Python3.13.12、pytest9.2.0、PyYAML6.0.3。仅 helper 逻辑测试使用既有 CPU torch2.10.0；它不是要求的原 torch2.9.0+cu130 栈，未被用于代签 F1。没有下载解释器/大栈或修改共享依赖。

```sh
/tmp/schema-b05-a3-venv/bin/python -B -m pytest -q \
  tests/workflows/test_contracts.py tests/workflows/test_artifacts.py \
  tests/workflows/test_api.py tests/workflows/test_reference.py \
  tests/workflows/test_standalone.py tests/workflows/test_evaluation_logic.py \
  tests/workflows/test_evaluation_execution.py \
  tests/workflows/test_fa4_source.py tests/workflows/test_fa4_case.py
```

结果 **143 passed，585.20s**，包含专属 20 项；不是额外 143+20 个不同测试。core 源码在该批回归期间没有再变更。收尾的 case 来源目标说明修正后，再跑专属两文件，**20 passed，40.32s**。没有无差别重复整个 320 测试。

覆盖真实 codec/完整 12 文件 Source、closure/import/装载/模式/额外文件与 symlink 拒绝、两→一删除、交换后中断恢复、正式 Task/evaluator/ref 绑定、原生成器调用次序和常量/13 ABI、四节点 effective/root 错绑、未知 variant/非 F1 scope、来源意图区别、synthetic 不升资格，以及小 CPU tensor 的 NaN/finite/逐元素/只读变更/返回 clone/原 out 换 storage 反例。另有新进程准备后 sys.modules 检查：没有 torch/cutlass/cuda/原包导入。

`git diff --check`、本次四份 Markdown 的本地链接、原码入口 `--help` 均通过。首次工程 prepare 曾揭露 Template effective 中 Decimal 不能直接 JCS；已只在派生 effective 摘要中转为十进制字符串，原 YAML/raw 未重写；最终准备成功。

## 下一任务精确接线清单

这些工作要在新的无并发写冲突任务完成；本批没有把部分工程叫做端到端实现。

1. **Task dispatch**：`implement/reference.py`、`evaluation/channel.py`、`writing.py`、`_writing_policy.py` 仍显式使用 RowSoftmaxTask。为 FA4 接入对应 finite Task/Scope seam；draft-document 检查需要明确实现，不能照搬 row-softmax 的 full_task_only 或简化 when。新增 Schema/variant 仍 unknown，root 不能当 fallback。
2. **K3 Source 输入/输出**：`runtime/claude.py` 的角色交付和 `candidate_source` 输出仍是单个 kernel.py/string。引入最小 typed 完整文件输出，精确 file-qualified 内容/模式，全部 source/input 回放与恢复一致；用 core `source_files`、`put_source_files` 和完整替换，不重新写 parser/codec。新格式 capability 必须随真实支持一起启用，保留旧格式负例。
3. **K4 fidelity 源码定位**：`fidelity.py` 当前读 `bundle.source(...).decode()` 和 kernel.py 行号/softmax AST。接收全部实际 Python 文件与相应精确行证据，读取当前完整 selected Schema；只读权限/静态图/有限 AST 不授 fidelity。补 empty parity、TMA padding/page tail、TMEM、输出 staging 和必需原语义务，未知仍 unknown。
4. **AKA adapter**：A5 完成后再接 `aka.workflow.schema` 及 execution session 多文件编码/安装/终态 Source freeze；公有 C→native source identity→真实 stage/upload/评测文件显式映射。后续 source 删除必须体现在 native 文件集，不能 overlay；原 C 复验零 coder/零修改/零 promotion。只读原子安装是已有 core 支持，不能另复制预算或原操作账本。
5. **source 信息界限**：original_verify 可以审固定原 C；seed_continuation 显式选 seed；schema_rebuild coder 不能获 baseline/old Source/annotated code/campaign/history。特别需要抑制 K3 现有 Task baseline 交付例外；当前 tree 早拒只是防止误交付，不算此路径已实现。
6. **真实原栈执行通道/profile**：当前 Agate `_worker.py`/six-job row-softmax 资格不能原样签 FA4。接完整 case evaluator 和候选 sandbox，限定原 CuTe/SM103/Python/driver/compiler/TVM-FFI，实测 readonly、完整 stdout、文件模式/上传 identity、同原操作 submit/query/settle 与费用。用户已说明认证已配置；本批没有做 case 认证实测，不作“未配置/服务不可用”的断言。
7. **候选与 baseline 比较**：当前 `stage_original` 只准入固定完整原码。候选执行需要上述隔离及 native predicate 机制后才可扩展；baseline/候选各自 fresh process，用同一完整 comparator/计时，不继承旧 kernel-sum 或旧 R 的 fidelity。将实际 owner/native usage receipts 与 reports 一起绑定，才可能进入合格结果的五门判断。

## 仍未闭合的真实门与操作边界

本批模型调用=0，GPU job=0，网络提交=0。未运行 F1 原 CUDA 源、未建立/观察 native launcher、未生成原栈 F1 数值、未测冷/稳态样本、未作完整 source-to-current-Schema fidelity 判断。N1 仍有空 parity/TMA/TMEM 义务；不能由有限无环图签安全。

原 L20D alias/实际 SM103/驱动/编译器/Python/TVM-FFI/hardware Wiki、Agate 认证/实际 profile、readonly/完整 stdout 和候选隔离均尚无该 case 的资格证据。源码许可仍存在 NVIDIA BSD header 与旁置 LICENSE/缺失 notices 等 K7 缺项；两固定提交匿名取得证明也未补。原件可核哈希和继续本地工程，不等于重新授予原源码许可。

未访问/修改/运行 EXP；未修改 AKA；未改远端、bench、根计划/progress/history、Git 分支/配置/commit/push 或 AKA KS pin。预算/运行时恢复核心未修改，只有 generic intent 的完整 Source 验证一行兼容修改。小 demo 和 A5 fixed 不受 F1 真实门阻塞。
