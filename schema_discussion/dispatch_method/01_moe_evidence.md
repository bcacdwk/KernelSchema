# Fused MoE：设计差异、组合规模与返工证据

2026-11-13，只读研究；本报告是建议，不是新规范或新批准。历史 YAML 的 `phase`、`repeat`、`protocol` 等字段仅用于核对历史数据，现行语法依据仍是 DOC0–6。

- **事实：60 与 154 的实现选择范围不同，不能解释为同一组设计从 60 膨胀到 154。** 初始 60 包含显式配置的 FC1/FC2 族；后续冻结范围排除了手动配置，却要求保留默认入口的初始化、缓存、验证、回落与错误出口。该范围文件明确不收窄数学 Contract，见[冻结 scope:5–11][scope]。
- **事实：暂停时是 154 项计划、96 项批准，尚未完成本轮 P6/发布。** 另有 9 份完整候选、1 份待续写/返工、5 个失败 seed、43 项未开始。能打开 111 个文件也不等于 111 份完整设计；5 个失败 seed 都只有 17 行且 `when` 为空。
- **事实：两代产物暴露了不同的组合问题。** 初始 60 中有 10 种 FC1 组织 × 5 种 FC2 组织的 50 个命名组合；较新 154 可分为 15 个基础条目及 10 类前缀组合。后者已经删去了大量非默认配置，仍产生大量重复完整稿，不能只靠缩小调优范围解决。
- **事实：有可直接证明的重复，也有必须保留的局部信息。** `fused_small_flat` 与 `no_scan_fused_small_flat` 的三个有效 Design 区域完全相等；后者的 Realization 多两项重建服务与 workspace 寿命说明。`init_no_scan_fused_big` 增加查表工作后，原有七段 pipeline 全部相等，但三个整区都变成本地定义。不能把必要的初始化、版本、可见性和寿命条件随 API 细节一起删掉。
- **事实：FC1/FC2 的组合受接口约束。** 路由 map 的 BM、压紧行排列、ActQ/ActS 表示、生产完成边界必须一致；REDF 改变 FC2 的输出对象和后续归并，不能退回普通写 rout 的内核。TSTORE 还必须区分完整/部分 quadrant，以及共享源读完与全局写完。这些不是一句“微调”能表达的变化。
- **建议：完整 YAML 记录代表性的端到端设计；局部组织和调用协议另写一次明确的设计说明。** 对默认范围可先保留融合链、split-K 链、单 kernel direct、带验证的计算链四类代表；若继续覆盖初始显式配置，则还要保留 last-arriver、in-grid prepare、BF16 两族、REDF 的新增设计信息。这个分组是讨论起点，不是九节点最小性证明，也不宣称每个组合已经重新展开或审定。
- **事实与边界：成本支持优先研究重复创作/审阅，但不支持节省比例。** 已记录 MoE 2.5.0 系列 P5 为 248 次尝试、$1,863.86；其中既有覆盖扩展，也有转录遗漏、源码缺陷、提案证据格式拒绝和 CLI 失败。单 root 的 P3 闭环本身已有约 $15.83，不能把所有费用或回修归因于分支数。

## 1. 版本、范围和数量的可复核口径

| 时点/状态 | 核对结果 | 意义与证据 |
| --- | ---: | --- |
| 初始 2.2.2 交付 | 60 个 YAML | 实际目录中逐文件解析得到 60；旧索引末尾“后续 P5 生成”是过期文案，不能据此改称 60 份计划。[归档说明:22、32][archive]、[阅读勘误:5–7][errata]明确其交付身份及仍保留的质量意见。 |
| 2026-11-09 暂停计划 | 154 项 | `state.plan` 的 ID 集合与 `nodes.csv` 中 MoE 的 154 行一致；没有 MoE 退役项需要扣除。 |
| 暂停时批准 | 96 项 | root 加 95 个派生节点；所选 working checkpoint 恰有这 96 个 YAML，文件名集合与清单批准项完全一致。 |
| 完整候选、待审/复审 | 9 项 | 使用清单各自的 candidate checkpoint；不能从批准 working checkpoint 缺少这些文件推断从未写过。 |
| 需继续写作/返工 | 1 项 | `init_no_scan_split_mono`，已有 1064 行稿；长度不构成完整性或批准证据。 |
| 失败，暂无已记录创作结果 | 5 项 | 都有 17 行 seed：元数据、空 `when`、三个 `from_parent`。例如 [seed:10–17][seed]。它们不是可使用的完整候选。 |
| 未开始 | 43 项 | 不计入已创作或完整稿。 |

暂停分类取自[暂停清单:5–9][pause-moe]和[交接:10–31][pause-readme]，本轮没有改变任何状态。批准数是该 run 的历史批准，不表示这些旧语法文件通过了今天的 DOC0–6 检查。

较新原稿统一入口是：

```text
.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/
  checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/
```

下文 `Q/` 表示该目录，`I/` 表示 `schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/`；`C`、`P` 分别表示实际 `candidate/nvfp4_moe.cu`、`candidate/solution.py`。每处引用均链接到具体文件和起始行。[较新 root 完整位置][q-root]、[较新 dispatch 完整位置][q-dispatch]可直接打开。

**范围限定的性质。** 冻结文件 `nvfp4_fused_moe_default_scope.md` 是输入绑定中的独立 `scope` 材料，并非源码移除了其他分支。它在第 7 行排除手动 cfg/cfg2、环境覆盖、诊断、实验与手工 ABI；第 9、11 行明确任务动态输入范围、数学、IO、验收不变。Q/root 的 `when` 第 14–15 行引用该实现选择范围。源码仍在 [P:1170–1172][p-selection]、[P:1202][p-cfg2]接受显式覆盖。因此应分别描述“共同数学 Contract”与“本次选取哪些实现方案”，不能说较新 Contract 只允许默认数学，也不能说默认设计有效性只由原阈值定义。

本轮对 `solution.py`、`nvfp4_moe.cu`、`nvfp4_fast.c` 计算 SHA-256，三者均与该 run 的冻结 `inputs.lock.json` 相等；这里的源码行号对应实际冻结内容。全部 111 个所选文件也与清单的 `own_sha256` 相等，见[可重现数据][snapshot]。

## 2. 两种组合规模，分别说明什么

### 初始 60：计算组织的配对

逐一核对文件名，以下每行都有 `flat / mono / redf / w1 / ws` 五个 FC2 后缀，合计 50 个 YAML。它们是实际文件，不只是索引中的空行。

| FC1 前缀 | 代表节点 | 新增的主要设计信息 | FC2 配对数 |
| --- | --- | --- | ---: |
| `mono_fused` | `mono_fused_mono` | 四 warp 单体 FC1，融合激活与 packed QDQ | 5 |
| `mono_split` | `mono_split_mono` | split-K 的 FP32 Part 物化，独立 activation/量化提交 | 5 |
| `mono_arrival` | `mono_arrival_mono` | 最后到达 CTA 读取 Part、归并并量化；取消独立 activation | 5 |
| `w1_fused` | `w1_fused_mono` | 一个 warp 执行主循环，其他 warp 在 item 边界参与 epilogue | 5 |
| `w1_split` | `w1_split_mono` | 上述主循环组织配 split-K 与独立 activation | 5 |
| `ws_fused` | `ws_fused_mono` | A/B producer、MMA、epilogue 分工；融合量化 | 5 |
| `ws_split` | `ws_split_mono` | 分工 FC1 配 split-K 与独立 activation | 5 |
| `ws_arrival` | `ws_arrival_mono` | 分工 FC1 的 epilogue 承担 last-arriver 归并 | 5 |
| `ws12` | `ws12_mono` | 独立的 12-warp FC1 组织与双累加器协议 | 5 |
| `selfprep` | `selfprep_mono` | FC1 grid 内重算路由 map，改变路由生产者与准备边界 | 5 |

另外 10 项为 root、`route_fused`、`bf16_tc`、`bf16_warp`，以及 `zero_fused_*`、`zero_split_*` 各三项。来源是[初始索引:5–64][i-dispatch]；可重现脚本显式生成 `10×5` 集合并检查每个文件存在。

这支持“已有相同 FC2 知识随不同 FC1 重写”的判断，却不能证明 50 个任意强制配对都正确。初始索引自己也在[第 76–78 行][i-limits]披露 map 不匹配、错侧配置和 BF16 BM 不一致；下节直接检查其源码条件。这个表不是一张合法组合证明书。

### 较新 154：默认计算主干与调用前缀的配对

普通基础条目有 15 个：融合 FC1 的大准备四种 FC2、小准备四种 FC2、一个关闭 PDL 的大 R 变体；split-K FC1 的五种 FC2；以及一个 direct。这里的四/五种 FC2 是 bulk-store WS、单体 TMA、flat、cp.async 回退，加 split 路径的极小 R 直接写回 WS。基础条目全部已批准，但不因此变成 15 个不可再分的核心设计。

| 前缀类别 | 计划 | 已批准 | 前缀增加什么 |
| --- | ---: | ---: | --- |
| 无前缀 BASE | 15 | 15 | 普通 ABI 复用的上述基础路径 |
| `no_scan_` | 15 | 13 | 重建 ABI，查表和零 scan 都未提交，正常进入主干 |
| `init_no_scan_` | 15 | 11 | 在上一类之前实际提交查表并同步 |
| `probe_` | 14 | 12 | 重建时已提交零 scan，但 `zero_ok=false`，随后主干 |
| `init_probe_` | 14 | 9 | 查表 + 上述 scan + 主干 |
| `verify_` | 15 | 13 | 14 种带设备验证的完整主干，另有 `verify_only` |
| `probe_verify_` | 15 | 5 | 重建 scan 接受后，带设备验证的主干或仅验证 |
| `init_probe_verify_` | 15 | 2 | 再增加查表 |
| `double_verify_` | 12 | 8 | 独立验证图，然后含第二次验证的完整图 |
| `probe_double_verify_` | 12 | 5 | 重建 scan 后执行上述两图 |
| `init_probe_double_verify_` | 12 | 3 | 再增加查表 |
| 合计 | **154** | **96** | 分类由最长前缀匹配，未把 `init_probe_verify_` 重复归入 `init_probe_` |

这不是无约束的笛卡尔积。direct 在 `run_pipeline` 前部返回，不进入一般零验证链；`split_ws`、`split_mono` 在本 Contract 的默认门下达不到 ZERO2 的 `T*H` 门，所以没有双验证子项。[Q/dispatch:162–176][q-selection]说明这些排除；[P:1478–1549][p-zero]、[C:9102–9133][c-direct-submit]可核对真实门与提前返回。`verify_only` 也没有再与每一种 FC2 配对，这是计划已经采用的一处合理压缩。

**推断：** 初始矩阵主要放大“局部计算组织的组合”，后续矩阵主要放大“同一计算知识遇到不同调用前缀”。两种重复不能靠同一种删除口径处理：前者要检查 Act/map/归并接口，后者要检查初始化、数据版本和事件关系。仅改为默认 scope 已不足以避免组合规模。

## 3. 实际 YAML 差异：相同到什么程度

比较通过 PyYAML 解析，沿实际 `from_parent` 逐区展开，比较全部 Design 字段，**包括参数、动作说明与绑定字符串**；未做字符串清洗或语义归一化。元数据、`when`、Realization 另比，不混入“Design 相等”。这些是内容相等测试，不是新的正确性批准。

| 实际比较 | 本轮结果 | 应如何解释 |
| --- | --- | --- |
| `Q/fused_small_flat` → `Q/no_scan_fused_small_flat` | math、schedule、pipeline 三个有效区域全部相等。子文件[第 47–53 行][q-no-scan-flat]直接引用父三区。 | 不值得再展开一份完整核心 Design。Realization 的 `require` 从 4 项增至 6 项，新增[第 80–111 行][q-no-scan-flat-services]的重建服务/错误边界和 workspace 初态、绑定、寿命；这些信息应另记录一次。`when` 也确实从复用改为重建。 |
| `Q/no_scan_fused_big` → `Q/init_no_scan_fused_big` | 原 count、scan、scatter、gather、fc1、fc2、finalize 七个 pipeline 子树逐项相等；增加 `init_tables`。但 math/schedule/pipeline 三个整区都不同。 | [新增数学:47–64][q-init-math]、[launch:229–238][q-init-flow]、[pipeline:465–480][q-init-pipe]是完整的局部前缀变化。Realization 只改原第 4 条：说明 init_tables 普通提交。不能据三区不等判断三个核心设计都变了。 |
| `Q/root` → `Q/no_scan_fused_big` | math、pipeline 全等；schedule 只在 thesis、count 的 strategy 两个文本位置不同，flow 和其他字段一致。 | 这里主要是调用服务/初态的说明，不能只因“local schedule”推断出现新调度方案。 |
| `Q/root` → `Q/fused_small` | gather、fc1、fc2、finalize 四个 pipeline 子树全等；三次路由计算提交改为一个 prepare。 | [root flow:375][q-root-flow]对比[small flow:203][q-small-flow]。这是值得单独说明的准备算法，但没有理由把四个未变计算段随每种 FC2 再写一遍。 |
| `Q/fused_small_mono` → `Q/fused_small_flat` | math、schedule 全等；prepare、gather、finalize pipeline 全等；FC2 pipeline 改写。FC1 只有 `AProducer.info.verb.prime` 一段文案不同。 | 不能夸称“只改 FC2 的所有字节”：父写“fc1 用 GatherS、fc2 用 ActS”，子在 FC1 处专写 GatherS。两份 when 均选 FC1 cfg97，源码实例相同；没有发现额外 FC1 设计变化。[flat:38、102、317][q-flat]、[mono:114、332][q-mono]。 |
| `Q/fused_small` → `Q/split_store` | gather 消失，增加 activate；ActQ/ActS 的 producer 从 fc1 改为 activate，出现 Part；prepare、finalize pipeline 全等。 | [split math:88、128][q-split-math]与[flow:252][q-split-flow]保留真实物化/依赖改变。默认 cfg97 与 99 同时改变 split 和 A 搬运；不能把这对样本当成单变量 split 实验。 |
| `Q/verify_root` → `Q/double_verify_root` | 原七段主干 pipeline 全等；增加独立 probe，第二次 verify 的发布说明改变；三整区均有版本/账目/提交变化。 | [double math:46–112][q-double-math]区分 `ProbeClean/Clean`、`ProbeOutput/ZeroOutput`；[flow:370][q-double-flow]实际有两次验证。可以共用主干，不能把两次验证合成一次。 |
| `I/mono_split_mono` → `I/mono_arrival_mono` | pipeline 删除独立 activate；路由计数初始化、FC1 归并组织也改变。 | [独立归并:340–356][i-split-act]与[last-arriver target:298–323][i-arrival]、[C:3756–3795][c-arrival]相符；不只是去掉一个 launch 名字。 |
| `I/mono_split_mono` → `I/mono_split_redf` | finalize 被 convert 替换；FC2 输出与 tile reduction 改变。 | [REDF 数学:170–198][i-redf-math]、[调度:380–403][i-redf-flow]与[C:3948–3982][c-redf]相符。 |
| `I/ws_split_mono` → `I/selfprep_mono` | 新增 selfprep、删除 gather；activate 与 FC2 pipeline 原文相等。 | [selfprep flow:233–240][i-selfprep]保留 `selfprep,fc1` 同次提交；[C:4034–4060][c-selfprep]给出 producer、同值写、局部与下游完成条件。 |
| `I/bf16_tc` → `I/bf16_warp` | decode、准备、finalize pipeline 相等；增加独立 activate，FC1 从融合 BF16 Act 改为先物化 FP32 Fc1。 | [BF16 tc:192–199][i-bf16-tc]、[warp:199–207][i-bf16-warp]；对应源码 [C:9281–9301][c-activation-submit]。不能视为只换 MMA 指令名字。 |

另核对：95 个批准的非 root 节点里，24 个用了显式整区引用，共引用 45 区。这个计数说明整区共享已经在用，也说明“支持引用”本身没有减少 154 项计划。无引用的区域不代表没有文件内 alias；局部原文相等也不自动证明可在任意节点上下文共享。

## 4. 分支轴、可组合程度与必须保留的接口

| 分支轴与实例 | 数学/物化/分解/角色/依赖实际变化 | 独立组合的边界与源码证据 |
| --- | --- | --- |
| **FC1：融合、split-K、last-arriver**；`root`、`split_store`、`mono_arrival_mono`、`ws_arrival_mono` | 融合直接写 packed Act；split-K 写 FP32 Part 后单独归并量化；last-arriver 仍写 Part，但最终到达者在 FC1 内归并，增加计数器与发布协议。可能伴随归约顺序、折叠激活与舍入变化。 | 三者最终可提供同一 packed Act 接口，但必须保留 Part 的 slice 顺序、计数初值与量化前归并完成。WS last-arriver 仅 epilogue 的 128 线程进 named barrier，不能换全 CTA 屏障；`FUSE` 要求 G1、SK>1、EPI=4，排除 PP/EARLYDONE/TSTORE。[C:4233–4240][c-assert]、[5270–5311][c-ws-arrival]。 |
| **FC1 A 搬运：预 gather 与直接 gather4**；cfg97、cfg98、cfg99 | 预 gather 物化 GatherQ/GatherS，FC1 消费连续 route 行；直接 gather4 在 FC1 内按 token 映射取输入，删除全局 gather 缓冲和提交。 | 它与 split 在默认选择中相关，但不是数学上必然绑定：cfg98 是 SK2+AMODE2，cfg99 是 SK2+AMODE3。[C:7780–7815][c-99]。必须同步修改 workspace 分配表、实际 gather 提交表、tma_set 输入和 FC1 AMODE；不能独立切一个布尔值。[P:978–1001][p-workspace]、[C:9245–9272][c-gather-submit]。 |
| **FC1/FC2 主循环分工：mono、w1、WS、ws12** | 同样的 contraction 可有不同发起者、每 stage 的 CTA rendezvous、跨 item 缓冲与 epilogue 重叠；不是数量变化一项。 | w1 的 G1 要求 AMODE2，主循环由 warp0 独占，其他 warp 在 item 边界加入；WS 分 A/B producer、MMA、epilogue，ws12 有双累加器。[C:5420–5463][c-w1]、[4215、4451、4541、4843][c-ws]、[6201–6212][c-ws12]。相同 BF16/packed 接口允许有条件配对，不允许把所有分工写成一个隐藏的参数开关。 |
| **FC2：per-item、flat、cp.async**；`fused_small_mono/flat/cp` | 保留 BF16 RouteResult 与 finalize，但 flat 把多个 item 的 stage 串成流，改变 ring、累积提交及 item 尾部 epilogue；cp.async 改 payload 就绪和复用协议。 | FC2 读取相同 ActQ/ActS 和专家路由 map，所以可避免为每个 FC1 重写其逻辑。flat 的 NK1 实例仅 I=256；一般 flat 另有多 stage 实例，不可把 NK1 条件删掉。[C:7224–7257][c-flat-select]、[5822–5867][c-flat]。 |
| **FC2：WS bulk store 与直接 store**；`split_store/split_ws`、cfg81/57 | packed contraction/最终 BF16 route 语义一致；cfg81 还有 DEEP、TSTORE 等组合变化，共享 Sink 的寿命不同。 | TSTORE 只允许 G2、BN128、EPI4、非 PP。完整 32 行 quadrant 才能 bulk 写；部分块必须 masked STG，否则覆盖下一专家。`wait_group.read` 只保证源已读，最终 `wait_group 0` 才给输出完成。[C:4233][c-assert]、[5210–5246][c-tstore]。两种存储可以写一次局部协议，不能只保留“快写回”四字。 |
| **FC2：REDF**；`mono_split_redf` 等 | 先 BF16 舍入，再乘路由权重，以 FP32 原子汇入 token Acc32；不写/读 rout，finalize 改 convert；加旁流清零和 GEMM2 前 join，改变归并顺序。 | 改动横跨 FC2、输出归并和初始化。不能复用普通 FC2 的后继链；源码明确禁止 map 拒绝时回退 rout 写者，因为后继已按 cfg2 选择 convert。[C:3948–3982][c-redf]、[9164–9191][c-redf-init]、[9330–9356][c-finalize]、[8722–8740][c-refusal]。 |
| **准备：大准备、小准备、SELFPREP**；`root/fused_small/selfprep_mono` | 大准备有 counts/fill 清零及 count→scan→scatter；小准备单 CTA 内生成 map；SELFPREP 在每个 FC1 CTA 重算并发布 map。 | 小/大准备可通过同一逻辑接口接相同下游，但不是必须生成逐字节相同排列：scatter 的原子顺序可不同，必须保持 `ptok/pslot/pexp/route_of/offs/toffs/texp/tm0` 一致对应。[C:657–697][c-route-map]。SELFPREP 要 R≤256，并且 PREP2/KEEP_PREP 可令独立准备仍发生，[C:9211–9223][c-small-submit]；不能无条件删除所有准备。 |
| **PDL 开关**；`root/fused_serial` | 计算提交和数学不变，启动许可与 predecessor 结果就绪的关系改变。 | 源码门是 `g_pdl_shape_off`，[C:9092–9095][c-pdl]；不能只改 launch 说明而保留不匹配的 role 入口。应在一个局部接续协议里说明 PDL 的 launch_ready 与最终可见性、普通 stream 完成接续各自关系；原阈值属于实现选择证据。 |
| **初始化/缓存/零 scan**；`no_scan_*`、`init_*`、`probe_*` | ABI cache 命中本身不改变 contraction；查表和 scan 则是额外设备工作。scan 的数据依赖与 OR 归并有新信息，查表可能不被所选 packed 主干消费。 | `ensure_tables` 按设备 ready-set 决定提交并设备同步；probe 的 -1/-2 在 scan 前，-3/-4 在 scan 调用后，Python 异常/负返回仍可走完整主干。[P:1005–1016][p-tables]、[P:1484–1495][p-zero]、[C:9585–9614][c-probe]。重建不等于一定新建 workspace，更不等于隐含跨 stream 排他；保留初态和寿命，简化 API 叙述。 |
| **零验证/双图/回放**；`verify_root`、`verify_only`、`double_verify_root` | 输出可能由验证 kernel 写零，其他已提交 kernel 经 ZGUARD 早退；双图实际两次验证共享存储但有不同版本；超时仍可提交完整图。 | 只有携带 guard 的家族可接验证；有限轮询的 host 读值不是 kernel_done，也不自动等于本次 Clean。实际存在跨调用 mirror 无代次问题，见下一节。graph 记录固定 ABI/选择，执行时读取当前张量；重放不能按新直方图重新选路。[C:208–214][c-guard]、[1707–1722][c-zero-contract]、[9471–9491][c-graph]。 |
| **描述符拒绝/错误出口**；`fused_big_cp` 等 | cfg23 拒绝先转12，再与44汇合到有效 fc8；这一路改变搬运协议，主干其余部分可共享。许多其他配置则直接报错。 | `tma_set` 同 key 缓存成功或拒绝；但 WS、split 和 REDF 不能按同样办法回退。[C:7165–7186][c-map-cache]、[7257][c-flat-select]、[8722–8750][c-refusal]。错误码不是新的算法；已发生的工作与可能写零的副作用仍须保留。 |

### 两个容易被“可组合”掩盖的约束

**FC1/FC2 的共享 buffer 不是仅凭 shape 相同成立。** [P:1367–1391][p-family]先按中间表示族校正配置；packed FP4 与 BF16 Act 不能直接混用。两 GEMM 还共用一份 BM 路由 map，见[P:1173–1174][p-selection]和[C:7222][c-flat-select]。初始 BF16 文件已经披露独立 BM 导致覆盖/多写问题；这不是可以补造“一般 BM 自动相等修正”的地方。安全复用要写清值的表示、有效行域、route 索引的正反对应和完成事件，而非只引用 `Act.shape`。

**验证协议不能整体降为无关包装。** 已批准的 `Q/verify_only` 第 145–154 行明确区分 `HostObserved` 与当前 device `Clean`，并保留镜像无请求代次的源码限制，[见当前稿][q-verify-only]。本轮直接检查 [C:9480–9488][c-graph]：CPU 重置共享 word 后提交图并有限轮询，既不等待前次验证完成，也不校验版本；[C:1764–1786][c-zero-publish]中验证最后到达者还能在稍后写该 word。由此可构造“前次超时后仍在途发布 1，后次读到 1 而自身最终 Dirty”的交错；这与原 [P5 退修回执][r-verify]相符。这里是源码静态协议反例，未运行 GPU 复现，不能改写为已测故障率。

## 5. 建议的压缩分组与记录深度

以下是**讨论方案**。完整 YAML 仍应在所声明的代表范围内自洽；局部说明只承诺明确的变化和条件，**不充当可以自动展开的 patch，不把未展开的组合称为已审设计**。本方案有意不复刻全部运行时选路，因此需要修改目前“每个完整节点互斥穷举”的职责，不能在现规则下直接宣布完成。

| 完整记录组 | 代表与覆盖的知识 | 为什么值得完整 YAML | 其余如何处理 |
| --- | --- | --- | --- |
| A. packed 融合主干 | Q/root：路由压紧、预 gather、WS FC1 融合量化、FC2、BF16 route、slot-order finalize | 给出共同的数学改写、物化图、两 GEMM 接口和端到端依赖，是主要设计起点。 | 小/大准备、FC2 搬运/分工、PDL 以局部说明组织；不把 A 与每个前缀的配对重新算一份核心设计。 |
| B. split-K 主干 | Q/split_store：Part→独立归并/量化→FC2，代表默认 gather4 输入形式 | Part 的存储、slice reduction、生产者变化和完成边界跨 math/schedule/pipeline，读者需要贯通整条链。 | SK 合法数值按公式参数化；AMODE2/3 的差异单独说明，不能假称全部 split 必须用 gather4。 |
| C. 单 kernel route direct/MEGA | Q/direct，兼参照 I/route_fused | 消除原准备/两 GEMM 链，变成单 kernel 的 census、路由计算与完成计数，工作分解和资源寿命整体改变。 | capture 时 empty 只作来源选择，保留回放时非空处理；STAGED 权重缓冲作为局部数据通路说明。 |
| D. 验证输出捷径 | Q/verify_root 作为带 guard 的完整主干代表；结合 verify_only/double_verify_root 的协议事实 | 验证成为实际输出生产者，guard 与验证完成耦合，不能只有一句“零时省略计算”。 | 单验证图/两图的提交、版本、host 观察、超时和源码限制写成一次完整协议说明，再声明可接的主干接口；不与所有 FC2 交叉复制。 |
| E. last-arriver split-K（扩展范围） | I/mono_arrival_mono 或 ws_arrival_mono 选一个明确实例 | Part 仍全局物化，但归并移到最后到达者；增加计数初值、发布/读取和角色内屏障，取消独立激活。 | mono/WS 两种执行角色的对应差异记录在局部组织说明；不展开五个 FC2 后缀。 |
| F. in-grid 路由准备（扩展范围） | I/selfprep_mono 的一个有据实例 | 路由 producer、执行位置、重复工作与本 CTA/下游可见性都改变；这是新的数据供给方案。 | R≤256、map 一致性、PREP2/KEEP_PREP 等范围写清。超出已核对范围不承诺随意组合。 |
| G. BF16 warp MMA 链（扩展范围） | I/bf16_warp 的一个 BM 匹配实例 | hidden 解码与 FP32 Fc1 物化、独立激活、BF16 Act、warp MMA 构成另一条数据通路。 | tile 数值参数化；有缺陷的强制 BM 配对作为来源限制，不补造成功设计。 |
| H. BF16 tensor-core 融合链（扩展范围） | I/bf16_tc 的一个 BM 匹配实例 | 使用 BF16 operands/TMEM，并在 FC1 内融合激活，取消完整 Fc1 临时量；与 G 的提交和存储边界不同。 | 公共解码、路由和接口可共享定义；不能仅因二者最终都是 BF16 Act 就合并正文。 |
| I. REDF 输出归并（扩展范围） | I/mono_split_redf 或其他明确配对选一个 | 数值舍入点、归并顺序、Acc32、旁流初始化/join 与 convert 必须连起来理解。 | 与 E/G 等新组合没有证据时不保证；描述符错误边界独立保留。 |

这给出默认范围四组、扩展范围另五组的候选，而不是先设九个名额。**A–I 并不代表全部微流水线知识只有九种。** mono、w1、WS、ws12、flat、TSTORE、DEEP/PP 等仍有值得理解的方案；建议为这些变化写局部设计说明，而不是再复制完整算子。若某一微流水线本身正是学习或实现目标，也可以把其代表升级为完整 YAML。是否升级由新增信息和使用目标决定，不由“历史索引已有多少个 ID”决定。

局部说明至少给出：改动的数学值/step/role；输入输出接口；存储与版本；必要等待及删除/新增的真实提交；适用条件与来源。具体到本案例：

| 记录为一次局部说明 | 必须保留 | 可以不继续展开的内容 |
| --- | --- | --- |
| 小准备替换大准备 | count/scan/scatter 到单 CTA 的工作分解，路由 tuple，计数初态，下游等待的 producer 改名 | 每种 FC2 再复制不变的 gather、fc1、fc2、finalize |
| mono/w1/WS/ws12 与 flat | 谁发起搬运/MMA，谁参与同步，item 间 carry，ring/accumulator 退休 | 不变的任务 Contract、路由与输出归并正文 |
| TSTORE、DEEP、PP 等协议 | 完整/部分 quadrant、Sink 复用与全局完成、参数互斥、双累加器映射 | 每个数值槽数的独立节点；“换一种更快存储”的空泛说明 |
| ABI 重建与 workspace | 与计算绑定的值/地址布局、必要初态、调用/图的存续期、既有跨流限制 | cache 类名、查找数据结构、Python/C 包装调用方式、普通分配 API 的逐条复述 |
| 查表/scan 前缀 | 真正提交时做什么、是否消费、全局归并、完成边界；未提交/已提交后失败的区别 | 与 14/15 个主干做重复完整 YAML 配对；不影响设计条件的错误码数字 |
| 验证与双图选择 | 当前数据谓词、输出零写、device verdict 与 host 观察的版本区别、真实提交、guard、超时行为 | 每个 FC2 的重复 guard 链；精确轮询次数可留来源，不能省掉“有界且可能超时” |
| 描述符回退/拒绝 | 实际 fc8 的搬运协议与适用接口；REDF/WS/split 的不兼容回退；已经发生的副作用 | 为 cfg23→12→8 与 cfg44→8 再各记一份相同的 fc8 全链 |

**正例：不值得再次完整展开。** `no_scan_fused_small_flat` 的 Design 已与父完全相同；保留“ABI 重建可落到相同五次计算提交”以及新增 workspace/服务条件即可。不能为了省一个节点又把这些条件写成未经确认的默认行为。

**反例：不能合并为一句微调。** `mono_split_redf` 相对 `mono_split_mono` 删除 rout 的读写路径，改 Acc32 和 convert，同时引入旁流清零/join、保留 BF16 后再权重相乘，并改变 FP32 加和顺序。把它写成“FC2 融合 finalize”会遗漏初始化、数值顺序和错误回退约束；它应有完整代表设计。last-arriver 的计数和角色屏障同样不能省略。

## 6. 哪些额外节点来自规则，哪些来自误用或源码

| 来源 | 具体证据 | 判断 |
| --- | --- | --- |
| 规则把节点当完整选路分区 | 现 DOC0 [146 行][doc0-when]要求包括 root 的条件独立求值、互斥且覆盖全域；冻结 scope 又要求全部默认入口。[P4 2.2.7 回执:108–127][r-coverage1]拒绝把正常初始化/eager 放进“覆盖缺口”；[2.2.8 回执:101–103][r-coverage2]要求枚举 ABI 前缀与主干组合。 | 这是后续矩阵增长的直接制度性推动，不是 Reviewer 无故追求数量。要采用少量设计目录，必须改变该覆盖职责，而不是让作者悄悄少写。 |
| 真实结构差异须形成完整节点 | DOC0 [162–164 行][doc0-split]与 DOC4 [245–247 行][doc4-submit]要求实际提交/生产者/物化/事件不能用正文删改；同类工作两次提交有不同 step 身份。 | 小准备、查表、probe、第二次 verify 都有真实差异；现规则下它们会进入完整 dispatch。事实“有差异”成立，是否值得成为完整算子设计则仍是新研究问题。 |
| 共享粒度和来源约束 | DOC0 [111–134 行][doc0-sharing]要求有效完整 Design，整区只能引用直接父同名区域，不能局部 patch 或跨来源引用；必要 Realization 独立填写。 | init_tables 只新增局部工作，却使三整区都不同；FC1 与 FC2 来自不同组织时，单父不会自然提供两者的最优共享源。该限制放大正文重复，不能据此证明所有节点划分本来就是错的。 |
| 旧 Agent 越过结构语义 | I/ws_fused_ws 的 flow 同列 Small/大准备/可选 gather，再靠正文条件取消；成员登记最大范围再靠说明取消。初验 [OPERATOR_REVIEW:19–23、35–37][old-review]有直接记录。 | 这是旧稿与旧审阅的误用。不要用它证明“条件分支天生不能表达”，也不要把不合法的旧参数化当作今天可直接采用的压缩方案。 |
| 现规则已经能参数化的变化 | DOC5 [286–301 行][doc5-if]允许动作条件，当前 [hoist 示例:683–695][doc5-hoist]保留低 fill 的 release→await→compute，以及两侧共同的 release→matmul；成员端点也可参数化。 | Hoist、合法成员数、tile/数量不要求再生完整节点。历史 0/1 repeat 不是当前建议写法；不能把所有源码 if 都归为新 dispatch。 |
| 源码本身的接口与缺陷 | map/BM、REDF 后继、验证 mirror 版本、BeginCapture 失败后仍在私有流调用 pipeline。最后一项可直接对照 [C:9533–9577][c-capture-failure]和[Q/dispatch:188–192][q-errors]。 | 减少节点无法修复源码问题。可以集中保留缺陷/适用条件与实际工作前缀，不应为每个错误出口发明一个“正常算法”。 |

DOC4 [263、405 行][doc4-boundary]已经允许简明写调用前提，并明确“分配 API 不属于本区，但必要初值不能删”。这提供了区分轻量实现细节与必要设计条件的现成依据；仍不足以让完整 dispatch 条件任意失去覆盖，需要单独讨论 DOC0 职责调整。

## 7. 成本与返工：测量支持到哪里

以下费用直接汇总既有去重 `attempts.csv`，采用当时用户给定的 30 分钟缓存写入价格情景，**不是账单，也不是从零提取对照实验**。不含未记录的其他模型工作。

| 已记录集合 | 尝试 | 费用 USD | 说明 |
| --- | ---: | ---: | --- |
| 所有 MoE 正式记录 | 479 | 3,208.42 | 跨 2.2.1、2.2.2、2.2.7、2.2.8、2.2.10、2.5.0 系列，范围和输入稿不同。 |
| MoE 2.5.0 系列 | 253 | 1,902.20 | 不是最终 154 项已经全部完成的价格。 |
| 其中 P5 | 248 | 1,863.86 | 140 Worker、108 Reviewer，涉及 110 个非 root 节点。响应 decision 为 134 complete、102 approve、2 revise、10 空；response approve 不能当暂停批准数。 |
| 上述 P5 中有 `rejected.json` 的四次提案 | 4 | 30.97 | 两次缺绑定入口 anchor，两次未引用实际 raw reference/provider 行。是已付工作中的证据提交/接收成本；不是四次算法设计错误，也不代表可全部删除。 |
| 近期单 root P3 首次批准闭环 | — | 15.83 | 历史 root 复盘隔离 P3，MoE 约 8.575M 累计输入、65 次模型响应。[ROOT_ANALYSIS:29、68][root-cost]；不依赖 154 个派生节点才会产生。 |

逐次原因不能混写为“分支太多”：

| 实际事件 | 可归因的原因 | 与组合重复的关系 |
| --- | --- | --- |
| 初始 `ws_fused_mono` P5 revise | 显式 cfg2=17 的 TMA B box 是 64 行，而 FC2 BN128 的事务等待按 128 行计，旧稿遗漏不能保证进展的源码限制。[回执][r-map] | 是源码错误/限制的审计遗漏；不由复制数量直接推出。 |
| 初始 `zero_fused_ws` P5 revise | AMODE2 的 cfg97 不直接消费 ptok，但父复制稿残留 `Route.consumer=[gather,fc1]`。[回执][r-route]明确称为转录残留。 | 是重复派生正文导致错误的具体证据；仍没有“一份共享即可消灭所有此类错误”的对照。 |
| 默认范围 P4 两次被接受的 coverage revise | 首次 ABI 重建、正常 eager、probe 负返回后继续等未有节点归属。[两次回执][r-coverage1]、[后一次][r-coverage2] | 现规则的全入口覆盖要求与作者漏项共同推动计划扩大；新目标若只记录设计，应把前缀知识分离出来。 |
| 同期另外两次 P4 revise 响应未被接收 | dispatch 已有 154 行，结构化 `plan=[]`；属于提案格式缺项，详见 snapshot 的 `revision_evidence`。 | 不能统计成两次新的分支语义发现，也不能说增加节点修好了它。 |
| 较新 `verify_only` P5 revise | 把共享 host word 的 1 错当本次 device Clean=1，遗漏跨调用版本限制。[回执][r-verify] | 即便仅保留一份验证设计，也必须修；全当“包装细节”省略会丢正确性条件。 |
| 较新 `probe_split_store` P5 revise | 新 scan 的跨 shard OR 只写在数学/正文，`target.tile_reduction` 缺失。[回执][r-scan] | 局部新增阶段没有完整落入正式字段。一次写准并共享该 scan 说明可能减少重复，但已有数据未隔离节省。 |
| 较新 P5 的失败/空结果 | CSV 记录 9 个 cli_error，另 1 个 outcome 为空；还有上表 4 个已完成调用的提案拒绝。 | 路径数、失败重试、工具接收要求、上下文成本混在总额中；不能按“删去几成节点”推算美元减少几成。 |

单 root 复盘还显示六个 P3 样本的上下文缓存读写占费用 81.6%，但这是**六个 P3 样本**的比例，不是 MoE P5 全集的比例。[ROOT_ANALYSIS:48–71][root-context]支持关注重复材料与长交互；它不证明 YAML 行数、引用数或节点数与费用线性相关。这里也没有把 GEMM/FMHA 的全树重审费用套到 MoE 上。

## 8. 未验证项与后续试写应解决的问题

- 本轮复核了节点/状态/哈希和代表差异，没有证明 154 个 `when` 全部互斥可达，也没有重新批准初始 60 或较新 96。初始质量保留意见仍有效。
- 微流水线的每一个实验开关没有逐实例穷尽审计。表中的组合条件来自已读源码和代表 YAML；不能把“存在标准 Act/map 接口”推广成所有 cfg 的合法性证明。
- 默认门的数值阈值描述原实现选择，并不自动给出设计正确性的最大有效域。候选分组应另写成立条件，不应把 `when` 阈值直接当硬件定理。
- 验证 mirror 竞态、描述符不匹配和 BeginCapture 失败后的跨流工作，本轮只做源码静态核对；没有 GPU 执行或故障频率测量。完整记录不能补造来源没有的隔离、版本标记或等待。
- 局部说明能否让独立读者正确组合，需要以两到三组实例试写检查接口；尤其应试 `split + REDF`、`small prepare + flat`、`verify + bulk store`。这项审阅不等同于为所有组合再生成完整 YAML。
- 当前报告支持的直接收益是减少重复定义、减少必须单独维护的组合文档。减少模型调用、费用、耗时与回修传播的幅度都待后续受控验证。

复现入口：

```bash
python schema_discussion/dispatch_method/evidence_moe/reproduce.py
```

脚本只读取归档/原始状态/源码，输出 JSON，不运行提取或检查器。保存结果是 [evidence_moe/snapshot.json][snapshot]；其中有精确节点集合、状态、文件哈希、source 对照、15 组 YAML 比较的不同字段路径、成本分组与原回执位置。`different_design_paths` 是结构定位，不是设计差异数量，尤其列表长度变化只记一个路径。

[scope]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/snapshots/8b281029059476dcfe78d7588953be485983ff2d084fdc1216ff44d2e3b565f0/source/schema_extract/inputs/nvfp4_fused_moe_default_scope.md:5
[archive]: /home/shaohanyong.shy/KernelSchema/schema_wiki/archive/six-operators/README.md:22
[errata]: /home/shaohanyong.shy/KernelSchema/schema_wiki/archive/six-operators/errata/nvfp4_fused_moe.md:5
[pause-moe]: /home/shaohanyong.shy/KernelSchema/schema_extract/archive/six-operators/20261109-pause/moe-nodes.md:5
[pause-readme]: /home/shaohanyong.shy/KernelSchema/schema_extract/archive/six-operators/20261109-pause/README.md:10
[seed]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/98a2fc5608299c823aa883c103f2c1e3d2a0d801bff9269adae166c93d99f3f3/artifacts/init_no_scan_split_ws.yaml:10
[snapshot]: /home/shaohanyong.shy/KernelSchema/schema_discussion/dispatch_method/evidence_moe/snapshot.json
[q-root]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/root.yaml:14
[q-dispatch]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/dispatch.md:5
[q-selection]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/dispatch.md:162
[q-errors]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/dispatch.md:188
[q-no-scan-flat]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/no_scan_fused_small_flat.yaml:47
[q-no-scan-flat-services]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/no_scan_fused_small_flat.yaml:80
[q-init-math]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/init_no_scan_fused_big.yaml:47
[q-init-flow]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/init_no_scan_fused_big.yaml:229
[q-init-pipe]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/init_no_scan_fused_big.yaml:465
[q-root-flow]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/root.yaml:375
[q-small-flow]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/fused_small.yaml:203
[q-flat]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/fused_small_flat.yaml:38
[q-mono]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/fused_small_mono.yaml:114
[q-split-math]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/split_store.yaml:88
[q-split-flow]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/split_store.yaml:252
[q-double-math]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/double_verify_root.yaml:46
[q-double-flow]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/double_verify_root.yaml:370
[q-verify-only]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/verify_only.yaml:145
[i-dispatch]: /home/shaohanyong.shy/KernelSchema/schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/dispatch.md:5
[i-limits]: /home/shaohanyong.shy/KernelSchema/schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/dispatch.md:76
[i-split-act]: /home/shaohanyong.shy/KernelSchema/schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/mono_split_mono.yaml:340
[i-arrival]: /home/shaohanyong.shy/KernelSchema/schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/mono_arrival_mono.yaml:298
[i-redf-math]: /home/shaohanyong.shy/KernelSchema/schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/mono_split_redf.yaml:170
[i-redf-flow]: /home/shaohanyong.shy/KernelSchema/schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/mono_split_redf.yaml:380
[i-selfprep]: /home/shaohanyong.shy/KernelSchema/schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/selfprep_mono.yaml:233
[i-bf16-tc]: /home/shaohanyong.shy/KernelSchema/schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/bf16_tc.yaml:192
[i-bf16-warp]: /home/shaohanyong.shy/KernelSchema/schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/bf16_warp.yaml:199
[p-selection]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/solution.py:1170
[p-cfg2]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/solution.py:1202
[p-family]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/solution.py:1367
[p-workspace]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/solution.py:978
[p-tables]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/solution.py:1005
[p-zero]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/solution.py:1478
[c-direct-submit]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:9102
[c-arrival]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:3756
[c-ws-arrival]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:5270
[c-redf]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:3948
[c-redf-init]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:9164
[c-finalize]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:9330
[c-selfprep]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:4034
[c-activation-submit]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:9281
[c-assert]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:4233
[c-99]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:7780
[c-gather-submit]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:9245
[c-w1]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:5420
[c-ws]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:4215
[c-ws12]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:6201
[c-flat-select]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:7222
[c-flat]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:5822
[c-tstore]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:5210
[c-route-map]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:657
[c-small-submit]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:9211
[c-pdl]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:9092
[c-probe]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:9585
[c-guard]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:208
[c-zero-contract]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:1707
[c-zero-publish]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:1764
[c-graph]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:9471
[c-map-cache]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:7165
[c-refusal]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:8722
[c-capture-failure]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:9533
[doc0-when]: /home/shaohanyong.shy/KernelSchema/schema_docs/0.common.md:146
[doc0-split]: /home/shaohanyong.shy/KernelSchema/schema_docs/0.common.md:162
[doc0-sharing]: /home/shaohanyong.shy/KernelSchema/schema_docs/0.common.md:111
[doc4-submit]: /home/shaohanyong.shy/KernelSchema/schema_docs/4.design.work_schedule.md:245
[doc4-boundary]: /home/shaohanyong.shy/KernelSchema/schema_docs/4.design.work_schedule.md:263
[doc5-if]: /home/shaohanyong.shy/KernelSchema/schema_docs/5.design.tile_pipeline.md:286
[doc5-hoist]: /home/shaohanyong.shy/KernelSchema/schema_docs/5.design.tile_pipeline.md:683
[old-review]: /home/shaohanyong.shy/KernelSchema/schema_extract/archive/six-operators/20261103-trial/OPERATOR_REVIEW.md:19
[root-cost]: /home/shaohanyong.shy/KernelSchema/schema_extract/archive/six-operators/20261109-usage-review/ROOT_ANALYSIS.md:29
[root-context]: /home/shaohanyong.shy/KernelSchema/schema_extract/archive/six-operators/20261109-usage-review/ROOT_ANALYSIS.md:48
[r-coverage1]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/53d4bebe-882d-4a66-98b1-287d132dd6f3/attempts/ad7c88fe-7f10-414f-b0b6-7e364c6fbbbc/receipt.json:108
[r-coverage2]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/2c4207c8-000e-43ea-8da0-2eb6a6079aee/attempts/a87ed212-0fc6-4b8a-9e60-a6729b9ab903/receipt.json:101
[r-map]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/3563c3b7-2b7c-4967-9d41-9c2042e53d2c/attempts/ed5b6e28-f535-458a-9ed2-46d0c60b0837/receipt.json:115
[r-route]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/3563c3b7-2b7c-4967-9d41-9c2042e53d2c/attempts/c4d57ca6-8d54-4f59-84fa-df36adbd1ba9/receipt.json:155
[r-verify]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/attempts/95072422-523c-4655-9c39-80916c0f6bc9/receipt.json:194
[r-scan]: /home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/attempts/b93580ed-e502-4ff0-8e75-430ddf58dd56/receipt.json:231
