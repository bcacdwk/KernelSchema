# M04：EXTRACT_SCHEMA / MLA 提取试走草稿

本稿是按任务授权手工走到 P2 的工作记录，不是 P1/P2 放行回执，不是正式 schema 或 dispatch。没有运行 Python、extract.py、checker、GPU、构建或测试，没有修改正式材料，也没有启动子代理。只使用当前 docs、problem 和 candidate；没有阅读旧 schema 的 Design/dispatch 或此前讨论报告。

## 阅读路径与身份

先读 AGENTS.md、EXTRACT_SCHEMA.md、P1/P2 阶段页、EXECUTOR_ALIGNMENT.md，随后读 common、Contract 规范相关段、模板说明与 starter 硬件占位；按实际疑问读 REVIEW 的候选降级/来源缺项及 DOC4/DOC5。为核实片段边界，最后补读 P3 的逐段职责，没有执行 P3。

本次只读核对时 `git rev-parse HEAD` 为 `0c662ebcf2cb57d6b417c12498c3adbf39071a44`；对 candidate/solution.py 与整个 problem 的 `git status --short` 无输出。下文代码行号均指目录 `kernel_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/` 内当前文件。可确认的固定实现入口是 `candidate/solution.py::Model.forward`；reference 是 `problem/reference.py::Model.forward`；`problem/solution.py` 则是 FlashInfer baseline，三者不可混用。未核对仓库远端标识，不在本试稿拼造完整 meta_data.source。

EXECUTOR_ALIGNMENT 明说“不能用当前工作树的 CLI 端到端运行这套新方法”；模板也说“复制骨架不能直接得到符合现行规范的草稿”。这些文字没有要求停止文字分析。P2 的正式准入“P1 的任务元数据和共享 Contract 已通过”在本实验未满足，因此以下是试做，不冒充阶段推进或工具放行。

## P1：已确认事实与缺口摘要

- 任务是 prepared ragged expanded MLA attention，Q/K 为 E4M3 D192、V 为 E4M3 D128，结果写入调用者 BF16 out；可有 FP32 log2-LSE、每头 attention sink。projection/RoPE/cache writer 不在调用内。
- shape_range 的最终交叉约束是 `query_lengths == kv_lengths`，且头几何实际列出 H=Hkv∈{12,24,32,96,128}，请求数 1..32，单请求长度 1..16384，总 Q≤16384。causal、sink、return_lse 都有真假域。metadata 长描述中的 H16/48/64、首次 causal 等不能盖过这些具体域；candidate tiny 注释里的“shape_range allows q<k”也与当前文件不同。详见 M04-02。
- input.py:66–133 从带页准备数据产生连续 ragged Q/K/V，page_size 最终不作为 attention kernel 的页访问策略。scale 构造为 `bmm1_scale=192**(-0.5)*q_scale*k_scale`、`bmm2_scale=v_scale`；FP8 默认 q/k/v scale 为 .5/.25/.75。out 为带前后保护存储的调用者视图；return_lse=false 时构造 lse=None。
- reference.py:6–46 对存储值转 FP32，score=`dot(Q,K)*bmm1_scale`，按 causal/right-aligned 可见域及 window 做掩码，在线累积 softmax 和 V*bmm2_scale。sink 相当于只增加分母的虚拟 logit；输出按其质量重归一。Model.forward:76–90 用 FP32 临时 out，最后复制到 BF16，LSE 除 ln(2) 写回。返回结构/身份由 returned() 保留。正式 Contract 仍需逐一登记签名里的 ABI 对象，本稿没有逐项完成。
- metadata:13–51 明确 out/lse 全逻辑区、保持返回别名和非写区，workspace 是许可 scratch；默认 atol=.01/rtol=.05/max_mismatch_fraction=0，mutated_inputs.out 改为 .005 mismatch 并有 max_rel_l2=.01。未获得 comparator 实现，relative_to、rel-L2 归一化、非有限值处理与返回值/原地值的检查路由尚不完整；不把 candidate 文件头的“frozen checker”说明当独立验收实现。当前计算事实可确定，按来源验收记录继续，不能宣称数值保证。M04-03。
- candidate 的 Model.forward:2076–2158 固定传 `_PV_MODE=0`；会接受某些任务域外 legacy 4D/缺省 out 分支，但这不扩大共享任务。对 lse 的写入由 lse 是否存在决定，不应仅由 return_lse 推断。构造器当前把两者关联，其他组合尚未完整审阅。
- 硬件保持 starter 占位；本稿不确认设备能力/SM 数、不引用测量性能为保证。源码使用 CTA、TMEM、TMA 的事实仍是必要设计证据。

## P2：有据的机制概览

### 入口与 host 准备

1. `_PV_MODE=0`（candidate:80、2150），所以生产调用实际会选择 `<0,3>` 或 `<0,4>`。MODE1 的 FP16-V preparation 和 `<1,3>` 在源码存在，但本次入口无参数可切换，暂放在已见但不收录的范围说明，不能称为当前 workload 的 fallback。
2. `launch_impl`:1648–1681 用静态 nonblocking side stream 把当前 seq/cu 元数据 DtoH，等待该 side stream，然后在 CPU 生成 ItemGeo 表。表包含绝对 Q/KV 地址、head、qvalid、kv_tiles 等。不是文件头所说的 device-side ragged 扫描。
3. 默认表序为 head 升序、batch 升序、query tile 降序（1782–1818）；对 ragged 输入，在所有 qlen≥128、scatter 未触发等条件下，稳定按 kv_tiles 降序重排（1820–1907）。这是相同独立工作集合的排列变化，输出仍绝对寻址。
4. 表 HtoD 与 event record 在 1908–1912，计算 stream 在 2011 等 table_event。已见这条上传到主 kernel 的完成边。没有找到 side stream 读输入前等待当前 stream 生产者，以及下一次写静态 d_table 前等待上次 kernel 最后读者的边。M04-07 阻塞完整调用保证，不抹掉前面已确认的机制。
5. 源选择：`items*2 <= nsm_l*3` 时 grid=items，否则 grid=nsm_l（2008）；scatter 依据当前几何及 runtime cache/SM 属性选择 `<0,3>` 或 `<0,4>`（1696–1718、2016–2027）。这些是来源 selector，不是我已证明的可行域。它们应在目录保存，不放成一份 YAML 的配置选择器。

### 核心计算与角色

- 一个 worktile 是一个 head 的一个 128-query tile，局部遍历 128-key tiles。QK 的 D192 拆成 4+2 个 k32 步（1281–1300），FP32 S 在 TMEM。
- warp0..3 做 mask/max/softmax；warp4..7 correction/epilogue；warp8 发 MMA；warp9 发 TMA；warp10 初始化协议并在 queued 模式充当 pop server。warp11 无主要动作，不因 launch=12 warps 就虚建空 role。
- softmax 的实际数值比文件头更复杂：maxfreeze 阈值 .75，只在 drift 超阈值时推进 max（965–976）；exp 参数转 f16x2、近似 exp2、f16x2 分组累加，MODE0 将 P×448 转 E4M3（995–1063）。不能用文件头“unquantized fp32 row sum”替代这条链。
- softmax 先发布 acc_scale，再算/pack P；correction 等上一 PV 和当前 scale，warp-uniform scale==1 时跳过 TMEM 乘法，但仍发布 o_rescaled（1128–1167）。这些是当前方案必要的状态判断，不另按每个 scale 值建候选。
- PV 仍由 warp8 使用 TMEM P、SMEM V 产生 TMEM O，4 个 k32 步；epilogue 等 PV，归一化、sink、BF16 store，输出可选 log2-LSE（1181–1247）。
- tiny：MODE0、kv_tiles==1、qlen≤32、row<32 且 warp-uniform mcol≤32 时，softmax 额外重读 S、用 FP32 exp2/FMA 算实际 O 的 SMEM 累加器与分母（698–738、1078–1087）。epilogue 从 SMEM 取结果（1210–1217）。原 E4M3 P store 和 PV MMA 仍发生，不可在 schema 中删除“无用 PV”。注释说 scratch 覆盖 stage1 K，实际当前使用独立 SMEM_TINY；以实际访问为准。

### 工作分配与寿命

- bootstrap：grid=items，每 CTA 初始 ItemGeo[blockIdx.x]，无补领，完成自身工作后退出（770–785、863–884）。
- queued：grid<items，仍先一 CTA 一份 bootstrap。softmax 在当前工作最后一个 key tile 开始时发 pop_req（899–903）；warp10 等请求，atomicAdd，领取 `gridDim.x+old_counter`，通过交替 ItemM 槽发布，失败发布 kv_tiles=-1 sentinel（1457–1486）。
- queued 的 Q 预取等待下一 ItemM 和当前末 QK 完成后覆盖单 Q buffer（1427–1442）。K/V ring 的累计 T 跨 worktile 不重置，S ring 同样如此；O/fin/tiny scratch 复用受 o_done 控制。epilogue 普通路径在最后一次 TMEM read 后、最后 BF16 pack/store 前可发布 o_done（1220–1227），不能把它写成全局 O 已写完。
- KVST=4 与 S=3 时 loader 的覆盖边用额外 kv_free，issuer 在 PV commit 后也 commit kv_free（1329–1330、1361–1362、1391–1414）。KVST=3 则 loader 直接等同周期 pv_done；所以 3→4 不是只改 depth 的纯数值替换。

## 拟 root 与候选去处

这里只规划 2 份完整代表；未来文件名均是普通文本“仅计划”，没有链接到不存在的 YAML。共同 Contract 的完整域不会缩到以下子域。

| 候选/线索 | 暂定去处与理由 | 依据与未决边界 |
| --- | --- | --- |
| MODE0、4-slot K/V、bootstrap、head-major、普通 E4M3-P 输出 | 拟 root（root.yaml，仅计划）。一 CTA 一 worktile、无跨工作补领，适合作为清楚的基线。实例族可选所有请求长度>32且至少一个<128，以关闭 tiny 和 LPT，同时满足实际 bootstrap 与非-scatter 条件；如单请求长度64、H12，其 items=12，是否选此 profile 仍依实际 runtime gate。本稿没有替它编造 SM 数。 | 80、1078、1848–1859、2008、2023–2026。代表必须写自身范围；尚非正式选定。 |
| MODE0、4-slot K/V、queued、固定采用 LPT 排列、普通输出 | 拟保留第二完整代表 persistent_lpt.yaml（仅计划）。主要领取/退出、pop role、跨工作 Q 预取、T 累计相位、O 槽寿命需要贯通解释。不能用“root + queue”一句局部记录包住。 | 863–904、1257–1280、1370–1486；符合 common 明举的持久工作独特机制。跨调用前提仍受 M04-07 阻塞。 |
| head-major 与稳定 LPT | 暂降为局部排列记录，不按 static/queued 配对新建完整 YAML。变化定位既有 ItemGeo 工作编号与 schedule，固定工作/输出/局部 kernel 组织继续成立。 | 1782–1907。不能把原 gating 留成 YAML 内模式开关；LPT 表达试读见 M04-06。 |
| K/V depth=3 与 depth=4 | 暂降为局部记录，包含容量、slot、release 协议联动；不是仅参数改值。边界是 LOAD 的 K/V slot 覆盖与 ISSUER 的 PV 完成发布，主体 QK/softmax/PV/工作领取不变。 | 1329、1361、1393–1414。是否所有跨工作相位边都闭合尚未证明，P4/P5 仍可推翻降级。M04-05。 |
| tiny FP32 输出覆盖 | 暂倾向局部记录，但保留待审，不声称已解决。QK、普通 P/PV、分配及原角色仍在；额外数值链/SMEM 产物/fin_sum 语义/epilogue 来源可围绕已有值与 role 定位。另一方面它确实改变主要输出累加者，不能只写“精度优化”。 | 698–738、1078–1087、1182–1217。当前规则不足以让我自信地断言一定局部或一定完整，M04-04。本稿 root 排除 tiny，故没有在正文藏输出选路。 |
| causal、sink、可选 LSE、有效行掩码 | 任务分支，留在所选具体设计的实际执行中；不是配置候选列表。 | reference 和 shape_range 明确允许；mask/epilogue 的当前执行分支。 |
| acc_scale 早发布、MAXFREEZE、scale==1 跳过 rescale | 当前固定入口已经采用的共同机制，直接写代表正文；未找到用户可选择的另一个版本，不从历史注释制造候选。 | 965–994、1137–1167。不同历史版本性能叙述仅作为线索。 |
| MODE1 FP16-P/V prep、legacy 4D | 当前固定任务入口不收录；说明源码范围。不称为参数自动可选或已交付独立方案。 | `_PV_MODE=0`；Model.forward 的 legacy 分支自称 non-frozen；MODE1 prep 1913–1949。 |
| persisting L2 table window、descriptor/allocation cache | 当前来源行为先简述；未见改变数学和主工作组织的证据，不凭 cache API 数量建完整节点。必要性能选择/寿命事实后续到对应字段和 Realization。 | 1730–1779、1950起；cache 复用的真实保证受 M04-07 限定。 |

## 最小字段试写

### root 的工作分配

下面只展开 DOC4 中 decomposition 的工作坐标、mapping 与 schedule；target/输入块/上游符号未写，因此绝不是完整 work_schedule。`Items` 是待在正式上游定义的 ItemGeo 条目数，本段试写前已在 prose 定义其值为 `H*sum_b ceil_div(Qlen[b],128)`。这里没有把 LPT 作为可选开关。

```yaml
decomposition:
  strategy: Each item selects one head and one query tile from the current geometry table; table order is head ascending, batch ascending, query tile descending
  partition:
    item:
      meaning: One concrete query tile and head described by ItemGeo
      size: Items
      tile: 1
mapping:
  worker: {unit: CTA, unit_count: 1}
  worker_count: worktile_count
schedule:
  worktile_order:
    - {by: item, direction: ascending}
  worker_assignment:
    preassigned: {start: worker_id, count: 1}
    refill: none
    exit: assigned_done
```

这个片段最初我差点只写 `refill: none`；DOC4:218 明确要求同时给 preassigned，因此补了 `{start: worker_id, count: 1}`。没有靠 mapping 的数目暗示归属。用 item 维度避免把 ragged batch/tile 硬拆成带 padding 的笛卡尔积；需要在正式数学/分解中保留 ItemGeo 的坐标定义，而不是只剩一个神秘表号。

### 4-slot K/V 的一条等待

这是 LOAD role 下的单条试写；不是完整 pipe。假定现有 `load:KV(g)` 已声明为 TMA 载入 K/V 到第 g%4 槽，`ISSUER::matmul:P(g)@V(g)` 是对应 PV，done 包含 V 源已被消费完；g 是当前工作按顺序的 key-tile 索引。完整稿仍须声明这些 noun、索引与所有 QK/P/输出依赖。

```yaml
waiting:
  released:
    - "{issue load:KV(g+4)} ← [{done ISSUER::matmul:P(g)@V(g)}]"
```

源依据是 loader 等 ring-matched kv_free，issuer 于对应 PV 后 commit。`released` 是前驱提供旧槽可覆盖条件，不要求 verb 本身叫 release；字符串加引号，左右都显式 issue/done。对 bootstrap 用局部 g 足够；persistent 需要 DOC5 flatten 的累计序号并连续穿过 worktile，不能每 worktile 令相位归零。本稿没有把这条边当作已经证明实际完整协议安全。

## 未审范围与下一步判断

P1 没有完整 Contract/逐输出路由审阅；P2 未取得独立 reviewer 结论。主要工作组织已有足够线索支持“bootstrap 与 queued 分别有完整价值”的初选，但 tiny 的降级、K/V ring 局部闭合和 host 共享资源前提尚未全部解决。不会据此宣布 P2 已通过。

未逐条审 TMA map/descriptor、全部 PTX 原语完成保证、输入域外 ABI、线程寄存器/容量、所有 rejection 前缀、MODE1 内核、计数器在并发调用下的行为，也未穷尽所有 shape。现有 P2 选择没有用“未读就一定有另一设计”推定无限调查。

最重要恢复点：先取得或核实调用者与 side stream 之间的真实输入就绪/静态表复用前提（M04-07）；再用实际局部接口判断 tiny 与 K/V ring 是否足以降级，不为争取两份 YAML 的结论省略必要事实。工具对齐前继续手工可做部分，但不声明工具检查或正式阶段批准。
