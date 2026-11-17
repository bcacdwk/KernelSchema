# FA4 prefill：第一轮选型交接

建议交付 **3 份完整设计**：`root`（compact static）、`packed_persistent`、`split_kv`。本文是下一位独立审阅和实施者的选型依据，尚未修改 YAML；拟共享须以新正文的有效整区再核对。

## 来源与共同 Contract

固定来源是 `KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019`，详见 [SOURCES.md](../SOURCES.md)。下文 `P` 指其中 `fa4_prefill/`，`A` 指 `P/aka_solution/`，`F` 指 `A/flash_attn/cute/`。实现为打包 v37 延续树（v43 包标签），不是 `CAMPAIGN_V53_NOT_SHIPPED.json`，也不是 `P/solution/kernel.py` 的 page128 upstream seed。真实调用为 `A/solution.py:Model.forward` → `F/interface.py:_flash_attn_fwd` → `F/sm100_hd256_2cta_fmha_forward.py:BlackwellFusedMultiHeadAttentionForward`，必要时接 `F/flash_fwd_combine.py:FlashAttentionForwardCombine`。

- 任务是 FP8 E4M3FN Q 和 HND paged KV 的右对齐 causal GQA，16 Q heads / 1 KV head / head_dim 256 / page 64。输出是传入 `out` 的 BF16 更新及同对象返回；workspace 是 scratch，所选实现忽略它及页数前缀，但 Contract 保留全部 ABI 参数。`reference.py:11–121` 定义显式页 gather、FP32 score/softmax/value product 及 BF16 写回；`input.py:19–35,88–127` 给出正 query 长度、query≤KV、page 容量和输入构造。
- **旧 Contract 有实质遗漏**：公开域不是从 30 个样例推断。`P/shape_train.json` 的 `generator.domain_policy` 是 `widened_non_enumerating`（18）；`shape_domain.seq_lens_values.length` 明定 1..16（243–270），`query_start_loc_values.length` 明定 2..17（186–213）；两者 items 分别为 1..131072（216–242）、0..16384（159–185）。顶层 `invariants` 明定 Q prefix 起点 0、总 Q≤16384、max KV≤131072（22–26）。`operator_contract.fixed_parameters`（29–43）固定 TableColumns=16434、WorkspaceBytes=413138944、bmm1_scale=0.0625、bmm2_scale=1 及上述 heads/dim/page。`input.py` 的通用函数不扩大这份明确公共域。
- `metadata.json:2–52` 交叉确认 HND、out mutation、standalone callable、composite scales、workspace 首次隔离使用前清零，并分别为 return output / mutated out 给出 atol=0.06、rtol=0.04。不能给 Q/K/V 另外添加量化 scale，也不能把外部 workspace 换成 split partials。
- 固定验证记录在 `R/validation-20261021/results/fa4_prefill/{validate.json,profile.json}`；其 environment 明记 L20D、SM count=148、capability 10.3。仅用已有事实固定代表的部署参数；Hardware 区按 COMMON/P1 保留相应占位，不能把旧 YAML 里的整套硬件声明原样当已核实资料。`AGATE_VALIDATION.json` 是另一份历史 30/30、atol/rtol 记录，不是本次重跑；新文档不声称任意公共输入或每个 dev knob 均验证通过。

## 新完整设计

共同已选设置：pack GQA；cluster=2 CTA，每 CTA 384 threads；cluster Q rows=256（16 tokens）、KV block=128、channel halves=128；Q slots=2、score/P slots=2、KV stage=4；page64 TMA；SM103 native fast exp2（无多项式替代）；max offset=4、rescale threshold=4、masked/unmasked loop 分段、conditional correction、coalesced epilogue、early O release。它们在 YAML 中固定，不保留环境变量选择器。

| 稳定 ID / 文件（仅计划） | 真实已选配置与入口 | 独特完整价值 / 范围 | 可共享整区 |
| --- | --- | --- | --- |
| `root.yaml` | compact **static**；双 K/V load；请求按 KV 长度降序，request 内 `m_raster=2`；不共享 alpha；`persist_mode=0`。实际代表是 `shape_valid["6"]`：B=3，Q=[2019,584,4200]，KV=[6243,6067,4200]，T=6803，host K bound=6272，split=1，small persistence gate=false。入口为 `F/interface.py:935–1020,1146–1299` 和主 kernel `:1669–1791`。 | 一份工作独占一个 CTA pair；compact 网格去掉矩形请求乘积中的大部分空工作，CTA 前置映射完成后四类主角色处理同一 `(b,m)`。作为来自真实 scored shape 的单工作阅读入口。保守描述 small-batch compact static 子域；原 selector 何时选择它写入 dispatch，不能把拒绝 persistence 的阈值做成正文配置开关。 | root 三区全本地。数学使用 request-local `(b,m)`，不要把 grid ordinal 写进主公式，以便 persistent 直接复用。 |
| `packed_persistent.yaml` | `persist_mode=1`，compact grid，双 load，不共享 alpha；保留原请求次序；每 CTA 一次建 req_map，四步二分查段，request 内 `m_raster=2`。实际代表 `shape_valid["3"]`：B=9，T=16384，max Q=3584，K=7808；Cpad=1032，74 个 CTA pairs，split=1。入口同上；kernel `:1577–1662,1170–1263,1828–3210`。 | 多份 query 工作静态步进复用一组 pipelines / TMEM；Q 的下一工作加载、旧工作 PV/correction/写回可重叠，主要资源寿命跨工作。范围保守取 5≤B≤16 的这套已选配置。 | `Math_reshape: {from: root}`；Work_schedule、Tile_pipeline 本地。若实现时数学绑定/retention 名称不相容，应统一 root 的通用工作标识后复核，不抄相同数学两遍。 |
| `split_kv.yaml` | **Ssplit=5 固定**；rectangular static，**联合 K/V load**（一个 load warp），`m_raster=2`；每 split 输出 FP32 normalized partial O + natural-log LSE；combine 固定 tile 8×128、256 threads、4 stages、max_splits=32。实际代表 `shape_valid["0"]`：B=1，Q=200，KV=8648，host K=8704；148 SM 时 `_hd256_split_heuristic` 返回 5，joint-load gate成立。`F/interface.py:562–606,1005–1020,1758–1791,3896–3940`。 | 新增 KV 工作轴、空 split、全局 partial 及另一 kernel 的最终输出归并，需要整条输出形成重新贯通。保守取 B=1、所选 five-split / joint-load 调用子域；split 数及 dual-load 为目录变化。 | 三区均本地；不能只从 root/persistent 引用数学再由目录补 combine，也不能隐藏 Ssplit/Smax 的 selector。 |

这里选择 root 不以旧文件名为约束。旧 root 的 `B>16` 在明确公共域外；其 B=1 的矩形分支仍真实存在。例如 B=1、Q=1024、KV=131072，在 148 SM 上不 split、不 persistent、dual load，域内且走该分支，但它不是另一种主要 attention 协作。

### 旧 11 份的去向

| 旧节点 | 新去向与理由 |
| --- | --- |
| `small_static_lpt` | 成为新 `root`；固定代表及设置，不再携带整个 host selector。 |
| `root` | 相对新 root 的**矩形网格局部变化**：改变候选集合、坐标解码、worker 数及映射准备；同一有效 `(b,m)` 仍由一对 CTA 完成，主数学、单工作寿命和角色协议不变。删除旧域外 B>16 主张。 |
| `rect_joint_load` | 上述矩形变化与联合 load 变化的**已知组合**，不单独建节点。 |
| `packed_persistent` | 保留，重写为当前完整 persistent 代表，数学整区共享。 |
| `single_persistent_dual_load` | persistent 的**请求映射局部变化**：B=1 时 req_map 退化、禁用多请求排序、线性 trivial lookup；跨工作主协作不变。 |
| `single_persistent_joint_load` | 单请求映射与联合 load 的**已知组合**。 |
| `small_persistent_lpt` | persistent 的**请求映射局部变化**：B=2..4 排 KV 长度、短线性查段，保留 stride worker / 连续环位置。 |
| `small_persistent_shared_rescale` | 上述 small-request 映射与 shared-alpha 的**已知组合**。 |
| `small_static_shared_rescale` | root 的 **shared-alpha 局部变化**。 |
| `splitkv_joint_load` | 成为新 `split_kv` 的真实选中代表。 |
| `splitkv_dual_load` | split_kv 的**独立 K/V producer 局部变化**及所需范围放宽；split partial 和 combine 主链原样成立。 |

compact static / rectangular static 的降级依据必须由下一位重新核对：变更封闭在已有 `MAIN` 的 work index、前置映射和空候选过滤，没有独立输出 producer、跨工作领取协议或不同 accumulator 寿命。若新审阅发现这些边界不成立，应据具体事实重开判断，不能为维持三份数量而强行合并。

## dispatch 应记录的独立变化

正式对象名随新正文落实；以下名称是定位建议，不能当作已存在定义。每条直接链接完整 YAML 和其已有对象；组合引用这些条目时仍直接列完整基线。

1. **单工作网格（局部）**：基线 `root` 的 `MAIN` 分解 / assignment / `RequestMap`。compact 使用 `Cpad=ceil((T+15*(B-1))/16)` 个候选 cluster，按 `C_b=ceil(Qlen(b)/16)` 前缀归请求；矩形改为每 request 各 `Crect=ceil(T/16)` 个候选，移除 compact prologue，直接解码 `(request,query_cluster)`，raster 作用于 Crect；两者均每 pair 一候选，局部排除 `m>=C_b`，其后主角色动作相同。不要把矩形无效候选从成本中抹掉。源码 `kernel:650–672,1669–1808`。
2. **persistent 请求映射（局部）**：基线 `packed_persistent` 的 `RequestMap` / `MAIN.Schedule`。B=1 不读其他请求；B=2..4 用固定比较网络按 `seq_lens` 排序、查段改线性；B>4 保留输入顺序/四步二分；prefix 表及 lookup 的 producer/consumer、CTA 初始化屏障和 request-local raster 联动说明一次。`flatten_varlen_lpt` 参数在 kernel 只有保存，没有控制这段排序；真正 small-batch 排序直接由 `request_count<=4` 控制。不要把同名 flag 误当 LPT 开关。源码 `kernel:1577–1662,1170–1263`。
3. **联合 / 独立 K/V load（局部）**：直接并列 root、packed_persistent、split_kv 的 load role、KV rings 及 MMA consumption。双 load 为 K/V 各四槽独立 readiness；联合 load 关闭第二 producer，用一个 acquire/advance 流按 `K0, K1,V0, K2,V1,…,Vlast` 提交，每 K/V 有两个 channel halves，K/V保留独立SMEM数组，只共用四槽协议token的推进（kernel:1511–1523）；不能把共享token误写成存储别名。两端 slot 序号、QK/PV 消费、覆盖前等待、末尾 tail 均一起修改。主 online attention、工作归属和最终 producer 不变。默认 host 条件是 B>1 或 rounded K≥98304 才双 load（`interface:973–982`）；kernel `:1327–1390,2116–2261,2620–2807`。相同差额只写一次。
4. **shared alpha（局部）**：直接列 root、packed_persistent 的 `Stats` / `SOFTMAX` / `CORRECTION`。由两项 `(old_max,safe_max)` 改为单 FP32 alpha；SOFTMAX 的 denominator rescale 和 CORRECTION 复用同一已计算 alpha，后者取消 exp2；Stats shape、发布/读取和 aliased TMEM 覆盖等待相应调整。warp ballot 的 conditional O rescale仍保留。原 host 只在 B=4、T=16384 的 flattened path开启；与 static/persistent 请求映射的已知组合记录一次。源码 `interface:966–972`、`kernel:3523–3587,3644–3651,3694–3759`。
5. **split 数 / combine 容量（参数）**：基线 split_kv 的 Ssplit、partition、PartialO/LSE、combine split reduction 和环。其余可选数按 `_hd256_split_heuristic` 的 underfill、最少16 blocks、每 split至少2个宽度界 blocks、24MB partial-O预算、最多64来记录；combine max_splits 是至少32的下一二次幂、可到64，必须联动存储和归约范围。所选 YAML 仍 Ssplit=5 / max_splits=32。不能把每个计数建文件或留自动公式在该完整设计内。
6. **实现参数 / 闭合局部路径**：raster 0/1/2、KV stage其他取值、native/polynomial exp2、ldred rowmax、max-offset/threshold、conditional rescale、coalesced/early-release epilogue、req_map hoist/search，以及 page64 cp.async assembly 是源码提供的 dev 变化；只有确有需要时以所属现有对象分组记录，数值改变须说明舍入和近似差额，传输改变须说明完成/发布配套。不要复制所有笛卡尔配对，也不把未实际审阅的 dev 覆盖说成已验证。

原 selector 的 small-persistence 决策整体记录到 persistent 条目的条件依据：wave floor 4；D_bound≤0.3 或 near-pinned 或 D_tight≤0.60（且 K≤8Q）；默认 plain stride，B=1需 ceil(Q/8)>SM+floor(SM/2)。公式源码 `interface:951–965,1212–1287`。它们只决定原实现采用哪套固定配置，不能污染单份正文。

**已研究但未完整展开**：默认关闭的 CLC `persist_mode=4` 具有专用 scheduler warp、取消待启动 cluster、response 消费/退休和提前保留下一工作，会改变主要领取与退出；它是独立候选，不是 stride 的参数。源码 `interface:116–138,1168–1182`、`kernel:1449–1533,3214–3232`、`tile_scheduler:1398–1643`；当前交付聚焦默认路径，目录范围明确未交付 CLC schema。环境 override 的 unpacked GQA / 非 page64 路径不属于本次默认 ABI提取；不按 44题同名实现补全。

## 实施必须保留的数学与执行事实

- **数值主线**：`kernel:3431–3674`。QK 分两个128-channel contraction得到FP32 S；右对齐 causal key集合后求 max。保留 max-lag判定（scaled old-new≥−4 时保留旧基准）、offset=4、fast exp2、FP8 P narrowing、FP32未窄化 exponentials的 denominator。PV numerator使用P，denominator不能也改成sum(P)。denominator的源码四组 packed-FP32累加树用短 arithmetic 说明；最终零/NaN sum guard如实保留。首 PV以不累加方式建立 Y=0 的数学初态，不虚构独立清零 kernel。
- **masked/unmasked**：`kernel:2890–3005` 的 full-block prefix / masked suffix决定 rowmax可复用硬件 ldred；算法 causal边界、末page复制和query tail都保留为本设计的动态条件，不能当待拆分配置。
- **page64 TMA**：`kernel:2116–2261`。K每 CTA取对应64-token page；V每 CTA以两个64-token page搬入128-token×64-channel半块。末半page越界时重用该128-token块首page地址，随后mask保证无效key不入结果；不能伪造零填充完整KV。两CTA transaction完成量覆盖两边提交；不能以host顺序或某一CTA提交代替pair readiness。
- **主流水**：score/P共两槽且物理重叠；P压入原FP32 score的前32列，Stats在其后；Y独立一槽。MMA leader进行2CTA MMA，QK领先PV一块；N=1路径有实际Q释放点。SOFTMAX读score完成、两种Stats/P消费者完成才允许各自覆盖；CORRECTION初次只消费/释放首Stats，后续在 O token ready 后可能执行rescale，跳过乘法也仍取得/释放O ownership。源码 `kernel:1327–1450,2598–2811,3476–3478,3585–3587,3758–3816`。
- **persistent 寿命**：`kernel:658–667` 把逻辑M CTA数补为偶数，避免同pair一个有效另一个无效的2CTA死锁。req_map CTA一次初始化后，四个角色段分别以同一stride序列映射/前进。pipelines participant索引不按 worktile重置；Q在当前工作最后QK消费后可覆盖，PV/correction/旧输出仍可继续；KV ring和score/P按跨工作连续序号连接。LOAD和MMA的producer尾等待在整个worker循环后，最终cluster同步后才free TMEM（2511–2519、2808–2812、3239–3245）。无效padding work仍推进scheduler，但不产operand/Stats/P/O动作。
- **early O release**：`kernel:3891–3984`。CORRECTION先把末channel half从TMEM读完并转换到SMEM，fence后释放O token，再SMEM→register→global；输出stage的group barrier及复用仍成立。不要把O release统一拖到global store完成，否则丢掉persistent相邻工作的实际重叠。
- **split空工作与combine**：`kernel:3250–3289,3136–3147,3987–4033`。split span依据完整used-KV块数，和每query工作causal trip相交；空split只写−inf LSE，不写PartialO、不等待任何未产生的pipeline token。非空partial是FP32的局部归一化O；LSE含`scale*retained_max + log(sum) - 4*ln2`。combine是另一次提交，需显式主kernel完成handoff；`interface:1758–1769` 的LSE transpose只是视图。
- **combine不得理想化**：`flash_fwd_combine.py:398–445,488–513,516–593,595–711`。先搬LSE与最初3个PartialO stages，再计算normalized exp(LSE−max)权重；partial O加载不以权重为条件，空split未写O仍可能实际被搬入。只有FP32累加由正weight保护，避免0×未定义值/NaN污染；`max_valid_split`截短后续循环，4槽cp.async环保留load/commit/wait与线程自身SMEM读取关系。不要写成「zero-weight partial从不读取」，也不要无依据追加完整排空；producer尾行为按源码交代。其输出row顺序head-major，attention packed row为head-fast，需明确coordinate转接。

`A/solution.py` 直接调用 `_flash_attn_fwd`，**不会进入**公有 `flash_attn_varlen_func` 的 `_hd256_fastpath` replay入口（`interface:3748`）。full path有编译cache及可能的fast-record逻辑，但不能从该函数注释虚构本ABI会重用已缓存partial/output的设计。

## 有限图样本与审阅边界

- root建议图样本 B=3，Q=[32,32,32]，KV=[512,384,256]，选首个有效worktile：仍是默认compact static / dual，不split；首请求N=4，可看score两槽和K/V四槽复用。另以无效compact尾候选检查「无payload仍终止」，不另交付完整YAML。
- persistent建议 B=9、每请求Q=512/KV=512、SMCount=148；Cpad=296、74 workers。图展示一个worker连续至少2份工作，允许 `--worktiles 2`；仅画1份不能证实连续slot索引和early-release跨工作关系。绘图参数属于合法缩小输入示意，不改选中的tile/stages/persist_mode。
- split建议仍Ssplit=5，B=1、Q=200、KV=2048（host heuristic亦为5）；完整used-KV blocks=16，span=4，第5个split为空；同时展示一个非空split的4轮及一个empty split/LSE分支，combine维持max_splits32、8×128/4stage。工具若仅能画一worktile，记录主图选哪一split，并另做有限empty路径检查，不能声称图覆盖所有split。
- 新语法应直接使用 `Step / Index / Role / Pipe / Waiting`；条件用`if`，删除旧active/empty假0–1循环。persistent连续序号用DOC5的`flatten:[t,i]`，每worktile不同N不能手写一个固定N乘t掩盖边界。槽覆盖、pair成员完成、aliased TMEM及联合load的共享token/独立K/V存储是工具核对重点。
- 本轮只做源码与公开合同选型；用Python AST单独执行源码split heuristic并人工代入selector，核实shape6=single/static/dual、shape3=single/persistent/dual、shape0=five-split/joint，未import GPU库、未运行kernel、未跑YAML检查。暂未认定新工具缺陷；图/规范化结果由下一位实际执行后记录。fast-exp2/log/division的单原语误差界仍无材料，不补造。

无需用户决定的问题。下一轮应独立审定三份的完整价值、compact/rect局部闭合和上述Contract修正，然后实施；若有新源码事实再调整集合。
