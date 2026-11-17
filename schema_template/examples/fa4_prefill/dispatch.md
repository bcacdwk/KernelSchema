# 分页因果分组查询注意力 Prefill

## 范围

收录打包 v37 延续实现（v43 包标签）的三种已选设计。共享 ABI 为 FP8 E4M3FN Q/K/V、16/1 heads、head_dim 256、page64、右对齐 causal、BF16 原位输出；公开域 B=1..16、总 Q≤16384、KV≤131072。目录记录原 host 依据，不规定隐式优先级，也不把 selector 放进单份 YAML。

固定来源：`KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution`。下文文件名均相对其中 `flash_attn/cute/`；“kernel”指 `sm100_hd256_2cta_fmha_forward.py`。部署证据为 L20D/SM103、148 SM；历史样本不能证明全部公共域或开发开关均验证通过。

已研究但未展开：默认关闭的 CLC `persist_mode=4` 用专用 scheduler warp、取消待启动 cluster、response 消费/退休和提前保留下一工作，改变主要领取与退出，是独立候选。本次聚焦默认路径，没有交付 CLC schema；依据 `interface.py:116–138,1168–1182`、kernel `:1449–1533,3214–3232`、`tile_scheduler.py:1398–1643`。unpacked GQA、非 page64 override 和未打包 v53 不属于本次提取来源或范围。

## 完整设计

| 设计 | 具体机制与保留理由 | 关系 |
| --- | --- | --- |
| [root](root.yaml) | Compact static，按请求 KV 长度降序映射，每 CTA pair 一份 query 工作；双 K/V producer，直接 BF16 输出 | 阅读入口，唯一 Contract；真实代表 shape_valid[6] |
| [packed_persistent](packed_persistent.yaml) | Compact stride；请求表只建一次，四步二分；环跨工作连续，下一 Q 可与旧输出重叠 | 数学整区来自 root，调度/流水本地；真实代表 shape_valid[3] |
| [split_kv](split_kv.yaml) | 固定五路 KV、joint load、FP32 partial O/LSE、独立 combine；空 split 只写 LSE | 三个 Design 区均本地；真实代表 shape_valid[0] |

## 变体与组合

### 单工作矩形网格（局部）

- 基线：[root](root.yaml) 的 `MAIN.Decomposition`、`MAIN.Schedule`、`requestmap`。
- 差异与联动：每请求启动 `crect=ceil(totalq/16)` 候选，worker 数改为 `batchcount*crect`；取消 compact 扫描和共享映射发布，直接解码请求/query ordinal，raster 作用于 crect。无效候选仍启动并在主循环前退出；有效 `(b,m)` 的数学、单工作寿命和主角色协议保持一致。
- 条件与依据：kernel `:650–672,1669–1810`。域内例 B=1、Q=1024、KV=131072 在148 SM上为 unsplit/static/dual；旧 B>16 条件不在公共域。

### Persistent 请求映射（局部）

- 基线：[packed_persistent](packed_persistent.yaml) 的 `MAIN.Schedule`、`requestmap`、各 role 的 `load:work`。
- 差异与联动：B=1 退化为单请求 lookup；B=2..4 以比较网络按 KV 长度降序、短线性查段；B>4 保留输入顺序、四步二分。prefix 表、CTA 初始化屏障和 request-local raster 联动；跨工作 stride 和连续环保留。
- 条件与依据：kernel `:1170–1263,1577–1662`。排序由 `request_count<=4` 控制，保存的 `flatten_varlen_lpt` 不控制该段。host 默认 B>4 使用 persistence；B=1 另需 `ceil(Q/8)>SM+floor(SM/2)`；B=2..4 需至少四波，且 dominance bound≤0.3、near-pinned 或 tight bound≤0.60（KV≤8Q）之一成立，公式见 `interface.py:951–965,1212–1287`。

### K/V 加载分工（局部）

- 基线：[root](root.yaml)、[packed_persistent](packed_persistent.yaml) 的 `LOAD_QK/LOAD_V`、`keyring/valuering`、`MMA`；[split_kv](split_kv.yaml) 的 `ATTENTION.LOAD/MMA`。
- 差异与联动：双 producer 各用四槽 readiness；joint 关闭第二 loader，按 `K0,K1,V0,K2,V1,…,Vlast` 的 channel halves 共用四个 acquire/advance token。K/V 仍是独立 SMEM；两端槽号、覆盖等待、QK/PV 消费和 tail 随 token 序共同变化。online、工作归属与最终 producer 保留。
- 条件与依据：默认 B>1 或 rounded KV≥98304 使用 dual，否则 joint；`interface.py:973–982`，kernel `:1327–1390,1511–1523,2116–2261,2620–2807`。split 改 dual 时可放宽 B，partial/归并链保留。

### 共享 rescale（局部）

- 基线：[root](root.yaml)、[packed_persistent](packed_persistent.yaml) 的 `scoreprobability.stats`、`SOFTMAX`、`CORRECTION`。
- 差异与联动：Stats 的 old/base 两项改成一项 FP32 alpha；分母修正和 CORRECTION 复用同值，后者取消 exp2。Stats 区域、发布、捕获及别名覆盖等待同步调整；warp vote 与跳过乘法时的 Y ownership 保留。
- 条件与依据：默认仅 B=4、总 Q=16384 的 flattened path；`interface.py:966–972`，kernel `:3523–3587,3644–3651,3694–3759`。

### Split 数与归并容量（参数）

- 基线：[split_kv](split_kv.yaml) 的 `ssplit`、`ATTENTION` partition、`partialo/partiallse`、`COMBINE` reduction 和环。
- 差异与联动：其他 split 数联动 KV partition、partial 容量及归约；combine 容量为至少32的下一二次幂，可到64。正文固定五路与32槽。
- 条件与依据：`interface.py:_hd256_split_heuristic:562–606` 要求机器未填满、宽度界至少16个 KV blocks、每 split 至少2 blocks；受24MB partial-O预算和最多64路限制。空 split 只写 LSE、combine 可能读未写 O而仅正权重累加的语义保留。

### 数值、环与传输细项（局部）

- 基线：[root](root.yaml)、[packed_persistent](packed_persistent.yaml)、[split_kv](split_kv.yaml) 的 `stream/online`、KV rings、`SOFTMAX/CORRECTION`、输出 stage。
- 差异与联动：raster 0/1/2、KV stages、offset/threshold 为参数；polynomial exp2、非 ldred rowmax、关闭 conditional correction、非 coalesced/非 early-release epilogue、未 hoist/线性查段、page64 cp.async assembly 为局部选择。数值变化需保留舍入/近似差额；环与传输须联动完成、发布及覆盖关系。
- 条件与依据：源码开发开关；正文固定 native exp2、ldred、offset/threshold=4、四槽、conditional correction、coalesced/early release、page64 TMA，不声称所有组合已验证。

### 已有决定的组合（组合）

- 基线：[root](root.yaml)、[packed_persistent](packed_persistent.yaml)、[split_kv](split_kv.yaml) 及以上对应对象。
- 差异与联动：矩形+joint、单请求 persistent+joint、small-request persistent+shared alpha、small static+shared alpha 已在源码出现，组合上述网格/映射、加载及 Stats 联动；无新增整体组织，不逐对建文件。
- 条件与依据：遵守上述 host gate 与接口限制。本 ABI 的 `solution.py` 直接调用 `_flash_attn_fwd`，不进入公有 `flash_attn_varlen_func` 的 `_hd256_fastpath` replay。
