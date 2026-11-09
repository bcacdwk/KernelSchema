# KernelSchema 六典型算子独立恢复审计 A

> 归档说明：六算子恢复阶段的历史记录。正文中的运行状态、预算、命令和分工对应记录时点，不作为新的执行授权。 [归档索引](../../README.md)。

审计日期：2026-11-05。工作树 HEAD 实测为 `63e437695fef892eec925c5b5009ec91d9b88bd2`；`schema_extract/README.md:3` 声明 2.5.0。此次只读评估，没有执行 extract/resume/continue/revise/upgrade-runtime/cancel/pause/export，没有改源码、schema、规则、运行状态或 Git。报告和小型证据仅写 `/tmp/kernelschema-recovery-audit-20261105-a*`。没有读取其他审计代理结果，也没有委派下级代理。

## 核心判断

六项与 historical six-families 完全对应，不能把目前 81 个推荐节点视为 2.5.0 产物，也不能把历史 COMPLETION.json 当成后来修订已完成的证明。

- **GEMM：keep 2.2.4 三节点校正版。** 已验证正式 export、P6 和文件绑定，已知 graph 销毁返回码的转录修订确实存在；没有基于本次抽查要求整项重提取的理由。
- **GDN：keep 2.2.7 六节点修订版。** `published=false` 是原目标冲突，独立 export 已完成；不要 resume，也不要回退到两节点原稿。
- **MLA：保留 Design，针对当前 Contract 作有限 patch/review。** 原单 root 仍有价值，但输入 attention_sinks/lse 与输出 LSE 缺 `nullable: true`，不满足当前明确的 None 语义。这是规则表达迁移，不是数学或 kernel 错误；若要求正式当前规则发布，因原 run 已 published，需要新目录的新 extract，不能原地连续修订。
- **FMHA：保留现有修订的 29 个有效节点及全部节点自己的草稿，review/有限 patch 后完成剩余 14 个。** 优先同 run 的正式语义维护以采用 2.5.0，而不是从头重提取。当前先要由旧执行器清算 2 个未确认调用；需要经审阅的旧 Design 语义中间执行器桥接。
- **Sparse：保留当前 2/8 的 root/mma_direct 与未完节点稿，review/有限 patch 后完成 6 个。** 当前停止不是旧 tc_direct 源阻塞，而是 1 个未确认调用。布局缺陷已按现有证据分流，不需要先修 kernel 或求出破坏后的随机值；不能把这点当成草稿已完整。
- **MoE：默认范围保持不变。优先当前 run 的 review/有限维护接续；若桥接建设不划算，可对当前 P2 run 显式 full-restart 到 2.5.0。** 目前只有 P1 新批准、无有效 Design 节点，full-restart 的损失显著小于 FMHA；仍保留 root 候选为可编辑起稿。原 60 节点属于不同实现选择 scope，不能沿用其批准，也不能直接把 60 节点当新计划。

以上 keep 表示保留内容及既有身份，不宣称旧批准已变为 2.5.0 有效批准。若用户只要求完成原冻结语义而不采用 2.5.0，三条未完成任务可选择清算、解除相应控制、追加墙钟后普通 resume；这条路最短，但须在到 ready 前决定是否要新语义，ready/export 后不能再用连续修订。

## 路径与身份矩阵

以下相对路径均相对仓库 `/home/shaohanyong.shy/KernelSchema`：

- `O = schema_wiki/aka-cuda-operators-20261030/operators`
- `R224 = schema_wiki/revisions/2.2.4/aka-cuda-operators-20261030/operators`
- `R227 = schema_wiki/revisions/2.2.7/aka-cuda-operators-20261030/operators`
- `RUN(u) = .schema_extract/v2/runs/u`
- `CP(u,h) = RUN(u)/checkpoints/h/artifacts`，这是可核对的 checkpoint 原始 YAML 目录，不等于整树正式交付。

| 算子/家族 | 正式原始发布 | 当前推荐正式版本 | 未完成的当前身份与准确产物路径 | 本次权威状态 |
|---|---|---|---|---|
| fp8_scaled_grouped_gemm / grouped GEMM | `O/gemm/grouped/fp8_scaled_grouped_gemm`，`4aac1ea4-94fa-4b4c-93ca-d48a8eb08996`，2.2.2，3/3 | `R224/gemm/grouped/fp8_scaled_grouped_gemm`，`b61c62b1-b0dd-4ed8-a21f-e9b7227a0d3b`，2.2.4，3/3 | 无待处理后继；校正版原 bound output 为 `.schema_extract/trials/six-families/verification/2.2.4/fp8_scaled_grouped_gemm`，另正式 export 到 R224 | ready/P6，published=true，exports 对应 R224 published；checkpoint `301b901245cd78343bca60556eabfe08c29939e3bc90720321db7da1f0c429eb` |
| fmha_mxfp8_qkv / FMHA | `O/fmha/fmha_mxfp8_qkv`，`acb68909-aaaa-4bc2-8b16-94904752af8e`，2.2.2，8/8 | 原 8 节点，仍带表达保留意见 | `45ea6608-66b2-4915-b47d-59d2c115602e`；`CP(45ea6608…,812602cc9003742176580b62ea078814a926c3a8d230aa42da277aa2ad48e85f)` | termination_unconfirmed/P5；P1–P4 有效批准；29/43；7 cli_error、5 pending、2 termination_unconfirmed；未 P6/未发布/无 export |
| mla_expanded_fp8 / MLA | `O/mla/mla_expanded_fp8`，`e3f7704f-061f-48ff-8131-cc317afe6003`，2.2.2，1/1 | 同原版 | 无待处理后继；旧 2.2.1 P1 blocked 前任已被接续，不能当当前任务 | ready/P6、published=true；checkpoint `2a16c9dcfb5d65b04ee50108349ec9040de6ee479a7c055729f7110a52461ede` |
| dual_cache_latent_bf16 / Sparse | `O/sparse_attention/dual_cache_latent_bf16`，`d5a73c01-7c3a-45b2-a3ce-0d5eec634808`，2.2.2，3/3；阅读索引 `schema_wiki/errata/dual_cache_latent_bf16.md` | 原 3 节点，勘误仅修索引阶段文案 | `8f1be03a-af4a-44f2-a2ba-b8deda6ae3e7`；`CP(8f1be03a…,0876fd66dce267bb9c46cd383bddc47ddfb87d53521d0c2a26425d534d11a356)` | termination_unconfirmed/P5；P1–P4 有效批准；2/8（root、mma_direct）；1 cli_error、4 pending、1 termination_unconfirmed；未 P6/未发布 |
| gdn_prefill_bf16_state_bf16 / GDN | `O/linear_attention/gdn_prefill_bf16_state_bf16`，`97902388-eda0-4cbd-b8a6-ba71440e86ab`，2.2.2，2/2 | `R227/linear_attention/gdn_prefill_bf16_state_bf16`，`45f9bb51-77f3-4285-b7a7-251cf2744fa1`，2.2.7，6/6 | 无待恢复工作；历史 `9699c87b-4ae0-412a-b966-defc914407ef` cancelled/P1 仅历史证据 | ready/P6；原 output published=false/conflict，但 export `4cae56f4243c563903c7102fbb2361c3d7abcd1014b3707be9c486e3aa251a48` 已 published；checkpoint `349f0fdbad54effd79e2557888dffcc88de418c142438fd2989db2b4937d0a2d` |
| nvfp4_fused_moe / MoE | `O/moe/nvfp4_fused_moe`，`3563c3b7-2b7c-4967-9d41-9c2042e53d2c`，2.2.2，60/60；阅读索引 `schema_wiki/errata/nvfp4_fused_moe.md` | 原 60 节点，仍有表达保留意见 | `feb0c655-4e13-43ac-817d-1f457f31c029`；`CP(feb0c655…,ed3b30e50697640fc0b86c8205bddf0e10bdc4308134b7bb1596e9c29c622354)` | cli_error/P2 reviewer，P1 有效批准；0 个已批准 Design 节点，P4 未建立当前 plan；root v6 1025 行是候选，不能因 YAML 检查通过称 P3 完成 |

FMHA 和 Sparse 的 bound output 仍为 O 的原发布目录，已占用；历史“intended revision directory”不是执行器已改输出的证据。完成后应 `export` 到新的修订目录；普通原目标发布将冲突并保留原版。MoE bound output 是 `R227/moe/nvfp4_fused_moe`；历史记录希望 export 到 2.2.10 目录不等于已经 export。采用 2.5.0 后应使用准确的新修订目录/回执，不把目录标签当 rules 身份。

### source/config、rules、executor

原六条 source.ref 都是 `b5c8e870e5534ae629325e56aa0ba28c3f00486f`。同源码后继的 ref 可变为 `6e4ca91…`，但这不等于 candidate 被改；我按 lock 的逐文件表及 role bindings 对比，GEMM 原版→2.2.4、GDN 原版→2.2.7、FMHA 原版→当前、Sparse 当前与前任均 source files 和 bindings 一致。当前候选源码路径的实际工作树 SHA 与所查固定 manifest 一致。

| 身份类别/版本 | rules_id | executor digest |
|---|---|---|
| 原六项 2.2.2 | `5173c550e7fd4cd936098ccf6f9db5c6546ecdc9d2b89cf436f759ee2a7a43c3` | `95fe1cfcd79a8d3a4389855303f3377cc83f967d21dda574a3df18aaad209ac8` |
| GEMM 校正版 2.2.4 | `79680055ec6e2017c4862bf01ea43d31a1403fd096715cfe9492fd6967f10507` | `942a592bea18516801efcea7d6b6f2c88d7539f5b2ffa83cda6a33109c68737e` |
| GDN 修订/FMHA 当前 2.2.7 | `66554a12e7c54257f0c3c98947a4daf62b5e406d7c5b6db45474054230da5d97` | `5ebe83bf45aaad1d50202207b4560f8a0697056105c14ba5122e81916a9e1d69` |
| Sparse 当前 2.2.9 | `1ab0e7ce27619a83a415c2a6a5785d58c192cd5771427cfb4cf5767a1d8ca534` | `025f94a1b1edb82d622882d2082290bcce722e35bcb6e695fc3a80386140e7e2` |
| MoE 当前 2.2.10 | `114b8626ae972f48fd42b7c8e29159191b0c5ba452948f315d09f6fa1cdcf323` | `b0ca2d045e6ef0e8bc4d714b21655317a0f2fe5eb531ff5a36aa6c31041f4c5b` |

三条当前未完成任务的 execution 均为 `gpt-6-astra`、worker/reviewer xhigh、Codex CLI 路径 `/usr/local/lib/node_modules/@openai/codex/bin/codex.js`。没有登记 semantic_revision 或 execution_binding，executor capabilities 均不含 `semantic_revision.v1` 和 `scoped_recovery.v2`；不能据工作树新代码假定可直接 `continue` 或 `resume --scope`。

FMHA source_id `7aa9d67d65afbe0703332ba3037319a1907f0d2f9ef6e7e6e73f4315e4365247`（35 files）；Sparse `b15be39f9906641000dce91a3aaf0d8c52faef47faa3014d82bdfa108bd0e356`（31 files）；MoE `880672bc1e079f44262d9d5d2e95300b29fc7501f43e4c6e87d9d2bdcb3f5456`（41 files，dirty-source ref `ce1b7d5e8e7560b5c80abd87549bc2328c52df34`）。MoE 的原 60 节点与新 run 唯一 source-files 增量是 `schema_extract/inputs/nvfp4_fused_moe_default_scope.md`，bindings.scope 同时变化；因此它是独立的新 scope，不是原发布 run 的同语义维护。该文件限制默认实现选择，但保留完整数据域、非默认接口参数值、自动探测、PDL、eager/graph/cache、fallback 和错误出口。不能再次简化成 10 个样例或“参数默认值”。

### lineage，避免恢复前任

- GEMM：`b259d14c…`（2.2.1 P1 comparator blocked）→ `4aac1ea4…`（原发布）；`b61c62b1…` 是独立验证/校正任务，不是该链 successor。
- FMHA：`166fc15f…` → `acb68909…`（原发布）→ `45ea6608…`（当前）。
- MLA：`2dfb565c…` → `e3f7704f…`（已发布）。
- Sparse：`e6b585c3…` → `d5a73c01…`（原发布）→ `059ed88a…`（paused/P1 2.2.6）→ `2d252bf9…`（blocked/P5 2.2.7、旧 3/7）→ `b6d50336…`（cli_error/P1 2.2.9）→ `8f1be03a…`（当前 2/8）。
- GDN：`67778ed7…` → `97902388…`（原发布）→ `9699c87b…`（cancelled/P1 2.2.6）→ `45f9bb51…`（6 节点 export）。
- MoE：`d2be6c2b…` → `3563c3b7…`（原 scope 发布）；新默认 scope 独立链 `53d4bebe-882d-4a66-98b1-287d132dd6f3`（2.2.7 P4 cli_error）→ `2c4207c8-000e-43ea-8da0-2eb6a6079aee`（2.2.8 P4 invalid_receipt）→ `feb0c655…`（2.2.10 P2 cli_error）。

旧 continuous `continue` 在当时实现中会建立从 P1 重审的 successor；历史名称不能与当前同 UUID 的 semantic maintenance 等同。原 ready→旧 successor 的历史事实不构成今天再次这样调用的授权或支持。

## 具体内容判断

本节是独立抽查并核对源码的内容判断，不是独立 P6。覆盖六个推荐 root 的 Contract 主干、数学/调度/realization 关键段、六份索引，代表旧分支及当前 FMHA flat/segmented、Sparse root/mma_direct、MoE 当前 P2 root/dispatch。没有全文逐行审完 81 个推荐节点，也没有全文审完 FMHA 当前 29 个有效节点；未对每个协议事件和每个 when 做全域证明。

### GEMM：校正版值得保留

`R224/.../root.yaml:42–111` 分离 stored FP8、scale 布局、完整 out 写回/别名和 guards；:114–127 把外部比较器缺口保留为来源记录，没有用默认公式补全。:153–197 按 128 reduction scale block 作未缩放收缩，再按序 FP32 scale promotion/FMA，区分舍入位置并计入 padded row 工作，符合当前 Contract/Design 分层。

具体已修转录：root:287–294 与 `umma_wide.yaml:183–190` 明确 destroy 返回码被忽略；源码 `candidate/fp8gg_umma.cu:1844–1877` 中 graph hit launch 检查返回码，1847–1848 destroy 无返回码检查，1868–1872 create/add/instantiate/launch 才控制 fallback；wide 同类位置 1723–1751/1780–1808。应保留该修订，不回用原 root/umma_wide 的过宽“所有 graph API failure”概括。

host memo/cache、跨调用参数一致性与输出边界的来源限制仍有价值，不能因追求短文而删除。没有看到必须新建大量 dispatch 的直接证据；两种 UMMA 设计与 mma_sync 的并存有实际来源。当前检查三节点通过只证明结构门禁；本次没有重新证明全域数值和全部同步。首选 keep，若要作 2.5.0 当前批准则新目录独立 review/re-extract，不原地换 receipt。

### MLA：小范围 Contract 迁移，保留单 root 设计

`O/mla/mla_expanded_fp8/root.yaml:27–38` 的 attention_sinks/lse、:52 的 LSE 有 None/可选文字却没有 nullable。当前 `schema_docs/1.contract.algorithm_requirement.md:56,214–232` 规定 tensor-or-None 必须 nullable=true，default 不代替缺席性，None 分支与返回/写回要由 compute 定义。实际 `problem/input.py:101,113,123,129` 构造可选 sink/LSE 并传入 prepared ABI，不能删除可选域来消除问题。所需 patch 是这些 nullable 声明及对 LSE 缺席分支的统一核对；既有 :65–67、:100–107 大部分语义已经写明。此问题是新旧规则表达差异，原 2.2.2 回执仍按原身份保存。

值得保留：:144–242 的 regular half-exp/E4M3 numerator 路径、tiny 额外 FP32 重算（regular PV 仍执行）和 sink/log2 LSE；:295–325 描述 host side-stream 几何表、排序、队列及缺少生产流/前次表读取等待；:636 起 realization 将 tcgen05 完成/读退休机制与因果联系。它们属于设计/源限制，不应被改写成“源 kernel 正确”或删除重算成本。单 root 的 MODE=0 固定入口和纯内部数学数量分支有根据，本次没有证据要求为了 sink/LSE 或 tiny 另拆 host launch 树。

MLA 旧 frozen source 不含 annotated 辅助文件，独立检查只发现旧锁没有这项；当前 collector 支持它不会补造旧运行已经读过的证据。原 candidate 仍为主要事实来源。首选保留原版供读、有限 patch 的新交付；不因 2.5.0 新引用语法整项重写 Design。

### GDN：六节点已修复实际表达问题

`R227/.../root.yaml:38–70` 正确把初态 bf16、返回 FinalState fp32、CoreWritten/FinalState nullable 区分；:95–118 的逐 token recurrence 保留 FP32 carried state，不因算子名 state_bf16 错写最终状态。`problem/input.py:242–246` 明确提供 cu_seqlens/chunk_indices/chunk_offsets、`use_qk_l2norm_in_kernel=False`。因此六节点版只抽实际输入域，不是任意删掉 reference 可选接口。

root:8–30 的 segmented/direct_qk/ordered 三项 host 判定，与 dispatch 六项的 prep、单 chain、MODE1/2 两 pass、Perm 生产和读取区别一致。`dispatch.md` 明确分段与 direct-QK 不并存，VTT/环深保留参数化；弱衰减仍是 MODE2 内退出，不能取消已经提交的 MODE1/2。应保留这种按真实提交/物化拆分，而不是把六节点机械合并或改为全引用。

来源限制（core 非连续副本未回写、末 mini-chunk 读取退休、共享 workspace 跨流/重入、L2 cache device-key、外部 comparator）仍保留于索引/root；其中上游同步/副作用风险不是本轮要修 kernel 的理由。应在将来当前规则适用性审阅中核实必要限制是否应简化为一句并链接证据，不应把坏复用变成 realization 的实现硬要求。现有完整本地三区域合法，无需为了采用新引用能力重提取。

### FMHA：新修订确实有价值，当前前沿应保护

原 root:60–79/94 也有旧 optional/None 写法；当前 root:64/72/79/98 已使用 nullable，保留 MXFP8 解码与概率分组舍入，当前共享 Contract 有独立 P1 后重审证据。

原 `split_tcg.yaml:267–272` 明说 TwoPass 时 segment 才存在、否则 scan 直接依赖 gm；但是 flow/phase 已把 segment/segmax 写进同一结构。这违反当前 `schema_docs/0.common.md` 的参数/结构分流规则，也不是合法 no-op。源码 `candidate/fmha_mxfp8_qkv.cu:1642–1678` 明确 fuse_scan 完全不发 scan，scan2p 发 segmax+scan，否则发 flat_scan。当前 `tcg_d128_flat_tma_direct_narrow.yaml:208` 和 `tcg_d128_segmented_tma_direct_narrow.yaml:232` 分别列 4 和 5 个真实 launch，后者有独立 SegMax handoff；flat 与 segmented 的 realization 还明确 PDL launch permission 与读取完成不同。这个方向应保留。

当前 43 节点计划含 D128/D256、TMA/cp、flat/segmented/fused scan、direct/xpose、narrow/wide 等实际差异，不能只因节点多就合并，也不能根据节点名保证全部必要；适用性 review 要逐 when、producer/consumer 和选择域验证。29 个有效节点文件 SHA、generation、merged 状态已一致，不应让 full-restart 丢掉其作为可编辑起点的资格。2.5.0 maintenance 首选先 review 当前全文，发现内容错才给对应字段 patch；未完作者稿保留，不自动用父内容覆盖。当前从停滞服务错误不能推出这些内容需从头写。

### Sparse：拓扑重构与 kernel 缺陷必须分开

原 root:267–278 用 S=1 取消 combine 却常驻 combine/handoff；`probe_dual.yaml:224–239` 用正文选择 packed/mma、取消 probe/prefix 及完成边。旧表达确需修订，索引勘误没有修复它。原 Contract:94–124 的两池共享 K=V、重复索引保留 multiplicity、联合 softmax、empty output 和 sink 语义则值得保留。

当前 root:284 和 mma_direct:121 已分别表达 split+combine 与单 attention；当前 root sinks:48–52 有 nullable。新八节点区分 tc_heads/tc_packed、serial/hidden probe 和实际选择的后段，已审 root/mma_direct 不应退回旧三节点。

固定源码 `candidate/dsv4_kernel.cu:2111–2115` 的 bar_land、tmem_base_s、ukey 共址；2497–2515 初始化/分配、2753–2761 继续读 ukey、3068–3070 等 landing，确实构成源布局风险。`schema_extract/issues/archive/kernel_wiki/sparse-tc-control-layout-overlap.json` 固定 SHA `c9389387d35bd26954754a5ae13e2d202df06fc9799a77fddfb0654306909eb2`，并记录 1879–1881 已预算第五组 barrier 和返回槽，支持同一抽象设计而无需重定资源策略。当前 REVIEW:47–55 允许这种边界之外落实偏差分流；不要要求先查明被破坏的具体 TMEM/key 值。反之，TMEM、完整 UnionKey 容量/寿命、生产消费/复用依赖仍须完整写，不得用 issue 为缺稿开脱。首选保持当前 2/8 前沿并 review 后完成。

### MoE：默认 scope 决定保留什么，旧 60 节点不能当新计划

原 root:94–141 的 NVFP4 nibble/scale 解码、两种激活、block16 QDQ 和每条 route 的 FC2 先 BF16 再加权是重要数值边界；当前 P1 root 对这些有独立批准，且补齐 clamp nullable，值得保留。

旧 `ws_fused_ws.yaml:196–197` 用正文取消 small/large 准备链和 gather、重接实际 predecessor；:357 说 member 列的是最大范围，实际只实例化 warp[3:3+EPI]，但 :503/:592 仍写 `warp[3:11]`。实际 `candidate/nvfp4_moe.cu:5384,5407` 是 `__launch_bounds__((3+EPI)*32,...)`，:4986 根据 EPI=8 选择 parity。这是 schema 表达错误/旧 checker 绕行，不能视为 kernel 缺陷；当前规范支持符号成员端点，应使用真实范围并审核跨 role release 覆盖全部实际 epilogue 成员。

旧 when:16–23 纳入大量 cfg/cfg2 和显式控制，而新 source scope 排除非默认控制才可达的实现。新 P2 dispatch 已区分小/大准备、PDL 串行门、split-K、FC2 实现、零验证、ABI 建立和异常实际前缀，注明拟议族不能作笛卡尔积、不能用正文删除已声明提交。这个草案方向合理，但未获 P2 reviewer 成功签收；当前 root 1025 行通过检查也不能替代 P3。特别需要 review graph 失败后 eager 接续是否确实产生额外工作、初始化是否已提交、验证已录制和回放时内容的区别。首选保留当前 P1/候选、在同 run 维护后完成；若全新当前运行成本更小，只有该项可合理优先考虑 full-restart。

## 最短合法恢复路径：实际工具支持与条件

### 先对账，不先换语义或释放全部 hold

当前正式 inspect 的三个 runner 均 lock=free、alive=false。FMHA 两个未确认 attempt：

- `85cfe993-9236-4aec-bb0d-c76a276262af` → `P5:tcg_d128_segmented_tma_xpose_wide`
- `01c1e189-86a7-4e22-8006-da604bba958a` → `P5:tcg_d256_flat_cp_wide`

Sparse：`adbba921-2156-4bfd-9fc1-6b8359f3a9f1` → `P5:tc_heads`。

它们有 process/result/finished，result.exit_code=1、response=null、outcome=termination_unconfirmed；reason 是 invocation group 尚有无法确认归属的 survivors。当前 ps 查询这些 recorded survivor PID 都不存在；这只是有利的新观察，不能替代正式 termination.json 和 Runner 结算。MoE 最后 `9eb6167a-1108-4b9e-ba7c-714786b0ae5b` 则 termination.confirmed=true、cli_error、response=null，故仅是未取得新审阅结果。

**可以构造、不在本次执行的无新模型清算命令：**

```sh
python3 schema_extract/extract.py cancel 45ea6608-66b2-4915-b47d-59d2c115602e
python3 schema_extract/extract.py cancel 8f1be03a-af4a-44f2-a2ba-b8deda6ae3e7
python3 schema_extract/extract.py inspect --run 45ea6608-66b2-4915-b47d-59d2c115602e 8f1be03a-af4a-44f2-a2ba-b8deda6ae3e7 --json
```

代码依据：当前 `tools/cli.py:543–559` 登记 cancel 并转冻结入口；2.2.7 冻结 `cli.py:430–446` 调 `run(...,resume=False)`。旧 runner:890/948 消费控制，:960 调 `_reconcile`，:971 在 control.action 非空时 `next_calls=[]`，仍处理 pending 的既有 result，:1069 置 cancelled；旧 recovery:18–45 再读 process 身份、用 terminate_owned 确认并保存 termination，:159–173 规定 cancel 不被后续 pause 弱化。因此当前两个 termination_unconfirmed 状态可经此命令由旧唯一写者清算，不发新 worker/reviewer。全局 hold 不挡非 resume 控制路径（runner:933 注册/保留 hold 后继续对账）。`authority_changed` 特例 :953–958 只 terminate_only，不提供全结账，但这两 run 当前不是该原因。

这三版本的 cli.py、runner.py、recovery.py、store.py 实测 SHA 分别相同，故上述旧代码路径同时适用于当前 Sparse 2.2.9 和 MoE 2.2.10。命令退出 0 不证明清算完成，随后必须验证 active_attempts/unconfirmed_attempts/unsettled_attempts **全部空**、旧 writer/launch 死亡、checkpoint/节点稿未丢。若仍有 unknown ownership，保持 fence，不能手改 state 或借新 run 绕开旧调用。

当前有 active global hold `15837dcb-5f2f-4e3a-a4d8-d99631834cd6`（不是历史已经 released 的 `77e95fe9…`）。它限制派发/发布。应保持到恢复策略及目标身份已确定；后续获执行授权后才用 `resume --all --hold 15837dcb-5f2f-4e3a-a4d8-d99631834cd6` 释放该 hold，并复核 later manual cancel 仍有效，不能把释放 hold 当作解除所有控制或充值。已取消任务再由显式 resume 启动。

### 采用 2.5.0：未完成任务需要桥接，已发布任务新目录

当前三条冻结执行器不支持 semantic_revision.v1，直接 `continue` 会在 `revision_control.py:80–82` 拒绝。直接 `upgrade-runtime` 到 2.5.0 又被 `runtime.py:310–315` 的 design_region_derivation.v1 语义边界拒绝。合法顺序：安全清算 → 构造并独立审阅旧 Design 语义的中间执行器 → compatibility-report/upgrade-runtime → 正式 semantic revision/maintenance → 当前 P6。

真实存在的候选开发基线为 Git `60285f0`（README 2.4.1，runtime.EXECUTOR_CAPABILITIES 含 semantic_revision.v1、scoped_recovery.v2，不含 design_region_derivation.v1）。**我没有证明整份 2.4.1 checkout 对这三个旧 run 都是 compatible，也没有建立或虚构已经可用的桥接版本。** compatibility 协议检查全部差异，不是只看能力位；跨版本的 Contract/政策/检查/提示语义差异不能谎标 runtime_fix。尚缺一个明确绝对路径的完整中间工具树、逐文件兼容审阅及必要离线验证，可能要从该真实基线定向回植运行修复而保留旧规则材料。三条源语义可能需要不同桥接审阅，不能跨 run 复用 report 身份。

下面命令模板的 BRIDGE_PATH/REPORT/REVIEW/PLAN 是**尚待形成的有界材料**，不是已存在文件。每次 checkpoint/状态变化后重新构造 report，不能使用本次预清算摘要冒充升级时 basis：

```sh
python3 schema_extract/extract.py compatibility-report RUN_UUID --checkpoint CURRENT_DIGEST --target-repo BRIDGE_PATH
python3 schema_extract/extract.py upgrade-runtime RUN_UUID --report REPORT.json --review REVIEW.json --target-repo BRIDGE_PATH
python3 schema_extract/extract.py revise request RUN_UUID --reason '保留已有原始稿，按当前规则独立审阅适用性并完成未完范围' --target-repo /home/shaohanyong.shy/KernelSchema
python3 schema_extract/extract.py revise prepare RUN_UUID
python3 schema_extract/extract.py revise prepare RUN_UUID --plan PLAN.json
python3 schema_extract/extract.py revise apply RUN_UUID
python3 schema_extract/extract.py revise status RUN_UUID
```

request/prepare 不等于维护已完成；cancel/paused run 采纳后不会凭空执行，之后还要在控制/预算符合时 resume。计划先保留各节点 own raw，对现有 P1/P2/P3/P4/已完成 P5 独立 `review`；有据问题才 `patch/reextract` 并写 fields。尚未完成的普通稿继续作者当前逻辑步骤，不能把 maintenance 当成自动补完它的办法。**本报告没有完成全部 52 个变化和每个范围的适用性审阅，所以不足以直接形成 adopt assessment。** 本次 FMHA→2.5.0 正式只读 compatibility-report 有 52 个变化，默认 approvals/executor compatible=false、execution_authorized=false、suggested_start=P1；后者是未知差异的保守建议，不是强制全项重写命令。

P4 换父要记录 identity_migration，保留旧 raw/版本，worker 显式改 parent/from_parent 或落地旧有效整区，再独立批准；fields 缺 parent/ref/设计区时修订授权范围，不自动扩大。root/shared Contract/P3/P4/provider 变化按依赖使消费者失效，P6 资格亦失效但账本不重置。全本地 YAML 仍合法，不需要把 29 个 FMHA 节点全部转 from_parent；若未来选引用，只可三个区域各自引用直接父同名整区，不能局部 patch、跳祖先或继承 realization。review 要覆盖 child/direct_parent source manifest 与引用链 raw 锚点、跨区绑定、独立 when 和本地 realization；旧父批准不能代替子适用性。

GEMM/GDN/MLA 原 run 已 ready/published/export。当前 `revision_control.py:48–53` **和** `continuation.py:145–147` 都拒绝它们，所以 `continue --full-restart` 也不是已发布修订通道。若确需正式当前规则交付，使用新输出目录的 `extract`，明确冻结目标来源与 role/scope，原版保持字节不变。可从 catalog 使用 `--id`，但会重新冻结现有实际 source 集合；若必须保持旧锁来源集合，要先从旧 inputs.lock 的 source/role bindings 构造合法 source.v2 描述，报告尚未生成它，不应假称 `--id` 自动保持旧 fingerprint。只有 API 目标为读视图时才 standalone；生成视图不修复 nullable、拓扑或产生新 P6。

### full-restart 的真实保留程度

当前 `continuation.py:39–98` 将**选定 checkpoint**全树封存为 `continued/artifacts`，initial 可编辑目录只放 root.yaml；非 root node 标为 removed，draft_checkpoint/approved_checkpoints/approval/approved_sha256/last_draft_checkpoint 清空，plan/plan_draft/merged_checkpoint 清空，阶段回 P1。仅保留旧节点账目、预算及只读历史线索；未合入节点的私有最新草稿并不会自动复制到新的可编辑工作区。它不是保存 FMHA 29 节点作业前沿的等价替代。

MoE 因没有已审 P5，full-restart 比其余两项损失小，可用下面已确认 CLI（仅拟执行）：

```sh
python3 schema_extract/extract.py continue feb0c655-4e13-43ac-817d-1f457f31c029 --full-restart --checkpoint ed3b30e50697640fc0b86c8205bddf0e10bdc4308134b7bb1596e9c29c622354 --reason '默认可达 scope 不变，在 2.5.0 从保留 root 重新独立审阅；不继承批准' --add-seconds SECONDS --add-calls CALLS
python3 schema_extract/extract.py resume SUCCESSOR_UUID
```

`SECONDS/CALLS` 是预算决策占位，不应照抄字面值；successor UUID 必须使用命令实际返回的值。full-restart 禁止 `--target-repo/--plan/--request-id`；目标来自所用当前执行器的 rules tree，来源/config 固定继承；不重置原来 P4 已用 20 次和 2 次退修。新 source/scope 应另 `extract` 到新目录，不能 full-restart 偷换。

### 墙钟/调用余额

本次观察已晚于三条截止：FMHA `2026-11-04 21:06:16 UTC`，Sparse `19:10:06 UTC`，MoE `23:26:35 UTC`，已开始的相应 stage/node 截止也均过期。停止原因为 cli_error/termination_unconfirmed 不代表恢复时预算还够。恢复必须按实际 scope 的剩余额度追加墙钟；普通 resume `--add-seconds N` 将已开始范围延期至 `max(now,old_deadline)+N`。未开始范围仍使用原 stage_seconds，FMHA/Sparse 3600、MoE 7200；需据复杂程度安排后续追加。

三者 max_calls=32；FMHA 未完节点当前多为 2–6 次（其中 tcg_d128_fscan_tma_xpose_narrow 6 次），Sparse probe_serial_mma 5、tc_heads 3；MoE P2 8、历史 P4 20。当前看不出所有节点都需要 add-calls，不能仅因总 attempts 多就统一充值；semantic review 与全文 P6 也消耗原账目，应按维护范围估算。普通 continue 无 --add-*；升级不充值、不启动；`resume --scope` 不接受充值，且三条目前还缺 scoped_recovery.v2。前任使用的 P6 calls/p6_starts 也保留，不得把 P6 pending 中的历史 calls 误当当前终审已进行或可免费重置。

## 优先级与风险

1. 先形成 FMHA/Sparse 清算证据，保持 hold；保护当前私有稿、checkpoint、scope 账目。风险是误把残存 process fence 清掉、错恢复前任或释放整批控制后无意启动。
2. 若目标明确是 2.5.0，优先验证可复用的旧语义桥接方案与当前 maintenance；FMHA 29 已审节点的重复写作成本最高，最不宜 full-restart。Sparse 次之。
3. MoE 根部尚浅且默认范围独立；可以跟随已验收桥接作局部接续，也可在对比成本后显式 full-restart。主要风险是再次过度扩张 cfg 组合、遗漏零验证/graph/错误已提交前缀、把旧 60 节点当默认计划。
4. MLA 有具体小表达差异，应作为有限 current-rule revision 候选，不应被“旧版基本通过”遮盖；正式新发布成本由现有工具限制决定，而不是 Design 内容本身需要全重写。
5. GEMM/GDN 保留推荐正式版本；没有必要仅为版本一致重做。当前 reader 可以继续使用旧身份与已知限制。

## 实际做过的只读验证及边界

- 读取所要求的 EXTRACT_SCHEMA、三份提取说明/REVIEW、common 与 P1–P6 阶段页关键部分，并定向读 Contract/Design/引用/nullable/maintenance/compatibility 规则；阅读导航、SUPERVISOR、COMPLETION、当前 repairs/followups、TRIAL_REPORT/OPERATOR_REVIEW/TRIAGE 和相关 archive。没有把这些历史记述代替 runtime。
- 浅层枚举 23 个 state，提取 lineage、source lock、executor、stage/node approval、计划及停止原因；读取三条当前正式 inspect，查看实际 runner/launch/hold/unconfirmed；核对记录的 surviving PIDs 当前均不在进程表。
- 六原发布加两修订共 8 个目录的 `.extraction/receipt.json` 文件表与全部实际文件 SHA 一致；8 次最终 P6 的 canonical JSON receipt digest、validated.receipt_sha256、artifact_digests 与交付表一致，check.json returncode/exit_code=0 且等于 validated.check。注意 receipt_sha256 使用工具 canonical JSON digest，不是缩进文本原始字节 hash。
- 用当前 `python3 -B schema_template/check_yaml.py --skip-hardware <root>` 检查上述 8 目录，均返回 0（3/60/8/1/3/2 原节点及 3/6 修订）。这是当前结构/关联检查，不证明旧稿的正文条件拓扑或 nullable 表述都语义合规。
- 同样检查三条当前 checkpoint 目录，FMHA 29、Sparse 2、MoE 1 都返回 0；核对 FMHA/Sparse approved nodes 的 merged=true、approved_generation==generation、raw SHA 对得上。它们仍不是完整计划交付：检查器不从 runtime 得知还有 14/6 个计划节点未完成。
- 逐文件比较关键同源前后 lock、role bindings；核对 candidate 工作树字节与相关 frozen manifest 一致。定向读上述源码点，未运行 GPU 或比较器。
- 执行一次只读 FMHA→当前规则 `compatibility-report`，保存 `/tmp/kernelschema-recovery-audit-20261105-a-fmha-target-diff.json`；52 个差异，无独立 report review，不登记 runtime。
- 读取并比较旧 cli/runner/recovery/store，以及当前 full-restart/runtime/revision 代码，得出上述操作路径。没有实际验证 cancel/upgrade/revision 的运行结果，不能保证尚未生成的桥接树 compatible。
- 未执行全 lineage 的新 audit --verify；没有声明全部历史 usage 证据完整，也没有汇总 token/费用。GDN 历史 cancelled run 缺 result 与后来已发布的最终 P6 成功是两件事。
- 工作树 `git status --short` 起止均为空；AGENTS.md 未改。

## 附录：本次实际 state 定位

以下 UUID 均由当前 state 枚举；snapshot 路径表用于定位冻结规则/source，不能替代 state 中现行执行绑定。

| run UUID | 算子 | status/stage | 冻结版本 | snapshot fingerprint | predecessor |
|---|---|---|---|---|---|
| `059ed88a-3be3-4642-b578-8ed5959eed37` | dual_cache_latent_bf16 | paused/P1 | 2.2.6 | `01ca6dd17b86740615c148557de8e8bdc202627b46557768fe2cba74a5bf346a` | d5a73c01-7c3a-45b2-a3ce-0d5eec634808 |
| `166fc15f-e49e-42ef-b429-c7f507f5ed53` | fmha_mxfp8_qkv | blocked/P1 | 2.2.1 | `7212157be7b4732268af55eb0c78752f374882bfddcf1256979245636f0851a6` | — |
| `2c4207c8-000e-43ea-8da0-2eb6a6079aee` | nvfp4_fused_moe | invalid_receipt/P4 | 2.2.8 | `c496cab3ab2d8a0423da57009a8c094fd75922814a317dc46024d8c9bc024a84` | 53d4bebe-882d-4a66-98b1-287d132dd6f3 |
| `2d252bf9-d9a3-4fa4-96d2-90a98e25eb34` | dual_cache_latent_bf16 | blocked/P5 | 2.2.7 | `9e832c203bee2dd503043a8ccdbe8e8c8233a19d0e7d40f3cd53e6d6a1123717` | 059ed88a-3be3-4642-b578-8ed5959eed37 |
| `2dfb565c-28a6-48de-9cdc-ab31f1758e4a` | mla_expanded_fp8 | blocked/P1 | 2.2.1 | `02ae95832373f9c65161d24cacff87ebc012e187e25ab9fd16c2d7de8326bfeb` | — |
| `3563c3b7-2b7c-4967-9d41-9c2042e53d2c` | nvfp4_fused_moe | ready/P6 | 2.2.2 | `57d15d0d2382bee43a2a7a4e852dc801b6fca69ad2725388e0df89faa0382ad1` | d2be6c2b-8a06-4f4b-9f23-61ed6cda8db1 |
| `45ea6608-66b2-4915-b47d-59d2c115602e` | fmha_mxfp8_qkv | termination_unconfirmed/P5 | 2.2.7 | `138d555b31ab141c588c735d7b260b5b139f3188daa26393fd447ef066ffa3b8` | acb68909-aaaa-4bc2-8b16-94904752af8e |
| `45f9bb51-77f3-4285-b7a7-251cf2744fa1` | gdn_prefill_bf16_state_bf16 | ready/P6 | 2.2.7 | `0b48041c858b11c265d3ea0c03ec39714fe16909c958ea31e4eb25ab55266b31` | 9699c87b-4ae0-412a-b966-defc914407ef |
| `4aac1ea4-94fa-4b4c-93ca-d48a8eb08996` | fp8_scaled_grouped_gemm | ready/P6 | 2.2.2 | `6077811896c6fdaebc5b046da6665a47ced0278c676771a25a4df910076d2be4` | b259d14c-3496-4802-88df-6f4632d90fd0 |
| `53d4bebe-882d-4a66-98b1-287d132dd6f3` | nvfp4_fused_moe | cli_error/P4 | 2.2.7 | `c16d1221691cd538dbf4bee46d963523018e83eb2e8ca3226cd7b2274f3b4f46` | — |
| `67778ed7-dde4-41a4-aaa5-34a6d58228f4` | gdn_prefill_bf16_state_bf16 | blocked/P1 | 2.2.1 | `5b13603cc9e05c8c27f1b440eb824c076b1c2c7499780ed3f602989189ca8302` | — |
| `8f1be03a-af4a-44f2-a2ba-b8deda6ae3e7` | dual_cache_latent_bf16 | termination_unconfirmed/P5 | 2.2.9 | `ece761e6f1d0183735c539148e66c1965a62f883d7152ad3d20b70d66c1d0160` | b6d50336-fe0e-4689-a52a-a50ffefcbd6a |
| `9699c87b-4ae0-412a-b966-defc914407ef` | gdn_prefill_bf16_state_bf16 | cancelled/P1 | 2.2.6 | `671e9c1f4d8791af10cde06009122c4e53301cd96dc2eedb5dc758d897d6b9d3` | 97902388-eda0-4cbd-b8a6-ba71440e86ab |
| `97902388-eda0-4cbd-b8a6-ba71440e86ab` | gdn_prefill_bf16_state_bf16 | ready/P6 | 2.2.2 | `14029f9bd6b055d1edeccba2e9c9e4aa7fcdf92b089aa63060b7e7dd09ec772b` | 67778ed7-dde4-41a4-aaa5-34a6d58228f4 |
| `acb68909-aaaa-4bc2-8b16-94904752af8e` | fmha_mxfp8_qkv | ready/P6 | 2.2.2 | `9c8a9c68e11015f00da32484b44d5455baa83482558840c61c4fab4d7b1fb3a9` | 166fc15f-e49e-42ef-b429-c7f507f5ed53 |
| `b259d14c-3496-4802-88df-6f4632d90fd0` | fp8_scaled_grouped_gemm | blocked/P1 | 2.2.1 | `314b77af686cd587221a47614bd3bf2a0cd9ecde7fe5f6a699cf80d9475d6c00` | — |
| `b61c62b1-b0dd-4ed8-a21f-e9b7227a0d3b` | fp8_scaled_grouped_gemm | ready/P6 | 2.2.4 | `a42ac9bf81434da90b1a322bc2bfba2777bb87f8fe9a4aefffc844116b3d0bdc` | — |
| `b6d50336-fe0e-4689-a52a-a50ffefcbd6a` | dual_cache_latent_bf16 | cli_error/P1 | 2.2.9 | `b47f229dbcad3c8aeeaf6d52363f2ca6334ea4244e53cf727a78e8fd1d8b7489` | 2d252bf9-d9a3-4fa4-96d2-90a98e25eb34 |
| `d2be6c2b-8a06-4f4b-9f23-61ed6cda8db1` | nvfp4_fused_moe | blocked/P1 | 2.2.1 | `0faa932d47de53695cb5ea95d95fdac1fa2631fe6ae0a00ecc109af28c5c7b32` | — |
| `d5a73c01-7c3a-45b2-a3ce-0d5eec634808` | dual_cache_latent_bf16 | ready/P6 | 2.2.2 | `11a983b2954512798f55391301436a6044249208032128ebd6211551bba264b3` | e6b585c3-f060-45cc-8f41-aba22512717a |
| `e3f7704f-061f-48ff-8131-cc317afe6003` | mla_expanded_fp8 | ready/P6 | 2.2.2 | `b97eec335391249ef6078863930904b62a52dcd4dbf8a0ad72a0021c35a91fd6` | 2dfb565c-28a6-48de-9cdc-ab31f1758e4a |
| `e6b585c3-f060-45cc-8f41-aba22512717a` | dual_cache_latent_bf16 | blocked/P1 | 2.2.1 | `97e5c151527d578b7c65ae26a7f3c771e13b1604376198436ae164b0bd492b26` | — |
| `feb0c655-4e13-43ac-817d-1f457f31c029` | nvfp4_fused_moe | cli_error/P2 | 2.2.10 | `8b281029059476dcfe78d7588953be485983ff2d084fdc1216ff44d2e3b565f0` | 2c4207c8-000e-43ea-8da0-2eb6a6079aee |

全部 snapshot 的绝对根为 `/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/snapshots/<fingerprint>`；原三身份锁在该目录与 `RUN(UUID)/inputs.lock.json`。

### fmha_mxfp8_qkv 当前完整范围表

Run：`45ea6608-66b2-4915-b47d-59d2c115602e`。当前 checkpoint 原始目录：`/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/45ea6608-66b2-4915-b47d-59d2c115602e/checkpoints/812602cc9003742176580b62ea078814a926c3a8d230aa42da277aa2ad48e85f/artifacts`。

| scope/node | status | parent | 当前批准 attempt 或未完 draft checkpoint |
|---|---|---|---|
| P1 | approved | — | 82d20c2b-58ba-4b84-92e4-900aa3f46a1b |
| P2 | approved | — | a409ae84-ec2c-4d83-8523-4fc534fc2bd1 |
| P3 | approved | — | 8724fd8b-a13e-406e-83f7-e6ea128e1295 |
| P4 | approved | — | 49ef36d8-c0a1-4203-be6d-ea5c511d22c7 |
| P5 | running | — | — |
| P6 | pending | — | — |
| fused_d128_cp | approved | root | `03e94ca8-52b7-426a-a4f1-9f5607b2d93d` |
| fused_d128_group | approved | fused_d128_key | `3adba96f-fd6e-40e5-b927-4bb4995a3cdd` |
| fused_d128_key | approved | root | `fe6a8d7c-cbdf-4231-a804-445f0a02cbbd` |
| fused_d256_cp | approved | fused_d256_tma | `775d70f8-9e90-4f3c-a356-1a82e8610e4d` |
| fused_d256_tma | approved | root | `6a1b4897-3eb8-4976-8669-7ddc3ea7971f` |
| root | approved | — | `8724fd8b-a13e-406e-83f7-e6ea128e1295` |
| tc_fused | approved | root | `334d262d-a570-4cb7-9bb5-7814a98d1f9b` |
| tc_split_flat_narrow | approved | tc_fused | `297c99ef-d709-4a79-a1b6-921128c3e4a3` |
| tc_split_flat_wide | approved | tc_split_flat_narrow | `7ce12f84-1ece-4986-8eb0-8cc7c493857e` |
| tc_split_segmented_narrow | approved | tc_split_flat_narrow | `2cc2b11a-9e69-47f3-be2b-7d85832494fb` |
| tc_split_segmented_wide | approved | tc_split_segmented_narrow | `8a0a933e-6ea6-4ecd-b926-8ce8e96c9f74` |
| tcg_d128_flat_cp_direct_narrow | approved | tcg_d128_flat_tma_direct_narrow | `5f619e3c-bfd6-4558-894f-14359ded5986` |
| tcg_d128_flat_cp_direct_wide | approved | tcg_d128_flat_cp_direct_narrow | `56207c60-8e81-42ef-af5c-a9158ffbaa93` |
| tcg_d128_flat_cp_xpose_narrow | approved | tcg_d128_flat_tma_xpose_narrow | `e307c4a7-9d0b-4fd0-a56e-b7bc9eece36b` |
| tcg_d128_flat_cp_xpose_wide | cli_error | tcg_d128_flat_cp_xpose_narrow | `0ff4cd1cef861832420840e688148de07a48d79ebdb1b87a49d284b4c49bba64` |
| tcg_d128_flat_tma_direct_narrow | approved | root | `3563fd53-a2bb-4c5d-9645-b4ed72241923` |
| tcg_d128_flat_tma_direct_wide | approved | tcg_d128_flat_tma_direct_narrow | `c61f0a54-6df5-4225-954a-faca8b5b2ea9` |
| tcg_d128_flat_tma_xpose_narrow | approved | tcg_d128_flat_tma_direct_narrow | `3060a43e-771e-4563-b450-872543f9af92` |
| tcg_d128_flat_tma_xpose_wide | approved | tcg_d128_flat_tma_xpose_narrow | `14a5d1c9-4e5f-4266-a06d-bb74011da95a` |
| tcg_d128_fscan_cp_direct_narrow | approved | tcg_d128_fscan_tma_direct_narrow | `4a0117ba-fe4a-4a48-afd3-d9bf6a03d746` |
| tcg_d128_fscan_cp_direct_wide | cli_error | tcg_d128_fscan_cp_direct_narrow | `67e5d5d916c128a4a7903ea9a42b173707d4a6d01852425ad460cd76e43ccd37` |
| tcg_d128_fscan_cp_xpose_narrow | pending | tcg_d128_fscan_tma_xpose_narrow | `—` |
| tcg_d128_fscan_cp_xpose_wide | pending | tcg_d128_fscan_cp_xpose_narrow | `—` |
| tcg_d128_fscan_tma_direct_narrow | approved | tcg_d128_flat_tma_direct_narrow | `6bb561a9-b305-4dd0-8f33-321d3da83ab1` |
| tcg_d128_fscan_tma_direct_wide | approved | tcg_d128_fscan_tma_direct_narrow | `fe0e0cac-5226-4aed-acca-69e9ee9d4666` |
| tcg_d128_fscan_tma_xpose_narrow | cli_error | tcg_d128_flat_tma_xpose_narrow | `b75e19699a447cc9a537abb1133f74d3d57395e666193b6ef6dd4e28ebbf71f2` |
| tcg_d128_fscan_tma_xpose_wide | pending | tcg_d128_fscan_tma_xpose_narrow | `—` |
| tcg_d128_segmented_cp_direct_narrow | approved | tcg_d128_segmented_tma_direct_narrow | `d6fe6c37-5056-45c3-9ab3-17bbab211807` |
| tcg_d128_segmented_cp_direct_wide | cli_error | tcg_d128_segmented_cp_direct_narrow | `986c15b4b9ca9a2cbed34a881f13faaff0bbca371856e055d4b765176c717f39` |
| tcg_d128_segmented_cp_xpose_narrow | cli_error | tcg_d128_segmented_tma_xpose_narrow | `d31e5d9f06edd8cccebd299a38bf766b9596bd21150dd167ffb19e67df7d9942` |
| tcg_d128_segmented_cp_xpose_wide | pending | tcg_d128_segmented_cp_xpose_narrow | `—` |
| tcg_d128_segmented_tma_direct_narrow | approved | tcg_d128_flat_tma_direct_narrow | `6fcf0b52-f62a-452d-bfbf-287cd3151042` |
| tcg_d128_segmented_tma_direct_wide | approved | tcg_d128_segmented_tma_direct_narrow | `930a73d3-2505-4aff-8932-0f790e0859b4` |
| tcg_d128_segmented_tma_xpose_narrow | approved | tcg_d128_flat_tma_xpose_narrow | `2e5d787b-dc75-4249-8e2d-ce9f3da5d4d8` |
| tcg_d128_segmented_tma_xpose_wide | termination_unconfirmed | tcg_d128_segmented_tma_xpose_narrow | `bc86bed279e5bfc47f432e7f60d6258623e888acb5ac51d25bcbe815cad65ee8` |
| tcg_d256_flat_cp_narrow | approved | tcg_d256_flat_tma_narrow | `da21e951-dab1-4cd9-a913-e5008c392aa7` |
| tcg_d256_flat_cp_wide | termination_unconfirmed | tcg_d256_flat_cp_narrow | `6cc77c8cb7b488081b7874392f0117bc4026fe29a030bf991dfd7b5f26e91962` |
| tcg_d256_flat_tma_narrow | approved | fused_d256_tma | `99013039-803a-4618-8946-f5ee5b0fd655` |
| tcg_d256_flat_tma_wide | approved | tcg_d256_flat_tma_narrow | `a42a998d-d011-4cdb-88ef-6f40448489e1` |
| tcg_d256_segmented_cp_narrow | cli_error | tcg_d256_segmented_tma_narrow | `44724b4004807f2ac8cb6d11fb975106c5e8067723ff9d89738d7356385e40e3` |
| tcg_d256_segmented_cp_wide | pending | tcg_d256_segmented_cp_narrow | `—` |
| tcg_d256_segmented_tma_narrow | approved | tcg_d256_flat_tma_narrow | `a9b48270-cf0c-4fc1-8f4a-c1fef90a6b83` |
| tcg_d256_segmented_tma_wide | cli_error | tcg_d256_segmented_tma_narrow | `107cc823c3c4ef36bd924847ac9a38c9fd340a4aece74cd06624ab4c001e719f` |

### dual_cache_latent_bf16 当前完整范围表

Run：`8f1be03a-af4a-44f2-a2ba-b8deda6ae3e7`。当前 checkpoint 原始目录：`/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/8f1be03a-af4a-44f2-a2ba-b8deda6ae3e7/checkpoints/0876fd66dce267bb9c46cd383bddc47ddfb87d53521d0c2a26425d534d11a356/artifacts`。

| scope/node | status | parent | 当前批准 attempt 或未完 draft checkpoint |
|---|---|---|---|
| P1 | approved | — | 40f7c04f-39c8-4d0a-a1fa-2b9584f62c73 |
| P2 | approved | — | dddc9777-0a50-459d-a845-597f10cf52ea |
| P3 | approved | — | e37b992e-879e-4652-8656-06a0a9a95307 |
| P4 | approved | — | 5d732e2d-c09a-4c35-b606-b6965b420c16 |
| P5 | running | — | — |
| P6 | pending | — | — |
| mma_direct | approved | root | `8f9bc9a4-c7a2-4fbe-9373-9f2c213a6701` |
| probe_hidden_mma | pending | probe_hidden_packed | `—` |
| probe_hidden_packed | pending | probe_serial_packed | `—` |
| probe_serial_mma | cli_error | mma_direct | `7f704e92dfc35bc9c7c34745d69acc57d6901f48ee5570570226023951f1d128` |
| probe_serial_packed | pending | tc_packed | `—` |
| root | approved | — | `e37b992e-879e-4652-8656-06a0a9a95307` |
| tc_heads | termination_unconfirmed | mma_direct | `64c36fc83b16cb5262a59683004bf58e2374f0f781fd5bdf7cc281f285723477` |
| tc_packed | pending | tc_heads | `—` |

### nvfp4_fused_moe 当前完整范围表

Run：`feb0c655-4e13-43ac-817d-1f457f31c029`。当前 checkpoint 原始目录：`/home/shaohanyong.shy/KernelSchema/.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/ed3b30e50697640fc0b86c8205bddf0e10bdc4308134b7bb1596e9c29c622354/artifacts`。

| scope/node | status | parent | 当前批准 attempt 或未完 draft checkpoint |
|---|---|---|---|
| P1 | approved | — | f5b90086-7e74-4a5e-84c2-c6182a6eac99 |
| P2 | cli_error | — | — |
| P3 | pending | — | — |
| P4 | pending | — | — |
| P5 | pending | — | — |
| P6 | pending | — | — |
| root | draft | — | `ed3b30e50697640fc0b86c8205bddf0e10bdc4308134b7bb1596e9c29c622354` |

额外 realization 抽查锚点：GEMM 校正版 root:556–571 把 TMA swizzle、描述符、staging 坐标及容量解释为对应关系而非成功保证；GDN 六节点 root:676–690 区分 dense 输入物化、128B swizzle 相位、PDL 许可/数据可读/bulk drain；Sparse 当前 root:547–576 分开 cp.async 组完成与 graph/scratch 绑定。这些因果交接内容值得保留，仍需在任何新引用组合下由使用节点重新核对，不自动继承父 realization。

补充核验：当前 FMHA 29 个、Sparse 2 个 approved 节点的实际 attempt/receipt.json canonical digest 均与 node.approval.receipt_sha256 和 validated.receipt_sha256 一致。旧 cancel 清算可使这些失败调用转为已确认的 invalid_receipt 再停 cancelled，不能把它写成内容批准；其后仍须按实际节点原因进入普通恢复或维护。
