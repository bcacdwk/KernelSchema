# MoE 候选设计目录

**试写，尚未迁移任何完整 schema。** 下列“完整主线”是拟保留的目录条目；链接的 YAML 是旧语法代表证据，不是符合新规则的交付。局部记录不能自动施加或展开成组合。

本目录重点试写[后续默认范围][q-index]：154 项计划中 96 项曾批准，范围含调用准备与验证。它与[初始 60 份显式配置稿][i-index]不同；后者的扩展边界单列于文末，不计算“154 压成几份”的成绩。数量与状态依据见[范围核对](../../01_moe_evidence.md#1-版本范围和数量的可复核口径)。

## 拟保留的完整主线

全体共用 root 的数学与硬件 Contract。root 保存实际完整设计且三区本地；来源关系写在下表，不设 parent。各条先限于链接来源已经说明的调用子域；旧 selector 阈值只是出处，不是设计的最大适用域。准备、验证和错误前缀不会因列在后文就自动进入主线范围。

| 候选条目 | 代表证据与保留的整体决定 | 关系与当前边界 |
| --- | --- | --- |
| `root`：packed 融合链 | [Q/root][q-root]：路由压紧、预 gather、WS FC1 融合激活与再量化、FC2 写 BF16 RouteResult、按 slot 顺序 finalize。 | 默认阅读入口。以来源的 ABI 复用、普通主干为起点；保留原有 workspace 初态、存续与接续限制。 |
| `split_k` | [Q/split_store][q-split]：FC1 两个 reduction slice 写 FP32 Part，独立 activate 归并、激活和量化；本实例直接 gather4 取输入。 | 与 root 共用路由/packed 接口，新增全局 partial 接续。小准备见下文；gather4 和 split 在此样本共存，不宣称二者必然绑定。 |
| `route_direct` | [Q/direct][q-direct]：单 kernel 重复路由 census、按中间通道块直接计算、FP32 原子累加、最后到达者完成 token。 | 独立工作组织。构建 ABI 时的空路由选择不能替代回放时的非空处理；完整主线必须保留当前 empty/非 empty 两条计算事实。 |
| `verified_packed` | [Q/verify_root][q-verify]：验证先无条件写零输出并计算当前 device Clean；原主干仍提交，各 grid 依 guard 决定是否做 payload。 | 以 root 的 packed 链为 dirty 分支；验证是输出生产者。先保留不经前置 verify-only 图的具体实例；host 选择与双图版本问题只在下文解释一次。 |

## 一次记录的局部决定与配套条件

以下都只补相对上述代表或明确链接基线的差额；下游仍须消费同次调用匹配的 route map、ActQ/ActS、BM 和完成事件。名字/shape 相同不能证明接口相同。

### small_prepare

用一个 CTA 的 prepare 代替 root 的 count→scan→scatter：暂存路由、局部分组与前缀扫描，一次生成相互对应的路由和 tile map；Counts/前缀/RoutePos 完整写出，packed 与 tile map 仅写有效前缀。独立 Counts/Fill 清零不再需要。下游 handoff 改接 prepare 完成，PDL 的早启动许可不能替代此等待；不要求复现大准备的原子到达排列。`4R` 字节路由暂存及固定局部表须满足单 CTA 容量；本次只取[融合][q-small]、[split][q-split]来源子域，不把原性能阈值当容量证明。

### fc2_mono

相对 root 的分工 FC2，把搬运、MMA 和 epilogue 交给四个协作 warp；payload/scale 均用三槽，payload transaction 与 scale copy group 都完成才消费。每 item 重新 prime 两组，但环位置/完成 parity 跨 item 延续；前一 MMA 完成后才 refill，item 尾完成 TMEM drain 与直接 BF16 写回提交后复用。packed Act 和 RouteResult/finalize 接口不变。[单体实例][q-mono]。

### fc2_flat

相对 `fc2_mono`，把 item 的 stage 串成连续流，用三 payload/两 scale 槽位，payload、scale 与 metadata 预取跨 item；边界仍须完成前一乘积的 drain 和写回提交，才能复用 accumulator/metadata。[此代表][q-flat]是 `I=256` 的 NK1 实例；不把它推广为所有 flat 配置。

### fc2_cp_fallback

相对 `fc2_mono`，payload 改用 cp.async copy group，scale 改为同步暂存；item 尾 drain 后重启槽位但延续完成 parity。[实例][q-cp]保留空尾 group、两 stage lookahead 和匹配的 wait。来源中两种 TMA/flat 拒绝入口汇入这个实现，不另建两条记录；不能据此为 WS 内核、split-K FC1 或 REDF FC2 补造同样的[回落][refusal]。

### fc2_ws_direct_store

这是[split_ws][q-ws]的具体分工变体：payload/scale 在两 stage 距离联动复用，四次串行 drain 直接写回，完整 epilogue 后才释放 accumulator。它同时改变 DEEP/early-release/store，不简写成 `TSTORE=false`。对照 root 的 bulk Sink：只有完整 quadrant 可 bulk 写，部分块须 masked store；源读完只许可 Sink 复用，全局写完才许可 finalize 消费。[bulk 基线][q-root]。

### serial_handoff

关闭 PDL 时，同一计算链依普通 stream 的前驱完成接续，不再依靠早启动许可。[源码 helper][pdl]仍无条件保留，在普通 launch 下不产生该重叠；无需删去入口动作。[来源实例][q-serial]的规模阈值只作来源选择。

### abi_rebuild

ABI 重建可以落到与复用完全相同的计算 Design，但仍要保存 workspace 的初态、地址/布局绑定、图与异步工作的存续要求；重建不意味着一定重新分配，也不提供跨 stream 排他。具体新增服务条件集中读[no_scan_fused_small_flat 的 Realization][q-rebuild]；其三区与[普通 flat][q-flat]相同，不再单列主线。

冷准备还可能实际提交并同步 `init_tables`，保留模块期常量；[所选 packed 主干][q-init]并不消费它，不另当一项设计决定。这里保留来源事实，不重复格式表算法。

另有[已知错误接续][capture-failure]：BeginCapture 失败后仍可能在 nonblocking 私有流实际提交主干，再走 caller-stream fallback；未见两者 join。caller 的完成不能替代那份在途工作的完成/寿命保证，本次成功也不证明前次私有流访问已排空。本目录不把它包装成安全的普通重试，也不穷举错误码组合。

### eligibility_scan

ABI 重建时的零 probe 扫描 payload 零、scale 非 NaN 和必要标量有限性，跨 shard OR 归并，并据结果准备验证状态；它不替代执行时重新验证。probe 还检查两份权重 scale，执行时 verify 不再读它们；共同 Contract 已限定其有限值，不额外假设权重不可修改。[代表][q-probe]与[`moe_zero_probe`][zero-probe]区分提交前/后失败；负返回仍可进入主干，已提交 scan 不能随结果删除。

### graph_observation

图固定 ABI 选择/绑定，执行读取当次设备数据；不按回放时的新直方图重新选路。[双图实例][q-double]先执行独立 probe 图，必要时再执行带第二次 verify 的完整图；两次分别生产 ProbeClean/Clean 与零输出，不能合成一次验证。仅完整 guard 家族可承接 device 验证，direct 在此链之前返回。

`HostObserved` 不是当前 device Clean，也不是 kernel_done。[`graph_replay`][replay]重置同一 host word 后有限轮询；前次超时的[device writer][verdict]可能在后次重置后发布，无代次标签可区分。host 读到 1 而省掉完整图的来源路径因此存在版本缺口；这里不补造调用隔离保证，也不把 [verify_only 的旧批准][q-only]当成当前调用正确性的证明。仅链接到其他主干不证明双图组合正确。

## 初始 60 份显式配置：扩展边界

默认四条主线没有完整保存初始显式族。初始 `10 种 FC1 × 5 种 FC2` 的 50 个配对也不是合法性证明；[旧索引][i-index]已有 map、BM 与强制配置限制。若继续扩展，按下列新增决定审视，而不逐旧 ID 迁移：

| 初始族 | 本次判断与证据 |
| --- | --- |
| last-arriver split-K | 应另保留一个完整实例：[mono_arrival_mono][i-arrival]把 Part 归并交给最后到达者，计数初态、发布与参与屏障不能只写“少一次 activate”。本次未试写该完整主线。 |
| in-grid prepare | [selfprep_mono][i-selfprep]把路由 producer 移入 FC1 grid；须保留 `R≤256`、同值 map 写入与 PREP2/KEEP_PREP 仍可能保留外置准备的边界。本次未推广到所有 FC2。 |
| BF16 两族 | [warp][i-bf16-warp]先物化 FP32 Fc1、再独立激活；[tensor core][i-bf16-tc]在 FC1 内融合成 BF16 Act。应分别贯通保留；不能因 Act shape 相同而合并，更不能补造源码未提供的 BM 匹配修正。 |
| REDF 输出归并 | [mono_split_redf][i-redf]应保留完整实例：FC2 **先 BF16 舍入再乘路由权重**，[FP32 原子汇入 Acc32][redf]；[旁流清零][redf-init]须在 FC2 前 join，后继改 convert，不再读 rout。描述符拒绝不能[退回 rout 写者][refusal]。本次仅保留此扩展判断，未迁移完整实例。 |
| 其他 FC1/FC2 组织 | mono、w1、WS、ws12、last-arriver 的角色差别与 DEEP/PP 等仍有知识；它们和后端的全部耦合本次未完成逐项核对，不能计为“已保存，只省组合”。 |

## 信息保存对照

| 去向 | 本次实际保存到哪里 |
| --- | --- |
| 完整主线入口 | 四行各指向一个端到端代表；包括数值/物化、输出归属与依赖。完整 YAML 仍是来源，不是本次新交付。 |
| 一次局部说明 | 上述九条。验证的 device 输出在主线，host/双图版本在 `graph_observation`；两者不混作包装。ABI 条目保存必要条件与来源限制，不自称新的计算算法。 |
| 已知组合不另立条目 | 小准备+split、small+mono/flat/cp、重建+flat、tables+大准备、双验证+同一 packed 主干已有来源；只链接相关记录，不复制所有配对。未核接口的组合不归入此项。 |
| 主动不记录的实现精度 | cache 容器/分配 API、轮询确切次数、不影响已述协议的错误码数字、原 selector 的逐条件决策树；必要初态、完成、错误前已有工作仍保留。 |
| 尚未研究/完成 | 初始显式族的完整迁移、所有默认前缀与后端组合、完整异常轨迹覆盖、实际 GPU 正确性。已指出的 host 镜像/私有流缺口只做静态分析，没有修复或 GPU 复现。 |

[q-index]: ../../../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/dispatch.md
[q-root]: ../../../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/root.yaml
[q-split]: ../../../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/split_store.yaml
[q-direct]: ../../../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/direct.yaml
[q-verify]: ../../../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/verify_root.yaml
[q-small]: ../../../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/fused_small.yaml
[q-mono]: ../../../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/fused_small_mono.yaml
[q-flat]: ../../../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/fused_small_flat.yaml
[q-cp]: ../../../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/fused_small_cp.yaml
[q-ws]: ../../../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/split_ws.yaml
[q-serial]: ../../../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/fused_serial.yaml
[q-rebuild]: ../../../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/no_scan_fused_small_flat.yaml
[q-init]: ../../../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/init_no_scan_fused_big.yaml
[q-probe]: ../../../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/probe_fused_big.yaml
[q-double]: ../../../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/double_verify_root.yaml
[q-only]: ../../../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/verify_only.yaml
[i-index]: ../../../../schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/dispatch.md
[i-arrival]: ../../../../schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/mono_arrival_mono.yaml
[i-selfprep]: ../../../../schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/selfprep_mono.yaml
[i-bf16-warp]: ../../../../schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/bf16_warp.yaml
[i-bf16-tc]: ../../../../schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/bf16_tc.yaml
[i-redf]: ../../../../schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/mono_split_redf.yaml
[pdl]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:180
[refusal]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:8722
[zero-probe]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:9585
[replay]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:9471
[verdict]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:1764
[capture-failure]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:9533
[redf]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:3948
[redf-init]: /home/shaohanyong.shy/KernelSchema/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu:9164
