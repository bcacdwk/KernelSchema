# 六类 schema 的恢复与返工评估

> 归档说明：六算子恢复阶段的历史记录。正文中的运行状态、预算、命令和分工对应记录时点，不作为新的执行授权。 [归档索引](../../README.md)。

本页保留 2026-11-05 初始评估时的状态和判断。后续采纳方案、凭据恢复及实际交付进度见[执行记录](EXECUTION_LOG.md)，阅读入口以 [Schema Wiki](../../../../schema_wiki/README.md) 为准。

评估日期：2026-11-05。基线为 `63e437695fef892eec925c5b5009ec91d9b88bd2`，当前 extract 版本为 2.5.0。本轮先评估已有资产与恢复路线，不代表六项已经通过新的 P6。两名子代理分别使用全新上下文和 `xhigh` effort，独立覆盖全部六项；Supervisor 核对运行状态、文件摘要、关键源码及命令边界后形成以下处置建议。后续独立任务继续使用新的子代理，不复用已完成审计代理的会话。

**建议保留现有投入：GDN 保留；GEMM、MLA 局部修订；FMHA、Sparse 优先保稿接续；MoE 保留 Contract 和 root 候选，重新审定默认路径计划，可作为优先考虑重启的一项。** 当前没有证据支持把六项一起从零重写。

六项都有历史正式交付，后来又启动了部分修订。当前推荐版本合计 81 个 YAML；计入被替代的 GEMM/GDN 原版，共有 8 套正式目录、86 个 YAML 和 8 份 dispatch。历史“六项完成”与当前“三条修订未完成”同时成立。不能从历史 COMPLETION、旧 post-batch 总结或 `published=false` 单一字段判断现在要恢复什么。

| 算子 | 当前可用资产与真实进度 | 内容处理决定 | 后续推进方式 |
| --- | --- | --- | --- |
| fp8_scaled_grouped_gemm | 2.2.4 校正版，3 节点，已终审并正式 export | 保留数学、调度及 graph 销毁错误码修订；局部修正 root 的 MMA 成员范围 | 原 run 已完成，正式新规则交付需新输出目录；当前工具不能原地 revise 已发布结果 |
| mla_expanded_fp8 | 2.2.2 单 root，已发布，无待恢复后继 | 保留 Design；补三个 nullable 及对应缺席分支，修正三个下游符号绑定 | 同上；内容上是有限修补，流程上需要新目录重新完成正式审阅 |
| gdn_prefill_bf16_state_bf16 | 2.2.7 六节点版 ready，正式 export 已完成 | 保留；本次未发现足以要求返工的新问题 | 不恢复旧两节点或 cancelled 前任；无需仅为使用新引用语法重提取 |
| fmha_mxfp8_qkv | 历史 8 节点已发布；当前修订 P5，29/43 节点在旧规则下有效合入 | 保留 29 个节点及其余节点自己的稿件；当前规则适用性 review，具体问题才 patch | 清算两次未确认调用后，优先经兼容桥接进入 2.5.0 连续修订，再完成剩余工作和 P6 |
| dual_cache_latent_bf16 | 历史 3 节点已发布；最新修订 P5，2/8 有效，另有两份在写稿件 | 保留当前 root、mma_direct 和各自草稿；先 review Contract 任务域，再继续新的拓扑拆分 | 清算一次未确认调用后采用正式修订，完成余下 6 个范围；不要恢复旧 3/7 的前任 |
| nvfp4_fused_moe | 历史 60 节点已发布；当前默认实现范围任务停在 P2 reviewer，P1 已批，root Design 候选尚无当前 P3 批准 | 保留 P1 与完整 root 候选，重审 P2–P4；旧 60 节点仅作有来源的参考 | 桥接方案可用时优先接续；若桥接成本明显较高，选择该项 full-restart，比丢弃 FMHA 前沿更合理 |

这里的“保留”保留的是内容和原有身份，不把旧批准自动认作 2.5.0 批准。29/43、2/8 均包含 root，不计已 removed 的历史节点。剩余节点数也不等于必须从空白写作的数量。

已核实的具体内容依据：

- **GEMM 的 root 需要局部 patch。** [校正版 root](../../../../schema_wiki/archive/six-operators/revisions/2.2.4/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/root.yaml) 第 33–35 行明确覆盖两种 dual；第 427 行登记 `warp[1:3]`，第 433–434 行却取消 dual=0 时的 warp 2。固定源码 `candidate/fp8gg_umma.cu:415` 仅在 `p.dual` 为真时让 warp 2 承担该角色，`:1830` 根据工作量选择 dual。[成员规则](../../../../schema_docs/5.design.tile_pipeline.md) 第 55、192–194 行要求真实参与集合，允许事先确定的整数参数端点，禁止最大集合再用正文取消。应确认 dual 的 0/1 定义后参数化成员范围，例如 `warp[1:2+dual]`，并复核相应完成/释放语义。定位到原版和校正版的 root；当前推荐修订针对校正版，wide/mma_sync 没有发现同类问题。数学与已修正的 graph 错误处理应保留。
- **MLA 需要有限 Contract 迁移。** [原 root](../../../../schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/root.yaml) 第 27–38、52 行仅用正文/default 表达可选对象，缺少 `nullable: true`。[Contract 规则](../../../../schema_docs/1.contract.algorithm_requirement.md) 第 214–232 行明确 nullable 与 default 不互相代替，且要求写明缺席条件、返回和写回。固定 `problem/input.py:101,113,123,129` 实际构造并传入可选 sink/lse；不能缩小任务域来回避。修补三个声明并核对 LSE 缺席时只返回 out 的行为即可，本次没有依据要求增加分支或重写 regular/tiny 的 Design。
- **MLA 另有三个绑定方向需要修正。** 同一 root 第 165–167 行将 Bq/Bk/N 的 `binding` 值写成 `ref:design.work_schedule::...`，但这三个值由下游调度固定。[数学区引用规则](../../../../schema_docs/3.design.math_reshape.md) 明确这种情况用 `bind:`。应修正值的前缀并核对目标，保留 `binding` 字段和数学主体；不是把字段本身改名为 bind。
- **GDN 六节点修订已经解决原表达问题。** [修订索引](../../../../schema_wiki/archive/six-operators/revisions/2.2.7/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/dispatch.md) 区分真实缓存/直接 QK、分段和 Perm 路径；root 已有 nullable，固定输入设置 `norm=false` 并提供 packed 元数据。原 `published=false` 是原输出目录冲突，不影响另行 export 已交付的事实。全本地 Design 在 2.5.0 仍合法。
- **FMHA、Sparse 当前草稿优于旧拓扑。** 旧 FMHA split_tcg 用正文取消 segment 提交；旧 Sparse 用正文取消 combine、选择 probe 后 packed/mma，不能把已声明流程当作可任意删除。新稿已把实际不同提交集合拆开：FMHA flat/segmented 等，Sparse direct/split、tc_heads/tc_packed 和 probe 路径。应沿此方向复核节点必要性、选择域和跨阶段接口，不能因为数量多就直接合并。FMHA 未有效的 14 个节点都有历史写作记录，其中一些保留旧批准或草稿；不能只保存 merged checkpoint 而遗失各节点的私有稿件。
- **Sparse 的源布局缺陷与未完成 schema 分别处理。** `dsv4_kernel.cu` 的 landing barrier/TMEM 返回槽/ukey 共址已有[归档证据](../../../issues/archive/kernel_wiki/sparse-tc-control-layout-overlap.json)。按当前 REVIEW 保留抽象设计与来源落实偏差，不把修 kernel 或猜测破坏后的具体值设为恢复前提；资源容量、寿命和依赖仍须完整表达，归档 issue 不能代替完整稿件。
- **Sparse 先复核 P1 的有效域。** 最新 root 第 47–55 行给两个 scale 默认值并允许 sinks=None，第 99–104 行又说明构造器固定使用 `D^(-0.5)`、1 和总存在的 sink；`problem/input.py:116–120` 确实如此。这是 B 发现、Supervisor 核对到的范围疑点，尚未排除其他 scope 材料允许更宽接口，不能直接判定删字段。先审 source/metadata/scope 是否支持额外可选域；若只来自 reference 签名，则局部纠正 Contract 转录并复核全树。不能为赶进度缩到样例，也不能未经依据扩为通用 API。
- **MoE 当前任务不等于旧 60 节点的 scope。** [默认范围](../../../inputs/nvfp4_fused_moe_default_scope.md) 已被当前 source lock 冻结：排除仅靠非默认实现控制到达的路径，但保留全部任务输入域、自动选择、零探测、PDL、graph/cache/fallback 和错误出口。旧稿的条件流程和 `warp[3:11]` 加正文取消成员问题必须修正，旧 60 节点不可直接当默认计划。当前 root 已有较完整候选内容，只是没有当前 P3 批准；P2 停止不表示只有一份空 Design。

恢复时必须处理的运行边界：

1. 当前三条未完成任务的冻结执行器分别为 2.2.7、2.2.9、2.2.10，均没有 `semantic_revision.v1`、`scoped_recovery.v2`。工作树 2.5.0 不会自动作用于旧 run；普通 resume 继续原规则，直接 `resume --scope` 也不可用。
2. 2.5.0 的 Design 引用、依赖与批准含义属于规则语义变化，不能伪装成 compatible runtime upgrade。保进度采用新规则，需要先确认旧调用结清，再登记一个经独立完整兼容审阅的旧语义中间执行器，之后正式 semantic revision。Git `60285f0` 的 2.4.1 是真实存在的开发候选基线，但本轮没有认证任何可直接采用的桥接树；这一步仍是后续工程工作。
3. 已有全本地 YAML 合法；维护优先保留每个节点 own raw，不自动转引用、不拿父 Design 覆盖子稿。P4 换父需显式 identity_migration 和字段范围，review 必须覆盖当前全文和实际依赖；父批准不能替代子节点适用性。
4. `continue --full-restart` 只把选定 checkpoint 全树封存为只读线索，新的可编辑起点只保留 root；分支置 removed、当前批准与草稿指针清空，阶段回 P1。它保留账目，但不保留 FMHA 的可继续作业前沿。不能把它称为普通中间恢复。
5. ready、published、已有 export 或已提交发布的 run，既不能连续 revise，也不能使用当前 full-restart 通道。GEMM/MLA 若要求正式新规则修订，使用新输出目录 extract，并明确 source/role/scope。`--id` 会重新冻结当前来源，不自动维持旧 fingerprint；需要完全保持旧材料集合时，先从旧 lock 构造 source.v2 描述。
6. FMHA/Sparse 原绑定输出仍是已占用的历史发布目录。恢复完成后应正式 export 到新修订目录；不能改写原交付或把历史“期望输出路径”当作已登记的新位置。
7. 三条任务的墙钟期限均已过。后续按实际维护和剩余范围追加时间；调用额度按 scope 核算，不能因总 attempts 多就统一充值。升级、修订、hold 释放均不自动增加预算，也不重置退修、P6 starts 或单次裁决。

当前目标 run 为 FMHA `45ea6608-66b2-4915-b47d-59d2c115602e`、Sparse `8f1be03a-af4a-44f2-a2ba-b8deda6ae3e7`、MoE `feb0c655-4e13-43ac-817d-1f457f31c029`。B 从最后失败调用事件定位到了 HTTP 403 的模型配额错误；FMHA/Sparse 还留下未确认终止。当前供应商配额是否恢复未做新调用探测；续作前须核实实际服务条件，不能把配额失败当成 schema 质量失败。旧 config 的 model/effort 也不能用 resume 的不存在参数静默替换。

后续执行先处理运行结算，再做迁移与内容；每个独立任务交给新的 xhigh 子代理，Supervisor 复核其证据与产物。第一项可明确构造的动作，是由旧 Runner 清算 FMHA/Sparse 的未确认调用。当前只读进程观察显示三个旧调用均已退出，但状态尚未结算；空锁或 PID 消失不能替代结算。

```sh
python3 schema_extract/extract.py cancel 45ea6608-66b2-4915-b47d-59d2c115602e
python3 schema_extract/extract.py cancel 8f1be03a-af4a-44f2-a2ba-b8deda6ae3e7
python3 schema_extract/extract.py inspect --run 45ea6608-66b2-4915-b47d-59d2c115602e 8f1be03a-af4a-44f2-a2ba-b8deda6ae3e7 --json
```

以上为拟执行命令，本轮未执行。已检查当前 CLI 转旧冻结入口、旧 Runner 消费 cancel 后不发新模型并 reconcile 既有调用的路径；执行后仍须验证 active/unconfirmed/unsettled 全空。当前 active hold 是 `15837dcb-5f2f-4e3a-a4d8-d99631834cd6`，不能误用历史已释放 hold。迁移方案明确后再处理该 hold 和各 run 的手动控制，不用批量释放代替逐 run 检查。

桥接方案成立后的顺序是 `compatibility-report → 独立审阅 → upgrade-runtime → revise request/prepare → 逐范围计划 → revise apply → resume → 当前 P6 → 新目录 export`。桥接路径、原始兼容审阅 JSON、实际清算后的 checkpoint 和维护计划尚待形成，因此本报告不提供可直接粘贴的虚构参数，也不宣称可以立刻无损迁移。完整 CLI 语法、原稿定位和节点表见两份独立审计记录。

建议先以 FMHA 验证保稿修订通路，收益来自保护 29 个有效节点及未完成稿；随后用于 Sparse。MoE 保留默认范围与已有候选，桥接可复用则接续，否则从该项显式重启。GEMM/MLA 的小范围修订可由各自新代理单独处理，GDN 继续使用六节点版。新规则 P6 是后续交付条件，本轮报告不代替它。

本次验证覆盖 23 个真实 run 的状态与 lineage、8 套正式目录的 94 个产物摘要、当前冻结能力及关键命令代码路径。两位审计者均运行了当前 `--skip-hardware` 检查；8 套正式目录及当前已合入子集全部通过。Supervisor 另核实正式文件 SHA 无漂移、GEMM/MLA 当前 candidate/problem 字节与对应冻结材料一致，并只读确认三次未结算调用的进程退出证据。正文审计覆盖 root 主干、全部索引和代表分支，没有逐行独立验收全部旧稿或全部新节点；不能因此宣称六项当前语义全部通过。按提取流程，本轮无需 GPU 验证。

独立判断及 Supervisor 取舍：A 建议保留 GEMM，B 定位到 MMA 成员范围的具体错误；Supervisor 核对 root.when 与 CUDA 实际分支后采用 B 的局部 patch 建议。两份报告均支持 GDN 保留、MLA 有限 Contract 修订、FMHA/Sparse 保护已有进度、MoE 按默认 scope 决策。独立报告原文保留各自结论，不把分歧回写成事先一致。

记录校正：B 末尾“AGENTS.md（保持空）”应读作“保持原样”；原文件内容为 `KEEP THIS FILE EMPTY`，本轮未修改。该表述不影响其算子评估。

原始评估：[审计 A](audit_a.md) · [审计 B](audit_b.md)。

后续组织与派发见[推进方案](CONTINUATION_PLAN.md)，其中进一步区分 FMHA 的离线保稿验证和 Sparse 的首次正式迁移顺序。

已开始执行的动作和运行定位见[执行记录](EXECUTION_LOG.md)。
