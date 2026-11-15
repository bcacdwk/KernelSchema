# M03：进展与真实卡点

## 进展记录

已从给定 Contract 推导访问域、同一行 GQA 复用、整体流量与预处理代价；选择每 `(row,kv_head)` 一份工作作为工作组织主路线，试写单 step 的 work_schedule、一个有跨 warp 就绪/释放边的 pipe，以及 cache repack 的跨 kernel handoff。比较了分 head、逐行 gather 物化、全 cache 重排、split-KV、跨行选择复用。已做分支身份判断，不要求每项都获得完整 YAML。

读取顺序是 AGENTS/DESIGN_SCHEMA 与输入，随后 common/math/schedule 和作者模板入口，接着 DOC 5/6 及 DOC 1/2 的范围/数值/资料规则。没有读旧 QSA Design、旧 dispatch 或其他报告。若批量读取被工具输出截断，会针对实际使用的条款重新读小段；不把截断部分当成已逐字核对。

没有遇到“必须先跑工具才能思考”的规范阻碍。DESIGN_SCHEMA 开头及模板入口都允许缺资料时继续独立部分；本轮受任务约束不运行 checker，故没有工具适配成功或失败的证据。

## M03-01：默认有限输入已澄清，但全域 FP32 语义仍卡住

- 分类：第一部分是自身理解已纠正；第二部分是输入/验收问题。
- 场景：准备把 q/gate 的输入范围用于在线 softmax 精度分析。
- 位置：输入 `qsa_prefill.md` 的 q/gate range 省略、K/V range；DOC 1 `schema_docs/1.contract.algorithm_requirement.md:60`、`:150` 起；DESIGN_SCHEMA 数学改写段要求检查中间量范围。
- 第一理解：未写 range，可能连 q/gate 的 NaN 行为也得处理。
- 重查：DOC 1 明定浮点 range 省略是“任意有限值”，所以不能以 q/gate NaN 为 Contract 缺口。当前采用这一规则。
- 尚存的两种可能读法：一是默认 FP32 公式就定义可能 overflow/NaN 的参考行为；二是任务希望在有限输入上得到数学上有意义的 attention，但公式未写安全求值路径。前者需要 tolerance 对非有限结果的比较规则，后者需要给出与默认 FP32/参考之间的关系。我不能代选。
- 实际尝试：取大有限 BF16 q/K，FP32 dot 能溢出；max-shift 不能逆转点积溢出。考虑更宽 dot 只解决数值范围，不自动等于既有参考。
- 影响：不能交付涵盖所有 Contract 有效输入的数值保证。结构工作能继续。
- 需要：确认极端有限输入及参考中非有限输出的验收语义；不是请求悄悄收紧范围。

## M03-02：online U 的新溢出反例推翻了初版精度论据

- 分类：自身方案不完整，非规范错误。
- 场景：用 online softmax 不物化 scores/attention 输出，拟采用 FP32 的未归一化 U。
- 位置：DRAFT 第 3 节；DESIGN_SCHEMA 第 2 节“输入有限、输出有界或保留 FP32，都不自动保证中间状态安全”；DOC 3 `arithmetic` 与 ledger。
- 第一理解：softmax max-shift + FP32 累加是合理初版，误差可留实现验证。
- 推演后改变：q=K=0、V 为最大有限 BF16 时，U 累加 2048 份 V 会溢出；原 normalized 权重路线没有同样数量级的中间值。这项不能只写“待验证”。
- 尝试：归一化均值块合并、宽 U；没有完成舍入误差/成本分析，也没有把它们当已选修复。即使修复 U，也还有 M03-01 的 dot 问题。
- 影响：可以选工作组织方向，但不能声称 FP32 OnlineAttention 数学层已全部收束。下游只是独立结构的草稿表达。
- 需要：继续选择可覆盖有效域的数值路径，或拿到任务方明确的参考处理规定。当前不是 schema 字段缺失。

## M03-03：知道使用共享槽，却不能从占位资料绑定 MMA/驻留保证

- 分类：已给输入资料不足。
- 场景：12 heads 共享 32-token 的 K/V 块，想用矩阵单元提高利用率，并判断单/双缓冲能驻留多少 CTA。
- 位置：Contract hardware_specification.wiki 全部 pending；DOC 2 `schema_docs/2.contract.hardware_specification.md:125`；DOC 4 执行映射与 DOC 5 execution_mode；DOC 6 实现交接。
- 当前理解：SM103、TMEM 和 async feature 名字不自动给出 tile shape、可用容量、CuTe 4.6.1 表达能力及原语完成保证。编造 API 名也不能补齐。
- 尝试：先做 32 KiB K/V 槽、12 KiB U 等算术容量账，使用普通串行计算的局部 pipe，不假定 async_ordered。没有为占位硬件跑调查；输入允许保留缺口。
- 不知道具体写什么：`realization.require.choice` 若要绑定异步 MMA，当前不知道应指定哪个已核定原语及其适用限制；不能只写“use Tensor Core”充作完成协议。
- 影响：无 Tensor Core 利用率、寄存器分配、实际驻留与候选时间优劣的可靠结论。不妨碍访问域、payload 成本及工作覆盖分析。
- 需要：相应硬件容量/原语与 DSL 资料。工具版本适配问题未实际触发。

## M03-04：全 cache repack 是局部变化还是整体方案，我仍拿不太准

- 分类：分支判断尺度，可能只需案例澄清。
- 场景：增加一次只处理映射页的 PackCache，约 512 MiB 最低预处理流量，MAIN 用紧凑 Kpacked/Vpacked，其他分组 attention 和 gate 不变。
- 位置：DOC 0 `schema_docs/0.common.md`“参数变化与结构变化”“局部记录与完整方案”；DESIGN_SCHEMA 第 5 节。
- 第一理解：多一个 kernel 与全局物化，应独立 schema。
- 重查后：文档明确“提交次数不决定记录深度”，并以能否围绕已有值/step/存储闭合为准，所以目前倾向局部变化：增加源视图生产与 handoff，主要递推/归属不变。
- 两种可能读法：可在 MAIN 已有 Ksel/Vsel 入口外围闭合，是局部；或者预处理是计入目标的一项主要计算，形成新的全局输出形成链，需要完整贯通，是整体。前一种读法更符合我当前方案，但“主要”的判定仍带主观性。
- 实际尝试：写出两次 launch、kernel_done 前驱与 Kpacked/Vpacked global handoff；没有给它起完整节点 ID，也没有复制全套方案。
- 影响：当前局部方案能表达；正式交付时不确定应把哪些事实放目录、哪些值得完整 YAML 身份。
- 需要：一个关于“仅重排输入、主 attention 组织保持”的边界示例或审阅判断。不据此宣称文档自相矛盾。

## M03-05：一个多 warp role 的 serial 不是整个 CTA 的隐含屏障

- 分类：自身理解经查阅已基本解决；字符串粒度仍需谨慎。
- 场景：四个 warp 合作加载共享 K/V，各自计算三个 heads。我想先写最简单单 role 流水。
- 位置：DOC 5 `schema_docs/5.design.tile_pipeline.md:206` 起 execution_mode、`:498` 默认依赖、`:185` describe 要求和成员规则。
- 第一理解：全部动作写在 MAIN role，serial pipe 可以自然顺次读写共享槽。
- 重查后：serial 的作用域包含“成员”，不会天然等待其他 warp；应有 all-member prepared/released，或明确且有依据的原语保证。
- 尝试：加入 `{issue matmul:Scores(i)} ← [{done MAIN[*]::load:KV(i)}]` 与 `{issue load:KV(i+1)} ← [{done MAIN[*]::matmul:Update(i)}]`。确认最后一轮没有下一 load，初轮无上轮等待，Update 完成覆盖本 warp 对 V 的最后读取。
- 字符串真正难写的地方：`load.KV.describe` 既要说 quad 解码/无非法读取，又要绑定共享槽、写入范围与可见性；短写“gather KV”缺关键保证，写成 warp/lane 地址代码又越过设计层。当前停在“每 warp 不重叠的部分、全部有效 selected rows、shared K/V slots、done 可供 CTA 读者使用”，不指定线程 layout。
- 影响：依赖关系已能局部表达；槽正式登记和 CuTe completion 绑定还未完成。不是 waiting 语法表达不了。
- 需要：后续实现绑定确认 load done 的具体跨 warp 可见性实现；当前未要求新增字段。
