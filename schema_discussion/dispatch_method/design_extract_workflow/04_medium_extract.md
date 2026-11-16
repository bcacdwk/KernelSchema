# Medium Extract 限域试读交接

本次代入未来 Worker，再从接手 Reviewer 角度自查；不是独立审批。只读真实材料和给定旧稿的入口片段，没有运行 Python、Runner、checker 或 GPU，没有修改 YAML、源码、正式规范或运行状态。没有新的 P1 批准，因此以下是机制判断与后续写作建议，不是 P2–P5 完成结论。

## 实际阅读与范围

已读 `EXTRACT_SCHEMA.md`、P2/P3/P4/P5 全页、`REVIEW.md`、`schema_docs/0.common.md` 和 `EXECUTOR_ALIGNMENT.md`；按问题查看了分区规范导航。没有填写正式字段，数学/调度/流水线完整写作时仍需读取对应分区规则全文。源码阅读集中在：

- `problem/reference.py` 全文、`metadata.json` 与 `shape_range.json`；定位 `input.py` 的生成调用、量化、路由、可观察输出，以及 `problem/solution.py` 的原库调用。未逐项读取全部 shape cases，也未复核原库内部实现。
- `candidate/solution.py` 的 shape/family gate、cfg/cfg2 选择和 small threshold；`candidate/nvfp4_moe.cu` 的 `run_pipeline`、97/98/99 与 FC2/81 实例、对应 WS 主体、split partial、激活、prepare 和 finalize。未审完整图服务、ZERO2 或其他 FC2 族。
- 给定旧 `root.yaml`、`split_store.yaml`、`fused_small.yaml` 的元数据及开头数学片段，仅作定位，没有审完整有效稿件。

本次工作树 HEAD 为 `36175e14a5e32bb91b419b7938c198067518821d`。candidate 两文件实际 SHA256：`solution.py` 为 `0a71613fe4beec9314570cf128f2d455e91adf7182467c261a36ac11e45f0688`，`nvfp4_moe.cu` 为 `cf0b42e2e4809cf71b2442f1b64285da4cc0a69214f43eb82efbdac938978f6f`。下文源码行号以这些字节为准；旧稿的 source 身份未直接当作本次冻结身份。

## 给下一位作者的机制说明

任务计算边界先保留清楚：输入是已提供的路由、packed E2M1 payload/E4M3 scale 和 prepared weights。`reference.py:69–92` 的实际链是解码、按专家 FC1、clamp/激活、每 16 通道 NVFP4 QDQ、FC2；**每条 route 的 FC2 先转 BF16，再乘路由权重并 FP32 求和，最后写 BF16 output**。`reference.py:100–101` 写入传入 output 并返回 `[output]`。`input.py:112` 固定本地 offset=0、无 bias 等生成调用条件；模型 TP 的跨分片求和不在这个本地调用中。不能把 source 的某个 selector 范围缩成共享 Contract。

本次建议保留两项拟完整代表，都是计划，不新建文件：

| 拟代表 | 当前已核实的实际机制及完整价值 |
| --- | --- |
| root：FC1 cfg97 + FC2 cfg2=81，正常大 prepare 链 | `cu:7728–7779` 的 cfg97 为 WS3、SK=1、NPAIR、预 gather A；`cu:5044` 后直接在 FC1 epilogue 做激活和 packed QDQ。FC2/81 消费 packed activation，写 route BF16，再 finalize。可作为完整 packed fused 计算链的阅读入口。 |
| split_store：FC1 cfg99 + FC2 cfg2=81 | `cu:7797–7819` 实参是 SK=2、AMODE=3、NPAIR=true、SFCH=true、SFPF=2；没有预 gather slab。`cu:5021–5043` 将两片原始 FP32 up/gate 写到 `part`，`cu:9297` 后提交独立 `k_act_packed`，再接同一 FC2/81。主要 FC1 归约工作轴、中间物化、激活生产者与交接都变了，值得完整贯通，不能只记“SK 从 1 改成 2”。 |

这里只选择了计算实例，未声称 root 已有完整调用范围。`solution.py:1202–1310` 独立选择 cfg2，故“cfg99 路径”不能自动等同“FC2 也用 99”。本次明确固定配对 81，不研究其他 FC2 族；`cu:8330–8363` 的实际 FC2/81 实参包含 DEEP、TSTORE、EPITB、CPHOIST、EARLYDONE，不能只抄旁边旧的“override-only”注释。`solution.py:1308` 已可选择 81；98/99 注释里的“override-only”同样不能替代当前 selector（例如 `solution.py:810` 返回 99）。

输出去向在两份代表中相同：FC2/81 对完整 32-row quadrant 使用 TMA store，对尾部 quadrant 用有效行掩码的普通 store（`cu:5164–5200`），都先执行 BF16 RN。`k_finalize` 按原 token/slot 的 `route_of` 找 route，跳过负索引，按 slot 顺序 FMA 加权（`cu:1464–1501`）。完整 quadrant 与尾部是当前固定方案的执行情形，不另立配置节点。

### 差异记录的建议落点

1. **cfg98 相对 split_store 的输入准备与 scale 搬运，记局部条目。** 它仍是 SK=2、NPAIR、相同 `part → k_act_packed → act_q/act_sf → FC2` 链。实际差额是 AMODE=2 的 route-order 预 gather/2D TMA，SFCH=false、SFPF=0；99 为 clipped gather4 与 chunked scale staging（`cu:7797–7817`、`9250–9275`）。记录需同时交代 gather slab 的生产与寿命、A payload/scale 来源、权重 scale 行映射和等待字节数，不能只记模板取值。`cu:4669` 的 `nbox` 控制有效 gather4 box，`expect_tx` 随之调整；它是 99 固定机制内随有效行变化的搬运范围。主体 split 归约、激活、FC2 输出归属仍成立，当前没有理由额外创建 98 的完整 YAML。
2. **small prepare，记局部条目并复用于已核对基线。** 大路径为清零、count、scan_tiles、scatter；小路径 `k_prepare_small` 用一个 1024-thread CTA 完成私有直方图、前缀、scatter 和 tile map（`cu:918–1011`、`9194–9240`）。差额要定位 Counts/Offs/TilePrefix、route 双向映射与 prepare 工作及交接；小路径的分组内 atomic 排序不同，不能理解为所有中间数组逐字节相同。主链只依赖一致的紧凑路由映射和逆映射，各专家 tile 与最终 slot 求和可继续成立。cfg98/99 没有 SELFPREP 跳过条件；同一 small prepare 能喂给二者。不必再按“98×small、99×small、97×small”各写完整节点。
3. **NPAIR 的收益及数值边界保留在选定正文。** 它将 up/gate 两个 N64 MMA 合为 N128，并联动 B-scale 落点（`cu:4757–4800`、`4957–4965`）；这是当前所选实例的一部分，不再为“99 相对 98”重复登记。未调查非 NPAIR 全族，不在此声称所有相关组合均成立。

上述是 P2 级线索，引用的对象名尚是建议。P4 必须依真实已审 root/其他基线对象落实 common 的“基线、差异与联动、条件与依据”；不能让这份交接或目录补足 YAML。旧 `fused_small.yaml` 已存在并不构成保留完整身份的理由。

## 实际疑点、分类与下一步

**ME-01：数值路径不能合并成一句“相同激活”（已核实差异，正常待写）。** `cu:506` 的 `activation2` 使用 folded prescale；`cu:537` 的 `activation2_nf` 先按 alpha/倒数缩放，再调用 sigmoid/tanh 的非折叠形式。cfg97 fused epilogue 调前者（`cu:5077–5091`），98/99 的 `k_act_packed` 调后者（`cu:1417–1431`）；此外 split partial 按 slice 升序累加。影响是数学区必须保留求值顺序与近似/舍入，不能宣称两代表数学整区相同或直接 `from`。此项不是流程文案问题；P3/P5 分别沿 helper 完整写出，不把源码注释的误差量级直接升级为已验证保证。

**ME-02：仍未完成的执行关系审阅（源码可用，不称输入缺项）。** 已看到 WS payload/scale 分别按 STAGES/SFS 退休等待、epilogue 对 accumulator 的复用约束，以及 TSTORE 的 source-read wait 与全局完成不是同一件事。尚未逐个贯通所有 role 首轮/稳态/末轮、跨 item CPHOIST/EARLYDONE、PDL 交接及 empty/tail 情形；`k_act_packed` 按 Rmax 而非有效 route 数处理、预取 partial 在 expert 检查之前（`cu:1357–1391`），也需写清无效行实际发生的读取与消费者边界。当前仅判断计算链与代表价值，不对同步正确性或全域有效输出作结论。P3/P5 应继续读这些函数及 `run_pipeline`/实际调用边界，若发现动摇主要机制的证据再回 P2；不能预先补造等待或以 host 返回成功证明完成。

**ME-03：验收来源边界（给定材料尚不足以复刻全部外部检查）。** `metadata.json` 同时给出了 default 与 `mutated_inputs.output` 的 atol/rtol、mismatch fraction、max_rel_l2，描述中还给出 upstream `torch.allclose(reference_float, candidate_float, ...)`。本次输入未指定完整标准 runner config 与该分支比较路由/rel_l2 公式；我未追索外部实现。正式 P1 须保留已知比较对象、阈值和材料差别，依据 REVIEW“来源缺项”及 DOC1 tolerance 来源记录处理；不能自行选择一组阈值当全部定义，也不能把这个边界变成当前机制调查的阻塞。没有核实到任务计算本身缺少来源。

**ME-04：旧稿与工具状态（已明示边界，无新增框架问题）。** 旧 `split_store` 含 `parent: fused_small`，`fused_small` 含 `parent: root`。现行 common 的 dispatch 只有 `id?/when`，共享是实际整区 `from`，不是树形继承。`EXECUTOR_ALIGNMENT.md` 明说“不能用当前工作树的 CLI 端到端运行这套新方法”。因此旧稿可辅助定位，不能据旧 parent、旧批准或旧 checker 认定共享和阶段前提。后续正式检查等待匹配工具能力；本次不迁移旧 run。

下一步先由 Supervisor 确定正式接续及固定输入身份；缺失的新 P1 批准不能由本报告补齐。在正式前提成立后，P2 可据本说明继续确认限域代表与初步范围；P3 只写具体 root 的数学、调度、流水线及必要 Realization，并落实自身 when。P4 依据完整 root 确认两份集合、将 small prepare 落为目录记录、给 split_store 明确仅计划的写作任务。P5 写 split_store 完整有效全文；只有实际存在已审且真正整区相同才能 `from`，不能因 FC2 相同便引用混合了 FC1 的整个调度/流水线区。依赖 split_store 的 cfg98 条目先留计划，正文有对象后经 P6 集中落记录。普通新记录留交接汇集；改变集合/未来依赖回 P4、改变重要选型回 P2、Contract 错误回 P1。

## 接手 Reviewer 会怎样核

我会先从当前正式 root、目录及实际引用闭包建立理解，独立核对上述代表价值，而不是拿本报告或旧稿补定义。核查重点是：路由 inverse map 到所有输出的闭环；FC1 split partial 和激活真正的生产者；BF16 route 物化不能消失；folded/non-folded 激活差异；small prepare 与 gather 差额外的主体确实仍成立；每个引用在使用节点的参数、数值、完成关系、存储寿命和本地 Realization 是否成立。

未完成正文回到该 P3/P5 作者；实际 provider 定义错误回定义处，provider 正确但使用处不适用则在本节点本地化；重要选型与集合变化按上面阶段返回。缺比较器定义按已明确的来源缺项规则保留，未读同步源码继续研究，旧工具问题留在执行器对齐边界，不混称文案错误。阶段放行仍需完整范围的独立审阅与匹配检查。

此次流程使用体验较顺：P2 的“入口注释只作导航”、common 的局部闭合判据和 P4 的未来基线规则，足以帮助我从旧三文件拆法转向两份代表加必要记录；不需要另造分类或报告字段。最容易遗漏的是同一族中不同阶段的数值实现与 cfg2 配对，现有 P2 第 2–3 步已经要求核实。没有遇到必须修改正式流程文字才能继续的歧义，也没有为增加问题数量提出文案修改。
