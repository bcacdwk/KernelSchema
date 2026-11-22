# R1 执行、案例、验收与维护复审

日期：2026-11-20。本轮只提出问题，不修改正式计划或实现。审查依据为本目录 `before/` 中的计划快照及下述源码只读核对；没有运行模型、GPU、远端服务、安装或资格测试，没有提交、fetch、fork，也没有派子代理。

简称：P=`../04_implementation_plan.md`；W=`../05_work_packages.md`；C=`../planning/r5_contract_execution_review.md`；V=`../planning/r5_validation_delivery_plan.md`。KS、AKA、EXP 分别为 `/home/shaohanyong.shy/{KernelSchema,atrex-kernel-agent,AKA_experiment}`。计划定位的行号对应审查前版本。

已读两仓 AGENTS、用户简报、已确认 D1–D3、主计划/工作包、两份采用的 R5 附录；另外核对了 AKA 的 Task/Source/V0/评测/生产审阅边界，EXP 的源码投影/加载与现有 evaluator hooks，以及 FA4 原 Contract、direct Design、dispatch、输入/reference/host 来源。

## 结论与优先级

没有发现应推翻 D1–D3、改换首题或延迟全部 CPU 开发的理由。建议先处理 E-01、E-02；E-03 是首期动态设计验收的缺环，E-04 只影响后续 pilot 的性能解释。以下没有把“资格尚未实际运行”本身列作 bug。

| ID | 优先级 | 问题 | 证据性质 | 相关交付 |
| --- | --- | --- | --- | --- |
| E-01 | P1 | `return is out` 不能保证写回调用者原输出缓冲区 | 已证实规范缺项；具体候选未执行 | KS-01/04、AKA-02、V01 |
| E-02 | P1 | 未完成 coder 骨架被映射到要求先通过测量的 native V0 source | 已证实计划映射与现代码冲突 | AKA-02/05/06、V11 |
| E-03 | P2 | 新 Rd/R' 的 fidelity 判据从何产生、如何与 R0 分离仍未闭合 | 已证实验收未覆盖；实现风险为推断 | KS-06/07/09、AKA-07/10 |
| E-04 | P2 | 用同批计时选最快候选又报告其性能，引入随候选数变化的偏差 | 离线精确演算；非性能实测 | EXP-01/02 |

## E-01 — 输出对象身份检查遗漏原 storage/view 身份

- **计划定位：** P:136–138；V:36–49、63–65、75–82；W:29–33。规范明确要求 x 的 storage 身份不变，却只对 out 规定返回同一对象、数值、写满与外围 canary。
- **触发反例：** 实现者算出正确的 `tmp` 后误用 `out.set_(tmp); return out`，或给 `out.data` 重新绑定同形状结果。`return is out`、shape/dtype/device/有限性、数值、x 不变、旧 backing 的前后 canary 都可以成立；调用者另持有的原输出 storage/view 内部仍是毒值。只检查“调用后的 out”会跟随被替换的 storage，漏掉真正的目标缓冲区。
- **证据：** V:46–49 的检查对象是返回对象及调用后的 out；V:63 只明确检查 canary。与之相比，现有 EXP `assets/bench/src/atrex_bench/eval/source_hooks.py:72–73,118–149` 已分别保存入场 view/storage 签名与原始 backing，并在 `check()` 中拒绝 metadata/storage 改变；这里有可参考的现实实现边界。
- **影响：** 公共 evaluator 可能把不符合预分配输出 ABI 的实现判 pass；即使 R0 reviewer 有机会发现，也不能用 fidelity 代替独立正确性判据，尤其 B 组没有 R0。不同 backend 若一侧采用现有 BackingGuard、另一侧照本计划实现，还会得到不等价的 Task 判定。
- **最小修正：** 在 Task 及 V01 中固定 out 的入场对象、storage、storage_offset、shape、stride、dtype/device，并检查原输出区域已写满；加一个“同对象换 storage”的坏候选。任意合法 storage_offset 的承诺还应至少给一个明确的非零 x/out view 样本，或将首期布局支持范围收窄到实际要验的范围。无需另建内存安全证明框架。
- **首期相关性：** 直接进入 KS-P1 和 AKA-P1 的公共 ABI 验收；应在编写 evaluator 之前修正。这里证实的是规范漏洞，没有声称该候选已在 GPU 上通过。

## E-02 — native InitialSource 不能直接等同于可不完整的 coder seed

- **计划定位：** V:141 将 native `initial` 定为“共同骨架/显式 seed”；P:87、C:134 允许 Task.starter_source 为骨架且 seed 可为未完成代码；W:103–107 要完成该映射。AKA-05 虽列 V0/V1/helper 职责，但未修正这条已采用的映射。
- **触发反例：** 新 Task 的 `Model.forward` 骨架抛 `NotImplementedError`，或用户给一个尚未能构建的合法 UTF-8 partial seed。若按 V:141 把它交作 native InitialSource，Light 在第一次 coder 准入之前就测该源码，V0 不合格，fixed implement 还没开始便失败。`framework_baseline=never` 只跳过后续 V1，不能解决此问题。
- **证据：** AKA `src/aka/contracts/problem.py:17–21` 将 InitialSource 定义为选中的实现；`src/aka/workflow/light/campaign.py:182–203,258–280` 从选中 SourceProvider 取得代码、核对源码身份、记录 V0，随后立即调用 `_run_v0_evaluator_with_correctness_prefetch()`。`src/aka/workflow/light/orchestration/campaign.py:1109–1118` 在 `all_pass` 不成立时抛错。`src/aka/workflow/light/long_horizon/main_adapter.py:30–42` 的先后顺序是 setup_baseline，再 ensure_framework_baseline。
- **影响：** 首次从骨架实现和从部分代码续写是已确认的核心能力；这不是可在真实资格阶段简单标 unsupported 后仍声称 AKA 首题已完成的边缘输入。若为让 V0 通过而暗用完整 reference 作为 coder 起点，则又破坏共同骨架与 seed 的约定。
- **最小修正：** 删除“native initial 就是共同骨架/seed”的一对一映射。明确 native bootstrap/V0 使用哪份可测的 Task baseline，公共 starter/seed 仅在第一份 coder draft 的准备边界安装；分别记录 V0、G、C0，费用仍归同一父账户。若采用另一种 V0 helper，必须同样明确两个来源，不能伪造 baseline pass。由 AKA-02/05/06 合作即可，无需新增工作包。
- **验收补点：** V11 使用真正不可执行的骨架和非空 partial seed，证明 V0 测的是明确的 baseline，首调实际收到的是指定 C0，且中断恢复不重做 V0、不把 G 当 seed。已有 V04 的“e1 后 C1→e2”不能单独覆盖这个冷启动缺口。
- **首期相关性：** 阻塞 AKA-P1 的 first implementation；KS 独立链不受此问题阻塞。

## E-03 — generated Rd/R' 需要自己的 fidelity 解释，现有必验反例只围绕手审 R0

- **计划定位：** W:41–45 的具体交付为“R0 条目/方法 manifest”；V:86 明确 design 的 Rd 不必等于 R0，V:96 允许通过“一 program 多行”形成实质 R'；W:59–63 要做真实 design→implement→optimize。C:235、258 规定条目在规则/请求闭包内固定且全部必需项通过，P:117、123 规定新 R 重判 fidelity。
- **触发反例：** planner 合法选择一 program 处理多行。对应源码对手审 R0 必须 fail，对这个新 Design 则可能 pass。若“重判”只是重新运行随 fixed fixture 带入的 R0 清单，永远无法完成新设计；若仅换 receipt 的 R 摘要而不改变条目的解释，则新的设计义务可能完全没有受检。
- **证据：** V02（V:198）只列 R0、外部 softmax、省 max-subtract；V06（V:202）要求换来源后不继承旧 pass；V09（V:205）检查发布/准入。它们都没有要求证明“同一 C 对两个实质不同的 R 得到应有的相反 verdict”，也没有明确 writing 输出交 implement 时由谁建立适用于当前 R 的必需条目。这不是要求模型证明任意程序等价。
- **影响：** fixed-first 可能完全通过，但三入口组合一遇到合法的新设计就沿用旧判据，或漏检新增约束。真实 optimize 若只产生 keep，也不会揭示这条缺环。
- **最小修正：** 明确选用一种简单机制：通用审阅方法按当前完整 Schema 解释全部必要义务，或在发布/请求准备时派生并冻结当前 R 的清单；由 KS-06/07 持有，AKA 只消费。条目不能由 planner/coder 通过删清单自行豁免。给 V02/V09 加上述 ownership 变化的 synthetic 正反例即可，不要求真实 planner 强行 revise，不必增加新服务或公共字段。
- **首期相关性及把握：** 应随 KS-P1 的 design demo 闭合；也影响后续 AKA dynamic。已证实的是计划验收缺项，尚无新实现，不能据此断言未来实现一定会硬编码 R0。若裁决认为现有“按当前 Schema 审阅”已隐含覆盖，至少应留下这一个可判定的 fixture，避免只证明 receipt 换绑。

## E-04 — pilot 的“最快候选”选择与最终性能估计复用同一批噪声

- **计划定位：** V:267–274、284–295；P:191–193；W:207–217。默认从已完成测量的历史候选中选几何平均耗时最低者，再把其计时作为终态性能；没有独立的选定后测量，亦未将其明确标作选择用估计。
- **触发反例：** 两组生成的所有候选真实速度完全相同，某组因更早完成/不同 planner 与 reviewer 开销而有两次候选测量，另一组只有一次。共同总预算不会使测量机会数相等；取观测最小值的组会显示更快，即使代码实际上没有改善。
- **离线证据：** 只用 Python 标准库穷举：每个候选真实耗时 10 μs，观测独立取 9/11 μs、各一半；候选数为 1、2、3 时，选中观测的期望分别为 10、9.5、9.25 μs。四 shape 若共享该次测量抖动，同样影响几何平均指标。这只是精确演算，不是实测噪声模型或性能结果。
- **影响：** D−F/F−B 的终态速度可能混入不同搜索机会带来的选样偏差；“同选择规则”“只有三次重复”“不声称普遍收益”都不消除它。pilot 仍能记录运行事实，但不能把同批选中值直接解释成所交付 C 的独立性能估计。
- **最小修正：** 更简单的首期选择是删去按观测最快选胜者，预先采用与实测速度无关的规则，例如选最早完成硬项的 C。若确实保留 best-of-budget，则冻结所选 C 后用预留额度做一次独立公共计时，选中后不得再挑；预算不足就明确无独立终态测量，保留选择用数据。所有失败/历史测量继续保留。
- **首期相关性：** 只约束 EXP-01 的预登记与 EXP-02 的解释，不应反向阻塞 KS/AKA fixed 发布，也不需要为了本轮复审启动更多实验。

## 核对后未另列为 bug 的部分

- FA4 F1 与现有来源吻合：`q4_direct.yaml:8,94–99` 接受 B1/Q4/L64，原 `input.py:33–40,64–78` 给一页、有效首项 0，其余表项 -1；root 13 参数、FP8/BF16、完整容差及 scratch 可写语义可定位。F2 十配置仍属 N=1 的 direct 域；计划没有据此替 N≥5/PageIndex 或 global 空 split 签证明。冷启动/native launcher 限制也在原 `dispatch.md:40–44` 和计划中保留。没有理由为“更复杂”而把这些范围全部提到 KS-P1 前面。
- Agate/Atrex 能否完整运行判据，计划已经安排了实际 hook、来源摘要及本地退路，不能仅因尚未调用服务而报 bug。只读发现 EXP 已有 `aka.source_hooks.v1`，可作为资格调查线索；它依赖的实际 runtime 与传输闭包仍须冻结，不能把本轮阅读当成独立 AKA 已获得该能力。
- 三仓投影处理与当前代码方向相符：EXP `tools/lib/aka_vendor.py:93–99,144–151` 确实排除 gitlink 且要求 excluded materialized=false；单独物化 KS 比把它硬塞回 AKA projection 更合理。新配对、旧 capture/reader 和新 run 回退的区分已明确，没有发现必须重写旧捕获的矛盾。
- 按 W 的接口/集成/发行限定核对主要依赖后，未发现仍存在的硬循环。AKA-11 不等 MIG-04、KS-07 开发不等 KS-06 实测、KS-08 不等 KS-07 的修正已落实；不建议重复增加同一门或把环境缺项升级为全局停工。
- 小题可以证明四入口、固定实现与准确续写的产品链；计划已限制“借公开参考设计”的结论，未把教程型 softmax 或 keep 当作复杂 Schema 有效性/提速证明。本轮没有据此要求换题或强制真实 revise。

## 关键源码核对身份

源码可能处于他人活动树；下列 SHA-256 仅标识本轮实际读取内容，不代表提交或准许迁入。正式计划快照由本目录 `baseline.json` 管理。

| 文件 | SHA-256 |
| --- | --- |
| AKA `src/aka/workflow/light/campaign.py` | `3569a26d3d13411cced5d219988d493aeb4dcd31bcc1c12080275dc9e2d747d2` |
| AKA `src/aka/workflow/light/orchestration/campaign.py` | `6163243ed09d16303f9d305e0c3bf85b6d597e4d165d7c5a6183f5a1cdacaf84` |
| AKA `src/aka/workflow/light/long_horizon/main_adapter.py` | `d58971ff71e2a7dac42a21a7ccb09a6a15a861a81845ba68f0acbcc22d6e0c2e` |
| EXP `assets/bench/src/atrex_bench/eval/source_hooks.py` | `e467ea32bdf8a56641d340b3786cafac0000ec9c033920491dff26f923a46609` |
| EXP `tools/lib/aka_vendor.py` | `ceb0e53fef1dac69a615c6a99521b17217ff331a06e6233922476dc37b2bd3d4` |

本轮唯一执行的演算是 E-04 的有限枚举；没有运行待实施的 workflow、Task/evaluator 或模型。正式修改留给裁决后的最后一轮。
