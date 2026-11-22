# 第一轮 Schema 审计：事实、最小修复与验收边界

本报告是审计与方案，**没有实施修复，也不是新的 P6 批准**。只写本目录的报告/索引；没有修改规范、Template、Extract、正式 schema、kernel 或运行态，没有启动 extract/resume/monitor、真实模型或 GPU。

基线为 HEAD `484902855fe7be82a74fe7c394503d1d7d222dbb`。范围取 [ISSUE_RECAP](../../extract_twelve_20261119/ISSUE_RECAP.md)、[FINAL_REPORT](../../extract_twelve_20261119/FINAL_REPORT.md)、[精确十二个 run 的交付索引](../../extract_twelve_20261119/DELIVERY_INDEX.json)，按已经发生的问题定向回查当前正文、原生回执、必要源码和检查实现；没有重新逐算子通审全部 33 份设计。历史两轮审修及未定的 workflow architecture 只作为背景，不把历史批准当本次结论。

本次还读取了精确十二个 run 的所有 `receipt.json` 中的 `framework_candidates`，去重得到 **24 个 ID**；完整根因映射及回执坐标见 [SCHEMA_FINDINGS.json](SCHEMA_FINDINGS.json)。候选中的“独立确认”仍须核实：下文既有采纳，也有不采纳和更正原判断。

优先处理三件事：规范化不得合并不同数学对象（S-06）；验收记录不得为满足 100 字符限制丢掉已知事实（S-07）；已发布正文中的明确漏项应定点修订并绑定新的当前清单（S-01/02/03/05/09）。大部分既有内容修正方向正确，不能据此宣布现有产物没有残余错误。其余工作主要是少量规则澄清和提示归位，不需要新的 schema 层、证明框架或全量重提取。

## S-01 数学归约域与精度链：保留正确修复，补掉不成立的精确性声明

**已确认。** Dense GEMM 的 [weight_band_mma.yaml](../../../schema_wiki/aka-cuda-operators-20261030/operators/gemm/dense/bf16_nvfp4_gemm/weight_band_mma.yaml) 第 33–49 行已把四个 quarter 写成 `g=16*t+4*q+u`、`r=4*t+u`，覆盖全部 `k/256` 个 stage，并同步 accumulator 的终点及 retention。原缺陷回执为 run `2b64ba84…` / attempt `a4939137-59cf-4702-a771-bd1067041737`；对应源码 `candidate/kernels.cu:13745–13792` 是循环外清零、循环内四组累加。这是完整修复归约漏项，不应再统一成单 stage 的短式。

KDA [prepare_chain.yaml](../../../schema_wiki/aka-cuda-operators-20261030/operators/linear_attention/kda_raw_prefill_state_bf16/prepare_chain.yaml):46 的 `ki_f32` 分流正确：`prepare.cuh:803–815,839–843` 先得到 FP32 `ki_pair`，`kr_pair` 直接使用它，之后独立窄化。root 的 restore 遍确实重读 BF16 `ki`，因此两族写法不应为了整齐再合并。终审回执 `bda2a4ba…/5ab1ac5e-582b-4ac7-87c6-8194434b6672` 明确区分了这点。

**仍需调整。** 同次定向回查发现 masked GEMM [root.yaml](../../../schema_wiki/aka-cuda-operators-20261030/operators/gemm/grouped_masked/fp8_mxfp4_grouped_gemm_masked/root.yaml):39、89、117–118 把“尾数位少、缩放是二次幂”直接升级为全声明域的 FP32 精确乘积/仅归约次序不同。其 Contract 允许 scale 指数 `[-126,127]`：两个解码值各为 `2^100` 时，乘积 `2^200` 不能用有限 FP32 表示；`2^-127 * 2^-23 = 2^-150` 位于 FP32 最小次正规数的一半。尾数位数不能排除指数溢出、下溢及不同缩放求值路径的差异。这证明现有文字过强，**不证明 GPU 已出现错误**，也不授权缩小 Contract。

**最小修复。** DOC3 已有“实际求值、显式转换、精度依据”的职责，无需新增精度字段。保留 GEMM/KDA 的修法；在 masked 的 Contract/Math/Arithmetic 中按来源区分 FP32 求值与数学实数恒等式，删除无成立范围的“全部精确/仅次序变化”保证，写清仍未给出的数值覆盖。不要为追求一句精确性主张补造源输入限制。P3/P5 逐段方法增加一个简短审阅要求即可：沿完整归约到最终输出核对初值、迭代域、终值和每个窄化点，声称精确时同时核对有效位与指数范围。

**验收。** 对 GEMM 枚举一个多 stage 的组号集合，确认每个 K 位置恰好覆盖；对 KDA 保留两个不同的窄化数据流正例；masked 的数值反例只检验文字主张，不作为 kernel 测试。自然语言数学关系须语义审阅，普通结构 checker 不负责证明浮点等价。两种 FMHA 的全掩码、归约/近似指数及 mixed Sparse 的 FMA/FTZ/量化修正，按原回执归入本类；本次没有另外宣称它们的全域数值正确性。

## S-02 写回与布局：align16 修法正确，但当前 eff 定义漏了 clamp

**已确认残余错误。** masked GEMM root:112 目前写 `eff = masked_m[e]-128*mb in [1,128]`，缺少 `min(128, ...)`。合法 `masked_m[e]=256, mb=0` 会推得 `eff_a=256`，与 128 行瓦片、TMA box 梯度、寄存器/TMEM 容量和相邻工作写回范围冲突。源码 [kernel.cu](../../../kernel_wiki/aka-cuda-operators-20261030/operators/gemm/grouped_masked/fp8_mxfp4_grouped_gemm_masked/candidate/kernel.cu):1181–1190 在两条编译分支都先截到 `bm`，然后向上对齐 16。这不是 kernel 缺陷，是正文定义错误。

原 P6→P3 修正“padding 永不写回”为实际 `align16(eff)` 范围是必要且正确的；现在第 119–142 行已保留最多 15 行无数值契约的写入、零 active expert 不产生工作及缓冲外不写。终审 `c8dd5678…/cb950a08-f477-47a8-ad36-62495f1119b3` 不能补足漏掉的 clamp。

**最小修复。** 在定义处修为 `eff=min(128, masked_m[e]-128*mb)`，沿 `eff_a` 的实际消费者复核 Math、Target、TMA box、存储次数和目录。第 118 行 `acc_j=y` 的对应关系还须限定到 `i_l<eff`：Contract 的 y 未对 inactive padding 定义，不能把对应关系扩到整个 `eff_a`。不另外引入 padding 字段，不改“非活跃结果无数值契约”为“永不访问”。P3 的存储/输出核对直接区分三个范围：要求有值的区域、实际读取/写入的区域、已分配且允许访问的区域。

**验收。** 有限 CPU 枚举 `masked_m=0,1,15,16,17,127,128,129,255,256,1024`，核对非空块 `1<=eff<=128`、`eff<=eff_a<=128`、写入不超 capacity、不同 mb 不重叠，尾部额外写入至多 15 行。它只验证明确索引公式。正式字节修改按 S-09 处理。

## S-03 容量按同时存活副本计；已有存储规则修正没有完全落实到现行稿

DOC5 第 355–363 行已定义 `total_size=depth×单槽字节数×持有成员数`，区分 worker 成员容量、thread 上登记的 register 名称与硬件上限。这些历史修正应保留；[上一轮审修](../../schema_audit_and_cleanup_20261119/README.md) 已纠正过寄存器副本账，不应再退回去重后的数学 shape 计物理容量。

**当前仍不一致。** FP8 FMHA [root.yaml](../../../schema_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_fp8_per_token_head/root.yaml):338–341 的 `m_run/l_run` 各记 512 B；[decode_split_kv.yaml](../../../schema_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_fp8_per_token_head/decode_split_kv.yaml):229–232 即使增加了 `slice:4`，各仍只记 256 B。源码 `candidate/fmha.cu:421,584–596` 为每线程 `float rm[MT][2], rl[MT][2]`，且 row 由 `lane>>2` 决定，四 lane 保留同一行的状态副本。root 为 `MT=2`、128 线程，各 **2048 B**；decode 为 `MT=1`、128 线程，各 **1024 B**。源码第 1192 行只让 `(lane&3)==0` 发布 m/l，也印证同值复制。原候选 `register-storage-replica-accounting` 指出问题，但 decode 加 slice 只解决 warp 分片，未计 lane 副本。

**最小修复。** DOC5 在“跨动作保留的寄存器副本”后补一句：总占用包括同时存活的独立副本，逻辑值形状仍归 Math；必要时用已有 shape 轴表示复制数。修上述 m/l 条目并复核依赖这些数字的资源说明；同次维护定向核对该候选指出的 `page_regs` 口径，不借机全量调查硬件。不要把寄存器区域字节直接等同编译器总寄存器数，也不要从 launch bound 推出实际驻留保证。

**验收。** 源数组数量 × 实际持有线程数 × 4 的 CPU 算术核对，加当前目录结构检查；无需 GPU。机制类型相同不等于副本数相同；多 CTA worker 的整组总数不能直接和单 SM 容量比较。MLA/Sparse 的 CTA/warp/head 分工、TMEM 位置及 GEMM 两次 32 B store/双 CTA 账均使用同一方法复核，不添加算子专用字符串规则。

## S-04 动作、条件域和资源寿命：保留实际保证，不靠改标签补依赖

**已确认有效修法。** Dense stream-K 的 `load:partial_chunk` 只在 `kind(j)==2` 产生；源码 `kernels.cu:5655–5687` 是普通向寄存器载入，紧接使用。改为同 role `serial` 并删除覆盖更宽 `compute:out_chunk` 域的两条边是正确修法，见终审 `2b64ba84…/a202cfe7-76fd-414f-8074-e01e6060b47b` 及当前 `pair_stream_k.yaml`。它利用真实同 lane 完成顺序，不是任意把异步动作改成同步以消除图错误。

本类还覆盖 FP8 MoE 的空 cp.async commit 组与小形状 tile decode、mixed Sparse 单槽复用的块会合、BF16 MoE 的任务序号/原子次序、absorbed MLA 的 allocation permit 与 TMEM 退休。后者当前 root:358–359、412–413 已分清“取得内存后 relinquish permit”与“最后读取后的 dealloc”，不能再合成一个释放动作。

**最小系统修复。** DOC5 已区分条件为假时动作不存在、已执行 no-op 仍有事件、serial 的作用域以及在途访问；不新增动作类别。把 P3/P5 已有边界贯通要求落到参数退化点：当某个范围变成空/单点或守卫恒假时，按当前真实配置核对动作是否仍发生、依赖两端是否共存、最后读写是否先于复用。`pipe-guard-parameter-range-coverage` 的问题成立，但普通 checker 没有能力从源码推断“这里应有一个动作”；有限图检查应由审阅者选取明确样本，不声称自动穷尽。

`handoff-single-storage-per-value` 暂不采纳新的 storage 映射/重复键方案。当前 FP8 MoE `split_k_decode.yaml:23,35,256–259` 已明确 `routing_tables` 是包含 row/tile 前缀及 totals 的实际全局聚合表示，本地分量经片上存储给 SK1，聚合体给后续 step；DOC4:305–307 允许已有数学视图与明确区域对应。把这段包含关系写清即可；候选所称“包含关系无处登记”与当前 Math 不符。若将来有不能用真实值/视图表达的不同实例，再给最小反例讨论，不为此任务增加 schema 结构。

**验收。** 复用现有 pipeline 图测试的真实条件动作样例，分别覆盖产生/不产生前驱、空提交组及同 lane serial 正例；在相关人工审阅中保留“最后访问→退休/复用”的来源坐标。GDN fence、indexed Sparse 零 tile、BF16 FMHA 同步风险属于源限制：不改 kernel，不虚构屏障，不把抽象 Design 批准说成源码已正确。已有逻辑关系足够时按现行源缺陷分流继续；须新造逻辑依赖时仍不能批准。

## S-05 Contract、when、ABI 与 workspace：现有职责正确，防止再用收窄任务掩盖缺口

Contract 定义外部任务及允许的副作用；Design 定义这份具体机制；`when` 声明它承担的域和必要调用条件；dispatch 分开记录源 selector、其他变体、收录范围和实验覆盖。DOC0:142–150、DOC1:25/271–289 与现有 P1/P3/P6 已提供该分工，无需新增覆盖字段或把 `when` 做成 selector 语言。

**确认保留。** GDN 三份 `when` 现都包含 out 基址 4 B 对齐及 stride 偶数条件；标量 out 分支只由奇 stride 触发，不能凭基址不对齐就宣称源码会回退。目录新增 vec==0 时只取消 state 向量对齐条件，保留 mixed/gate/out 的独立条件，方向正确。P6→P3 修 root、再限域改兄弟的做法尊重唯一定义，并未修改共同 Contract。未覆盖调用应如实保留，不能把它挪成非法任务输入。

absorbed MLA 的 23 个可省略 ABI 默认值与 workspace 写入白名单修正正确；`out=None` 虽在 Python 签名中出现，但候选入口明确拒绝，reference 同样需要 caller out，不能机械把所有签名默认值都当合法任务调用。P1 应把签名、构造、reference 和任务域一起核对。KDA 的 `scale=0` reference truthiness 差异留在 `Golden_reference.Gap`，不改 Contract 去迁就 reference。

**仍需小修。** KDA `prepare_chain.yaml:581` 的 Forbid 把四个未用参数一概说成 null/zero；原终审已确认状态索引实参是一个元素的占位缓冲地址。这不是 source gap。最简写法只保留“不能启用间接状态寻址，kernel 仍按 request 定 slot”的实质约束，删掉不真实且无必要的地址要求；或者准确区分占位指针、两个 null FP32 指针和零开关。`slot_start` 的 meaning 可补“cadence>0 时使用”；Math 和 Pipe 已守卫读取，因此不把这点升级成新的 nullable 机制。

**候选处置。** `inplace-output-dual-registration`、`inplace-output-naming-doc-gap` 重复历史已处理根因：DOC1:107 要求名称唯一，现有诊断已指明同时保留 Input buffer 与异名 Output result。补一个短实例和 Tolerance 按 Output 名映射的说明即可，不能删除必需 Input 或新增 alias 类型。`extent-input-name-overlap` 同理，当前 checker 区分输入对象与自动引入的 extent；在 DOC1:119 旁给出真实 scalar Input 与 `expert_count = local_num_experts` 的关系示例，不依靠改成怪异表达式躲避检查，也不无据开放同名两身份。

**验收。** 用现有 nullable/in-place/非 Tensor 回归确认两角色与默认值语义；对 GDN 采用地址/stride 的小型整数分析与原 host 分支核对。它们不证明 GPU 访问正确。FMHA 短 KV、wave-cap、生产路径；GEMM SM 数方向/工作区失败回退；MoE 队列及 finalize 分区等已发生修正归入同类审阅，不从样本通过推出整个 Contract 都被实现覆盖。

## S-06 高优先级：数学标识符规范化可合并不同对象

**现行代码已复现。** `normalization.py:283` 将公式左值的 casefold 命中标为 `related`，`_Names.add:125–129` 因此跳过不同原名的碰撞错误。Contract `Input.v` 与 Math 中新的临时量 `V` 可以被合并。本次使用完整 scalar fixture，在内存中执行公开 API：

```text
Contract: y = v + 1
Design 原文: V = v + 1; y = V
Design 候选: v = v + 1; y = v
原文 validate_documents(skip_hardware=True): PASS
候选 validate_documents(skip_hardware=True): PASS
normalizer: 仅 identifier+reference warning；再次规范化无变化
```

这里的 `V` 是公式局部临时名，不是需要小写的具名映射键。即便把它认定为风格不合规，也只能拒绝歧义或要求作者改名，不能合并数学身份。普通静态检查不读取公式含义，无法替独立审阅发现这个变更。

**调用路径与影响边界。** 正常 Runner 在 `schema_extract/tools/artifacts.py:927–975` 规划、写候选再严格重读；上述结构检查本身不会阻止改义，后续独立语义 reviewer 仍可能发现。本次未运行 Runner 写盘，也没有证明十二份交付有此腐化。absorbed MLA 的作者改用 `merge`，BF16 MoE 改局部下标，是产物绕开风险，不是框架修复。`template-identifier-case-fold-subscript` 的直接 `I/i` 反例在本次明确注册外部 I 时正确抛 `capture outer name`，不能把该候选关于所有局部作用域都绕过 guard 的解释原样采纳；它与已确认全局碰撞应合并研究，但保留复现程度的区别。

**最小修复。** 扩充现有名称身份/作用域碰撞检查，不能仅凭 casefold 相同认定两个定义 related。只有已知为同一声明的引用可以同步大小写；新增定义与现有对象拼写不同且目标撞名时，拒绝整个规范化事务。未知数学文本保留原字节，无法证明绑定时给作者诊断；不继续增加正则推测公式语义，不关闭安全的字段大小写、引号和布局规范化。

**验收。** 在现有 `test_normalization` 中加入上述 Contract-v/Math-V 反例及跨 provider/consumer 冲突；保证拒绝时原稿和兄弟文件不变。保留目前能正确拦截的 I/i、T/t shadow 正例及安全重命名/幂等测试。必要时加一项 Extract normalization transaction 的离线回归，证明歧义候选不能落地；无需真实模型。报告末尾给出本次可重现片段。

## S-07 高优先级：Tolerance 预算已经造成已知验收信息丢失

DOC1:265 与 P1:35 要求保留已知阈值、基准和局部标签；Template `text_policy.py:58` 却给每个 Tolerance **100 Unicode 字符/1 行**。本轮三个候选 ID、多个算子重复报告同一冲突，应合并成一个修复。

**现行事实。** absorbed MLA `performance/ACCEPTANCE.json:268–274` 明确给出 `atol/rtol/max_mismatch_fraction/max_rel_l2/relative_to/mode=elementwise_and_l2`；当前 root:125–126 省略 mode，且对 lse 把源码显式 `max_rel_l2: null` 写成 absent。KDA 原回执也承认 mode 因预算没有进入正文。已提供的标签丢失不能称为“来源缺口”；来源未给出的组合公式与计算/路由定义则仍是合法缺口，不能从 mode 名推导出来。

**最小修复。** 保持 Tolerance 为现有字符串映射，仅把该字段预算调整到足以保留本批已知标签及限制的档位；建议复用 **300 字符/2 行**，数值只在 `BUDGETS` 维护。这是消除规范冲突，不是取消篇幅门禁。按固定验收材料恢复相关当前输出的已知声明，区分 null、absent 与实际阈值；保留“公式/基准/路由缺失、无默认解释”限制。已有 `relative_to` 绑定校验继续保留，不新增会假装证明所有自然语言标签等价的 parser。

**验收。** 用本批六字段验收记录加缺口说明作为正例；预算超限仍报准确 count/path。独立核对对应 Output 和源声明，确认没有改阈值、没有从局部标签推断完整比较公式。SOURCE 原题 hash、比较器本体、逐输出/机制 A/B、NCU 6/10 与性能例外不在修复范围；mixed Sparse 的这些事实继续保留。

## S-08 机械约束已有大部分实现；提示与诊断应直接复用同一入口

**Version。** DOC0:37/43、Template starter、P1/P5、`artifacts._versions:523–533` 一致要求新稿从 1 开始、内容修订递增；invocation 已有 `first_draft`。GDN 首稿 2 是实际不合规，不是 checker 误判。P5 只需明确：Runner 提供的未 authored、Version=1 占位仍是首次稿件；不能因为文件已存在就加到 2；经过 P1 共享元数据同步的更高版本继续按普通修订处理。不让工具替作者随意改版本。

**字符预算。** 除 S-07 的真实冲突，已有 `text_size/check_values` 按解码后 Unicode 字符与逻辑行统一计数，helper 会批量报字段；无须另写计数器或在多处复制整表。writer 应用既有只读 helper 处理正文；handoff 是另一种非空行计数，应由 Extract 审计负责把同一校验数值送入提示及本地预检，不能套到 P5。

`execution-summary-budget-unenforced` 不应被解释成漏掉一条本来承诺执行的门禁：`--skip-hardware` 明确跳过整个硬件区及其预算，placeholder 也本来不满足普通完整硬件检查。保留这个显式豁免，不偷偷改变 flag 语义。P1 只保留执行层级、存储名称/作用域；单 launch 配置、容量和寄存器预算应归相应 Design/Realization，过长且越层的摘要按此精简，不因篇幅另开硬件调查。

**诊断改善属于次优先级。** flow plain scalar 中未加引号的逗号产生幻影键，仍应拒绝，不猜原意修 YAML；可在现有 unknown-field 诊断补有 source-style 依据的引号提示。layout 现为每文件一次 warning，既有 pending_corrections 允许 Runner 后续落地；可附首处差异行/顶层区域并标记其是否可写，避免 P3 为只读 Contract 试错。不要把 warning 变成新门禁或扩张 worker 权限。

**验收。** 复用 Version 首稿/续修/同步、预算边界及 dispatch 测试；新增测试只覆盖改变的行为。文案澄清不用测试逐字镜像。

## S-09 阶段目录、正式文字与已发布元数据

**草案与最终目录不是同一状态。** DOC0 和 Template 已正确要求每条正式变体直连当前完整基线；P2/P4 允许未来节点只有计划，P5 无共享目录写权限，P6 集中落地。三个 `p2-*-baseline-*-leads/placement` 候选反映指引不够具体，不需要放松最终直链检查：P2/P4 的“范围”保存未来基线的记录计划（普通文件名、仅计划、具体差额和依据）；已有基线的条目才进入正式变体节；P6 按实际文件/对象改成正式条目。不得伪挂 root，也不先建立空 YAML 让链接通过。

P6→P3 返回 masked/GDN 是正确使用原 run 的定点修复。KDA 在 P6 改非 root 的明确 Math 范围也符合现行权限。扩大 P6 root 写权限不是这类返工的必要修复。P2 revise 空 plan 造成的假 selection 诊断另见同伴 Extract 的 X-15；它不是已批准代表被回滚，不能照搬候选原推断。

**正式文字仍有局部遗留。** mixed Sparse dispatch:11 的“本次未运行 GPU”等过程口吻确实存在；GDN dispatch:17 也保留“无 GPU、不复现”“已记 source_gap issue”。应改成当前可用证据及其覆盖范围，保留数值、性能、硬件来源限制。不是全局删除“本次/未验证”等词：例如“来源未验证全域”的含义不能删，研究取舍也不是流程垃圾。

**修改已发布字节的最小协议。** 本批现有 `.extraction/receipt.json` 精确覆盖当时 publication proof；任何语义或纯编辑修订之后都不能继续声称它批准新字节。沿上一轮六算子的既有做法：保全旧 receipt 与原字节/出处，新增或更新 `.extraction/current.json` 绑定当前 YAML、dispatch、Version、Source、人工审阅范围、checker 与历史 receipt 指针；明确这是当前清单，不是新的 Runner/P6 receipt。即使只改 dispatch，摘要也必须刷新；语义改动递增实际修改的 YAML 版本并复核实际消费者，纯格式不制造版本增长。不要改旧 run/proof、自动 republish 或用当前清单冒充旧批准。

具体 metadata 工具接口与重新发布/恢复安全由 [Extract 审计](EXTRACT_AUDIT.md)统一设计；这里给出所需身份边界，不另建平行发布框架。当前十二项无需为了编辑文案重建运行。若用户另要求新 P6，才采用明确的新审阅流程，不能把人工修订记录改名成 P6。

**验收。** 改动前后 diff 与源/规则身份、当前 manifest 摘要、历史 receipt 原字节、真实链接和 checker 一起核验；旧证据不能覆盖新文件，合法来源缺口不能被编辑消失。此项应与语义修复同批收尾，避免又出现两种互相矛盾的“当前”。

## S-10 条件短路：一句规范澄清足够，执行器已有正确行为

`if-condition-guarded-divisor` 来自 KDA：`cadence > 0 and c % checkpoint_stride == 0` 在 cadence=0 时依赖短路。DOC5:293 允许 and/or/not，但没有明确求值顺序；现行 `pipeline/expressions.py:255–264` 已按从左到右短路，并在真实求值的除数为零时抛错。

建议 DOC5 写明 and/or 从左到右短路，未求值分支不要求其整数表达式在本次调用有定义；DOC0 仅引用这条控制语义，不泛化为 shape 可以无定义。复用或补一项 `False and 1%0`、`True or 1%0`、实际求值 `True and 1%0` 的有限求值测试即可。无需改 KDA 为计数器、强制所有守卫分母恒正或新增表达式语言。

## S-11 concurrent 与物理执行单元接管：需要明确语义取舍

**真实表达张力。** BF16 MoE [root.yaml](../../../schema_wiki/aka-cuda-operators-20261030/operators/moe/bf16_fused_moe/root.yaml):183–264 用一次 concurrent 提交，三种 Mapping 分别为 146 CTA、73 双 CTA cluster、1168 warp；GEMM EPI 在 sentinel/TMEM 退休及会合后由同一物理 warp 进入 COMBINE。源码 `candidate/kernel.cuh:1175–1194,1240–1252` 支持该接管。DOC4:277 写“互不相同的 worker，可独立推进”，DOC5 的复用说明偏向同一 worker 实例。把它直接改成 per_worker/all 又违反前三种 order 的同 Mapping 要求。当前文字能说明真实过程，但 enum 的边界需更清楚。

**推荐最小方案。** 保持四种取值，不加新 enum 或通用调度图。明确 concurrent 表示同次提交中的不同逻辑 worker 群体、没有隐含全局 step 次序或同时驻留保证；允许同一物理执行单元在显式交接/退出之后承担另一群体，必须写清映射、旧资源退休、接管动作及不能同时活跃的约束。已有数据交接/事件/Realization 承担这些关系，不能只把同一 warp 写两遍就假定安全。

这是本报告真正需要确定的 schema 语义选择：是否接受上述“逻辑群体可在物理单元上先后接管”的 concurrent 含义。推荐接受；如果产品坚持 concurrent 必须物理互斥，则应另设计受限的接管表达，不能仅靠修改当前正文几个词宣布矛盾已消失。第二轮须独立挑战该选择及其与已有检查/图边界的兼容性，不把它当已批准规范。

## 改动职责与最小执行顺序

| 顺序 | 负责位置 | 工作与完成标准 |
| --- | --- | --- |
| 1 | `schema_template/tools/normalization.py`；现有 normalization/事务测试 | 修 S-06，歧义不能落地；安全规范化保持。 |
| 2 | `text_policy.py`、DOC1/P1 少量说明；对应正文 | 修 S-07，完整保留已知验收信息；不推导未知比较器。 |
| 3 | DOC3/5、P3/P5/Reviewer 的现有段落 | 以完整归约/转换、实际访问范围、物理副本、条件动作/退休为简短核对落点；S-10 短路澄清；S-11 单独定语义。已有重复要求合并，不再建一层 checklist 或独立证书。 |
| 4 | P2/P4/P5/P6、`response.md`/helper 和现有诊断 | 首稿身份、草案基线、handoff 限制及可操作诊断各归唯一入口；P2 假诊断/证据补正等由 Extract 报告实施。 |
| 5 | 本报告明确列出的已发布文件及其真实使用者 | 定点修 eff、精确性、m/l 容量、Tolerance、KDA Forbid、过程措辞；对改动内容做独立语义复核及完整受影响目录检查。不得扩成全部算子重审或 kernel 修复。 |
| 6 | `.extraction/current.json`、discussion 当前索引/历史证据入口 | 绑定实际新字节与有限审阅证据，旧 publication receipt 不变；新旧身份没有混称。 |

可直接删减/不新增：逐算子字符串补丁；重复的尺寸/副本计数器；以新 alias/type/storage 结构解决已有表达可用的问题；把长交接复制入 dispatch；所有动作都加显式等待的“保险边”；用硬件/GPU 新验证补足提取门禁；不区分 raw 与 current 的重复 manifest。真正存在的限制与来源实测不得因清理而删除。

## 本次验证与可复现性

实际运行：`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=schema_template python -B -m unittest tools.tests.test_normalization tools.tests.test_dispatch tools.tests.test_pipeline_graph -q`，**63 tests / OK**。这些既有测试通过，同时 S-06 新反例仍成立，说明现有覆盖不能证明此类数学保真性。没有运行模型 review-eval；其现有 fixture 只有 scalar copy/矛盾 Thesis 两项，不能冒充本批语义问题的检测能力。若以后评估提示效果，可在现有 review-eval 加少数缩小的归约、精度、padding、条件动作正反例，单列模型实测结果，不把离线 mock 成绩叫 reviewer 质量。

本次 S-06 复现全部在内存，以下使用仓库原有完整 fixture，仅替换需要的字段：

```python
from pathlib import Path
from schema_template.tools import normalization, yaml_source, schema
src = Path('schema_extract/tools/tests/fixtures/review_eval/base.yaml').read_text()
src = src.replace('Input: {x:', 'Input: {v:').replace('Invariant: {x:', 'Invariant: {v:')
src = src.replace('y = x', 'y = v + 1').replace('Read x and write y', 'Read v and write y')
src = src.replace('Copy x to y', 'Compute y from v')
src = src.replace('Thesis: Retain the scalar copy without intermediate values',
                  'Thesis: Add one to the input scalar')
a, b = src.split('DESIGN:', 1)
src = a + 'DESIGN:' + b.replace('y = v + 1', 'V = v + 1\n          y = V', 1)
files = {'root': Path('/virtual/root.yaml')}
d = yaml_source.loads(src, 'root', path=files['root'])
schema.validate_documents(files, {'root': d}, skip_hardware=True)
p = normalization.normalize_documents(files, {'root': d}, skip_hardware=True)
schema.validate_documents(p.files, p.documents, skip_hardware=True)
print(p.documents['root'].value['DESIGN']['Math_reshape']['Rewrite'][0]['Math'])
# 当前输出 v = v + 1 / y = v；未调用任何写回函数。
```

fixture 的其他说明仍是合成 copy 背景，故这个实验明确检验“公开规范化+完整静态检查是否保真/阻止改义”，不把整个合成稿当作语义审阅合格设计。实际待修定义 `V=v+1; y=V` 相对 Contract 的数值关系本身明确；Source/dispatch/真实 Runner 事务没有在该实验中执行。现有源码证据、当前字节和历史回执分别承担自己的证明范围。
