# View 专项研究

**主代理建议：保留可选的一层 view，只保留 `slot_index / part`；完全删除 view 的 dtype 职责，放宽整槽顺序复用时的 part 填写要求。** 不引入父 storage 默认槽公式再由 view 覆盖的机制，也不另造 flat + backing/alias 体系。普通独立区域本来就能平列；多个值复用同一片区域时，一张逐值映射仍然直接。

两位全新 gpt-6-astra、max effort 代理独立调查，主代理随后核对源码、数量和反例。本轮只有研究与建议，没有修改正式 docs、算子或工具。输入与分工见 [BRIEF.md](BRIEF.md)，现行规则快照见 [inputs.json](inputs.json)；前一轮已经改名为 storage/index/event，本轮重新评估 view，不以先前保留结论为前提。

## 推荐的职责划分

| 内容                         | 放在哪里                                      | 本轮建议                                                        |
| ---------------------------- | --------------------------------------------- | --------------------------------------------------------------- |
| 区域的容量、份数、存储位置   | storage 的现有字段                            | 保留 shape/dtype/total_size；父 dtype 是容量与坐标单位。        |
| 某个值、版本使用哪个槽和区域 | view                                          | 只保留 slot_index/part，不再定义如何解释数值。                  |
| 值的数值类型和转换           | 已有数学／值定义                              | 沿用选定的 Design 类型，输入输出沿用 Contract；不重复一份类型。 |
| 必要的本地记录或物理编码     | 产生或处理它的动作契约、必要 realization 绑定 | 定义一次，所有使用者引用同一含义；不增加类型表。                |
| 不影响这些约束的地址和布局   | Coder                                         | 不要求 Agent 在设计阶段写出无关偏移、线程地址或普通打包临时量。 |

删除 view.dtype 要同时取消“省略时继承父 dtype”的解释。比如 FP8 的 P 放在按 FP32 单元计量的 TMEM 区域，P 仍按已定义的 FP8 表示读写；父 FP32 只说明容量与 part 坐标。只删 YAML 字段、保留原有继承规则，会改变含义。

## 数据支持什么

六 root 共有 **82 条 storage、88 个 view、24 处显式 view.dtype**。其中 **14 处**与已登记同名值的 dtype 直接一致；其余 10 处是本地别名、metadata 或 scale packing，不能盲目按名字推导。主代理复算见 [census.json](data/census.json)。

进一步回查表明，这些局部格式也不要求由 view 拥有：Sparse 的 build 已定义 `8192-word uint64 HashTable`；FMHA 的 E8M0 atom、MoE 的 E4M3 scale bytes 已由动作及实现绑定说明。一个 uint8 字段本身也表达不了这些编码。若某份旧稿只有 view 中留下必要格式，迁移时应把该事实补到已有对象／生产动作的定义中，不能直接丢掉。

**类型可沿用，不代表占用可机械推导。** GDN 的 Ti 是逻辑上的 64×64 BF16 矩阵，但实际只存十个 16×16 下三角 tile，物化量为 **5120 B**，不是完整矩阵的 8192 B。打包子集、padding、有效分配粒度和独立副本仍须有足够说明，不能用数学 shape × dtype 覆盖实际容量。见[语义报告 E2–E4](reports/01_semantics.md)和[形式报告 F01](reports/02_forms.md)。

## part：可以少定位置，但不能丢掉并存和覆盖关系

建议省略 part 时，将整个父槽作为该用途的**保守占用／冲突范围**；这不声称值填满、读写全部字节，也不给未用空间自动增加并发用途。

- 已有交接保证冲突读者和在途写者都已退休时，整槽可以先后承载不同大小、不同格式的值，不必另写任意的首地址和末地址。
- 若设计依赖两个值同时存在、局部读者尚在使用、或某个部分覆盖关系，则保留 part，或按现有规则把互不重叠的区域分列为 storage。
- 省字段不得靠新增全量等待来成立。关键区域关系要保留；可替换的偏移与布局由实现选择。

**可简化的真实例子：MoE/direct 的 Exchange。** 它先保存 WarpCounts，消费并完成交接后才保存 FinalRoutes。父区域是 16 个 int32 容量单元，64 B；不必要求后一个较小值从 element 0 开始。建议片段：

```yaml
Exchange:
  depth: 1
  location: SMEM
  shape: [element: 16]
  dtype: int32
  total_size: 64
  view:
    WarpCounts: {}
    FinalRoutes: {}
```

两个本地对象的格式、有效数量、读写绑定和原有交接仍须明确。Sparse root 的 Q_ring 搬入寄存器并 join 后让出 ring，以及所有 Gather 使用结束后的 PartialStage，也有可省具体 part 的同类依据。

**本轮不能直接省的例子：MLA 的 P 子区。** 各 warp 的本地“读 S 后写 P”不构成全组退休。`s_free` 在每个 warp 读完后 arrive，但 P 写者不等待它；`p_ready` 在写 P 之后发布。主代理提出定向反例后，语义代理撤回了最初的整槽草图。主建议仅删 dtype，保留原 part：

```yaml
ScoreProbability:
  depth: 3
  location: TMEM
  shape: [row: Bq, column: Bk]
  dtype: fp32
  total_size: "3*Bq*Bk*4"
  view:
    "S(n)": {slot_index: "n%3"}
    "P(n)": {slot_index: "n%3", part: {column: [32, 64]}}
```

该片段仍依赖原有行分工、表示与复用保证，并保留源码已知限制。它不证明 column 32 永远不能替换；本轮只是没有证明省掉这个区域约束后仍保留相同并行关系。具体反例与修订见[语义报告 E1](reports/01_semantics.md)，另一代理独立采用同样的保守结论，见[形式报告 F03–F04](reports/02_forms.md)。

## 为什么不把所有属性提到 storage

父 storage 可以有一个容量，却有多个值／版本各自的槽和子区。把所有逐值属性提升，会出现父默认与子覆盖，或按值分别列出的 slot/part 平行表；需要跟踪的关系仍然存在。

六 root 的 38 处 slot_index 中，只有三个父区域出现相同公式的重复：GDN 的九种输入共用 c%NSTAGE，GEMM 的三个视图共用 n%3，MLA S/P 共用 n%3。提升最多省掉这三组的 11 次公式复写，但 GDN 无 c 形参的 FinalState 仍要固定为 0。GEMM/mma_sync 也同时有 p%3 与 Output(t)=0；父模板还需要定义参数绑定、是否求值以及覆盖优先级。

语义代理认为这份节省值得引入公共模板；形式代理与主代理认为当前目标下不值得。**最终主建议保留一个填写位置：映射写在对应 view 上。** 选择哪个版本轴、是否跨工作持续、哪些代次共用槽仍是设计事实；槽的统一编号重命名和真实地址可以留给实现。不要为了统一公式把局部 p 硬换成连续 n，或丢掉 terminal/空 copy-group 的身份。

## 哪些情况可以直接平列

单槽、同名的普通单值沿用免 view 规则；互不重叠的独立设计区域也可以直接平列。FMHA 的 ScoreHalf、scale 区、NumState、Prob8 本来不相交，若不需要表达跨区覆盖，可以分列为独立 storage，让实现安排合法原点并合并申请。

但它会重复容量头，不一定减少总文字；若后来有一个用途跨越多个区，父区域加 view 更容易表达。Sparse 的输出可能同时覆盖两个 Gather 槽，而两个槽彼此不重叠，这不是一个简单的“同址”等价类。flat + shared_with/backing 仍须补回容量只算一次、跨区相交和引用身份，因而不作为另一套正式写法。

格式只说明一次，也不意味着存储副本只有一份。MLA 的 TMEM/寄存器副本、GEMM 的双槽 DrainRegister 仍有独立容量和生命周期。完整同例比较与填写量见[形式报告](reports/02_forms.md)。

## 建议规则的最小范围

1. view 只定义版本到存储区域的绑定，删除 dtype 字段及父类型继承。
2. 格式沿用已有值／动作／必要原语定义；只在影响设计时说明本地编码，不让各消费者重新选择。
3. 保留逐 view 的 slot_index；不新增父默认、覆盖表或另一套 backing 引用。
4. part 按设计需要填写。整槽顺序复用可省精确位置，局部并存和部分覆盖仍须明确；不能靠增加等待换取省略。
5. 保留容量、物化足迹、有效粒度、独立副本与真实退休条件；布局细节不按源码逐项搬入。

两位代理都支持前两项及有条件的 part 简化，对公共槽模板保留上述分歧。这里没有证明所有旧 view 都可自动迁移；特别是本地格式来源和部分区域的并发安全仍须逐项核对。本轮完成的是规则建议，现行规范尚未改变。

## 报告与核验

- [01 内存语义、必要信息与早期决策](reports/01_semantics.md)；[数据](data/01_semantics.json)。
- [02 删除、提升、扁平化的同例比较](reports/02_forms.md)；[数据](data/02_forms.json)。
- [主代理复算](data/census.json)、[综合判定](data/synthesis.json)、[输入和引用核验](data/validation.json)。

两份报告均覆盖六 root 和四个有针对性的 approved 分支；两者选择同一组分支，但分别独立阅读相关源段、形成初始判断。后续仅对 MLA 省略 part 的具体反例做定向复核。源码身份沿用父研究的十八份固定副本；没有 GPU 执行，也没有新增源码副本。

复核入口：

```bash
python schema_discussion/phase_grammar/view_review/scripts/census.py
python schema_discussion/phase_grammar/view_review/scripts/01_build_data.py
python schema_discussion/phase_grammar/view_review/scripts/02_forms.py
python schema_discussion/phase_grammar/view_review/scripts/check_evidence.py --write-report
```

这些检查验证数据来源、引用和有限算例，不将草图认证为完整新规范实例。
