# Flow 规则评估

**建议保留 `launch + handoff` 的骨架，移出 `protocol`，补清调用边界与少数默认规则。** 当前内部 phase 组合基本够用；主要问题是重复记录、默认含义不够直接，以及外部前置只能散落在文字里。本轮提出建议，尚未修改正式规则或算子。

`flow` 负责三件事：哪些 phase 组成同一个 kernel，提交之间怎样接续，数据经由哪里交给谁。它应保留这些设计关系，不承担完整 host、stream、缓存和错误恢复程序。

## 推荐取舍

| 处理 | 建议与理由 |
| --- | --- |
| 保留 | `launch[].phase / order / start_after`。kernel 边界、同一 kernel 内的接续粒度、启动前驱各自独立；不能只靠数据交接推导。 |
| 保留 | handoff 的值、`producer / consumer / storage / ready / wait_at`。值在哪里、谁使用、何时可用、何处等待都可能改变设计。 |
| 移出 | **`launch[].protocol`**。启动许可仍由 `launch_ready` 表达；数据就绪和等待位置不变。具体协议与必要能力由已有 pipeline 动作契约及 `realization` 绑定兑现。 |
| 收紧 | `initialize` 只记录本次交接版本仍需建立的初值；入口已给定、旧值存续、首次分配清零、每次重置分别按真实含义处理。暂保留该字段，其直接样本依据弱于 `ready/wait_at`。 |
| 精简 | 相同 handoff 属性用已有 YAML anchor/alias 共享；删除已有启动路径能保证的重复 `ready/wait_at` 与专用别名。API 调用过程、录图失败分支等案例说明移出通用规则正文。 |
| 补清 | 统一计算请求的入口/出口语义；特定外部供给只约束实际需要它的提交。实例保留必要的一句前提，`realization` 只绑定机制，不首次决定依赖。 |

删除 `protocol` 后，必须同步说明 `launch_ready` 的产生条件和保证来源，保留真实的早启动能力。只删除名字并让事件自行获得保证，不构成完整修改。

## 不能再压掉的关系

| 实证 | 推翻的简化 |
| --- | --- |
| [GDN](../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:326)：chain 可先准备独立输入，随后等 prep 数据 | 启动许可、数据就绪、等待位置不能合成一个 `after`。 |
| [GEMM 重算](../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gemm/wide_recovery_chain.yaml:149)；[模板 GDN m64](../../schema_template/examples/gdn_prefill/m64_single_sequence.yaml:85)：有启动次序而没有数据 handoff | 不能从 handoff 自动生成全部启动边。 |
| [MoE direct](../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/direct.yaml:188)：一个 CTA 做完一次 census 后处理多份 unit，Census 留在片上 | `per_worker` 不能换成逐 worktile 配对、全体屏障或两个 kernel；`ready` 不能一律默认 `kernel_done`。 |
| [Sparse hidden](../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/probe_hidden_packed.yaml:299)：prefix 与 remainder 无完成依赖，最后共同交付输出 | launch 列表不能默认串行；出口汇合不能靠把两支串起来替代。 |
| [FA4 decode](../../schema_template/examples/fa4_decode/root.yaml:167)：空 split 保留有限的旧 payload，workspace 仅首次独立使用前归零 | “生产者未写”不等于“每次调用必须清零”。 |

## 外部边界：补共同含义，暂不加执行图

对于本轮关注的单次计算，建议共同约定：入口计算接在调用者前序设备工作之后；调用者后续设备工作在相关输出完成后使用它们；这些关系不要求 CPU 阻塞，也不在内部独立分支之间添加顺序。

共同约定之外，仍需保留特定前提。例如 [MLA 的 Geo](../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml:321) 完整供给后 main 才启动；MoE 的 Counts/Fill 在本次使用前建立零初值。它们可以各用一句现有设计说明表达，不拆出虚假的 host phase。

这套约定只约束所声明的计算边界，不能替外部供给或源码补同步：MLA 的 side-stream metadata 回传没有 caller fork；跨次 table 覆写、host gate 旧值和共享 scratch 仍须保留真实前提或来源限制。完整 host 阻塞、失败恢复与跨请求并发不是本次默认模型的覆盖目标。

**这里仍有结构化表达边界。** 共同约定和短说明足够保存当前样本的关键设计，但外部事件仍不可被机械寻址。若要比较任意外部接点、分批输入或分批输出，再考虑让已有 `start_after` 等关系引用具名外部就绪事件；当前证据不要求先增加完整 `boundary / stream / service` 体系。详见[边界审阅](reports/03_boundaries.md)。

## 三处默认规则应直接改清楚

1. **局部交接先认同一 worker 的存储副本，再匹配值、版本与区域。** 多个 CTA 各有一份 Census，不意味着消费者要等全部 CTA 的 Census。
2. **`wait_at` 点名实际等待位置。** 它可以引用循环外的 worker 入口动作；空 worktile 集合不删除这个动作。默认入口或已有前驱已经保证所需关系时可省略。
3. **各层约束共同成立。** 较早 `ready` 或较晚 `wait_at` 不撤销 `order/start_after`；`concurrent` 保留并发机会，不承诺同时驻留。

初始化的填写条件同步采用上表的窄定义，避免把已有初态重复变成本次工作。

## 主代理对分歧的选择

- **保留显式 `storage: global`。** 模板审阅提出纯跨 launch 时可省略，推导成立；我选择保留一眼可见的全局物化标记，避免为省一个字面量新增依赖 launch 拓扑的默认规则。
- **先用现有 alias，不增加交接组。** 十个 root 的 28 项 handoff 只有 12 组不同属性；GDN 的 8 项完全相同。已有复用能缩短表达，暂无必要改变按值登记的身份方式。数据见[重复盘点](data/handoff_repetition.json)。
- **保留四种 `order`，不增加递归组合。** 真实多 phase 同 kernel 的正例集中在 `per_worker`；其余取值有独立含义，未使用不构成删除理由。当前样本也没有迫使我们增加任意嵌套、独立 launch ID 或通用 retry 控制。

## 材料、独立审阅与核对

四个模板的 21 份 YAML 已与当前模板逐字节核对；六个提取算子使用暂停工作稿，FMHA、MoE 不以旧发布稿代替。机械盘点覆盖 198 份固定文件，语义判断使用十个 root 和报告明确列出的代表分支；候选、未完成和占位材料保留状态，不据文件数量宣称设计覆盖率。入口见[方法](METHOD.md)、[材料清单](corpus.json)和[字段盘点](data/census.md)。

五位新代理均为 `gpt-6-astra / max`，分批独立判断，最多两位同时运行；主代理核源、审阅反例并裁定分歧：

1. [字段语义与默认](reports/01_semantics.md)
2. [六算子与源代码](reports/02_six_operators.md)
3. [外部前置与调用边界](reports/03_boundaries.md)
4. [四模板与组合能力](reports/04_templates.md)
5. [最小方案的反证](reports/05_minimality.md)

主代理另行核对了 GDN 复用写法展开等价、移出 protocol 的片段只改变该字段、FMHA 的重复 ready 已由启动链保证、MoE 指定 verify 路径中的 19 个目标事件已有前驱路径。记录见 [root_checks.json](data/root_checks.json)。文件哈希、证据位置、链接与 YAML 草图检查见[验证结果](data/validation.json)；这些检查不代表整份算子迁移或 GPU 验证。
