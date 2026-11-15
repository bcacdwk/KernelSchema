# 候选组织试写：MoE 与 FA4 prefill

**结论：作为保存设计知识的目录，这个组织能工作；本次没有完成新 schema 迁移，也不支持“旧资料已无损压缩”的结论。** 直接读试写：[MoE](case_trials/moe/dispatch.md)、[FA4 prefill](case_trials/fa4_prefill/dispatch.md)。二者都以少量端到端代表为入口，局部新增决定写一次，源码出现过的配对不再全部登记。

这项收益主要来自改变记录单位，并非整区引用。MoE 仍有写不短的输出/版本/寿命问题；FA4 的 folded static 可降为组合与局部坐标发布，而 split 必须完整保留。完整主线内数值、资源和事件仍需闭合，来源链接不替代这项工作。

## 实际产物与计数

| 试写 | 拟完整主线条目 | 具名局部记录条目 | 每条局部记录全文出现次数 | 额外全文复述 | 新完整 YAML |
| --- | ---: | ---: | --- | ---: | ---: |
| [MoE 默认主线](case_trials/moe/dispatch.md) | 4 | 9 | 各 1 次 | 0 | 0 |
| [FA4 prefill](case_trials/fa4_prefill/dispatch.md) | 3 | 4 | 各 1 次 | 0 | 0 |

计的是本次 Markdown 的完整主线入口、`###` 局部记录及其正文，**不是已经交付的完整 schema 数量**。MoE 的 ABI 记录含必要调用条件与来源限制，不把它说成新数学算法；显式族的扩展判断和下文 Sparse 反例不计入已试写完整主线。表中“复述”指同一记录全文在这三份新 Markdown 中的重复，不计标题/链接提及，也不声称机器已证明不存在语义重复。

[复核脚本](case_trials/review_inventory.py)只读这些产物与所列源文件，输出[保存结果](case_trials/review_inventory.json)，可运行：

```sh
python schema_discussion/dispatch_method/case_trials/review_inventory.py
```

## 两例分别保存了什么

| 案例 | 完整主线负责 | 只写一次或保留为来源 | 明确没有完成的工作 |
| --- | --- | --- | --- |
| **MoE 后续默认范围**：154 项计划、96 项历史批准 | packed 融合链、split-K+独立激活链、单 kernel direct、含 device 验证输出的 packed 链 | 准备、FC2 局部分工/搬运/写回、串行接续、ABI 条件、eligibility scan、图观察；每项对象和联动在[目录](case_trials/moe/dispatch.md#一次记录的局部决定与配套条件)一处说明 | 所有前缀配对、完整异常轨迹、新规则下范围与完整 YAML 的重审；历史批准不保证当前来源没有缺陷 |
| **MoE 初始显式范围**：60 份交付 | 本次没有另交付这批主线 | last-arriver、selfprep、BF16 两族、REDF 各有[扩展判断及代表](case_trials/moe/dispatch.md#初始-60-份显式配置扩展边界)；不隐藏其与默认链的差别 | 未完成显式角色变体/强制配置的逐项保存或合法组合核对，不能算作“四主线已包含初始 60” |
| **FA4 prefill**：现有 11 份来源 | 矩形 static、folded persistent、split-KV | 同映射的单/多请求合并；LPT、静态 folded 坐标发布、alpha、joint loader 在[目录](case_trials/fa4_prefill/dispatch.md#一次记录的局部决定)各记一次 | 未重写 11 份 YAML，也未证明 source selector 之外的最大可行域或其他组合 |

选择证据的边界见 [MoE 范围核对](01_moe_evidence.md#1-版本范围和数量的可复核口径)。154 的默认实现范围与 60 的显式配置范围不同，不能拿它们合成一个压缩率。两例主动省去 selector 全树和无设计影响的 API/查找精度；未研究的机制则明确列为缺口，二者没有混作“已无损保存”。

## 试写改变了哪些判断

- **同映射合并有具体依据。** FA4 两份 persistent 的 math/pipeline 相同，schedule 仅两个说明字符串不同；[源码](https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py#L1213)让 B=1 直接退化为同一前缀映射。局部查找实现不需要设计身份。
- **“已知组合”仍可能需要一句接续说明。** folded static 没有必要为便于阅读再占一个完整主线，但其 pipeline 也不等于 root：它多出 WorkCoord 发布与首次加载等待。这项差额留在 `static_folded_coordinates`，没有误归为所有 LPT 排序的要求。
- **未被消费的冷工作不必成为设计决定。** MoE 表初始化仍可作为 ABI 冷准备事实追溯，但不在每条主干复制，也不保留独立具名设计条目。
- **重要局部决定不必变成小 schema。** alpha 只记新载荷、生产位置与所关联的旧协议；joint loader 则必须多说 token 顺序和谁归还哪类 operand。没有逐条复填完整 IO、shape、存储表和成本。

## 强耦合反例：Sparse hidden_mma

[完整来源](../../schema_wiki/archive/six-operators/revisions/2.5.0/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/probe_hidden_mma.yaml)借用 packed 与 mma 两种计算，但[实际 `dual_dispatch_hidden`](../../kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/candidate/dsv4_kernel.cu)又定义了新的调用组织，不能只放 A+B 两个链接：

| 必须贯通的决定 | 来源中的具体关系 |
| --- | --- |
| 工作与输出切分 | 整批输入、两后端均不 split；`K=SMCount-8≥1`、`GT>1`、`ceil(T/GT)≥2K`。packed 前缀写 `[0,K·GT)`，mma 余段写 `[K·GT,T)`。这些是本次代表的来源条件，不宣称最大适用域。 |
| 已执行前缀 | side probe 先等 caller 已排队输入；caller 在 host 等待观察结果前已经提交 packed 前缀。拒绝 packing 不会让它消失。 |
| 余段与接续 | 实际观察拒绝后用初始 plan 的 alt 参数，在 side stream 的 probe 之后执行余段，不重新规划前缀，也不等 packed 前缀完成。host 有界轮询所得观察不是 probe 的最终完成证明。 |
| 最终可见性 | 两段各自建立局部状态、写不重叠区域；caller 等 side join，再结合自身已排队前缀，才使后续 caller 工作看到整份输出。 |

这些关系重新决定一次调用的输出归属与 fork/join，应保留一个具体完整组织。任何单个旧区域都不同时包含两套计算及新边界，`from` 不会补齐它们。本轮仅做这个反例核对，没有第三份迁移目录。MoE REDF 的完整实例要求同样已在其扩展边界保存，不再在此复述数值和回落细节。

## 能工作到哪一步

**可以采用候选的记录方式，但不能把目录完成视作实施完成。** `when` 可保守采用已研究子域、允许重叠与空缺；不继承来源条件。root 保留唯一 Contract 与实际完整 Design，不承担运行时 fallback 或强制祖先职责。FA4 的一行 `{from: root}` 只示范已核相同的数学整区，没有片段 patch、组件库或自动组合。

仍有三项代价/缺口需要接受：

1. **MoE 的版本与在途生命周期不能再短成“缓存命中/失败回退”。** 目录已把 device 输出、host 观察和此前在途访问分别说清；不承诺消除源码竞态，不补造 drain、跨流隔离或额外输入不变条件。旧 `when` 排除本次失败，也不能证明此前访问已经结束。
2. **完整新组织仍要写全。** MoE split-K/direct/验证输出、FA4 split、Sparse hidden_mma 需要端到端实例。MoE 多条完整主线仍含相似计算正文；同名整区不相同就不能只因后半条链相似而引用。此次没有用片段共享抹平这项维护成本。
3. **本次只验证目录表达与静态证据。** 所有源 YAML 未迁移；尚未检验读者能否据局部记录实施具体组合，没有 GPU 验证、全历史组合可达性或成本对照。正式规则若仍要求互斥穷举或局部记录独立展开批准，这两份目录就不能满足它；必须先接受设计知识而非完整原调用轨迹的交付范围。

本轮直接核查了代表 YAML、MoE 准备/FC2/验证/捕获/REDF 源码、FA4 固定历史版本映射与 alpha，以及 Sparse hidden 调度。复核脚本同时检查本次本地文件链接及几项逐区对照。正式 docs、template、wiki 和提取工具均只读；没有运行提取或 GPU。
