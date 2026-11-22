# 本轮问题与处理回顾

范围仅为本轮十二项 extract。以下按类型合并重复事件，不是独立 issue 数量或逐条审稿意见统计。回顾依据为本轮审阅回执、恢复记录、巡检记录及最终交付证明；没有开展新的源码审计或修改。

本轮改动集中在 schema/dispatch、生成回执及 discussion 内的取证脚本。未修改 Extract Python/执行器或 kernel，未做 repair-runtime 迁移，未重建失败 run、追加调用额度或放宽门禁。最终 12/12 发布、33 个设计通过原门禁；来源缺陷和流程改进观察仍应单独保留。

## 已发生的 schema 内容修正

| 类型 | 代表问题与已采用的修正 |
| --- | --- |
| 数学与精度链 | dense GEMM 的 weight_band_mma 漏 K-stage 循环、公式仅覆盖前 256 元素，补为全 K 归约。KDA prepare_chain 错把 kr 写成经 ki 的 BF16 舍入，改用 ki_f32 并分别窄化。两种 FMHA 修正全掩码 softmax、归约及近似指数表述；混合 Sparse 区分 FMA、exp2/FTZ 和量化舍入路径。 |
| 布局、写回与容量 | masked GEMM 原称 padding 永不写回，改为实际 align16(eff) 写回范围，最多覆盖 15 个无数值契约的非活跃行。修正 MLA/Sparse 的 CTA、warp、head 分工及寄存器/TMEM 位置，FMHA lse 输出归属，GEMM 两次 32B 存储及双 CTA 容量账。 |
| 同步、动作与资源寿命 | dense GEMM stream-K 补实际 partial load→compute 完成保证，将存在的 load 标为 serial，并删除覆盖不存在动作的依赖边。FP8 MoE 补 cp.async 空提交组和小形状瓦片解码；混合 Sparse 补单槽复用的块级会合；BF16 MoE 修正序号、原子分配顺序和同步表述；MLA 修正 TMEM relinquish 时机。 |
| 调度与适用域 | GDN 三份 when 补 out 基址 4B 对齐，并补 vec==0 标量访问变体。FMHA 修正短 KV 的 n_split=1、wave-cap 和真实生产路径条件；GEMM 修正 SM 数方向及工作区失败回退范围；MoE 修正队列/取票守卫和 finalize 分区。其他来源条件与状态池覆盖按实际实现收紧。 |
| ABI 与最终文档 | absorbed MLA 补可省略参数的 23 个默认值，修正 workspace 可写与其余存储不变的矛盾；KDA 明示 scale=0 的参考实现差异。目录中的“仅计划”描述改为真实 YAML 链接和变体记录，并修正 Require/Prefer 强度、实测计数及失真源码表述。 |

两次明显的 P6→P3 返回分别涉及 masked GEMM 写回范围与 GDN out 对齐条件，均在原 run 内修订定义，没有重启整个提取。

关键内容证据：

- [GEMM 全 K 归约修订回执](../../.schema_extract/v2/runs/2b64ba84-64b3-4d10-86fe-65c351ce571c/attempts/a4939137-59cf-4702-a771-bd1067041737/receipt.json)
- [KDA 精度链最终核验](../../.schema_extract/v2/runs/bda2a4ba-47f7-4ae2-8a9b-9e0081c41cf9/attempts/5ab1ac5e-582b-4ac7-87c6-8194434b6672/receipt.json)
- [masked GEMM 修订确认](../../.schema_extract/v2/runs/c8dd5678-2935-4efa-943a-02e848d03611/attempts/cb950a08-f477-47a8-ad36-62495f1119b3/receipt.json)
- [stream-K 最终修法](../../.schema_extract/v2/runs/2b64ba84-64b3-4d10-86fe-65c351ce571c/attempts/a202cfe7-76fd-414f-8074-e01e6060b47b/receipt.json)
- [GDN 条件、变体与来源缺陷](../../.schema_extract/v2/runs/e0311f19-50e5-4760-8331-5805dbb23a09/attempts/68ba856e-0bf6-4e90-b80b-bf8fb1059166/receipt.json)

## 流程、回执与辅助工具问题

| 类型 | 本轮处理及状态 |
| --- | --- |
| 输出目录已有空占位 | 启动前确认仅有空 .gitkeep 后精确移除占位与空叶目录，随后正常注册输出。 |
| CLI 模型名提示 | unrecognized_model 提示经原生交互核实未阻断执行，沿用配置。未修改模型；不能据此额外证明服务端上下文容量或 effort 的实际落实。 |
| 回执入口类别遗漏 | 同一绑定入口仅有 host 或 implementation 类别，另一类别仅指向下游 CUDA；补齐对应绑定入口引用。属于回执缺项，不是验证器误判。 |
| 证据坐标错误 | 行号越界、source_evidence 索引越界、quote_offset 无效；由原 worker/reviewer 补正后重新核验。 |
| 稿件机械规则 | dispatch 缺基线直链、GDN 首稿 Version 写成 2、多处 YAML 文字超字符/行数；分别补链接、首稿改为 1、压缩说明后通过门禁。 |
| handoff 超长 | P2/P3/P4 approve 的短交接超过 1200 字符或 20 非空行，模型压缩后仍有计数失准；根据新结果有限续接直至合规。该限制不能套到 P5。具体回执已修，提示与计数辅助机制未改。 |
| 累计补正额度触顶 | 阶段内累计补正已用完，但调用预算仍有余额；核实终止与计数后普通 resume，或对 P5 仅恢复失败 scope，保留健康兄弟及所有已用次数。累计策略、前置错误遮蔽后置错误的行为未改。 |
| 控制入口竞争 | 一次 busy/committed=false 的控制请求未提交；确认无状态或调用变化后串行重放，仅新增一次真实模型调用。未改控制机制。 |
| 状态时效与旧到期记录 | 对被更新观察超越的采样重新 inspect；KDA stale summary 用 status --full 核权威状态。重启 monitor 继承旧 due 导致首条立即提醒，按实际 ack/scan 登记；完成后关闭。缓存与旧 due 行为未改。 |
| 取证脚本自身错误 | CLI 文本误按 JSON、参数位置、摘要值类型、proposal 与规范化 receipt 的比较、过早读取未结算 usage 等问题，按实际格式及时序修正。仅影响检查整理，没有修改 Extract 执行器或增加模型调用。 |

对应原始过程见 [monitoring](monitoring/)、[recovery](recovery/)；[GEMM](recovery/gemm_p5_split_k_anchor/README.md) 和 [GDN](recovery/gdn_p5_warp_row_version/README.md) 恢复报告也记录了辅助脚本的调整。

## 已发现但本轮没有修复的来源风险与观察

- 源码审阅保留了 GDN fused 的每线程 fence 到 tid0 arrival 之间缺块内聚合、indexed Sparse 非 split 零 tile 路径的屏障失配，以及 BF16 FMHA 的 warp 通知汇合、split_range 潜在分歧、fused 计数跨调用状态前提。它们被记为来源风险或成立条件，没有修改 kernel，也未通过本轮 GPU 运行复现或消除这些风险。
- [indexed Sparse 的正式边界说明](../../schema_wiki/aka-cuda-operators-20261030/operators/sparse_attention/paged_indexed_latent_fp8/dispatch.md)与[BF16 FMHA 的最终回执](../../.schema_extract/v2/runs/e9af332e-dbf4-476b-8283-a88e3cd3cd98/attempts/fa98a491-7fa8-4c6c-8117-07c0ee7d5ec1/receipt.json)保留相应证据。
- 混合精度 Sparse 的原题验证回执/最终题目哈希、比较器、逐输出及机制 A/B 资料仍有缺口；NCU 仅 6/10 达标，shape 0/1/3/9 为用户性能豁免。本轮如实记录，没有补齐来源资料或把它改成全域验证通过。
- 部分原提示未明示 handoff 数值限制，累计补正策略与逐项诊断会带来额外回合，KDA summary 持续 stale；均保留为后续流程优化观察。
- mixed Sparse dispatch 的“本次未运行 GPU”等过程措辞仅记录待清理，没有在发布后改写已经核验的正式字节。

上述来源及后续观察详见 [最终交付报告](FINAL_REPORT.md)。十二项提取交付完成，不等于这些来源问题和通用机制已经全部修复。
