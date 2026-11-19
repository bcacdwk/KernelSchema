# 验证状态与性能证据

[首页](README.md) · [总索引](CATALOG.md) · [按状态检索](FACETS.md)

本 Wiki 保留原库的验收结论、范围与例外。导入做了文件哈希、接口、链接和元数据检查，未重新执行 GPU 正确性或性能评测；“SOTA 实现库”不额外表示所有 shape、环境或设备上的最优性证明。

## 三个维度分别看

| 维度 | 含义 | 当前数量 / 依据 |
| --- | --- | --- |
| 实现来源 | 当前 `candidate/` 选自 AKA 包还是参考实现包 | 20 AKA、24 REFERENCE；[MANIFEST](aka-cuda-operators-20261030/MANIFEST.json) |
| 候选验收 | 当前选定版本的认证、采样验收或人工例外 | 20 `certified`、22 `passed_sampled`、2 `accepted_by_user_with_performance_waiver`；逐题 SUMMARY |
| 题目绑定 | 当前 problem 是否与验收使用的合同和输入对应 | 20 全部七文件核对、21 核心绑定核对、3 待最终回执；[PROBLEM_AUDIT](aka-cuda-operators-20261030/PROBLEM_AUDIT.json) |

`REFERENCE` 也是当前选定候选的来源标签，不是让调用者改用 `problem/reference.py`。`candidate/solution.py` 是所选实现入口，`problem/solution.py` 是生产对照，`problem/reference.py` 是正确性参考。

`problem/metadata.json.validation` 还保留历史原生实现的验证状态，可能出现 `numerical_failure` 或 `native_unsupported`。它不自动代表当前候选失败，也不能被抹去；判断当前候选需要查其 `performance/SUMMARY.json` 与绑定的认证／验收文件。三题缺少最新回执的限制仍单独成立。

## 三题待最终回执核验

| 算子 | 当前候选状态 | 尚缺什么 |
| --- | --- | --- |
| [fmha_nvfp4_qkpv](aka-cuda-operators-20261030/operators/fmha/fmha_nvfp4_qkpv/README.md) | 人工接受性能例外 | 最新题目验证回执，尚不能完成最终题目哈希确认 |
| [gdn_prefill_bf16_state_fp32](aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_fp32/README.md) | 采样验收通过 | 同上 |
| [paged_latent_mxfp8_nvfp4](aka-cuda-operators-20261030/operators/sparse_attention/paged_latent_mxfp8_nvfp4/README.md) | 人工接受性能例外 | 同上 |

这些题的 problem 已配齐、接口与历史题库一致，原库按既有决定带标记保留。其余 21 个参考实现的核对覆盖输入、reference、合同与固定用例；辅助 `shape_range.json` 和 `roofline.json` 未记录独立的最新版验收哈希。20 个 AKA 题目的七个 problem 文件已逐字节核验。详见原始 [备注](aka-cuda-operators-20261030/PROBLEM_NOTES.md)与审计文件。

## 两项人工性能例外

| 算子 | 原始 NCU 门槛结果 | 保留的超限 shape |
| --- | --- | --- |
| [fmha_nvfp4_qkpv](aka-cuda-operators-20261030/operators/fmha/fmha_nvfp4_qkpv/USER_ACCEPTANCE.md) | 9 / 10 达到门槛 | 8 |
| [paged_latent_mxfp8_nvfp4](aka-cuda-operators-20261030/operators/sparse_attention/paged_latent_mxfp8_nvfp4/USER_ACCEPTANCE.md) | 6 / 10 达到门槛 | 0、1、3、9 |

两者保留为人工接受性能例外；不将这些超限记录改写为自动通过。接受背景与对应源码身份保留在各自 `USER_ACCEPTANCE.md`、`ACCEPTANCE.json` 和统一摘要中。

## 两种主要计时口径

| 主测量 | 算子数 | 当前主要对照 | 如何解读 |
| --- | ---: | --- | --- |
| `ncu_kernel_sum` | 25 | 生产实现 | GPU kernel 耗时之和；CPU 调度、编译、copy-engine DMA 等不计入 kernel sum，逐题边界另查记录 |
| `cuda_event_forward` | 19 | 历史候选实现 | 完整 forward 的 event 协议；主要采用配对、顺序平衡比值，不能直接当作对生产实现的加速比 |

这是每题 `primary_measurement` 的分类。整个快照共有 **80 组测量**，一题可以保留多组历史／辅助结果；优先读主测量，再追溯相关记录。

- 时间单位是 µs；`candidate_over_baseline` 是候选／对照，越小越快。对照是谁必须看 `comparison`，不能只读数字。
- `paired_ratio` 和 `balanced_order_ratio` 按原配对与分层协议计算；不能用两个总体耗时中位数相除替代。
- 顺序平衡结果可能只有比值、绝对耗时为 null。不能凭比值补出未测的绝对耗时，也不能跨 GPU 合并绝对耗时。
- `null` 表示未记录或不适用。`baseline_unsupported` 是生产对照不支持，既不是零耗时，也不是对照比较通过。
- [KDA raw prefill](aka-cuda-operators-20261030/operators/linear_attention/kda_raw_prefill_state_bf16/README.md)的固定用例 0、2 的生产对照不支持；保留其布局／边界说明和缺测标记。
- 固定用例与动态支持域不是同一集合。不能把已有采样验收推广到 `shape_range.json` 中所有参数的笛卡尔积；头数、布局、状态、形状往往还有跨字段约束。
- `metadata.json` 中标为 `measured: false` 的评分归一化值不是测量结果。真实性能从 SUMMARY 及其引用的原始记录读取。

各题的 `PRECISION.md`、`VARIANTS.md`、`BASELINE_BOUNDARY.json`（若有）解释量化、变体或生产对照的限制。不同输入 dtype、scale 布局、输出别名和预处理边界不宜直接横比。本 Wiki 因此提供检索维度，不生成跨题性能排行榜。

## 导入完整性与复现入口

| 检查内容 | 结果 |
| --- | --- |
| 原清单文件 + CHECKSUMS 文件 | 1,266 个文件，51,906,145 字节，与提供目录逐字节一致 |
| 候选源码 | 429 个文件，保留原哈希 |
| 题目文件 | 44 × 7 = 308 个，保留原哈希 |
| 接口与性能摘要 | 44 个接口、44 份 SUMMARY、80 组测量通过原库检查 |
| Python 静态语法 | 原库 201 个 Python 文件通过检查 |
| 元数据转换测试 | 原库 8 项测试通过 |
| 新增 Wiki 索引 | 检查机器索引、Markdown 索引、路径与原始哈希一致性 |
| GPU 重新评测 | 未执行；继承原包记录及其边界 |

运行命令见[首页](README.md#验证与维护)。原库未附通用 GPU 评测执行器；实际 GPU 复现还需合同要求的构建环境、依赖与评测 Harness。不能把导入校验成功描述为本次重新测得了数值或性能结果。

[IMPORT.json](IMPORT.json)固定本次来源与清单哈希。原 MANIFEST 中的 archive 路径、验收记录中的远端 job 路径属于历史出处；原始压缩包未包含在用户提供的目录中，不是本 Wiki 的本地文件入口。当前快照已经包含其清单列出的源码、题目和原始性能文件。
