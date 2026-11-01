# 题库约定

按计算功能分类，题目 ID 保持稳定。每题包含 `candidate/`、`problem/`、`performance/` 和一份简短 README。

| 内容 | 唯一入口／职责 |
|---|---|
| 实现 | `candidate/solution.py:Model.forward`；复用整个 candidate 目录。 |
| 接口清单 | `MANIFEST.json` 的 `interface`：构造、forward、实际别名目标、`_make_inputs`、可选 `prepare_model`。参数照源码登记；构造为 null 表示继承。 |
| 题目 | problem 中的 metadata、input、reference、shape_cases、shape_range、solution、roofline 七个文件；solution 是生产对照。 |
| 性能 | `performance/SUMMARY.json` 提供统一摘要；原始记录、样本和证书保留原样。 |
| 文档 | 首页→分类索引→逐题 README；固定记录来源、入口、初始化、题目、范围、依赖、性能、状态、补充资料。 |
| 可选资料 | PRECISION、VARIANTS、USER_ACCEPTANCE、LICENSES 按来源保留；缺少独立许可文件需说明。 |

接口签名含 `self`，别名必须解析。存在 `prepare_model` 时按 input.py 钩子初始化；不同算子的参数及写入／返回语义以各自合同为准。

性能摘要统一使用 `ncu_kernel_sum` 或 `cuda_event_forward`，时间单位 `us`。`primary_measurement` 指向当前采用的结果；原有状态和人工性能例外如实保留。

每组 measurement 记录协议、环境、对照身份、统计方法及逐 shape 的 candidate_us、baseline_us、candidate_over_baseline、ci95、by_order、status。比值均为候选／对照，越小越快；配对中位比值不能用两个耗时中位数相除替代。跨作业平衡结果只保留比值，绝对耗时为 null。

null 表示数据未记录／不适用。无生产对照标为 `baseline_unsupported`，不能填 0 或当作通过。不同 metric、环境和对照身份的数字分别解释；摘要不生成新的验收结论。

摘要的 source_records 保存原记录路径及哈希；candidate_manifest_sha256 是 MANIFEST 中 candidate_files 按 UTF-8、排序键、紧凑 JSON 序列化后的 SHA-256。

源码、题目和原始性能证据须保持哈希对应；修改后需要相应的新验收。`python3 tools/refresh_metadata.py` 仅更新派生清单／文档／摘要，不重绑已改源码；随后运行 `python3 tools/verify.py` 和 `python3 tools/test_metadata.py`。

三题待核验情况见 [备注](PROBLEM_NOTES.md)。旧四题的生产用例／历史解整理和 QSA 扩充尚未实施，当前仍为 44 题。
