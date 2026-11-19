2026-11-18：模板自动格式化警告可读性修复

已核实并修复 `normalizer-diff-not-visible`：长字段不再只展示相同的开头，而是标出首处差异在解码后字段值中的字符位置（从 1 开始），并展示修改前后的短上下文。只改当前模板源码的警告展示；schema 字段、字符档位、符号绑定、格式化算法和规范化结果不变。

证据来自 `nvfp4_fused_moe` run `74bcac91-808a-41b2-aab4-85143935ab40`。公开只读入口：

```sh
python3 schema_extract/extract_yaml.py candidates 74bcac91-808a-41b2-aab4-85143935ab40
```

返回该候选 `status: confirmed`。worker attempt 为 `7cabb084-f7f8-406e-89de-4c07eddcd825`，独立 reviewer attempt 为 `9dabf120-87ee-42c1-8b8b-8d696b7104d1`（`decision: approve`）。两者的 `receipt.json.framework_candidates` 均记录本问题；记录保存在 `.schema_extract/v2/runs/<run>/attempts/<attempt>/`。

worker attempt 的 `original/root.yaml`、`normalized/root.yaml` 和 `normalization.json` 可直接复核：`root.CONTRACT.Algorithm_requirement.Compute[2].Math` 实际长 968 个 Unicode 字符，首处差异为第 158 字符的 `L → l`，后续还存在同字段内的大小写变化。候选中“约 1300 字符”是估计，不采用为测量值。旧警告的前后两段均为：

```text
'The declared prepared layout stores the expert weight payloads and scales out of logical row orde…'
```

修复前，[当前 yaml_format.py](../../schema_template/tools/yaml_format.py) 与本 run 的冻结文件 `.schema_extract/v2/snapshots/adcbd97155004fe0f7ad81f8bc907c0fb57e5c0d4ab26b059d304c727aff1bcc/rules/schema_template/tools/yaml_format.py` 内容完全一致，SHA-256 为 `f818e0fa740668539e9903fc4b6e6a79ac34333cef77c1bc688a2138095ff470`。旧第 673–683 行把完整字段分别交给 `shown()`，超过 100 字符就截成前 97 字符加省略号，因此该真实变化被隐藏。

最小修复共用 `_shown_change(before, after)`：短字段继续完整展示；长字段取首处差异之前最多 24 字符、从差异起最多 72 字符，截掉的两端标省略号，仍用 `repr` 显示换行及空白。真实输入的新警告为：

```text
identifier+reference: first change at character 158: '…ogical rows:\n  half_row(L) = 2*L for L < nrow/2 and 2*(L - nrow/2) + 1 for L >= nrow/2, applied …' → '…ogical rows:\n  half_row(l) = 2*l for l < nrow/2 and 2*(l - nrow/2) + 1 for l >= nrow/2, applied …'
```

一致修复范围是 `identifier+reference` 与 `period`，后者在长说明末尾删句号时有同一截断缺陷。现有 `normalization.py` 把固定字段更名提示重标为 `field`，因此自然沿用这项展示，无需新增分支。保留现有 `Diagnostic` 的路径、类别、提示及 warning 严重度；`quote`、`layout` 和其他错误诊断不涉及该前后截断，不改。提示明确只展示首处差异附近，不声称列出了字段内全部变化。

验证：

- 新增两项针对性测试，覆盖长字段开头／中部／末端的大小写变化、Unicode 字符位置，以及长说明删除句号后保留尾部空白。
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=schema_template python3 -m unittest tools.tests.test_yaml tools.tests.test_normalization tools.tests.test_cli`：68 项通过。
- 只读加载上述真实 `original/root.yaml`，调用当前 `normalize_documents` 在内存中产生候选；候选与已保存 `normalized/root.yaml` 逐字节相同，SHA-256 仍为 `b70d1fd360cad94b305ed9bda75d27e1644b1ebf947900661535a0f097960e86`，原 Document 未改变。
- 六份已冻结 `yaml_format.py` 仍为上述 `f818e0fa…` 哈希；本次没有调用真实模型、重跑整批、修改冻结 code/rules、运行状态、checkpoint、回执或产物，也未提交或推送。

当前六算子批次继续使用其冻结工具，**不会热更新**这项警告展示。当前源码调用和以后冻结的新运行才会得到新提示；本报告中的新提示来自内存复核，不是对本批历史警告或产物的改写。
