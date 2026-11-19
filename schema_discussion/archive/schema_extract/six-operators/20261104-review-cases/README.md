# 六算子试用：Reviewer 机制验证材料

本目录整体归档六算子初验后用于修复和验证 Reviewer 机制的材料，记录时点为 2026-11-04 / Schema Extract 2.2.5。相关问题见[算子结果初验](../20261103-trial/OPERATOR_REVIEW.md)，验证结论见 [VALIDATION](VALIDATION.md)。

| 文件 | 内容 |
| --- | --- |
| [base.yaml](base.yaml) | 合成标量复制设计的基线 |
| [suite.json](suite.json) | 十个语义样例的变更及期待判断 |
| [reference.py](reference.py) | 合成任务的输入、实现和 reference |
| [source.md](source.md) | 来源与任务说明 |
| [VALIDATION.md](VALIDATION.md) | 当时的测试结果和真实 GDN 隔离终审 |

评测工具保留，默认 suite 路径已指向本归档；也可通过 `--suite` 明确选择其他材料。以下说明保留手动复核的方法，本次归档没有启动模型评测。

这组样例检验 [Reviewer 方法](../../../REVIEW.md)对纯文本表达的判断，不衡量真实 kernel 数值或性能。所有样例都通过普通结构及已实现关联检查，避免用未知字段等检查器本就能发现的错误代替语义评估。

`base.yaml` 是完整的合成标量设计，`suite.json` 定义独立样例及维护者期待。四个正例覆盖简洁设计、合法自定义动作、始终存在的 no-op 事件和允许保留的验收来源缺口；六个反例覆盖历史定义、未定义对象、正文取消阶段或成员、下层改写上层，以及把审阅历史写入 thesis。

从仓库根目录运行；默认沿用当前 Codex provider/model 配置，reviewer effort 为 xhigh，每例只启动一次独立只读调用，默认最多 240 秒：

```sh
python3 -B -m schema_extract.tools.review_eval --output .schema_extract/review-eval/first
```

可用 `--case ID` 限定样例，或用 `--call-seconds N` 调整有限调用预算。输出目录必须是新目录；重跑使用新目录，保留之前的失败与结果。`--model` 只在明确需要评估其他模型时指定。

运行前冻结当前规范、Reviewer 方法和来源，工具为 reference 建立真实的固定本地来源身份。每个调用只看到当前规范、当前产物及固定来源，不看到样例 ID、期待结果、其他样例或历史批准。期待数据留在调用 workspace 之外。硬件明确跳过。

`report.json` 保存全部逐例结论、期待与实际决策、评分、提示和输入身份、耗时与原始 usage；每个 attempt 保存真实 prompt、CLI events、response 和 result。反例须拒绝且定位到对应字段和问题类别，正例须通过；“全部拒绝”不能通过回归。CLI 失败与未完成不计为正确拒绝，未知 token 不按零相加。

离线测试只验证样例可解析、结构通过、评分器及审阅门禁工作；上述命令才实际调用模型。样本通过不保证未来所有审阅正确，新增真实反例时继续扩充本集。

已完成的 [2.2.5 验证记录](VALIDATION.md)与样例放在同一归档内，保留当时的真实 GDN 终审和验证范围；使用新规则重新评估时应另存新结果。
