# Contract 名称冲突诊断

已通过公开只读 `candidates 74bcac91-808a-41b2-aab4-85143935ab40` 核实 `inplace-output-input-name-diagnostic` 为 `confirmed`；确认 reviewer attempt 为 `2e075c1d-1ee7-4eea-98a4-a77f80442d26`。依据现行 [DOC1](../../schema_docs/1.contract.algorithm_requirement.md):43–44、107，调用者提供的目的地与写回结果分别登记为 Input、Output，契约名称保持唯一。

[symbols.py](../../schema_template/tools/rules/symbols.py) 的冲突错误现在保留当前定义路径，并给出重复名称和上一处定义的完整路径。Input/Output 撞名时，现有 `Fix` 提示要求保留两种角色、分别命名；若 Input 是写回目的地，在 `Output.meaning` 和既有 Compute 中声明关系。复用现有路径表与 `Diagnostic.hint`，不依据 shape/dtype 推断原地关系，不自动删除、合并或改名，不新增字段或改变名称合法性。

定向验证通过：10 种 Constant/Input/Output/Compute.Symbol 间或 Compute 条目间的撞名组合；4 种 Input/Output shape/dtype 相同或不同组合；三重撞名、异名写回正例及诊断渲染。验证前后输入对象一致。另运行 4 项现有 Contract/只读 API 单测，全部通过；`git diff --check` 通过。

修复用于当前源码及后续冻结的新 run。本批相关 I/O 登记产物已维护，本次不重跑本批，不修改其 frozen 工具/规则、运行状态或产物；未调用模型、提取或 GPU，也未提交或推送。
