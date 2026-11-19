# 正式目录清理独立验收

**结论：pass。** 正式目录卫生目标已满足，没有待修清理项。本次直接读取正式 README、六份 dispatch、六份当前元数据与相关规范入口，并独立计算摘要、验证迁移和链接；不是仅复述清理报告。验收覆盖最终 Sparse 补正后的实际字节。

独立检查发现的唯一遗漏是 Sparse dispatch 的“本 run 未运行 GPU、未复测”过程措辞。原清理代理已作[单段补正](sparse_followup/changes.json)：保留 L20D fixed 十用例、domain 十采样点和源码测量声明，明确这些证据不能证明四份设计及完整适用域均获独立验证，并直接陈述外部比较器缺口。补正前六份文件完整保全；差额与原编辑链衔接，25 份不应受影响的正式文件摘要不变。

验收结果：

- 六个成品目录实际仅有 **15 YAML、6 dispatch、6 `.extraction/current.json`**。15 YAML 全部逐字节匹配 GEMM/MoE/MLA/GDN 的最新 `after.json` 及 FMHA/Sparse 的既有专项 `after.json`；Version、Source、当前 manifest 一致，六个来源文件同时匹配声明的 Git 提交。
- README、dispatch 和当前元数据没有审修批次、模型角色、P2/P4 交办、旧 pending/invalid receipt、前后稿或失败检查过程残留。dispatch 的所有文字改动均可从保全原件按记录精确重建；选择、调用条件、来源测量和真实限制未被误删。MLA 当前 tiny 传递依赖说明保留，没有恢复已撤销的伪缺陷。
- 当前索引、六份元数据、语义证据与最终 checker 的路径和 SHA 一致。人工证据只声明当前 YAML 字节延续，dispatch 编辑另有精确证据链；旧 receipt 只覆盖原批准文件，未冒充当前 Runner P6。硬件和完整动态域数值验证范围也未被夸大。
- **856 个迁移文件**与映射摘要全部一致，其中 **836 个原已跟踪档案**另与 Git blob 独立核对；六份旧 receipt、四份完整人工记录均匹配清理前摘要。125 条旧路径别名有效。两份清理前已删除的 issue Markdown 仅按明确 Git 提交保全，没有声称恢复未提交版本。
- **85 个已删 `.pyc`**均有保留的源码及正确源码摘要；8 个无现存源码的旧缓存已归档。旧历史目录已移出正式树，活跃测试源码和 fixtures 保留，正式树没有非测试原始日志、diff 或字节码残留。
- **FRAMEWORK 160 = 原位不变 94 + 同 SHA 历史迁移 66**。8 个运行基线摘要不变，原六次运行共 201 次调用；monitor 仍为 disabled / generation 8，successor 的 cancel 控制保留。
- 41 份正式 issue JSON 原字节、稳定 ID 和决定均未改变；本批 15 项仍为 **12 resolved、2 declined、1 partially_adopted**。最终 **59 份活跃 Markdown 的 624 个本地链接**全部存在，包含 Template 入口；当前 JSON 指针和补正新增证据也有效。

六目录的最终只读、跳过硬件检查记录均绑定当前字节，退出 0、共 15 YAML、零规范化变更；Sparse 补正后的单目录检查也为 0。11 项 prompt/freeze/input 离线测试日志为通过，相关活动代码摘要保持不变。独立验收核对了这些结果与最终文件的对应关系，没有重复整套测试或重跑六目录 checker；最终 `git diff --check` 通过。

必要保留项：正式 issue JSON 是当前工具读取的决定台账；规范中的失败、恢复及审阅角色说明和活跃故障 fixtures 是当前机制；GDN/Sparse 的 P1/P2 和 GEMM 的迭代“本轮”属于源码语义。Git 忽略的 `.schema_extract/v2`、`.schema_extract/archive` 是原位运行证据。归档中的旧状态、相对路径和报告保持各自时点含义，由位置映射追溯。这些均不属于正式成品中的过程垃圾。源码缺陷、比较器缺口及调用前提继续有效。

验收绑定：

| 最终记录 | SHA-256 |
| --- | --- |
| [当前索引](../CURRENT_SCHEMA_INDEX.json) | `6927547cdf0bc0ead5a9e606bfeca3e3db06bf67be712fd1d15df8813da1b0e3` |
| [最终检查清单](directory_checks.json) | `fb5e6a4a66a40c310ad724553e2e69508061d139a617751d1881b938c9873a79` |
| [清理验证](verification.json) | `3419474e490e929a69dabc8470a21231b94ed664dd14a6d7468e5e6515eb1823` |
| [Sparse 补正绑定](sparse_followup/after.json) | `173c602133736f646a024e2196cf1a61eade2a477173258f538486432fd9c2c3` |

本独立验收只写本报告；未修改正式文件、索引或历史证据，未执行 extract、恢复、取消、monitor、模型或 GPU 调用，未 commit/push。算法专项审阅不在此次收尾验收的重做范围。
