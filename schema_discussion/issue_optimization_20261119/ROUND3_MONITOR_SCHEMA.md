# 第三轮精简：monitor 与 Schema

代理：`/root/issue_r3_monitor_schema_high`，`gpt-6-astra/high`。

本组审查覆盖相对 `24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2` 的旧未提交修法和第二轮实现。仅精简三个文档；Python、测试、正式实例与证据记录均未改动。

| 文件 | 精简及理由 |
| --- | --- |
| `schema_docs/1.contract.algorithm_requirement.md` | 删除 meaning 的泛义复述及开放类型的重复前言，直接保留对象/元素原则和三个样例；dtype 表已明确不是封闭枚举。 |
| `schema_docs/5.design.tile_pipeline.md` | 删除纯存储无需单独执行单元键的重复附句；该规则仍由 DOC2 直接定义，location 继续明确引用所属单元摘要。 |
| `schema_extract/prompt/phases/06_final.md` | 合并两次 Version 更新要求；保留随获准非 root 内容修改的隐含写权限、不单列 `revision_scope` 和 Runner 校验。 |

监控实现与回归各自承担采样、竞争、身份、显示及真实中断边界，没有删去整项测试。normalizer 首差异诊断和同名对象诊断已足够紧凑，保留现状。DOC2、P1、maintenance 与 wakeup 提示保留必要规则。

保留边界：

- 采样与完成时间分离；逐 run 采样时间不倒退，旧观察不覆盖 latest，采样元数据不制造 changes；claim 仍核对 boot/start 身份，unknown 不抢占。
- 等待通知在锁外；真实 SIGINT 后同 wake 可续接和 reused。北京时间固定 UTC+8，仅白名单输出副本增加显示；epoch、ID、调度及 `inspect --json` 不改。
- normalizer 只诊断首差异；撞名提示保留 Input/Output 双角色，不推断 alias。Version 不成为新的 scope。
- 非 Tensor 对象、逻辑元素、default、None 成员与整个对象 nullable 仍分明。纯存储在所属执行单元摘要登记，不成为 worker/role，也不构成容量、编译分配或无 spill 保证。

两处实例的新读者核对：

- MLA：`sage_attn_sfs` 的 `shape: []` 与 `python_tuple` 一起表达整体对象；整数 tuple 保留四元素轴和 `python_int`，meaning 说明容器。候选、参考及输入构造支持现有默认值；未用参数不能推出固定 range。四处变动（含 Version 4→5）必要且清楚，无需再改。
- GEMM：thread 摘要的 thread-local `register` 为 root/wide 已有 Storage.location 提供名称与范围；两段 epilogue 的局部累加数组及明确 register prefetch 支持此表达，不推导无 spill。两处变动（含 Version 3→4）必要且清楚，无需再改。

实际验证：

- 与 `after_implementation` 字节快照对比本组 14 个文件，仅上表三处文档不同；代码和测试字节未变。
- 离线 Python 核对两实例全部 after manifest、原 receipt 字节及摘要、root 深层差异的精确 4/2 个字段，以及 11 份依据文件与固定 commit 的字节：全部通过。正式人工修订记录仍是作者自核，不是新的独立 P6。
- 三个精简文件的 `git diff --check`：通过。本轮仅删减文字，未重复执行未变代码的单元测试或 YAML 全目录检查；第二轮结果不冒称本轮运行。

本组编辑与验证已结束，无运行中测试。未改 AGENTS、正式 issue 状态、归档、历史 receipt 或 `.schema_extract/v2`；未运行真实 extract/resume/cancel/monitor、模型或 GPU，未 commit/push。第四轮核对精简后边界并收口。
