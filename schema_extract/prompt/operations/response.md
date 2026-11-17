# 提交与证据

按 output-schema 返回一个 schema_extract.proposal.v1 JSON，原样回传本次 attempt_id。summary 给出简短、非空的语义结论。文件摘要、实际差异和调用身份由 Runner 生成，不另写回执文件。

证据由你亲读并选择相对 path、line_start、line_end、quote_offset 和解释语义的 note；源码另选 category。行号从 1 开始，每段最多 12 行，quote_offset 是该范围原文中从 0 开始的字符偏移。不要回传 sha256、source_id 或 quote。conclusions/issues/framework_candidates 的 evidence 索引本次 source_evidence；不能挪用过程回执里的旧索引。

complete/approve 覆盖 required_conclusions 与 required_evidence；每类先覆盖对应绑定入口，再补必要调用链。reviewer 的各种决策均提供实际阅读的源码证据，artifact_evidence 覆盖 required_artifact_evidence；P5 的每条 from 链还须覆盖当前引用、各中间引用和实际定义的 raw 行。展开视图及规范化预览的位置不是证据。

四项 document_* 结论由当前 artifact_evidence 支持，所需文件均已覆盖时 evidence 可为 []；P6 的 directory 结论在最终文件集合与所选集合一致、目录有效且逐文件引用齐全时也可如此。其他阶段结论仍引用支持其语义的源码证据。

revise/blocked 的纯文档问题可用 kind=submission_format 或 schema_conformance，scope 写出当前文件名及具体位置，由该文件的 artifact_evidence 支持，issue.evidence 可为 []。源码缺口、转录、计算、数值和选择问题仍须相关源码依据；reopen 也保留源码要求。reviewer 的实际源码核查与文档问题锚点分别填写，不用无关源码充当文档缺陷依据。blocked/reopen 还须给出非空 recovery_condition。

P2/P4 的普通或维护 reviewer 独立认可当前选择或安排后，approve 的 plan 原样回传 plan_draft；需改变时按本阶段职责退修或返回，说明待改判断。P6 修改给出 revision_scope，affected_nodes 包含范围内的 YAML 节点及实际受影响者；目录使用 {path: "dispatch.md", fields: ["*"]}。worker 原样保留授权 revision_scope，approve 的 affected_nodes 覆盖已复核的 pending_rechecks。具体可写字段见阶段页。

framework_candidates 只填写 id、scope、summary、evidence。id 由小写字母数字组成的词以单个 - 或 _ 连接，同一根因沿用原 ID；scope 定位规则或工具，summary 写清问题、可复查反例、最小建议及不改的影响。reviewer 仅将独立核实的项写入此数组，沿用候选 ID 并使用本次证据；未确认项在 conclusions 说明理由。候选不改变当前 schema 语义或写权限。

checks 仅记亲自执行的命令及退出码；未执行时填 []，在 checks_note 说明交 Runner 检查。handoff 简短保留已定事项、理由、入口、新事实去向和缺口，不复制正文；P2/P3/P4 的 approve 必须提供，其余无必要交接时可为空。
