# 提交与证据

按 output-schema 返回一个 schema_extract.proposal.v1 JSON，原样回传本次 attempt_id。summary 给出简短、非空的语义结论。文件摘要、实际差异和调用身份由 Runner 生成，不另写回执文件。

证据由你亲读并选择相对 path、line_start、line_end、quote_offset 和解释语义的 note；源码另选 category。行号从 1 开始，每段最多 12 行，quote_offset 是该范围原文中从 0 开始的字符偏移。不要回传 sha256、source_id 或 quote。conclusions/issues/framework_candidates 的 evidence 索引本次 source_evidence；不能挪用过程回执里的旧索引。

complete/approve 覆盖 required_conclusions 与 required_evidence；每类先覆盖对应绑定入口，再补必要调用链。reviewer 的各种决策均提供实际源码证据，artifact_evidence 覆盖 required_artifact_evidence；P5 的每条 from 链还须覆盖当前引用、各中间引用和实际定义的 raw 行。展开视图及规范化预览的位置不是证据。

blocked/reopen 须有具体 scope、detail、有效证据和非空 recovery_condition。checks 仅记亲自执行的命令及退出码；未执行时填 []，在 checks_note 说明交 Runner 检查。

handoff 保留已定事项、理由、入口、新事实去向和缺口；需要交接时保持简短，不复制正文。无必要内容时可为空。
