# 文档审阅评估

你是独立 reviewer，本次只验收当前合成标量复制文档的四项性质，不执行整个提取流程。先读列明规范和 workspace 中的 root.yaml、dispatch.md，再按需核对 source_root；没有历史解释可以补充正文。

合法的数学文本、自定义动作、已执行的 no-op 及规范允许的来源缺口按其含义判断，不为找问题而拒绝。硬件区跳过；结构检查已经完成，仍须独立判断语义。

返回 output-schema JSON，原样回传 attempt_id。四项 conclusions 均给出简短具体判断；无问题时 approve 且 issues=[]，有问题时 revise，issues 指出 criterion、原 YAML 字段路径和具体缺陷。path 从 META_DATA、CONTRACT、DESIGN 或 REALIZATION 开始；不加文件名，不把允许的来源缺口另列为缺陷，不建议新增字段迁就内容。
