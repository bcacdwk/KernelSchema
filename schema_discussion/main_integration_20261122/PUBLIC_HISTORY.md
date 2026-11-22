# 公开历史与证据边界

本次公开的是原交付基线、已合入的 workflow 自然历史、批准方案及有限实施证据。原十二项交付的 57 个文件在 `41b7db0dd636f01a1d8bc490bc159c2d65b7b034` 保留旧字节；`6c3859e58cbc25e4094590c5e50f39ffcdb773a3` 正常合入 workflow 分支，保留其 37 个自然提交。W1、W3、W4 分别为 `1a221e4`、`d444101`、`f46b14a`。W2 与公开收口由包含本报告的 main 提交及后续 Git 交付记录绑定。

[HISTORY_PUBLICATION.json](HISTORY_PUBLICATION.json) 是唯一历史 stage allowlist；每个列入文件均经全文审阅并绑定 SHA。它也逐项记录三个旧 discussion 的 **695** 个原件的路径、SHA、大小、公开去向与理由。五个旧候选全文审阅通过：原十二项的 DELIVERY_INDEX/FINAL_REPORT，以及审计的 FINAL_PLAN/COVERAGE/FINAL_REVIEW。其余 **690** 个原件默认 local-only、原位保全；其中为本轮有限源复核读取的少量研究摘要单独标注，其余标记 `withheld_without_content_review`。没有按后缀或目录批量公开。

公开记录不包含 raw/native 模型捕获、原始思维、账户或运行账本。695 原件只按已有精确清单流式核 SHA，不为筛选公开材料展开捕获正文。全部路径元信息也经过检查，使用仓库相对路径；未发现需用稳定匿名标识替代的敏感字面量。两个大型 research JSON 和早期调试日志保留本地，必要结论写入本轮轻量报告。失败及修正的来源仍在 W1、两轮报告和执行日志中如实区分；旧研究的验收次数没有计入本轮实施测试数。

fresh clone 可以从 Git 验证原 57 文件、workflow 祖先关系、当前正文/Version、原 receipt、新 current、公开 checker/测试记录和有限复核的相互摘要。九个新 current 覆盖完整 19 YAML/9 dispatch，明确不是 Runner P6；另外三目录的 14 YAML/3 dispatch 仍匹配旧 receipt。旧 receipt 的批准对象可以从原基线提交还原，不能据此批准新字节。

fresh clone **不含** `.schema_extract/v2` 的旧 state、supervisor、账户、attempt、checkpoint 或冻结 run 规则，也不含 690 个 withheld 原件、未公开研究资料和 `/tmp` 的原测试环境/临时 fixture。公开旧文档内指向这些材料的相对链接及本机绝对路径是历史定位，并不保证 clone 存在目标。masked 的 4616 个历史索引组合只复用其已冻结结果；[source_facts](round2_d/source_facts.json) 给出公开摘要、原结果 SHA 与固定源码坐标，未冒称本轮重跑。SHA 能核对已有字节的身份，不能凭摘要还原或证明缺失材料的内容。

本轮将旧 BASELINE 中 42 个 archive 路径作为元信息排除，没有访问其目标；clone 中即使包含已有档案，也不构成本轮验证来源。其余 277 个保护路径按精确允许清单检查，变化限于授权产品/测试。12 个原 state、supervisor、12 份原 receipt 和用户 AGENTS 的保全结果见 [preservation.json](round2_d/preservation.json)。不移动、不改签或迁移历史。

CPU 验证需要具备 PyYAML/jsonschema 的 Python；workflow 还需 pytest。记录中的 `/opt`、`/tmp` 解释器是当时实际环境，clone 不携带该环境；记录显式证明本轮导入的是 main。测试使用合成 backend、临时 Store/Publication、假时钟或受控子进程。固定 Kernel Wiki 源码与 ACCEPTANCE 只支持声明的源对照：比较器定义缺口、null 含义边界、同步风险与性能例外仍保留。没有新的模型、GPU、真实 Extract/monitor 调用或 P6 认证，也不声称可由这些记录复现真实模型成本或 GPU 效果。

过程边界另有一项明确偏差：D 初次定位 AGENTS/计划文件的 `rg --files` 虽按文件名过滤，却把仓库根加入范围；未返回 archive/AKA 条目、未打开其中正文，但不能宣称零目录级探测。后续全部按精确范围操作，archive 不作为验证来源。见[过程边界记录](round2_d/process_boundaries.json)。
