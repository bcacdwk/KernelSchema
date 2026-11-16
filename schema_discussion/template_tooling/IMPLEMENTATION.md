# Template 实施记录

已完成 [PLAN](PLAN.md) 的阶段 0–5，落实 DOC0–6、starter、工具与必要流程说明的当前格式；本次已实现批准的[最新文本补充表](TEXT_RULES_PROPOSAL.md)，并完成新的 [xhigh 独立审阅与修正](TEXT_RULES_REVIEW.md)。阶段 6 的四案例整理单独延期；`schema_template/examples/` 整个目录，包括 YAML、PNG、README 与 dispatch，保持未改。`schema_wiki`、`kernel_wiki`、Extract 脚本及运行状态未改。

## 已交付

命名以 [FIELD_MAP](FIELD_MAP.md) 的逐路径分类为准：顶层全大写，结构字段首字母大写，记录属性小写，step/role 全大写，自取名称小写，动作语法与实际外部标识保留。DOC0–6 与 starter 已统一结构轮廓、字段路径和引用；starter 仍是 `version: 1` 的待填写骨架。Ledger 空值、range/default 简短表达式、必要引号与简单记录排版已对齐。

工具按 raw 读取、结构检查、引用展开和有效关联分层。三区 `from` 合并检查节点 DAG，在使用节点上下文绑定；来源保留实际 consumer→provider 链。raw、effective 和不同节点的展开副本不共享可变对象，来源身份只包含 root、自身与三个实际内容链。

共享 `pipeline_syntax` 覆盖有序 `for/if`、八个动作族、成员和显式事件。静态规则核对当前结构、声明/属性引用、提交、启动事件、交接、成员与可确定的存储关系。有限图按实际样本建立动作 issue→done、pipe 发起/serial 推进、跨 noun/verb/worktile 的 ordered 完成及四类显式时点边；不推导同名数据边，不证明跨 step 活性。

`text_policy.py` 集中实现 [PLAN](PLAN.md) 的路径预算、逻辑行计数和源样式 warning，预算只计 raw 定义。`yaml_format.py` 直接用源节点及同一标量 resolver 产生候选，保留精确类型、Decimal 原文、anchor 名、alias/merge 与 `from`，回读核对节点关系和幂等。唯一允许的值差异是已核准普通说明路径的终末一个 `.` / `。`，其余字符、空格和换行不变；缩写、引文、数学、路径或引用等不确定情形保留，共享 scalar 有非 prose 或被跳过硬件用途时保留并提示。原 Document 与检查 API 不变，默认 CLI 写回后给修正 warning，并按清理后的实际值检查预算。真实注释报错并拒绝丢弃，超限正文不截断。

最新补充已落实到既有路径与字段规则：自取单名、真实类型/硬件身份、结构化数值与固定值字符串、Stack、retention、内部语义引用和材料摘要分别检查。Constant 的原标量类型保持，Math Symbol 的数组逐字符串叶计；原 `300/2` 与 Contract Input `range/default` 的 `150/1` 保留。纯公式中无法机器识别的名称仍按约定审阅；当前分支预算集中列在[工具说明](../../schema_template/tools/README.md#文字与形式)。

调度排版补齐 Mapping.Worker/Placement、Worktile_order、Preassigned、Refill 和 exit。单机制映射及单个排列记录与父字段合行，多机制和多记录逐条书写，by 数组保留；exit 仅在 1–3 个短项且无逻辑换行时行内。折叠块标量转行内仍保留解码值及必要引号，格式器不为压行改变正文。DOC0–6 对齐名称分类、最小字符形式、引用/retention 和必要示例；starter 已符合本次相关布局，无需补造字段。

唯一日常入口 `check_yaml.py` 默认检查同目录全部 YAML 与 dispatch，并排版所选文件；`--all` 扩展排版和请求的图对象，`--read-only` 报格式差异而不写文件。解析/结构失败先停止写回，预算或关联失败可以伴随已完成的保真排版。原文复核、原子替换、权限保留与未变化不写已接通；需改写的文件 symlink 明确拒绝并保留。

`--graph check/png` 在同次请求中增加有限展开。所有所选分析完成后才发布 PNG；源漂移、缺样本、未知条件或失败不删除旧图，完整环可生成标红诊断图。PNG 保存最终 raw 摘要、来源链、effective identity、样本及实际范围。旧独立图入口与未启用的目录硬件策略已删除。

`rules/dispatch.py` 只检查 common 的必需章节、当前完整文件链接覆盖/存在、直接基线链接及十份上限，支持常用行内/引用式链接并忽略 fenced 模板；设计价值、已有对象含义与 prose 仍由审阅核对。目录不补 YAML 定义，也不作为隐式 patch。

## 只读接口与 Extract 边界

`yaml_source.load/loads` 保留 `Document` 的值、位置及源摘要，新增源节点、anchor、occurrence 和 comment 信息。`rules.validate_algorithm` 仍为 Contract-only，只读并包含相应文本预算；`Diagnostic.severity='warning'` 不表示失败。

`schema.validate_file/directory/resolved_file` 检查当前完整任务、dispatch 与可确定的本地仓库定位；`validate_documents(files, documents, ...)` 只使用给定最终快照，零文件 I/O，目录及本地资料由独立文件 hook 检查。仅输入文件所属仓库与 repo/ref 准确匹配时核对该提交中的路径，尖括号占位保留，入口语义交原件审阅；不搜索其他仓库或联网。`validate_standalone_file` 只读校验单文件及本地定位，不读兄弟 YAML 或 dispatch。`resolve_directory` 与 `effective_yaml` 用于可展开草稿及来源视图；`--show-effective` 明示 `validation: not_run`。

Extract 的后续外部检查需显式调用 `--read-only --skip-hardware`，或使用只读 Python API。Template 已对齐不表示 Runner 的阶段权限、计划、回执、冻结运行、恢复或发布已对齐；这些均未修改，见 [EXECUTOR_ALIGNMENT](../../schema_extract/EXECUTOR_ALIGNMENT.md)。

## 维护验证

本次最终审阅修正后完整合成套件 **82 项全部通过**（5.328 秒），共新增 12 个回归方法；38 个 Python 文件通过 Python 3.10 语法检查，`git diff --check` 通过。最终全套命令：

```sh
/opt/venv/python3.13.12/bin/python3 -B -m unittest discover -s schema_template/tools/tests -t schema_template
```

布局回归覆盖紧凑调度、必要引号、折叠块标量、exit 的列表阈值与逻辑换行，以及两行标量保真；CLI 回归验证本地资料门禁位于纯快照 API 之外。文字侧回归覆盖代表性的名称/分支预算、最小数值形式、引用与 retention、本地提交路径及合法占位、长动作和 Waiting 的原语法边界。最终审阅修复了 `item` 过早拒绝方括号的问题：先匹配已登记完整名称和映射路径，再检查未解析目标的下标形式；未开放列表下标身份。独立复现还验证了完整 300/2 与 Input 150、外部身份 100、零 I/O 快照、带空格和括号的本地目录及调度 alias/tag 保真。详见 [TEXT_RULES_REVIEW](TEXT_RULES_REVIEW.md)。

前轮合成套件为 70 项，独立审阅存于 [FINAL_REVIEW_CORE](FINAL_REVIEW_CORE.md) 与 [FINAL_REVIEW_PIPE_FORMAT](FINAL_REVIEW_PIPE_FORMAT.md)。此前 Graphviz 的临时合成任务验证了最终 source hash、manifest、effective identity 和来源链，并实际查看 862×535 的生成图；这些属于前轮已完成的验证，不作为本次新增规则的验收。

`text_limit_demo` 是上轮按当时预算填满的快照，包含用户后续现场内容。本次未迁移、未重新填满、未用其验收新规则；它及原 CSV/README/dispatch 统计附件保持原样。新形式规则可以使其中原值报错，不能据此放宽规则或称其已通过本次检查。

本次未整理或重画正式 examples，未修改 wiki、Extract 执行器、运行状态及 CHANGELOG 末尾用户原稿，未进行源码提取或 GPU 验证。
