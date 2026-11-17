# R4 A：有效 changelog 与 Docs / Template 落实核查

结论：完整读完 [schema_docs/CHANGELOG.md](../../schema_docs/CHANGELOG.md)，按后续最终决定合并检查，未发现已明确的当前规则在 Template 代码中尚未落实。本轮修正一处 Template 作者入口残留的隔离表述；四个当前案例、九份 YAML 和九张 PNG 均保持原样。Extract 与根作者入口由并行 B 核查，本报告不替代该部分。

核查同时使用当前 DOC0–6、[文本批准表](../template_tooling/TEXT_RULES_PROPOSAL.md)、[最终文本审阅](../template_tooling/TEXT_RULES_REVIEW.md)、[字段决定](../template_tooling/auto_normalization/FIELD_DECISIONS.md)及 [Storage 最终决定](../template_tooling/auto_normalization/STORAGE_LAYOUT_REVIEW.md)。末尾字符限制草稿按要求逐项核对，没有整体作废；替代项见下表。表中“已核”表示已追到实际代码、现例或对应验证，不表示工具可以证明数学、来源或设计独特价值。

## 有效要求与实际落点

以下路径以 `schema_template/` 为相对根，DOC 编号对应 `schema_docs/0.common.md` 至 `6.realization.md`。

| 合并后的有效要求 | 实际实现、现例或验证位置 | 状态 |
| --- | --- | --- |
| 一份 YAML 表达一套具体完整方案；root 唯一 Contract、完整本地 Design；完整集合含 root 最多 10 份；其他配置、局部决定及组合记入目录，不能由目录补正文 | DOC0 的设计划分、完整价值与范围规则；`tools/rules/structure.py:validate_task`、`tools/design_resolution.py:task_files`；四例分别 3/4/1/1 份 | 已核；独特价值、具体方案与任务覆盖仍需审阅 |
| `Dispatch.id/when`、同任务 Name/Describe、非 root 无 Contract；范围允许重叠或未覆盖；中文目录必需“范围”和完整设计表，变体标注参数/局部/组合并直接定位当前完整基线 | `tools/rules/{structure,declarations,dispatch}.py`；四例的 `dispatch.md` 已通过；DOC0 保留范围、依据、对象和未展开独立候选要求 | 已核；记录语义及组合相容性不由最小格式检查代替 |
| 同任务同名整区 `from`、无覆盖字段、跨区节点图无环、使用节点绑定；只传播实际区域；raw/effective 分离，追溯 root、自身、中间链及 provider | `tools/design_resolution.py:resolve_documents`、`ResolvedTask`；`tools/rules/symbols.py`；`test_design_resolution.py` 的来源链、consumer 绑定、来源诊断、独立副本与跨区环检查 | 已核；两份现例仅从 root 引用数学区 |
| 唯一当前字段：Step、Launch 的小写 `step/order/start_after`、Preassigned.stride；Architecture、Mapping.Member、三处 Thesis、Arithmetic.Approx/Accumulate、Ledger.Baseline/Gain/Cost；独立 REALIZATION.Feedback | `tools/rules/structure.py` 导出的 CANONICAL_FIELDS/STRING_FIELDS；`tools/rules/hardware.py`；`starter.yaml`；`test_normalization.py:test_final_field_spellings_and_lowercase_attributes_share_source_recovery` | 已核；不接受旧单词或层级的兼容别名 |
| Rewrite 无步骤 ID，Ledger 归每个 Rewrite、Baseline 键必需且三项可 null；新增工作块/Partition/Storage 的 meaning、Index 的 meaning/definition、Info.Describe；view 只有 slot_index/region | `tools/rules/structure.py` 的 rewrite、ledger、工作记录、pipeline_step、view；DOC1–6 的结构与片段；当前例子通过结构检查 | 已核 |
| 说明文字分工、unknown 指明缺口及影响、Strategy/Refill/Choice 只补必要信息；输入默认域、数值转换和动作本地对象沿正式定义；实现 Require/Prefer/Forbid/Feedback 保持权限与强度 | DOC0/1/3–6；`tools/rules/{values,symbols}.py` 核对属性/Item 路径、等价目标、被引用重名及 Feedback 对应 Prefer；现例保留来源缺口 | 已核；表达是否充分、实验是否真实由审阅判断 |
| 固定源码 P1 可最小登记有据 Execution，其余占位；`--skip-hardware` 显式跳过整个硬件区、预算和依赖硬件关系，不隐含只读；本地可确定的 Wiki.Hardware 必须为文件 | DOC2；`tools/schema.py`、`rules/{structure,declarations,values}.py`、规范化的 skip 分支；现例无 skip 也通过，Wiki 占位仍可见 | 已核；资料占位不是已完成绑定 |
| Flow 每个 step 一次提交、不同提交无隐式次序、四种接续粒度；片上 worker 身份、默认 ready/按 consumer 的 wait_at、入口事件及补值 initialize；launch_ready 保留，具体协议由动作与 Realization 承担 | DOC4/5；`tools/rules/{declarations,schedule}.py` 的提交、启动环、共享映射、存储身份与 consumer_waits；`test_schema.py` 相关检查；四例 Flow | 已核；跨 step 实际可见性/存续和调用前提仍需语义证据 |
| 八动词白名单、noun 固定签名、三种 noun 级模式；有序非空 Pipe 仅 for/if、词法索引作用域、flatten 恢复冲突、动态域 count/last 限制 | `tools/pipeline_syntax.py`、`tools/rules/pipeline.py`；有限展开 `tools/pipeline/{graph,expressions}.py`；`test_pipeline_graph.py` | 已核；无 while / 专用 prologue 等节点 |
| Waiting 三分类、显式 issue/done 和非空前驱列表；省略 member 为当前 worker 成员，不按 warp 数扩实例；只建立模式默认边及显式边；async_ordered 跨 noun/verb/循环/worktile 累积 | `rules/pipeline.py`；`pipeline/graph.py:add_operations/explicit_edges`；四种时点边、同名 noun 无数据边、跨成员与累积完成既有检查 | 已核；不推导同名数据边或额外 drain |
| Storage 的 meaning/depth/location/shape/dtype/total_size，容量按字节及 worker 持有成员计；view 不重复计账、无默认槽映射，region 为父容量坐标；共享字节不产生同步或清零 | `rules/{structure,declarations,pipeline}.py`；DOC5 的固定两行例子、共享 S/P 和槽映射；`test_schema.py:test_known_storage_size_and_member_bounds` | 已核；符号容量、覆盖安全及原语保证仍需审阅 |
| Event 为已有事件集合；Flow 实际调用代入形参，未调用别名不求值；缺样本与确定缺失事件分开；有限图保留空工作入口、局部末项、flatten/padding 边界及固定参数相关性 | `pipeline/__init__.py:analyze`、`graph.py:event_aliases`、`windows.py`；`test_pipeline_graph.py` 的 EventCallTests 及窗口边界检查 | 已核 |
| 有限图使用 Mapping.Member 可证明的单点数学坐标，局部绑定遮蔽同名坐标，普通 for 端点在外层作用域；peer 窗口逐成员检查；支持有界整数异或和左移 | `pipeline/{graph,expressions,windows}.py`；成员/局部作用域/固定联合投影及位运算定向检查；decode cluster 现图 16 个成员 | 已核；多元素/未知区域不猜代表值 |
| 全任务规范化定义与消费者，涵盖 from、单 Step/Role MAIN、公式/循环/Event/view 的局部绑定；只修可确定大小写，碰撞/歧义停止；文件名、Dispatch.id、from 来源值不改 | `tools/normalization.py`；`yaml_format.py:ScalarEdit/_edit_plan`；`test_normalization.py` 的跨文件、局部作用域、开放量词、属性引用及共享 alias 检查 | 已核；开放 Math/Strategy 正文无法可靠归属的填写名称不猜改 |
| 原稿精确读写：Decimal、必要引号、合法换行、anchor/alias/merge 与源位置保真；明确 opt-in 恢复与严格 load/loads 分开；检查/规范化快照 API 零 I/O | `tools/yaml_source.py`、`yaml_format.py`、`schema.py:validate_documents`；精确值/源图/幂等及只读 API 定向检查 | 已核 |
| 默认整任务生成候选、安全写回后严格重读再检查；warning 分类列实际修改；预算错可伴随已完成修正；原文摘要、权限、mtime、外部新修改与 symlink 保护 | `check_yaml.py:_normalize/_write_candidates/_unchanged`；`test_cli.py`、`test_normalization.py` 的候选/发布验证 | 已核；规范化不递增 Version |
| `--read-only` 检查相同候选并标未写入；`--all` 仅扩图范围；`--show-effective` 标 validation:not_run；独立视图保留节点身份、来源与只读 standalone 校验 | `check_yaml.py`、`schema.py`、`design_resolution.py:effective_yaml`；`README.md`、`tools/README.md` 的当前 API 表；standalone 定向检查 | 已核；导出前完整任务验收与展开成功分开 |
| PNG 同名输出、最终 raw/provider/chain/effective 身份和确切样本入图；全部请求先分析；缺样本/来源漂移/渲染失败保留对应旧图，完整环可生成诊断图；普通检查不清旧 PNG | `check_yaml.py:_graphs`、`pipeline/{output,render}.py`；来源漂移定向检查；九 PNG 实际元数据核对 | 已核；图只证明给定样本的局部事件关系 |
| 四个固定来源案例从 21 份收束为 9 份，参数/局部/组合进目录、保留来源限制及确切图参数，现行作者入口可使用 | `examples/README.md`、各例 README/dispatch/YAML/PNG；[前轮独立验收](../template_examples_20261117/FINAL.md)；本轮四目录只读与九图身份核对 | 已核；未重新跑语义提取、GPU 或大图 |

## 文本预算与静态布局逐项核对

预算按解码后的 Unicode 字符/逻辑行计，不裁剪文本；同一个 raw 定义不在 from consumers 重复计数。实现为 `tools/text_policy.py:BUDGETS/check_document`、`rules/values.py` 与 `pipeline_syntax.py`；没有把开放公式变成新的强制表达式语言。

| 最终要求 | 实际落实 |
| --- | --- |
| Name 50/1；全部 Describe/describe 150/1；meaning 50/1；Math 1000/10；三处 Thesis 300/2；Dispatch.when 500/5 | BUDGETS 按各自结构路径覆盖元数据、Compute/Rewrite、Step/Role/noun 及所有 meaning 条目 |
| Input.range/default 150/1，range 字符串端点逐项；Tolerance 每输出和 Gap 100/1；Device/Architecture 50/1；Execution/Feature 摘要 100/1 | BUDGETS；保留最终 DOC1 已定义的表达式类型，不采用草稿中的一律禁字符串 |
| Arithmetic.Approx/Accumulate、Ledger 三项、unknown 属性、三个 Strategy、每个 Refill 机制均 150/1；Ledger null 不消耗预算 | BUDGETS 与值分支；保留 Ledger 最终重新确定的预算，不沿用历史中间态的“无上限” |
| Require/Prefer/Forbid 的 Item/Choice/Reason 与 Feedback.Item 300/2；Feedback.Choice 150/1；列表引用逐项 | BUDGETS；既有两行预算保留，无额外全串压成一行 |
| 可解析自取单名 50/1；普通 dtype、Storage.location、硬件登记身份 100/1；结构化整数表达式/端点和固定值字符串叶 100/1 | `rules/values.py` 分支选限额；`pipeline_syntax.py` 逐单名/表达式检查；真实类型/硬件身份不叠加单名 50 |
| Stack 各叶、retention 150/1；Exit、Tile_reduction 输入/输出/方法、Handoff.initialize 的字符串、内部 binding、自由来源摘要 300/2；Wiki 各叶单逻辑行 | BUDGETS、`values.binding/source/repository`；纯仓库定位保留真实相对路径及合法空格，不截断为摘要 |
| 数值串有实际内容；Unicode 数学、允许语境的 ∞/∅、切片省略及事件 * 保留；retention 只有既定外壳；引用先匹配真实完整身份 | `text_policy.numeric_form_message`、`values.semantic_reference/identity`、`structure.retention`；文件入口才核可确定的本地提交路径，快照 API 不读仓库 |
| 只保留 YAML 必要引号，普通说明可证末句点自动清理并警告；公式/路径/缩写和不安全共享用途保留 | `yaml_source.plain_scalar_allowed`、`yaml_format.py`、`text_policy.sentence_period_changes`；quote/period/Unicode/共享用途定向检查 |
| 具名数据记录行内、结构分区及 Pipe 展开；Worker/Placement、Worktile_order、Preassigned、Refill 使用已定紧凑布局；Exit 仅 1–3 项且各 ≤50 字符、无逻辑换行时行内 | `yaml_format.py:RECORDS/SCHEDULE_RECORDS/_flow_style`；布局按路径和类型决定，不按字符串宽度自动试排 |
| Storage 固定两行：名字及 meaning/depth/location/shape；第二行 dtype/total_size/member/view，缩进两格；必要正文换行、alias/merge 和 anchor 顺序保真 | `yaml_format.py:STORAGE_FIRST_LINE/STORAGE_SECOND_LINE/_render`；DOC5 三例与现行九 YAML；固定断点/共享图/Decimal 定向检查 |

## 本轮补齐、验证与边界

- 补齐：`schema_template/README.md` 将 Extract 的“阶段检查、隔离与运行环境”改成“阶段检查、执行边界与运行环境”，与用户已决定移除 Extract 自建隔离一致。未改工具接口或规则。
- 四目录执行 `python3 -B schema_template/check_yaml.py schema_template/examples/<case>/root.yaml --read-only`：3+4+1+1 份 YAML 全部退出 0、零待写变化；结构、预算、整区来源、关联和 dispatch 均通过。
- 九 PNG 用 `pipeline.output.metadata` 实际解析；逐一与当前 `ResolvedTask` 比较 `source_sha256 / source_manifest / effective_identity / origins`，全部相同；逐 step 样本与各例 README 一致，均记录无环。没有仅按文件存在性判定完成。
- 28 项既有定向离线检查通过，3.992 秒：覆盖 Schema/预算/Item、精确 YAML/Storage/调度布局、来源解析/standalone、成员与窗口/Event、默认安全写回及来源漂移保图。执行环境为 `/opt/venv/python3.13.12/bin/python3`；一次在子目录使用系统 `python3` 的先行尝试因缺 PyYAML 未执行相关测试，换用文档指定环境后通过，未安装依赖。没有新增测试或运行整套回归。
- `git diff --check -- schema_docs schema_template` 通过。当前四例已通过的完整语义审阅、九个主图和七组边界验证来自[既有验收](../template_examples_20261117/FINAL.md)，本轮不冒称重做这些工作。

| 被否决、替代或延期的历史项 | 当前处理 |
| --- | --- |
| while；旧 repeat/prologue/steady/epilogue、旧动词及旧等待分类/事件记法、默认同名数据边 | 已被 DOC5 最终 for/if、八动词、三模式和显式依赖替代，不恢复 |
| phase→Step、Preassigned.step→stride、resource/domain/milestone→Storage/Index/Event、part→region、view.dtype、区域级 Ledger、prefer 内嵌 feedback、dispatch.parent/from_parent、Launch.protocol | 采用当前唯一结构；Extract 的 P1–P6 phase 不是本次工作组织改名对象 |
| 全禁引号、range/default 禁字符串、所有 entry 一律单行、Storage 动态列宽、所有相同单词一律首字母大写 | 以必要引号、已定值类型、按路径静态布局及明确小写属性边界替代；Storage 固定两行 |
| 对所有填写字符串套标识符小写规则 | 用户明确填写值不受该规则约束。已声明身份及引用仍同步；开放 Math/Strategy 中无法可靠归属的 `T_b/N_r/N` 等不猜改，保留现例和图 |
| 历史条目写着“Template 工具/正式案例/Extract 后续对齐” | Template 工具和四个现例已由后续实施完成；Extract 的现状由本轮 B 核查，不能以早期“后续”文字判定仍未实施 |
| schema_wiki 批量重新转换、Wiki 资料绑定、真实模型全流程/GPU 验证 | Wiki 重转换由用户明确留后续；硬件 Wiki 占位与四例来源限制仍保留；真实模型完整试跑已按用户要求停止，本轮均未执行，不计为完成 |

本代理改动文件仅本报告和 `schema_template/README.md`；保留基线以来全部已有修改，未写 Kernel、schema_wiki、archive、历史讨论/演示快照，未 commit。
