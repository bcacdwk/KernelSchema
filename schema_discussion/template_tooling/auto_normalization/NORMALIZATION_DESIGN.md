# 自动规范化：任务级接口与安全边界

目标是同一个 `check_yaml` 先产生可验证的规范化候选，再检查字段、预算和关联。
字段、文本类型和闭集值从结构检查器的 `CANONICAL_FIELDS / STRING_FIELDS / CANONICAL_VALUES` 取得，
纯整数/布尔叶的引号恢复由同源 `LITERAL_TYPES` 限定，
不维护历史拼写兼容表。A/B/C 未决字段仍采用当前结构的唯一拼写。

## 顺序

1. 读取原始字节并保存摘要；严格 YAML 解析，必要时请求窄范围缺引号候选。
2. 按唯一路径表建立只读字段视图，并在每个消费节点的实际作用域规划声明和引用；
   固定字段、闭集值与具名用途最后一次性共同授权，避免先改某一用途导致共享 scalar 被误拒绝。
3. 生成候选，核对类型、精确值、源节点图、别名和 merge；之后统一句号、引号和既定布局。
4. 全部所选候选验证成功后安全写回，重新解析实际字节；只读模式只预览。
5. 独立检查字段、字符/逻辑行预算、from 和关联；真实超预算不能挡住安全文本规范化。

## 小接口

- `yaml_source.scalar_contexts(document)`：枚举源 scalar 的全部实际字段用途，包含集合 alias、merge 和被覆盖默认值。
- `yaml_source.repair_missing_quotes(text, name, *, path, corrections, skip_hardware=False)`：
  收集各自有确定字段/边界的补引号，最终严格重解析并核对原 lexeme 后才返回；普通 `loads/load` 不自动修复。
- `yaml_source.recover_scalar_quotes(document, *, corrections, skip_hardware=False)`：
  在所有共享用途均为文本叶、无显式 tag 时，把 bool/int/float 原 lexeme 包裹成字符串。
  全部用途仅接受同一种 int/bool 时，也可对普通单行 quoted scalar 去引号；同一 core resolver 和严格解析必须确定该类型，原 lexeme 及复用图不变。null、显式 tag、联合表达式类型、空白、转义或多行内容不猜转。
- `yaml_format.ScalarEdit(node, value, contexts, main_names=())`：只授权原始用途上的字符串修改。
  默认仅大小写；`main_names` 只额外允许明确绑定的完整身份 token 改成 `MAIN`，不能任意重命名。
- `yaml_format.field_case_edits(document, fields)`：内部唯一路径表驱动的 casefold 匹配；`None` 为具名映射键，`int` 为列表位置。
- `yaml_format.canonical_parts(parts, fields)`：只规范固定字段组件，保留具名键与列表位置。
- `format_document(document, *, edits=(), corrections=..., skip_hardware=...) -> str` 保持无写入、无原地修改。
- `fields` 不是用户配置或版本表；后轮仅用一份当前规范字段定义调用，不保留旧名兼容。

任务入口位于 `tools/normalization.py`：

- `load_for_normalization(path, name, *, skip_hardware=False)` 与
  `loads_for_normalization(text, name, *, path="<memory>", skip_hardware=False)` 返回
  `LoadedDocument(document, source_sha256, corrections)`；摘要属于原始字节，不能换成恢复后的摘要。
- `normalize_documents(files, documents, *, canonical_fields=None, skip_hardware=False)` 接受
  `Document` 或 `LoadedDocument`，返回 `NormalizedTask(files, documents, candidates, source_hashes, corrections)`。
- `documents` 是全部最终候选；`candidates` 只含确实变化的 `{node: text}`；`source_hashes` 是各文件原始摘要。
  所有文件应作为同一次任务快照传入，库绝不写回；CLI 决定是否预览或按原摘要核对后一起写回。

## 可以自动处理

- 只有固定字段所在的位置匹配规范字段；具名对象刚好叫 `Name` 不因此被当字段。
- `arch` 到 `Architecture` 是正式文档/规则改名，运行时只匹配 `Architecture.casefold()`。
- 键和 scalar 的变化按原始节点及原始用途逐项授权；同一个共享节点必须覆盖所有用途且目标一致。
- 规范化造成显式键或合并后键碰撞时整份候选拒绝，不能覆盖、合并或只改一半。
- 缺引号覆盖无注释、非 flow 的单行已知文本字段中的第二冒号或保留起始字符。
- 完整 Event/Waiting 以现有括号语法证明范围，可在 block/flow 列表中补包裹；
  多个独立、可证明的缺引号可连续收集，但任何剩余歧义都使整份恢复候选失败。
- 必须由 scanner 确立父字段和 scalar 边界；仅包裹该段原文，候选仍须通过完整严格解析。
- 未定义 alias、已有坏引号、坏容器、null 占位、未能证明的类型误写和多行/flow 边界不明不猜修。
- 原有高置信说明末句号规则继续适用；数字、缩写、公式、路径或混合共享用途不删。
- 引号由同一 resolver 和实际 flow/key 上下文决定；布局只采用已有小记录规则。
- 物理折行与 decoded LF 区分；300 字/2 行仍是允许上限，不能截断或把所有值强制一行。

## 命名与引用

先建作用域内声明映射，再同步结构化 ref/bind、Step/Role、Execution/Feature、member 和可解析动作。
可识别表达式只改已绑定且唯一的标识 token；外部属性链、普通正文和不透明公式不是 global lower 目标。
若待改声明存在无法消除的作用域歧义、名字碰撞或可疑未覆盖引用，说明具体原因并拒绝该候选。
跨区域及 from 的使用上下文以同一次任务快照解析；同一物理 scalar 在所有消费者中必须得到相同目标。
只产生全任务计划，不改文件名、Dispatch.id 或 from 来源。单一 Step/Role 可明确映射到 MAIN，
若不同消费者需要 MAIN 与其它身份等不同目标则报出各消费者的冲突，不能只规范提供者。
无需构建完整代码重构器；无法证明绑定关系的引用应留给作者，不能伪造可靠性。

公式只同步已登记、运行时 extent、shape 轴及明确赋值建立的数学名称；简单函数形参和局部数学索引
限定在可证明的公式行，Pipe 的局部 for 名称限定在其词法 body，Event/view 形参限定在自己的条目。
外部属性链、引号中的字面字符串、dtype、设备型号、源码函数/路径及普通说明正文保持原样。
不能唯一绑定的多行形式定义、名称碰撞和共享消费者分歧会给出具体错误，不建立通用数学语言解析器。

## 保真与诊断

两层校验都保留：decoded 值只能发生精确计划内键/值变化，源节点图保留数字原文、tag、anchor、alias、merge。
字段数量、列表顺序、未授权字符串及 from 来源仍受保护；候选需达到幂等。
correction 使用 `field: / identifier+reference: / quote: / period: / layout:` 五个稳定前缀，
名称/字段/句号提供简短 before→after，布局每文件汇总；位置已经随计划内的键改名映射到最终候选。
CLI 按类别汇总并根据是否写回说明候选或实际文件位置。
无变化不 warning、不 rewrite；未写入候选不能报告“已修正”；语义合法性错误继续单独报错。

## 交接

本轮库已经提供可用全任务计划；CLI、安全写回、维护回归与作者说明由第三轮接入。
类型恢复仅在独立源阶段授权，不放宽 formatter 对 scalar 类型、精确 Decimal、列表顺序或复用关系的保护。
不会把缩写 arch 迁移成 Architecture，也不会把无法精简的超长正文截断；这些仍由严格检查报错。
详细接口、smoke 与复核边界见同目录 `TASK_NORMALIZATION.md`。
