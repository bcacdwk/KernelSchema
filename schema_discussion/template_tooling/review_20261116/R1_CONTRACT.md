# R1 契约与有限图审阅

基线：`0778456`；2026-11-16。本轮只读取规范、代码并运行最小复现，没有修改正式代码或案例。主代理已运行基线全套测试：113 项通过，22.317 秒；本审阅没有重复全套测试。

结论：确认四类静态漏检，均可在现有规则内修复。没有发现需要新增字段、封闭取值或改变数字预算的理由。本轮未确认有限图主链的错误完成结论；它的有限样本边界应继续保留。

## 已确认问题

### R1-C1 · P2 · 裸 Item 的已知对象路径被当作开放目标，漏掉属性与重复身份检查

- 依据：DOC6 `Item` 先匹配已有完整名称，深层路径遵循上游发布结构；裸名称与限定名称指向同一对象时等价，不得重复同一对象（`schema_docs/6.realization.md:97–113`）。
- 位置：`tools/rules/symbols.py:414–424` 只尝试裸对象/步骤名的完整匹配，随后直接走 `semantic_item` 和 `unresolved`，没有检查已登记对象的后缀路径。限定写法则在 `:431–448` 遍历属性。
- 最小复现：以 `tools/tests/fixtures/minimal/root.yaml` 为底稿，增加一条 `REALIZATION.Require`，`Choice/Reason` 为普通短字符串；通过 `schema.validate_documents` 检查。

| Item | 当前结果 | 应有结果 |
| --- | --- | --- |
| `x.no_such_property` | PASS | `x` 是唯一已登记 Input，应报告其不存在的属性 |
| `CONTRACT.Algorithm_requirement::x.no_such_property` | 正确报错 | 保持 |
| `[x.dtype, CONTRACT.Algorithm_requirement::x.dtype]` | PASS | 两者是同一个已发布属性，应报重复 |

- 影响：已经能够静态定位的错误进入人工开放目标分支；相同问题还会使 `Feedback` 的对应关系过度放宽。另一位审阅者观察到 `X.SHAPE → x.SHAPE` 后长期不再纠正，这与这里的身份遗漏相交。
- 建议：在现有 Item 解析中，先完整名称、再已知对象路径，返回确定的规范身份；只有无法由结构确定的公式/Strategy 名称才保留开放解析。不要强迫所有作者补 `owner::`，不要从正文提取伪符号表，也不要把数学限定名一律解释为字段路径。
- 验收：上表三项；合法 `x.dtype` 与其限定写法均通过；现有 `formula_value`、限定公式目标、带空格/括号的完整登记名继续通过。

### R1-C2 · P2 · 被引用的重复 Compute/Rewrite 名称失去歧义信息

- 依据：DOC6 `:99–107` 要求 Item 唯一解析；DOC0 `:186` 要求按唯一 rewrite 名称定位。该要求不意味着所有未引用标题都必须全局唯一。
- 位置：`tools/rules/symbols.py:68–69,399–400` 把步骤名称收进 set；`realization.item` 对限定的数学步骤名也不计候选数量。
- 最小复现：底稿的 `Math_reshape.Rewrite` 改为以下两条，再加 Item 为 `DESIGN.Math_reshape::Cast` 的 Require；`schema.validate_documents` 返回 PASS。

```yaml
Rewrite:
  - {Name: Cast, Math: a = fp32(x)}
  - {Name: Cast, Math: y = fp32(a)}
```

- 影响：引用不能说明约束哪一次 cast，却被当作已经解析的目标；区域限定也无法消除同一区内的重名。
- 建议：保留结构化名称的候选身份/数量，在实际解析命中的引用处报歧义。与 C1 共用最小解析逻辑即可，不另建兼容层，不全局禁止未被点名的重复标题。
- 验收：该引用报歧义；两个不同名称分别引用通过；没有引用时不要仅凭本项审阅新增标题唯一性限制。相同检查适用于实际引用的 Contract Compute 名称。

### R1-C3 · P2 · dispatch 的最低表格与标题类型未检查，测试固化了缺项通过

- 依据：DOC0 `:155–182` 明确唯一最小格式，完整设计使用表格，变体标题类型只能为“参数 / 局部 / 组合”。机制、保留理由和条件是否充分仍是人工审阅，不应由工具猜测。
- 位置：`tools/rules/dispatch.py:83–90,110–122` 只检查节非空及节内链接；`tools/tests/test_dispatch.py:71` 显式断言没有表格也通过。
- 最小复现一：`dispatch.validate` 对下面正文及 `{'root': Path('/task/root.yaml')}` 返回空诊断。

```markdown
# 标量复制
## 范围
收录标量复制
## 完整设计
[root](root.yaml)
```

- 最小复现二：在合法完整设计表后写 `## 变体与组合`、`### 更快`、`[root](root.yaml)`，也返回空诊断。它没有声明允许的变体类型；目前连这个可确定的结构缺项也不报。
- 建议：补最低表格存在性、当前文件在表中登记，以及变体标题的允许类型检查。正文三个信息项可以合为一句，不能强制新增列表、字段或逐项标签；正文质量继续留给审阅。不要把普通节外 YAML 链接当成完整表登记。
- 验收：两例不再通过；DOC0 最小表格和合句变体仍通过；保留已有 reference link、行内 code 标签、围栏模板和外部 URL 的处理。

### R1-C4 · P2 · 可确定的 Wiki.Hardware 目录被当作资料文件接受

- 依据：DOC2 `schema_docs/2.contract.hardware_specification.md:117` 原文：“普通入口可以指向一个文件或目录；`Wiki.Hardware` 必须指向具体型号的资料文件”。这里只对这一叶增加已定文件要求的落实，其他 wiki 入口继续允许目录。
- 位置：`tools/rules/declarations.py:34–37` 仅看尾部 `/`；`tools/rules/values.py:547–550` 查询 Git tree 后只保留名字，丢弃已知对象类型。
- 最小复现：底稿的 `Wiki.Hardware` 改为 `KernelSchema@0778456:schema_docs`，Document 对应路径使用本仓库中的 fixture 绝对路径。`validate_documents` 与 `validate_local_repositories` 均 PASS；同一修订的实际对象为：

```text
$ git --literal-pathspecs ls-tree 0778456 -- schema_docs
040000 tree ed5e0da1ff20725f99476d2f064fa077580e47cc  schema_docs
```

- 建议：沿已有本地仓库核对保留 Git entry 类型，对明确指向 tree 的 `Wiki.Hardware` 报错；不靠扩展名猜文件、不联网、不扩大到普通资料入口。设备材料内容是否确为该型号仍需审阅。
- 验收：该本地目录被拒；本地已存在文件通过路径/类型检查；`Wiki.Programming.Language` 等普通入口指向同一目录仍通过；占位和 `--skip-hardware` 保持原边界。

## 本轮覆盖与边界

- 已读当前 DOC0–6、Template 与 tools README、starter，走读 `schema.validate_documents → structure → raw text → resolve → declarations/symbols/schedule/pipeline`，以及 `pipeline_syntax.py`、`pipeline/` 的展开、窗口、边、渲染与发布调用。CHANGELOG 仅作背景，未用历史附录制定规则。
- 最新字段、`Flow.Launch[].step/order/start_after`、noun 级 `execution_mode`、Waiting 分类、Storage 当前两行示例与 starter 未见需要改动的契约差异。两行布局的实现保真由另一位审阅者主查。
- 按已同意文本提案复核了数字/类型/硬件身份、retention、内部 binding 等分支。`300/2` 有明确适用位置，不能据本轮审阅给所有 `ref:/bind:` 增加兜底预算。自动格式和源恢复问题由另一位审阅者记录。
- 新做三个有限图小检查：含一条越界移位前驱和一条合法前驱的等待保留 3 条合法边、仅 1 项边界跳过；生产者与目标同在假分支时不制造等待；`for(j=0:⌈n/2⌉)` 语法接受而有限求值返回 `ExpansionError`，没有把开放数学转成 schema 非法。结果符合现有边界。
- 整区引用的节点 DAG、provider/chain 与 consumer 绑定、单文件阅读身份已对照现有测试走读。本轮未重复来源写回/CLI 自动化主线，未使用展示 demo 作算法证据。
- 第三轮宜继续独立核对带形参 Event 的调用域、跨局部作用域的 Waiting、依赖域中空工作与固定/移位坐标组合；本轮没有把这些未充分验证的组合列成缺陷，也不宣称已穷尽。图仍只证明所选样本的局部事件关系，不应扩成跨 step 活性或存储安全证明。

## 可在 high 精简轮核实后删除的冗余

- `pipeline_syntax.common_domains`（`:93`）、`pipeline.actions.operands`（`:12`）在当前 tools 与测试中没有调用点，README 也未列为 API；可再检索消费者后删除。
- 数学属性 `_target` 在 `descriptor_properties` 与 `math` 中重复解析。最小错误 `copied.shape: ref:CONTRACT.Algorithm_requirement::absent.shape` 在 `rules.validate_relations` 中产生两条路径/消息完全相同的诊断。可复用同一解析或移除重复检查，保留定义顺序、属性存在性、循环及绑定约束各自的职责；不需要新增通用诊断去重层。
- C1/C2 应收拢已有名称解析，而非在校验末尾再追加一套字符串补丁。

## 用户待决项

没有阻止上述四项修复的用户待决项。开放表达式/公式身份、变体正文质量、设备材料内容和设计唯一价值仍按当前规范人工审阅；本轮不据工具遗漏扩充 schema 语言。
