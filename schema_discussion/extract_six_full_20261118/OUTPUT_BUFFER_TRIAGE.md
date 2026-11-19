# Sparse output buffer：独立只读分流

结论：现稿存在可按既有规则修正的 P1 输入登记缺项，不能仅作为“两种同样合规写法”的规范建议结案。当前计算、输出及副作用语义已完整；这不是算法或实现错误，也不需要新增字段或先修改规范。建议通过现有 maintain 的 P1 限域 review，由真实 reviewer 独立判断，必要时 worker 修正后 verify；本记录不代替该审批。

审阅对象：run `8f0844f9-f279-4781-a19d-0216b7fc4ef5`，读取 `state.checkpoint` 为 `e6ec946a1bcebd260efee3ae524089b7b822fc2fd05b7a7d4dc87c6145d38cef`，状态为 P4 worker。固定来源 commit 为 `24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2`。以下简记：

- `root`：该 checkpoint 的 `artifacts/root.yaml`。
- `source`：快照 `e5db05dc281797bc4a7350cc1d7eb35b2d1331efb10f304bd0ab3b1955ff6388/source/kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/`。
- DOC1、common：该快照冻结规范；已逐文件核对，与当前 `schema_docs/1.contract.algorithm_requirement.md`、`schema_docs/0.common.md`、`schema_extract/prompt/common.md` 内容一致。

## 为什么属于现有登记义务

DOC1:43 要求 Input 登记“调用者提供的全部对象，包括可缺省的输入”；:44 要求原地写回输入的结果“也登记为输出”。这两个条件分别成立，不以传入对象的旧元素是否参与数学计算为前提。DOC1:5 的调用者视角与输入输出关系也没有撤销这两项字段定义。将纯目的地排除出 Input，需要补入现行规范没有的例外。因此这里确实需要分别登记传入缓冲角色与结果角色，而不是因为传统 ABI 惯例而附加新义务。

这也不是只凭签名扩充任务接口：`source/problem/metadata.json`:13–34 明确 `mutates_inputs=[out]`、返回相同输入对象、保留输入存储；`problem/input.py`:119–130 实际传入 out，并将它列为可观察写回对象。`problem/reference.py`:17、43–45、70、82–83 则确认参数可缺省/取 None、否则写回既有对象并返回。candidate 的 `solution.py`:187、223、237–240 保持对应行为。Input 已有不贡献数值结果的 workspace_buffer（root:17），也支持本例读取，但判定依据仍是 DOC1 的明确文字。

DOC1:59、61、214、226 已规定 nullable 与 default 的职责。现稿 root:38 在散文中描述参数 None/省略，并不能替代 Input 条目的相应字段。反过来，Output.out 始终是结果张量，不应因为传入目的地可为 None 就给结果增加 nullable。

## 当前语义及实际影响

root:32 登记 BF16 输出的完整 shape；:38 说明原地写回、保留并返回同对象以及 None/省略时分配；:67、73 给出空集合和非空集合结果；:80–84 明确全部元素覆写、不保留旧值及其他副作用。就本候选而言，读者已经能理解并实现正确行为；没有证据表明数值、写回范围或验收结果因此错误。

缺项影响的是 Contract 自身声明的输入对象及其缺省/空值信息。不能把现有散文完整性当作免除固定字段义务的理由，也不必由此引出完整 ABI 字段、泛化检查器或全 Design 重审。reviewer 的候选指出真实写法分歧，但“都能理解”不等于“都满足当前字段定义”。

## 建议的最小 P1 范围

仅 `root.yaml` 的以下现有字段位置：

1. 新增一个 Input 条目，例如 `CONTRACT.Algorithm_requirement.Input.out_buffer`。在 meaning 中映射源码参数 `out`，说明其为调用者提供的写回目的地、旧内容不参与计算；填已知 shape/dtype，以及既有 `nullable`、`default`、必要的 `range`。
2. `CONTRACT.Algorithm_requirement.Output.out.meaning`：明确结果与传入 out_buffer 的条件性对象同一关系。保留输出 ID `out` 及其 shape/dtype。
3. 既有 Compute 的 `Interface preconditions.Math`（当前第 0 项）：只把传入缓冲的条件、None/缺省分支和返回对象绑定到新的输入名。
4. 既有 Compute 的 `Output write scope and side effects.Math`（当前第 5 项）：只同步“输入中的写回目的地”名称，保留全量覆写及原有副作用语义。

DOC1:107 要求输入、输出等名称在同一契约内唯一，因此不建议直接同时使用 `Input.out` 与 `Output.out`。新增不同输入名并在现有 meaning/Math 中建立关系即可，不必改输出 ID，也不牵连依赖 `out` 的 Design、Tolerance 或参考入口。上面是语义上的最小字段范围；维护请求应沿用工具现有的字段路径语法，不添加新的列表选择记法。

该 Input 条目的 range 需要特别保留事实：`problem/input.py`:119 用 `torch.full_like(q, float('nan'))` 初始化目的地。DOC1:60、200 在省略 range 时默认浮点输入有限；若只补 shape/dtype/nullable/default，会错误排除当前固定输入。应在现有 range 内表达旧内容不受有限值限制，并与“全部覆写、不读旧值”一致；不能反过来将某个 NaN 样本收窄为唯一允许的初态。这是新增条目自身的必要完整性，不是额外扩大维护范围。

后续规范说明可以补一个目的地/结果异名、可缺省且允许 None 的简例，帮助读者同时遵守登记、唯一命名与 range 规则；它只是既有规则的说明待办，不是本次修稿的前置条件，更不是新增强制登记规则。

本次只读取有关规范、固定来源、当前 checkpoint 和该候选回执；仅写入本讨论记录。没有修改源码、工具、运行状态、产物或快照，没有调用阶段模型、维护/恢复，也没有运行测试。
