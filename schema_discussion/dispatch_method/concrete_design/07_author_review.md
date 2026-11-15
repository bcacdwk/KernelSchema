# 第 4 轮：独立作者视角验收

**结论：当前作者规则可接受，满足本轮用户预期。必改项 0，可选修改项 0；没有未决实质问题，也不建议为凑意见增加规则。** 这个结论针对方法与 Markdown 说明，不表示旧算子、starter 或 checker 已迁移。

先读 AGENTS.md、当前 common、DOC5、设计／优化／实现入口及模板与必要分区定义，完成下面的作者演练并向主代理报告初步结论；之后才读 PLAN.md 和 03_rule_audit.md 对照。未读另一位验收者的报告，未派子代理。

## 1. 原创单 root 的开工演练

给定一个已完成的 Contract：有限 FP32 向量 X，长度 N>0，输出 Y 为逐元素绝对值，设备包含 CTA/warp，参考与验收由 Contract 给定。没有可借用的设计源码。我选择固定 256 元素一份工作、每 CTA 四个 warp、一个协作 MAIN role，直接从 Contract 计算输出。N=257 时仍是同一方案，产生两个 worktile，第二份只访问最后一个有效元素。

以下是实际试写的元数据和完整 Design 片段；给定 Contract 不在审查报告中重复抄写。这不是新建的正式算子文件，也没有经过旧 checker 验收。

```yaml
meta_data:
  name: Absolute value
  describe: Elementwise absolute value of a finite FP32 vector
  version: 1
design:
  math_reshape:
    thesis: Preserve the elementwise computation without intermediate materialization
    rewrite:
      - name: Preserve absolute value
        math: |
          Y[i] = abs(X[i]), 0 <= i < N
  work_schedule:
    thesis: Assign one fixed 256-element worktile to each CTA to expose independent memory traffic
    step:
      MAIN:
        describe: Compute the valid output elements of one contiguous worktile
        decomposition:
          partition:
            element: {meaning: Contiguous output element interval, size: N, tile: 256}
          worktile:
            invariant:
              X: {meaning: Input interval of this worktile, shape: [element: 256], dtype: ref:contract.algorithm_requirement::X.dtype}
          target:
            single_tile:
              Y: {meaning: Output interval of this worktile, shape: [element: 256], dtype: ref:contract.algorithm_requirement::Y.dtype}
        mapping:
          worker: {unit: CTA, unit_count: 1}
          worker_count: worktile_count
        schedule:
          worktile_order: [{by: element, direction: ascending}]
          worker_assignment:
            preassigned: {start: worker_id, count: 1}
            refill: none
            exit: assigned_done
    flow:
      launch: [{step: [MAIN]}]
  tile_pipeline:
    thesis: Use one cooperative role without inter-role handoff or a reusable shared buffer
    step:
      MAIN:
        role:
          MAIN:
            info:
              describe: Read the valid input interval and write its absolute values
              member: warp[0:4]
              using_verb:
                compute:
                  Y:
                    execution_mode: serial
                    describe: Read X directly from the Contract input and write Y by the upstream definition within the current worktile's valid interval; never access padded elements
            pipe: [compute:Y]
```

这次试写不需要 `meta_data.source`、`meta_data.dispatch`、`realization`、额外 symbol、空 waiting 或虚构的 Load/Epilogue role。依据是 [common:36–41、51、69](../../../schema_docs/0.common.md#元数据字段)、[DOC3:58–66](../../../schema_docs/3.design.math_reshape.md#主体字段)、[DOC5:76–108](../../../schema_docs/5.design.tile_pipeline.md#role-字段)和 [DESIGN:53](../../../DESIGN_SCHEMA.md#5-完成交接检查与交付)。如果原始任务连参考入口也没有，DOC1:81 允许先由 compute 写出并固定参考实现；这是既有验收要求，不是强制存在设计源码，也不要求走提取流程。

该任务的 dispatch 只需以下内容，沒有额外空节：

```markdown
# 逐元素绝对值

## 范围

当前设计承担 Contract 的全部有效输入。

## 完整设计

| 设计 | 具体机制与保留理由 | 关系 |
| --- | --- | --- |
| [root](root.yaml) | 固定 256 元素一份工作，每 CTA 四个 warp 协作直接写回，保留简单且无跨角色交接的完整方案 | 阅读入口，保存唯一 Contract |
```

[common:144–170](../../../schema_docs/0.common.md#选择条件与索引)明确只有“范围”和“完整设计”必需；[DESIGN:5–7](../../../DESIGN_SCHEMA.md)也允许简单设计与逐层自检，不新增审批。此场景的现规则足够。

## 2. 固定配置、输入工作量与等待语义

| 具体演练 | 按现规则得到的写法与后果 | 判断依据 |
| --- | --- | --- |
| 固定 Bk=64、KV 槽数=4、split=4、EPI=4，再讨论其他取值 | 四个当前选择在各自定义位置固定；其他取值和容量、槽映射、释放距离等联动进入一次性的参数记录。改写为 `min`、查表或自然语言也不能把 selector 留回 YAML。 | common:110–116 |
| 同一固定方案处理不同序列长度、尾块与空 split | `ceil_div` 工作数、span/lo/hi、有效区域和 N=0 的实际动作留在 Design。空 partial 不得被当作已经产生的数据。 | common:112；DOC4:363–367；DOC5:297–322 |
| 当前设备判定 Clean、队列成功／失败和 terminal descriptor | 保留判定的产生、发布、必要等待、相应动作及退出；这些是所选算法的执行状态。 | common:112、180；DOC4:213–230；DOC5:271、347 |
| DOC5 固定等待例子，分别展开 t=0 和 t=1 | t=0 不产生 Drain；t=1 的 Drain 可以提前 issue，但 done 等前一工作的全部 release，serial 再约束 scale 的 issue。首 matmul 的直接 release 依赖仍保留。其他操作数就绪约束没有被取消。 | DOC5:655–697 |
| 成员端点使用 EPI 或 Nt/32 | EPI 示例明确为已选 4；表达式只引用已定义组成或实际输入实例化，不能以空区间、最大成员范围加文字取消事件的方式暗藏另一配置。 | DOC5:572–574 |

用少量实际材料复核了上述判定：FA4 旧 split 稿的 [105–108、131 行](../../../schema_template/examples/fa4_prefill/splitkv_dual_load.yaml)同时包含 Ssplit/Smax selector 和必须保留的 span/尾块/空工作；两者可按“决定的对象”分开。MLA [实际提交的 2019–2024 行](../../../kernel_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/candidate/solution.py)明确选择三槽或四槽，不能因 Scatter 源于输入便继续选配置。MoE [4233–4234、5163–5189 行](../../../kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu)将 TSTORE/EPI 配置约束与当前完整 32 行／尾块写回区分开；固定 TSTORE 后仍须保留后者。没有发现需要新增规则的反例。

## 3. 多基线记录、组合与完整设计名额

纸面设立两个已经完整写出的基线 A、B：A 每 CTA 只处理一项；B 有队列、提前补领及跨工作环寿命。二者都是固定 Bk 与槽数的当前方案。这是分类演练，不把计划中的 A/B 冒充真实交付文件。

- 调整 K/V 环槽数时，只写一条参数记录，直接列 A/B 各自的环、加载及释放对象；共同的容量、取模与复用距离差额只写一次，单工作索引与跨工作累计索引的区别在该条说明。
- Softmax 产生共享 alpha、由 Correction 消费时，只写一条局部记录，落到两份基线已有的值、动作与存储，保留数值、载荷、生产消费及归还边界。改变多个区域本身不要求增加完整 YAML。
- 组合上述决定时，仍直接列 A/B 和原始两条记录，只增加已核对的相容关系与限制；不能再让“组合二”以“组合一”为唯一基线，也不能按条目顺序应用 patch。
- 如果 A/B 其实尚未完整存在，不能在最终目录中据此建立记录；若 B 需要重新贯通主要领取、主循环及跨工作寿命，也不能先把 B 包成一个“队列对象”再声称是 A 的局部改动。

[common:114–124、174–182](../../../schema_docs/0.common.md#局部记录与完整方案)足以决定这些写法；其中明确保护数学相同但主调度／主协作不同的独特设计。对照 03_rule_audit 的 MLA 双基线试写后，结论一致，没有依赖旧报告来补规则。

另演练了已经保留十份完整设计后发现第十一项独立主机制：按 [common:130、168、178](../../../schema_docs/0.common.md#文件与共享-contract)，可以替换已有完整代表，或在范围中诚实说明它的独特机制、依据和未完整展开理由；不能伪装为参数、拆目录、保存为“历史”完整文件或在 MD 藏一份全算子定义。记录未展开也不解除任务明示的完整交付义务。这个规则没有上限例外，也没有强迫每项候选取得完整身份。

## 4. 整区 from 与正式定义

把 A 的数学整区通过 `bind:design.work_schedule::Bk` 使用 A 的 Bk=64，让有独特主调度的 B 用相同完整数学定义、在自己的调度中固定 Bk=128：依 [common:186–203](../../../schema_docs/0.common.md#design-整区引用)与 [DOC3:141–173](../../../schema_docs/3.design.math_reshape.md#引用与缺项)，B 可以直接 `from: A`。相同的是整区定义，外部绑定在 B 的完整上下文解释；B 仍是一套固定配置。必须核对边界、数值、接口和余下两区，而不是只比符号名。

反过来，若被引用的流水线整区内部固定 depth=4，B 不能依目录把它改成 3；有任何内容需要改动，就完整本地化该区，其余相同区继续 from。root 全本地、不能建立只有 provider 用途的节点、不能靠三区来源不同制造完整身份，均有直接规定。没有片段覆盖空间。

MLA 旧 root 的 [8–14、317 行](../../../schema_wiki/archive/six-operators/revisions/2.5.0/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/root.yaml)把 WorkCount/LaunchSMCount 定义放在 when。按 common:112、142、178、199 的现规则，删除原 selector 时仍须在相应 Design 定义这些工作量及其必要来源；不能只移到 dispatch，也不能从 provider 的 when 自动继承。DOC3:143、155 又要求语义引用解析到实际定义，源码函数不能冒充公式。现规则能够阻止迁移时把必要工作量一并删掉。

## 5. 实施义务和工具状态

[IMPLEMENT:9–13、44、63](../../../IMPLEMENT_SCHEMA.md#1-确认任务设计与实现起点)要求明确本次选择和权限，目录不会自动选配置或让 Coder 改 Design；已授权的局部实验无需先补成完整算子。[OPTIMIZE:47、75](../../../OPTIMIZE_SCHEMA.md#3-按影响范围逐层修订)让同一设计原位修订，其他配置／局部／组合记录一次，并按实际使用关系核对影响。未发现另加每条笔记补 YAML、固定报告或审批义务。

模板 [23–30、52–80 行](../../../schema_template/README.md#2-创建文件或进入已有任务)、[工具说明:3–7](../../../schema_template/tools/README.md)与 [执行器对齐:3、15–18](../../../schema_extract/EXECUTOR_ALIGNMENT.md)明确当前 starter、字段结构、parent/from_parent 解析与新方法尚未对接。视图与检查命令有实际协议的范围限定，不能据此声称当前 checker 支持 from。

[examples:3、11–16](../../../schema_template/examples/README.md)明确旧 YAML/PNG 未迁移、未按新规则验收，并保留 FA4 prefill 的真实 11 份及总计 21 份。只读计数也得到 11+7+2+1=21；这里的旧数量是现状说明，不是新上限的例外。工具实现及冻结协议中的旧名称没有被当成作者规范残留。

## 6. 审查范围与最终判定

完成了当前 common、DOC5、DESIGN/OPTIMIZE/IMPLEMENT、根 README、模板及 examples/tool 状态说明的作者视角审阅；按需读取 DOC1、DOC3、DOC4 和执行器对齐入口。随后对照 PLAN.md、03_rule_audit.md，并只读抽查上述 FA4/MLA 旧 YAML 与 MLA/MoE 局部源码。未进行提取流程全阶段验收，也不以本报告替代另一位验收者的提取视角审查。

自写的只读 Markdown 检查覆盖 15 个当前文档、262 个代码围栏外的本地文件／章节链接，未发现缺失文件或锚点；示意 `root.yaml` 等代码块内容没有当成实际文件链接。没有运行 checker、extract、绘图、GPU 或生产测试，没有修改正式文件、源码、YAML、运行状态或历史材料；唯一写入为本报告。

**作者规则验收通过。没有尚待修正的实质问题，也没有需要新增用户决定的事项。** 后续仍需按已声明范围对接工具并迁移实际算子，不能把本次文档验收当作那些工作已经完成。
