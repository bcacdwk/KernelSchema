# 第 2 轮：具体设计规则的反例审计与目录试写

依据 [BRIEF](BRIEF.md)、[第 1 轮作者规则](01_schema_rules.md)、[第 1 轮提取方法](02_extract_method.md)，只读核对当前规范及 FA4、MoE、MLA 的少量真实定义。本轮没有修改正式规范、代码、旧 YAML 或运行状态，没有运行提取，也没有重新批准旧产物。

**第 1 轮的主方向可以落地，但须修正五处：** 明确 MLA 这类输入启发式也不能继续选择 YAML 的配置；完整设计准入不能把主调度或主流水线装进一个黑盒后称为局部；目录的最终基线须直接落到真实完整 YAML；空变体节省略；删除“实现任何目录变体都必须先补一份完整 schema”的普遍义务。含 root ≤10、越少越好、相同整区直接 `from` 均保持。**本轮没有必须再由用户选择的内容。**

## 1. 具体配置与输入相关量：看决定的对象，不看公式长相

### 1.1 三个算子的实际反例

| 当前字段或源码事实 | 错误读法及后果 | 最小修正 |
| --- | --- | --- |
| FA4 [split 稿](../../../schema_template/examples/fa4_prefill/splitkv_dual_load.yaml) 的 `Ssplit=min(floor(G/C),floor(Kblocks/2),floor(24000000/(TotalQ*Hq*D*4)),64)`，以及 `Smax=32 if Ssplit<=32, otherwise 64` | “只有整数公式，没有配置枚举”会把并行度、workspace 与 combine 容量的选择器原样保留。它的结果虽是运行时参数，也仍是其他配置选择。 | 完整代表选定一个有据的 `Ssplit` 及其 `Smax`；其他取值、联动与预算规则在一个参数条目记录。不能只去掉 `if`，再写等价的 `min`、查表或自然语言选择。固定来源见 [FA4 heuristic][fa-heuristic]。 |
| 同一 FA4 稿中的 `span=ceil(ceil(L(b)/Bk)/Ssplit)`、`hi=min((s+1)*span,ceil(L(b)/Bk),Nfull)`、`N=max(0,hi-lo)` | “参数都要具体”若被理解成冻结 `N` 或删除 `N=0` 分支，会导致空 split 读取未生产的部分结果，或等待不存在的 operand 操作。 | 固定 `Bk/Ssplit` 后仍按当前请求计算 span、边界和有效工作；保留空 split 的 LSE、未写 partial O 与 combine 跳过零权重项的完整语义。它们不是另选配置。 |
| FA4 [persistent 稿](../../../schema_template/examples/fa4_prefill/packed_persistent.yaml) 的 `worker_count=ceil(min(SMCount,2*Cpad)/2)`、`C_b/Cpad` 与 `worker_assignment` | 机械禁止 `min` 会连已选 persistent 调度的实际工作实例数也禁止；反过来，把任意分块选择称为“实例数”又会放回 selector。 | 当前已选 CTA-pair 组成、worktile 定义和固定步长分配规则不变时，按工作数与已选设备上限产生 worker 数可以保留。若表达式选择的是 split 分块、成员组成、槽深或协议模式，则进入目录。不能仅用字段叫 `worker_count` 判定。 |
| MoE `f4_body_ws` 的 `TSTORE`、`EPI`、`PP` 和静态断言：TSTORE 只接 GEMM2、BN=128、EPI=4、非 PP | 将 `EPI=4 or 8` 留在 role 成员范围，或把 TSTORE 当普通数据条件，会让一份 YAML 包含不同角色组成和写回协议。 | 当前方案明确选择 EPI、TSTORE、PP；其他选择记目录，保留限定条件。来源见 [模板约束][moe-assert]。 |
| 同一 TSTORE=true 路径的 `rows_valid >= (wl+1)*32` 与后续 `tok_ok` | 因为正文包含两种写法，就将全块/尾块拆成两个设计或移出正文；完整 quadrant 的 bulk store 随后可能覆盖下一专家的 route。 | 这是所选写回机制的必要边界处理：完整 32 行才 bulk，尾块 masked STG。保留 source-read 完成许可复用 Sink 与 global-write 完成许可 finalize 消费的区别。来源见 [写回代码][moe-tail]。 |
| MoE [verify_root 稿][moe-verify] 的 `Clean=1-OR_s DirtyPart(s)`、先写 `ZeroOutput`、`Active=1-Clean` | “具体方案不得有选择”若连 Clean 都禁掉，会丢掉验证本身作为输出生产者及其他已提交 grid 的 guard 行为。若反过来把 host 的旧镜像读值叫 Clean，又会补造本次调用的保证。 | 固定“验证并以本次 device verdict 守卫主链”这一机制；保留 verdict 的产生、发布、等待、两种结果及实际副作用。host 是否选择验证图是另一配置/组合问题，`HostObserved` 不能冒充当前 Clean 或 kernel_done。 |
| MLA [root](../../../schema_wiki/archive/six-operators/revisions/2.5.0/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/root.yaml) 的 Loader 定义 `KVSlots = 3 if Scatter, otherwise 4`；schedule 又在 `every q_b>=Bq and not Scatter` 时选 LPT | 这两处同样由当前输入算出，但仍是在三/四槽及 LPT/head 顺序之间选配置。第 1 轮只强调 FA4，容易留下这个看似“自适应”的漏洞。 | 代表固定槽数与表序。`ScatterBytes`、L2 比较阈值和源码如何联动两个选择在目录记录一次；不能因计算 Scatter 的输入来自张量长度，就继续放在 YAML 里选择配置。源码确实据此提交 `<0,3>` 或 `<0,4>`，见 [启发式][mla-scatter]、[表序][mla-order]、[提交][mla-launch]。 |
| MLA 的 `WorkCount=H*sum ceil_div(q_b,128)`、`qv/N`、Queue 成功补领或发布 terminal descriptor；Contract 的 `Causal`、`ReturnLSE` | 一律冻结输入 bool 或删除失败 pop，会把可变任务误当多配置，把必要队列终止误当另一设计。 | 当前算子的输入含义、有效行数、实际工作数与所选队列算法的状态机都保留；`ReturnLSE` 决定契约所要求的输出，terminal 仍须发布完成并排空在途工作。不能因 branch 是动态的就认定它合法或非法，要看它是任务语义/当前机制的执行，还是在选另一个配置。 |

可供作者判断的一句话是：**先明确本份选定的分块、worker 组成、缓冲和机制，再表达这个方案对当前输入产生多少工作、哪些有效值和哪些必要动作；用于挑选前述决定的表达式另记目录。** 它不是编译期/运行期之分，也不是 `if/min` 的语法黑名单。

还须修正一个迁移细节：MLA 旧稿把 `WorkCount/LaunchSMCount` 的定义放在 `when`，正文再写“defined in dispatch.when”。去掉 selector 文字时，真正的工作量和设备参数定义必须移回对应 Design 字段；不能一起删掉，也不能让目录成为正文中工作量符号的唯一来源。`when` 仍可保守选择已核子域，不必追求最大域；但不负责 YAML 内的模式选择。

“已选自适应机制”也不能成为免检名称。Clean/队列例子有本次数据或协议状态的产生者、消费者和全路径后果；MLA 的缓存容量启发式明确选择两套可各自固定的槽深/表序。这些对象足以区分本轮真实例子，不需要创造一个通用的自动分类器。

### 1.2 Hoist 教学保留 for/if，只选一个方案

当前 [DOC5 条件等待示例](../../../schema_docs/5.design.tile_pipeline.md#条件等待示例) 同时解释 `Hoist=0/1`，并非一个由当前数据产生 Hoist 的算法。建议直接选“先等旧累加器释放，再准备 scale”，删除示例的 Hoist 符号，保持现有 `N>0` 的索引声明与动作定义，关键片段改为：

~~~yaml
pipe:
  - for(t):
      - if(t > 0):
          - await:Drain(t)
      - for(g[t]):
          - compute:ScaleTM(g)
          - matmul:PayloadA(g)@PayloadB(g)
waiting:
  released:
    - "{done await:Drain(t)} ← [{done Epilogue[*]::release:Accumulator(t-1)}]"
    - "{issue matmul:PayloadA(g[t,0])@PayloadB(g[t,0])} ← [{done Epilogue[*]::release:Accumulator(t-1)}]"
~~~

配套解释只需说明：首 worktile 不产生 Drain；声明窗口排除首轮的 `t-1`；后续 worktile 的 Drain 完成依赖全部旧 release，serial await 再约束后面的 scale 准备。直接的 release→首 matmul 约束仍保留。这样继续教会 for、if、缺席动作、移位窗口和 issue/done，不再借教学将两套配置并列塞入正文。

原 Hoist=1 的差异可作为一次局部记录：基线是该 MMA 的 Drain、ScaleTM 和首 matmul；允许独立的 ScaleTM 提前准备，取消该显式 Drain，仍保留 release→首 matmul 及 operand 就绪约束；前提是提前 scale 准备不覆盖仍被旧工作使用的存储。无需在 DOC5 再放第二份 YAML，末尾一行指向 common 的目录规则即可。

DOC5 成员示例也改成“当前方案 `EPI=4`，成员为 `warp[3:3+EPI]`”；删除“数量变化可以沿用同一份参数化设计”。`Nt/32` 这类合法端点仍可使用，但必须来自当前选定组成或真正的输入实例化，不能替代其他组成的目录记录。

## 2. 主体/局部闭合判据需要防止两个极端

| 实质问题与反例 | 后果 | 给编辑者的最小修正 |
| --- | --- | --- |
| “存在输入输出接口就局部”。MLA bootstrap 与 persistent 都接受 Geo、产生 O/LSE，也可将整个 main 包成一个对象；MoE mono 与 WS 同样可共享 GEMM 签名。 | 任何主流水线都能被称为替换一个函数，完整设计准入随作者抽象粒度消失，数学不变的调度/协作创新被压掉。 | 局部边界必须落在基线**已经定义的对象与保证**上，边界外保留其主体组织。把主计算、主要领取/退出或整条主循环生产消费组织整体包进新黑盒，不算局部闭合。MLA Queue、提前补领、next-Q 预取与跨工作 ring 寿命，不能只写“bootstrap + queue”。 |
| “影响多个区就完整”。FA4 shared alpha 改 Stats 的数学含义、shape、Softmax 生产与 Correction 消费。 | 数值/存储/协议的一次有界决定又按每种调度展开 YAML。 | 只要原 online 递推、输出归属、Stats capture 后归还及 Y 修正所需 PV 完成边界仍成立，一次局部记录即可。必须说明这几个影响，不能只写 `stat: 2→1`。 |
| “必保留的机制一定给完整 YAML”。十个完整名额用尽，随后发现另一个独立 main 组织。 | 第十一份获得例外，或被伪装为参数；也可能直接抹掉事实。 | 独立性与当前完整交付是两次判断。保留其名称、独特机制、已知范围/依据和未展开原因，明确它没有完整 schema；不强认最近基线，不藏一份全算子 YAML。若要完整交付它，替换当前代表或重新选择集合，始终 ≤10。 |
| 第 1 轮提取稿写“超过候选上限”“进入完整代表候选的集合必须 ≤10”。 | 调研者可能在第十一个线索前停止，或将尚未比较的独立方案偷偷归类为局部。 | “发现/研究的候选”可以超过十个；“拟完整交付的设计集合”和最终在用完整集合含 root ≤10。P2/P4 如何比较选择属于 extract，不需在 common 再写一次调查算法。 |

必要信息可以是完整方案、可核对的差异记录、未完整展开的独立候选或明确来源限制；这些身份并不等价。保留未展开事实也不表示当前任务要求的完整交付已经完成。反之，若任务未要求逐套完整交付这些候选，目录登记并不自动产生未来欠账。

当前没有证据证明“用户点名必须逐套完整交付的不可约方案超过十份”。因此不用先让用户批准一套超限政策；即使以后出现真实冲突，也应拿具体取舍处理，不能使十份上限获得例外。

## 3. 只保留一个目录格式，修掉虚基线和空节

推荐继续用第 1 轮的三部分骨架，其中**范围、完整设计必需；变体与组合按实际内容出现**。条目写清三项事实即可，不增加 ID、版本、审批状态、逐分区小表、必填性能数字或完整源码清单。短条目可合成一段，不要求为了标签填充句子；这不是另一种记录格式。

最终完整设计表列全当前交付的 YAML，摘要足以辨认其具体机制与保留理由；精确参数、when 和版本仍以 YAML 为准。原创作者可以先提出有边界的局部候选并说明推导，不必制造源码出处；提取者则须忠实区分已读出的机制、未研究范围与尚未完整化的候选，不能把源码中的配置关系当成普遍可组合性。

~~~markdown
# <任务名称>

## 范围

当前完整记录的范围，以及影响读者使用的主要缺口。
确有独立但未完整展开的候选时，在此简述名称、独特机制、范围/依据与未展开原因。

## 完整设计

| 设计 | 具体机制与保留理由 | 关系 |
| --- | --- | --- |
| [root](root.yaml) | ... | 阅读入口、唯一 Contract |
| [other](other.yaml) | ... | 独立方案或必要的来源关系 |

## 变体与组合

### <有意义的名称>（参数 / 局部 / 组合）

- 基线：完整 YAML 链接及其中已有对象；多基线直接并列。
- 差异与联动：相对这些基线的变化及必要配套，已有定义直接引用。
- 条件与依据：适用的接口/范围、已有依据及影响使用的缺口。
~~~

格式必须同时守住以下边界：

1. **最终基线直接落到完整 YAML。** 局部/组合标题可相互引用已有决定，但不能靠“上一变体的下一变体”才找到完整基线。组合直接列完整基线和原始参数/局部条目，不再把组合层层包装成新基线。引用用于说明事实，没有 patch 应用顺序。
2. **同名不等于同对象。** 同一条目适用于多个基线时，直接说明对象对应及必要区别，如 MLA persistent 的累计 `n` 与 bootstrap 的局部 `t`；不可写“两个文件按同样行号替换”。不额外创设一套全局组件身份。
3. **相同轴一次写完。** 基线并列，共同差额只写一次；只有某个基线有额外联动才在本条指明。组合只写已知相容关系或新限制，不将两个基线、两种槽数、两种表序展开成八份记录。
4. **依据按实际有的写。** 源码事实固定来源；原创设计可以给推导与待验证判断，不能被要求补一个不存在的 source。成立条件、原 selector 何时选择、实验覆盖是三种事实，知道哪种就说明哪种，不制造三个必填子栏。
5. **身份与范围不隐式改变。** 目录不修改当前 YAML 的参数或 `when`，也不赋予变体完整计划身份。变体可讨论基线 `when` 外的方案，但须说明自己的条件及证据；不能默默继承原 selector 的全部限制，也不能默默扩大基线范围。

第 1 轮 FA4 试写的“也适用于 folded static 变体”是具体缺陷：完整表没有该基线，前文也没有闭合解释从矩形 root 到 folded static 的坐标发布；随后 alpha 和组合再引用它，读者无从确认需要哪些改变。最小修复是在一条“folded static 坐标发布”记录中直接链接 root，指明 folded 映射、elected thread 的坐标解析、WorkCoord 发布及首次加载等待；其后组合仍直接列 root 与这条原始差额。不能为了修链接而新增一份没有必要的完整 YAML，也不能把“来源存在这份旧文件”当作当前完整基线。

没有变体时直接省略第三节；没有独立未展开候选时也省略对应文字。无需“当前未记录其他变体”“暂无独立候选”等填充句。作者仍须如实说明影响当前声明的实际缺口，省略空节不是隐瞒已知缺口。

### 3.1 MLA 的实际短试写

下面以真实 [MLA root][mla-root]、[bootstrap][mla-bootstrap] 及其源码为材料试写。设后续将它们各整理为一份 **FP8 PV、Bq=Bk=128、四槽 K/V、稳定 LPT 表序** 的具体版本；`root.yaml/bootstrap.yaml` 在代码块中是该拟定交付的文件名。此处没有创建 YAML，也没有声称旧文件已符合新规则；旧稿中的 KVSlots/排序 selector、语法与已知来源限制仍须后续处理。设计表展示最终目录的形状，不能作为现有文件的合规证明。

~~~markdown
# Expanded MLA FP8

## 范围

当前两份完整设计都选择 FP8 PV、Bq=Bk=128、四槽 K/V 与稳定 LPT 表序；
输入范围以各 YAML 为准。本文还记录同一实现中的三槽与 head 顺序变化。
旧源码已披露的 TMEM/描述符寿命限制继续由受影响设计说明，不因改用目录而消失。

FP16 V/PV 路径尚未完整展开：它物化 FP16 V 副本并改变 PV 数值路径，属于独立候选；
本次集中记录 FP8 路径，仅保存其[来源][fp16]与该差异，未交付该候选的完整 schema。

## 完整设计

| 设计 | 具体机制与保留理由 | 关系 |
| --- | --- | --- |
| [root](root.yaml) | persistent CTA：最后一轮提前补领，Queue 发布下一项或终止描述符，next-Q 与旧工作排空相接 | 阅读入口，保存唯一 Contract |
| [bootstrap](bootstrap.yaml) | 一项一 CTA：没有补领队列和跨工作缓冲接续，初始化与退出围绕单项闭合 | 数学整区相同，直接 from root；调度与流水线本地定义 |

## 变体与组合

### K/V 缓冲槽数（参数）

- 基线：[root](root.yaml)、[bootstrap](bootstrap.yaml) 的 main.Keys/Values、Loader 的 KV 加载及 MMA 的 PV 完成；root 用跨工作累计 n，bootstrap 用单工作 t。
- 差异与联动：四槽改三槽时，两块容量、取模和重用距离一起改为 3；三槽可用与三槽 S/P 同周期的 PV 完成通知。四槽仍须匹配四槽周期的释放通知，不能只改 slot_count 而沿用不匹配的 barrier 周期。root 继续跨工作累计，bootstrap 仍在本 CTA 内结束。
- 条件与依据：保持 FP8 K/V 表示、相同消费与旧槽读完再覆写的保证。[配置和释放代码][ring]提供两种实现；原实现中的组合范围见下条组合记录，源码存在不等于所有槽数/表序组合都已测得正确或更快。

### 保持 head 顺序（局部）

- 基线：[root](root.yaml)、[bootstrap](bootstrap.yaml) 的 main 调度中 Geo 表序、Item 与表上传交接。
- 差异与联动：不执行 LPT 重排，保留 h 升序、b 升序、a 降序的 Geo 表。每项完整的请求、head、query 区间和绝对输出位置一起保留；上传完成后主 kernel 才消费。root 按此表补领，bootstrap 的 CTA 直接领取对应表项；各自输出归属和退出协议沿用基线。
- 条件与依据：工作集合与每个有效输出恰好一个归属不变，变化在完整 Geo 表上传边界闭合。[原表序代码][order]给出该模式；不将 LPT 性能选择条件当作正确性的最大有效域。

### 槽数与表序的已见组合（组合）

- 基线：[root](root.yaml)、[bootstrap](bootstrap.yaml)，以及上面的“K/V 缓冲槽数”“保持 head 顺序”两项。
- 差异与联动：两个决定分别作用于 K/V 环和 Geo 表；沿用各条已说明的接口及基线自身的索引、跨工作寿命，不另复制差额正文。
- 条件与依据：在 FP8 PV 入口，源码令 Scatter = CachedL2>0 且 4*min(H*B, CachedSM 为正时取其值否则取 148)*max_b(k_b)*320 > 3*CachedL2。Scatter 为真采用三槽+head 顺序；为假且所有 q_b≥128 采用四槽+LPT，否则采用四槽+head 顺序。这一关系对 queued 与 bootstrap 共用，见[选择及提交][select]。这是原选择依据，148 不是已证明的设备属性；本条没有登记三槽+LPT，也不改变当前两份 YAML 的 when。

[ring]: https://github.com/bcacdwk/KernelSchema/blob/448ab852500e244b07dff024418c27e6d097f320/kernel_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/candidate/solution.py#L1384
[order]: https://github.com/bcacdwk/KernelSchema/blob/448ab852500e244b07dff024418c27e6d097f320/kernel_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/candidate/solution.py#L1763
[select]: https://github.com/bcacdwk/KernelSchema/blob/448ab852500e244b07dff024418c27e6d097f320/kernel_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/candidate/solution.py#L1691
[fp16]: https://github.com/bcacdwk/KernelSchema/blob/448ab852500e244b07dff024418c27e6d097f320/kernel_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/candidate/solution.py#L1904
~~~

试写中的引用位置在本轮已逐段核对：`ring` 可对照 [Cfg 说明][mla-ring-config]、[Loader 重用][mla-ring-reuse]；`order` 对应 [表构造与 LPT][mla-order]；`select` 可继续对照同一文件的[共同提交][mla-launch]；`fp16` 对应 [V 副本准备][mla-fp16]。当前源码的 FP8 kernel 模板确实在 queued 与 bootstrap 两种 grid 中共享三槽/四槽提交，不是根据旧 YAML 名称推测。

这个短例有两份完整主体、一个参数轴、一个局部变化和一条组合关系；没有创建跨基线的参数笛卡尔积。两种槽数的必要协议差异保留在参数条目，两个主体的生命周期差异仍留各自 YAML。它也说明“局部”并不等于“没有需要认真审的正确性细节”。

## 4. 实施选择不应新增普遍补 YAML 与审批义务

第 1 轮末句“确需实现某个目录变体时，Planner 先选定配置并落实为当前完整设计，再交 Coder”过强。反例是任务明确要求只测某个 KVSlots 变化：真正缺的是本次具体取值、匹配的释放协议及实验范围，并不是所有 Markdown 笔记都欠一份完整 schema。现行 [IMPLEMENT_SCHEMA](../../../IMPLEMENT_SCHEMA.md) 已要求按选定设计实现、关键决定不明时反馈，并明确任务沟通即可承载反馈，无需固定报告或额外审批。

作者规则采用以下替换文字即可：

> 实施当前 YAML 时遵守其完整有效 Design；目录中的其他变化不会自动应用。任务明确选择某项参数、局部变化或组合时，应说清本次具体选择、必要联动及适用范围，使实施者不必补选设计。只有任务确需交付该方案的完整 schema 时，才将实际选定方案整理为完整设计，并继续遵守数量与独立价值规则。

角色权限仍按现有 README：需要改变 Design 的决定仍由 Planner 承担，Coder 不因阅读目录取得自行改设计的权限。明确的既有任务选择无需再次审批；若关键设计决定确实未定，按原反馈路径处理。**common 只写内容与身份关系，不写“Planner 先做 A、Coder 才可做 B”的新工作流。**

## 5. from 与四种身份：不要让复用目标反过来制造设计

| 对象 | 身份与范围 | 不允许的替代 |
| --- | --- | --- |
| root | 唯一 Contract 加一份真实、具体、完整 Design；默认阅读入口，其范围由自身定义说明 | 全域兜底、把所有配置拼成通用 root、为了供引用而只写抽象区 |
| 非 root 完整设计 | 同一 Contract；有自身独特价值、具体选择、when 和必要 Realization；本地内容加实际整区引用闭合；计入 ≤10 | 仅换参数/来源组合就取得身份；未完整说明的新主体仅靠 MD 条目补齐 |
| 独立未展开候选 | 有名称和有据的独特机制，可以没有可局部继承的基线；范围与缺口明示 | 伪装成局部、假装已批准的完整文件、作为 from provider，或在 Markdown 藏第十一份完整方案 |
| 参数/局部/组合记录 | 直接定位完整基线并保存必要差额与条件；不自动修改任何 YAML | plan ID/版本、隐式 patch、目录顺序产生覆盖、所有笔记都成为待完整化任务 |

相同完整区直接 `from` 是要求，不是可选优化。比如 FA4 root 与 packed persistent 的完整数学定义相同，不能因“正文更好读”继续复制；将其余不同区本地写全即可。MLA bootstrap 与 persistent 的数学可共用，并不要求二者共享调度或流水线。

这里要保留第 1 轮对外部绑定的正确解释：共享数学区通过既有 `bind:design.work_schedule::Bk` 使用当前节点的 **已选** Bk，不要求 provider 的外部 Bk 恰好同值；但全部公式、值含义、有效域、精度和边界必须仍适合当前节点。若共享区内部自己固定了四槽或 Hoist 模式，就不能靠目录覆盖；也不能专为共享把它改写成未选的 `Hoist∈{0,1}` 或 `KVSlots∈{3,4}`。

有一行必须变就意味着该整区不相同；真正决定保留的独立设计可将这个区域完整本地化，其他共同区仍直接 from。减少微调重复的首要方式是不给每个微调完整文件身份。不能新增只有 provider 用途、没有自身完整设计价值的节点；多个 provider 的组合本身也不构成新增身份。

## 6. 第 3 轮最小落地内容

### 6.1 common 应有的规范句，不搬入研究算法

以下内容足以完成规则增量；已有字段、来源、when 和整区语法尽量原位修订，不把本报告全文搬进去。

1. 每算子的在用完整设计含 root 最多十份，能少则少；每份在自身范围内由本地定义与显式整区引用构成一套具体闭合方案，不能依靠目录条目补齐。
2. 当前 YAML 定义自己选定的配置及由输入确定的尺寸、工作与必要条件；其他参数取值、局部改动、已有决定组合在 dispatch.md 记录。配置选择无论写成开关、公式或查表都不进入正文。
3. 必须保留的信息不一定取得完整 YAML 身份。局部变化围绕基线已有对象说明全部联动，边界外主体组织继续成立；重写主要计算、工作归属/领取或主循环协作的变化可作为独立完整候选，数学相同不排除它。
4. dispatch.md 的范围与完整设计表必需；有变体才出现变体节。条目说清完整基线及对象、差异与联动、条件与依据；同一轴一次定义，多基线直接并列，组合只补必要关系，链接与排列不构成 patch。
5. 未选择完整展开的独立候选明示独特机制、范围/依据及缺口；它不是局部记录或已完整交付的设计，不为它保留假 YAML 身份，也不绕过总数上限。
6. 当前 YAML 的范围以自身 when 为准；目录里的变体、原选择策略和实验不能隐式改变它。范围允许重叠，root 不自动兜底；记录缺口不解除实际任务的完整交付要求。
7. 共同完整区域直接 from；已固定的区内决定不能被覆盖，不为复用重新引入配置开关或无独立价值 provider。允许当前节点对既有外部参数绑定作具体定义，并审阅整个区在该上下文中的适用性。
8. 完整 schema 只整理实际要求完整交付的方案；目录笔记本身不产生未来完整化或审批义务。记录仍须保留受影响的数值、输出归属、初态、就绪、复用与完成保证，已有实际副作用不会因记录形式改变而消失。

“源码先看什么、如何找轴、何时停止调查、P2/P4/P5/P6 怎么交接”都不属于上述规范句。原创作者不应先沿 host 选路扫描、提交固定源码比较表或走一次提取阶段才有资格使用 common。

### 6.2 可直接执行的编辑清单

| 文件/当前位置 | 修改动作 |
| --- | --- |
| common 的“单份 plan”与“文件与共享 Contract” | 写入一份具体设计、含 root ≤10、越少越好；保留唯一 Contract 和实际完整 root。不新建候选 YAML 字段。 |
| common 当前 110 行、114 行、120–122 行 | 110 行的源码调查先后顺序移 P2；114 行“参数变化保留在同 YAML”改为当前值在 YAML、备选在目录；补上当前工作量/算法条件的区别与主调度/主流水线反例，避免只保留“结构变化”这个含混标题。 |
| common 当前 140–156 行 | 用第 3 节的唯一格式替换松散目录说明；空变体节可省略；未展开独立项放范围；保留局部不创建 plan 身份。实施文字用第 4 节替换，不加入 Planner/Coder 新关口。 |
| common 当前 158–177 行 | 保留三区整区 from、root 全本地、同目录、实际引用图无环；补“相同整区直接共享”和禁止可选配置/provider 造节点，不添加 patch。 |
| DOC5 当前 572–574、664–695 行 | EPI 示例只选 4；Hoist 示例采用第 1.2 节片段，保留 for/if、窗口、issue/done 和 release→首 matmul。不是删除控制语法。 |
| DESIGN_SCHEMA / OPTIMIZE_SCHEMA / schema_template README | 删“参数能表达就继续并入同节点”、强制互斥覆盖或不同 workload 才能并列等旧表述；指向 common 的具体方案与目录规则，不重复一套三项模板。 |
| IMPLEMENT_SCHEMA | 同步 from 和实际内容来源；加第 4 节的选定方案说明，保留原角色权限与反馈，不新增每项 MD 笔记补 YAML/批准义务。 |
| P2/P4 与 REVIEW（由流程侧编辑） | P2 识别机制与选择有限完整集合，候选线索可超过十个；P4 用真实 root 与明确的拟议非 root 整理记录，不重复全源码分组。草稿可以说明尚未写出的计划基线，但不得宣称对象已存在；P5 落实后，P6 核对最终真实对象和完整文件。 |
| examples README 与集中工具能力说明 | 明示旧例尚未迁移，FA4 prefill 仍有十一份；旧 checker 的成功不证明新规则、from 或目录格式已经实现。只改说明，不更改旧 YAML 或启动工具迁移。 |

P2/P4 的具体读深、收敛与返修安排由 [02_extract_method](02_extract_method.md) 及同轮流程审计落地；common 不复述其阶段次序。第 3 轮可据上述规则直接编辑，没有需要先追加用户确认的格式分歧。

[fa-heuristic]: https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/flash_attn/cute/interface.py#L562
[moe-assert]: https://github.com/bcacdwk/KernelSchema/blob/448ab852500e244b07dff024418c27e6d097f320/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu#L4208
[moe-tail]: https://github.com/bcacdwk/KernelSchema/blob/448ab852500e244b07dff024418c27e6d097f320/kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/candidate/nvfp4_moe.cu#L5200
[moe-verify]: ../../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/verify_root.yaml
[mla-root]: ../../../schema_wiki/archive/six-operators/revisions/2.5.0/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/root.yaml
[mla-bootstrap]: ../../../schema_wiki/archive/six-operators/revisions/2.5.0/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/bootstrap.yaml
[mla-scatter]: https://github.com/bcacdwk/KernelSchema/blob/448ab852500e244b07dff024418c27e6d097f320/kernel_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/candidate/solution.py#L1691
[mla-order]: https://github.com/bcacdwk/KernelSchema/blob/448ab852500e244b07dff024418c27e6d097f320/kernel_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/candidate/solution.py#L1763
[mla-launch]: https://github.com/bcacdwk/KernelSchema/blob/448ab852500e244b07dff024418c27e6d097f320/kernel_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/candidate/solution.py#L1995
[mla-ring-config]: https://github.com/bcacdwk/KernelSchema/blob/448ab852500e244b07dff024418c27e6d097f320/kernel_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/candidate/solution.py#L399
[mla-ring-reuse]: https://github.com/bcacdwk/KernelSchema/blob/448ab852500e244b07dff024418c27e6d097f320/kernel_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/candidate/solution.py#L1384
[mla-fp16]: https://github.com/bcacdwk/KernelSchema/blob/448ab852500e244b07dff024418c27e6d097f320/kernel_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/candidate/solution.py#L1904
