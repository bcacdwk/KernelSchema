# Phase 公共定义：六算子调查与建议

**建议保留三类定义，名称改为 `storage / index / event`，补清登记条件和引用语义，删掉没有新增信息的具体条目。** 五位独立调查者在所读六 root 和代表性分支中，都没有找到必须增加第四种 phase 公共定义的反例；名称及个别字段的取舍存在分歧，下面是主代理综合判断。

本页保留三类 phase 定义的调查与当时建议；`storage / index / event` 已进入[正式规范及变更记录](../../schema_docs/CHANGELOG.md)。调查语料是六算子暂停时的旧格式稿，规则采用本页所列快照；旧稿的字段名、批准状态不等于符合现行规范。

后续针对 view 的字段、抽象程度和层级重新评估，见 [view 专项研究](view_review/README.md)；它单独记录删除 dtype 职责、按需保留 part 及不引入槽映射继承的建议。

## 推荐名称与职责

| 当前名 | 推荐名 | 直接含义 | 保留边界 |
|---|---|---|---|
| `resource` | **`storage`** | worker 的片上存储区域及占用账 | 包括 SMEM、TMEM、寄存器状态；不扩成分配器或全部硬件资源表。 |
| `domain` | **`index`** | 需要命名的有序索引、范围和编号关系 | 包括 worktile 序列与 flatten；不是普通变量表或共享可变游标。 |
| `milestone` | **`event`** | 供跨 phase 引用的已有事件集合的名字 | 单事件和多事件都可；由实际动作提供保证，不创建同步动作。 |

三者按需登记，不要求每个 phase 填齐。声明放在 phase 下，实际存储和动作仍按 worker、成员及工作实例解释；登记位置不等于分配 API 或资源寿命边界。

`storage` 比 `buffer` 更自然地包含寄存器状态，比 `allocation` 更少暗示 API 粒度；`index` 比 `range` 更能包含工作序列和连续编号。`event` 最短且能指出引用内容，但第一句必须明确它是已有事件的引用名。备选 `event_group` 更字面，一位调查者推荐它；另一位建议保留 milestone，主要顾虑都是 event 与 CUDA event 对象的混淆。这里选择靠简短定义消除歧义，不为命名再增加结构。

## 建议修改的核心规则

**1. storage 按设计区域记账，保留真正的存储映射。**

- 同一共用区域只计一次，重叠值使用 view；互不重叠的设计区域可以分列，不强求一条记录对应一次分配 API。
- 保留 `view.slot_index / part / dtype`。MLA 的 S/P 逐行字节重解释、Sparse 的 Q/KV/partial 跨槽共用，各有实际用途；槽数与等待关系都不能代替这些映射。
- 已确定的有效分配粒度仍计入容量。例如 MoE 的一块 TMEM 使用 192 列、实际占 256 列，不能把余下 64 列写成已节省的容量。
- 保留决定容量、寿命或重叠的寄存器副本，例如 GEMM 的双缓冲读出、FMHA 的 Q 预取。普通标量临时量、指令寄存器名、barrier 编号和存储字节不逐项列入。
- 保留 `total_size` 的现有必填规则，作为容易读取、可生成和核对的派生值；本轮不采纳将它删除或改为可省略的建议，也不增设第二套字节字段。

MLA 源码一次申请 512 列 TMEM，稿中把互不重叠的三槽 S/P 区与 Y 区分开记录，说明记录粒度不宜绑定 API。FMHA 的设计 SMEM 账为 **83,968 B**，初始实际请求为 **102,912 B**，且后者还能按占用查询增加；影响驻留的额外预留需要在 realization 保留短约束，不能把前者当作完整物理申请量，也不为凑数虚构数学值。见 [ST01–ST06](reports/01_storage.md)、[PG04-06](reports/04_prose_gaps.md)、[N4](reports/05_names_minimality.md)。

**2. index 只保留需要命名的索引，允许单 role 使用工作序列。**

- 多个 role 共用身份、需要跨条目引用，或需要绑定 worktile/flatten 序列时登记；单 role 也可以需要这些定义。
- 局部范围写 `for(z=0:2)`，条件写 `if(Active == 1)`；普通参数、布尔判定不因此进入 index。
- view 键上的变量是形参。局部循环变量传给 view，不应仅因同名就被强制再次登记公共索引。
- 动态端点在实际遍历时必须已知并就绪；索引声明本身不执行读取、领取或等待。

MoE gather 只有一个 role、没有 resource，仍需要 `t: worktile`；这直接说明现有“多 role 或 view 使用时登记”的条件应放宽。FMHA 的 z、GDN 的 r 和 Active 假域、MoE 的常量 c 则提供了具体删减对象。GEMM/MLA/MoE 的跨工作连续版本编号仍须保留，物理槽号不能替代它。见 [I01–I06](reports/02_indices.md)、[N2](reports/05_names_minimality.md)。

**3. event 明确为引用集合，消费端逐事件施加约束。**

- 仅在跨 phase 需要点名局部事件时登记；默认交接、内建事件或其他已声明关系已足够时不另建别名。
- 别名复用被引用动作的实例身份。索引作为形参在引用处代入，不捕获某个 role 的当前循环；跨 phase 的实例匹配沿用已有的值、版本和区域关系。
- `ready / start_after` 引用事件集合时，所需前驱必须全部满足；`wait_at` 引用事件集合时，就绪约束施加到其中**每个**实际等待或读取端点。
- 事件存在性由 pipe 决定；空 worktile 集合不自动删除 worker 入口处仍真实执行的动作。

第三条是最重要的语义澄清。若数据在时刻 10 就绪，两个消费等待点分别在 5、15 完成，只检查“整组在 15 完成”会漏掉第一个点过早通过。集合用于消费目标时必须展开，不能只比较一个组完成时间。两位代理独立指出此问题；主代理的[小模型](data/event_reasoning_check.json)复核了这一反例。见 [EV02–EV03](reports/03_events.md)、[N3](reports/05_names_minimality.md)。

多 consumer 的 wait_at 部分覆盖也宜明确为按消费者覆盖：被点名者使用指定位置，其余仍保留默认入口约束。这样局部调整不会取消其他消费者的等待，也不必为默认位置补造别名；这是规则建议，当前样本未据此做批量改写。

## 确实有可删的，也有不能删的

| 实例 | 数据支持的处理 |
|---|---|
| GDN：八个交接值共用 `chain.prep_ready` | **保留别名**。它指向独立输入准备之后的 PDL 等待，避免在八处暴露 role/action 路径。 |
| MoE `double_verify_root`：Clean 的七个 guarded 别名 | **可删这些别名及该 wait_at**。既有启动链已经保证 verify 完成早于全部消费入口；实际 guard 动作及其他依赖保留。 |
| GEMM/MLA：不同深度的环及跨工作版本 | **保留共享编号和显式落槽**。例如同一 n 可以分别映射 n%3 与 n%2，编号不等于槽号。 |
| Sparse：Q、Gather 环和 FP32 partial 共用 arena | **保留 part/dtype**。用一块区域表达不同阶段用途已足够，无需新增跨槽 alias 字段。 |
| MoE `direct`：prelude/direct 在同一 worker 上接续 | **沿用现有合账与 handoff**。Census/Exchange 只定义一次，不增设 owner_phase 或 scope。 |
| FMHA z、GDN r/Active、MoE 固定 c | **缩回局部 for/if 或直接常量实参**，减少公共索引登记。 |
| GDN final_store 的末轮未使用临时赋值、普通 barrier/地址细节 | **删除实例中的讲解**，保留转置、重叠、存储存续和完成要求。 |

主代理独立复核了 MoE 这一删减：不使用 Clean.handoff 边，仅用现有启动路径及“局部事件发生在所属提交开始之后”，即可推出七个别名内 **19 个事件**都晚于 verify.kernel_done。推导见[结果](data/event_reasoning_check.json)和[脚本](scripts/check_event_examples.py)；不是对整个 kernel 的正确性证明，也不推广到没有这条启动链的分支。

## 文字中还藏着什么

| 文字内容 | 处理位置与理由 |
|---|---|
| 本次从哪个生产者、哪个版本/副本读取，写入哪个 view | noun 的短 describe；这是动作绑定，不能仅靠同名推导，也无需再建一套 I/O 字段。 |
| 独立发布、全部旧读者结束、条件改变等待位置 | 用现有动作、执行模式、for/if 和 waiting 表达。 |
| 原语保证的细部读取顺序 | 保留简短且有依据的保证及 realization 绑定；仅有 done 顺序不一定能推出。 |
| 精确 lane 地址、barrier 编号、临时赋值、普通缓存维护过程 | 交给 Coder；若某项改变容量或交接，则保留那个影响，而非整段实现过程。 |
| MLA 的 Geo 上传先于 main 启动；Sparse 的 caller fork/双输出 join | **真实的外部端点缺口**，在 docs4/调用边界另议；局部 event 不能引用一个未登记的外部供给者。 |
| 跨调用 workspace 身份、CPU 观察哪一代结果、再次复用条件 | 保留必要的短调用约定；不把 storage 扩成 process/cache/allocator 模型。 |
| GDN final-mini、FMHA 空 split、MoE host mirror 的来源限制 | 属于来源或旧稿问题；区域和事件名称不能补出源码不存在的保证。 |

因此，“没提 issue”不能证明结构完备：说明字段允许作者合法写入很多事实，其中混合了必要语义、错放的关系和实现细节。当前数据支持改清边界和使用方法，没有支持把这些内容全部结构化为新的 phase 字段。详细删改清单及真实片段见[文字专项](reports/04_prose_gaps.md)，外部端点论证见[事件专项 EV06–EV07](reports/03_events.md)。

## 最小改名效果

以下是来自真实材料的建议片段，非完整 plan；只展示相关声明，原有动作、等待和工作组织保留。

MoE/direct 的片上存储，登记在 prelude 下：

```yaml
storage:
  Census: {depth: 1, location: SMEM, shape: [], dtype: int32, total_size: 4}
  Exchange:
    depth: 1
    location: SMEM
    shape: [element: 16]
    dtype: int32
    total_size: 64
    view: {WarpCounts: {}, FinalRoutes: {part: {element: [0, K]}}}
```

handoff 的资源引用随表名改为 `tile_pipeline.phase.prelude.storage.Census`。同一 worker 的 direct phase 沿用这些区域。

MLA 的工作与连续版本编号：

```yaml
index:
  w: worktile
  t: [0, "N(w)"]
  n: {flatten: [w, t]}
```

GDN 的消费等待点，登记在 chain 下：

```yaml
event:
  prep_ready: ["{done MAIN[0]::await:Prepared}"]
```

调度层仍写 `wait_at: chain.prep_ready`；别名本身不必跟着表名改变。三处示例的完整原文、上下文和改写边界见[存储报告](reports/01_storage.md)、[索引报告](reports/02_indices.md)、[命名报告](reports/05_names_minimality.md)。

## 数据与独立报告

重点审阅六个 root，并定向审阅 **17 个去重分支的相关片段**。另对 **172 份有设计内容的稿件**做结构扫描：145 个历史 approved、13 个候选、14 个待改；5 份 scaffold、43 个未开始节点不计为有效设计样本。全量扫描与局部语义审阅分别记录，不把 alias/from_parent 的展开量当成作者重复书写量。

六 root 合计 14 个 phase、32 个 role、82 条存储声明、88 个 view、23 个 domain、1 个 milestone；逐算子表见 [census](data/census.md)。扩展到全部有设计稿的本地字段，milestone 有 294 条，来自 GDN/FMHA/MoE；其中 245 条单事件、49 条多事件。所有 alias 键均未带索引，这说明索引事件族缺少本批实证，不能据此判定应删。

六 root 与其原 working checkpoint 的字节一致；18 份入口/Kernel 源码副本与各 run 的 inputs.lock 匹配。本轮当前规则另存 [sources/rules](sources/rules)，位置与 SHA256 见 [corpus.json](corpus.json)。复用了指定的[文字密度调查](../../schema_extract/archive/six-operators/20261109-methodology-review/SCHEMA_VERBOSITY.md)和[十算子文字问题总结](../../schema_extract/archive/20261109-ten-operator-text-audit/SCHEMA_DISCUSSION.md)，没有重新统计其写作谱系或声称新的 GPU 验证。

五位代理均为新上下文、gpt-6-astra、max effort；按根 AGENTS.md 分批运行，每次最多两个。各自先独立提交，主代理随后复核共同证据、容量计算和事件推导并综合。

| 报告 | 独立重点 | 命名建议及主要分歧 |
|---|---|---|
| [01 存储](reports/01_storage.md) | 区域、别名、容量、寄存器副本、跨 phase 共用 | storage/index/event；建议 total_size 可省，综合建议暂不采纳。 |
| [02 索引](reports/02_indices.md) | worktile/flatten、稀疏编号、局部循环、view 形参 | storage/index/保留 milestone。 |
| [03 事件](reports/03_events.md) | 别名价值、范围、目标组展开、可推导删除、外部端点 | storage/index/event。 |
| [04 文字](reports/04_prose_gaps.md) | 旧长说明的去向、可删 lowering、源缺陷与层级边界 | buffer/index/event_group。 |
| [05 最小性与命名](reports/05_names_minimality.md) | 尝试消去概念、首次阅读语义、真实边缘例子 | storage/index/event；保留 total_size。 |

共同输入与引文核验见 [validation.json](data/validation.json)。复核命令：

```bash
python schema_discussion/phase_grammar/scripts/inventory.py
python schema_discussion/phase_grammar/scripts/check_event_examples.py
python schema_discussion/phase_grammar/scripts/check_evidence.py --write-report
```

各报告另附生成/复核脚本和 YAML 草图。检查范围是固定输入摘要、引用行段、例子与局部论证；不代表旧稿已完成新规范迁移，也不替代 Kernel 正确性或完整执行图审阅。完整研究口径见 [METHOD.md](METHOD.md)。
