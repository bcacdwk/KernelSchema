# 四模板源码调查：while 是否必须进入设计级 pipe

本报告独立调查 FA4 prefill、FA4 decode、GDN prefill、QSA prefill。结论采用 [METHOD.md](../METHOD.md) 的抽象层与消去判据；未修改任何算子、trial 或正式规则。冻结源来自 `01fff90d714abc6e84904ea9c44523ba66d34a0e`。路径、哈希、循环分组与逐位置清单见 [01_templates.json](../data/01_templates.json)。

## 判断

**四个实际 root 在当前动作粒度下，没有发现必须增加 pipe `while` 才能表达的循环。这个结论不能推出“所有合理算子都不需要 while”。** 原因不是现有 trial 恰好只用了 for/if，而是逐项控制来源不同：FA4 root 是单份工作加输入确定的 KV 遍历；decode 是长度、split 与流水填排决定的遍历；GDN 是固定工作分配与固定 chunk 数的递推；QSA 的探测是 Hash 动作内部算法，票据循环则实现已有动态工作调度。

三个需要保留的反向约束：

- **QSA 的票据 while 是真实跨 role 反馈，而且是合理设计。** 不能说 while 出现在源码里就说明设计坏了；本例能用动态 `for(t)`，依赖的是已经独立定义的实际工作序列，以及没有被省掉的票据动作。
- **哈希探测也是真实结果反馈。** CAS 返回值决定继续探测；它并不是一个事先已知成功轮次的 for 域。root 的容量约束给出真实上限，但在实现层改成 `for+if` 仍可能增加大量空检查。无需把这个实现选择提升成 pipe 控制。
- **空遍历不等于没有协议。** QSA 的 terminal ticket 与 FA4 扩展 sparse 路径的 empty tile 都有必须保留的交接。删掉这些事件的所谓 for 改写不成立。

## 材料、路径与覆盖边界

以下缩写均指 `schema_discussion/pipe_grammar/trial_roots/sources/` 内的冻结目录：`P=template_fa4_prefill`，`D=template_fa4_decode`，`G=template_gdn_prefill`，`Q=template_qsa_prefill`。表中的 `file:line` 可按此前缀复核；JSON 使用完整仓库路径。

| family | 实际 root 路径和选择依据 | 与库中其他路径的区别 |
| --- | --- | --- |
| FA4 prefill | `P/solution.py:95–109` 调用 `_flash_attn_fwd`，FP8、hd256、page64、causal、pack_gqa；`interface.py:951–979,1549–1580` 选择 `sm100_hd256_2cta_fmha_forward.py`。root 的 `not Split` 与 batch 条件排除 flatten/persistent，启用独立 K/V loader。 | 实际 root 外层 while 只容纳一次静态工作。hd256 文件 `:351–353` 明确拒绝 block sparsity；`flash_fwd_sm100.py`、MLA、backward 中的 while 不能算该 root 的机制。其他已写 family 节点的 persistent 是固定步长分配；CLC 环境分支不是当前 root。 |
| FA4 decode | `D/solution.py:10–39` → `operator.py:269–288,727–749` → `gqa_decode_paged.py:534`，q4、非 compact、单 Softmax 组、global partial + 独立 combine；combine 位于 `gqa_decode.py:382`。 | 全部六个冻结文件没有 Python while/break。`try_wait` 是原语探测 token，随后交给 `wait_and_advance`；不是 role 重试整个工作。BLASST 是库内可选分支，编译实参 `operator.py:379` 为 None，root 不开启。 |
| GDN | 实际入口是压缩后的 `G/solution.py:812`，强制 M64；`:782` 的 B>1、HQ=4/HV=32 选择非 static descriptor、非 compact event。`build_solution.py:19–22,92–100` 说明它由 scheduler/core/adapter 拼接和压缩而来，正文用可读 core 与入口中的对应调用交叉核对。 | 两份文本的循环不能算两套机制。core 的六个 kernel role while 对应入口 `:164,174,193,203,222,244`；`mma_issuer_warp` 的第七个 while 在冻结包内仅发现定义，没有调用，不计入实际 root。 |
| QSA | `Q/solution.py:7` → `qwen38_core_qsa_cutedsl/operator.py:139–159`，先 union preprocess，再 persistent attention。root 为 Br=10、SelectedWidth=2051、Rows=4096、KVHeads=2。 | preprocess 的主 quad 探测与 `_insert_lane` 是同一种开放寻址机制的两个入口；合法 root 的 quad 打包与有效页约束走主路径。attention 的 Scheduler、Softmax、MMA、Gather 所见 while 共同实现一条工作票据协议。 |

对 manifest 的 **77 个冻结源码文件、4 个 baseline** 重新计算 SHA256，均匹配。先用 `rg` 定位 while、break、active/keep flag、动态 `cutlass.range`、PTX 分支和等待，再用 Python AST 区分真实控制节点与注释中的 while，并回读下述代表循环的上下文。77 是文件覆盖数，绝不是机制数。

额外检查了四 family 共 21 份 YAML 的 dispatch / work_schedule：FA4 prefill 的 11 个节点、decode 的 7 个节点、GDN 的 2 个节点与 QSA 的 1 个节点。该检查用于确认分支范围，**不等于对其全部 action/waiting 做了第二次完整验证**。vendor 的 general-SM100 sparse、CLC、varlen 映射与等待辅助函数作了定向源码复核；其余 backward/MLA 循环只保留定位与范围排除，未冒充逐体审阅。

## 按职责分组

| ID | 代表位置 | predicate / 体内动作 / 进展 | 层次与 for/if 判断 |
| --- | --- | --- | --- |
| P1 | `P/.../sm100_hd256_2cta_fmha_forward.py:1829,2529,2817,3022`；`tile_scheduler.py:1303–1345` | `work_tile.is_valid_tile` 来自 static scheduler；体分别装载、MMA、softmax、correction；末尾 `advance_to_next_work` 清掉 `_is_first_block`。 | root 为零或一份静态工作；有效 worktile 的 pipe 不需 while。不能把邻近 persistent 分支也说成单轮。 |
| P2 | 同文件 `:1903–1919,2124,2142,2628,2922,2945,3162` | trip count 由 seqlen、causal/window 与 split 范围得出；每次消费下一 KV 块；mask 前后缀、首末 QK/PV 与 correction 各有边界。 | 输入确定的有限数学遍历。for 保持每块顺序及两半片批次；首末动作、Stats 初次确认和输出阶段不随空中间域消失。最大值/alpha 改变数值和写入选择，不改变块数。 |
| P3 | `tile_scheduler.py:1335–1345`；hd256 `:200–213,3221–3224`；general SM100 `:2998–3013` | static persistent 增加固定 worker 步长；CLC 则提交下一硬件领取，等待响应、消费并归还响应槽，响应 invalid 后退出，再做 producer tail。 | 前者为有限预分配工作遍历；后者为实际动态 worktile 调度。可用允许的动态 for，但领取、响应交接、terminal 响应与 tail 必须显式；没有证据可把 CLC 失败当本地无限重试。CLC 硬件实现未冻结。 |
| P4 | `P/.../block_sparse_utils.py:707–786,792–829,951–1140`；`flash_fwd_sm100.py:1681–1704,2065–2160,2507–2673` | mask/full counts 和索引来自稀疏元数据；两个列表按规定逆序遍历、可再按 split/subtile 切分。`total_block_cnt==0` 时不做 QK/PV，但仍发 empty stats 与写零输出。 | 有真实有限域；不存在由本轮 QK/softmax 结果生成下一稀疏块的反馈。for+if 可以表达，不能把 empty worktile 直接过滤掉。此机制不在 hd256 root。 |
| P5 | `P/.../tile_scheduler.py:949–973` | `group_end_tile<=next_tile_idx`；每轮扫描后续 31 个 batch 的长度、前缀和，batch 索引加 31；到 `num_batch` 后显式设置退出哨兵。 | 调度坐标解析内部的有限元数据搜索；最多 `ceil(B/31)` 次后续轮，包含越界终止处理。for+if 可沿真实 batch 组遍历，原搜索状态即可保持“已找到后不再动作”；多余尾检查仍存在。不是收敛迭代。 |
| P6 | `P/.../utils.py:843–855,942–955` | warp reduction 的 offset 每轮减半；binary search 的 `[lo,hi)` 每轮严格缩小，读取 prefix boundary。 | 固定归约阶段 / 单动作搜索算法。binary search 可用真实对数上界加 `if(lo<hi)`，但 while 在实现中更自然。前者冻结包内只定位到定义，后者用于非 root 的 MLA/稀疏 metadata。 |
| P7 | `P/.../barrier.py:56–61`；`blackwell_helpers.py:542–549,901–908` | 前者 repeated acquire-load 等待通知值；后者 PTX 反复 `mbarrier.try_wait.parity`，成功才离开。进展来自其他参与者的通知。 | 原语等待。没有从源码得到有意义的有限重试上限；PTX 的 10000000 是单次等待 hint，不是正常退出 cap。应表达 await/acquire 的保证，不在 pipe 编造 retry 域。上述本地 helper 并非 hd256 root 使用的 CUTLASS helper 实现。 |
| P8 | `P/.../cache_utils.py:117–137` | 主机文件锁非阻塞尝试，失败 sleep，成功 break，deadline 到期抛异常。 | 冷编译/缓存基础设施；不能提升成 GPU role 的动作反馈，也不能把超时异常改叫正常循环完成。 |
| G1 | `G/gated_delta_net_chunked.py:1528,1598,1730,1769,1896,2059`；scheduler `:237,263–270` | 六个 role 各自沿相同线性工作 ID 推进；M64 总工作数 `2*B*HV`（core `:1039–1044`）；每份工作后 ID 加 resident CTA 数。 | 有限固定步长调度；自然 `for(t)`，无新锁存状态。各 role 的资源尾与每工作状态装入/末存仍须保留，for 边界不自动 drain。 |
| G2 | core `:1553,1627–1630,1738,1776–1826,1943–1996,2109–2213` | pair/chunk 数来自 Cu32；状态递推依赖前块 S，ScalarOutput 按 `j+4<Np` 前瞻。padding 使用中性输入但仍推进协议。 | 数值递推有结果反馈，**继续与否却不由结果决定**。`Np=2*ceil(L/128)` 的 for 自然，前瞻用既有范围/if。不能因 padded chunk 无有效输出就删其事件。 |
| G3 | core `:3999–4086,4207–4233` | 8×8 的 N−1 个 pivot，再 8→16→32→64 合并与 rendezvous；无残差阈值或迭代至收敛。 | 单动作内部有限层级算法。源码函数还兼任后续发布，函数边界不等于动作边界；不能以 inverse 一词推断 while。 |
| D1 | `D/.../gqa_decode_paged.py:1104–1107,1228,1325,1415,1491,1654,1937`；`gqa_decode.py:442,503–529` | `iters_s=ceil((ceil(L/256)-split)/work_splits)`；掩码相位由 CausalMask 生成；填排与全部名义 split 的归约均有有限域。 | for 与首尾 if 足够。try-wait token 不让 tile 重试；N_r=0/1 的统计、握手及空 O 槽退役不得从结果为空推断删除。 |
| D2 | paged `:1707–1724,1497–1506,1939–1944` | 库中 BLASST 把 score 与既有 max 比较，跨角色发布 keep_tile；决定当前块是否做 P/PV，随后仍去下一输入块。 | 是数据依赖的真实跨角色选择，但不是反馈终止。开启时需要显式发布/等待 flag 的 for+if；root 编译传 None 禁用。未独立证明该近似分支的数值契约。 |
| Q1 | `Q/.../qsa_preprocess.py:218–232,340–359` | `not placed`，CAS 返回空槽或同 key 则 OR 位图并置 placed；否则 h 加一并环绕。键只从 0 变为某 key、不删改。 | 单个 Hash build 内的并发开放寻址；合法 root 有真实有限容量证明。不能把“成功前的轮次”写成预先域；当前 pipe 不需要暴露每次探测。详见下节。 |
| Q2 | `Q/.../qsa_kernel.py:324–334,461–646,673–731,762–840` | Scheduler 原子领取、发布、等待全部复制确认后再看 terminal；其他 role 根据收到的 u 是否在工作范围内处理并收下一票。 | 合法跨角色反馈，职责为动态工作调度。for(t) 可保持全部动作，但只有在 t 是 work_schedule 已定义的实际工作序列时成立；不能换成静态分配或隐式领取。 |
| Q3 | kernel `:480,513–557,683–727,792–831` | 每票 N 由已完成 preprocess 的 Count 得出；softmax vote 决定 rescale，MMA 的 ahead 决定首尾填排。 | 有限 union tile 遍历加条件数值修正；vote 不减少或增加 N。保持 P 发布、上一 PV 完成及输出释放边界，不需局部反馈循环。 |

`P/...` 中的省略段为 `flash_attn/cute/`，`D/...` 为 `qoder_greenfield_decode_attention/`，`Q/...` 为 `qwen38_core_qsa_cutedsl/`。JSON 还单列 GDN 未调用 legacy issuer、FA4 未逐体复核的 vendor work loops，以及 backward 中有限 `iter_count` 倒计数，避免把它们混进 actual-root 结论。

## 候选一：QSA CAS / 哈希探测

[主循环](../../trial_roots/sources/template_qsa_prefill/qwen38_core_qsa_cutedsl/qsa_preprocess.py#L218) 并非“先尝试占锁，失败就等待原持有者”。失败意味着遇到另一 key，下一轮改探下一槽；成功或找到同 key 后，每个 inserter 都对 membership 做 atomic OR，包括创建 key 的线程。初始清零后 CTA 会合（`:130–137`），全部插入后再次会合（`:253`），之后才 compaction。因而看到 key 不需要再等创建者发布 membership：所有 OR 都在第二次会合前结束，没有 creator-only 非原子覆盖窗口。

**root 的终止论证。** 构造器 `:62–65` 计算 `MaxE=Br*(SelectedWidth//4+1)` 并断言小于 H；此处 MaxE=5130、H=16384。root 输入要求每行 512 个不同的完整四元组，至多再有一个尾组，故整个 CTA 的不同 key 数 D≤5130。槽内 key 在此阶段只增不删；一次失败 CAS 消耗一个不同槽的位置。某次插入的 key 如果已经存在，线性探测最终会遇到它；如果不存在，至少一个槽必空，且同一 key 的并发插入会在同一探测序列的第一个成功槽合流。至多 H 次尝试必遇到空槽或同 key，不能在全表循环后仍合法失败。该证明依赖原子操作能完成与通常执行进展，不声称给出墙钟时间上界。

H 是真实候选槽域，故实现层可以写 `for probe_offset in range(H): if not placed: <原 CAS/OR/推进体>`。`placed` 本来就在源码中，不需要新增锁存值；成功后它不再变假。它保留实际 CAS/OR 次序和结束条件的论证，须理解为在对应原子线性化顺序下比较；**没有证明实际 GPU 重排、最终 union 排序或浮点输出逐 bit 一样**。额外 H−实际探测次数个条件检查是真实控制成本，不是“免费把 while 删除”。若根本不需要改实现，做这种改写没有优势。

现有设计中 preprocessing 只有一个 Main role，`build:Hash` 与 `compact:Union_Count` 之间才有设计边界。按最新八词、两字段 noun 规则，同一粒度可以这样写：

```yaml
info:
  using_verb:
    compute:
      Hash:
        execution_mode: serial
        describe: "Clear the shared hash table, read and filter the declared input quads, merge each group's membership with CAS and atomic OR, and complete the CTA rendezvous before compaction."
      Union_Count:
        execution_mode: serial
        describe: "Compact occupied hash slots into the global Union, preserving each key and membership mask; finish all entry writes before exposing Count. Warp reservation order is unspecified."
pipe:
  - "compute:Hash"
  - "compute:Union_Count"
```

两条动作内部仍有 ordinary read/write 和线程协作，未把原有发布、交接或释放边界吞并。若提升成 `while(r, not Placed(e,r))`，还得给每个线程负责的 entry、探测与成功 OR 建立事件身份；现有消费者不依赖中间探测事件，所得描述是另一种更细的线程算法层，而不是补上当前 pipe 缺失的设计边界。

**判定：真实算法反馈，非坏设计；在当前 pipe 粒度下把它当 while 必要性证据属于层次混淆。** `_insert_lane` 的注释声称支持不同打包，但其 MaxE 推导仍使用 quad 数；不能把以上容量/输出空间论证无条件推广到任意 mixed-group packing。那会成为扩域后需要重审的容量问题，不是合法 root 的反例。

## 候选二：QSA terminal ticket 与动态工作领取

[Scheduler 源码](../../trial_roots/sources/template_qsa_prefill/qwen38_core_qsa_cutedsl/qsa_kernel.py#L319) 的体很短，但控制关系真实存在：

1. `atomic_add(mTicket,1)-tbase` 得到 u；
2. 写 sUnit，再 `unit_posted` 通知；
3. 等待十一消费者 warp 的 `unit_taken`；
4. 从本票 u 得出 terminal，决定是否再领取。

总工作 W=ceil(4096/10)*2=820。每张有效票唯一映射到固定 `(row_block,kv_head)`；领取不会由 softmax 计算生成新工作。每 CTA 取得一张 terminal 后退出，所以在串行 eager、计数基址一致和协议正常推进的条件下，全 launch 消耗 W+P 张票（`operator.py:155–158`）。单 CTA 的实际工作数不预先可知；不能把它换成 `for u in range(worker_id,W,P)`。

与源码同粒度的反馈写法可以是以下控制示意。`Terminal` 初始为 false，由 claim-and-post 动作保存本票比较结果；条件自身没有读取 counter 或执行等待：

```yaml
pipe:
  - "while(r, not Terminal)":
      - "signal:Ticket(r)"
      - "await:TicketTaken(r)"
```

这里 `signal:Ticket` 继承原 `publish` 的 **claim-and-post** 契约，内部领取没有被省略；`await:TicketTaken` 等所有消费者复制确认。它们均为 serial。`signal` noun 的 `describe` 必须明确包含按 work_schedule 领取并发布本票，不能只写一个无来源的通知。如果未来基础词归并要求把领取单列为 compute，两种版本都应等量拆开，不影响循环判断。

已有 work_schedule 定义实际领取形成的有效 worktile 序列 t；以每张**已取得的有效票**绑定一次 t，等价的设计级形式是：

```yaml
pipe:
  - "signal:Ticket0"
  - "await:Ticket0Taken"
  - "for(t)":
      - "signal:Next(t)"
      - "await:NextTaken(t)"
```

各 noun 均为 serial，分别沿用 first/next claim-and-post 与全部确认的说明；实际登记仍应各自只含 `execution_mode`、`describe`。这个对照没有把 t 定义为“直到成功的领取轮次”：t 的成员是真正的有限算子工作，成员身份与排序在工作层已有定义；循环发起前的票据已给出当前工作，循环体尾部实际产生下一票或终止票。

若某 worker 获得 m 张有效票，两式都发出 m+1 张票并等待 m+1 次确认。事件对应为 `Ticket(0)↔Ticket0`，`Ticket(j+1)↔Next(t_j)`；消费者相同地把初始 accept 放在 for 前，把下一 accept 放在每工作尾部。此对应是任意 m 的协议展开论证，不是仅列几个 trial 日志。**for 头不执行 atomic_add、不发布、不替消费者确认，也不使整个 worktile 完成。** Scheduler 可以在消费者复制本票后领先一份工作，原有异步重叠必须继续存在。

必须额外保留这些源码事实：

- terminal 本身仍发布和确认；m=0 时不能吞掉 Ticket0 / Taken0。
- SoftmaxA 初始票即 terminal 时，仍对钳制后的有效坐标加载 Q 并发 qready（`:386–398`）；MMA 仍等待这个初始 qready（`:669`）。
- 工作尾部 terminal Qnext 跳过 Q copy（`:584–588`），却仍在 Y_A 最后读完、`odrained` 之后发 qready（`:617–624`）。旧 baseline 把 terminal Qnext 称为 no-op 的说明不完全对应源码；本调查记录差异，未修改继承契约。
- Gather 的累计 gt、KV ring phase 不在 for(t) 边界复位；最后一次 copy 的 done 在下一次 issue 之后发布，跨工作延续；terminal 后还有 drain（`:828–840`）。MMA 的下一工作首 PV 等两半输出释放（`:703–707`），不是只等 ticket acknowledgement。

**判定：合法跨 role 反馈，非伪需求或坏设计；在本研究明确允许动态工作序列 for 的前提下，不构成 while 必要性反例。** 如果将来把 for 收紧成只能预先枚举的域，这个结论立即需要重审。不能借当前投影宣称 QSA 从来没有控制反馈。

## 候选三：FA4 CLC / sparse 的空工作与交接

这是定向调查的 vendor 扩展，不是 FA4 hd256 root。CLC 的 [ClcState](../../trial_roots/sources/template_fa4_prefill/flash_attn/cute/tile_scheduler.py#L77) 在 producer acquire 后向硬件提交下一领取；消费者在 wait 后解析坐标并 release（`:1604–1618`）。[scheduler warp](../../trial_roots/sources/template_fa4_prefill/flash_attn/cute/flash_fwd_sm100.py#L2998) 每份工作预取下一响应，invalid 响应结束循环，最后做 producer tail。源码没有“领取失败后等待更多工作再重试”的另一层反馈。和 QSA 相同，只要保留这些动作与响应寿命，就可以遍历实际提供的动态 worktile；不能把这些调用都塞进 for 头。

稀疏列表本身来自 `get_curr_blocksparse_tensors`。masked/full 的 split 范围在处理前就已确定，两个列表按源码顺序处理。最危险的消去错误发生在零列表：

- Load 不加载 Q/K/V，MMA 不发布 P/O 完成；相应 pipeline state 不跟着一份空列表假推进。
- Softmax 仍 acquire stats 空槽，然后发一次 EmptyStats；Correction 会合、归还 stats 槽，给输出写零，并在独立 epilogue 存在时发布输出槽。
- 因而 empty→empty、empty→nonempty、nonempty→empty 三种接续都要保持相位；不能用“t 只枚举非空 sparse tile”的域删掉空工作。

证据不仅是注释的 [mbarrier contract](../../trial_roots/sources/template_fa4_prefill/flash_attn/cute/block_sparse_utils.py#L96)：实际 Softmax 代码在 `flash_fwd_sm100.py:2109–2111` 无条件 acquire sparse stats；helper 在 `block_sparse_utils.py:1093–1094` 发 empty 通知；Correction 在 `:907–938` 准备 stats、会合、归还，并执行 scale=0 的 epilogue。

下面只投影 **Empty(t)=true 的分支**，用于比较控制外壳与交接；它不是覆盖非空 softmax 的完整迁移稿。SparseMeta 的串行动作实际读取元数据并建立 Empty；统计槽申请、通知、会合与归还保持各自可引用的边界：

```yaml
# 两个 role 的空路径投影；完整稿还需保留非空分支。
Softmax:
  pipe:
    - "for(t)":
        - "load:SparseMeta(t)"
        - "acquire:StatsSlot(t)"
        - "if(Empty(t))":
            - "signal:EmptyStats(t)"
Correction:
  pipe:
    - "for(t)":
        - "load:SparseMeta(t)"
        - "if(Empty(t))":
            - "compute:EmptyStatistics(t)"
            - "await:EmptyStats(t)"
            - "release:StatsSlot(t)"
            - "compute:EmptyOutput(t)"
            - "store:Output(t)"
```

SparseMeta / EmptyStatistics / EmptyOutput 是 serial；通知、等待、资格与使用权按八词定义登记 `execution_mode` 与英文 `describe`。这里选择 correction 自己输出的分支；若输出经独立 epilogue，要继续保留输出槽 acquire / signal / release。片段末省略的 CLC response consume/release 和 producer tail **在两个控制版本中都必须保留**；把外围 `for(t)` 换成 `while(r, ValidTile)` 不会使这些交接可省。

**判定：没有发现本地结果驱动的新无限域，for+if 的表达能力足够；空列表协议是其等价改写的必要条件。** 本次没有冻结或验证底层 CUTLASS CLC 实现，也未对这一 vendor 分支建立完整 waiting 图，故结论是控制来源与必要事件的源码论证，不是整条异步 pipeline 正确性证明。

## 局限与交付状态

- 未运行 GPU、CuTe 编译或数值比较，未迁移 trial。源码哈希匹配、AST 清单与范围计算不证明并发死锁自由、硬件公平性、浮点误差或所有 waiting 正确。
- 保留了无法自然变成有界 for 的原语等待；没有用硬件有限状态空间、任意巨大 cap、超时或验证预算冒充正常退出。
- 库中其余 MLA/backward 的 worktile while 已定位、排除 actual-root 归属，未逐个证明其正常退出与所有资源尾。JSON 对此使用 `inventory_only`，不把搜索命中算已审机制。
- QSA Hash 容量证明只覆盖 root 合法输入；QSA ticket 消去以当前动态 for 规则和现有 work_schedule 为前提。它们不能承担所有合理算子 while 可删除的普遍论证。
- 源码实际不同于 baseline 的 terminal qready 等细节单独标出；没有因为目前 pipe 足够表达循环，就认定现有设计已经完全正确或所有事件已经表达完整。

本轮只新增本报告和对应 JSON。供后续取舍使用的短结论是：**四 root 不提供必须保留 pipe while 的反例；QSA 提供“真实反馈可以属于合理调度”的反例，FA4 sparse 提供“空遍历仍须保留交接”的反例。两者都阻止把 while 的去留简化成关键词或循环次数问题。**
