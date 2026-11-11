在本次固定证据范围内，**六个提取 root 及下列代表分支没有提供“设计级 pipe 必须有 while”的强例**。最强候选分别是 MLA 的动态工作队列、MoE direct 的跨 CTA 到达票据、Sparse packed 的 CAS 探测与位置发布等待。前两者分别可由真实 worktile 域和有限 unit 域加 if 表达；第三者的无界等待确实存在，但属于既有协作构建动作内部的原语协议。这个结论不支持“所有合理算子都不需要 while”。

源码中的 while 可以正确、合理而且比低层 for 更自然。这里没有要求删除源码 while，也没有用任意大上限、“直到成功域”或新大动作消去反馈。对无上限的等待，保留 await/acquire 所兑现的通知、可见性与资格语义，而不是声称可以有界轮询替代。

**覆盖边界。** 搜索覆盖 [trial manifest](../../trial_roots/manifest.json) 给出的六算子全部 **27 份冻结源码**，包括 while、无条件 for、break、CAS、等待辅助函数及 PTX 回跳。逐机制读控制上下文，而不把宏 `do…while(0)`、注释、同一函数的版本副本和 host 选择器相加。数学、资源、契约取六个原 root 和 14 个代表分支；trial 只作 pipe 投影对照。路径和 SHA-256、逐例谓词/进展/替代表述见 [JSON](../data/02_extracted.json)。没有运行 GPU、改写原源码或迁移 schema；未逐条审查全部 172 份提取有稿节点，也没有证明所有编译开关和调用路径。

| 算子 | root 的选定路径 | 代表分支及核对重点 |
| --- | --- | --- |
| GDN | 缓存 Q/K、MODE=0、无 permutation | `segmented` 的两遍状态链和 weak 回退；`direct_qk_ordered` 的稳定排序与同一有限 chunk 链 |
| GEMM | `umma_gemm<0>` pair 管线 | `wide` 的 UMMA 宽列管线；`mma_sync`；组坐标查找与等待不改变归约域 |
| MLA | 队列 refill 的五角色 persistent kernel | `bootstrap` 的同体单工作分支；真正终止票与非末/末键块协议 |
| Sparse | mma.sync split + FP32 combine | `tc_packed` 的 union 构建；`probe_hidden_packed` 的相同构建与 host 有界观察 |
| FMHA | D128、QK64、TMA、fused DPV | D128 TMA direct narrow 的 `flat`、`segmented`、`fscan` 三种扫描路径 |
| MoE | cfg97/cfg2=81 的分角色 FC1/FC2 | `direct` 的 cfg92 到达票据；`fused_small_mono`；`split_ws`；`double_verify_root` |

判据采用 [METHOD](../METHOD.md)、当前 [pipe 规则](../../README.md)及 [八词/两个字段规则](../../../verb_vocabulary/README.md)：动作族由 `execution_mode`、`describe` 解释，但 describe 不能代替已命名边界之间的 waiting。源码的 `while(!bar_try_wait(...))` 也不能直接抄为 pipe 条件：读取、等待、通知须属于明确动作，条件只能使用已就绪的值。下文保留 baseline 的旧动作名方便复核，并不提出 verb 迁移。

**机制表。** “可 for/if”是在标明的抽象层判断，不是声称重新编译后的指令、时序或任意并发轨迹相同。

| 实际机制与证据 | 谓词来源、体内工作、进展/退出 | pipe 对应与判断 |
| --- | --- | --- |
| GDN chunk 递推：[CUDA 2058–2063、2319–2343](../../trial_roots/sources/extract_gdn/gdn_kernel.cu#L2319)；[root 551、621–675](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml#L551) | 输入 request 长度给 `chunks=ceil(seqlen/64)`；每轮载入、KS/QS、rhs、solve、读出与状态更新。R 依赖上轮，但是否有下一 chunk 不依赖 R。 | `for(c)` 保留五动作及下一 advance 的读退出；状态递推不是继续条件反馈。 |
| GDN 三角子块与排序：[CUDA 2591–2604、2648–2672](../../trial_roots/sources/extract_gdn/gdn_kernel.cu#L2591)、[276](../../trial_roots/sources/extract_gdn/gdn_kernel.cu#L276) | `ks>mw` 只由 warp 行块决定；插入排序依已读 chunk counts 向左移动，最多 i 次。 | 三角域可 `[0,mw+1)`；排序是既有 `enter:Metadata` 内的有限坐标构建。低层 while 自然，不构成 pipe 需求。 |
| GDN weak 第二遍：[CUDA 2227–2240](../../trial_roots/sources/extract_gdn/gdn_kernel.cu#L2227)；[segmented 119–140、617–626](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/segmented.yaml#L119) | 等 pass A 后扫描 G 个 SegA，一次建立 weak、Run、Count；weak 时仅 z=0 跑全请求，否则各段跑自己的范围。 | `join_a → classify → if(Run) → for(cl=0:Count)`；后续结果不再触发重试，三 phase 没有反馈回边。 |
| GEMM 组查找：[UMMA 363–388、798–838、1162–1180](../../trial_roots/sources/extract_gemm/fp8gg_umma.cu#L363)；[sync 133–147](../../trial_roots/sources/extract_gemm/fp8gg.cu#L133) | 执行时 m_indptr 给非降 prefix；`while(prefix[e+1]*NB<=w) ++e` 跳空组。已知 `w<W=prefix[E]*NB` 保证找到 e<E。内层 PAIRS/KB 由 K 决定。 | Schedule/Work 坐标解码；外部 `t=worktile`、`n=flatten(t,q)` 自然。若细化查找，有真实 E 候选域，无算法性无界继续。 |
| GEMM/MLA/MoE/FMHA/GDN/Sparse 等待：[GEMM 181–183](../../trial_roots/sources/extract_gemm/fp8gg_umma.cu#L181)、[MLA 150–153](../../trial_roots/sources/extract_mla/solution.py#L150)、[MoE 2558–2574](../../trial_roots/sources/extract_moe/nvfp4_moe.cu#L2558)、[FMHA 259–268](../../trial_roots/sources/extract_fmha/fmha_tcg_ptx.cuh#L259)、[GDN 300–305](../../trial_roots/sources/extract_gdn/gdn_common.cuh#L300)、[Sparse 1614–1637](../../trial_roots/sources/extract_sparse/dsv4_kernel.cu#L1614) | barrier 固定世代的完成位来自对端 arrival 或异步操作完成；体内只有测试/可选休眠。正常退出依赖该通知，不依赖重复发起计算。等待次数没有有意义的输入上限。 | 已有 acquire/await、动作完成契约和 waiting 承担此职责。必须保留世代、成员及可见性保证；不能把 poll 换成固定次数后继续。源码 while 合理，pipe 无需按 poll 拆事件。 |
| MLA 动态领取：[863–870、1458–1483](../../trial_roots/sources/extract_mla/solution.py#L1458)；[root 340–353、620–641](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml#L340) | 五个 `for(;;)` 外循环；Queue 等请求，atomicAdd 取 Geo 索引，越界发布 kv_tiles=-1 后退出。每个实工作请求一次，表长度 WorkCount 有限。 | 真实 upstream Geo/atomic_queue 形成 `for(w)`；保留 request、publish、四方 wait/latch 和终止发布。不是“直到计算成功”伪域。 |
| Sparse root 与 packed 数据域：[1002–1009、1198–1206](../../trial_roots/sources/extract_sparse/dsv4_kernel.cu#L1002)；[packed 38–56](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/tc_packed.yaml#L38) | root 的 Klen 来自长度元数据，split 区间固定；无效 ID 仍占迭代并读替代行。packed 的 Nu 由 union 构建得出，完成后 C=ceil(Nu/KC) 固定。 | root `for(c0:c1)`、packed `build:Union` 后 `if(C>0)` 与 `for(c)`；数据依赖的已就绪范围不等于反馈循环。 |
| Sparse 开放寻址与位置等待：[2305–2379](../../trial_roots/sources/extract_sparse/dsv4_kernel.cu#L2305)；[Setup 契约 363–384](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/tc_packed.yaml#L363) | 碰撞推进 slot；最多 2560 个输入条目、8192 个槽。相同 key 找到后可能等待创建线程把低位位置从 0 发布为 pos+1。 | 探测有真实槽域；同 key 等待没有时长上界。两者都在现有全 CTA `build:Union` 的协作算法内，后续 Loader 未开始。详见候选三。 |
| FMHA 64/16/1 及分段跳过：[flat 932、1062、1099](../../trial_roots/sources/extract_fmha/fmha_tc.cuh#L932)、[segmented 1422–1497](../../trial_roots/sources/extract_fmha/fmha_tc.cuh#L1422)、[fscan 4409–4452](../../trial_roots/sources/extract_fmha/fmha_tcg.cuh#L4409) | flat 批长由 T 给定；segmented 以已载入 summaries 和 incoming m 计算 Cut，前缀填常量、后缀顺扫，每批仍向前 16 个 segment；fscan 扫 `[0,min(cLo,T))`。 | 分段已有 `for(c)` 中 `test:Prefix(c) → replay:Chunk(c)`。若细分，可在 test 后遍历 `[g0,Cut)` 与 `[Cut,g1)`；不需要重复 test 直到 m 稳定，也不能因跳过读取而删除 snapshot stores。 |
| MoE root persistent 与二分：[4371–4384、4448–4461、4544、4858、4990](../../trial_roots/sources/extract_moe/nvfp4_moe.cu#L4371)、[775–783](../../trial_roots/sources/extract_moe/nvfp4_moe.cu#L775) | TilePrefix 给 J，item 按 grid stride 遍历 J*N*SK，局部 H/BK 或 I/BK 次；二分找 expert，每步区间缩半。 | root 是 `refill:none`，不是动态领取队列。保留跨 item 连续 g 与环世代；二分是 `produce:TilePrefix` 内算法，最多 ceil(log2 E) 步。 |
| MoE direct 到达票据：[1985–1989、2210–2256](../../trial_roots/sources/extract_moe/nvfp4_moe.cu#L2210)；[direct 345–407](../../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/direct.yaml#L345) | 固定 unit 各贡献一次 Partial，再 atomicAdd Arrivals；返回值等 n(t)*S-1 的参与者做最终归并与清零，其他参与者继续自己的固定 unit 域。 | `for(v)` 内保留 accumulate、arrive、`if(Last)` 的 gather/finalize、advance；确有跨 CTA 交接，但不存在重复试做、回流或重试领取。 |
| MoE 验证票及 host 决策：[1723–1789、9480–9488](../../trial_roots/sources/extract_moe/nvfp4_moe.cu#L1723)；Sparse [probe_end 5593–5614](../../trial_roots/sources/extract_sparse/dsv4_kernel.cu#L5593) | 每 CTA 验证一次，最后票发布 clean；host 对 mirror/samples 有实际 cap，随后一次决定是否提交后续图/哪种余段。超时后的观察不等于最终 GPU 判定。 | 验证 kernel 保留 store/scan/publish；host 观察属 dispatch，不是 role 反复算数。不能改为无条件无限 await：超时/观察变化会改变原选择语义。 |

JSON 还单列了非默认 fused-combine CAS、watchdog 和 host 配置搜索，以保留排除依据。

**候选一：MLA，真实动态工作流，但不是数学结果决定是否再算一轮。**

这是六 root 中最值得质疑的外层控制：工作数不能由单个 CTA 预先知道，Queue 与四类消费者不同步前进，源码明写无条件循环并靠 sentinel break。其域却不是事后捏造的成功轮次：Geo 在进入 kernel 前已经完整上传，每个条目唯一拥有自己的输出，剩余条目由真实 atomic_queue 分配。执行完成不会追加新 Geo，也不会把失败 worktile 放回队列。

保留当前 `for(w)` 能保持每项 request→pop/发布→消费的事件边界：Softmax 在末键块等待前请求；Queue 在该请求之后执行 `publish:Next(w)`；Softmax、Correction、MMA、Loader 各自显式 `wait:Next(w)`。**最后实工作仍有一次失败领取和 terminal publication，四类消费者都要观察它；终止票不产生一个零 tile 假工作。** 普通几何 latch、两槽 reuse waiting、连续 n 也保留。无需把这些动作合并进 Work，也不能仅以“循环结束”代替通知。低层 wait 保持无界，不能拿 WorkCount 当轮询次数上限。

有意义的退出依据是表耗尽：WorkCount-worker_count 次成功 refill，加每 CTA 一次终止领取，共 WorkCount 次 pop。进展仍以受等待约束的当前工作能完成、请求/通知会被服务为条件；这是调度与原语的活性义务，不是仅由 for 语法证明的事实。baseline 已指出 early descriptor reuse 与 TMEM 行地址的源码偏差；本次没有用循环形式为其担保。

**候选二：MoE direct，确有跨 CTA 协作，for+if 仍保留全部独立边界。**

`direct.yaml` 的 cfg92 可以在建立 ABI 时看到空路由、在后续 replay 收到非空路由，不能只审查空路由快速返回。非空时固定域是 u=(t*K+k)*S+s；无效 unit 读 ID 后跳过，实 unit 完成 FC1、激活、FC2 Partial 后才发到达票。每个 token 总到达数 n(t)*S 由当前输入的有效路由数决定，而不是由收敛质量决定。

对应 pipe 可以自然写成下面的局部控制示意，动作全部来自原分支；a/f 原来专作 0/1 门控，换为常数 0 不删除实数据索引 v：

```yaml
"for(v)":
  - "inspect:Unit(v)"
  - "if(ValidUnit(v) == 1)":
      - "prepare:X(v,0)"
      - "contract:UpGate(v,0)"
      - "quantize:QChunk(v,0)"
      - "accumulate:Partial(v,0)"
      - "arrive:Ticket(v,0)"
      - "if(Last(v,0) == 1)":
          - "gather:FinalRoutes(v,0,0)"
          - "finalize:Y(v,0,0)"
      - "advance:Unit(v,0)"
```

`arrive` 包含每线程 Partial 的 device-scope 发布、CTA join、单次 atomicAdd 和 Last 的 CTA 发布；这建立了 if 所需的已就绪值。finalize 的读取、BF16 边界、输出、route accumulator 清零和 counter 清零继续可识别；非最后实 unit 也必须执行 advance，保护下一 unit 的 Last/输入/中间存储。没有把通知塞进 if，也没有令失败的最后票重新执行 accumulate。source 与 baseline 对这些边界一致；跨 CTA 本身不等于反馈循环。

**候选三：Sparse packed，无界 while 是必要的等待实现，不能被“有 8192 槽”掩盖。**

必须拆开两个循环。开放寻址遇到不同 key 时向下一槽推进；合法容量保证新 key 最多在一个槽周回内找到空位/同 key，这个有限域有真实数据结构依据。命中同 key 却 pos 尚未发布时，`while(pos<0)` 重读同一个位置，直到创建线程的 atomicExch；它可能等待任意久，8192 和 2560 都不是等待次数上限。任意 capped for 后继续会改变可见性和退出语义。

这仍不是当前 pipe 的反例：在原稿里，所有八个 warp 共同担任 Setup，创建者/争用者是每条 key 临时形成的线程分工；HashTable 与 IndexStage 是此构建暂用的环存储。不存在 Loader 已独立消费半成品、再把结果回传决定下一次构建的循环。`build:Union` 原本已经声明“安装 key → 分配位置 → 原子发布 → 同 key 等待 → membership/重复项处理”，最终 CTA rendezvous 才向后续角色发布完整 UnionKey/UnionMask/Nu，同时退休临时存储。**保留这个已存在、契约公开的协作动作，不是为删 while 临时发明大动作。**

若将来明确把按 key 的安装、位置发布、争用者等待设计为独立 pipe 角色，就需要在那个新粒度重新检查，不能继续引用本结论将其吞并。即使如此，单个 position wait 也可以由等待指定发布条件的 await 表达；本次没有证明所有更细的并发哈希设计都无需 while。这里的进展依赖创建者能执行到 publication，未运行 GPU，因此不声称对任意调度证明终止。该分支原稿记录的活动 barrier/metadata alias 问题也未修复。

**证据的取舍。** FMHA 分段 scan 是最强的“数值结果影响后续读取量”例子，但上一批 m 只影响下一批 Cut；批数、key 区间和必须写出的全部 snapshots 仍有限且自然。GDN segmented 也有数值判定，但只决定第二遍的有界范围。它们说明 for 域和 if 可以依赖先前动作产生的已就绪值，不能说明数值依赖必须写成 while。反过来，本报告没有尝试把真正的收敛判定、回流工作或失败后跨角色重试伪装成这类域。

完整性检查重新计算了 27 份源码和六份 root 的 SHA-256，均匹配 trial manifest；为代表分支另记实际读取 hash，所有逐例源码范围附首行与片段 hash。此检查证明引用字节可追溯，不证明 kernel 数值正确、任意并发无死锁或原有 waiting 完整。支持的结论限于：**已审的六提取路径不迫使当前设计级 pipe 保留 while；是否为更广的合理设计保留，须由独立正向反例审查决定。**
