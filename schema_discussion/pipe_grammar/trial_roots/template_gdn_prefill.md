可以完整表达；有序列表使每工作初末、整批先后与标量前瞻更清楚，动作内部复杂度仍由原契约说明。覆盖 **2 phase、7 role、42 个静态动作、17 个 for；if/while 均未使用**。源码的 scheduler while 只按预分配步长推进工作（[advance_to_next_work:263](sources/template_gdn_prefill/gated_delta_net_tile_scheduler.py#L263)），没有需要反馈判定的局部循环，现有 `for(t)` 足够；本例并未检验 while 的一般能力。

采用 manifest 固定的 `01fff90d714abc6e84904ea9c44523ba66d34a0e` 源码。[Model.forward:812](sources/template_gdn_prefill/solution.py#L812) 转换 Cu32 并强制 M64；[入口选择:782](sources/template_gdn_prefill/solution.py#L782) 对本 root 的 B>1、HQ=4、HV=32 关闭静态描述符与紧凑事件。[build:19](sources/template_gdn_prefill/build_solution.py#L19) 读取调度器、核心、适配层，[build:92](sources/template_gdn_prefill/build_solution.py#L92) 拼接后压缩；同时核对了入口内核心方法调用。下表 K 为同目录 `gated_delta_net_chunked.py`，采用 packed 分支。

| phase / role | for / 动作 | 原动作对应与关键源码位置 |
| --- | ---: | --- |
| index_cast / Convert | 0 / 1 | `produce:Cu32`；上述 forward:812 |
| gdn / Load | 2 / 4 | bound、K/Q/V；[kernel:1896](sources/template_gdn_prefill/gated_delta_net_chunked.py#L1896)、[tma_qkv_warp:2472](sources/template_gdn_prefill/gated_delta_net_chunked.py#L2472) |
| gdn / MMA0 | 2 / 4 | KK0、KK1、QK0、QK1；[mma_cg0_pair:2793](sources/template_gdn_prefill/gated_delta_net_chunked.py#L2793) |
| gdn / Inverse | 5 / 8 | Pair、BetaPair、KK/A、invert、Tinv、QK/Bmat；[compute_group_0_pair:3595](sources/template_gdn_prefill/gated_delta_net_chunked.py#L3595)、[_finish_pair_inverse:4041](sources/template_gdn_prefill/gated_delta_net_chunked.py#L4041) |
| gdn / MMA1 | 2 / 5 | KS、QS、U、Y、S；[mma_cg1_chunk:2891](sources/template_gdn_prefill/gated_delta_net_chunked.py#L2891) |
| gdn / Recurrence | 2 / 11 | 初态、九项递推、末存；[_load_initial_state:4742](sources/template_gdn_prefill/gated_delta_net_chunked.py#L4742)、[compute_group_1_chunk:4969](sources/template_gdn_prefill/gated_delta_net_chunked.py#L4969)、[_store_final_state:4861](sources/template_gdn_prefill/gated_delta_net_chunked.py#L4861) |
| gdn / ScalarOutput | 4 / 9 | bound、GateRegs、Gate、Beta、O；[kernel:2059](sources/template_gdn_prefill/gated_delta_net_chunked.py#L2059)、[load_gate_beta_warp:2643](sources/template_gdn_prefill/gated_delta_net_chunked.py#L2643)、[epilogue_warp:5543](sources/template_gdn_prefill/gated_delta_net_chunked.py#L5543) |

- 域完整列出：index_cast 为 `{}`；gdn 保留 `t=worktile`、`c=[0,Np)`、`p=[0,Np/2)`、`n=flatten(t,c)`。j 仍是各局部遍历重新绑定的索引。没有删域、改名或依赖替换；全部 waits_for、数学、调度、成员、执行模式、资源、契约与里程碑继承原 root。
- 每工作仍是一序列、一值头的六十四行半片，两半独立。正序列长度使空序列分支不可达，显式初态保证首块也执行状态投影；零初态不会省去加载。补齐块仍使用中性填充值并产生完整动作事件，输出越界由描述符抑制，不能将无有效行误写为无动作。边界转换只执行一次，初末状态动作则属于每次工作。
- 列表决定角色的发起顺序；装载异步性、两个矩阵发行者各自的累积完成语义仍生效。下一条开始不等于上一条数据已完成，读者仍依赖原发布及完成等待。串行计算组沿原契约前进，循环结束不隐式归还、同步或重置资源。矩阵分量与工作层次因此保留，避免将独立角色误读成同步推进。
- Inverse 保留“两份 A 都准备好→invert→两份 Tinv 都发布→逐项 QK/Bmat”。三个相邻 j 遍历不能合并。MMA0 的固定四次发射继续短展开。KK/QK 的 read 仍包含提前归还累加槽；prepare 归还 gate、Tinv 归还 beta，W 保留至 Bmat，未合并这些边界。
- ScalarOutput 先准备最多四项；主体每次先补 j+4，再存 j；尾部仅存。源码 [K:2109](sources/template_gdn_prefill/gated_delta_net_chunked.py#L2109) 先准备两项，Np>2 再准备两项，[K:2211](sources/template_gdn_prefill/gated_delta_net_chunked.py#L2211) 在存储循环内判断前瞻。Np 为至少二的偶数，故以下相邻遍历保留同一动作序列，无须 if。

| Np（对应 L_b） | 初备 j | 主体：补备→存储 | 尾存 j |
| --- | --- | --- | --- |
| 2（1–128） | 0、1 | 空 | 0、1 |
| 4（129–256） | 0…3 | 空 | 0…3 |
| 6（257–384） | 0…3 | 4→0，5→1 | 2…5 |

- 多工作 `[2,6,4]` 的 n 窗口为 `[0,2)`、`[2,8)`、`[8,12)`，槽号持续推进。每工作均加载初态、等待自己的最后 S 后末存；`n[t,c+1]` 不跨工作递推，`n+1` 等复用移位仍跨工作。各窗口偶数长度保证 `n+1-n%2` 始终指向本对后块。Q/K 的双读者、Gate 的双读者、Beta 的 Tinv 归还以及唯一 O 槽的前次存储完成等待均保留；循环边界不排空资源。例中第二工作的 `load:Q(3)` 仍等第一工作末对 QK 完成，`publish:O(2)` 仍等 `store:O(1)`。同一对也可跨三槽环尾部，第三槽与第一槽不能当成一次发布；连续序号保留了这层区别。
- 原稿与源码粒度不同：Load 把有效前块、末有效块和补齐块的同名调用归为一次 n 遍历；bound 契约概括描述符更新和首次各输入访问前的 fence。invert 包含 8→16→32→64 层与 rendezvous；源码 `_finish_pair_inverse` 还兼任两个 Tinv 发布，不能把函数边界当动作边界。前缀扫描、矩阵子块和线程协作留在动作内部；残差的两次舍入、更新操作数的不同变换与浮点状态保留也未改动。一次性描述符初始化、分配与 worker tail 原稿未单列，本次不造新动作。原稿所述旧在途请求跨描述符更新的顺序未证明，仍是局限。
- 验证：PyYAML 解析及单键控制检查通过，原 root 和五份源码 SHA256 均匹配 manifest。对 Np∈{2,4,6} 的一至三个工作共 39 组展开，12,483 次动作与原稿逐项同序同身份；13,966 条显式等待边的窗口内端点存在，并核对了源码标量顺序、Inverse 整批边界及每工作初末。此检查覆盖有限域与原动作粒度，未运行 GPU，也不证明所有异步时序或性能。
