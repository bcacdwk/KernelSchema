# FA4 decode 选型交接

选型保留 **4 份完整设计**：`root`、`q4_cluster`、`q4_direct`、`q1_global`。旧 `q4_compact` 降为一次局部记录；旧 `q1_cluster / q1_direct` 降为两个已有机制的组合记录。没有另一个已识别而隐去的独立整体方案。此文件保留最初选型依据；四份现已实施并经独立复核，最终检查与图范围见 [REVIEW.md](REVIEW.md)。未运行 GPU 或提交 Git。

## 固定来源与公共 Contract

来源统一为 `KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019`。下文 `R` 指其 `fa4_decode/`，`K` 指 `R/aka_solution/qoder_greenfield_decode_attention/`。选定的是 v55，不能代入 `R/solution/` 或 44 题库的相似实现。

已全文核对 [operator.py][host] 的 `run / _select_policy / _compile_kernel`，以及 [gqa_decode_paged.py][paged]、[gqa_decode.py][combine]、[mask.py][mask]。paged 文件 SHA-256 为 `ec930c32758ea4f34e2df9fefa2de90f3ef345dc11fa66948fef74ae016fbe92`，与固定材料一致。

| 事实层级 | 应进入共享 Contract 或范围说明的内容 |
| --- | --- |
| [shape_train.json][train] 的明确域 | `1<=B<=17`；`Q∈{1,4}`；每项序列长度的公开上限为 131072。固定 `Hq=16, Hkv=1, D=256, page_size=64, block_table_columns=16434, workspace_bytes=413138944, bmm1_scale=0.0625, bmm2_scale=1.0`。不能沿旧 root 写成 `seq_lens<=64*16434`。 |
| [input.py][input] 对域的补充约束 | `L_b>=Q`，故 Q4 不接受 shape_domain 单独阅读时可能误取的 1、2、3。`max_seq_len=max L_b`；query/out 为 `[B*Q,16,256]`，KV 为 `[P,2,1,64,256]`，query/KV 是 E4M3FN，out 是 BF16，表和长度是 int32，workspace 是 uint8。张量在同一 CUDA 设备且连续。 |
| [reference.py][reference] 的数学和外部结果 | 以 block table 解码 HND cache；第 t 个 query 仅见 `0<=k<=L_b-Q+t`；FP32 score、softmax、PV，最后乘 bmm2_scale 并转 BF16。返回值 alias 传入 out，并覆盖全部 out；无 LSE 输出。workspace 是 scratch，不是数学输出。 |
| 固定调用参数 | `window_left=-1, sinks=None, o_sf_scale=None`；None 是必须传入的 `python_none` 单值，`shape: []`，不伪装成零张量或 nullable tensor。复合 bmm scales 已包含边界上的量化缩放，不新增 Q/K/V scale ABI。 |
| 输入构造与样例观察 | 构造器取 `P=sum ceil(L_b/64)`，各请求分配连续、不共享的物理页，未用表项填 -1，workspace 初始全零；随机幅度校准值不是数学值域。reference 的逻辑页映射本身不要求 identity。表的有效前缀须映射合法物理页；不要把样例列表、连续 page ID 或随机分布当作明示公共域。 |
| 实现额外接受范围 | `_validate` 同时接受 E5M2，类构造器允许别的 page/tile/head 参数；它们不是本题公开 Contract，也不是本轮新代表。BLASST、sink、LSE、非因果/滑窗入口在选定 host 均未启用。 |

[metadata.json][metadata] 明示 returned output 与 mutated out 均使用 `atol=0.06, rtol=0.04`。2026-10-21 [harness.py][harness] 的 `checks`（44–51）给出 FP32 比较公式，并要求 shape/dtype 一致、双方有限；`correctness`（70–96）另外检查 out alias、未声明输入不变，并在计算前把 out 填 NaN。可保留完整已知判据，不能把 AGATE 的汇总最大相对误差改写成新的阈值。

## 完整代表（现已实施）

所有代表固定 page=64、D=256、head tile=16、KV tile=256、E4M3FN→BF16、PDL 开启、table_stride=16434、BLASST/sink/LSE 关闭。每份采用单次已编译调用的稳态范围：native launcher 已建立且命中同一输入对象/签名、线程和当前设备，调用不处于 stream capture；不把首次 native probe 包含在稳态性能或 launch 数中。YAML 的 `when` 还须包含下表实际承担范围，不能只写模式名称。

`L=max_seq_len, P=kv_cache.shape[0], skew=(128*P<B*L)`。以下范围刻意选取 host 的简单子域；其余同机制配置在 dispatch 记录，不由 root 兜底。

| id | 具体配置与真实实例 | 源码路径 | 保留理由 | 建议 when 子域 | 可共享整区 |
| --- | --- | --- | --- | --- | --- |
| `root` | Q4；prediction tile=2；单 softmax WG；384 threads；S=9；非 compact；warp-parallel combine。`shape_valid[14]`：B=15、L=31420、P=4067，host 返回 `(9,kernel)`。 | `K/operator.py:205–224,269–290`；`K/gqa_decode_paged.py:459–530`；`K/gqa_decode.py:328–568` | 独立 CTA 写 FP32 `(Opartial,Mpartial,Lpartial)`，另一个 kernel 按 query head 归并、归一化并唯一写 BF16 输出；这是完整的跨 kernel split-KV 方案。 | `Q=4, B∈{15,16}, 24576<=L<=131072, not skew`；固定 S=9、compact=false。 | root 三区本地。 |
| `q4_cluster` | Q4；prediction tile=2；单 softmax WG；384 threads；S=16、cluster=16。`shape_valid[0]`：B=1、L=17844、P=279，host 返回 `(16,atomic)`。 | `K/operator.py:196–203`；`K/gqa_decode_paged.py:796–827,1847–1917,2086–2105`；`K/gqa_decode.py:205–326` | 16 CTA 通过 DSMEM 蝶形交换 max/sum，各自计算归一化贡献；split 0 清零、发布 zero_done，所有活跃 split 向 BF16 out 做 TMA reduce-add。最终输出者、舍入路径和主要 CTA 协作均不同。 | `Q=4, B=1, 4096<=L<24576`；固定 S=16。 | 与 root 的输出数学、worker 和 epilogue 均不同，预计三区本地。 |
| `q4_direct` | Q4；prediction tile=2；单 softmax WG；384 threads；S=1、cluster=1。域内构造实例 B=1、L=64、P=1，host 返回 `(1,none)`；30 个记录样例没有该短序列实例。 | `K/operator.py:84–87`；`K/gqa_decode_paged.py:1138–1145,1547–1553,2025–2051,2106–2120`；`K/gqa_decode.py:128–162` | 每个 query tile 的单 CTA 自己形成分母和最终 out，无 split workspace/外部 combine/跨 CTA 统计。必须完整保留 N=1 时空 parity 初始化和收尾，不能把它写成理想化的一次 GEMM。 | `Q=4, 1<=B<=17, 4<=L<=256`，且每个 `L_b>=4`；固定 S=1。 | 预计三区本地；不能为共享强留多迭代选择器。 |
| `q1_global` | Q1；prediction tile=1；双 softmax WG；512 threads；S=16；warp-parallel combine。`shape_valid[1]`：B=16、L=31413、P=4344，host 返回 `(16,kernel)`。 | `K/operator.py:134–165,269–290`；`K/gqa_decode_paged.py:621–681,1568–1579,1726–1790`；`K/gqa_decode.py:328–568` | 两个 softmax WG 按奇偶迭代分工，交替取得共享 max 的互斥轮次，各自推进 P/S 游标、生产一个 L parity；须重接 Score→Softmax→Value/Correction 的整条主循环生产消费关系。完整保留这一主要协作机制，但不与三种输出组织做笛卡尔积。 | `Q=1, B=16, 24576<=L<32768, not skew, 5*B*L>=576*P`；固定 S=16。 | `Math_reshape: {from: root}` 可成立：同样的 stripe、双 FP8 概率、双 numerator parity 和 FP32 combine；QueryTile/PackedColumns/S 通过本节点绑定。Work/Pipeline 本地。实施时逐区核实，不能共享 root 已固定的 Q4 值。 |

Q4 使用 64/168/256 的 issuer/softmax/correction register 配置；Q1 没有该 `setmaxregister` 重分配。它们是已选实例的固定事实，不在 YAML 内写 `Q==4` 选路。S/P/PageIndex 的四槽和 O/L 的两槽按源码落实；容量 helper 所决定的其他槽数需保留来源推导，不把旧稿中的常量未经核验升级为硬件保证。

## 旧 7 节点去向

| 旧节点 | 新去向 |
| --- | --- |
| `root` | 保留同 ID，重写为上表固定 Q4/S9/global 代表，删除原宽范围选择公式。 |
| `q4_cluster` | 保留同 ID，固定 S16，完整保留 cluster 输出协议。 |
| `q4_direct` | 保留同 ID，固定单块直接输出。 |
| `q1_global` | 保留同 ID，固定 Q1/S16，并复用经核实相同的数学整区。 |
| `q4_compact` | 删除 YAML/PNG；dispatch 记录请求相关 WorkSplit 与 serial combine 的联动。 |
| `q1_cluster` | 删除 YAML/PNG；dispatch 直接定位 `q1_global` 的双 softmax 主循环及 `q4_cluster` 的 cluster 统计/输出组织，记录 Q1 的列宽、线程数和寄存器差异。 |
| `q1_direct` | 删除 YAML/PNG；dispatch 直接定位 `q1_global` 的角色组织及 `q4_direct` 的单块收尾。Q1/N=1 时 odd WG 只提供空 L parity 的通知，不伪造第二份 softmax 计算。 |

## dispatch 分组与直接基线

下表保留选型阶段的对象定位草案；现行对象名和联动以 examples/fa4_decode/dispatch.md 为准。每条共同差额只写一次。

| 类型/分组 | 完整基线与对象 | 应记录的差异、联动、条件与依据 |
| --- | --- | --- |
| 参数：occupancy/wave split 数 | `root`、`q1_global` 的 `SplitCount / stripe / partial / combine`；`q4_cluster` 的 `SplitCount / ClusterMax / ClusterSum / zero_done` | `_select_policy`（78–224）完整分成 Q1 与 Q4 的长/短/skew 带，表 `Q4_LONG_WAVE_SPLITS` 取 14/12/10/9/6/5 等非二次幂也是相同 global 组织。global 计数最大64；atomic 仅2/4/8/16。改变 S 联动 grid、循环步长、workspace、cluster 轮数；不新增 YAML。combine 的 unroll16 仅在 S%16=0 使用，其余 unroll8（gqa_decode.py:495–555）；不是新的数学方案。 |
| 局部：compact Q4 请求内 split | `root` 的 `WorkSplit / stripe / Mpartial / Lpartial / Opartial / combine` | host `compact_q4_splits` 门控在 operator.py:730–747，device 规则在 paged:1078–1103。名义 grid 与 workspace S 不变；每 CTA 读 batch min/max，S32 按 `max(4,ceil(L_b*S/Lmax))`，S8 按相近长度/批大小规则取6/7或比例数。超出 WorkSplit 的 CTA 走既有空 split；逻辑 KV 步长改为 WorkSplit。combine 同时改为非 warp-parallel 路径，按 `Mpartial>-inf` 跳过空 O 读取。固定 worker 预分配、归并拥有者、主要 KQ/PV 与 correction 流水线不变，局部边界闭合；不是持久领取或新增整体调度。 |
| 组合：Q1 cluster/direct | `q1_global` 的 `SoftmaxEven / SoftmaxOdd / Maximum / P_pair / ExponentialSum`，并直接列 `q4_cluster` 的归并对象或 `q4_direct` 的单块输出对象 | 两个组合均被同一 host 实际选择。Q1 cluster：如 shape_valid[2]，B1/L17845/S16；Q1 direct：域内 B1/L64/S1。双 WG 只更换局部统计的生产分工，统计接口仍为每列 FP32 max/sum/两 parity numerator；cluster/direct 原归并边界成立。列宽由32改16，CTA 由384改512，无 Q4 register 重分配，mask用Q1；不能泛称任何开关都可自由组合。 |
| 局部：调用前缀与 native 缓存 | 四份完整基线的 `partial / combine` 或 `MAIN` 提交及 `out / workspace` | operator.py:474–583、652–843：miss 后先正常调用 fmha，再尝试 clone 当前 out、重复同一 fmha、设备同步、比较相对范数（阈值1e-2）、失败恢复 out；成功后缓存 weakref/签名/thread/keepalive。命中稳态才是单次 ffi 调用。该自校验不替代 Contract tolerance，不新增完整设计，也不宣称稳态覆盖首次。stream capture 下不建立 native launcher；同一 kernel 组织仍适用。 |

实现额外接受的 E5M2、其它 page/head/tile、BLASST、sink、LSE 仅在范围中明确排除；不以本题变体为它们建立完整节点。

## 实施必须保留的机制与边界

1. **数值链**：`KQ` FP32 累加；mask 后基于 `bmm1_scale*log2(e)` 的 fast exp2；最大值归约用 finite `min_f32` 防全 mask 产生 `-inf-(-inf)`。`P_hi=fp8(448*P)`，`P_lo=fp8(448*P-fp32(P_hi))`；PV 在每个原生 K slice 先 hi 后 lo 累加到同一 FP32 numerator；分母用未量化 P。host `scale_o=bmm2_scale/448`。不得删掉 residual 的第二次 PV 或把分母改成量化 P 的和（paged:1664–1788）。
2. **两 parity correction**：O 与 L 有两个槽，correction 用 `m(i-2)-m(i)` 重标度旧槽，最终再合并两 parity。Q4 单 WG 交替 L phase；Q1 双 WG 各负责一个 phase，共享 max 按迭代次序交接。N=1 仍初始化两个 O 槽和空 L parity，Softmax 发空 parity 通知，VP 补一个空 O commit，Correction 读两槽后归并；不要用“只有一个 tile”删除这些实际动作（paged:1547–1579,1834–1845,1919–2051）。
3. **PDL 不是输出完成**：decode 的 reduction warp 在统计 epilogue 后调用 `griddepcontrol_launch_dependents`（paged:2120）；O 的发布/搬运还有自己的完成关系。combine 在读取 partial 前执行 `griddepcontrol_wait`（gqa_decode.py:441–462），末端发 launch_dependents 在最终 out store 之前（557–565）。Flow 分开 launch-ready 与数据可读/全部输出可见；不能把 ready 当作 done，也不能把 host 返回当作完成。
4. **issue turn 与真正完成分开**：K/V TMA、KQ/VP MMA 在获取最后一个 slice 后、尚未执行该 slice 的 copy/gemm 前通知对方轮次。`yield_turn` 只表示对方可以推进；KV operand、S/P/O 的完成和释放由各 pipeline handle/fence 保证。四个 KV slices 共用环；同一 role 的异步引擎不能无依据地统称共同有序完成（paged:1249–1269,1343–1364,1418–1448,1510–1544）。
5. **PageIndex 四槽环**：首次按固定 table stride 和 split index 提前 cp.async，甚至空 split 也完成这次准备；先 wait_group(0)+CTA sync，再判断 exit。每轮提前加载下一组4页，wait_group(1)取得本轮；V 稍后读同一个槽。Lead=2，`pt_stages=4>Lead+1` 仅为容量断言；K3 在等待 V0 turn 之前就预取 Page4，未找到 V0 必先捕获槽0的保证。最初“轮次保证安全”的选型判断已撤回，YAML 不添加源码没有的覆盖保护边，保留未决，见 [pageindex_order.md](pageindex_order.md)（paged:1016–1063,1224–1256,1336–1379）。
6. **空 split 是已提交工作**：先完成 TMEM/共享统计/必要 cluster barrier 初始化，再走早退。`M=-inf,L=0`，TMEM 释放；kernel 模式发布中性统计，atomic 模式仍参加 DSMEM 归约；不发 O load/store/reduce-add。不能直接删除整个 CTA（paged:796–827,972–989,1138–1145,2063–2120）。
7. **空 numerator 的真实边界**：普通 global combine 对所有名义 split 读取 Opartial 并乘权重0；空 split 没有覆盖 Opartial。metadata/input 的首次清零保证首次空槽为有限零，固定布局反复使用可沿已有有限 Opartial；不能无据保证同一 workspace 跨不同 `(S,B,Q)` 布局复用后仍有限，因为旧 Mpartial 的 -inf 可能落入新 Opartial 区域。记录这个来源边界，不补造每次清零或 guard。compact combine 的 `Mpartial>-inf` guard 确实跳过空 payload。`m_bsh` 虽仍由 host 切片传入，但 kernel 不使用、更不做旧注释声称的 atomic-max/-inf prefill（paged:434–437）。
8. **cluster 输出协议**：所有成员在早退前初始化 DSMEM 接收 barrier，cluster rendezvous 只发布初始化。split0 的 correction warp 在 ramp 中清 out，GPU fence 后用 DSMEM 消息通知每个 zero_done；每个活跃 split 的 TMA output 先 wait zero_done，再 async-proxy fence。统计先蝶形 max，再校正本地 sum，后蝶形 sum；贡献转 BF16 后按实际完成次序 reduce-add。没有唯一 FP32 numerator 汇总/最终一次 BF16 cast（paged:796–827,1390–1398,1847–1917；gqa_decode.py:205–326）。
9. **资源退休与物化**：K/V 共用 SMEM，最终 O staging 又覆盖该区；必须等所有 K/V operand 读取结束。TMEM 释放由 Correction 捕获最后两个 O 槽并释放后允许，不能提前到 MMA issue 结束。workspace 中 unused `m` 的实际切片保留在必要实现说明，不能声称所有划分出的 scratch 都是数学输出。

## 有限图与验证边界

建议只保存每份一个代表 PNG；再用少量 `--graph check` 验证空工作/单轮边界。最终参数名由正文决定，下表给出真实整数实例，不是已执行的命令。所有文件使用已选固定 S/角色/槽配置，不为压图改 S 或算法。

| 节点 | 主图代表 | 必要边界检查 |
| --- | --- | --- |
| root | shape_valid[14] 的最长请求 `L_b=31420,T_b=123,r=0,S=9,N_r=14`；一个 partial worker 和一个 combine worker | 同一固定 S9 机制的 `N_r=0/1`；空 O 不写但 combine 的零权读取仍出现。 |
| q4_cluster | shape_valid[0]：`T_b=70,S=16`，成员0–5为5轮、其余4轮；保留16成员和4轮蝶形 | 短请求存在空成员时，中性统计/接收 barrier 仍有事件；split0 始终非空，zero_done 总会发布。若同一 when 内 B1 范围没有空成员，空成员属于记录中的同机制其他 batch/S 参数样本，不伪装成主图范围内。 |
| q4_direct | `B=1,Q=4,L_b=64,S=1,N=1` | 显示空 parity 的初始化、通知、读取与最终 TMEM 释放；没有空 CTA。 |
| q1_global | shape_valid[1] 最长请求 `L_b=31413,T_b=123,r=0,S=16,N_r=8` | 双 WG 奇偶轮次、N=1 的 odd 通知以及 N=0 初始化收尾。 |

root 的14轮和 Q1 的8轮足以触及 PageIndex 四槽复用及多个 correction parity；cluster检查必须给各成员真实循环长度。最终PNG采用when内B1/L4096/T_b16的真实16个一轮成员；T_b70的不等长主输入另做完整图检查，见REVIEW。源码 `yield_turn` 与 done 分离、全局 PDL 事件绑定、空分支 Event 是否存在，是最可能需要细化动作的地方。若现有语法/工具确实无法表达，提交最小复现和 DOC 依据，不改 kernel、不添加虚构同步。

已有证据：[AGATE_VALIDATION.json][agate] 记录30/30通过；[2026-10-21 validate.json][validate] 记录30/30，绑定所读完整 Python 源哈希，包含 out/alias/未修改输入检查；[evidence summary][evidence] 标明 v55、L20D/SM103、30 shape。它们支持记录样例，不证明整个公开域、短序列 direct、所有空 split 或跨布局 scratch 复用。已有可核实 Hardware/Execution/版本事实应换成当前字段保留；未知资料入口用 DOC2 正式占位，不开展新的硬件调查。

选型阶段实际完成：静态解析并执行仅含 Python 标量算术的 `_select_policy` AST，核对全部30个 shape_valid 的 `(S,reduction,compact)` 与上表实例，未导入 torch/cutlass、未执行 kernel；核对 paged SHA-256。后续四份实施、独立全文审阅及最终 YAML/有限图验收已完成，准确结果见 [REVIEW.md](REVIEW.md)。

[host]: ../../../kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/qoder_greenfield_decode_attention/operator.py
[paged]: ../../../kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/qoder_greenfield_decode_attention/gqa_decode_paged.py
[combine]: ../../../kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/qoder_greenfield_decode_attention/gqa_decode.py
[mask]: ../../../kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/qoder_greenfield_decode_attention/mask.py
[train]: ../../../kernel_wiki/aka-four-problems-20261019/fa4_decode/shape_train.json
[input]: ../../../kernel_wiki/aka-four-problems-20261019/fa4_decode/input.py
[reference]: ../../../kernel_wiki/aka-four-problems-20261019/fa4_decode/reference.py
[metadata]: ../../../kernel_wiki/aka-four-problems-20261019/fa4_decode/metadata.json
[harness]: ../../../kernel_wiki/aka-four-problems-20261019/validation-20261021/harness.py
[agate]: ../../../kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/AGATE_VALIDATION.json
[validate]: ../../../kernel_wiki/aka-four-problems-20261019/validation-20261021/results/fa4_decode/validate.json
[evidence]: ../../../kernel_wiki/aka-four-problems-20261019/evidence/fa4_decode/summary.json
