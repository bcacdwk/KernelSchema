可以完整表达，顺排各批工作和独立发布后更清楚。覆盖原 root 的 **2 phase／7 role、48 个静态动作字符串**；使用 **12 个 for、2 个 if**。**while 未使用**：该路径为有界遍历，等待原语内部轮询留在契约内；这不代表验证了其他反馈流程。

- 选固定源码的 q4、每 tile 两查询、非 compact、global partial＋combine。[Model.forward:10](sources/template_fa4_decode/solution.py#L10) 经 `run` 进入 [operator.py:269](sources/template_fa4_decode/qoder_greenfield_decode_attention/operator.py#L269) 的编译入口，277 行指定单 Softmax 组，727–749 行选择策略。[__call__:468](sources/template_fa4_decode/qoder_greenfield_decode_attention/gqa_decode_paged.py#L468) 启动 decode，518 行启动归约。未审其他分支。
- [decode:534](sources/template_fa4_decode/qoder_greenfield_decode_attention/gqa_decode_paged.py#L534) 的 LoadK／LoadV／Score／Value 分支始于 1150／1294／1402／1470 行，Softmax／Correction 始于 1558／1790 行。[Correction:1920](sources/template_fa4_decode/qoder_greenfield_decode_attention/gqa_decode_paged.py#L1920) 先读最多两个最大值，1951 行提前通知 FinalMaximum，2022 行发布 FinalSum 并会合，2032 行归还最终 Y 槽。[reduction_kernel:382](sources/template_fa4_decode/qoder_greenfield_decode_attention/gqa_decode.py#L382) 的 442 行等待前驱，467 行选择 warp 归约，503／517 行遍历全部名义 split，包括零权重空槽。
- 完整重列 domain，只删 `active`；Qpack 和 `zero:Y(0)` 分别置于 `if(N_r > 0)`。原契约的初始化标签改为 0，仍指一次双槽初始化。保留 `i/single/head/corr/corr_prefix/corr_final/tail/u`；循环自身空域已足以表达不发生。combine domain 为 `{}`，`r` 仅绑定于 `for(r=0:SplitCount)`。
- **完整替换 Value、Softmax、Correction 的 waits_for**：初始化两端改为 `zero:Y(0)`；首次握手仅挂 `publish_max:m(0)`，没有扩散到每个 i。等待随目标发生域实例化，无额外门控域。`single` 是单块空槽的真实依赖窗口，`corr/tail` 及 +2、+4 位移继续约束对应版本；其余依赖继承。
- head、corr_prefix 先各走完整批；两段 corr_final 分居 FinalMaximum 两侧；全部 tail 和读取后先发 FinalSum，再逐项捕获 tail。空域不吞后续发布。`N_r=0` 保留统计初始化、三次发布、LoadV/store、Score 收尾及 Value 退役，Ostage／输出无 payload；`N_r=1` 的 EmptySum 信号与空奇偶槽 1 排空仍在原动作契约中，且 `prepare:EmptySum` 在其他长度也作为 no-op 存在。

| N_r | head／corr_prefix／corr_final／tail 索引 | partial 动作／等待边 |
| --- | --- | --- |
| 0 | ∅／∅／∅／∅ | 14／9 |
| 1 | 0／∅／∅／0 | 33／28 |
| 2 | 0,1／∅／∅／0,1 | 50／44 |
| 3 | 0,1／∅／0／1,2 | 68／61 |
| 4 | 0,1／0／1／2,3 | 86／80 |
| 6 | 0,1／0..2／3／4,5 | 122／122 |

- PyYAML 检查通过；控制均单键接非空列表，无重复键。6 个源码及 baseline 哈希匹配。对 `N_r=0..64`（Lead=2、ScoreStageCount=4）和 combine 的 SplitCount=2/4/8/9/12/18/32/64，动作身份、顺序及依赖边与旧稿相等。先解析签名再展开目标；删除 Qpack 或窗口内 ExponentialSum(2)，以及在空域中拼错签名，三个反例均报错。
- 粒度仍沿旧稿：四切片、主／残差概率、短循环无 payload 轮转在契约内，不新增大动作。源码 1951 行在末次读取协议内部发布最大值，旧稿在 `read:m` 后独列发布；源码归约 439 行先置零寄存器、442 行才等待前驱，旧稿把置零放在统计装入后，均保留旧稿顺序。数学、资源、模式、契约及里程碑继承。未运行 GPU；原契约记录的 Lead=2 时 PageIndex 四槽预取覆盖保护未证明，本次声明等价检查不补足该证明。
