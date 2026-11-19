# MLA 直接审修交付

当前正式目录保持 **1 份完整设计**，`root.yaml` 从 Version 5 修订为 **Version 6**，并同步修订 `dispatch.md`。本代理 `/root/mla_schema_review_xhigh`（gpt-6-astra/xhigh）真实承担 worker 与 reviewer 两项职责，直接审阅固定实现、修改正文后回读自审；没有另一个独立 reviewer，也没有调用 extract、恢复旧 run 或运行 GPU。

正式目录：`schema_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8`。完整结构化记录见 [review.json](review.json)，当前 raw manifest 与记录摘要见 [after.json](after.json)。修改前的 root、dispatch 和两份 `.extraction` 文件按原字节保存于 [before/](before/)，清单见 [before.json](before.json)。

## 完整自审范围

- 对照 `schema_template/README.md` 与现行 DOC0、DOC1、DOC3、DOC4、DOC5、DOC6 核对正文语义、版本、变体闭合、Storage 和事件规则；未修改或放宽规则。
- 从固定提交 `24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2` 核对整个算子来源目录的 24 份文件。候选和七份 problem 文件的 SHA 同时与来源 `SELECTION.json` 一致。全部源码读取均以该固定材料为事实依据，旧审修报告只用于定位线索。
- 完整复核 29 个 Input、两个逻辑 Output、caller-owned 别名与 guards、所有 dtype/shape/default/nullable/range、参考数学及输出返回条件。AST 核对显示 Contract 的 29 个输入名和顺序均与 candidate/reference 的 `Model.forward` 一致。
- 独立复核准备与 forward 边界、数学改写和所有数值分支、工作表和队列、14 个 Storage 条目、五个角色的完整 Pipe/Waiting、最终释放，以及三种可达参数/局部变化和组合的必要联动。

## 确认保留的契约

`shape_range.json` 明确写出 `query_lengths == kv_lengths`、五组相关 head 几何、长度与批量范围、`window_left=-1`，并允许 causal、sink、return_lse 变化。当前 root 的域据此保留；没有把十个样本、输入生成默认值或源码中宽泛的 helper 签名当作完整域。metadata 描述中的更宽 head 家族文字不覆盖具体 `shape_range` 约束。

`sage_attn_sfs` 是一个整体 `python_tuple`，`shape: []` 与四 None 默认值正确；`num_elts_per_sage_attn_blk` 是四个 `python_int` 逻辑元素及全零默认值。两项没有固定 singleton range；本轮没有重改它们，也没有因为未参与数值计算而删除任何 ABI 对象。`out=None` 的候选自行分配路径和 4-D legacy 输入不属于冻结调用；`lse` 只在 return_lse=True 时由调用者提供。workspace 是允许改写的 scratch，当前 MODE0 不使用它。

外部比较器仍未绑定。metadata 中 `mutated_inputs.out` 和 default 的阈值分别保留，未挑选更宽阈值、合并未知输出组或套用 Schema 的默认相对误差公式。源码注释中的 `|a-r| <= .01+.05|a|` 只被记录为注释主张；它没有补齐外部 comparator 的路由、mismatch/rel-L2 定义和归一化。

## 实质修订

1. **数值链与误差界。** 按源码补准 FP32 device scale、`-FLT_MAX` 初态和哨兵、offset 的 `fmaf(-safe,c,8)`、四条模 4 的 FP16 标量求和链及其三次 FP32 合并、输出 LSE 的 FMA/log1p，以及 tiny 路径的逐 key FMA 递推。移除“实际 u 等于理想指数”“任意权重舍入相对误差不超过 2^-4”“全链永不饱和”和 tiny 路径精确实数/bitwise 保证。保留正常数的舍入界，明确次正规数、快速指数及任意正 scale 的来源边界。修正 32 行 accumulator 重标定的数量，tiny FMA 次数按上界表达。
2. **存储与物理表示。** 补记 SOFTMAX 的 packed probability 寄存器副本和 CORRECTION 的 accumulator/输出 drain 寄存器容量；按一个 CTA 计量，归入各 role 已有预算。明确 Q/K 两段 swizzle、TMA 全局 stride、SMEM 基址、Meta 和 tiny padding，以及 TMEM 的 score/P 子区与 O 区分界。原 207184/209920 字节 smem 账本保留，不能把剩余 2736 字节当可额外并发的工作空间。K/V 四槽比三槽多 40960 字节，不能沿用源码注释的“32KB”。
3. **同步与调用生命周期。** 加入实际最终 `__syncthreads` 对应的 `ISSUER.await:exit_join` 及其他角色退出前驱，再释放 TMEM。原 `release:o_slot` 的早释放保留：它保证源读取完毕，允许目标全局 store 仍在途，符合现行 DOC5。还修正 Q completion object 只在 ordinal 0 初始化、QUEUE 执行者为 warp 10 lane 0、sentinel 只写 kv_tiles 字段的表述。额外记录共享静态表/queue/host staging 的调用隔离和单设备条件；主流 launch 串行本身并不能保护非阻塞侧流提前改写表。调用者 metadata 就绪也不能由 restore-before-call 自动推出。
4. **tiny 存续关系。** 去除原 dispatch 对 head-major tiny 重读的伪缺陷和“必须加等待”的要求，改为实际 tail-PV 传递依赖。未修 kernel，未给 tiny 路径添加同步，也未要求 LPT 单调性。
5. **目录与来源范围。** 更新 K3 的新增地址账及静态网格的 exit-join 变体联动；MODE1 是不可达的完整源码方案而非已交付设计。区分 NCU kernel-sum、eager 调用与 CUDA Graph 捕获。记录 host 除 MODE0 申请 209920 字节外，还因配置所有模板而检查 opt-in >=218112。来源证书、注释实验和当前检查分开表达。

## tiny 重读的精确证据

以下行号均指固定 `candidate/solution.py`：

- 945：SOFTMAX 已读取两个 score 半区后发送 `s_free`。单独看到这一行，不能推出 tiny 重读不安全。
- 1064–1090：末 tile 等待 `o_done` 后，tiny helper 完成首 32 列 score 与对应 V tile 的读取；helper 的 TMEM load/wait 位于 698–734。
- 1091–1104：tiny helper 返回之后才写 P、等待 TMEM store 完成并发送 `p_ready`。
- 1333–1371：每个 unit 的尾部 PV 先等待其最后一个 `p_ready`；只有尾部发射结束后才递增 `i_item`、进入下个 unit 的 QK 循环。
- 1359–1369、1403–1411：V 槽的复用须等该 PV 的 `pv_done`（K3）或其后的 `kv_free`（K4）。

因此依赖是 `tiny_read_done → p_ready → tail_PV_issue → next_unit_QK`；V 侧是 `tiny_read_done → p_ready → tail_PV_done → slot_release → refill`。即使局部 key-tile 数是 `[1,5]`，会覆盖旧 score 槽的 QK(n+3) 也不可能先于 tiny 重读。提前发布下一份几何、甚至 loader 提前搬运其他槽，都不能绕过 ISSUER 的尾部 PV 顺序。此证明与表序无关。

输出早释放亦按原件核对：普通 drain 在 1222–1226 的最后一次 TMEM load/wait 后 arrive `o_done`，后续 BF16 pack/store 仍可继续；tiny 分支在 1209–1215 读完 scratch 后 arrive。最终 1490–1492 的 CTA join/deallocation 与新增退出事件相符。没有把目标 store 的 `done` 提前解释为仅源读取完成。

## 最终检查

所有最终检查均返回 0，命令与原始输出见 [checks.json](checks.json)。

- 完整目录检查：`/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py <root.yaml> --read-only --skip-hardware`，一份 YAML，零规范化差异，结构、预算、引用、关系和 dispatch 均通过。
- 有限事件样本：每 unit 1 个 key tile × 5 units，无 LSE，174 actions / 617 edges；4 × 2，有 LSE，197 / 666；7 × 3，无 LSE，482 / 1641。三者均无环，各有两个预期边界跳过。覆盖 score 三槽、K/V 四槽、跨 unit Q 预取、首 credit、哨兵退出和最终 join；不表示验证全部运行轨迹。
- [cpu_checks.py](cpu_checks.py) / [cpu_checks.json](cpu_checks.json)：24 份来源身份、META 布局、SMEM/TMEM/register 数量、SW128 16384 字节双射、tiny stride、两个 causal 模式下 33816 个输出行唯一覆盖、223/288/3840 items 的队列闭合、tiny `[1,5]` 传递关系均通过。固定十例在假设 148 SM 下的 selector 计算也与目录记录一致，未将该假设升级为硬件事实。
- 数值反例：FP16 可精确表示的 `2^-11` 转 E4M3 得 0，相对舍入误差为 1；在 prepared coefficient 下 `safe=4.75`，offset FMA 得 `7.938180446624756`，先乘后加得 `7.938179969787598`。它们只证明旧泛化界和算式替换不成立，不是 GPU 指令误差界或验收比较。

编写中出现的规范化、字数和定位路径诊断已经在当前规则内修正；初次 CPU helper 错把省略的 `sink` 当作必有键，错误日志保存在 [cpu_initial_failure.log](cpu_initial_failure.log)，随后依实际输入默认值处理并通过。没有为这些诊断改变 checker。

## 剩余边界与统一清理交接

硬件区仍保留提取占位，相关规则明确跳过；未运行 GPU、编译或性能测试。快速 exp/log/division、完整动态域数值验收、真实 register spill/occupancy、外部 comparator 与未附 domain/实验配置仍没有新的保证。静态缓存与缺失反向跨流 join 是真实来源限制，必须保留，不能由清理工作删除。

两份 `.extraction` 原字节未改动：receipt 只批准其 `files` 中的旧 root/dispatch；原 P6 的 root SHA 是 `4fec3e…`，并不覆盖本轮进入时已是 Version 5 的 root，更不覆盖 Version 6。旧 `issue-review-20261119.json` 仍含“等待后续复核”的阶段措辞，由后续统一清理代理处理；本报告和 after manifest 是本次真实人工审修的当前依据，不是新 P6 凭据。正式正文未写本轮日期、模型档位、角色安排或 before/after 过程。
