# gated_delta_rule_prefill

## 范围

记 B=`batch_count`、T=`total_tokens`。共享 Contract 是固定 BF16/FP32 的 packed-varlen GDN core：B=1..16，总 T≤16384，每序列非空。完整 YAML 收录 B>1 的具体实现；B=1 的同机制局部变化在下文记录。来源固定为 `KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019/gdn_prefill/aka_solution`，主入口 `solution.py:Model.forward` 强制 M64。

M64/M128 是每 CTA 的 value/state 行数，两者的 token chunk 都为 64。公开接口不支持 None 初态、缺失终态、其他 dtype、checkpoint 或 state pool；可读内部 adapter 的这些功能不扩张本任务。源码 descriptor 更新对旧在途读取的顺序尚未确立，见 root 对 LOAD 的限制。

## 完整设计

| 设计 | 具体机制与保留理由 | 关系 |
| --- | --- | --- |
| [root](root.yaml) | 每个持久 CTA 保留一份序列头的 64 行状态；双 MMA issuer 配合成对 inverse、chunk 递推与同 chunk 输出，动态 descriptor 隔离序列尾部 | 阅读入口，保存唯一 Contract |

## 变体与组合

### 单序列输入与通知（局部）

- 基线：[root](root.yaml) 的 `GDN.LOAD`、`GDN.SCALAROUTPUT`、`GDN.RECURRENCE`、`GDN.MMA1` 和 `gatering/betaring`。
- 差异与联动：B=1 时，scheduler 直接以 CTA id 定位 head/value-half，边界是 `[0,T)`；Q/K/V/O 用 embedded descriptor，省去逐工作更新。`INDEX_CAST` 仍真实提交，Cu32 结果在 core 中未读取。ScalarOutput 给 alpha/beta 各增加两个 `[64,4]` FP32 raw TMA 槽，先发 chunk 0/1，以 `stage=c%2, phase=(c//2)%2` 等待传输；读出当前 head 列后 `sync_warp` 与 async.shared fence，再复用 raw 槽预取 c+2。其 gate 扫描结果和 beta 写入五槽后，warp 同步并由 elected lane 执行 ready arrival；两消费者以 `stage=c%5, phase=(c//5)%2` 等待该次发布。Inverse 和 Recurrence 不再归还 scalar empty-stage。五槽安全来自既有四块预取和输出推进：写 c+5 前已完成 store O(c)，传递保证 c 的 gate/beta 消费完成。原来的显式 gate/betachunk Released 边由该推进约束承担。
- 差异与联动：Recurrence→MMA1 的 StateOperand、R、Ubar/Udec 合为同一个 barrier 的三次通知；最后一次同时发布 Ubar 和 Udec，MMA1 不再单独等待 Udec。phase 依次为 `c&1`、`(c+1)&1`、`c&1`；前一通知被消费后所产生的 KS、U、S 结果约束后续通知，防止套圈。其余 inverse、KS/QS/U/Y/S 值、FP32 state、输出生产者和工作归属均沿用 root。
- 条件与依据：`gdn_prefill.py:get_gdn_launch_metadata`、`gated_delta_net_chunked.py:load_gate_beta_tma_warp/compute_group_0_pair/mma_cg1_chunk/compute_group_1_chunk`，继承上述固定来源。只有 B=1 且 T%128==0 时才关闭标量尾块 predicate；T%64==0 仍可能需要一个全 padding chunk。`use_static_scale` 比较 Python `scale==key_dim**-0.5`；实际 input.py 用 `1/math.sqrt(key_dim)`，相邻 binary64 值并不相等，两者转 FP32 相同。zero/continuation 都加载显式初态。

### Value 行块大小（参数）

- 基线：[root](root.yaml) 的 `GDN.Decomposition.Partition.value_dim`、`GDN.Mapping`、`GDN.kring`、`GDN.RECURRENCE`、`GDN.INVERSE`。
- 差异与联动：内部 override=128 让一个 CTA 覆盖整个 value 维，工作数从 B*64 变成 B*32，K 从四槽改三槽，V/O/state 局部形状扩大；state 先保留整块再衰减，TMEM/SMEM 搬运布局相应变化。七个乘积、64-token chunk、inverse 和输出生产者不变。M128 的 QK accumulator 在加权写出之后归还，M64 在完整寄存器捕获之后提前归还；需同步调整相应 Released 时点，不能只替换数字。
- 条件与依据：固定来源 `gdn_prefill.py:get_gdn_launch_metadata`、`gated_delta_net_chunked.py:_setup_attributes/compute_group_0_pair/compute_group_1_chunk`。M128 仅是内部可调用参数，本候选 Model.forward 不选择它；历史 forced-M128 记录绑定另一 SHA，不能当作当前候选验收。
