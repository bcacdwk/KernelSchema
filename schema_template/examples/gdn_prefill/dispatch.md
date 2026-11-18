# gated_delta_rule_prefill

## 范围

记 `B=batch_count`、`T=total_tokens`。唯一共享 Contract 定义固定 BF16/FP32 的 packed-varlen GDN 核心计算：`B=1..16`，总 `T <= 16384`，每条序列非空。完整 YAML（`root.yaml`）收录 `B > 1` 方案；`B=1` 的同机制局部变化与内部 M128 参数见下文。固定来源为 `KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019/gdn_prefill/aka_solution`，公开入口 `solution.py:Model.forward` 固定选用 M64。

M64 与 M128 区分每 CTA 承担的 value/state 行数，两者的 token chunk 长度均为 64。公开接口不支持 `None` 初态、省略终态、其他 dtype、checkpoint 或 state pool，内部 adapter 的扩展分支不改变本任务 Contract。多序列更新 TMA descriptor 相对旧在途读取的顺序保证尚未确立，详见 `root.yaml` 中 `GDN.LOAD` 与 `REALIZATION` 的记录。

## 完整设计

| 设计 | 具体机制与保留理由 | 关系 |
| --- | --- | --- |
| [root](root.yaml) | 每个持久 CTA 保留一份序列头的 64 行状态；双 MMA issuer 配合成对 inverse、chunk 递推与同 chunk 输出，动态 descriptor 隔离序列尾部 | 阅读入口，保存唯一 Contract |

## 变体与组合

### 单序列输入与通知（局部）

- 基线：[root](root.yaml) 的 `GDN.LOAD`、`GDN.SCALAROUTPUT`、`GDN.RECURRENCE`、`GDN.MMA1` 和 `gatering/betaring`。
- 差异与联动：`B=1` 时直接按 CTA id 定位 `(head, value_half)`，序列区间固定为 `[0, T)`；Q/K/V/O 改用 embedded TMA descriptor，省去逐工作 descriptor 更新（`INDEX_CAST` 仍照常提交，核心不读取 `cu32`）。`SCALAROUTPUT` 为 `alpha/beta` 各增设两个 `[64, 4]` FP32 raw TMA 槽：先预取 chunk `0/1`，按 `stage=c%2, phase=(c//2)%2` 等待到齐，取出当前 head 列后执行 `sync_warp` 与 `async.shared` fence 再复用槽位预取 `c+2`；算出的前缀与 `betachunk` 写入 `gatering/betaring` 五槽后同步并发一次联合 ready 通知，`INVERSE` 与 `RECURRENCE` 按 `stage=c%5, phase=(c//5)%2` 等待且不再回传 empty-stage，五槽覆写安全改由四块前瞻与 `store:o(c)` 完成推进传递保证（替代显式的 `gate/betachunk` `Released` 边）。同时，`RECURRENCE -> MMA1` 的 `stateoperand`、`r`、`ubar/udec` 合并到同一 barrier 的三次通知（phase 依次为 `c&1`、`(c+1)&1`、`c&1`），末次通知一并发布 `ubar` 与 `udec`，`MMA1` 不再单独等待 `signal:udec`，由前序 `ks/u/s` 消费链防止套圈；其余求逆、`ks/qs/u/y/s` 数值、FP32 `s`、输出生产者与工作归属均沿用 `root`。
- 条件与依据：同源 `gdn_prefill.py:get_gdn_launch_metadata` 与 `gated_delta_net_chunked.py:load_gate_beta_tma_warp/compute_group_0_pair/mma_cg1_chunk/compute_group_1_chunk`。仅当 `B=1` 且 `T%128==0` 时关闭标量尾块 predicate（`T%64==0` 但非 128 倍数时仍含一个全 padding 合成块）。`use_static_scale` 检查 Python `scale == key_dim**-0.5`，与 `input.py` 的 `1/math.sqrt(key_dim)` 差 1 ULP binary64（转 FP32 相同）。zero 与 continuation 模式均加载显式初态。

### Value 行块大小（参数）

- 基线：[root](root.yaml) 的 `GDN.Decomposition.Partition.value_dim`、`GDN.Mapping`、`GDN.kring`、`GDN.RECURRENCE`、`GDN.INVERSE`。
- 差异与联动：内部参数设为 128 时，单个 CTA 覆盖完整 `value_dim=128`，总工作数由 `B*64` 减为 `B*32`，`kring` 由四槽改为三槽，V/O/state 局部片形状加倍；状态先保留整块再做衰减，TMEM/SMEM 搬运分片相应调整。七次矩阵乘、64-token chunk、成对求逆和输出生产者不变，但 `INVERSE` 中 `qk` TMEM 槽在加权写出 `bmat` 后才归还（M64 在 `load:qkcaptured` 读入寄存器后即提前归还），须同步移动对应的 `Released` 时点，不能只替换维度常数。
- 条件与依据：同源 `gdn_prefill.py:get_gdn_launch_metadata` 与 `gated_delta_net_chunked.py:_setup_attributes/compute_group_0_pair/compute_group_1_chunk`。M128 仅为内部可传参路径，当前 `Model.forward` 不选用；历史 forced-M128 测试绑定另一源码提交，不作为当前候选的验收依据。
