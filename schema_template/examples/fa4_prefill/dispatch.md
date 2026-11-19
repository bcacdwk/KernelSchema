# 分页因果分组查询注意力 Prefill

## 范围

固定来源：`KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution`（v37 延续实现，v43 包标签；下文文件名相对 `flash_attn/cute/`，kernel 指 `sm100_hd256_2cta_fmha_forward.py`），目标设备为 L20D/SM103（148 SM）。共享 Contract 在 `root.yaml`：FP8 E4M3FN Q/K/V、16/1 heads、`head_dim=256`、`page=64`、右对齐 causal、BF16 原位输出，公开域 `B=1..16`、总 `Q <= 16384`、`KV <= 131072`。任务入口 `solution.py` 直接调用 `_flash_attn_fwd`（不经过 `flash_attn_varlen_func` 的 `_hd256_fastpath`）；各节点不设隐式优先级。

源码限制：空 split 仅写 `-inf` LSE、不写 `partialo`，`COMBINE` 可预取未写槽但仅累加正权重项；native fast `exp2/log/division` 无独立误差界。默认关闭的 CLC（`persist_mode=4`）引入专用 scheduler warp 与待启动 cluster 取消/提前保位，改变主领取与退出协议，本目录未展开其完整设计；unpacked GQA、非 `page=64` 与未打包的 v53 不在范围内。

## 完整设计

| 设计 | 具体机制与保留理由 | 关系 |
| --- | --- | --- |
| [root](root.yaml) | Compact static，按请求 KV 长度降序映射，每 CTA pair 一份 query 工作；双 K/V producer，直接 BF16 输出 | 阅读入口，保存唯一 Contract |
| [packed_persistent](packed_persistent.yaml) | Compact stride；请求表只建一次，四步二分；环跨工作连续，下一 Q 可与旧输出重叠 | Math_reshape 整区引用 root，其余本地定义 |
| [split_kv](split_kv.yaml) | 固定五路 KV（`ssplit=5, smax=32`）、joint load、FP32 partial O/LSE、独立 combine；空 split 只写 LSE | 独立完整定义 |

## 变体与组合

### 单工作矩形网格（局部）

- 基线：[root](root.yaml) 的 `MAIN.Decomposition`、`MAIN.Schedule`、`requestmap`。
- 差异与联动：每请求固定启动 `crect=ceil(totalq/16)` 个候选，worker 总数改为 `batchcount*crect`；省去 compact 扫描与共享 `requestmap` 发布，直接从网格坐标解码请求与 query 序号并对 `crect` 做 raster 折叠。越界候选仍启动并在主循环前退出；有效 `(b,m)` 的数学计算、单工作寿命与各角色协议不变。
- 条件与依据：kernel `:650–672,1669–1810`。域内例如 `B=1, Q=1024, KV=131072` 在 148 SM 上走 unsplit/static/dual；源码中 `B > 16` 的触发分支不在公开域内。

### Persistent 请求映射（局部）

- 基线：[packed_persistent](packed_persistent.yaml) 的 `MAIN.Schedule`、`requestmap`、各 role 的 `load:work`。
- 差异与联动：`B=1` 退化为单请求直接查表；`B=2..4` 用比较网络按 KV 长度降序排列并做短线性查段；`B>4` 保持输入顺序并做四步二分。前缀表内容、CTA 初始化同步与请求内 raster 随之联动，跨工作 stride 分配与连续环保持不变。
- 条件与依据：kernel `:1170–1263,1577–1662`（排序实际受 `request_count <= 4` 控制，已保存的 `flatten_varlen_lpt` 属性未参与该分支）。host 默认在 `B > 4` 启用 persistence；`B=1` 另需 `ceil(Q/8) > SM + floor(SM/2)`；`B=2..4` 需至少四波且满足 dominance bound `<= 0.3`、near-pinned 或 tight bound `<= 0.60`（`KV <= 8*Q`）之一，公式见 `interface.py:951–965,1212–1287`。

### K/V 加载分工（局部）

- 基线：[root](root.yaml)、[packed_persistent](packed_persistent.yaml) 的 `LOAD_QK/LOAD_V`、`keyring/valuering`、`MMA`；[split_kv](split_kv.yaml) 的 `ATTENTION.LOAD/MMA`。
- 差异与联动：dual 模式由两个 producer 各自维护四槽就绪；joint 模式关闭第二 loader，按 `K0,K1,V0,K2,V1,...,Vlast` 的半通道序列共用四个 acquire/advance token。K 与 V 仍驻留各自独立的 SMEM 数组，槽映射、覆写等待、QK/PV 消费和尾部排空随共享 token 序列一并切换；`online` 计算、工作归属与输出生产者不变。
- 条件与依据：默认 `B > 1` 或对齐后 `KV >= 98304` 选 dual，其余选 joint（`interface.py:973–982`，kernel `:1327–1390,1511–1523,2116–2261,2620–2807`）。`split_kv` 切到 dual 时可放宽 `B`，partial 与 combine 交接链不变。

### 共享 rescale（局部）

- 基线：[root](root.yaml)、[packed_persistent](packed_persistent.yaml) 的 `scoreprobability.stats`、`SOFTMAX`、`CORRECTION`。
- 差异与联动：`stats` 视图由保存 `old/base` 两列改为保存单列 FP32 `alpha`，供 `SOFTMAX` 分母修正与 `CORRECTION` 分子重标定直接共用，`CORRECTION` 省去本地 `exp2`。`stats` 区域宽度、发布/捕获与别名覆写等待同步调整，warp 投票及跳过乘法时的 `numerator` 所有权交接不变。
- 条件与依据：默认仅在 `B=4`、总 `Q=16384` 的 flattened 路径启用（`interface.py:966–972`，kernel `:3523–3587,3644–3651,3694–3759`）。

### Split 数与归并容量（参数）

- 基线：[split_kv](split_kv.yaml) 的 `ssplit`、`ATTENTION` partition、`partialo/partiallse`、`COMBINE` reduction 和环。
- 差异与联动：`ssplit` 联动改变 KV 切分、`partialo/partiallse` 容量与归并遍历；`COMBINE` 的 `smax` 取不小于 32 的 2 的幂（上限 64）。`split_kv.yaml` 固定 `ssplit=5`、`smax=32`。
- 条件与依据：`interface.py:_hd256_split_heuristic:562–606` 要求机器未填满、总宽至少 16 个 KV 块且每路至少 2 块，并受 24 MB `partialo` 预算与最多 64 路限制。

### 数值、环与传输细项（局部）

- 基线：[root](root.yaml)、[packed_persistent](packed_persistent.yaml)、[split_kv](split_kv.yaml) 的 `stream/online`、KV rings、`SOFTMAX/CORRECTION`、输出 stage。
- 差异与联动：数值路径含多项式 `exp2`、非 `ldred` 行最大值与无条件 correction；环与调度参数含 raster 排列（0/1/2）、KV 槽数与 `offset/tau`；传输与收尾含非合并或非提前释放输出及 `page=64` 的 `cp.async` 汇编。切换数值路径须同步更新近似与舍入说明，切换环深度或传输原语须同步调整完成、发布与覆写等待。
- 条件与依据：均为源码开发开关；三份完整 YAML 固定采用 native `exp2`、`ldred`、`offset=tau=4`、四槽 KV、条件 correction、合并写出与提前释放、`page=64` TMA。

### 已有决定的组合（组合）

- 基线：[root](root.yaml)、[packed_persistent](packed_persistent.yaml)、[split_kv](split_kv.yaml) 及以上对应对象。
- 差异与联动：源码实际启用的“矩形网格 + joint 加载”、“单请求 persistent + joint 加载”、“小批量 persistent + 共享 `alpha`”和“小批量 static + 共享 `alpha`”均由上述网格/映射、K/V 加载与 `stats` 局部变化正交组合而成，未引入新的主体协作组织。
- 条件与依据：遵循上述各条 host 门控与接口约束。
