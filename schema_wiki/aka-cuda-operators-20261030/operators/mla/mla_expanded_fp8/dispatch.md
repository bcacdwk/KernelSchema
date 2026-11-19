# mla_expanded_fp8

文件版本、来源与检查信息见[当前元数据](.extraction/current.json)。

## 范围

本目录仅含一份完整设计 [root](root.yaml)，保存唯一共享 Contract，三个 Design 区域全部本地定义，无其他 YAML 或整区 `from` 引用。固定来源为 `KernelSchema@24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2:kernel_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/candidate/solution.py`。

**入口与运行期门控。** `Model.forward` 恒以模块常量 `_PV_MODE = 0` 启动单个融合模板 `mla_fp8_kernel<0, KVST>`，无跨核归并。工作单元为 `item = (请求 b, 128 行 q_tile, head h)`，共 `item_count = tiles_total * heads`，KV 按 128 键分块（causal 右下对齐）。Host 在每次启动前分别计算三道可相交的门控（`solution.py:1690–1717,1855–1866,2008–2024`）：
- **scatter 门**：`min(kv_heads * batch_count, sm_count) * max(kv_len) * 320 * 4 > l2_bytes * 3` 命中时，同时选择 `KVST = 3`（`<0, 3>`）与跳过 LPT 的 head-major 表序；否则选 `KVST = 4`（`<0, 4>`）。
- **degenerate 门**：任一请求 `q_len(b) < 128` 时关闭 LPT，保留 head-major 表序。
- **网格门**：`item_count * 2 <= sm_count * 3` 时选静态网格（`grid = item_count`，一 CTA 一 item，无队列补领），否则选持久领取网格（`grid = sm_count`）。两种网格共用同一模板族，`KVST` 仍由 scatter 门独立决定。

**内部算法分支与未交付方案。** 单 KV tile 且 `q_len(b) <= 32` 的 tiny 单元走未量化权重 FP32 累加路径，可选 `attention_sinks` 重归一与 log2 空间 LSE 均属 `MODE0` 内部机制；源码概率实际按 $2^8 = 256$ 缩放（`kLog2_448 = 8.0f`，448 仅为 E4M3 上限余量）。**MODE1**（`_PV_MODE = 1`：独立 `v16_prep_kernel` 将 V 转为跨核 workspace FP16、FP16 概率、锁死 `KVST = 3`）是源码中存在但从冻结入口不可达的独立研究方案，未交付完整 schema。非冻结 4D/自建输出分支、域内恒关闭的左窗（`window_left = -1`）及死代码不构成设计。

**必需调用限制。**
- **侧流元数据就绪**：每次 `forward` 在非阻塞侧流异步 DtoH 拷贝 `seq_lens`、`cum_seq_lens_q`、`cum_seq_lens_kv` 并同步侧流以构建上传 `ItemGeo` 表；侧流对调用者主流无输入等待事件，三个元数据张量在侧流拷贝发出前必须已就绪（`root.yaml:644–645`）。
- **跨调用隔离**：静态 pinned 缓冲、设备 `ItemGeo` 表、队列字、事件及首次查询缓存的 `sm_count/l2_bytes` 跨调用共享；复用前前次调用必须已完成（仅在同一主流排队不能阻止侧流覆盖在读设备表），且无并发 host entry 或设备切换（`root.yaml:647–648`）。
- **不可整图捕获**：每次调用均执行 host 几何解析与侧流同步，不能对整次调用做 CUDA Graph 捕获。既有样本不证明完整动态域误差上界。

## 完整设计

| 设计 | 具体机制与保留理由 | 关系 |
| --- | --- | --- |
| [root](root.yaml) | `MODE0`（`mla_fp8_kernel<0, 4>`）持久领取方案，`sm_count` 个 384 线程（12 warp）CTA。Host 构建按 `kv_tile_count` 降序稳定 LPT 排序的 `ItemGeo` 表，CTA 先执行预分配首项再经 `Refill.atomic_queue` 原子补领（末次领取自复位队列字为 0），`LOADER` 跨 item 预取下一 `q_tile`；QK 分 `[0,128)` SW128 与 `[128,192)` SW64 两段共 6 个 `k32` 步覆盖 192 通道，3 槽 TMEM `score_prob` 环与 128 列 `o_slot` 占满 512 列，概率按 256 缩放量化为 E4M3 写回同槽供 4 步 FP8 PV 消费，`CORRECTION` 重标定 `o_run`；epilogue（含 tiny 未量化 FP32 分支）归一化、折入可选 sink 后原位写 BF16 `out` 与可选 log2 `lse`。 | 阅读入口，保存唯一 Contract，三区本地定义；`when` 要求 `item_count * 2 > sm_count * 3`、非 scatter、各 `q_len(b) >= 128`、正 `sm_count/l2_bytes` 及上述调用隔离条件。 |

## 变体与组合

下列记录定位 [root](root.yaml) 中的已有对象，不修改其参数或 `when`。

### K/V 共享内存环深 3（局部）

- 基线：[root](root.yaml) 的 `Tile_pipeline::MAIN`（`k_ring/v_ring`、`ISSUER.release:kv_slot`、`LOADER.load:kv_tile` 与 `await:kv_slot`）及 `REALIZATION` 的动态 smem 请求与周期匹配释放 `Forbid` 条目。
- 差异与联动：实例切为 `mla_fp8_kernel<0, 3>`。`k_ring/v_ring` 深度由 4 减至 3（槽索引 `n % 3`，动态 smem 请求由 209920 B 降至 168960 B，`Meta<3>` 移除专用 `kv_free` mbarrier），`LOADER` 复用等待守卫由 `if(n >= 4)` 改为 `if(n >= 3)`；因 K/V 环与 3 槽 TMEM `score_prob` 环同周期，取消 `ISSUER.release:kv_slot` 专用释放动作，`LOADER.await:kv_slot(n)` 改为直接等待同周期的 `ISSUER.matmul:p(n-3)@v_tile(n-3)` 完成通知（`root` 为 4 槽配周期 3 的 PV，必须保留专用 `kv_free` 防止相位累积死锁）。
- 条件与依据：scatter 门 `min(kv_heads * batch_count, sm_count) * max(kv_len) * 320 * 4 > l2_bytes * 3` 命中时选用（`solution.py:1711–1717,2019–2021`）。

### item 表序改回 head-major（参数）

- 基线：[root](root.yaml) 的 `Work_schedule::MAIN.Schedule.Strategy`（按 `kv_tile_count` 降序稳定 LPT 排序）。
- 差异与联动：Host 跳过稳定 LPT 排序，保留建表时的 `head-major → 请求 → 重 q_tile 优先` 顺序，以维持按 head 的 K/V L2 复用或避免子 tile 堆积在尾波；item 划分、领取协议、输出地址，以及 tiny 路径在 `signal:prob_ready` 与尾部 PV 发出前完成重读的跨槽存续顺序均不变。
- 条件与依据：degenerate 门（任一 `q_len(b) < 128`）或 scatter 门任一命中即启用（`solution.py:1855–1866`）。

### 静态网格：一 CTA 一 item（局部）

- 基线：[root](root.yaml) 的 `Work_schedule::MAIN.Schedule.Worker_assignment`（`Refill.atomic_queue`）、`Tile_pipeline::MAIN`（`Role.QUEUE`、`geo_slot`、`LOADER` 跨 item 动作及各角色 `await:queue_end`）与 `REALIZATION` 的队列初态条目。
- 差异与联动：`grid = item_count`，`queued = false`。取消 `Refill.atomic_queue` 与哨兵退出（完全不触碰设备队列字，既不依赖也不执行队列自复位，但共享设备表与 host 缓冲的跨调用隔离仍须保持）；`QUEUE` 仅执行 `compute:barrier_init` 后退出，`SOFTMAX` 仅发布首项 `geo_slot[0]`，`LOADER` 取消 `await:next_item`、`await:q_consumed` 与跨 item 的 `load:q_tile(t+1)` 预取，各角色完成单个预分配 item 后经退出同步释放 TMEM。
- 条件与依据：`item_count * 2 <= sm_count * 3` 时选用（`solution.py:2008–2009`）。

### scatter、degenerate 与静态网格的门控组合（组合）

- 基线：[root](root.yaml) 及上述三项变化。
- 差异与联动：scatter、degenerate 与静态网格三道门在 host 侧分别独立求值、可相交生效（`solution.py:1690–1695,1855–1863,2008–2024`）——scatter 成立时必同时启用 `KVST = 3` 与 head-major 表序（无论 degenerate 是否同时成立）；仅当 degenerate 成立且 scatter 不成立时，才只将表序改为 head-major 而保持 `KVST = 4`；上述任一配置若同时满足 `item_count * 2 <= sm_count * 3`，均叠加静态网格（`<0, 3>` 或 `<0, 4>` 按 `grid = item_count` 启动）。
- 条件与依据：三个 launch 谓词的正交组合，不引入新的片上角色或跨核阶段。
