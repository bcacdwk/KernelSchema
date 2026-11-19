# Sparse root 寄存器声明核查

结论：需要一次真实、很小的 P3 限域维护。**208 有可复查的主数组容量推导，不能判作凭空数字；但该推导不能支持当前“总寄存器数约 208，且无 spilling”的表述。** 这是内容证据范围问题，与 P6 approve 工具 gate 无关。本次只读核查，唯一新增文件为本记录；未改 source、冻结材料、run/state、schema 产物或回执，未调用模型、维护、resume、检查器或测试。

- Run：`8f0844f9-f279-4781-a19d-0216b7fc4ef5`；核查 checkpoint：`66ad31f314d7ca7e894527d9641bbf3daad59f4ba35278e71e7841dd5e3901da`。
- 当前文件：`.schema_extract/v2/runs/<run>/checkpoints/<checkpoint>/artifacts/root.yaml`；`META_DATA.Version: 4`。
- 两处声明：第 218 行 `DESIGN.Tile_pipeline.Thesis` 的 `Costs ~208 registers/thread`；第 341 行 `REALIZATION.Require` 第三项 `Choice` 的 `~208 registers per thread without spilling`。同项第 343 行 `Reason` 将 register 设计、驻留预算与 launch check 混写，需要同步收窄。

## 来源与推导

固定快照为 `.schema_extract/v2/snapshots/e5db05dc281797bc4a7350cc1d7eb35b2d1331efb10f304bd0ab3b1955ff6388`。以下源码均为其中 `source/kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/candidate/dsv4_kernel.cu`，对应固定 commit `24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2`，SHA-256 `c9389387d35bd26954754a5ae13e2d202df06fc9799a77fddfb0654306909eb2`。

第 6058 行选择 `run_cfg<64,8,16,1,64,2>`；第 5308–5310 行的默认参数及 root 的非 split 实例给出 `HB=64, WARPS=8, QKS=16, KSPLIT=1, KC=64, STAGES=2, SPLIT=false, QOV=false, MODE=0, EI=0, DQ=1`。第 850–859 行可算出 `MI1=1, NI1=4, MI2=4, NI2=8`。

| 主数组 | 固定源码 | 每线程 32 位槽数 |
| --- | --- | --- |
| `qreg[QKS][4]` | 763–765 行明确说明 `QKS*4` 成本；1162–1174 行声明、载入 | `16*4 = 64` |
| `sacc[KSPLIT][MI1][NI1][4]` | 1210–1218 行 | `1*1*4*4 = 16` |
| `oacc[MI2][NI2][4]` | 1176–1182 行 | `4*8*4 = 128` |

三数组合计 **208 个 32 位槽/线程**。当前 Storage 第 235–240 行的 `q_fragments`、`scores`、`o_acc` 容量也满足 `(65536+16384+131072)/(256*4)=208`。`qreg` 和 `oacc` 跨 chunk 保留，`sacc` 在每轮 phase 1 产生并用于 softmax；这是可复查的设计存储账目，不仅是搜不到字面 208 后的判断。

原 Qwen P3 worker 回执 `attempts/95cb56ee-7d01-4fb9-9a45-e1c72dfc729a/receipt.json:136` 已写 `64+16+128=~208/线程`；P3 reviewer `attempts/99b28dbc-3db7-44dd-9864-504e6b91f2a0/receipt.json:184` 进一步对应 `QKS*4、MI1*NI1*4、MI2*NI2*4`。故该数字来源可以追回，问题在于把这份小计提升成了更强的实现声明。

## 证据边界

1. 上述小计未包含第 1223 行 `b[4]`、1264 行 `a[MI1][4]` 等临时 fragment，以及地址、控制、归约状态；不能从三个数组的元素数直接得到编译后总寄存器数，也不能据此判断 spilling。第 1244–1246 行还记载过另一展开形态使 128 个 fragment 寄存器落入 local memory 的历史事例；它支持展开方式重要，不证明当前 cfg2 无 spilling。
2. 冻结 `performance/ncu/0.csv` 至 `9.csv` 确实含 `launch__registers_per_thread` 和 `launch__registers_per_thread_allocated`，但核对全部 Kernel Name 后没有 root 的 `attn_kernel<64,8,16,1,64,2,false,false,0,0,1>`。不能移用其他实例的数字。源码第 6767–6770 行的 `200–204 registers per thread` 明指 heavy **tcgen05**，亦非 root 证据。固定 source 清单未包含 cfg2 编译资源或 spill 报告。
3. 第 817 行的 `__launch_bounds__(WARPS*32,1)`、共享存储函数及第 5311–5318 行的动态 smem 检查/设置能支持保留 launch 规模与共享存储申请；`TORCH_CHECK` 比较的是 `bytes <= max_optin_smem()`，不是寄存器数或 spilling 检查。无需改动现有 207,232 B 存储账目来修复本问题。

## 最小修订范围

由原 Qwen worker 按既有 P3 维护流程处理，独立 review/verify；本核查不代写 schema。工具层字段上限建议仅：

```json
[{"path":"root.yaml","fields":["DESIGN.Tile_pipeline.Thesis","REALIZATION.Require"]}]
```

其中 `REALIZATION.Require` 为列表，语义授权**仅第三项的 `Choice` 和 `Reason`**：

- `Thesis`：删除 `~208 registers/thread, `，保留此前已写明的 16 步 register-cached A fragments、64×512 FP32 accumulator，以及 207,232 B smem 和驻留成本。主要数组的定量设计成本继续由原 Storage 完整承载，不丢失设计信息。
- 第三项 `Choice`：仅删除末尾 `; ~208 registers per thread without spilling`，保留 256 线程、launch bound、动态 smem 申请与容量依据。
- 同项 `Reason`：收窄为共享存储布局必须由申请覆盖、`run_cfg` 在 launch 前检查 device opt-in 上限，避免原句继续把 register 小计当作必要预算或检查保证。可用：`The shared-memory request must cover the Tile_pipeline storage layout; run_cfg checks it against the device opt-in limit before launch.`
- 不改 Storage、Math、Schedule、Pipe、代表/Source、when、Contract、dispatch 或其他条目，不新增字段或硬件调查；版本仅按工具既有内容修订规则递增。

若维护 reviewer 选择保留 208，只能明确写成三个已登记数组的 **32 位槽小计**，不能继续称实际总寄存器数或无 spilling；上面的删除方案更小，且避免重复 Storage。

冻结 `03_root.md` 第 7 段要求必要实现条件和容量判断有匹配版本依据，不将实现记录升级为设计保证。冻结 `06_final.md` 明确全文审阅包含未改内容、未知内容不得用作保证，并规定 root 定义错误回 P3；因此 P6 的 root 只读权限不能成为保留该保证直到“后续任意修订”的理由。`aa4176a8-3767-4ee4-b045-cb95b15c06fa/response.json` 的 approve summary 已指出该精度项，但旧结论和维护批准均不能代替修订后的当前 P6。修复完成后按既有流程重新满足 P6 准入即可，不借此绕过或解释另一项工具 gate。
