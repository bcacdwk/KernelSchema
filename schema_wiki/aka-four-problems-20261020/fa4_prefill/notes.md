# FA4 prefill schema

本目录包含 11 份独立完整的 plan。文件名为 `fa4_prefill__<path>.yaml`，dispatch ID 为 `fa4_prefill.<path>`；精确选择条件见各文件的 `meta_data.dispatch.when`，修订号见 `meta_data.version`。

| Schema 路径 | Dispatch 概览 |
| --- | --- |
| [splitkv_joint_load](fa4_prefill__splitkv_joint_load.yaml) | KV 分段，K/V 联合加载 |
| [splitkv_dual_load](fa4_prefill__splitkv_dual_load.yaml) | KV 分段，K/V 独立加载 |
| [single_persistent_joint_load](fa4_prefill__single_persistent_joint_load.yaml) | 单请求 persistent，K/V 联合加载 |
| [single_persistent_dual_load](fa4_prefill__single_persistent_dual_load.yaml) | 单请求 persistent，K/V 独立加载 |
| [small_persistent_lpt](fa4_prefill__small_persistent_lpt.yaml) | 2–4 请求，LPT 排列，persistent，普通 rescale |
| [small_static_lpt](fa4_prefill__small_static_lpt.yaml) | 2–4 请求，LPT 排列，static，普通 rescale |
| [small_persistent_shared_rescale](fa4_prefill__small_persistent_shared_rescale.yaml) | 4 请求且 TotalQ=16384，LPT 排列，persistent，共享 rescale |
| [small_static_shared_rescale](fa4_prefill__small_static_shared_rescale.yaml) | 4 请求且 TotalQ=16384，LPT 排列，static，共享 rescale |
| [packed_persistent](fa4_prefill__packed_persistent.yaml) | 5–16 请求，保留请求次序，persistent |
| [rect_joint_load](fa4_prefill__rect_joint_load.yaml) | 矩形 static 分派，K/V 联合加载 |
| [rect_dual_load](fa4_prefill__rect_dual_load.yaml) | 矩形 static 分派，K/V 独立加载 |

LPT 按 KV 长度降序排列请求；persistent 按固定步长接续多份工作，static 每个 worker 只承担一份工作；共享 rescale 传递已计算的 alpha，普通 rescale 传递最大值信息。
