# gdn_prefill schema

本目录包含两份独立 plan，均为 `version: 1`。目标为 NVIDIA L20D / SM103，HQ=4、HV=32、DK=DV=128。

`B=cu_seqlens.numel()-1`；各序列长度为正，总 token 数 `T<=16384`。完整适用条件见各文件的 `meta_data.dispatch.when`。

| Schema 文件 | Dispatch ID | 路由条件 |
| --- | --- | --- |
| [gdn_prefill__m64_single_sequence.yaml](gdn_prefill__m64_single_sequence.yaml) | `gdn_prefill.m64_single_sequence` | `B=1` |
| [gdn_prefill__m64_packed_sequences.yaml](gdn_prefill__m64_packed_sequences.yaml) | `gdn_prefill.m64_packed_sequences` | `2<=B<=16` |

长度对齐情况及 zero/continuation 初态模式均不另拆 dispatch。
