# Sparse 历史候选阅读说明

本包保留同一任务的原始源码与角色材料，并附五份历史候选供当前阶段阅读、核对和转写。任务的数学要求、shape、验收阈值和实现保证不因这些参考材料改变；任务事实以本包 `kernel_wiki/` 内的原角色材料和实际实现为准。

`artifacts/` 中的 root、split_kv、tcgen05、tcgen05_packed 与 dispatch 原字节保留。旧稿的 Source、Version、选择、from 和旧判断均是历史内容，没有当前批准效力；旧四项选择只供 P2 独立选择时参考，不是强制计划。当前 run 从 starter 和 P1 正常建立各阶段、节点及 P6 批准；只转写当前阶段允许的字段。源实现和比较语义必须回查实际实现与原角色材料，不能用旧 YAML、历史回执或本说明替代源码证据。当前 Source/reference 与版本按新 run 的真实输入身份和写作规则建立。

[REGISTER_CLAIM_TRIAGE.md](REGISTER_CLAIM_TRIAGE.md) 记录已知证据边界：208 是 qreg、sacc、oacc 三个主数组的每线程 32 位槽小计，不能称编译后实际总寄存器数，也不能保证 without spilling。P3 应独立核实、收窄 root 的 Tile_pipeline.Thesis、REALIZATION.Require 第三项 Choice 及相关 Reason；其余内容仍须正常全文核查。报告中针对旧 run 的维护建议、内部 checkpoint/attempt/snapshot 路径和旧审批语境仅用于历史定位，不是本包可读 Source 或当前授权。

本包导航（以下均相对本说明）：

- 历史原稿：[root](artifacts/root.yaml)、[split_kv](artifacts/split_kv.yaml)、[tcgen05](artifacts/tcgen05.yaml)、[tcgen05_packed](artifacts/tcgen05_packed.yaml)、[dispatch](artifacts/dispatch.md)。
- 来源身份、逐文件 SHA 与包内路径：[origin.json](origin.json)。
- 寄存器核查的实际实现：[candidate/dsv4_kernel.cu](../kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/candidate/dsv4_kernel.cu)；入口/host：[candidate/solution.py](../kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/candidate/solution.py)。
- 原任务角色：[problem/metadata.json](../kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/problem/metadata.json)、[problem/input.py](../kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/problem/input.py)、[problem/reference.py](../kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/problem/reference.py)、[shape_range.json](../kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/problem/shape_range.json)、[shape_cases.json](../kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/problem/shape_cases.json)。
- 原验收：[performance/CERTIFIED_BEST.json](../kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/performance/CERTIFIED_BEST.json)；寄存器测量核查材料在同目录 `ncu/0.csv` 至 `ncu/9.csv`，不得挪用其他实例的数字。
