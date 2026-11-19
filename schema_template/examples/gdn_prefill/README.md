# GDN prefill

本目录包含一份完整设计 [root.yaml](root.yaml)：面向多序列，按每 CTA 64 行 value/state 与 64-token chunk 执行成对块内求逆和片上状态递推。单序列的输入与通知局部变化、内部 M128 行块参数见 [dispatch.md](dispatch.md)（M128 指每 CTA 128 行 value/state 而非 128-token chunk，公开 `Model.forward` 固定选用 M64）。

固定来源为 `KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019/gdn_prefill/aka_solution/solution.py`（SHA-256 `e216c76d8c434233cba9042be79500438a522da13cc42090281cc138ae427c69`），可读模块拆分与单文件产物的对应关系见同目录 `build_solution.py`；未混用 44 题主批次。

从仓库根目录运行：

```sh
python3 -B schema_template/check_yaml.py schema_template/examples/gdn_prefill/root.yaml
python3 -B schema_template/check_yaml.py schema_template/examples/gdn_prefill/root.yaml --read-only
python3 -B schema_template/check_yaml.py schema_template/examples/gdn_prefill/root.yaml --graph png --set 'chunk_count(0)=6' --set 'chunk_count(1)=2' --worktiles 2
python3 -B schema_template/check_yaml.py schema_template/examples/gdn_prefill/root.yaml --read-only --graph check --set 'chunk_count(0)=2' --worktiles 1
```

[root.png](root.png) 展示单个代表 worker 连续处理两份工作的局部事件关系：对应 `B=3, lengths=[257,1,1], num_sm=148` 时 CTA 0 的本地 `t=0,1`（按步长 148 分别映射到全局工作 `0,148`，即序列 `b=0,2`，`chunk_count(0)=6`、`chunk_count(1)=2`，连续索引 `n` 在本 worker 内展平 `(t,c)`），涵盖半满尾块、全 padding 合成块、各环槽复用与跨工作接续。每条序列非空且按两块成对补齐，最小合法单工作样本为 `chunk_count(0)=2`，0 或 1 个 padded chunk 不在有效域内。

公开域上界 `B <= 16`、总 `T <= 16384` 取自 `shape_train.json` 的显式 domain/invariants。调用入口要求显式传入 FP32 初态与终态 buffer（zero 模式同样加载完整初态张量）；输入 `k` 已在外部完成 BF16 L2 归一化，`q` 不归一化，kernel 不在内部生成 gate；计时包含 `Model.forward` 内的 `cu_seqlens` `int64 -> int32` 转换提交（单序列即使核心不读 `cu32` 也照常提交）。参考判据同时检查返回值与原位改写 buffer：输出 `o` 为 `atol=0.01, rtol=0.01`，终态 `s_final` 为 `atol=0.005, rtol=0.001`。`metadata.json` 所列 `scale` 末三位为 `.845`，而 `input.py` 的 `1/math.sqrt(128)` 为相邻 binary64 值 `.843`：二者转 FP32 相同，但在 host `scale == key_dim**-0.5` 的 Python 浮点相等比较下不同。

源码限制：多序列逐工作更新 Q/K/V TMA descriptor 相对旧在途读取缺少顺序保证；descriptor workspace 按设备跨调用缓存，无并发或跨流互斥；fast `log2/exp2` 无独立误差界。硬件与实现栈记录 L20D/SM103、CuTe DSL 4.6.1，`Wiki` 字段保留 DOC2 占位。
