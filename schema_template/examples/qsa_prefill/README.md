# QSA prefill

[root.yaml](root.yaml) 收录行块 hash union 预处理与基于 ticket 动态调度的持久稀疏 attention；[dispatch.md](dispatch.md) 说明承担范围与两槽分数环局部变化，[root.png](root.png) 展示代表 worker 的局部事件依赖。

固定来源为 `KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019/qsa_prefill/aka_solution`，BF16 FIT v3 `fc30f74`。主要入口是 [operator.py](../../../kernel_wiki/aka-four-problems-20261019/qsa_prefill/aka_solution/qwen38_core_qsa_cutedsl/operator.py)、[预处理](../../../kernel_wiki/aka-four-problems-20261019/qsa_prefill/aka_solution/qwen38_core_qsa_cutedsl/qsa_preprocess.py)、[主 kernel](../../../kernel_wiki/aka-four-problems-20261019/qsa_prefill/aka_solution/qwen38_core_qsa_cutedsl/qsa_kernel.py) 和 [reference.py](../../../kernel_wiki/aka-four-problems-20261019/qsa_prefill/reference.py)。

从仓库根目录执行：

```sh
python3 -B schema_template/check_yaml.py schema_template/examples/qsa_prefill/root.yaml
python3 -B schema_template/check_yaml.py schema_template/examples/qsa_prefill/root.yaml --read-only
python3 -B schema_template/check_yaml.py schema_template/examples/qsa_prefill/root.yaml --graph png \
  --set 'tile_count(t)=4' --set 'has_next(0)=1' --set 'has_next(1)=0' \
  --set 'correction(0)=0' --set 'correction(1)=1' --set 'correction(2)=0' --set 'correction(3)=0' \
  --set 'correction(4)=0' --set 'correction(5)=0' --set 'correction(6)=0' --set 'correction(7)=0' \
  --set last_loaded=7 --worktiles 2
```

主图将每份工作的 union 遍历缩至 4 片，覆盖双槽 KV 复用、单次修正（`correction(n)=1` 表示 `SOFTMAX_A` 任一 warp 触发 32 行 `move_ref` 投票，命中时仅投票 warp 等待前序 PV 并重标定对应 Y 行）、跨工作延迟发布与 Q/输出交接。真实输入域每份工作至少 32 片，该样本仅用于局部协议检查与示意，不代表合法任务输入。空 worker 另用 `--read-only --graph check --set 'tile_count(t)=4' --set last_loaded=-1 --worktiles 0` 检查：即使初始已减 `base` 的票据 `u` 即为终止票，仍完成全员确认及 `min(u, worktile_count-1)` 的初始 Q 读取与发布。

Contract 的误差口径依据[原验收器](../../../kernel_wiki/aka-four-problems-20261019/evidence/qsa_prefill/reproduce/checker/atrex_bench/eval/correctness.py)：转 FP64 比较、以实现值 `x` 为相对误差基准、逐元素满足阈值且拒绝非有限输出（历史 [correctness.json](../../../kernel_wiki/aka-four-problems-20261019/evidence/qsa_prefill/correctness.json) 中 4 个 shape × 4 组采样全部通过）。独立的 [20261021 harness](../../../kernel_wiki/aka-four-problems-20261019/validation-20261021/bundles/qsa_prefill/harness.py) 采用 FP32 参考基准判据并通过对应 4 个 shape，两套历史记录分开保留、不混成同一标准；案例整理未重跑 GPU。

保留的源码限制：未写的 `qblk` padding 行参与修正投票；单槽 `halfmax` 仅在两半写入后同步一次，下一片覆写前无 peer 读完屏障（按提取边界如实记录该限制，不补造 `Released` 等待或双缓冲）；fast `exp2` 与 `rcp_fast` 无来源误差界；缓存 workspace 无跨 stream 隔离，录图重放不更新 host ticket `base`。硬件与实现栈记录已核实的 L20D/SM103、CuTe DSL 4.6.1，`Wiki` 保留 DOC2 占位。
