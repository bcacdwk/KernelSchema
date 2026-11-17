# QSA prefill

[root.yaml](root.yaml) 提取一套行块 union 与 ticket 持久 attention；[dispatch.md](dispatch.md) 记录范围及两槽分数环的局部变化。[root.png](root.png) 展示代表 worker 的事件关系。

固定来源为 `KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019/qsa_prefill/aka_solution`，BF16 FIT v3 `fc30f74`。主要入口是 [operator.py](../../../kernel_wiki/aka-four-problems-20261019/qsa_prefill/aka_solution/qwen38_core_qsa_cutedsl/operator.py)、[预处理](../../../kernel_wiki/aka-four-problems-20261019/qsa_prefill/aka_solution/qwen38_core_qsa_cutedsl/qsa_preprocess.py)、[主 kernel](../../../kernel_wiki/aka-four-problems-20261019/qsa_prefill/aka_solution/qwen38_core_qsa_cutedsl/qsa_kernel.py) 和 [reference.py](../../../kernel_wiki/aka-four-problems-20261019/qsa_prefill/reference.py)。

从仓库根目录执行：

```sh
/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_template/examples/qsa_prefill/root.yaml
/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_template/examples/qsa_prefill/root.yaml --read-only
/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_template/examples/qsa_prefill/root.yaml --graph png \
  --set 'tile_count(t)=4' --set 'has_next(0)=1' --set 'has_next(1)=0' \
  --set 'correction(0)=0' --set 'correction(1)=1' --set 'correction(2)=0' --set 'correction(3)=0' \
  --set 'correction(4)=0' --set 'correction(5)=0' --set 'correction(6)=0' --set 'correction(7)=0' \
  --set last_loaded=7 --worktiles 2
```

图将每份工作的 union 遍历缩到四片，展示双槽复用、一次修正、跨工作延迟发布和 Q/输出交接。真实题目每份工作至少 32 片，故这组绘图参数不是合法任务样本；它只检查展开的局部事件关系，不验证完整输入合同、数值结果或所有控制路径。空 worker 另以 `--read-only --graph check --set 'tile_count(t)=4' --set last_loaded=-1 --worktiles 0` 检查；初始终止票仍经过确认和 Q 发布。

Contract 的误差口径依据 [原验收器](../../../kernel_wiki/aka-four-problems-20261019/evidence/qsa_prefill/reproduce/checker/atrex_bench/eval/correctness.py)：FP64 比较、candidate 作为相对误差基准、每个元素满足判据、拒绝非有限输出。历史 [correctness.json](../../../kernel_wiki/aka-four-problems-20261019/evidence/qsa_prefill/correctness.json) 的四个 shape 各四个输入 draw 均通过。独立的 [20261021 harness](../../../kernel_wiki/aka-four-problems-20261019/validation-20261021/bundles/qsa_prefill/harness.py) 用 FP32/reference-relative 判据，其四个 shape 记录也通过；两套结论不混作一个验收标准。本轮未运行 GPU。

保留的事实边界：Q padding 未写且参与修正投票；HalfMax 交换没有已确立的 peer 读完后覆盖保证；fast exp2/reciprocal 没有来源误差界；缓存没有跨 stream 隔离或捕获重放时 host base 更新。硬件与实现栈沿用已核实的 L20D/SM103、CuTe DSL 4.6.1 环境；Wiki 入口仍为 DOC2 占位，不表示资料已完成绑定。
