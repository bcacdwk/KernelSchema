# GDN prefill

当前只有 [root.yaml](root.yaml)：多序列、每 CTA 64 行 value/state、64-token chunk 的成对变换与片上递推。单序列的输入和通知变化、内部 M128 参数记录见 [dispatch.md](dispatch.md)。M128 不表示 128-token chunk，也不是已验证 Model 的自动 fallback。

来源固定为 `KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019/gdn_prefill/aka_solution/solution.py`，SHA-256 `e216c76d8c434233cba9042be79500438a522da13cc42090281cc138ae427c69`。可读拆分与单文件关系见同目录 `build_solution.py`。未混用 44 题批次。

从仓库根目录运行：

```sh
/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_template/examples/gdn_prefill/root.yaml
/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_template/examples/gdn_prefill/root.yaml --read-only
/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_template/examples/gdn_prefill/root.yaml --graph png --set 'chunk_count(0)=6' --set 'chunk_count(1)=2' --worktiles 2
/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_template/examples/gdn_prefill/root.yaml --read-only --graph check --set 'chunk_count(0)=2' --worktiles 1
```

默认检查与只读检查均通过，未跳过硬件；只读预览零变化。同名 [root.png](root.png) 展示一个 worker 的两份连续工作：GDN 为 249 actions / 979 edges / 30 boundary skips，无环；INDEX_CAST 为 1 / 1 / 0。它可对应 B=3、lengths=[257,1,1]、num_sm=148 时 CTA 0 的工作 0 和 148，包含 partial 和全 padding chunk、各环复用与跨工作继续。最小合法样本 chunk_count=2 为 GDN 68 / 242 / 19，无环。输入序列非空且 chunk 成对补齐，因此 0/1 个 padded chunk 不在该设计域内。图只说明所给有限事件样本；不证明所有输入、数值结果或真实 GPU 调度。

Version 9 经独立全文审阅，结果见 [REVIEW.md](../../../schema_discussion/template_examples_20261117/gdn_prefill/REVIEW.md)。单序列按源码核对 raw/ready-only/CG1 的局部差额，并用元数据替身执行实际单文件与可读拆分的 selector：T=1/64/65/128/129/257 结果一致，只有 128 关闭尾块标量 predicate。这是 host 选择条件检查，未把 root 改参数当作单序列完整图。

公开域上界来自 `shape_train.json` 的 domain/invariants，而非样例最大值：B≤16、总 T≤16384。入口要求显式初态和终态 buffer；zero 模式也加载完整 FP32 tensor。输入已做 BF16 K normalize，Q 不归一化，kernel 不生成 gate。计时包含 `Model.forward` 的 int64→int32 转换；单序列即使不读结果也保留这次提交。

容差按 `metadata.json` 与独立 `validation-20261021/results/gdn_prefill/validate.json` 的逐元素四路检查：返回值和被修改 buffer 都检查，O 的 atol/rtol=0.01/0.01，终态为 0.005/0.001。原 `reference.py` 与 Agate 适配参考字节相同，以 FP32、32-token block inverse 和 log-space decay 计算后转换输出；已有独立记录覆盖 30 几何 / 180 种子。`gdn_agate_eval/summary.json` 的统一 runner rtol=0.05 不能替代这些逐输出阈值。本轮没有 GPU 重跑。metadata 的 scale 小数末位为 .845，而实际 input.py 的 `1/math.sqrt(128)` 是相邻 binary64 .843；二者 FP32 相同，selector 的 Python 相等比较则不同。

保留两项来源限制：多序列更新 Q/K/V descriptor 时，对旧在途 descriptor 读取的顺序没有充分依据；descriptor workspace 跨调用缓存，源码没有共享缓存调用间的互斥。YAML 不补造 drain 或独立 workspace。Wiki 链接仍按 DOC2 占位，fast log2/exp2 没有可靠误差界；这些也不由有限图补齐。
