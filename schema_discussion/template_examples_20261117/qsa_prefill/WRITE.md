# QSA 写作交接

现行完整集合只有 [root.yaml](../../../schema_template/examples/qsa_prefill/root.yaml)，同时更新 dispatch、README 和同名 PNG。选择结论见 [SELECTION.md](SELECTION.md)：固定 v3 的两次提交是一套完整设计；构造器两槽分数环仅作局部记录，无隐式 host selector。

保留行块 hash union、lo/hi 行成员位、warp-prefix 压缩、两列半 softmax、未掩码最大值与稀疏重定标、BF16 P 和两次输出舍入。流水线包含 ticket 发布/全部确认、明确的 Count 读取、跨工作双槽 KV 延迟发布、S/P 三阶段别名、Q refill、分半 Y 释放及终止 drain；初始即终止的 worker 仍确认并完成无效 Q gather/qready。

来源均固定为 `KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019`。实际核对：

- `qsa_prefill/aka_solution/qwen38_core_qsa_cutedsl/operator.py:52–160`：ABI guard、group/br、缓存分配、两次 current-stream 提交、ticket/tbase 生命周期。host fallback 没有消除主 kernel 的双 KV 头假设。
- `qsa_preprocess.py:98–313`：清表/头信息、前瞻 quad load、有效性检查、CAS/OR 与压缩；warp 预留顺序不确定，未沿用源码注释中“全局 slot 顺序确定”的过强说法。
- `qsa_kernel.py:205–339,372–394,462–647,665–738,758–856,872–897`：存储、票据、初始 Q、softmax/修正、epilogue、MMA 和 gather。SHA-256 为 `f5232531709cea8de34e90a0aa578e8e3d580feddf31ab20cdab63361de80b8c`，与 SOURCES 一致。
- `metadata.json`、`shape_train.json`、`shape_valid.json`、`input.py:385–460` 与 `reference.py`：4096 行、两个 GQA 组、2052 字 packed ABI、映射页有效内容、可选 out 与返回 alias。没有加入额外因果 mask。
- `evidence/qsa_prefill/reproduce/checker/atrex_bench/eval/correctness.py:233–340`：FP64、candidate-relative、零 mismatch、非有限拒绝；`reproduce/verify.py` 和 `correctness.json` 绑定四 shape×四 draw。20261021 harness 的 FP32/reference-relative 判据另行记录，没有混合成新标准。
- `evidence/qsa_prefill/environment.json` 与旧 Template 已有执行/机制摘要：保留 L20D/SM103、CuTe DSL 4.6.1、CUDA13.0 等；Wiki 使用 DOC2 占位，没有新增硬件调查。

最终实际检查（仓库根目录，Python 均为 `/opt/venv/python3.13.12/bin/python3`，全部退出 0，**不使用 `--skip-hardware`**）：

| 命令后缀（共同前缀 `-B schema_template/check_yaml.py schema_template/examples/qsa_prefill/root.yaml`） | 结果 |
| --- | --- |
| 无额外参数 | 规范化、结构、预算、关系及目录通过，见 `check-default.log` |
| `--read-only` | 0 个待改文件，见 `check-read-only.log` |
| README 中完整 `--graph png` 命令 | PREPROCESS 2 动作/4 边；ATTENTION 158 动作/586 边，均无环；root.png 已生成，见 `graph.log` |
| `--read-only --graph check --set 'tile_count(t)=4' --set last_loaded=-1 --worktiles 0` | ATTENTION 9 动作/26 边，无环，见 `graph-empty.log` |

PNG 样本是两个工作、每工作四片，只有连续片 1 触发 correction，has_next(0)=1、has_next(1)=0、last_loaded=7。它检验局部协议，实际题目每工作至少32片；不是新增 GPU 测量。早期按 COMMON 创建 starter 硬件占位曾导致默认检查失败；按主代理续修授权恢复已有已核实硬件记录后，最终普通检查通过。

工具问题见 [TOOLS.md](TOOLS.md) 与可运行最小复现：有界 i 域依赖当前动态 t 时，`last(i)` 被错误禁止；root 等价使用 `tile_count(t)-1`，未改设计或公共工具。

请 fresh reviewer 独立核对当前全文，重点检查：

1. `halfmax` 的 peer 读完后覆盖缺口与未写 Q padding 对 vote 的影响；当前正文保留事实，未补造同步或初始化。
2. `last_loaded` 作为终止后最终已发片号，对 delayed publication/drain 的两条时点约束是否准确；空 worker 不产生 drain。
3. 以每个 softmax warpgroup 为 role 时，correction 的“任一 warp 投票”条件及仅投票 warp 的累加器修正边界，是否足够忠实。
4. ticket acknowledgement 与独立 Count 读取、下份首 PV 的双半 Y release、终止 nextq 通知是否完全闭合。

本轮未运行 Runner/GPU，未修改源码、其他案例、公共工具或文档，未提交 Git。
