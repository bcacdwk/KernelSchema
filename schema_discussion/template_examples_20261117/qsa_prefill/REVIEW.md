# QSA 独立审阅

审阅对象为当前 [root.yaml](../../../schema_template/examples/qsa_prefill/root.yaml)（Version 11）、[dispatch.md](../../../schema_template/examples/qsa_prefill/dispatch.md)、[README](../../../schema_template/examples/qsa_prefill/README.md) 和重绘的同名 PNG。独立读取当前 DOC、合同、host、预处理和主 kernel 后，确认仍只需这一份完整设计；没有重新选型或增加隐含 selector。当前内容在独立理解、分层、全文一致和内容必要性方面可交付，来源限制如下保留，不等同于证明 kernel 没有缺陷。

## 来源与已确认关系

以下源码均固定于 `KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019`。根 MANIFEST 的版本是 BF16 FIT v3 `fc30f74`；实际 `qsa_kernel.py` 的 SHA-256 为 `f5232531709cea8de34e90a0aa578e8e3d580feddf31ab20cdab63361de80b8c`，与来源表一致。

- `qsa_prefill/{metadata.json,shape_train.json,shape_valid.json,input.py:280–460,reference.py:10–81}`：固定 4096×24×256 查询、两个 KV 头、2052 字 packed ABI、K/V 页内视图及偏移、两个 CP rank 的 fitted step-15 选择、末尾 valid count、可选 out 和返回同一 out。标准答案先将 attention 舍入 BF16，再门控并舍入；没有额外因果掩码。当前 YAML 单独登记返回结果 o，并明确它也是写回 out 的同一结果。
- `evidence/qsa_prefill/reproduce/checker/atrex_bench/eval/correctness.py:233–340`、`reproduce/verify.py`、`correctness.json`：比较先转 FP64；逐元素 `abs(candidate-reference)<=3.2e-5+.02*abs(candidate)`，允许 mismatch 为零，拒绝任一非有限输出；返回值及 mutated out 均检查，无 L2 放行或附加门槛。历史四 shape×四 draw 记录通过。20261021 harness 的 FP32/reference-relative 记录没有混入 Contract。
- `qsa_prefill/aka_solution/qwen38_core_qsa_cutedsl/operator.py:52–160`：仅当前 stream 的 union 预处理及主 kernel 两次提交；group=12 对应 br=10。workspace/count/ticket 按设备、块数和容量缓存；device ticket 单调推进，host 每次预留 total_units+ncta，传入 base 才做 int32 归一。其他 group 的 br fallback 不消除主 kernel 对两 KV 头的硬编码，不能升级为支持范围。
- `qsa_preprocess.py:98–313`：清 hash/count 并缓存行头和页表；前瞻 quad 读取、有效性过滤、CAS 线性探查和每个插入者 OR；warp-prefix 后按 warp 原子预留段落压缩。warp 预留顺序不固定，源码注释不能证明全局条目顺序固定。union/count 的全局 handoff 与后继 kernel 完成等待匹配。
- `qsa_kernel.py:205–339,372–394,462–647,665–851,854–951`：单 CTA 的 M128/N64、两槽 K/V、三槽 S/P 别名、两列半 softmax、完整 ticket 确认、按接受票据分别读 Count、Q refill、半 Y 释放及最后条带写出均已贯通。MMA 实际顺序为 QK0/QK1/QK2/PV0/…；原 docstring 的示例次序不作为依据。最大值未先掩码；E 在全部列求 exp2 后选择，P 转 BF16，而分母保留未舍入值。修正由 warp vote 决定，只有 A 的投票 warp 重标定其 32 行的全部 D 列；两半更新各自 L/reference。

具体合法形状的静态贯通采用公开 rank0/seed1001 条件：410 个行块×2 KV 头，共 820 份工作；票据 0/1 对应尾块的两个 KV 头，尾块仅六个真实查询行、72 个 packed 行，最终覆盖全部 4096×24 个输出行。union 至少含一行的 512 个完整组，因而 `ceil_div(union_count,16)>=32` 是保守下界。此推导不是运行 generator 或 GPU 得到的测量。

## 实际修正

1. 四处 `packedacked / packedrior / packedaged` 是作者字符串替换误伤，不是规范化行为；还发现同次误伤将 `p_a/p_b` 定义写成 `packed_a/packed_b`，与 Symbol 不一致。临时完整目录 `/tmp/qsa-word-review-izoy0ctl` 中仅还原正确文字，默认检查退出 0、0 changed，前后文本字节完全相同；见 [review-typo-normalization.log](review-typo-normalization.log)。无需公共工具修复。
2. Math 将 K/V gather 展开为 union 条目、页表和 cache 的简短公式，补齐 b/v/i 范围；删除重复的未掩码、reference 复制和更新可下降等说明，公式仍保留这些事实。padding 参与 vote 的限制放入 qblk 存储说明。`p_a/p_b` 与两半发布动作现在直接对应。
3. 按 DOC5 八动词语义，包含 reference/L 数值更新的 `store:p` 改为 `compute:p`，纯发布最终半分母的 `compute:halfsum` 改为 `store:halfsum`；Pipe 与 Waiting 一并更新，保留原动作位置。明确 weights 形成 alpha/tsum，correction 消费 alpha，而 PV 加到已经修正的 y，首 PV 覆盖旧 y。
4. Contract 增补固定 fitted generator 定位和 seed 域，避免将三个 packed 不变量误读为任意 512 组采样均属于评分域；可选 out、alias、全部输出和误差口径保持完整。host base 的说明区分 host 预留和传入 int32 归一。
5. dispatch 修正 accumulator 名称。README 明确 2 worktiles×4 片低于真实 union 下界，不是合法任务样本，只是有限局部事件示意，不能声称完整合同、数值或全控制路径验证。

## 独立验证

所有最终命令使用 `/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_template/examples/qsa_prefill/root.yaml`，不带 `--skip-hardware`：

| 参数 | 实际结果 |
| --- | --- |
| 无附加参数 | 退出 0；1 YAML，0 changed；结构、预算、关系及 dispatch 通过，见 [review-check-default.log](review-check-default.log) |
| `--read-only` | 退出 0；0 would change，见 [review-check-read-only.log](review-check-read-only.log) |
| README 的完整 `--graph png` 命令 | 退出 0；PREPROCESS 2 动作/4 边，ATTENTION 158 动作/586 边，均无环；同名 PNG 已重绘并目视检查，见 [review-graph.log](review-graph.log) |
| `--read-only --graph check --set 'tile_count(t)=4' --set last_loaded=-1 --worktiles 0` | 退出 0；ATTENTION 9 动作/26 边，无环，见 [review-graph-empty.log](review-graph-empty.log) |

图中 `has_next(0)=1,has_next(1)=0`，仅 `correction(1)=1`，其余 0，`last_loaded=7`。末片 done 由 drain issue 放行，drain done 等实际片完成；空 worker 没有片和 drain，但初始终止票仍被全部确认，SOFTMAX_A 仍 gather 安全工作坐标的 Q 并发布 qready。最终仅澄清 Contract/Math 的文字与索引范围，流水线和图参数未再变动；默认/只读已针对最终全文重跑。

## 真实遗留

- HalfMax 只有两半写入后的 rendezvous，没有发现 peer 读完再覆盖的来源保证；未添加虚构 Released 边。未写 Q padding 仍可影响 vote；没有擅自改成清零。两个问题均随 YAML/README 保留。
- fast exp2 与 reciprocal 没有来源误差界；历史正确性覆盖不变成任意有限输入上的保证。
- 缓存没有跨 stream 隔离，捕获重放没有 host base 更新；root 仍采用串行 eager 调用条件。终止票、初始无工作和尾部在途操作不因该条件省略。
- [TOOLS.md](TOOLS.md) 的动态 worktile 依赖局部域 `last(i)` 误判仍是原工具问题；当前使用等价 `tile_count(t)-1`，没有修改公共工具或设计。
- 已核实的 L20D/SM103、CuTe DSL 4.6.1、CUDA 13.0 等继续保留；Wiki 是 DOC2 尖括号占位，不声称资料绑定完成。

本轮未运行 Runner/GPU，未修改固定源码、其他案例或公共工具/DOC，未提交 Git。至此停止写入本案例。
