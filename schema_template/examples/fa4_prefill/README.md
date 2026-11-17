# FA4 prefill

三份完整设计见 [dispatch.md](dispatch.md)：[root](root.yaml) 是 compact static，[packed_persistent](packed_persistent.yaml) 保留跨工作环与提前输出释放，[split_kv](split_kv.yaml) 固定五路 KV 和独立归并。共享 Contract 只在 root；persistent 的数学整区显式来自 root。

固定来源为 `KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution`，v37 延续树、v43 包标签。入口为 `solution.py:Model.forward` → `flash_attn/cute/interface.py:_flash_attn_fwd`；核心为 `sm100_hd256_2cta_fmha_forward.py`，归并为 `flash_fwd_combine.py`。公开 Contract 的批量上界16来自 `shape_train.json` 的显式 domain，并非从样例估计。硬件和栈沿用匹配证据的 L20D/SM103、CuTe DSL 4.6.1；Wiki 仍为 DOC2 占位。

从仓库根目录运行默认规范化与只读检查：

```sh
/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_template/examples/fa4_prefill/root.yaml
/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_template/examples/fa4_prefill/root.yaml --read-only
```

同名 PNG 的确切命令：

```sh
/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_template/examples/fa4_prefill/root.yaml --graph png \
  --set 'tile_count(t)=4' --set 'valid(t)=1' --set 'rescale(n)=1' --worktiles 1
/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_template/examples/fa4_prefill/packed_persistent.yaml --graph png \
  --set 'tile_count(0)=1' --set 'tile_count(1)=3' --set 'valid(t)=1' --set 'rescale(n)=1' --worktiles 2
/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_template/examples/fa4_prefill/split_kv.yaml --graph png \
  --set 'tile_count(t)=4' --set 'valid(t)=1' --set 'rescale(n)=1' --set lastvalid=3 --worktiles 1
```

图是给定局部参数的事件关系，不是时间线或评分输入复现。root 对应 B=3、Q=[32,32,32]、KV=[512,384,256] 的首份工作，展示四轮及两槽/四槽复用。persistent 对应 B=9、各请求 Q=KV=512、74个 workers 的 worker0 前两份工作：全局候选0与74映射至 `(b,m)=(0,0),(2,20)`，因果遍历为1、3轮。该缩小输入仍走相同默认调度。rescale=1 是显式选择的分支示意，非测得的输入数值轨迹。

split 图对应 B=1、Q=200、KV=2048：host 仍选五路，span=4个 KV blocks；图的 ATTENTION 取 split0，COMBINE 的最后有效 split=3。第五 split 为空，另外以 `--read-only --graph check --set 'tile_count(t)=0' --set 'valid(t)=1' --set 'rescale(n)=0' --set lastvalid=3 --worktiles 1` 检查，只有 LSE 写入而没有 attention payload。跨提交依赖通过 Flow 的 kernel_done 与全局 handoff 表达，图仅画各 step 的局部事件。

另外已检查 root 的单轮（如 B=3、各 Q=17/KV=64 的 m=1，包含 query 尾和缺第二 page）和无效 compact 候选、persistent 不触发 rescale、joint 单轮，以及 combine 全零权重时仍初始化输出的协议边界。空 split 的 PartialO 保持未写；combine 可以预取这些未写字节，只有正权重进入 FP32 累加，未补造零初始化或最终 copy-group drain。

有效 persistent 工作后接无效 padding 的检查现已通过：B=9、各请求 Q=KV=128 时，cpad=80、有效候选72、worker 数74；worker0 的候选0为单轮工作，下一候选74无效。确切命令：

```sh
/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py schema_template/examples/fa4_prefill/packed_persistent.yaml --read-only --graph check \
  --set 'tile_count(0)=1' --set 'tile_count(1)=0' --set 'valid(0)=1' --set 'valid(1)=0' --set 'rescale(n)=0' --worktiles 2
```

公共工具冻结后，三张同名主图均已重绘：root 194 actions/700 edges，persistent 217/782，split ATTENTION 194/705 与 COMBINE 20/58；padding 检查为80/259，均无环。默认检查和只读预览均为3 YAML、零变化，没有跳过硬件。首次自由槽获取也明确保留；复用依赖采用等价正向索引，joint 保留连续 token 流并区分初次 S/Y 获取与后续复用。

历史窗口问题与本次有限表达调整见 [TOOLS.md](../../../schema_discussion/template_examples_20261117/fa4_prefill/TOOLS.md)，完整独立结论、源据和检查结果见 [REVIEW.md](../../../schema_discussion/template_examples_20261117/fa4_prefill/REVIEW.md)。本轮未运行 GPU，fast exp2/log/division 的独立误差界仍无材料。
