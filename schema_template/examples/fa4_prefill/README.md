# FA4 prefill

本目录收录三份完整设计，范围与变体见 [dispatch.md](dispatch.md)：[root.yaml](root.yaml) 采用 compact static 每 CTA pair 单工作调度，[packed_persistent.yaml](packed_persistent.yaml) 保留跨工作连续环与提前输出释放，[split_kv.yaml](split_kv.yaml) 固定五路 KV 切分与独立 combine。唯一共享 Contract 在 `root`；`packed_persistent` 的 `Math_reshape` 通过 `{from: root}` 整区复用。

固定来源为 `KernelSchema@3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution`（v37 延续树，v43 包标签），入口链为 `solution.py:Model.forward` → `flash_attn/cute/interface.py:_flash_attn_fwd`，主核与归并分别为 `sm100_hd256_2cta_fmha_forward.py` 和 `flash_fwd_combine.py`。公开 Contract 的批量上界 `16` 取自 `shape_train.json` 的显式 domain；硬件与实现栈记录 L20D/SM103 与 CuTe DSL 4.6.1，`Wiki` 保留 DOC2 占位。

从仓库根目录运行默认规范化与只读检查：

```sh
python3 -B schema_template/check_yaml.py schema_template/examples/fa4_prefill/root.yaml
python3 -B schema_template/check_yaml.py schema_template/examples/fa4_prefill/root.yaml --read-only
```

三张同名 PNG 的生成命令：

```sh
python3 -B schema_template/check_yaml.py schema_template/examples/fa4_prefill/root.yaml --graph png \
  --set 'tile_count(t)=4' --set 'valid(t)=1' --set 'rescale(n)=1' --worktiles 1
python3 -B schema_template/check_yaml.py schema_template/examples/fa4_prefill/packed_persistent.yaml --graph png \
  --set 'tile_count(0)=1' --set 'tile_count(1)=3' --set 'valid(t)=1' --set 'rescale(n)=1' --worktiles 2
python3 -B schema_template/check_yaml.py schema_template/examples/fa4_prefill/split_kv.yaml --graph png \
  --set 'tile_count(t)=4' --set 'valid(t)=1' --set 'rescale(n)=1' --set lastvalid=3 --worktiles 1
```

图像展示给定局部参数下的代表 worker 事件依赖（非时间线，其中 `rescale(n)=1` 为显式分支示意）：

- `root`：对应 `B=3, Q=[32,32,32], KV=[512,384,256]` 首份工作，展示 4 轮局部迭代及双槽/四槽复用。
- `packed_persistent`：对应 `B=9`、各请求 `Q=KV=512` 时 74 个 CTA-pair worker 中 worker 0 的前两份工作（全局 pair 候选 `0` 与 `74` 映射至 `(b,m)=(0,0)` 与 `(2,20)`，因果遍历分别为 1 轮与 3 轮；逻辑 CTA 跨度在源码中补齐为偶数 `2*cpad`）。
- `split_kv`：对应 `B=1, Q=200, KV=2048` 的五路切分（每路跨度 4 个 KV 块），`ATTENTION` 取 split 0，`COMBINE` 取 `lastvalid=3`。空的第 5 路（仅写 `-inf` LSE，无 attention 计算）另可用 `--read-only --graph check --set 'tile_count(t)=0' --set 'valid(t)=1' --set 'rescale(n)=0' --set lastvalid=3 --worktiles 1` 检查；跨提交依赖由 `Flow` 的 `kernel_done` 与全局 handoff 表达。

有效 persistent 工作后接无效 padding 候选（如 `B=9`、各请求 `Q=KV=128` 时 `cpad=80`、有效候选 72、worker 数 74，worker 0 的候选 0 有效单轮、候选 74 无效）检查命令如下：

```sh
python3 -B schema_template/check_yaml.py schema_template/examples/fa4_prefill/packed_persistent.yaml --read-only --graph check \
  --set 'tile_count(0)=1' --set 'tile_count(1)=0' --set 'valid(0)=1' --set 'valid(1)=0' --set 'rescale(n)=0' --worktiles 2
```

源码限制：空 split 的 `partialo` 保持未写，`COMBINE` 可预取未写字节但仅累加正权重项（全零权重时仍置零输出）；native fast `exp2/log/division` 无独立误差界。
