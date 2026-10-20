# GDN Prefill solution：FlashInfer SM100/SM103 CuTeDSL

这里保存的是 FlashInfer 固定提交
`2ab910c58fdd2392914ea05e2a8714946ac0eef6` 中真实的 Blackwell GDN
Prefill kernel，不是对 `flashinfer.gdn_prefill` 的 API 转发。

- `gated_delta_net_tile_scheduler.py`：上游 persistent tile scheduler，原样保存。
- `gated_delta_net_chunked.py`：上游完整 CuTeDSL device kernel，原样保存。
- `gdn_prefill.py`：上游 SM100/SM103 编译、缓存与 launch adapter，原样保存。
- `solution.py`：将以上源码单文件化并接入题目 `Model.forward` ABI；这是实际提交 Agate 的候选。
- `build_solution.py`：可复现的打包脚本。只调整模块边界、内联两行 `get_num_sm`、增加 ABI adapter，随后在保留局部变量名和类型标注的条件下压缩源码。
- `agate_reference/`：仅供 Agate 网关使用的等价题面副本。原题 `input.py` 所需的官方 reference 函数被直接内联并压缩，以通过输入生成器的安全扫描和大小限制；输入分布、reference、shape 和容差未改变。

## 题目契约

固定结构为 `HQ/HK/HV=4/4/32`、`DK=DV=128`，输入为 packed-varlen
BF16 Q/K/V、FP32 alpha/beta 与 FP32 `[B,H,V,K]` state。候选必须写满并返回
预分配的 BF16 output 和 FP32 final state。目标机器为 L20D/SM103，要求 CUDA 13。

候选直接调用抽取出的 `chunk_gated_delta_rule_sm100`。该实现是 persistent
CuTeDSL chunked GDN kernel，内部使用 TMA、TMEM 和七段矩阵乘流水；题目采用固定
`use_cp=False` 路径。

## 构建

```bash
uv run --with 'python-minifier==3.3.0' python build_agate_reference.py
uv run --with 'python-minifier==3.3.0' python build_solution.py
```

单文件需要压缩是因为 Agate 候选源码上限为 131072 字节；未压缩组合源码约
226 KB，生成的候选约 123 KB。三个上游源文件始终保留为可读、可审计副本。

## Agate

```bash
agate eval --gpu L20D \
  --candidate solution.py \
  --reference-dir agate_reference \
  --mode correctness_only --job-timeout 1800
```

真实作业结果记录在 `AGATE_VALIDATION.json`。正确性作业
`ev_94ec39c2dc41` 与完整作业 `ev_4e1e2792ff14` 均已通过全部 30 个 shape、
180 个随机输入；完整作业的 30 个性能 shape 也全部完成，中位延迟的几何平均为
`141.824 us`。官方逐元素门限为：output
`atol=0.01, rtol=0.01`，final state `atol=0.005, rtol=0.001`；返回值和原地
写入的两个缓冲区都会检查。

完整作业请求了锁频，但共享 worker 没有实际施加锁频；采样期间观测到 SM 时钟为
2032 MHz（100% 最大频率）。因此该性能数字是有效实测，但不能标记为“硬件锁频结果”。
