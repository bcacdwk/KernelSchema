**四个冻结 SOTA 算子复验报告 — 2026-10-21**

结论：四个 `aka_solution/` 的冻结实现均通过原题正确性检查。按 NCU 口径对照，四题速度与 wiki 基本对应；FA4 Production 依赖版本未能完全对齐，具体限制见下文。验证期间未修改任何原始题面、solution 或 `AKA_experiment` 文件；本报告及全部中间产物均在本目录。

下表为完整 wiki shape 集合；耗时单位 µs，取每 shape 多次采样的中位数，再对全部 shape 取几何平均。加速比为 Production / AKA，包含必要的预处理、合并与 dtype 转换 kernel。

| 算子 | 正确性 shape / case | Wiki AKA | 本次 AKA | 耗时差异 | Wiki 加速比 | 本次加速比 | 更快 shape |
|---|---:|---:|---:|---:|---:|---:|---:|
| FA4 Prefill | 30/30；30/30 | 1043.419 | 1035.592 | -0.75% | 1.1990× | 1.2009× | 30/30 |
| FA4 Decode | 30/30；30/30 | 27.774 | 26.564 | -4.36% | 1.1126× | 1.1349× | 19/30 |
| QSA Prefill | 4/4；16/16 | 424.101 | 428.667 | +1.08% | 4.2724× | 4.2492× | 4/4 |
| GDN Prefill | 30/30；180/180 | 194.418 | 196.088 | +0.86% | 1.1591× | 1.1594× | 25/30 |

**需要区分的口径**

- 当前 `AKA_experiment/assets/bench` 的 GDN 是 17 个 shape，是 wiki 30 个的原样子集；输入生成器、reference 与 correctness contract 一致。这 17 个全部通过。在 NCU base 时钟下，本次为 **1.1270×**，历史证据按相同 17 个重算为 **1.1248×**；不能直接引用 30-shape 的 1.1591×。
- GDN 的旧 NCU 绝对延迟在 `--clock-control base` 下复现：AKA 196.088 µs，对比历史 194.418 µs。默认未锁频对照为 118.579 µs、1.1765×。此前约 39% 的绝对时间差与时钟条件一致，不能把它当成源码加速。旧记录没有完整时钟证据，base 复跑提供了可复现实验对照。
- GDN wiki 另外列出的标准 Agate benchmark 也复现：固定 `atrex-bench==0.1.0.post20260917160311`、100 bench iters、6 correctness cases、eager，实测 **121.457 µs**，历史均值 **121.297 µs**，差 **+0.132%**；30/30 编译、正确性、性能及 180/180 case 通过。该值与 NCU base 时钟的 196 µs 是不同测量条件。
- QSA 使用冻结实现已验证的 eager 调用方式，未启用 CUDA Graph replay。
- FA4 表格使用远端镜像的 `flashinfer-python==0.6.16.post1`；wiki Production 依赖声明是 0.6.18。尝试在独立 job 中装入精确 0.6.18 wheel，但两个准备命令退出码为 1、`command_ok=false`、缺失评测产物，不能当成通过。它们的网关顶层 `status=succeeded` 仅代表作业容器完成。AKA 源码、Torch 与 CuTeDSL 版本已对齐，Production 的版本差异仍限制精确环境复现。失败回执和短超时安装诊断均保留。

**正确性与源码证据**

直接读取每道题原有 `metadata.json` 的逐元素阈值；未放宽阈值或以 relative-L2 代替逐元素检查。检查返回 tensor 的形状、dtype、原地缓冲区地址、被写入的输出，以及所有非 scratch 输入的逐字节不变性。输出在调用前填 NaN，40 次 eager 测量后再检查一次，防止遗漏写入或陈旧输出。

| 算子 | atol / rtol | 本次最大绝对误差 | 最大误差 / 容差 |
|---|---|---:|---:|
| FA4 Prefill | 0.06 / 0.04 | 0.0703125 | 0.895522 |
| FA4 Decode | 0.06 / 0.04 | 0.0625 | 0.260417 |
| QSA Prefill | 0.000032 / 0.02 | 1.52587891e-05 | 0.294364 |
| GDN Prefill | output: 0.01 / 0.01；state: 0.005 / 0.001 | 0.00048828125 | 0.082531 |

原包 MANIFEST 中的 222 个文件在验证前后 SHA-256 全部一致。`manifest_check.json`、`problem_audit.json`、`runtime_source_check.json` 保存文件与题面核对结果。原 `verify_package.py` 会因包内预先存在的 `.DS_Store` 以及本次新增结果目录而拒绝“文件集合完全相等”，因此本次单独验证全部清单条目的字节数与哈希，没有删除这些文件或改动原校验脚本。

**测量方法与环境**

目标为 NVIDIA L20D / SM103，148 SM，Torch 2.9.0+cu130、Triton 3.7.1、CuTeDSL 4.6.1。FA4 Production 采用原包 `production_ncu.py` 的复用 counter buffer 适配器；QSA/GDN Production 使用原包 `solution/solution.py`。FA4 seed `solution/` 不是本次要验证的 SOTA，也没有被当成 NCU Production。

正确性参考为原题 PyTorch reference。自定义 Agate harness 的用途是保留多文件源码闭包、原始输入 ABI，并在同一进程、相同输入上配对测量 Production / AKA；标准单候选 profile 接口不能直接表达这组配对过程。GDN 另经标准 `agate eval` 独立复核。

NCU 采集 `gpu__time_duration.sum` 与 `gpc__cycles_elapsed.avg.per_second`；先 warmup 10 次，再仅在目标调用周围打开 profiler。成功完成的四题每侧每 shape 均 4 次；未完成的 0.6.18 补测原计划为 8 次，顺序交替 AB / BA；包含全部必要 kernel。保留全部原始样本，没有剔除慢样本。FA4/QSA `--clock-control none`；GDN 额外使用 `base` 复现旧 NCU 数字。

| 算子 | 本次 Production µs | NCU 最大 CV | NCU job | 正确性 job |
|---|---:|---:|---|---|
| FA4 Prefill | 1243.670 | 13.67% | `dv_3e2128a5beb1` | `dv_3e2128a5beb1` |
| FA4 Decode | 30.146 | 3.36% | `dv_2703beb26d56` | `dv_2703beb26d56` |
| QSA Prefill | 1821.472 | 0.31% | `dv_17b5efef6a3b` | `dv_17b5efef6a3b` |
| GDN Prefill | 227.336 | 3.09% | `dv_a8c5ae1ccfbe` | `dv_27f813348f5c` |

Prefill 较长 shape 的部分首样本较慢，因此单次耗时并非全部稳定；以上采用全部样本中位数。逐 shape 的原始样本、CV、历史差异与胜负见各结果目录的 `per_shape.csv`。

**文件索引与复算**

- `final_summary.json`：最终所选对照、当前 GDN 17-shape 子集、标准 Agate 结果和全部 job ID。
- `results/<name>/validate.json`：逐 shape、逐 seed、逐输出正确性；`profile.json`：实际采样顺序与源码 SHA-256；`ncu.csv`：原始 NCU 指标；`kernel_launches.csv`：全部 launch 归属；`per_shape.csv` / `summary.json`：逐 shape 和汇总。
- `results/*.job.json`：原始 Agate 回执；`*.submission.json` / `*.submission.log`：提交记录；`results.tar.gz`：下载的原始产物。
- `bundles/` 与四个算子 `.tar.gz`：实际提交的题面与源码快照；`harness.py`、`run_remote.sh`、`run_remote_pinned.sh`、`run_clock_base.sh`：测量代码；`dependency_pin.json`：0.6.18 wheel 的 SHA-256。
- 在本目录运行 `python3 -B analyze.py` 和 `python3 -B write_report.py`，可从已保存证据复算，无需提交新 GPU 作业。
