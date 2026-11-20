# FA4 decode：F0/F1 准入材料

本页保留 **K7.materials 材料预审**依据。F1 固定为 `q4_direct` 的 `B=1, Q=4, seq_lens=[64], P=1, block_tables[0,0]=0, N=1, S=1`。后续 K8 已提供 [CPU 准备、正式 Task/Source、独立原栈 evaluator 及接口说明](../../../examples/workflows/fa4_decode/README.md)。没有执行原 kernel、GPU 或模型，不能据此声明 runtime 支持或通过 KS-P2。

可消费文件位于 [examples/workflows/fa4_decode](../../../examples/workflows/fa4_decode/)：

| 文件 | 内容 |
| --- | --- |
| [materials.json](../../../examples/workflows/fa4_decode/materials.json) | 27 项固定原件的 Git 定位、SHA-256、大小、模式；完整 12 文件 `aka_solution/`；原比较器 harness；源码与 Schema 版本差异 |
| [f1-parameters.json](../../../examples/workflows/fa4_decode/f1-parameters.json) | 一个固定输入点、原 helper 的完整 kwargs、种子、13 参数映射、完整比较/副作用及实际 dispatch 条件 |
| [preflight-checks.json](../../../examples/workflows/fa4_decode/preflight-checks.json) | 本次实际 CPU 检查、工具身份和检查范围；非 GPU 收据 |
| [licenses-and-gaps.md](../../../examples/workflows/fa4_decode/licenses-and-gaps.md) | available/missing/incompatible、许可及 K8/公开发行缺项 |

这些 K7 JSON 保留为案例内材料记录，不是 `ks.workflow.v1` Task/Materials/Source/request。[prepare.py](../../../examples/workflows/fa4_decode/prepare.py) 读取所选固定 Git 原件，核摘要后冻结真正的协议对象和 Ref；不能把 Git SHA 当文件 SHA，也不能将本页的检查当 qualified。

## 两个固定来源

`kernel_source_commit=3d64ad35900a0e6a66d6f93da34108fbceba0daf` 固定四题快照中的 `kernel_wiki/aka-four-problems-20261019/fa4_decode/`，选中 `aka_solution` v55。`schema_commit=484902855fe7be82a74fe7c394503d1d7d222dbb` 固定当前 `schema_template/examples/fa4_decode/` 的四节点 Schema。

两者有直接对应依据：[root.yaml](../../../schema_template/examples/fa4_decode/root.yaml) 第 4/64 行和 [dispatch.md](../../../schema_template/examples/fa4_decode/dispatch.md) 第 5 行分别指向上述源码/参考固定提交。两个提交间原题目录只删除 `.DS_Store`；所有选中源码、reference、input、原 shapes、metadata 和原验证记录逐字节相同。本次同时核验 `PROVENANCE.json` 中六个 Python 文件的原始 SHA-256。

`3d64ad…` **并非四节点 Schema 的版本**：该提交中的 YAML 是旧七节点形式；后续删除 `q1_direct/q1_cluster/q4_compact` 并整理四份完整设计。清单保留旧/新 raw 的真实摘要；不回退旧 YAML，也不把当前 YAML 冒称为旧提交。四节点版本的 root 只覆盖 Q4、B15/16、L≥24576、非 skew 的 S9 global；它的共享 Contract 才是完整 Task 域。

必要源文件可从本仓库任意 checkout 的 Git 对象读取，例：

```sh
git show 3d64ad35900a0e6a66d6f93da34108fbceba0daf:kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/solution.py
git show 484902855fe7be82a74fe7c394503d1d7d222dbb:schema_template/examples/fa4_decode/q4_direct.yaml
```

按清单逐项核验输出的原字节再安装，不能依赖作者目录、工作树“同名文件”或在线补齐活动运行。历史 Schema 记有 `https://github.com/bcacdwk/KernelSchema` 来源；本次没有验证匿名下载这两个固定提交。公开取得证明和许可闭包仍缺失。

## F0：完整 ABI 和验收来源

以下行号按清单中相应固定提交。`aka_solution/solution.py:10–40` 的 `Model.forward`、`reference.py:11–26` 和 `operator.py:652–665` 都是以下顺序；不是删除 sentinel 或 scratch 后的简化 ABI。所有张量连续、位于同一 CUDA 设备，原 Contract 有 `B=1..17, Q∈{1,4}, Q≤L_b≤131072`。

| # / 参数 | root Contract 行 | F1 值或布局 | 副作用 |
| --- | --- | --- | --- |
| 1 `query` | 19 | `[4,16,256]` FP8 E4M3FN | 不变 |
| 2 `kv_cache` | 20 | `[1,2,1,64,256]` FP8 E4M3FN，HND，selector 0/1 为 K/V | 不变 |
| 3 `workspace_buffer` | 21 | `[413138944]` uint8，首次使用全零 | 可变 scratch |
| 4 `block_tables` | 22 | `[1,16434]` int32，有效前缀 `[0]`，其余由 helper 填 `-1` | 不变 |
| 5 `seq_lens` | 23 | `[1]` int32，值 `[64]` | 不变 |
| 6 `max_seq_len` | 24 | Python int `64` | 不变 |
| 7 `bmm1_scale` | 25 | Python binary64 float `0.0625` | 不变 |
| 8 `bmm2_scale` | 26 | Python binary64 float `1.0` | 不变 |
| 9 `window_left` | 27 | Python int `-1` | 不变 |
| 10 `sinks` | 28 | Python `None` | 不变 |
| 11 `o_sf_scale` | 29 | Python `None` | 不变 |
| 12 `out` | 30 | `[4,16,256]` BF16，原 helper 用 `torch.empty` | 完整覆写 |
| 13 `q_len_per_req` | 31 | Python int `4` | 不变 |

`reference.py:45–88` 按有效页表取 K/V，将输入转换为 FP32，计算右对齐 causal scores、softmax 和 value product，再窄化写入原 `out` 并返回它。F1 四个 query 的可见 key 数依次为 `61,62,63,64`。参考实现是数值参考，不自动成为合格性能 baseline。

完整比较式来自当前 `root.yaml:63–67`：对返回值与原位 `out_after` **分别**检查相同 shape/dtype、actual/reference 每元素均有限，并在 FP32 中逐元素满足 `abs(actual-reference) <= 0.06 + 0.04*abs(reference)`。`root.yaml:38–43` 只允许修改 out 与 workspace；`metadata.json:11–37` 记录两个输出容差、`mutates_inputs=[out]`、`scratch_inputs=[workspace_buffer]`。因此不能把 workspace 误列只读，也不能只测返回值、仅报 L2 误差或忽略 out。

K8 的 [完整 evaluator](../../../examples/workflows/fa4_decode/evaluator/entry.py) 已实现以下工程边界：保留原 out 的 storage/view 身份并核返回别名；检查每个 out 元素覆写；对 query/kv_cache/block_tables/seq_lens 做前后字节不变检查；独立准备 reference destination，避免参考调用覆盖候选证据。helper 后将同一 out 填 BF16 NaN，准备 launcher 后再次 poison，均在计时外且不换对象/存储。每次 cold/warmup/measurement 后均完整复查，不用包装器补写 out。CPU helper 测试未执行 F1 GPU，原 30/30 也不能证明新 evaluator 的真实资格。

已有可复用的真实比较器来源是同一 `3d64ad…` 的 [历史 harness.py](../../../kernel_wiki/aka-four-problems-20261019/validation-20261021/bundles/fa4_decode/harness.py)：`44–51` 行执行 FP32 有限/逐元素比较，`54–62` 行检查只读输入字节和 poison，`69–96` 行分别比较返回值/原位输入、核 alias 和未变输入，`120–121` 行禁用 TF32。K7 纳入原件；K8 仅在显式 synthetic CPU helper 测试中执行所选函数，未执行历史 main 或 F1 GPU。旧 alias 只比较调用后的 data_ptr，没有冻结调用前 out storage/view；`157–167` 行 post-timing 也仅复查返回数值。K8 接入实际 helpers 并补上述边界，真实执行资格仍待验证。

`shape_valid.json` 与 `shapes.json` 字节相同，各有 30 项；没有 F1 的 `[64],Q4` 这一项。`shape_train.json` 是 widened public domain，`development_cases=[]`，不是第三组 shape 列表。F1 来自既有 Schema README 第 17 行代表样本，不冒称历史 30 shape 中的一项。

## F1：固定输入、effective 和实际分派

`f1-parameters.json` 可直接提供原 `_make_inputs(**kwargs)` 的十个参数。固定种子 `20261120` 是本次选择，尚无生成张量；真实运行须冻结 torch/CUDA/设备及输入摘要。保留 `input.py:6–10,41–62` 的 Q4 分布：CUDA BF16 `randn` 乘 `1.22/0.96` 后转换 FP8，query 先于 KV 消耗 RNG；保留完整 workspace 常量。不得改原 YAML/helper 或换 BF16/page128 桥以适应环境。

材料条件满足 `q4_direct.yaml:8` 的 `Q=4, B∈[1,17], L∈[4,256], L_b≥4`。原 host `_select_policy:78–86` 的纯整数部分在 F1 返回 `(1,"none")`；`_pick_tiles` 给 `(16,2)`，编译选择 `bk=256`、单 softmax WG、`compact=false`。因此 `T_b=N=S=1`，每个 query tile 由一个 CTA 完成，完整 F1 有 **两个** query tile/worktile。root 的 B/L/S 条件拒绝该点；Q1 绑定 root 同样不合法。

CPU 通过既有 Template `effective_document("q4_direct")` 验证：读取视图含 **root Contract + direct Design/Realization**；三个 Design 区 provider 都为 `q4_direct`，依赖 raw 为 root/direct。q1_global 的 Math_reshape provider 则是 root，说明仅靠文件名或本地 raw 不能推断闭包。保留实际 dispatch 原件及完整四节点 raw 供全任务覆盖审查；单独 direct raw 没有 Contract，不能作为完整任务。effective 是阅读产物，不替 raw。

当次 Scope 只选该 F1 点；合报告必须分开 coverage/fidelity，`task_uncovered` 保留除此点外的完整 Contract 域，包括其余 direct 输入、Q1、cluster、global 和 compact 分支。材料条件通过尚不等于全部 when 通过：四个 YAML 还限定已建立 native launcher、同对象/签名/线程/当前设备且非 stream capture，这些 runtime 条件本次为 `not_run`。

## 原栈、源码格式与冷启动

保留 `root.yaml:68–93` 的 L20D/SM103、CuTe DSL 4.6.1、torch 2.9.0+cu130、cuda-python 13.1.1。原 requirements 仅钉 `nvidia-cutlass-dsl==4.6.1`，torch 为 `>=2.9`、cuda-python 未钉；TVM FFI 版本、Python、实际驱动/编译器、设备身份与精度开关未形成当前合格 profile。硬件 Wiki 多项仍是 `@pending`。旧 production metadata 的 torch 2.11 开发版和 roofline 的 B300/SM100 标签属于别的记录，不能补作该原栈版本。H20/Triton 首题资格不能代签 CuTe/实际 SM103。

`solution.py` 必须与 `qoder_greenfield_decode_attention/` 同目录；六个 Python 文件都在原清单。即使 direct 不启动 global combine，paged 模块仍导入 `gqa_decode.py` 中符号，不能只取入口或一个 kernel 文件。有限 `python_tree` codec、完整冻结/装载/导入、原子安装/替换及删除语义已通过 CPU 测试；[实际格式与 reference 入口](../../../examples/workflows/fa4_decode/README.md) 已连接 K3 typed 全文件输出、reference implement/verify、多文件 fidelity 和 FA4 Agate 原作业通道。AKA 仍需自身兼容接线；原栈/只读/CUDA/完整输出/独立 reviewer 的真实资格仍未取得，不由 CPU 工程代签。

`operator.py:474–583,652–843` 区分首次调用和 cache hit。冷路径先执行正常 fmha，再可能克隆 out、经 FFI 重放、设备同步并探测误差；失败/异常可恢复 out。`delta/denominator`（分母≤`1e-6` 时取 delta）≤`1e-2` 只是 launcher 探测门，绝非 Task 的 `0.06/0.04` 容差。记录冷启动/准备事实后，只有满足 native 条件的完整 `Model.forward` 和完成边界才可量声明的稳态。原性能 README 第 44–49 行明确 kernel-sum 排除 host。

## 已排除的路径与仍未决的义务

| 限制及证据 | 对 F1 的结论 |
| --- | --- |
| root `:167–185`：普通 global combine 读取空 split 旧 opartial；跨 `(S,B,Q)` 布局可把旧非有限位型作为零权重 payload | F1 是 direct，没有 global combine/空 split；不为 global 补签 |
| [pageindex_order.md](../../../schema_discussion/template_examples_20261117/fa4_decode/pageindex_order.md) `:5–18`；paged `:1228–1249,1343–1353`：N≥5 四槽预取可能早于 V 捕获同槽 | F1 N1 不发生第 5 组复用；不能从有限图无环推导全部异步安全 |
| direct `:295–302,383–398`：N1 仍通知、读取空 sum/numerator parity，随后释放 TMEM | **仍需**核源码、原语及实际执行，不是“只有一轮所以安全” |
| direct `:18–20,443–466`：初轮含未使用页项、TMA/KV 环、TMEM fence、output staging 生命周期 | **仍需**审有效页/负页号的 TMA 语义、页尾遮罩和完成可见性；F1 的物理页满 64 token，但逻辑 bk=256 仍有 padding |

本次只读 Template 检查通过四份 YAML、dispatch 和 F1 的有限图：`88 actions / 284 edges / 8 boundary skips / acyclic`；图仍是每 step 一个代表 worker，`worktiles=2` 不是两个 CTA 的 GPU 执行。CPU 不证明共享地址无竞态、数值、完整副作用、native 支持或性能。

在固定 Schema checkout 中安装 Template 自身的 PyYAML 依赖后，可从仓库根目录复核，命令只读且不生成 PNG：

```sh
python3 -B schema_template/check_yaml.py schema_template/examples/fa4_decode/q4_direct.yaml --read-only --graph check --set N=1 --set splitcount=1 --worktiles 2
```

本次实际使用 Python 3.13.12 与临时隔离的 PyYAML 6.0.3；默认 Python 首次因缺 PyYAML 返回 2，补齐临时 CPU 依赖后返回 0。此环境记录不构成计划中 Python 3.11/CuTe runtime 的资格。材料清单可用 Python 标准库对 `git show <commit>:<path>` 原字节计算 SHA-256/size，逐项与清单比较；13 参数检查仅解析 AST，原 host 检查只提取四个纯整数函数，未导入原 CUDA 模块。

K7.materials 保持“预审完成且如实披露缺项”。K8 的 KS 工程现已接通 K3 typed 全树、reference implement/verify、多文件 fidelity 和显式 FA4 Agate channel；独立原码观察导入仍只能产生 unknown，不能代替合格 channel。AKA 的 FA4 接线需要独立完成。K8.real/KS-P2 仍缺原栈真实资格、有限 fidelity、冷/稳态证据及公开取得/许可闭包。历史 L20D 30/30、1.1126× kernel-sum 和既有异步 stress 叙述只保留为原码材料。
