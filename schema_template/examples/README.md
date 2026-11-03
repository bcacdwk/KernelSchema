# Schema 示例

本目录保存按现行规范维护的完整任务。日常使用同一个[作者检查入口](../README.md#4-检查与处理反馈)；传入任一节点都会检查所在任务的全部 YAML。

## 第一批四个任务

以下任务从 `aka-four-problems-20261020` 提取结果复制并规范化。[来源快照](https://github.com/bcacdwk/KernelSchema/tree/e8605fd9107dec1339cc4593c01049cea4570b0d/schema_wiki/aka-four-problems-20261020) 固定在现行字段迁移前；原 wiki 目录已在 `448ab85` 从当前树移除，可查阅[移除前版本](https://github.com/bcacdwk/KernelSchema/tree/3d64ad35900a0e6a66d6f93da34108fbceba0daf/schema_wiki/aka-four-problems-20261020)。本页的旧源码与测量记录也固定到该提交，保留历史依据。节点集合、dispatch 条件、算法、精度路径、分块和缓冲容量沿用来源，微流水线按现行规范明确动作身份、完成保证、依赖与空分支。

| 任务入口 | YAML 节点数 | 当前表达与核对范围 |
| --- | --- | --- |
| [FA4 prefill](fa4_prefill/dispatch.md) | 11 | 补实硬件摘要和实现环境；显式登记 Stats、FinalSum 的生产者及成员依赖，明确累积完成保证，规范 combine 尾部预取的空区间。 |
| [FA4 decode](fa4_decode/dispatch.md) | 7 | 明确必传 `None` 输入和库容量；交接引用实际等待完成；分别表达轮次放行、操作数到达和结果完成，保留空分支动作及 no-op 保证。 |
| [QSA prefill](qsa_prefill/dispatch.md) | 1 | 补实环境与 TMEM 容量；明确 ticket 交接、延迟 gather 发布及空 worker 的 drain 行为，保留未验证事项。 |
| [GDN prefill](gdn_prefill/dispatch.md) | 2 | 还原前置类型转换提交；补实环境与 TMEM 容量；统一具名乘积及角色引用，显式登记 U 的生产者。 |

共 21 个 YAML 节点，各算子目录保留自己的 `dispatch.md`、YAML 和同名 PNG。四份 `dispatch.md` 与来源快照一致；生成命令及验证边界见本页，节点关系、选择条件和修订号以 YAML 为准。

本批 L20D 的融合 GDN 与之前 B200 上的多阶段示例具有不同的任务要求和数值路径；这里保留的是本批原有的两条设计，不从另一示例移植算法。

## 生成 PNG

需要 Python 3.10+、PyYAML 和 Graphviz 的 `dot`。从仓库根目录执行下面的命令，即可重建全部 21 张图：

```sh
python3 schema_template/tools/pipeline_graph.py schema_template/examples/fa4_prefill/root.yaml --all --set N=4 --set Ssplit=6 --worktiles 2
python3 schema_template/tools/pipeline_graph.py schema_template/examples/fa4_decode/root.yaml --all --set N_r=6 --set ScoreStageCount=4 --set Lead=2 --set SplitCount=2 --set KVStageCount=8 --worktiles 1
python3 schema_template/tools/pipeline_graph.py schema_template/examples/gdn_prefill/root.yaml --set Np=6 --worktiles 2
python3 schema_template/tools/pipeline_graph.py schema_template/examples/gdn_prefill/m64_single_sequence.yaml --set Np=6 --worktiles 1
python3 schema_template/tools/pipeline_graph.py schema_template/examples/qsa_prefill/root.yaml --set 'N(b)=4' --worktiles 2
```

每个 `name.yaml` 直接在同目录生成 `name.png`。去掉 `--all` 并换成目标 YAML 路径，可只重建一个节点。需要检查依赖而不写图片时加 `--check-only`，此时不需要 Graphviz。

这些整数是明确选定的展示规模，不是新增的设计参数或真实 workload：prefill 展示环复用与跨工作接续，decode 展示短 split 的阶段关系，GDN 展示标量环及两份工作。QSA 的 `N(b)=4` 是缩小的示意展开，真实任务要求至少 32 个 union tile。生成器不猜测缺失取值、不按算子名称选择配置，也不从旧 PNG 读取生成参数。

图中保留 phase、role、动作、依赖类型和展示取值，没有实测时间尺度。PNG 内保存来源文件名、内容指纹和生成取值，不再产生旁置配置或图报告；当前 YAML 与图片是否对应，由回归测试核对。

## 规范化依据

四个任务的 L20D / SM103 身份沿用任务与运行记录。共用的执行层级、存储与异步行为摘要依据 [CUDA Programming Guide 13.0](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#thread-block-clusters) 和 [PTX ISA 9.0](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#tensorcore-5th-generation-family-instructions)，不从算子配置推断设备容量或峰值性能；完整设备资料的 wiki 绑定仍未完成。

环境版本只采用与原实现源码哈希匹配的记录：[FA4 prefill](https://github.com/bcacdwk/KernelSchema/blob/3d64ad35900a0e6a66d6f93da34108fbceba0daf/kernel_wiki/aka-four-problems-20261019/validation-20261021/results/fa4_prefill/profile.json)、[FA4 decode](https://github.com/bcacdwk/KernelSchema/blob/3d64ad35900a0e6a66d6f93da34108fbceba0daf/kernel_wiki/aka-four-problems-20261019/validation-20261021/results/fa4_decode/profile.json)、[QSA](https://github.com/bcacdwk/KernelSchema/blob/3d64ad35900a0e6a66d6f93da34108fbceba0daf/kernel_wiki/aka-four-problems-20261019/evidence/qsa_prefill/environment.json)及其[源码清单](https://github.com/bcacdwk/KernelSchema/blob/3d64ad35900a0e6a66d6f93da34108fbceba0daf/kernel_wiki/aka-four-problems-20261019/evidence/qsa_prefill/source_manifest.json)、[GDN](https://github.com/bcacdwk/KernelSchema/blob/3d64ad35900a0e6a66d6f93da34108fbceba0daf/kernel_wiki/aka-four-problems-20261019/validation-20261021/results/gdn_prefill/profile.json)。记录覆盖 CuTe DSL 4.6.1、PyTorch 2.9.0+cu130 和 cuda-python 13.1.1；`cu130` 不被解释成已核实的 nvcc 或驱动版本。

### 库提供的容量与转换

FA4 decode 原代码查询 `sm_100` helper 得到的值为 `T=512` 列、`Csm=232448` 字节；保留它们原有的 stage 与领先距离公式。依据固定 CUTLASS 修订的 [TMEM 表](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/python/CuTeDSL/cutlass/cute/arch/tmem.py#L31-L58)、[SMEM 表](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L145-L158)和[查询实现](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/python/CuTeDSL/cutlass/utils/smem_allocator.py#L118-L135)，并与官方 4.6.1 wheel 交叉核对。GDN 的同一 TMEM helper、QSA 的源码容量断言也给出 512 列；这些是原库调用的结果，不是重新选择缓冲大小。

GDN 的 `cu_seqlens.to(torch.int32)` 仍是计时范围内的独立提交。其[原 NCU 数据](https://github.com/bcacdwk/KernelSchema/blob/3d64ad35900a0e6a66d6f93da34108fbceba0daf/kernel_wiki/aka-four-problems-20261019/validation-20261021/results/gdn_prefill/ncu.csv)记录一个 CTA、128 个线程。[PyTorch 2.9.0 的 copy 路径](https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/cuda/Copy.cu#L171)、[展开提交](https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/cuda/CUDALoops.cuh#L481)和[线程配置](https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/cuda/thread_constants.h#L16)表明，每 CTA 容量为 512 个元素，因此整个契约内的 `B+1` 个边界（2–17 个）总是一份完整逻辑工作；四个 warp 只处理有效元素。没有改成新的转换 kernel，也没有增加 host 同步。单序列主 kernel 不读取转换结果的事实保留。

### 表达依据

- FA4 decode 的 `sinks`、`o_sf_scale` 在[原入口](https://github.com/bcacdwk/KernelSchema/blob/3d64ad35900a0e6a66d6f93da34108fbceba0daf/kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/qoder_greenfield_decode_attention/operator.py)中均为必传 `None`。使用已定义的非数值单值类型 `python_none`，不删输入、不增加 `default`，也不把 YAML `null` 当作未知值。[算法要求](../../schema_docs/1.contract.algorithm_requirement.md#rangedefault-与空值)补充了这一场景。
- root、q1_global、q4_compact 原有 `launch_ready` 局部别名重复了内建事件；删除别名，保留实际发布动作及 PDL。`read_partial` 从等待动作的 `issue` 改为 `done`，对应[原 combine](https://github.com/bcacdwk/KernelSchema/blob/3d64ad35900a0e6a66d6f93da34108fbceba0daf/kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/qoder_greenfield_decode_attention/gqa_decode.py)先执行 `griddepcontrol_wait`、再读取 partial 的顺序。等待可提前开始，完成后才消费；[交接规则](../../schema_docs/4.design.work_schedule.md#交接字段)补充了这个区别，没有把所有 `wait_at@issue` 都列为非法。
- FA4 prefill 的 Stats、FinalSum 捕获显式等待 Softmax 的发布；[原实现](https://github.com/bcacdwk/KernelSchema/blob/3d64ad35900a0e6a66d6f93da34108fbceba0daf/kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py) 中的统计量等待、读取与释放是这些边的依据。combine 尾部预取用 `min(s+3,Ssplit)` 表达空区间，不增加预取动作。
- FA4 decode 的 [paged 实现](https://github.com/bcacdwk/KernelSchema/blob/3d64ad35900a0e6a66d6f93da34108fbceba0daf/kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/qoder_greenfield_decode_attention/gqa_decode_paged.py) 在取得最后一个 slice 或操作数后放行对方，随后仍有 copy、GEMM 或 commit。`yield_turn` 因此与整块数据或结果的完成分开表达；空 split 仍保留原实现实际执行的初始化和收尾。
- QSA 的 [原实现](https://github.com/bcacdwk/KernelSchema/blob/3d64ad35900a0e6a66d6f93da34108fbceba0daf/kernel_wiki/aka-four-problems-20261019/qsa_prefill/aka_solution/qwen38_core_qsa_cutedsl/qsa_kernel.py) 在发布 ticket 后等待接收确认，利用 `wait_group(1)` 延迟发布前一块 gather，并在已有请求时排空末块。对应动作和依赖保留这些顺序；无请求时的 drain 不产生数据访问。
- GDN 原来的 `mma:KK(n)` 等动作按已定义的乘积结果命名；通用 `mma:A@B` 则按操作数解释。自定义 `matmul:KK(n)` 沿用原契约与事件引用。Recurrence 的 `read:U(n)` 显式等待 `MMA1/matmul:U(n)`，同一 serial role 中此前的 `release:V(n)` 已通过相同前驱保证这个顺序。
- GDN 单序列保留了未被主 kernel 消费的物化结果。[数学区](../../schema_docs/3.design.math_reshape.md#符号属性)允许如实说明这种已存在的工作，不要求为了满足“keep_value 有用途”而虚构消费者或删除开销。

## 尚未解决的事实与验证边界

| 范围 | 保留的缺口 |
| --- | --- |
| 所有任务 | 完整 wiki 资料绑定；部分 CUDA 编译器、驱动版本未单独记录。 |
| FA4 prefill | 部分快速数学操作的独立误差依据。 |
| FA4 decode | fast-exp2/reciprocal 误差界、FP32 contraction、BF16 reduce-add 行为，以及 PageIndex 环复用的顺序依据。 |
| QSA | 未初始化 Q padding 对投票的影响、HalfMax 覆盖安全、近似误差界。 |
| GDN prefill | 快速 log2/exp2 误差界，以及多序列描述符更新与旧在途读取的顺序依据。 |

这些缺口在相应设计说明中保留。显式动作和依赖表达已有实现的顺序，不为消除未决事项增加同步或替换数值路径；库容量仍需追踪到实际定义。

算子已经真实跑过 GPU；当前算子和 schema 都不需要 GPU 正确性验收。示例规范化与工具回归核对 schema 表达、结构及关联。
