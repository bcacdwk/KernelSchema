# Schema 示例

本目录保存按现行规范维护的完整任务。日常使用同一个[作者检查入口](../README.md#4-检查与处理反馈)；传入任一节点都会检查所在任务的全部 YAML。

## 第一批四个任务

以下任务从 [aka-four-problems-20261020](../../schema_wiki/aka-four-problems-20261020/) 复制并规范化。原目录保留作历史对照；节点集合、dispatch 条件、算法、精度路径、分块和核心流水线保持原设计。

| 任务入口 | YAML 节点数 | 本次主要整理 |
| --- | --- | --- |
| [FA4 prefill](fa4_prefill/dispatch.md) | 11 | 补实硬件摘要及有同源记录的实现环境；Design 与所有派生文件未改。 |
| [FA4 decode](fa4_decode/dispatch.md) | 7 | 明确两个必传 `None` 输入；补实库容量；删除重复的内建里程碑声明，将三处等待点定位到原等待动作完成。 |
| [QSA prefill](qsa_prefill/dispatch.md) | 1 | 补实环境与 TMEM 容量；将未验证结论的伪引用占位改为明确说明。 |
| [GDN prefill](gdn_prefill/dispatch.md) | 2 | 还原已存在的前置类型转换提交；补实环境与 TMEM 容量；统一具名乘积动作及相关引用。 |

共 21 个 YAML 节点，四份 `dispatch.md` 与历史版本一致。数据发生变化的节点递增修订号，未变化的节点保留原版本。

本批 L20D 的融合 GDN 与之前 B200 上的多阶段示例具有不同的任务要求和数值路径；这里保留的是本批原有的两条设计，不从另一示例移植算法。

## 规范化依据

四个任务的 L20D / SM103 身份沿用任务与运行记录。共用的执行层级、存储与异步行为摘要依据 [CUDA Programming Guide 13.0](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#thread-block-clusters) 和 [PTX ISA 9.0](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#tensorcore-5th-generation-family-instructions)，不从算子配置推断设备容量或峰值性能；完整设备资料的 wiki 绑定仍未完成。

环境版本只采用与原实现源码哈希匹配的记录：[FA4 prefill](../../kernel_wiki/aka-four-problems-20261019/validation-20261021/results/fa4_prefill/profile.json)、[FA4 decode](../../kernel_wiki/aka-four-problems-20261019/validation-20261021/results/fa4_decode/profile.json)、[QSA](../../kernel_wiki/aka-four-problems-20261019/evidence/qsa_prefill/environment.json)及其[源码清单](../../kernel_wiki/aka-four-problems-20261019/evidence/qsa_prefill/source_manifest.json)、[GDN](../../kernel_wiki/aka-four-problems-20261019/validation-20261021/results/gdn_prefill/profile.json)。记录覆盖 CuTe DSL 4.6.1、PyTorch 2.9.0+cu130 和 cuda-python 13.1.1；`cu130` 不被解释成已核实的 nvcc 或驱动版本。

### 库提供的容量与转换

FA4 decode 原代码查询 `sm_100` helper 得到的值为 `T=512` 列、`Csm=232448` 字节；保留它们原有的 stage 与领先距离公式。依据固定 CUTLASS 修订的 [TMEM 表](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/python/CuTeDSL/cutlass/cute/arch/tmem.py#L31-L58)、[SMEM 表](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L145-L158)和[查询实现](https://github.com/NVIDIA/cutlass/blob/e05f953a5b3d38adc240df2ff928e0421c2abba3/python/CuTeDSL/cutlass/utils/smem_allocator.py#L118-L135)，并与官方 4.6.1 wheel 交叉核对。GDN 的同一 TMEM helper、QSA 的源码容量断言也给出 512 列；这些是原库调用的结果，不是重新选择缓冲大小。

GDN 的 `cu_seqlens.to(torch.int32)` 仍是计时范围内的独立提交。其[原 NCU 数据](../../kernel_wiki/aka-four-problems-20261019/validation-20261021/results/gdn_prefill/ncu.csv)记录一个 CTA、128 个线程。[PyTorch 2.9.0 的 copy 路径](https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/cuda/Copy.cu#L171)、[展开提交](https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/cuda/CUDALoops.cuh#L481)和[线程配置](https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/cuda/thread_constants.h#L16)表明，每 CTA 容量为 512 个元素，因此整个契约内的 `B+1` 个边界（2–17 个）总是一份完整逻辑工作；四个 warp 只处理有效元素。没有改成新的转换 kernel，也没有增加 host 同步。单序列主 kernel 不读取转换结果的事实保留。

### 表达修正与规则反馈

- FA4 decode 的 `sinks`、`o_sf_scale` 在[原入口](../../kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/qoder_greenfield_decode_attention/operator.py)中均为必传 `None`。使用已定义的非数值单值类型 `python_none`，不删输入、不增加 `default`，也不把 YAML `null` 当作未知值。[算法要求](../../schema_docs/1.contract.algorithm_requirement.md#rangedefault-与空值)补充了这一场景。
- root、q1_global、q4_compact 原有 `launch_ready` 局部别名重复了内建事件；删除别名，保留实际发布动作及 PDL。`read_partial` 从等待动作的 `issue` 改为 `done`，对应[原 combine](../../kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/qoder_greenfield_decode_attention/gqa_decode.py)先执行 `griddepcontrol_wait`、再读取 partial 的顺序。等待可提前开始，完成后才消费；[交接规则](../../schema_docs/4.design.work_schedule.md#交接字段)补充了这个区别，没有把所有 `wait_at@issue` 都列为非法。
- GDN 原来的 `mma:KK(n)` 等动作按已定义的乘积结果命名；通用 `mma:A@B` 则按操作数解释。将前者统一命名为自定义 `matmul:KK(n)`，沿用原动作契约，并同步全部事件引用；未改变动作、依赖或完成保证，也未引入新的隐式数据依赖。
- GDN 单序列保留了未被主 kernel 消费的物化结果。[数学区](../../schema_docs/3.design.math_reshape.md#符号属性)允许如实说明这种已存在的工作，不要求为了满足“keep_value 有用途”而虚构消费者或删除开销。

这四个任务在迁移开始时的检查器上也全部通过，不需要放宽检查规则或增加字段；示例继续使用现有作者入口和示例回归。增加的类型回归区分明确的非数值哨兵与真正缺失的 dtype/default。

## 尚未解决的事实与验证边界

| 范围 | 保留的缺口 |
| --- | --- |
| 所有任务 | 完整 wiki 资料绑定；部分 CUDA 编译器、驱动版本未单独记录。 |
| FA4 prefill | 部分快速数学操作的独立误差依据。 |
| FA4 decode | fast-exp2/reciprocal 误差界、FP32 contraction、BF16 reduce-add 行为，以及 PageIndex 环复用的顺序依据。 |
| QSA | 未初始化 Q padding 对投票的影响、HalfMax 覆盖安全、近似误差界。 |
| GDN prefill | 快速 log2/exp2 误差界，以及多序列描述符更新与旧在途读取的顺序依据。 |

这些缺口在相应设计说明中保留；本次没有新增同步、替换数值路径或编造证明。库容量曾藏在说明字符串中，即使结构检查通过，也仍需追踪到实际定义。

已有 GPU 记录在本次用于核对来源、环境和固定实现行为；没有重新运行 GPU 验收。通过 Python 检查表示结构及已实现的关联检查通过，不表示这些 schema 的数值、复用安全、活性或性能已经全部得到验证。
