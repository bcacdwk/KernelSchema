**已确认的基本动作词：四类八词。** 本文记录研究结论；正式 schema、算子 YAML 与检查器的更新另行实施。

| 类别 | 动作词 | 定义 | `done` 保证 |
| --- | --- | --- | --- |
| 数值计算 | `compute`（非矩阵运算） | 按上游定义执行逐元素、归约、转换等非矩阵乘类运算 | 指定结果可供声明的消费者使用 |
| 数值计算 | `matmul`（矩阵运算） | 执行上游定义的矩阵乘法及其累加；具体公式与数值路径由上游确定 | 指定矩阵结果可供声明的消费者使用 |
| 搬运访存 | `load`（读取） | 从绑定的源读取既有值到指定目标 | 本次源读取结束，目标数据可用；不自动归还源使用权 |
| 搬运访存 | `store`（存储） | 把已有值写入绑定的目标 | 目标写入完成，并满足声明的消费者所需的可见性 |
| 顺序同步 | `await`（等待其他人通知） | 等待其他参与者的已声明通知条件，可涉及多个通知 | 声明的条件及其交接保证已满足 |
| 顺序同步 | `signal`（通知其他人） | 向声明的参与者发出指定含义的通知 | 本方通知义务完成，不表示对方已经继续 |
| 资源占用 | `acquire`（申请资源） | 取得指定资源本次使用所需的资格 | 已实际取得使用资格，仅发出请求不算完成 |
| 资源占用 | `release`（释放资源） | 归还本参与者对指定资源或区域的使用权 | 本参与者已结束该部分使用；不代表所有参与者完成，也不必物理销毁资源 |

源、目标、访问区域、资源及参与者必须由已有声明或显式引用绑定；读取与存储的方向不能靠词名猜。通知还须绑定条件、多个通知的组合关系及含义。数据交接必须保证相应消费者按约定访问方式正确读取；启动许可等其他通知则遵循各自声明的含义。实现负责兑现所需的完成与可见性保证。

普通读写与结果落位可以包含在计算中，不必把每次计算拆成多个动作；有独立搬运、通知或使用权交接边界时，再组合相应动作。`release.done` 与 `store.done` 分别描述源使用权归还和目标写入完成，两者不能互相替代。动作如何推进仍由 `execution_mode` 与已声明前驱决定，`await` 不隐含等待全部在途工作。

**选择八词的依据。** 四组对照分别表达非矩阵／矩阵、读取／存储、等待／通知、申请／释放，便于辨认主要作用；数据变化、通知协议与使用资格各有明确归属。具体算法由上游数学定义与动作组合表达，词表只规定基本作用，减少实例自由造词和重复解释。较高层粒度保留关键设计，也便于跨平台对应。[原始统计](data/verb_census.json)覆盖 193 份有稿 YAML、141 种动作名，为这次归并提供了语料依据。覆盖目标是常见流水线动作的组合，而非逐条复刻完整指令集。

**`info.using_verb`。** 键只能从上述固定词表选择，并与该 role 的 pipe 用词一致。每项可写 `explain` 与 `execution_mode`，也可用 `{}`；`explain` 可省，至多一句，只解释对象、范围或上游绑定，不得重定义基础意义、改变 `done` 或隐藏前驱。按词填写的 `execution_mode` 覆盖 role 默认值，省略时沿用默认。

新格式局部示意：上游已定义非矩阵结果 `R`，并绑定其本地缓冲与写出目标；串行推进保证计算完成后再写出。

```yaml
info:
  execution_mode: serial
  using_verb:
    compute: {}
    store: {}
pipe:
  steady: ["compute:R", "store:R"]
```

**跨平台依据。** [NVIDIA PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) 是虚拟 ISA，提供矩阵、访存与同步指令族；AMD 的 [CDNA MFMA](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html) 与 [RDNA3 WMMA](https://gpuopen.com/download/rdna3-shader-instruction-set-architecture-feb-2023.pdf) 提供矩阵计算对应。`matmul` 的含义来自上游数学，不绑定 NVIDIA 指令的线程组织、布局或完成方式。[昇腾 Ascend C](https://gitcode.com/cann/asc-devkit/tree/82b89a15f26329511fc7fbc807d5132a1f9b4ba7/docs/zh) 的 `Mmad`、`DataCopy`、`SetFlag/WaitFlag`、`AllocTensor/FreeTensor` 分别提供相关能力的 API 对应。平头哥 [V900](https://www.t-head.cn/product?id=12) 公开资料可确认张量计算、互联与软件栈能力，尚不足以逐项确认 PPU 的具体动作指令。跨平台统一上述语义，具体实现遵守各平台及型号的契约。
