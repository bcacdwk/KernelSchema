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

源、目标、访问区域、资源及参与者须能从上游定义和本动作的 `describe` 唯一确定；读取与存储的方向不能靠词名猜。通知还须绑定条件、多个通知的组合关系及含义。数据交接必须保证相应消费者按约定访问方式正确读取；启动许可等其他通知则遵循各自声明的含义。实现负责兑现所需的完成与可见性保证。

普通读写与结果落位可以包含在计算中，不必把每次计算拆成多个动作；有独立搬运、通知或使用权交接边界时，再组合相应动作。`release` 的 done 与 `store` 的 done 分别描述源使用权归还和目标写入完成，两者不能互相替代。动作如何推进仍由 `execution_mode` 与已声明前驱决定，`await` 不隐含等待全部在途工作。

**选择八词的依据。** 四组对照分别表达非矩阵／矩阵、读取／存储、等待／通知、申请／释放，便于辨认主要作用；数据变化、通知协议与使用资格各有明确归属。具体算法由上游数学定义与动作组合表达，词表只规定基本作用，减少实例自由造词和重复解释。较高层粒度保留关键设计，也便于跨平台对应。[原始统计](data/verb_census.json)覆盖 193 份有稿 YAML、141 种动作名，为这次归并提供了语料依据。覆盖目标是常见流水线动作的组合，而非逐条复刻完整指令集。

**`info.using_verb`。** 每个基础动词下直接登记本 role 使用的 noun；每项只含必填的 `execution_mode` 与 `describe`。定义位置、索引与执行语义统一采用下文 [Noun 与执行模式](#noun-与执行模式)，不再使用 role／verb 级模式、继承或空条目。

**跨平台依据。** [NVIDIA PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) 是虚拟 ISA，提供矩阵、访存与同步指令族；AMD 的 [CDNA MFMA](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html) 与 [RDNA3 WMMA](https://gpuopen.com/download/rdna3-shader-instruction-set-architecture-feb-2023.pdf) 提供矩阵计算对应。`matmul` 的含义来自上游数学，不绑定 NVIDIA 指令的线程组织、布局或完成方式。[昇腾 Ascend C](https://gitcode.com/cann/asc-devkit/tree/82b89a15f26329511fc7fbc807d5132a1f9b4ba7/docs/zh) 的 `Mmad`、`DataCopy`、`SetFlag/WaitFlag`、`AllocTensor/FreeTensor` 分别提供相关能力的 API 对应。平头哥 [V900](https://www.t-head.cn/product?id=12) 公开资料可确认张量计算、互联与软件栈能力，尚不足以逐项确认 PPU 的具体动作指令。跨平台统一上述语义，具体实现遵守各平台及型号的契约。

## Noun 与执行模式

**已确认：每个 role 在 `info.using_verb.<verb>.<noun>` 定义自己的动作族，每项仅有 `execution_mode`、`describe` 两个必填字段。** noun 键只写基名；实参写在 pipe 和 waiting。这里记录设计结论，正式规范、十个案例和检查器的迁移尚待实施。

### 目的、问题与取舍

八个 verb 固定动作的基本性质；同一 role 中求最大值、计算指数、求和都可以使用 `compute`，具体区别由 noun 和本条目的说明表达。目标是让 Agent 在一处找到这个动作处理什么、如何推进，既不扩充 `compute[sum]` 一类二级动词，也不留下没有解释的自由名称。

现有十例暴露了三个问题：

- **单个 verb 的说明和模式太粗。** GEMM Epilogue 的预取、Fragment 读取是异步，Scales 捕获是同步；归并成 `load` 后仍须保留这些差异。
- **noun 与 resource 不是包含关系。** FA4 的 S、P、Stats 共用存储；MLA 的同一个 S／P 又同时存在于 TMEM 与寄存器。Ticket、Turn 等还涉及调度或通知含义，不能仅用存储容量解释。
- **换成基础动词会暴露身份冲突。** GDN 多个求解、合并阶段原来都作用于 Ti(r)；不能仅替换动词后，让不同阶段生成同一个动作事件。

曾考虑集中 noun 表、对象分类、引用层和模式继承。最终选择每个 role 的局部声明：上游继续负责数学与工作定义，resource/view 继续负责存储，`describe` 直接说明本动作的具体使用方式。无需另建 `phase.noun`、`role.noun`、`ref` 或分类体系；模式逐项明确，避免读者追查父级或猜测默认值。

### 唯一结构

下面是登记与调用的格式示意；Scores、RowMax、Exp、RowSum 的数学关系及其工作范围已由上游确定：

```yaml
info:
  using_verb:
    compute:
      RowMax:
        execution_mode: serial
        describe: "Compute the row maximum of Scores(i)."
      Exp:
        execution_mode: serial
        describe: "Form the unrounded exponentials from Scores(i) and RowMax(i), following the softmax formula."
      RowSum:
        execution_mode: serial
        describe: "Sum the unrounded Exp(i) values along the key dimension."
pipe:
  - "for(i)":
      - "compute:RowMax(i)"
      - "compute:Exp(i)"
      - "compute:RowSum(i)"
```

- `using_verb` 的键只能来自八个基础动词；动词下面直接列 noun，不再套一层 `noun:`。只登记本 role 使用的动作族，每个 pipe 动作必须唯一匹配一个声明。
- 每个 noun 条目仅有两个字段，固定先写 `execution_mode`、再写 `describe`；二者均必填，不接受 `{}`。`describe` 为非空英文说明，必要时可多行，不硬限一句。
- `execution_mode` 只接受 `serial`、`async`、`async_ordered_completion`。role 与 verb 层均不设模式，也没有默认或继承；本条目的模式适用于它的全部实例。

### Noun 的范围、参数与身份

**“只定义一次”指每个 `(role, verb, noun)` 动作族一次。** noun 是这个作用域里的对象名称，不是全局数学值或物理分配的登记。不同 role、不同 verb 可以使用同名 noun；例如 `compute:P` 与 `store:P` 各有自己的模式和说明。同名只有在描述的对象、版本及范围一致时才指同一个东西，不能仅靠拼写认定。

声明键统一只写 `P`、`Fragment` 这样的基名。矩阵乘采用操作数表达时，整个 `A@B` 是一个键，使用处写 `matmul:A(i)@B(j)`；不为 A、B 再建登记层，`describe` 须交代两操作数及结果。

不把 `P(i)`、`A(n)@B(n)` 写成声明键，是为了避免另造形参绑定和模式匹配规则。代价也明确：声明本身不编码参数个数或索引相等条件，这些按下列规则检查：

- 同一 role、verb、noun 基名结构只允许一种固定参数个数；复合动作分别检查各操作数。`P` 与 `P(i)` 不在同一动作族中混用，不按参数数量或常量索引重载。
- 从 pipe 语法树核对全部调用，包括空域和未执行分支中的动作；waiting 的引用必须匹配相同动作族及参数个数。实参继续使用既有常量、移位、`last()` 和 flatten 记法。
- `A(i)@B(j)` 是否允许、是否要求 i=j，由实际数学与索引关系决定，不能从声明键推导。声明不建立循环域、不生成动作，也不改变索引作用域。
- 动作实例由 phase、worker 实例、成员、role、verb、noun 及实际参数共同确定，展开后必须唯一。重复动作要带足够的真实轮次、版本或区域索引；不能靠源码行、控制块位置或未声明的阶段区分同名事件。

### Describe 要说明什么

最少要让读者确定“对象是什么、本动作做什么”。上游已经明确的公式、精度、工作范围直接沿用，可用普通对象名和步骤名说明对应关系，不引入结构化引用等级，也不重抄整份算法。无法唯一确定的副本、来源目标、访问区域、原地更新版本、参与者和附带结果，则须在本条目说清楚。

例如，MLA 的 P 既可在 PackedRegisters，也可在 ScoreProbability；搬运描述须明确从哪一份到哪一份。FA4 的 P 经过 FP8 窄化，但 ell 仍使用未窄化的 E；描述不能只说“产生 P”而丢掉后续需要的 E。协议 noun 须说明通知的含义与参与范围，资源 noun 须定位实际使用区域或分配，不能把整个 TMEM 层级当成一次具体分配。

noun 的命名自由不能改变基础 verb 的性质或降低它的 done 保证。`describe` 也不能代替已命名动作之间的必要等待；需要独立引用的部分完成、使用权归还或通知边界，仍使用已有基础动作及 waiting 表达。

### 执行模式作用于哪条序列

每个动作自身都是先 issue、后 done，所有动作按 pipe 顺序 issue。模式的声明位置虽在 noun 条目，约束仍作用于**同一 phase、worker 实例、成员、role 的实际动作序列**：

| 模式 | 推进与完成保证 |
| --- | --- |
| `serial` | 自身 done 后，才能发起后续动作；不自动排空此前其他异步动作。 |
| `async` | 按顺序发起，后续动作不必等自身 done；不提供动作间的完成顺序。 |
| `async_ordered_completion` | 所有采用此模式的动作按发起顺序累积完成；后一个 done 覆盖该序列此前动作的 done。 |

累积完成**跨 noun、跨 verb**，跨循环与 worktile 接续；中间的 serial 不重置该序列，其他 async 动作不被纳入。不能为每个 noun 暗中建立一条独立完成队列。FA4 的 QK/PV 共用累积完成流，MoE 的 ScaleTM 搬运与 MMA 也有跨动词覆盖的实例。

只有原语确实提供上述共同保证时，才采用 `async_ordered_completion`。同一 role 若有互不覆盖的完成机制，使用 `async`，通过已有 waiting 表达需要的先后。累积完成本身不提供操作数就绪、别名覆盖安全或提前归还源缓冲的保证。

### 依赖、迁移与审计边界

采用 [Waiting 规则](../pipe_grammar/README.md#waiting-依赖规则)，按完整动作事件连接依赖。**取消仅凭 noun 同名自动推导 load／publish 到消费者的数据边。** 局部名字不具备全局对象身份，照搬旧推导可能连错对象；由模式或其他已明确依赖传递保证的重复前驱可以省略。

迁移必须处理语义，不能只更名和移动字段：

- 先找出旧规则隐式生成的数据边；新顺序仍不能保证的必要前驱须补入 waiting，不能随旧规则一起丢掉。
- 基础动词归并后的 noun 与实例身份必须重新核对。GDN 的 Ti 各阶段须用实际阶段、版本或区域区分，并选择适当的基础 verb。
- 旧 FMHA 的 qk/pv.done 仅表示入队，不能机械移成 `matmul` 的 done；矩阵动作应按真实异步推进和结果完成保证记录。源读取结束早于目标写完时，仍用独立 `release` 表达。
- QSA 的 load.done 在下一次 load.issue 后才发布则可以保留：done 是可依赖的完成保证，可以晚于最早物理完成时刻。

上述判断来自[十份 pipe 试写](../pipe_grammar/trial_roots/README.md)及其 baseline 的资源、契约核对。规则审计支持本方案定稿，不代表十例已经完成新格式迁移或数据流验证。

机械检查可覆盖字段、八词与模式枚举、参数个数、引用，以及工具能够展开或证明的事件身份与顺序。`describe` 中对象绑定、公式与精度、原语保证及依赖完整性仍需语义审阅；动态循环的检查预算耗尽只表示未验证完。
