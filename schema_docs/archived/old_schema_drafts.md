# 旧版 schema 草稿（原设计文档第 8 节）

> 本文件从原设计文档 `Kernel流水线设计old.md` 的第 8 节拆出，收录各区 YAML 表述的旧版草稿：字段划分、语法规则、原则与具体 case。正文按原文保留，只补充了本说明、目录、分组标题和一张草图的说明文字，并把原文导出时带入的不换行空格统一换成了普通空格。
>
> - 现行版本见 `schema_docs/1–6`；背景、动机、分区思路、通用原则与现状见仓库根目录的 `README.md`。本文件只读，不再维护。
> - 旧版的部分字段名在现行文档中已经调整，以现行文档为准。
> - 各节按写作先后排列，后写的章节会引用先写的章节：math reshape 的记号对齐 tile pipeline 的动作名，并引用 work schedule 的版本一、二、三；contract 与 realization 沿用 math reshape 的记号和案例。
> - tile pipeline 一节的 `render.py` 与同节的 YAML 样例没有对齐：脚本要求 lane 下的键为 `info / pipeline / depends`、`info` 下为 `mode / verbs / completion_order / registers_per_thread`，样例使用的是 `pipe / edge` 与 `order / registers`。直接用脚本处理该样例会报错 `TMA warp: unsupported keys ['edge', 'pipe']`。
> - 图片均为原文档的外链。

## 目录

| 章节 | 主要内容 | 现行文档 | 原文日期 |
| --- | --- | --- | --- |
| 早期 CUE template 示意 | 最早的 template 字段设想 | template 尚未编写 | — |
| tile pipeline 的前身：按角色分组的前驱表 | FA4 局部示例；发射与完成、buffer 复用、迭代距离三个语义问题 | `5.design.tile_pipeline.md` | 26/09/08 |
| tile pipeline的YAML语言表述 | lane 的 `info / pipe / edge` 写法；FA4 case；`render.py` 与 DAG 渲染 | `5.design.tile_pipeline.md` | 26/09/08–09 |
| work schedule的YAML语言表述 | `decomposition / mapping / schedule`；FA4 的三个调度变体 | `4.design.work_schedule.md` | 9/14 |
| math reshape的YAML语言表述 | `thesis / rewrite / ledger`；FA4 因果单 CTA、FA4 非因果双 CTA、GDN、MoE | `3.design.math_reshape.md` | 26/09/14 之后 |
| Contract algorithm的YAML语言表述 | `constants / inputs / outputs / compute / reference`；FA4、MoE、GDN | `1.contract.algorithm_requirement.md` | — |
| Contract hardware的YAML语言表述 | `device / arch / features / reference`；B200、RTX PRO 5000 Blackwell、RTX 4090 | `2.contract.hardware_specific.md` | — |
| realization的YAML语言表述 | `require / prefer / forbid`；FA4、MoE、GDN | `6.realization.md` | — |

## Template（早期示意）

### 早期 CUE template 示意

原文上下文：template 计划用 CUE 编写，选型理由见 `README.md` 第 5 节。下面的字段是最早期的设想，已经过时。

例如 Template 可以保持接近这种密度：

```plaintext
#PipelinePlan: {
    spec_version!: string

    workload!: {
        input_dtype!:  "fp16" | "bf16" | "fp8"
        accum_dtype!:  "fp32" | "fp16"
        causal!:       bool
        variable_len?: bool
    }

    macro_pipeline!: {
        tile!: {
            q?:  int
            kv?: int
            m?:  int
            n?:  int
            k?:  int
        }

        dispatch!: "static" | "persistent" | "clc" | "custom"
    }

    novel_idea?: {
        ...
    }
}
```

它明显比完整 JSON Schema 紧凑。

## Design 区

### tile pipeline 的前身：按角色分组的前驱表

![FA4 warp-spec 手绘草图：上半部分是 TMA、UMMA、softmax0、softmax1、correction 各角色的时间线，下半部分是对应的依赖图（原文附于 26/09/03 的记录之后）](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/1aa6afbb-4e8c-4717-ad09-0b6e7c9ea5c4.jpeg)

**26/09/08：**

思考什么样的语言可以承载“warp-spec”的异步流水线的信息，简洁干净、容易记录和理解

OSDI 2026：Optimal Software Pipelining and Warp Specialization for Tensor Core GPUs

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/9511888e-97a9-4325-a90f-a88d868d241a.png)

**这个做法虽然看起来能分析，但是完全没办法迅速的get核心，AI去理解也会很慢**

**按角色分组的前驱表**

核心只需要这条约定：

> `**操作: [前驱操作……]**`**，表示所有前驱完成后，该操作才允许开始。**

按角色把这些条目分组，已经很接近 Excel 风格了：

```plaintext
kernel: fa4

warp_spec:
  version: 1
  inputs: [m_prev, y_prev]

  ops:
    tma:
      q:  []
      k0: []
      k1: [k0]

    umma:
      qk0: [q, k0]

    softmax:
      load_s0:     [qk0]
      max_update0: [load_s0, m_prev]
      alpha_ready0: [max_update0]
      exp0:       [alpha_ready0]

    correction:
      rescale0: [alpha_ready0, y_prev]
```

这是**局部示例，不是某个 FA4 版本的完整还原**：省略了 V、PV、其他迭代和 buffer 复用关系；`alpha_ready0` 在这里包含 alpha 的计算及发布，表示 correction 已可消费它。

这个片段明确表达了：`k0` 完成后，`k1` 搬运和 `qk0` 计算可以具备重叠条件；`alpha_ready0` 之后，`exp0` 和 `rescale0` 分别向下推进，correction 不必等整个 softmax 结束。**没有依赖表示不被这张图禁止重叠，不表示硬件必定同时执行。**

这里最重要的是：**角色分组不产生任何隐含的顺序边。** `q`、`k0` 的发射顺序没有在这个片段中记录，不能从它们的文本排列顺序推断。

这不是一个现成命名的行业标准，而是一个很小的项目 schema。不过其核心“节点→前驱集合”就是常见图表示；Python 标准库 `graphlib.TopologicalSorter` 接受的正是这种关系，分组展开后可以直接用于拓扑处理。

**为什么我更倾向这种写法**

每个节点只声明一次，角色和依赖都在附近，Agent 改一条关系不需要调整坐标，也不需要重排整张表。根操作用 `[]`，窗口外的状态放 `inputs`，含义也清楚。

以后增加属性，可以把单个条目展开，而不改变其余节点：

```plaintext
exp0:
  after: [alpha_ready0]
  latency_cycles: 1024
```

这里的 `1024` 只是格式示例，不是 FA4 实测值。项目约定短写 `[alpha_ready0]` 等价于长写 `{after: [alpha_ready0]}` 即可。

**建议用** `**load_s0`****、`**alpha_ready0` **这样的稳定 ID，而不是纯数字编号。** 增删一步不应导致后面的编号全部变化。

**比语法更重要的三个语义问题**

1. 发射顺序，不能误写成完成顺序

这是 warp-spec 最容易画错的地方。异步指令可以先返回控制权，工作随后才完成；PTX 也明确区分发起与完成观测机制。

因此：

**“同一 warp 先发射 A，再发射 B”不等于“A 完成后，B 才能开始”。**

需要表达这种区别时，把关键操作拆成 `a_issue`、`a_done` 两个节点：后续发射依赖 `a_issue`，数据消费者依赖 `a_done`。不必把所有操作都拆开，只拆那些确实存在“发射后继续干别的事”的边界。

这也是为什么不建议把 `TMA: [Q, K0, K1]` 自动解释成一条完成依赖链。**这串列表只说明任务分配，还没有说清楚顺序的类型。**

2. 不仅要记录“数据 ready”，还要记录“buffer 可以再次写”

如果只记录 `load_k0 → qk0`，图会漏掉复用约束。对于双缓冲，同一个物理槽再次装载 `K2` 前，必须确认旧的 `K0` 已经不再被消费者使用。CUTLASS 的 producer acquire / consumer release 正是在表达这类关系。

所以需要类似：

`**k0_read_done → load_k2**`

这里应使用“真正允许复用”的事件，而不是随意替换成某个更早或更晚的操作。它会直接影响你分析出来的重叠空间。

初版可以把这些关系都放进前驱表；需要区分原因时，再给边增加 `data`、`order`、`reuse` 等类型。

3. 展开后的图是 DAG，折叠后的循环模板不一定是

把前四个迭代展开，使用 `k0、k1、k2、k3`，很适合核对启动阶段。

但长期要避免重复记录，可以使用编译器依赖图里的 **iteration distance**：例如一条 `u → v, distance: 2`，表示 `u[i] → v[i+2]`，不是同一迭代里的回边。Twill 使用的循环依赖模型就包含这种跨迭代信息

### tile pipeline的YAML语言表述

#### 1. 核心设计

**一个 lane 对应一个 warp 或 warp group，在同一处记录：它是什么角色、按什么顺序做事、每个动作在等谁。**

目标是用最少的文字表达 kernel 的流水线依赖，而不是描述执行代码或绘图坐标。这些是我们约定的语法规则，循环和依赖不是 YAML 自带的功能。

每个 lane 固定分成三块：

| 字段     | 内容                                                                                                                 |
| -------- | -------------------------------------------------------------------------------------------------------------------- |
| `info` | 角色属性：同步/异步模式`mode`、顺序约束 `order`、动作词表 `verbs`，以及可选的每线程寄存器数 `registers`。 |
| `pipe` | 动作序列：`prologue` 写开头，`steady` 写重复部分，`epilogue` 写收尾。                                       |
| `edge` | 当前 lane 的动作所需的额外前驱，分为`data / reuse / sync`。                                                  |

例如：

```plaintext
TMA warp:
  info:
    mode: async
    order: [issue]
    verbs: [load]

  pipe:
    prologue: [load:K(0), load:Q_high, load:Q_low]
    steady:   [load:V(i), load:K(i+1)]
    epilogue: [load:V(N-1)]

  edge:
    reuse:
      load:V(i+1): [mma:Q_low@K(i)]
      load:K(i+2): [mma:P_low(i)@V(i)]
```

#### 2. 语法规则

**动作命名：**`**动词:对象_说明(序号)`****。** 例如 `load:S_high(i)`、`compute:alpha_high(i)`；矩阵乘法写成 `mma:P_high(i)@V(i)`。动作名称同时作为引用 ID，展开后必须唯一。名称负责可读性，**不会自动推导数据依赖**。

**流水线只写“开头＋循环＋收尾”。** `prologue`、`epilogue` 各展开一次；没有内容就写 `[]`。`steady` 的 `i` 从 0 开始，只有整轮动作的索引都落在 `0..N-1` 内才展开。因此，含 `K(i+1)` 的循环自然少一轮，不必单独声明循环范围。

**列表顺序有意义，但发射顺序不等于完成等待。** 同步 lane 按动作完成顺序推进；异步 lane 的 `order: [issue]` 只约束发射先后，`[issue, done]` 则分别约束发射顺序和完成顺序，**不额外加入“前一个完成后才能发射下一个”**。这里的同步模式不妨碍不同 lane 并行。

**依赖统一写成** `**当前动作: [前驱动作……]`****，表示所有前驱均须就绪。** 左侧必须属于当前 lane 的 `pipe`，右侧可以来自其他 lane；同一动作命中多条规则时，前驱取并集。三类名称只是区分等待原因：

| 分类      | 等待什么                       |
| --------- | ------------------------------ |
| `data`  | 操作数或计算结果就绪。         |
| `reuse` | 旧内容使用结束，允许覆盖存储。 |
| `sync`  | 额外的同步协议放行。           |

含 `i` 的依赖也按索引窗口展开；越界实例整条跳过，窗口内引用不存在则报错，不能静默忽略。

**这样压缩的是重复描述，不是必要语义：** lane 内顺序只写一次，循环体只写一次，额外依赖放在等待方附近；high/low 属于不同 WG，就分别明确记录。

#### 3. 如何渲染成 DAG

```plaintext
YAML
  → Python 展开循环和依赖
  → 检查动作归属、重复 ID、缺失引用和有向环
  → 构建事件 DAG：异步动作拆成 issue / done
  → 输出 DOT：按 lane 分组，按依赖类型设置线型
  → Graphviz 的 dot 自动布局
  → SVG / PNG
```

**Python 负责语义，Graphviz 负责排版。** YAML 不记录节点位置；脚本使用统一的分组和样式规则，不需要逐节点摆放。

最终得到的是**带 lane 分组的依赖图，不是实际时间线**：节点间距不代表耗时，同一高度也不代表同时执行。

#### 4. 具体case

目前的，在Warp-spec这边的YAML效果：

```yaml
kernel:
  name: FA4
  arch: B200-SM100
  dtype: bf16
  head_dim: 128
  tile: [128, 128]
  q_stage: 2
  cta_group: 1
  causal: false
  kv_smem_slots: 3

warp_spec:
  version: 3
  parameters: {N: 4}

  # steady 和含 i 的 edge：i 从 0 起，索引须在 0..N-1；越界整轮/整边不展开。
  # 固定项越界、窗口内引用缺失、或 edge 左侧不属于本 lane，直接报错。
  # pipe 自带 lane 内顺序；edge 的左侧属于本 lane，右侧是它等待的前驱。
  # data=结果就绪；reuse=允许覆写；sync=额外协议放行。
  # 沿用操作级模型：P 整块发布，未展开 split-P 和最终输出阶段。
  lanes:
    TMA warp:
      info:
        mode: async
        order: [issue]
        verbs: [load]
      pipe:
        prologue: [load:K(0), load:Q_high, load:Q_low]
        steady:   [load:V(i), load:K(i+1)]
        epilogue: [load:V(N-1)]
      edge:
        # 此配置的 K/V 共用 3 个 SMEM 槽，覆盖前必须等旧数据读完。
        reuse:
          load:V(i+1): [mma:Q_low@K(i)]
          load:K(i+2): [mma:P_low(i)@V(i)]

    UMMA warp:
      info:
        mode: async
        order: [issue, done]
        verbs: [mma]
      pipe:
        prologue: [mma:Q_high@K(0), mma:Q_low@K(0)]
        steady:   [mma:P_high(i)@V(i), mma:Q_high@K(i+1),
                   mma:P_low(i)@V(i),  mma:Q_low@K(i+1)]
        epilogue: [mma:P_high(N-1)@V(N-1), mma:P_low(N-1)@V(N-1)]
      edge:
        data:
          mma:Q_high@K(i):         [load:Q_high, load:K(i)]
          mma:Q_low@K(i):          [load:Q_low, load:K(i)]
          mma:P_high(i)@V(i):      [store:P_high(i), load:V(i)]
          mma:P_low(i)@V(i):       [store:P_low(i), load:V(i)]
          # 后续轮的 PV 还要等历史 Y 完成 correction；首轮不需要。
          mma:P_high(i+1)@V(i+1):  [rescale:Y_high(i+1)]
          mma:P_low(i+1)@V(i+1):   [rescale:Y_low(i+1)]

    Softmax WG0:
      info:
        mode: sync
        order: [done]
        verbs: [load, rowmax, compute, exp2, store, rowsum]
      pipe:
        prologue: []
        steady:   [load:S_high(i), rowmax:S_high(i), compute:alpha_high(i),
                   exp2:S_high(i), store:P_high(i), rowsum:L_high(i)]
        epilogue: []
      edge:
        data:
          load:S_high(i): [mma:Q_high@K(i)]
        # stats 的放行约束，不是 rowsum 的数学输入。
        sync:
          rowsum:L_high(0):   [wait:alpha_high(0)]
          rowsum:L_high(i+1): [rescale:Y_low(i+1)]

    Softmax WG1:
      info:
        mode: sync
        order: [done]
        verbs: [load, rowmax, compute, exp2, store, rowsum]
      pipe:
        prologue: []
        steady:   [load:S_low(i), rowmax:S_low(i), compute:alpha_low(i),
                   exp2:S_low(i), store:P_low(i), rowsum:L_low(i)]
        epilogue: []
      edge:
        data:
          load:S_low(i): [mma:Q_low@K(i)]
        sync:
          rowsum:L_low(i):   [rescale:Y_high(i+1)]
          rowsum:L_low(N-1): [release:stats_low(N-1)]

    Correction WG:
      info:
        mode: sync
        order: [done]
        verbs: [wait, rescale, release]
      pipe:
        # 首轮只消费通知；这里的 alpha(0) 是初始化通知的抽象名称。
        prologue: [wait:alpha_high(0), wait:alpha_low(0)]
        steady:   [rescale:Y_high(i+1), rescale:Y_low(i+1)]
        epilogue: [release:stats_low(N-1)]
      edge:
        data:
          wait:alpha_high(0):  [compute:alpha_high(0)]
          wait:alpha_low(0):   [compute:alpha_low(0)]
          # rescale 包含检查、按需缩放、发布就绪；动作仍属于 Correction。
          rescale:Y_high(i+1): [compute:alpha_high(i+1), mma:P_high(i)@V(i)]
          rescale:Y_low(i+1):  [compute:alpha_low(i+1), mma:P_low(i)@V(i)]
```

**第一，同一个动作命中多条依赖规则时，取前驱的并集。** 例如 `mma:P_high(1)@V(1)` 同时需要 `store:P_high(1)`、`load:V(1)` 和 `rescale:Y_high(1)`，后面的规则不会覆盖前面的规则。

**第二，Correction 用** `**(i+1)`** **表达“从第 1 轮开始修正”。** 这样既不需要写循环范围，也不会凭空产生第 0 轮 rescale。Softmax 的全 `(i)` 列表则自然覆盖全部四轮。

**检查脚本（render.py）：**

```python
#!/usr/bin/env python3
"""Render a small warp-specialization YAML into a checked event DAG and DOT.

Requirements: Python >= 3.9, PyYAML, Graphviz executable (dot).
No model calls, network requests, kernel-specific graph edges or image editing.
"""
from __future__ import annotations
import argparse
import ast
import re
from dataclasses import dataclass
from graphlib import TopologicalSorter, CycleError
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import yaml


class UniqueLoader(yaml.SafeLoader):
    """Reject duplicate mapping keys rather than silently replacing entries."""


def unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


@dataclass(frozen=True)
class Event:
    id: str
    op: str
    lane: str
    phase: str


# Bound expansion; only N, i, integer constants and +/- are permitted.
LIMIT = 20000


def check_keys(value, allowed, where):
    if not isinstance(value, dict):
        raise ValueError(f'{where}: expected a mapping')
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f'{where}: unsupported keys {sorted(map(str, unknown))}')


def integer(expression, env):
    """Safe integer expressions: literals, parameter names, + and -. No eval."""
    text = str(expression)
    if len(text) > 128:
        raise ValueError('Integer expression is too long')
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name) and type(env.get(node.id)) is int:
            return env[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            a, b = visit(node.left), visit(node.right)
            return a + b if isinstance(node.op, ast.Add) else a - b
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            a = visit(node.operand)
            return a if isinstance(node.op, ast.UAdd) else -a
        raise ValueError(f'Unsupported integer expression: {text!r}')
    try:
        return visit(ast.parse(text.strip(), mode='eval').body)
    except (SyntaxError, RecursionError) as exc:
        raise ValueError(f'Invalid integer expression: {text!r}') from exc


# All parenthesized numbers are iteration indices, not physical buffer slots.
# N is global; i is local to each complete steady block or dependency rule.
INDEX = re.compile(r'\(([^()]*)\)')
TEMPLATE = re.compile(
    r'^[a-z][a-z0-9]*:[A-Za-z_][A-Za-z_0-9]*(?:\([^()]+\))?'
    r'(?:@[A-Za-z_][A-Za-z_0-9]*(?:\([^()]+\))?)*$'
)


def inspect_operation(text, params):
    """Validate syntax before bounds filtering, including inactive boundary rules."""
    if not isinstance(text, str) or not TEMPLATE.fullmatch(text):
        raise ValueError(f'Expected verb:noun(index), with no space after colon: {text!r}')
    uses_i = False
    for expression in INDEX.findall(text):
        integer(expression, dict(params, i=0))
        tree = ast.parse(expression.strip(), mode='eval')
        uses_i |= any(isinstance(n, ast.Name) and n.id == 'i' for n in ast.walk(tree))
    return INDEX.sub('()', text), uses_i


def instantiate(text, env):
    """None means index out of domain, never an unknown operation or syntax error."""
    values = [integer(expression, env) for expression in INDEX.findall(text)]
    if any(not 0 <= value < env['N'] for value in values):
        return None
    iterator = iter(values)
    return INDEX.sub(lambda _: f'({next(iterator)})', text)


def compile_templates(document, overrides=None, trace=None):
    """Version 3 lane-local syntax -> the unchanged concrete v1 event graph."""
    if not isinstance(document, dict) or 'warp_spec' not in document:
        raise ValueError('Expected kernel and warp_spec mappings')
    spec = document['warp_spec']
    check_keys(spec, {'version', 'parameters', 'lanes'}, 'warp_spec')
    if spec.get('version') != 3:
        raise ValueError('Expected warp_spec.version: 3')
    check_keys(spec.get('parameters'), {'N'}, 'parameters')
    params = dict(spec['parameters'])
    for name, value in (overrides or {}).items():
        if name not in params:
            raise ValueError(f'Unknown parameter override: {name}')
        params[name] = value
    if type(params.get('N')) is not int or not 1 <= params['N'] <= LIMIT:
        raise ValueError(f'N must be an integer in 1..{LIMIT}')
    lanes = spec.get('lanes')
    check_keys(lanes, lanes if isinstance(lanes, dict) else (), 'lanes')
    if not lanes:
        raise ValueError('lanes must be nonempty')
    flat = {'version': 1, 'lanes': {}, 'async_lanes': [], 'completion_order': [],
            'after': {}, 'reuse_after': {}, 'sync_after': {}}
    report = {'index_domain': [0, params['N'] - 1], 'lanes': {}, 'rules': []}
    families, owners = {}, {}
    for lane, definition in lanes.items():
        if not isinstance(lane, str) or not lane:
            raise ValueError('Lane names must be nonempty strings')
        check_keys(definition, {'info', 'pipeline', 'depends'}, lane)
        info = definition.get('info', {})
        check_keys(info, {'mode', 'verbs', 'completion_order', 'registers_per_thread'}, f'{lane}.info')
        mode, verbs = info.get('mode', 'sync'), info.get('verbs')
        if mode not in ('sync', 'async'):
            raise ValueError(f'{lane}: mode must be sync or async')
        if not isinstance(verbs, list) or not verbs or not all(
            isinstance(v, str) and re.fullmatch(r'[a-z][a-z0-9]*', v) for v in verbs
        ) or len(verbs) != len(set(verbs)):
            raise ValueError(f'{lane}: provide distinct verbs')
        ordered = info.get('completion_order', False)
        if type(ordered) is not bool or (ordered and mode != 'async'):
            raise ValueError(f'{lane}: completion_order must be boolean and requires async mode')
        registers = info.get('registers_per_thread')
        if registers is not None and (type(registers) is not int or registers <= 0):
            raise ValueError(f'{lane}: registers_per_thread must be a positive integer or null')
        pipeline = definition.get('pipeline')
        check_keys(pipeline, {'prologue', 'steady', 'epilogue'}, f'{lane}.pipeline')
        families[lane] = set()
        parts = {}
        for part in ('prologue', 'steady', 'epilogue'):
            items = pipeline.get(part)
            items = [] if items is None else items
            if not isinstance(items, list):
                raise ValueError(f'{lane}.{part}: expected a list or an empty value')
            uses_i = False
            for text in items:
                family, symbolic = inspect_operation(text, params)
                families[lane].add(family)
                uses_i |= symbolic
                if text.split(':', 1)[0] not in verbs:
                    raise ValueError(f'{lane}: undeclared verb in {text!r}')
            if part != 'steady' and uses_i:
                raise ValueError(f'{lane}.{part}: i is only allowed in steady')
            if part == 'steady' and items and not uses_i:
                raise ValueError(f'{lane}.steady: a nonempty steady must contain i')
            parts[part] = items
        sequence, active = [], []
        for part, items in parts.items():
            contexts = (dict(params, i=i) for i in range(params['N'])) if part == 'steady' and items else [params]
            for env in contexts:
                block = [instantiate(text, env) for text in items]
                if any(op is None for op in block):
                    if part != 'steady':
                        raise ValueError(f'{lane}.{part}: index outside 0..N-1')
                    continue  # Skip the entire boundary block, never individual operations.
                if part == 'steady' and items:
                    active.append(env['i'])
                for op in block:
                    if op in owners:
                        raise ValueError(f'Duplicate operation ID: {op}')
                    owners[op] = lane
                    sequence.append(op)
                    if len(owners) > LIMIT:
                        raise ValueError('Expanded operation limit exceeded')
        if not sequence:
            raise ValueError(f'{lane}: the expanded pipeline is empty')
        flat['lanes'][lane] = sequence
        report['lanes'][lane] = {'steady_i': active, 'operations': len(sequence)}
        if mode == 'async':
            flat['async_lanes'].append(lane)
        if ordered:
            flat['completion_order'].append(lane)

    all_families = set().union(*families.values())
    for lane, definition in lanes.items():
        groups = definition.get('depends') or {}
        check_keys(groups, {'data', 'reuse', 'sync'}, f'{lane}.depends')
        for kind, rules in groups.items():
            rules = {} if rules is None else rules
            check_keys(rules, rules if isinstance(rules, dict) else (), f'{lane}.depends.{kind}')
            section = {'data': 'after', 'reuse': 'reuse_after', 'sync': 'sync_after'}[kind]
            for target, sources in rules.items():
                if not isinstance(sources, list) or not sources:
                    raise ValueError(f'{target}: predecessors must be a nonempty list')
                target_family, uses_i = inspect_operation(target, params)
                if target_family not in families[lane]:
                    raise ValueError(f'{lane}: target not declared in this pipeline: {target}')
                for source in sources:
                    family, symbolic = inspect_operation(source, params)
                    uses_i |= symbolic
                    if family not in all_families:
                        raise ValueError(f'{lane}: unknown predecessor family: {source}')
                active, skipped = [], []
                for i in range(params['N']) if uses_i else [None]:
                    env = dict(params, i=i) if i is not None else params
                    block = [instantiate(text, env) for text in [target, *sources]]
                    if any(op is None for op in block):
                        if not uses_i:
                            raise ValueError(f'{lane}: fixed dependency index outside 0..N-1: {target}')
                        skipped.append(i)
                        continue
                    dst, *preds = block
                    if owners.get(dst) != lane:
                        raise ValueError(f'{lane}: dependency target not in this pipeline: {dst}')
                    for src in preds:
                        if src not in owners:
                            raise ValueError(f'{lane}.{dst}: unknown predecessor: {src}')
                    flat[section].setdefault(dst, []).extend(preds)
                    active.append(i)
                report['rules'].append({'lane': lane, 'kind': kind, 'target': target,
                                        'applied_i': active, 'boundary_skipped_i': skipped})
    if trace is not None:
        trace.update(report)
    return {'kernel': document.get('kernel', {}), 'warp_spec': flat}, params


def expand(document):
    spec = document['warp_spec']
    if spec.get('version') != 1:
        raise ValueError('Expected warp_spec.version: 1')
    lanes = spec['lanes']
    if not isinstance(lanes, dict) or not lanes:
        raise ValueError('lanes must be a nonempty mapping')
    async_lanes = set(spec.get('async_lanes', []))
    ordered = set(spec.get('completion_order', []))
    if not async_lanes <= set(lanes) or not ordered <= async_lanes:
        raise ValueError('Unknown async lane, or completion_order is not an async lane')
    ops, events, entry, done = {}, {}, {}, {}
    for lane, sequence in lanes.items():
        if not isinstance(sequence, list) or not sequence:
            raise ValueError(f'{lane}: expected a nonempty operation list')
        for op in sequence:
            if not isinstance(op, str) or not op:
                raise ValueError(f'{lane}: operation IDs must be nonempty strings')
            if op in ops:
                raise ValueError(f'Duplicate operation ID: {op}')
            ops[op] = lane
            phases = ('issue', 'done') if lane in async_lanes else ('done',)
            for phase in phases:
                eid = f'{op}::{phase}'
                events[eid] = Event(eid, op, lane, phase)
            entry[op] = f'{op}::{phases[0]}'
            done[op] = f'{op}::done'

    edges = set()
    def add(src, dst, kind):
        if src == dst:
            raise ValueError(f'Self-dependency: {src}')
        edges.add((src, dst, kind))

    for lane, sequence in lanes.items():
        for op in sequence:
            if lane in async_lanes:
                add(entry[op], done[op], 'async')
        for a, b in zip(sequence, sequence[1:]):
            add(entry[a] if lane in async_lanes else done[a], entry[b], 'program')
            if lane in ordered:
                add(done[a], done[b], 'completion')

    for section, kind in [('after', 'data'), ('reuse_after', 'reuse'), ('sync_after', 'sync')]:
        values = spec.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f'{section} must be a mapping')
        for op, predecessors in values.items():
            if op not in ops:
                raise ValueError(f'{section}: unknown operation {op}')
            if not isinstance(predecessors, list):
                raise ValueError(f'{section}.{op}: expected a list')
            for pred in predecessors:
                if pred not in ops:
                    raise ValueError(f'{section}.{op}: unknown predecessor {pred}')
                add(done[pred], entry[op], kind)

    predecessors = {eid: set() for eid in events}
    for a, b, _ in sorted(edges):
        predecessors[b].add(a)
    try:
        order = list(TopologicalSorter(predecessors).static_order())
    except CycleError as exc:
        raise ValueError(f'Cycle detected: {exc.args[1]}') from exc
    return lanes, ops, events, sorted(edges), predecessors, order


def q(s):
    return json.dumps(str(s), ensure_ascii=False)


# Appearance belongs to the renderer, not to kernel metadata.
STYLES = {
    'async': ('#718096', 'solid', 1.0),
    'program': ('#8291A1', 'dashed', 1.0),
    'completion': ('#A0AEC0', 'dotted', 1.0),
    'data': ('#244866', 'solid', 1.3),
    'reuse': ('#A16112', 'dashed', 1.2),
    'sync': ('#7752A3', 'dashed', 1.2),
}
FILLS = ['#E7F0F7', '#E5F0EB', '#FAEBDD', '#F7E4EA', '#ECE6F6']


def dot_text(document, lanes, events, edges, params):
    k = document['kernel']
    title = " | ".join(str(k[x]) for x in ("name", "arch", "dtype") if x in k)
    if params:
        title += " | " + ", ".join(f"{key}={value}" for key, value in params.items())
    out = ['digraph WarpDAG {',
           f'graph [rankdir=TB, compound=true, newrank=true, bgcolor="white", pad=0.25, nodesep=0.18, ranksep=0.24, label={q(title + chr(10) + 'Operation-level event DAG | NO TIME SCALE' + chr(10) + 'Solid: data / async completion   Dashed gray: program order' + chr(10) + 'Dotted: completion order   Orange: reuse   Purple: synchronization')}, labelloc=t, fontname="DejaVu Sans", fontsize=14];',
           'node [shape=box, style="rounded,filled", fontname="DejaVu Sans", fontsize=10, margin="0.07,0.04", color="#BAC6D1"];',
           'edge [arrowsize=0.55];']
    for idx, lane in enumerate(lanes):
        out.append(f'subgraph cluster_{idx} {{ label={q(lane)}; fontname="DejaVu Sans"; color="#D9E0E8";')
        for event in events.values():
            if event.lane == lane:
                label = event.op + ('\nISSUE' if event.phase == 'issue' else '\nREADY' if f'{event.op}::issue' in events else '')
                out.append(f'{q(event.id)} [label={q(label)}, fillcolor={q(FILLS[idx % len(FILLS)])}];')
        out.append('}')
    for a,b,kind in edges:
        color,style,width=STYLES[kind]
        out.append(f'{q(a)} -> {q(b)} [color={q(color)}, style={q(style)}, penwidth={width}, tooltip={q(kind)}];')
    out.append('}')
    return '\n'.join(out)+'\n'


def run_graphviz(dot_path, base):
    engine = 'dot'
    exe=shutil.which(engine)
    if not exe:
        raise RuntimeError(f'{engine} not found; install Graphviz and add it to PATH')
    outputs=[]
    for ext in ('svg','png'):
        target=Path(str(base)+'.'+ext)
        args=[exe]
        args += [f'-T{ext}', str(dot_path), '-o',str(target)]
        proc=subprocess.run(args, capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(proc.stderr or f'{engine} failed')
        if proc.stderr.strip(): print(proc.stderr.strip(),file=sys.stderr)
        outputs.append(target)
    return outputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('yaml_file',type=Path)
    parser.add_argument('--out',type=Path,default=Path('out'))
    parser.add_argument('--set', action='append', default=[], metavar='NAME=INT')
    args=parser.parse_args()
    try:
        raw=args.yaml_file.read_bytes()
        document=yaml.load(raw,Loader=UniqueLoader)
        overrides = {}
        for item in args.set:
            name, value = item.split('=', 1)
            overrides[name] = int(value)
        trace = {}
        concrete, params = compile_templates(document, overrides, trace)
        lanes,ops,events,edges,pred,order=expand(concrete)
        args.out.mkdir(parents=True,exist_ok=True)
        dot=args.out/'dag.dot'
        dot.write_text(dot_text(concrete,lanes,events,edges,params),encoding='utf-8')
        run_graphviz(dot,args.out/'dag')
        (args.out/'inference.json').write_text(json.dumps(trace, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
        (args.out/'expanded.yaml').write_text(yaml.safe_dump(concrete, sort_keys=False, allow_unicode=True), encoding='utf-8')
        expanded={'events':[vars(e) for e in events.values()],
                  'edges':[{'from':a,'to':b,'kind':kind} for a,b,kind in edges]}
        (args.out/'events.json').write_text(json.dumps(expanded,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        report={'operations':len(ops),'events':len(events),'edges':len(edges),'acyclic':True,
                'parameters':params,
                'yaml_sha256':hashlib.sha256(raw).hexdigest(),
                'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'graphviz':subprocess.run(['dot','-V'],capture_output=True,text=True).stderr.strip(),
                'pyyaml':yaml.__version__,'python':sys.version.split()[0]}
        (args.out/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report,indent=2))
    except (KeyError,ValueError,TypeError,OSError,yaml.YAMLError,RuntimeError) as exc:
        parser.exit(1,f'Error: {exc}\n')


if __name__=='__main__':
    main()

```

流程仍然是：

```plaintext
lane-local YAML
    → 按统一边界规则展开
    → 检查动作归属、引用、重复 ID 和环
    → issue / done 事件 DAG
    → DOT
    → Graphviz SVG / PNG
```

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/09750d35-ba40-4cbf-ac92-75ff4d1561c8.png)

DAG在Graphviz的渲染效果，已经接近了预期的流水线样式：

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/3bbbbd5e-0f92-4c86-b200-c0bda1af4a42.png)

### work schedule的YAML语言表述

总的来说，分成了三个部分：

1）decomposition：任务在数学/算法层面的分治。通过partition描述数学对象的工作划分，通过worktile描述每个任务的具体信息，通过target描述每个任务的预期输出和格式

2）mapping：任务在编程对象/物理硬件的映射。通过worker/placement给出一个具体的worker的资源和形式，数量和范围

3）schedule：分治后的子任务（worktile），如何在具体的硬件对象（worker）上面进行任务调度。包括计算的先后顺序order，以及任务的分配方式assignment（动态/静态等等）

初版的草稿示例（有很多问题，表达不完备或者不合理）

```yaml
work_schedule:
  thesis: "双 Q 共用 KV，有限 CTA 按固定步长接续任务。"

  partition:
    batch: {extent: batch_count, per_task: 1}
    head:  {extent: head_count, per_task: 1}
    query: {extent: query_length, per_task: 256}
    kv:    {extent: kv_length, per_task: all}

  within_task:
    query: {block: 128, names: [Q_high, Q_low]}
    kv: {block: 128, order: descending, shared_by: [Q_high, Q_low]}
    tail: {query: clip, kv: mask}

  result:
    tensors: [O, LSE]
    region: task_query_rows
    kind: final

  placement:
    unit: CTA
    units_per_task: 1
    grid: ["min(sm_count, task_count)", 1, 1]

  assignment:
    fastest_first: [query, head, batch]
    direction: ascending
    first_index: cta_id
    index_step: cta_count
    index_limit: task_count
```

核心：workerassignment

**它应该描述一个 worker 获配的工作序列，以及这个序列如何被补充、何时处理完毕；而不只是“第一项是什么、下一项怎么算”。**

把原来的：

```plaintext
first: ...
next: ...
end: ...
```

改成：

```plaintext
initial: ...   # 初始获配的工作序列
refill: ...    # 是否从别处补充工作
exit: ...      # worker 的退出条件
```

一个 worker 已经获配了 `0、8、16、24`，那么从 8 走到 16，只是在消费自己的序列；不需要再用一个“领取机制”描述它。只有从共享工作池、待启动 worker 等来源取得额外工作，才属于 `refill`。

这个区分比 `persistent/static/dynamic` 更接近真正的设计决定。

#### 1. 核心思想

**Work schedule 是一张设计决策表，描述：计算如何分解、由谁执行、怎样分配。** 它不描述线程级实现，也不重复 tile pipeline 的动作、依赖和缓冲协议。

两个基本对象：

**Worktile**：可以独立分配的一份计算工作，由 `decomposition.partition` 定义。它可以产生完整结果，也可以产生待合并的部分结果。

**Worker**：承担 worktile 的协作执行单元，可以是单个 CTA、多个 CTA 或其他平台的执行组织。一个 worker 可以连续处理多个 worktile。

核心分离是：

> **Worktile 的排列，不等于 worker 的工作归属；预先获配的工作，不等于运行时新领取的工作。**

#### 2. 固定字段

| 区域              | 字段                      | 记录内容                                                                             |
| ----------------- | ------------------------- | ------------------------------------------------------------------------------------ |
| 顶层              | `thesis`                | 整体设计主张及目的。                                                                 |
| 三个主区          | `strategy`              | 一句简短英文，概括该区的策略，不重复逐项解释。                                       |
| `decomposition` | `partition`             | 真正产生独立 worktile 的划分维度。每项使用`size / tile`，必要时加 `range`。 |
|                   | `worktile.invariant`    | 在当前 worktile 的局部迭代中保持不变的输入块。                                     |
|                   | `worktile.iteration`    | 随局部迭代更换的输入块，记录其`shape / dtype / direction`。                    |
|                   | `target.single_tile`    | 一个 worktile 的产物及其`shape / dtype`。                                      |
|                   | `target.cross_tile`     | **可选。** 多个 worktile 合并后的产物。                                     |
|                   | `target.tile_reduction` | **可选。** 合并的维度、输入、输出和数学方法引用。                             |
| `mapping`       | `worker`                | 一个 worker 的组成：`unit / count`。                                           |
|                   | `placement`             | **可选。** 一个 worker 所需的物理资源范围，不登记没有设计意义的绑定信息。   |
|                   | `count`                 | 本次提交的 worker 总数，可以是固定值或必要的运行时数量关系。                       |
|                   | `members`               | **可选。** 协作成员之间的数学区域分工。                                       |
| `schedule`      | `worktile_order`        | Worktile 的逻辑排列规则。                                                           |
|                   | `worker_assignment`     | Worker 的`preassigned / refill / exit`。                                      |

`partition` 下的维度名、operand 名、产物名都是**当前算法的名称，不是模板专属字段**。名称与 Contract、数学定义和 tile pipeline 保持一致。

#### 3. 数据与范围的写法

**固定参数写实值，运行时规模写引用及适用范围。**

```plaintext
head:  {size: 8, tile: 1}
query: {size: QueryLength, range: [1, null], tile: 256}
```

`size` 是本次调用的实际规模；`range` 是允许的取值范围，`null` 表示本记录不额外限定该端的边界。已有 Contract 明确规定的范围不必重复填写。**数量取决于输入，不意味着划分策略未确定。**

`partition` 只列参与独立工作划分的维度。完整遍历 KV 时不写 `kv: all`；沿 KV 切成独立 worktile 时才加入 KV 维度。

Shape 使用**有序、带名称的维度列表**：

```plaintext
Q_high: {shape: [query: 128, head_dim: 128], dtype: bf16}
Q_low:  {shape: [query: 128, head_dim: 128], dtype: bf16, offset: [128, 0]}
```

`offset` 可选，按 shape 的维度顺序表示相对当前 worktile 的数学偏移，不是物理地址。名称与偏移的关系只需明确一次，不再增加 `tensor / views` 等重复描述。

所有 shape 默认以**整个 worktile**为作用域；成员局部范围由 `mapping.members` 给出。成员列表按编号排列，区域使用左闭右开区间。

`invariant` 不表示编译期常量；`iteration` 不重复登记可由运行时输入推导的迭代次数。合法迭代范围来自既有数学语义和输入定义，不能因省略次数而变成无约束访问。

#### 4. 排列与分配的语法

`worktile_order`：只定义宏观排列

```plaintext
worktile_order:
  - {by: [batch, head], group_size: 8, direction: ascending}
  - {by: query, direction: descending}
  - {by: [batch, head], direction: ascending}
```

列表从外层到内层读取：后续条目只在前面排序键相同的范围内继续排列，无需另写优先级编号。

`by` 可以是单个名称或有序名称组合。`[batch, head]` 按 batch 外层、head 内层展平；`group_size: 8` 表示**每组八个** `**(batch, head)`** **坐标对**，不是两个维度分别分八组。末组只包含实际存在的成员。

这里排列的是 worktile，**不包含 K/V 的内部遍历，也不隐含跨 worker 的启动顺序、完成顺序或同步依赖。**

`worker_assignment`：从 worker 视角描述工作归属

```plaintext
worker_assignment:
  preassigned: {start: worker_id, step: worker_count}
  refill: none
  exit: assigned_done
```

`**preassigned**` 描述预先确定的工作序列，而不只是第一项。`start + step` 表示有序 worktile 列表中的等步长序列，直到列表末尾；`start + count` 表示从起点开始的指定数量。也可以引用已有的预分配表。它不要求物化一个队列。

`**refill**` 描述运行时额外取得工作的机制。沿预分配序列前进不属于 refill；`none` 表示不补充工作。预取容量只在确实属于设计决定时记录。

`**exit**` 描述当前 worker 的退出条件。`assigned_done` 包含预分配及后续取得的全部工作；条件列表表示同时满足，例如：

```plaintext
exit: [refill_failed, assigned_done]
```

局部领取失败不等于全局完成。

`worker_id` 和 worktile 编号均从零开始；worktile 编号由 `worktile_order` 决定。`worker_count` 是提交数量，不是同时驻留数量。

#### 5. 取舍原则

**一个决定只写一次。** 最终算子语义属于 Contract；局部计算与同步属于 tile pipeline；本区只保留工作分解、执行组织和分配策略。

没有跨 tile 合并，就省略 `cross_tile / tile_reduction`；没有成员分工，就省略 `members`。不为格式对称填空壳，也不添加 `notes` 式的兜底区域。

`persistent / CLC` 等名称可以出现在 `strategy` 中帮助人类理解，但**有约束力的是下面的具体决策，不是策略名称**。

**精简的是重复描述与实现细节，必须留下的是：一份工作是什么、由谁承担、如何取得后续工作，以及何时结束。**

#### 6. 具体case

FA4在sm100的3个变体（FA开源库的版本）

这组示例选择：**每卡 8 个 MHA head，Q/K/V 的 head dimension 均为 128，BF16 输入和 O，FP32 LSE**。这里的 8 是这组 Plan 选择固定的参数，不是说所有 FA4 都必须固定为 8 个 head。

运行时量则保持为：

| 名称            | 含义                  | 本组示例的范围          |
| --------------- | --------------------- | ----------------------- |
| `BatchCount`  | 本次调用的 batch 数 | 正整数，不额外限定上界  |
| `QueryLength` | 本次调用的 Q 长度   | 正整数，不额外限定上界  |
| `KVLength`    | 本次调用的 KV 长度  | 正整数，不额外限定上界  |
| `SMCount`     | 目标设备的 SM 数    | 来自 hardware profile |

同一次调用内使用等长序列，但长度可以在调用之间变化；**不预设** `**QueryLength = KVLength`**。三份都不包含 paged KV、varlen、split-KV 或 local window。第一份非 causal，后两份 causal，具体因果对齐语义沿用 Contract。

**版本一：双 CTA 协作，预分配固定步长的 worktile 序列**

这份保留原来的双 CTA 分支：每个 worktile 覆盖 512 行 Q，一个 worker 内两个 CTA 协作；提交的 worker 数受 worktile 总量和 SM 数量限制。源码的 cluster 级任务计数与步长分配对应这一组织。

```yaml
work_schedule:
  thesis: "两个 CTA 协作处理双 Q 块，按固定步长接续 worktile。"

  decomposition:
    strategy: "Partition Q rows into 512-row worktiles."
    partition:
      batch: {size: BatchCount, range: [1, null], tile: 1}
      head:  {size: 8, tile: 1}
      query: {size: QueryLength, range: [1, null], tile: 512}
    worktile:
      invariant:
        Q_high: {shape: [query: 256, head_dim: 128], dtype: bf16, offset: [0, 0]}
        Q_low:  {shape: [query: 256, head_dim: 128], dtype: bf16, offset: [256, 0]}
      iteration:
        K: {shape: [kv: 128, head_dim: 128], dtype: bf16, direction: descending}
        V: {shape: [kv: 128, head_dim: 128], dtype: bf16, direction: descending}
    target:
      single_tile:
        O:   {shape: [query: 512, head_dim: 128], dtype: bf16}
        LSE: {shape: [query: 512], dtype: fp32}

  mapping:
    strategy: "Map each worktile to a two-CTA worker."
    worker: {unit: CTA, count: 2}
    placement: {unit: SM, count: 2}
    count: "min(worktile_count, floor(SMCount / 2))"
    members:
      - {query: [[0, 128], [256, 384]]}
      - {query: [[128, 256], [384, 512]]}

  schedule:
    strategy: "Preassign strided worktiles to persistent workers."
    worktile_order:
      - {by: batch, direction: ascending}
      - {by: head, direction: ascending}
      - {by: query, direction: ascending}
    worker_assignment:
      preassigned: {start: worker_id, step: worker_count}
      refill: none
      exit: assigned_done
```

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/099ee316-5940-4f89-a8fc-5797a7e464b1.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/ab9d8d70-17f4-40e2-9288-8936104f9cab.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/d348e2ab-3394-4d0d-9c67-0ecc01f09f7b.png)

**版本二：单 CTA 单 worktile，按 KV 工作集分组排序**

这一版有一个必须随运行时范围同步修改的地方：原来的 groupsize: 8 只是 KVLength 为 8192 时的结果，不是这个源码分支在所有输入上的固定参数。

源码按完整 K/V 工作集大小调整分组宽度，使用 50 MiB 调度预算，并向下取 2 的幂。因此，扩大为运行时 KV 范围后，这里需要保留一条输入相关规则。50 MiB 是调度预算，不是物理 L2 容量的声明。

```yaml
work_schedule:
  thesis: "按 KV 工作集分组，组内较长 worktile 优先编号。"

  decomposition:
    strategy: "Partition Q rows and iterate over causal-visible KV blocks."
    partition:
      batch: {size: BatchCount, range: [1, null], tile: 1}
      head:  {size: 8, tile: 1}
      query: {size: QueryLength, range: [1, null], tile: 256}
    worktile:
      invariant:
        Q_high: {shape: [query: 128, head_dim: 128], dtype: bf16, offset: [0, 0]}
        Q_low:  {shape: [query: 128, head_dim: 128], dtype: bf16, offset: [128, 0]}
      iteration:
        K: {shape: [kv: 128, head_dim: 128], dtype: bf16, direction: descending}
        V: {shape: [kv: 128, head_dim: 128], dtype: bf16, direction: descending}
    target:
      single_tile:
        O:   {shape: [query: 256, head_dim: 128], dtype: bf16}
        LSE: {shape: [query: 256], dtype: fp32}

  mapping:
    strategy: "Map each worktile to one CTA."
    worker: {unit: CTA, count: 1}
    placement: {unit: SM, count: 1}
    count: worktile_count

  schedule:
    strategy: "Group KV working sets and number longer worktiles first."
    worktile_order:
      - {by: [batch, head], group_size: "max(1, floor_pow2(102400 / KVLength))", direction: ascending}
      - {by: query, direction: descending}
      - {by: [batch, head], direction: ascending}
    worker_assignment:
      preassigned: {start: worker_id, count: 1}
      refill: none
      exit: assigned_done
```

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/573bfd67-38c0-4b9e-af8e-0fb9a09def1e.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/464cc4bb-144d-4bd4-823f-6309ca400778.png)

**版本三：单 CTA 预分配一项工作，运行时接管更多工作**

这一版的数学分解与第二版相同，取消工作集分组。每个 worker 预先对应一个 worktile，随后可以取消尚未启动的 worker 并接管其工作。源码的这个分支使用原始提交坐标对应的工作，并进行 Q 块反向映射，不走上一版的分组重排。

```yaml
work_schedule:
  thesis: "每个 worker 先分配一项工作，再提前接管未启动 worker 的工作。"

  decomposition:
    strategy: "Partition Q rows and iterate over causal-visible KV blocks."
    partition:
      batch: {size: BatchCount, range: [1, null], tile: 1}
      head:  {size: 8, tile: 1}
      query: {size: QueryLength, range: [1, null], tile: 256}
    worktile:
      invariant:
        Q_high: {shape: [query: 128, head_dim: 128], dtype: bf16, offset: [0, 0]}
        Q_low:  {shape: [query: 128, head_dim: 128], dtype: bf16, offset: [128, 0]}
      iteration:
        K: {shape: [kv: 128, head_dim: 128], dtype: bf16, direction: descending}
        V: {shape: [kv: 128, head_dim: 128], dtype: bf16, direction: descending}
    target:
      single_tile:
        O:   {shape: [query: 256, head_dim: 128], dtype: bf16}
        LSE: {shape: [query: 256], dtype: fp32}

  mapping:
    strategy: "Map one preassigned worktile to each CTA."
    worker: {unit: CTA, count: 1}
    placement: {unit: SM, count: 1}
    count: worktile_count

  schedule:
    strategy: "Prefetch work by cancelling unstarted workers."
    worktile_order:
      - {by: batch, direction: ascending}
      - {by: head, direction: ascending}
      - {by: query, direction: descending}
    worker_assignment:
      preassigned: {start: worker_id, count: 1}
      refill: {cancel: unstarted_worker, prefetch: 1}
      exit: [refill_failed, assigned_done]
```

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/87afd7c6-c30c-4bf6-b26a-52cfdab1a904.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/d2150a3e-1dce-43ef-a383-2df8e8635c77.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Mp7ld75BJ9MdpOBQ/img/57a350c7-b7b0-450e-b58d-4faf7cbb6633.png)

**以上展示的过程中，可以发现，其实work schedule也能画图（类似于tile pipeline），但是感觉会比较麻烦（看起来信息量似乎和纯文字的差不多），所以似乎没有必要专门在这里绘图了，YAML已经足够清楚和简洁**

### math reshape的YAML语言表述

总的来说，比较类似于tile pipeline，以多个lane并行为核心，这里也是以多个step为核心：

1）thesis简单交代核心的设计思路

2）rewrite内部由多个串行/并行的step1、2、3组成，每个step可以理解为是“一段关键的数学变换和优化”，有自己的name（称呼）、math（这里通过接近数学公式的语言来描述）、symbols（新引入的数学记号以及他们的含义）、arithmetic（这里的计算的关键数学精度/舍入/累加等等的选取）

3）ledger作为账本，通过不同的类别记录这个reshape对应的收支取舍，比如memory的traffic、计算、深度、存在的精度妥协、减少的计算/访存，强调这个方法带来的理论优势和开销

#### 1. 核心思想

**1.1 这一层做什么**

Plan 的 Design 部分有三层：mathreshape 修改、优化要实现的任务本身；workschedule 把任务分治成可以分配的 worktile；tilepipeline 确定每个 worktile 内部的执行细节。mathreshape 在最上游：在不违反 Contract 的前提下，决定算出哪些值、这些值怎样定义、以什么精度存在、存在多久。

只有这一层定下来，下游才知道有哪些量、哪些量跨迭代携带、哪些中间量要写回全局内存，才能开始划分工作。例如 FA4 的改写确定了“沿 kv 分块迭代、每行携带 (m, L, Y)”，workschedule 才能把 K、V 写成迭代输入，把查询行写成划分维度。

这一层在源码里最难看见。调度器、warp 角色和 barrier 在代码里都是显式的，数学改写却常常只是几行 FMA、一个阈值判断或一次类型转换：FA4 的滞后最大值，在源码里只是 `softmax.py` 中的一行 `if acc_scale_ >= -self.rescale_threshold`。把它单独写出来，专家和 Agent 才能直接讨论、修改、比较 kernel 的算法形态，而不必先读懂或重写代码。

**1.2 改写带来什么**

这一层是算法与硬件的协同设计：动机来自硬件，改写本身用数学表达。常见的收益如下：

| 收益                 | 做法                                       | 案例                                                              |
| -------------------- | ------------------------------------------ | ----------------------------------------------------------------- |
| 状态放得进片上       | 用固定大小的状态代替随输入长度增长的中间量 | FA4：每行 Lk 个分数压成 d + 2 个 fp32                      |
| 避免物化与重复搬运   | 中间量只以块的形式存在                     | FA4：S、P 不写回 HBM                                            |
| 串行工作变成矩阵乘   | 把逐步递推折叠成块内的矩阵运算             | GDN：L 次秩一更新变成 L/64 次块更新                            |
| 分散的工作变得稠密   | 改变表示，把零散的计算聚成能复用数据的块   | MoE：逐槽位 GEMV 变成按专家分组的 GEMM                         |
| 换用更便宜的运算单元 | 在精度允许的范围内使用近似或低精度         | FA4：部分 exp2 从 MUFU 挪到 FMA；MoE：MMA 直接吃 fp8 码值 |

收益总有代价：GDN 多出约 1.8 倍的 MAC 和五趟之间写回 HBM 的中间量，MoE 多出补齐行和每 token 约 250 KB 的中间张量。代价与收益一样记在 ledger 里。

**1.3 精度是改写的一部分**

改写几乎总会改变舍入点、累加次序或数据格式，一次错误的舍入就可能让整个 kernel 不可用。本规则区分三类，并给每一类固定的位置：

* **精确等价**：实数下与原定义相同的重排，例如在线 softmax、UT 变换、把路由权重移进 GEMM2。它们直接体现在计算本身。
* **有限精度差异**：舍入点与累加精度的变化，例如 P 舍入到 bf16、W 和 U 以 bf16 存储。它们由 math 中的显式 cast 表达。
* **主动近似**：函数求值的近似，例如多项式 exp2。它们写在 `arithmetic.approx`，并给出误差界。

这一层不能为了性能放宽 Contract 的精度要求。Contract 规定的量化（例如 MoE 的 q(·)）直接引用，本层只决定它作用在哪里。

**1.4 核心约定**

**math 只记录 kernel 实际执行的计算。** 不写证明、恒等式、不变量或解释。改写是否与原定义等价，由实现与 golden reference 的数值对比来检验；设计为什么值得，由 ledger 的定量增减来体现。这样 math 的每一行都能对照源码或实现逐行核对。

**1.5 边界：一个决定属于哪一层**

| 这个决定改变了什么               | 归属                          |
| ----------------------- | --------------------------- |
| 算出哪些值、如何定义、什么精度、存在多久    | mathreshape                 |
| 谁在何时计算哪一块               | workschedule / tilepipeline |
| 只替换实现，数值差异在 fp32 ulp 以内 | realization                 |

* FA4 的多项式 exp2 误差 8.8e-5，超出 fp32 ulp，属于本层；`rcp.approx`、`lg2.approx` 只有 ulp 级差异，属于 realization。
* FA4 的 kv 块长、MoE 的每块槽位数、访问顺序属于 workschedule；GDN 的块长 64 决定三角系统的规模与串行步数，属于本层。

#### 2. 字段划分

```yaml
math_reshape:
  thesis: ...          # 一两句：改写的主张与目的
  rewrite:             # 有序步骤，从 Contract 的输入写到 Contract 的输出
    - step: 1          # 互不依赖的兄弟步骤写 1(A)、1(B)
      name: ...        # 这一步做的决定
      math: |          # 实际计算：定义、递推、分情形、显式 cast
      symbols:         # 公式里看不出的信息
      arithmetic:      # 公式表达不了的精度例外
  ledger:              # 相对 baseline 的客观增减
    baseline: ...
    <resource>: [{item, change, by, when}]

```

**2.1 thesis**

一两句写出改写的主张与目的，也点出主要代价。读者只看 thesis，就应知道这份改写的核心想法；“提升性能”“优化访存”之类的泛化口号不写。

**2.2 rewrite、step 与 name**

rewrite 是有序的步骤列表，从 Contract 的输入一直写到 Contract 的输出。每个 step 是一个设计决定：

* `step` 是编号，供 ledger 回指。互不依赖的兄弟步骤写 1(A)、1(B)，例如 MoE 的输入量化与槽位分组。列表顺序表达数据依赖，不承诺执行必须串行。
* `name` 用一句话写出这一步做的决定，例如“参考最大值允许滞后 τ”，而不是“softmax 阶段”。

**2.3 math**

kernel 实际执行的计算：定义、递推、分情形、显式 cast。它是 step 的主体，也是这一层最需要逐行审阅的部分，写法见第 3 节。

**2.4 symbols**

只登记公式里看不出的信息。公式已经定义清楚、又没有下面这些属性的中间量，不登记。

* `life` 表示一个量存在多久，是本层最重要的属性：
  * 省略：只在本步的一次计算中存在，例如 FA4 的 E、P，GDN 的 A、R；
  * `carried`：跨迭代携带的状态，例如 FA4 的 m、L、Y，GDN 的 S。它们的总大小决定片上能否放下；
  * `stored`：写回全局内存、由后续 pass 读取的中间量，例如 GDN 的 G、T、W、U、S̄、R̄。它们决定额外的访存，也标出了 pass 的边界。
* `value`：本层固定的取值，例如 GDN 的块长 64。
* `shape`、`dtype`：公式推不出时才写。
* `meaning`：公式看不出含义时写一个短语；由下游决定的量在这里注明，例如“kv 块长（由 workschedule 决定）”。

用 `life` 而不另设一个“物化”区，是因为物化和状态压缩正是访存与片上容量的来源。把它们写在记号定义处，每个量的代价一眼可见。

**2.5 arithmetic**

只记公式表达不了的精度例外：

* `approx`：函数求值的主动近似，写方法与误差界，例如 FA4 的 3 次多项式 exp2；
* `accumulate`：非默认的累加方式，例如 GDN 求逆时子块合并用 tf32 输入，MoE 的 FP8 MMA 累加器位宽低于 fp32。

fp32 计算、fp32 累加与 RNE 舍入是默认值，不写；窄化由 math 中的 cast 表达。四个案例合计只有 5 条 arithmetic。

**2.6 ledger**

客观记录改写相对 baseline 多了什么、少了什么、换成了什么。

* `baseline`：一句话写清被比较的直算方式，例如 GDN 的“逐 token 递推，fp32 状态常驻片上”。资源类条目以它为基准，`error` 以精确计算 Contract 公式为基准。
* 资源键是固定词表，只写有变化的资源：
  | 键                                         | 计量                                    | 为什么单列                                                                                                       |
  | ------------------------------------------ | --------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
  | `matmul`、`vector`、`transcendental` | 张量核、通用 ALU、特殊函数单元上的计算 | 把工作从一种单元换到另一种，本身就是常见的改写：GDN 是 vector → matmul，FA4 是 transcendental → vector |
  | `traffic`                                | 片外访存字节                            | 决定带宽开销                                                                                                     |
  | `footprint`                              | 每个并行单元必须常驻片上的状态          | 决定能否放下，与 traffic 是两回事                                                                              |
  | `depth`                                  | 串行依赖链的长度                        | 决定可用的并行度，例如 GDN 从 L 缩短到 L/64                                                                 |
  | `passes`                                 | kernel 数或全量扫描趟数                | 包含 launch 开销与多趟中间量                                                                                   |
  | `comm`                                   | 跨 SM 或跨设备的通信                  | 本轮案例未用到                                                                                                   |
  | `error`                                  | 相对精确计算新增的误差来源              | 精度代价也是代价                                                                                                 |
* 条目有四个键：
  * `item`：对象；
  * `change`：用 `+`（新增）、`−`（消除）、`A → B`（转换），并且定量；公式用 Contract 与 rewrite 的记号写，具体形状下的数值附在括号里；
  * `by`：回指造成它的步骤；
  * `when`：只在特定输入区域成立时写，例如 MoE 的 `T·k·4 ≤ E`。

不叫“得失”，是因为同一个变化在不同场景下好坏不同：GDN 的 matmul 增加恰恰是收益。ledger 只记事实，判断留给评审与实验。要求定量，是因为定性描述无法核对，而且会无限膨胀。

#### 3. 语法规则

**3.1 记号**

| 记号                                                 | 含义                                                          |
| ---------------------------------------------------- | ------------------------------------------------------------- |
| `X(i)`、`X(c)`                                   | 迭代 i、块 c 上的实例；`X(−1)` 或 `X(0)` 为初值    |
| `X[·]`、`blk(j)`                                | 下标；`blk(j) = [j·B, (j+1)·B)` 为第 j 块的位置区间 |
| `·`、`⊙`、`ᵀ`                               | 矩阵乘或数乘、逐元素乘（向量按行广播）、转置                  |
| `Σ v`、`Σ(范围) 表达式`、`max v`          | 向量元素求和、按范围求和、向量元素取最大                      |
| `⟨a, b⟩`                                        | 一个量化组内的内积                                            |
| `cumsum`、`tril`、`‖`、`⌊·⌋`、`⌈·⌉` | 前缀和、保留 r ≥ s、拼接、取整                            |
| `bf16(·)`、`fp8(·)`、`q(·)`                 | 舍入到 bf16、舍入到 fp8（饱和）、Contract 规定的量化       |
| `a 若 条件；否则 b`                             | 分情形，也用于按输入区域分派                                  |

**3.2 迭代与递推**

迭代 i 上的值写成 X(i)，跨迭代依赖直接写成从 X(i−1) 到 X(i) 的递推，不另设循环语法；初值写成 X(−1) 或 X(0)。例如 FA4 的 `L(i) = α(i)·L(i−1) + Σ E(i)`，`L(−1) = 0`。块内位置用方括号，块的位置区间用 blk(j)；访问顺序记为 σ，由 workschedule 决定。

**3.3 分情形**

条件更新与按输入区域分派都用“若 … 否则 …”，各分支纵向对齐。FA4 的参考最大值：

```plaintext
m(i) = m(i−1)   若 m(i−1) > −∞ 且 c·(m̂(i) − m(i−1)) ≤ τ
       m̂(i)     否则

```

MoE 按批量大小选择两种分组方式：

```plaintext
B(b) = {b},    ε(b) = e(b)                                            若 T·k·4 ≤ E
B(b) = π[b·BM, (b+1)·BM),    ε(b) = e(π[b·BM])                        否则

```

**3.4 精度**

* 窄于 fp32 的值一律由显式 cast 定义：`bf16(·)`、`fp8(·)`，或 Contract 的 `q(·)`。没有 cast 的计算默认为 fp32 计算、fp32 累加；舍入为 RNE，`fp8(·)` 饱和到 ±448。
* 精度分叉必须分别命名。GDN 的修正量 R 有两个 bf16 版本：`R̄(c) = bf16(R(c))` 存给输出，`bf16(δ(c) ⊙ R(c))` 用于状态更新。两者都由 fp32 的 R 算出，不能互相代替；如果共用一个名字，读者就无法知道状态更新用的是哪一个。
* 只在最终存储时舍入的值，把 cast 直接写在输出上，例如 `O = bf16(Y(N−1) / L(N−1))`。

**3.5 记号来源与命名**

* Contract 的记号直接使用，例如 Q、K、V、scale、visible、q(·)，不重新定义。
* 局部视图用公式命名，例如 GDN 的 `X(c) = X[blk(c)]`，而不是用文字描述“第 c 块的输入”。
* 新记号与 tilepipeline 对齐：FA4 的 S、P、α、L、Y、N 对应 tile pipeline 中的 `S_high(i)`、`alpha_high(i)`、`L_high(i)`、`Y_high(i)` 与参数 N。希腊字母在下游可用拼写名。

**3.6 不写什么**

* 不用 LaTeX：原始的 LaTeX 在 YAML 里不会渲染，读者要先在脑子里编译一遍。
* 不用 torch 写法：view、transpose、dtype 参数带着实现的形状，会掩盖数学关系。
* 不写说明性的文字：math 中只保留“若 / 否则”，含义放进 `symbols.meaning`。

**3.7 示例**

一个完整的 step（取自 FA4）：

```yaml
- step: 3
  name: 分母累加 E，分子累加 bf16(E)
  math: |
    E(i) = exp2(c·S(i) − c·b(i)),    P(i) = bf16(E(i))
    L(i) = α(i)·L(i−1) + Σ E(i),       L(−1) = 0
    Y(i) = α(i)·Y(i−1) + P(i)·V(i),    Y(−1) = 0
  symbols:
    L(i): {dtype: fp32, life: carried, meaning: 分母}
    Y(i): {shape: [d: 128], dtype: fp32, life: carried, meaning: 分子}
  arithmetic:
    approx: 1/8 的 E 元素（每块每行 128 列中的 16 列）用 exp2(x) ≈ 2^⌊x⌋·p₃(x − ⌊x⌋)，p₃ 为 [0, 1) 上的 3 次 minimax 多项式，最大相对误差 8.8e-5

```

* 第一行把同一个指数结果分成两种精度：E 保持 fp32，P = bf16(E)。
* 第二、三行是跨块递推：分母 L 累加 E，分子 Y 累加 P·V，两者共用同一个缩放因子 α(i)。
* symbols 只登记两个 carried 状态；E、P、α 由公式定义、只在本步存在，不登记。
* arithmetic 只有一条：部分 exp2 用多项式近似，给出方法与误差界。

一条 ledger 记录：

```yaml
traffic:
  - {item: S 与 P, change: "− 16 · Lq · Lk B：fp32 的 S、P 各写一次、读一次", by: [1, 3]}

```

`item` 写对象，`change` 写带符号的定量变化，`by` 指向造成它的步骤 1（只访问 kv 块）与步骤 3（在线累加）。

#### 4. 原则

1. **Less is more。** 一条信息只写一次，写在唯一合适的位置；不为格式对称填空壳。判断标准是：删掉它，读者会不会读错或漏掉设计？不会就删。按这条标准，本规则删去了 given、恒等式、不变量与数值范围，也删去了 FA4 非因果分支里不会出现的空行分支。
2. **只写实际计算。** 不写恒等式、不变量、数值范围，也不写“次序任意”这类声明。证明改写正确不是这一层的任务；math 只需能逐行对照源码或实现核对。
3. **一步一个决定。** 一个 step 对应一个可以单独审阅、替换或否决的决定，在两种地方切开：
   * pass 边界，即产生 `stored` 之后；
   * 迭代结构变化处，即“每块一次”与“结束时一次”之间。

     精度分叉不单独成步，而是在同一步里用两个名字表达。例如 FA4 的步骤 2 可以单独撤销，令 τ = 0 就退回每块都重缩放的在线 softmax；GDN 的五步正好对应五个 kernel。
4. **精度写进公式。** 每个窄于 fp32 的值都由显式 cast 定义，读者从公式本身就能看出每个值走哪条精度路径。舍入点因此成为设计的一部分，Coder 不能悄悄移动。`arithmetic` 只记公式表达不了的例外。
5. **省略有边界。** 未改变的计算只引用、不重写，例如 SiLU、q(·)、mask；改写后的计算完整写出，一直写到 Contract 的输出。
   * 在 per-tensor FP8 的 MoE 里，GEMM 可以不展开，因为 scale 只是标量；
   * 在 block FP8 的 MoE 里，GEMM 必须展开，因为“scale 提到组外”这一改写就发生在 GEMM 内部。
6. **不替下游做决定。** 块大小、访问顺序由 workschedule 决定，只在 `symbols` 中注明。只有数学上必须固定的量才写 `value`：GDN 的块长 64 决定了三角系统的规模、串行步数与中间量的大小，所以由本层固定。
7. **ledger 客观、定量、单一基准。** 只写有变化的资源，不记流水重叠、指令选择与定性描述。基准必须写明：同一个 bf16 中间张量，相对“全融合”是新增，相对“逐步不融合”则不是变化，基准不明就会时算时不算。
8. **不设分类字段。** 常见的改写类别彼此不正交：FA4 的在线 softmax 同时是分解、算法与状态表示，MoE 的按专家分组同时是表示变换、分派与分解。强行归类只会误导，而这些类别都能从结构直接读出：
   | 类别           | 结构中的签名                          |
      | -------------- | ------------------------------------- |
      | 状态压缩       | `life: carried`                    |
      | 多阶段、物化   | `life: stored`                     |
      | 表示变换       | 引入索引类记号，数据张量本身不重排    |
      | 按输入区域分派 | math 中的分情形，配合`ledger.when` |
      | 重算代替存储   | 派生量在多步中现算，没有`stored`    |
      | 跳过计算       | 迭代域收缩                            |
      | 精度分叉       | 同一个量的两个 cast 版本            |
9. **按真实源码的单一分支填写。** 源码中每一处偏离精确 fp32 计算的地方，都必须出现在 math 的 cast 或 `arithmetic` 里；不同分支不拼成一份。
   * FA4 的 workschedule 版本二、三调度不同，数学完全相同，共用一份；版本一在块集合、空行分支与多项式比例三处不同，单独一份。
   * 一份独立写出的版本漏掉了多项式 exp2。这类近似藏在一次函数调用里，不逐处核对就看不见。

#### 5. 具体case

**1）FA4 · SM100 · 因果单 CTA**

* **来源**：Dao-AILab/flash-attention `ce088ab9`，`flash_attn/cute/flash_fwd_sm100.py`、`softmax.py`；因果、单 CTA、hdim 128、bf16，对应 workschedule 版本二与版本三。
* **内容**：4 步。只访问含可见 key 的 kv 块；参考最大值允许滞后 τ = 8；分母累加 fp32 的 E、分子累加 bf16(E)，其中 1/8 的 exp2 用多项式；结束时一次归一化得到 O 与 LSE。ledger：消除 S、P 的 HBM 往返，每行状态压到 d + 2 个 fp32，3 个 pass 合为 1 个。

```yaml
math_reshape:
  thesis: >-
    每行只带 (m, L, Y) 按 kv 块在线合并，S、P 从不整体存在；参考最大值允许落后至多 2^τ 倍，
    历史状态只在最大值跃升时重缩放；因果下整块不可见的 kv 块不参与计算。

  rewrite:
    - step: 1
      name: 只访问含可见 key 的 kv 块
      math: |
        blk(j) = [j·Bk, (j+1)·Bk)
        J = { j | ∃ q ∈ R, k ∈ blk(j)：visible(q, k) },    N = |J|
        S(i) = Q[q] · K[blk(σ(i))]ᵀ,    V(i) = V[blk(σ(i))],    S(i)[k] = −∞  若 ¬visible(q, k)
      symbols:
        R:  {meaning: 共用同一组 kv 块的查询行，q ∈ R（由 work_schedule 决定）}
        Bk: {meaning: kv 块长（由 work_schedule 决定）}
        σ:  {meaning: J 中块的访问次序（由 work_schedule 决定）}

    - step: 2
      name: 参考最大值允许滞后 τ
      math: |
        c = scale · log2(e),    τ = 8
        m(−1) = −∞,    m̂(i) = max(m(i−1), max S(i))
        m(i) = m(i−1)   若 m(i−1) > −∞ 且 c·(m̂(i) − m(i−1)) ≤ τ
               m̂(i)     否则
        b(i) = m(i)   若 m(i) > −∞；否则 0
        α(i) = exp2(c·(m(i−1) − b(i)))
      symbols:
        m(i): {dtype: fp32, life: carried, meaning: 参考最大值}

    - step: 3
      name: 分母累加 E，分子累加 bf16(E)
      math: |
        E(i) = exp2(c·S(i) − c·b(i)),    P(i) = bf16(E(i))
        L(i) = α(i)·L(i−1) + Σ E(i),       L(−1) = 0
        Y(i) = α(i)·Y(i−1) + P(i)·V(i),    Y(−1) = 0
      symbols:
        L(i): {dtype: fp32, life: carried, meaning: 分母}
        Y(i): {shape: [d: 128], dtype: fp32, life: carried, meaning: 分子}
      arithmetic:
        approx: 1/8 的 E 元素（每块每行 128 列中的 16 列）用 exp2(x) ≈ 2^⌊x⌋·p₃(x − ⌊x⌋)，p₃ 为 [0, 1) 上的 3 次 minimax 多项式，最大相对误差 8.8e-5

    - step: 4
      name: 结束时一次归一化
      math: |
        O = bf16(Y(N−1) / L(N−1)),    LSE = ln2 · (c·m(N−1) + log2 L(N−1))      若 L(N−1) > 0
        O = 0,    LSE = −∞                                                   若 L(N−1) = 0

  ledger:
    baseline: 按 Contract 直算：物化 fp32 的 S = Q·Kᵀ 与 P = softmax(S)，再算 P · V
    traffic:
      - {item: S 与 P, change: "− 16 · Lq · Lk B：fp32 的 S、P 各写一次、读一次", by: [1, 3]}
    matmul:
      - {item: 不可见块, change: "− 约一半的 4 · Lq · Lk · d FLOP（Lq = Lk 时；跨对角线的块仍整块计算）", by: [1]}
    vector:
      - {item: Y 重缩放, change: "+ 每行每次参考更新 d 次乘法；首块之后每行最多 c · (M − m(0)) / τ 次，M 为该行最终最大值", by: [2, 3]}
    transcendental:
      - {item: exp2, change: "访问到的每个分数一次 ex2，其中 1/8 改为约 6 条 FMA/ALU 指令（含 packed f32x2）", by: [3]}
    footprint:
      - {item: 每行状态, change: "Lk 个分数 → d + 2 个 fp32：Y 的一行、m、L", by: [2, 3]}
    depth:
      - {item: kv 方向, change: "对 Lk 的并行归约 → N 次串行合并；并行只来自行组、head 与 batch", by: [2, 3]}
    passes:
      - {item: kernel, change: "3 → 1", by: [1, 2, 3, 4]}
    error:
      - {item: P 舍入, change: "+ P = bf16(E) 进入 P · V（2^−8），而分母 L 累加未舍入的 E", by: [3]}
      - {item: exp2 近似, change: "+ 1/8 的 E 元素带 ≤ 8.8e-5 的相对误差", by: [3]}

```

**2）FA4 · SM100 · 非因果双 CTA**

* **来源**：同一 commit；非因果、双 CTA（要求 Lq > 256，且非 varlen、非 split-KV），对应 workschedule 版本一。
* **内容**：与 5.1 相同的四步，差别只有三处：访问全部块，只屏蔽尾块中 k ≥ Lk 的位置；没有空行分支；多项式 exp2 覆盖 3/16 的元素。

```yaml
math_reshape:
  thesis: >-
    每行只带 (m, L, Y) 按 kv 块在线合并，S、P 从不整体存在；参考最大值允许落后至多 2^τ 倍，
    历史状态只在最大值跃升时重缩放。

  rewrite:
    - step: 1
      name: kv 分块
      math: |
        blk(j) = [j·Bk, (j+1)·Bk),    N = ⌈Lk / Bk⌉
        S(i) = Q[q] · K[blk(σ(i))]ᵀ,    V(i) = V[blk(σ(i))],    S(i)[k] = −∞  若 k ≥ Lk
      symbols:
        Bk: {meaning: kv 块长（由 work_schedule 决定）}
        σ:  {meaning: 块的访问次序（由 work_schedule 决定）}

    - step: 2
      name: 参考最大值允许滞后 τ
      math: |
        c = scale · log2(e),    τ = 8
        m(−1) = −∞,    m̂(i) = max(m(i−1), max S(i))
        m(i) = m(i−1)   若 m(i−1) > −∞ 且 c·(m̂(i) − m(i−1)) ≤ τ
               m̂(i)     否则
        α(i) = exp2(c·(m(i−1) − m(i)))
      symbols:
        m(i): {dtype: fp32, life: carried, meaning: 参考最大值}

    - step: 3
      name: 分母累加 E，分子累加 bf16(E)
      math: |
        E(i) = exp2(c·S(i) − c·m(i)),    P(i) = bf16(E(i))
        L(i) = α(i)·L(i−1) + Σ E(i),       L(−1) = 0
        Y(i) = α(i)·Y(i−1) + P(i)·V(i),    Y(−1) = 0
      symbols:
        L(i): {dtype: fp32, life: carried, meaning: 分母}
        Y(i): {shape: [d: 128], dtype: fp32, life: carried, meaning: 分子}
      arithmetic:
        approx: 3/16 的 E 元素（每块每行 128 列中的 24 列）用 exp2(x) ≈ 2^⌊x⌋·p₃(x − ⌊x⌋)，p₃ 为 [0, 1) 上的 3 次 minimax 多项式，最大相对误差 8.8e-5

    - step: 4
      name: 结束时一次归一化
      math: |
        O = bf16(Y(N−1) / L(N−1)),    LSE = ln2 · (c·m(N−1) + log2 L(N−1))

  ledger:
    baseline: 按 Contract 直算：物化 fp32 的 S = Q·Kᵀ 与 P = softmax(S)，再算 P · V
    traffic:
      - {item: S 与 P, change: "− 16 · Lq · Lk B：fp32 的 S、P 各写一次、读一次", by: [1, 3]}
    vector:
      - {item: Y 重缩放, change: "+ 每行每次参考更新 d 次乘法；首块之后每行最多 c · (M − m(0)) / τ 次，M 为该行最终最大值", by: [2, 3]}
    transcendental:
      - {item: exp2, change: "每个分数一次 ex2，其中 3/16 改为约 6 条 FMA/ALU 指令（含 packed f32x2）", by: [3]}
    footprint:
      - {item: 每行状态, change: "Lk 个分数 → d + 2 个 fp32：Y 的一行、m、L", by: [2, 3]}
    depth:
      - {item: kv 方向, change: "对 Lk 的并行归约 → N 次串行合并；并行只来自行组、head 与 batch", by: [2, 3]}
    passes:
      - {item: kernel, change: "3 → 1", by: [1, 2, 3, 4]}
    error:
      - {item: P 舍入, change: "+ P = bf16(E) 进入 P · V（2^−8），而分母 L 累加未舍入的 E", by: [3]}
      - {item: exp2 近似, change: "+ 3/16 的 E 元素带 ≤ 8.8e-5 的相对误差", by: [3]}

```

**3）GDN · FLA Triton 分块前向**

* **来源**：fla-org/flash-linear-attention `954438d1`，`fla/ops/gated_delta_rule/` 与 `fla/ops/common/` 中的五个 kernel；chunk 64、dk = dv = 128、bf16 q/k/v；未安装 FlashQLA 时的默认路径。
* **内容**：5 步，每步一个 pass。块内重置的 log2 累积衰减；严格下三角系数与以 bf16 存储的 (I + A)⁻¹；同一个逆作用于两个右端项得到 W、U；块间状态推进，状态 S 保持 fp32，快照 S̄ 与修正量 R 的两个 bf16 版本分别命名；由快照与块内项得到输出。ledger：串行链从 L 缩短到 L/64，递推从 vector 转到 matmul；代价是 MAC 约 1.8 倍，访存约为最小值的 4.5 倍。

```yaml
math_reshape:
  thesis: >-
    把逐 token 的秩一 delta 更新改写成 64-token 块上的 UT 变换：块内修正由一次单位下三角求逆解出，
    串行链从 L 缩短到 L/64，主体计算全部变成矩阵乘；代价是约 1.8 倍的 MAC 和写回 HBM 的中间量。

  rewrite:
    - step: 1
      name: 块内重置的累积衰减
      math: |
        blk(c) = [c·C, (c+1)·C),    NT = ⌈L / C⌉,    X(c) = X[blk(c)]，X ∈ {g, β, Q, K, V}
        G(c) = log2(e) · cumsum(g(c))
        γ(c) = exp2(G(c)),    δ(c) = exp2(G(c)[C−1] − G(c)),    Γ(c)[r, s] = exp2(G(c)[r] − G(c)[s])  (r ≥ s)
      symbols:
        C:    {value: 64, meaning: 块长；尾块取有效长度}
        G(c): {shape: [C], dtype: fp32, life: stored}

    - step: 2
      name: 块内依赖写成单位下三角系统
      math: |
        A(c)[r, s] = β(c)[r] · Γ(c)[r, s] · (K(c)·K(c)ᵀ)[r, s]  (r > s)；0  (r ≤ s)
        T(c) = bf16((I + A(c))⁻¹)
      symbols:
        T(c): {shape: [C, C], life: stored}
      arithmetic:
        accumulate: (I + A)⁻¹ 由 16×16 对角块前代（fp32）与子块合并（tf32 输入、fp32 累加）求得

    - step: 3
      name: 同一个逆作用于两个右端项
      math: |
        W(c) = bf16(T(c) · bf16(β(c) ⊙ γ(c) ⊙ K(c)))
        U(c) = bf16(T(c) · bf16(β(c) ⊙ V(c)))
      symbols:
        W(c): {life: stored}
        U(c): {life: stored}

    - step: 4
      name: 块间只推进一次状态
      math: |
        S(0) = S_init,    S̄(c) = bf16(S(c))
        R(c) = U(c) − W(c) · S̄(c),    R̄(c) = bf16(R(c))
        S(c+1) = γ(c)[C−1] · S(c) + K(c)ᵀ · bf16(δ(c) ⊙ R(c)),    S_final = S(NT)
      symbols:
        S(c): {shape: [dk: 128, dv: 128], dtype: fp32, life: carried, meaning: 进入块 c 前的状态}
        S̄(c): {life: stored}
        R̄(c): {life: stored}

    - step: 5
      name: 输出由快照与块内项组成
      math: |
        B(c) = bf16(tril(Γ(c) ⊙ (Q(c) · K(c)ᵀ)))
        o[blk(c)] = bf16(scale · (γ(c) ⊙ (Q(c) · S̄(c)) + B(c) · R̄(c)))

  ledger:
    baseline: 按 Contract 逐 token 递推：fp32 状态常驻片上，每个 token 做一次状态衰减、两次矩阵-向量乘与一次秩一更新
    depth:
      - {item: 串行链, change: "L 次秩一更新 → L/C 次块更新；步骤 1、2、3、5 对所有块并行，步骤 4 只在 batch、head 与 S 的列块之间并行", by: [2, 3, 4]}
    vector:
      - {item: 逐 token 递推, change: "− 每 token 3 · dk · dv 次 MAC 与 dk · dv 次乘法（约 4.9 万 MAC + 1.6 万乘法）", by: [4]}
    matmul:
      - {item: 块化计算, change: "+ 每 token 3 · dk · dv + C · (2.625 · dk + 2 · dv) 次 MAC（约 8.7 万；不计子块合并）", by: [2, 3, 4, 5]}
    transcendental:
      - {item: 衰减因子, change: "每 token 1 次 exp → 约 107 次 exp2；Γ 在步骤 2、5 各重算一次，而不是存储 C × C 矩阵", by: [1, 2, 5]}
    traffic:
      - {item: 多趟中间量, change: "每 token 每 head 约 1.0 KB → 至少 4.6 KB：G、T、W、U、S̄、R̄ 的写回与重读，K 读 4 次", by: [1, 2, 3, 4, 5]}
    passes:
      - {item: kernel, change: "1 → 5，每个 step 一个 kernel", by: [1, 2, 3, 4, 5]}
    error:
      - {item: bf16 中间量, change: "+ 五个存储量 T、W、U、S̄、R̄ 与四个矩阵乘操作数（β ⊙ γ ⊙ K、β ⊙ V、δ ⊙ R、B）各舍入到 bf16（2^−8）；S 与 R 本身保持 fp32", by: [2, 3, 4, 5]}

```

**4）MoE · vLLM TritonExperts · FP8 block**

* **来源**：vllm-project/vllm `28282ffe`，`fused_moe/experts/triton_moe.py`、`fused_moe.py`；SM90、TP = 8、权重 128×128 块 scale、激活 1×128 动态量化，DeepSeek-V3 路由专家形状。
* **内容**：6 步。1(A) 在展开前量化 X，与 1(B) 并行：槽位按专家成块、只存索引，小批量时每槽位一块；GEMM1 把两个 scale 提到 128 组内积之外；SiLU·mul 之后直接量化，不经 bf16；GEMM2 把路由权重乘在 fp32 累加器上；最后 top-k 合并。ledger：权重读取从每槽位一次降到每块一次，MMA 直接吃 fp8；代价是补齐行、6 个 pass，以及每 token 约 250 KB 的中间张量。

```yaml
math_reshape:
  thesis: >-
    把每个 (token, 专家) 槽位上的两次 GEMV 改写成按专家分组的 GEMM：分组只存在于索引表，不产生重排副本；
    FP8 的 scale 按 128 组提到内积之外，MMA 直接吃 fp8 码值；路由权重借线性并入 GEMM2 的 fp32 累加器。

  rewrite:
    - step: 1(A)
      name: 输入在展开前量化
      math: |
        (Xq, sX) = q(X)
      symbols:
        Xq: {shape: [T, H], dtype: fp8e4m3, life: stored}
        sX: {shape: [T, H/128], dtype: fp32, life: stored}

    - step: 1(B)
      name: 槽位按专家成块，只存索引
      math: |
        p = t·k + j,    t(p) = ⌊p / k⌋,    e(p) = idx[t, j],    w(p) = w[t, j]
        B(b) = {b},    ε(b) = e(b)                                            若 T·k·4 ≤ E
        B(b) = π[b·BM, (b+1)·BM),    ε(b) = e(π[b·BM])                        否则
        π = pad(P(0)) ‖ … ‖ pad(P(E−1)),    P(e) = { p | e(p) = e }
      symbols:
        pad: {meaning: 以哨兵 T·k 补到 BM 的倍数}
        BM:  {meaning: 每块槽位数，由 work_schedule 按 T 选取}
        π:   {shape: [≤ T·k + E·(BM − 1)], dtype: int32, life: stored}
        ε:   {shape: [块数], dtype: int32, life: stored}

    - step: 2
      name: scale 提到 128 组内积之外
      math: |
        H1[p, n] = bf16( Σ(g < H/128) sX[t(p), g] · S13[e][⌊n/128⌋, g] · ⟨Xq[t(p), g], W13[e][n, g]⟩ ),    p ∈ B(b)，p < T·k，e = ε(b)，n < 2I
      symbols:
        H1: {shape: [T·k, 2I], life: stored, meaning: "[gate | up]，按槽位存放"}
      arithmetic:
        accumulate: ⟨·, ·⟩ 在 FP8 MMA 的累加器中求和，Hopper 上其位宽低于 fp32（约 14 位）

    - step: 3
      name: 激活后直接量化，不经 bf16
      math: |
        A[p, i] = SiLU(H1[p, i]) · H1[p, I + i],    i < I
        (Aq, sA) = q(A)
      symbols:
        A:  {shape: [T·k, I], dtype: fp32}
        Aq: {shape: [T·k, I], dtype: fp8e4m3, life: stored}
        sA: {shape: [T·k, I/128], dtype: fp32, life: stored}

    - step: 4
      name: 路由权重乘在 fp32 累加器上
      math: |
        Y3[p, n] = bf16( w(p) · Σ(g < I/128) sA[p, g] · S2[e][⌊n/128⌋, g] · ⟨Aq[p, g], W2[e][n, g]⟩ ),    e = ε(b)，n < H
      symbols:
        Y3: {shape: [T·k, H], life: stored}
      arithmetic:
        accumulate: 同步骤 2

    - step: 5
      name: top-k 合并
      math: |
        Y[t] = bf16( Σ(j < k) Y3[t·k + j] )

  ledger:
    baseline: 按 Contract 逐槽位直算：每个槽位在片上完成 GEMV → SiLU·mul → q → GEMV → 加权累加，专家权重按槽位各读一次
    traffic:
      - {item: 专家权重, change: "T·k·3·H·I B → Σ(e) ⌈n_e / BM⌉ · 3·H·I B，n_e 为专家 e 的槽位数（不计 L2 复用）", by: ["1(B)", 2, 4], when: "T·k·4 > E"}
      - {item: 中间张量, change: "+ 每 token 2·k·(2·2I + I + 2·H) B：H1、Aq、Y3 各写一次读一次；本形状约 250 KB，其中 Y3 占 92%，T = 4096 时约为权重流量的 1/3", by: [2, 3, 4, 5]}
    matmul:
      - {item: FP8 MMA, change: "反量化后的 bf16 MMA → 直接用 fp8 码值的 MMA（SM90 峰值 2 倍）", by: [2, 4]}
      - {item: 补齐行, change: "+ 每个 GEMM 至多 E·(BM − 1) 行空算", by: ["1(B)", 2, 4], when: "T·k·4 > E"}
      - {item: 空行, change: "+ 每槽位 BM − 1 行空算", by: ["1(B)", 2, 4], when: "T·k·4 ≤ E"}
    vector:
      - {item: 反量化, change: "每槽位 3·H·I 次逐元素反量化 → 3·H·I / 64 次 scale 乘法（每个输出元素每组 2 次）", by: [2, 4]}
    passes:
      - {item: kernel, change: "1 → 6：量化 X、分组、GEMM1、激活与量化、GEMM2、合并", by: ["1(A)", "1(B)", 2, 3, 4, 5], when: "T·k·4 > E"}
      - {item: kernel, change: "1 → 5：不做分组", by: ["1(A)", "1(B)", 2, 3, 4, 5], when: "T·k·4 ≤ E"}
    error:
      - {item: 中间舍入, change: "+ H1 与每个槽位的 Y3 各舍入一次到 bf16（2^−8）；A 由 fp32 直接量化，不增加舍入", by: [2, 3, 4]}

```

## Contract 区

### Contract algorithm的YAML语言表述

**Contract**区，包括algorithm requirement，以及 hardware specific，planner和coder都不可修改

纯数学要求，纯硬件规格

1）algorithm requirement：算法要求的计算I/O、从外向内作为黑盒抽象的算子定义。包括：math definition，也就是数学语义的公式表达、数学记号、特殊语义；输入输出和内部涉及的变量命名、意义状态；基于Pytorch等实现的golden reference的oracle（通过文件索引地址来记录，精简代码）；代表性的shape，输入的workload，目标优化的指标latency/throughput、精度的容忍范围等特殊要求和target；**不看 Design，能否仅凭这一段写出一个朴素 reference，而不需要猜掩码、量化、路由和状态语义**

从 FA4、MoE、GDN 三个已有 case 的调用边界与 reference 出发，先写出各自的算法契约，再归纳共同字段：`constants / inputs / outputs / compute / reference`。

`compute` 沿用 math reshape 中 `rewrite` 的步骤格式，但表达的是**黑盒必须满足的计算**，不是优化后的实现方法。

#### 1. 核心思想

**1.1 这一层做什么**

algorithm requirement 属于 Contract：从调用者的视角，定义算子接收什么、返回什么，以及输入与输出之间必须满足的数学关系。Planner 与 Coder 都不能自行修改。

判断是否写完整，不是看它有没有覆盖源码中的每个函数，而是：**不看 Design，能否仅凭这一段写出一个朴素 reference，而不需要猜掩码、量化、路由和状态语义？**

一个算法契约可以跨越多个源码函数，也可以只覆盖某个函数的一种配置。边界来自选定的计算任务，不直接照搬现有的 kernel 划分。

**1.2 格式怎么来的**

三个 case 分别暴露了不同的必要信息：

| Case | 不能省略的外部语义                                                            |
| ---- | ----------------------------------------------------------------------------- |
| FA4  | Q/KV 长度、因果对齐、无可见 key 时的结果、LSE 定义。                      |
| MoE  | 量化码值与 scale 的解释、给定路由的使用方式、输出是否只是权重分片的部分和。 |
| GDN  | 门控输入的含义、初始状态、更新与读出的先后关系、最终状态。                    |

这些差异不需要变成三套字段。固定数值放进 `constants`；对象含义与取值限制跟随 `inputs / outputs`；计算关系写进 `compute`；对照代码与参数映射放进 `reference`。

因此，不再设置顶层 `symbols` 或单独的 `domain`；不在 `reference` 外套一层 `check / oracle`；也不写没有新增约束的 `target: end_to_end_latency`。

**1.3 与 Design 的边界**

`algorithm.compute` 写“要求算什么”，`math_reshape.rewrite` 写“怎样改写这份计算”。

例如，因果可见性、MoE 的量化函数、GDN 的状态递推属于 Contract；在线 softmax、按专家分组、块内三角求解属于 Design。量化组长决定数值含义，不能当成普通的 GEMM tile size 自由调整。

`compute` 中出现一个中间量，只是在定义数学关系，**不要求实际分配、保存或物化这个量**。因此，本区不继承 `life / arithmetic / ledger`，也不记录 kernel 数量、存储位置或执行组织。

#### 2. 字段划分

| 字段          | 记录内容                                                                                     |
| ------------- | -------------------------------------------------------------------------------------------- |
| `constants` | 本份契约固定的数值，每项使用`value / meaning`。                                          |
| `inputs`    | 调用者提供的对象，每项使用`meaning / shape / dtype`，必要时加 `range / default`。 |
| `outputs`   | 调用者得到的对象，每项使用`meaning / shape / dtype`。                                  |
| `compute`   | 从输入到输出的计算定义，由`step / name / math / symbols` 组成。                     |
| `reference` | 对照代码的位置，以及契约对象与函数参数、返回值之间的映射。                                   |

**2.1 constants、inputs 与 outputs**

`meaning` 用一两个词或一个短语表达对象的角色，例如“隐层维度”“专家索引”“分片部分和”，不重复解释公式。

`constants` 只放固定值。函数在首次使用处定义，dtype 在对象上声明，不把函数说明、类型解释混进常量表。固定尺寸可以直接写在 shape 中，不必为了命名而重复登记。

输入、输出与常量的名称属于当前算法实例，不是模板新增的专属字段。

**2.2 compute**

每个 step 包含：

| 字段        | 内容                                                           |
| ----------- | -------------------------------------------------------------- |
| `step`    | 步骤编号；互不依赖的兄弟步骤写`1(A)`、`1(B)`。             |
| `name`    | 一句话说明这一步的计算关系，例如“下投影后按原路由权重求和”。 |
| `math`    | 定义、递推、分情形与显式 cast，是本步的主体。                 |
| `symbols` | 可选。只补充公式中看不出的局部记号含义。                       |

`symbols` 不重复登记已经由公式说明的中间量，也不填写存储生命周期。没有额外信息就省略。

**2.3 reference**

`reference` 固定使用三个字段：

* `source`：`仓库@commit:文件路径::函数名`，固定代码版本与入口。
* `inputs`：左侧为函数参数名，右侧为契约对象、转换表达式或固定参数。
* `outputs`：左侧为契约输出名，右侧为函数返回值或转换表达式；`result` 表示返回值。

这里的映射是调用说明，不是可执行的 YAML。未显式指定的参数采用固定版本的默认值；改变本例语义的非默认参数必须写出。

#### 3. 语法规则

**3.1 Shape 与范围**

Shape 使用有序、带名称的维度列表：

```yaml
X: {meaning: 输入特征, shape: [token: T, hidden: H], dtype: bf16}
```

固定规模写实值或引用 `constants`；运行时规模使用符号。同名维度在同一份 algorithm 内表示相同的大小，不同 case 的符号作用域独立。

`range` 只在需要限制输入取值时填写。`[0, 1]` 表示两端包含；含符号或开端点的区间写成字符串，例如 `"[0, E)"`。`default: 0` 用于张量时，表示缺省为给定 shape 的全零张量。

本节三个 case 均限定为正整数规模、有限数值输入；没有写出的运行时尺寸上界不额外限定。公式明确规定的 `−∞` 掩码与输出不受有限输入约定影响。

**3.2 步骤与依赖**

步骤按计算关系排列，从 Contract 的输入一直写到全部输出。列表顺序表达数学依赖，不承诺执行必须串行；`1(A) / 1(B)` 也不要求拆成两个 kernel。

递推直接写出初值与状态转移，例如：

```text
S(0) = S_init
S(t+1) = ...
```

编号不是执行调度。GDN 可以先定义完整的状态递推，再定义各位置如何读出；这不要求实现先保存全部状态、再启动输出计算。

**3.3 数学表达**

记号沿用 math reshape：`X[t]` 表示下标，`S(t)` 表示递推实例；`· / ⊙ / ᵀ / ‖` 分别表示矩阵乘或数乘、逐元素乘、转置、拼接；按范围求和写成 `Σ(范围) 表达式`。

公式可以省略相互独立的 batch/head 下标，但归约维度、迭代范围与边界分支必须明确。`exp` 使用自然底数，`logsumexp` 使用自然对数；FA4 的 softmax 与 logsumexp 均沿 key 维计算。

辅助函数在首次使用处定义，后续直接引用。`dq(q(z))` 表示把 `q` 返回的码值与 scale 传给 `dq`，不要求真的生成反量化副本。

`math` 不写 LaTeX 源码、torch 的视图操作或说明性段落；对象解释放在 `meaning`，分支使用“若 / 否则”。

**3.4 精度**

沿用已有约定：没有显式 cast 的计算默认使用 FP32 计算与累加，舍入为 RNE；`bf16(·)` 表示舍入到 BF16，`fp8(·)` 表示舍入到 E4M3FN 并饱和到 ±448。

`q(·)` 不等于 `fp8(·)`：前者还包含分组、scale 计算及数值编码规则，必须由 Contract 定义。

这里给出的是参考计算的精度口径，不要求实现使用相同的指令或物化相同的中间量。Design 引入的有限精度差异仍须满足另外确定的误差要求。

#### 4. 原则

1. **一个对象、一处定义。** 固定值、外部对象、局部记号分别放在自己的位置；下游直接引用，不重新解释或改变含义。
2. **一步一段计算关系，不按源码 pass 切分。** Contract 不描述优化主张，也不因当前实现拆成多个 kernel 就增加多个步骤。
3. **删重复，不删边界。** 无可见 key、路由权重的使用位置、量化 scale 的解释、初末状态等会改变结果，必须留下；泛化目标、实现细节与没有内容的字段不写。
4. **数学定义、reference 与数值验收不能混为一谈。** Reference 必须与选定的计算边界和配置一致，不能直接把待优化实现当作独立 golden。输出 dtype 也不能代替误差指标与阈值；本轮尚未定稿的 `accuracy` 不填空壳，不据此宣称验收已经完成。

#### 5. 具体case

下面仅展开 `contract.algorithm`，省略外层 `contract`。三个例子均包含常量、输入、输出、计算定义与 reference 映射。

**1）FA4 · SM100 · 因果**

**来源**：Dao-AILab/flash-attention `ce088ab9`；对应已有 FA4 因果设计，reference 为 `flash_attn/cute/testing.py::attention_ref`。

**范围**：8 个 MHA head，Q/K/V 均为 128 维；同次调用内各序列等长，但 Q 与 KV 长度可以不同。因果采用右下对齐，返回 O 与自然对数 LSE。Reference 以 FP32 输入调用，避免 LSE 被转换回 BF16。

```yaml
algorithm requirement:
  constants:
    scale: {value: "1/√128", meaning: 分数缩放}

  inputs:
    Q: {meaning: 查询向量, shape: [batch: BatchCount, query: QueryLength, head: 8, d: 128], dtype: bf16}
    K: {meaning: 键向量,   shape: [batch: BatchCount, kv: KVLength, head: 8, d: 128], dtype: bf16}
    V: {meaning: 值向量,   shape: [batch: BatchCount, kv: KVLength, head: 8, d: 128], dtype: bf16}

  outputs:
    O:   {meaning: 注意力输出,   shape: [batch: BatchCount, query: QueryLength, head: 8, d: 128], dtype: bf16}
    LSE: {meaning: 对数归一化项, shape: [batch: BatchCount, head: 8, query: QueryLength], dtype: fp32}

  compute:
    - step: 1
      name: 按右下对齐定义因果分数
      math: |
        visible(q, k) = 0 ≤ k < KVLength 且 k ≤ q + KVLength − QueryLength
        Z[q, k] = scale · Q[q] · K[k]ᵀ  若 visible(q, k)；否则 −∞

    - step: 2
      name: 对可见位置归一化并加权求和
      math: |
        O[q] = bf16(softmax(Z[q]) · V),    LSE[q] = logsumexp(Z[q])  若 ∃ k：visible(q, k)
        O[q] = 0,                        LSE[q] = −∞               否则

  reference:
    source: Dao-AILab/flash-attention@ce088ab9:flash_attn/cute/testing.py::attention_ref
    inputs: {q: "fp32(Q)", k: "fp32(K)", v: "fp32(V)", causal: true, return_lse: true}
    outputs: {O: "bf16(result[0])", LSE: "result[2]"}
```

**2）MoE · vLLM TritonExperts · FP8 block**

**来源**：vllm-project/vllm `28282ffe`；对应已有 DeepSeek-V3 路由专家、TP=8 的分片规格，reference 入口为 `tests/kernels/utils.py::torch_experts`。

**范围**：输入量化、专家计算与本分片内的 top-k 加权合并均在边界内；路由选择与跨分片归约在边界外。`I=256` 是本分片的中间维度，`Y` 是分片部分和，不是完整模型层的最终输出。`W13` 按门控、上投影的顺序拼接。

**量化基准**：采用本轮选定的普通 FP32 scale，amax 下界为 `1e−10`，scale 不取整为 2 的幂。两次激活量化引用同一个 `q`，不把融合 kernel 特有的数值规则反过来当成算法要求。

```yaml
algorithm requirement:
  constants:
    H: {value: 7168, meaning: 隐层维度}
    I: {value: 256,  meaning: 分片中间维度}
    E: {value: 256,  meaning: 专家数}
    k: {value: 8,    meaning: 路由槽位数}
    G: {value: 128,  meaning: 量化组长}

  inputs:
    X:   {meaning: 输入特征, shape: [token: T, hidden: H], dtype: bf16}
    idx: {meaning: 专家索引, shape: [token: T, slot: k], dtype: int32, range: "[0, E)"}
    w:   {meaning: 路由权重（原值）, shape: [token: T, slot: k], dtype: fp32}
    W13: {meaning: 门控与上投影权重, shape: [expert: E, out: 2I, in: H], dtype: fp8e4m3fn}
    W2:  {meaning: 下投影权重,       shape: [expert: E, out: H, in: I], dtype: fp8e4m3fn}
    S13: {meaning: W13 块缩放, shape: [expert: E, out_group: 2I/G, in_group: H/G], dtype: fp32, range: "(0, +∞)"}
    S2:  {meaning: W2 块缩放,  shape: [expert: E, out_group: H/G, in_group: I/G], dtype: fp32, range: "(0, +∞)"}

  outputs:
    Y: {meaning: 分片部分和, shape: [token: T, hidden: H], dtype: bf16}

  compute:
    - step: 1(A)
      name: 输入按行分组量化
      math: |
        blk(g) = [g·G, (g+1)·G)
        q(z) = (zq, s)
          s[g] = fp32(max(max|z[blk(g)]|, 1e−10) / 448)
          zq[i] = fp8(z[i] / s[⌊i/G⌋])
        dq(zq, s)[i] = fp32(zq[i]) · s[⌊i/G⌋]
        x[t] = dq(q(X[t]))
      symbols:
        z: {meaning: 一行待量化值}

    - step: 1(B)
      name: 按块 scale 解释专家权重
      math: |
        Ŵa[e, r, c] = fp32(Wa[e, r, c]) · Sa[e, ⌊r/G⌋, ⌊c/G⌋],    a ∈ {13, 2}

    - step: 2
      name: 按给定路由计算门控与上投影
      math: |
        u[t,j] ‖ v[t,j] = x[t] · Ŵ13[idx[t,j]]ᵀ
      symbols:
        u: {meaning: 门控分量}
        v: {meaning: 上投影分量}

    - step: 3
      name: 门控乘积重新量化
      math: |
        A[t,j] = dq(q(SiLU(u[t,j]) ⊙ v[t,j]))

    - step: 4
      name: 下投影后按原路由权重求和
      math: |
        Y[t] = bf16(Σ(0 ≤ j < k) w[t,j] · (A[t,j] · Ŵ2[idx[t,j]]ᵀ))

  reference:
    source: vllm-project/vllm@28282ffe:tests/kernels/utils.py::torch_experts
    inputs: {a: "fp32(X)", w1: W13, w2: W2, topk_weight: w, topk_ids: idx,
             w1_scale: S13, w2_scale: S2, quant_dtype: fp8e4m3fn, block_shape: [G, G]}
    outputs: {Y: "bf16(result)"}
```

这里以 FP32 的 X 调用 reference，是为了让非量化中间量保持 FP32，而不把 BF16 测试路径的中间舍入固定为算法要求。

**待完成**：`torch_experts` 仍调用共享的量化与激活 helper，需要固定上述普通 scale 分支并隔离这些 helper。当前记录的是 reference 的适配起点，不是已经独立、已经验收的 golden；现有 Design 与这份量化基准的相容性仍需验证。

**3）GDN · FLA Triton 分块前向**

**来源**：fla-org/flash-linear-attention `954438d1`；对应已有 GDN 分块前向设计，reference 为 `fla/ops/gated_delta_rule/naive.py::naive_recurrent_gated_delta_rule`。

**范围**：等长序列、相同 head 数、dk=dv=128。Q/K 不在本算子内归一化；g 是单步自然对数衰减，β 是已激活的更新系数；接收初始状态并返回最终状态。这里定义逐 token 的数学递推，不固定 chunk=64 或分块算法。

```yaml
algorithm requirement:
  constants:
    scale: {value: "1/√128", meaning: 读出缩放}

  inputs:
    Q:      {meaning: 查询向量, shape: [batch: B, token: L, head: H, dk: 128], dtype: bf16}
    K:      {meaning: 键向量,   shape: [batch: B, token: L, head: H, dk: 128], dtype: bf16}
    V:      {meaning: 值向量,   shape: [batch: B, token: L, head: H, dv: 128], dtype: bf16}
    g:      {meaning: 单步对数衰减, shape: [batch: B, token: L, head: H], dtype: fp32, range: "(-∞, 0]"}
    β:      {meaning: 更新系数（已激活）, shape: [batch: B, token: L, head: H], dtype: fp32, range: [0, 1]}
    S_init: {meaning: 初始状态, shape: [batch: B, head: H, dk: 128, dv: 128], dtype: fp32, default: 0}

  outputs:
    o:       {meaning: 读出向量, shape: [batch: B, token: L, head: H, dv: 128], dtype: bf16}
    S_final: {meaning: 最终状态, shape: [batch: B, head: H, dk: 128, dv: 128], dtype: fp32}

  compute:
    - step: 1
      name: 先衰减历史状态，再写入 delta 修正
      math: |
        S(0) = S_init
        Ŝ(t) = exp(g[t]) · S(t)
        r(t) = β[t] · (V[t] − K[t] · Ŝ(t))
        S(t+1) = Ŝ(t) + K[t]ᵀ · r(t),    0 ≤ t < L
      symbols:
        S(t): {meaning: 写入 token t 前的状态}

    - step: 2
      name: 从更新后状态读出，并返回最终状态
      math: |
        o[t] = bf16(scale · Q[t] · S(t+1)),    0 ≤ t < L
        S_final = S(L)

  reference:
    source: fla-org/flash-linear-attention@954438d1:fla/ops/gated_delta_rule/naive.py::naive_recurrent_gated_delta_rule
    inputs: {q: Q, k: K, v: V, g: g, beta: β, scale: scale,
             initial_state: S_init, output_final_state: true}
    outputs: {o: "bf16(result[0])", S_final: "result[1]"}
```

### Contract hardware的YAML语言表述

2）hardware specific：硬件平台的具体信息。硬件的具体性能参数和指令集手册（wiki，地址索引）；microbench关键计算/访存指令的细粒度延时吞吐测试结果（wiki，地址索引）；该schema的design设计时所利用的硬件和特性；

#### 1. 定位

`hardware_specific` 属于 Contract，记录**目标设备的身份、核心硬件能力及其边界**，以及完整规格与 microbench 的引用。

这部分独立于 Kernel。同一设备、同一资料版本对应同一份描述，不随算法或 Design 的变化而增删特性。Planner 与 Coder 可以选择如何利用这些能力，但不能修改硬件事实或替换任务指定的设备。

内容从公共硬件资料中提炼，而不是从某个 Kernel 使用了哪些指令反推。筛选标准是：这条信息是否会改变可行的计算、存储或协作方式？不是“这一代新增了什么”，也不是“当前设计打算使用什么”。

#### 2. 字段与填写规则

| 字段                | 内容                                                       |
| ------------------- | ---------------------------------------------------------- |
| `device`          | 明确的设备型号，必要时包含产品版本。                       |
| `arch`            | 设备架构标识；具体指令的适用条件查`spec`。               |
| `features`        | 能力名称到简短描述的映射，说明关键行为、作用域或限制。     |
| `reference.spec`  | 固定版本的设备资料入口，包含资源规格、指令语义与支持范围。 |
| `reference.bench` | 固定版本的 microbench 入口，包含测试配置、吞吐与延时。   |

features 不只列名称，也不展开成手册。每条用一句话说清能力及必要边界。例如“支持 CLC”不够，应说明它接管的是未启动的工作，而不是抢占正在执行的任务。计算、存储、搬运、协作与调度不另设固定子区，没有必要的类别不填。

**只陈述条件，不替 Design 做选择。**“支持 CTA-pair MMA”属于这里；“采用双 CTA”“使用三重缓冲”“某单元是当前瓶颈”不属于这里。软件后端暂时无法表达某项能力，也不能写成硬件不支持。

摘要不是完整白名单。未列出不代表不支持；容易被误认为继承自其他架构的关键缺项，可以明确写“不支持”。不为对称而枚举全部功能开关。

参数外置，但约束仍须检查。SM 数量、资源容量、分配粒度与完整类型组合查 `spec`；吞吐、延时和组合执行结果查 `bench`。不在本区重复抄表，也不能因摘要没有列出就忽略资源限制。本次任务额外限制的可用资源由运行配置固定，不改写设备能力。

公共硬件条目是唯一维护来源，Plan 引用或只读展开。一份任务固定设备与资料版本；纠错、补测通过新版本发布，不静默改变已有任务的依据。

#### 3. 具体 case

以下展示摘要的粒度，不是完整能力表。`reference` 中的尖括号是占位内容，正式使用时替换为实际仓库、commit 与条目路径，不表示这些测试已经存在。

**1）NVIDIA B200**

```yaml
hardware_specific:
  device: NVIDIA B200
  arch: SM100

  features:
    tcgen05.mma: 单线程异步发射；累加器位于 TMEM；支持单 CTA 与同 cluster 内的 CTA-pair 协作
    TMEM: Tensor Core 专用片上存储，经专用指令访问，不等同于通用 SMEM
    TMA: 基于 tensor map 在 GMEM 与 SMEM 间异步搬运；支持向 cluster 内多个 CTA multicast
    cluster: 支持 cluster 内跨 CTA 的共享内存读写与原子操作
    CLC: 尝试取消本 grid 尚未启动的 CTA 或 cluster，成功后取得其索引
    L1/SMEM: 共享物理容量，SMEM carveout 与可用 L1 容量相互制约

  reference:
    spec: "<硬件资料仓库@commit:B200 条目>"
    bench: "<测试仓库@commit:B200 测试配置与结果>"
```

**2）NVIDIA RTX PRO 5000 Blackwell**

```yaml
hardware_specific:
  device: NVIDIA RTX PRO 5000 Blackwell
  arch: SM120

  features:
    mma.sync: warp 协同执行；操作数与累加器位于寄存器；支持 block-scaled 低精度 MMA
    tcgen05/TMEM: 不支持，不能沿用 B200 的 TMEM 与 tcgen05 CTA-pair MMA 路径
    TMA: 支持基于 tensor map 的 GMEM 与 SMEM 间异步搬运
    CLC: 尝试取消本 grid 尚未启动的 CTA 或 cluster，成功后取得其索引

  reference:
    spec: "<硬件资料仓库@commit:RTX PRO 5000 Blackwell 条目>"
    bench: "<测试仓库@commit:RTX PRO 5000 Blackwell 测试配置与结果>"
```

**3）NVIDIA GeForce RTX 4090**

```yaml
hardware_specific:
  device: NVIDIA GeForce RTX 4090
  arch: SM89

  features:
    mma.sync: warp 协同执行；操作数与累加器位于寄存器；支持 FP8 MMA
    cp.async: GMEM 到 SMEM 的异步拷贝，无需用通用寄存器中转数据
    TMA: 不支持；cp.async 不提供 tensor map 搬运语义
    L1/SMEM: 共享物理容量，SMEM carveout 与可用 L1 容量相互制约

  reference:
    spec: "<硬件资料仓库@commit:RTX 4090 条目>"
    bench: "<测试仓库@commit:RTX 4090 测试配置与结果>"
```

三个例子记录的是设备能力差异，而不是针对某个 Kernel 的推荐方案。B200 与 RTX PRO 5000 Blackwell 即使同属 Blackwell，也必须分别绑定各自的硬件资料，不能按代际名称继承全部特性。

## Realization 区

### realization的YAML语言表述

#### 1. 核心思想

**1.1 这一层做什么**

Realization 是 Planner 与 Coder 之间的实现交接区：只记录 **Contract 和 Design 尚未表达、但需要双方共享的实现决定**。它不是第四个 Design 区，也不是源码剩余细节的收纳区。

Contract 规定任务与硬件；Design 确定数学改写、工作分配与流水线；Realization 补充既定设计依赖的实现条件，以及少数值得优先尝试的实现选择。具体代码组织、地址计算、barrier 编号与 phase、精确 layout 等，默认留给 Coder。

**1.2 格式怎么来的**

沿用前面的方法：先对照三个固定版本的源码，分别写出 realization，再归纳共同字段，不预先建立指令、布局、编译器等分类。

三个 case 暴露了不同的边界：FA4 的原语选择与专职 warp、异步发射及 TMEM 数据流耦合；MoE 的 FP8 计算路径已经由 math_reshape 表达，没有必要再写一遍；GDN 的指数求值可以保留为与重计算代价相关的实现建议。

因此，沿用 `require / prefer / forbid` 三类，将每条决定统一压成 `item / choice / reason`。**源码采用了某个选择，只能证明它存在；是否需要交接、是否必须采用，还要根据它与 Design 的关系判断。**

#### 2. 字段划分

**顶层只区分约束强度，不区分实现类别：**

| 字段 | 记录内容 |
| --- | --- |
| `require` | 既定 Design 成立所需的硬条件，Coder 不得自行替换。 |
| `prefer` | 与既定设计的代价直接相关的优先选择，Coder 可以用实验支持替代方案。 |
| `forbid` | Planner 已考虑并明确排除、值得避免再次尝试的路径。 |

**三类条目共用相同结构：**

| 字段 | 记录内容 |
| --- | --- |
| `item` | 选择作用于哪个对象或计算，直接引用上游名称。 |
| `choice` | 具体的实现选择或兼容性要求。 |
| `reason` | 这一选择依赖哪个设计前提，或影响哪项既定代价。 |

`item` 可以是单个名称或名称列表；同一选择和理由适用于多个对象时合写，不逐项复制。对象名称与原语名称是当前实例的内容，不为它们新增模板字段。

本区不另设 `thesis`：设计主张已经在 Design 中，单项选择的依据由 `reason` 说明。

#### 3. 填写与反馈规则

**一个决定只写一次。** 数学公式、舍入点、tile 大小、warp 角色、buffer 深度和依赖边已经由上游确定，不在这里重复。Realization 的省略不取消这些约束。

**是否保留，先做替换测试。** 换掉这一选择后，若既定角色、数据流或重叠方式不再成立，进入 `require`；若设计仍然成立，但有具体理由优先采用，进入 `prefer`；其余留给 Coder。`reason` 必须说明这种关系，不能只写“提升性能”。

**空区直接省略。** 三类字段均可省略；整个区域没有新增决定时，可以省略 `realization`，示例中用 `realization: {}` 明确表示没有额外交接项。`forbid` 不机械地列出 `require` 的反面，也不收集所有可能的错误实现。

**反馈只附在** `prefer` **条目下。** 可选字段 `feedback` 内只保留 `choice / evidence`：前者记录 Coder 提出的替代选择，后者引用固定版本的实验记录。原条目的 `choice / reason` 保留，完整的编译配置、精度结果与性能数据放在实验记录中；没有实验依据就不填写反馈。

`feedback` 不是修改上游设计的入口。若替代方案需要移动舍入点、增加搬运、改变角色或 buffer 生命周期，应回到对应的 Design 区域讨论；若 `require` 无法实现，应报告阻碍，不能自行降为 `prefer`。

#### 4. 具体 case

下面只展开三个案例的 `realization`，Contract 与 Design 沿用前文。FA4 选择 work_schedule 版本二，不混入版本三的 CLC 调度。本轮是固定版本源码的静态核对，没有 GPU 实验反馈；`prefer` 是实现起点，不是已经验收的性能结论。

**1）FA4 · SM100 · 因果单 CTA**

**来源**：Dao-AILab/flash-attention `ce088ab9`，`flash_attn/cute/flash_fwd_sm100.py`、`softmax.py`；因果、单 CTA、hdim 128、bf16，对应 work_schedule 版本二。

保留两项原语硬条件与一项求值建议，不重复单 CTA、双 Q、流水线顺序和缓冲关系。

```yaml
realization:
  require:
    - item: [Q·Kᵀ, P·V]
      choice: tcgen05.mma
      reason: 独立 MMA warp 发射，累加器经 TMEM 由 Softmax、Correction 消费。

    - item: [Q, K, V]
      choice: TMA
      reason: 专职搬运 warp 通过 tensor map 发射整块传输，不承担逐段搬运。

  prefer:
    - item: E 的多项式求值
      choice: fma.rn.f32x2
      reason: 成对求值，压低从 SFU 转移到 ALU 的指令发射开销。
```

多项式本身、覆盖比例及误差界已经属于 math_reshape；这里仅建议其求值方式。替换 packed FMA 不等于允许取消多项式近似或修改其精度要求。

**2）MoE · vLLM TritonExperts · FP8 block**

**来源**：vllm-project/vllm `28282ffe`，`fused_moe/experts/triton_moe.py`、`fused_moe.py`；SM90、TP=8，权重 128×128 块 scale、激活 1×128 动态量化，沿用已有分片规格。

```yaml
realization: {}
```

空区是完整结果，不是待填项。FP8 码值直接参与 MMA、scale 在组内积之后相乘、激活后直接量化、路由权重在最终舍入之前相乘，都已经由 math_reshape 表达，再写 `require` 或 `forbid` 没有新增信息。

本例不额外固定具体 MMA 指令族，也不把源码使用 Triton 变成实现限制。空区只表示没有额外交接项，不表示已有 Design 已通过数值验收；它与 Contract 量化基准的相容性仍沿用前文的待验证状态。

**3）GDN · FLA Triton 分块前向**

**来源**：fla-org/flash-linear-attention `954438d1`，`fla/ops/gated_delta_rule/`、`fla/ops/common/` 及 `fla/ops/utils/op.py`；chunk 64、dk=dv=128、bf16 Q/K/V，沿用已有 Triton 分块前向分支。

```yaml
realization:
  prefer:
    - item: [γ, δ, Γ]
      choice: ex2.approx
      reason: 块内衰减因子反复求值，优先采用硬件指数路径而非通用函数实现。
```

已有 ledger 记录了衰减因子重计算带来的指数求值代价，因此保留这一建议；块长、五个 pass、状态与中间量的精度、求逆合并中的 TF32 输入均已在 Design 中确定，不在这里重复。

`ex2.approx` 表示优先尝试的原语家族，不额外授权 `ftz` 或放宽误差要求。是否采用仍须接受既有数值验收。
