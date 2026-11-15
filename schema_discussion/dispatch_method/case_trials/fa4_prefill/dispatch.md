# FA4 prefill 候选设计目录

**试写，尚未迁移任何完整 schema。** 下列条目链接[现有 11 份 YAML][old-index]作代表证据；旧语法、旧 `when` 与批准状态不会因链接而变成新规则的完整交付。目录保存设计决定，不重建原 selector，也不自动组合局部记录。

## 拟保留的完整主线

root 保存唯一 Contract 和本地三区；其余是共用 Contract 的并列设计，不必连成 parent 树。当前记录范围先取各来源已说明的子域，同映射合并仅取两份子域的并集；阈值不是最大可行域，未核对的组合留空。

| 候选条目 | 完整代表与核心决定 | 关系 |
| --- | --- | --- |
| `root`：矩形 static | [root][root]：一个 CTA pair 完成一个位置，局部拒绝无效位置；GQA 打包、lagged-max FP8 online attention、独立 K/V ring 与 Y/Stats 交接。 | 默认阅读起点，完整保留数值路径；root 不是其他范围的 fallback。 |
| `folded_persistent` | [packed_persistent][packed]和[single_persistent_dual_load][single]：按请求前缀定位、折叠 query，有限 worker 按固定步长接续多项工作；mapping 与 ring phase/Q/Y 复用跨 worktile 延续。 | 两者采用同一映射；B=1 的前缀查找退化，不另建设计。四步二分/线扫留给实现。 |
| `split_kv` | [splitkv_dual_load][split]：KV 连续分段，写归一化 FP32 partial 与 LSE，独立 combine 归一到共同参考后形成输出。 | attention 部分延续 root 的局部算法，新增跨 CTA 物化与第二次提交。不能仅换 worker_count 或链接旧 attention+无说明的 combine。 |

`folded_persistent` 合并依据不仅是两个区域逐字相同：两份都以 `C_b=ceil(Qlen(b)/16)` 建 Prefix，用同一 folded m 映射；B=1 时 Prefix 只定位请求 0，`Cpad=C_0`。见[源码前缀映射][mapping]。这不证明 LPT、joint-loader、split 的任意组合成立。

## 一次记录的局部决定

### request_lpt

将 folded 序列的请求映射换成按 KV 长度降序的排列；前缀、长度与真实请求索引必须使用同一排列，结果仍写原请求。[static][static]和[persistent][lpt]共用这一逻辑决定，不保证等长请求的稳定次序；排序网络不另记。

### static_folded_coordinates

[small_static_lpt][static]是 root 单工作生命周期、folded 前缀映射与 LPT 的已见组合。相对 root，还须由 elected thread 解析本项坐标，Softmax 前奏发布 `WorkCoord`，LoadQK/LoadV 在首次加载前等待它；persistent 则每 CTA 保存并复用 `RequestMap`。这项静态坐标发布不能省成“只是排序”，也不必再成为完整 attention 身份。

### shared_alpha

相对 root 的 `Stats=(old maximum, base)`，由 Softmax 计算并发布一列 FP32 alpha，同时用于自身分母递推与 Correction 的 Y 缩放；Stats 的载荷/视图和两个 role 一起改变。沿用基线的 Stats 捕获后归还、前一 PV 完成后修正 Y 与首块处理，不能只把存储列数改小。[static][static-alpha]、[persistent][persistent-alpha]为两个已有实例；[源码生产和消费][alpha]。

### joint_loader

把独立 K/V producer 换成一个 loader，以 `K0,K1,V0,K2,V1,…`（单块为 `K0,V0`）形成共同 token 序，每项仍分两半；各自 K/V 数组共用四 token acquire/release 流。K 由 QK 消费后归还、V 由 PV 消费后归还，相应 slot index、等待目标与退休顺序一起改变。[static][joint]、[persistent][persistent-joint]、[split][split-joint]已有实例，完整环序列以 `KVStage` 为准。

## 信息保存对照

| 去向 | 本次实际保存到哪里 |
| --- | --- |
| 完整主线入口 | 三行各指向具体完整代表；static/persistent 的跨工作生命周期、split 的输出形成方式均保留。 |
| 同规则合并 | 两份保持请求序的 persistent 记录合为一条；不把单请求查找另当局部变体。 |
| 一次局部说明 | request_lpt、static_folded_coordinates、shared_alpha、joint_loader 各一条；静态组合新增的坐标发布单独定位，不误当所有 LPT 都需要的协议。 |
| 已知组合不单列 | LPT+static/persistent、LPT+alpha+static/persistent、joint+矩形 static/单请求 persistent/split；相应旧文件仍作证据。不由这些实例推出完整组合矩阵。 |
| 主动不记录的实现精度 | 线扫/二分实现、排序网络、原 selector 的性能阈值算式；物理列号无需成为 alpha 变化身份。 |
| 尚未研究/完成 | 新规则下完整 YAML、最大适用域、上述已见实例以外的组合与 GPU 复验；没有证明旧 11 份的所有信息已无损迁移。 |

例如未来迁移 `folded_persistent` 时，已核相同的数学区可以写成以下**非完整片段**；schedule/pipeline 仍须完整填写。它不改变主线条目数，也不继承 root 的 `when`、元数据或 Realization。root 三区仍本地，同任务内容引用图须无环。

```yaml
design:
  math_reshape: {from: root}
```

[old-index]: ../../../../schema_template/examples/fa4_prefill/dispatch.md
[root]: ../../../../schema_template/examples/fa4_prefill/root.yaml
[packed]: ../../../../schema_template/examples/fa4_prefill/packed_persistent.yaml
[single]: ../../../../schema_template/examples/fa4_prefill/single_persistent_dual_load.yaml
[static]: ../../../../schema_template/examples/fa4_prefill/small_static_lpt.yaml
[split]: ../../../../schema_template/examples/fa4_prefill/splitkv_dual_load.yaml
[lpt]: ../../../../schema_template/examples/fa4_prefill/small_persistent_lpt.yaml
[static-alpha]: ../../../../schema_template/examples/fa4_prefill/small_static_shared_rescale.yaml
[persistent-alpha]: ../../../../schema_template/examples/fa4_prefill/small_persistent_shared_rescale.yaml
[joint]: ../../../../schema_template/examples/fa4_prefill/rect_joint_load.yaml
[persistent-joint]: ../../../../schema_template/examples/fa4_prefill/single_persistent_joint_load.yaml
[split-joint]: ../../../../schema_template/examples/fa4_prefill/splitkv_joint_load.yaml
[mapping]: https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py#L1213
[alpha]: https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py#L3523
