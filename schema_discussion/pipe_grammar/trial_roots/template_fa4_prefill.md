可完整表达，而且首尾与批次顺序更清楚。[试写](template_fa4_prefill.yaml)覆盖 attention 的全部五个角色；只有动作、有序列表和 `for(...)`，无额外控制节点。固定的两个 d 分量短展开后共 5 个 for、0 个 if、0 个 while、27 处静态动作；原稿的 20 处变多只因展开分量，运行事件没有增加。

- **真实 dispatch。** 固定源码来自 `01fff90d714abc6e84904ea9c44523ba66d34a0e`。[包装器](sources/template_fa4_prefill/solution.py#L95)传入 FP8、hd256、page64、causal、pack_gqa 和 num_splits=1；后者仍可被内部 split 启发式改选。[root 条件](../../../schema_template/examples/fa4_prefill/root.yaml#L9)先排除 Split，再限定 B>16，或 B=1、ceil(Q/8)≤SM+floor(SM/2)、K≥98304。[interface:951](sources/template_fa4_prefill/flash_attn/cute/interface.py#L951)说明这两类均不进入 flatten/persistent，并由 973–979 行启用独立 K/V loader。实际类在 [1549 行](sources/template_fa4_prefill/flash_attn/cute/interface.py#L1549)选择 `sm100_hd256_2cta_fmha_forward.py` 的两 CTA 路径；本稿继承静态矩形调度，不覆盖该文件的其他调度分支。
- **顺序和可读性。** LoadQK 先发完两份 Q，再逐块装 K；LoadV 独立遍历。MMA 明列首 QK，中间每轮完整发完 QK0、QK1 才发前块 PV0、PV1，最后补末 PV。Softmax 在遍历后发布 FinalSum；Correction 先确认 Stats(0)，只对非首块 rescale，最后交替 stage/store。相邻批次不能合成逐 d 的 QK/PV 交替；列表只约束发起顺序，异步完成和 P 先读后覆写仍由继承契约保证。

下表的 kernel 均指上述实际文件的 `kernel`（定义于 1266 行）；动作数按书写位置计，不乘成员或迭代数。

| 角色 | for / 动作数 | 关键源码位置 |
| --- | ---: | --- |
| LoadQK | 1 / 4 | [kernel:2095](sources/template_fa4_prefill/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py#L2095)、[2124](sources/template_fa4_prefill/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py#L2124)：先 Q，后 K |
| LoadV | 1 / 2 | [kernel:2142](sources/template_fa4_prefill/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py#L2142)：独立 producer，每半片两次 TMA |
| MMA | 1 / 8 | [kernel:2604](sources/template_fa4_prefill/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py#L2604)、[2688](sources/template_fa4_prefill/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py#L2688)、[2774](sources/template_fa4_prefill/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py#L2774)：首轮、末 QK、末 PV |
| Softmax | 1 / 5 | [softmax_step:3431](sources/template_fa4_prefill/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py#L3431)、[store_sum_max:3987](sources/template_fa4_prefill/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py#L3987) |
| Correction | 1 / 8 | [首次确认:3159](sources/template_fa4_prefill/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py#L3159)、[correction_rescale:3677](sources/template_fa4_prefill/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py#L3677)、[correction_epilog_coalesced:3891](sources/template_fa4_prefill/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py#L3891) |

- **域与依赖。** 完整保留 i=[0,N)、later=[1,N)、d=[0,2)；later 仍限定非首项依赖，d 仍是资源及依赖的数据索引。没有 waits_for 替换，五角色全部继承原边；K/V 四槽、S/P 两槽、Stats 归还和单槽 OutputStage 复用关系不变。键和值每块各占两个半片，装入第 i+2 块须等第 i 块相应乘法完成；分数槽相隔两块复用，仍须等待采集完成。两个成员共同构成操作数就绪，只有领先成员发射矩阵乘，不能在展开半片时遗漏成员等待。输出缓存只有一槽，先暂存两半再统一存出会覆盖前半，因此两个存出完成边界也不能合并。
- **为何未用 while/if。** 源码外层 `while work_tile.is_valid_tile` 在本路径仅处理一次：[scheduler:1344](sources/template_fa4_prefill/flash_attn/cute/tile_scheduler.py#L1344)随后清除首工作标志。KV 是已定长的有序遍历，没有需要新控制的反馈。越界工作由继承调度拒绝；不凭源码出现 while 就扩展范围，也不为展示语法新增 if。
- **动作隐藏的流程与源差异。** Softmax 源码分成无掩码前缀和有掩码后缀（2922、2945 行），每轮事件顺序相同，掩码、最大值和指数仍属动作实现。源码末 QK 单列以归还 Q（2708 行），新循环由原 qk 契约保留此区别。rescale 包含两半片及片段双缓冲；warp vote 跳过改写时，3815 行仍归还 Y，因此动作不能消失。源码默认提前归还 Y（3968–3971 行），位置在第二半 stage 后、store 前；原 root 未单列该事件，本稿也未补入。静态路径没有下一工作消费该归还，但这一源粒度差异仍未独立可引用，不能由 store 完成推定归还时刻。
- **边界与验证。** N=1 时两个非首项循环为空，首 QK、末 PV、Stats(0) 无载荷确认、FinalSum 和两次输出均保留；N=0 不在有效工作契约内。PyYAML 解析及结构检查通过；基线和 61 个源码文件哈希与 manifest 相符。N=1…9 的五角色轨迹逐项等于原稿，并独立对照源码 MMA 分支；逻辑动作总数为 14N+7，N=1/2/3 为 21/35/49。按两 CTA 实例化继承依赖、只裁去原移位窗口外边，窗口内前驱均存在且图无环。发生域从控制树建立，不从完成日志倒推：先解析动作签名，再按目标动作实例查找前驱。若区间内发布缺失，核对即失败，不能和前驱集合取交来掩盖。多轮核对跨过环槽回绕，既检查身份，也检查发射顺序；单轮则特别检查空循环之后仍有收尾。这验证有限事件顺序，不证明 GPU 异步行为或数值精度；未跑 GPU，也未据此宣称覆盖其他反馈流程。
