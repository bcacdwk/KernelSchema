可以完整表达，准备范围与重复范围更清楚。本稿依照 [pipe-control-v0.2](../README.md#pipe-规则)，覆盖原 root 的 `prep.Main`、`chain.Main` 两阶段、两个角色；共 26 个静态动作、2 个 `for`、1 个 `if`，未使用 `while`。只投影 pipe、完整 domain 及需要更新的依赖，数学、调度、资源、成员、执行模式、动作契约和里程碑沿 [原 root](../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml) 继承。

- 范围严格取非 segmented、非 ordered、非 direct_qk 的缓存路径：`MODE=0`、`perm=nullptr`、`KQTMA=false`，原 dispatch 继续限定适用输入。固定源码来自 manifest 的 `6e4ca91bd91d998cac99e8dadd83838e153c679d`；没有把其他五个变体混入本次覆盖。
- `enter:Metadata` 保留在条件外。它装入实际 chunk 数、请求边界及夹紧后的索引，初始化屏障并完成 CTA join；因此当前位置的 `Active` 已就绪。无效 slot 的 launch trigger 与返回仍在 enter 契约中。`if(Active == 1)` 只决定后续动作是否发生，既不读设备状态，也不暗中等待。
- prep 完整域只剩 `r: [0, "ceil_div(HeadsPerPrep,GroupWidth)"]`，删除专供门控的 `a: [0, Active]`；chain 完整域仍为 `c: [0, c_n]`。`Active` 仍是调度给出的 0/1 值。原 enter 文本中“a 域为空”现在对应 if 为假；数学中的同名字母不属于此次域替换。
- `save_image:KQImage(a)`、`drain_image:KQImage(a)` 均改为索引 0，保持实际事件身份。prep.Main 的完整 `waits_for` 仅有 `data` 下的 `drain_image:KQImage(0)@done: [save_image:KQImage(0)@done]`。等待按目标 if 的发生域实例化：目标不存在则没有边；目标存在而前驱缺失必须报错。没有新增 q/a 假域；chain 未提供替换，继续继承原依赖。
- `r` 是一份已分配 CTA 工作内的头组遍历，`GroupWidth` 为 1 或 2，三头组的末轮只有一头。它不领取后续工作。skeleton 只在循环前初始化一次，Q/K/F 跨组存活，下一组复用 scratch 前仍依赖 publish_group 的显式 join。非图像 owner 的 save/drain 保留事件，不能再用传输条件删掉它们。
- chain 将五个准备动作、`for(c)` 的五个动作和最后写回直接顺排。PRE0 的输入 Q/K/V 与初态读取先于 join_prep；prime 再补首块系数，复用同一次完整字节期望。prime 完成仅代表发出，当前槽可读仍由 advance 的 acquire 与 CTA join 保证；列表顺序不把异步传输改成同步完成。

| 实际源码位置 | 核对内容 |
| --- | --- |
| [Model.forward:52](sources/extract_gdn/solution.py#L52)；[gdn_forward:3320](sources/extract_gdn/gdn_kernel.cu#L3320)、3343–3358、3462–3477 | 扩展入口、缓存/顺序/PRE0 选择及 MODE=0 发射 |
| [gdn_prep:225](sources/extract_gdn/gdn_kernel.cu#L225)–242；658–665、693–714、1274–1292 | 元数据先于退出，图像发出，skeleton 先于两种头组循环 |
| [gdn_prep:852](sources/extract_gdn/gdn_kernel.cu#L852)–1262；1686–1740 | Qd 先发出、求逆及发布、跨组归还、最终图像 drain |
| [gdn_chain:1983](sources/extract_gdn/gdn_kernel.cu#L1983)–2024；2211–2224、2248–2343 | 屏障初始化、PRE0、初态/PDL 顺序，预装与逐块获取/复用 |
| [gdn_chain:2448](sources/extract_gdn/gdn_kernel.cu#L2448)–2550；2598–2660、2753–2863 | mini、完整 solve 的读退出，输出扫写及最终状态暂存/转置 |
| [mbar_wait:300](sources/extract_gdn/gdn_common.cuh#L300)–305 | 等待原语内部的 PTX 轮询，沿动作契约保留 |

| 有限边界 | 应有事件与顺序 |
| --- | --- |
| `Active=0` | 仅 enter；没有 save/drain，也没有对应等待实例 |
| `Active=1`，头组数 1/2/4 | 每组完整执行八动作；skeleton、save、drain 各一次 |
| 单 chunk，`c_n=1` | 准备一次、五动作一次、final_store 一次；mini 也保留 solve 的 issue/done |
| 多 chunk，`NSTAGE=2/3` | 按 c 整组推进；下一轮 advance 的 join 先结束旧读取，再发出存在的未来块 |

- mini 仅在 `VTT=64` 且有效行数为 1 时出现；rhs 发布 DU0 后，solve 无读写和 join，finish_chunk 执行秩一更新。末轮和不请求最终状态的情形也保留原动作事件。final_store 不借用退出后的 c；其契约指向已暂存的末态。
- 原同步缺口仍在：请求 FinalState、末块为 mini 且 `(c_n-1)%NSTAGE=0` 时，2511 行的 Kc 行读取与 2835–2836 行的槽 0 写入之间缺少跨 warp 读退出保证；2839 行的 join 在写入之后。有序列表及现有 finish_chunk 粒度都不能补此保证。普通非末块由下一次 advance 归还槽，full 路径则保留 solve 的预装与 join。
- 已通过 PyYAML 解析、角色覆盖、单键节点与索引作用域检查，并核对原 root 和三份源文件的 SHA-256。6 组 prep、10 组 chain 的新旧事件顺序完全一致，覆盖三头末组、两种槽深、单块与多块、最终状态开关。另有 3 个依赖负例：前驱声明仍在但被独立条件屏蔽时必须失败；目标不发生时也不能掩盖拼名或签名错误。槽检查只核对有限发出/获取顺序，不构成 GPU 并发证明。
- 未运行 GPU。旧同步、数值及非连续 core 回写问题沿 baseline 保留，有限控制检查不提供修复或精度结论。此路径没有需要 pipe 反馈控制的循环，底层等待轮询不升格为 while；未使用也不等于验证过其他反馈流程。
