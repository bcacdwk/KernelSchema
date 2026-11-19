# MoE 直接 Schema 审修

MoE 三份完整设计已完成源码审阅、必要修订与回读：`root` 4→5、`split_k` 1→2、`mega` 2→3，保持三稿、所有 Design 区域仍本地定义。正式目录只改三个 YAML 和 `dispatch.md`。原完整目录与 `.extraction/receipt.json` 已逐字节保存于 [before](before/)，基线见 [before.json](before.json)。

本次按用户指定由同一名全新 gpt-6-astra / xhigh 代理先作 worker 修订，再作 reviewer 核对；这是直接人工审修，没有新 extract、Runner P6 或旧批准对新字节的重绑定。原 run `74bcac91-808a-41b2-aab4-85143935ab40` 的 39 次调用与正常 P6 发布事实按既有交付记录保留，原 receipt 未改写。

## 已修正

| 问题 | 当前结果与依据 |
| --- | --- |
| 参考 QDQ 的 scale 舍入点错误 | `reference.py::nvfp4` 实际先乘 FP32 的 `1/6` 常量。Contract 原来的 `/6` 改为该乘法；CPU 算得 peak=0.1113281175494194 时，两写法最终 E4M3 code 分别为 10、9，差异不是排版问题。 |
| root type-10 的 folded prescale 被写成 non-folded | 按 `act_k_constants`、`activation2` 显式写出先形成 `k_gu/k_uu` 再乘 gate/up 的舍入顺序。beta=25、up≈0.00025 的 CPU 标量结果已展示两个次序不同；split_k/mega 保持源码 non-folded 形式。三稿的 L2E 均直接写真实 FP32 常量，避免与 expert 变量 e 混淆。 |
| 验收配置只抄了较宽的一组 | 保留 metadata 的两组 source-only 事实：default 为 atol=.01/rtol=.05/max_mismatch_fraction=0；`mutated_inputs.output` 为同 a/r、max_mismatch_fraction=.001/max_rel_l2=.01。没有推断它们对同一个结果同时生效，也没选一组作为真实比较器；`Gap` 明写比较公式和 return/writeback 应用映射缺失。 |
| 激活参数缺少来源关联 | 补明 input construction 的 type-10→alpha=4、beta=25，以及 type-3 参数范围、向量在专家维一致的事实；保留原动态域/固定样本的两分支并集，不用实现限制改变任务。 |
| ACT 载入数少了一半 | 每个线程每切片各读一个 up float4 和一个 gate float4，总计 `2*split_count`，即 SK=2 时四个 float4。 |
| clipped gather4 过度声称不读取 padding | 末个有效 box 的不足四行部分实际重复首个有效 token；只省略后续整个 box。A scale 的 padding 不载入并保留 shared 槽字节。128 种 tile_rows 的坐标及 tx-byte 数已验算。 |
| FC2 padding 一概称未生产字节 | 明确可能读下一专家的有效 act 行、allocation 尾部或 descriptor 界外零填充；scale staging 重复本 tile 首个有效行。输出 mask 丢弃当前 tile 的 padding，不能把全局有效行说成未写。I=256 时 scale 只 prime 一个 stage，也已修正。 |
| 字节账及 MEGA 分配/物化混淆 | BF16 到 packed+scale 是每元素 2→9/16 byte，非减半；gather payload/scale 为同一 allocation 的互不重叠 region。MEGA 省略中间量读写，但 host 仍分配未消费的 route/act/rout slabs；当前描述区分这两者。act 的真实 byte-view 账在自核记录保留。 |
| 源注释/编译观测被当成全域保证 | 去掉激活误差的全域界限口吻；ACT 的“无 launch_bounds 所以总会≤85 regs、3 CTA/SM”改成有来源成本依据的 Prefer，并注明依赖工具链。MEGA 的空路由选择不能证明非空 replay 全域数值验收。 |
| root 完整参数与默认 selector 混用 | root `when` 明示保留大准备、fin8、PDL-on。源码确有可选择这套方案的控制：`SMALL_THRESH=1`、`FIN16MB=1024`、`PDL=1`、`PDL_BIG_R=-1`（全名均以 `NVFP4_MOE_` 开头），CFG/CFG2 不覆盖。原 shapes/97–81 门仍在，输入域不缩窄。默认 fixed 3 对应原正文；fixed 4/5/7 的 fin16、fixed 6 的 fin16+PDL-off 是显式变体。原 NCU/耗时不作为强制配置的性能保证。 |
| MEGA 跨调用 self-clean 前提遗漏 | REALIZATION 说明同一 acc32/cnt workspace 要在前次调用完成自清后复用。源码没有为跨 stream 并发复用另加同步，未把该能力当成已存在，也没新增 Contract 前提。 |

## 无需改动的核对

- 36 个调用参数与 candidate/reference 签名一致；`output_buffer` 是源码 `output` 的 schema 名称。返回 `[output]`、同对象全量覆写、不读旧内容、只允许该 view 被改动均由 source/input/metadata 对上。当前任务要求提供 buffer，因此不因函数签名 `output=None` 添 nullable/default；没有移入 Sparse 的 NaN 前态规则。
- `topk_ids` 的二维逻辑元素为 `(int32, fp32)` pair，`Compute.Routing` 已说明物理对象为两个 tensor 的 tuple；保持现有表示，未机械改成整体标量对象。
- E2M1 nibble 次序、E4M3 格式、准备权重行置换和 scale swizzle、route 无效分支、up 双侧/gate 单侧 clamp、FC2 的 per-route BF16 边界及最后 FMA 路由和均对上来源。参考单独乘 score 再加与候选 FMA 的数值区别继续保留。
- root 七节点、split_k 五节点、mega 单节点的实际生产/消费不同，三份各有独立设计价值；未引入 from，未新增/删除设计。所有有效稿的 Contract 解析到当前 root，Realization 为自身定义。
- 两 GEMM 的 payload ring 3、scale ring 2、MMA 累积 commit、EPI accumulator handoff、TSTORE `wait_group.read 0` 归还 sink 后再写下一 chunk、退出 `wait_group 0` 等全局可见性均沿源码核对。无证据支持增加新的等待或替换 source 协议，故不改 Pipe/Waiting 拓扑。
- Storage 的 register 形状按 CTA worker 聚合；sf_prefetch=4096 bytes/CTA，非每线程 4096。TMEM 使用 192 列、分配 256；shared allocation 是 106640/114960 bytes。mega 的动态与静态 shared 分开计量；未从占位硬件区推导运行驻留保证。

## 校验与边界

[semantic_selfcheck.json](semantic_selfcheck.json) 记录了实际通过的 CPU 自核：32768 个合法几何实例执行提取出的纯源码 `_pick_cfg`；30160 个 root 几何实例满足已述来源控制；24064 个 payload 行逆置换、2408448 个 scale 地址、148 个 Storage 算账实例、512 个 shared scale word 和 128 种 clipped gather 计数。8 份核心来源文件均与固定 commit `24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2` 逐字节一致。该检查未执行 torch、CUDA、native kernel 或完整数值参考。

[normalize_final.json](normalize_final.json) 与 [check_final.json](check_final.json) 均 exit 0，覆盖完整目录三份 YAML 与 dispatch，最终为 `3 YAML; 0 would change`。中途一次逗号引号错误、文本预算超限和规范化差异的失败输出也完整保留于 `check_revision_1/2/3.json`、`normalize.json`。未放宽 checker。

[effective_readback.json](effective_readback.json) 对三稿逐区比较 raw/effective、当前 root Contract、当前来源 SHA 和本地 Realization，一致。`--show-effective` 的 `validation:not_run` 不被当成验收；目录验收来自上面的完整 check。

仍有明确的材料/源码限制：外部比较器公式与应用映射缺失；硬件区按固定提取保留占位；近似误差、MEGA 非空 replay、source 宣称的 bit-exact 与性能收益均无本次 GPU 复现。MEGA host 仅检查 H%256，但其 FC1 需要 H%512，当前 H 集合满足；ticket 数组依赖 K=8，当前任务满足。这些旧来源限制保留，未修 kernel、未扩大输入域，也未删去不足。

## 留给统一清理

当前 `.extraction/receipt.json` 仍是旧批准事实。`dispatch.md` 原有 P1/P2/P3 叙事、PROMOTED/旧 job/历史快照与“本次未独立验证”等过程记录未整体清理；请最后的统一清理代理区分来源事实与过程历史后迁移。本次审修日期、模型档位、失败检查、before/after、未新增 P6 等只在本 discussion 目录中记录。

机器可读结论见 [review.json](review.json)，当前字节清单及对它的摘要引用见 [after.json](after.json)。没有 SHA 自引用。
