# C：独立综合验收

本轮文字结果可以收束为 [ISSUE_REGISTER](../ISSUE_REGISTER.md) 的逐号登记和 [FIELD_ISSUES](../FIELD_ISSUES.md) 的两项规则选择。**52 个原始观察全部有去处，不能称为修复了 52 个漏洞，也不能称所有字段表达已经闭合。** 两个保留选择直接来自 M02-04、M06-01、M08-03；输入、数值、源码保证和调查未完事项另行保留。

## 核实范围与基线

C 先读 AGENTS、MAX_BRIEF、RAW_INDEX、全部八份 ISSUES 和 A/B 独立报告，再按争议复读原始草稿、输入、正式规范及源码。独立重点是：默认有限域与 QSA 溢出；状态/工作区域/存储；worker 成员与 warp；ragged item/LPT；执行登记/nSM；ZERO2 提交和 mirror；局部/整体边界；正式审阅与限域手工结论。

本轮正式增量以 `/tmp/schema_role_trial_before_mfYTjS/` 为基线，其 `reading.sha256` 的 16 份采集材料副本一致。A 的独立增量还对照 `/tmp/role_max_a_before_noivxsrh/A_DOC_DIFF.patch`。C 编辑前的 27 份正式文档副本保存于 `/tmp/schema_role_max_c_before_fc8g6fl0/`。没有用包含多轮旧工作的整个 git diff 推断本轮成果，也未回滚先前用户修改。

C 定向读取了 GEMM 的 abox、双 issuer/commit、nSM、weakref memo 与 graph/direct 回退；MLA 的 tiny 输出、3/4 槽通知、ItemGeo 排列、侧流与设备数量查询；MoE 的 REDF 输出、cfg44/46、cfg88 的 last-arriver 片段、ZERO2 verify/两图提交/host mirror；GDN 按输入固定提交用 `git show` 读取 scalar 消费和 forward 边界。未逐行重审全部算子和全部模板实例，其他证据明确沿用原稿或 A/B 的已记录范围。

## 对 A/B 的采纳与纠正

| 主题 | C 的结论 |
| --- | --- |
| 默认有限域和数值反例 | 采纳 A/B 对 M07-02 的纠正：query/kv_cache 不因 fp8 编码支持 NaN 就包含 NaN 输入。独立核 QSA：有限 BF16 的 2^64 点积可溢出 FP32；2048 个最大有限 BF16 V 的归一化加权和可有限，未归一化 U 两项即溢出。两者分别保留参考/验收与算法问题，不归入已解误读。 |
| GDN scalar 与包装边界 | 未把 A 的后续 source 定位写成问题已解。固定 oracle 的 Python scale 乘 FP32 张量位置已找到，PyTorch 2.9.0 实际消费舍入仍未独立核完；输入的 int64→int32 计时描述与 oracle 求差/tolist 不同，不能据 oracle 删除固定包装或新增转换 launch。 |
| 同一 CTA 中 Role[*] | 采纳 A 对 M03-05 原稿残留误读的纠正，并与采集前 DOC5 比较。worker.unit 的成员实例原已定义；一个 CTA 里的四个 warp 不因此各有事件。serial 只在已承诺的动作 done 后推进，不创造 CTA 屏障。 |
| 数学接口导航与同步 matmul | A 新表的“动作间传递的数值结果”可能读成额外登记条件。C 改为已作为下游接口的值，明确沿用原 DOC3 判据；本地对象的跨动作使用不自动新增 symbol。同步例压掉重复公式文字，点名 QSlot/KSlot/ScoreSlot，保留全参与 warp 的读取结束和结果可见性，不承诺一条硬件矩阵指令。 |
| ragged 与排序 | item 前缀映射保持原乘积工作数，没有允许 size 捕获其他 partition 坐标；P 不自动物化数组。C 将变长轴与独立 head 轴写清，并把 LPT 的 item 明确为排序后的表项位置，每项对应实际工作，消除“按原编号升序是否真的排序”的歧义。 |
| 局部/整体判据 | 认可 abox 固定机制内的尾块范围、双 issuer 的可定位局部协议、输入重排的局部接口；四槽不能只改 depth。tiny、B3→B2、REDF 须沿实际输出生产和工作组织判断。C 保留 M04 原局部倾向及 M08 独立完整候选判断，不把“双方赞成”当成最终完整选集批准。 |
| P1 最小 execution | DOC2、P1、EXTRACT、REVIEW、template 及工具入口已一致：只登记给定材料明确的执行层级/组成，其余硬件占位；DOC4/5 的名字引用范围未放宽。未登记某次 launch 配置为设备上限，也未从 CUDA 名称推导未给的能力。旧 Runner 写权限未支持，在 EXECUTOR_ALIGNMENT 如实保留。 |
| nSM 来源 | 同意 A/B 没有解决数值部分，进一步核到查询在 GEMM 初始化时针对设备 0 缓存，MLA 也有查询及源码 fallback。已有字段可以记录合规硬件资料中的查询关系，但现在没有符合 DOC2 的完整型号资料绑定。直接接受源码为数量来源须讨论放宽 DOC4；C 在 P1 补“来源规则仍有效、冲突按框架问题定位范围”，没有创造临时豁免。 |
| ZERO2 | 独立确认固定 flow 不包含 host 条件存在性。补充 A/B 较少展开的目录路径：选定固定图可完整表达，其他两图关系/调用条件/独特机制可如实保存，未完整 ZERO2 可保持独立候选身份。该路径不等于单份完整 Plan 表达 host 算法；有价值控制不被一概降成包装。mirror 迟到写入风险单列为来源/跨调用问题，未用新字段充当源码修复。 |
| 提取与正式审阅 | 采纳 B 对材料冲突、真实实例、未研究/未展开/P2 暂拟对象、P4 实际定位及 reviewer 独立止点的修订。正式有效全文、W/R 分工、独特价值/必要降级、P6 新终审都保留；限域评议没有增加完整 P1/P3 义务，也没有获得正式 approve。 |

## C 实际追加的文案

正式文档仅追加修订四份，均在 A/B 已改文件内：

- [DOC3](../../../schema_docs/3.design.math_reshape.md#符号属性)：数学值与流水线本地对象仍按原判据分工。
- [DOC4](../../../schema_docs/4.design.work_schedule.md#工作分解)：澄清变长 item 与其他独立轴；LPT 例明确排序后的真实表项位置。
- [DOC5](../../../schema_docs/5.design.tile_pipeline.md#describe-要说明什么)：收束接口导航的登记条件，压缩同步 matmul 例并补明确槽绑定。
- [P1](../../../schema_extract/phases/01_contract.md#第-3-段硬件占位)：硬件资料豁免不放宽既有引用/来源规则；不相容表达按既有框架问题处理。

本轮相对采集基线共改变 18 份正式 Markdown。DOC1、DOC6、P5、P6 等没有本轮增量；十份上限、具体方案、配置差异进目录、字段结构、封闭值、必填/默认、引用范围和事件解释保持原定内容。C 未改 README 导航，由主代理汇总。

## 验收结果与实际限制

- 原始八份 ISSUES 提取出 52 个唯一编号；综合登记有 52 行、无漏号和额外编号。A/B 独立报告均覆盖该集合，FIELD_ISSUES 也给出了全部编号的字段判断或非字段去处。17 个组只是主题归档。
- 采集时 `raw.sha256` 的 20 份 medium/input 原稿全部一致。没有修改八份 DRAFT/ISSUES、输入或 A/B 独立报告来抹去疑问。
- 18 份本轮正式修改文档与 A/B/C、综合登记、字段登记共 23 份 Markdown 的本地链接/锚点、代码围栏及行末空白已核对；本轮正式文件的 `git diff --check` 通过。
- 两段新增 YAML 例子（DOC4 紧凑 item 分解、DOC5 同步 matmul 动作）经普通 PyYAML 解析为映射，并按所示范围人工核对字段与既定语义。它们是明示省略外围定义的片段，不作为完整 Plan 或 GPU 可实现性的证明。
- 未运行 extract、schema checker、构建、GPU、数值探针或生产测试，未新增仓库测试；未改 Python、starter、原算子 YAML/PNG、源码、状态、archive 或旧批准。工具仍未适配，不把链接/解析通过写成执行器已实现。

F1 仍影响依赖设备规模查询的完整 mapping 来源；F2 仍影响要求一份完整 YAML 包含 ZERO2 host 数据相关提交的任务。已有目录能力可保留相关知识，不消除明确的完整交付要求。普通设计的材料不足、QSA/其他候选数值分析、GDN 的接口事实、缓存/侧流/mirror 保证、MoE 未完成调查仍分别开放；本轮完成的是提示修订与可追溯核实，不是替八份草稿补全算子实现。
