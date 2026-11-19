# extract_gdn：全量文字审计 v1.1

本算子的核心设计目前依赖自由叙述，不能只读结构与共享规则还原。最集中的是 stage 交易与预取、单个多 warp role 内部的子集同步、共享存储的全部读者退休、原始 perm 字节暖读、外部存储副作用，以及跨调用 workspace。公式本身不构成语言缺陷；自定义 verb 合法也不证明这些事实已被结构承载。

样本固定为 manifest 中 working checkpoint `349f0fdbad54effd79e2557888dffcc88de418c142438fd2989db2b4937d0a2d` 选定的 6 份 YAML 与 dispatch：root v11，cached_ordered/direct_qk/direct_qk_ordered v2，segmented/segmented_ordered v3，均为 approved，无骨架节点或未完成分支。本结论不是重新授予批准，也不把 approved 当作 GPU/数值验证。规则使用本审计冻结 README、DOC0/1/3/4/5/6、REVIEW 与 examples/README。root 的 hardware Contract 是本批提取允许的 null 占位。

来源统一为 `meta_data.source` 的 commit `6e4ca91bd91d998cac99e8dadd83838e153c679d`。以 `git show` 取得并亲读 solution.py、其 gdn_forward/关键 CUDA 路径、reference.py 及 input.py 的写区域定义；副本在 work/extract_gdn。C++ 行号均属于此提交。没有使用不匹配的当前源码行号。

范围与计量

- 已全文审阅 336 个精确文本组、790 个声明位置（含 dispatch 的 13 段）；原 316 组加 v1.1 的 20 组 name/step，全部人工语义分类。全部使用节点/作用域均核对；没有未审字段的默认标签，也没有 occurrence override。
- 去重开放字段面积 88,227 字符，其中英文 YAML 85,996、中文/混合 dispatch 2,231。YAML 开放文本声明 148,990、alias 实例 149,851。字符按规范化空白后 len 计，不是 token，也不是纯自然语言字数。
- YAML 全部字符串声明 176,432、实例 177,558，包含公式、引用与标识；有效 Design 字符实例 151,948，单独统计，不与 raw 相加。六节点三个 Design 区均本地定义，没有 from_parent。
- 全量 scalars 复核后，新增 name/step 已纳入；剩余为来源、引用、标识、类型、成员选择及 shape/domain/retention 表达式，未发现尚遗漏的独占正文。

| 节点/类型 | 状态 | 全字符串声明/实例 | 开放文本声明/实例 | 有效 Design 实例 | verb 声明/实例 |
|---|---|---:|---:|---:|---:|
| cached_ordered：缓存QK、排序 | approved | 27,382/27,382 | 23,243/23,243 | 24,001 | 27/27 |
| direct_qk：直接QK、输入顺序 | approved | 25,649/25,649 | 22,053/22,053 | 22,166 | 25/25 |
| direct_qk_ordered：直接QK、排序 | approved | 27,752/27,752 | 23,948/23,948 | 24,260 | 24/24 |
| root：缓存QK、输入顺序、PRE0 | approved | 32,111/32,111 | 27,794/27,794 | 22,756 | 26/26 |
| segmented：分段A/C、输入顺序 | approved | 31,687/31,930 | 25,986/26,099 | 28,902 | 38/38 |
| segmented_ordered：两段A/C、排序 | approved | 31,851/32,734 | 25,966/26,714 | 29,863 | 39/40 |

节点审阅均贯通 Contract → 本地 math → 工作范围/参数/目标 → launch/handoff → role/resource/pipe → Realization。root 的首块 PRE0、排序路径先暖读后重新解析请求、direct-QK 的图像物化/传递删除、A/C fresh worker、G=2 的 SegB(1) 无读者及无 mini 分支都分别核对。segmented 的 mapping alias，以及 segmented_ordered 的 mapping/schedule/warm alias 只节省文档内声明；它们不共享存储身份。分支间相同文字仍是独立声明，不能用大树重复量替代 root 判断。

全字段分类（字符是字段面积）

| 类别 | 精确组 | 去重字符 | 声明位置 | 声明字符（含 dispatch） |
|---|---:|---:|---:|---:|
| DUPLICATE | 10 | 1,481 | 15 | 2,072 |
| NONCORE | 1 | 186 | 2 | 372 |
| STRUCTURABLE | 8 | 1,230 | 9 | 1,426 |
| ESSENTIAL_TEXT | 18 | 5,634 | 31 | 10,158 |
| FORMAL | 39 | 2,888 | 153 | 5,835 |
| SOURCE_LIMIT | 9 | 2,784 | 17 | 4,124 |
| EXPLANATION | 135 | 13,484 | 338 | 25,617 |
| MIXED | 113 | 59,617 | 219 | 99,771 |
| UNRESOLVED | 3 | 923 | 6 | 1,846 |

root 单独有 150 个声明位置、27,794 开放文本字符：MIXED 35/17,436，EXPLANATION 73/4,864，FORMAL 27/918，SOURCE_LIMIT 5/1,714，ESSENTIAL_TEXT 6/2,272，STRUCTURABLE 1/146，UNRESOLVED 1/240，DUPLICATE 2/204（每项为位置数/字符）。root 自身已有必要自由叙述，结论不是由分支复制面积产生。

全算子有 18 组 ESSENTIAL_TEXT（31 个声明、6 节点），另有 113 组 MIXED。只看 pipeline.verb、schedule.strategy、realization.choice，110 组含至少部分核心事实（209 个声明，6 节点；85 个 verb、18 个 strategy、7 个 choice）。这不是“110 个结构缺口”：其中既有公式/参数定义，也有已能迁移的内容、开放原语语义和表示成本问题。39 组 FORMAL 含正式引用及成本关系，不能算作冗余叙述。

去重机制 finding（完整 unit ID、节点、反例及源码/规范锚点见 annotations/extract_gdn.json；组数可重叠）

| ID | 判断 | 涉及组/声明/YAML节点 |
|---|---|---:|
| F01 | 外部 prefix/readonly/fresh/no-alias：开放存储契约，nullable 不足 | 8/8/1 |
| F02 | 数值公式、求值分组、正式引用：保留；可缩默认说明 | 62/197/6 |
| F03 | 比较器、近似误差及 compiler contraction：来源限制 | 23/45/6 |
| F04 | 摘要、释义、标题、导航：有阅读价值，不能自动整字段删除 | 134/332/6 |
| F05 | when 的独立定义与共享派生式：跨节点复用成本 | 18/18/6 |
| F06 | 参数/owner 定义与现有 partition/iteration/target 混写：可部分结构化 | 31/52/6 |
| F07 | 静态工作分配/grid轴说明：已有结构与公式覆盖 | 10/15/6 |
| F08 | ring/PRE0/A交易字节与到达语义：原语内协议，表示成本 | 17/29/6 |
| F09 | PDL启动/完成分离：已有 flow/milestone，可通用 wait 短写 | 25/44/6 |
| F10 | prep subset joins、扫描/搬运并发与成员分工：表示成本 | 38/98/6 |
| F11 | inactive/empty/weak/no-op：现有数学+repeat域可表达 | 41/75/6 |
| F12 | alias全读者退休与最终mini竞态：结构有存储、文字有执行保证 | 40/83/6 |
| F13 | raw perm暖读与数学Perm区别：开放访问语义 | 6/6/3 |
| F14 | swizzle/对齐与完成原语绑定：有意开放实现语义 | 16/42/6 |
| F15 | core临时副本偏差、dense输入实际复制提交：保留来源未决 | 14/24/6 |
| F16 | process workspace/L2缓存跨调用身份与复用：缺字段/作用域候选 | 10/11/6 |
| F17 | 空bulk wait及末尾重复解释：少量非核心 | 3/8/6 |
| F18 | 部分rewrite/verb跨文件复用：整区引用限制带来的表示成本 | 75/233/6 |

现有结构能解决的部分

F06/F07：保留已有 `partition`、`worktile.iteration`、`target` 和 `worker_assignment`；例如 `preassigned: {start: worker_id, count: 1}, refill: none, exit: assigned_done` 已给出每 CTA 一份静态工作。长 strategy 中“chunk 不是独立 worktile”和“rank order 不保证 CTA 完成次序”可缩；SU/Active/Shared/HeadsPerPrep、NSTAGE、PRE0、输出有效行和唯一 owner 的公式仍保留。cached_ordered 的排列数学可迁到现有 rewrite.math/symbol.perm，不需要新字段。

F09：以 root 的等待为例，现有八条 handoff 仍保留 `ready: prep.kernel_done, wait_at: chain.prep_ready`；把自定义 `join_prep:Prepared` 换为通用 `wait:Ti`，并令 `prep_ready: ["Main/wait:Ti@done"]`，保持 prologue 的 setup → early_inputs → wait → prime → seed 顺序。Realization 继续将该等待绑定 `griddepcontrol.wait`，所以没有用动作名冒充未定义机制，也没有把 `start_after: prep.launch_ready` 改为完成屏障。A/C 等待同理。

F11：C 已有 `run: [0, Run]`、`cl: [0, Count]`。`Run=0 if weak and z>0 else 1` 不移除 setup、输入解析和 predecessor wait；将定义完整的 weak 计算换为通用 `compute:weak` 后，再进入 `repeat: run` 即可。空 A 的 summary 发布位于 cl 循环外；已有结构不需要新增条件删边机制。

仍需文字或尚未闭合的部分

F08/F10/F12 不等于已证明缺新事件。现有 issue/done、role.member、release、waits_for 可表达更细动作；但当前 `Main: warp[0:8]` 没有给每个内部 warp/subset 各自的事件身份。把扫描、搬运、merge和输出全部拆成角色，要协调同一物理 warp 不同阶段的复用，也要保留真实发布/退休保证。本轮未得到更短且等价的完整拆分稿，故归为 representation_cost。最关键的反例是：全 CTA 先等 Q/K 再scan，仍符合现有粗粒度动作顺序，却失去既定 subset overlap；同动作内部各 warp 立即写 final stage0，也不能凭 `serial` 解释为所有 warp 已完成 K 行读取。

F16 是更强的结构缺口候选：`keep_value invocation`、`handoff.storage: global`、worker 资源均不命名跨调用的 process-static 分配/host cache。源码 [3258](../work/extract_gdn/gdn_kernel.cu:3258) 的 grow-only workspace 及 [3316](../work/extract_gdn/gdn_kernel.cu:3316) 的无设备键 L2 cache，让不同调用共享容量、地址和选择状态。每调用重新分配、错误跨 stream 复用和当前设备重新查 L2 都会改变主要开销或选择，却无法仅由现有 invocation 结构排除。新结构仍只是候选，未在本次改规范。

F01/F13/F14 是有意开放语义：外部别名/只读写区域、投机 raw-byte 读取和 swizzle/原语完成保证。现有字段给它们位置，不等于已经以结构表达。公式中的 BF16 cast、scan tree、宽度合并次序则属于 FORMAL/数学定义，不能据此宣称扩语言。

保守短写候选

共 26 个精确、不重叠的局部候选，涉及 25 组。按独立声明计净减 4,523 字符（YAML 开放声明量的约 3.04%）；每组候选各计一次为 2,783 字符。没有把 whole MIXED、跨文件复制或 alias 实例扩张算入可删量；没有独立短稿等价审阅，因此只称候选。

| 原 quote | replacement | 保留依据 |
|---|---|---|
| `The residual uses the decayed incoming state; readout uses the updated state.` | 删除 | 同字段 delta 含 a*S，Out 读取 S(l+1) |
| `Inputs are finite; floating ranges additionally constrain g and beta.` | 删除 | DOC1 finite 默认和 g/beta.range |
| `No chunk is an independent chain worktile.` | 删除 | 四个单链节点 partition 无 chunk，iteration/math 保留串行 c |
| `Keeping this suffix in one action does not order those reads before other warps' writes.` | 删除 | 保留无 CTA join、精确 final-mini 冲突条件和无退休保证 |
| `Hierarchical rounded inverse with different merge order at the final level` | `Hierarchical rounded inverse` | 正式 math 完整保留 w32/w64 两种括号次序 |

所有完整 quote/replacement、适用 occurrence 和证据都在机器交付。最长 1,300 字符的 finish_chunk 只删上述解释句；真正的 warp 输出 sweep、RhsOutput alias、final stage0 写入和 mini 读取退休缺口完整保留。约 1,400 字符的 pass_c strategy、1,900 字符的 chunk math 和接近千字的 stage_inputs 也未按长度截断。

边界与限制

- 首次/末次：PRE0 首次只预取输入，后补系数不得二次 arrival；最终 Rh 即使无人读仍重建。没有下个 worktile（每 CTA count=1），没有域外 speculative future chunk。final_store 保留下一 float4 预载于当前 store 前的流水。
- 空分支：inactive prep 仍执行 metadata/trigger；空 A 发 EntryA/1 summary，强制 weak；C 的 weak z>0 只移除 run 主体。mini solve no-op 仍有事件。segmented_ordered 的 G=2/VTT16 排除 mini，并使 SegB1 无消费者，不能套用一般 G 的归约结构。
- 所有读者/alias：full path 的 K/Qd preload 和 CTA join保护 Hrhs→Out；普通环由下一轮 loop-top CTA join退休。final mini stage0 重用缺跨 warp K 行读取退休保证，源码只在写后 join，本审计没有替源码加边。
- 跨调用/捕获：源码入口没有显式 capture/replay 分支；依 DOC0，录图固定 host 选择和按值绑定，回放设备读取当次内容。没有图回放或跨 stream 证据，不能由同 stream 安全断言并发/回放安全。
- 3 组 UNRESOLVED 共 923 去重字符、6 个声明，覆盖所有节点的 dense-input require：源码确有 `.contiguous()`，现有 flow 未展开其可能的复制 kernel。缺匹配 PyTorch lowering/版本，无法据此写准前置提交、线程分配与编码失败前缀；这是来源不足及已有字段未用的待确认问题，不是已证明无语法可表达。
- 比较器公式、relative_to、算术精度、NaN/Inf及 core 检查路由未给；近似指数和 combined error 无全域界。非连续 core 写临时副本无 copy-back 已亲读源码确认。硬件占位、编译器 contraction 和未运行GPU限制均保留。

验证：`python3 -B scripts/summarize.py --operator extract_gdn` 返回 VALID：336/336 组，6/6 节点；所有证据行、候选唯一匹配及区间不重叠通过。机器汇总为 data/extract_gdn/audit_summary.json。只写本审计 operator/annotation/work 交付及统一脚本生成的 audit_summary；未动原 schema、规范、Runner 或 GPU。
