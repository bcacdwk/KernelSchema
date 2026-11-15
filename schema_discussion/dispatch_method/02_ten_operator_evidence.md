# 十算子的横向证据：设计身份不等于源码路径

第 1 轮独立研究。只读核对模板、归档 YAML、暂停清单及固定源码；没有提取、恢复运行、修改正式规则或执行 GPU 验证。以下是研究建议，不是已经批准的分组或最少节点数。

- **需要减少的是重复设计决定，不能拿文件数代替设计数。** 四个模板有 21 份 YAML，六个初始交付有 77 份；六算子暂停计划为 220、批准 145。它们是不同版本和范围，不能相加或直接解释成同范围膨胀。
- **FA4 prefill 有可以具体证明的合并正例。** `packed_persistent` 与 `single_persistent_dual_load` 使用同一请求前缀映射、同一 worker 分配与流水线；二分查找和单请求线扫改变查找成本，不改变工作归属。无需用两个完整设计记录同一规则在不同 batch 上的选路。
- **小变化仍可能是重要设计。** FA4 prefill 的共享 alpha 改变生产者和交接载荷；decode compact 改变空 split 是否读取、归约分组及 scratch 前提；GDN 单序列的 ready-only 标量环靠输出进度保护复用。它们不能以“只是优化”删去，也不必在每种组合上复制完整 YAML。
- **不同 topology 不自动意味着独立核心设计，相同 topology 也不证明相同设计。** GEMM 恢复链重复相同矩阵计算；FA4 decode 的 global/cluster 则改变物化、归约位置及 BF16 舍入。应检查新计算、所有权、数值和事件决定，而不是数 launch、producer 或 if。
- **组合必须保留真实耦合。** FMHA 融合 scan 的 `PartM/MTail` 必须配加权 combine；GDN 请求排序会禁止 PRE0；Sparse hidden 探测只允许不 split 的两个后端，且必须保留已执行的 packed 前缀。这些不是任意可交换的开关。
- **单 root 树可以做阅读起点，不足以表达所有真实设计来源。** Sparse 的 packed 前缀与 mma 余段是实际混合设计；FMHA 扫描方式和宽归并跨搬运后端组合。强行只认一个直接父会把组合重写成不同树枝，或让读者误认为另一来源不存在。
- **推荐少量完整代表设计，加有界的局部变化记录和来源选路引用。** 完整设计在声明的适用域内自洽；局部记录只承担解释和实施提示，不冒充已展开、已审阅的完整方案。选路阈值与实现恢复历史不再决定设计身份。不要用巨型条件 YAML 或几十条完整长说明伪装压缩。

## 1. 样本、版本与计数口径

`YAML 数`按任务目录直接包含的文件计数，包含 root，不计 PNG、receipt、standalone view 或重复归档。`暂停选定文件`按暂停 CSV 每个非退役节点的 `source_file` 计数；文件存在不意味着完成写作或获批。可重现脚本和逐文件版本、SHA、来源见 [inventory.py](evidence_cross_operator/inventory.py) 与 [inventory.json](evidence_cross_operator/inventory.json)。

| 样本 | 本文主要读取版本；root/其他节点修订号 | 初始 YAML | 本文可用的较新已交付 YAML | 暂停选定 YAML；批准 / 计划 |
| --- | --- | ---: | ---: | --- |
| FA4 prefill 模板 | 当前模板；root v6，其他 v4–5 | 11 | 同一组 11 | 不适用 |
| FA4 decode 模板 | 当前模板；root v9，其他 v7 | 7 | 同一组 7 | 不适用 |
| GDN prefill 模板 | 当前模板；root v7，单序列 v6 | 2 | 同一组 2 | 不适用 |
| QSA prefill 模板 | 当前模板；root v9 | 1 | 同一组 1 | 不适用 |
| GDN wiki | 初始 2.2.2：root v5/segmented v1；2.2.7：root v11，其他 v2–3 | 2 | 6 | 6；6 / 6 |
| MLA wiki | 初始 2.2.2：root v3；2.5.0：root v4/bootstrap v1 | 1 | 2 | 2；2 / 2 |
| grouped GEMM wiki | 初始 2.2.2：root v4，其他 v1；2.5.0：root v4，其他 v1–2 | 3 | 7 | 7；7 / 7 |
| Sparse wiki | 初始 2.2.2：root v4，其他 v1；2.5.0：root v13，其他 v1–3 | 3 | 8 | 8；8 / 8 |
| FMHA wiki | 初始 2.2.2：root v4，其他 v1；暂停工作 root v9，逐节点稿见 CSV | 8 | 仍为初始 8；43 节点轮未发布 | 43；26 / 43；另 4 待审、13 待继续写作或返工，后者含 5 份有 Design 正文但 `when` 为空的稿 |
| MoE wiki | 初始 2.2.2：root v4，其他 v1–2；暂停工作 root v7 | 60 | 仍为初始 60；154 节点轮未发布 | 111；96 / 154；其中 5 个存在文件仅为空 `when` 骨架，非创作成果 |

四模板加初始六 wiki 为 **98** 份；四模板加各算子最新已发布版本为 **112** 份；这两个数是替代口径，不能相加。六算子暂停计划排除 Sparse 6 个、FMHA 7 个退役历史节点后为 **220**，批准 **145**；选定文件存在 **177** 份。其中 FMHA 有 5 份带区域正文但条件为空的待续稿，MoE 有 5 份仅三区 `from_parent` 加空条件的骨架；这些不能算完整方案。GEMM 的 2.2.4 三节点是另一个历史修订，不加到七节点上。版本/发布关系依据 [归档总表 19–26 行][versions]、[暂停表 10–17 行][pause]；逐文件事实以统计 JSON 为准。

特别是 MoE，初始 60 节点包含显式 BF16、MEGA 等配置；暂停计划的 `when` 明确要求冻结默认实现控制。**60→154 不是同一范围的自然增长实验。** 本文只核对其清单和必要对照，不对 154 种组合的可达性或最少设计数作独立结论。[初始索引 5–23 行][moe-initial]、[暂停 root 8 行起][moe-paused-root]可直接对照范围。

来源身份也不能混用：

| 来源组 | 固定身份和核对方式 |
| --- | --- |
| 四模板 | `bcacdwk/KernelSchema@01fff90d714abc6e84904ea9c44523ba66d34a0e:kernel_wiki/aka-four-problems-20261019/...`。这批源码在当前工作树中已经不存在；本文用 `git show <ref>:<path>` 读取，末尾链接固定同一 commit。统计 JSON 留有原路径与源码 SHA。 |
| 六 wiki 初始 | `source@b5c8e870e5534ae629325e56aa0ba28c3f00486f:kernel_wiki/aka-cuda-operators-20261030/operators/...`；均为工具 2.2.2 初稿。 |
| GDN/Sparse 后续及 FMHA 暂停 | `source@6e4ca91bd91d998cac99e8dadd83838e153c679d:...`；本文引用的本地 `.cu/.cuh` 已与该 commit 字节比对。 |
| MLA 2.5.0 | `SchemaExtractSource_f91523a50e3d@228934a2253bde4d6f0c4463dd77aeada1bf07a8:...`；本地源码与该 run 冻结 `source` 文件字节一致。 |
| GEMM 2.5.0 | `SchemaExtractSource_2c7b5a74c0b3@497a8077ac54ed1431bc00772871337f9a309852:...`；同样比对冻结 `source`，不假定该 commit 属于主仓库对象库。 |
| MoE 暂停 | `SchemaExtractSource_c98aa54e58ac@ce1b7d5e8e7560b5c80abd87549bc2328c52df34:...`；本文只用暂停清单、root 范围及初始索引作数量/范围对照。 |

历史 YAML 中的 `phase/resource/repeat` 等是**证据原文**。现行语法以 DOC0、DOC3–5 为准；本文不把历史稿当新语法模板，不通过运行现行检查器给旧稿追加批准。

## 2. 十算子的设计轴总览

下表的“可压缩”不等于删去该事实；参数化、一次性记录局部变化、引用已有定义、只引用原实现选路，承担的职责不同。

| 算子 | 已存在的划分轴 | 应保留的核心设计信息 | 可压缩或移出独立设计身份的部分 |
| --- | --- | --- | --- |
| FA4 prefill | static/persistent；矩形/请求折叠；输入序/LPT；单/双 K/V loader；共享 alpha；split-KV | lagged-max FP8 online attention、两 CTA 所有权、QK 领先 PV、跨工作环寿命；split 归一化 partial 与独立 combine | 请求查找算法、单请求退化、split 数和选路长公式；LPT、共享 alpha、joint-loader 协议各写一次，不能随调度轴重复扩成整棵树 |
| FA4 decode | q1/q4；global/cluster/direct；q4 compact | 两 FP8 概率分量、双分子 parity；FP32 partial+PDL combine；DSMEM 统计+BF16 reduce-add；一块直接归约 | split/cluster 成员数；q1 两 softmax 与 q4 单组分工的重复组合；compact 是需保留的局部归约/空载荷差异 |
| GDN 模板 | packed 多序列/单序列 | chunk inverse 的 BF16 merge 舍入、FP32 状态；双计算组/双 issuer；标量与输出共享 warp | 单/多序列不是只改 B：描述符边界、标量 TMA 和 ready-only 复用需保留；只有两份，暂无迫切减文件需求 |
| QSA 模板 | 单 root；若干 API 输入分支不分设计 | query 行并集、membership mask、persistent 调度、双 softmax 半区；attention 先 BF16 再 gate 再 BF16 | `use_prefill_config` 被源码忽略；`out` 分配与否不改变计算设计；拟合域的限制不能因一个 root 被省略 |
| GDN wiki | cached/direct QK/segmented × ordered/unordered 六种合法组合 | prep 系数与 Q/K 图像；直接输入 TMA；分段 summary→入口组合→重跑；排序生产者与 PRE0 限制 | VTT、环深、GQA；同一个排序决定不必三份完整重复。segmented 与 direct-QK 不能自由组合 |
| MLA wiki | queued persistent/bootstrap；每节点内还有排序、K/V 三/四槽、sink/LSE/tiny-row | Queue 服务、终止描述符、跨工作 Q 预取；bootstrap 无补领；FP16 指数/局部和、FP32 分母递推及 FP8 PV 权重舍入 | 排序和环深已经在节点内；不能把“两份 YAML”进一步简化成仅 worker_count 不同 |
| grouped GEMM | UMMA V0 / mma.sync / wide；恢复计算次数 1–4；wide F/G 已参数化 | block-scale FMA 次序、K 成对或逐块推进、异步产品所有权；wide 的暂存/环深取舍 | graph/ctypes 入口本身、缓存命中、恢复重算次数；后者需要实际提交记账，通常不新增核心 GEMM 设计 |
| Sparse | split mma/direct mma/tc_heads/tc_packed；serial/hidden probe；probe 后端选择 | 双池并集与重复次数、QIS2 两 QK issuer、sink 生命周期；split/直接归约；投机 packed 前缀、side stream、join | head/KV/split/DQ/ring 数；probe 选择结果与同一后端组合重复。已提交 probe 不能随失败结果删除 |
| FMHA | 融合 7 类；mma split 4 类；D128 tcg split 24 类；D256 tcg split 8 类 | delayed-max 递推及概率舍入；融合 key/group 分工；外置/内置 scan；partial 格式与加权位置；TMA/cp 退休关系 | scan×copy×写回×combine 的重复乘积；准确接口仍需记录。不能把43个计划行叫43个已证独立方案 |
| MoE | 初始 FC1 组织 × FC2 组织、准备/激活/last-arriver、BF16 路径；新计划再含 ABI 生命周期等 | 路由、量化/激活边界、跨 CTA 合并、FC1/FC2 消费关系；详细独立性另作专项研究 | 组合是数量的重要来源，但本文没有验证每条轴可独立替换，不给压缩后数字 |

表中每项均由以下展开核对支撑；MoE 只承担明确标注的对照，结论不依赖其专项分析。

## 3. FA4 prefill：一个合并正例，与两个必须保留的局部决定

### 3.1 单请求 persistent 并非新的工作映射

解析后的 `packed_persistent` 与 `single_persistent_dual_load`：`math_reshape`、`tile_pipeline` 完全相同；`work_schedule` 只有 `phase.attention.describe`、`phase.attention.schedule.strategy` 两个字符串不同。**这个结构统计只定位问题，不单独证明语义等价。**

两份稿都定义 `C_b=ceil(Qlen(b)/16)`、`Cpad=ceil((TotalQ+15*(B-1))/16)`，按请求前缀解出 `(b,m)`，同一矩形折叠规则，同一 worker 起点及固定步长。同一请求序都是 identity。具体差异是 5–16 请求用四步二分，B=1 用线扫。[packed 74–97 行][fp-packed]、[single 74–97 行][fp-single]。

固定源码 `_persist_remap_coord_hoist` 1213–1250 行可检验：两种查找都返回最大的 `seg` 使 `prefix[seg] <= cl`；`local_cl=cl-prefix[seg]`，`mapped_batch=req_map[seg]`。有效工作内，B=1 的线扫次数为零，`seg=0`，`Cpad=C_0`；把 B=1 代入多请求公式直接得到同一工作映射。源码二分也是对 `[1,16]` 的同一个单调前缀表求解；后续折叠没有因查找算法改写。[固定源码 1170–1255 行][fp-map]；[host 1290–1300 行][fp-bsearch]。

**建议分组：**一个“保持请求次序的 folded persistent”设计可覆盖两种适用子域；保留 map 的值、构造时机、容量和一次发布，查找用线扫还是二分可留作实现选择。源码当前选择哪些 batch、长度阈值，不需要两份完整 Design 才能追溯。没有据此证明 LPT、joint load、split 可以任意组合。

### 3.2 alpha 共享是真实设计变化，但不必随 static/persistent 各复制一次

普通路径中，Softmax 发布两列 `Stats=(old maximum, base)`，Correction 读取后计算 alpha；共享路径发布一列 `Stats=alpha`，Softmax 在更新分母时也使用这个已计算值。Y 仍由 Correction 在前一 PV 完成后 rescale，原有 Stats 捕获/槽位归还与 Y 所有权握手仍成立。[root 181–287 行][fp-root-pipe]、[共享稿 39–62 行][fp-alpha-math]、[共享稿 113–129 行][fp-alpha-storage]。

源码并非仅改名称：3523 行先 acquire Stats，3525–3545 行计算和写一列 alpha；3644–3648 行复用 alpha 更新分母；3689 行等待 Stats，3692 行起读取对应的一列布局。[固定源码 3523 行][fp-alpha-source]、[3644 行][fp-alpha-use]、[3689 行][fp-alpha-read]。

应该保留的最短记录是：`Stats` 新含义/形状、唯一生产者、两类消费者、旧/新数据通路和仍须成立的槽位/Y 握手。它改进指数计算复用和传输，不需要用新“完整 attention 算法”夸大，也不能简单标成 `alpha=true` 而省略接口。

### 3.3 单/双 loader 与 split 必须按实际协议理解

root 的 K 和 V 使用各自四槽流、两位生产者；joint load 把两类搬运串入与 QK 领先 PV 对应的共同 token 顺序，K/V 的 slot index、等待目标、退休事件都改变。源码 `split_kv_loads` 控制第二个 loader 及独立 V pipeline，现稿 `rect_joint_load` 的共同 `KVStage` 序列体现了它。[root 183–245 行][fp-root-pipe]、[joint 96 行起][fp-joint]、[源码 1306–1387 行][fp-loader-source]。这是一种值得说明的不同局部通路，不能叫普通 tile 参数。

split-KV 则增加跨 CTA FP32 partial/LSE、归一化后的部分输出和独立 combine；它是端到端分解的显著变化。[split 稿][fp-split]。将“单次 attention”和“split+combine”分别做完整设计很合理；是否将 static/persistent 分别完整保留，是阅读/实施粒度的选择，不能只由 `worker_count` 决定。persistent 的环 phase 和 Q/Y 复用跨工作延续，必须解释。

## 4. FA4 decode：同一数学任务不意味着同一种数值和同步方案

| 对照 | 已核变化 | 判断 |
| --- | --- | --- |
| global → cluster | global 物化 FP32 `Mpartial/Lpartial/Opartial`，combine 读齐后归约；cluster 交换 DSMEM max/sum，member 0 清零输出后，每个有效 split 将归一化贡献先转 BF16，再无固定累加顺序地 reduce-add | 应保留不同完整方案，或至少保留可独立审阅的整段数值/collective 设计；不能只把两者看成 split 数变化 |
| global → direct | 仅一个 KV block；CTA 内结束计算，不存在 split workspace、collective 或第二 kernel | 可以做一个清楚的单块完整方案；不能保留 global 的提交后用正文把 combine 删掉 |
| q4 → q1 | q4 每请求两个 query tile、单 softmax WG；q1 一 tile，两 WG 交替接收 score、发布概率与统计，奇偶状态/交接不同 | 属于重要局部分工方案；无需自动将这一变化与 global/cluster/direct 的每一种都升级成独立基础知识 |
| 普通 global → compact | 普通 combine 对全部 nominal split 读 Opartial，空 split 权重为零，要求未写的 scratch 保持有限；compact 只读有效 split，且每个输出线程串行重算 max/denominator | 即使两 kernel topology 相同，也有额外 scratch 前提和归约方式变化，必须记录 |

global 的明确式子是 `U=sum_{r=0}^{SplitCount-1} w[r]*Opartial[r]`，并说明空 payload 被读取且应有限；compact 是 `sum_{r in R}`。这决定了 `0*NaN` 一类边界能否被跳过，不能作为纯排版变化省略。[global 161–190 行][fd-global]、[compact 85–112 行][fd-compact]。

cluster 稿 83–100 行显式记录 BF16 窄化及未确认的 native reduce-add 内部算术；216 行的写回等待 member 0 `zero_done`，不能用普通 CTA barrier 代替。[cluster 83 行起][fd-cluster-math]、[216 行][fd-cluster-store]。固定源码 `reduction_cluster` 用 xor peer、DSMEM copy 和 mbarrier 逐轮交换，而 q1 的两 softmax WG 建立 phase 互斥及首轮特殊释放。[源码 248–283 行][fd-cluster-source]、[paged 1568–1579 行][fd-q1-source]。

**建议分组：**global、cluster、direct 是三条易理解的端到端主线；q1/q4 分工和 compact 合并方式分别保留一次有边界的变化说明。是否把 compact 再保留成完整实例，取决于实现者是否需要直接领取这条方案；它的数值/初始化差异必须先写全。不能声称已有七份稿可无条件变成三份同等完备的当前 YAML。

## 5. Sparse：真实多来源组合，与“探测失败就没有探测”的反例

较新八节点有四种注意力组织：split mma、direct mma、单 query 的 tc_heads、多 query 的 tc_packed。后者先构建按池组织的索引并集，保留每个 query 的成员/重复次数，QIS2 使用两套部分 score 后再相加；源码 cfg90 给 GT=1/2/4/8 的实例，但 cfg91/92 只接受 GT=1，已经说明“同后端名称”不是任意 packing 的保证。[索引 5–12 行][sparse-index]、[源码 5927–5947 行][sparse-qis]。

其中 `probe_hidden_mma` 特别重要：

1. side stream 等待 caller 之前的输入生产；probe 在 side stream 提交。
2. caller 不等 probe 结果，先按 **packed tcgen05** 计算 `[0,K*GT)` 前缀。
3. host 实际读到的 ratio 不接受 packing 时，side stream 在 probe 之后用 **mma.sync** 计算 `[K*GT,T)` 余段。
4. 余段不等待前缀完成；caller 最后等待 side join，使之后工作看见两段输出。

这是源码 `dual_dispatch_hidden` 6153–6190 行的真实组合，不是理论上的“多继承可能有用”。YAML 的 math 同时包含 packed 和 mma 两套计算，flow 为 probe、prefix、remainder，并明确 remainder 只等 probe。parent 为 `probe_hidden_packed` 只表达了一侧来源；另一侧的计算方案来自 mma direct。[源码][sparse-hidden-source]、[YAML 342–367 行][sparse-hidden-yaml]。

组合还有严格接口：`pl.splits=pl.alt_splits=1`，调用不能预先缩小 token 范围，`K=SMCount-8>=1`，至少两波 groups；两段输出区间不重叠，各自独立局部状态；fork/join 不可遗漏。因此“把 probe、packed、mma 随意拼装”不是压缩方案。

serial probe 的统计不可用与 `probe_begin` 根本未提交是两件事。源码先 `probe_mean_ratio` 再选择后端，YAML 保留实际 probe；hidden 选择 mma 也不能抹去此前已跑的 packed 前缀。[源码 6433–6453 行][sparse-serial]。sink 同样必须保留到最终归一化：源码 1131–1134 行保存 `sink2_s`，1525–1526 行使用它；它不是一轮 softmax 后就能复用的临时区域。[sink 源码][sparse-sink]。

**建议分组：**先保留四种计算设计；hidden 的投机前缀和跨流接续值得一个完整、具体的组合实例。serial probe 的共同控制过程只记录一次，原选路结果到后端的关联引用原索引即可。压缩后不承诺完整复刻每条异常/超时路径；若任务要求复现该调用，必须补齐它真实发生的提交。源码 gate 复用竞争、DQ 重复写及外部验收缺口仍是限制，不能因设计压缩消失。

## 6. FMHA：43 是组合展开数，并非已经证明的独立设计数

直接解析暂停计划：7 个 fused + 4 个 mma split + 24 个 D128 tcg split + 8 个 D256 tcg split = 43。D128 的24来自 `3 scan × 2 copy × 2 partial-store × 2 combine`，D256为 `2 scan × 2 copy × 2 combine`。这只是**计划分解**；本文核对了代表源码和选择稿，未证明每个组合都有可达输入，更未证明都可自由互换。[暂停计划 5–55 行][fmha-plan]。

对比 `tcg_d128_flat_tma_xpose_narrow`（已批准 v4）、对应 cp/宽归并/FSCAN/segmented 四个选择稿（均为待审 v2；各自 checkpoint 见统计 JSON），可以从 actual flow 读出：

| scan 组织 | 计算提交 | 物化与消费变化 | 不能丢的接口 |
| --- | --- | --- | --- |
| flat | group_max → flat_scan → split → combine，4 次 | `GroupMax→MState/MTail→PartA/PartB`；split 已把 partial 乘回全局 delayed-max 参考 | split 消费 scan 完成后的状态；combine 消费已加权 partial |
| segmented | group_max → segment_max → segmented_scan → split → combine，5 次 | 新增 `SegMax`，筛除不必重放的 prefix 读取，仍形成完整 `MState/MTail` | segment max 与 replay 的依赖；不是把 flat 的线程数调大 |
| FSCAN | group_max → split-local scan → weighted combine，3 次 | 不生成全局 MState；各 split 重放 prefix、写未加权 partial 和 `PartM`；末 split 写 `MTail` | `PartM/MTail` 的生产者迁移；combine 必须启用 FWGT，不能在 split 内跨 CTA 读取刚发布的 MTail |

源码 1640–1692 行把 scan、split 和 combine 一起选择；`fuse_scan` 取消的是整条 scan 链。`fmha_tcg.cuh` 5181–5222 行解释并执行末 split 写 MTail、split 写未加权 partial，5480–5488 行写 PartM；`fmha_chunk_wgt` 在 combine 侧计算 `exp(ms-mtail)`。源码还记录了先前跨 CTA 读取竞争与无界指数溢出。因此 FSCAN 是**带耦合的算法/物化改写**，不是一个能脱离 combine 的 flag。[提交源码][fmha-launch]、[FSCAN epilogue][fmha-fscan]、[PartM 写回][fmha-partm]、[combine 权重][fmha-weight]。

copy、partial-store、combine 也不是无意义的名字：

- TMA/cp.async 代表稿有相同四提交与 global handoff，但 payload/scale 的完成等待不同；fused 的 TMA 还允许延迟 PV 退休，而 cp 版本逐组退休。不能只引用 TMA pipeline 然后把动作名换成 cp。
- D128 非 PSK 转置写回复用已空闲 Q/K/V shared 区、加 CTA join，再由全体线程搬出。源码明确 staging 只对 split 有足够空间；搬到 group-max 会越过静态容量约束。[fmha_tcg.cuh 5225–5246 行][fmha-xpose]。
- wide combine 由四组分担 split，再经 shared partial/join 汇总；narrow 有每 split 的 shared 容量限制。源码宽归并条件既有成本阈值，也有 `nsplit > FMHA_COMBINE_NARROW_BT` 的正确性边界。[launch_combine 916–959 行][fmha-combine]。

同时，fused key-split 与 group-alternating 是不同的持有/交接策略：前者两 WG 处理同组 key、交换组最大值；后者两 WG 轮流处理整组、提前交接 max、之后交接 denominator。其 score、K/V ring 和归约顺序不同。这些局部方案有独立理解价值，不能为凑少数节点全部压成一个 `KSPLIT` 数量参数。[已批准 key/group 稿的 thesis/资源][fmha-key]、[fmha-group]。

**建议分组：**保留 fused、外置 scan split、内置 scan split 三条端到端主线，再按上述融合分工选择必要完整实例。segmented scan、转置 partial、宽归并与搬运协议各记录一次具体变化及前提；对已出现的组合给短来源链接，不重写整个模型。本文尚未逐项核对七种 fused 的合并边界，**不填一个精确压缩总数**。43 的独立最小性与全部可达性也未获证明。

## 7. GDN、MLA、GEMM、QSA 对照

### GDN 模板：同一核心算法下，也有不能删的协议变化

两份模板都是64行 value/state 切片、chunk inverse 和两计算组/两 issuer，数学区主线相同。B=1 用固定描述符与 scalar TMA，RawScalar 为两槽；处理后的 G/Gamma/Beta 共用五槽 ready-only 环，最多提前四 chunk。关键保证是 `prepare(c+5)` 在 `store:O(c)` 之后，借输出进度证明旧标量读者结束，而非增加每个 reader 的 empty 返回。[单序列 222–246 行][gdn-template-scalar]；源码构造条件明确 `use_scalar_tma=split_value_rows && use_static_tma_descriptors`，`use_ready_only_scalar=use_scalar_tma && compact_cg1_events`。[源码 239–259 行][gdn-template-source]。

多序列则要更新请求边界，Gate/Beta 分别有读者退休；两者仍都执行前置 `cu_seqlens` 转换，即使单序列主 kernel 不消费其结果。不能从“不消费”推出“未提交”。现有两份已足够少，保留两份或一份完整加精确变化说明均可；不会因为“B不同”继续拆分对齐/初态情况。也不能把本模板与 wiki GDN 当作同 Contract 的两条路径：硬件及任务/数值要求不同，模板说明已明确区分。[模板说明 18 行][template-readme]。

### GDN wiki：三种数据通路 × 排序，但排序有启动耦合

cached 的 prep 生产 Qc/Kc 图像，direct-QK 删除这两个 global handoff、chain 从输入 TMA 读；segmented 则物化 SegA/SegB，MODE1→MODE2 两次 chain，通过分段摘要组成入口后重跑输出。这三类有明确的数据通路或算法差异。[direct-QK 183–202 行][gdn-direct]、[segmented 119–150 行][gdn-segmented]。

排序 `perm` 由 prep 生产，chain 必须等完再知道请求身份；这会禁止依赖预先知道请求的 PRE0。源码 `kqtma` 仅在 `Gseg==1`，`pre0` 要求 `!kqtma && Gseg==1 && vtt==16 && perm_ptr==nullptr`。[源码 3320–3354 行][gdn-wiki-source]、[ordered handoff 181–201 行][gdn-ordered]。可把排序记录成一个有启动约束的局部决定，避免在三条完整算法上重复；不能把它概括成任意纯排列，或让 direct-QK 与 segmented 自由相乘。

### MLA：保留两个完整设计有理由，原因不只是 grid 数量

host 确实按 `2*items <= 3*nSM` 选择 bootstrap grid，否则限于 nSM。root 有 Queue server、原子补领、终止描述符、跨工作 Q 预取；bootstrap 删除这些职责而保留本工作依赖。`bootstrap.math_reshape` 已整区引用 root，另外两区仍本地。[host 1998–2009 行][mla-host]、[root refill 342–351 行][mla-refill]、[bootstrap 17–27 行][mla-bootstrap]。

这是“少量完整方案+共用数学”的好实例。要进一步合并，必须说清 Queue/终止/预取如何消失，不能只让 `worker_count=items` 并宣称其他不变。另一方面，排序、三/四槽和可选数值功能已留在各自节点内，说明现行规则本来就没有要求每个参数各一份 YAML。[MLA 索引 12–16 行][mla-index]。

### GEMM：恢复历史放大执行节点，通常不增加计算设计

V0、mma.sync、wide 有不同 tile、issuer、产品存储、K-loop 和尺度 FMA 组织，值得保留三种核心实现策略。wide 已用 `J=1/2` 统一 F/G：暂存 `32*J KiB`、输入环 `5-J`、写回批数 `2/J`，不是每个数值各一份。[wide 191–226 行][gemm-wide]。

恢复节点则将同一次请求的实际计算列表分成 `[V0,V0]`、两次 wide、三次 wide、`[G,G,F,F]`。源码依次尝试 G graph→G direct→F graph→F direct，确实可能留下已提交前缀；API 返回错误本身既不能证明已经执行，也不能删去已经执行的那次。[源码 1718–1818 行][gemm-recovery-source]。`wide_recovery_full` 数学说明是四次相同类型乘积、各自完整覆盖 out、无前次结果依赖。[恢复 YAML 34–61 行][gemm-recovery-yaml]。

需要准确保留的事实是：最多四个实际计算调用位置、同 stream 顺序、各次局部资源独立、后次重写输出、仅当确有提交才计数。**这不意味着四次重算值得一份新的完整 GEMM 数学/微流水线。** 面向设计知识可记录一次恢复约定及原调用链；面向逐调用执行复现才需要 L 的精确实例化。本文没有验证每个错误码组合在驱动上可达，也没有将失败路径当作有效数值方案。

### QSA：单 root 没有缺少设计信息，但仍须保留数值边界

QSA 的完整主线已有 preprocess union→persistent attention+gate；membership、双半分母、共享 lagged reference、ticket/资源交接都在一份稿内。源码明确 `del use_prefill_config`，`out` 为 None 时只分配同形状缓冲，计算相同。[operator 49–55 行][qsa-operator]。这些调用分支无需新设计。

不能因 attention 与 gate 融合就删掉中间 BF16 舍入：实际 `attn=o.to(self.dtype)` 再乘 sigmoid，最后还要 BF16 store；YAML 明确两次舍入。[YAML 127–134 行][qsa-math]、[kernel 628–640 行][qsa-gate-source]。单 root 也不意味着泛化到所有 QSA workload：该稿只代表 single-request fitted-step15 的既定范围。

## 8. 如何认定“独立完整设计”

建议把判定对象定为**能够独立审阅并实施的一组新决定**，而不是源码条件的一个叶子。审阅至少能回答：计算什么以及精度；谁持有中间量、何时物化；谁取得工作；主要资源何时可写、何时可复用；相对已知方案为何可能受益、代价在哪里。

| 问题 | 本文建议 | 核对例子 |
| --- | --- | --- |
| 所有 topology/producer/launch 差异都新建 YAML？ | 不自动。先判断是否新增端到端分解，还是可定位的一次变化/重复执行；需要完整独立交付时再展开。记录层级改变不能抹去实际事件。 | GEMM 重算可一次说明；FA4 split 的跨 CTA partial 值得完整方案 |
| topology 相同可以合并？ | 不够。还看数值分组、物化值含义、读者/写者与资源寿命。 | decode compact 的 empty scratch；FMHA key/group 分母分组 |
| 局部创新何时重要？ | 能改变性能因果、精度、并行依赖或一个必要正确性前提，就必须记录；篇幅小和影响小无必然关系。 | alpha 单一生产者；GDN ready-only；QSA 中间 BF16 |
| 设计相同但适用域不同？ | 合并设计身份，用适用条件列出限制；运行时阈值/缓存规则作为来源选路引用。 | prefill folded persistent 的 B=1/5–16 |
| 相同输入有两种有效方案？ | 设计适用条件可以重叠；只有某个具体 selector 必须决定当前选谁。 | FA4 static/persistent 改分配而不改各 worktile 数学，源码保留两种编译并给出同输入 A/B 记录；不是数学契约迫使阈值一侧只能成立一个方案 |
| 只有多种“实现了的路径”能否称有效方案？ | 不能。来源可达、完整表达和满足 Contract 是三项不同判断。 | FMHA mma group-max 漏覆盖、Sparse probe 竞争、GEMM 错误重试均有来源限制 |

FA4 `interface.py` 1146–1185 行把 static/persistent 当同输入下的调度比较，静态 stride 的代价是负载不均；它不是只在“被选中”那一侧才数学成立。[固定 source 1146 行][fp-persist-policy]。这里的证据支持**设计域可重叠**，不替代对每个范围的 GPU/误差验证。

DOC0 当前要求 `when` 是选择充要条件，所有节点独立求值、互斥穷尽，直接父是设计来源；它适合记录一个确定实现的完整 dispatch。若目录要保存可竞争的设计，把这种 selector 约束施加给设计本身就不合目标：会强迫知识目录重放原阈值，或人为划掉另一方案的适用域。[DOC0 49–52、111–132、144–162 行][doc0]。建议概念上分开“设计成立的输入/调用前提”和“某来源本次选择依据”；暂不在本报告规定新字段名。

现行有限 `for/if` 已能表达局部行为；DOC5 的 `if` 明确要求判定值已就绪，no-op 的完成事件仍存在。[DOC5 286–301 行][doc5-if]。它不能成为把所有工作提交和后端方案的并集塞进一份 YAML 的理由。DOC4 又明确一次 step 是一个实际提交实例，重复计算不是多走一次 worktile。[DOC4 提交身份 243 行起][doc4-launch]。若目标是设计压缩，需要修改目录的交付职责，而不是偷偷绕过现行规则。

## 9. 两种组织候选与推荐

| 候选 | 完整 YAML 的单位 | 局部/重复信息如何保留 | 代价与适用范围 |
| --- | --- | --- | --- |
| A. 保持完整执行分区，先压缩同义选择与公共定义 | 每个不可统一的实际提交/协议组合仍完整；首先合并 folded persistent 这类同规则子域 | 相同 Design 区域引用一次；所有原当次选路、异常前缀仍逐节点记录 | 能保持精确源码路径覆盖；少量身份可消除，但 FMHA 的组合数和 GEMM 的恢复节点仍在。整区复用主要减少文字，不保证减少需理解的方案 |
| B. 以设计知识为主的少量完整实例 | 一份代表一个值得独立学习、比较和实施的方案，明确适用域；可以共享 Contract、可以适用域重叠 | 不重复的局部决定写一个短记录：定位对象、改变什么、必要前提、收益/代价、来源；已有完整端到端组合给短引用；原 selector 用固定源码引用 | 直接降低独立维护的设计数。主动放弃每次原实现调用都能从目录唯一选出一份完整 YAML 的承诺。需要实施局部组合时，再完整展开、核接口并审阅 |

**推荐 B。** 这与 README “YAML 供人和 Agent 阅读、不被编译，尽可能压缩核心信息”的目的更近。A 可以作为不改变当前交付含义的保守过渡，但无法只靠 `from_parent` 实现用户所要的少量核心设计。

| 记录深度 | 具体内容 | 减掉之后仍须保证什么 |
| --- | --- | --- |
| 不能丢 | 算法/中间值的含义与精度、global 物化、输出舍入、工作所有权、资源退休、跨流/跨 CTA 接续、已知来源限制 | 读者能从完整设计判断算了什么、何时可读、何时可重用；如 FSCAN 的 PartM→权重、cluster 的 BF16 累加 |
| 可从设计正文直接省略 | 不改变所选设计的缓存键字段、Python/ctypes 包装分支、排序网络逐次 compare-swap、未采用的诊断宏和实验配置列表 | 保留真正的工作映射、顺序要求与已选资源条件；不承诺照抄原实现的缓存/诊断设施 |
| 用来源引用保留 | 原 host selector 的长阈值式、同算法的编译实例名、图回放/直接提交如何进入同一计算、性能实验原始记录 | 精确复现原选路时仍能找到固定 source；完整设计自身不能靠源码函数名代替必要计算或参数定义 |
| 用一次局部记录保留 | alpha producer/Stats 变化、compact 的空载荷与归约规则、GDN Perm/PRE0 联动、GEMM 恢复调用链 | 说明改变对象和必要前提，不自动宣称它在每个后端都适用，也不冒充已展开的完整 YAML |

局部记录应有严格边界，不建通用组件语言、不定义 patch 执行顺序、不声称自动可组合。以 alpha 为例，四句足够：一列 FP32 Stats 由 Softmax 计算并发布；分母与 Correction 复用同一值；Stats 覆盖仍须等 Correction 捕获、Y rescale 仍须等上一次 PV；收益是消除 Correction 重算与一列传输，来源仅确认现有 static/persistent 两种组合。它不是“几十份 YAML 各改写成一页”。若一个变化要重写多数数学、资源和事件，直接保留另一份完整设计。

可试写的分组规模与**已经核实的计数**分开如下；这只是下一轮候选，不是本文已经完成的 schema 转换：

| 样本 | 建议完整实例的起点 | 只记录一次的变化/边界 |
| --- | --- | --- |
| FA4 prefill | 3 个起点：static、folded persistent、split+combine | joint-loader 完整局部协议；alpha 共享；LPT 请求顺序；二分/线扫实现选择。3 不宣称覆盖11份稿全部组合的同等表达能力 |
| FA4 decode | 3 条主线：global、cluster、direct | q1 双 softmax 交替；compact 合并；需要直接实施 compact 时可保留第4个完整实例 |
| GDN 模板 / QSA / MLA | 维持 2 / 1 / 2，先去重复、不追求继续减数 | 准确保留上述标量环、gate舍入、Queue/终止差异 |
| GDN wiki | 3 个起点：cached、direct QK、segmented | 一份排序/Perm/PRE0 联动说明；禁止 segmented+direct-QK |
| GEMM | 3 个起点：V0、sync、wide | F/G 参数关系；一份恢复调用链与实际提交记账说明 |
| Sparse | 4 个计算方案，加1个具体 hidden 混合编排示例 | serial probe 过程及选择结果链接；hidden 不能删除 packed 前缀；不推广 split hidden |
| FMHA | 先按3条宏观链和各融合分工试写，**完整实例总数待验证** | scan/combine 配对；copy 退休；partial 转置；分母分组。不能硬凑一个小于十的数字 |
| MoE | 本横向研究不定数量 | 需用专项源码结论确定 FC1/FC2、准备与 ABI 生命周期之间的真正边界 |

这张表不相加为一个“十算子最少节点数”。其中完整实例与局部记录的分界，需要下一轮实际试写来验证读者能否正确实施。

## 10. root、共享 Contract 与来源关系

root 作为共同 Contract 容器、默认阅读入口可以保留；**“其他设计必须都是它的单父派生后代”不是 GPU 设计的事实**。

FMHA 的 tcgen05 原生 MXFP8 产品与 mma.sync 显式 block-scale、BF16 V/PV、寄存器分子是同 Contract 下的不同实现架构；后者不需要从 QK64/TMEM root 逐字段变出来才成立。GEMM 的 sync 与 UMMA 同样共享矩阵 Contract，却没有必须服从 TMEM 主线的设计来源关系。树可以任选一条线为阅读起点，不应把目录位置解释为算法继承证明。[FMHA tc_fused][fmha-tc-fused]、[GEMM 索引 5–8 行][gemm-index]。

真实的多来源组合有两类：

- **一份调用同时采用多个设计：**Sparse hidden_mma 的 packed 前缀与 mma 余段，上文已核代码、区间和 join。
- **同一个局部决定在多个完整设计复用：**FMHA 宽归并同时出现在 flat、segmented、FSCAN，以及 TMA/cp 路径；FSCAN 还要求 combine 的加权形式。prefill alpha 同时出现在 static/persistent；GDN permutation 同时接到三种数据通路，但带 PRE0 限制。这些关系不是树上一个父节点能完整表明的。

推荐把三种关系分别表达清楚：共用哪个 Contract；实际参考/借用了哪个设计决定；实际复用了哪份内容。它们可以在短索引和来源说明中呈现，不必先引入多继承语言。共享只减少重复书写；使用者仍应检查输入、数值、buffer ownership 和事件接口，不因原文相同或链接存在就自动获批。[DOC0 126–132 行][doc0]已经包含“原文相同不证明新组合成立”的关键约束，应保留。

## 11. 证据边界与下一轮最小验证

已核：十个样本的文件数/元数据版本；六算子暂停状态与源文件摘要；FA4 两类映射的实际解；alpha 数据通路；decode global/cluster/compact 和 q1交接；Sparse hidden/serial 提交与实际混合；FMHA 三种 scan 链及耦合、代表 copy/store/combine 差异；GDN、MLA、GEMM、QSA 上述源码/稿件对照。

建议而未批准：完整实例候选、把局部变化降成记录、允许适用域重叠、root 只作入口/Contract容器。尚未验证：全部 FMHA/MoE 组合可达性、最少独立设计数、所有建议组合的数值误差/GPU正确性、文字和调用成本节省比例。源码注释中的性能数字只用作说明设计意图，没有重新测量，也不当成压缩方法的性能证明。

下一轮最有辨别力的是两次小试写：一是将 folded persistent 的 B=1/5–16 合成一份完整规则，再用 alpha 记录检验能否正确保留接口；二是用 Sparse hidden_mma 或 FMHA FSCAN 写一个明确多来源、带实际接口约束的组合。若试写仍需搬入大半原稿，就保留完整方案，而不是继续扩大简记语言。

[versions]: ../../schema_wiki/archive/six-operators/README.md#L19
[pause]: ../../schema_extract/archive/six-operators/20261109-pause/README.md#L10
[template-readme]: ../../schema_template/examples/README.md#L18
[doc0]: ../../schema_docs/0.common.md#L103
[doc4-launch]: ../../schema_docs/4.design.work_schedule.md#L243
[doc5-if]: ../../schema_docs/5.design.tile_pipeline.md#L286
[fp-packed]: ../../schema_template/examples/fa4_prefill/packed_persistent.yaml#L74
[fp-single]: ../../schema_template/examples/fa4_prefill/single_persistent_dual_load.yaml#L74
[fp-root-pipe]: ../../schema_template/examples/fa4_prefill/root.yaml#L181
[fp-alpha-math]: ../../schema_template/examples/fa4_prefill/small_persistent_shared_rescale.yaml#L39
[fp-alpha-storage]: ../../schema_template/examples/fa4_prefill/small_persistent_shared_rescale.yaml#L113
[fp-joint]: ../../schema_template/examples/fa4_prefill/rect_joint_load.yaml#L96
[fp-split]: ../../schema_template/examples/fa4_prefill/splitkv_dual_load.yaml#L16
[fp-map]: https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py#L1170
[fp-bsearch]: https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/flash_attn/cute/interface.py#L1290
[fp-alpha-source]: https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py#L3523
[fp-alpha-use]: https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py#L3644
[fp-alpha-read]: https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py#L3689
[fp-loader-source]: https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py#L1306
[fp-persist-policy]: https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_prefill/aka_solution/flash_attn/cute/interface.py#L1146
[fd-global]: ../../schema_template/examples/fa4_decode/root.yaml#L161
[fd-compact]: ../../schema_template/examples/fa4_decode/q4_compact.yaml#L85
[fd-cluster-math]: ../../schema_template/examples/fa4_decode/q4_cluster.yaml#L83
[fd-cluster-store]: ../../schema_template/examples/fa4_decode/q4_cluster.yaml#L216
[fd-cluster-source]: https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/qoder_greenfield_decode_attention/gqa_decode.py#L248
[fd-q1-source]: https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/fa4_decode/aka_solution/qoder_greenfield_decode_attention/gqa_decode_paged.py#L1568
[sparse-index]: ../../schema_wiki/archive/six-operators/revisions/2.5.0/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/dispatch.md#L5
[sparse-qis]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/candidate/dsv4_kernel.cu#L5927
[sparse-hidden-source]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/candidate/dsv4_kernel.cu#L6153
[sparse-hidden-yaml]: ../../schema_wiki/archive/six-operators/revisions/2.5.0/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/probe_hidden_mma.yaml#L342
[sparse-serial]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/candidate/dsv4_kernel.cu#L6433
[sparse-sink]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/candidate/dsv4_kernel.cu#L1131
[fmha-plan]: ../../.schema_extract/v2/runs/45ea6608-66b2-4915-b47d-59d2c115602e/checkpoints/e2cb480fdbadab88d555c766adc019cca0681c59889a63929553d81fa7789938/artifacts/dispatch.md#L5
[fmha-launch]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu#L1640
[fmha-fscan]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_tcg.cuh#L5181
[fmha-partm]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_tcg.cuh#L5480
[fmha-weight]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_common.cuh#L292
[fmha-xpose]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_tcg.cuh#L5225
[fmha-combine]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/candidate/fmha_mxfp8_qkv.cu#L916
[fmha-key]: ../../.schema_extract/v2/runs/45ea6608-66b2-4915-b47d-59d2c115602e/checkpoints/e2cb480fdbadab88d555c766adc019cca0681c59889a63929553d81fa7789938/artifacts/fused_d128_key.yaml#L26
[fmha-group]: ../../.schema_extract/v2/runs/45ea6608-66b2-4915-b47d-59d2c115602e/checkpoints/e2cb480fdbadab88d555c766adc019cca0681c59889a63929553d81fa7789938/artifacts/fused_d128_group.yaml#L26
[fmha-tc-fused]: ../../.schema_extract/v2/runs/45ea6608-66b2-4915-b47d-59d2c115602e/checkpoints/e2cb480fdbadab88d555c766adc019cca0681c59889a63929553d81fa7789938/artifacts/tc_fused.yaml#L26
[gdn-template-scalar]: ../../schema_template/examples/gdn_prefill/m64_single_sequence.yaml#L222
[gdn-template-source]: https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/gdn_prefill/aka_solution/gated_delta_net_chunked.py#L239
[gdn-direct]: ../../schema_wiki/archive/six-operators/revisions/2.2.7/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/direct_qk.yaml#L183
[gdn-segmented]: ../../schema_wiki/archive/six-operators/revisions/2.2.7/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/segmented.yaml#L119
[gdn-wiki-source]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/candidate/gdn_kernel.cu#L3320
[gdn-ordered]: ../../schema_wiki/archive/six-operators/revisions/2.2.7/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/cached_ordered.yaml#L181
[mla-host]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/candidate/solution.py#L1998
[mla-refill]: ../../schema_wiki/archive/six-operators/revisions/2.5.0/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/root.yaml#L342
[mla-bootstrap]: ../../schema_wiki/archive/six-operators/revisions/2.5.0/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/bootstrap.yaml#L17
[mla-index]: ../../schema_wiki/archive/six-operators/revisions/2.5.0/aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/dispatch.md#L12
[gemm-wide]: ../../schema_wiki/archive/six-operators/revisions/2.5.0/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/wide.yaml#L191
[gemm-recovery-source]: ../../kernel_wiki/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/candidate/fp8gg_umma.cu#L1718
[gemm-recovery-yaml]: ../../schema_wiki/archive/six-operators/revisions/2.5.0/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/wide_recovery_full.yaml#L34
[gemm-index]: ../../schema_wiki/archive/six-operators/revisions/2.5.0/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/dispatch.md#L5
[qsa-operator]: https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/qsa_prefill/aka_solution/qwen38_core_qsa_cutedsl/operator.py#L49
[qsa-gate-source]: https://github.com/bcacdwk/KernelSchema/blob/01fff90d714abc6e84904ea9c44523ba66d34a0e/kernel_wiki/aka-four-problems-20261019/qsa_prefill/aka_solution/qwen38_core_qsa_cutedsl/qsa_kernel.py#L628
[qsa-math]: ../../schema_template/examples/qsa_prefill/root.yaml#L127
[moe-initial]: ../../schema_wiki/archive/six-operators/initial/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/dispatch.md#L5
[moe-paused-root]: ../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/root.yaml#L8
