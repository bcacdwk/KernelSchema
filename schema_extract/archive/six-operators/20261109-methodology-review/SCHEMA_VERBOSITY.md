# 本轮 Schema 的文字密度与创作谱系

2026-11-09；以本次明确暂停后的只读快照为准。

本轮 YAML 确实普遍比模板示例更密，最直接的差异是 **`tile_pipeline.phase.*.role.*.info.verb` 的单条契约更长**，而不只是 role 更多；`work_schedule` 的策略和少数 `realization.choice` 也是长文集中处。六个当前 root 的单条动词契约中位数为 175–438 字符，模板四个 root 为 91–159 字符。可是“更多文字”不等于“都能删除”：其中既有必要的完成、可见性、空分支和复用保证，也有过宽容器中的过程性叙述、可检查结构被正文代替，以及后续纠错留下的解释。

**现存证据不支持“主要是 Reviewer 反复要求加字，Worker 首稿原本简洁”的总归因。** 六算子最早 2.2.2 P3 的实存 Worker 稿已长，首次语义审阅均批准；实际 2.5.0 的 MLA、GEMM 又从 Design 占位稿重新创作，长契约同样先由 Worker 写出。Reviewer 会认可这种密度，也确实促成过实质增补；FMHA 分支是一条明确反例，但它后来又被缩短，不能把调用数等同于持续加字次数。

## 1. 比较口径与当前身份

研究只读暂停快照、原始尝试和规范；提取保持暂停，未运行模型、Runner、GPU 或改写任何 YAML。当前稿由[暂停清单](../20261109-pause/README.md)和 [inventory](../../../../.schema_extract/supervision/pause-closeout-20261109/inventory.json)确定。FMHA/MoE 是暂停时 **working checkpoint** 中的已批准 root，整树仍未完成，不能用旧发布稿冒充当前工作稿。节点版本是文件修订号，不是规则版本。

统计覆盖六个当前 root、8 个有意选择的分支，以及[四个模板任务的全部 21 YAML](../../../../schema_template/examples/README.md)。分支选择覆盖继承、较多提交、重修和待复审稿，不是随机样本。模板来自旧提取并经规范化，硬件信息更完整；任务和源实现不同，不能视作同复杂度、同提示的对照实验。

字符数使用 `norm(v) = re.sub(r"\s+", " ", v).strip()`，对字符串值求 `len(norm(v))`。不计键名、注释、缩进和数值类型的值；**计入表达式、动作名、引用等字符串，不是纯自然语言字数或 token 数**。全部 35 份 raw 样本的字符串值均无中文字符，不存在中英 UTF-8 字节倍数造成的主差异。

有三层必须分清：

- **标量声明量**：源 YAML 中同一个源标量节点只计一次，alias 不重复计字；分开声明但值相同的两段仍计两次；正文体量主表用此数。
- **字段实例量**：每个字段位置都计数，YAML anchor/alias 已解引用；最适合说明每个 role 承担多少契约，但不能当成作者重复写了多少文字。`verb_chars` 只计短契约或长形式的 `meaning`，不含 `execution_mode`。
- **raw / effective**：raw 尚未展开 `from_parent`；effective 展开三个 Design 区域并补入 root Contract。两者分列，绝不相加。动作数是 pipe 中编写的动作出现数，未按运行迭代展开；结构项数是映射值条目数加序列元素数。

主代理独立复算了10个root的动词契约总量与中位数，全部一致；并用生产 `resolve_documents().effective_document()` 复核 MLA `bootstrap`、MoE `verify_only`（含 `verify_root` 祖先）的 effective Contract/Design 字符量，和CSV一致。此为只读统计核对，不是新的语义批准。

精确路径、SHA256、checkpoint、node version 和两类长度见[字段 CSV](verbosity_fields.csv)；定义、最长字段、逐尝试谱系见[JSON 索引](verbosity_provenance.json)。

### 六个当前 root 与四个模板 root

以下为**标量声明字符数**；C 为完整 Contract，M 为 math_reshape，W 为 work_schedule，T 为 tile_pipeline，R 为 realization。非 math 小计只取 W+T+R，避免把 Contract 中的数学公式混进去。

| 样本 | C | M | W | T | R | W+T+R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 当前 GDN | 5,962 | 8,822 | 4,288 | 9,646 | 1,804 | 15,738 |
| 当前 MLA | 7,373 | 7,535 | 3,118 | 6,884 | 913 | 10,915 |
| 当前 GEMM | 3,545 | 3,173 | 2,192 | 6,113 | 1,861 | 10,166 |
| 当前 SPARSE | 7,342 | 6,907 | 2,571 | 7,069 | 1,799 | 11,439 |
| 当前 FMHA | 9,790 | 7,727 | 2,744 | 6,672 | 2,275 | 11,691 |
| 当前 MOE | 9,009 | 8,520 | 5,022 | 10,090 | 1,395 | 16,507 |
| 模板 fa4_prefill | 3,780 | 3,264 | 977 | 2,614 | 666 | 4,257 |
| 模板 fa4_decode | 3,962 | 6,436 | 1,801 | 10,026 | 1,692 | 13,519 |
| 模板 gdn_prefill | 3,380 | 4,055 | 1,134 | 4,743 | 1,139 | 7,016 |
| 模板 qsa_prefill | 3,536 | 4,551 | 1,324 | 6,852 | 897 | 9,073 |

当前六 root 的 W+T+R 中位数是 **11,565**；模板四 root 是 **8,044.5**，全部 21 个示例是 **6,280**。这说明总体差异，但不是每个 extract 都比每个模板长：模板 FA4 decode 为 13,519，超过当前 MLA/GEMM/Sparse/FMHA。六 root 的硬件区仍为占位，没有贡献字符串；模板四 root 的硬件区分别占 918, 911, 898, 867 字符。Contract 的差异也不能归给硬件说明。

| root | phase / role / 动作数 | 契约实例 / 标量声明数 | verb 字段实例 / 声明字符 | 单条契约中位数 / 最大值 | T 结构项数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| gdn | 2 / 2 / 26 | 26 / 26 | 8,079 / 8,079 | 288.5 / 1300 | 353 |
| mla | 1 / 5 / 30 | 20 / 20 | 5,026 / 5,026 | 175.0 / 573 | 344 |
| gemm | 1 / 3 / 23 | 16 / 16 | 4,742 / 4,742 | 242.5 / 648 | 258 |
| sparse | 2 / 2 / 13 | 13 / 13 | 6,068 / 6,068 | 370 / 1266 | 255 |
| fmha | 1 / 1 / 15 | 13 / 13 | 5,824 / 5,824 | 438 / 619 | 194 |
| moe | 7 / 19 / 96 | 80 / 24 | 25,286 / 7,629 | 311.5 / 680 | 765 |
| fa4_prefill | 1 / 5 / 20 | 10 / 10 | 1,496 / 1,496 | 139.0 / 253 | 275 |
| fa4_decode | 2 / 7 / 48 | 33 / 33 | 6,987 / 6,987 | 159 / 1198 | 503 |
| gdn_prefill | 2 / 7 / 42 | 19 / 19 | 2,329 / 2,329 | 91 / 457 | 476 |
| qsa_prefill | 2 / 7 / 40 | 24 / 24 | 4,499 / 4,499 | 104.5 / 1076 | 399 |

GDN/FMHA 的 role 与结构项都少于多份模板，却把多数执行细节集中进少量契约；这是“单字段更长”的直接证据。五个非 MoE root 中，verb 文本占 T 的 73.0%–87.3%；四模板 root 为 49.1%–69.7%。MoE 的声明量占比是 75.6%，若按字段实例算则是 85.5%。

**MoE 不可漏算 alias，也不可把 alias 误报成重复写作。** 其 verb 为 25,286 个字段实例字符，实际仅 7,629 个标量声明字符；`enter` 的同一 328 字符契约通过锚点供 FC1/FC2 的 14 个 role 使用，`retire` 也共享。17,657 的差额不是可直接删除的冗余段落。全区域实例量 W+T+R 为 35,975，实际声明量是 16,507，后者才适合和物理 YAML 文字体量比较。

### 分支补充

“本地文字”列仍只计 raw 的标量声明；后两列分别为 raw/effective 的 W+T+R 字段实例量。它们同时显示 alias 与区域引用两种不同效果。

| 分支 | 暂停状态 | W+T+R 本地文字 | raw 实例 | effective 实例 | 主要说明 |
| --- | --- | ---: | ---: | ---: | --- |
| gdn/segmented | 已批准 | 19,831 | 20,074 | 20,074 | GDN：3 phase，分段增加工作 |
| mla/bootstrap | 已批准 | 8,260 | 8,260 | 8,260 | MLA：math 引 root，W/T 本地 |
| gemm/wide_recovery_full | 已批准 | 11,497 | 36,967 | 36,967 | GEMM：12 role，alias 已复用大量定义 |
| sparse/probe_hidden_packed | 已批准 | 23,565 | 36,458 | 36,458 | Sparse：15 role，探测和 packed 路径 |
| fmha/tcg_d128_flat_tma_xpose_narrow | 已批准 | 17,799 | 17,799 | 17,799 | FMHA：15 次调用，版本4已批准 |
| fmha/tcg_d128_flat_tma_xpose_wide | 完整候选待审 | 19,036 | 19,036 | 19,036 | FMHA：完整候选，尚待复审 |
| moe/fused_big_mono | 已批准 | 12,399 | 20,674 | 25,692 | MoE：math/W 引 root，T 本地 |
| moe/verify_only | 已批准 | 5,905 | 5,905 | 5,905 | MoE：单 role 校验路径 |

`bootstrap` 的 math raw 仅是 `from_parent: root`（字符串值 4 字符）；不能据此说数学表达本身被压缩成 4 字符。`fused_big_mono` 同时引用 math/W，effective W+T+R 增长来自 W 展开；GEMM 的实例/本地差额主要来自 alias，不能归到区域继承。

### 长在什么字段

几个可直接核对的位置比文件 KB 更说明问题：

- **GDN `chain.Main.info.verb.finish_chunk`：1,300 字符。** 合并 full/mini、末块/非末块、最终状态 staging、未排空写回和源复用风险。开头是 “Each warp follows its own instruction sequence without a CTA join…”，末段还说 “Keeping this suffix in one action does not order those reads before other warps' writes.” 这是动作粒度与完成含义的集中说明，既不宜视为单一动词的普通释义，也不能无条件删掉后半段。（[当前 root](../../../../.schema_extract/v2/runs/45f9bb51-77f3-4285-b7a7-251cf2744fa1/checkpoints/349f0fdbad54effd79e2557888dffcc88de418c142438fd2989db2b4937d0a2d/artifacts/root.yaml)，L638–662。）
- **Sparse `attention.Main.info.verb.prepare`：1,266 字符；`realization.require[1].choice`：1,112 字符。** 后者覆盖 workspace 选择、单项 graph cache、capture 失败、实例化失败、回退及完成边界，说明非 math 长文并不全在 verbs。（[当前 root](../../../../.schema_extract/v2/runs/8f1be03a-af4a-44f2-a2ba-b8deda6ae3e7/checkpoints/a2e71c99c568d4c81e6a91a14a598e03b6962f3b592864ecda33ed32f18264fa/artifacts/root.yaml)，按字段定位。）
- **MLA `main.schedule.strategy`：1,174 字符。** 同一段说明 host 拷贝、等待、排序、缓存参数、Geo 上传、外部录图和可选输出；例如 “the main/current stream waits for that upload before the one main kernel.” 必要提交事实存在，但不同问题聚在一个自由字符串中，阅读与分层审查负担很高。（[当前 root](../../../../.schema_extract/v2/runs/531567f1-bcae-45ad-acf6-fd674c055fdc/checkpoints/05f6f97e9f84c73f3af897059f8a331e08626e326ba026f8d9b2bd0ee6d0558c/artifacts/root.yaml)。）
- **FMHA root 单个 Main 的 13 条全是自定义动词。** `advance` 为 619 字符，包含 NumState 读改写、概率发布、线程汇合及临时值死亡。GDN 的 26 条也全是自定义词；本轮不能简单用“删掉重复的通用动词释义”解决。（[FMHA 当前 root](../../../../.schema_extract/v2/runs/45ea6608-66b2-4915-b47d-59d2c115602e/checkpoints/e2cb480fdbadab88d555c766adc019cca0681c59889a63929553d81fa7789938/artifacts/root.yaml)。）

这些是定位线索，不是自动判删清单。完整容量、完成保证和现存来源限制可支撑设计判断；精确指令或缓存过程是否应迁到别层，需要逐句判断。

## 2. 起源：先固定首稿，再归因审阅

这里的“首稿”是该次 Design 创作调用结束后保留的最早实存 workspace，不是模型第一次写文件的瞬间；同一调用内可能已多次自查修改。P1/P2 也可能提前决定前提。因此可以判定“首次 P3 Reviewer 之前已出现”，不能推断作者最初的心理动机。

已读取全部 55 条 root P3 尝试和 FMHA 所选分支的 15 条尝试（精确数量以 JSON 为准），核对 `attempt.json` 的 base checkpoint、workspace、response/receipt、validated、result 及必要 repair。所选 35 条 Reviewer 尝试的目标 YAML 均与其 base 字节相同；正常 Reviewer 只读，以下“Reviewer 增补”均指**意见促使 Worker 写入**，不是 Reviewer 直接改稿。

### 六算子最早 2.2.2 的横切面

六份首个 P3 的 base Design 都是占位模板，字符串值仅 26 字符，不能误判为沿用了完整 Design。表中首稿与首次批准的数字用 W+T+R **标量声明量**；GDN/MoE 把失败或无效回执时已留下的稿件也纳入。

| 算子 | 最早实存 Worker 稿 | 首次获接收的批准稿 | 当前稿 | 首次 P3 语义 Reviewer |
| --- | ---: | ---: | ---: | --- |
| GDN | 17,101 | 17,173 | 15,738 | approve |
| MLA | 12,930 | 12,930 | 10,915 | approve |
| GEMM | 10,930 | 10,930 | 10,166 | approve |
| Sparse | 11,493 | 11,493 | 11,439 | approve |
| FMHA | 11,779 | 11,779 | 11,691 | approve |
| MoE | 14,584 | 14,584 | 16,507 | approve |

这不是六条连续同源版本差分：MLA/GEMM 后来重新创作，MoE 的当前谱系在 2.2.7 也重新起稿。它证明的是：**早期首次语义 Reviewer 之前，长文就普遍存在**，且初审没有普遍要求补长。

### 链 A：GDN——最长契约在首次 Reviewer 前形成

连续来源是 `97902388`（2.2.2）→`9699c87b`（2.2.6 continuation）→`45f9bb51`（2.2.7）。后两 run 的 `state.json.continuation.predecessor/checkpoint` 可核对，完整 ID 见 JSON。

| 时点 / attempt 前缀 | 动作 | W+T+R | verb 字符 | 核对结论 |
| --- | --- | ---: | ---: | --- |
| `19549753` | 首个 Worker timeout | 17,101 | 8,285 | 实际 YAML 已保存；repair checkpoint `63e3929b…` |
| `32ea7a6a` → `400e5551` | Worker 续写，再补无效回执 | 17,173 | 8,390 | `finish_chunk` 已为1,300字符；后一调用零字节改动 |
| `b6b6938f` → `a28d78cd` | Reviewer approve，补 summary 后接收 | 17,173 | 8,390 | 没有语义退修或 YAML 改动 |
| `8f49e40a` | 2.2.7 Worker 修订 | 15,653 | 8,079 | 总体缩短并收窄/修正设计 |
| `cba95a83` → `678f8c35` → `d8f1db1b` | Reviewer revise → Worker → approve | 15,653 | 8,079 | 仅补数学 `symbol.DU0`，M 增82字符 |
| `30816cbf` → `d54a4fb8` | 后续 Worker / approve | 15,738 | 8,079 | 调度 reader 说明增85字符；动词契约不变 |

[首个 timeout](../../../../.schema_extract/v2/runs/97902388-eda0-4cbd-b8a6-ba71440e86ab/attempts/19549753-70ae-42f3-bed9-60e93f804bcc)、[Worker 续稿](../../../../.schema_extract/v2/runs/97902388-eda0-4cbd-b8a6-ba71440e86ab/attempts/32ea7a6a-bb5d-4f7f-8db6-2da18728a820)、[2.2.7 退修意见](../../../../.schema_extract/v2/runs/45f9bb51-77f3-4285-b7a7-251cf2744fa1/attempts/cba95a83-0234-429b-983d-353e220b6448/receipt.json)保留实际身份。1,300 字符的 `finish_chunk` 是 `32ea7a6a` 把若干后缀动作归并时形成的，早于任何 P3 Reviewer；其内容仍保留在当前稿。退修 DU0 是“公式已有定义但跨区 symbol 缺登记”的真实缺项，并不是 Reviewer 让动词继续解释。那次冻结 checker 退出 0，语义审阅仍发现数学接口不全。

### 链 B：MLA——真正的 2.5.0 新作，初审直接批准

旧 2.2.2 的 `e3f7704f/372430c3` 从占位稿写出 W+T+R声明量12,930（实例量13,566），`d41cfb84` 批准。实际 2.5.0 run `531567f1` 没有这条 continuation，且 P3 base checkpoint `1d5e25f7…` 的 Design 仍只有 26 字符的占位结构，**不是把旧稿“复核 complete”当首稿**。

`43589d1a` 的[Worker 首稿](../../../../.schema_extract/v2/runs/531567f1-bcae-45ad-acf6-fd674c055fdc/attempts/43589d1a-cbbb-4c3f-8a5d-1d24ac8d1eaf)已含 W+T+R=10,923、20 条动词契约=5,026字符，以及上述1,174字符的 schedule.strategy；`80fa3023` 首审直接 approve，其 `document_relevance` 认为保留了“必要公式、策略、动作契约及来源限制”。这证明审阅认可了当前写法，不能证明 Reviewer 创造了它。

后来 P4 发现 Geometry 首次复用边界遗漏，`226f1b78` 改 `Queue.waits_for.reuse`：

```yaml
# 前稿
"publish:Next(w+2)": ["Softmax/latch:Item(w+1)", ...]
# 修订
"publish:Next(w+1)": ["Softmax/latch:Item(w)", ...]
```

`1da6e4cd` 批准；W+T+R 减8字符到10,915，verb 仍为5,026。这里需要的是正确索引的结构边，长契约和首次 approve 都没有代替这项检查。（[修订 attempt](../../../../.schema_extract/v2/runs/531567f1-bcae-45ad-acf6-fd674c055fdc/attempts/226f1b78-cd69-4c04-87fa-d6ac512a9a46)。）

### 链 C：MoE——区分两次新作、后续继承与 alias

原始 2.2.2 run `3563c3b7` 的 Worker `657f0c8a` 已写出 W+T+R=14,584（实例量25,941）、verb声明=6,947；回执缺 summary 未被接收。`4834acf2` 明言“本次仅补齐回执摘要，当前草稿保持不变”，随后 `8b99a5bf` 首审批准。

当前谱系的创作起点则是 **2.2.7 新 run `53d4bebe/79a589d1`**：它的 base 也为占位稿，不能把它与2.2.2画成内容继承箭头。之后 `2c4207c8` 与 `feb0c655` 的 continuation 明确接续这份完整稿。

| 当前谱系时点 | W+T+R 声明量 | verb 声明 / 实例量 | 审阅与变化 |
| --- | ---: | ---: | --- |
| 2.2.7 `79a589d1` Worker | 16,998 | 7,829 / 25,678 | `84ed1a7a` 直接 approve |
| 2.2.8 `18ee7825` Worker | 16,519 | 7,629 / 25,286 | 条件等待改为显式结构；`b77ffa00` approve |
| 2.5.0 receiver `ba2b0c5c` Worker | 16,507 | 7,629 / 25,286 | 沿完整稿局部修订；`355de9cb` approve |

[2.2.7 首稿](../../../../.schema_extract/v2/runs/53d4bebe-882d-4a66-98b1-287d132dd6f3/attempts/79a589d1-098c-4522-ad05-4bbc51418f21)已经有大部分当前措辞；2.2.8 两个 MMA 的 `prepare` 从每处676缩为304字符，新增具名 wait 和依赖，结构更多、文字更少。2.5.0 把跨调用 workspace 生命周期从 realization 移到 work_schedule.thesis，并补数学计数，W+T+R 净减12。近期 P1 `feb0c655/1d8dde38` 的 Worker 也零字节改动；Sparse P1 `8f1be03a/d740b118` 同样如此，二者不是本轮 Contract 从零创作证据。

### 链 D：FMHA 高调用分支——Reviewer 确曾促增长，但不是单调膨胀

`45ea6608/tcg_d128_flat_tma_xpose_narrow` 的首个 P5 Worker 从父设计工作副本派生，已有大量完整内容；以下总量不是它从空文件新增的字数。其15次调用包括 CLI失败、无效回执及重开，不能说“退修了15轮”。

| 时点 / attempt | W+T+R | verb | 实际变化 |
| --- | ---: | ---: | --- |
| 首稿 `208b8735` | 19,961 | 11,316 | 33条契约、4 role；首次成功语义审阅前已有长文 |
| 首次有效 Reviewer `00bbccc2` | 19,961 | 11,316 | revise：空split的安全复用和零PartA承诺缺依据 |
| Worker `971cb08b` / approve `16a6224b` | 21,134 | 12,331 | 非math增1,173（5.9%），其中verb增1,015 |
| 版本3 Worker `f4efea31` / `cb8c8087` | 18,205 | 10,070 | 前者工具/回执未接收，后者同稿接收；随后 `3060a43e` approve |
| 版本4 Worker `34d6133f` / approve `ce57b90d` | 17,799 | 9,712 | 2.5.0维护；相对v3减406，恢复设计依赖并集中来源限制 |

[第一次退修](../../../../.schema_extract/v2/runs/45ea6608-66b2-4915-b47d-59d2c115602e/attempts/00bbccc2-3a73-4c16-847c-ab00b6bf46f8/receipt.json)明确要求：“记录空 split 的 Q 写与暂存写未获跨 warp 排序、全局 PartA 零值不受保证”，随后 Worker 增补了 prepare/replay/stage_partials/write 等说明，因果链成立。它并非笼统要求“多写一点”，而是在当时抽象口径下纠正源代码保证的转录错误。

版本3的 base Design 与此前版本2不相同，不能把跨重开的缩短全归因于紧接着的 Reviewer；JSON保留 base checkpoint与两边指标。最终[版本4](../../../../.schema_extract/v2/runs/45ea6608-66b2-4915-b47d-59d2c115602e/attempts/34d6133f-7741-4a4f-b005-d36eee06379e/workspace/tcg_d128_flat_tma_xpose_narrow.yaml)把 `prepare.done → stage_partials.issue` 写进 `waits_for.reuse`，把数学PartA恢复为有据的设计值，源码没有正确落实同步的事实浓缩为一句来源限制。当前定义与来源缺陷分流见下文。`b1137a89` 的早先维护稿未被接收，`34d6133f` 才是被接收且版本递增到4的 Worker，不能用前者冒充最终批准。

## 3. 规范与 checker 到底要求什么

规范并未要求把每条 role 动词写成小作文。现行[微流水线规范](../../../../schema_docs/5.design.tile_pipeline.md) L94、123、137、493 的组合要求是：自定义动作须有契约，通用动作可收紧；值是“一句契约”；依赖原语顺序时写明保证及适用范围；仅保留特有、影响设计的内容，“不是源码说明书”。“一句”没有机械字数上限，也不能代替语义必要性判断。

| 层面 | 必须/允许保留 | 明确边界 | 实际自动检查 |
| --- | --- | --- | --- |
| Contract / math | IO、计算、精度、符号接口及有据来源限制 | 不能用下层说明补上层定义 | 结构、属性引用等；数学等价和多数正文引用仍人工核实 |
| work_schedule | 工作、映射、调度、提交与交接 | `strategy/describe/thesis` 不复述已有字段；不提供条件删边语义（第4篇L380–382） | phase/launch/handoff等既有关系；strategy的条件含义不执行 |
| tile_pipeline | 非通用动作含义、完成/可见性、复用和必要原语保证 | 跨role不可推导的前驱用waits_for；不能把条件等待藏进契约（第5篇L98） | verb存在、格式、mode枚举、动作族/索引作用域、事件目标等 |
| realization | 上游尚未表达的必要原语/兼容条件和具体理由 | 删除测试、最小交接；理由不能新增角色/依赖（第6篇L7、20–32、87） | item/choice/reason的形状和非空；item语义绑定仍需人工 |
| Reviewer | 全文独立理解、分层、一致、内容必要 | 过程、待办、历史批准和重复定义移出YAML；不按篇幅判好坏 | Runner核对结论、证据绑定及权限；不证明结论正确 |

实现位置可以核对到函数：[values.py](../../../../schema_template/tools/rules/values.py) L22 的 `text` 只要求字符串非空、非 `-`；没有英文语言检测、长度上限、重复度或事实蕴含检查。[structure.py](../../../../schema_template/tools/rules/structure.py) L178–186 的 `verb_contract` 接受短字符串或只含 meaning/execution_mode 的长形式；[pipeline.py](../../../../schema_template/tools/rules/pipeline.py) `validate()` L47–52 只要求自定义 verb 有契约，并继续做动作/事件关联校验。不存在把自然语言中的“wait”“after”自动编译为依赖的机制。

这不意味着自由文本完全无法审计。[模板校验边界](../../../../schema_template/README.md#校验边界) L126–130 清楚分开结构接受与语义正确；[REVIEW.md](../../../../schema_extract/REVIEW.md) 四项判断要求结合源码、公式与结构人工核查。有限图工具还可按显式参数展开动作、检查实例边界，但不能读懂契约、证明所有参数范围内复用安全，或证明源实现兑现了保证。

### 提示是否在鼓励越写越长

[P3](../../../../schema_extract/phases/03_root.md) 第6段要求检查首轮/稳态/末轮/空工作/终止、相邻工作和跨调用状态；第7段要求实现选择做删除测试。Reviewer必须覆盖全文和源码依据，允许删除重复决定和不影响理解的细节。当前 [prompts.py](../../../../schema_extract/tools/prompts.py) L184 指定 Reviewer只读，L235 要“忠实、简洁”，并嵌入阶段和审阅方法；它没有要求按role增加说明段数。

必须按历史版本区分：原始2.2.2快照没有独立 REVIEW.md，但 P3 L37/51 已要求末尾、相邻工作和复用核对，也已有“重复决定及不影响理解的细节可删”；第5篇L119/458已有“一句契约”和“不是源码说明书”。2.2.7引入更明确的当前文档/内容必要验收。实际MLA2.5.0快照有“不要求找问题或增加篇幅”；**不能把今天这句话反投到所有早期或bridge receiver调用**。冻结快照身份已随每次attempt记录在JSON。

可证实的是：检查要求密集，而允许的自由字符串可容纳很多细节；Worker需要自行决定压缩粒度，Reviewer经常把长文判为必要。无法仅凭提示或最终稿证明模型“因害怕退修而主动堆字”，也没有短版/长版盲审对照能量化这种激励。

## 4. 三类真实边界：文字、结构、来源限制

**文字没有补全结构：GDN DU0。** 2.2.7 root 已在 `chain.math` 定义 DU0，resource.view也使用它，但 symbol没有shape/dtype登记。Reviewer `cba95a83` 在 checker=0 后指出缺项；Worker增加 symbol，而不是再给资源说明加一段。这个例子说明字段非空、公式中出现名字和跨区接口已完整是三件事。

**文字曾代替应有依赖：MoE 条件 scale-copy等待。** 2.2.7首稿的 `fc1/2.MMA.info.verb.prepare` 写道：

> “At i=0,t>0, if R<16*E, first wait for the preceding worktile's four epilogue release:Accumulator(t-1) events … These are internal branches of this action…”

同稿 `waits_for.reuse` 只把前一worktile release连到首个MMA，没表达该条件下 **prepare之前** 的跨role前驱；按后续明确的规则（[既有问题记录](../../../../schema_extract/issues/archive/extract_workflow/conditional-flow-topology.json)绑定 `nvfp4_moe.cu:4843–4915`），不能以动作“内部条件分支”移动已命名事件的等待。2.2.8 [修订](../../../../.schema_extract/v2/runs/2c4207c8-000e-43ea-8da0-2eb6a6079aee/attempts/18ee7825-6ae9-4e11-934f-80249aca3c23)使用现有语法：

```yaml
- repeat: {q: [0, "min(1,t)*(1-Hoist)"]}
  steady: ["wait:Drain(t,q)"]
# waits_for.reuse 中：Drain@done 等待四个 Epilogue/release:Accumulator(t-1)
# 首个 MMA 继续保留其必要前驱
```

这不是要求自然语言全部可执行，而是将已经属于依赖图的事实放回依赖图。单处prepare文字676→304，结构变多而全文更短；关于是否需要新分支的讨论见[分支专题](BRANCH_DERIVATION.md)。

**来源缺陷说明不等于 Design 的损坏后结果：FMHA。** 首稿说共享区在staging前已空闲，又说空split也执行该动作；Reviewer追源发现空split跳过了保证该顺序的循环屏障。这是实质反例，不能由“有一大段解释”自动免责。早期修订把可能污染的字节继续写进PartA数学和输出说明，文字增长。最终2.5.0维护根据有据的生产者、容量、消费者和复用设计，将设计依赖前移并写明：

> “Source limitation: the empty-split path lacks a pre-staging join for Query writers and can violate this reuse ordering.”

当前稿较短，且明确区分应有设计关系与源同步落实偏差；这不代表kernel已修复或批准可证明kernel正确。此处的缩短是抽象归属更明确的结果，不是删掉所有风险语句。仅当无需猜测/重选算法、数值版本、角色、容量或依赖即可确定完整设计时，才适用[源实现缺陷分流](../../../../schema_extract/REVIEW.md#源实现缺陷与设计分流)。

## 5. 根因归因与后续验证

| 候选原因 | 本次证据支持到哪里 | 不能推出什么 |
| --- | --- | --- |
| 源实现与任务复杂度 | MoE多phase/role，MLA有host建表，Sparse有graph/cache控制，GDN有mini/末块复用；这些直接增加需表达事实 | 不能解释所有长句都必要；模板同样有复杂异步流水 |
| 动作粒度与容器过宽 | 少role的GDN/FMHA仍有很长契约；动作序列、条件、生命周期和限制可压到一个meaning中 | 不能据长度自动判错，也不应把协作计算虚拆成多个role |
| Worker创作风格 | 2.2.2六首稿及真正2.5.0新稿都在首次Reviewer前已有长文；GDN最长段形成于作者续写 | 尚未审计每次调用内首次落笔，不能归因心理动机或指定模型倾向 |
| Reviewer验收倾向 | 多个长首稿直接approve，甚至document_relevance明确认可；FMHA一次有据退修带来增长 | 不能说全部Reviewer在强制加字，也不能把高调用数当高退修数 |
| 自由文本与结构错位 | DU0漏登记、MLA索引边界、MoE等待前驱均需改结构；FMHA源缺陷曾扩散进多层说明 | 无自动语义校验不等于规范允许，也不等于人工审阅总能发现 |
| 旧稿/重复定义传播 | continuation保留旧文，局部重审常不触动verb；alias/整区引用已节省部分文字。GEMM源限制需传播到多个本地T区域的实例见分支专题 | 不能把当前稿中的每段限制叫“历史残留”；独立有效定义可能仍需要它 |

本题需要的后续工作不是先定统一字数上限，也不是马上设计大型新语法，而是做以下有边界的验证：

1. **逐句必要性标注。** 先取 GDN.finish_chunk、Sparse.prepare/realization.choice、MLA.schedule.strategy 和 MoE共享enter，各句标成“独有语义、已有结构的重复、来源限制、所属层不当、纯实现细节”，每个删改结论给出当前结构或源码依据。保留未能证明可删的句子；不要靠关键词如barrier、unknown或句长直接删。
2. **同源短稿的语义对照。** 在另存的实验副本中，用已有字段重写一小组节点，冻结相同源码/规则/选择域。比较动作身份、首末/空分支、全部读者完成、alias槽位和跨调用状态是否保留，再让独立审阅盲判完整性、分层和查错效率。此项尚未执行，不能预报节省比例。
3. **将漏掉的结构事实作为反例。** 回放 DU0、MLA首次复用、MoE条件等待和FMHA stage issue/done四例，检验“短且正确”和“长却矛盾”均能区分；既测漏检，也测把合法no-op、必要原语契约和source_gap误删的误拒绝。
4. **分别衡量作者与审阅影响。** 以后保留每次首个完整Worker稿、Reviewer具体issue、修订字段及增减；按新作/沿稿、规则版本、源码复杂度分组。一次revise只把直接对应的差分归给该意见；P1/P3重开和跨版本重写单列。成本仍须读取用量，不能用输出文件短了多少同比折算模型费用。
5. **单独验证复用与重复。** 保留当前raw、alias声明量、effective实例量三套指标；共享相同文字只有在使用域/参数/依赖都一致时才成立。GEMM传播问题与节点拆分方法见[分支专题](BRANCH_DERIVATION.md)，本专题关注每个字段为什么需要那么长。

本报告没有给出“可安全删除X%”的结论。已证实文字密度、出现时点、具体增补/缩短及几类错位；仍未知的是在保持相同表达质量时的最短长度、全量分支的冗余比例，以及换提示或审阅流程后的因果效果。

## 原始身份与证据入口

当前六root均在暂停inventory记录的已批准working checkpoint；已交付四项与[归档映射](../../../../schema_wiki/archive/six-operators/LOCATIONS.json)对应。正文短ID只为可读，CSV/JSON保留完整run/attempt/checkpoint、文件路径与SHA256。JSON覆盖全部所选P3历史及分支15次尝试，不把原始回执内容重新解释成运行授权。

| 算子 | 当前run | node version | 读取入口 |
| --- | --- | ---: | --- |
| GDN | `45f9bb51-77f3-4285-b7a7-251cf2744fa1` | 11 | [root.yaml](../../../../.schema_extract/v2/runs/45f9bb51-77f3-4285-b7a7-251cf2744fa1/checkpoints/349f0fdbad54effd79e2557888dffcc88de418c142438fd2989db2b4937d0a2d/artifacts/root.yaml) |
| MLA | `531567f1-bcae-45ad-acf6-fd674c055fdc` | 4 | [root.yaml](../../../../.schema_extract/v2/runs/531567f1-bcae-45ad-acf6-fd674c055fdc/checkpoints/05f6f97e9f84c73f3af897059f8a331e08626e326ba026f8d9b2bd0ee6d0558c/artifacts/root.yaml) |
| GEMM | `d4e41124-e29c-458c-a1ce-2a9bf87ee0a8` | 4 | [root.yaml](../../../../.schema_extract/v2/runs/d4e41124-e29c-458c-a1ce-2a9bf87ee0a8/checkpoints/6a1253c6db2be19c8849e0c2a2b76470a3247730a632a62da41255cb0c3f6d5c/artifacts/root.yaml) |
| SPARSE | `8f1be03a-af4a-44f2-a2ba-b8deda6ae3e7` | 13 | [root.yaml](../../../../.schema_extract/v2/runs/8f1be03a-af4a-44f2-a2ba-b8deda6ae3e7/checkpoints/a2e71c99c568d4c81e6a91a14a598e03b6962f3b592864ecda33ed32f18264fa/artifacts/root.yaml) |
| FMHA | `45ea6608-66b2-4915-b47d-59d2c115602e` | 9 | [root.yaml](../../../../.schema_extract/v2/runs/45ea6608-66b2-4915-b47d-59d2c115602e/checkpoints/e2cb480fdbadab88d555c766adc019cca0681c59889a63929553d81fa7789938/artifacts/root.yaml) |
| MOE | `feb0c655-4e13-43ac-817d-1f457f31c029` | 7 | [root.yaml](../../../../.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/root.yaml) |
