# LIVE-14：证据选择的 12 行上限是否必要

结论：建议取消 `source_evidence` 与 `artifact_evidence` **选择范围**的固定 12 行上限，保留已有的 320 字符精确摘录和全部真实性、身份、审批校验。12 行有“迫使局部定位”的实际收益，但没有查到必须取 12 的依据；它既不证明相关性，也不控制当前读取或摘录大小。换一个固定数字会保留同类机械负担。本结论是产品规则建议，不改变任何冻结 run，也不使旧拟 approve 获得批准。

本次仅抽样公开提案、对应固定源码及 raw 产物，查看相关实现和历史；唯一新增文件是本报告。未读取模型隐藏推理，未调用模型/GPU、运行测试或修改产品、state、snapshot、原稿、旧回执。

## 出处与实际作用

- 当前限制在 [reading.py](../../schema_extract/tools/reading.py) 的 `read_evidence` 和 [receipts.py](../../schema_extract/tools/receipts.py) 的 `_anchor` 中分别执行；两处都是将 `1 <= start <= end <= 实际行数` 与 `end - start < 12` 放在同一个范围错误分支。现有 `read_evidence` 已独立用 `selected[quote_offset:quote_offset + 320]` 生成精确短摘录，`_anchor` 再核对摘录非空、长度和原文匹配。
- 沿 `git log --follow -S 'last - first >= 12'`，当前校验的可追溯起点是 `f984cc8`（2026-11-03）的 `schema_extract/receipts.py`；当时手册同时规定“引用最多 12 行、短引文最多 320 字符”，并明确语义关系由 reviewer 核实。`29875a7` 后来新增 `read_evidence`，把摘要和引文生成移交 Runner，沿用同一上限。所查实现、对应手册和提交说明没有说明为何必须是 12。
- 这是 Extract 内部回执协议，不是用户固定的正式 schema 字段规则。[0.common.md](../../schema_docs/0.common.md) 的“来源与证据”规定真实仓库、修订、路径和入口；[Algorithm_requirement](../../schema_docs/1.contract.algorithm_requirement.md) 规定固定参考入口。当前七份正式字段文档没有 `source_evidence`、`artifact_evidence`、`quote_offset` 或 12 行限制。不能把修改内部范围策略称为放宽正式 Source 规则。
- `read_evidence` 在判断 12 行之前已经通过 `_read_file` 读取、哈希整个文件，并解码为行。12 行没有限制这部分 I/O；实际存入回执的 quote 已由 320 字符控制。上限限制的是定位跨度，并非事实量、文件真实性或语义支持程度。

## 六份公开提案的抽样

口径：四份已知范围失败提案，加上 QSA r03、FA4 r04 各一份已被正式接收的 P2 worker 完整候选作对照。不是随机样本，也不是全部调用的失败率。每项按原 `line_end - line_start + 1` 统计，未裁剪、拆分或替换证据。表中范围为最短–最长。

| 提案 | source 数 / 行数 | artifact 数 / 行数 | 超 12 行 | 越文件尾 |
| --- | --- | --- | --- | --- |
| FA4 r02 P2 worker [98ca1cea](../../.schema_extract/live_acceptance_20261117/runs/a9a16cd1-7232-46c0-9721-1e8d98a227d9/attempts/98ca1cea-0547-483c-9685-d1d18893f1c7/response.json) | 85 / 1–13 | 0 | source 1 | 0 |
| QSA r03 P2 worker [eb61bbe9](../../.schema_extract/live_acceptance_20261117/runs/f1a26608-6adf-463a-acbe-144e3a2bc69d/attempts/eb61bbe9-0cd2-4ca8-a4cf-da570314a4db/response.json) | 74 / 2–12 | 9 / 3–11 | 0 | 0 |
| QSA r03 P2 reviewer [0b7794c2](../../.schema_extract/live_acceptance_20261117/runs/f1a26608-6adf-463a-acbe-144e3a2bc69d/attempts/0b7794c2-1885-4753-b000-0de0257c21a7/response.json) | 116 / 1–14 | 23 / 3–38 | source 3、artifact 7 | 0 |
| QSA r03 维护 reviewer [aac442e7](../../.schema_extract/live_acceptance_20261117/runs/f1a26608-6adf-463a-acbe-144e3a2bc69d/attempts/aac442e7-0dab-46ec-a9ad-776aee3eb859/response.json) | 113 / 2–12 | 30 / 1–14 | artifact 1 | artifact 2；其中 1 条同时超 12 |
| FA4 r04 P2 worker [9d74ca7d](../../.schema_extract/live_acceptance_20261117/runs/ebe745a0-e758-43c4-9679-2e8aa6e2a842/attempts/9d74ca7d-9f3a-4cb5-8dcc-f79495eb34a2/response.json) | 132 / 2–12 | 0 | 0 | 0 |
| FA4 r04 P2 reviewer [e6336933](../../.schema_extract/live_acceptance_20261117/runs/ebe745a0-e758-43c4-9679-2e8aa6e2a842/attempts/e6336933-1f7c-4bdc-875b-4aa5913d3456/response.json) | 87 / 2–22 | 15 / 3–12 | source 10 | 0 |

合计 607 条 source、77 条 artifact；22 条超 12，其中 21 条只有长度超限，1 条同时越文件尾；另有 1 条只越文件尾。267 条恰为 12 行。该分布已受现行上限影响，不能据此推导一个自然的新上限。抽样文件实际存在；source 字节均匹配固定清单，reviewer artifact 字节均匹配其候选依赖摘要；未发现另外的负/倒置行号或无效 offset/空摘录。这里只检查上述范围及绑定条件，不据此断言移除上限后其余门禁和语义独审必然通过。

需要纠正“只看首个报错”的归因：

- `0b7794c2` 的三条 source 是已知的 `operator.py:1–14`、`qsa_preprocess.py:109–122`、`environment.json:1–14`。此外 artifact `[13,14,16,19,20,21,22]` 还分别引用 `root.yaml:19–32、33–45、57–69、96–111、112–127、128–155、156–193`，共 `14、13、13、16、16、28、38` 行；均在真实 193 行内。normalize 先处理 source，旧拒绝摘要没有展示这些后续范围问题。38 行末段确是完整的 DESIGN starter 骨架；不是引用不存在的尾行。
- `aac442e7` 与 [LIVE13](LIVE13.md) 一致：artifact `[13] dispatch.md:71–76` 仅 6 行，但文件只有 75 行；`[29] root.yaml:181–194` 是 14 行，且文件只有 193 行。取消上限后这两条仍无效。不能把这次维护阻塞完全归因于 12。
- `e6336933` 的 10 条超限长度为 `13、20、14、14、13、16、19、13、22、20`，均未越文件尾；artifact 均合法。旧拟 approve 仍只是一份被拒提案。

## 收益、负担与选择

真实内容说明“行数”不等于“摘录大小”或“一个事实”：FA4 的两个单 shape JSON 对象分别占 20/22 行、406/439 字符；tiler/cluster 的相关赋值块占 20 行。QSA `environment.json:1–14` 只有 239 字符却被拒；维护提案中合法的 `metrics.csv:3–5` 只有 3 行、却有 5682 字符，最终同样只生成 320 字符 quote。短范围也可只引用无关内容；把相关块机械拆成多项不会自动增加真实性，反而需要维护更多 note 和 evidence 索引。

**三个方案共同必须保留**：固定清单内真实存在的普通文件、合法相对路径与无符号链接；整数行号且 `line_start >= 1`、`line_end >= line_start`、`line_end <= 实际行数`；`quote_offset` 落在所选原文内且生成非空精确摘录；固定 source/候选摘要、run/attempt/nonce/依赖身份与证据身份；实际字段、必需证据、比较声明、引用链覆盖和完整 P1–P6 独立语义审阅。任何方案都不能靠截断原范围、代填证据或保留未经通过的 approve 来修复旧提案。

| 方案 | 实际收益与风险 | 代码和模型负担 |
| --- | --- | --- |
| 保留 12，仅用 LIVE13 精确定位和共用提示 | 维持机械局部定位，抑制整文件式引用；精确错误有助于正规重提。但 13 行相关块仍会拒，短而无关的块仍可能过范围门禁。没有证据表明 12 是真实性必需值。 | 不再改代码；模型继续计首尾行、拆相关块、重编索引。LIVE13 新提示改善多少尚无足够样本，不夸大重审耗时都由上限造成。 |
| 改成另一统一局部上限 | 可以减少一部分拆段，仍有局部定位约束；但格式稀疏的 JSON/YAML 和较长语句继续遇到新边界。当前样本不支持任何特定新值，不能按本次最大 22 或 38 倒推政策。 | 仍维护两处阈值、说明与对应测试；模型仍承担计数/拆段。代码复杂度几乎不减。 |
| 取消选择范围的固定上限，保留现有短摘录（建议） | 允许按实际相关代码块/对象定位，字节身份、真实行尾和摘录长度仍独立校验。风险是宽范围更容易掩盖定位不足，相关性仍需独审。 | 删除两处上限条件，更新共用说明及现有上限用例即可；无需字段、配置、迁移或新摘录系统。减少机械拆段与索引重排；审阅应继续检查范围是否确实支持结论。 |

这里不能把取消上限说成完全没有校验能力损失：`receipts._region_anchored` / P5 引用链覆盖用范围是否包含定义行作机械证据，宽范围会更容易满足该检查；12 的局部定位收益在此是真实的。但包含定义行从来不等于理解定义，12 行也不能保证理解。P1 比较器还要求具体声明文本出现在 quote 内，这项约束应原样保留。

最小建议是只取消两处任意的跨度上限，把共用提示改为“选择支持当前结论的相关原文范围，避免无关扩张；行尾不得超出实际文件”，沿用现有 320 字符摘录和独审职责。不要新设一个没有依据的数字，也不要建立新的摘录或恢复机制。若主代理决定必须机械保证每个锚点局部性，则保留 12 并只用已做的精确定位/提示是可解释的保守选择，但应明确承担它带来的格式性拒绝，而不再称其为正式 schema 的来源真实性要求。
