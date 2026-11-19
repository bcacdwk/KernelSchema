# 正式目录历史清理

六算子的正式目录现在只保存 15 份当前 YAML、6 份 dispatch 和各自一份 `.extraction/current.json`。15 份 YAML 与专项审修后的字节完全一致，没有调整算法、Version、Contract 或源码。当前入口为 [Schema Wiki](../../../schema_wiki/README.md)，机器索引为 [CURRENT_SCHEMA_INDEX.json](../CURRENT_SCHEMA_INDEX.json)。

| 算子 | 当前版本 |
| --- | --- |
| GEMM | root 5 / wide 2 |
| MoE | root 5 / split_k 2 / mega 3 |
| MLA | root 6 |
| GDN | root 3 / segmented 3 |
| FMHA | root 8 / mma_sync 4 / split_kv 4 |
| Sparse | root 5 / split_kv 3 / tcgen05 2 / tcgen05_packed 3 |

`current.json` 绑定实际文件摘要、Version、精确 Source 与源文件 SHA，指向对应语义审修证据和最终 checker。它明确是当前成品清单及人工检查元数据，不是 Runner P6 receipt。六份原 receipt 只覆盖其中列出的原批准字节；其旧状态、marker 与批准范围没有被改签。

## 迁移与删除

逐文件旧路径、新路径、SHA-256 和删除依据见 [LOCATIONS.json](LOCATIONS.json)。归档导航见 [archive/README.md](../../archive/README.md)。

| 原材料 | 处理 |
| --- | --- |
| `schema_wiki/archive`，131 文件 | 原字节迁入 `schema_discussion/archive/schema_wiki` |
| `schema_extract/archive`，704 文件 | 原字节迁入 `schema_discussion/archive/schema_extract`，包括纯历史脚本、旧结果、失败/中间日志与设计讨论 |
| `schema_docs/archived`，3 文件 | 原字节迁入 `schema_discussion/archive/schema_docs` |
| 六份 receipt 与四份旧人工 JSON | 迁入 `schema_discussion/archive/schema_wiki/extraction_records/<operator>/` |
| 11 份被编辑的导航与 dispatch | 原件保存在 [before](before/)；精确文字替换见 [content_changes.json](content_changes.json) |
| 85 个现存源码可再生的 `.pyc` | 删除；清单保留旧摘要与对应源码摘要，不删除源码或测试 |
| 8 个已无现存源码的旧 `.pyc` | 原字节迁入 `schema_discussion/archive/generated_bytecode` |

合计迁移 856 个现存文件。没有删除唯一历史证据，也没有为去重重写旧稿、回执或带 hash 的报告。125 条更早的六算子位置记录已组合成旧路径→中间位置→实际位置的可核对映射。归档旧 README 与历史报告保持原文，原相对链接按旧目录解释；当前入口使用真实新位置，不改写旧 hash。

`issues/archive/TRIAGE.md` 与 `DESIGN_REGION_DERIVATION.md` 在清理前已经从工作树删除，见 [git-status-before.txt](git-status-before.txt)。另将 Git 提交 `24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2` 中的两份历史原件直接保全到 discussion；这不是恢复未提交的最新内容。搜索 discussion 与忽略快照后仅找到更早的 2.2.8 TRIAGE 维护备份，原位保留，未发现更近保存版；位置、摘要与时点见 [preexisting_deletions.json](preexisting_deletions.json)。

## 当前正文与证据边界

删除了正式入口中的批次、角色、旧审修状态叙述，以及 dispatch 中的 P2/P4 交办、原 pending/invalid receipt、前后稿与“本轮未新增 P6”等过程叙述。保留完整设计选择、来源实验与变体依据、真实来源限制及依赖证明。四个新审修的 `reviews/*/{review,after}.json` 与 FMHA/Sparse 的既有审修记录均未改写；当前索引连到这些证据，dispatch 的纯编辑差额另在本清理记录中说明。

- GEMM 的后置 scale 溢出边界、缓存失效条件及来源 NCU 范围仍保留。
- MoE 的比较器映射缺口、MEGA workspace 串行复用与来源限制仍保留。
- MLA 的静态缓存/侧流调用隔离、数值范围与实际 tiny 传递依赖仍保留，没有恢复已被推翻的 tiny 伪缺陷。
- GDN 的接受域/写回缺口、G≥3 标量段组合误差缺口及局部 vt64 mini-tail 跨 warp 冲突仍保留。
- FMHA 的外部比较器、NO_KTINY 全域界缺口与 mma.sync×split 源实现缺陷仍保留。
- Sparse 的 `bar_land`/`tmem_base_s`/packed union 控制区重叠、cfg2 编译资源未知与比较器缺口仍保留。

这些人工语义证据及结构检查不证明完整动态域数值正确性、GPU 行为或硬件资源保证。

## 验证

机器结果见 [verification.json](verification.json)。

- 六个完整目录执行现行 `check_yaml.py <root.yaml> --read-only --skip-hardware`，全部退出 0，15 YAML、零规范化变化；原始输出及绑定的正文摘要见 [directory_checks.json](directory_checks.json)。
- 15 YAML 的摘要与最新语义审修记录一致，版本与 Source 一致；每个 Source 文件均匹配声明的 Git 提交，见 [source_references.json](source_references.json)。六份当前 metadata 及当前总索引的指针/摘要均核验。
- `test_prompt_materials`、`test_executor_snapshots`、`test_inputs` 共 11 项离线测试通过，见 [日志](offline-prompt-freeze-tests.log)。使用纯净 `PATH=/opt/venv/python3.13.12/bin:/usr/bin:/bin`、`PYTHONDONTWRITEBYTECODE=1`，没有真实模型调用。冻结材料的三个入口仅移除历史旁白/链接，规范和 prompt 内容未改，也未新增对 discussion 的运行依赖。
- 所有迁移原件及旧路径映射的 SHA 核验；原历史目录内已跟踪文件另与原 Git blob 核对。活跃 Markdown 链接和当前 JSON 指针有效；规范的 fenced 示例不作为实际文件链接。
- 41 份正式 issue JSON 原字节保持；其中本轮 15 个 ID 与决定仍为 12 `resolved`、2 `declined`、1 `partially_adopted`。原 `issues` CLI 只读调用退出 0，输出见 [issues-cli.json](issues-cli.json)。
- `FRAMEWORK_BASELINE.json` 的 160 项全部匹配：94 项代码原位不变，66 个纯历史 Python 文件仅位置改变。`RUNTIME_BASELINE.json` 的 8 项摘要不变，原六次运行合计 201 次调用，monitor 仍 disabled/generation 8，successor 仍保留 cancel 控制。
- 正式树复扫没有原始运行日志、中间 diff、旧 `.extraction` 过程记录或字节码缓存。余下 P1/P2 是 GDN 源 kernel 名或 Sparse 数据区名，GEMM 的“本轮”指 kernel 迭代；均保留真实语义。`git diff --check` 通过。

## 必要例外与实施记录

正式 issue JSON 是稳定决定台账，保留历史证据和残余限制；没有因算子审修完成而改写旧决定叙述。当前工具/规范说明中的失败状态、恢复机制及测试故障 fixtures 是产品内容，保留不动。Git 忽略的 `.schema_extract/v2` 和 `.schema_extract/archive` 是真实运行证据，未迁移、删改或恢复；本清理没有启动 extract、模型、GPU、commit 或 push。

清理助手的中途诊断没有影响正式成品：迁移脚本遇到清理前已删除的两份 issue Markdown 后，按 Git 来源单独保全并完成其余迁移；索引脚本适配了 FMHA/Sparse 历史 `versions` 记录的 `{before,after}` 表示后完成写入；核验脚本将自身尚未生成的结果链接延后到写出后检查。最终核验使用成品实际字节，未改旧记录格式或 hash。

本清理的所有完整过程记录均位于此 `cleanup/`。旧历史索引仍是各自时点快照，新的 `CURRENT_SCHEMA_INDEX.json` 是当前入口；最终独立收尾复核由 Supervisor 另行安排。

独立收尾复核后的最小补正见 [Sparse 差额记录](sparse_followup/changes.json)：只将 dispatch 中“本 run 未运行 GPU、未复测”和沿用旧 source_gap 的过程口吻改为当前证据范围说明。L20D 的 fixed 十用例/domain 十采样点仍明确是已有 GPU 来源证据，其覆盖不等于四份设计及完整适用域均已独立验证；公式与相对误差基准缺口保留。原清理记录、旧 checker 输出及补正前的索引/metadata/验证原件均保留，Sparse 单目录重新检查通过，受影响的当前绑定已刷新，15 YAML 与其余正文保持原字节。
