# Schema Extract

本地执行器从显式选择的 kernel 材料调用已登录的 Codex CLI，形成可追溯的 schema。恢复、非硬件检查和合作审阅已有回归验证，本机真实调用与试点的依据、边界见 [ISSUES.md](ISSUES.md)。

## 当前政策

- **GPU 验收与实测不在提取范围。** 算子已经真实跑过 GPU；当前算子和 schema 都不需要 GPU 正确性验收。extract 忽略这部分，不将其列为待办、缺口或交付条件，也不在收尾重复提醒。
- **硬件完全不在提取范围。** `contract.hardware_specification` 只保留 starter 原样占位；撰写、搜索、检查、审阅、收尾都跳过。extract 固定调用现有检查器的 `--skip-hardware`，复用缺硬件的关联处理。普通检查默认不变。旧硬件问题标为 `ignored`，不影响 ready/export。
- **没有任务级硬性额度。** extract 不设 schema 正文的总字数、行数或字节上限；也不按时长、总 token、调用次数、修订轮次、无进展次数或续接次数停止。历史 plan 中的旧 limit 字段不再生效。超过 60 分钟只供 Supervisor 调查，不会自动终止。单次工具输出、并发名额、进程启动／清理等待和解析器容量另计：例如单个 YAML 展开超过 100,000 个解析节点会报容量错误，这不是正文长度限制。
- **Reviewer 是只读合作者。** 重点源码忠实性、实质遗漏、值得简化的冗余；具体可修问题合并交 worker 小修，由 reviewer 自行判断什么值得改。不凑问题数量、不因格式偏好或额外硬件／实验／误差证明扩大工作。已知原实现缺陷与尚未确定的源码关系如实备注，不自动否决提取。只有基础任务定义确实不清楚或必要源码缺失才 `needs_input`。
- **真实要求保持不变。** 不为迁就实现缩小 Contract 或伪造 fallback；来源明确的转录可自动纠正并重新查看受影响节点。数学、调度与流水线仍按规范逐段读写回查。

## 流程和记录

工作阶段为 `contract → root node → 其余 node → index`，Contract 使用单独会话协作审阅；单节点合并 root 与最终审阅。一次连续运行内，作者跨阶段和小修沿用同一 workspace／thread，独立 reviewer 只读已接受检查点的副本；返回作者前核对其文件仍与检查点一致。续接提示只给当前阶段与新反馈，不重复整个来源背景。会话引用仅存于当前 runner 内存，暂停／失败后的显式恢复仍新建目录与会话。节点内部依次写数学、调度、流水线和适用 realization，需要继续时保存检查点并续接，没有“两次续接”限制。具体修订不设轮数上限。

索引写作聚焦链接、parent、实际选择关系和未覆盖范围。程序统一核对共享 Contract、目标／root 的语义、选择关系修订及输入 fingerprint；只有仍有效的独立审阅可复用。新增派生节点、仅润色索引 reason 不撤销未变节点的审阅；修改既有 when／parent 或节点内容会撤销相应结论。

多节点 index/final 的上下文从当前检查点机械生成，只提供节点 metadata、已有审阅摘要、Contract 与来源定位，不重复塞入 Design 正文。metadata 由同一固定 YAML 解析器取值并读回核验，保留字符串类型和 Decimal 精度。`review_context.json` 只是调用审计副本，不能作为恢复权威。单节点完整审阅和每个节点第一次独立语义核验照常进行。

仅修改 `dispatch.md` 的意见使用 `impact=index`，修正后直接回 final。若实际修改了 YAML，程序按实际语义变化补齐独立节点审阅，再回 final，不额外重派已完成的 index 写作；不能靠回执标签跳过审阅。

固定输入接入现有 `kernel_wiki/catalog.json`，只选择指定 operator；公共说明仅供当前算子导航，不读取整库上下文。干净来源使用实际仓库 commit，选定的未提交来源才生成独立私有快照。提示词、规范、检查器和执行器内容摘要一同冻结，worker 只写其独立暂存目录，reviewer／supervisor 使用只读沙箱。提取与复核显式 `gpt-6-astra / xhigh`，总提取调用并发最多 2，reviewer 最多 1。

独立 Supervisor 每轮可另有 1 个只读巡检调用，不占转换名额，也不因两个 worker 忙而跳过调查；不为巡检设置 agent 时长额度。

```text
schema_extract/
  runner.py             计划、调度、恢复、导出
  codex_cli.py          无 agent 超时的 CLI 调用、事件与取消确认
  records.py            固定输入、原子 task.json、不可变检查点
  protocol.py           提取与合作审阅提示／回执
  reading.py            有界批量读取与无状态续页
  review_context.py     从当前检查点派生索引审阅上下文
  supervisor.py         独立的巡检与可选 15 分钟调度入口
.schema_extract/        忽略的运行记录
  inputs/<fingerprint>/repo/
  runs/<run-id>/plan.json
  tasks/<key>/task.json
  tasks/<key>/attempts/...
  tasks/<key>/checkpoints/...
<work-root>/<key>/<attempt-id>/   仓库外的会话产物目录
```

每个 `task.json` 是唯一状态权威。先保存完整不可变检查点，再原子更新任务指针；中断半成品不被接受。恢复先确认旧调用停止，再从已接受检查点建立新目录／会话。相同已结束任务不会因再次 `run` 自动重做。

新 workspace 初始为空且无 Git 元数据，提示给出冻结规范的绝对路径和当前阶段的阅读入口；材料按该算子的 catalog 路径定位，混读源码与产物时使用各自的绝对路径。`reading.py` 是由 agent 原生 `exec_command` 调用的本地读取器，原生 shell 工具仍可用。一个批次共享页大小，保留原行号、列号和换行，页尾明确给出未读位置。92 字符的 v2 游标绑定版本、请求／文件摘要及精确坐标；损坏、输入变化与旧版本分别明确报错。旧版游标须重新建立批次。校验用于检测损坏，不是权限认证。

每页最多 6,000 UTF-8 字节（包括额外一层 JSON 字符串包装），默认列号为 1，省略 `ends_line` 表示原行已完整返回。每次外层调用只返回一页，按 `next_cursor` 原样续页；`complete=true` 表示所请求范围读完。提示提供固定 8k 内外工具输出预算的调用模板，由客户端暂存原命令和游标，不需手抄长 token；当前页可按 `replay` 重放。这些临时值不参与恢复或交付。此保证覆盖读取入口，不替任意外层合批兜底；不是 agent 时长、调用次数或总读取量限额。模块只用标准库，无临时文件、持久索引或后台服务。

找内容时可在同一 request 加 `"find":["tolerance","rtol"]`，在指定行段中按大小写敏感的字面子串 OR 定位。返回 `matching_lines` 与匹配行总数，不回显匹配正文；再按需要的行号读取原文。查询条件同样绑定游标，混合定位／正文请求也共享一页上限。`rg --files` 仍可用于发现文件名。

回执 `source_evidence.path` 相对固定仓库，`artifact_evidence.path` 相对 workspace，行号放在独立字段；`repository@commit:path` 用于 YAML 来源引用。误把当前固定来源前缀或当前目录内绝对路径写入回执时，程序只机械还原相对路径，仍严格核对文件、行号和原文，记录 `receipt_normalization.json` 并保留原始 response。错误 commit、越界路径、错误引文仍拒绝，一次列出所有无效锚点及行范围；建议短的单行精确引文。需要模型修正时，同会话只收到具体错误及必要规则，产物保持不变。作者的一个 attempt 可跨多个阶段，各次调用的 `prompt.txt` 保存当时的阶段、节点和 checkpoint 绑定。

回执意见分 `fix / note / needs_input / ignored`。ready/export 需要非硬件检查通过、当前产物已完成相应协作审阅、待修错误已处理；源码备注可随交付保留。正式输出只创建新目录；相同内容重复导出幂等，目标内容不同则保留原样。

检查结果原文完整保存在记录中；提示只缩短诊断位置／文档链接重复的绝对目录，不删除错误、Actual 值或 Fix 提示。分页边界与索引审阅复用的依据、验收情况见 SE-018／SE-025。

## 命令

需要 Python 3.10+、现有检查器的 PyYAML 和已登录本地 `codex`：

```sh
python3 -m schema_extract discover
python3 -m schema_extract plan --run-id NEW_RUN --id OPERATOR \
  --work-root /absolute/path/outside-repository \
  --output-root schema_wiki/NEW_COLLECTION --jobs 2
python3 -m schema_extract run NEW_RUN
python3 -m schema_extract status NEW_RUN
python3 -m schema_extract status NEW_RUN --full
python3 -m schema_extract pause NEW_RUN
python3 -m schema_extract cancel NEW_RUN
python3 -m schema_extract resume NEW_RUN --id OPERATOR
python3 -m schema_extract retry NEW_RUN --id OPERATOR --reason '说明重新尝试原因'
python3 -m schema_extract export NEW_RUN --id OPERATOR
```

没有默认 all，不自动导出。全局 `--repo / --state-root / --codex` 放在子命令前；`--codex` 可明确本地可执行文件。pause 等待当前调用边界，cancel 请求终止所属进程并确认；状态以 task.json 为准。`run` 只派发 queued，不自动恢复 stopped/ended 或用户的暂停／取消。

读取入口也可直接使用（替换为该 task 的实际绝对路径）：

```sh
python3 -B /absolute/snapshot/schema_extract/reading.py \
  --root /absolute/snapshot --root /absolute/workspace \
  --requests '[{"path":"/absolute/snapshot/EXTRACT_SCHEMA.md","start_line":1,"end_line":40}]'
```

后续页保持相同 roots／requests，增加 `--cursor` 及上一页的 `next_cursor`。source 和 workspace 可在同一批次选择；文件改变后，针对变化材料重新发起所需范围。不要把多页重新合并成一个大工具输出。

`status` 默认只读紧凑进度：阶段、检查点、检查结果、已审阅范围、问题计数及当前调用 PID／thread／持续时间／事件更新时间。用量只含已结束调用，明确不包含活跃调用尚未报告的 token；没有事件更新本身不表示卡死。`--full` 返回完整 task 记录，供需要回执／历史的调查使用。

维护工具时，既有 plan 的执行器摘要不会随代码修改更新。需要让一次试跑继续使用其已冻结版本，可从该 task 的 `inputs.lock.json` 取得 `snapshot`，以该目录为 cwd 执行 `python3 -B -m schema_extract --repo /Users/hanyong/KernelSchema run RUN_ID`；自定义 state-root 时仍须显式传入原绝对路径。所有读取和检查继续核验原摘要。要测试工具修改，建立新的无 seed 计划；不要编辑旧快照或 task 来绕过校验。

`plan --seed-run OLD_RUN` 逐文件核对来源、规范、检查器与政策，复用已接受草稿及其回执、来源、用量和问题身份，清空旧审阅后重新协作复核。已有 Design 直接进入复核，不重新提取。

旧政策 plan 仍不能直接续跑。此次已审计的过渡可显式增加 `--seed-policy-update skip-hardware`：只接受 SE-012 记录的五组政策文件摘要，其余原有来源／规范必须一致；仅在新检查点副本中恢复 starter 硬件占位、递增 root 版本并记录变更。旧 task、检查点与回执保持原样；这不是通用迁移或跳过来源核验的入口。

结束时生成 `.schema_extract/runs/<id>/issues.patch`，仅包含变化条目的上下文补丁，由 Supervisor 审阅后使用内置 `apply_patch` 更新台账。收尾简写不同于初次发现时，从可用的首个不可变 worker 回执保留原发现／源码证据，并列当前结论；原记录不可用时不臆造来源。程序不直接改仓库 ISSUES；已知 issue 沿用原 ID／`ledger_task_key`，不因新 fingerprint 复制问题。

## 观察与 Supervisor

CLI 原始 `usage` 保留。已实证本机 0.163.0 的同 thread 续接报告累计用量，新调用按相邻累计差额观察；cached/reasoning 子项不再加一次。缺失、未知基线或下降记为观测异常，不产生负增量，也不据此阻断流程。历史状态／回执不改写，旧累计数字保留其历史口径。

Supervisor 独立入口为 `python3 -m schema_extract.supervisor tick|enable|disable|status <run-id>`；全局路径参数同样在子命令前。默认不启用。macOS `enable` 为选定 run 安装 15 分钟 LaunchAgent，使用绝对 Python/Codex/仓库路径，非常驻循环。每次只读紧凑状态及活动摘要，给出建议；固定程序只可启动该 run 已 queued 的工作，不恢复用户暂停／取消，不自动发布。pause 等待调用边界，cancel 会终止正在进行的巡检并确认。调度启停与 tick 互斥，活跃巡检期间拒绝变更，待其退出后再操作。会话按明确 thread ID 续接，`tick --new-session` 可显式换上下文。

## 2026-11-02 读取与多节点收尾验收

两个全新上下文的 agent 分别实现有界读取和审阅复用，父端集成并用 `refine_20261102_fp8` 从零提取 `fp8_scaled_gemm`。最终 `cp-0004-36afb82b91` 为 ready，root v8、small_m_transpose v4，8 次执行器 CLI 调用完成四阶段及对应独立审阅。最后只改索引，两份 YAML 字节不变。index 作者没有展开节点正文；final 读 root 至第 130 行、派生节点至第 9 行，均未进入 Design。

冻结运行中成功返回 288 个有界页（242 次续页），页内没有截断；5 个 thread 的 324 个工具回包没有外层截断。三处内层截断均来自绕过入口的直接 `rg`。其后增加同一入口的字面定位和带完整性校验的游标，并让客户端自动保存原命令／游标。这三次实际查询已用冻结输入重放：600 和 532 条匹配各一页返回相同有序行号，第三组用 5 页逐字节重组全文；600 行案例含包装 3,833 字节，原输出 161,243 字节。后补功能不冒充冻结 run 中已执行的模型行为。

最终 95 项执行器相关测试、147 项检查器测试通过。精确绑定、位置损坏、失效补审、索引小修、暂停恢复、模型可见读取范围及验证边界见 SE-018／SE-025 和 `.schema_extract/runs/refine_20261102_fp8/verification.json`。成本信息随真实任务保留，不另做成本对照，也不从不同算子的总用量推断降幅。

## 前一轮从零试跑（2026-11-02）

两个独立、全新上下文的使用者按同一执行器政策提取 `bf16_grouped_gemm_masked` 与 `mla_expanded_fp8`，均无 seed、旧 schema 或旧会话。第一题冻结旧执行器收集基线；第二题使用本轮改进版。两题类型不同，总 token／耗时不能当作严格 A/B 对照；效果按具体失败、续接与读取行为验证。

| 算子／run | 结果 | 调用与产物 |
| --- | --- | --- |
| `bf16_grouped_gemm_masked` / `trial_20261102_masked_baseline` | `ready`，非硬件检查与全部独立审阅通过 | root v8、single_cta v5、mma_sync v4；28 调用，12,758,962 差分 reported tokens，3,641.907 CLI 秒 |
| `mla_expanded_fp8` / `trial_20261102_mla_improved` | `ready`，非硬件检查与完整独立审阅通过 | root v8；4 调用，6,016,647 差分 reported tokens，1,761.009 CLI 秒 |

masked 的最终检查点为 `cp-0005-949e9eb079`，保留实现可能写入 inactive tail rows、外部比较器未核实两项 note。18 次回执修正占 659.988 秒／2,387,484 tokens；首个拒绝原因是 17 次路径、1 次引文，但重放发现部分路径首错掩盖了引文错误，不能全算作机械可消除的成本。`verification.json` 保存用量、检查、版本和进程终止核验，`receipt-replay.json` 区分可归一化与仍须修正的回执，均位于相应 `.schema_extract/runs/<run>/`。未导出、未提交或推送代码。

MLA 最终检查点为 `cp-0002-b297748059`。作者跨 Contract/root 保持同一 thread，两次 reviewer 独立，4 份回执一次通过；保留任务域／目录叙述漂移、比较器缺失、源码注释漂移、数值／顺序证明及跨调用同步五项 note。算法 Contract 字节不变。两题所有调用已确认停止，runner 锁空闲，未启用周期 Supervisor。

该轮 52 项执行器／巡检回归通过。真实拒绝回执离线重放中 13 份可归一化通过、5 份仍正确拒绝；MLA 没有触发归一化分支。当时多节点 final 收敛只完成回归，单节点完整 final 已实测；批量输出有 3 次外层、2 次内层截断。后续有界读取及多节点验证见上一节。MLA 冻结启动后增加的诊断路径缩写通过真实提示离线重放：13,093→9,153 字符，18 条错误／Actual／Fix 不变，见 `feedback-replay.json`。

该轮总核验见 `.schema_extract/maintenance/verification_20261102.json`，当时的回归日志为同目录 `extractor-tests-20261102.log`；其中记录两题冻结版本与工具摘要、源码／正式库保持、无 seed／导出、进程结束和验证边界。

## 已有试点与验证边界

2026-11-02 已按用户要求确认停止并清理下面三轮 FP8／GDN 的运行记录、专属快照和暂存产物；源码、正式产物、问题台账及独立 CLI／Supervisor 探针保留。下述路径与数字是历史依据，问题和结论摘要保留在 ISSUES，不应再把已清理路径当作可恢复的 seed。

旧政策下，两题都只在运行区保存，未导出：

| 算子 | 已完成 | 尚未完成 |
| --- | --- | --- |
| `gdn_postconv_mtp_bf16_state_fp32` | root v12，Contract 与 final 均独立审阅为带备注草稿 | 本轮仅核验旧产物的非硬件检查 |
| `fp8_scaled_gemm` | root v10、小 M v8，Contract 和两个节点均独立审阅 | 旧 index / final 因 8M 预算未派发；新政策复用进展见 SE-005 |

本轮新 run `policy_xhigh_20261101_fp8` 已复用上述 FP8 草稿达到 `ready`：root v11（仅硬件占位政策变更）、小 M v8，算法 Contract 和两个 Design 不变；完成 index 与全部独立复核，保留两项非硬件 note。最终检查点 `cp-0003-0bd256ca55`，依据见该 run 的 `verification.json`。未导出；探针与实际 run 的调度均已停用。GDN 没有重新提取。

第二轮 FP8 曾遇旧 900 秒超时并从检查点恢复。该自动超时与预算逻辑现已移除。旧 GDN 累计计量存在续接重复相加，实证差分为 5,874,314；FP8 已报告 8,777,991，另一次超时用量未知。历史报告保持原样，见 `.schema_extract/runs/pilot_xhigh_20261101_r2/supervisor-verification.json`。

回归入口：`python3 -B -m unittest discover -s schema_extract -t .`；检查器使用 `python3 -B -m unittest discover -s schema_template/tools/tests -t schema_template`。独立真实探针保存在 `.schema_extract/probes/policy_20261101/`；已清理运行的历史结论见 ISSUES。
