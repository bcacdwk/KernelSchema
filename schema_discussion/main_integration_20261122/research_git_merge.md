# Git、合并与交付保全研究

日期：2026-11-22 UTC。研究对象仅 KernelSchema；本轮只读 Git/产品/历史证据，仅新增本报告与同目录 `research_git_manifest.json`。未 stage、commit、push、checkout、stash、reset、clean，未启动模型、GPU、Extract、恢复或 monitor。报告是待执行建议，不表示四包已实现。

决策入口：[Extract 审计 README](../extract_schema_review_20261121/README.md)、[FINAL_PLAN](../extract_schema_review_20261121/FINAL_PLAN.md)。完整 SHA、文件类型、分支提交/变更清单与讨论保全元信息见 [机器清单](research_git_manifest.json)。机器清单不含捕获正文、凭据或模型思维。

## 1. 已查明的 Git 事实

| 对象 | 事实 |
| --- | --- |
| main / 本地 origin/main | 都是 `484902855fe7be82a74fe7c394503d1d7d222dbb`；本轮未 fetch，origin/main 是已存在的本地远端跟踪值。 |
| workflow | `feat/schema-workflows-v1`，HEAD `7be9263fe9a6f5fba97ed7ad7685de13e4d6146f`；已知工作树 `.worktrees/schema-workflows-v1` 的 tracked 内容干净。 |
| 共同祖先 | 精确等于 main HEAD；main 独有 **0**，workflow 独有 **37** 个提交。它们的逐项 SHA/标题在机器清单中。 |
| workflow 差异 | **148** 文件：README 1、docs 6、examples 55、kernelschema 47、pyproject 1、schema_design/schema_implement/schema_optimize 各 1、tests 35。只有根 README 是修改，其余为新增。 |
| 无分支差异的共享区 | `schema_extract/`、`schema_template/`、`schema_docs/`、`schema_wiki/`、`AGENTS.md`、`.gitignore`。没有另一份需要择优的 Extract 实现。 |
| 文本合并检查 | 只读旧式 `git merge-tree BASE main feat/schema-workflows-v1` 退出 0；无冲突标记、无 changed-in-both。未用会写树对象的 `--write-tree`。 |
| 未跟踪覆盖检查 | 对 workflow 的 148 个实际 incoming 路径和各父目录逐项检查：main 当前 **0** 个未跟踪同名文件、**0** 个父路径类型阻塞。 |
| 索引/原修改 | 索引为空；只有原来的 `AGENTS.md` 修改及十二个 `.gitkeep` 删除，没有本轮产品修改。 |

已有 Extract 修正属于共同历史：`24d2b3d` 已有首轮运行后修正，`8724f5d` 已有后续 Extract/六算子修订，`4849028` 再精简文档。`runner.py`、`monitor.py`、`wakeup.py`、`executor_repairs.py` 最近产品提交均为 `8724f5d`；名称规范化与文本预算的最近实现提交仍为 `7a8030f`。workflow 完整继承这些修正，不能把四包待办误判为“merge 即完成”。十二项本批 YAML 修正则还在 main 的未跟踪成品里，不存在于任一分支提交。

## 2. 当前文件的三种交付边界

### 2.1 可形成公共历史基线的正式产物

[DELIVERY_INDEX](../extract_twelve_20261119/DELIVERY_INDEX.json) 的十二目录共有 **33 YAML + 12 dispatch + 12 原 publication receipt = 57 文件**。本轮重算 57 个 SHA，全部与 DELIVERY_INDEX 一致；12 个 `.extraction/` 里当前各只有 `receipt.json`。没有 current 文件被混在本批原批准里。

| 算子目录末段 | YAML 数 |
| --- | ---: |
| `bf16_nvfp4_gemm` | 7 |
| `fp8_mxfp4_grouped_gemm_masked` | 1 |
| `bf16_fused_moe` | 1 |
| `fp8_scaled_fused_moe` | 3 |
| `fmha_bf16` | 4 |
| `fmha_fp8_per_token_head` | 3 |
| `mla_absorbed_fp8` | 3 |
| `mla_nvfp4` | 1 |
| `paged_indexed_latent_fp8` | 3 |
| `paged_latent_mxfp8_nvfp4` | 2 |
| `gdn_conv_mtp_bf16_state_bf16` | 3 |
| `kda_raw_prefill_state_bf16` | 2 |

精确相对路径与 SHA 见机器清单的 `formal_delivery_baseline`。基线提交应显式 stage 这 57 个文件及同十二目录的 `.gitkeep` 删除；不要 stage 整个 `schema_wiki` 或所有 `.extraction`。这些 receipt 是 `publication.py` 明确生成的 `schema_extract.delivery.v2` 公共投影，含来源/文件摘要/批准身份，并非原始 reviewer 回复或运行账户。57 文件的凭据形态/私钥/原始 thinking channel 标记启发式扫描无命中；此检查不等于其余讨论捕获已通过公开审查。

### 2.2 必须原位保全的本地材料

- `.schema_extract/v2` 的 snapshots、原 rules、runs、attempt、raw、checkpoint、账目、supervisor 和 clock 状态；本报告没有递归读取或复制这棵运行树。不得靠本次 merge 启动/迁移/重签它们。
- 现有 workflow 资格捕获与已知外部 `/tmp` 原始证据；不复制到仓库，也不把旧实际调用变成新验证。
- 三个未跟踪 discussion 目录的全部原稿、回执捕获、失败记录、历史方案，保留原路径与字节。本轮只对它们做字节哈希与文件类型统计，不把“保留历史”解释成“推送全部私有捕获”。
- `AGENTS.md` 用户新增末段：要求等待子代理主动回报、避免频繁轮询。当前字节 SHA 已记录；**全程保留本地、不混入产品提交**。workflow 未修改此文件，正常 merge 不需要覆盖或 stash 它。最终 main 有这一个已知本地修改不妨碍 main 产品交付。

现有 `.gitignore` 只忽略缓存和 `.schema_extract/`，`.git/info/exclude` 另有 `/.worktrees/` 和 `/kernelschema/runtime/qualification/`。它们**不保护 discussion 下的原始捕获，也不普遍忽略 `.extraction`**。因此禁止 `git add .`、`git add -A`、`git commit -a` 或按目录批量 force-add。没有必要为本任务修改 ignore 规则。

### 2.3 discussion 的公开历史与本地保全

| 未跟踪目录 | 原文件数 | 总字节 |
| --- | ---: | ---: |
| `extract_twelve_20261119` | 550 | 17,910,689 |
| `extract_schema_review_20261121` | 29 | 527,275 |
| `workflow_architecture_20261120` | 116 | 1,868,288 |

机器清单固定了这 **695** 个普通文件的 SHA/大小；未发现符号链接或其他特殊文件。文件名初筛只是保守分流，**不是逐文公开审查**，不能把其中的 candidate 标签直接转为 stage 清单。

推荐由主代理维护单一公开 allowlist：

1. 优先逐项审阅并提交十二项的 `README/FINAL_REPORT/ISSUE_RECAP/DELIVERY_INDEX`、审阅方案的最终入口和必要前轮报告/离线 fixture/机器结果，以及 workflow 的既有设计/实施历史和本次仲裁、执行记录。它们都使用精确路径，不能按 `.md` 或 `.json` 后缀盲选。
2. `response/result/native/stream/stdout/stderr/prompt/invocation/process/account` 等原始运行捕获，及所有未审查内容，仍本地保留。capture 文件中即使某个字段看起来只是状态，也不整文件推送。
3. 新建公开历史索引，为每份原文件记 path/SHA、公开或本地保留、理由；遇到不能公开的原报告，写单独的公开摘要，原稿不改。已公开报告所依赖的私有或 `/tmp` 证据必须在索引中明确为本地历史依据，不能声称 fresh clone 已包含或验证了它。
4. 本次新增状态放 `main_integration_20261122/`；不要把旧 `FINAL_PLAN` 的“待实施”、旧验证计数或旧失败记录改成新的完成记录。按 FINAL_PLAN 要求新增的 `CURRENT_SCHEMA_INDEX.json` 可放原审阅目录，但这是新文件，不改签旧结论。

这样最终 main 包含全部产品、可公开设计/修订历史和完整历史元信息；私有原字节仍完整留在本地。本轮未完成 695 文件的逐文公开裁定，这项必须在最终 stage 前由主代理完成。

## 3. 真正的共享接点与语义风险

| 接点 | 现状及合并后要求 |
| --- | --- |
| `kernelschema/template_api.py` | 以私有命名空间动态载入现有 Template，没有复制实现；W1 名称修复和 W2 文本预算会直接影响严格路径。复核 normalization 的对象身份与第二次无改动。 |
| `kernelschema/_writing_policy.py`、`writing.py` | strict 路径调用原 normalize/validate/text_policy，并比对冻结源码 SHA；修复后旧冻结规则可能拒绝继续，这是既有身份边界。新运行重新冻结实际规则，不能改写旧 bundle 来“通过”。 |
| 默认 write_only | 不进行 strict 的 Template normalize/text_policy/reviewer 门禁；保留用户已采用的默认行为，不把 W3 Extract 回执义务移植到 writing。 |
| `resources.freeze_rules` | 记录实际 docs 和 Template Python 的完整字节；W1/W2 后测试与新请求必须冻结新字节。只改归档旧规则或只改一个 tool version 都不够。 |
| `extract_bridge.extract` | 独立解释器调用原 `extract_yaml.py`，应保持参数、帮助和错误退出码透传；测试只调用 help/无效参数，不能真正启动提取。 |
| `extract_bridge.consume_raw` | 可选 receipt 只能是原 `schema_extract.delivery.v2`，其 `files` 必须匹配当前目录。W2 改后用 `origin=curated, receipt=None` 导入新 raw，current 另作普通 provenance；带旧 receipt 必须仍因 SHA 不同拒绝。无需为 current 新增 P6 门或修改 publication。 |
| `pyproject.toml` | 默认 pytest 只发现 `tests/workflows`。一次仓库根 pytest 通过不能说明 Extract/Template 的修正已验收；两套旧 tests 必须显式运行。 |
| 入口说明 | root README 由 workflow 修改，`schema_wiki/README.md` 目前仍只列旧六项。最终公共入口需显示十二项并区分 current/原 receipt；由末轮整合 owner 单写。不要宣称完整旧 receipt 验了修订新字节。 |

没有发现需要先改 AKA 的 KernelSchema 合并阻断。既有 optional backend 保持按显式 implement 请求加载；本次不导入、不测试或调整外部 AKA。

## 4. 唯一推荐顺序：旧交付基线 → 正常 merge → 四包 → 审定历史 → push main

主代理始终留在现有 main 工作树即可；本次不需要切换旧工作树、另建产品副本或制造一轮暂时丢失正式目录的 checkout。Git 动作全部由主代理负责，执行代理只改分配文件、交付检查结果。

1. **固定不可变旧交付。** 重核 HEAD、workflow HEAD、57 文件 SHA、AGENTS SHA 和索引状态；从机器清单生成精确 allowlist，提交 57 个公共文件与 12 个 `.gitkeep` 删除。可同提已审查的 DELIVERY_INDEX 和简短历史说明。基线 commit 明确“原交付快照，已知审计修复尚待实施”，不宣称重新验证。它让全部旧批准字节可由 Git 直接还原。
2. **保留 workflow 原历史。** 在 main 执行普通 `git merge --no-ff feat/schema-workflows-v1`，保留原 37 个提交及一个明确汇合点。以最新状态再次检查 incoming 路径；若出现新阻塞，先保全并解释新增来源，不能 `clean/reset` 消掉它。
3. **第一轮先 W1，内部设屏障，再做 W3/W4。** 两位代理按下面文件所有权工作；两半 W1 都完成定向验收后，再各自继续 W3/W4。不因为同轮并行而越过 FINAL_PLAN 的先修要求。主代理按精确 diff 收口；可保留 W1 与 W3/W4 独立提交，或让执行报告明确各包完成证据。
4. **第二轮 W2 与独立整合。** W2 完成有限来源复核、实际 raw/preview 对齐和新 Version；独立整合 owner 在最终字节交接后核对 current、依赖闭包、bridge、文档入口与必要 CPU 回归。W2 相关更正、预算、规范、证据与 current 必须同一交付单元，不能先推其中半包。
5. **最终提交与公开历史审定。** 逐文件确认 staged 集只有授权产品、已审历史、轻量新证据；原 receipt/DELIVERY_INDEX/规则与本地捕获 SHA 保持原义。旧讨论无需为产品落 main 先全部 stage，也不能遗忘 final allowlist/索引。
6. **发布。** 由主代理 fetch 核远端并确认正常快进关系，检查整条 main 已包含 workflow HEAD 与四包完成提交，然后正常 `git push origin main`，核远端 main 等于交付 HEAD。禁止 force-push。远端若有并发新增，正常整合该新增并做受影响复核，不覆盖。用户已授权最终进入 main/push，不再另设无依据审批。

预期历史形状：`4849028 → 旧交付基线 B → merge M(第二父=7be9263) → W1 → W3/W4 → W2+current → 完成交付记录`。其他合法小提交可以保留，关键是有可还原 B、原 workflow 祖先关系和真实实施记录。

未采用其他顺序：

- **先 merge 后保存成品**：当前技术上可行且无同名覆盖，但在 W1/W2 改写前尚无 Git 旧字节还原点，审计收益较低。
- **先修四包后 merge**：增加在非最终共享 Template/workflow 环境验收的机会，最后仍须重核共享缝；不必要。
- **复制文件、squash 或 cherry-pick 37 提交**：没有冲突需要如此处理，会丢掉自然分支汇合事实或制造重复历史。
- **stash/clean/reset，再 clean merge**：不必要；未跟踪交付与本地证据保全风险明显，AGENTS 本来也不与 incoming 冲突。

## 5. 恰两轮、四位执行代理的建议所有权

| 轮次/代理 | 唯一写入所有权 | 交付与等待点 |
| --- | --- | --- |
| R1-A | W1 的 `schema_template/tools/normalization.py`、相关 normalization/事务 tests；P2 的 `runner.py`、`prompts.py`；随后 W3 的 `receipts.py/reading.py/agent.py/runner.py/prompts.py`、对应 response/receipt_repair prompt 和 tests。 | 先交名称/P2 W1 反例与回归；等待 B 的控制 W1 完成，随后贯通 W3。runner/prompts 始终同一人写。 |
| R1-B | W1 控制与 W4 的 `monitor.py/cli.py/store.py/wakeup.py`、`schema_extract/README.md`、对应控制/观察/时钟 tests。 | 先交提交事实、ID 对账和副作用计数；与 A 完成 W1 屏障，再交 W4。inspect/record_scan/Clock 由一人保持字段一致。 |
| R2-C | W2 的 `text_policy.py`、指定三份 docs、P2/P4 phase prompt、九 YAML/两 dispatch、九个受影响目录的 current、`CURRENT_SCHEMA_INDEX` 和有限源复核/检查证据；对应必要 Template tests。 | 等第一轮稳定；在候选副本处理 normalization，最终落盘再验同一字节；保留旧 receipt，旧字节用 B 还原。 |
| R2-D | workflow 共享接点的有限回归及必要补充、`tests/workflows`、root/Schema Wiki 入口、公共整合记录；为主代理准备历史公开 allowlist/安全摘要，不改历史原稿。对 C 的更正与 current 独立只读复核。 | 在 C 改稿期间可做共享 API/旧规则行为验证；接到 C 最终 SHA 后再复核来源/Version/current/raw。发现问题反馈 C 修，不同时编辑 YAML/current。 |

其他文件若确实必要，先向主代理交代依赖并重新分配，不能代理间自行抢写。尤其 A 不顺手改 monitor/README，B 不顺手改 runner；D 不建立平行的 Template 或 receipt 修复机制。执行代理不 stage/commit/merge/switch/push；主代理负责这些动作与最终精确 allowlist。这里的“等待”是必要的包依赖交接，不要求高频查询子代理。

## 6. 必要验证与可复用环境

本研究未跑产品 test suite。已做的检查只有 Git 关系/路径阻塞、文件 SHA/类型、公共基线敏感标记初筛和 Python 包版本检查，不能把它们写成产品修复验收。

- `/opt/venv/python3.13.12/bin/python3`：Python 3.13.12、PyYAML 6.0.3、jsonschema 4.26.0，**无 pytest**；可复用跑 Extract/Template 的 unittest 与只读目录检查。
- `/tmp/schema-b18-model-budget-usage-20261121/venv/bin/python`：同 Python/PyYAML/jsonschema，pytest 9.2.0，已有 `kernelschema` console script；**默认导入的是旧 `/tmp/.../fixed-ks/kernelschema`**。运行新代码时必须显式设置 `PYTHONPATH=/home/shaohanyong.shy/KernelSchema` 并先核 `kernelschema.__file__`、`resources.checkout_root()`。不要为了测试改装此历史环境。
- 两个 venv 的 bin 目录均无 `claude`/`codex`；执行时仍限定 PATH 并核找不到真实模型 CLI，使用 `PYTHONDONTWRITEBYTECODE=1`、`-B`、`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`。workflow pytest 可加 `-p no:cacheprovider`；只用 fake backend、临时 Store、假时钟。

针对性范围：

1. W1：FINAL_PLAN 指定的 normalization/artifact transaction/stage flow/maintenance receipt 测试及真实离线反例；控制登记前、登记后 Popen 前、Popen 后写回前故障注入。不能只看 checker exit0，还要证明数学对象未合并、持久 request ID 和副作用次数正确。
2. W3/W4：对应 proposals/receipt_repair/call_limits/backends、monitor_evidence/wakeup_retry/usage observation 的必要测试；计费与身份门禁必须留存。测试模块可能包含多项场景，按实际改动选择，不以触发真实 monitor 来验证。
3. W2：九个受影响完整算子目录的 `check_yaml.py root.yaml --read-only --skip-hardware` 与 dispatch 检查，确认 preview/raw 同内容后固定 current。九 YAML 的增版与有限来源审阅、两 dispatch 限制保留、九条 Tolerance 原 JSON 对应、provider/consumer 和容量副本核对独立记证据。
4. workflow：`test_template_extract.py`、严格 writing 中冻结规则/原文本预算/规范化字节测试、`test_write_only.py` 的默认行为，以及实际受影响的 iteration/API smoke。旧 receipt 验旧字节成功、验改后字节失败，`receipt=None` 导入新完整目录成功。帮助/非法参数透传保持退出码；不启动模型、GPU 或真实 Extract。
5. 主代理只汇总实际命令、退出码、固定源码与文件 SHA、未覆盖限制。以上相关检查通过且无新增问题后，不扩成全仓库、全部历史 run、全六阶段或真实模型/GPU 验收。

## 7. 保全与本轮范围记录

- 新增文件只有本报告与机器清单；结束时 main/workflow HEAD 不变，index 仍空，tracked 原修改未变。
- 未操作其他旧工作树，未访问 AKA 仓库或实验仓；未打开认证文件或运行账户，未展开或输出模型原始思维。对 discussion 原始捕获只计算字节哈希，不输出正文。
- 有一项只读命令范围偏差已向主代理报告：查询 `8724f5d` 的历史提交统计时，宽 `schema_extract` 路径选择意外列出了该提交里的历史 archive 路径名。未打开这些文件内容或当前 archive 目录；随后已停用宽统计，改为精确活跃文件查询。后续执行不应复用该宽查询。
