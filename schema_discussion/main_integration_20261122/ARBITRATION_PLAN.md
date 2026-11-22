# main 整合仲裁与四人执行计划

状态：**plan_ready；产品尚未开始修改。** 2026-11-22。唯一需求依据仍是用户已批准的 [FINAL_PLAN](../extract_schema_review_20261121/FINAL_PLAN.md)、[COVERAGE](../extract_schema_review_20261121/COVERAGE.json)、[FINAL_REVIEW](../extract_schema_review_20261121/FINAL_REVIEW.md)。本文件只裁定合并、依赖、归属和验收的落地方式，不恢复旧轮次被删除的建议。机器分工与精确路径见 [execution_map.json](execution_map.json)。

主代理已获实施及最终正常推送 main 的授权，无须再次确认。本计划使用**恰两轮、每轮两名新的 xhigh 执行者 A/B、C/D**；同轮内交接不是另一个执行轮。不增派代理，不改 AKA，不运行真实模型、GPU、Extract 或 monitor。

## 1. 独立裁定

1. **在原 main 工作树先提交原交付，再正常 no-ff merge，安全且应直接执行。** 仲裁重读 HEAD、workflow HEAD、空索引，重新哈希全部 57 个正式文件，均与研究清单相符；AGENTS 用户改动 SHA 也未变。workflow 只修改根 README、其余新增，未碰本批成品或 AGENTS。无需 stash、checkout、复制产品树、reset 或 clean。
2. **W1 屏障必须验证同一组最终源码。** A/B 各自完成 W1 后先停止产品写入；双方冻结后才做屏障定向测试、报摘要。主代理核摘要并提交 W1，再同时放行 W3/W4。各自先测试旧的另一半、随后口头宣布通过，不足以跨过屏障。
3. **W2 的正文、规则、独立审阅、current 有明确单向顺序。** C 冻结 19 YAML/9 dispatch 及规则 → D 独立有限源复核并冻结证据 → C 生成九份 current/总索引 → D 只读终检。D 不同时改 C 的正文；current 不反向成为语义审阅报告的摘要对象。
4. **不需要因合并而重写 workflow。** 直接检查了私有 Template loader、规则快照、consume_raw 和既有测试：共享真实 Template 已足够。默认 write_only、显式 strict、标准 AKA 委托仍是各自已有接口。D 先验证，只有已证实的共享接口回归才做最小兼容修补。
5. **公开历史审阅归 D，不留给主代理逐读 695 份原稿。** D 审阅少量候选全文，给可公开精确 allowlist、SHA 与全部旧文件的保全元信息索引；其余默认 local-only。主代理只核 allowlist/摘要并精准 stage。不得以“保留自然历史”推送 raw、thinking、账户或未审查捕获。
6. **入口数量应为 18 项/48 YAML。** 根 Schema Wiki 原有六项/15 YAML；本批增加十二项/33 YAML。D 保留旧六项导航，追加十二项，并分别标注本批九 current 与三原 receipt；不能把旧六项覆盖成“全库只有十二项”。
7. **环境事实以当前实测为准。** `/opt/venv/python3.13.12/bin/python` 有 PyYAML/jsonschema，当前没有 pytest。既有 B18 环境有 pytest 9.2.0，但默认 editable 指向旧源码。采用第 7 节的两种解释器与显式 main 导入，不安装软件或改共享环境；这不是产品阻塞。
8. 当前没有必须用户另选的技术或语义事项。只在出现无法通过现有来源、授权及本计划自行裁定的真实冲突时报告问题；预算降低、新 P6 或 AKA 修改不属于当前任务。

仲裁实际只做文件/接口读取、57 文件哈希、Git 只读及 Python import 探测；没有跑产品 test suite，也没有将研究阶段的检查写成修复完成。

## 2. 主代理合并前步骤与不可变边界

工作树固定为 `/home/shaohanyong.shy/KernelSchema`，main 起点 `484902855fe7be82a74fe7c394503d1d7d222dbb`，workflow 起点 `7be9263fe9a6f5fba97ed7ad7685de13e4d6146f`。主代理已用 ls-remote 核远端仍为这两个提交；最终推送前仍正常 fetch。

顺序：

1. 重核 HEAD、空 index、57 文件 SHA 和 incoming 路径无覆盖。以 `research_git_manifest.json.formal_publication_stage_boundary` 的精确路径，仅 stage **57 文件 + 12 个 .gitkeep 删除**。不按目录 add；本基线提交暂不带 discussion，避免把历史公开审阅提前交给主代理。
2. 提交历史基线 B，提交说明明确“原交付字节，后续审计修订待实施”。逐项核 `B:path` 与原 SHA 相符。33 YAML、12 dispatch、12 receipt 由这个提交可还原；这足以保全原批准对象，不再复制一个历史目录。
3. 执行普通 `git merge --no-ff feat/schema-workflows-v1`，记合并提交 M，第二父为 workflow HEAD，37 个原提交全数可达。merge 不触发任何 run、模型或 monitor。若出现与已查事实不符的新阻塞，保留现场，先核来源；不得清理未跟踪文件。
4. 仅此后启动 A/B。W1 屏障通过时形成独立 W1 提交；第一轮全部完成后收口 W3/W4，再启动 C/D。W2 正文、必要规范/预算、有限证据与 current/index 作为同一可 review 的交付单元收口，不先推半包。

全过程保全：

- `AGENTS.md` 当前 SHA `e21f835840354b48eb2d1b90fccbb5cf0d7e8c50c6054d2c3de0d0149d4180d1`，留本地、不 stage。
- 12 原 receipt、旧 `DELIVERY_INDEX.json`、旧 FINAL_PLAN/COVERAGE/FINAL_REVIEW 及原稿，SHA 与旧义务不变；Source 仍指固定来源提交，不能改为 merge HEAD。
- 三个旧 discussion 的 695 个原件按既有元信息原位保全；不搬移、不重写。D 最后按现成精确路径核 SHA，不通过递归搜索发现材料。
- `.schema_extract/v2` 的原 rules、checkpoint、attempt、账户、state 和 supervisor 不写入。本批 12 run 已 ready/P6/published、无 active/unconfirmed/unsettled；clock disabled/generation 10。旧 daemon.status 字段不构成启动或修状态的理由。只按已知精确 state/supervisor 路径核摘要，不遍历运行树。
- 禁止访问、枚举、运行或修改 AKA_experiment 和任何 archive 目录；基线中的 42 个 archive 条目只从已读元信息中排除，不访问目标。禁止 home 递归搜索。本轮不访问或修改 AKA 仓库。

禁止 `git add .`、`git add -A`、`git commit -a`、目录批量 force-add、force-push，以及为合并使用 stash/reset/clean。子代理均不执行 stage/commit/merge/switch/push。

## 3. 唯一文件所有权和交叉边界

下列路径相对仓库根；完整展开路径在 execution_map 中。所有者可读别人的文件，只有唯一所有者可写。新测试优先放现有对应模块；不改共享 fixture/support 来省事。

| 所有者 | 产品写入 | 测试写入 | 本次记录 |
| --- | --- | --- | --- |
| A，第一轮 | `schema_template/tools/normalization.py`；`schema_extract/tools/{runner,prompts,receipts,reading,agent}.py`；`schema_extract/prompt/operations/{response,receipt_repair}.md` | Template `test_normalization.py`；Extract `test_artifact_transactions.py`、`test_stage_flow.py`、`test_maintenance_receipts.py`、`test_proposals.py`、`test_receipt_repair_policy.py`、`test_call_limits.py`、`test_backends_runtime.py`、`test_prompt_materials.py`、`test_current_returns.py`、`test_current_recovery.py`、`test_review_protocol.py`、`test_issue_evidence.py`、`test_source_gaps.py` | `w1_a.json`、`ROUND1_A.md`、`round1_a/` 的轻量证据 |
| B，第一轮 | `schema_extract/tools/{monitor,cli,store,wakeup}.py`；`schema_extract/README.md` | Extract `test_monitor_evidence.py`、`test_wakeup_retry.py`、`test_cli_recovery.py`、`test_maintenance_polling.py`；必要新增观察/控制用例放前三个模块 | `w1_b.json`、`ROUND1_B.md`、`round1_b/` 的轻量证据 |
| C，第二轮 | `schema_template/tools/text_policy.py`；DOC1/4/5；P2/P4 phase prompt；第 5 节九 YAML/两 dispatch；九 current；新增 `extract_schema_review_20261121/CURRENT_SCHEMA_INDEX.json` | Template `test_yaml.py`、`test_schema.py`、`test_dispatch.py`、`test_design_resolution.py` 的必要增量；不为短文案写镜像测试 | `W2_SUBJECT.json`、`ROUND2_C.md`、`round2_c/` 的检查/预算证据 |
| D，第二轮 | 根 `README.md`、`schema_wiki/README.md`；确有共享回归才最小改 `kernelschema/{template_api,extract_bridge,resources,_writing_policy,writing}.py` | `tests/workflows/{test_template_extract,test_write_only,test_writing,test_iterations,test_standalone,test_api}.py` | `W2_REVIEW.json`、`ROUND2_D.md`、`IMPLEMENTATION_COVERAGE.json`、`PUBLIC_HISTORY.md`、`HISTORY_PUBLICATION.json`、`FINAL_REPORT.md`、`round2_d/`；最终本目录 README/progress |
| 主代理 | 无并行产品编辑；Git 与流程交接 | 不承担第二套产品测试 | `W1_GATE.json`、Git 提交/远端核验记录；最终交付消息 |

DOC1/4/5 是 `schema_docs/1.contract.algorithm_requirement.md`、`4.design.work_schedule.md`、`5.design.tile_pipeline.md`；phase prompt 是 `schema_extract/prompt/phases/02_dispatch_draft.md` 和 `04_dispatch_plan.md`。所有本次记录默认位于 `schema_discussion/main_integration_20261122/`。

关键边界：A 的 runner/prompts 从 W1 到 W3 始终同人写；B 的控制 ID、inspect、record_scan、Clock 字段始终同人写；C 的正文/Version/current 始终同人写；D 仅只读复核 C。A 不改 cli/monitor，B 不改 runner；C 不把义务改动塞进 operations prompt；D 不另建 Template 或 receipt 实现。

如确需表外文件或第二轮发现第一轮真实缺陷，先由主代理在本次记录中精确转移单个文件所有权，确保原 owner 已停写，再由现有执行者完成和复验。不得抢写或增派第五人，也不得把必要修复列为第三轮待办。`api.py`、`iterations.py` 和默认 AKA 路径默认不需要修改。

## 4. 第一轮屏障与验收交接

### A 的 W1

- S-06：声明身份、原始拼写与目标拼写分开；casefold 只发现潜在碰撞。Math 仅同步有明确绑定的引用；未登记临时量/开放记法保留，必须改的登记名可能捕获或穿过未知绑定时整组写前拒绝。不建立全 Math 解析器，也不只删 `related`。
- 保留事务、原稿、provider/兄弟不变；覆盖普通与 `:=` 的 `V=v+1; y=V+v`，v=1 必须保持 3 或写前拒绝，不能得到 4；反向大小写、局部/外层、双局部、provider/consumer、安全同对象改名与二次幂等都核到。
- X-15：普通及维护 P2 reviewer 检查已经读到的 `state.plan_draft`。`revise/plan=[]` 合法，不能构造空候选产生“缺 root”；approve 仍核相等，adjudication select/patch 保持自己的候选；首稿待审不误标 removed。坏目录、偷改 plan、reviewer 改 raw 仍拒绝，不改旧 check 计数。

### B 的 W1

- X-08：操作级 request/launch 的已提交事实和原 ID 必须在后续 busy/投递异常中保留；明确未提交 false、已提交 true、无法确认 null/unknown。不能以全局 ledger revision 代替本操作事实。
- 核 restart/start、maintain 后启动、extract 登记后启动（单项与批量）、hold release 后启动、resume/export 预约后运行、clock 登记后 spawn/signal；cancel 保留已有 request/delivery 分离。inspect 按 scope 投影待消费/最近请求，丢返回后能只读对账。
- 在真实临时持久账本注入登记前、登记后/Popen 前、Popen 后/写回前三处 busy；核 ID、ledger 与副作用计数。重复请求不第二次启动/增额；旧 launch 不能 claim 新预约；最新 pause/cancel/hold 优先。锁等待只限单事务，不重放整命令/Popen/signal/apply。

### 两边共同屏障

1. 各自完成 W1，发 `W1_CODE_READY` 给主代理与对方，列精确产品/测试路径、SHA 和准备运行的定向命令，然后停止产品写入。此时不写 W3/W4 产品。
2. 主代理收到双方 ready 后发 `W1_VERIFY_FROZEN`。A、B 各在同一冻结树运行自己的 W1 定向验收；Template 与 Extract 使用独立解释器进程。完成分别写 `w1_a.json`、`w1_b.json`：W1 项目、输入源码/测试 SHA、命令、解释器、exit code、真实断言结果、日志路径/SHA、限制，并报 `W1_VERIFIED`。测试只写临时目录。
3. 若任一失败，由 owner 修复，再共同冻结、只重跑受影响验证；此前 ready/verified 自动失效。两份通过记录对同一冻结树有效后，主代理核摘要、精确提交 W1，写 `W1_GATE.json`（引用两份证据及 W1 commit），发 `W1_GATE_RELEASED`。
4. A 此后做 W3，B 做 W4。两人不退出任务去等待“下一轮”；等待只为当前必要屏障。W3/W4 改到 W1 同文件，第一轮最终定向套件须再次包含各自 W1 回归，不能只引用早先通过。双方最终代码 ready 后保持产品字节稳定，各自完成验收并记录同一最终共享树的输入摘要；若此前测试所依赖的另一半代码随后变了，只重跑受影响验证，再由主代理收口第一轮。

### A 的 W3 完成条件

- 现有 receipts gate 为唯一权威：P2/P3/P4 且 approve 的 handoff 非空、1200 Unicode 字符/20 非空行；schema description 与动态 prompt 同源，修改 schema 副本不污染其他阶段；P5 不套这个限制。
- 类别/locator 从当前 STAGES/bindings 投影；同址 host 不抵消 implementation，每类任一允许 locator 即可。工具不代造引文、坐标或 summary。
- 贯通 schema_errors，在原路径合并能独立安全判断的类别/坐标/索引/handoff 错误，保留原数组索引和真实计数；前置失败不读依赖材料。截断说明省略数量与未检查部分；新 proposal 全量重验。
- 先分 native 双通道矛盾，再只取消普通 receipt repair 的额外累计 cap；沿已有 scope `call_exhaustion`，不清次数、不加额度。坏回执拒绝批准、可信草稿未批准保留、原角色可否重交分别判定。普通第三次有额度可继续；错误 decision/清单外选择可补正但当前错误不执行、不越界读；nonce/身份/权限/来源/字节失败仍停。native/CLI 有限策略、未终止 fence、预留 reviewer/裁决/签收、单次裁决/签收、维护/用户控制均保留。

### B 的 W4 完成条件

- `read_summary` fallback 只读一次 state，显示此次样本；stale/missing/invalid 是缓存状态，保留 revision/采样时间/ctime 和 `deep_verified=false`。读中变化报变化/未知，不写缓存或状态。
- 当前 anomalies 仅在 `record_scan` 产生，旧 rows/superseded IDs 保留、展示前标历史，Clock 消费同一结果。迟到旧失败不能覆盖新 ready 或倒退当前计数/行动提示；下游不复制过滤器。
- disabled→显式新代际重置 initial scan/due；同 enabled 代际 daemon 恢复保留原 ID/ack/claim/due。旧发送未结清仍 fenced，旧代际不能确认新提醒，不重复发送。
- README 只补输出叶目录须不存在，并说明缓存/样本、提交/投递、未知提交不可据此重放。用假时钟/临时 Store 测，不启动实际监控。

## 5. W2 的确切范围与单向交接

以下简写以 `schema_wiki/aka-cuda-operators-20261030/operators/` 为根。只改这九 YAML 的语义/文字并各增一版；另 24 YAML 原字节、Version、Source 不变。

| 文件 | Version | 必须修正 |
| --- | --- | --- |
| `gemm/grouped_masked/fp8_mxfp4_grouped_gemm_masked/root.yaml` | 4→5 | 非空 tile `eff=min(128,masked_m[e]-128*mb)`、`eff_a=16*ceil_div(eff,16)`；Contract y 等式只在 `i_l<eff`，额外非活跃行保留实际计算/窄化/写回且不引用未定义 y；撤销全域解码/乘积精确和仅改变归约顺序的保证，分别写 FP32 求值与实际 block-scaled 路径。不缩输入域或改阈值。 |
| `fmha/fmha_fp8_per_token_head/root.yaml` | 4→5 | m_run/l_run 各 512→2048 B，page_regs 512→2048 B；现有 shape 加复制轴、total_size 同步，数学行数不变。 |
| `fmha/fmha_fp8_per_token_head/decode_split_kv.yaml` | 2→3 | m_run/l_run 各 256→1024 B，保留 slice:4 并计四 lane 副本；page_regs 的 1024 B 活跃 payload 不再乘四。 |
| `linear_attention/kda_raw_prefill_state_bf16/prepare_chain.yaml` | 2→3 | 删除未读 state-index 占位指针必须 null 的断言；仍不启用间接状态索引/FP32 state，CHAIN 按 request 取 slot，保留 ki_f32 两条窄化路径。 |
| `moe/bf16_fused_moe/root.yaml` | 4→5 | y_result 恢复完整 Tolerance 标签、mode、显式 null。 |
| `mla/mla_absorbed_fp8/root.yaml` | 4→5 | attention_output/attention_lse 恢复 mode；lse 的 absent 改显式 null。 |
| `mla/mla_nvfp4/root.yaml` | 2→3 | out/lse 恢复 mode、显式 null、完整 max_mismatch_fraction 标签。 |
| `sparse_attention/paged_indexed_latent_fp8/root.yaml` | 4→5 | out 恢复 mode、完整 max_mismatch_fraction/max_rel_l2 标签。 |
| `linear_attention/kda_raw_prefill_state_bf16/root.yaml` | 3→4 | attention_output/final_state/checkpoint_states 恢复 mode/null。 |

两 dispatch：`linear_attention/gdn_conv_mtp_bf16_state_bf16/dispatch.md`、`sparse_attention/paged_latent_mxfp8_nvfp4/dispatch.md`。只将过程措辞改为当前来源支持范围/限制；保留 GDN fence→tid0 arrival 缺聚合、out4B 对齐/vec==0 等具体条件，以及 mixed Sparse 原题回执/最终题目哈希、比较器/逐输出/机制 A/B 缺口、SM103 采样边界、NCU 6/10、shape 0/1/3/9 性能例外。两目录 YAML 不因此增版。

九条 Tolerance 全部保留 `atol=0.01, rtol=0.05, relative_to=candidate, mode=elementwise_and_l2`，再按下表保留完整字段名与数值；null 与缺失不同，不推断免检或通过。保留 source-only、比较器定义未齐和不补默认值的边界。

| 输出 | max_mismatch_fraction | max_rel_l2 |
| --- | --- | --- |
| BF16 MoE y_result（input.y） | 0.0 | null |
| absorbed MLA attention_output（out） | 0.001 | 0.01 |
| absorbed MLA attention_lse（lse） | 0.0 | null |
| NVFP4 MLA out、lse | 0.0 | null |
| indexed Sparse out | 0.001 | 0.01 |
| KDA attention_output、final_state、checkpoint_states | 0.0 | null |

五个固定 ACCEPTANCE.json 的 path、SHA、JSON pointer、输出别名见 `execution_map.json.w2.tolerance_sources`，原依据是 `round2/schema_contract_evidence/result.json`。仅补这五 root，不能给其他算子添加臆测默认值。

完整检查/清单为九目录、19 YAML、9 dispatch：

| 目录末段 | YAML 闭包 |
| --- | --- |
| fp8_mxfp4_grouped_gemm_masked | root |
| bf16_fused_moe | root |
| fmha_fp8_per_token_head | root、decode_split_kv、tcgen05_prefill |
| mla_absorbed_fp8 | root、simt、split_kv |
| mla_nvfp4 | root |
| paged_indexed_latent_fp8 | root、cluster_m128、split8_merge |
| paged_latent_mxfp8_nvfp4 | root、tcgen05_tmem |
| gdn_conv_mtp_bf16_state_bf16 | root、fused_single_launch、warp_row_reduction |
| kda_raw_prefill_state_bf16 | root、prepare_chain |

每个非 root 隐式继承 root Contract；五个 Tolerance root 覆盖十个有效节点。显式 from 还包括 GDN fused_single_launch 的 Math_reshape→root。用现有 Template resolver 取得真实闭包，不能只看显式 from 或只查九个改动文件。另三未改目录仍为 14 YAML/3 dispatch，匹配原 receipt。

C 同包完成：Tolerance BUDGETS 仅 300 Unicode 字符/2 逻辑行；DOC1 的来源字段保全、Input buffer/异名 Output result/Tolerance 对应、真实 scalar Input/异名 extent 等式短例；DOC4/5 的逻辑 worker/物理先后接管、旧存储与在途访问退休；DOC5 独立存活副本容量、and/or 左到右短路且只免未求值分支定义；P2/P4 未来线索归“范围”，正式变体只链实际完整基线。不改 enum/evaluator、其他预算、P3/P5 checklist。

交接过程：

1. C 按 B 固定旧 SHA/Version/Source/receipt，在临时候选副本做必要规范化审查。只将批准的九 YAML/两 dispatch 更正落盘。未改节点出现 normalization 差异应先查 W1 兼容问题，不能自动批量改稿或虚增版。
2. C 对九完整目录运行最终 main 的原 checker `--read-only --skip-hardware`，另核 dispatch；比较 raw 与 preview 实际字节，不能用“预览 exit0”替代实际落盘验收。冻结 `W2_SUBJECT.json`：19 YAML/9 dispatch SHA、Version/Source、规则/检查器/证据 SHA、真实依赖与命令退出码；报 `W2_SUBJECT_FROZEN`，停止这些产品字节写入。
3. D 针对同一 subject 做独立有限源复核，覆盖九 YAML、两 dispatch 及真实继承消费者。masked 复用既有 4616 输入索引算术；FMHA 核四 lane/四 warp 独立存活副本，区别 payload 与编译器分配；KDA 核未读非空占位地址和两窄化路径；九 Tolerance 逐项对五个固定 JSON 全标签/null/别名；原风险/阈值不漂移。公式或来源不变不重算旧枚举，不重审全 33 稿。
4. D 问题由 C 更正；受影响 subject/check/review 全部刷新后才通过。D 输出并冻结 `W2_REVIEW.json`，只引用 subject/来源/检查证据，不包含尚未生成的 current SHA。报 `W2_REVIEW_PASS`。
5. C 生成九 `.extraction/current.json`，复用 `schema_wiki.current_schema_manifest.v1`；列整个当前目录、Version、Source/源 SHA、隐式与显式依赖、有限语义/结构证据、原 receipt 路径/SHA；明确 `acceptance.runner_p6_receipt:false`。新增旧审计目录 `CURRENT_SCHEMA_INDEX.json` 仅投影九 current。current 引用 D 冻结报告的 SHA；C 此后不改该报告或 subject。
6. D 终检 current/index 与实际字节、source/依赖/证据一致，旧 receipt/DELIVERY_INDEX 原样，另 24 YAML 不变；做 bridge 的旧 receipt 拒绝新字节与 curated 无 receipt 接受检查。任何 reviewed 字节变化使相应检查/审阅/current 失效，必须在第二轮闭环，不能写“稍后补审阅”。

## 6. workflow 边界与最小公开历史

### workflow 必须保持

- 私有 `kernelschema._template_tools` 使用原 Template，normalize plan、resolve/effective_document、异常语义保留；不复制 parser，不以全局 `sys.path` 覆盖两个 tools。
- 默认 design/optimize 仍 write_only：保留原 YAML/诊断，单次 writer、不隐式 review/evaluator、无新增 Template/效果/GPU 门。W3 是 Extract 原角色回执规则，不套给 writing/native workflow receipt 或 iterate。
- 显式 strict 使用新冻结规则中的真实 normalization/BUDGETS；新 Rules 冻结实际 docs/Template 完整源码。旧冻结规则漂移按已有身份拒绝，不修改旧 Rules/usage，也不假装只改 tool.version 就够。
- Extract bridge 只测试原 CLI `--help`、非法参数与退出码透传；不启动提取。`committed=null` 是未知，不能被 wrapper 转成可重放 false。
- 提供 receipt 时只接受原 `schema_extract.delivery.v2` 且摘要匹配；修订新字节配旧 receipt 必须拒绝。修订目录可 `origin="curated", receipt=None`，current 仅普通 provenance，不能塞进 receipt 参数。不改 publication、不加 P6 gate。
- 标准 `api.implement(schema=...)` 原样 mock 委托 `aka.workflow.schema.standard`；不要求 Extract receipt/current、旧 qualification 或写作批准。不读 AKA 仓库。iterate 仍按 Schema 版本计 rounds、每版完整标准 campaign、keep 停、异常/未结清停且不自动复放；本轮只 mock 验证。

### D 的公开审阅交付

只新增一个轻量 `HISTORY_PUBLICATION.json` 和一篇 `PUBLIC_HISTORY.md`，不建新归档框架。695 原件全部原位保留，metadata 使用既有 `research_git_manifest.json.discussion_file_metadata`；不得为公开任务展开 raw 内容。

D 逐个全文审阅以下最小旧文件候选：

- `extract_twelve_20261119/DELIVERY_INDEX.json`、`FINAL_REPORT.md`；
- `extract_schema_review_20261121/FINAL_PLAN.md`、`COVERAGE.json`、`FINAL_REVIEW.md`。

以上路径均相对 `schema_discussion/`；候选并非自动可 stage。不安全或含大段 captured/raw 的候选维持 local-only，由 PUBLIC_HISTORY 用结论/路径/SHA 写短摘要，原稿不删改。原 workflow 的 37 个自然提交及已带入的产品 docs/examples 本身保留；旧 workflow discussion 默认 local-only，由公开摘要交代其历史位置，无须再挑整套资格捕获。

必要新公开集合为本次仲裁/执行映射、W1 与两轮完成证据、W2 subject/review/轻量 checker 日志、最终覆盖/交付摘要、CURRENT_SCHEMA_INDEX 和九 current；只发布实际需要的日志与合成 fixture，不发布模型原始输出。D 同样逐项审阅新文件全文及其中元信息；根 README 和当前目录 README 最终只把可公开入口作为主导航。研究报告可按必要性纳入，但两个大 research JSON 不因后缀自动公开。

HISTORY_PUBLICATION 至少给：每个旧文件原 path/SHA/bytes、`public` 或 `local_only`、原因；每个计划 stage 的历史/新证据文件 path/SHA、全文审阅结论；private/captured 未读文件明确 `withheld_without_content_review`，不能假称全审。全 695 元信息也需检查路径字段是否含敏感字面量；若有，公开索引用不泄露该字面量的稳定标识和内容 SHA，本地精确索引保留。该索引不包含自己的 SHA，不与 FINAL_REPORT 形成循环；自身由 Git 提交绑定。

PUBLIC_HISTORY 说明 fresh clone 能验证原基线、当前正文/清单、公开有限复核及测试记录；本地运行捕获、账户、未公开历史和外部临时环境只保留 hash/定位，clone 不含其字节，也不能仅凭 SHA 证明其内容或复现真实模型/GPU效果。历史文档中指向 local-only/外部材料的链接列为证明边界，不以追链接为由访问 archive。

D 交付已审 allowlist 和本次产品精确路径列表，主代理核 SHA 后 stage。无需主代理再读 695 份原文；未列路径一律不提交。D 还从旧 BASELINE 的元信息先排除 42 个 archive 路径，再按精确允许路径核其余 277 个保护文件：只有本次明确分配的产品差异可变化；原 state/supervisor 用 research_revision_coverage 的精确 run ID/摘要核对，不遍历运行树。695 全量原文公开、旧账目改签、旧规则迁移均明确排除。

## 7. 精确 CPU 验证方式

所有命令 `cwd=/home/shaohanyong.shy/KernelSchema`；`PYTHONDONTWRITEBYTECODE=1`、`-B`、`PYTHONPATH=/home/shaohanyong.shy/KernelSchema`、`PATH=<所选解释器bin>:/usr/bin:/bin`，确认该 PATH 下无真实 claude/codex。只用 fake backend、临时 Store/Publication、假时钟、受控 fixture 子进程；日志留本次目录，运行态落临时目录。Template、Extract、workflow **分开解释器进程**。

unittest/checker 使用 `/opt/venv/python3.13.12/bin/python`。必要模块（按修改跑，不为文案扩大全套）：

| 阶段 | 精确测试位置 |
| --- | --- |
| A/W1 | `schema_template/tools/tests/test_normalization.py` 单独进程；Extract `test_artifact_transactions.py`、`test_stage_flow.py`、`test_maintenance_receipts.py` 另进程；吸收第二轮真实 S-06/accept_result 反例到合成测试，不改旧证据脚本。 |
| B/W1 | Extract `test_monitor_evidence.py`、`test_cli_recovery.py`、`test_wakeup_retry.py`、`test_maintenance_polling.py`；三故障时点与全部多步入口必须能对应断言，不能只报 suite exit0。 |
| A/W3 | Extract `test_proposals.py`、`test_receipt_repair_policy.py`、`test_call_limits.py`、`test_backends_runtime.py`、`test_prompt_materials.py`、`test_review_protocol.py`、`test_issue_evidence.py`、`test_source_gaps.py`，涉及恢复/可信保稿时运行 `test_current_returns.py`、`test_current_recovery.py`；最终再包含 A/W1 模块。 |
| B/W4 | Extract `test_monitor_evidence.py`、`test_wakeup_retry.py` 及 B 修改的控制模块测试；fallback 零写入、旧样本及代际反例直接放现有模块；最后包含 B/W1 回归。 |
| C/W2 | Template `test_yaml.py`、`test_schema.py`（预算与 Output 绑定）、`test_dispatch.py`、`test_design_resolution.py`（root/from）；完整记录接受、300 字符/2 行边界及超限拒绝，其他预算不变。九目录 checker 逐一 `python -B schema_template/check_yaml.py <dir>/root.yaml --read-only --skip-hardware`，不加 `--all`。 |
| D/集成 | `tests/workflows/test_template_extract.py`、`test_write_only.py`、`test_writing.py`、`test_iterations.py`、`test_standalone.py`、`test_api.py`；上述共享接口、规则漂移、write_only、标准委托、keep/异常停止、旧 receipt 拒新字节与 curated 无 receipt 均覆盖。 |

unittest 用完整模块名，例如 `-m unittest schema_template.tools.tests.test_normalization`，Extract 另起 `-m unittest schema_extract.tools.tests.test_stage_flow ...`。不要依赖根 pytest 默认发现，因为 workflow 的 pyproject 只默认收集 `tests/workflows`。

workflow 使用 `/tmp/schema-b18-model-budget-usage-20261121/venv/bin/python`，并设 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，命令 `-B -m pytest -q -p no:cacheprovider <上表六个显式路径>`。此环境的 `kernelschema` console script 可复用，但进程及子进程必须继承指向 main 的 PYTHONPATH。运行前断言 `kernelschema.__file__`、`resources.checkout_root()`、`template_api.tools().__file__` 均在当前 main（Template 的私有 schema 路径应为 `schema_template/tools/schema.py`），记录实际路径。不可直接用旧 console script 的默认导入证明新 main。

D 可在 C 写稿时完成公开候选审阅、整理源对照和测试准备；最终共享规则测试绑定 C 冻结后的字节。若提前运行某项测试，只有其所有输入摘要在最终版本相同才可复用结果；无需无理由重复全套。所有新证据区分“历史已运行”“本轮真实运行”“未运行”。

## 8. 四份可直接转发的任务

每份任务附本文件和 execution_map 的路径即可执行；共同要求：读适用 AGENTS、唯一批准 FINAL_PLAN/COVERAGE/FINAL_REVIEW 及本计划；不 spawn、不做 Git 写动作、不碰旧运行态/receipt/Source、不访问 AKA_experiment/archive/AKA、不 home 递归、不跑模型/GPU/真实 Extract/monitor。只写分配文件，精确记录命令、退出码和摘要；遇到归属交叉向主代理交接，不双写。

### 发 A：第一轮采用对象、P2 与 W3

你是第一轮新的 xhigh 执行者 A。先完整实现本计划第 4 节 A/W1（S-06、X-15），文件/测试唯一归属见第 3 节。使用真实临时 normalize_candidate→accept_result fixture 证明数学身份，不只测试字符串；覆盖 P2 普通/维护 revise=[]、approve 相等和只读门禁。完成 W1 后发 W1_CODE_READY，暂停产品写入，待双方冻结命令后跑 A/W1，写 w1_a.json，报 W1_VERIFIED；收到主代理 W1_GATE_RELEASED 前不做 W3。之后在同一任务完成 X-01～05 的全部 W3：义务同源→安全完整诊断→native 原因分类→普通 repair 只沿 scope 总预算，保稿/批准/重交分别断言；新 nonce、真实来源、fence、独立 reviewer/裁决/签收预留和用户控制不可退化。最后跑受影响既有模块及 W1 回归，写 ROUND1_A.md/轻量证据，交精确路径、行为、测试/限制、确认无需真实模型。不能留下 W3 cap 或诊断“后续再做”。

### 发 B：第一轮提交事实与 W4

你是第一轮新的 xhigh 执行者 B。先完整实现第 4 节 B/W1（X-08），唯一写 monitor/cli/store/wakeup 及分配测试。核全部多步入口；在登记前、提交后/Popen 前、Popen 后/写回前三点注入 busy，保留本操作持久 ID 与 true/false/null、投递状态、inspect 对账、幂等和最新控制；不能重放整个命令或以 ledger revision 猜提交。完成后 W1_CODE_READY 并暂停写入；双方冻结后跑 B/W1 写 w1_b.json，W1_VERIFIED；主代理 W1_GATE_RELEASED 后才继续 W4。用同一 owner 完成只读 fallback/缓存区别、record_scan 当前 anomalies/历史、显式新 clock 代际/同代际恢复，并补 README 叶目录和字段解释。用临时账本/假时钟/假 Popen/signal；不真实启动 monitor。最终包含 W1 回归，写 ROUND1_B.md，交三时点/多入口测试覆盖和旧运行态零写入结论。

### 发 C：第二轮全部 W2、正文/规范/current

你是第二轮新的 xhigh 执行者 C。第一轮收口后，完整实现第 5 节全部 W2：九 YAML/两 dispatch、Tolerance 300/2、DOC1/4/5、P2/P4 提示。按本计划精确数值/Version/来源修正；另 24 YAML 不变，保留固定 Source 和旧 receipt/DELIVERY_INDEX。先候选副本审阅 normalization，再落最终 main 字节；九完整目录19 YAML/9 dispatch做 checker/dispatch/raw==preview/真实隐式root与显式from闭包检查。冻结 W2_SUBJECT.json，报 W2_SUBJECT_FROZEN，等 D 对同一实际字节独立有限源复核；D 指出问题由你更正并重冻/复验。D 的 W2_REVIEW_PASS 与固定报告 SHA 到达后才生成九 current 和 CURRENT_SCHEMA_INDEX，明示 runner_p6_receipt:false、有限审阅范围与原receipt。不得生成“待补审阅但已完成”清单，不改 D 报告。写 ROUND2_C.md，给 D 最后核验；所有必需项本轮完成。

### 发 D：第二轮独立有限审阅、workflow 与公开收口

你是第二轮新的 xhigh 执行者 D。负责第 6/7 节共享接口 CPU 回归及最小兼容修补、C 新字节的独立有限源复核、最终入口/覆盖/历史 allowlist。先在 C 写稿期间审阅第 6 节最小历史候选和新证据、整理695原件保全元信息/clone边界与源对照。C 冻结后独立核十一处更正和真实继承消费者，五份 ACCEPTANCE 的九输出完整值/标签/null、masked/FMHA/KDA及两dispatch风险；问题交 C 改，不改其文件。冻结 W2_REVIEW.json并发 W2_REVIEW_PASS，再等 C current/index后只读终检，报告实际subject/check/review/current一致。最终六个 workflow 测试必须导入main，保留write_only、strict真实冻结规则、标准AKA mock委托/iterate边界；旧receipt配新字节拒绝、curated无receipt可用。更新根README、Schema Wiki（18项48YAML、保留旧六项）、本目录入口。写全26发现/24候选104出现/20 recap/11已有修正/16删减及FC01/02的 IMPLEMENTATION_COVERAGE，区分已实现/保留/明确排除，不改旧COVERAGE。完成 HISTORY_PUBLICATION/公开摘要与已审精确stage allowlist、FINAL_REPORT；核695原件、277允许保护文件及精确state/supervisor，42个archive元信息只过滤不访问。公开审阅不能转交主代理或只标候选。你不提交Git、不改AKA、不发布raw/账户/原始模型思维。最后向主代理交所有必需项完成证据及准确限制。

## 9. 完成判据与 main 交付

- B 可还原全部原 57 文件；M 保留 workflow 37 个自然提交，最终 main 的祖先包含两个原 HEAD、B、M 和完成的四包。
- W1 有共同冻结树测试及独立提交；W3/W4 完整实现并保留 W1 回归；没有将已删 T01～T16、新恢复/诊断/预算平台、可选 Execution/KDA slot_start 文案恢复为隐藏待办。
- W2 恰九 YAML 增版/两 dispatch 修订；九 current 覆盖完整19 YAML/9 dispatch，有限源复核/实际规则/检查对象与字节一致；其余24 YAML、12原receipt、旧DELIVERY_INDEX/历史、原run/rules/account与AGENTS均保全。不是新 P6，不证明全33稿数值正确或 GPU 性能/真实模型返工率。
- workflow 共享接口定向 CPU 回归通过，默认write_only、标准AKA接口未追加门禁；最终入口无“审阅已完成所以产品已实施”的混称。
- D 给完整实施覆盖、已审公共allowlist和695历史元信息，不留 mandatory pending；本地未公开原字节完整保留。
- 主代理按产品路径及 D 审定 allowlist 精确 stage，核 staged bytes 与证据相同；不得 stage AGENTS、运行态或未审记录。正常 fetch；远端有新增则正常 merge 并对受影响内容复核，不能 force。正常 `git push origin main` 后核远端 main 等于交付 HEAD，并报告提交与验证限制。允许工作树保留明确的 AGENTS 本地修改及 local-only 未跟踪历史，不能为“干净状态”删除它们。

只有这些完成条件全满足，当前 discussion 的实施状态才由 plan_ready 改为 complete。旧审计 FINAL_PLAN 的“待实施”仍是历史事实，不能回写为当时已经修好。
