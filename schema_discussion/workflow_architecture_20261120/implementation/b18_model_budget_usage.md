# B18：模型阈值与矛盾 usage 的有界修复

2026-11-21 UTC。本项接续 B17，完成本地源码修复、只读反例解析和定点合成测试。**真实模型 calls0 / GPU jobs0 / 外部网络调用0**；没有重跑原失败 scope，没有改真实账户、原 Q/profile、原日志、收据或失败结果。KS 与 AKA 的提交由主代理完成，本代理不提交或推送。

## 原问题与保留边界

B17 唯一新增 coder 在原600秒 scope 内运行57676ms后以 CLI exit1、timed_out=false、`error_max_budget_usd` / `budget_exhausted` 失败。原 profile 阈值0.50，原 terminal 的 `total_cost_usd` 与 `modelUsage.costUSD` 都为0.758005，说明这是 CLI 停机阈值，可在超过后才停止。原 assistant text 虽可解析且逐字等于 current_source，也不能改失败终态或追认 transport qualification；原 scope 已 done，reviewer0，qualification=null。

同一 terminal 的 aggregate 四项 tokens 全0，modelUsage 却报 input6/output2219/cacheCreation112400/cacheRead0。组件合计114625不是已证完整真实总量；assistant事件也不能重复相加。旧解析器据aggregate写入 `tokens=0, measurement=exact`，该值不能作为真实零用量或免费调用的证明。真实总tokens仍unknown，原CLI报告成本0.758005保留。

开工时，唯一原 parent `/tmp/schema-b14-agate-20261121-twmkup4i/.runtime/owner.json` 仅通过文件读取核验，没有实例化 OwnerAccount：三个operation都已终，model_calls1/jobs1/active122688ms，active_run/active_since=null。冻结限额仍24models/16jobs/10800000ms，roles planner6/coder8/reviewer16/extract0，tokens=null。原descriptor与余额没有修改；剩余23models/15jobs/10677312ms不能因错误tokens0被夸大。该快照的全会话实际模型累计8，GPU job1；历史unknown使全会话tokens/费用仍unknown。

## 修改前的原实现保全

只读快照 `/tmp/schema-b18-model-budget-usage-20261121/original-b17` 保存原 KS `7d5fb60cf1d6f9ef7cf63dff757250bb4f614ab1`、AKA `89fb61b696a93a255f44668658f79bd8bd0e6ce3` 的必要公开源码、KS选定release资源、原600秒/0.50profile与安装身份。目录0555，文件0444。未复制整套private captures或凭据，也未覆盖先前e867保全件。

| 文件 | SHA256 |
| --- | --- |
| ks-source.tar | `1ba0b2c70b3568543633e7f6594f70090062b2f7e6f6828e8caa8410057811ad` |
| aka-source.tar | `1f5fe4b31b81fd157534a45463ff53f90d3c98b3ab81b9f0bb14e15999207642` |
| installed-before.json | `113134a3621f085e44208c07ca03d61884f1e88e60cd457958943f5cae91b8cb` |
| b17-original-profile.json | `f86190b7c579afe95fede90ecc762670297293b283249db6d3d181f5533cb0f0` |

原安装 `/tmp/schema-b05-a3-venv`：schema为AKA editable，framework/legacy/light为原B10 wheel，KS为AKA固定子模块。原framework wheel SHA `62eb0a630eaa3840561d20f2f828166ac44bf90d074d9ba4ec7d0d94109eb645`。因此修改 framework-owned parser源码不会自动更新原wheel；跨仓测试必须明确选择新源码，后续实际新scope还须核对安装来源。

## 最小修复

- KS `ClaudeProfile.max_budget_usd` 默认保持字符串 `"0.50"`，允许显式正且有限的十进制字符串，例如 `"2.00"`；拒绝零、负数、NaN/Infinity及非字符串。仍传实际 `--max-budget-usd`，没有添加货币账本、硬美元保证或重试。
- AKA probe移除另设的0.50固定拒绝。exact profile/Claude hash、scope ceiling、原Q和owner限制仍逐项冻结。usage parser的base源码加入资格implementation摘要，防止遗漏这条解析依赖。
- AKA 的 ClaudeLikeAdapter 与通用stream fallback共用terminal解析函数。当aggregate解析为0却出现正modelUsage组件/成本或正terminal成本时，返回unavailable，禁用modelUsage或assistant delta回填；NativeInvocation保存原stdout，两组原始字段仍在。正常零用量、正常正用量、原缺aggregate兼容路径不改。
- KS reference Claude runtime同样将这种矛盾标为 `tokens=null`、`accounting_complete=false`、`conflicting_native_usage`，原始事件保存在receipt。CLI预算失败/exit1始终失败，不接受assistant文本替代terminal。
- 文档明确阈值可被越过后才停，新阈值只属于新profile/Q/scope，原已准入scope必须使用原实现，不迁移、不重跑旧失败。

## 定点验证与来源冻结

所有数据和进程均为本地合成，未执行模型、GPU、业务Task或真实native probe。

KS首组：43 passed / 61 deselected / 39.93s；覆盖profile校验、矛盾usage、正常零、原正常用量、失败和replay兼容。Owner审计组3 passed / 0.59s。最后代码整理后，对新阈值、两种矛盾终态和正常零作4项复核，4 passed / 7.15s。日志位于工作目录 `ks-targeted.log`、`ks-budget.log`、`ks-final.log`，`git diff --check`通过。KS五文件已冻结为 `ks-frozen-files.json` 与 `ks-owned.patch`，交主代理提交；后续固定SHA及AKA跨仓结果在末节补充。

对B17原NativeInvocation Ref `artifacts/57fc89a2d039483cb080ce30dcc2f01f/native-invocation.json`（SHA256 `1ffd2052a3558613e16f16f5fa836bea8c460be20651e72ddd8ed0644b1081e5`）作只读parser replay：旧安装parser仍0，新KS parser为null/accounting_complete=false，cost仍0.758005、output未接受，保留native_terminal_failure。该诊断使用合成parse intent，仅核验原stream用量，不打开原run。证据 `ks-original-b17-replay.json`。最初诊断脚本选到非init system事件导致KeyError，修正为显式subtype=init后成功，没有副作用。

AKA独立parser组9 passed / 0.71s，覆盖两组同时矛盾、仅model tokens、仅model cost、仅terminal cost、重复assistant delta、正常零/正、缺aggregate及缺terminal的原兼容语义。后续新scope/receipt/failure组合测试等待主代理固定KS后运行。

## 原错误零值的最小审计修正方案（本阶段未执行）

现有 OwnerAccount 已足够，不需要新账本或迁移平台。`absorb`接收同producer/run/operation/role的更高sequence累计row，保留原history，latest可由已知0改为null；lower仍0只是数学下界，不是总量。累积calls仍1、evaluator_jobs仍0、gpu_ms仍0、terminated仍true，不增加operation或active time。

可供主代理审阅的追加方式：

1. 只读核验原invocation Ref/hash、session/op、原sequence1 receipt及原failed终态，使用新parser产生独立审计勘误文档，明确 `old tokens=0`、`corrected tokens=null`、理由和两组原usage字段。文档引用同一原NativeInvocation，不能把新观测冒称新调用。
2. 在现有account针对原operation追加sequence2 row：同producer `aka.native:90bc0a6376cc4488ae09e1d4ac5cad0f`、同role/purpose，新的勘误receipt_id/Ref；state=settled、calls1/jobs0/tokens=null/gpu_ms0、terminated=true。不新建op，不begin、不reserve、不注册新run、不更新descriptor，不重复calls/time，不续额。
3. 保留旧 NativeInvocation、sequence1收据、原result与原failed资格结论；如需展示新owner投影，另发布带勘误说明的usage index。旧不可变result仍显示历史零投影，不能被静默改写。

合成账户测试已验证：sequence2追加后history恰[原row,勘误row]，重放任一row均幂等；operation仍1，calls1/jobs0、active57ms、原reservation/lower/descriptor均不变；tokens变null，已终且active_run/active_since=null。因tokens维度unlimited，unknown统计本身不会新增fence，也不改变有限calls/jobs/time余量。

现有 `Transactions.absorb_invocation` 固定sequence1并绑定原invocation publication；不能调用它来覆写或重新解释已存原件。原scope也不能在变更实现下continue来做勘误。本阶段只交可审阅方案和证据，真实账仍保持原样；明确审计勘误是当前事实解释，不能据原0宣称完整真实总用量或更大余额。

## 下一明确scope提议（未执行）

新profile显式 `max_budget_usd="2.00"`、timeout600000ms，新scope ceiling600000ms；仍coder/reviewer各最多1，首失败即停，保留完整reviewer最小收尾容量，沿同原parent剩余额度创建新明确Q。需先固定并核验新KS/AKA源码与安装identity，再准备新profile/Q；原B17失败不重跑，旧Q/profile和收据继续用原实现追溯。该提议不构成执行记录或运输/GPU资格。L20D sandbox/GPU资格由另项研究，本项未改变K1/Task/GPU通道或实验内容。

## AKA固定依赖与勘误预览补充

主代理已核对KS五文件并提交 `77ad741ca5b088d24f4cd23018eb0efcd491f15f`。AKA从本地Git对象库fetch该精确SHA并detached checkout，未访问网络，也未安装KS活跃tree。299文件release manifest SHA256为 `10bf642db61b25797ba0cc8485ce646f5374bfa98faea01cdcb01610b7bc781b`。

AKA对同一B17原stdout的两条新解析路径均为total_tokens=null、measurement=unavailable、normalized_terminal_events=0，原NativeInvocation文件SHA保持 `1ffd2052a3558613e16f16f5fa836bea8c460be20651e72ddd8ed0644b1081e5`。来源与结果保存在 `aka-original-b17-replay.json`；该进程显式选AKA新src和固定KS子模块，不把旧installed framework当新解析器。

具体审计预览位于工作目录：`b17-usage-erratum-proposal.json` 与 `b17-usage-row-proposal.json`，明确apply=false、status为proposal，receipt path用 `REVIEWED_PUBLICATION_ID` 占位，不能直接feed到absorb。原row的JCS SHA256是 `3b5c8cbf0dd0726df26ab8931437f36986a59625a6eb6532ecaee849b9116248`，原terminal行SHA256是 `68475f84660ce38a7bedd9aa74e17bca1ca52592950034dc4e5ccd1507ae7b76`，原session是 `bb595a61-adb9-4d0c-8af6-c5c480daf945`。原row producer/run/operation/role/purpose及所有非token计数逐字沿用，仅新sequence2、独立勘误Ref/receipt_id、tokens=null。预览勘误SHA256 `3bfbe5a5fe6ae7f77362725b761f8388bff71e65eb0b5a41cb84249e51822059`；实际发布前由主代理审阅并冻结最终文档摘要。

主代理同期明确通知B19已启动授权L20D诊断，因此共享parent整体hash和活动值会合法变化，本项不能再把开工时model1/jobs1/active122688当当前快照。只读复核确认B17原operation的latest/lower/终止事实与descriptor未变，见 `b17-owner-operation-unchanged.json`。本项没有调用真实OwnerAccount。

AKA固定KS跨仓定点组最终 **18 passed / 66.81s**，日志 `aka-cross.log`。覆盖profile2.00/600s确实进入CLI参数、exact profile hash与implementation依赖、同scope两角色最多各1、changed-profile继续拒绝、两种矛盾usage均无资格、原失败不重试，以及旧scope/RoleSession兼容。第一次命令拼错一个已有测试selector，收集阶段no tests ran，日志保留为 `aka-cross-selector-error.log`；纠正selector后执行本组。没有全套重跑。AKA8文件+gitlink逐SHA冻结为 `aka-frozen-files.json` 和 `aka-owned.patch`，交主代理提交；`git diff --check`通过。

## 最终固定安装与交付

主代理逐SHA提交AKA **`b89d061e606f8eae5415e0bdfcbd88261a5d58dd`**；本代理没有提交或推送。独立环境为 `/tmp/schema-b18-model-budget-usage-20261121/venv/bin/python`，KS源码/资源是同目录 `fixed-ks` 的 **77ad741固定Git快照**；不是活跃KS工作树。AKA新framework与schema包均从b89d061的Git archive，用原public builder的文件ownership选集在本地stage，`python -m build --no-isolation`先sdist后wheel。新环境以 `--no-index --no-deps` 安装；legacy/light复用已核对的原B10 wheel，不重复构建。15个明确公开依赖按已安装dist RECORD逐文件复制到独立环境，来源/版本/文件SHA保存在 `offline-dependency-provenance.json`，没有复用旧环境sys.path，没有复制凭据。全程外部网络0。

| 新构建产物 | SHA256 |
| --- | --- |
| atrex_aka_framework-0.2.0-py3-none-any.whl | `ceba7941990d48f747ca01deef63d7fc8e317ff31625c4de607086069fb5d39e` |
| atrex_aka_framework-0.2.0.tar.gz | `8ca1ea66a7ec6af04dd9dd8e40c38aedb70c107709eb2fa5326b68e9dc25bb04` |
| atrex_aka_schema-0.2.0-py3-none-any.whl | `41e6a089842b7dcbed3c348f65dee6ca2fee0e7cc6356e9854b74fe80d734692` |
| atrex_aka_schema-0.2.0.tar.gz | `1c82bef9438fd31dc2bce9633942b93d7d84e35e4dc271207423f9b0fd5b7746` |

`verify-installed.py`在 `-I -B` fresh process下核验十个关键模块字节：AKA全部来自新venv、KS全部来自固定快照，逐项与固定source一致，release manifest核验通过。实际安装的两个parser再次只读处理B17同一原NativeInvocation，均返回unknown；原文件仍保留tokens0、exit1、timed_out=false。记录 `installed-final.json`。候选2.00/600s profile仅本地构造与argv核验，没有inspect CLI或发起probe。`pip check`结果No broken requirements found。

安装环境下的定点组 **14 passed / 12.74s**，日志 `installed-targeted.log`，包括新阈值、原B17形状的矛盾/失败、正常零/正及fallback兼容、fresh-child installed source/resource与实际stdin child传递。未重跑全套。原framework wheel字节再次核验未变，证据 `original-installed-framework-preserved.json`。须精确区分：原venv未被本项重装，但其editable schema/KS路径随reviewed生产源码/gitlink前移；它既不是B17完整冻结环境，也没有B18新framework parser。追溯原B17应结合只读原源码快照与原wheel/identity恢复；新的scope应使用这里已核验的完整新安装，不能把原wheel冒充已修复。

B19结算后的只读共享parent快照 `owner-readonly-final.json`：四op全终，model_calls下界1/jobs2/active207911ms，active_run/active_since=null，descriptor与B17原operation的latest/lower/终止事实均保持原样。本项真实新增仍models0/jobs0。该快照当时剩余23models/14jobs/10592089ms，之后如有其他授权活动须再核验；tokens维度仍unlimited，总tokens仍unknown。根安排的后续勘误尚未由本项执行。

### 源码逐文件摘要

| 仓 / 文件 | SHA256 |
| --- | --- |
| KS `docs/workflows/standalone.md` | `379df28d04bb6c9ceafa123e60261b6c929c54853353b13d426b5c98c81fbfd1` |
| KS `kernelschema/runtime/claude.py` | `b2f39adf72ee4a8c2c5189001fcb3651f3aabb53503d090ce039a1394c6f531b` |
| KS `tests/workflows/fixtures/runtime/fake_claude.py` | `cae50a3fc6facbb32a4eb348609d3926d9363f061b4eab756ba19f46a0ee5440` |
| KS `tests/workflows/test_budget.py` | `f62723afe1bd67a4bd4d3e1af23c64e37af5a1ff361c65efe0e56eaaa278cd6e` |
| KS `tests/workflows/test_claude_runtime.py` | `77cc9c52b053e2bb7dee8b47bf78982921b6c94546004f59dc6e6b6303aade03` |
| AKA `docs/schema-source-release.md` | `01f333d7c54bc79bbe497745a4b4971a671351b383cf01680876eafdbc999eff` |
| AKA `src/aka/agent/backends/__init__.py` | `171fdc92bce39ce912990795350ece1682eb4dd7bc0af6878d96e987c5b552e6` |
| AKA `src/aka/agent/backends/base.py` | `4bb5b9059b1804278862cb2b197369ab84d64cb920590391e0632271ef35c1b5` |
| AKA `src/aka/agent/runtime.py` | `de2f1f34dd8e333225897bbc166d30da24823c225c78381a5f2200e408f0eb29` |
| AKA `src/aka/workflow/schema/native_probe.py` | `4f0dfa9a87709687eed499929e3386f1089e63a3109a010f79dd27a5ba8c9732` |
| AKA `src/aka/workflow/schema/release.json` | `10bf642db61b25797ba0cc8485ce646f5374bfa98faea01cdcb01610b7bc781b` |
| AKA `tests/test_schema_native_limits.py` | `93ff59df36901ec105b4dd384536c2ee06cf10d138c7ede858f69c231786ce9d` |
| AKA `tests/test_schema_native_probe.py` | `3e5bd1ab8812a5a490d23e6b7424b321d18fa4f1122828f7df26a052561c5808` |

AKA gitlink为 `77ad741ca5b088d24f4cd23018eb0efcd491f15f`。源码冻结清单、补丁、测试日志、构建产物与安装身份均在工作目录；最终机器可读交接为 `b18-final-handoff.json`。
