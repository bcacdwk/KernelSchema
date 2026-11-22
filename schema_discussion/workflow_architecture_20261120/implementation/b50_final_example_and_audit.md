# B50：公开例子、使用说明与最终审计

2026-11-21 UTC。**公开例子与本地审计已完成：B49 独立 native host 验证 accepted=true、PASS、reused=false，五个实际 GPU job 全终态；原字节 kernel 和精简 results 已加入示例。** 主结果为该次同 allocation 三 shape 的原 AKA ABBA 投影 **1.503522×**。B45 旧 host 额度失败与 optimize keep 原样保留。本 agent 模型/GPU/安装/服务启动均为0，不派生子代理、不操作运行账户/进程/RPC、不stage/commit/push、不改AKA gitlink/release。

实测来源为 KS `13efd1554d112d56e4d767ba43b66257b2d75669` / AKA `55bc39f3239e2d9fa65d2a799182a26df413368c`，均已由根确认推送，同分支 `feat/schema-workflows-v1`，AKA当时配KS13efd。B50五个公开文件待根固定，再单独完成AKA配对；文档提交不能替代实测源码身份。

## 公开材料与精确产品manifest

[product-manifest.json](/tmp/schema-b50-final-example-audit-20261121/product-manifest.json) 的 SHA256 为 `57a88967450e7b8004e406b0bb1f582368e792f273a28031d7a4a6810677b310`，仅含KS独立工作树这五个待提交文件：

| 文件 | 用途 |
| --- | --- |
| `README.md` | 极短导航到任意task有限入口和可用vector-add示例 |
| `docs/workflows/standalone.md` | 版本间kernel继承边界与真实示例导航 |
| `examples/workflows/vector_add/README.md` | 安装、静态implement、KS/AKA有限iterate、直接使用kernel、结果读取、B45/B49真实范围 |
| `examples/workflows/vector_add/kernel.py` | B49实际candidate逐字复制，2089B、SHA105d89d7432e0662625b9388df68ac1f36ecfcecb1bd4494074b3adc7a5f7e5f |
| `examples/workflows/vector_add/results.json` | 5275B精简公开摘要：source/candidate/promotion身份、UTC结束时刻、三shape、5job用途/receipt hash、host成对时延与未知环境 |

`root.yaml`和`task.md`均与原KS13efd Git对象逐字相同。root仍9106B/SHA `58a83e9be44e4d1d5f656454fbec1de65ec948f49c79aad06f9daeb6b6d30e1c`；生成来源KS907c99、B45运行AKA88d773、本次B49运行AKA55bc分开保留。没有workflow/AKA运行实现变更；新增Python只为原字节公开kernel示例。没有新运行壳/调度程序、巨大receipt、模型thinking/base64、私有账户/CLI/profile/认证进入产品。

用法准确区分：静态`--schema --schema-path`在原coder episodes提供固定Schema；`api.iterate`与AKA薄转发`--schema --schema-iterate`共用唯一有限循环。`rounds`计Schema版本、`max_iters`计版内原coder轮数，planner只在版本间工作；keep、blocked、unsettled、缺产物/反馈按既有规则停止，已有out拒绝重放。每版新workspace不会自动把前版kernel设为incumbent；原docs/schema-doc可明确给代码参考，版本修订不保证性能单调改善。

## 最终真实证据与来源

已读[B49原报告](b49_final_native_promotion.md)、[root-handoff](/tmp/schema-b49-final-native-20261121/root-handoff.json)、[stage-summary](/tmp/schema-b49-final-native-20261121/stage-summary.json)和[native-jobs](/tmp/schema-b49-final-native-20261121/native-jobs.json)。对原5job receipt只做原字节SHA核对，没有输出原CLI流或思维块，也没有重跑测量。

B49通过现有docs显式提供B45真实candidate参考，模型在新workspace逐字materialize；**不是第二次仅从Schema生成**。候选2089B/SHA105d89…7e5f与B45/B49原artifact相同。此次candidate_commit为 `21563597f24e9ab4aca4ffcc2a812e9da8d18363`，promotion_commit为 `4b41a98516f7ba3681e75d18410c47e09bb385ed`，不混用两者。

| 本轮job | 结果 |
| --- | --- |
| V0 `ev_4c605098b3bc` | 参考PyTorch，三shape各6case通过 |
| candidate full `ev_7102dba05dfd` | 三shape各6case/max_abs_err0 |
| Agent ABBA `ev_7840da47d4aa` | 完整有效，原AKA投影1.556388× |
| CandidateGate `ev_9e107207e34c` | correctness_only，三shape各6case/max_abs_err0 |
| host独立ABBA `ev_53c06a9e4f27` | 完整有效，accepted=true/PASS/reused=false，baseline11.226587979µs、candidate7.466858430µs，原AKA投影1.503522276× |

host record `3ea86742e7fbd66078184d11901469be19ed5932dffd0ae970e6aa9817b8d1a6`来自独立trusted request，原receipt SHA `1d382376bda582586a0740bae8d3350d506fb4ebe0b7bdfb45dc36c97d606a58`。ABBA为1完整SDK block，三shape各2baseline/2candidate采样，每个完整run1correctness case；六case来自独立full/gate，不称2blocks或ABBA每run六case。使用原AKA投影，不混成SDK另一speedup字段。

两次ABBA、B45旧2.350190×都各属自己的allocation；不拼接成跨record性能变化。identity=null/comparability=unknown保持，跨record reuse/minima仍拒绝。原时延有离散/离群样本；单次完整block可报告实测值，不证明统计显著性、SOTA或未测任务范围。

B49实际1host coder/5jobs/657102 exact tokens/request active811673ms，无writer、无新Schema、无handoff resume；原第二coder未启动并已释放。parent增量811674ms与request的1ms投影差异保持。native receipt未报告USD，本次/全会话USD unknown。旧parent最终10models/16jobs/5773511ms/40ops，idle、无unterminated/uncertain；剩coder1/model14/jobs0/active5026489ms，旧15requests/33ops/descriptor摘要全不变。已知tokens下界3118640不是总tokens。收费slot释放，后续0调用。

## B45与历史保留

B45真正从B40原root经`api.iterate`产出新candidate；full、六case gate和Agent ABBA通过，host独立ABBA因原4-job group用尽而未提交第5job。accepted=false、verification ERROR的旧事实不倒改。真实失败与五份feedback逐字交write-only optimize后completed/keep、无第二版；证明反馈与自动停止，而非强制revise或性能单调提升。

B45实际native1coder/4jobs/965010tokens/984471ms，独立writer1planner/0GPU/34155tokens/188933ms/USD0.364455；合计2host invocations/4jobs/999165tokens/1173404ms，native/总USD unknown。只核获准的[B45 summary](/tmp/schema-b45-finite-native-20261121/stage-summary.json)、[handoff](/tmp/schema-b45-finite-native-20261121/root-handoff.json)、[integrity](/tmp/schema-b45-finite-native-20261121/final-integrity.json)及原报告；未读取账户。B33旧API失败与B35/B36/B41无候选失败保持。

B48的45项CPU/10次自然调度证据按[原报告](b48_native_shell_wait_fix.md)复用，不重跑。根后来应用并固定55bc；只原Python wrapper临时monitor-off、等待真实0/7/147并恢复状态，原暂停/保护保留。B49本次公开工具未见147/Stopped是该实际路径证据，不扩成所有Bash/绕过wrapper/外层pipeline保证。

## 本地审计与有限检查

[audit-manifest.json](/tmp/schema-b50-final-example-audit-20261121/audit-manifest.json)单列原main指定discussion的六文件：`implementation/README.md`、`progress.json`、`next_steps.md`、`delivery_audit.md`、本报告与`01_history.md`。不混入KS产品提交。H097–H101顺序追加B45、B48、B49启动/终态与B50交接；H001–H096原前缀、B01–B13原正文、progress units/gates/旧rollup与旧计划保留。最终source pair/push只待根回执，未提前写成已提交。

- 公开最终51个本地链接、10个shell/Python片段语法、kernel AST和精确diff空白通过；最初2条未变AKA外链HTTP200证据复用，未重复联网。见[final-public-verification](/tmp/schema-b50-final-example-audit-20261121/final-public-verification.json)。片段仅解析，不执行GPU代码。
- 公开kernel与B45/B49原字节相同，root/task与Git固定对象相同；5job原receipt SHA、所有job终态/gate、三shape六case、host block完整性及host投影比值一致。见[B49 evidence check](/tmp/schema-b50-final-example-audit-20261121/b49-evidence-check.json)。
- 原历史前缀/正文/旧JSON字段与[修改前精确副本](/tmp/schema-b50-final-example-audit-20261121/before)比对保持；最终审计链接及JSON解析见[final-audit-verification](/tmp/schema-b50-final-example-audit-20261121/final-audit-verification.json)。

未重跑旧大套、真实模型/GPU、安装或新测试平台。没有读取、枚举、执行或修改AKA_experiment及archive变体，没有home递归搜索，原main算子提取未触碰。下一步仅由根按产品manifest提交推送，并单独更新匹配AKA子模块/release及最终审计SHAs。
