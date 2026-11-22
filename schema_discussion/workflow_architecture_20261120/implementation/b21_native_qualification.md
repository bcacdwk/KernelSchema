# B21：B18 修复后的有界真实 native 资格

2026-11-21 UTC。本项沿原 B14 parent 完成 B17 append-only usage 勘误，并成功完成唯一一次新的 2.00 USD／600 秒 native 运输资格。**新增真实模型calls2（coder1/reviewer1）、tokens227586、CLI报告USD1.46283525、GPU jobs0**；原scope已完成、两原进程均终止。没有业务 design/coder/GPU 执行，不修改生产源码、不提交或推送。未访问、枚举、修改或运行禁止访问的实验仓，未派生代理。

## 原账勘误

开工只读核验与 B19 快照一致：4 个原 operation 全终，model1/jobs2/active207911ms，active_run/active_since=null、无 fence。原 descriptor 首冻结限额保持24model/16jobs/10800000ms，roles planner6/coder8/reviewer16/extract0、tokens=null。原 H20 拒绝、两次 L20D 成功诊断及 B17 失败全部保留。

已核对 B18 审阅提案、原 row JCS SHA `3b5c8cbf0dd0726df26ab8931437f36986a59625a6eb6532ecaee849b9116248`、原 Invocation SHA `1ffd2052a3558613e16f16f5fa836bea8c460be20651e72ddd8ed0644b1081e5`、原 terminal 行 SHA `68475f84660ce38a7bedd9aa74e17bca1ca52592950034dc4e5ccd1507ae7b76` 及已终止事实。新安装两条 parser 均把矛盾 aggregate0/modelUsage正数解释为 unknown；CLI 报告成本仍0.758005，原失败 `error_max_budget_usd` 不变。

根明确授权原 B17 capture 中仅新增一个 Publication 及其自身封存元数据。真实勘误 Ref 为 `artifacts/2acfb6f5f4b04a3ab34919cf4d008652/b17-usage-erratum.json`，SHA `8d1bb253dc73dfac715cf979932a7d22bad161ab031682dcbb9be80d797e3208`。其文本为已发布的独立观察勘误，未使用 proposal 占位 Ref。幂等 intent 与真实发布 Ref 均在吸收之前 fsync 保存。

既有 `OwnerAccount.absorb` 已针对原 run `90bc0a6376cc4488ae09e1d4ac5cad0f`、原 op `aka_b52d235c2d7e463a84bdcc2a6b693ddf` 追加 sequence2，保留 history `[1,2]`。producer/role/purpose不变，仅 tokens0→null、sequence、新receipt_id/Ref变化；calls1/jobs0/gpu_ms0/settled/terminated均保留。原 account active/calls/jobs/lower/limits不变，无新operation、无begin或register、未continue旧scope；原result、state、Q/profile、Invocation按SHA复核不变。

勘误前/后 owner SHA分别为 `e435a2875da0fef6a22c7b7f0d42d88194840fb420f2a300be629fc6d9f49ad0`、`57d39ca030bb4f4748715e475e4a01c478178bd28c7dcd7ba7fd5ac9a89b3692`。差异严格只有原operation的history/latest及正常事务clock基准；累计active仍207911ms，tokens unknown不在unlimited维度新增fence。

## 固定安装与新的明确输入

独立Python `/tmp/schema-b18-model-budget-usage-20261121/venv/bin/python`；framework/schema均来自 AKA `b89d061e606f8eae5415e0bdfcbd88261a5d58dd` wheel，KS module/resource来自同目录fixed-ks `77ad741ca5b088d24f4cd23018eb0efcd491f15f`。10个关键module逐字SHA、5个dist安装元数据、299文件release manifest均复核匹配；manifest SHA `10bf642db61b25797ba0cc8485ce646f5374bfa98faea01cdcb01610b7bc781b`。未使用旧framework venv、未安装KS活跃worktree。

官方 `ClaudeProfile.inspect` 的version/help与原profile匹配，`claude auth status` exit0且loggedIn=true；未读取或复制凭据。CLI2.1.316、qwen3.8-max[1m]、原allowed model列表、existing_cli_auth/no-tools配置保持；executable SHA `fd13c70000fec3f2116dfbcbb6121210b4a29328bf0e46210a59dd594a67d824`。

新exact Claude profile只把原0.50阈值改为字符串 `"2.00"`，timeout600000ms不变，SHA **`3e1766d7fad58da4e76b77c28a296b0adcb225c04e7e56704dac45a848efc530`**。这是CLI停机阈值，可越过后才停止，不是硬计费上限；原B17 profile/Q和失败scope未修改或重跑。

从 B16 明确业务 fixedQ `8ebbdebfb3606945e528ad7ae26a66b082f9c5fab4f335de8ed4ecea638c8e1f` 导入完整选定闭包。Task `4162e75b3f340e13dbe27714c17ad575f37823afc23096b473ab4e1d46bb48c2`、R0 `285960e80e5170b9ee5f3c77c93f849ffdcbe93cea42c49fedba3e60c7b62a97`、Rules/Scope/seed/materials及原parent保持。首先冻结保留完整原业务目标的新profile业务Q，再单独派生运输Q。

| 冻结输入 | 路径（相对B21目录） | SHA256 |
| --- | --- | --- |
| 新业务 Q | `business-prepared/request.json` | `c2538bb3a1d66c4ad1bda64975943952aa1e671a4bab2d508300ac2343fa37e2` |
| 新运输 Q | `prepared/request.json` | `063044c20b2495f86d866808694bfe3dce3eb1de464f331cd408193418a51cdb` |

运输目标只授权 coder 完整逐字返回 current_source、reviewer 返回运输限定 review；其余原材料正文及aliases不删减。preview coder416497bytes/reviewer417344bytes，各27正文、13/15直接alias、29个host implementation身份，每项SHA/size核验通过。preview仅是预览；实际投递另核验原capture。

## 实际执行边界与命令

最终零调用预检证明原parent idle、4op全终、剩余23model/14jobs/10592089ms；原descriptor与authority在准备前后逐字不变。新Q/native allowance为coder1/reviewer1、model_calls2/jobs0/active1250000ms/tokens=null，minimum_active_ms coder/reviewer各600000、evaluator1000。scope创建时冻结600000ms ceiling，保留完整reviewer窗口，首失败停止；不覆写旧scope、不并行模型或GPU，不换scope重试。

```sh
KS_BUDGET_AUTHORITY=/tmp/schema-b14-agate-20261121-twmkup4i/.runtime/owner.json \
 /tmp/schema-b18-model-budget-usage-20261121/venv/bin/python -I -B -m aka.workflow.schema.native_probe \
 --request /tmp/schema-b21-native-qualification-20261121/prepared/request.json \
 --out /tmp/schema-b21-native-qualification-20261121/native-capture \
 --scope-timeout-ms 600000
```

所有本项准备脚本、日志、intent和前后快照在 `/tmp/schema-b21-native-qualification-20261121`。原B17脚本只读复制并在新目录改写；新增profile日志序列化曾因tuple不符合JCS被拒，已在任何模型调用之前改为list；期间官方inspect/auth均成功且无收费调用。未新增或运行无关测试。


## 真实成功终态与用量

公开probe进程exit0；原结果 `completed`、reason=`original native transport captured`、synthetic=false、recovery=none、unsettled=[]。唯一新run为 **`c2ba514b22d84a98937e0e0f498bbf79`**。两次CLI均exit0、timed_out=false、terminal subtype=`success`；没有continue或重试。

| 真实事实 | coder | reviewer |
| --- | --- | --- |
| operation | `aka_808b02bb3fb04d75ad1321bb56a8cba1` | `aka_0aa4986512d946d5830948ec809e555e` |
| session | `70cfd85b-c16b-4b14-86e2-63ffa7f44161` | `ce13219e-911e-4070-b1da-b4f7912226fc` |
| 原PID / start_ticks | 1125380 / 162861942，已终 | 1125641 / 162864044，已终 |
| tokens | 112853 | 114733 |
| CLI reported USD | 0.713555 | 0.74928025 |
| input bytes | 416497 | 417344 |
| 实际input SHA | `334924f6183a8e3b886fc4a44f453b00fae1551c8d1e584e9cd5f94e1bcd0dab` | `fd437fd16370809649d9e856d45b122da960289709afa2d6d227ea34b0ef6d70` |
| 原Invocation SHA | `ba4f4ec4babf594f117249a3c7c90a891151e1341c80577c8022099b3ca3b320` | `0b701a2a243512454a0f402df41bf14c362fe73b64236c86e1884b3ba9dd1067` |

coder tokens组件为input6/output439/cacheCreation112408/cacheRead0；reviewer为input6/output1875/cacheCreation112339/cacheRead513。两原terminal的aggregate与modelUsage相符，CLI total_cost_usd与对应modelUsage.costUSD相符；无矛盾tokens0。报告成本按十进制精确保留，非账单总额。

每个原init均为qwen3.8-max[1m]、permissionMode=dontAsk，tools/MCP/skills为空，profile允许的原builtin plugins匹配。原exec/locator/session/op/input/assignment摘要及600000ms ceiling逐项核验。实际input与preview的Ref字节不同，因为真实capture另行发布delivery；正文、完整材料hash/size、运输objective、assignment保持一致，未用preview摘要代替实际capture。

coder返回完整current_source逐字不变；reviewer返回 `{ "transport_probe": true, "qualification": "transport only; Task and fidelity not tested" }`。两者均来自各自真实成功terminal并通过既有 `validate_capture`；没有借B17失败assistant输出制造资格。

原资格Ref：`artifacts/3c6584d2326c4c94b6d3b751c0b8dc69/native-qualification.json`，SHA **`7593f2d91103aa56a8a0a4354aa5a25df01009ddfbd217be3bff27f35b064301`**。原usage index为 `artifacts/8c59cf6a744a0d1d5f8350fc30298fdb/usage.json`，SHA `b6960e2bdb50407cc1a55ce9314fb3bb7722ac01281bdb017d6edc862105dac1`。原process/exec/Invocation/usage及scope全部保留在native-capture。

新run active78475ms，精确累计 **78475235895ns**。parent active由207911变286387ms，显示差78476ms；原parent自带824862ns尾数跨越毫秒，因此两个独立取整结果相差1ms。只读审计先以整数毫秒相等断言被拒，随后按既有账本的ns精确相等核验通过；无源码或账目修正，未重跑收费scope。

最终快照：6个operation全终，原4个operation及其request贡献逐字保持勘误后原样；总model3/jobs2/active286387ms，active_run/active_since=null，无fence。parent tokens仍unknown，lower227586只是数学下界。限额/descriptor未变，剩余 **21model/14jobs/10513613ms**，roles剩planner6/coder6/reviewer15/extract0。快照owner SHA `39b113367665d5a0d936867fd2cdc31027682bd563f32d2a8f8d5cca1505fd8f`；收费slot已交回根，此后该快照不冒称后续活动的最新状态。

## 导出与后续边界

`qualification-export`通过公开选定Ref闭包导出原资格、两角色的input/exec/Invocation，不复制凭据或运行时owner。`business-selection`通过既有 `native_probe.select_request` 从**新profile业务Q**生成，保留原业务objective、Task/R/Rules/Scope/seed、其它materials、原parent和exact Claude profile；没有用运输objective覆盖业务目标。新增资格材料均为host-only roles=[]。

- 业务selection Q：`business-selection/request.json`，SHA `247764077bf38e713c712b14b1b266905308494aa45c8acd4b6d9e64716ba94b`。
- selection描述：`business-selection/selection.json`，SHA `3efc93feeb87da1940b374a6199232711e3375cdad400398e8727e1d6d52a102`。
- 资格export描述：`qualification-export/export.json`，SHA `a629bbc15b439d2a5306745ae32e697137fe48f937798fff0f3f589d69cf105a`。

selection是未执行建议，`business-run-not-started`目录不存在；原业务profile没有K6完整GPU channel/六作业资格。本项native运输资格不构成Task、GPU、fidelity或性能资格，也不授权自动启动业务design/coder/GPU。后续L20D新evaluation profile由根安排，沿同原parent结算。

全会话真实模型累计10（原8+本项2），GPU jobs累计2；CLI已知费用部分累计2.453513000000000010 USD，历史timeout及B17矛盾用量使整体tokens/总成本仍unknown。没有把部分已知费用当全账总额。

本项没有修改生产源码或运行全套测试。定点验证为原证据只读parser复核、独立安装逐SHA/CLI inspect/auth、完整输入闭包/alias和headroom校验，以及真实两role的原生终态/receipt/进程/usage验证与公开selection生成校验。机器交接为 `/tmp/schema-b21-native-qualification-20261121/b21-final-handoff.json`；详细证据索引见同目录`evidence-manifest.json`。
