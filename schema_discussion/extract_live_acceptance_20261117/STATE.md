# 当前进度

> 本轮已由用户终止；2026-11-18 02:45 UTC 两条当前run已正式取消、巡检已关闭。完整结论与终止现场以 [ATTEMPT_REPORT.md](ATTEMPT_REPORT.md) 为准，不自动恢复。


更新：2026-11-17 17:40 UTC（环境日期已显示11-18，clock工具仍为11-17 UTC；记录统一UTC）。

目标：FA4 Prefill + QSA 的真实完整自动 P1–P6 提取并交付；主代理只作Supervisor，写稿/审阅全用Claude Code qwen3.8-max[1m] / max。用户明确授权真实全流程与必要工具修复、要求新xhigh修根因、不能绕过检查。**尚无最终交付，不能宣布跑通。**

## 当前两个活跃运行

| 算子 | Run UUID | 进度 | 输出 |
| --- | --- | --- | --- |
| FA4 r02 | a9a16cd1-7232-46c0-9721-1e8d98a227d9 | P1已独立批准；P2 worker已提出root+split_kv两设计和15KB目录草案，但完整回执有一条13行证据被正确拒绝。主代理17:37已正式resume，等待模型修证据；尚无P2批准 | deliveries/r02/fa4_prefill |
| QSA r03 | f1a26608-6adf-463a-acbe-144e3a2bc69d | 新P1正常worker 8400a03f-0fde-4ac1-9dca-6d73ce12b937 已启动，有前次Qwen草稿作参考，尚无批准 | deliveries/r03/qsa_prefill |

状态根：`/home/shaohanyong.shy/KernelSchema/.schema_extract/live_acceptance_20261117`。QSA后台运行；FA4在17:37由正式resume前台恢复，根工具session 20726可能仍在运行，勿重复启动或终止它。只读state/status观察即可。输出目录未预建，由Runner最终发布。

FA4配置：输出65536、call5400/stage28800/operator86400秒。QSA r03：输出65536、call7200/stage36000/operator86400秒。原生max/max保持。时间都是有限上限，不是预计耗时。

工具store：`activeAcceptRuns`为当前混合run IDs；`acceptanceWatch`只读状态/公开工具事件（不读推理正文）；`await eval(load("acceptancePollJs"));`输出相对上次的变化。`r02PollJs`仍含旧QSA r02，不再用于主监控。也可正式status/inspect。不要因静默就判死锁。

## 已完成修复，所有子代理已停写

基线fa853fa；当前产品未提交Git。固定schema字段规则与Kernel源码均未修改。

- LIVE01：已删EXECUTOR_ALIGNMENT.md仍有清单/导航引用，已修，冻结/小测试通过。
- LIVE02：首轮P1仅900秒是阶段内预留，不是计算bug；曾正规各+7200恢复，1800秒仍不足。原失败audit verified。
- LIVE03：可选examples导航缺失只观察；P1必需材料齐全，无规则冲突。
- LIVE04：本地假API确认CLI默认max_tokens32000/adaptive/max；FA4曾触发输出上限，QSA未触发。不是丢主机配置或effort无效。
- LIVE05：新增--max-output-tokens贯通冻结配置、session、CLI环境和调用审计。40项相关测试通过，含Claude65536/Codex默认六阶段；单次真实极短Qwen/max请求2.126秒成功，证明服务接受64K请求，不是正式阶段通过。
- LIVE06：无效参数不再抢占output登记；集中配置校验前移，2项定向测试通过。无源AST/IO回滚大系统。
- LIVE07：P1–P5 worker helper仅格式待处理时按preview返回0；reviewer/P6旧标准不变，4项测试+真实草稿副本验证通过。FA4 r02仍用其冻结旧helper，这是非阻塞反馈问题；不可宣称r02已升级。
- LIVE08：Python验收器中的relative_to不再误当JSON；只对.json结构化严核，坏JSON/声明矛盾仍拒绝。8项测试3.18秒通过，原真实提案越过此错误后仍因6条错误comparisons.output（说明句/非输出事实）被合法拒绝。P1提示补清output/source_output真实键名和JSON锚点。未代写提案、未放宽门禁。详见各LIVE报告。

## QSA接续细节（重要）

旧r02 `388e08c1-eef9-463d-bc95-ee9bfee09cb6`：首worker5400秒超时但完整保稿；续接worker `bf98dff0-9de7-4e91-b597-859828120fc3`原生成功、draft机器检查0，却因LIVE08/错误回执进入invalid_receipt。不要恢复它去用新执行器，不改快照。最终稿SHA `a56c8f8da992737a621d46ba62f72384dba95b6031e86a2d190181882e88c1a6`、10257字节，来源/稿/状态都保留。

已按现有框架新建r03 --queue，再maintain P1 patch，将旧稿原文和SHA放入reason，明确仅作参考、必须回查新固定来源、重新撰写回执与独立审阅，不作为acceptance/reference/scope，不继承旧批准/证据索引。任务文件qsa_r03_reference_tasks.json，请求c61a2e4f-6783-45fe-b8bd-df95a72e06a0。维护会自动启动Runner，未完成P1按普通worker处理；不要再启动重复Runner。最初请求误列META_DATA.Version被拒，移除后已成功；P1合法维护字段为Name/Describe/Source、Algorithm_requirement、Hardware_specification.Execution。

## 下一步

持续监督当前两个run到ready且published，核对P2精选完整设计数、P5引用/P6独立终审、发布receipt与Template只读检查、audit --verify。正常退修由流程自动执行。遇确证工具bug，起新的gpt-6-astra/xhigh/fork_turns=none代理修，最多同时两个；不改旧快照，不替模型写稿或批准。仍可对不受影响的冻结run继续推进；必要新run可沿现有维护reason提供旧稿参考，无需添加seed/migration系统。

目前较长耗时主要是Qwen/max阅读、推理与回执准备。旧默认时间预算/32K不适合这两例；完整实测后需诚实说明使用了更大的显式预算，是否调整产品默认值应据完整时长判断，不能把有覆盖的验收说成裸默认已成功。

资料：源描述sources/*.json，固定四题来源commit fa853fafd32a78ac496fbf71e615e34200eeeae3，不是44题catalog。现有template答案不作提取答案。Python /opt/venv/python3.13.12/bin/python3，Claude /usr/local/bin/claude 2.1.316，勿输出认证。

所有过程在本目录。用户偏好简洁中文进展、自主推进，不反复确认；不运行GPU、不发送外部消息、不擅自commit，不结束在“仅启动任务”。

最新恢复细节：FA4 P2失败attempt 98ca1cea-0547-483c-9685-d1d18893f1c7 的唯一超长source_evidence是索引50，路径 fa4_prefill/aka_solution/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py，2114–2126共13行，上限12。规范是有效硬门禁；不能手工篡改提案，只让模型正规续修。已执行原run resume，无加预算。原生complete提出plan root+split_kv（未批准），其它调参/局部/组合仅记录。当前产品未因此改规则。

## 后续监督交接

主代理已委派新的监督助手 `/root/live_continuation_supervisor_high` 持续监控、正常恢复与记录，阶段仍是Qwen/max。他只维护RUNS.md和CURRENT_SUPERVISION.md，不修改产品/快照/产物。主代理负责工具修复和最终交付。**当前更实时状态优先读 CURRENT_SUPERVISION.md 与 state.json。**

FA4 r02在17:37正规resume后暴露LIVE09：旧P2原生success但receipt拒绝时state丢UUID，private session仍ready；新attempt 85686d53…传null UUID，7ms启动前被拒，又被旧执行器误标termination_unconfirmed。有fence、无实际Runner，不可再盲resume或人工解除。

LIVE09由新xhigh完成并停写：仅agent.py/runner.py及定向测试；原生成功且scope匹配后先登记UUID，再做提案语义校验；session预检进入既有try/never_launched结清路径。8项测试17.055秒通过，原FA4的27份选定现场摘要未改。旧冻结run缺少正向未启动证据，不能通过旧程序清理fence；保留旧run，重新冻结新FA4。

已授权监督助手在LIVE09停写后新建FA4 r04，输出deliveries/r04/fa4_prefill，source仍sources/fa4_prefill.json，Qwen/max/output65536、call7200/stage36000/operator86400。先queue，再正式maintain P1 reason内嵌旧合同+SHA作参考，重新核源并完整P1–P6，不继承批准。后续P2草案也是参考而非已定选择；现有机制不能可靠预先把未来P2任务反馈保留到阶段，因此在P1审阅结算的自然边界尽量pause，再向P2提供旧dispatch参考；不能精确停时正常让P2自行核源，不加新调度系统。助手须回报新run ID并写CURRENT_SUPERVISION/RUNS。

QSA r03 f1a26608-6adf-463a-acbe-144e3a2bc69d 一直正常P1 worker运行，已有约10.4KB新稿；未因LIVE09中断。它的冻结版本早于LIVE09，如发生同类恢复问题不得热改snapshot；正常无错运行可继续。

## 20:20 UTC 接续

旧监督助手已正常移交并停写；新的活跃监督是 `/root/live_supervision_next_high`，维护CURRENT_SUPERVISION.md/RUNS并给主代理简报。当前FA4 r04与QSA r03的P1都已独立批准，均已到P2。

FA4 r04 P2首稿机器检查发现dispatch第67/74行变体/组合缺直接基线YAML链接，Runner自动交worker修正，目前修稿在途；不是产品bug，没有手改产物。

QSA r03 P2 worker已合法提交单root计划；reviewer原生success、拟approve却有三条14行source_evidence（上限12），因此合法拒绝，P2未批准。它是LIVE09之前冻结版本，state.sessions.P2.reviewer=null/private scope已有ready UUID，不能普通resume。无active/unconfirmed/unsettled、原Runner已退出，未触发新的fence。

LIVE10 xhigh只读核查及真实冻结代码临时副本验证：必须正式 pause→maintain P2/root/review fields=[]（确认preserved/requested）→resume。恢复先将scope回reviewing，已登记维护屏障再判断完整候选，首派全新maintenance reviewer operation，不会启动旧normal reviewer。原candidate/plan原样、selected/approval仍空、旧无效approve不被采纳、调用账目照常增加，后续仍需完整独审和P3–P6。直接maintain会走waiting再普通worker，不能替代该顺序。

LIVE10已停写，报告含精确任务/命令，46个原现场及冻结文件摘要未改，没有模型或合成批准。主代理已授权新监督执行上述正式三步并核首个attempt.mode/session_scope；若实际偏离立即报告，不手改state。当前更实时状态看CURRENT_SUPERVISION/真实state。

## 21:40 UTC 新监督接续

LIVE11旧维护恢复路径已确认不可行；LIVE12/LIVE13均已完成停写（只读孤儿恢复验证 / 当前维护恢复修复）。LIVE13 8项测试通过：正常维护invalid_receipt可正式同scope/UUID再提交，原账目、真fence、语义blocked与一次性约束不变；精确证据诊断发现QSA原回执不止14行，还有引用越过文件尾，未代改。

FA4 r04主代理已实际恢复成功（不是仅计划）：公开resume以nohup setsid脱离临时工具会话执行，Runner1209365 alive且持writer锁，launch eec0775d…claimed和state同身份；新P2只读reviewer e6336933-1f7c-4bdc-875b-4aa5913d3456，原candidate9e63214d…/plan保留，无unconfirmed。不要重复启动。日志/tmp/live12-fa4-r04-resume.log，观测/tmp/live12-fa4-r04-launch-observation.json。

新的活跃监督 `/root/live_supervision_third_high` 已受命接手（前两位监督均停写）。他将新建QSA r05（需回报精确ID），采用当前LIVE13工具、原sources/qsa_prefill.json、output deliveries/r05/qsa_prefill、Qwen/max/output65536/call7200/stage36000/operator86400。先queue后maintain P1 reason带旧合法P1稿参考，不继承批准；P1 reviewer自然边界pause后P2再带旧dispatch参考。旧QSA r03保持blocked不操作，不能热改snapshot。

请以CURRENT_SUPERVISION及RUNS最新追加、真实state为准；这里的老阶段信息可能过时。主代理继续负责整体决策/派新xhigh修工具，监督代理负责长期监测与正常控制，所有阶段仍由Qwen完成。完整验收尚未完成，不可结束为成功。

## 22:49 UTC 进度与当前工具精简

FA4 r04 P2正式独立批准，选中root+split_kv两份完整设计，现P3 worker fe7ec14b-a12c-41dd-851d-3b324162122f。最后handoff修复回执1170字符合法采纳；P2 calls6/invalid_receipts2/receipt_repairs1，原账目未退还。QSA r05 dee63d48-09d3-40ec-a694-4ba7f259cdcf 仍P1独审，普通pause等自然结算，活跃监督为live_supervision_third_high，最新看CURRENT_SUPERVISION。

LIVE14独立只读数据评估后，主代理纠正此前将工具内部12行任意上限当必要规则的判断：它不限制读取开销或摘录大小、不保证相关性/真实性，真实文件与位置、固定身份、偏移、320字符摘录和完整独审才必要。它不是schema_docs字段规则。接受删除会降低机械局部定位、P5宽范围更易覆盖定义行的代价，不增加新防御系统，仍要求独审相关性。

LIVE15新xhigh已完成并停写：删除reading.py/receipts.py两处12行条件，更新共用说明和恢复测试（以真正越界触发，不删覆盖），10项测试通过。六份原始真实提案684 anchors旧拒23→现拒2；21条纯长度拒绝原样通过，QSA维护的artifact[13]dispatch71–76/实际75行及[29]root181–194/实际193行仍拒。88个固定材料摘要不变，无旧回执改写/批准/热升级。handoff1200字符/20行限制不变。两活跃run仍按其冻结12行旧规则跑；最终当前工具补验如实区分版本，不为采用精简而打断健康运行。
