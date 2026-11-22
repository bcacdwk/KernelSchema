# 当前简化实现的交付审计

2026-11-21，B50 公开材料与本地审计、根的最终提交和推送均已完成。**KS 和 AKA 两个公开入口目标已实现；B45 真实有限迭代已有候选正确性与 Agent ABBA，host 复核因 4-job 额度失败；真实失败经 optimize keep 自动停止。B49 在新 request/workspace 完成独立 host PASS、accepted=true/reused=false，五个新 GPU job 均终态。** 按 [用户简化决定](../simplified_execution_20261121.md)，写作无需独立效果验收，旧 strict 资格仅保留历史。

## 两个目标的准确落点

| 目标 | 当前行为 | 真实证据边界 |
| --- | --- | --- |
| KS 直接写作和有限迭代 | 任意 task 的 design/optimize；唯一 api.iterate/CLI 调用标准 implement，真实结果和声明反馈交 optimize，revise 才下一版 | B40 completed/revise；B45 completed/keep 自动停，候选正确性与 Agent ABBA 已过、host promotion 未过 |
| AKA switch 接入 Schema | `--schema --schema-iterate` 薄适配同一 KS 循环，原 WorkflowLaunch/coder/工具/评测保留；静态 switch/off 不变 | CPU 接线通过；B49 从 api.implement 使用同一 standard/native 路径完成独立host复核；AKA switch的薄转发由既有CPU证据覆盖 |

`rounds` 计 Schema 版本/native campaigns（默认2）；每版新 workspace，原 `max_iters` 控制版内 coder 轮数。writer 在版本间运行，非每个 coding episode 都调用 planner。新 workspace 不会自动把前版代码设为 incumbent；可用原 docs/schema-doc 显式提供候选为参考。revision 不承诺性能单调改善。

settled failed 可交 writer；blocked/interrupted/unsettled/缺反馈/无 completed 可读稿停止；已有输出拒绝重放。B46 只读投影显式 owner 的本 stage unsettled IDs，防止继续收费；默认 OwnerAccount 非必需。B44 保留单次 typed 终态、源绑定、完整测量和 SDK 同 allocation 完整 ABBA；identity=null/comparability=unknown 不改，跨记录 reuse/minima、显式 unsupported/SSH unknown 仍拒绝。

## 当前运行来源与公开材料

最终交付 KS `7be9263fe9a6f5fba97ed7ad7685de13e4d6146f` / AKA `87404feb703891efa9d03221d2ae4fdb5068efdc` 均已推送；gitlink 与 release 精确选择 KS7be9263，301 文件来源验证通过，两个开发工作树干净。相对下述实跑来源，后续仅有公开例子/文档与配对变更；核心工作流代码未改。详见 [最终交付清单](final_delivery_20261121.json)。

KS `13efd1554d112d56e4d767ba43b66257b2d75669` / AKA `55bc39f3239e2d9fa65d2a799182a26df413368c` 均已由根推送，同分支 feat/schema-workflows-v1。AKA gitlink/release 选择 KS13efd，原301-file release SHA `9d2c4f4789b75b4e9e2f7c791cbf0a7965d3768137bbcc54739a5e9c8e1e8da4` 与 verifier 的 B46 证据保留。55bc 含 B48 wrapper 修复；B45 固定88d773快照不热改。

B50 仅整理 KS README、vector-add 示例和 standalone 使用说明，逐字复制最终kernel并选取公开results字段；产品清单与本地审计精确manifest分开，见[B50报告](b50_final_example_and_audit.md)。不自行stage/commit/push或调整AKA gitlink/release。B40 root 保持9106B/SHA `58a83e9be44e4d1d5f656454fbec1de65ec948f49c79aad06f9daeb6b6d30e1c`；公共文件不含私有回执、CLI thinking/base64、账户或凭据。原 main 只写指定 discussion，原提取/实验仓/旧账/运行快照不触碰。

## 真实结果与用量

| 阶段 | 可确认事实 | 限制 |
| --- | --- | --- |
| B32 | reference baseline 三 shape PASS，ev_1861beac3248 | 参考 PyTorch，非新 kernel |
| B33/B35/B36/B41 | 原失败、无候选终态保留；B41 1model/1V0job/661725ms，partial239774tokens | V0 与 carry-forward PASS 不冒充候选，完整 tokens/USD unknown |
| B40 | write-only completed/revise；planner1/reviewer0/GPU0；40303tokens、USD约0.58199875、393087ms | 无效果判定 |
| B45 native | 候选 SHA105d89d7432e0662625b9388df68ac1f36ecfcecb1bd4494074b3adc7a5f7e5f，revision fe100e75632180c73a63f7e257f8eed36bd20a7a；full/6case 每shape过、max_abs_err0；Agent ABBA 2.350190× | host 别次 ABBA allocation exhausted，没有第5job；accepted=false、verification ERROR |
| B45 writer | 原真实反馈逐字交 optimize；completed/keep，无第二版 | native1coder/4jobs/965010tokens/984471ms，writer1planner/0GPU/34155tokens/188933ms/USD0.364455；合计999165tokens/1173404ms，总USD unknown |
| B49 | completed/exit0、accepted=true、host PASS/reused=false；新full/6case均3shape各6case/max_abs0；host独立ABBA baseline11.226588µs / candidate7.466858µs = 1.503522× | 显式docs参考B45同字节候选，非第二次从Schema独立生成；1coder/5jobs/657102 exact tokens/811673ms，无writer/新Schema，USD unknown |

B45 原4jobs依次为 reference V0 `ev_5f4a8683ce92`、候选full `ev_ea85b6fe0359`、Agent ABBA `ev_9e3c06b15bbc`、correctness_only `ev_84872b7fccff`。Agent与host分别提交ABBA且unknown identity禁止跨record reuse时，正常完整路径需要5jobs；没有把第5次拒绝写成GPU job。full latency为6.144/7.168/8.208µs；ABBA基线GeoMean17.545749µs、候选7.465671µs仅属于该次同allocation比较。

B49 的五个job为V0 ev_4c605098b3bc、候选full ev_7102dba05dfd、AgentABBA ev_7840da47d4aa、six-case ev_9e107207e34c、host独立ABBA ev_53c06a9e4f27；host record3ea86742e7fbd66078184d11901469be19ed5932dffd0ae970e6aa9817b8d1a6和trusted request独立，reused=false。Agent本轮1.556388×、B45旧2.350190×与本轮host1.503522×各属不同allocation，不能混成性能变化或统计显著性结论。两次ABBA各1完整block/shape各2基线2候选采样，每完整run1case；full和gate另各6case。

B49 candidate_commit21563597f24e9ab4aca4ffcc2a812e9da8d18363、promotion_commit4b41a98516f7ba3681e75d18410c47e09bb385ed分开；最终kernel2089B/SHA105d89…7e5f与B45参考逐字相同。parent最终10models/16jobs/5773511ms/40ops，active=null、unterminated=[]、uncertain=[]；剩coder1/model14/jobs0/active5026489ms，旧15requests/33ops/descriptor摘要全不变。request811673ms与parent增量811674ms的1ms投影差异按原数保留。B45独立writer不混入parent，全会话tokens/USD unknown，已知下界3118640不当总tokens。收费slot已释放，后续0调用。

## 验证与审计范围

复用 B43 KS27/AKA21+2、B44 28、B46新增4+兼容2 CPU；B45 安装、两help、六个边界节点、固定unknown-stop probe与终态完整性已存证。B48 45项CPU/10次自然调度证明原 Python wrapper 暂时关闭monitor后等待真实终态并恢复monitor/真实exit0、7、147；原暂停、active冻结和保护保持，绕开wrapper或任意outer pipeline不作承诺。

B50 仅检查新文档链接、shell/Python片段语法、公开原稿字节及历史保存；不重跑旧大套，不调用模型/GPU，不另建平台。H001–H096、B01–B13原文、progress units/gates/旧rollup和旧requests原位保留。B49原报告、summary、五个receipt哈希和实际kernel字节已只读核对；全部通过，不重跑测量。最终实测与manifest见[B50](b50_final_example_and_audit.md)，只待根提交公开材料并完成后续源码配对。

---

下方为 B01–B13 原审计的历史正文，仅调整标题层级。其中所有“当前/最终”、7次模型/GPU0、401以及严格六作业前置都只描述原阶段。它们不是现在的运行总量、认证状态或默认准入条件；原失败/unknown事实仍照录。

## B01–B13 历史交付审计原文

状态：两条有限工程路径已完成实现、独立审阅及推送，最终固定KS `1c9e592` / AKA `804e26f`；两仓工作树干净，远端同SHA已验证。真实模型/kernel/GPU闭环尚未验收通过。工程实现与真实运行资格分别验收，不能把可调用入口或 CPU 通过记作真实 kernel 成功。

### 两个目标的工程落点

| 目标 | 已实现路径 | 当前限制 |
| --- | --- | --- |
| KernelSchema 独立迭代 | 唯一 CLI/API；原 extract 透传；共享 design/optimize；reference implement；有限 driver 执行初始设计或已有 Schema→实现→至多一次优化→合法后续实现/验证 | 真实业务模型、GPU、完整 fidelity/性能 demo 未资格 |
| AKA 内 switch 开启 Schema | 默认 off；可选包与固定 KS gitlink；原 Agent/NativeSession/loop；fixed 实现与阶段式 dynamic planner；同一 workspace 和账户 | 首发 row-softmax/single_file；真实 native transport、Task、动态业务资格待实际证据 |

两仓分支均为 `feat/schema-workflows-v1`。KS 在 `.worktrees/schema-workflows-v1` 开发；原 main 留给另一个进行中的 12 算子提取任务。已完成的 6 算子和对应 21 份 YAML/Markdown 已逐字核对保留。AKA 的 origin 仍是 alibaba，上游基线 `bf5557ac04f6ff9988901d511cf25e1cbe61a1bc`，开发推送到 bcacdwk fork。

EXP 不参与依赖、安装、测试或运行。其将来只需采用更新后的 AKA 及该 AKA 固定的 KS 子模块。本任务没有 EXP 工作包。

### 用户离线期间采用的主要决定

| 决定 | 依据与影响 | 记录 |
| --- | --- | --- |
| 工程继续推进，真实条件缺失单独保留 | Agate 用户已配置；当前会话的官方 SDK 请求返回缺签名头401，未发现可用标准配置。继续实现并检验本地协议、通道和恢复，不据此说服务不存在 | H033、H037–H040 |
| AKA 首发用原 native 框架的 no-tools profile | 宿主固定完整资料并安装模型返回源码，避免复制 AKA 或建立另一套 agent 平台；当前模型不交互调用 read/effective/check | H035、H037 |
| D3 只在完成固定阶段且有显式优化目标时触发 planner | 普通编译/数值/fidelity失败、unknown、耗尽和未晋升继续属于同 R 实现；不凭失败猜测设计障碍 | H040、H047–H048 |
| 同 parent 账户贯穿所有 Q/R，unknown 不记零 | 新 Schema 发布不是费用准入；所有实际操作仍 reserve/authorize，恢复缺绑定不能重建空账 | H039–H040、H047–H049 |
| FA4 本轮选择 KS reference 路径 | 完成 F1 有限多文件 Source、Task/evaluator/runtime/fidelity/channel 工程；AKA 不额外扩大到所有后端/案例组合 | H041、H043、H046、H048 |
| 先减少真实投递冗余，再有界验证 native transport | 同一输入从1,012,929到354,400 bytes；必要非host正文逐ID同字节、去重前校验权限。原超时记录和unknown费用不改 | H044、H046、H048 |
| 维持 Python 3.13 隔离环境 | 停止可选 Python 3.11 下载，不替换系统解释器或干扰提取；3.11兼容性未签 | H033及implementation记录 |
| 业务期限及资格scope期限均显式配置 | 真实超时暴露全产品/工具120秒硬cap；保留默认120，允许明确较长的新profile/资格scope，均取原预算和绝对deadline紧限，续跑不覆盖。三个已失败scope仍120不变 | H051–H053、H057–H059 |
| 不代权利人决定许可证 | 保留AKA现有LICENSE/NOTICE与第三方来源；KS项目/FA4材料的许可缺项记录为发布前待决 | H040、公开release-gaps说明 |

### 已知失败与真实费用

本节统计工作流运行时的CLI模型调用，不包含开发协作子代理的对话。

已确认的历史K3记录：KS CLI profile实际4次，其中首次planner失败保留；最终三角色运输profile通过。已知15,291 tokens、CLI reported USD `0.232672750000000010`，并非核实账单。

AKA旧native probe实际1次coder，120秒超时终止；没有terminal usage，tokens/cost均unknown，reviewer未启动。旧输入、进程与原账户记录保留。本轮KS unbound design另实际调用planner1次：120秒内有init/thinking、无完整proposal，原进程终止，reviewer未启动；tokens/cost同为unknown。原失败结果可零新增调用恢复，108文件/10项内嵌原件已核。见[b10_k5_design_trial.md](b10_k5_design_trial.md)；该报告的“产品120秒硬cap”是原76版本历史事实，已在1c9修复。

AKA本轮运输scope又实际调用coder1次，正式输入355,182 bytes，120秒超时终止、reviewer0/GPU0、tokens/cost未知；原model/session/无tools权限init符合，但缺终态，因此未资格。原安装和源码已只读保全，旧B08原件未改。

当前合计7次模型调用，总tokens和总费用未知；15,291仅为已知token下界。三个超时scope均停止，后续期限配置修复只做CPU，不另建账户重试。

真实GPU jobs=0。CPU transports、fake CLI、synthetic反馈、静态范围检查、历史性能均不作为真实模型/kernel资格。

### 最终审阅与版本

- 当前KS：`1c9e592c6e0ec021f91e17af5a8c39956ce39178`，已推送；最后审阅修复先固定`76ef9b1427e4d24174557054d049e7b134c9958d`，之后增加期限窄修复。
- 当前AKA：`804e26f79d094250a5852ad2b8a6f652f03f2928`，已推送，KS gitlink固定1c9e592；原已验证/实际失败版本`e867333dfcbbce553465635bcf57a9bde842b06f`作为永久源码锚点保留。
- KS最后审阅：修前稳定版本完整CPU **380 passed / 0 skipped**；修复后定点及独立安装以 [B10 KS报告](b10_ks_final_review.md) 为准。
- KS期限修复：34项初轮通过、修正测试收据存放位置后9项定点通过、4项writing兼容通过；默认/字段格式不变，不新增真实调用。旧76公开源码只读快照保留，原失败run身份/reopen已核。见[b11_runtime_deadline_config.md](b11_runtime_deadline_config.md)。
- AKA最后审阅：155完整CPU属于76配对的安装；26角色/期限+5收据/finishing属于1c9最终配对。10次真实CLI/continue零费用smoke属于76配对；新进程来源、完整发行资源和off不变性另有验证。详见[B10 AKA报告](b10_aka_dynamic_final_review.md)，不把分轮证据冒称最终全套。
- B12已核21份算子字节/旧提取不变、KS297资源、680项原安装归档、四wheel源码与e867一致、LICENSE/NOTICE保留、私有记录未打包、公开链接有效；见[最后交付审计](b12_final_delivery_audit.md)，最终8文件、9项证据工件、28个Schema wheel源码与安装来源已核对一致；未发现待修的工程交付阻断。
- 最后配置修复已固定804e26f：新资格scope显式期限、默认120、冻结后continue不覆盖；11scope/probe、10原恢复、3owner缺失补验、3最终安装检查分轮通过。没有新增真实调用或重跑大套，三个历史失败仍保留。见[b13_probe_scope_limits.md](b13_probe_scope_limits.md)。

### 仍需真实条件或后续决定的事项

1. 当前执行会话可用的 Agate 官方配置/认证入口，以及所选GPU和dev ACL。只需配置路径或运行环境，不在文档、对话或Git中记录密钥。
2. 实际原通道六作业资格、只读执行和完整输出/cleanup；随后才运行row-softmax业务demo/AKA fixed与dynamic，再按原栈验证FA4 F1。
3. 实际业务 planner/coder/reviewer 输出及完整用量；本次design失败不重跑。较长业务profile已经可以配置，但未作真实资格，不能继承历史120秒profile的通过声明。profile运输成功不能替代业务成功。
4. core anchor已存在而budget binding丢失的bootstrap缺口仍保守拒绝；不自动补账、换账户或重跑。
5. KS发行许可、FA4逐文件授权/notice及Python3.11兼容性尚未确认；不影响当前源码工程交付，但限制对应发布声明。

完整gate状态见 [progress.json](progress.json)，执行交接见 [next_steps.md](next_steps.md)，逐次决定见 [01_history.md](../01_history.md)。
