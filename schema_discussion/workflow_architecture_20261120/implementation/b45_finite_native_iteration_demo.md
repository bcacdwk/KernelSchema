# B45：薄有限迭代入口与下一次 native demo

2026-11-21 UTC。**第1版已产出并实际测量了 Triton候选，full、原六case正确性与Agent发起的同allocation ABBA通过；host promotion未通过，因为原4-job group无余额执行host另一次ABBA。** 唯一获批 `api.iterate` 已把此真实失败交给 `optimize-002`，writer完成并选择keep，循环自然停止，没有第二版实现。没有扩大预算、重投job或更改原状态；本次没有host promotion。收费slot已交回根代理。

实际候选：[kernel.py](/tmp/schema-b45-finite-native-20261121/artifacts/implement-001/kernel.py)，2089B，SHA256 `105d89d7432e0662625b9388df68ac1f36ecfcecb1bd4494074b3adc7a5f7e5f`，逐字来自模型public-candidate，核对full/ABBA/六case记录均为该源。原 candidate revision 为 `fe100e75632180c73a63f7e257f8eed36bd20a7a`；日志中另一commit字段不替代这个候选身份。工作区当前incumbent和v1 carry-forward不能代替它。

| 第1版实际原job | 结果 |
| --- | --- |
| 新V0 `ev_5f4a8683ce92` | 3shape各6case通过，参考PyTorch，非新候选 |
| 候选full `ev_ea85b6fe0359` | 3shape各6case通过、max_abs_err0；时延6.144/7.168/8.208µs，GeoMean7.123534µs，原performance_score1.492038 |
| Agent ABBA `ev_9e3c06b15bbc` | 原typed同allocation完整收据；3shape、1 ABBA block，每shape各2baseline/2candidate采样，完整通过；原AKA投影baseline GeoMean17.545749µs、candidate7.465671µs、speedup2.350190 |
| 原CandidateGate `ev_84872b7fccff` | correctness_only，3shape各6case全pass，max_abs_err0 |
| host收尾ABBA | 新trusted record `2cfcc9248fed3aa44dd621a83de9a0ca7a39290e9bd0001f14d8370f127c00d5` failed，**没有第5个job ID**；原错误 `native evaluator allocation exhausted` |

[原job收据](/tmp/schema-b45-finite-native-20261121/native-evidence)、[按字段提取的完整事实](/tmp/schema-b45-finite-native-20261121/native-records-summary.json)、[原实现终态](/tmp/schema-b45-finite-native-20261121/native-001-terminal-summary.json)。环境identity仍null/comparability仍unknown，未伪造稳定身份或跨记录可比性；ABBA数字只属于该次同allocation比较。

原 `RecordedABBAValidator.verify` 会走 `execute_trusted` 生成新的trusted request ID。已有Agent ABBA使用不同request ID，`SharedMeasurements.execute` 的跨record reuse仍要求environment_identity，因此当前环境下host另一次ABBA仍需job。此前4-job最小估算只覆盖“Agent不提前ABBA”的路径；这次Agent按原注入的可选ABBA指令提前测量，所以真实正常host收尾需5jobs。第4job继续正常完成；未暂停或热改现场。原native结果为 candidate_ready/accepted=false/verification ERROR，API则completed、stop=`stall: 3 episodes without promotion`、usage.unsettled=[]；driver因此将这个已结账结果及五份原feedback逐字交给optimize，而不是擅自改为blocked或promotion成功。

原coder receipt：1 host invocation、exit0、未超时，terminal exact tokens965010=input114+cache-read871598+cache-write67481+output25817。原telemetry另有 `usage_delta_exceeds_terminal` 警告，独立保留；它未抹掉上述原receipt。4 accepted GPU jobs均终态，B14 native request active984471ms后auto idle，writer独立记账。写作已completed/keep、unsettled=[]，独立owner亦idle。两stage合计2 host model invocations、4 accepted GPU jobs、999165tokens、1173404ms active。写作原receipt报告USD0.364455；native coder标准receipt未报告USD，故总费用unknown，不能从tokens换算伪造。

本轮实际出现ABBA与record-read的EXIT147/Stopped；两条命令使用分号而非`&&`，后续read/head成功读到完整JSON并继续episode-report。没有因此重投GPU，也不能把本次host promotion失败归因为147。只查看owned session的tool_use/tool_result白名单，未选取/输出thinking块；[终态公开工具证据](/tmp/schema-b45-finite-native-20261121/owned-tools-terminal.json)。

## 独立入口与打包检查

工作目录：[b45](/tmp/schema-b45-finite-native-20261121)。没有改 B43/B44 产品源码；没有访问、枚举、执行或修改禁止的实验仓及其归档目录，也没有 home 递归搜索。只读参考了原 B40/B41 的指定材料和 SDK/解释器。

- KS 固定 `13efd1554d112d56e4d767ba43b66257b2d75669`，精确选择源码与写作资源共 135 个文件，位于 [ks-source](/tmp/schema-b45-finite-native-20261121/ks-source)。[manifest](/tmp/schema-b45-finite-native-20261121/ks-source-manifest.json) 保存全部原文件 SHA256；snapshot SHA256 `7c08af0d14b817bbbd1a03c8efb72a1ee8a6d662fbae189657d5b242ff1f8287`。
- `kernelschema iterate --help` 与 `aka optimize --repo-root … --help` 均真实 exit0，并包含新有限迭代选项。保存于 [KS help](/tmp/schema-b45-finite-native-20261121/ks-iterate-help.txt)、[AKA help](/tmp/schema-b45-finite-native-20261121/aka-iterate-help.txt)。AKA 这一步只读当前树，不是最终固定配对的运行证明。
- 本地 `build --wheel --no-isolation`、`PIP_NO_INDEX=1` 检查 KS wheel：新 `kernelschema/iterations.py` 包含其中，解包后 `api.iterate` 与该模块可真实导入。见 [wheel check](/tmp/schema-b45-finite-native-20261121/ks-wheel-check.json)。写作需要仓库资源，仍使用文档支持的 source/editable 安装；不将一个普通 wheel 单独称为完整独立写作环境。
- 新 [venv](/tmp/schema-b45-finite-native-20261121/venv) 使用原 B18 的同一已安装 Python，通过本 venv 的 `.pth` 读取原依赖 site。只向新 venv 安装固定 KS editable；没有下载 Python/依赖或修改原 B18 的包。实际 KS 导入均来自 B45 固定 source，见 [import check](/tmp/schema-b45-finite-native-20261121/isolated-ks-import.json)。AKA最终固定 `88d773b3acdd60ed13334dc46404e2d0caefe49f` 后，四包已按原builder的文件归属构建sdist→wheel（no-isolation/no-index），只安装到B45 venv。7个关键模块逐字匹配该提交，安装态两条help真实exit0；见[四包manifest](/tmp/schema-b45-finite-native-20261121/aka-package-manifest.json)、[安装态检查](/tmp/schema-b45-finite-native-20261121/installed-package-check.json)。
- 只补跑六个小型 CPU 节点：已有 Schema 的 keep 停止；缺失 feedback 停止；已有空/非空 out 拒绝；AKA 将原 launch 转交共用循环；off 分支不 import optional 代码。**6 passed in 1.16s**，见 [JUnit](/tmp/schema-b45-finite-native-20261121/boundary-checks.xml)。未重跑旧 strict 大套。

## 发现的原生未结账边界缺口

[配置正确的 CPU 复现脚本](/tmp/schema-b45-finite-native-20261121/probe_unsettled_configured_before.py) 将新 synthetic owner/limits 通过原 standard 公开选项传入，只替换 native host 与 writer 调用 seam；保留真实 `StandardBudget.model/close`、`standard.implement` 和 `api.iterate`。它从提交 `437e30228c82918398fc6b3f3179693361d8696d` 的 Git 对象加载原 standard 模块到当前 Python 进程，不修改源文件；在 native 回调返 receipt 前抛 `RuntimeError`，没有实际模型/GPU调用，也没接触 B14 authority。

原 `StandardBudget` 正确将 coder operation 标记为 `terminated=false` / uncertain；但提交版 `standard.implement` 将异常映射为普通 `failed`，没有 `usage.unsettled`。共用循环随后进入 optimize stub 一次。见 [修复前结果](/tmp/schema-b45-finite-native-20261121/unsettled-configured-before-probe.json)。这会使实际未结账的原生失败继续触发 planner，违背既定“不确定停止”边界。

问题已由 B46 在 standard 的结果投影处修复。B45对 B46当前工作树的[相同配置探针](/tmp/schema-b45-finite-native-20261121/probe_unsettled_configured.py) 已观察到 writer调用0、`usage.scope=native_stage`/unsettled含本coder、iteration blocked，原unknown operation保持；见[修复后工作树结果](/tmp/schema-b45-finite-native-20261121/unsettled-configured-probe.json)。最终固定88d773后，正确配置的探针已再次通过，writer0、iteration blocked，见[固定版探针](/tmp/schema-b45-finite-native-20261121/unsettled-fixed-probe.json)。初版探针未把 synthetic account 传入 standard 选项，作为历史保留；以上配置正确的前后对照是正式证据。全部是 synthetic 错误场景，不是新的真实收费失败记录。

## 只读预算与已同意的最小计划

[B14 只读快照](/tmp/schema-b45-finite-native-20261121/parent-readonly.json) 确认总计 8 models / 7 jobs / 3977366ms / 28 operations，active=null、unterminated=[]、uncertain=[]。角色已用 planner2/coder5/reviewer1，故剩余 **coder3**、planner4、reviewer15，总 model 余额16、GPU jobs9、active6822634ms。历史 tokens 仍 unknown；1496528只是已知下界，不作为总 token 数或可退还预算。

根已同意以下计划，并在最终pair提交后移交本次唯一收费slot：

| 范围 | 参数与真实边界 |
| --- | --- |
| 共用入口 | `api.iterate`，已有真实 B40 Schema 起步，`rounds=2`；AKA switch 也薄转发同一循环 |
| 每版原实现 | `max_iters=1`、`handoff_resumes=0`、最多1 host coder invocation |
| 时间 | 原 agent 1800s / episode2100s；native request 两版合计 active≤4800000ms |
| GPU | 每版原有限 group 4 jobs；两版 request 最多8 accepted jobs |
| 4-job 标准路径 | 新 V0、候选 full Evaluate、原 CandidateGate 六 seed correctness、独立 native ABBA |
| 写作 | 最多一次独立 write-only optimize；planner1/reviewer0/coder0/GPU0，profile600s/USD2.00，owner active610000ms |
| 数值/性能 | L20D、Triton、3 shapes 一批、native-abba、measurement_repetitions1、verify_repeats2、verify_run_timeout120 |
| 其他原参数 | framework_baseline=never、convert_after=0、max_stall1、setup_timeout600、原 token admission600000、output_limit_recovery_retries0 |

`owner_limits` 原接口只接受 model_calls/evaluator_jobs，不能表达每版 active 上限。因此准确采用原 B14 内单一新 request：coder2/model2/jobs8/active4800000ms；每版各用原 1/4 group、同一 run_id，原 reserve_group 自动 begin/end。中间 writer 使用自己的原生独立 owner，不手工保持 native request active。**没有每版2400000ms硬界**，没有新增预算字段、分配器或循环。USD2.00是原 CLI 停止阈值，不冒称硬账单上限；native token admission也不是硬 token 上限。

keep、blocked、不确定、缺写作产物均按原循环停止。只有 revise 才有下一版；写作完成不验效果。新 out 与原 workspace 分离，已有 out 一律不能自动重放。正常每版最少4 jobs，因此没有额外探索性 compile/profile 或分 shape 提交的预算。

## 输入与待运行脚本

真实 B40 root 从公开例子逐字复制：9106 bytes、SHA256 `58a83e9be44e4d1d5f656454fbec1de65ec948f49c79aad06f9daeb6b6d30e1c`。任务仍为 bf16 `Model(alpha).forward(x,residual)=x+alpha*residual`；数字 shape0 `(N1024,alpha1)`、shape1 `(N65536,alpha0.5)`、shape2 `(N1048576,alpha2)`。B41 官方 Task 的 reference/input/shapes/metadata 与固定 benchmark runtime 逐字复制，新 [input manifest](/tmp/schema-b45-finite-native-20261121/input-manifest.json) 和 [runtime manifest](/tmp/schema-b45-finite-native-20261121/bench-runtime-manifest.json) 保存长度与 SHA。

[run_iterations.py](/tmp/schema-b45-finite-native-20261121/run_iterations.py) 只调一次公开 `api.iterate`，不重写循环。AST已通过。`prepare` 硬禁止 owner create/register/authorize、模型 invoke 与 native host run，核对真实 prepare_only、composition 和原参数；`run` 还需最终 source manifest、零费用 preflight、根 slot 授权文件完全匹配后才能创建单次标记和注册 request。最终pair/slot已逐项匹配；真实prepare_only与composition boot通过，保持原Light/Agent/LegacyExecutor/LegacyEvaluator/Problem/Source/Candidate组成。预检前后B14完全相同、out未创建、模型/GPU调用0，见[preflight](/tmp/schema-b45-finite-native-20261121/preflight.json)。原CLI hash/version/auth通过，原Executor L20D GET support=supported、reasons=[]、identity=null、availability=unknown，见[环境预检](/tmp/schema-b45-finite-native-20261121/sdk-environment-preflight.json)。随后单次启动，见[命令](/tmp/schema-b45-finite-native-20261121/run-command.json)与[运行日志](/tmp/schema-b45-finite-native-20261121/run.log)。

配置复用 B39 minimal patch，只调整已同意的时间并显式 wiki=false/research_records=false。保留 measurement-session/research-records、gpu-measurement、原 sandbox、原六 seed验收和 ABBA。notes 要求从给定 Schema直接实现并提交真实 episode-report，禁用无关探索。

只读发现现有 `episode-report` bundle 可以选择 report-only 资料，但安装名为 runtime-report，而通用 prompt/错误提示仍链接 runtime-records，直接替换存在悬空链接。根决定本轮保留已有 B39资源，不为未使用 bundle 改名或扩框架；这里只记录发现。

已读取并遵循原 [atrex-gpu-gateway skill](/home/shaohanyong.shy/atrex-kernel-agent/src/aka/execution/skills/atrex-gpu-gateway/SKILL.md)。本次实际提交均由原 AKA runtime/Supervisor负责，现有认证只通过原私有配置交给 SDK；没有打印、复制或写入模型材料。

最终AKA固定source共1676文件，snapshot SHA256 `3157c35c05f986856d336810e8cb8369950e7f9197c1aca4d9c04f2c468f8585`。精简快照首次预检缺原legacy profile仍声明的Wiki身份目录，已从同一固定Git对象补足；Wiki插件/查询继续禁用，未作为模型资料投递，不为此修改产品profile。首次提取也安全拒绝了packages/atrex-aka-schema/src的内部symlink，确认其仅指向快照内../../src后保留并记录。

## keep、最终结账与最小交接

writer run `5d20eff62a114fb0b1871e5b5d76a7c7`，session `79976e60-295c-4c78-8e45-987b52d87a18`：completed/keep、diagnostics=[]、native succeeded/accounting_complete=true/terminated=true、physical_retries=null。它认为预算分配失败不是设计缺陷，沿用当前B40原稿，未改Schema以冒充解决host问题；[原rationale](/tmp/schema-b45-finite-native-20261121/iteration/optimize-002/artifacts/ff605620c4bb4434931f1e5bbe54c630/rationale.md)、[原change_report](/tmp/schema-b45-finite-native-20261121/iteration/optimize-002/artifacts/ff605620c4bb4434931f1e5bbe54c630/change_report.json)保留模型原话，模型说明本身不是新的效果证据。

[keep产物root](/tmp/schema-b45-finite-native-20261121/iteration/optimize-002/artifacts/eb432a81521d41c2a4a8395d4ad42fde/schema/root.yaml) 与[保留原稿副本](/tmp/schema-b45-finite-native-20261121/iteration/optimize-002/artifacts/defdb02c095b4637a4f54357718f4ad7/root.yaml)均为9106B/SHA58a83e9be44e4d1d5f656454fbec1de65ec948f49c79aad06f9daeb6b6d30e1c，逐字等于B40。没有第二版native workspace。writer为planner1/reviewer0/GPU0，34155tokens；runtime active186131ms、request active188933ms，reported USD0.364455。

B14最终累计 **9 models / 11 jobs / 4961837ms / 33 operations**；角色coder已用6/8、planner2/6、reviewer1/16。最终剩余 **coder2 / planner4 / reviewer15 / model15 / GPU jobs5 / active5838163ms**。active=null、unterminated=[]、uncertain=[]。B14历史tokens仍unknown，2461538只是当前已知下界；独立writer的34155tokens没有混入B14旧账。旧14个request完整摘要SHA逐项不变。本次原native group的5个operation均终态；未执行第二版，因此第二版的1coder/4job容量从未被预留或消费。没有新旧账迁移、回滚或预算重置。

[最终stage摘要](/tmp/schema-b45-finite-native-20261121/stage-summary.json)、[writer终态摘要](/tmp/schema-b45-finite-native-20261121/writing-terminal-summary.json)、[parent-after](/tmp/schema-b45-finite-native-20261121/parent-after.json)。运行进程exit0，收费slot已释放；本任务不再启动任何模型、GPU或重投。

收尾核对：AKA1676个固定source文件、KS135个固定source文件、输入7文件与benchmark runtime37文件均SHA不变；5份declared feedback逐字复制未改，full native返回JSON保留；见[完整性](/tmp/schema-b45-finite-native-20261121/final-integrity.json)和[feedback逐字校验](/tmp/schema-b45-finite-native-20261121/feedback-integrity.json)。源码和旧运行未热修，88d773/13efd配对之外没有暗中换包。

后继新任务最小只读复用路径见[handoff](/tmp/schema-b45-finite-native-20261121/root-handoff.json)：保留root、实际候选、原Task/bench、B39式配置和本次真实反馈。新native campaign须新workspace/request，并按根已知的5-job路径准备；现有standard `docs=(candidate_path,)` 可显式把已测候选正文交给原Agent，无需新coder/迁移框架。若根固定B48或其他新源码，应建新的匹配源码/venv，不修改本任务的固定环境和旧结果。根的后续动作不属于B45本次收费slot。
