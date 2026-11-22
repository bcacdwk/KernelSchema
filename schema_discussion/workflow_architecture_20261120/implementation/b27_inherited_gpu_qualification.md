# B27：inherited proc 独立复核与 L20D 资格

2026-11-21 UTC。**固定新 KS `5341470ba5ac03aea857b6480cc7a813c0e3682f` 的原六作业资格，在首个 probe POST 被 gateway 以 `HTTP400 / dev_sensitive_target_denied / process_root_escape` 拒绝；没有 job_id，后续五个 job 均未提交，L20D 新 profile 资格未建立。** 这是一项真实提交前拒绝，不是 worker 权限失败、CUDA pass 或 cleanup 终态。

本轮只有 **1次原POST、0次GET、0个已创建GPU/evaluator jobs、0次模型调用**。原拒绝按同 parent 已使用的明确4xx拒绝结算规则，由原 `OwnerAccount.absorb/end` 收口：本请求 active **94502ms / 94502267911ns**，原9个operation全部终止、无active/fence，slot已交回根。没有伪造job/terminal，没有更改原Q/profile/qualification state，原KS首次返回的 `blocked/settle_only` 原件保留。

最终证据根为 [/tmp/schema-b27-inherited-gpu-qualification-20261121-final](/tmp/schema-b27-inherited-gpu-qualification-20261121-final)。主要原件是 [拒绝结算](/tmp/schema-b27-inherited-gpu-qualification-20261121-final/rejection-settlement.json)、[只读原件审计](/tmp/schema-b27-inherited-gpu-qualification-20261121-final/rejection-audit.json)、[固定新准备](/tmp/schema-b27-inherited-gpu-qualification-20261121-final/preparation-handoff.json) 与 [完整选定导出](/tmp/schema-b27-inherited-gpu-qualification-20261121-final/rejection-selected-export.json)。原 e5 的独立审阅与零调用准备保存在 [/tmp/schema-b27-inherited-gpu-qualification-20261121](/tmp/schema-b27-inherited-gpu-qualification-20261121)，没有准入或operation。

本代理没有改生产源码、AKA、Task判据、Template、driver、runtime或预算；没有安装依赖、提交、推送、派生代理，未读取/枚举/修改/运行 AKA_experiment。原 main 仅写本报告。实际运输遵循已读 [atrex-gpu-gateway skill](/home/shaohanyong.shy/atrex-kernel-agent/src/aka/execution/skills/atrex-gpu-gateway/SKILL.md) 的原key/locator、终态及不重投规则；认证只由官方SDK读取既有私有配置，凭据没有输出或复制。

## 独立固定与机制审阅

用 `git archive e5a0748b81498cdecd97e12986491e250044b204` 的明确 KS 路径建立自己的 `fixed-ks` 快照；178 个相关 Git blob、B25 的10文件 SHA/长度、原六 evaluator 模块与五份 Task 判据均核对。官方 SDK 解释器复用 `.worktrees/.tooling/k1-evaluation/bin/python`，实际相关模块和 Template 来自本快照，不导入另一代理正在编辑的活跃树。依赖为 PyYAML6.0.3/jsonschema4.26.0/atrex-gateway-client0.14.37，没有安装或下载。

新 `agate-dev-stage-ro-inherited-proc-v1` 明确保留受限 proc 与 masks/submounts。六 namespace 独立性 guard 先于任何 mount；private propagation 和完整 stage bind/remount RO 后清 Inh/Prm/Eff/Bnd/Amb、设 NNP=1，进入独立 scratch cwd、只持标准 IO。完整 stage 包括所有模块、stage.json、remote.json，以及执行时 execution.json/frozen_inputs.json；拒绝 symlink、hardlink、特殊文件、子挂载与同 backing 可写别名。worker 从已核 hash 的 byte buffer 编译实际模块，并由 worker 与 outer 在退出时重核 stage 原字节。

实际权限 verifier 要求已知 outer parent 所持原可写目录/文件 FD、root 和 cwd 别名均 EACCES/EPERM；自身 root/cwd/新开的 stage FD 写开必须 EROFS，ENOENT 不作权限证据。unshare helper 直接父链、相同新 user/mnt/net/uts/ipc 而旧 pid namespace、scratch cwd、非零 permitted caps 都经核验；只接受 stdio 与至多一个由 fdinfo 验证的 self pidfd，drop 后所有 helper 别名亦被拒绝。

已逐 SHA/长度/Git blob 核保存的官方 Linux v6.6 commoncap/ptrace/proc/namespace 源码及 util-linux2.39.3 unshare 源码。commoncap 要求同 user namespace 的 capability-superset 或目标 namespace 的 CAP_SYS_PTRACE；nested 新 user/mount 获得的 caps 不能解除 locked RO 与 locked child mounts。unshare2.39.3 的等待父进程可保留 self-pidfd，child 于 exec 前关闭，与具体允许的 FD 集合相符。远程四工具路径/hash由原真实原件逐项核对：unshare/mount/setpriv 来自 B19 job `dv_dca077c701c8`，Python 路径/hash 来自 B16 job `dv_37ffeb8cd2f2`，没有猜测。

明确不承诺 whole-root readonly、private PID/进程元数据隐藏，或同 Python 地址空间内任意恶意 native code 的隔离；没有换标签复用旧 fresh-proc/bwrap 资格。B22 原 `dv_80d527ccf029` 的 fresh `/proc` mount-denied 失败仍完整保留。

## 本机有限验证

固定源码的新旧 profile 原测试为 **50 passed、1 skipped / 34.68s**，见 [日志](/tmp/schema-b27-inherited-gpu-qualification-20261121/local-tests-final.log)。唯一 skip 是旧 fresh proc 在本机被拒；新 inherited proc 的真实权限、旧视图 FD 泄漏拒绝、self-pidfd 身份和 nested 提权逃逸测试都通过。初次收集因有限 archive 漏带 `reference_support.py` fixture 失败；只补入同一固定 e5 的测试辅助文件后通过，原日志保留，未改生产实现或弱化测试。

额外真实 [生命周期检查](/tmp/schema-b27-inherited-gpu-qualification-20261121/lifecycle-check.json) 在2秒超时后2019ms返回；已知 helper/worker 在7ms有界检查内均消失，scratch 被清理，stage 原字节不变且 outer仍可写。第一次立即检查曾短暂看到正在退出的 worker 为 R，后来同一 PID 已消失；因此没有把 helper.wait 返回当成完整 cleanup。生产仍要求原 gateway terminal、`cleanup_pending=false` 与非 quarantine 才结算，UTC/monotonic 截止与远端剩余时限均保留。

原72+2调用、x/out/storage/view/guard、调用前CPU FP64 golden、逐元素容差、完整 forward 和80个 AB/BA批逐字不变。本机结果只证明被测 CPU 权限与接口路径，不能代签 CUDA/Task/Schema fidelity。

## 零调用准备和版本耦合

通过原公开 `Demo.create(start="design", parent=原Ref, parent_bundle=原Bundle, authority=原路径)` 建立 e5 `standalone-demo`，使用 B23 `full-one-episode` 显式预算入口。writing2700000ms、planner/reviewer minimum各600000ms、完整 finish floor1320000ms（coder1/reviewer1/job1）；code1500000ms一轮、verify800000ms；reference minimum coder600000/evaluator120000/reviewer600000ms，配置一次冻结。B21 exact 2.00 USD/600000ms runtime SHA仍 `3e1766d7fad58da4e76b77c28a296b0adcb225c04e7e56704dac45a848efc530`；L20D target SHA仍 `d4b9c90c350b4db10664ac9ee66ec118c852a03f5c9ffd955204ea89e31ddb0f`，精确设备名末尾8空格保留。

准备前后 owner SHA同为 `f5cf57f0bc32b4f218d9a81b7c21a1e18bfa2e3872de1ef819ddaaa1b27b0cad`；8 operation全终、无active/fence，累计model4/job3/active944642ms，剩20models/13jobs/9855358ms，roles剩planner5/coder6/reviewer15/extract0，tokens仍unknown。资格加两次writing/两次code的冻结请求窗口上限向量为12models/10jobs/9600000ms、planner2/coder4/reviewer6，均有余量；这是预算检验，没有预约或预扣。

初始零调用Task SHA为 `3c24f5d2e825f3645c7558e5b37dc930ff9358831bdbcbc4304bf2743a40e74b`，evaluator SHA `4c1535ba943dc6d4b6360e46528cfd39a6f05be146f44b48332204d352b2cbbb`。原 prepare_agate 将生成的资格Q预览SHA为 `d35c2dda865a1b9d625d22074871010a881038adf55d1d669e46888299439546`，但该 Q 尚未准入，尚无job或key。

在首POST前，根要求核B26 writing耦合；B27与B26独立确认：e5 Rules冻结METHOD/writing.py/_writing_policy.py，WritingBackend._rules逐SHA核实际实现，Demo后续Q持续使用config.rules/writing_profile。不能原位换新writing实现。根因此在首POST之前等待B26固定提交，再从新的独立快照创建新Demo；旧e5准备目录没有qualification运行目录，原父账在两次Demo.create前后都逐字不变。没有热改已冻结Q或重投旧scope。


## 新固定源码、Rules 与实际提交

B26提交后用 `git archive 5341470...` 再建 `final/fixed-ks`，181个Git blob逐字核验；B25原10文件、新B26五文件、六evaluator模块和五Task判据都与各自已签收SHA/长度一致。新 source manifest SHA为 `24f494dd8865edbbf376b76f76abebd2afadf0bae1780ffaacadda5f7ca6db83`。全部实际模块仍来自自己的快照，未安装或导入活跃树；未重复已有50+1权限回归或扩展能力测试。

重新通过原 `Demo.create(start="design")` 一次冻结完整 full-one-episode 配置、同B14 parent、exact runtime、L20D target和新channel。新writing Rules由实际 `WritingBackend._rules` 只读核验通过；selection catalog20项、text_policy80条，完整MODE/ID/字符计数合同进入本Demo冻结材料。没有调用任何writing业务、model或 `Demo.run()`。

| 最终冻结对象 | SHA-256 |
| --- | --- |
| Task | `f58622933aef7c7e529ca44898d3247ac5f8ff232935fc5a5c7de984f11579c0` |
| Evaluator | `9587809eefc3ae37296db73e9ebf291c346a8d5b10836c780563fb4d9b4bc997` |
| 原资格Q | `f951be3a65998baa2ec32cf52d17d5c7e74058bedc5b36f6c31e11fed4145329` |
| Reference profile | `50644393a390bb3554011bf5e9f2b27819037a99bf8bc82b2d5728c772368820` |
| Writing profile | `4bf21f2eb0d323873c3cb8f1c433194e963485b57098a6b97fb21043570ab5b6` |
| Rules | `f8541ae7e22069ecea8a978e3c9bdf056a33c39354362edb01b5372f02d501ab` |
| Writing METHOD | `bc90e3c710b867acc22ee3c9f248ed79e8e19e352aee7483cd51d8ec8bc76349` |
| Request/actual POST body | `7699c60163d8a80e899ff33344f04386612f019bbb02631279688256a8d55e5e` |

原 `Demo.prepare_agate(transport=原SDK运输审计包装)` 执行的是未改六阶段入口，qualification预算仍为6jobs/1200000ms/零模型；业务长预算已在Demo初始配置中显式冻结。审计包装只核原POST内容并记录原ACK，没有替换响应或原 evaluator 结果。

唯一run_id为 `82ec4cd30c784f4ca6b415c718656fff`，operation/idempotency key为 **`82ec4cd30c784f4ca6b415c718656fff_qualify_probe`**。原reserve/authorize准入后向 `/v1/jobs/dev` 发出1次POST：L20D、1GPU、timeout1194s、recycle=true、replay_policy=never、dev_intent=custom_harness。新Task、原Q、实际上传的 `_stage_ro/_worker/_remote` 和全部原判据字节再次核验；实际Q与无准入预览SHA相同，SDK运输身份 `post_retries=0`。

原207-byte `acceptance.raw` SHA为 **`d8249a8c12a54200f06e07526712b86ce49ce6c02ef6cda14249d061211d24ea`**，原status文件为HTTP400。原回应完整如下：

```json
{"error_class":"validation","reason":"dev_sensitive_target_denied","message":"dev payload references a protected platform target","details":{"rule_ids":["process_root_escape"]},"trace_id":"req-fdc97b290227"}
```

这是平台对实际上传payload的明确拒绝，回应未给job_id。没有远程权限proof、torch/CUDA environment capture、36输入capture、baseline72+2、80批timing或任何负例结果；没有尝试编码/变换proc路径、删除canary、换profile、换parent、重投或退回普通allclose。具体平台matcher实现不在本项证据范围，不能把该标签解释成已发生远程逃逸或已执行kernel。

## 原拒绝结算、导出与下游边界

原 `AgateDevExecutor.query` 对任何缺job_id ACK都返回unknown，没有针对本种明确HTTP400拒绝的内置分支；原Demo因此保留 `blocked/settle_only`，parent当时原operation有fence且active继续计时。这是host拒绝恢复的有限缺口，不能将该状态直接冒称已取得worker终态。

同parent原B14 `diagnostic_owner.py:42–44` 已用“明确永久4xx validation/auth拒绝且无job_id”作为提交前拒绝，记录jobs0。本项按同规则严格匹配本次原HTTP400、validation、reason、rule_id、trace_id及原key，再以固定公开 `OwnerAccount.absorb(..., terminated=True)`、`RunContext.record_terminal`、`OwnerAccount.end` 只结算这个原operation。收据标记 **`rejected_before_job`**，引用原response/status/request/intent/submission-start；不是伪native job terminal。未调用网络、未改owner JSON、未调用release_unstarted、未修改原qualification/demo state，未清除旧失败记录。原worker cleanup在此没有可查询的job，不能虚构 `cleanup_pending=false` 服务证据。

原拒绝证据Ref为 `artifacts/0a849d8905c54bee8ad23e795dae9da7/gateway-rejection-evidence.json`，SHA **`f4a5075b4cce8aec36e59ae82571efc28075a31b0ff9c954ffaba64eec0b1284`**。唯一sequence1 receipt实际为calls0/evaluator_jobs0/tokens0/gpu_ms0，明确基于提交前拒绝；本轮没有native GPU时长或货币账单。usage Ref为 `artifacts/c891497921684fd5ab570350a08e2975/usage.json`，SHA `712f9eef7381c461b49723938f449adee304fc041cbcd6bb324d0ffc337fe75c`。

[rejection-selected-export](/tmp/schema-b27-inherited-gpu-qualification-20261121-final/rejection-selected-export) 经公开 `import_bundle` 导出112文件/1495024bytes：原Q及完整公开Task/Rules/Materials、原拒绝、status、request/intent/key及usage，逐Ref/SHA核验，未包含认证材料或运行时authority。初次本项导出检查发现任意byte evidence内nested Ref不自动遍历；随后通过公开 `import_bundle` 显式发布原immutable staging rawrefs，再将它们全部列作export roots。没有改原receipt、原字节、Q/state或owner；较早不完整导出不作为最终交接。完整消费入口是 `rejection-selected-export.json` 的roots/subjects，不能只从一个任意byte文档推定隐含闭包。

| 父账结算事实 | 值 |
| --- | --- |
| operations | 9个全终，原8个operation/request逐字不变 |
| 累计 / 剩余 model_calls | 4 / 20 |
| 累计 / 剩余 evaluator_jobs | 3 / 13 |
| 累计 / 剩余 active_ms | 1039144 / 9760856 |
| 本请求精确active | 94502267911ns / 94502ms |
| 剩余roles | planner5 / coder6 / reviewer15 / extract0 |
| active_run / active_since / fences | null / null / [] |
| 原限额与descriptor | 24models / 16jobs / 10800000ms、roles6/8/16/0，全部不变 |
| 原账tokens | unknown；370133仅数学下界 |

结算后snapshot SHA为 `aa186482fc30c8c1dd6183ec9f4d690180406622f32825eecd194f553770566e`，独立审计时仍相同。全会话真实模型累计仍11、已创建GPU jobs仍3，已知CLI费用部分仍 `3.897061750000000010 USD`；历史unknown未被覆盖，整体tokens与费用仍unknown。

本轮没有六作业资格、没有新业务Schema或Code、没有fidelity/performance/AKA配对资格。父账已结算不等于原资格通过。原 `Demo.prepare_agate()` 不应再对这个scope调用：它仍保留缺locator的原状态且没有HTTP400恢复分支；后续若修复，应有明确的新源码/新请求安排，保留本次拒绝与结算原件，不能换key重投本payload。当前实质阻断是平台拒绝完整必需权限探针，需平台明确支持所需边界或提供经批准的等价执行能力；本项没有绕过该控制，也没有降低原Task/权限判据。业务writing可用性的选择由根独立安排，不能据此执行依赖GPU资格的完整链。
