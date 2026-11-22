# R3：补缺与精简的独立裁决

日期：2026-11-20。只裁决；本轮唯一新增文件为本文。正式计划、实现和原 Schema 均未修改。
D1 真实独立 implement＋AKA 增强、D2 fixed 先／尽早 planner demo／提前复杂预审、D3 固定实现阶段交 planner，全部保持。
定位简称：P=`../04_implementation_plan.md`，W=`../05_work_packages.md`，C=`../planning/r5_contract_execution_review.md`，V=`../planning/r5_validation_delivery_plan.md`；定位均指审查前快照。
证据分类：**确定缺项/冲突**指文本或现有源码可直接核对；**规范选择**指须定一种语义；**未实测**指不能由阅读替代技术资格。没有把缺少运行证据当作方案被证伪。
以下 `R3-*` 是最终修改报告应逐项勾连的稳定裁决标识；K1–K8、A1–A7、X1–X2 是本文确定的交付单元，共 17 个，旧 ID 映射见后表。

结论：R1 的主要补缺成立，但 C-05、C-06/E-03 主要是规范未收口，不能写成已发生的实现 bug；C-04 是有私有依赖时的条件冲突。
R2 的删除方向成立；调整运行恢复、闭包证据、外部预算与 pilot 选择的具体语义，避免精简后损失真实能力。拒绝“只准同进程”及“换选择规则即可得到独立无偏测量”的读法。
最终不用新增权威方案文件，不新增服务、不增加工作包来对应每个审计问题，也不要求所有包等待 GPU。

## 一、R1 逐项裁决

### R3-C01 — C-01：接受，首次 result 前的恢复缺口必须补

- **依据：** P §3 与 C §3 只给 `implement --continue …/result.json`；C §6 又允许资格先持久、result 后重建，入口覆盖确实不完整。
- **最终语义：** design、optimize、implement 都接受 `--continue <run目录>`，CLI 核对其 operation；Python API 使用同一 run 定位与验证规则。extract 仍用原恢复协议。
- 首次收费前持久化本地锚点：run 身份、原请求/Q、后端/profile 身份、预算 owner、原生运行定位。尚无 result 也能找到已发生的操作，结算并重建投影。
- 写作至少保证结算／重建发布结果；新写作调用仅在请求未完成、有明确下一行动、原操作已结算且原额度允许时准入。completed/keep 无行动不因 continue 再收费。
- **落点/检验：** C 入口与持久边界；W 的 K2/K4/K5/A5；V08/V09 给三个入口各一条“已有收费事实、首份 result 尚未生成”的恢复轨迹，外部 AKA 另按 R3-S02 核定位。

### R3-C02 — C-02：接受，缩窄去重承诺而非建全局服务

- **依据：** 相同 Q 在新输出根内无法得知另一运行根的费用；C §6 的“新输出目录不归零”若跨任意 run 理解，与私有状态设计不相容。
- **最终语义：** Q 标识固定输入；首次启动分配 durable run_id，同 run 恢复沿用。显式新 run 可执行相同 Q，产生独立真实费用；request_id 不充当全局去重键。
- `--request … --out …` 只创建新运行，已有运行根拒绝并指向 continue；继续原 run 才有原账目和恢复保证。不存在跨任意目录的自动去重承诺。
- 子请求即使使用新 out、新 Q、新 R，也必须沿用外壳 owner 的累计账户；“新 run 可新计费”不能成为同一父流程重置余额的漏洞。producer/operation 身份包含实际 run 命名空间。
- 同操作收据重放零新增费用；显式 continue 可在结算后按原预算准入新操作。丢失/损坏 owner 的旧 run 不得作为恢复重新发额度。
- **落点/检验：** C 身份/预算；K2/K4/A5/X2；V08 并列同 run 重放、明确新 run、父账户跨 R 不重置三轨迹，不增注册中心。

### R3-C03 — C-03：接受，修正直接文字冲突

- **依据：** P §5 将全部 unknown fence，C §6 却允许已结算 fidelity unknown 补证；两条不能同时作准入规则。
- **最终语义：** 已准入操作身份/终态或计费结算未闭合、现有限额的可用余额无法确定时触发未结算 fence；此时 blocked/settle_only，不能新提作业、换通道、换 R 绕过。已终止操作缺少某项 token/GPU 统计仍记 null，不单凭 null 推断还在途；相关限额无法核定余额时仍须 fence。
- 已终止且用量结算的检查 unknown 仍不 qualified；有具体补证动作和余额时，可同 R 继续或显式 verify_only。unknown 本身不是设计阻碍，也不触发 planner。
- pause/耗尽的 fence 保留；尚有未结算操作时，即使历史 C 合格也不得用 completed 掩盖恢复责任，产物仍可返回。
- **落点/检验：** C 唯一状态表，P 只链接其规则；K4/A5；V07 保留未知 job 零重提，V02/V08 增已结算 unknown 可补证的相反轨迹。

### R3-C04 — C-04：接受条件冲突，采用 R3-S01 的选定闭包

- **依据：** P §4 强制完整 artifacts，V §7 允许公开脱敏；AKA `contracts/evaluation.py:60–67` 明确测量产物有 host-private 边界。冲突发生在不可公开原件被强制带入时。
- **最终语义：** 删除无关历史的强制复制；被选根实际依赖的原件仍必须齐全。公开说明包、可导入产物包、私有完整归档不再冒充同一对象。
- 脱敏 L′ 是新字节，不能替 L 满足旧 Ref；若选中证据确实依赖 L，则不能声称该证据原闭包公开可导入。仅复用不依赖 L 的 Schema/C 不受牵连。
- **落点/检验：** C 发布/导入，V 公开交付；K2/K6/A2；缺实际依赖拒绝、缺无关私有日志成功，均纳入现有 V06/V13。无需先停掉纯本地 fixed 开发。

### R3-C05 — C-05：调整后接受，使用既有 Materials 承载目标

- **依据：** DESIGN_SCHEMA.md/OPTIMIZE_SCHEMA.md 要求当轮目标与 workload，核心请求又拒绝未知字段；目前没有唯一的授权承载位置。这是规范空白，不是 demo 已证明不能运行。
- **最终语义：** design/optimize 的 Materials 必须有唯一 `id=workflow_objective` 条目，file 使用现有 Ref，至少对 planner/reviewer 可见；内容写当轮目标、优先 workload 和计量范围，允许简单写“按 Task 默认目标”。
- 该保留 ID 是当前任务指令而非普通参考，纳入 materials→Q；没有新目标 DSL 或额外核心字段。后续 implement 若需该说明，通过其自身 allowlist 显式投递。
- 目标不能改 Contract、scope 或硬判据；`performance_goal` 仍由 Task 的固定接受口径计算。改当轮目标生成新 Q，不必改 Task 或增加 design_revision。
- **落点/检验：** C Materials/写作请求；K2/K5/K6/X2；模板验证目标唯一、缺失拒绝、目标变化改变 Q，且不能以目标放宽 Task。不要新增独立测试包。

### R3-FID — C-06 与 E-03：合并，接受缺环判断，采用 S-08 的固定方法

- **依据：** W KS-06 只交 R0 条目，V §3 又允许 Rd/R′ 改 work ownership；当前 V02/V09 没有证明判据会随实质设计正确解释。未来实现不一定硬编码 R0，不能称为已发生故障。
- **最终语义：** Rules 冻结一个有限方法及工具/prompt 版本；方法读取当前完整 Schema 和 scope，逐项解释适用的数值路径、work ownership、读写/寿命、Require/Forbid、覆盖与必要设计决定。
- 报告必须列实际设计义务、Schema 定位、代码位置和理由；笼统的三个 pass 标签不足。方法不足以处理的新义务为 unknown，不让 planner/coder 删义务豁免，也不增清单 planner。
- R0 的“一 program 一行”只是该设计的义务，不是所有新 R 的固定验收常量。读完整来源以解释当前 scope，不要求有限 scope 实现所有未选节点。
- 下一固定请求绑定当前 Schema/scope 和同一方法；新 R 重判 fidelity，数值报告只能按原规则单独核适用性。单换收据 hash 不算重判。
- **落点/检验：** C fidelity，V §3/V02/V09；K4 持有方法，K5/K6 交接，A5/A6 消费。用同一多行 program 的 C 对 R0 fail、对合法 Rd pass 的 synthetic 配对，并保留新增义务无法解释→unknown；不强迫真实 optimize revise。

### R3-E01 — E-01：接受，原 out storage/view 是正确性硬项

- **依据：** `out.set_(tmp); return out` 可保留对象身份而替换 storage；V §2 只明确保护 x storage。EXP `eval/source_hooks.py:68–79,118–149` 已有入场签名和原 backing 守卫，可借鉴而不能据此替新 adapter 发证。
- **最终语义：** 保存 out 入场对象、原 storage 活引用、offset、shape、stride、dtype/device及 backing 边界；调用后核对签名，并从原输出区域核对写满/数值与外围不变，不能只跟随调用后的 out。
- x 保持原来的只读/身份义务；维持合法 storage_offset 支持。现有 72 次中为每 shape 至少一个实例明确设置 x offset=3、out offset=5，保存布局/输入摘要，无须扩大 shape/输入值家族。
- 新增“同对象换 storage”的坏候选，保留漏 out、改 x、超差；它应在两个实际 adapter 上拒绝。有限非零 offset 样本不宣称穷尽任意布局或设备内存安全。
- **落点/检验：** V Task/V01，K1/A2；C 只引用统一 Task 规范，避免再抄一份不完整的 ABI。源码事实已核，GPU 反例本轮未跑。

### R3-E02 — E-02：接受，V0 bootstrap 与 coder seed 必须分离

- **依据：** AKA `light/campaign.py:182–203,258–280` 测所选 initial；`orchestration/campaign.py:1109–1118` 拒绝失败 V0；`main_adapter.py:30–42` 先 setup_baseline 再 V1。never 不会跳过这次 V0。
- **最终语义：** native InitialSource/V0 使用明确的、可测的 Task.baseline_source（首题为 FP32 torch.softmax＋copy 的 baseline）；starter/显式 partial seed 是首次 coder draft 的 C0，不能被拿去冒充合格 baseline。
- 在首次 coder 准入前安装并核对 C0；后续 e 用准确终态 C 和当前 HEAD 的准备关联，G 只作 native 比较基准。V0/C0/G/HEAD 分列，不承诺它们相等；V0 pass 不等于当前 R qualified。
- baseline 及私有 evaluator 不因 bootstrap 绕过资料 allowlist；固定 assignment 不因原生“优化 baseline”正文覆盖。恢复不能重新装 seed 覆盖 draft，也不能偷偷从 G 起步。
- V0/V1/helper、原生预取 reviewer 和 production 检查的真实费用都计入 owner；baseline 在当前 native/profile 是否可用仍须资格，不伪造 pass，不降低 ABBA/框架约束。
- **落点/检验：** V 原生角色映射/V11；A2 定映射，A4 闭合 bootstrap+seed，A5 记账。真正不可执行骨架、非空 partial seed 两例证明 V0 来源与首调 C0 各自准确，中断不重做已结算 V0；KS 独立链不等待此门。

### R3-E04 — E-04：调整后接受，删除最快观测选终态

- **依据：** 前轮有限枚举可直接核算：相同真实 10 μs、观测 9/11，各候选数 1/2/3 的最小值期望为 10/9.5/9.25。共同预算不保证相同测量机会；这是数学反例，不是 GPU 噪声实测。
- **最终语义：** 采用 R3-S09 的末次已登记合格候选规则，拒绝默认最早合格 C，否则 D 的后续真实 R′ 容易永远不进入终态。
- 不为首期增加独立复测流程；保留正常 Task 计时，并将其作为本次自适应运行所交付候选的描述性观测，不能称独立或无偏的终态性能估计。
- **落点/检验：** V pilot/X2；CPU 候选轨迹证明终态选择不读取耗时数值，且能选择已合格新 R、正确标注未完成新 R 后的历史交付。此项不阻塞 KS/AKA fixed。

## 二、R2 逐项裁决及删除边界

### R3-S01 — S-01：调整后接受，闭包按对象语义而非目录历史

- 导入根为调用者明确选择的 Schema、source、证据；递归保留已支持对象的 Ref、source.files 实体、Schema.parent/change_report、raw 的实际解析来源及相应字节。不能仅做通用 JSON 字典扫描而漏掉源码/原生协议依赖。
- 删“完整 artifacts 子树”；保留 publication_id 稳定前缀、原路径/字节/摘要、逐 Ref 核验、暂存后发布、同路径异字节/模式整次拒绝。原生报告按原协议解释，不把普通文本路径都变成强制 Ref。
- **补足摘要主体边界：** 仅导入 C 不必带其历史 Q；但声称可复核某份证据时，须同时选入其 Q/Task/R/C/方法等主体的可定位原件。subject 里的裸摘要不是这些文件已随包存在的证明。
- 当前 request 已提供的主体可复用；历史证据缺主体时补选相应不可变 manifest/request 快照，不能猜路径或重贴新 Q。若快照真实依赖私有材料，完整证据仍不能公开；不把这条限制传播给无关 C 的导入。
- K2 的导入器、A2 映射与 K6 公开说明共同兑现；V06/V09/V13 一组“schema＋parent/raw、source、选中证据”正例与真依赖缺失/无关历史缺失反例即可，不新建导出协议或服务。

### R3-S02 — S-02：调整后接受，单一用户定位不等于删除 backend 定位

- 用户与公共 API 统一用 run 目录恢复；删除公共核心 `continuation` 字段及 result 路径/token 作为用户恢复门票的规则。保留 `recovery` 动作提示；v1 尚未发行，不造兼容别名。
- **不能误删：** backend 内部可继续传 opaque token/原生运行句柄，宿主锚点持久保存。外部已安装 AKA 的真实 native run 路径、版本、操作键与必要句柄必须可定位，不能只写 backend 名称。
- 外部调用须在收费前建立可恢复定位：宿主先固定运行/操作键，adapter 先持久其 native run 关联后才执行；即使首个返回/token 丢失，也能按该关联取回原事实。只在最终 result 才返回不可重取句柄的接法不合格。
- 这是本地安装后端及合格本地文件系统内的恢复，不承诺迁机、移动私有 runtime 后继续、分布式全局去重；可移动的公开 artifacts 与可恢复的私有 run 必须区分。
- K2/A5 负责锚点/边界，K4/K5 实现三操作恢复。V08/V09 增外部 native root 不在公开 artifacts 中且 result 前中断的例子；仍必须找回原 operation，零重提。

### R3-S03 — S-03：接受，删除三套脚本与配置副本

- 删除拟议 `schema_{design,optimize,implement}/tools/run.py` 和目录专属配置模板。保留统一 CLI、同实现 Python API、原 extract 命令及透传。
- 三能力目录可各留简短 README，链接角色方法、唯一命令和同组 examples；不复制参数、安装和恢复说明。四项能力由真实入口/证据兑现。
- P 目录表、C 入口、W K2/K6、V 发行文件表一起更新；V13 在任意 cwd 检查唯一入口，无需为未发行别名保兼容。

### R3-S04 — S-04：接受，就地收敛四份权威正文

- 删除 C 的第二份 DAG、目录/技术 spike 重复表和历史整合命令；删除 V 的第二份发行门、逐门 README 表和开工顺序。独有语义先迁往下述唯一归属，再删原段。
- 删除 P 的全量字段/JSON规则、Task 数值细则、状态表与重复 DAG，改为精准链接；W 是唯一交付 owner/依赖表，V 是唯一 V01–V15 判据表。
- C 中旧 `inputs/`、`sources/`、`evidence/` Ref 样例必须改为稳定 publication 前缀，或删冗余大块样例，只留一份内部一致的最小例。不能再用 P 的优先级覆盖错误示例。
- 两份 R5 文件改标题/导语为当前合同与验收附录，不再同时声称“仅供最终整合”。原推导在 before/与历史链可查；不增加第三份最终规范。
- W K2/K6 负责未来入口/资源义务，第四轮只做文档迁移核对；每一条必需语义应有一个权威落点，不按整节标题盲删。

### R3-S05 — S-05：调整后接受，17 个交付单元取代 34 包

- 接受 KS-05/06、KS-09/14、AKA-05/06、AKA-07/08/09、MIG-04/05 的行为合并，见第三节映射；不把内部 gate 变成新包。
- 不保留 high 草案中的独立 M1“fork 补丁”交付包：必要迁入与来源映射归 A3；驱动、seed、资格接点的实际代码仍各由 A1/A4/A5 持有，不额外造一层总前置。
- MIG-01 是实施前各仓基线检查；MIG-03 拆为 A3 安装接口与 A7 固定发行配对；MIG-06 删除为工程包，仅保留维护原则。
- 每单元明确材料/接口、CPU 集成、真实资格的子 gate。只消费接口的工作不能等该包全部完成；X2 预登记也不能等自己的运行结束才供 X1 消费。
- W 重写为唯一依赖表；第四轮静态核对不存在合并后回边，尤其 K1↔K2、A3↔A4/A5、X1↔X2 不得由“整包完成”产生互等。

### R3-S06 — S-06：调整后接受，复用证据但保留独立发行边界

- 删除独立 KS-13：对接 fixture/语义映射归 A2，成对证据索引归 A7 的配对子 gate。AKA-P1 只依赖 A7.fixed，不等待 KS 的真实资格或 A7.pair。
- 同一后端精确 Task/R/scope/C、evaluator/环境、profile/adapter/审阅方法及适用性都一致时，一条既有记录可支持多个声明；变动只补受影响路径，不固定再跑一整轮。
- 两后端分别取得真实证据；KS pass 不代签 AKA。原三关键坏候选外，加 E-01 的同对象换 storage，在实际 adapter 上拒绝；不能沿用“仅三例”漏掉新硬项。
- C 保留证据适用性规则，V 保留测试内容，W A2/A7 写 owner；配对索引仅说明复用证据和差异，不另建资格项目。

### R3-S07 — S-07：调整后接受，单 owner 可跨进程调用外部 AKA

- 接受一个顶层 owner、串行子请求、单个活动时钟；删每条 usage 行强制 `active_ms`，公共时间由 owner 转换记录汇总。任意深度账户树/通用分布式区间合并不在首期。
- **拒绝同进程限制：** KS 可同步调外部 AKA，AKA 也可调 KS 单次 planner。外部子调用前 owner 持久预约受限额度；子后端按这份授权及同一 pause/fence 做内部准入，真实操作收据回归原 owner，预约不是新余额。
- 子后端可持久授权内消费/原生状态，但不能另复制一个可自由加额的父账本。所有新收费接点受父／请求／native 最紧上限，未知在途额度不释放；恢复原预约，不重复授权。
- 唯一 owner 不意味着 native 内部永无并发：当前 `orchestration/campaign.py:1122–1149` 可并行 V0 评测与 reviewer 预取。它们均须预约/记账，owner 一直活动，直到全部在途操作静止；不需为此实现一般区间并集引擎。
- 父等待与子调用不重复加时间；pause 后在途仍计时，确认静止才停；崩溃间隔不明保守计入。父汇总不叠加 child totals/native totals，同收据只吸收一次。
- 请求 `usage.scope=request_cumulative` 保持：其 active_ms 从 owner 的当前 Q 活动段投影，不能填整个父账户的历史时间；父总时间直接由同一时钟累计，不再相加子投影。
- **落点/检验：** C 预算，K4/A5，V08/V11；一条外部子调用＋native 准备/review＋中断恢复轨迹核总调用/费用/时间与 fence。额度/暂停无法贯穿的外部 profile 暂不合格，不删除外部后端方向。

### R3-S08 — S-08：接受，以 R3-FID 为唯一细则

- 固定方法读取当前 R，不新增清单生成链、审批包或 planner 自选合格标准；R0 专属清单保留作 fixture。C 方法条目、V 正反例、W K4/K5/A5 交接同时改。
- “有限”不等于只查几个名称；新增适用设计义务必须列出，无法判断则 unknown。真实 keep 可正常发布，不制造一次假 revise 来交验收。

### R3-S09 — S-09：调整后接受，预登记末次候选而非最快值

- 终态选择按运行内首次登记的 `(所属 Schema, C)` 候选顺序，取顺序最大的合格者；B 按 C 首次登记。登记序号在候选冻结时固定，重复计时/迟到收据/重放不重新编号，不读取速度选回退项。
- 合格指预算内完成 Task、本次 scope 和 F/D 所属 R 的必要检查及规定测量；B 没有 Schema fidelity 义务。新 R 同字节 C 是新的所属关系，须取得适用资格，不自动继承。
- 新 R′ 合格可成为终态；R′ 未完成则可以展示历史合格 `(R,C)`，同时报告当前请求 partial/blocked 与 latest/qualified/G。历史交付不冒充当前 R qualified，也不遮盖 unknown 操作。
- 删除 best-of-budget、额外 P/C 实验臂及预排 F1 第二批九次；不转成必须关闭的 TODO。保留首批 B/F/D 全样本、9-run 建议及资格后预登记预算，不自动扩大运行授权。
- 选择器不按计时值选优，仍可能受完成机会、反馈驱动搜索和噪声影响。结论仅描述已登记策略的交付/成本/观测性能；不声称独立测量或消除全部自适应偏差。
- V pilot、W X2 修改；用“旧 R 合格→新 R 合格”“新 R 失败→旧 R 合格”“旧收据晚到”三短轨迹核选择，不新增 GPU 流程。

### R3-S10 — S-10：接受，合报告不合判定

- 一份 fidelity 报告同时含 coverage_id→节点/变体/代码位置、task_uncovered、设计义务结果；assessment 的 coverage/fidelity 可引用同一 Ref，但两种状态各自保留。
- 调用前仍核 scope/when/必需 dispatch；coverage_id 仍是 Task 输入分区而非测试编号。正确性 pass 不能代替 scope 覆盖或 fidelity，缺 dispatch 与设计违规可分别诊断。
- C 删除独立 coverage 文件的强制义务并定义合并报告；K4/A5 投影；V02/V06/V15 复用既有例核对各自状态，无需通用报告引擎。

### R3-S11 — S-11：调整后接受，删除无对象工程承诺，F1 不后置成愿景

- 直接删 wheel 工作项及离树 wheel gate，源码 checkout/editable 的任意 cwd、资源/许可/固定版本验收保留；删 MIG-06 包，维护原则留 P。
- KS-11 的完整源树扩展并入 K8，只实现 F1 实际需要的格式；AKA-12 不保留抽象扩栈包，只有 K8 选定 AKA 执行时纳入该案例的 adapter 子交付。
- K7 预审与首题同时，K8 真实 F1 紧随小 demo，其工程不等 README 收尾；保留 CuTe/实际 SM103、完整 evaluator、host 冷启动/稳态与有限 fidelity 的真实门。
- F2 十配置与后续 F3 扩域明确后置，由 F1 结果及实际下一请求决定，不先承诺固定排程。任何实际选中请求的必要 dispatch/完整快照/删除语义均不能因此省掉。
- V §5 留 F1 精确输入与 root 不覆盖全任务的证据，F2/F3 仅保留范围限制和扩域原则；W K7/K8 保持明确 owner。历史 30/30、kernel-sum 或 H20/Triton 不能替新复杂资格。

### R3-J01 — 新增的一处收口：避免 change_report 与发布审阅形成哈希环

- **性质：** 条件性的发布顺序空白，不断言旧规范一定要求互指。C 定义 Schema.change_report Ref，同时 writing_review 绑定输出 Schema；最终合并报告时容易错误复用同一文件。
- **最小语义：** change_report 是冻结前的 diff/影响检查，可绑定输入与待发布 raw 内容；Schema 引用它。之后 writing_review 绑定已冻结 Schema/scope，作为结果证据，Schema 不再反向引用该终态审阅。
- C 发布顺序/K5/V09 用现有写作发布例核无自引用/哈希环；不加新服务、新包或新版本，也不要求同一结论生成两套完整审阅。

## 三、最后一轮应采用的权威布局与交付边界

### R3-DOC — 四份当前正文，各有唯一职责

| 文件 | 唯一权威内容 | 应删除/迁走的重复内容 |
| --- | --- | --- |
| 03_decision_questions.md | 已确认 D1–D3 与历史备选 | 无需改决定或再问用户 |
| P / 04 | 范围、仓库/循环 owner、阶段交接、依赖方向、发布门、维护原则 | 完整合同/状态/Task参数、第二份 DAG/技术 spike 清单 |
| W / 05 | 下表 17 单元、改动区域、交付、子 gate/依赖、V ID归属、首批顺序 | 旧 34 包和重复判据全文；旧 ID 映射留本报告/修改报告即可 |
| C / 既有合同附录 | API/请求/目标、身份/发布闭包、source、恢复/预算、fidelity/结果、证据适用性 | 历史修订表、重复 DAG/目录/开工指令；错误路径/token示例 |
| V / 既有验收附录 | row-softmax/FA4材料与判据、V01–V15、公开文件/证据要求、pilot | 旧工作包映射、重复发布门/README时序/开工顺序 |

P 明确分工并直链 C/V，不再让读者按覆盖优先级拼规范。C/V 双向链接只指各自内容；R4/research/before 是历史依据，不再兼作当前规则。
公开文件清单归 V，具体 owner 归 W；P 发布门的支持声明引用这两处。状态仍全部未开始，本文不使任何 CPU/模型/GPU资格通过。

### R3-WORK — 最终 17 个单元与完整旧 ID 去向

| 单元 | 明确交付边界 | 旧 ID 映射 |
| --- | --- | --- |
| K1 | 首题 Task/材料及唯一完整 evaluator、所选执行通道资格；先交材料 | KS-01＋KS-04 |
| K2 | 唯一包/CLI/API、合同/资源/codec/选定闭包/恢复锚点、extract 消费桥 | KS-02＋KS-08；KS-14 包资源建设部分 |
| K3 | Claude 单调用、角色投递与真实 profile 资格 | KS-03 |
| K4 | 固定参考实现、准确续写/预算恢复、有限审阅与 qualified 投影 | KS-05＋KS-06 |
| K5 | design/optimize 共享写作与当前 R 交接/发布 | KS-07 |
| K6 | 真实小 demo、源码公开发行、证据/README | KS-09＋KS-14 当前发行部分 |
| K7 | FA4 来源/范围/原栈及 F1 预审 | KS-10 |
| K8 | 真实 F1 与实际必要的 source 格式/栈/adapter | KS-12＋按需 KS-11；仅实际选择 AKA 时吸收 AKA-12 |
| A1 | 指定 episode、生命周期/预算接点 CPU spike 与 driver 结论 | AKA-01；相关接缝实现仍在本单元 |
| A2 | Task/source 映射、实际 Atrex hook 资格与共同对接 fixture | AKA-02＋KS-13 对接部分 |
| A3 | 可选安装/off、固定角色正文/资料、必要迁入补丁的来源归口 | AKA-03＋AKA-04＋MIG-02＋MIG-03 建设部分 |
| A4 | native V0/V1/helper 与 C0/跨 e seed 准备、准确来源 | AKA-05＋AKA-06 |
| A5 | 固定 backend 的资格/三audit、额度/恢复、完整公共返回 | AKA-07＋AKA-08＋AKA-09 |
| A6 | 按阶段交 planner、keep/新 R 准入的有限动态链 | AKA-10 |
| A7 | AKA 独立 fixed 源码发行；另汇合跨后端证据 | AKA-11＋MIG-03 发行部分＋KS-13 配对结论 |
| X1 | EXP 固定 vendor/KS 物化、新捕获、唯一产品来源与回退 | MIG-04＋MIG-05 |
| X2 | B/F/D 预登记/CPU配置；资格后有限运行与全样本分析 | EXP-01＋EXP-02 |

MIG-01 变成实施前基线检查记录，不是产品单元；MIG-06 变成“独立更新、固定新配对、保留旧 capture、先 off 后受影响能力”四条维护原则。
MIG-02 来源归口不意味着 A3 承包所有接点代码；A1/A4/A5 的改动各维护来源记录，A7 发行时汇合，不能反向让早期接口等待全部补丁完成。

### R3-GATE — 首批顺序与必要内部 gate

1. 实施前重读三仓状态并隔离；首批同时交 K1 材料、K7 F0/F1 预审、A1 CPU spike。没有真实通道不阻塞这些工作。
2. K2 消费 K1 的 Task/材料接口；K1 evaluator 随后消费 K2 合同接口，不依赖 K2 全部完成。K3、K5、A2 及 X2 预登记随后并进。
3. K4 的方法/结果接口可先定义，CPU 集成再需 K2/K3/K1 相应接口；真实 fixed 需实际模型/执行/审阅资格。K5 开发不等 K4 的 GPU，真实 optimize 才消费其真实反馈。
4. K6 资源/说明从 K2 即可做，KS-P1 才汇合 K1/K3/K4/K5 的适用真实证据与 K2 消费能力；不等待 A6、X1/X2 或 K8。K8 工程可在 K4接口与 K7 材料齐后启动，真实交付紧随小 demo。
5. A3 接口消费 A1 结论、A2/K2 接口和已审来源；A4 消费这些接口，A5 的预算/资格接口从 A1 同期做，CPU 集成再汇合 A4。没有“先 A5 全部完成才能开发 A4”的回边。
6. A7.fixed 汇合 A1–A5 适用工程/真实证据和独立安装配对；不等 EXP、K6 或 A7.pair。A7.pair 在两后端适用真实记录都有时关闭，仅控制跨后端声明。
7. A6 CPU 需 A5 固定工程及 K5 可调用接口，真实动态另需所用模型/固定资格；A7.fixed 不依赖 A6。X1 安装/捕获工程可在 A5 CPU/A3 固定源码接口后做，能力消费再等对应真实门。
8. X2.protocol 先供 X1 配置核对；X2.run 后依赖 X1消费、A7.fixed、A6真实动态及冻结R0/evaluator/profile。预登记与 launch 不互为整包前置，真实运行不由本裁决授权。

## 四、修改后只需重点核对这些反例与一致性

### R3-CHECK — 有限复核清单

- **恢复/费用：** 三新入口首 result 前崩溃；外部 AKA 定位仍在；同 run 重放、显式新 run、同父新 R 三种费用语义不同；未结算 fence 不被历史 qualified 遮盖。
- **unknown/预算：** 已结算 reviewer unknown 可有据补证，未知 job 不重提；外部子调用、V0及预取reviewer均受同 owner限额，暂停在途计时且无双计。
- **导入/发布：** 选定 Schema 的 parent/raw 真依赖齐全，缺无关私有历史仍能导入；证据主体可定位；脱敏件不替原件；同路径异字节拒绝；writing发布无哈希环。
- **来源/正确性：** V0 测正确 baseline，首调收到不可执行骨架/partial C0；恢复不覆盖draft、不退 G；原 out storage换绑及非零 offset在统一 Task/V01中明确。
- **fidelity/scope：** 同 C 对 R0与合法Rd得到应有相反结论，新增不明义务unknown；scope是输入分区，root的when不覆盖direct时拒绝；合报告不合两个判定。
- **pilot：** 合格新R可进入终态、未完成新R保留历史所属、迟到旧收据不抢终态；删除最快选择与额外实验臂后，结论仍标描述性观测和全样本失败。
- **文档/DAG：** C-01～06、E-01～04、S-01～11和J01均有修改映射；旧34包各有去向；C仅一套字段/恢复，V仅一套Task/V规则，P/W链接无失效ID或旧result/token入口。
- **不能误删：** D1–D3、独立真实资格、准确C/latest/qualified/G/HEAD、固定assignment/allowlist、三audit、off零KS依赖、源码公开闭包、F1真实门、旧capture冻结与必要dispatch均留有owner及验收落点。

核验记录：已读指定文件、两份当前 R5 和必要角色/案例文档；直接核对 AKA V0/准备、source安装/终态捕获、native reviewer预取、EXP BackingGuard/vendor及FA4 when/launcher边界。
五份正式决定/计划/附录与 before/及 baseline.json 的字节/摘要核对均未变。仅执行文件读取和该离线一致性检查；没有运行模型、GPU、服务、安装、资格测试、凭证、提交或远端操作，也未派子代理。
