# M02 试走疑问与阅读记录

## 阅读与阶段进度

已读 AGENTS.md、EXTRACT_SCHEMA.md、P1/P2阶段页、EXECUTOR_ALIGNMENT；DOC0公共约定、DOC1算法要求、template资料/创建/检查边界；为试写读P3逐段方法、DOC4工作分解/映射/分配/提交/调用边界、DOC5动作/pipe/index/waiting/跨工作/条件等待、REVIEW完整保留与来源缺项章节。starter仅查看占位，不读取外部硬件资料。

问题材料：input、reference、baseline solution、metadata、shape_range；未把10个样例当成全集，也没有用performance记录证明candidate机制。源码：Python入口与build flags、C binding/memo、sync主体、UMMA host selector和launch、pair producer/issuer/实际SAREG0 epilogue、wide主体与两种store。未逐行覆盖旧SAREG1、全部primitive编码或历史诊断。

P1事实稿完成、Contract未完成；P2已作机制与分支判断但未放行；只试写少量P3表达。没有工具强制“先运行才准判断”的新阻塞：EXTRACT_SCHEMA明确免GPU验证，EXECUTOR_ALIGNMENT明确当前CLI不能端到端运行新方法，P2:36明确不要求完整YAML检查。正式阶段P1批准前提被本次文字试走授权绕过，不把这种试走当成正常运行已批准。

以下是实际写作时产生的问题，不全是规范缺陷；保留已自查解决的状态。

## M02-01：任务是生成域，还是完整 upstream 签名域？

- 暂分类：输入不足／任务边界未理解；未解决完整P1，未阻断已知FP8路径机制。
- 规范：`schema_extract/phases/01_contract.md` 第2段：“生成参数、函数签名和候选实现限制不能单独决定任务要求”；DOC1 nullable/default要求分别落实对象缺席和参数省略。
- 输入位置与短片段：`problem/metadata.json:4` 描述“exact upstream API”“shape_range.json defines dynamic request ranges”；`problem/input.py:67–68`总传BF16 out和固定参数；`reference.py:9,26–30`签名允许out=None，但expected立刻读 `p['out'].shape`。`problem/solution.py:8–10`把全部可选参数传给未在当前小目录内展开的FlashInfer API。
- 当时理解/尝试：我先想在Contract按签名写out nullable=true/default=None，并按candidate分配结果；随后发现这用candidate补了reference没有的分支。另一做法是按本题构造域要求caller-owned out，记录签名更宽；但不能仅凭生成器就断言外部API域收窄。当前只写已确认传入out的事实。
- 影响：out/out_dtype的合法缺省及输出类型域不宜拍板；metadata提到“optional C alias”但签名没有C，也不能据此发明输入。它不改变眼下三种设备计算组织。
- 需要什么说明：明确这次共享Contract以本题shape/固定生成ABI为准，还是须覆盖FlashInfer上游全部可选参数；若后者，提供对应固定API的需求与参考分支。不需要先运行测试。

## M02-02：自适应搬运box是尾块执行，还是应从YAML移出的配置选择？

- 暂分类：文字边界／自身未理解；未解决，会影响pair/wide正文范围。
- 规范：`schema_docs/0.common.md:106–110`要求“其他取值、模式或组合写入dispatch”，同时“当前方案随输入产生的尺寸、索引、有效区域”留正文；例子允许min截取尾块，禁止三槽/四槽配置选择。
- 代码：`candidate/fp8gg_umma.cu:377–396`：“tightest A box for this tile's valid rows”；`boxh = vrows > 64 ? 128 : (vrows > 32 ? 64 : (vrows > 16 ? 32 : 16))`，mapA随之选择，expected bytes=`boxh*BK`；stage容量依旧固定。wide `1172–1184`相同。host `1682–1686`map创建失败则关闭abox。
- 当时理解/尝试：方案A在Producer动作describe写“当前tile按有效行选择最小覆盖box”，保留actual load byte和masked MMA行。这像尾块执行，不更换worker或缓冲。方案B固定boxh=128，另三个高度写目录；但一个调用中不同组尾tile会选择不同box，固定后YAML可能不再忠实描述实际自适应机制，或者只能把when限制为全128行tile。
- 影响：会决定root/非root是否只能取很窄子域、需要多少局部记录，以及能不能把这个输入相关表达式留在正文。我未用四份YAML掩盖疑问。
- 需要什么说明：判断的对象若是固定容量内的实际搬运区域/descriptor选择，是否属于当前边界执行？最好给这种离散向上取整传输box的正反例；不是请求增加字段。

## M02-03：单issuer→双issuer，局部闭合到哪里才够？

- 暂分类：分支判断／自身拿不准；倾向局部，但不声明已完成降级。
- 规范：DOC0差异表说能围绕基线已有值、step、role或存储说清必要联动且主体继续成立则局部；`0.common.md:122`又说“将主计算、主要领取与退出或整条主循环协作包成一个新对象，不构成局部闭合”。REVIEW“完整设计的保留与降级”要求解释基线加目录差额为何不足。
- 代码：pair `fp8gg_umma.cu:336–339,415–523`，dual时两个warp分别对kb0/kb1发MMA与commit，empty_bar计数为2；单时一个warp顺序发两块并commit一次。producer等待、pair stage、两个TMEM块、promotion均沿用。host `1828–1833`以pairs_ub/grid选择dual。
- 当时尝试的局部记录：基线PairDot拆成PairDot0/PairDot1，第二个由新MMA1 role执行；两者都等同一PairPayload与对应旧TMEM release；下一次覆盖stage和promotion都等两者done。KB为奇数时第二issuer即便无第二payload也仍需履行commit通知，不能丢事件。原分配与pair循环不改。
- 两种理解：这些差额有明确value/role/释放边界，因此局部；或者主循环从单issuer变成两个长期协作issuer，恰是警句不准打包的主协作变化，因此应独立完整候选。单纯“多一个role”或“只有几十行”都不是证据。
- 影响：目前三份拟完整代表可能还需一份；不应在没解决时把dual直接归参数，也不因超过某预想数量而改名。
- 需要什么说明：主协作组织与局部角色分工之间的尺度判例；可以直接对上述差额判断，不需要要求worker预先写完整两份YAML才知道。

## M02-04：硬件占位豁免下，CTA/warp和nSM怎样合法绑定？

- 暂分类：固定字段/跨章边界；手工推演继续，未把它当硬件缺项要求补资料。
- 规范：P1:39要求硬件placeholder原样、跳过搜索与检查、不填设备能力；DOC4:9/152要求执行单元来自hardware_specification.execution登记，设备规模来自绑定资料；DOC5核心思想也要求role单元来自execution。starter:14–19只有`execution: {Unit: null}`。
- 代码：sync `fp8gg.cu:340–344`使用`min(wub,2*nSM)`作grid；Python `_Engines`从device_properties取得multi_processor_count，worker和role实际是CTA/warp。UMMA的selector也读nSM。
- 当时理解/尝试：拟写`worker: {unit: CTA, unit_count: 1}`与含nSM的worker_count；不想把nSM伪造成Contract输入，也不想偷偷填硬件占位。文字可以说明来自host查询，但目前没有确定放在哪个正式定义位置，单靠豁免检查似乎不等于完成名称绑定。
- 影响：不影响3类机制判断，却会在正式work_schedule落字段时卡住。使用硬编码设备SM数会违反本次不读硬件资料且改变动态源码表达。
- 需要什么说明：固定实现提取豁免是否同时允许直接用源码执行单元名和host取得的设备规模符号；若允许，给定其定义/引用落点。这里不是请求真实设备资料。

## M02-05：缓存命中和失败回退，哪些保证可从本源码成立？

- 暂分类：来源限制／输入不足；不是已证明的kernel缺陷。
- 规范：P2:17要求沿缓存/拒绝前缀核对实际提交与副作用，host顺序不证明设备完成；DOC4调用边界与common局部记录要求不补造等待/隔离保证。
- 代码：`fp8gg_bind.cpp:13–23`注释称tensor data_ptr/shape/dtype/strides构造后不可变；try_launch6只比六个活weakref对象身份即复用指针/维度。`fp8gg_umma.cu:1542–1547`sched cache只比indptr地址与shape；`1723–1759` graph launch失败会继续direct，direct失败又可能engine2→1。成功API提交后没有host等待输出。
- 当时理解/尝试：我能确定正常成功路径graph与direct都提交一个主kernel，构图本身不是额外计算；不能从weakref身份推出metadata/storage不可变，也不能只凭返回错误推出先前未提交任何工作。若写“固定allocation/metadata期间复用memo；命中仍读当前payload内容”，前半是必要调用前提还是修复源码保证，目前没证据完全定性。
- 影响：暂将host缓存作为局部/实现记录，但这不足以批准跨调用生命周期或所有错误路径。不能为每条缓存路径复制三份完整设计，也不能为了省事说全是无关ABI细节。
- 需要什么说明：本任务是否保证同对象跨调用metadata/storage固定、并发调用或stream切换有什么域约束；异常提交语义若要形成保证，需要对应API资料。按当前授权不追硬件资料、不运行；保留受影响调用范围，不猜测修复。

## M02-06：异步store的源读取释放与最终写回完成如何表达？（已自查解决）

- 暂分类：自身未理解，现有文档足以解决，无新增规范请求。
- 代码：pair `1015–1039,1055`和wide `1320–1346,1391`分别用`bulk_wait_group_read0`保护staging复用、`bulk_wait_group0`在退出时排空写回。
- 初次尝试：下一tile Pack等待前一Store.done。这样安全但抹掉源码在global写入仍在途时已可覆写staging的重叠；只等Store.issue又不保证安全。
- 自查依据：DOC5“事件引用”`588–609`附近给出store:O和serial release:OSlot例子，明确“全局写入此时仍可在途”；动作表也区分release与store的done。
- 现在写法：独立定义源槽release完成保证，Pack等release，最终退出等store完成；同步原语放实现交接。尚未把full/tail交替的条件事件展开完，不把解决该概念当成完整pipe通过。
- 影响：不需要为read-wait/full-wait拆完整设计。这个例子帮助我落实了实际写法。
