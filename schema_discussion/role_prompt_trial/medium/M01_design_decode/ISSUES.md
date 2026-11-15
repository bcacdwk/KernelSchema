# M01：实际阅读与卡点

## 阅读范围与推进位置

已读 AGENTS.md、DESIGN_SCHEMA.md、本次 `inputs/fa4_decode.md`；按问题查阅了 schema_docs/0.common.md 的参数/结构、节点与引用规则，DOC 3 的数学、符号、保留、精度和账本，DOC 4 的分解、mapping、分配、launch/handoff、数据块与事件，DOC 5 的 role/verb、执行模式、pipe、索引、storage、waiting 和事件引用，DOC 1 的默认浮点域及数学表达，DOC 2 的资料和占位规则，DOC 6 的实现交接开头。作者入口查了 schema_template/README.md 的资料、创建、检查与交付段落，以及 README 的角色/权限入口。

一次较大的并行 cat 输出被截断，随后按标题、sed 区间补读实际使用的规则；不把未见到的截断内容算作通读。未读旧 fa4_decode 的 Design/dispatch，未读其他试走报告，未读历史研究内容。文件名搜索曾列出旧文件名，没有打开这些文件。

推进到：选定在线 softmax 主线，比较并写出 split-KV 数学和交接候选；完成一个最小 work_schedule 文本及单角色 serial pipe。未完成全量 symbol/storage/realization，未绑定硬件原语。没有运行 Python、extract、checker、GPU、构建或测试。以下问题既含实际缺输入，也含阅读后已解开的疑问；不是规范缺陷清单。

## M01-01：硬件占位下，能把“方案已选”推进到哪一层

- 分类：输入不足；推进边界最初拿不准。
- 具体选择：Bk=64、8 warps/CTA、单缓冲、FP32 计算；是否改 Tensor Core、双缓冲、多少 CTA 驻留。
- 文档位置：DESIGN_SCHEMA.md 开头“当前路线已选、计算关系与所需能力有依据时……继续向下推演”；DOC 2“省略与占位”又说“在依赖该入口作出设计或实现判断之前，应补齐对应资料”；输入 hardware/wiki 全是 pending。
- 初读不明点：A 是所有硬件入口未齐都只能停在数学；B 是可先选候选并估算需求，只不能宣称资源/原语可行。
- 尝试理解：回读 DESIGN 的“不依赖它的工作”和模板 README 的缺项规则，采用 B。Contract 的 CTA/SMEM/warp 定义足以写抽象协作边界，但没有 SM 数、SMEM 上限、寄存器容量、FP8 MMA 形状或 CuTe API 支持。没有从设备名称猜出这些事实。
- 临时处理及影响：把 64/8/单槽当作真实初稿选值，计算 36 KiB SMEM 与约 40 KiB 寄存器逻辑值需求；不称配置可驻留。继续写同步 pipe，但停止完整 realization 与异步设计。这是资料不足，不认为需要增加固定字段。

## M01-02：split-KV 能否依赖 workspace，是否可以自行分配

- 分类：输入/接口条件不足。
- 具体选择：跨 kernel 的 Mpart/Zpart/Upart 全局物化；预估容量 `BatchCount*q_len_per_req*Smax*16,512` 字节。
- 文档位置：输入 `workspace_buffer` 仅写“Scratch zeroed before first isolated use”，shape 是 WorkspaceBytes，未给容量下界；计时边界“includes any partial-result combination”。DOC 4“存续与初始化”说初始化“省略不表示清零”，DOC 0“选择条件与索引”允许 when 写必要调用条件，同时提醒范围缺口不能解除完整任务义务。
- 两种可能：A 传入 workspace 是这次实现唯一可用 scratch，因而 split 只能承担容量足够的子域；B 实现可以自行申请辅助张量，但是否包含分配成本及可接受的额外分配仍未说明。
- 尝试理解：公式先推完整分段归并，再算实际字节需求。用全部写入的部分槽避免依赖首次清零；这个处理解决重入初态，却不能解决容量。
- 临时处理及影响：不自行选择 B，不给 Contract 增加下界。主线保持不使用 workspace；split 是值得独立 schema 的未完成候选，不能宣布覆盖所有输入。需要任务方明确 scratch 资源与计时口径才能完成它的调用条件。

## M01-03：缺少 batch/长度分布时，如何选短长 KV 的具体职责

- 分类：任务输入不足；当前固定结构并未声明 workload 字段。
- 具体选择：是否把少 batch、长 KV 交给 split，以及切换阈值；q_len=4 是不同节点还是当前工作量。
- 文档位置：模板 README 第 1 节说“还需从任务说明取得实际 workload 和优化目标；当前规范尚未确定它们的字段格式，不自行增加字段”。DOC 0“参数变化与结构变化”区分配置选择和由输入形成的工作量，并禁止把选路隐藏在 min/公式内。
- 初读疑问：只知道“少量 query、短/长 KV”，能否将 max_seq_len 大直接作为 split 选择依据；或者必须先有 BatchCount、长度偏斜和提交成本。
- 尝试理解：算了无 split 的工作数 `BatchCount*q_len_per_req`、split 的矩形工作数及额外流量。即使 max_seq_len 很大，足够多 batch 也可能已经充满设备；长度偏斜还会造成空段。因此不能仅凭“长”固定性能阈值。
- 临时处理及影响：选择可比较的 1024-token 段作为候选参数，不写假精确 when 阈值。q_len=1/4 在同一 partition 规则下只改变工作量。说明性能选择需要的材料即可，未自行增加 workload 固定字段，也未要求本试验补齐调查。

## M01-04：在线 softmax 的保留量与有限输入范围

- 分类：自身起初未完全理解，阅读后大部分解开；残余是数值验证不足。
- 具体字段/选择：m/z/u 的 `retention` 与 shape，score/E 的 FP32 数值路径；输入中没有显式有限 range 是否允许 NaN。
- 文档位置：DOC 3“符号属性”“值的保留”说 carry_forward 是逻辑携带，不决定全局位置；“数学表达与 shape”说圆括号递推索引不列入 shape。DOC 1“range、default 与空值”说“浮点输入默认取有限值”。DESIGN 第 2 节要求检查中间量，不能以 FP32 作为安全证明。
- 最初两种读法：把每页 m/z/u 写成带 page 轴的完整快照；或只保存当前递推状态。另一个疑问是 fp8 格式有 NaN 编码但输入未写 finite，能否进行有限值上界分析。
- 尝试理解与变化：读后采用当前状态的 shape，保留范围写 `(b,t); i=-1..Nb-1`，不额外物化页快照；按照默认域排除输入 NaN。随后手推 score、z、u 范围及首轮分支，避免把 -inf 初态无条件代入不合法表达式。
- 临时处理及影响：能继续工作调度；但“中间不溢出”不能保证原 atol=0.06、rtol=0.04，尤其长递推和 V 抵消。归约树与 exp 实现的具体误差仍未验证，不把这个缺口塞成新的状态字段。

## M01-05：单角色 pipe 是否必须再重复写 waiting

- 分类：表达疑问，查阅后已解开。
- 具体字段/选择：MAIN 的 serial load/compute/matmul 顺序；下一页覆盖 K/V 的 released 前驱。
- 文档位置：DOC 5“执行模式”规定 serial 的 done 在后续动作 issue 之前；“默认的依赖”说 role 顺序与 execution_mode 是默认依赖，并允许省略由其传递保证的重复前驱；“role 字段”明确单 CTA 可以只有一个 role，没有 waiting。
- 起初疑问：已知每页要等 K/V 就绪，并在读完后才能覆盖，是否仍强制列 prepared/released 两条边；还是所有动作 serial 时默认顺序已足够。
- 尝试理解：逐轮检查 `load KV -> Scores -> Weights -> WeightedValue -> UpdateU -> next load KV`；前一轮全部读取结束在下一轮覆盖之前，Output 在末轮更新之后。因此选择省略 waiting，而不是抄双角色例子添加虚构依赖。
- 临时处理及影响：可以写出最小 pipe。但 action 的 CTA 内完成/可见性保证必须由真实 CuTe 协作原语实现，不能把 serial 标签当作底层屏障的证据；该未绑定问题仍归 M01-01。没有观察到必须新增 verb、while 或 waiting 形式的需求。

## 工具与流程观察

没有遇到“必须先跑工具才准继续数学/调度”的实际阻塞。DESIGN 第 5 节及模板检查入口确实要求正式交付时检查，但同时允许中间稿、检查提前结束和继续独立核对；本次明确要求文字试走，所以按任务约束未执行。旧 starter 仍未迁移的说明使我选择直接按 DOC 3–5 的结构写片段，不复制旧骨架。当前没有由本次具体需求证明必须修改的固定字段。
