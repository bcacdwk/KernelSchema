# M05 问题与理解变化记录

这里只登记真实推演中遇到的阻力；不是断言每项都是 schema 缺陷。位置均指当前工作树材料。未运行工具检查，没有把“未能检查”写成语法错误。

## M05-01：优先 workload 和硬件相对代价不足，主路线只能保守选择

- 场景：比较驻留逐 token、chunk 三角、affine scan。块化缩短主链但 FP32快照约1GiB，能否换来收益取决于真实 FP32矩阵路径、带宽和工作规模。
- 位置：输入 Contract 的 hardware_specification.wiki 全部 pending；`DESIGN_SCHEMA.md` §1“从任务说明明确优化目标、优先 workload 和计量边界”；作者入口 §1“当前规范尚未确定它们的字段格式”。DOC2 §省略与占位说明 pending 不是已有文件。
- 最初读法/尝试：先把给定设备名和 Tensor Core 概念视为块化方向的充分动机；查 DOC2 和 math 精度规则后撤回。没有把 FP32 R/W/S 自动窄化为 BF16或改TF32。
- 影响：能手算流量与容量、选择R继续表达，不能声称R或B胜出。B3快照流量是实质反证风险，不能用链长收益盖过。
- 需要帮助：目标长度/B分布、性能计分方式；适用L20D/SM103/CuTe4.6.1的容量、FP32输入计算路径与相对成本资料。类别：输入不足，未认定文字流程错误。

## M05-02：binary64 scale 与默认FP32的结合如何具体交给Coder

- 场景：O=bf16(scale*sum State*q)，scale输入明确是Python binary64 0.08838834764831843；sum默认FP32。最终舍入可受 scale是否先转FP32影响。
- 位置：输入 `scale.dtype: fp64` 和“Mandatory Python output multiplier”；DOC1 §数学表达与参考精度“浮点计算与累加默认使用FP32”“窄化…必须…写出”；DOC3 §精度与辅助运算。
- 最初读法/尝试：先打算沿用Contract公式不改写；随后意识到输入dtype和计算精度分离，保留 `scale` 名字并不足以告诉实现者它在这里是否按FP32标量参与。显式写 fp32(scale) 又可能是在解读而非已知事实。
- 影响：R和B的最后输出转换仍不能称为完全具体数值路线。不自行把乘法改FP64、不改scale值或误差要求。
- 需要帮助：确认这里默认FP32是否已唯一确定Python标量的消费路径，或查给定固定oracle如何提升/转换该标量（本次仅输入转录未展开oracle代码）。类别：材料与语义衔接/自身未理解。没有理由因此停掉状态/工作推导。

## M05-03：int64-to-int32边界转换的工作归属不清

- 场景：R只需每CTA读两个cu_seqlens值；可安全在索引运算中用32位，因为T<=16384。但输入计时描述专门提到Model.forward及其int64转int32转换。
- 位置：输入 compute：“Timing includes Model.forward and its int64-to-int32 boundary conversion”；cu_seqlens为int64。DOC4 §调用边界与§工作分解；DESIGN_SCHEMA §3要求辅助工作也有归属。
- 最初读法/尝试：想让kernel直接读原int64，省额外转换数组；又不能确定用户固定了wrapper转换步骤，还是仅说明oracle实现的计时边界。若固定wrapper给内核int32，则正式输入仍int64，二者接口也须说明。
- 影响：DRAFT的主要调度只展开递推，boundary与scale入口尚未全部登记；不能悄悄把预处理排除计时，也不先新增重复转换launch。
- 需要帮助：确认可设计的forward边界、内核实际收到的boundary对象、转换是否强制存在。类别：输入不足/接口归属；不需要为此改固定字段。

## M05-04：递推状态、invariant、target和storage起初容易混为一谈，DOC3/4大体解开

- 场景：J行状态在一个worktile内常驻，但数值每token变化；O的target覆盖L_b个token，却不应分配L_b*J片上输出。
- 位置：DOC3 §值的保留“retention描述逻辑用途”；DOC4 §工作分解与§数据块描述；DOC5 §片上存储“register…成员内逻辑值总量”。
- 最初读法/尝试：初稿倾向把S写进worktile.invariant，因为它不换地址；查文档后修正：initial_state是invariant输入，S是math carry_forward，StateRows为实际storage，target是工作产出区域而不是驻留预算。
- 影响：误解已基本解开；仍想确认 work_schedule 不需额外登记一个“内部携带值”栏目，是否只靠math+storage就足够。当前没有自行添加state字段。
- 需要帮助：一个直接递推的完整小例子能验证这个跨区放置；现有条款已足以继续。类别：已缓解的自身理解问题，非固定字段变更要求。

## M05-05：变长chunk的独立工作坐标如何既具体又不把准备成本藏掉

- 场景：B3中PREP/READOUT按(b,h,c)独立，STATE里c是局部迭代。Nc_b随b变化，无法把同一矩形partition的chunk.size含糊写成Nc而不说明哪个b。
- 位置：DOC4 §step粒度“chunk先是独立工作维度…状态递推…顺序迭代”；§工作分解允许运行时size或上界空工作；§数据块描述“一个维度通常产生ceil_div(size,tile)个工作坐标”。
- 最初读法/尝试：想写partition batch/head/chunk，其中chunk.size=ceil_div(L_b,C)。随后不能确定依赖外层partition坐标的size是不是规范支持的ragged工作域。改想扁平Nc_total加(b,c)映射，或全局上界网格+无效worktile早退，两条都可继续，但实际准备/成本不同。
- 影响：未完成B3的partition/schedule，没有虚构新的ragged/lookup字段。数学和handoff仍可独立写出。
- 需要帮助：规范是否允许partition维度size依赖另一维坐标；若不允许，建议明确给一个展平变长域并计入映射来源的现行表达示例。类别：固定字段表达范围可能相关，尚未认定必须改字段。

## M05-06：具名动作describe能写多深，普通同步计算的完成保证在哪里绑定

- 场景：R的Advance要读五种输入槽、旧状态、完成DK归约、更新状态并写O。全部合成一个compute动作很长；拆成每个数学小步又没有独立发布或重叠价值。
- 位置：DOC5开头“动作采用设计级粒度”；§describe要求跨动作输入、存储绑定、原语保证；§默认依赖“唯一不写也成立…role内…顺序”；DOC6 §先做删除测试。
- 最初读法/尝试：担心Advance过于黑盒，把状态数学藏进字符串。查阅后沿用上游公式，describe只列输入版本/槽及done读完条件；用两个role和prepared/released把真正的交接显式化。不再按数学公式逐项拆动作。
- 影响：手工片段能检查环和覆盖。但Loader.done对其他warp的可见性目前是必须实现的保证，没有真实CuTe原语资料；不能把“serial”当作跨warp同步。缺少底层资料时，require里也不能编一个API名。
- 需要帮助：确认普通CTA共享内存的原语兼容条件需要在realization具体到什么程度；是否允许这个阶段先给语义保证并保留绑定缺口。DESIGN_SCHEMA推进原则倾向允许草稿继续。类别：字符串深度/资料不足，部分已由文档解开。

## M05-07：B3合并输出为B2究竟是局部还是独特整体方案

- 场景：三提交版本保存S_entry和U供chunk独立READOUT；融合版将READOUT搬到STATE的chunk循环内，删除快照/U的全局保存，主输出生产者与并行轴改变。
- 位置：DOC0 §参数变化与结构变化“新的整体设计…重新贯通解释主要计算、输出形成、工作归属或主协作组织”；“局部新决定…围绕基线已有值、step、role或存储…边界外主体组织继续成立”；§局部记录与完整方案要求直接定位完整基线。
- 最初读法/尝试：起初把去一个launch视作融合局部变化；按文档不能靠提交数判定，转而列出数学物化、输出归属、chunk并行度的联动。倾向将最终选定的融合路线看作另一个整体候选，但它与B3共享很多计算，局部边界也能描述，所以未作确定裁决。
- 影响：只记录候选，不创建两个完整节点，也不强行给“局部”身份。本次没有完整基线，不能把草稿标题当正式dispatch基线。
- 需要帮助：用这个具体例子说明“边界外主体组织继续成立”的判定粒度；是否STATE+READOUT合并已足以达到整体主组织变化。类别：目录判断/自身不确定，无证据表明固定字段必须修改。

## M05-08：工具未适配不是手工设计停止条件——记录已解开的潜在阻力

- 场景：作者入口要求日常检查、交付前检查；同时现有checker/starter还是旧结构。本次明令不执行工具。
- 位置：`schema_extract/EXECUTOR_ALIGNMENT.md` 开头“不能用当前工作树的CLI端到端运行这套新方法”；DESIGN_SCHEMA §5“中间稿可以尚未完成下游”“检查提前结束…继续可独立完成的核对”；作者入口 §1“可以继续不受影响的部分”。
- 读法/尝试：没有把CLI禁止扩大成Planner必须停写；使用文档的新pipe/step/work_schedule表达，明确未检查，不为旧工具填写parent、phase或pipe.steady。
- 影响：本次手工设计未被工具前置阻塞；完整结构/关联验收仍未完成。这项没有留下待批准的问题。
- 需要帮助：无需本实验补工具；后续正式交付应有与现行规范匹配的校验路径。类别：已解开的文字流程边界。
