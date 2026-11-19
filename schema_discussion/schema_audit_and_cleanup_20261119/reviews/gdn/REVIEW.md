# GDN 当前 Schema 审修

本次由同一 worker/reviewer 完成固定源码核对、直接修稿、成品回读和目录检查。保留两份独立设计：`root.yaml` 与 `segmented.yaml` 均从 Version 2 递增为 3，`dispatch.md` 同步修正；未修改 kernel、框架、规范、issue、原 receipt 或运行历史，未调用 extract，也未运行 GPU。

当前正文可作为固定实现的设计记录交付，源码的接受域和数值覆盖缺口仍然存在。结构检查通过不代表该实现满足完整 Contract 的数值验收。

## 来源与任务边界

固定来源为 `KernelSchema@24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2` 下 `kernel_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16`。核对了 candidate/solution.py、gdn_kernel.cu、gdn_common.cuh，problem/reference.py、input.py、metadata.json、shape_range.json、shape_cases.json，以及存储的性能/NCU记录。26 个来源文件的当前字节均与该提交一致，完整摘要见 [source_manifest.json](source_manifest.json)。NCU result 中的七个 problem 文件摘要也与当前固定材料一致；NCU 仍然只是其固定捕获样本的证据。

原三正文及 `.extraction/receipt.json` 原字节保全于 [before](before/)，身份见 [before.json](before.json)。原 receipt 只绑定修改前字节，本次没有回填或借用原 P6 批准。

- ABI 返回 `(out, final_state)`；`core_attn_out_updated` 表示调用者可观察的写回，不是第三个返回值。out 是新 BF16 结果，所请求的 final 是新 FP32 结果；候选从同一次新分配中切出两个不重叠区域。初态按 `[request,head,value,key]` 读取、内部状态按 `[key,value]` 运算，输入初态不写回。
- `output_final_state` 必传；`cu/ci/co/core` 默认 None，归一化开关默认 True，生成器通常显式传 False。reference 接受 cu 张量配 ci/co=None，且完全不读取 ci/co。原 Contract 的此可空分支有参考实现依据，未由候选较宽签名扩张任务；标准非 None 枚举则由 input.py 的 counts/pairs/co 构造支持。
- `shape_range` 给出 H∈{2,4,8,16}、HV/H∈{2,3,8}、1–32 个正长度请求、总 token≤16384；没有为方便实现改成固定 workload 或固定 flag。未把生成器的 Gaussian/softplus/sigmoid 样本变成未声明的浮点硬值域。
- input.py 的 prepare 产生调用前历史状态，不属于 Model.forward。Model 构造主要承担编译；forward 仍包括 wrapper 默认分支、连续化、内部缓存与 tensor map 查询、可选归一化、prep/chain 和结果分配。

## 修正与证据

| 项目 | 发现与修正 | 固定源码位置 |
| --- | --- | --- |
| 数值路径 | 补记 `gram=bf16(dot32(k,k))`；qk 保持 FP32 到 qkd 缩放。明确 system/inverse/rhs/u/du/state plane 的 BF16 RNE，权威状态和 final 仍为 FP32。将“等于 Contract 递推”改为带这些舍入的近似。8 级前代只写 `d<8-j`，16/32 与64合并分别表达，补全64级 BF16逆块写入。 | kernel:434–498、543–566、1329–1389、1422–1687、2377–2434、2553–2754；common:258 |
| 尾块和有限性 | 保留真实 g/beta 零填、外来 token/TMA OOB 行、输出 i<Lc 掩码；删除“有限输入必然无污染”“b 的尾部严格平坦”“累计 \|b\|<35”等过度保证。扫描树及指数锚点存在舍入、溢出/下溢边界。 | kernel:285–328、346–399、1339–1347、2130–2171、2340–2367、2555–2593 |
| 分段近似 | 原文把 G≥3 省略跨段 delta 反馈归为约2^-24的重排误差，并声称 theta 兜住全域。现保留实际标量 a_seg 组合，明确其不同于含 k/beta 反馈的矩阵转移、全域误差界未知。新增独立 pc/pc_hi，区分 pass A 的零/初态传播与 pass C 的组合状态。 | kernel:1782–1804、2201–2310、2798–2827 |
| 片上账与别名 | y_scratch 原 view 按 FP32行单位选择，不能覆盖实际 BF16 40元素步距。改为等容量 BF16单位，y只占前32行的32列。补 pred/qpred 奇偶累加器的跨动作寄存器账。final_stage 明确复用 stage-0 的 K/Q字节；vt64局部记录额外说明跨入v首部508B。 | kernel:164–178、1920–1942、2377–2434、2700–2719、2835–2877 |
| 生产消费顺序 | 保留 PREP 的装载、rho 与 box 在途重叠、全CTA join后再gram；纠正旧性能公式暗示的 gram/rho 重叠。B1发布rhs，B3结束rhs读取并发布u/du，之后rhh才存out。次迭代顶端join退休上一chunk读取并允许环槽复用；最终暂存所覆写的K/Q已在B3/先前迭代join退休。 | kernel:336–417、425–648、2588–2658、2725–2795、2835–2877 |
| 控制与寿命 | 为所有分段worker补显式geometry动作，sg>0不再依赖仅sg=0执行的state0动作来取得nseg/slot0。保留pass C在wait后读取a_seg并决定退出/回退；空段a_seg=1强制回退。修正norm并非工作区内区段，a_seg按value tile复制计量，并保留跨流复用未隔离的源限制。 | kernel:2040–2072、2186–2261、2823–2827、3257–3303 |
| vt64迷你尾块复用 | 额外识别固定源码中stage-0最终暂存与其他warp的K0读取可能冲突，移除把完整路径B3保证套用迷你路径的表述；未补造barrier或改kernel。 | kernel:2498–2528、2530–2550、2835–2852 |
| ABI缺口与分派 | 新说明非连续core的contiguous副本缺少copyback；ci/co None组合的已知拒绝保留。限定输出对象关系而不缩小Contract。同步vt16/vt32/vt64寄存器/SMEM差额、精确平均chunk门和seg/GQA交叉可达性。 | solution:53–68；reference:36–46；kernel:3086–3146、3163–3251、3320–3401、3424–3478 |

没有增删设计或增加 `from`：root 的三区均为本地定义；segmented 的三区也均本地定义，只继承唯一 root Contract。分段改动涉及额外状态对象、两次链提交、跨 grid transfers、输出生产者和动态回退，其独特性充分；有效展开和 source_files 已重新取得。公共骨架的修订在两份实际本地区域分别落实，dispatch 的局部引用也跟随当前对象复核。

## 有限验证

[checks.json](checks.json) 保存命令、退出码、输出文件及执行时的 YAML 摘要；[cpu_checks.json](cpu_checks.json) 和 [abi_cpu_checks.json](abi_cpu_checks.json) 保存可复现的具体 CPU 结果。

- 当前目录完整检查使用 `/opt/venv/python3.13.12/bin/python3 -B schema_template/check_yaml.py <root.yaml> --read-only --skip-hardware`，两份 YAML、dispatch、字段、预算、整区依赖和关系全部通过；最终零规范化差异。编辑期间的引号/预算问题在正文中修正，没有改宽 checker；已有中间诊断留在本 discussion，不进入正式正文。
- 有限事件图覆盖 root 的4-chunk环复用、单chunk/空prep槽/不返回final，以及 segmented 的G=3普通组合和weak时sg>0退出；均无环。图只展开已声明step的代表worker，不能证明CUDA全grid活性，不能替代host可选L2NORM提交条件或执行compute动作后的动态变量求值；sg=0回退的nseg扩展与空段行为由源码审查和CPU算术检查补证。
- BF16 gram 示例的FP32值为0.5333480835、落入BF16为0.53515625；beta=.1时system系数因漏掉此舍入而从0.053466796875变为0.05322265625。三个负g构成的实际shuffle树在零填尾部得到两个相邻FP32值，证实“尾部严格平坦”不能作为位级保证。
- G=3反例使用选中本segmented实例的 H=2/HV=4/N=1/T=3072、q=k=e0、v=0、BF16初态2^22、g≈-0.007445917，仅token2047的beta=.5。每段a≈0.000488282，小于theta；进入第三段时保留反馈的状态约0.500002、标量组合约1.000002，段首BF16输出分别0.0439453125与0.087890625。这是CPU对源块关系和BF16舍入的有限反例，不是CUDA intrinsic逐位仿真，也不冒充外部GPU验收失败记录。
- 显式CPU张量运行固定reference，确认cu张量+ci/co=None、cu=None默认分支和final=None。stride-2可flatten的core被reference正确更新前缀，尾部及交错guard不变；复现候选core.contiguous()数据流时写入副本，调用者core仍为21，证实缺少copyback的影响。
- 所有Storage的 `depth × shape × dtype` 逐项核对。root登记SMEM 121856B，对应源dynamic申请123024B；segmented PASS_C登记163840B，对应165016B。PASS_A登记143360B，未使用但仍申请的u与qkd共20480B，加上相同控制/对齐余量得到同一165016B。PREP设计区域66304B，源按最大ratio分配96512B，不把额外预留误计为本实例活动工作面。寄存器账按每CTA逻辑值量，未声称编译器精确分配。
- 每个5120B三角打包平面的2560个BF16位置唯一覆盖；y_scratch实际2048B值区域、40-BF16行步距得到精确覆盖。有限dispatch样本覆盖root、G=2/G=3、seg+GQA、wide-seg和长短混合空段；未声称穷举动态域。另以合法 H=8/HV=64/N=1/T=4097 定位vt64末chunk cl=64/槽0：warp0的final_state暂存 `stg[key=0,value=8]` 写字节32，warp1的K0[key=16]读同字节32，两者均在迷你路径唯一CTA barrier之后，epilogue覆盖前缺少第二次退休等待。此为源码地址与顺序风险证明，不是GPU竞态实测；两个完整YAML实例均为vt32/vt16，风险属于dispatch列出的vt64局部路径。

## 保留限制与交接

仍保留外部比较器公式/逐输出路由缺失的 source-only 容差记录。metadata中原库的numerical_failure与CERTIFIED_BEST中的候选有限作业通过是不同记录，均不能当作完整动态域证明。任意有限输入的指数/乘积可产生非有限中间量，G≥3标量转移近似没有全域误差界；ci/co缺席接受域、非连续core写回、跨流共享缓存隔离，以及vt64迷你尾块的stage-0退休缺口均为固定源码问题，未以修改Contract掩盖，也没有修kernel。

`gdn-slot-padding-coverage` 旧issue实际指另一个 postconv_mtp FP32状态算子，不能移用为本算子的事实。`source-defect-abstraction-boundary` 仅用于分清实现缺口与Schema表述职责，不作为跳过当前源码核对的依据。

后续统一清理只需处理正式 `.extraction/receipt.json` 的历史过程内容及当前身份导航：完整原件已备份，原批准不得改绑Version 3。本轮没有改动此元数据。当前正文不添加审修日期、模型/角色安排、失败检查或原P6状态叙述；真实实现限制仍留在正文。
