# M08：NVFP4 fused MoE 的独立 P2 评审试走

## 结论与审阅边界

本次不批准整个 P2。保留 cfg97/81 root、cfg99/44 split-K、cfg92 MEGA direct 三个完整候选；定向读完 REDF 输出路径后，建议增加一份 REDF 完整候选。ZERO2 保留为重要机制，但其条件提交仍受当前表达规则阻碍，不能以已批准完整代表计。单体 FP4 与 BF16/warp MMA 族不能统一降为参数；草稿尚未给它们足够明确的去处。另外发现 cfg88 的最后到达者 split-K 归并需单独记录，不能被“split-K 已有代表”自动覆盖。

这不是完整 P1/P3 复审或正式回执。所给 M06 DRAFT 明示尚未完成 P1，任务也禁止补做完整 P1/P3；我按手工实验审查当前事实摘要与 P2 选择，不以缺少正式 YAML 为独立退修理由。下文的“保留”仅指候选身份有依据，不能当作完整方案定义获批。

先读了 AGENTS.md、EXTRACT_SCHEMA.md、P2 阶段页、REVIEW.md、EXECUTOR_ALIGNMENT.md，以及 common 的差异/范围规则、DOC4 提交与调用边界、DOC5 动作、索引、依赖、跨工作接续。仅阅读指定 M06 DRAFT，没有读 M06 ISSUES、其他试走或旧研究/旧 schema。随后核对 problem/reference.py 全文、input.py 的准备与输入/输出构造，以及候选 host 和 CUDA 的定向区段。未执行 Python、提取器、checker、GPU、构建或测试。

证据缩写：`O = kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/`；`S = O/candidate/solution.py`；`C = O/candidate/nvfp4_moe.cu`。行号为当前工作树，HEAD 为 `0c662ebcf2cb57d6b417c12498c3adbf39071a44`。没有重新验证 M06 的全部 SHA256 与文件清洁声明，故不继承该部分保证。

## 当前任务理解

reference.py 的实际计算支持草稿主线：给定路由而非计算 router；逆准备排列后解码 NVFP4；FC1、clamp/activation、每 route 每 16 intermediate 的 QDQ、FC2；每 route 先 BF16 舍入，再乘 score 做 FP32 归并，最后写 BF16 output。`expected` 的 `index_add_` 与 `forward` 的 `output.copy_(result); return [output]` 支持归并及调用方输出身份。input.py:101–126 支持 situ 的 alpha=4/beta=25 与 output 可写区域说明。

这足以判断这里的主要数值接口，不能代替全部签名、域、输出验收材料的正式 P1。没有重新核完 metadata、shape_range、全部 shape_cases、baseline、外部比较器或 CFAST；草稿关于它们的全部结论仍是未独立覆盖范围。特别是误差阈值路由不在本次批准对象内，也不把源码性能/bit-exact 注释当证据。

## 独立候选判断

| 候选 | 判断 | 相对基线的独特价值与证据 |
| --- | --- | --- |
| cfg97/81 packed 主链 | 保留 root 候选 | 是真实配对而非拼接：S:798–838 的 pick 逻辑、S:1270–1308 的 G2 gate、C:7769–7779 的 cfg97 实例、C:8330–8359 的 cfg81 实例支持。C:9250 附近 gather、C:9285–9363 的 FC1/act/G2/finalize 选择支持完整输入到输出主线。root 的作用是建立 packed、WS、route buffer 输出基线，无需再证明相对某个更早设计的独特性。 |
| cfg99/44 split-K | 保留完整候选 | 相对 root，K-slice 成为独立工作轴，FC1 的最终 activation 生产者从 GEMM epilogue 移到独立归并/QDQ 工作，产生全局 FP32 partial。C:7780–7821 的 SK2/AMODE3 实例，C:1352–1400 的 slice_stride 与 `for (sl...)` 固定顺序相加，C:9298–9327 的独立 act 提交支持。不是改一个 tile 数便可保持原主体。配对的 mono G2 与 gather 差额须分别记录，不能把它们统称 split-K 的必然后果。 |
| cfg92 MEGA direct | 保留完整候选 | 相对 root，expert tile 的两段 GEMM 链改为 raw route×I/32 unit；一个 CTA 完成本 unit 的 FC1、QDQ、FC2 partial，跨 CTA 累积 route 值，token 的最后到达者归并输出并清理状态。C:1927–1975 的 `S=p.I>>5`、无路由清零与 census，C:2160–2257 的 atomicAdd、ticket、route BF16 舍入、score 和 self-clean，C:9099–9136 的单 kernel 出口支持。不能降为“empty 时 memset”，因为 graph replay 可带非空当前路由，源码存在完整计算。 |
| REDF，建议具体化为 cfg99/46 | 建议完整保留，数值边界疑点已消除 | 以 cfg99/44 为最近基线，可隔离输出组织差额。C:8234–8262 显示 44/46 同属 mono FP4 G2，46 切 REDF；C:3948–3987 实际先 `__floats2bfloat162_rn` 再转 FP32 乘 score，执行 `red.global.add.v4.f32`。原先每 route 独占写 BF16 rout、finalizer 拉取全部 slots 的归并，变为 G2 worker 对 token 共享 FP32 acc32 推送；C:9164–9192 清零侧支、C:9329–9338 join/convert、C:1687–1703 最终 BF16 写出共同定义新输出形成。这个差额涉及整段输出的生产者、共享冲突、初值、完成和数值求和顺序，不是仅替换 store 指令。 |
| ZERO2 | 保留重要独立机制，阻塞其完整表达/放行 | C:9471–9491 的 host spin 后 `if (*zh != 1) cudaGraphLaunch(full)` 是实际条件提交；clean/full 不是同一固定提交的空工作。outer capture 又只记录，之后 replay 才执行。相对普通链，验证节点既可直接生产全部输出，又由 host 本次观察决定是否提交后续主体。不能用普通 graph 缓存说明抹去它。详见下节。 |
| 单体 FP4、BF16/warp MMA 族 | 不接受整族降级，尚不能给出完整选集 | C:3053–3110、3531–3718 的 mono 主循环包含 CTA 同步的装载/矩阵乘/退役关系，WS 则分 producer/MMA/epilogue 并跨 item 接续；这不是角色数量参数。与此同时 cfg99/44 已含 mono G2，不能据“又有 mono”再造一份重复候选。BF16 路径还由 S:1357–1378 的表示族、C:9285 附近 decode、C:9297 附近 act 分支提示不同表示/物化。需先明确已有代表承载哪些机制，再决定新增完整、组合或明确未展开独立方案。 |

REDF 的判断不是因为差额跨多个字段就必须完整。它替换的是路由输出归并的主体；若只把这段封装成一个“REDF epilogue”新对象，会掩盖由谁收集全部 route、何时可转成最终输出。相对 MEGA 也不重复：MEGA 的 acc32 是 `[R,H]`、归并的是 I chunk，随后最后到达者做 token/slot 收集；REDF 的 acc32 是 `[T,H]`、直接归并加权 route，后续 convert 不再负责归并。两者的共享位置与完成判定不同。

root 的 fixed_3 示例足以支持可达性线索，但未实际生成路由，不能批准“这个 seed 必走此路径”。P3 仍须固定具体配置与范围，说明非零/zero probe、TMA 成功、所选 gather/finalizer/CPHOIST 配置；不能只列 shape 就称其涵盖所有输入状态。S 中 G2 gate 的 lower-bound 推理也不能据此变成“所有有 masked route 的输入均有此实际 tile 下界”的数学保证；selector 的代码行为与实际工作量是两件事。

## 降级、合并与需补的线索

- **DEEP：局部记录。** C:4624–4660 改的是同一 payload 三槽、scale 两槽的复用等待；producer、MMA、epilogue、值接口仍成立。必须记录等待由 g−2 拆成 payload g−3 / scale g−2，以及保证来自哪个退役事件，不能删成普通“更深流水”。
- **NPAIR：局部记录。** C:4944–4968 用一枚 n128 替代 up/gate n64 对，需联动 B-scale 图像、TMEM/SMEM 和 MMA 数量；四角色族及整体输出形成未重组。重复配置名应合并为一次差额记录，而不是再建完整 YAML。
- **tile、槽深等：仅在同机制及必要联动已核实后作为参数记录。** 本次没有批准所有取值及配对，尤其不能从两个 stage 值相近推出生命周期相同。
- **pre-gather vs clipped gather4：保留局部候选判断。** C:4386–4400、C:4661–4690 支持 route-view 物化消失、直接按 ptok load、有效 box 数与 expect_tx 同步改变；边界仍是同一 packed GEMM A 值。草稿的方向合理，但本次没有覆盖全部 padding 与 metadata 初始化，不给局部记录全文批准。
- **small/big prepare：仍是待核局部候选。** 相同 map 接口是判断起点，不是结论；本次只核 C:9194–9248 的提交组织，未核全部 map 值和 PDL 出口。
- **普通 graph 缓存/CFAST：不增加完整设计候选。** 草稿对普通缓存与 ZERO2 已作区分；但本次未全文审 CFAST，不能批准其在途副作用完全可忽略的强断言。
- **MEGA staged：若后续记录，优先相对 direct 做权重搬运/存储局部差额。** C:1894–1931 的同一 `k_mega_body<STAGED>`、C:2180 附近同一计算改从 staged 或 global 权重取值提供方向；未审完所有 staged 同步，暂不批准具体记录。

新增重要线索是 **cfg88 的 fused split-K last-arriver**。C:8031–8062 存在真实 SK2/AMODE3/FUSE 实例；C:5270–5309 的 partial fence、`atomicAdd(cnt)==SK-1`、固定 slice 归并，将 activation 生产者移到最后到达 CTA 的 epilogue。其位置在 s_epidone 之后，使该 CTA 的主循环可以接续下一 item。C:9299 明确 cfg88 不提交独立 k_act_packed。

它不能按“只少一个 launch”或“数学与 split-K 相同”忽略，也不能说 MEGA 有 ticket 所以组合已知：ticket 的工作组是 `(M tile,N tile)`，完成 activation 而非最终 output，且与 producer/MMA 的跨 item 进展作用不同。本次将其列为独立机制、完整展开与否待选；没有追完计数初值、全部 partial 生产者/可见性和后继接口，因此不伪装已确定完整闭合。后续定向补这条链即可，无需枚举所有 cfg 配对。

## 工作调度与 pipe 片段是否已说明关键关系

草稿的文字已足以支持 P2 的主要识别：CTA 是 worker、item 为工作坐标、grid-stride 预分配不是动态队列，g 跨 item；C:4450–4505 的 `start/istride` 与持续 `g` 支持。实例 `dim3(224)` 和 wrapper `(3+EPI)*32`（C:5378–5390）确认是七 warp，不能被函数内旧 `0..191/0..5` 注释误导。

两条 released 边与 DOC5 的半开索引、flatten、声明窗口是相容的；C:4628–4660、4950–4968 支持 payload g+3 与 BScale g+2 覆盖前等待同一旧 stage MMA commit。它们解释了 **operand 槽复用距离**，没有解释整个 WS 关键协作：未列操作数 prepared、A-scale 的保护、accumulator 退役、epilogue 完成如何放行下一 item、CPHOIST 的跨 item 等待移动、TSTORE sink 的复用/完成以及 handoff 到 finalizer。

故答案是：**足以作 P2 的局部事实示范，不足以批准完整调度/pipe，也不足以证明所有必要跨 item 关系已写全。** 本次不会因为它自称“极短片段”就要求补一整份 P3，也不会把两条正确边扩张为完整性保证。`matmul done` 汇总 scale cp 与多枚 MMA 的完成必须在正式动作声明中绑定源码保证；草稿已说明方向，本次未独立审所有原语定义。

ZERO2 的字段问题与上述可表达依赖不同。DOC4:243–259 要求 step 恰好在 launch 列表一次并以实际计算提交为准，只有 step/order/start_after；DOC4:395 又不允许说明文字条件删边。DOC5:286–301 的 if 是 worker 内动作控制，不取消 host 提交。common:112 却允许按本次 Clean 决定必要动作。我的查找没有找到可直接表达本次 host verdict/timeout 条件提交完整链的现有结构。给 full 添加 start_after 只改变启动时点，不让提交消失；把不存在工作伪写成空集合又违背 REVIEW 对未提交/空工作/no-op 的区分。

这是一个具体未解决表达阻碍，不宣称已证明所有合法表达都不存在。恢复条件是提供可复查的现有合规表达，或由有权限者澄清/调整规则；不能由本 reviewer 私加 launch.guard。若明确缩窄当前完整收录范围，也须在目录保留 ZERO2 机制与未展开原因，不把整个算子当作已解决。

## 四项判断与后续止点

- 独立理解：能理解主要任务与四个已选机制；尚不能从草稿确定整个实现范围的选集及取舍。
- 分层正确：已审文字未把 selector 当 Contract；ZERO2 的错误 flow 例子被草稿明确否定，不当作已提交设计。正式数值/调度层仍未完成。
- 全文一致：候选方向大体有据；mono G2 已在 split-K 中出现却又整族列“未研究”，需拆开责任；cfg88 去处缺失。ZERO2 仍无法批准其固定 launch 表达。
- 内容必要：DEEP/NPAIR 等保留差额足够；RED F/MEGA/split-K 的主体知识应完整承载。没有依据要求多写所有配置，也没有依据删掉重要未展开机制。

可先完成普通 root 写作；正式放行整个 P2 前，应处理 ZERO2 表达/范围、cfg88 机制去处、mono/BF16 的有限识别与取舍。RED F 的本次数值疑点已解决，不应重复按“未读 epilogue”扣住。未读范围不泛化成所有代码都要再审。

尚未审：全部入口签名/域/验收路由、CFAST 完整前缀、所有配置 fallback/失败副作用、全部 route-map 初始化、mono/BF16 全主循环、FUSE 完整生命周期、完整 ZERO 验证发布协议，以及任何完整 P3 schema。没有 GPU 正确性结论或性能保证。
