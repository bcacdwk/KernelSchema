# 当前验收规范：Task、V01–V15、公开材料与消费模式

日期：2026-11-20。本文是案例判据、有限验收矩阵、公开文件和消费模式的唯一正文；实际实施与各层资格状态见[实施记录](../implementation/README.md)，本规范本身不构成通过证据。路径保留供既有链接使用，原 R5 推导见[审查前快照](../review_20261120/before/planning__r5_validation_delivery_plan.md.txt)。
[合同](r5_contract_execution_review.md)定义 API/身份/恢复/预算/结果与方法，[主计划](../04_implementation_plan.md)定义阶段/发布门，[交付单元](../05_work_packages.md)唯一持有 owner/依赖；本文不另设发行顺序。已有材料只代表可定位来源，计划、CPU fixture 和历史复验不代表本轮模型/GPU 成功。
来源前缀：KS=KernelSchema，AKA=atrex-kernel-agent；FA4 原码位于 KS:kernel_wiki/aka-four-problems-20261019/fa4_decode/，GW 文档位于 AKA:src/aka/execution/skills/atrex-gpu-gateway/reference/。

<a id="task"></a>
## 1. row_softmax_fp32_v1：完整 Task

首题固定四个合法 shape，不增加非二幂、多 dtype 或布局平台；扩大输入域须新 Task 身份及资格。

| 项目 | 固定规范 |
| --- | --- |
| 入口 | kernel.py 暴露 class Model(torch.nn.Module)，Model() 无参数；forward(self,x,out)→out，支持 x=、out= 调用 |
| 张量 | x/out 均 CUDA FP32、同设备、shape=(M,N)、stride=(N,1)、requires_grad=False；M∈{1,32}、N∈{128,1024}，支持任意合法 storage_offset |
| 输入域 | x 有限且位于 [-20,20]；x/out 不共享 storage；候选不得依赖测试编号、运行次序或预存结果 |
| 输出/副作用 | return is out；原 out 输出区写满，out 对象/原 storage/view 签名保持；x 字节/对象及 storage/view 签名保持；只允许写原 out 输出区域 |
| 调用状态 | inference_mode、eager、单调用流；准备/预热后可重复调用并重算变化的合法 x；首发不支持 CUDA Graph 或多流并发 |
| 域外 | Task 准备/adapter 明确拒绝不支持 shape/dtype/别名；域外失败不冒充域内反例 |
| Golden | 调用前保存 x 的 FP32 字节，再提升 CPU FP64 算稳定 softmax；不从调用后 x 重算 |
| 性能/native baseline | 单独 baseline.py：out.copy_(torch.softmax(x,dim=-1)); return out，FP32、同 ABI；它是 Task.baseline_source，不是 FP64 golden |

令 z=FP64(x_before)，m_i=max_j z_ij，g_ij=exp(z_ij-m_i)/Σ_k exp(z_ik-m_i)。先拒 NaN/Inf，再逐元素检查 `abs(FP64(actual_ij)-g_ij) ≤ 2e-6 + 2e-5*abs(g_ij)`；全部通过，不能改成平均误差、native 默认 allclose 或以 actual 为相对项基准。
入场保存 x/out 的对象、原 storage 活引用、storage_offset、shape、stride、dtype/device 及 backing 边界；x 保存原字节，out 在前后 canary 之间毒化。调用后核对对象/签名，并从保留的原 backing 输出区检查写满/数值及外围不变，不能只沿调用后的 out 取值。即使 `out.set_(tmp); return out` 保持 Python 对象，也必须因原 storage/view/输出区违规失败。
合法非零 offset 不能被 adapter 清零以简化检查。有限 canary/offset 样本只给有限副作用证据，不宣称穷尽任意布局或全部设备内存安全；候选无 evaluator 文件写权限。

inputs.json 固定 4 shapes×下列 9 个实例=36 case_id；每个实例以 NaN 与 -123.0 初始化 out，各调用一次，共 72 次。
随机使用锁定 torch 的 CPU Generator，分别 manual_seed(17)、manual_seed(23)，FP32 `-20+40*rand`，再复制 GPU，保存实际输入字节摘要。

| 家族 | 每 shape 的实例 | 典型错误 |
| --- | --- | --- |
| 均匀随机 | seeds 17、23，共两个 | 错公式、跨行污染、只适配一个输入 |
| 常值 | 全 -20、全 0、全 +20，共三个 | 未除分母、全矩阵归一化 |
| 两端交替 | x_ij=20 当 (i+j)%2=0，否则 -20 | 极值/索引/数值路径 |
| 每行单峰 | 峰列 (N-1-17*i)%N 为 +20，其余 -20 | 漏末列、广播、均匀/全零输出 |
| 近等值 | x_ij=(-1)^i*19+((j%5)-2)*2^-12 | 低精度中间量、混用行统计 |
| 行间置换 ramp | x_ij=-20+40*(((j*(2*i+1)+i)%N)/(N-1))，FP32 生成 | 行列错位、只算首行、整批复制 |

每 shape 的 seed=23 实例明确使用 x offset=3、out offset=5，两个 out 初值均如此；其余保持已声明合法布局。保存布局签名/输入摘要/原 backing 检查结果，仍为上述 72 次，不增输入家族。
另用同一个 Model 连续处理两个已有合法 case：宿主在两次调用间覆写同一 x storage，重新保存 before 副本并毒化复用 out，验证未缓存结果；复用既有输入，不扩输入全集。
对象身份、原 storage/view、写满、x 不变与外围不变均为独立硬项，数值通过不能覆盖它们；有限样本通过不等于连续全域证明。

完整计时采用固定 timing.py：每 shape 用 seed=17，独立预分配 out，两方均调用 `model.forward(x=x,out=out)`。构造/首次编译/20 次预热单列，不计稳态。
每批先同步，perf_counter_ns 开始，循环 100 次完整 forward，设备同步后结束并除以 100；每 shape 10 对批次，AB/BA 交替，保留全部顺序和纳秒值，报告中位数/IQR/最小/最大/实际样本数。
这是含 Python 提交与批末同步的摊销完整调用时间，不是孤立单调用延迟或 NCU kernel-sum/CUDA event 纯设备时间。输入生成/毒化/比较在外，forward 内分配/copy/全部 kernel/必要同步在内，两方一致。
软目标为四 shapes 等权几何均值 `exp(mean(log(t_baseline/t_candidate))) ≥ 1`；样本齐全且环境/代码/计时身份明确才判 met/unmet，较慢但硬项全通过仍可 completed。噪声和缺样照实报告，略大于 1 不称已证明提速。

| 最小反例 | 必须失败的项 |
| --- | --- |
| 新张量返回且未写 out | return 身份、原 out 毒化/写满 |
| out 写满但 return out.clone() | return is out |
| 同一 out 对象 set_ 到装有正确值的新 storage 再返回 | 原 storage/view、原输出区与 backing 检查 |
| 只写 out[0] 或漏末列 | M=32/单峰/poison 的全量比较 |
| 原位改 x 后用被改 x 计算 | 入场 x 字节/签名及调用前 golden |
| 超容差或偷换 comparator | 固定逐元素数值判据 |
| FP16 归约、外部 softmax、删除 max-subtract | 按当前 R 的 fidelity；删除 max 在该输入域可能数值通过，不伪造必然溢出 |

<a id="materials"></a>
## 2. R0、参考材料与真实入口材料

手审 R0 服务 fixed 工程；design 输出称 Rd，不要求与 R0 相同，也不把 R0 答案偷偷投递给 design。

| R0 的设计义务 | coder 自由度 |
| --- | --- |
| max-subtract→FP32 exp→FP32 行内 sum→FP32 除法/写出；无 FP16/BF16 中间量 | FP32 归约树、寄存器命名、编译器调度；近似 exp 须满足已声明语义与 Task 容差 |
| 一 Triton program/CTA 拥有整行，grid=(M,)，每行宽 N，一轮读取/归约/写回 | num_warps∈{4,8}，不改变路径/ownership 的编译参数进入源码/证据 |
| x 与指数在 program 内使用，无显式全局中间张量、跨 CTA 通信/原子归约 | 编译器寄存器分配/spill 如实报告，不声称静态表证明无 spill |
| forward 一个自有 Triton kernel，不以参考/torch.softmax/外部实现代算 | host 封装、合法 shape 分支与 launch；不在 constructor 预计算输出 |
| out 全覆盖、x 只读，无异步跨 program 生命周期义务 | 二幂 N 无额外尾 tile 义务，不宣称任意 N |

这些是 R0 fixture，通用方法采用[C fidelity](r5_contract_execution_review.md#fidelity)。warp 自由参数不增 revision，多行 program 或物化/分工改变才是实质设计变化。报告必须解释当前完整 Schema/scope 的实际义务，不沿用 R0 表代审 Rd/R′。
synthetic 配对：同一个数值正确的多行 program C，对 R0 的单行 ownership 判 fail，对合法声明该多行设计的 Rd 在其余义务也有证据时判 pass；如果新 R 增加方法不能解释的义务，必须 unknown。覆盖与 fidelity 共用报告但分别判定；该配对不要求真实 optimize 一定 revise。

Materials 包括当轮 workflow_objective（唯一保留 ID，按 C）、Contract、角色方法、schema_docs/0–6、Template 版本、实际设备/栈资料及一个 Triton fused-softmax 参考。按角色 allowlist 投递，design 不读手审答案或私有 evaluator。
参考发现地址为 https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html；实施时固定匹配资格版本的官方源码/文档 revision，记录取得 URL/tag/commit/文档版本、摘要、许可、访问日期；moving main 不作冻结身份，本轮未下载。
设备资料同样取选中 profile 的官方版本；现有 catalog 只透明筛选相邻机制并说明差异，不称已有 softmax 近邻 Schema。资料不可再分发时使用合法固定获取方式；必要原件无法取得不假称公开闭包完整。
公开教程源码可以作为设计参考，因此小 demo 证明“借参考设计/实现”；纯 Schema 重建的信息边界另见[消费模式](#consumption)。

唯一 examples 请求组为 fixed-r0.json、design.json、optimize-template.json、from-partial.json、verify.json。demo 宿主从准确结果/证据和余额构造 implement-request/optimize.json，不能读 latest 猜输入或提交占位摘要。
真实 design 留选择依据、草稿/规范化/审阅、Rd 与可消费请求；真实 implement 从骨架生成并验收，另给明示人工来源的 partial seed，模型从准确字节继续。跨两次调用/e 的准确续写由 V04/V05 证明；真实首轮已完成不伪造第二轮。
一次真实 optimize 使用已结算反馈和目标；keep 照实发布理由/停止条件；revise 才发布新 R、明确旧 C 是否 seed 并重验。KS 有限单 owner driver 从 design 或已有 extract 消费开始，在 implement 后至多调用一次 optimize；有合法 revise、同 R implement 或 verify 行动时，按 C 的结算、准入与同父余额规则实际执行后续固定请求。keep 无行动停止，缺项/fence/耗尽照实返回；不只产出未执行的请求就宣称闭环。真实只有 keep 无行动时正常交付，其他行动/恢复分支用 V09 的 CPU 轨迹验证并标 synthetic，不称真实修订或收益。
命令与三操作 run 恢复统一见[C API](r5_contract_execution_review.md#api)。extract 保留原命令与透传，展示既有 curated FA4 消费/缺项即可，不为首发重做提取；原栈未资格时记录 blocked，复杂实现由 F1 单独验收。

<a id="adapters"></a>
## 3. 公共 evaluator 与原生执行资格

已知文档事实：Agate 的 Python Model/reference.py/input.py/shapes.json 见 GW:reference-dir.md；Light 非 SOL 需要 Atrex-Bench runtime_root，真实标记在 AKA:task/problem/layout.py。manifest/compile_targets 不证明 worker 可用或 out 副作用等价，稳定跨进程 operation key 要显式提供；这些不是当前服务资格。

| 资格项 | 必须取得的证据 | 判据 |
| --- | --- | --- |
| 唯一 Task | Task/evaluator manifest、V01 原始结果 | 完整 ABI/72 次/golden64/原 storage/offset/副作用/计时，通用 allclose 不够 |
| 包装与来源 | abi-map、candidate-loader trace、C→native→执行摘要 | 同 kernel.py::Model 原字节，包装不替候选写 out/复制返回值或屏蔽异常 |
| golden 映射 | comparator-map、坏候选结果 | native 如需 FP32 reference Model 可包装，完整比较仍使用公共 golden64 |
| native public | 实际角色投递清单 | 仅授权资料；baseline/private evaluator 不因 bootstrap 绕过 allowlist |
| native InitialSource/V0 | Task.baseline_source 摘要、V0 输入/终态/费用 | 测可执行 FP32 softmax+copy baseline；V0 pass 不等于当前 R qualified |
| 首次 coder C0 | starter 或显式 partial seed 摘要、首调 draft trace | 在 coder 准入前安装准确 C0；不可执行骨架不冒充 V0，原生“优化 baseline”正文不覆盖固定 assignment |
| 后续 C/G/HEAD | e/准备/源码/晋升关联 | 后续从准确终态 C 继续并关联当时 HEAD；G 仅 native 比较，V0/C0/G/HEAD 分列且不假定相等 |
| native runtime/hook | 固定 Atrex-Bench ref/源码/安装来源、task-hook 实际结果 | 找到实际 custom hook/受信 Evaluation 运行完整判据；空目录/valid.py 名称不算资格 |
| 公共计时 | timing-map、完整 samples | 可执行完整 forward recipe；native ABBA/晋升另报，不能替换公共计时 |
| 传输/恢复/费用 | API/profile/intent/native-root/job/terminal/artifact 关联 | 收费前可恢复定位，固定 C/evaluator 可核，未知原 job 零重提，V0/V1/预取 review/production 真实费用可归 owner |
| 两后端对接 | 各自的同字节正例与关键坏候选记录 | 漏 out、改 x、超差、同 out 换 storage 均在实际 adapter 拒绝；一边 pass 不代签另一边 |

V0 baseline 在当前 native/profile 是否可用仍需实际资格，不伪造 baseline pass，不降低 ABBA/框架约束。never 也不会跳过 V0；helper 足够则 never，否则只在固定设计/额度/提交前资格均闭合时采用受限 V1，职责结论和费用保留。恢复不能重装 seed 覆盖 draft、重做已结算 V0 或偷偷退 G。
先核 Agate native eval 是否能表达完整 Task，否则核获支持命令能否运行同 evaluator 并回收完整事实；都不能时 KS 选同 evaluator 的合格本地 GPU。AKA 无实际 native hook 则该 Task unsupported，不能抄 KS 本地结果。未知在途作业先结算，不在运行中切通道重复付费。
设备选择先检查已授权可用 H20，再选可取得的 NVIDIA/Triton 组合并固定 profile；记录实际设备/架构、驱动、CUDA、Python/torch/triton、CLI/请求与报告模型、镜像/evaluator 摘要及时间。H20 文档或旧 L20D 不代表当前资格，硬件变化需新固定身份。
材料与 CPU 开发可先行；真实模型资格限制对应真实调用，真实 GPU/服务资格限制对应支持声明，不冻结独立 CPU 工作。

<a id="fa4"></a>
## 4. FA4 decode：提前预审与真实 F1

原始 kernel/reference/input/shapes 固定为 KS@3d64ad35900a0e6a66d6f93da34108fbceba0daf 的四题快照 v55，不是 44 题主批次；F1 消费的四节点 Schema/dispatch 另固定为 KS@484902855fe7be82a74fe7c394503d1d7d222dbb 的 schema_template/examples/fa4_decode/，其 Source/Golden_reference 指向前述原件。源码提交不充当这份后续 Schema 的版本；对应证据见[K7 材料交付](../implementation/b01_k7_fa4_materials.md)。共享 Contract 是 B=1..17、Q∈{1,4}、Q≤L_b≤131072、FP8 E4M3FN 输入/BF16 out、13 参数 ABI。
完整比较在 FP32 下两输出有限且逐元素 `abs(actual-reference)≤0.06+0.04*abs(reference)`，另核 alias/非授权输入不变。root 只覆盖 Q4、B∈{15,16}、长 KV 的 S9 global；q1_global 还引用 root 数学区，root 不能代全 Task。
Schema 记录 L20D/SM103、CuTe DSL 4.6.1、torch 2.9.0+cu130、cuda-python 13.1.1，编译器/驱动未记、Wiki 有占位；这些都是需要核实的材料条件。

| 范围 | 输入/材料 | 能声明的结论 |
| --- | --- | --- |
| F0 来源预审 | 完整 aka_solution/、reference.py、input.py、原 shapes、许可/摘要，Model 13 参数→Contract 对照 | 原件可定位和公开取得缺项，不是运行资格 |
| F1 实际子请求 | q4_direct；B=1、Q=4、seq_lens=[64]、P=1、有效 block_table=[0]，其余常量严格按 root | 完整 direct 子请求，N=1/S=1；不是 q4_direct 全域或 full Task |
| 后续 F2/F3 | F1 结果与实际下一请求确定 direct 边界、混长/页映射或 root/global/q1/cluster | 每次固定 scope/when/必要 dispatch 和新资格，没有预排扩域配置批次 |

F1 来自既有代表样本，q4_direct 的 when 允许该点并给 S=1/bk=256。原 `_make_inputs(seq_lens_values,...)` 的 Q4 query/kv 分布倍率为 1.22/0.96、workspace 初零、out 未初始化；保留原生成语义并固定种子，另加 out poison/不变输入检查，不重做 extract 或修改原 YAML 迁就环境。
effective 交接包含 root Contract+选中 direct Design；不能单独拿无 Contract 的 q4_direct raw 充完整任务。实际请求需要的 dispatch、源文件快照/删除语义必须实现。
范围限制必须保留：global 可能读空 split 的旧 opartial；PageIndex 四槽在 N≥5 的顺序未证明（既有 pageindex_order.md）；F1 N=1 不经过这些多轮路径，仍须审空奇偶槽、TMA 页尾与 TMEM 生命周期，不能因避开一例就签全部异步安全。
四节点还要求已建立 native launcher、同对象/签名/线程/设备及非 capture 稳态。新 Task 记录冷启动/探测与准备过程，之后才量声明稳态；launcher 的 1e-2 探测阈值不是 0.06/0.04 数值容差。
原源树要求 solution.py 与 qoder_greenfield_decode_attention/ 同目录；在 F1 内实现实际需要的格式/codec/adapter。真实 CuTe/实际 SM103、资料、完整 evaluator 和 fidelity 资格不得由 H20/Triton 替代。
历史 L20D 30/30 和 kernel-sum 排除 host 的 1.1126x 只作原码材料，不证明新副作用/范围/重建或完整调用收益。预审交 available/missing/incompatible；F1 缺栈保持 blocked，不影响小题独立门。

<a id="validation"></a>
## 5. V01–V15：有限验收矩阵

CPU 表示真实状态/codec/恢复代码配 fake 角色/执行器；MODEL 表示实际锁定模型调用；GPU/服务表示实际 evaluator/profile。模拟报告必须标 synthetic，不能供产品资格冒充原始真实收据。
每份 E/index 记录方法、时间、Q/run/Task/R/C/rules/evaluator/实际环境/profile/adapter 身份、状态及原始文件；证据主体与依赖闭包按 C 可定位。owner 只见[W 验收归属](../05_work_packages.md#verification)。

| ID / 类别 | 输入或反例 | 必须观察的事实 | 最小证据 / 类型 |
| --- | --- | --- | --- |
| V01 正确性 | 本文 72 次，含每 shape offset=3/5；正例和坏候选 | 全量数值与原 out storage/view/输出区、x/外围硬项一致；漏 out、改 x、超差、同对象换 storage 均在实际 adapter 拒绝 | cases、布局/输入摘要、backing/比较器映射与 raw / GPU；CPU 只验解析 |
| V02 fidelity/权限 | R0 正例；外部 softmax/省 max/违规更快 synthetic；同多行 C 对 R0/Rd；新不明义务；已结算 reviewer unknown | 当前 R 义务/Schema与代码位置/理由齐；R0 fail、合法 Rd pass、方法不足 unknown；unknown 有动作/余额可补证但未补齐不 qualified；coverage/fidelity 独立，角色/Contract/evaluator 越权拒绝 | method、review、合报告、admission/投递 trace / CPU+MODEL，配对标 synthetic |
| V03 性能/慢合格 | 同 R 合格 C 比公共 baseline/native G 慢且未晋升 | completed+unmet 与 G/promotion 分列；计时完整；latest/qualified 准确 | samples、result、native-outcome / CPU 证明分流，GPU 记录实际计时 |
| V04 准确续写 | partial/未晋升 C1→同 R 下一调用/e→C2；改函数/删除旧注释 | 下次实际收到完整 C1，不退 G；C2 仅留新字节，关联当时 HEAD，预算累计 | C1/C2 manifest、invoke/HEAD/G trace / CPU；实际续写另标 MODEL |
| V05 seed/格式中断 | intent/安装中/ready/准入后中断；删 kernel.py、夹 solution.json；F1 必要格式删除例 | 准入前只见完整合法 seed，已准入 draft 不重装；损坏/超格式拒绝不补旧文件；实际扩展两文件到删除一文件仍为完整快照 | boundary trace、source manifests、result / CPU |
| V06 错绑/闭包/verify | C1 已测/C2 改一字节；同 C 换 Task/evaluator/R；选 Schema+parent/raw、source、证据 | 不继承错主体 pass；verify 零 coder/改码/晋升且必要执行审阅计费；选中证据的裸摘要主体可定位；缺真实依赖/同路径异字节模式拒绝，缺无关私有历史成功；脱敏不替旧 Ref | subjects、selected-roots、closure、rejected-links、usage/result / CPU |
| V07 未知作业 | 提交后断连/取消接受/404，后原 job 终态到达 | 原 key/定位保留，未知时零新提交/零换通道；原收据吸收一次，否则 blocked/settle_only；历史 qualified 不掩盖 fence | intent、job-events、receipts/result / CPU；所宣告服务恢复另需真实收据 |
| V08 run/预算/暂停 | implement 有费用未生成首 result；同 run 重放/明确新 run/同父跨 R；外部 AKA 原生根在私有位置；pause/余额0/未知统计 | 按 run 找回原 operation，重复收据零费、明确新 run 新费、同父跨 R 不重置；V0/并行预取 review/production 与外部子调用同 owner；在途 pause 仍计时、静止才停、父子无双计；已终止统计 null 不自动当在途，限额余额不明仍 fence | run-anchor、admission/reservation、native locator、clock transitions、receipts/usage / CPU；实际外部定位能力有对应真实资格 |
| V09 写作/新 R/恢复与资格归属 | design、optimize 各有收费事实而首 result 未生成；目标缺失/重复/改变；R′ 发布未准入/准入未建 e；keep 无行动、同 R implement/verify、revise；下列三条请求归属轨迹 | 两写作入口找回/结算/重建，keep 无行动零新费；目标唯一且改变 Q、不放宽 Task；change_report 先冻结、writing_review 后绑定无环；KS driver/AKA 动态外壳执行合法后续固定行动，先结算旧事务，新 R 用固定方法重判；新旧资格分清，迟到原收据不改当前归属 | publication/依赖图、requests、assignment/review、driver/action/admission/recovery 与 receipts trace / CPU；实际 keep/revise/后续行动各按 MODEL/GPU 标记 |
| V10 三 audit | 普通 seal 后、fidelity 后/ABBA 前未提交恢复、已提交恢复；audit 缺资格 | normal/未提交绑定原资格；已提交从 commit-bound audit 吸收；缺资格 unknown 保留历史、禁自动推进，不新评改历史 | eligibility、attempt、audit、recovery/usage / CPU |
| V11 指定 e/V0/C0 | 已终 e 再请求、双 native 上限；不可执行骨架 C0、非空 partial C0 两例；V0 中断；never/helper 或受限 V1 | 无 e+1/已结算 V0 重做；V0 测正确 baseline，首调准确 C0，恢复不覆盖 draft/退 G；metadata/pin/production/HEAD 职责分列，V0/V1/预取 reviewer 真费用在同账户 | driver结论、setup/finish/seed trace、V0/C0/G/HEAD、usage / CPU；所用 baseline/native profile 的可用性需真实资格 |
| V12 off | 默认安装/显式 off，KS 不存在且禁 import/下载，原 native 配置 | 归一化 prompt/tools/额度/晋升/导出一致，零 KS 扫描/初始化/挂载/导入/下载/模型调用；不要求随机输出逐字相同 | upstream/fork-off、normalized-diff、安装/开关 trace / CPU |
| V13 公开独立 | 新目录/任意 cwd、无 AKA、唯一 CLI/API 与有限 driver；原 extract/curated 消费；选定公开闭包 | 三新入口及迭代命令实际可调用，消费 optimize 行动/keep 停止与恢复证据可读，作者绝对路径非必需；原 extract 状态不变；缺无关日志不拒绝，选中证据私有真依赖不能由脱敏代替 | install/resources/cli/driver/closure/model/gpu/readback / CPU+MODEL+GPU |
| V14 AKA 固定源码发行与 KS 配对 | 固定 fork+可选 KS gitlink；独立目录/非原 cwd 的 AKA 自有源码安装；同一安装的 on/off switch、fixed 与 A6 动态入口；自身固定产物恢复 | A3 包 ownership/metadata、实际安装/资源闭包和缺项诊断；默认 off 满足 V12；A7.fixed 核来源/许可与 fresh-child 实际模块/资源；pair 另汇合两后端各自真实记录；动态仅在 A6.real 后声明 | release-pair、install/source/resources、fresh-child/restore、switch/run、pair-index / CPU；所宣称运行能力另附各自 MODEL/GPU 记录 |
| V15 复杂范围 | F1；root 绑定 Q1/direct 反例，N≥5/空 split 限制 | 当次 when/scope/必要 dispatch 合法才准入；合报告仍分 coverage/fidelity、列 task_uncovered；只选中子范围可能 completed，异步 unknown 不代签 | admission、combined-review、limitations、raw/samples / 材料+后续 MODEL/GPU |

V08 与 V09 合起来为 implement/design/optimize 各提供一条“已有收费事实、首份 result 尚未生成”的恢复轨迹；外部 AKA 的 private native root/版本/操作键在首返回丢失后仍可取回。CPU 注入边界不反复要求真实模型崩溃，实际 profile 另证明它宣告的定位/终态回收能力。
V03 的“慢”可由 CPU 确定构造；GPU 只记录实际观测，不用 busy-wait 或改日志伪造慢实现/提速。检查 unknown 的已结算补证与未知 job 禁重提是相反轨迹，不能合并成一条全面禁继续规则。
V09 的三条有界 CPU 轨迹只验证当前请求与历史证据的归属，由 K5/K6/A6 消费；不要求跨请求候选排序服务。

| 请求归属轨迹 | 必须观察的结果 |
| --- | --- |
| 新 R 已准入且其 C 合格 | 当前 qualified 属于新 R；较慢仍可满足硬条件，不按 native G/晋升替代 |
| 新 R 已准入但未完成，旧 `(R,C)` 曾合格 | 当前状态仍为 partial/blocked；旧资格明确标历史，不填成新 R 的 qualified |
| 旧 Q/R 的原收据迟到并重放 | 仅吸收到原 operation/资格/账户一次，不改当前 admitted_request、不挪用旧 pass、不绕过 fence |

<a id="public"></a>
## 6. 公开材料、源码安装与证据边界

下表是未来可交付的最小集合，owner 见 W。文件存在不表示已运行；公开支持必须能从 index 追到真实事实。

| 位置 | 文件/内容 |
| --- | --- |
| examples/workflows/row_softmax/task/ | task.json、contract.yaml、inputs.json、acceptance.md、materials.json、scope.json；完整输入/比较/计时及固定获取方式 |
| 同例 schema/、starter/、partial-seed/ | R0 root.yaml/dispatch.md、R0 方法 fixture、不可执行骨架与明示人工来源的非空 partial seed |
| 同例 evaluator/ | golden.py、baseline.py、input.py、check.py、timing.py、entry.py、manifest.json；统一完整 evaluator 与已支持 adapter |
| 同例 references/、requests/、run_demo.py | 固定来源清单/可分发资料、唯一五类请求模板（含 workflow_objective）、至多一次 optimize 并消费合法后续固定行动的有限 driver |
| 同例 evidence/<capture-id>/ | 选定 request/主体/原始来源、latest/qualified C、构建/调用/用量、正确性/布局、原始计时、独立合报告、writing_review、result/index 及完整实际依赖 |
| docs/workflows/ | quickstart.md、support.md、extract-to-implement.md、evidence.md、release-notes.md、cases/fa4_decode-admission.md；重跑/历史回放、有限支持/缺项 |
| 发行根及能力目录 | pyproject.toml、固定源码/profile/资源 manifest、许可/THIRD_PARTY_NOTICES/checksums；三个能力 README 链接唯一命令与 examples |
| AKA 发行与使用材料 | upstream/fork/KS/adapter/profile 固定版本与可访问来源；可选包 ownership/metadata、源码安装/资源定位及 fresh-child 证据；on/off switch、fixed/动态命令与自身产物恢复说明；适用配对证据索引 |

只承诺源码 checkout/editable；安装分 CPU 核心、模型运行时、GPU/Agate 三层。核心不装 AKA、不导入 torch/CUDA；在另一 cwd 定位规则/Template/角色/参考，不靠作者目录或 sys.path 顺序。认证由用户环境提供，不分发 home/凭证。
必要字节给可访问固定位置及摘要；项目/第三方许可未明确则取得授权或换合法可用材料后发布，不能默认全库可再分发。
AKA 的默认安装/off 不需要 KS；on 的文档化安装明确所需 gitlink、包和选中资源，使用 AKA 自有入口并实际验证，只取得必要固定内容。安装源码不等于 Python 包/资源已就绪；冻结运行的恢复不得临时联网补齐缺失来源。首发不承诺 wheel、无 `.git` 的任意分发形式或全仓递归安装。

| 发布对象 | 边界与可声明内容 |
| --- | --- |
| 公开说明包 | 可含脱敏 L′、导出说明和原件摘要关联；能解释结果，但不声称 L′ 满足旧 Ref 或原证据完整可导入 |
| 可导入产物包 | 显式列选定 Schema/source/证据根；保留稳定 prefix、原路径/字节/摘要及实际闭包。选中证据 Q/Task/R/C/方法等主体可定位；无关私有历史不必带 |
| 私有完整归档 | 保存原运行事实、私有日志/runtime/定位；按 C 支持本地恢复，不承诺迁机或将私有路径变成公共依赖 |

若证据真依赖不可公开 L，其完整证据闭包不能作为公开可导入包；可以公开合法说明，或只发布不依赖 L 的 Schema/C。不得删除失败、时间、实际用量/案例来美化结果。
无 GPU 回放只读取历史 evidence、核摘要、解释结果和查看源码，展示原运行时间/profile并标“没有新生成或测量”，不产生新 pass。README 按 P 的门逐能力链接实际证据，不另维护一套发布时间表。

<a id="consumption"></a>
## 7. extract 消费模式与运行入口

extract 的以下三种使用方式属于产品信息边界，均通过显式 Task、seed 和角色资料 allowlist 构造固定请求；不要求重做已有提取。

| extract 消费方式 | 信息边界与结论 |
| --- | --- |
| 原码复验 | 原字节、非空准确 seed、verify_only；只说明旧代码当前资格，无新生成/设计传递结论 |
| 原码 seed 续改 | 显式准确 seed 且继续成本全计；说明原实现适配/优化能力，不称纯 Schema 重建 |
| 纯 Schema 重建 | 同骨架，禁被提取优化源码/annotated/旧会话/二进制进入资料和工具；仅在明示信息边界内解释，不声称模型未预训练见过公开实现 |

消费过程的历史初始化费用与当前请求用量按 C 分列；所有运行保留实际失败、未知、费用和计时。正常 Task 计时只说明当前交付候选在固定测量条件下的观测，不由一次运行推导普遍收益。

最终交付须给以下两条可复制、已验证的使用路径，具体命令/参数由相应 owner 随实现固定到 quickstart，不以本计划中的占位命令代替可运行入口。

| 路径 | 最终使用验收 |
| --- | --- |
| 当前 AKA 的 Schema on/off | 按 A3/A7 自有说明安装固定可选 KS；同一 AKA 入口显式开关，off 通过 V12，on 在 A6 中按阶段调 planner 并实际消费合法后续行动。fixed 记录和动态记录分别可追溯；只通过 fixed 不算本次迭代目标完成 |
| 独立 KernelSchema 迭代 | 无 AKA 安装下运行有限 driver，接通 design 或 extract 消费→implement→至多一次 optimize→合法后续固定行动/keep 停止；V09/V13 的请求、代码、资格、同父用量与恢复证据闭合。真实 keep 无行动是合法终态，未发生分支只声明其 CPU 工程验证 |

F1 和跨后端支持仍依据 V15、V14 与主计划的独立发布门报告；任何缺项不由另一条路径的成功代签。
