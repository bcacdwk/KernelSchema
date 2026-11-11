能完整表达，而且顺序更清楚。一次准备、逐组主体和最终收尾直接排列；`for(z)` 每次完整执行 QK 与读出，再进入下一半，单个分数缓冲区的复用关系容易追踪。收益主要是去掉包装层、明确循环作用域，动作粒度没有改变。本稿覆盖原 root 的全部 `main.Main`，按 `pipe-control-v0.2` 编写。

| phase.role | 静态动作 | for | if | while | 最大控制嵌套深度 |
|---|---:|---:|---:|---:|---:|
| main.Main | 15 | 2（g、z） | 0 | 0 | 2 |

- 固定源码为 manifest 的 `6e4ca91bd91d998cac99e8dadd83838e153c679d`；原 root 与六个冻结文件的哈希均吻合。[Model.forward:246–257](sources/extract_fmha/solution.py#L246) 调扩展入口，快速入口在 [fmha_forward_fast:2350](sources/extract_fmha/fmha_mxfp8_qkv.cu#L2350) 转到 `fmha_mxfp8_forward`。该函数确定计划与 MapsValid；[launch_sequence:1890–1899](sources/extract_fmha/fmha_mxfp8_qkv.cu#L1890) 选 `launch_tcg<128,PASS_FUSED,1,true>`，[launch_tcg:1183–1195](sources/extract_fmha/fmha_mxfp8_qkv.cu#L1183) 进入 TMA 内核。由此确认 D128、QK64、单 warpgroup、DPV=true；该路径没有下一组 QK 提前发射。
- 完整域保持 `g:[cLo,cHi+1)`、`z:[0,2)`。数学、调度、资源、成员、执行模式和动作契约全部继承；原 `Main` 没有显式 `waits_for`，本稿也不提供替换。十五个动作模板及展开索引身份逐项相同，没有删除控制域或新增依赖。非零起点尤其不能重编号为零：`g` 仍是实际 key-group，槽号才按 `(g-cLo)%2` 计算。首轮判据始终是当前组等于区间下界；初始最大值与分母仍在一次准备中建立。首轮不读取未初始化的分子，第一次矩阵收缩使用覆盖语义，避免把旧内存中的非有限值带入状态。末尾动作直接引用其继承契约，也不借用已经退出作用域的循环变量。
- 每组先发下一组 K 的预取，等待当前 K；随后严格执行 `QK(g,0)→read(g,0)→QK(g,1)→read(g,1)`。两半完成后，`retire:Previous(g)` 才退休上一组 PV，接着发 ValueNext、等待当前 V、更新状态和概率、发当前 PV。这个位置同时保护 V 环槽、单份 V 尺度、概率和分子状态。两半分数同时保存在寄存器中，共同形成整个键组的最大值，状态更新每组只发生一次。第二半覆盖分数缓冲区前，动作内部先确认第一半的全部读者已完成；因此保留分量循环也保留了关键的读取完成边界。列表顺序只安排发起；可读性、读者完成和存储发布仍由 land、read、retire 等动作中的等待与屏障证明，循环边界没有排空含义。
- 保留首组 `retire:Previous(cLo)`，其契约是不读取上一组操作数的空操作；也保留末组 Head 与 ValueNext 两个预取事件，其契约是不搬运载荷。三者仍各有 issue/done。若在 pipe 外加条件删掉这些动作，事件身份就改变了；因此这里不增加 if。最终只有一个 PV 尚未退休，由显式 drain 等待，之后才输出并释放 TMEM。

下表位置均来自冻结的 `fmha_tcg.cuh`；主函数为 `fmha_tcg_kernel`，搬运辅助函数为 `tcg_issue_stage`。

| 对照点 | 函数内源码位置 |
|---|---|
| DPV 条件、首组 K/V 发起 | [3004–3005](sources/extract_fmha/fmha_tcg.cuh#L3004)、[3211–3227](sources/extract_fmha/fmha_tcg.cuh#L3211) |
| Q 预读、分配、写入、初始状态 | [3283–3364](sources/extract_fmha/fmha_tcg.cuh#L3283)、[3475–3476](sources/extract_fmha/fmha_tcg.cuh#L3475) |
| g 与槽号、K 预取/落地、两半 QK/read | [4462–4468](sources/extract_fmha/fmha_tcg.cuh#L4462)、[4543–4662](sources/extract_fmha/fmha_tcg.cuh#L4543)、[4708–4765](sources/extract_fmha/fmha_tcg.cuh#L4708) |
| retire、ValueNext/land、advance、PV | [4899–4922](sources/extract_fmha/fmha_tcg.cuh#L4899)、[4962–5094](sources/extract_fmha/fmha_tcg.cuh#L4962) |
| 最后等待、输出读写、LSE、释放 | [5173–5178](sources/extract_fmha/fmha_tcg.cuh#L5173)、[5260–5266](sources/extract_fmha/fmha_tcg.cuh#L5260)、[5334–5353](sources/extract_fmha/fmha_tcg.cuh#L5334)、[5490–5499](sources/extract_fmha/fmha_tcg.cuh#L5490) |
| K/V 各自的载荷与尺度事务计数 | [tcg_issue_stage:2121–2169](sources/extract_fmha/fmha_tcg.cuh#L2121) |

- **while 未使用，也不需要。** g 和 z 是已确定的有序遍历，没有动作反馈决定下一轮工作。[mbar_wait:259–268](sources/extract_fmha/fmha_tcg_ptx.cuh#L259) 的设备轮询仍属于等待原语内部；把它提升到 pipe 会改变本轮约定的动作粒度，不能说明现有流水更清楚。
- 定向检查解析 YAML、逐项比较新旧动作展开，核对完整角色覆盖、单键列表结构、两半交替、退休位置和首尾动作。比较覆盖每个位置的动作名称与索引，因此相同动作集合中的次序变化也会被发现；首组与末组的空操作另作身份检查。每组展开十一项，合计 `6+11*Ng`；下表均通过。计数包含空操作，是动作发生数，不是矩阵指令数。检查没有运行 GPU，也不构成数值或硬件异步完成证明。

| cLo / Ng | 展开动作数 | 边界核对 |
|---|---:|---|
| 0 / 1 | 17 | 首 retire 空操作；两个末轮预取空载；一个 PV 由 drain 退休 |
| 0 / 2 | 28 | g=1 两半读完后退休 PV(0)；最后等待 PV(1) |
| 0 / 3 | 39 | 两次非空 retire；槽序列 0、1、0；最后等待 PV(2) |
| 5 / 3 | 39 | 动作组号保持 5、6、7；槽仍从 0 开始并回绕 |
| 5 / 0 | 6 | 仅域外语法展开：循环为空，一次准备与最终收尾仍保留 |

- **Ng=0 不在原 root 适用域。** 原数学明确 `Ng>0`；零组检查只说明空 for 不删除兄弟动作，不声称其 prime/drain 契约有效。源码在首发与最后 PV 等待处使用 `any_compute` 或 `cLo<=cHi` 保护；若扩展到空区间，需另行定义动作契约及输出语义，不能用本表冒充空输入支持。
- 动作内仍包含四次原生 QK 收缩、四次 PV 收缩、掩码与最大值计算、概率量化和分母归约、32-channel 分子读改写及输出、TMA 事务屏障与相位、线程协作和 TMEM 分配释放；可选 LSE 与无效行也沿用内部条件。本次只验证既有抽象的顺序保持，未发现该 root 范围内的源码时序冲突。
