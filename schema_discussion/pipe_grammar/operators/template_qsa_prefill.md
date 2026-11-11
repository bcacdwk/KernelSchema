template_qsa_prefill 已核对唯一 root、7 个精确 pipe 组和全部 7 个角色上下文。当前模板与冻结输入同 hash；语义依据归档规则。A_decl／B_local／C_effective 均为 7 roles、12 blocks、5 nested、40 静态动作，最大深度 2；无继承或别名增量。动作数不展开运行时循环。逐组证据见 [review JSON](../reviews/template_qsa_prefill.json)。

| phase／role | blocks | nested | actions | pipe 的职责 |
| --- | ---: | ---: | ---: | --- |
| preprocess／Main | 1 | 0 | 2 | 单份工作的构建、压紧 |
| attention／Scheduler | 1 | 0 | 4 | 首票与后继票发布、确认 |
| attention／GatherK | 2 | 1 | 4 | 逐工作装入 K，worker 尾部排空 |
| attention／GatherV | 2 | 1 | 4 | 逐工作装入 V，worker 尾部排空 |
| attention／MMA | 2 | 1 | 9 | 工作内 QK 填充、QK/PV 稳态及 PV 排空 |
| attention／SoftmaxA | 2 | 1 | 10 | A 半块发布、Q 接续、输出 |
| attention／SoftmaxB | 2 | 1 | 7 | B 半块发布及输出 |

十二块中，三块同时具有首段、重复主体和尾段，四块有首段及重复主体，四块仅有重复主体，一块仅有不重复的主体。五个列表混合固定动作与子循环，没有相邻子循环；首段、主体、尾段分别登记十一、二十五、四个动作。这些数量描述文本声明及出现位置，不衡量工作量，也不能据此估算重叠程度或硬件耗时。

两个 phase 的实际 ID 是 preprocess、attention；build 是 `build:Hash` 的动作标识。Main 无 repeat，仅 steady 顺序列出 `build:Hash`、`compact:Union_Count`。清零、索引预取、插入会合、压紧写出和 Count 发布位于契约；每 CTA 预分配一份工作、attention 等 preprocess kernel 完成则由 schedule 给出。[root:152](../../../schema_template/examples/qsa_prefill/root.yaml#L152)、[root:209](../../../schema_template/examples/qsa_prefill/root.yaml#L209)。

attention 的顶层 repeat:t 覆盖 worker 生命周期。Scheduler 首次发布并等待 Ticket0，随后每份工作发布并等待 Next(t)；五个消费者先各自 accept:Ticket0，再在每份工作主体之后 accept:Next(t)，其中包括终止票。确认对象是十一 consumer warps 的票据副本，允许领先一份工作，并不表示旧工作全部完成。停止条件还包含在途访问排空。[root:241](../../../schema_template/examples/qsa_prefill/root.yaml#L241)。

四个嵌套块仅含 repeat+steady：GatherK/V 遍历 n[t] 装入，SoftmaxA/B 遍历 n[t] 分别发布 HalfMax、P。它们界定每份工作内的 tile 遍历，把循环后的收票、归一化、输出留在外层 steady。另一个嵌套块是 MMA 的完整三段：两次 QK 填充；n[t,0:count(i)-2] 中先发 QK(n+2)，再发 PV(n)；最后两次 PV 排空。这里内层 prologue/epilogue 每份工作执行，顶层 prologue 才是 worker 一次初始化。[root:299](../../../schema_template/examples/qsa_prefill/root.yaml#L299)。

n 是 flatten:[t,i] 的连续编号，双槽 KV 按 n%2、S/P 按 n%3 接续。Gather 当前 load 的 done 等下一 tile 的 issue，延迟发布跨 worktile 保留，最终由顶层 epilogue drain 交出；地址预取却在每份工作内初始化并钳制边界。这些差异靠 domain、契约和 sync/reuse 边共同确定，仅看内层列表无法读出。[root:216](../../../schema_template/examples/qsa_prefill/root.yaml#L216)、[root:257](../../../schema_template/examples/qsa_prefill/root.yaml#L257)。

SoftmaxA/B 是列区域分工，各占四 warp，并非独立递推。双方先捕获 S 的各半，再经 HalfMax 会合和对方发布形成 P 两半；MMA 显式等待两半。A 还负责下一 Q 的异步 prefetch，随后与 B 分别归一化、drain 各半 Y。drain 完成只归还 Y，尾片仍在寄存器等待 gate/store；下一工作首 PV 等两半归还。这些交接位于外层 steady，并非顶层 epilogue。[root:313](../../../schema_template/examples/qsa_prefill/root.yaml#L313)、[root:320](../../../schema_template/examples/qsa_prefill/root.yaml#L320)。

当前 pipe 没有显式条件域。i>0 且 move_ref 为真的 A warp 等待前一 PV、修正全部 D 列，写在 publish 契约；mask 预取、真实行输出和终止票 no-op 也不能由 n[t] 推出。N(b)>=32 保证 MMA 首尾索引存在；worker 首票仍可终止，此时 t 为空，但首票确认、A 的初始 Q 发布、MMA 的初始 Q 等待及两个 drain 事件仍存在，drain 无数据时不等末 tile。[root:89](../../../schema_template/examples/qsa_prefill/root.yaml#L89)、[root:258](../../../schema_template/examples/qsa_prefill/root.yaml#L258)。

顺序还取决于执行模式。Gather 按列表发射而允许旧搬运在途，收票动作的串行契约只约束自身完成；MMA 的累积完成包含操作数读取；Softmax 的串行推进中又单独含异步预取。数据可读、环槽可覆盖、票据已确认分别由不同依赖承担。操作数同名同索引且生产来源唯一时有默认等待，P 的两半发布则有显式边。移位前驱越出声明窗口时不生成该项依赖；这与空重复不产生主体动作、终止动作无载荷仍保留事件，是三种不同边界。

已知边界包括 HalfMax 覆盖前所有读者完成尚未证明、未写入的 padding 对共享修正 vote 的影响尚未确定，以及近似数值误差界缺失。S/P 别名复用的操作数读取保证来自契约及 realization，不能仅从列表推断。7 个精确组、6 个形状组的指纹均不含这些外围条件，不代表语义等价；本报告不验证 GPU 正确性。[root:88](../../../schema_template/examples/qsa_prefill/root.yaml#L88)、[root:327](../../../schema_template/examples/qsa_prefill/root.yaml#L327)。
