# template_gdn_prefill：pipe 现状审计

两节点的 pipe 把工作生命周期、成对准备、逐 chunk 递推和标量前瞻放在不同层次；动作内部还包含矩阵分块、舍入及协作。已核对全部 **2 节点、14 个 role 使用位置、11 个精确组、4 个 phase 上下文**，两份现行 YAML 与冻结输入相同，无 alias 或管线继承，A/B/C 三口径相同。

| 范围 | role | 块 | 嵌套块 | 静态动作 |
| --- | ---: | ---: | ---: | ---: |
| root | 7 | 18 | 11 | 42 |
| m64_single_sequence | 7 | 18 | 11 | 37 |
| 合计 | 14 | 36 | 22 | 79 |

深度 1/2/3 为 14/16/6；22 个嵌套块全为 `repeat+steady`，另有两个无 repeat 的 Convert 块。显式 prologue/epilogue 均为零；4 个列表含相邻循环，共6对，7个列表混合动作与循环。数字计书写位置，不展开运行次数；完整组与上下文见 [review](../reviews/template_gdn_prefill.json)，口径见 [census](../data/operators/template_gdn_prefill.json)。

两节点均先运行 `pipe: {steady: ["produce:Cu32"]}`。root 用 Cu32 确定 packed 序列；single 仍执行转换，但核心不读取结果。root 选择 B>1，single 选择 B=1；每工作负责一序列、一 value head 的64行半片，两半不合并。`t` 是 worker 的本地工作序号，`n={flatten:[t,c]}` 使环索引跨工作连续；`n[t]` 限定当前工作。single 的“每 CTA 一工作”对应记录的148-SM设备，外层调度仍保留参数 P（[single:84](../../../schema_template/examples/gdn_prefill/m64_single_sequence.yaml#L84)）。

`C=64`、`Np=2*ceil(L_b/(2*C))` 和正序列长度使 Np 为至少2的偶数。MMA0 按 `t→p` 发射两个 KK、两个 QK；Inverse 按 `t→p→j`，其中 `p:[0,Np/2]`、局部 `j:[0,2]` 选择配对的 `2*p+j`。补齐 chunk 仍有动作和事件，数学中的中性填充值、输出边界共同说明其用途。Load 按 `t→n[t]` 装 K/Q/V；root 每 t 另有 `bound:QKV(t)` 更新末界，single 使用固定 T 描述符。K/Q 归还需同时考虑两个 issuer；异步列表顺序不等于每次装载完成后才继续。

每个工作的块数都是偶数，因此连续 n 的奇偶与本工作 c 的配对一致；复用边中的 `n+1-n%2` 指向该对后块。此关系需要结合外层域及填充公式才能读懂。

Inverse 的原文局部块包括：

```yaml
{repeat: {j: [0, 2]}, steady: ['read:KK(n[t,2*p+j])', 'compute:A(n[t,2*p+j])']}
{repeat: {j: [0, 2]}, steady: ['publish:Tinv(n[t,2*p+j])']}
{repeat: {j: [0, 2]}, steady: ['read:QK(n[t,2*p+j])', 'publish:Bmat(n[t,2*p+j])']}
```

三块在原列表中依次出现，第一、第二块之间是 `invert:Pair(t,p)`：先形成两份 A，再成对求逆、交出两份 Tinv，最后捕获 QK 并发布 Bmat。KK/QK 捕获后提前归还累加槽，W 留到 Bmat；row Beta 已捕获，column Beta 留到 Tinv。root 把 `capture:BetaPair(t,p)` 单列，single 合在 prepare。单个 invert 内部还有8→16→32→64层、层间 rendezvous，以及每 chunk 两 warp 和16层四 warp的不同分工；这些不计作 pipe 子块（[root:233](../../../schema_template/examples/gdn_prefill/root.yaml#L233)）。

三个局部循环各自重新绑定同名索引，分别承担构造反馈矩阵、交出逆矩阵、处理读出系数的遍历。第一个循环全部完成后才进入成对求逆；后两个相邻循环也保持“两个逆矩阵都已发布，再处理读出系数”的顺序。矩阵本身的行列维度、对角小块和合并宽度没有成为这三个循环的迭代轴。角色共享同一个工作域也不表示锁步：装载、逆矩阵准备和状态递推各自推进，完成与复用边限定它们相遇的位置。

MMA1 两节点 pipe 完全相同：每 n 依次 KS、QS、U、Y、S；KS/QS/U 零起算，Y/S 分别累加到 QSscaled/Sdec。root 等独立 Ubar、Udec 发布；single 的 UpdateOperand 一次使两者就绪，Y 之后的 S 无第二次发布等待。同组因此不等于相同契约。Recurrence 在 t 的 steady 内先 `load:initial_state(t)`，再遍历 n，末尾 `store:S_final(t)`；这是每工作初末，不是 worker 仅一次。相邻 chunk 的 S 完成约束下一 StateOperand，末存储明确等本工作最后 S；初态为零仍加载。StateOperand 一次读取兼作快照及衰减，U 捕获后产生两种更新操作数，O 发布归还 Y 槽，均需结合契约理解。

这里的初末动作绑定外层工作，循环内动作绑定连续块号；本工作最后一次状态更新与下一工作的初态不是一条数学递推。输出发布还要等待矩阵乘完成，下一块覆盖唯一共享输出槽又要等待前一块存储完成。数学区分别记录残差的两次舍入、更新值的两种变换和浮点状态保留，单看动作数量无法知道计算精度或每个动作包含多少次访存。

ScalarOutput 的三个局部 j 域原样为 `[0, 'min(4,Np)']`、`[0, 'max(0,Np-4)']`、`['max(0,Np-4)', Np]`：先准备最多4块，中段先准备 j+4 再存 j，最后仅存剩余块。Np=2/4 时中段为空；首尾仍存在于 steady。root 分别发布 Gate、装 Beta，五槽归还分别等待两个 gate 读者和 Tinv；single 联合发布 Scalar，`prepare(c+5)` 在 `store:O(c)` 完成之后，输出进度传递保护旧读者。single 的 prime/prepare 内还含 raw 双槽、四 head 载入、取 h%4、捕获后预取 c+2，再计算前缀；外层三段不足以独立读出这些次序（[single:228](../../../schema_template/examples/gdn_prefill/m64_single_sequence.yaml#L228)）。

当前阅读边界是：层数不能代表内部计算或同步数，精确组不包含域、调度和动作契约，未显式列出的默认/传递依赖仍参与执行。root 已注明旧在途请求跨描述符更新的顺序尚未建立；本轮未补足此证明，也未做运行或性能验证。历史语义仅采用冻结 DOC5；以上均为现状、用途与不确定性。
