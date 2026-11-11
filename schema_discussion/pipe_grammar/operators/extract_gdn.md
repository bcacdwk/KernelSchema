# extract_gdn pipe 现状

范围为首批冻结 `inputs/extract_gdn` 的全部 6 个节点；统计取统一数据，语义以冻结微流水线文档为准。A（声明）、B（本地展开）、C（有效展开）相同：14 roles、36 blocks，其中 22 nested、180 静态 actions，最大深度 3；没有 pipeline 继承。所有角色均为 `Main`、`warp[0:8]`、默认 `serial`。计数不展开运行次数，也不代表耗时。

| 节点 | roles | blocks / nested | actions |
| --- | ---: | ---: | ---: |
| root | 2 | 5 / 3 | 26 |
| cached_ordered | 2 | 5 / 3 | 27 |
| direct_qk | 2 | 5 / 3 | 25 |
| direct_qk_ordered | 2 | 5 / 3 | 24 |
| segmented | 3 | 8 / 5 | 38 |
| segmented_ordered | 3 | 8 / 5 | 40 |

9 个精确组全部核对如下；表中哈希接在 `extract_gdn:pipe:` 后。相同 pipe 只表示块和动作字符串相同。

| 哈希 | 节点及 phase | 静态 actions / 出现数 |
| --- | --- | ---: |
| f953ca80acf829fe | root、cached_ordered、segmented、segmented_ordered：prep | 15 / 4 |
| fc9711ebbdc918ed | direct_qk、direct_qk_ordered：prep | 14 / 2 |
| 6c8ded194ceb5d7d | root、direct_qk：chain | 11 / 2 |
| bd710c88bd4fca76 | cached_ordered：chain | 12 / 1 |
| 140c70fe693e3ce2 | direct_qk_ordered：chain | 10 / 1 |
| ba1f6170745a8e8a | segmented：pass_a | 11 / 1 |
| 995cf8df9d6298ae | segmented_ordered：pass_a | 12 / 1 |
| 77edd46a16ca9551 | segmented：pass_c | 12 / 1 |
| a66c1ed456820b4f | segmented_ordered：pass_c | 13 / 1 |

22 个 repeat 都引用域名：`a`、`r` 各 6，`c`、`cl` 各 4，`run` 2。`a=[0,Active)` 是有效 slot 的 0/1 门控；`r` 遍历一至两头组成的组，承接共享 Q/K/F 与 scratch 复用。`segmented_ordered` 的 `GroupWidth=HeadsPerPrep`，相同 `r` 写法只走一组。非分段 `c` 遍历本请求全部 chunk；pass A 的 `cl` 对应段内 `lo_z+cl`；pass C 则对应 `StartChunk+cl`。这些域没有承担持久 worker 的下一份 worktile，schedule 均为每 CTA 预分配一份、无 refill。

pass C 的 `run=[0,Run)` 同样是门控：weak 时只有 z=0 重走整请求，其余段在分类后退出；非 weak 时各段重走自己的区间。原例取自 [segmented.yaml:666][seg-run]：

```yaml
- repeat: run
  steady:
    - "prime:Stages"
    - "seed:R"
    - repeat: cl
      steady: ["advance:Stage(cl)", "project:KS_QS(cl)", "rhs:Hrhs(cl)", "solve:U(cl)", "finish_chunk:Out_R(cl)"]
    - "final_store:FinalState"
```

这里 prime/seed 与 final_store 都位于 steady，却分别是活动遍历的初始化与终结。prep 的 skeleton 也在 `a.steady`、`r` 之前，初始化整套 inverse scratch。所有 22 个子块仅含 repeat+steady；8 张列表混合动作和循环，来自 6 个 prep 与 2 个 pass C。没有相邻独立循环，也没有 prologue/epilogue 内嵌循环。pass A 的空段仍完成 seed 并在 epilogue 发布 SegB/SegA；pass C 的空 run 则不产生 prime/seed/final_store。

组外契约决定实际用途。缓存 prep 有 save_image/drain_image，直接 Q/K prep 以无传输的 finish_prep 收尾；前者 4 处有显式 drain 等待边，6 个 prep 均将 readout_coeff 标为 async。ordered 的排序藏在 stage_inputs，rank 经等待后解析为原请求，静态顺序不保证 CTA 完成顺序。root 与 direct_qk 共用 chain pipe，但 root 可在 PRE0 提前装首块，后续读缓存图像；direct_qk 在等待后直接读取 Q/K。cached_ordered 显列 warm/resolve，direct_qk_ordered 则把预热、等待及重读合在 join_prep，初态加载移入 seed。

分段的两个 pass 属于不同批次的 CTA。前者只做状态递推，不装载查询及读出系数；后者等待摘要后再做带输出的遍历。摘要与准备数据都必须保持到后者读完。阶段间早发射只表示允许后继启动，数据可见性仍依赖网格完成等待。块内部同样区分发出和到达：prime 的完成承诺是装载发出，当前槽真正可读要经过 advance 的获取与协作屏障；这个屏障也结束上一块的共享读取，然后才发未来块。输出与右端项共用存储，最后状态另借用第零槽，生命周期约束跨越多个动作。默认串行顺序没有替代这些已写明的协作关系。

少量大动作仍包含多重局部过程：stage_inputs 的扫描及子集协作、gram 的归约、solve8 的列内代入、merge16/32 的局部块，以及 final_store 的 `VTT/8` 轮均未计为 pipe repeat。宽度 32 与 64 的合并次序、KS/QS 的分累加器次序、full/mini 分支也由 math/契约承载；mini 的 solve 保留事件但无读写。segmented 的 seed 内部逆序合成前段；ordered 限定 G=2，该中间段循环为空。因此角色数量不能作为它较模板 GDN 简单的依据。

这里的表达边界是：块树能直接看见初始化包围哪些循环，却不能单凭叶动作数读出内部归约、扫描和同步的数量。相同动作列表下，数学仍区分不同乘法结合次序、舍入位置和多累加器合并位置；相同域名下，上界又可能来自空段、整请求或单头组。统计中的符号上界标志只反映写法，并非互斥用途标签。冻结文档提供的是这些依赖和资源约束，不是实测重叠时间线。

现状问题与不确定项分开看：4 个非分段 chain 及 segmented.pass_c 明写末尾 mini chunk、最终状态写回且槽号回到 0 时，stage-0 写可能与其他 warp 的 Kc 行读取竞争；把它们列成一个 finish_chunk 并没有提供读退出保证。segmented_ordered 限定 VTT=16，无此 mini 分支。来源另明确未给出近似指数及组合输出误差界，segmented 的 0.001 测试也无统一精度保证；这不是本次运行失败结论。所有节点的非连续 core 输出写临时区却不回拷，是 math 已记录的输出缺陷，不能从 final_store 字符串判断。完整来源定位及 14 个角色上下文见 [审计记录][review]；本次没有执行 GPU 或提取。

[seg-run]: ../../../schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/segmented.yaml#L666
[review]: ../reviews/extract_gdn.json
