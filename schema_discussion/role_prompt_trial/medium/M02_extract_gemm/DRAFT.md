# M02：grouped FP8 scaled GEMM 文字试走

本稿是 medium 工作档位的手工 P1/P2 试走，不是通过验收的 schema。只读规范与指定 problem/candidate；没有读旧 Design、dispatch 或其他试走报告，没有运行 Python、提取器、checker、编译或 GPU。P1 没有独立批准，按本次明确授权继续做 P2 判断；以下不冒充正式阶段放行。

源码根路径记为 `O = kernel_wiki/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm`。工作树 HEAD 为 `0c662ebcf2cb57d6b417c12498c3adbf39071a44`；限定到 O/problem、O/candidate 的 git status 无修改输出。正式来源可定位到此提交下的 `O/candidate/solution.py::Model.forward`，各机制另以实际 CUDA 入口佐证。没有借用 metadata 所写 FlashInfer commit 冒充 candidate 的版本。

## P1：已经确认的任务事实

- 输入：A `[M,K]`、B `[E,N,K]` 为 e4m3fn；SA `[K/128,M]`、SB `[E,K/128,N/128]` 为 FP32；indptr `[E+1]` 为 int32。生成器传连续 scales 和 payload，并传 caller-owned BF16 out `[M,N]`。依据：`problem/input.py:53–74`、`reference.py:17–22`。
- 每组 `e` 对 indptr 给出的所有物理行计算。令 `q=floor(k/128)`、`v=floor(n/128)`，参考关系为 `Adec[m,k]=A[m,k]*SA[q,m]`，`Bdec[e,n,k]=B[e,n,k]*SB[e,q,v]`，`Out[m,n]=bf16(sum_k Adec[m,k]*Bdec[e,n,k])`。reference 先用 float 张量反量化、后 einsum；Model 禁用 TF32。没有从源码注释推导“FP8 块累加精确”或逐位保证。
- `m_indptr` 是 counts 的前缀和；空组跳过。`shape_range.json` 限定 M≤65536、E∈{8,16,32,64}、每组 0..8192 行且对齐4，以及五种相关 N/K 对。M 实际为各组行数和。`logical_group_sizes` 不产生额外掩码，`mode` 没有传入 forward；不能把 psum 标签再拆成另一外部任务。
- 输出是返回值与原地写入的同一个 out。`input.py:76–88` 只登记 out 为可写和可观测区域；metadata 要求非写区域保留。reference `Model.forward:30–31` 做 out.copy_ 后返回 out。
- `problem/solution.py` 是调用 FlashInfer 的 baseline；`reference.py::Model.forward/expected` 才是此处可读的参考计算；candidate 是独立 CUDA 实现。reference 中 uint8/alpha 分支没有对应本题生成入口，暂不扩成 FP4 或 alpha Contract。
- 已知验收声明：metadata `benchmark_contract.correctness_tolerances.default` 给 atol=.01、rtol=.05、max_mismatch_fraction=0。problem 内检索未见 comparator 的相对误差基准、完整公式、比较精度或非有限值处理。按 DOC1 来源验收记录保留这些事实，不套默认简写公式。它不妨碍目前机制判断。

P1 未完成：完整输入属性／固定参数登记、可省略 out/out_dtype 等 API 分支的任务有效域、所有 alias 约束没有落实到 Contract；见 M02-01。没有填硬件区，也没有把 metadata 的 sampled validation 当成本 candidate 的正确性证明。

## P2：入口、主要机制与暂定去处

`candidate/solution.py::_Engines` 在可加载 UMMA 时走 `fp8umma_run`，否则加载 `fp8gg_run`。两条是实际入口，不只是源码遗留函数。bind/ctypes/Python memo 改变 host 路径；成功 hit 仍调用 CUDA 入口计算。UMMA 内还有 host sched_policy、tensor-map cache、单 kernel node graph cache 和错误回退。它们必须保留调用条件，但不能按入口函数或 graph API 次数计设计数量。

### 拟选完整代表（均仅计划）

| 候选 | 具体实例与独特完整价值 | 暂定判断 |
| --- | --- | --- |
| `root.yaml` | `fp8gg.cu::fp8gg_bm64`，BM64/BN128/BK128、3槽 cp.async、256线程；8 warp 各自读 fragment、mma.sync、块 scale promotion；输出经共享存储重排后直接写 global。每个 CTA 静态步长连续处理 tiles。 | 选作 root。这个真实后备方案没有 UMMA 的多种 host 机制开关，方便先贯通数学、工作与输出；root 不必是主要性能路径。只承担该后备方案所支持的调用条件，不自动接管目录未覆盖范围。 |
| `pair_umma.yaml` | `fp8gg_umma.cu::umma_gemm<0>`，BM128/BN128/BK128，3个 pair stage，每 stage 放两块 K；4个128列 TMEM accumulator 组成两对；TMA producer、MMA issuer、128线程 promotion/epilogue 分工。先选单 issuer、A scale 两槽 cp.async 的真实模式。 | 保留完整候选。相对 root，主计算结果改在 TMEM 形成并跨角色交接，搬运和输出协议、跨工作流水均需贯通说明，不能只称 tile 参数变化。dual 的去处暂见 M02-03。 |
| `wide_umma.yaml` | `umma_wide<4,32768,0>`，BM128/BN256/BK128、4个48KB stage、2个256列 TMEM槽；每轮一块 K；8 warp epilogue 分两列组，各128列，输出复用32KB半 tile staging顺序写两半。 | 保留完整候选。并非仅 BN 加倍：pair 改为单 K 轮次，累加槽版本和 release 节奏改变，scale 对应两个 N块，主要角色与输出分段都需重新串联。 |

依据：sync `105–297,306–345`；pair `304–523,785–1063`；wide `1110–1398`。三个方案都在每个 K128 块内计算原始 FP8 dot，再乘 SA*SB 加到 FP32 fin，最后 BF16；没有 split-K 的跨 CTA 部分结果或外部归并 kernel。

这只是三份明确有完整价值的拟集合，不声称“正确数量就是3”。dual 的分类、自适应 A box 的正文身份及缓存调用边界仍未完全解决，因此本次不声明 P2 可正式放行。

### 参数、局部、组合及非实际入口

1. **wide 整 tile 单遍写回：局部，暂不保留第4份 YAML。** `umma_wide<3,65536,1>` 相对 wide 基线，用64KB staging一次 pack四象限、发四个 TMA store，替换32KB staging按 h=0,1串行复用；同时输入 stage 4→3。输入分配、K轮次、MMA、fin、epilogue成员和每元素舍入未改。完整差额可以指向现有 OutputStaging、Pack/Store、源槽释放及输入环 depth；这不是仅一条 depth 参数。源码 `1101–1109,1313–1391,1631–1632,1707–1762`。
2. **dual issuer：暂倾向局部，保留疑问，不硬降级。** 基线一个 issuer 顺序发 pair 两个 accumulator；dual 中 warp1负责 kb0、warp2负责 kb1，各发 commit，empty barrier计数1→2；producer和epilogue仍按原 pair等待两块完成。能写出对应值、role和完成汇合的差额，但这是主循环中长期协作变化，common 的边界警句让我不敢直接认定局部。见 M02-03。
3. **B 的 evict_first：局部缓存策略。** `pp=(mtiles==ne)` 后在两种UMMA中选 TMA B load hint；不产生新的值或输出生产者。适用条件和缓存命中造成的旧策略必须如实记录，不据注释保证真实访存只发生一次。`1640–1646,398–403,1186–1187`。
4. **A 搬运 box 高度：去处未定。** 主 stage 容量固定，按当前 tile 有效行选择16/32/64/128的搬运box和事务字节；只屏蔽无效输出行。但它也在源码中选择不同 tensor-map，读 DOC0 后不明白应算尾块执行还是配置分支。见 M02-02；没有枚举四个完整设计。
5. **host graph／direct launch、bind／ctypes、identity memo：暂记调用实现与局部条件，不逐设备机制配对。** graph create/add/instantiate不执行主计算，成功 graph launch 提交一次；正常 direct 分支也一次。异常回退是否一定排除了先前在途工作，尚无这里可读的 API 依据，不声称所有失败都无副作用。见 M02-05。
6. **SAREG=1和诊断宏：不当作当前正常候选。** build 的 flags 没有 FORCE_ENGINE/FORCE_SAREG/FORCE_DUAL、V0_8EPI 等宏；sched_policy当前正常路径固定sareg=0（初始化也是0）。头部历史注释仍提旧gate，但 `1572–1583` 实际赋值已改变。可保留“源码含诊断/旧实例”范围说明，不能因为模板实例被设置属性就声称 Model 正常选到它。

还没有写正式 dispatch 格式或给未来文件建链接；局部记录中的目标名字目前是试拟对象，不冒充已经存在的 YAML 定义。正式写作后必须落到基线实际对象。

## 最小表达试写

以下是为了检验能否表达而抽出的片段，不是可检查的完整 YAML；没有补齐的数学值、动作声明不能由本片段隐含提供。

### root 的工作归属

把组内 M tile 与 N tile 映射到一个扁平 `w`：`T_e=ceil_div(indptr[e+1]-indptr[e],64)`，`P_e=sum_{h<e} T_h`，`W=P_E*(N/128)`；唯一 e 满足 `P_e*(N/128)<=w<P_(e+1)*(N/128)`；`r=w-P_e*(N/128)`，`nt=floor(r/T_e)`、`mt=r%T_e`。空组没有 w。每CTA收到 `w=worker_id+t*worker_count<W`，不是动态队列领取。

```yaml
flow:
  launch:
    - step: [MAIN]
```

```yaml
partition:
  w: {meaning: Flattened group output tile, size: W, tile: 1}
```

```yaml
worktile_order:
  - {by: w, direction: ascending}
worker_assignment:
  preassigned: {start: worker_id, stride: worker_count}
  refill: none
  exit: assigned_done
```

源码确实在device端读取indptr并算prefix。DOC4容许运行时size表达式；我采用确定映射的静态分配，不因为host未先读实际W就改成atomic refill。worker_count 为 `max(1,min((ceil_div(M,64)+E)*(N/128),2*nSM))`，nSM来自candidate host查询；这个非Contract量怎样在硬件占位豁免下正式绑定尚不明确，见 M02-04。

### pair 的跨工作连续环

拟定义 Pairs=ceil_div(K/128,2)，t 是本worker的worktile本地序号，j为pair局部序号，g为跨工作连续pair序号。固定源码stage、pj没有在每个w重置；不能用局部j%3给输入ring索引。

```yaml
index:
  t: {meaning: Local worktile ordinal, definition: worktile}
  j: {meaning: Pair within one worktile, definition: [0, Pairs]}
  g: {meaning: Pair ordinal across worktiles, definition: {flatten: [t, j]}}
```

Producer片段（PairPayload含当前pair有效的A/B块；Matmul PairDot是两块独立dot的组合动作，done保证两块已算完且读完stage）：

```yaml
pipe:
  - for(t):
      - for(g[t]):
          - load:PairPayload(g)
waiting:
  released:
    - "{issue load:PairPayload(g+3)} ← [{done MMA::matmul:PairDot(g)}]"
```

MMA还必须等待本pair payload ready，以及 `g-2` 的TMEM读者release；这里没有写全。片段说明 `flatten` 足以表达静态持久工作，不需要while。没有把原始bar_spin写成缺少退出意义的循环。

输出staging的源读取完成和global写入完成不同：代码 `bulk_wait_group_read0` 允许下一工作覆盖staging，最终 `bulk_wait_group0` 才保证写回完成。DOC5“事件引用”的 OSlot 例子已直接提供 store/release 分离写法，暂时疑问已自查解决。完整pipe仍要处理全tile与tail交替；不能草率写成每个t都等待前一个t的store，因为tail可能根本没有TMA store事件。

## 明确停止处

已完成 bounded 主机制调查和代表的实质比较；P1只保留确认事实与缺口。P2仍有可影响分类的疑问，没有独立reviewer，也没有正式YAML、完整动作表、所有缓存错误路径的API语义核实或源码全覆盖。未进入完整P3–P6。不列GPU正确性或实测为待办；当前工具未对接是已知实施边界，未用它阻断手工推演。
