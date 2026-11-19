from pathlib import Path
B=Path('schema_wiki/aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16')
def rep(s,a,b):
 assert a in s,a[:100]
 return s.replace(a,b)
for p in B.glob('*.yaml'):
 s=p.read_text()
 s=s.replace('describe: all segments resolve n, start, chunks, seg_lo, nseg and slot0 from input cu/chunk_offsets; pass A initializes aprod to one','describe: "all segments resolve n, start, chunks, seg_lo, nseg and slot0 from input cu/chunk_offsets; pass A initializes aprod to one"')
 s=rep(s,'per-thread rhs = beta*(v - ai*pred) with pred merging the even/odd accumulator pairs','per-thread rhs = beta*(v - ai*pred), merging pred_acc even/odd pairs')
 s=rep(s,'with qpred pair-merged, staged bf16','with qpred_acc pair-merged, staged bf16')
 s=rep(s,'even/odd fp32 accumulators; yields pred/qpred','even/odd fp32 sets in pred_acc/qpred_acc yield pred/qpred')
 if p.name=='segmented.yaml':
  s=rep(s,'even/odd fp32 accumulators; yields pred','even/odd fp32 sets in pred_acc yield pred')
  s=rep(s,'single-term exact roundtrip (bitwise)','single-term transfer with no omitted intervening segment; same-VTT finite state arithmetic is unchanged')
  s=rep(s,'and the bitwise seg_count=2 composition claim, which relies on pass A and pass C evaluating identical products','and same-VTT seg_count=2 transfer, relying on matching pass-A and pass-C state products')
  s=rep(s,'shape: [{parity: 2}, {row: 64}, {value: 16}],\n            dtype: fp32, total_size: 8192','shape: [{parity: 2}, {row: 64}, {value: 16}],\n            dtype: fp32, total_size: 8192')
  s=rep(s,'sreg: {meaning: fp32 authoritative state columns of this tile','sreg: {meaning: pass-local FP32 state; c indexes pc(seg_lo+c) in A or sc(seg_lo+c) in C')
  s=rep(s,'b_seg(sg) = pc(seg_hi(sg));  a_seg(sg) = prod_{c=seg_lo(sg)}^{seg_hi(sg)-1} ai_c(63), 1 if empty.','b_seg(sg) = pc(seg_hi(sg));  a_seg(sg) = prod_{c=seg_lo(sg)}^{seg_hi(sg)-1} ai_c(63), 1 if empty; a_seg is physically replicated for every 16-column value tile.')
  s=rep(s,'amax > theta widens segment 0 to nseg(0) = chunks(r) (bitwise monolithic incl final_state) and sg > 0 produce nothing.','amax > theta widens segment 0 to nseg(0) = chunks(r) (the same-VTT monolithic sequence incl final_state) and sg > 0 produce nothing. An empty segment has a_seg=1 and forces this fallback.')
 else:
  s=rep(s,'average chunks per request < 8 (depth-2 ring)','floor((total_tokens+63*request_count)/(64*request_count)) < 8 (depth-2 ring)')
 # Preserve interface gaps in the actual binding description, without changing the Contract.
 needle='  Require:\n'
 entry='    - Item: CONTRACT.Algorithm_requirement\n      Choice: preserve the declared nullable metadata and caller-buffer writeback; the bound source only synthesizes chunk metadata when cu_seqlens is None and writes core through core_->contiguous()\n      Reason: source coverage gaps remain when cu is present but ci/co are None, or a noncontiguous core requires copyback; these do not narrow the Contract\n'
 s=rep(s,needle,needle+entry)
 p.write_text(s)
p=B/'dispatch.md';s=p.read_text()
s=rep(s,'fp32 累加）算 gram=K@K^T 与 qk=Q@K^T','fp32 累加）算 gram=bf16(K@K^T) 与保留 FP32 的 qk=Q@K^T')
s=rep(s,'（状态平面、Tinv、rhs、u、Qkd）','（gram、system、状态平面、Tinv、rhs、u、du、Qkd）')
s=rep(s,'（含同流序安全论证）','（同流串行复用条件）')
s=rep(s,'（工作区尾垫、pass A 网格完成后 pass C 才读）','（perm 尾垫之后的独立 transfers 区、pass A 网格完成后 pass C 才读）')
s=rep(s,'host 实现要求 cu_seqlens 为张量时 chunk_indices/chunk_offsets 必须一并提供，Contract 允许的“cu_seqlens 为张量而 chunk 元数据为 None”组合被拒绝，实现接受域窄于 Contract 有效域。','host 实现要求 cu_seqlens 为张量时 chunk_indices/chunk_offsets 必须一并提供，Contract 允许的“cu_seqlens 为张量而 chunk 元数据为 None”组合被拒绝。此 None 分支依据 reference.Model.forward 的默认参数及其完全不读取 ci/co、仅按 cu 切块的行为；标准非 None 枚举依据 input.py。候选对非连续 core 调用 contiguous()，之后没有 copyback，可丢失调用者写回；这些是实现覆盖缺口，Contract 的可空分支与前缀写回要求保持不变。')
s=rep(s,'host 分派只依赖张量元数据与运行时查询的 sm_count、L2 字节数，无 D2H 同步。','host 分派只依赖张量元数据与运行时查询的 sm_count、L2 字节数，无 D2H 同步。problem/input.py 的 prepare 是调用前的样本状态构造（32 个历史 token），不属于 Model.forward 计时计算；编译主要在 Model 构造阶段，forward 内仍含连续化、内部缓存、tensor-map 查询和结果分配。')
s=rep(s,'G>=3 为 ~2^-24 组合类加 theta 有界的跨段 delta 反馈误差，该数值类别已在节点内如实表达','G>=3 除 FP32 结合顺序外，还把含 k/beta 反馈的矩阵状态转移替换为标量 a_seg；theta 只筛掉弱衰减，不提供 Contract 全域误差界')
s=rep(s,'sg>0 先 compose_state 按 i 降序组合','sg>0 先 compose_state 按 i 降序近似组合')
s=rep(s,'包含性零填下行 >=1 的全部量恰为零/单位，输出与完整路径逐位一致，输出生产者与工作归属不变。','当中间运算有限时，beta 零填使无效 rhs/u/du 为零，退化计算具有相同的单项乘积；源码的逐位主张不能覆盖溢出或非有限中间量。输出生产者与工作归属不变。')
s=rep(s,'smem 随增一槽；stage_v','smem 改为 165,016B；pred_acc/qpred_acc 保留奇偶两组但 value 轴改为 16，各占 8,192B，sreg 为 8,192B；stage_v')
s=rep(s,'stage 环深度恒 2 同 root。','stage 环深度恒 2 同 root，dynamic smem 为 151,696B；pred_acc/qpred_acc 的 parity 轴各从 2 改 1、value 轴改 64，各仍 16,384B，sreg 为 32,768B。final_state 的 stage-0 暂存为 [128][65] FP32，末字节 33275，跨 stage_kq 的 32768B 并进入 stage_v 首部 508B；这些 v 读取已在 rhs/B1 后结束，K/Q 读取在 B3 前结束，不另分配存储。')
s=rep(s,'（Realization 的 123024B 预算相应上调）','（vt32 的 dynamic smem 从 123,024B 改为 171,160B）')
s=rep(s,'compute:decay_stats 用 gram 期间空闲 warp（tid 128-191）预算全部头的 cumsum(b)/rho','compute:decay_stats 用 tid 128-191 在 K/Q box 在途期间计算全部头的 cumsum(b)/rho，生产 E6 路径在 gram 前全 CTA 汇合')
s=rep(s,'（不再是 165,016B）','（151,696B）')
s=rep(s,'（G=2 组合逐位、G>=3 为 ~2^-24 组合类，源码注在 c8/c1 认证）','（相同 VTT 的 G=2 不省略跨段反馈，G>=3 有标量转移近似；c8/c1 的源码认证注释不构成全域误差保证）')
anchor='## 完整设计\n'
extra='''当前数值边界：g/beta 等未额外约束的浮点输入遵循 Contract 的任意有限值默认域，生成样本不隐含 |g| 或 |b| 的上界。零填 g/beta、TMA 边界零填和 i<Lc 输出掩码都有实际代码依据，但有限输入仍可能在归约、指数或状态乘积中溢出；0*inf 不会被 beta=0 消除。64 行 shuffle 扫描的零填尾部也未保证与最后有效 b 逐位一致。两稿据此保留实际扫描、BF16 RNE 舍入和未知的全域误差，不把 chunk 结果声明成 FP32 reference 的精确值。分段 G>=3 的 amax<=1e-3 门只看衰减积；它没有测量 k/beta 反馈或状态幅度，省略项并非仅 ~2^-24 的 FP32 重排误差。短请求若产生空段，pass A 发布 a_seg=1，pass C 由 segment 0 重跑整链，其他段退出；没有新输出空洞。

来源验收边界：阈值来自固定 problem/metadata.json 的 benchmark_contract.correctness_tolerances.default（atol=.01、rtol=.05、max_mismatch_fraction=0），可变输入观察对象由 problem/input.py 的 observable_inputs/writable_views 指定为 core 前缀；外部比较器本体及完整逐输出比较公式仍未随材料提供，未采用 DOC1 的默认判据来补齐。performance/CERTIFIED_BEST.json 记录候选 fixed/domain/reverse_fixed 三项有限作业通过；metadata.validation 的 numerical_failure 记录的是原库，两者不能互相覆盖。源码内注释的速度与精度数字是其所述实验记录，不是当前完整域实测。持久 workspace 与两份独立 norm 缓冲仅有同流复用顺序，源码没有跨流并发调用隔离；out/final 则是同一次新分配的互不重叠视图。所有已登记 Storage 的数量按每 CTA 总量计，不按每 warp 或每 thread 再乘；寄存器逻辑账不声称等于编译器最终寄存器占用。

'''
s=rep(s,anchor,extra+anchor)
p.write_text(s)
p=Path('schema_discussion/schema_audit_and_cleanup_20261119/reviews/gdn/cpu_checks.py');s=p.read_text().replace('be=np.float32(.731)','be=np.float32(.1)').replace('assert float(gram)!=float(gramh)','assert float(gram)!=float(gramh) and float(with_round)!=float(without)');p.write_text(s)
