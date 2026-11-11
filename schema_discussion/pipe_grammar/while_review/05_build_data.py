#!/usr/bin/env python3
"""Freeze the fifth audit's selected evidence; no CUDA or source mutation.

This is an evidence index, not a C++ parser or a semantic loop census.
Line ranges are inclusive. Fragment hashes retain the original line endings.
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OPS = Path("kernel_wiki/aka-cuda-operators-20261030/operators")
CATALOG = Path("kernel_wiki/catalog.json")
INDEX = HERE.relative_to(ROOT) / "data/broad_lexical_candidates.json"
OUT = HERE / "data/05_broad_kernels.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def evidence(relative, start, end, purpose):
    path = OPS / relative
    data = (ROOT / path).read_bytes()
    lines = data.splitlines(keepends=True)
    assert 1 <= start <= end <= len(lines), (path, start, end, len(lines))
    return {
        "path": str(path),
        "start_line": start,
        "end_line": end,
        "fragment_sha256": digest(b"".join(lines[start - 1:end])),
        "purpose": purpose,
    }


def ev(op, filename, start, end, purpose):
    return evidence(f"{op}/candidate/{filename}", start, end, purpose)


def case(cid, operators, classification, selection, predicate, actions,
         progress, rewrite, boundary, excerpts, scope="mechanism_and_route"):
    return {
        "id": cid,
        "operators": operators,
        "classification": classification,
        "review_scope": scope,
        "branch_selection": selection,
        "predicate_provenance": predicate,
        "body_actions_and_handoffs": actions,
        "termination_or_progress_basis": progress,
        "natural_for_if_or_await_form": rewrite,
        "abstraction_boundary_argument": boundary,
        "requires_design_pipe_while_in_this_path": False,
        "source_evidence": excerpts,
    }


bg = "gemm/grouped/bf16_grouped_gemm"
bm = "moe/bf16_fused_moe"
fm = "moe/fp8_mxfp4_fused_moe"
mm = "moe/mxfp8_fused_moe"
sp = "sparse_attention/dual_cache_latent_fp8"
sb = "sparse_attention/dual_cache_latent_bf16"
sx = "sparse_attention/paged_indexed_latent_fp8"
sm = "sparse_attention/paged_latent_mxfp8_nvfp4"
fa = "fmha/fmha_fp8"
fh = "fmha/fmha_fp8_per_token_head"
gc = "linear_attention/gdn_conv_mtp_bf16_state_bf16"
gp = "linear_attention/gdn_postconv_mtp_bf16_state_fp32"
gf = "linear_attention/gdn_prefill_bf16_state_fp32"
kd = "linear_attention/kda_raw_prefill_state_bf16"
mb = "mla/mla_expanded_bf16"
mf = "mla/mla_expanded_fp8"
ma = "mla/mla_absorbed_fp8"
mn = "mla/mla_nvfp4"

cases = [
    case(
        "numeric_rescale_scan_for_if", ["fmha_mxfp8_qkv"],
        "numeric_state_feedback_with_a_fixed_input_traversal",
        "当前 TC split 路径可在 scan2p gate 下选择 fmha_scan_kernel；B1/B2 是单独编译开关。本次审阅所列 scan，未覆盖所有 FMHA_SCAN_* 组合。",
        "segment maxima 与上一段状态 m 的差是否超过 RESCALE_THRESHOLD，确实是数值结果决定的局部条件；外层 sbase 仍以固定 stride 遍历输入 nseg，内部扫描覆盖给定 [t0,t1)。jmp 是 C++ 位图变量，非 PTX 跳转指令。",
        "读取 segment maxima -> 计算 jump mask -> if(mask==0) 写常值状态，else 写前缀后扫描后缀；scan_walk 逐输入 tile 更新 m 并可 store；最终写 mtail。",
        "segment/tile 数由 T/nseg 给定，m 不改变外层 sbase 或 T；没有因某轮算数失败重做相同多动作段。",
        "保留已有 for(segments) 与 if(数值 predicate)，并保留各分支不同的读写事件；不以强制全扫描替代原跳过行为。",
        "这是数值局部状态反馈也未必需要 while 的具体对照；无需否认数据依赖条件，也无需将跨段状态塞入调度伪域。",
        [ev("fmha/fmha_mxfp8_qkv","fmha_tc.cuh",1342,1356,"有限区间内的数值状态更新"),
         ev("fmha/fmha_mxfp8_qkv","fmha_tc.cuh",1406,1427,"外层 input segment 域"),
         ev("fmha/fmha_mxfp8_qkv","fmha_tc.cuh",1464,1497,"数值 jump mask 控制 if，非循环退出"),
         ev("fmha/fmha_mxfp8_qkv","fmha_tc.cuh",1534,1568,"B1 同类数值条件"),
         ev("fmha/fmha_mxfp8_qkv","fmha_mxfp8_qkv.cu",1639,1676,"scan2p 的 launch 分支")],
        "selected_branch_mechanism_route_not_exhaustive"),
    case(
        "grouped_gemm_scheduler", ["bf16_grouped_gemm"],
        "finite_metadata_tile_traversal",
        "Model.forward 经 dispatch 选择 MGroupedContiguous 或 MGroupedContiguousWithPsumLayout；同一 helper 中 MGroupedMasked/KGrouped 分支在本题未选。",
        "next_block_idx=(++current_iter)*kNumSMs+blockIdx.x；PSUM/grouped_layout 决定各组 M tile 数，current_group_idx 只向前走。GEMM 累加值不进入判定。",
        "TMA producer、MMA issuer、epilogue 各遍历同一 tile 次序；每个 tile 内仍有 load、wait、MMA、归还及 store。helper 的内层 while 只计算组号/坐标。",
        "输入组数 kNumGroups 与各组有效前缀给出自然有限域；组号及 tile 序号单调增加。",
        "for(按原栅格顺序的 group/M/N worktile)，在 tile 内保留原 load/wait/MMA/store；组定位作为 metadata mapping。",
        "Scheduler 已是坐标调度对象，不是为删除 while 临时发明的反馈容器；不得把 producer/MMA/epilogue 合并成一个动作。",
        [ev(bg,"dispatch.cuh",1023,1028,"实际 GEMM 类型选择"),
         ev(bg,"kernel_helpers.cuh",576,596,"输入 shape / grouped_layout 初始化"),
         ev(bg,"kernel_helpers.cuh",677,738,"get_next_block 与三种组定位分支"),
         ev(bg,"kernel.cuh",182,224,"调度器和 TMA producer"),
         ev(bg,"kernel.cuh",278,301,"MMA task loop"),
         ev(bg,"kernel.cuh",386,408,"epilogue task loop")]),
    case(
        "masked_gemm_stage_walk", ["fp8_mxfp4_grouped_gemm_masked"],
        "finite_metadata_and_ring_stage_traversal",
        "kernel.cu 的 STAIL 调度与 slot-constant MMA 编译分支；未闭合所有 FMGGM_* 编译组合的选择，仅审阅所列分支。",
        "pos=++cur_iter*num_sms+worker_id，positions/split_from 来自真实 tile 总量；cur_group 来自 masked_m；kb<nk_ 和 stage!=0 来自 K tile/环位置。",
        "组定位没有跨 role 通知；MMA 首段对齐环、中段固定 STG、尾段处理余数，三段都调用原 MMA_STAGE。",
        "group 与 pos 单调前进；kb 每轮递增至 num_k_blocks，环对齐至多 STG-1 个真实 K tile。",
        "for(实际有效 worktiles)；K 域拆为到下一环边界、整环、尾部三段或普通 for(k)。",
        "上限来自 K 维度与实际剩余工作，不是任意大 cap；仍需逐 stage 保留等待及提交动作。",
        [ev("gemm/grouped_masked/fp8_mxfp4_grouped_gemm_masked","kernel.cu",1038,1066,"positions 的来源"),
         ev("gemm/grouped_masked/fp8_mxfp4_grouped_gemm_masked","kernel.cu",1101,1122,"调度谓词"),
         ev("gemm/grouped_masked/fp8_mxfp4_grouped_gemm_masked","kernel.cu",2130,2152,"环对齐/尾段 while 与普通 K for")],
        "selected_branch_mechanism_route_not_exhaustive"),
    case(
        "reference_moe_task_schedule", ["bf16_fused_moe","fp8_mxfp4_fused_moe","mxfp8_fused_moe"],
        "real_dynamic_work_schedule_with_explicit_handoffs",
        "BF16 kernel.cuh 与两种 FP8 normal kernel.cuh 是当前入口注册/dispatch 路径；FP8 的 kernel_situ.cuh 虽登记在 resolver，所查 dispatch 只 load normal tag。三题合同均明确 single-rank，不视为多 GPU 已验收。",
        "先等 dispatch 发布完整 expert_recv_count；num_total_m_blocks=各专家 ceil(tokens/BLOCK_M) 之和。L1/L2 分别 atomicAdd 领取 tile idx；有效性是 idx<固定层 tile 总数。warmup/交替层选择来自计数和形状，不读 FC1/FC2 的数值。",
        "独立 scheduler warp：await task-info empty -> claim -> publish_task；consumer receive full -> metadata copy -> TMA/MMA/epilogue；epilogue 在 TMEM full 后 release_task_info；最后 scheduler 显式 publish None sentinel。",
        "每个成功 claim 消耗已有 L1 或 L2 tile。L2 的等待条件是所需 L1 已被领取，数据另由 full/mask 等待保护；没有 enqueue、requeue 或由结果生成新 tile。进展仍依赖 role 公平调度与 ring 容量/代际约束。",
        "for(work_schedule 实际派给 worker 的 task 序列)，保留 await-empty 在 claim 之前、publication、各 consumer receive、epilogue release，以及无效 claim/终止票的尾部动作。不能让 for 头执行 claim，不能丢掉最后一次 empty wait。",
        "任务单位原本就是独立 TaskInfo{phase,expert,m,n_cluster,pool,shape}；源码已有 MegaMoEScheduler、专用 warp、两个共享 slot 及全局 L1/L2 分配器。这里归到 work_schedule 有现成职责与载体，不是把本地数值反馈移走。",
        [ev(bm,"solution.py",29,35,"BF16 源码 tag"),
         ev(bm,"kernel_helpers.cuh",1286,1319,"TaskInfo 单位及 sentinel"),
         ev(bm,"kernel_helpers.cuh",1436,1447,"接收/归还独立动作"),
         ev(bm,"kernel_helpers.cuh",1474,1495,"调度域来自收到的路由计数"),
         ev(bm,"kernel_helpers.cuh",1534,1587,"领取、层选择、L1 领取依赖和发布"),
         ev(bm,"kernel_helpers.cuh",1589,1645,"共享任务、routed 主循环及终止票"),
         ev(bm,"kernel.cuh",650,709,"token loader 对 task 的消费"),
         ev(bm,"kernel.cuh",770,830,"MMA 对 task 的消费"),
         ev(bm,"kernel.cuh",838,891,"scheduler role 与 epilogue release"),
         ev(fm,"dispatch.cuh",1525,1540,"实际 normal tag load"),
         ev(mm,"dispatch.cuh",1413,1428,"实际 normal tag load"),
         ev(fm,"kernel_helpers.cuh",1474,1645,"与 BF16 该片段字节相同"),
         ev(mm,"kernel_helpers.cuh",1474,1645,"与 BF16 该片段字节相同")]),
    case(
        "reference_moe_topk_reduce", ["bf16_fused_moe","fp8_mxfp4_fused_moe","mxfp8_fused_moe"],
        "finite_input_bitset_with_double_buffered_actions",
        "正常 combine epilogue；三题都在选择的 normal kernel 中；shared-expert 项受输入配置控制。",
        "total_mask=ballot(input_topk_idx>=0，另可含一个 shared-expert 项)；move_mask_and_load 用 ffs 取最低位并清位，do_reduce 只表示刚发起的那一份是否存在。reduced[] 的计算值不会改变 mask。",
        "先 await 路由 peer 的 L2 ready，首项 TMA load；每轮 prefetch 下一项、wait 当前 load、FP32 accumulate、切换 phase/stage；最后 cast/store token chunk。",
        "每次成功 move 清除一个输入位，最多 popcount(total_mask) 个贡献；空 mask 仍执行后续零结果的 cast/store。",
        "先 if(nonempty) 发首项 load；for(按 ffs 次序的有效 top-k slots)：if(has_next) 发下一项 load -> await 当前 load -> accumulate -> 原 phase/stage 更新；保留末尾 cast/store。",
        "该 domain 由本轮计算开始前的输入路由定义，不能用‘成功为止的轮数’替代。load 与 accumulate 均留在 pipe，完全不需要吞并成 ReduceAll。",
        [ev(bm,"kernel.cuh",1242,1287,"初始位图、peer await、首项发起"),
         ev(bm,"kernel.cuh",1289,1337,"do_reduce、累加和最终 store"),
         ev(fm,"kernel.cuh",1448,1492,"FP8 normal 同机制"),
         ev(mm,"kernel.cuh",1448,1492,"MXFP8 normal 同机制")]),
    case(
        "reference_moe_waits_and_rank_mapping", ["bf16_fused_moe","fp8_mxfp4_fused_moe","mxfp8_fused_moe"],
        "primitive_readiness_waits_and_finite_metadata_mapping",
        "正常 routing、token loader 与 epilogue；generic 多 rank min-peeling 代码被读到，但验收合同只有 rank=1。wait_until 有超时断言，不是正常退出。",
        "pending_mask=ld_acq(mask_ptr) XOR generation_expected；L1/L2 count 是 producer/consumer 完成数。rank min-peeling 的 remaining 来自已收齐的 per-rank token counts，非 GEMM 数值。",
        "pending_mask/count while 体仅重新读就绪状态；rank peeling 在产生 token 地址前只做整数/warp metadata 运算。数据搬运、L1 full 发布、L2 empty 归还都在这些循环之外，必须保留。",
        "等待由相应 producer 发布/consumer 归还推动，不给轮询次数虚设上限；min-peeling 每次未命中至少去掉一个非空 rank，合法 token 一定落入某层。",
        "await 对应 full/empty/readiness；MapToken 计算有限路由 metadata 映射；for(已有 token 域) 发起 pull 和通知。",
        "把原语内部 load-acquire 轮询归入 await 不会吞掉独立通知；rank 元数据映射本来就在 dispatch 地址选择层。",
        [ev(bm,"kernel_helpers.cuh",1335,1361,"L2 K readiness 的发布与等待"),
         ev(bm,"kernel.cuh",425,508,"专家定位和 rank min-peeling"),
         ev(bm,"kernel.cuh",525,543,"L1 slot wait 与实际 pull/store 的分界"),
         ev(bm,"kernel.cuh",563,579,"L1 full 的独立发布"),
         ev(bm,"kernel.cuh",668,687,"L1 full/shared L2/mask waits"),
         ev(bm,"kernel.cuh",883,910,"release_task_info 与 L2 empty wait"),
         ev(bm,"kernel_helpers.cuh",873,885,"超时断言仅属失败诊断")]),
    case(
        "sparse_fp8_clc_work_stream", ["dual_cache_latent_fp8"],
        "hardware_scheduler_work_stream_not_local_compute_feedback",
        "runtime.py 可选 fp8_h8_h16_source_exact；SM103 source 的七个 role 完成谓词/交接审阅。SM100 对应七个循环只作同机制出口核对，不声称两个架构完整等价。",
        "首个 work 是 blockIdx；之后是 clusterlaunchcontrol.try_cancel 的响应 valid 与被接管 CTA 坐标。active_topk/num_steps 来自 sparse_topk_lens；attention 数值不影响 CLC。",
        "scheduler await throttle -> return throttle credit -> await response slot empty -> arm full -> try_cancel -> await full -> query response -> release work_empty；softmax/correction/MMA/page-loader/padding/KV-loader 各接收响应。invalid 响应也先归还再 break。",
        "成功取消一个尚未执行的既有 CTA 即消费一个网格工作；invalid 立即结束，没有失败后重新请求直至成功的循环，也没有新建 CTA/queue 回流。await 依赖真实 CLC completion 与角色 release。",
        "work_schedule 给出 initial CTA 加实际 CLC 接管序列；for(worktile) 内保持 load/QK/softmax/PV/store；独立表达 cancel request、completion、response-read、各角色 release 和 terminal response。",
        "真实工作单位就是 launch grid 中的 CTA，源码已有 scheduler role 与 hardware queue。动态序列不由 attention 的局部结果重新定义，故不是人为伪域。",
        [ev(sp,"runtime.py",161,202,"当前 entry 的 route 选择"),
         ev(sp,"build.py",574,609,"SM103 exact 源码登记"),
         ev(sp,"fp8_h8_h16_source_exact_sm_103a_kernel.cu",192,242,"softmax work 与有限 key tiles"),
         ev(sp,"fp8_h8_h16_source_exact_sm_103a_kernel.cu",469,534,"consumer response/return/invalid 次序"),
         ev(sp,"fp8_h8_h16_source_exact_sm_103a_kernel.cu",1239,1284,"scheduler request/full/valid"),
         ev(sp,"fp8_h8_h16_source_exact_sm_103a_kernel.cu",1321,1334,"scheduler 的 response slot return"),
         ev(sp,"fp8_h8_h16_source_exact_sm_103a_kernel.cu",1439,1466,"loader throttle 与 Q 信用")]),
    case(
        "gemm_clc_work_stream", ["fp8_scaled_gemm"],
        "hardware_scheduler_work_stream",
        "candidate/kernel.cuh 的正常 pipe_role/CLC 调度路径；未运行 launch。",
        "worker work>=0 与 CLC query_cancel.is_canceled；矩阵维度给出发射 tile 网格。",
        "promotion/AB/MMA/scale 是独立 role；scheduler await clc_empty -> expect bytes -> try_cancel -> await clc_full -> query valid。",
        "CLC 成功接管既有 grid tile；invalid 结束；各消费者的 release 推动 slot 再用。",
        "for(work_schedule 的实际 CLC worktiles)；保留 cancel 请求及 terminal response 所有动作。",
        "文件本来就有 PipeRole 与独立 warp==6 的 scheduler，不借调度名称掩盖数值反馈。",
        [ev("gemm/dense/fp8_scaled_gemm","kernel.cuh",270,298,"worker work 判定"),
         ev("gemm/dense/fp8_scaled_gemm","kernel.cuh",477,515,"原本的 role 以及 scheduler")]),
    case(
        "paged_indexed_existing_bounded_clc", ["paged_indexed_latent_fp8"],
        "hardware_scheduler_work_stream_with_real_launch_bound",
        "kernel_m64.cuh 的当前 CLC 路径，S4_SPLIT 分支随 launch；本次审阅 source work / loader 的边界，未逐一验证全部 MMA role。",
        "static_work<total_work_items+1 是 launch 工作数加 terminal 的真实上界；实际退出仍由 CLC valid；有效 key tiles 来自 min(seq_lens,sparse_topk_lens,sparse_topk)。",
        "索引读取/分页收集、Q/KV/TMA 和后续 attention 是已分开的动作；收到 CLC 响应后先归还 source_work_empty，再按 valid 退出或更新坐标。",
        "每次成功接管减少剩余 CTA；有限 launch grid 是实际工作域，非硬件状态空间 cap。",
        "当前 for 已能表达该路径；保留真实 worktile 域、empty wait 与 invalid response 处理，不把 total_work_items+1 泛化成任意反馈循环的模板。",
        "本题 while 词法命中为零，仍含动态工作接管；源码关键词不能代替设计控制分类。",
        [ev(sx,"kernel_m64.cuh",274,309,"实际 bound 和输入 key 域"),
         ev(sx,"kernel_m64.cuh",353,419,"响应读取、归还、退出"),
         ev(sx,"kernel_m64.cuh",2232,2335,"source scheduler 的 CLC 请求与出口")]),
    case(
        "fmha_persistent_pool", ["fmha_fp8"],
        "input_planned_pool_or_static_segment_traversal",
        "当前 host AUTO 默认 -1：MT 长链可走 dynamic pool、短链用 static；TC 路径用 static。tc_pers.cuh 虽留 dynamic 分支，但 host 在 tc_route 且 static 不可行时 flat 为空，回退普通 launch；不能把残留分支当默认选定。",
        "pool_aids 由 host 的 B/Hkv/query tiles/causal KV 几何构造。ticket=atomicAdd；ticket>=pers_kmax 返回 aid=-1。static 时 qa<seg_len。mma_qa_p/sm_read_ctr/pull_seq 只是角色 pacing。",
        "loader await 前一任务 metadata 已读 -> claim -> 写 pull_aid -> 发布 pull_seq；MMA/softmax await publication -> read -> ack；tile 内 QK/PV、softmax、输出和 buffer 归还保持独立。",
        "pool item 总量在 launch 前已定；每次 claim 前进一次；最后一个 terminal claim 重置计数器仅为下次 graph replay，本次任务不重入。",
        "static for(segment items)；dynamic for(实际 host pool 经 ticket allocator 分给本 worker 的 worktiles)，保留 claim/publication、MMA started/softmax read acknowledgements 及末票和 reset。",
        "真实 pool_aids、ticket_ctr 与 worker 分派已经存在；不能把一个包含所有事件的大 attention 动作用来藏循环。",
        [ev(fa,"fmha_fp8.cu",1879,1888,"AUTO route gate"),
         ev(fa,"fmha_fp8.cu",1923,1946,"host pool 元素构造"),
         ev(fa,"fmha_fp8.cu",1980,1989,"static/dynamic 选择"),
         ev(fa,"fmha_fp8.cu",2080,2108,"flat 的构造、TC 动态残留分支不可由这里进入"),
         ev(fa,"fmha_tcmt.cuh",1375,1409,"loader claim / publication / terminal"),
         ev(fa,"fmha_tcmt.cuh",1540,1562,"MMA receive / ack"),
         ev(fa,"fmha_tcmt.cuh",1644,1663,"softmax receive / ack，包括 terminal"),
         ev(fa,"fmha_tc_pers.cuh",148,183,"TC static/dynamic 源码分支"),
         ev(fa,"fmha_tc_pers.cuh",361,392,"TC MMA 接收"),
         ev(fa,"fmha_tc_pers.cuh",475,496,"TC softmax 接收")]),
    case(
        "sparse_bf16_union_build", ["dual_cache_latent_bf16"],
        "finite_input_collection_with_probe_retry_and_one_time_publication",
        "正常 packed tc_attn GT>1 的 DB=false 分支；DB=true 是 cfg65 对照分支。overlap_probe MODE0 是实际 host planner 的默认 metadata probe，MODE1 是非默认 variant；fuse_step 的实验融合默认 g_fuse_combine=0。",
        "items=sum(tokN)，原输入索引列表；for(;;) 的重试源是 atomicCAS 返回不同 key 时的 hash collision。相同 key 的 pos<0 等另一个 thread 完成插入。atomicOr 返回已占 token bit 时分配一次 duplicate entry，随后退出。",
        "冲突只换 slot；成功者只执行一次分配 dense pos -> 写 ukey/umask -> atomicExch 发布位置 -> break；命中者 await pos -> 更新 membership/必要的重复项 -> break。CTA barrier 后 nkeys 冻结，再进入 gather/QK/PV。",
        "TCUCAP=2560 限制输入 item 数，TCHT=8192 确保空 slot；线性探测有自然有限候选槽。位置等待依赖已占槽的插入者完成；重试轮数不伪设 cap。",
        "for(输入 items)：LocateOrReserveHashSlot（探测原语），if(owner) 分配/写/发布，else await 位置后更新；保留 barrier 再 for(完成的 union keys)。原子探测可保留内部 while，无需上升 pipe。",
        "不将整段 union build 吞成一个不透明动作；独立 publish/await 仍明确。重复反馈边只包围槽探测，不重试完成的多动作插入段；attention 计算不会再增加 union 工作。",
        [ev(sb,"dsv4_kernel.cu",1857,1861,"真实 hash/union 容量"),
         ev(sb,"dsv4_kernel.cu",2247,2249,"输入 item 数"),
         ev(sb,"dsv4_kernel.cu",2303,2379,"metadata 定位、CAS 重试、一次插入与位置等待"),
         ev(sb,"dsv4_kernel.cu",2446,2451,"完成 union 后固定 nkeys"),
         ev(sb,"dsv4_kernel.cu",3866,3875,"planner probe 的 item gate"),
         ev(sb,"dsv4_kernel.cu",3963,3997,"默认 probe 的 CAS / membership"),
         ev(sb,"dsv4_kernel.cu",5542,5551,"probe 的默认与非默认实例"),
         ev(sb,"dsv4_kernel.cu",5580,5590,"probe 是实际 host planning launch"),
         ev(sb,"dsv4_kernel.cu",5841,5848,"DB 仅 cfg65 分支"),
         ev(sb,"dsv4_kernel.cu",4795,4807,"packed 输入域的容量 gate")]),
    case(
        "sparse_split_merge_handoff", ["paged_latent_mxfp8_nvfp4"],
        "one_time_producer_consumer_wait_then_finite_merge",
        "当前 launch 在全部 split 可驻留且 one_cta_per_sm 成立时选 fuse_tc；否则独立 combine。catalog 对性能有 user waiver，不能理解为任意配置已验。",
        "count_acquire(arrive+peer)<splits，splits 是 host 选定 split 数；每 CTA 一次 count_release。",
        "先写其他 CTA 所需 partials -> fence -> arrive notify -> await 全部 arrived -> merge 自己的 head 范围 -> done notify；last done 将两个计数器清零。",
        "host 校验 BS*nhb*splits<=nsm 与单 CTA 驻留，source 的进展依据是每个 producer 均可运行且会到达一次；未作 GPU 公平性证明。",
        "notify / await / for(该 CTA 的 heads) merge / notify_done / if(last) reset，均单独表达。",
        "while 体没有重新发布或重算，只有获取同一个计数；不能为简化把 await 后的 merge 与 done/reset 合成等待原语。",
        [ev(sm,"launch.cu",369,387,"当前 fused gate"),
         ev(sm,"kernel_tcgen.cuh",532,577,"partial 发布、纯等待、merge、done/reset")]),
    case(
        "linear_fk_handoff", ["gdn_conv_mtp_bf16_state_bf16"],
        "one_time_conv_to_gdn_readiness_wait",
        "当前 FK hybrid 分支可选：vec==3、conv_vec、fk_flags!=nullptr；host 给 flags 的 gate 是 fk_conv_total<=160。",
        "__ldcg(fkflags[i])<cxb；conv block 每个完成后加一，与递推数值无关。",
        "conv 写 mixed/state -> fence -> arrival；GDN 可先 preload 初态/gates，再 await conv -> CTA barrier -> depart notify；last departed reset flags；后续 GDN 按输入 token 域递推。",
        "每个 conv 退出都发布且不等待 GDN；source 依赖较小 launch 和经验 block-order 驻留关系。本次不把注释中的 de facto 调度说成 CUDA 规范保证。",
        "保留独立 notify、await、depart、if(last) reset，然后 for(tokens)；不用 cap 模拟轮询。",
        "将纯读 flag 的自旋归 await 可行；FK_CONSUME 中归还及 reset 不能顺带吞入 await 的不透明实现。",
        [ev(gc,"solution.py",1253,1273,"source 自述协议与进展假设"),
         ev(gc,"solution.py",1377,1425,"实际 notify / wait / depart / reset"),
         ev(gc,"solution.py",1733,1744,"launch 分支"),
         ev(gc,"solution.py",1966,1982,"选定条件与 flags 分配")]),
    case(
        "linear_gdn_token_recurrence", ["gdn_postconv_mtp_bf16_state_fp32"],
        "value_recurrence_over_input_token_domain",
        "正常 post-conv MTP kernel；有效 token 范围由合同长度0..8/状态槽输入定义。",
        "num_tokens=cu_seqlens[request+1]-cu_seqlens[request]；t 递增。h、delta、decay、beta 虽逐 token 反馈，但都不修改 num_tokens。",
        "state decay、dot(h,k)、delta、state update、dot(h,q)、checkpoint store，后续 gated norm；checkpoint 写是独立可见输出。",
        "输入 token 数与 chunk 数是真实有限域；不存在 residual 收敛判断。",
        "维持现有 for(token)，显式索引每次 state/checkpoint 动作；不需 while。",
        "数值递推本身不是控制反馈；不能因为 h 更新就把 token 域叫‘成功直到收敛’。",
        [ev(gp,"kernel.cuh",298,317,"输入 token 域与无效分支"),
         ev(gp,"kernel.cuh",411,449,"数值递推"),
         ev(gp,"kernel.cuh",475,489,"每 token 的 checkpoint 输出")]),
    case(
        "linear_kda_chunk_recurrence", ["kda_raw_prefill_state_bf16"],
        "finite_metadata_mapping_and_value_recurrence",
        "Model.forward 根据并行度/checkpoint cadence 选择 engine 或 prepare+chain；两种路径的 outer chunk 谓词均已读。",
        "metadata 的 s<sequences && c>=chunks[s+1] 只定位 chunk 所属序列；engine chunk 数=ceil((eos-bos)/32)，chain=ceil((eos-bos)/16)。",
        "各 role 的 QK、state update、output 和 checkpoint 通过 barrier 交接；state 算数结果不会再生成 chunk 或改变退出。",
        "sequence 与 token/chunk 长度在输入中给定；checkpoint 条件是 token index mod cadence。",
        "for(sequence chunks) + if(checkpoint boundary)；metadata 搜索可留 mapping 动作内。",
        "engine 的 compute/epilogue 不是一个合并动作；已有固定 chunk 域覆盖循环控制，计算反馈由索引状态关系表达。",
        [ev(kd,"solution.py",98,124,"engine/chain 选择"),
         ev(kd,"preprocess.cu",5,28,"metadata chunk->sequence 查找"),
         ev(kd,"chain.cuh",580,607,"chain 输入 chunk 域"),
         ev(kd,"engine.cuh",850,864,"engine 输入 chunk 域"),
         ev(kd,"engine.cuh",1085,1100,"compute chunk loop 与 checkpoint 条件")]),
    case(
        "linear_gdn_prefill_fixup", ["gdn_prefill_bf16_state_fp32"],
        "finite_chunk_state_transfer_recurrence",
        "registry/plan 可选 state_fixup_simt_row4；审阅该 fixup 的外层控制，未展开全部 CP/MMA 变体。",
        "num_chunks=ceil((cu[end]-cu[start])/chunk_len)；next_chunk< num_chunks；没有状态范数/误差决定重复。",
        "按 chunk 读取 local transfer/local state，传递 carried state，写修正状态；prefetch next chunk 受真实边界约束。",
        "每轮处理一个输入序列分块，chunk_idx 单调。",
        "for(chunks)，保留状态传递和独立输出；已有源码即这样写。",
        "本题唯一 while 词法在 host plan 的二分搜索，不能据此忽略 device 数值递推，也不能反向把递推升级 while。",
        [ev(gf,"runtime.py",95,104,"实际 registry 文件"),
         ev(gf,"plan.py",149,162,"fixup 选择"),
         ev(gf,"fixup_simt.cu",72,89,"chunk 域"),
         ev(gf,"fixup_simt.cu",125,157,"state transfer 与下一块 prefetch"),
         ev(gf,"plan.py",89,110,"host 二分 plan 搜索")]),
    case(
        "mla_work_streams", ["mla_expanded_bf16","mla_expanded_fp8"],
        "finite_planned_item_pool_and_sentinel_handoffs",
        "BF16 tc2p/tc2p64 在 AUTO waves>=2 的一部分几何可选，tc2s/tc2c 已退到 force-only；FP8 当前 solution.py 有五个 for(;;) role loop。未以旧 trial hash 代替本次文件。",
        "BF16 atomicAdd(counter)<n_eff，decode_item 的 valid 由 cu_q/cu_kv 和 tile geometry 决定；FP8 pop+gridDim.x<P.items，geo_table 的 kv_tiles<0 标终止。",
        "发布 item geometry 与 NEXT/item_ready；各消费者 await/read/ack；原 QK/PV/output-buffer handoff 留在 per-item body；最后的 sentinel 必须发布后退出。",
        "有限 item table / geometry 定义工作；计数只增加；reset 属下次 launch。没有依赖 softmax 结果再生 item。",
        "for(实际分配的 item worktiles)，保留 publication/acks/terminal。BF16 内层空 item 跳过属于原 scheduler 的 metadata 筛选。",
        "源码有现存 task table/atomic pool/专用发布路径；不能把 NEXT/CONS 归还藏在域定义。",
        [ev(mb,"solution.py",1616,1645,"AUTO 与 force-only 状态"),
         ev(mb,"kernel_tc2p.cu",877,926,"领取/metadata 过滤/发布"),
         ev(mf,"solution.py",850,882,"FP8 worker 序列和 tile 域"),
         ev(mf,"solution.py",1117,1136,"correction 序列"),
         ev(mf,"solution.py",1263,1276,"issuer 序列"),
         ev(mf,"solution.py",1376,1387,"loader 序列"),
         ev(mf,"solution.py",1450,1484,"独立 pop/publish sentinel")]),
    case(
        "extra_for_headers", ["mxfp4_gemm","fmha_fp8_per_token_head","mla_absorbed_fp8"],
        "finite_tiles_or_metadata_scan_not_unbounded_feedback",
        "mxfp4 的 tloc 尾部重分配分支；FMHA prefill persist/fold 代码及一个旧 TC wait helper；MLA absorbed SIMT v0 分支在 !_USE_V1 时可选，不等于默认 V1。这里只闭合谓词与出口。",
        "tloc 根据 total_tiles/rI/rR 分段；FMHA which 根据真实 T_all ticket pool 或静态一/二 tile 结束；MLA b 查 cum_q 包含 row 的区间。",
        "有限工作 tile 的计算和发布保持原次序；metadata scan 不执行 tensor 计算。FMHA 的 timeout for 单独是 mbarrier polling。",
        "tile 尾域由实际 launch/shape 给定；合法 row 在 cum_q 的末前缀内；timeout 返回 false 是失败路径。",
        "for(原有限 tile/segment 域) 或原实际 ticket worktiles；metadata mapping；wait 留 await。",
        "这些有初始化/步进的无限 for 头被原 exact-for(;;) 索引遗漏，不能把遗漏误当成‘源码没有反馈’；所读出口仍未给出本地多动作重试。",
        [ev("gemm/dense/mxfp4_gemm","tcgen05_kernels.cuh",923,951,"真实 tile 尾域"),
         ev(fh,"fmha_pre.cu",1799,1835,"persist 与静态 fold 分支"),
         ev(fh,"fmha_pre.cu",3172,3202,"另一个有限/ticket 变体"),
         ev(fh,"fmha_pre.cu",3828,3851,"双组 persistent 变体"),
         ev(fh,"fmha_tc.cu",118,152,"轮询超时，不是正常计算终止"),
         ev(ma,"kernel_simt.cuh",29,55,"前缀索引映射"),
         ev(ma,"solution.py",159,160,"默认分支入口"),
         ev(ma,"solution.py",198,209,"可选 SIMT fallback")],
        "predicate_exit_review_selection_partial"),
    case(
        "host_planning_and_bit_conversion", ["mla_nvfp4","bf16_fused_moe","paged_latent_mxfp8_nvfp4"],
        "host_configuration_search_or_scalar_format_conversion",
        "正常 host planning / scalar conversion；不能当作 GPU pipe 的 tensor 反馈。",
        "num_bytes_per_pull 按 hidden bytes 缩半；MLA tiles_per_split/num_units 按 KV 几何/traffic 预算变化；FP16 subnormal m 左移至固定隐含位。",
        "host 计算参数/拼表，不发起每轮 tensor 段；转换循环只做 bit normalization。",
        "正整数严格缩小或位模式向高位移动；候选域来自形状/格式，非等待未知成功的伪域。",
        "保留 host 原算法；GPU action 参数在 launch 前给定。bit conversion 留在算术实现内部。",
        "主张核心 pipe 不需 while，并不要求把 host 或原语源码的 while 全部删除。",
        [ev(bm,"dispatch.cuh",936,946,"num_bytes_per_pull 是 host 配置"),
         ev(mn,"plan.py",140,180,"host split 粒度选择"),
         ev(mn,"plan.py",237,255,"有限分段规划"),
         ev(sm,"common.cuh",67,86,"scalar subnormal bit normalization")],
        "host_or_primitive_only"),
    case(
        "wait_macros_and_diagnostic_paths", ["bf16_nvfp4_gemm","nvfp4_fused_moe","fmha_nvfp4_qkpv","fp8_scaled_fused_moe","dual_cache_latent_bf16"],
        "primitive_wait_atomic_retry_host_wait_or_nonselected_diagnostic",
        "BF16xNVFP4 flag_wait 属 split fixup 原语；NVFP4 MoE mbar_wait 正常/带 hint，host graph clean verdict 为可选路径；FP4 FMHA prefix-publish wait 正常分支；scaled FP8 MoE AP 分支源码明确 rejected；BF16 sparse fused_combine 默认关闭。未重建各历史分支完整 dispatch。",
        "flag/phase/published 是 producer 完成；fuse_step 的 prev==old 是一次原子计数更新成功；host zh 是 clean graph 验证结果；scaled MoE tile_off 搜索为输入 metadata。",
        "等待体只 load/test/nanosleep 或 CAS；发布 partial、count notify、merge 和下一次 graph fallback 均在循环外。do{...}while(0) 宏执行一次。",
        "正常等待依赖对应 producer 完成；CAS 依赖竞争者让出更新机会，无源程序迭代 cap 证明；超时仅是诊断/保守 fallback，不能当成普通完成。",
        "await/atomic action/host wait + if(verdict)；不展开虚构最大轮数。实验 fused_combine 的 win mask 再按固定 set bits 遍历。",
        "合理原语 while 仍可留在 Kernel 中；局部失败分支的存在不自动证明设计级 while 必须保留或算子有漏洞。",
        [ev("gemm/dense/bf16_nvfp4_gemm","kernels.cu",5416,5428,"flag set/wait/clear"),
         ev("moe/nvfp4_fused_moe","nvfp4_moe.cu",2558,2574,"正常 mbarrier 等待"),
         ev("moe/nvfp4_fused_moe","nvfp4_moe.cu",9472,9491,"host 等 verdict 后至多一次 fallback graph"),
         ev("fmha/fmha_nvfp4_qkpv","fp4pv_sm_103a_attention_split.cu",731,745,"已发布 prefix 的等待"),
         ev("moe/fp8_scaled_fused_moe","moe_kernel.cu",658,664,"专家前缀二分"),
         ev("moe/fp8_scaled_fused_moe","moe_kernel.cu",1461,1484,"G1 等待与 AP rejected 状态"),
         ev("moe/fp8_scaled_fused_moe","moe_kernel.cu",4258,4273,"一次 consensus publication 后 await"),
         ev(sb,"dsv4_kernel.cu",692,703,"fuse_step CAS 原语"),
         ev(sb,"dsv4_kernel.cu",734,759,"每行 arrival 一次，固定 win set merge"),
         ev(sb,"dsv4_kernel.cu",5189,5195,"fuse 默认关闭"),
         ev(bm,"runtime.cuh",33,46,"do-while-zero 宏")],
        "local_primitive_and_branch_spot_check"),
    case(
        "explicit_ptx_wait_backedges", ["mxfp8_gemm","fmha_fp16","gdn_decode_core_bf16_state_fp32"],
        "inlined_mbarrier_wait_backedges",
        "选读正常 inline PTX wait helpers；全树只做 bra/goto/extra branch 词法补扫，不宣称已重建生成 PTX/SASS CFG。",
        "mbarrier.try_wait/test_wait 的 ready predicate 与 phase；不是数值残差、预测接受或任务再生判据。",
        "回跳只包围 wait test；动作发起和 producer arrival 是调用点/另一个 role 的独立事件。",
        "对应 barrier phase 的生产者到达；本次不证明任意架构/并发调度都能进展。",
        "保持一个有明确完成语义的 await，显式保留调用点周边的发起与归还。",
        "没有 C/C++ while 关键词也可有循环；不能把汇编回跳漏掉，也不能逐条提升到设计级 pipe。",
        [ev("gemm/dense/mxfp8_gemm","mxfp8_kernels.cu",82,116,"普通/cluster wait 的 PTX 回跳"),
         ev("fmha/fmha_fp16","context_sink0_causal0_lse0_primitives.cuh",124,145,"mbarrier wait 回跳"),
         ev("linear_attention/gdn_decode_core_bf16_state_fp32","solution.py",115,131,"内嵌 PTX wait 回跳")],
        "primitive_only"),
]


def main():
    catalog = json.loads((ROOT / CATALOG).read_text())
    index = json.loads((ROOT / INDEX).read_text())
    assert catalog["operator_count"] == 44
    assert len(index["source_files"]) == 428
    mismatches = []
    source_cache = {}
    suffixes = Counter()
    additional = []
    branches = []
    other_asm_branches = []
    for f in index["source_files"]:
        data = (ROOT / f["path"]).read_bytes()
        if digest(data) != f["sha256"]:
            mismatches.append(f["path"])
        source_cache[f["path"]] = data
        suffixes[Path(f["path"]).suffix] += 1
        lines = data.decode(errors="replace").splitlines()
        rawlines = data.splitlines(keepends=True)
        for n, line in enumerate(lines, 1):
            if re.search(r"\bfor\s*\([^;\n]*;\s*;", line) and not re.search(r"\bfor\s*\(\s*;\s*;\s*\)", line):
                additional.append({"path": f["path"], "line": n, "text": line.strip(),
                                   "head_fragment_sha256": digest(b"".join(rawlines[n - 1:n + 10]))})
            if re.search(r"\bbra(?:\.[a-z]+)*\b|\bgoto\b", line):
                start = max(1, n - 12)
                end = min(len(lines), n + 2)
                context = "\n".join(lines[start - 1:end])
                branches.append({"path": f["path"], "line": n, "text": line.strip(),
                                 "context_start_line": start, "context_end_line": end,
                                 "context_sha256": digest(b"".join(rawlines[start - 1:end])),
                                 "context_contains_mbarrier": "mbarrier" in context})
            if re.search(r"\bbrx\b|\bjmp\b", line):
                other_asm_branches.append({"path": f["path"], "line": n, "text": line.strip()})
    assert not mismatches, mismatches
    assert len(additional) == 27
    assert len(branches) == 218
    assert sum(x["context_contains_mbarrier"] for x in branches) == 217

    op_coverage = []
    selected_paths = set()
    for c in cases:
        selected_paths.update(e["path"] for e in c["source_evidence"])
    previous_six = {"gdn_prefill_bf16_state_bf16", "fp8_scaled_grouped_gemm", "mla_expanded_fp8",
                    "dual_cache_latent_bf16", "fmha_mxfp8_qkv", "nvfp4_fused_moe"}
    for o in catalog["operators"]:
        prefix = str(Path("kernel_wiki") / o["paths"]["candidate"]) + "/"
        cs = [c for c in cases if o["id"] in c["operators"]]
        # These are per-path mechanism scopes, never whole-operator proofs.
        op_coverage.append({
            "id": o["id"], "family": o["category"], "origin": o["solution_origin"],
            "catalog_acceptance_status": o["acceptance_status"],
            "previous_six_wiki_roots": o["id"] in previous_six,
            "entrypoint": str(Path("kernel_wiki") / o["paths"]["entrypoint"]),
            "lexical_screened": True,
            "while_do_lexical_lines": sum(x["path"].startswith(prefix) for x in index["occurrences"]),
            "exact_unconditional_for_lexical_lines": sum(x["path"].startswith(prefix) for x in index["unconditional_for_candidates"]),
            "additional_empty_condition_for_lexical_lines": sum(x["path"].startswith(prefix) for x in additional),
            "mechanism_case_ids": [c["id"] for c in cs],
            "mechanism_review_scopes": sorted(set(c["review_scope"] for c in cs)),
            "whole_operator_or_all_compile_variants_proved": False,
        })
        if cs:
            for key in ("entrypoint", "metadata", "shape_range", "readme"):
                selected_paths.add(str(Path("kernel_wiki") / o["paths"][key]))

    manifest = []
    for path in sorted(selected_paths):
        data = (ROOT / path).read_bytes()
        manifest.append({"path": path, "sha256": digest(data), "bytes": len(data)})

    snapshots = []
    for path in (CATALOG, INDEX, HERE.relative_to(ROOT)/"METHOD.md", Path("schema_discussion/pipe_grammar/README.md")):
        snapshots.append({"path": str(path), "sha256": digest((ROOT/path).read_bytes())})

    payload = {
        "schema": "while_review.broad_kernel_audit.v1",
        "report": "reports/05_broad_kernels.md",
        "independence": "先读 AGENTS.md/METHOD.md/catalog/当前 grammar 与本地源码，未读 01–04 报告或数据结论；previous six ID 由主代理提供，仅用于标新增覆盖。",
        "question": "44 题的更广源码是否遗漏必须在 pipe 多动作段之间表达的真实本地反馈？",
        "recommendation": "在当前 for 已支持真实 work_schedule 动态 worktile 序列的前提下，建议当前核心移除 while；本次未发现要求改变该取舍的已审阅源码反例。原语/host 中合法 while 不需删除；不推广为所有合理算子都不需要 while。",
        "selection_and_proof_limits": [
            "44题完成词法筛查，不是44题全部路径或编译组合的语义证明。case 仅闭合所列控制机制；scope 明确区分完整机制、谓词出口、host/primitive、未穷尽编译分支。",
            "944 是 while/do 候选行，含注释/宏/历史变体；18 是字面 for(;;) 候选行，其中2行只是注释；27 是本次补扫的带初始化或步进的无条件 for 头。均不得当语义 while 数量。",
            "217条相邻片段含mbarrier的 bra 行包括向前 DONE 分支；218总数另含1条 goto 注释。词法上下文标签不是对所有生成汇编的控制流证明。",
            "所有 source_files 的428个 SHA256 已重新验证；selected_source_files 另冻结本次引用路径及合同。只审阅本地源码，没有 GPU 运行、数值验证或完整 schema 迁移。",
            "参考MoE合同明确single-rank；通用多rank模板中的通信逻辑不能当多GPU验收。catalog 的 certified/passed_sampled/waiver 也不等于全支持域证明。",
            "未用巨大cap、有限硬件状态空间或‘直到成功的所有轮次’伪域消去；保留已存在的 publication/receive/release/terminal 边界。",
        ],
        "scope": {
            "operator_count_lexical_screened": 44,
            "candidate_source_file_count": 428,
            "while_do_lexical_candidate_lines": 944,
            "exact_for_semicolon_semicolon_candidate_lines": 18,
            "additional_empty_condition_for_lines": len(additional),
            "bra_goto_lexical_lines": len(branches),
            "bra_goto_contexts_containing_mbarrier": 217,
            "hashes_verified": 428,
            "source_suffix_counts": dict(sorted(suffixes.items())),
            "operators_with_some_mechanism_evidence": sum(bool(o["mechanism_case_ids"]) for o in op_coverage),
            "new_vs_previous_six_with_some_mechanism_evidence": sum(bool(o["mechanism_case_ids"]) and not o["previous_six_wiki_roots"] for o in op_coverage),
            "mechanism_case_count": len(cases),
            "selected_source_file_count": len(manifest),
        },
        "input_snapshots": snapshots,
        "operator_coverage": op_coverage,
        "mechanism_cases": cases,
        "additional_empty_condition_for_candidates": additional,
        "explicit_branch_candidates": branches,
        "brx_jmp_candidates": other_asm_branches,
        "brx_jmp_classification": "38条命中全部是fmha_mxfp8_qkv/fmha_tc.cuh两个rescale scan中的C++变量jmp；并非PTX跳转。对应数值条件/固定域语义见numeric_rescale_scan_for_if。未找到brx指令候选。",
        "selected_source_files": manifest,
        "verification": {
            "source_hashes": "all 428 index entries match current bytes",
            "fragment_hash_definition": "sha256 of inclusive line slice, preserving original line endings",
            "generated_with": "python3 -B schema_discussion/pipe_grammar/while_review/05_build_data.py",
            "gpu_executed": False,
            "schema_migrated": False,
            "original_source_modified": False,
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload["scope"], ensure_ascii=False))
    print("brx/jmp lexical candidates:", len(other_asm_branches))


if __name__ == "__main__":
    main()
