"""Rebuild only data/02_extracted.json from reviewed, immutable inputs.

The case judgments are manual source review. Hash and range checks are integrity
checks, not execution, termination, or numerical tests.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SRC = Path("schema_discussion/pipe_grammar/trial_roots/sources")
BASE = Path("schema_extract/archive/20261109-ten-operator-text-audit/inputs")
OPS = ["extract_gdn", "extract_gemm", "extract_mla", "extract_sparse", "extract_fmha", "extract_moe"]
BRANCHES = {
    "extract_gdn": ["segmented", "direct_qk_ordered"],
    "extract_gemm": ["wide", "mma_sync"],
    "extract_mla": ["bootstrap"],
    "extract_sparse": ["tc_packed", "probe_hidden_packed"],
    "extract_fmha": ["tcg_d128_flat_tma_direct_narrow", "tcg_d128_segmented_tma_direct_narrow", "tcg_d128_fscan_tma_direct_narrow"],
    "extract_moe": ["direct", "fused_small_mono", "split_ws", "double_verify_root"],
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_record(path: Path | str) -> dict:
    p = ROOT / path
    raw = p.read_bytes()
    return {"path": str(path), "sha256": sha(raw), "bytes": len(raw), "lines": len(raw.splitlines())}


def ref(path: Path, first: int, last: int | None = None) -> dict:
    lines = (ROOT / path).read_bytes().decode("utf-8").splitlines(keepends=True)
    last = first if last is None else last
    assert 1 <= first <= last <= len(lines), (path, first, last, len(lines))
    fragment = "".join(lines[first - 1:last])
    return {"path": str(path), "line_start": first, "line_end": last,
            "first_line": lines[first - 1].rstrip(), "fragment_sha256": sha(fragment.encode())}


def source(op: str, name: str, first: int, last: int | None = None) -> dict:
    return ref(SRC / op / name, first, last)


def baseline(op: str, name: str, first: int, last: int | None = None) -> dict:
    return ref(BASE / op / (name + ".yaml"), first, last)


cases = []


def add(case_id, operators, mechanism, scope, evidence, predicate, body, progress,
        bound, schema_actions, alternative, preserve, limitations=""):
    cases.append({
        "id": case_id, "operators": operators, "mechanism": mechanism,
        "selected_scope": scope, "evidence": evidence,
        "predicate_source": predicate, "loop_body": body,
        "progress_and_exit": progress, "meaningful_bound": bound,
        "existing_schema_actions_or_domain": schema_actions,
        "for_if_or_primitive_alternative": alternative,
        "events_and_boundaries_to_preserve": preserve,
        "requires_pipe_while_in_reviewed_scope": False,
        "limitations": limitations,
    })


add("gdn_chunk_recurrence", ["extract_gdn"], "finite_counted_recurrence", "root; shared chain body in segmented/direct branches",
    [source("extract_gdn", "gdn_kernel.cu", 2058, 2063), source("extract_gdn", "gdn_kernel.cu", 2318, 2343), baseline("extract_gdn", "root", 621, 675)],
    "seqlen from cu; chunks=ceil(seqlen/64), nseg=chunks for MODE0. cl<nseg does not inspect R.",
    "Stage wait/join and future load; KS/QS; Hrhs; triangular solve; output and R/Rh update.",
    "cl increments once per input chunk; exit at nseg, regardless of state values.",
    "Exact input-derived chunk count; each numerical recurrence still depends on previous R.",
    "domain c=[0,c_n); advance:Stage, project:KS_QS, rhs:Hrhs, solve:U, finish_chunk:Out_R; final_store:FinalState",
    "Natural for(c), with existing protocol waits. Carrying R is not a feedback continue condition.",
    "advance retires old readers before issuing future stage; mini solve keeps no-payload events; final_store occurs after traversal.",
    "Baseline's final-mini stage alias/read-retirement defect remains; no GPU proof.")

add("gdn_segmented_one_decision", ["extract_gdn"], "finite_data_dependent_range", "segmented",
    [source("extract_gdn", "gdn_kernel.cu", 2227, 2249), source("extract_gdn", "gdn_kernel.cu", 2265, 2279), baseline("extract_gdn", "segmented", 119, 140), baseline("extract_gdn", "segmented", 617, 626)],
    "After pass-A completion, max of G SegA values creates weak=(amax>0.001).",
    "One classification: weak z>0 returns; weak z=0 chooses full chunks; otherwise use fixed segment bounds and compose seed.",
    "Pass C traverses chosen Count once; no branch re-enters pass A or reclassifies after pass-C results.",
    "G from input/host geometry, capped at 8; Count equals c_n or m_z.",
    "join_a: summaries ready; classify establishes Run/Count; run gate; cl=[0,Count).",
    "join_a, classify, if(Run), for(cl=0:Count), existing finalize.",
    "Keep cross-phase summary publication and wait; classification precedes domain use; inactive workers retain earlier entry events.",
    "The scalar weak test is not an error bound or convergence test.")

add("gdn_local_search_and_triangular_break", ["extract_gdn"], "finite_action_internal_algorithm", "direct_qk_ordered sorting; shared triangular contractions",
    [source("extract_gdn", "gdn_kernel.cu", 268, 279), source("extract_gdn", "gdn_kernel.cu", 2589, 2608), source("extract_gdn", "gdn_kernel.cu", 2645, 2673), baseline("extract_gdn", "direct_qk_ordered", 62, 70), baseline("extract_gdn", "direct_qk_ordered", 376, 399)],
    "Insertion-sort predicate uses already loaded chunk counts and j>=0; ks>mw depends only on warp tile.",
    "Shift key/ch arrays leftward; or load/multiply lower-triangular subtiles.",
    "j decrements at most i times; ks advances across at most four subtiles.",
    "i<=N-1 for sorting; mw+1<=4 triangular subtiles.",
    "enter:Metadata establishes permutation; solve/project contracts contain fixed matrix subtiles.",
    "Finite local algorithm domains exist; sorting while may remain preferable inside action. Triangular loop is directly [0,mw+1).",
    "Final permutation publication and prep-to-chain wait; all retained contraction order.")

add("gemm_prefix_decode", ["extract_gemm"], "finite_metadata_search", "root umma_gemm<0>, wide, mma_sync",
    [source("extract_gemm", "fp8gg_umma.cu", 327, 388), source("extract_gemm", "fp8gg_umma.cu", 798, 808), source("extract_gemm", "fp8gg_umma.cu", 1152, 1181), source("extract_gemm", "fp8gg.cu", 115, 147), baseline("extract_gemm", "root", 241, 271), baseline("extract_gemm", "root", 331, 397)],
    "Current m_indptr supplies nondecreasing prefix; w is a valid strided work index below prefix[E]*NB.",
    "Advance e over empty/prior groups, decode row/column; each work subsequently traverses PAIRS or KB.",
    "e only increases, valid w guarantees an enclosing group e<E; work index advances by positive grid stride.",
    "E candidate groups and exact KB=K/ScaleK (Q=KB/2 for root).",
    "initialize:Schedule, prepare:Work; t=worktile, n=flatten(t,q) or flatten(t,kb).",
    "Keep coordinate lookup in existing schedule/work action; natural finite work and reduction for domains. If exposed, search can use real E candidates plus found guard.",
    "Empty workers retain initialization/drain; n and barrier generations continue across work; preserve product release before final register FMA.")

add("barrier_poll_family", OPS, "notification_primitive_poll", "selected waits in six operators; sparse TC waits from representative branches",
    [source("extract_gdn", "gdn_common.cuh", 300, 305), source("extract_gemm", "fp8gg_umma.cu", 173, 183), source("extract_mla", "solution.py", 133, 154), source("extract_sparse", "dsv4_kernel.cu", 1614, 1637), source("extract_fmha", "fmha_tcg_ptx.cuh", 259, 272), source("extract_moe", "nvfp4_moe.cu", 2558, 2574)],
    "Fixed barrier address/generation; readiness from peer arrival or asynchronous transfer/MMA completion.",
    "Test readiness, optionally nanosleep or suspend hint; no reissue of the guarded design computation.",
    "Exit only when required generation/condition is satisfied. Progress depends on notifier/operation and protocol.",
    "No meaningful input-derived upper bound on poll attempts; suspendTimeHint is not an overall timeout.",
    "Existing acquire/await/wait and action completion contracts, plus waiting reuse/data/sync edges.",
    "Retain unbounded low-level polling inside legitimate await/acquire. A capped for that proceeds anyway is not equivalent.",
    "Exact parity, transaction bytes, participant arrivals, scope/visibility and required release notifications; await does not drain unrelated work.",
    "This class alone proves neither deadlock nor liveness; low-level while is legitimate.")

add("mla_dynamic_work_queue", ["extract_mla"], "persistent_dynamic_work_schedule", "root; bootstrap is same body with queued=false",
    [source("extract_mla", "solution.py", 850, 903), source("extract_mla", "solution.py", 1117, 1136), source("extract_mla", "solution.py", 1263, 1277), source("extract_mla", "solution.py", 1376, 1387), source("extract_mla", "solution.py", 1447, 1484), baseline("extract_mla", "root", 339, 353), baseline("extract_mla", "root", 620, 641)],
    "Each consumer exits on published kv_tiles<0; Queue tests whether atomic old counter plus gridDim is below P.items.",
    "Consumer latches geometry and executes N(w) tiles. Softmax requests once on final key tile; Queue waits, pops, copies geometry or sentinel and publishes.",
    "Predetermined Geo table is exhausted; work completion never creates/requeues items. Exactly WorkCount-worker_count successful refills plus one failed pop per CTA.",
    "Finite global Geo length WorkCount. Per-CTA actual work sequence is dynamic, not knowable in advance; poll durations remain unbounded.",
    "w=worktile; n=flatten(w,t), t=[0,N(w)); request:Next, Queue/publish:Next, four wait:Next/latch:Item families.",
    "for(w) over real atomic_queue assignment, keeping scheduling actions explicit. This is not an invented 'successful rounds' domain.",
    "Terminal failed-pop publication and all four consumers; no zero-tile work for sentinel; request before last score wait; two-slot geometry lifetime; continuous n; final output/release.",
    "Baseline flags early descriptor overwrite lifetime and TMEM row-address discrepancies. No generic concurrency/termination proof.")

add("sparse_counted_attention", ["extract_sparse"], "finite_input_or_materialized_data_domain", "root split; packed after union",
    [source("extract_sparse", "dsv4_kernel.cu", 980, 1025), source("extract_sparse", "dsv4_kernel.cu", 1187, 1207), baseline("extract_sparse", "root", 175, 194), baseline("extract_sparse", "tc_packed", 38, 64), baseline("extract_sparse", "tc_packed", 236, 239)],
    "Klen from visible length/topk length; balanced c0/c1 fixed before loop. Packed Nu becomes ready after build:Union.",
    "Load indices/gather, QK, softmax, PV and reader retirement for each selected chunk.",
    "c increments to c1 or C=ceil(Nu/KC). Invalid indices do not reduce root loop length; they still gather replacement rows.",
    "Exact input Klen or already materialized Nu; Nu<=GT*IndexWidth<=2560 in selected packed path.",
    "root c=[c0,c1); packed build:Union then c=[0,C(grp)) with empty/nonempty gate.",
    "Natural for domains and if(C>0). Empty split differs from nonempty all-invalid chunk.",
    "Empty-group commits used by wait_group; sentinel statistics from empty split; no-payload event contracts; pool identities and later snapshots.")

add("sparse_open_addressing", ["extract_sparse"], "finite_action_internal_CAS_probe", "tc_packed and shared packed bodies; probe variant uses slot identity",
    [source("extract_sparse", "dsv4_kernel.cu", 2305, 2379), source("extract_sparse", "dsv4_kernel.cu", 3886, 3928), source("extract_sparse", "dsv4_kernel.cu", 3964, 3993), baseline("extract_sparse", "tc_packed", 363, 384)],
    "Loaded encoded key and atomic returned slot content determine installed/same-key/different-key cases.",
    "CAS installation; different key advances slot; creator allocates dense position and publishes; same-key handles membership/duplicates.",
    "Different-key probes monotonically advance through circular slot order. A matching or empty slot must exist under the actual entry-capacity gate.",
    "8192 physical probe candidates with <=2560 selected items for packed union; this bounds spatial probing, not position wait duration.",
    "Setup/build:Union with temporary HashTable/IndexStage; analogous probe/Main count action.",
    "Existing cooperative action is appropriate. If spatial probe is explicitly modeled, real hash-slot domain plus found guard is available; no synthetic until-success domain needed.",
    "Key/position publication, multiplicity-preserving membership/duplicate handling, final CTA union publication and temporary-ring retirement.",
    "Do not claim identical CAS timing/runtime union permutation after an actual source rewrite. Different probe variants must not be conflated.")

add("sparse_position_publication_wait", ["extract_sparse"], "notification_primitive_poll", "tc_packed and packed prefix/remainder; distinct from spatial probe",
    [source("extract_sparse", "dsv4_kernel.cu", 2346, 2377), source("extract_sparse", "dsv4_kernel.cu", 2380, 2380), baseline("extract_sparse", "tc_packed", 357, 384)],
    "Same-key CAS observes a key whose low position word is still zero; poll reloads that same word until pos>=0.",
    "Volatile same-slot loads only; creator separately publishes low pos+1 through atomicExch after allocation/membership stores.",
    "Normal exit follows creator's position publication. Capacity and total items provide no bound on waiting time.",
    "No meaningful finite poll count.",
    "All eight warps belong to Setup during build:Union; per-key creator/contender are transient thread roles. Loader/Soft roles start after completed union.",
    "Keep the documented cooperative action and its internal await-like protocol. Capped for is not equivalent. If later modeled as separate per-key design actors, reassess at that new boundary.",
    "Same-key publication guarantee and final all-CTA rendezvous; no consumer of the downstream pipeline may use a half-built union.",
    "Baseline records live barrier/metadata alias defects. No hardware liveness proof; no new opaque action was introduced to eliminate a design boundary.")

add("sparse_optional_fused_combine", ["extract_sparse"], "atomic_primitive_retry_and_finite_winner_set", "supplemental source; default g_fuse_combine=0, not selected root",
    [source("extract_sparse", "dsv4_kernel.cu", 690, 703), source("extract_sparse", "dsv4_kernel.cu", 731, 759), source("extract_sparse", "dsv4_kernel.cu", 5194, 5194)],
    "CAS compares previous tagged counter; completed arrivals produce a finite HB-bit winner mask.",
    "CAS retry increments/installs one arrival; winner bitset iteration removes low set bit and combines that row.",
    "Successful atomic update ends one arrival; m&=m-1 removes one winner per merge iteration. No attention re-execution.",
    "Winner set bounded by HB<=64; retry is atomic implementation. Default path does not invoke it.",
    "Default split uses separate combine phase. Supplemental semantics match arrival/election then finite head traversal.",
    "Primitive atomic update plus for(heads) if(winner); no need for pipe feedback.",
    "Every split including empty contributes arrival; last arrival observes published partials.",
    "Explicitly excluded from selected default coverage; not counted as an additional shipped mechanism.")

add("fmha_fixed_scan_and_recurrence", ["extract_fmha"], "finite_counted_recurrence", "root and flat/fscan representative branches",
    [source("extract_fmha", "fmha_tcg.cuh", 4462, 4489), source("extract_fmha", "fmha_tc.cuh", 915, 934), source("extract_fmha", "fmha_tc.cuh", 1062, 1106), source("extract_fmha", "fmha_tcg.cuh", 4409, 4452), baseline("extract_fmha", "root", 647, 658), baseline("extract_fmha", "tcg_d128_flat_tma_direct_narrow", 462, 489)],
    "Root g from causal/window bounds and z=0,1; flat scan T groups; fscan prefix tw=min(cLo,T). m changes values but never these limits.",
    "Root QK/read halves, retire prior PV, advance max/denominator/probability and PV. Flat scan loads 64, then 16, then scalar batches before ordered threshold updates/stores.",
    "Each input group consumed once in the defined order; exit at fixed endpoint, not when max stops changing.",
    "Ng, T or tw from geometry; batch partitions follow division/remainder by 64/16.",
    "root for(g)/for(z); flat scan:MState, tail:MTail; fscan uses the same finite recurrence locally.",
    "Natural ordered for batches. If exposing loads and updates separately, preserve all loads of each batch before its dependent updates.",
    "Root first/last no-payload events, PV retirement/drain and all snapshot stores; keep delayed-maximum recurrence ordering.")

add("fmha_segmented_data_dependent_cut", ["extract_fmha"], "finite_data_dependent_range", "tcg_d128_segmented_tma_direct_narrow",
    [source("extract_fmha", "fmha_tc.cuh", 1406, 1437), source("extract_fmha", "fmha_tc.cuh", 1468, 1497), baseline("extract_fmha", "tcg_d128_segmented_tma_direct_narrow", 100, 120), baseline("extract_fmha", "tcg_d128_segmented_tma_direct_narrow", 539, 579)],
    "One summary batch is compared against incoming ScanState; Jump/Cut is computed before replay. t0>=t1 break is an extent guard.",
    "Fill constant prefix without GroupMax loads, replay finite suffix with threshold recurrence; next batch uses resulting m.",
    "sbase advances 16 segments every iteration. Updated states do not trigger repeated summary testing within the batch.",
    "ceil((Nseg-1)/16) batches; each [g0,Cut), [Cut,g1) within finite input groups.",
    "Existing for(c) contains independent test:Prefix(c), replay:Chunk(c); ReplayStart has explicit register resource.",
    "Direct for(c). Even if decomposed, test establishes Cut then two natural runtime ranges; no while and no arbitrary cap.",
    "All summaries consumed before replay; prefix still writes snapshots; entire suffix replay order; PDL input wait and output visibility.")

add("moe_static_persistent", ["extract_moe"], "persistent_static_work_schedule", "root, fused_small_mono, split_ws",
    [source("extract_moe", "nvfp4_moe.cu", 4371, 4384), source("extract_moe", "nvfp4_moe.cu", 4448, 4475), source("extract_moe", "nvfp4_moe.cu", 4543, 4560), source("extract_moe", "nvfp4_moe.cu", 4855, 4867), source("extract_moe", "nvfp4_moe.cu", 4990, 5004), source("extract_moe", "nvfp4_moe.cu", 3186, 3220), baseline("extract_moe", "fused_small_mono", 379, 449)],
    "J=TilePrefix[E] and fixed columns/SK form total_items; start/stride from grid. Local stages are H/(BK*SK) or I/BK.",
    "A/B producer loads and scale publication, MMA, four epilogue drains/releases/stores; mono variant performs these roles in one CTA sequence.",
    "item increases by positive grid stride until total_items. No work pop, failed ticket, requeue or feedback-generated item.",
    "Exact metadata-derived work count and input channel reductions.",
    "t=worktile, g=flatten(t,i), refill:none; individual scale load/publish, prepare, MMA, drain/release/emit actions.",
    "Natural nested for; Hoist conditional wait is a single if, not a feedback loop.",
    "Keep g/ring generations across work; A's per-work priming; all four accumulator release contributors; worker-final bulk-store completion even for local empty workers.")

add("moe_binary_search", ["extract_moe"], "finite_action_internal_algorithm", "root scan and small-routing variants",
    [source("extract_moe", "nvfp4_moe.cu", 750, 783), source("extract_moe", "nvfp4_moe.cu", 4158, 4170)],
    "hi-lo>1, using stable per-expert prefix and fixed tile j.",
    "Compare midpoint prefix and retain containing half; then store TileExpert/TileRow.",
    "Integer interval strictly shrinks to one expert interval, including zero-count experts.",
    "ceil(log2(E)) comparison iterations; no arbitrary cap.",
    "scan/Main produce:TilePrefix; small prepare includes the same finite tile-map algorithm.",
    "Keep binary-search while inside compute action; optional bounded guarded for is possible but unnecessary at pipe level.",
    "Prefix computation and CTA publication before searches, all resulting maps before FC phases.")

add("moe_direct_arrival_ticket", ["extract_moe"], "finite_cross_CTA_reduction_election", "direct.yaml cfg92, including nonempty replay after empty ABI construction",
    [source("extract_moe", "nvfp4_moe.cu", 1976, 1989), source("extract_moe", "nvfp4_moe.cu", 2205, 2257), baseline("extract_moe", "direct", 111, 144), baseline("extract_moe", "direct", 345, 407)],
    "ValidUnit from current route ID; Last from the single atomicAdd return equal to n(t)*S-1, ready after CTA join.",
    "One fixed unit computes FC1/QDQ/FC2 Partial, publishes additions, takes one arrival; last arrival gathers route metadata, finalizes output and clears scratch; all valid units advance.",
    "Each valid unit contributes exactly once; n(t)*S is fixed by current route count and intermediate chunks. Non-last contributors never wait for a winning ticket or rerun Partial.",
    "u in a clipped interval of [0,R*S); per-token arrival count n(t)*S.",
    "for(v) inspect; if(ValidUnit) prepare/contract/quantize/accumulate/arrive; if(Last) gather/finalize; advance.",
    "Natural finite for plus two if gates. Keep separate accumulate and arrive; do not replace return-value decision with polling.",
    "Publication-before-ticket; all arrivals visible before finalizer reads; BF16 boundary; ordered output/route resets/counter reset; advance for non-last valid units as well.",
    "No claim that every possible GPU schedule is proven; the described protocol does not require a grid barrier. cfg91 staged variant not included in this branch claim.")

add("moe_validation_arrival_ticket", ["extract_moe"], "finite_cross_CTA_reduction_election", "double_verify_root; same k_zero_verify called by verification variants",
    [source("extract_moe", "nvfp4_moe.cu", 1723, 1789), baseline("extract_moe", "double_verify_root", 958, 1021)],
    "Each CTA's local dirty OR; single arrival ticket compared with gridDim.x-1.",
    "Finite output/validation scans, dirty publication, arrival; last CTA publishes clean/host mirror and clears counters.",
    "All fixed shards arrive once. Other CTAs complete after own contribution, without polling for final clean.",
    "Input byte extents and exact submitted CTA count.",
    "store:ProbeOutput/ZeroOutput, scan:ProbeDirtyPart/DirtyPart, publish:ProbeClean/Clean.",
    "Finite scans plus last-arrival if; publication protocol remains in its existing named action.",
    "Two verification submissions stay distinct; mirror publication is a submission hint, not completion of all output stores; ordered scratch reuse.")

add("host_bounded_observation", ["extract_moe", "extract_sparse"], "host_dispatch_poll", "MoE double_verify_root and Sparse probe_hidden_packed; outside role pipe",
    [source("extract_moe", "nvfp4_moe.cu", 9471, 9491), baseline("extract_moe", "double_verify_root", 14, 33), source("extract_sparse", "dsv4_kernel.cu", 5593, 5614), baseline("extract_sparse", "probe_hidden_packed", 15, 33)],
    "Host reads device-written pinned mirror or eight sample slots; existing spin counter also terminates observation.",
    "Repeated reads only; after poll, one observed-value decision selects full graph or remainder plan.",
    "Readiness or existing cap ends observation; no repeated numerical kernel submission in the polling body.",
    "MoE actual source 200000000 loop tests; Sparse actual source 2000000 passes. These are real host caps, not invented elimination bounds.",
    "dispatch.when and host flow; device verification/probe actions keep their independent publication and kernel completion events.",
    "At host source level bounded for/if is possible; at pipe level leave this host dispatch responsibility outside roles. Unlimited await would change timeout/observation semantics.",
    "Terminal observation can be partial or timeout, not eventual GPU verdict; outer capture records both MoE graphs without polling; later replay does not rerun host decisions.")

add("host_configuration_search", ["extract_gdn", "extract_sparse", "extract_fmha"], "finite_host_planning", "frozen entry configuration choices, not pipe feedback",
    [source("extract_gdn", "gdn_kernel.cu", 3222, 3244), source("extract_sparse", "dsv4_kernel.cu", 4801, 4805), source("extract_sparse", "dsv4_kernel.cu", 5043, 5052), source("extract_sparse", "dsv4_kernel.cu", 5125, 5132), source("extract_fmha", "fmha_mxfp8_qkv.cu", 807, 837)],
    "Input geometry, device attributes, resource API responses; no per-tile numerical convergence value.",
    "Shrink G/head block, enlarge split or shared-memory request; return selected launch plan.",
    "Finite candidate configurations or real attempt cap; errors/occupancy criterion stop search.",
    "G<=8, finite power-of-two tile options; FMHA 96 actual attempts and 4096-byte growth capped by source.",
    "work_schedule/dispatch host plan, before device action sequence.",
    "Bounded parameter search can use for/if, but adding its while to device pipe would cross abstraction layers.",
    "Final selected kernel/geometry and actual API failure behavior must remain visible in host selection description.")

add("diagnostics_and_scaffolding", OPS, "excluded_diagnostic_or_syntax", "macros/comments and explicit watchdog/probe builds",
    [source("extract_sparse", "dsv4_kernel.cu", 1639, 1668), source("extract_fmha", "fmha_tcg.cuh", 2608, 2638), source("extract_moe", "nvfp4_moe.cu", 6184, 6199), source("extract_moe", "nvfp4_moe.cu", 9835, 9839)],
    "Diagnostic timeout flags/time counters or literal false for do-while macro wrapper.",
    "Poll then record failure, or burn clock cycles for instruction probe; macros execute once.",
    "Watchdog exits on readiness or actual diagnostic timeout; macro while(0) never repeats.",
    "Actual diagnostic caps, intentionally not bounds for normal pipeline wait semantics.",
    "No selected normal-pipe feedback; diagnostic behavior may return with wrong data by design.",
    "Exclude from evidence for normal design loops; do not import watchdog early return as a legal normal await.",
    "Keep distinct from ordinary wait termination and from semantic operator output guarantees.")


manifest_path = Path("schema_discussion/pipe_grammar/trial_roots/manifest.json")
manifest = json.loads((ROOT / manifest_path).read_text())
source_files = []
baselines = []
origins = []
for op in OPS:
    info = manifest["operators"][op]
    origins.append({"operator": op, "source_reference": info["source_reference"],
                    "source_origin": info["source_origin"],
                    "cited_revision_available": info["cited_revision_available"],
                    "source_verification_from_manifest": info["source_verification"]})
    root_rec = file_record(info["baseline"])
    root_rec.update(operator=op, kind="root", manifest_sha256=info["baseline_sha256"],
                    hash_matches_manifest=root_rec["sha256"] == info["baseline_sha256"])
    assert root_rec["hash_matches_manifest"], root_rec
    baselines.append(root_rec)
    for b in BRANCHES[op]:
        rec = file_record(BASE / op / (b + ".yaml"))
        rec.update(operator=op, kind="representative_branch",
                   review_scope="dispatch and loop-relevant math/schedule/resources/contracts; not every action or every full-branch numerical detail")
        baselines.append(rec)
    for src in info["sources"]:
        rec = file_record(src["snapshot"])
        rec.update(operator=op, original=src["original"], manifest_sha256=src["sha256"],
                   hash_matches_manifest=rec["sha256"] == src["sha256"])
        assert rec["hash_matches_manifest"], rec
        source_files.append(rec)

pattern = re.compile(r"^\s*while\b|^\s*for\s*\(\s*;\s*;\s*\)|\}\s*while\s*\(\s*clock64")
headers = []
for f in source_files:
    for n, line in enumerate((ROOT / f["path"]).read_text().splitlines(), 1):
        if pattern.search(line):
            headers.append({"path": f["path"], "line": n, "text": line.strip()})

groups = {}
for c in cases:
    groups.setdefault(c["mechanism"], []).append(c["id"])
groups["genuine_multi_design_action_feedback_requiring_while"] = []

rule_paths = [Path("AGENTS.md"), Path("schema_discussion/pipe_grammar/while_review/METHOD.md"),
              Path("schema_discussion/pipe_grammar/README.md"), Path("schema_discussion/verb_vocabulary/README.md")]
payload = {
    "format": "while-review/extracted-six-v1",
    "report": "schema_discussion/pipe_grammar/while_review/reports/02_extracted.md",
    "method": "Manual responsibility/boundary review; integrity checks only. No GPU execution or operator migration.",
    "rules_read": [file_record(p) for p in rule_paths],
    "source_manifest": file_record(manifest_path),
    "conclusion": {
        "reviewed_cases_require_pipe_while": False,
        "claim": "The six frozen roots and listed representative branches do not force design-level pipe while; primitive polling can remain unbounded below pipe.",
        "not_claimed": ["All reasonable operators are expressible without while", "Every possible branch/build/call path was reviewed", "GPU numerical correctness or general concurrent termination", "An arbitrary finite cap can replace notification waits", "All internal operations may be collapsed into one opaque action"],
        "strongest_candidates": ["mla_dynamic_work_queue", "moe_direct_arrival_ticket", "sparse_position_publication_wait"],
    },
    "coverage": {
        "operators": OPS, "frozen_source_file_count": len(source_files),
        "root_count": len(OPS), "representative_branch_count": sum(map(len, BRANCHES.values())),
        "representative_branches": BRANCHES,
        "search_included": ["while", "for(;;)", "break and early exits", "CAS retry", "PTX wait backedges", "host versus selected-kernel control", "mathematical recurrence stopping criteria"],
        "not_exhaustive": "Not all 172 authored extracted nodes, all compile-time variants, or all historical/current source copies were semantically audited.",
        "duplicate_handling": "Alternate MoE .cu/.py files are retained and hashed as frozen inputs; repeated helpers are not counted as independent mechanisms.",
        "source_origins": origins,
    },
    "groups": groups, "cases": cases,
    "source_files": source_files, "baselines": baselines,
    "lexical_header_inventory": {
        "regex": pattern.pattern,
        "purpose": "Reproducible candidate locations only, not a parser, selected-path census, or count of independent mechanisms. PTX waits and counted for loops are reviewed in cases rather than this header list.",
        "headers": headers,
    },
    "validation": {
        "source_hashes_match_trial_manifest": all(x["hash_matches_manifest"] for x in source_files),
        "root_hashes_match_trial_manifest": all(x.get("hash_matches_manifest", True) for x in baselines),
        "all_evidence_ranges_exist": True,
        "fragment_hash_encoding": "UTF-8 bytes of inclusive original lines, retaining line endings",
        "no_gpu_run": True,
        "no_finite_trace_equivalence_or_liveness_claim": True,
    },
}
assert len(source_files) == 27
assert sum(map(len, BRANCHES.values())) == 14
dest = HERE / "data/02_extracted.json"
dest.parent.mkdir(exist_ok=True)
dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"output": str(dest.relative_to(ROOT)), "cases": len(cases),
                  "source_files": len(source_files), "representative_branches": 14,
                  "source_and_root_hashes_match": True, "references_checked": sum(len(c["evidence"]) for c in cases)}, ensure_ascii=False))
