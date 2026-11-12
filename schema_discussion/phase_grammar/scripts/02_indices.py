#!/usr/bin/env python3
"""Build the independent index audit; quote and digest checks are read-only.

Only the assigned data/02_indices.json is written. The 172-node scan was used
for lexical triage, not for a claim of complete semantic review. Root counts
below count domain entries by field position, with YAML aliases expanded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "schema_discussion/phase_grammar"
CORPUS = json.loads((BASE / "corpus.json").read_text())
INPUT = "schema_extract/archive/20261109-ten-operator-text-audit/inputs/"
WORK = "schema_extract/archive/20261109-ten-operator-text-audit/work/"
RULE = "schema_discussion/phase_grammar/sources/rules/5.design.tile_pipeline.md"
NODES = {
    n["snapshot"]: (op, n)
    for op, obj in CORPUS["operators"].items()
    for n in obj["nodes"]
    if n["snapshot"]
}
SOURCES = {s["path"]: s for obj in CORPUS["operators"].values() for s in obj["sources"]}


def ip(op: str, node: str = "root") -> str:
    return f"{INPUT}extract_{op}/{node}.yaml"


def ev(path: str, first: int, last: int) -> dict:
    lines = (ROOT / path).read_text().splitlines()
    assert 0 < first <= last <= len(lines), (path, first, last)
    out = {"path": path, "start_line": first, "end_line": last,
           "quote": "\n".join(lines[first - 1:last])}
    if path in NODES:
        op, n = NODES[path]
        out.update(operator=op, node=n["node"], category_at_capture=n["category"])
    if path in SOURCES:
        digest = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        assert digest == SOURCES[path]["sha256"]
        out["sha256"] = digest
    return out


branch_ranges = [
    ("gdn", "segmented", [[117, 210], [280, 370], [566, 672]],
     "Segment-local chunk coordinates, descending seed composition and Run/Count guard."),
    ("gemm", "wide_recovery_full", [[1, 104], [110, 484]],
     "Four separately submitted phases sharing YAML aliases; persistent ring and fragment indices."),
    ("mla", "bootstrap", [[1, 101], [170, 320]],
     "One preassigned worktile and no queue; contrast with root's worker-local flatten."),
    ("sparse", "tc_packed", [[1, 651]],
     "Runtime union order, multiplicities, ordinal-to-physical mapping and independent softmax warps."),
    ("fmha", "fused_d128_group", [[1, 115], [215, 591]],
     "Even/Odd groups with stride two, unequal local loop lengths, early maximum and later denominator/PV handoff."),
    ("fmha", "tcg_d128_fscan_cp_direct_narrow", [[90, 145], [215, 299], [316, 355], [515, 600]],
     "Bounded prefix walk before attention; no need to register every internal scalar loop."),
    ("fmha", "tcg_d128_segmented_tma_xpose_narrow", [[234, 254], [322, 348], [520, 590], [690, 735]],
     "Review candidate: independent row scans, mask-selected suffixes and the separately recorded empty-split source limitation."),
    ("fmha", "tcg_d128_fscan_cp_direct_wide", [[1, 26], [371, 389], [550, 663]],
     "Needs authoring: prefix loop plus four independent combine groups; used only for targeted relation checks."),
    ("moe", "fused_big_mono", [[1, 46], [330, 512]],
     "One-role FC2, continuous ring slots, local fragment loop and empty copy-group tail."),
]

roots = []
for op, obj in CORPUS["operators"].items():
    path = obj["root"]
    node = NODES[path][1]
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == node["sha256"]
    doc = yaml.load((ROOT / path).read_text(), Loader=yaml.CSafeLoader)
    phases = doc["design"]["tile_pipeline"]["phase"]
    roots.append({"operator": op, "path": path, "status": node["category"],
                  "sha256": node["sha256"], "read": "full root text",
                  "domain": {p: d["domain"] for p, d in phases.items() if "domain" in d},
                  "domain_entries_raw_field_instances": sum(len(d.get("domain", {})) for d in phases.values()),
                  "domain_entries_effective_field_instances": sum(len(d.get("domain", {})) for d in phases.values())})

branches = []
for op, node_name, ranges, focus in branch_ranges:
    path = ip(op, node_name)
    node = NODES[path][1]
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == node["sha256"]
    branches.append({"operator": f"extract_{op}", "node": node_name, "path": path,
                     "status": node["category"], "sha256": node["sha256"],
                     "read_line_ranges": ranges, "read": "targeted semantic review", "focus": focus})

source_ranges = {
    WORK + "extract_gdn/gdn_kernel.cu": [[714, 721], [2243, 2295]],
    WORK + "extract_gemm/fp8gg_umma.cu": [[350, 408], [423, 444], [790, 845]],
    "schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py": [[488, 514], [848, 905], [1447, 1492]],
    WORK + "extract_sparse/dsv4_kernel.cu": [[2330, 2380], [2428, 2475], [2716, 2758]],
    WORK + "extract_fmha/source/candidate/fmha_tcg.cuh": [[3605, 3675], [3707, 3753], [3780, 3854], [4398, 4468], [4688, 4753]],
    WORK + "extract_fmha/source/candidate/fmha_tc.cuh": [[1456, 1499]],
    WORK + "extract_moe/source_nvfp4_moe.cu": [[1189, 1225], [3590, 3624], [4840, 4924]],
}

findings = [
    {
        "id": "I01", "operator": ["extract_gemm", "extract_mla", "extract_moe"],
        "category": "existing_structure",
        "claim": "Keep shared ordered indices and flatten: worker-local continuous versions connect independent roles and ring reuse across worktiles. A global work id, a mathematical tile coordinate and a physical slot are different objects.",
        "recommendation": "Retain worktile plus dependent integer domains and flatten. Keep work assignment in work_schedule and version-to-slot mapping in resource.view; neither substitutes for the shared logical ordinal. Do not infer lockstep, resets or a known dynamic work-sequence length.",
        "related_concepts": ["resource"],
        "existing_mechanism_attempt": "For each role, for(w) and for(n[w]) use w: worktile, t: [0,N(w)], n: {flatten:[w,t]}; n+depth crosses work boundaries, whereas n[w,t+1] stays in the current worktile.",
        "evidence": [ev(ip("gemm"), 266, 269), ev(ip("gemm"), 360, 386),
                     ev(WORK + "extract_gemm/fp8gg_umma.cu", 423, 438),
                     ev(ip("mla"), 364, 367),
                     ev("schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py", 859, 882),
                     ev(ip("moe"), 633, 636), ev(RULE, 598, 617)],
    },
    {
        "id": "I02", "operator": ["extract_sparse", "extract_moe"],
        "category": "other_layer",
        "claim": "Ragged/sparse membership, repeated IDs and runtime list permutations are value mappings, not a need for arbitrary-valued shared domains. An occurrence ordinal remains necessary even when two entries name the same physical row.",
        "recommendation": "Keep UnionKey/UnionMask and compact-route maps in math_reshape, their work partition in work_schedule, and only the ordered chunk ordinal in phase.domain. A short describe binds a gather to that existing map; do not add set, filter, list-value or general variable declarations to domain on this evidence.",
        "related_concepts": ["resource"],
        "existing_mechanism_attempt": "Sparse packed uses c:[0,C(grp)] and k=c*KC+u; UnionKey[k] and UnionMask[k] supply the actual pool/id/membership. Repeated (pool,id) entries still have different k values and therefore do not collide as event instances.",
        "evidence": [ev(ip("sparse"), 106, 114), ev(ip("sparse", "tc_packed"), 43, 59),
                     ev(ip("sparse", "tc_packed"), 236, 239),
                     ev(WORK + "extract_sparse/dsv4_kernel.cu", 2364, 2374),
                     ev(WORK + "extract_sparse/dsv4_kernel.cu", 2448, 2463),
                     ev(ip("moe"), 224, 236)],
    },
    {
        "id": "I03", "operator": ["extract_gdn", "extract_fmha", "extract_moe"],
        "category": "existing_structure",
        "claim": "Some shared-domain entries are local binders or historical Boolean pseudo-loops. FMHA root z and GDN prep r can be local for variables; GDN a should be if. MoE fc1/fc2 c is instantiated as constants 0 and 1, without using a shared c loop.",
        "recommendation": "Use for(z=0:2), for(r=0:ceil_div(HeadsPerPrep,GroupWidth)) and if(Active==1); remove those unnecessary shared declarations. Constant fragment arguments keep their action identities without a global c declaration. Preserve protocol events for empty committed groups rather than deleting them as empty data.",
        "related_concepts": ["resource"],
        "evidence": [ev(ip("fmha"), 462, 464), ev(ip("fmha"), 644, 658),
                     ev(WORK + "extract_fmha/source/candidate/fmha_tcg.cuh", 4698, 4712),
                     ev(ip("gdn"), 443, 445), ev(ip("gdn"), 533, 545),
                     ev(ip("moe"), 821, 828), ev(ip("moe"), 851, 855), ev(ip("moe"), 958, 965),
                     ev(ip("moe", "fused_big_mono"), 382, 395)],
    },
    {
        "id": "I04", "operator": ["extract_fmha"], "category": "existing_structure",
        "claim": "Two independent warpgroups advancing through alternate groups do not require a stride domain or a new role-iteration scope. The existing local j mapping is short and exact; a shared global g plus per-role if is another legal control representation.",
        "recommendation": "Keep either local j with g=cLo+w+2*j stated once per role, or use for(g) with (g-cLo)%2 selecting the role's groups. Preserve separate maximum, denominator and PV handoffs and unequal tail lengths; sharing g does not create a barrier.",
        "related_concepts": ["resource"],
        "existing_mechanism_attempt": "Even traverses g under if((g-cLo)%2==0), Odd under if((g-cLo)%2==1). Their predecessor maximum is g-1. At Ng=1 Odd's loop has no payload actions while its setup/teardown still exist. Old j->g conversion must update every indexed action and dependency consistently.",
        "evidence": [ev(ip("fmha", "fused_d128_group"), 325, 338),
                     ev(ip("fmha", "fused_d128_group"), 502, 524),
                     ev(ip("fmha", "fused_d128_group"), 531, 547),
                     ev(WORK + "extract_fmha/source/candidate/fmha_tcg.cuh", 3656, 3659),
                     ev(WORK + "extract_fmha/source/candidate/fmha_tcg.cuh", 3733, 3751),
                     ev(RULE, 328, 332)],
    },
    {
        "id": "I05", "operator": ["extract_mla", "extract_moe", "extract_gemm"],
        "category": "existing_structure",
        "claim": "A slot alias or a YAML alias is not an event identity. MLA Next(w) writes slot (w+1)%2 but remains the publication requested by work w, including a terminal publication. MoE's same-index quadrant releases are distinct role events; repeated GEMM phase aliases denote distinct executions.",
        "recommendation": "Retain real occurrence indices, producer roles and phase identities. Keep the MLA reuse edge Next(w+1) after all Item(w) latches; do not replace it with same-slot equality or merge terminal Next into a nonexistent next worktile. Retain all four MoE readers in the reuse join; do not count YAML aliases as repeated authoring.",
        "related_concepts": ["resource", "milestone"],
        "evidence": [ev(ip("mla"), 457, 464), ev(ip("mla"), 626, 641),
                     ev("schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py", 1449, 1483),
                     ev(ip("moe"), 814, 819), ev(ip("moe"), 793, 795),
                     ev(ip("gemm", "wide_recovery_full"), 339, 344),
                     ev(ip("gemm", "wide_recovery_full"), 454, 456), ev(RULE, 172, 177)],
        "limitations": "This preserves the declared design identity and lifetime constraint; it is not a proof that the source's descriptor reuse, row addressing or cold-copy protocol is correct.",
    },
    {
        "id": "I06", "operator": ["extract_moe", "extract_gemm", "extract_sparse"],
        "category": "underspecified_rule",
        "claim": "The filing condition needs clarification in two directions: a single role still needs a named worktile domain, while a view's formal parameter alone should not force a same-named shared loop domain. Dependent bounds must be ready when the traversal is entered, not evaluated merely by declaring the domain.",
        "recommendation": "Explicitly allow domain for a single-role worktile/flatten sequence. Say that passing a local for variable as a view argument does not itself require a shared domain; view formals only substitute into slot/part expressions. Declare an index publicly when its identity or boundary is genuinely shared, and obtain runtime bounds by the relevant action/dependency before use. This is a proposed clarification of the current filing rule, not a claim that the strict current wording already permits all omissions.",
        "related_concepts": ["resource"],
        "existing_mechanism_attempt": "MoE gather keeps t:worktile despite having one role and no registered resource. GEMM Fragment(n,d) can alpha-rename its formal d to x without changing the slot function x%2: the formal is not the role's loop binder. A single role may bind d locally and pass its value, subject to action-window checks. Sparse C(grp) is established by union construction before loops/conditions read it; an explicit setup wait can precede the control node if needed.",
        "evidence": [ev(RULE, 69, 70), ev(RULE, 324, 328), ev(RULE, 360, 372),
                     ev(ip("moe"), 617, 631), ev(WORK + "extract_moe/source_nvfp4_moe.cu", 1189, 1203),
                     ev(ip("gemm"), 315, 321), ev(ip("gemm"), 476, 490),
                     ev(ip("sparse", "tc_packed"), 377, 382), ev(RULE, 263, 263)],
    },
    {
        "id": "I07", "operator": ["extract_gdn", "extract_fmha"], "category": "concise_semantics",
        "claim": "Several long descriptions contain necessary local traversal facts, not a missing shared-domain feature. GDN descending segment composition and FMHA bounded prefix/row-specific suffix walks can remain math-defined computations; they do not need every scalar iteration exposed as a phase event.",
        "recommendation": "Keep the input/version, traversal order, carried state, independent preparation overlap and real handoff points. Shorten the action description by citing the existing math, resources and explicit dependency. If a local iteration needs an independently referenced event, expose for/if then; descending access can use an ascending occurrence ordinal with an explicit z-1-j mapping. Do not copy CUDA's register batches or while loops into domain.",
        "related_concepts": ["resource", "milestone"],
        "evidence": [ev(ip("gdn", "segmented"), 131, 140), ev(ip("gdn", "segmented"), 619, 626),
                     ev(WORK + "extract_gdn/gdn_kernel.cu", 2265, 2278),
                     ev(ip("fmha", "tcg_d128_fscan_cp_direct_narrow"), 546, 552),
                     ev(WORK + "extract_fmha/source/candidate/fmha_tcg.cuh", 4398, 4413),
                     ev(ip("fmha", "tcg_d128_segmented_tma_xpose_narrow"), 566, 579),
                     ev(WORK + "extract_fmha/source/candidate/fmha_tc.cuh", 1489, 1497)],
        "limitations": "The segmented-transpose FMHA node is a review candidate; its empty-split storage race is a separate source/draft issue, not evidence for a new domain kind. The needs-authoring wide node was inspected only for its grouped traversal relations, not approved here.",
    },
]

sketches = [
    {
        "id": "S01", "operator": "extract_fmha", "source": ip("fmha"),
        "finding_ids": ["I03"], "status": "proposed partial rewrite using current control forms",
        "purpose": "Remove the shared z declaration while retaining two distinct half actions and their read-completion boundary.",
        "yaml": """domain:
  g: [cLo, "cHi+1"]
role:
  MAIN:
    info:
      member: "warp[0:4]"
      using_verb:
        matmul:
          ScoreHalf:
            execution_mode: async_unordered
            describe: "Compute half z of group g into TensorMemory.ScoreHalf from Query, Key(g) and their declared scales; done makes the half readable."
        load:
          Score:
            execution_mode: serial
            describe: "Read the completed half into the corresponding Score register region; all half readers finish before the shared ScoreHalf region is reused."
    pipe:
      - "for(g)":
          - "for(z=0:2)":
              - "matmul:ScoreHalf(g,z)"
              - "load:Score(g,z)"
    waiting:
      prepared:
        - "{issue load:Score(g,z)} ← [{done matmul:ScoreHalf(g,z)}]"
""",
        "omitted": "Only the two-half subgraph is shown. Original setup, K/scale readiness, other per-group actions, storage, fences and final drain remain required. This is not a complete schema or a validation of the whole verb migration.",
    },
    {
        "id": "S02", "operator": "extract_fmha", "source": ip("fmha", "fused_d128_group"),
        "finding_ids": ["I04"], "status": "proposed control and maximum-handoff projection using current forms",
        "purpose": "Show alternate group ownership without stride or mapping fields; retain the asynchronous publication contract and separate role instances.",
        "yaml": """domain:
  g: [cLo, "cHi+1"]
role:
  Even:
    info:
      member: "warp[0:4]"
      using_verb:
        compute:
          MaxState: &max_step
            execution_mode: async_unordered
            describe: "Each row waits for the preceding group's MaxTransfer before reading it, except at cLo where it uses the initial state; apply normalize and publish MaxState(g) to Exchange.MaxTransfer(g). Done covers all row publications without a producer-side join."
    pipe:
      - "for(g)":
          - "if((g-cLo)%2 == 0)":
              - "compute:MaxState(g)"
    waiting:
      prepared:
        - "{done compute:MaxState(g)} ← [{done Odd::compute:MaxState(g-1)}]"
  Odd:
    info:
      member: "warp[4:8]"
      using_verb:
        compute:
          MaxState: *max_step
    pipe:
      - "for(g)":
          - "if((g-cLo)%2 == 1)":
              - "compute:MaxState(g)"
    waiting:
      prepared:
        - "{done compute:MaxState(g)} ← [{done Even::compute:MaxState(g-1)}]"
""",
        "omitted": "This projection assumes GroupMax is already supplied by each role's original score readout. Original query setup, QK/denominator/PV dependencies, resource declarations, tail and source limitations remain. The first g-1 lies outside the declared window, so no predecessor is required there. The per-row acquire in describe protects reads; done-to-done alone would not do so.",
    },
    {
        "id": "S03", "operator": "extract_mla", "source": ip("mla"),
        "finding_ids": ["I01", "I05"], "status": "proposed partial rewrite using current index, resource and dependency forms",
        "purpose": "Keep the continuous ordinal and explicit descriptor generation despite physical slot aliases and terminal publication.",
        "yaml": """domain:
  w: worktile
  t: [0, "N(w)"]
  n: {flatten: [w, t]}
resource:
  Geometry:
    depth: 2
    location: SMEM
    shape: [field: 10]
    dtype: int32
    total_size: 80
    view:
      "Item(w)": {slot_index: "w%2"}
      "Next(w)": {slot_index: "(w+1)%2"}
role:
  Queue:
    info:
      member: "warp[10:11]"
      using_verb:
        compute:
          Next:
            execution_mode: serial
            describe: "After Softmax's request, claim work by atomic_queue and release-publish its geometry or the terminal descriptor to Geometry.Next(w); the terminal publication has a done event and creates no next worktile."
    pipe:
      - "for(w)":
          - "compute:Next(w)"
    waiting:
      allowed:
        - "{issue compute:Next(w)} ← [{done Softmax::signal:Next(w)}]"
      released:
        - "{issue compute:Next(w+1)} ← [{done Softmax::load:Item(w)}, {done Correction::load:Item(w)}, {done MMA::load:Item(w)}, {done Loader::load:Item(w)}]"
""",
        "omitted": "The other four roles, all request/latch actions and all descriptor-consumer waits remain as in the root after ordinary verb migration. No count(w) or last(w) is introduced. Geometry's 80 bytes are 2 slots x 10 int32 fields x one CTA. No new domain form is proposed.",
    },
]

data = {
    "coverage": {
        "method": "Independent index audit; no other new agent reports read; no GPU execution or extraction-state mutation.",
        "rules_snapshot": RULE,
        "corpus": "schema_discussion/phase_grammar/corpus.json",
        "root_count": len(roots), "roots": roots,
        "branch_count": len(branches), "branches": branches,
        "branch_status_counts": {s: sum(b["status"] == s for b in branches)
                                 for s in sorted({b["status"] for b in branches})},
        "lexical_triage": {
            "eligible_design_files": sum(n["semantic_eligible"] for obj in CORPUS["operators"].values() for n in obj["nodes"]),
            "method": "Read raw YAML with CSafeLoader; resolve tile_pipeline.from_parent recursively in memory; group serialized domain mappings to locate unusual forms. This scan is not a semantic review of all branches.",
            "not_reviewed_as_design": "Five unauthored scaffolds and 43 not-started nodes are excluded. Candidate and needs-authoring designs remain distinctly labelled.",
        },
        "root_domain_entries_raw_field_instances": sum(r["domain_entries_raw_field_instances"] for r in roots),
        "root_domain_entries_effective_field_instances": sum(r["domain_entries_effective_field_instances"] for r in roots),
        "counting_note": "Counts are domain-name entries at field positions, not runtime iterations. All six root tile_pipeline regions are local, so raw/effective domain counts are equal. YAML alias expansion is not author duplication; no tree-wide semantic or text-reduction percentage is claimed.",
        "source_files": [{"path": p, "sha256": SOURCES[p]["sha256"], "read_line_ranges": rs,
                          "read": "targeted source passages, plus lexical navigation"} for p, rs in source_ranges.items()],
        "prior_reports": CORPUS["reports"],
        "additional_navigation": [
            "schema_extract/archive/20261109-ten-operator-text-audit/SCHEMA_LANGUAGE_GAPS.md",
            "schema_extract/archive/20261109-ten-operator-text-audit/gap_review/code_examples_actions.md",
            "schema_discussion/phase_grammar/data/census.md",
        ],
        "conclusion_scope": "No missing_capability counterexample established within the six roots and nine targeted branches. This is not a universal sufficiency proof.",
    },
    "findings": findings,
    "naming": {
        "domain": {"recommendation": "index", "priority": "preferred rename",
                   "reason": "Names the object being shared: an ordered logical index and its bounds. The word does not imply lockstep execution or a stored data value.",
                   "candidates": {"index": "Direct and customary for an ordinal/coordinate; define ordering in the rule.",
                                  "range": "Clear for [lo,hi), but too narrow for worktile streams and flatten, and does not name index identity.",
                                  "iteration": "Sounds like an execution step; collides with work_schedule.iteration and obscures view formals and ordinal aliases.",
                                  "domain": "Mathematically valid but less direct and commonly read as an unordered value set or the whole Contract domain.",
                                  "shared_index": "Explicit but longer; phase context and the section title can supply shared."}},
        "resource": {"recommendation": "storage", "priority": "independent optional rename",
                     "reason": "The field registers on-chip storage capacity and regions, including in-place register state; it does not inventory arbitrary hardware resources or synchronization objects.",
                     "candidates": {"storage": "Broad enough for buffers, accumulators and in-place state; keep per-worker/on-chip scope explicit.",
                                    "buffer": "Familiar and concise for rings and staging, slightly narrower for persistent register state.",
                                    "allocation": "Accurate for the capacity ledger, but can suggest an allocation operation or allocator policy.",
                                    "resource": "Usable existing term but broader than the actual field."}},
        "milestone": {"recommendation": "milestone", "priority": "retain",
                      "reason": "An established workflow word for an externally named point reached when all listed existing events hold; keeping it avoids confusing it with a new hardware event or a barrier.",
                      "candidates": {"milestone": "Best concise fit for the existing all-of event alias.",
                                     "event_group": "More literal fallback if all-of grouping is otherwise unclear; longer.",
                                     "event": "Short but conflates existing issue/done events with their exported group and can imply a CUDA event object.",
                                     "checkpoint": "Suggests saved state or resumption, which this field does not provide.",
                                     "sync": "Incorrectly suggests that the declaration creates synchronization."}},
    },
    "sketches": sketches,
}

valid_categories = {"existing_structure", "concise_semantics", "underspecified_rule", "missing_capability",
                    "other_layer", "implementation_detail", "source_or_draft_issue"}
for f in data["findings"]:
    assert f["category"] in valid_categories
    assert f["evidence"]
    for e in f["evidence"]:
        span = "\n".join((ROOT / e["path"]).read_text().splitlines()[e["start_line"] - 1:e["end_line"]])
        assert e["quote"] in span
for s in data["sketches"]:
    assert isinstance(yaml.load(s["yaml"], Loader=yaml.CSafeLoader), dict)
output = BASE / "data/02_indices.json"
output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"output": str(output.relative_to(ROOT)), "roots": len(roots),
                  "targeted_branches": len(branches), "findings": len(findings),
                  "evidence_spans_checked": sum(len(f["evidence"]) for f in findings),
                  "source_files": len(source_ranges), "sketches_yaml_parsed": len(sketches)}, ensure_ascii=False))
