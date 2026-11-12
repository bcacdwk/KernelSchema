#!/usr/bin/env python3
"""Reproduce the bounded view-semantics audit; never modifies source materials."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[4]
REVIEW = ROOT / "schema_discussion/phase_grammar/view_review"
CORPUS_PATH = "schema_discussion/phase_grammar/corpus.json"
CORPUS = json.loads((ROOT / CORPUS_PATH).read_text())
INPUTS = json.loads((REVIEW / "inputs.json").read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def ev(path, start, end=None):
    end = start if end is None else end
    lines = (ROOT / path).read_text().splitlines()
    assert 1 <= start <= end <= len(lines), (path, start, end)
    return {
        "path": path,
        "start_line": start,
        "end_line": end,
        "quote": "\n".join(lines[start - 1:end]),
    }


def node_meta(operator, node="root"):
    return next(n for n in CORPUS["operators"][operator]["nodes"] if n["node"] == node)


def snap(operator, node="root"):
    return node_meta(operator, node)["snapshot"]


GDN, GEMM, MLA, SPARSE, FMHA, MOE = [
    snap("extract_" + n) for n in ("gdn", "gemm", "mla", "sparse", "fmha", "moe")
]
GS = snap("extract_gemm", "mma_sync")
SP = snap("extract_sparse", "tc_packed")
MD = snap("extract_moe", "direct")
FX = snap("extract_fmha", "tcg_d128_flat_tma_xpose_narrow")
BASE = "schema_extract/archive/20261109-ten-operator-text-audit/work/"
GC = BASE + "extract_gdn/gdn_kernel.cu"
GH = BASE + "extract_gdn/gdn_common.cuh"
GU = BASE + "extract_gemm/fp8gg_umma.cu"
GM = BASE + "extract_gemm/fp8gg.cu"
MC = "schema_discussion/phase_grammar/sources/kernel/extract_mla/solution.py"
SC = BASE + "extract_sparse/dsv4_kernel.cu"
FC = BASE + "extract_fmha/source/candidate/fmha_tcg.cuh"
OC = BASE + "extract_moe/source_nvfp4_moe.cu"
RULE5 = "schema_discussion/phase_grammar/view_review/rules/5.design.tile_pipeline.md"
RULE3 = "schema_discussion/phase_grammar/view_review/rules/3.design.math_reshape.md"
RULE1 = "schema_discussion/phase_grammar/view_review/rules/1.contract.algorithm_requirement.md"

# These are the portions used for semantic conclusions, not a claim that the
# entire source files or every branch of a multipath kernel was audited.
READ_RANGES = {
    GC: [[134, 179], [1159, 1208], [1879, 1945], [2083, 2113], [2140, 2168],
         [2318, 2345], [2494, 2555], [2785, 2852]],
    GH: [[44, 76]],
    GU: [[662, 724], [822, 855], [866, 958]],
    GM: [[25, 42], [174, 213], [259, 299]],
    MC: [[80, 80], [176, 229], [379, 416], [464, 592], [692, 737],
         [787, 999], [1054, 1133], [1250, 1365], [1382, 1492]],
    SC: [[878, 899], [964, 993], [1154, 1176], [1420, 1466], [1938, 1955],
         [2201, 2227], [2340, 2375], [3150, 3206], [3454, 3480]],
    FC: [[1916, 2005], [2986, 3010], [3396, 3470], [4637, 4675],
         [4809, 4839], [4890, 4922], [4990, 5040], [5225, 5268], [5381, 5444]],
    OC: [[1921, 1979], [2214, 2245], [3000, 3033], [4339, 4372],
         [4428, 4547], [4918, 4963]],
}


def yaml_node_at(node, path):
    for key in path:
        if isinstance(node, yaml.MappingNode):
            node = next(v for k, v in node.value if k.value == str(key))
        elif isinstance(node, yaml.SequenceNode):
            node = node.value[int(key)]
        else:
            raise TypeError(path)
    return node


def node_ev(path, tree, route):
    node = yaml_node_at(tree, route)
    # Flow mappings may end midway through the last line; include that full
    # line, but not the following mapping's line for block mappings.
    last = node.end_mark.line + (1 if node.end_mark.column else 0)
    return ev(path, node.start_mark.line + 1, max(node.start_mark.line + 1, last))


LOCAL = {
    ("extract_sparse", "Q_ring"): {
        "kind": "local_alias_requires_explicit_binding",
        "source": "Q",
        "assessment": "Q is BF16 upstream; Q_ring is a physical copy name. Its format follows Q after an explicit binding, not from the string Q_ring.",
        "evidence": [ev(SPARSE, 198, 200), ev(SPARSE, 444, 455), ev(SC, 1162, 1173)],
    },
    ("extract_sparse", "PartialStage"): {
        "kind": "local_alias_requires_explicit_binding",
        "source": "Acc / PartO",
        "assessment": "Both Acc and PartO are FP32. Staging preserves those bits; full-D stride and DQ-owned columns must not be inferred from dtype alone.",
        "evidence": [ev(SPARSE, 231, 245), ev(SPARSE, 489, 493), ev(SC, 1447, 1466)],
    },
    ("extract_sparse", "RequestBounds"): {
        "kind": "local_aggregate_of_typed_inputs",
        "source": "cum_seq_lens_q plus sparse_topk_lens",
        "assessment": "Integer words copied into future FP32 reduction scratch; the composite name has no upstream dtype, but its explicitly identified inputs are int32. Without that typed binding, a local int32 declaration remains necessary.",
        "evidence": [ev(SPARSE, 39, 40), ev(SPARSE, 53), ev(SC, 964, 990)],
    },
    ("extract_sparse", "RequestLengths"): {
        "kind": "local_alias_requires_explicit_binding",
        "source": "seq_lens",
        "assessment": "Copy of int32 seq_lens into future RowSum scratch; infer only after explicit value binding.",
        "evidence": [ev(SPARSE, 40), ev(SC, 979, 989)],
    },
    ("extract_fmha", "QueryScale"): {
        "kind": "local_hardware_packing_of_typed_codes",
        "source": "sf_q",
        "assessment": "E8M0 input codes remain uint8 bytes. The replicated/sparse TMEM word arrangement is a separate hardware representation constraint.",
        "evidence": [ev(FMHA, 40, 44), ev(FC, 3430, 3440)],
    },
    ("extract_fmha", "KeyScale"): {
        "kind": "local_hardware_packing_of_typed_codes",
        "source": "sf_k / KeyAtom",
        "assessment": "E8M0 uint8 codes are copied, not numerically converted; the name lacks an upstream object declaration. Format can follow an explicit code-copy binding, while TMEM representation/footprint cannot follow logical shape alone.",
        "evidence": [ev(FMHA, 45, 49), ev(FMHA, 565, 570), ev(FC, 4645, 4649)],
    },
    ("extract_fmha", "UnitScale"): {
        "kind": "local_protocol_encoding",
        "source": "E8M0 code for unit scale required by selected block-scale MMA",
        "assessment": "Mathematical one does not uniquely imply uint8. The selected E8M0 protocol supplies the 0x7f byte encoding; retain this local representation contract once.",
        "evidence": [ev(FC, 3440, 3445), ev(FMHA, 674, 680)],
    },
    ("extract_fmha", "ValueScale"): {
        "kind": "local_hardware_packing_of_typed_codes",
        "source": "sf_v / ValueAtom",
        "assessment": "E8M0 uint8 codes are copied into the single live TMEM scale region; its consumer retirement and word arrangement are additional facts, not a dtype change.",
        "evidence": [ev(FMHA, 50, 54), ev(FMHA, 571, 575), ev(FC, 4910, 4921)],
    },
    ("extract_moe", "ScaleTM"): {
        "kind": "raw_container_spelling_for_hardware_packing",
        "source": "AScale plus BScale (E4M3 codes)",
        "assessment": "Upstream scales are FP8 E4M3; uint8 describes their raw-byte container after tcgen05.cp. No numeric E4M3-to-integer conversion occurs. The combined sparse/replicated TMEM scale representation is not an upstream tensor; its physical footprint is independently required.",
        "evidence": [ev(MOE, 651, 676), ev(MOE, 764, 779), ev(OC, 3012, 3030), ev(OC, 4918, 4961)],
    },
}


def type_registry(document):
    result = {}

    def visit(obj, route=()):
        if isinstance(obj, dict):
            for key, value in obj.items():
                nxt = route + (str(key),)
                if (isinstance(value, dict) and "dtype" in value
                        and "tile_pipeline" not in route
                        and any(s in route for s in ("symbol", "input", "output"))):
                    result[key] = (value["dtype"], nxt)
                visit(value, nxt)
        elif isinstance(obj, list):
            for index, value in enumerate(obj):
                visit(value, route + (index,))

    visit(document)
    return result


selected = [(op, "root") for op in CORPUS["operators"]] + [
    ("extract_moe", "direct"), ("extract_sparse", "tc_packed"),
    ("extract_gemm", "mma_sync"), ("extract_fmha", "tcg_d128_flat_tma_xpose_narrow"),
]
documents = []
overrides = []
for op, name in selected:
    meta = node_meta(op, name)
    path = meta["snapshot"]
    actual = sha(path)
    assert actual == meta["sha256"], path
    document = yaml.safe_load((ROOT / path).read_text())
    tree = yaml.compose((ROOT / path).read_text())
    registry = type_registry(document)
    counts = {"storage": 0, "view": 0, "dtype_override": 0}
    for phase, body in document["design"]["tile_pipeline"]["phase"].items():
        for storage, spec in body.get("resource", {}).items():
            counts["storage"] += 1
            for value, attrs in spec.get("view", {}).items():
                counts["view"] += 1
                if "dtype" not in attrs:
                    continue
                counts["dtype_override"] += 1
                if name != "root":
                    continue
                base = value.split("(")[0]
                route = ("design", "tile_pipeline", "phase", phase, "resource", storage, "view", value)
                record = {
                    "operator": op, "node": name, "phase": phase,
                    "storage": storage, "value": value,
                    "parent_dtype": spec["dtype"], "view_dtype": attrs["dtype"],
                    "evidence": [node_ev(path, tree, route)],
                }
                if base in registry:
                    declared_type, definition_route = registry[base]
                    record.update({
                        "kind": "direct_registered_value_type",
                        "upstream_dtype": declared_type,
                        "upstream_definition": ".".join(map(str, definition_route)),
                        "matches": declared_type == attrs["dtype"],
                        "assessment": "Delete view.dtype when view is defined only as a byte-region binding; the explicitly bound value retains this upstream dtype. Retain capacity/packing facts independently.",
                    })
                    record["evidence"].append(node_ev(path, tree, definition_route))
                    assert record["matches"], record
                else:
                    local = LOCAL[(op, base)]
                    record.update({key: val for key, val in local.items() if key != "evidence"})
                    record["evidence"].extend(local["evidence"])
                overrides.append(record)
    documents.append({
        "operator": op, "node": name, "category": meta["category"],
        "selected_basis": meta["selected_basis"], "snapshot": path,
        "sha256": actual, "counts": counts,
        "structural_read": "All storage/view mappings in this selected YAML were parsed.",
        "semantic_read": "Relevant value definitions, bindings, allocations and access/reuse portions were read; this is not a complete operator re-extraction.",
    })

rule_hashes = []
for rule in INPUTS["rules"]:
    assert sha(rule["snapshot"]) == rule["sha256"], rule["snapshot"]
    rule_hashes.append({"path": rule["snapshot"], "sha256": rule["sha256"]})
sources = []
for op, data in CORPUS["operators"].items():
    for src in data["sources"]:
        actual = sha(src["path"])
        assert actual == src["sha256"], src["path"]
        sources.append({
            "operator": op, "path": src["path"], "sha256": actual,
            "logical_path": src["logical_path"], "source_ref": src["source_ref"],
            "matches_corpus": True, "new_source": False,
            "semantic_read_ranges": READ_RANGES.get(src["path"], []),
            "read_scope": "selected semantic sections" if src["path"] in READ_RANGES else "hash verification or navigation only",
        })

findings = [
    {
        "id": "VS01", "claim": "24 root dtype overrides are not 24 independent format decisions: 14 directly match registered upstream value dtypes; 10 refer to local aliases, aggregates or hardware representations.",
        "recommendation": "Delete view.dtype entirely and define view only as a byte-region/reservation binding. Numerical format comes from its bound design value; local/packed formats must be defined once by the materializing action or referenced realization/primitive contract. Parent dtype remains a capacity coordinate unit, not an access interpretation or implicit conversion.",
        "evidence": [ev(RULE1, 64, 67), ev(RULE3, 79, 92), ev(RULE5, 346, 374),
                     ev(MLA, 188), ev(MLA, 213), ev(MLA, 387, 394), ev(MC, 80), ev(MC, 464, 470),
                     ev(GDN, 197, 199), ev(GDN, 228, 233), ev(GDN, 264, 265), ev(GDN, 577, 586)],
        "root_dtype_overrides": overrides,
        "boundary": "The Contract compute dtype is a reference-computation dtype, not an automatic storage mandate. Prefer the selected design value, resolve refs and precision choices, then bind the view.",
    },
    {
        "id": "VS02", "claim": "Known logical dtype does not determine storage footprint: triangular selection, padding and hardware scale placement can change capacity without changing the stored numerical values.",
        "recommendation": "Preserve one physical footprint/representation contract per affected binding or aggregate; remove duplicate numerical dtype declarations. Do not replace capacity with product(logical_shape)*sizeof(dtype) unless dense unpadded materialization is established.",
        "evidence": [ev(GDN, 264, 265), ev(GDN, 478, 483), ev(GH, 50, 65),
                     ev(GC, 1171, 1189), ev(GC, 1903, 1907),
                     ev(MOE, 651, 676), ev(MOE, 764, 779), ev(OC, 3012, 3030), ev(OC, 4918, 4961),
                     ev(FMHA, 40, 54), ev(FC, 3430, 3445)],
    },
    {
        "id": "VS03", "claim": "An exact subrange can be omitted for whole-region sequential reuse only after all conflicting accesses, including those of other members, are proven retired. GEMM's drained epilogue supports that case; MLA S/P does not establish a whole-group S-read retirement before its first P write.",
        "recommendation": "Keep MLA P's existing part and row-ownership/source-limitation constraints; delete only its duplicate dtype and lift the common slot map. Whole-slot reservation remains a candidate for proven drained reuse such as GEMM, without implying every byte is written. Per-warp pipe order must not be promoted into an implicit whole-group join.",
        "evidence": [ev(MLA, 477, 491), ev(MLA, 499, 514), ev(MC, 805, 809),
                     ev(MC, 852, 858), ev(MC, 918, 945), ev(MC, 1064, 1100), ev(MC, 1273, 1307),
                     ev(GS, 158, 168), ev(GS, 201, 208), ev(GM, 28, 38), ev(GM, 263, 296),
                     ev(FX, 503, 512), ev(FX, 604, 612), ev(FC, 5225, 5246), ev(FC, 5394, 5424)],
        "caveat": "This is a proposed change to current part omission semantics, not permission to silently delete part under the current rules.",
        "directed_recheck": {
            "status": "MLA no-part whole-slot rewrite withdrawn; dtype deletion unchanged",
            "superseded_claim": "The earlier audit incorrectly concluded that local produce:Final before publish:P proved all S readers retired before any P write.",
            "counterexample_scope": "First work item, T=0, at least two KV tiles, after QK(0) completion; no old-P wait and no final-item o_done wait apply to this iteration.",
            "allowed_source_trace": [
                "Warp 0 observes QK(0), finishes both S half-loads, and arrives at s_free[0].",
                "Warp 1 remains before its first S half-load; no whole-group barrier forces it to catch up.",
                "Warp 0 computes/encodes P and writes its P region, waits for its own stores, then arrives at p_ready[0].",
                "Warp 1's S read may occur later. Thus first P write does not follow all members' S reads.",
            ],
            "synchronization_audit": {
                "s_free": "Four arrives; the future QK issuer waits for it, but the P writer does not. Arrive also precedes the optional tiny S reread, so it alone is not a retirement event for every necessary S read.",
                "p_ready": "Four arrives after the P writes; the PV issuer waits for it. It cannot retroactively retire readers before those writes.",
                "o_done": "Protects previous work-item output/final scratch, not the current S read set.",
                "pv_done": "Protects consumption of older P generations, not current S readers.",
                "syncwarp_and_tmem_wait": "Supply local warp/access ordering, not a rendezvous of all four Softmax warps.",
            },
            "strict_whole_reservation": "Requiring all 128 rows of S to retire before any P write adds a cross-warp execution constraint that the existing per-row pipeline does not supply.",
            "capacity_only_region": "Avoids that added join only by leaving placement unconstrained; necessary row correspondence, P span, overlap sets and reader retirement then still need explicit representation elsewhere. Bare sharing is insufficient.",
            "source_address_limitation": "All S loads use tb=tb0 while P stores add warp<<21. Do not assume the intended disjoint S row ownership is already implemented, and do not use this known defect as permission for freer placement. Keeping the current part is not a claim that the source defect is repaired.",
            "revised_scope": "No claim that column 32 is intrinsically immutable. Any later relocation proof must preserve row/column conflict topology, footprint, encoding and all-reader ordering; this audit has not discharged it.",
        },
    },
    {
        "id": "VS04", "claim": "Partial sharing cannot be reduced to a single unqualified shared-with edge when other subregions or old readers remain live. GDN's last mini chunk has a concrete stage-0 read/write conflict; FMHA maintains live accumulator and scale regions during score/probability turnover.",
        "recommendation": "Preserve overlap/disjointness, footprint, binding identity and required read/write retirement. Exact byte/column origins can be late variables if these constraints and hardware alignment are retained. Split independent disjoint design regions when that avoids irrelevant arena offsets.",
        "evidence": [ev(GDN, 565, 586), ev(GDN, 651, 659), ev(GC, 2499, 2516), ev(GC, 2543, 2550), ev(GC, 2819, 2839),
                     ev(FMHA, 495, 507), ev(FMHA, 548, 555), ev(FMHA, 594, 620), ev(FC, 1935, 1946),
                     ev(SPARSE, 404, 412), ev(SC, 887, 893), ev(SC, 979, 990)],
        "source_status": "The GDN race is an already-recorded source limitation supported by direct source reads, not a new GPU-tested result. No source fix is made.",
    },
    {
        "id": "VS05", "claim": "Most identical slot mappings can be declared once per storage, but slot depth alone does not select a version ordinal or reset scope. Descriptor aliases and an end-of-loop overlay explain actual same-backing exceptions.",
        "recommendation": "Lift a common explicitly bound slot template to storage; retain per-view exceptions or canonical stored-value aliases. Never infer modulo from the first argument, the loop variable name or depth. Allow uniform physical slot renumbering, but preserve collision classes and generation/reset semantics.",
        "evidence": [ev(RULE5, 364, 374), ev(GDN, 577, 586), ev(GEMM, 266, 321),
                     ev(GS, 194, 196), ev(MLA, 364, 367), ev(MLA, 457, 464), ev(MLA, 626, 641),
                     ev(MC, 493, 507), ev(MC, 1447, 1483), ev(MC, 1388, 1412),
                     ev(MOE, 633, 668), ev(OC, 4501, 4532)],
        "descriptor_alias": "Use a local Descriptor(g) value, including a terminal descriptor. Item(w) reads Descriptor(w); publish:Next(w) writes Descriptor(w+1). Action indexes need not be changed. Next(w) is not necessarily a successful next worktile.",
    },
    {
        "id": "VS06", "claim": "Local protocol values supply genuine cases where mathematics alone cannot choose a storage representation: a ten-int descriptor, packed uint64 hash publication, and the E8M0 encoding of unit scale.",
        "recommendation": "Keep local representation declarations or reference the selected primitive's format contract; do not fabricate upstream math tensors just to make dtype inference work. These cases do not require view.dtype: their encoding is already a property of the local object or its producing protocol. Unknown names must fail binding validation rather than inherit a guessed type.",
        "evidence": [ev(RULE3, 90, 92), ev(MLA, 298, 304), ev(MLA, 457, 464), ev(MC, 493, 507),
                     ev(SP, 252, 267), ev(SP, 363, 384), ev(SC, 2201, 2212), ev(SC, 2346, 2362),
                     ev(FC, 3440, 3445), ev(SC, 979, 990)],
        "delete_field_counterfactual": [
            {"case": "Sparse HashTable", "result": "No information loss if build retains its existing 8192-word uint64 and single-word key/position publication contract; view retains the byte span."},
            {"case": "FMHA UnitScale", "result": "No independent view format: unit is encoded as E8M0 0x7f by the selected primitive. dtype:uint8 alone never specified that fact."},
            {"case": "MoE ScaleTM", "result": "Deleting dtype:uint8 does not remove the E4M3 code-copy semantics already supplied by AScale/BScale and tcgen05.cp/MMA; keep the physical footprint and representation binding."},
            {"case": "MLA Geometry", "result": "One local descriptor definition supplies ten int32 fields and the terminal sentinel; Item/Next views only identify versions/regions."},
        ],
    },
    {
        "id": "VS07", "claim": "Whole-slot same-format reuse and independent copies are different relations. Sparse Scores/E and MoE WarpCounts/FinalRoutes can be concise, while MLA P in registers versus TMEM and GEMM's double-buffered drain still need distinct physical bindings.",
        "recommendation": "Retain a single allocation ledger with optional minimal value bindings. Omit empty/identity bindings for ordinary single-value storage. Do not flatten into duplicated allocation rows or collapse distinct-location/overlapping-lifetime copies under one value name.",
        "evidence": [ev(RULE5, 340, 360), ev(SP, 327, 334), ev(SC, 3196, 3205),
                     ev(MD, 293, 305), ev(MD, 370, 376), ev(OC, 1960, 1974), ev(OC, 2218, 2237),
                     ev(MLA, 387, 412), ev(GEMM, 315, 326), ev(GU, 866, 884), ev(GU, 912, 924)],
    },
]

proposals = [
    {
        "id": "VP01", "claim": "Recommend one hierarchy: allocation once, optional bound views, inherited common slot mapping, and subregions only when lifetime/capacity constraints need them.",
        "recommendation": "Keep a minimal view mechanism, delete view.dtype, lift identical slot mappings, and reserve subregions for necessary coexistence/partial-preservation constraints. Do not mandate a flat per-value allocation table or a source-API arena as the design unit.",
        "evidence": [ev(RULE5, 340, 360), ev(MLA, 387, 412), ev(FX, 503, 543)],
        "fields": [
            {"field": "storage id/location/depth/capacity", "decision": "retain", "reason": "Defines an allocation class, independent copies and budget; count once."},
            {"field": "storage.dtype", "decision": "retain as capacity coordinate unit; infer only for a clearly homogeneous dense slot", "reason": "Mixed arena byte units and TMEM FP32 columns are not the bound value format."},
            {"field": "view value binding", "decision": "retain when nontrivial; omit identity/single-value boilerplate", "reason": "Needed to distinguish copies and shared residents."},
            {"field": "view.dtype", "decision": "delete the field entirely; view has no independent numerical interpretation", "reason": "14/24 direct root overrides duplicate upstream types; the remaining local formats belong to object/producer/realization contracts. None of the audited exceptions requires a view-level field."},
            {"field": "view.slot_index", "decision": "lift common template to storage; keep exceptions", "reason": "The declaration, selected ordinal and reset scope remain explicit. No universal modulo default."},
            {"field": "view.part", "decision": "remove only for proven whole-region retirement; retain MLA's current part and any other necessary subregion constraints", "reason": "Per-member order is not whole-group retirement. Late placement must preserve extent, containment, row/column conflict sets and retirement without adding undeclared joins."},
            {"field": "exact byte/column origin", "decision": "late to realization unless it changes the selected overlap schedule or externally fixed encoding", "reason": "Uniform relocation/packing can be implementation choice; source extraction still records realized maps."},
            {"field": "physical packing footprint", "decision": "retain or reference selected representation contract", "reason": "Triangular selection, row padding, replicated scale format and allocation granularity break dense-shape byte inference."},
        ],
    },
    {
        "id": "VP02", "claim": "Removing the view format role is cleaner than retaining a format override that is usually copied from elsewhere.",
        "recommendation": "Resolve selected design value -> explicit refs/typed slices -> the numerical object used by the action, without creating a view format. An alternate materialized representation must cite a defined conversion/encoding. A local tuple/protocol object needs one explicit representation definition in its materializing action or a referenced realization/primitive contract; all accesses bind that same object/representation. No inference from spelling, mathematical-integer range, parent capacity dtype or diffuse prose hints.",
        "evidence": [ev(RULE1, 64, 67), ev(RULE3, 79, 92), ev(MOE, 651, 676), ev(OC, 3012, 3030), ev(SP, 363, 375)],
        "cases_requiring_an_explicit_non_view_source": ["Reference-only Contract intermediate whose Design precision differs", "Unbound local name", "Raw tuple with heterogeneous fields", "Packed atomic protocol word", "Hardware replication or sub-byte pack not described by scalar dtype"],
        "completeness_condition": "A reviewer must be able to name one source for each stored representation and its footprint. Deleting view.dtype is not permission to omit those facts or infer them from the parent capacity unit.",
    },
    {
        "id": "VP03", "claim": "Late part placement must carry checkable constraints, not missing information.",
        "recommendation": "For each subregion give a physical extent or proven upper bound and containment; give alias/disjoint constraints for simultaneously live uses; require overlapping writes to follow all conflicting reads and outstanding fills across all relevant members. Coder chooses legal origins/swizzles and discharges the constraints in realization. If a choice changes depth, copy count, overlap, synchronization or budget, it is a design change. An existing source-layout defect is not evidence permitting weaker constraints.",
        "evidence": [ev(GC, 1903, 1935), ev(GDN, 651, 659), ev(SC, 878, 893), ev(FC, 1935, 1946), ev(FC, 5225, 5246)],
        "minimal_constraints": [
            "R(value,version) is contained in the selected slot/arena and fits its capacity.",
            "Distinct simultaneously live values use disjoint bytes unless a defined same-bits alias permits sharing.",
            "Every write to Rnew follows all old reads over Rnew intersect Rold, including other warps/members, and all outstanding writers to the same bytes; local serial order alone cannot establish a group-wide edge.",
            "Partial-value preservation identifies the retained logical/physical region; a full-value name alone is insufficient.",
            "Only one capacity charge per backing; separate-location copies are charged separately.",
            "Selected transfer/compute representation supplies required packing, alignment and address-class constraints.",
        ],
        "late_placement_arithmetic_example": {
            "case": "GDN VTT=64 final-state staging",
            "stage_bytes": 52224,
            "final_reservation_bytes": 33280,
            "source_final_interval": [0, 33280],
            "Kc_interval": [0, 16384],
            "candidate_interval": [16384, 49664],
            "claim": "The candidate interval fits and avoids the entire Kc region and tail Ai/Dl. This demonstrates freedom left by region constraints, not a source change or a GPU-validated implementation; all other access and representation constraints still apply.",
            "evidence": [ev(GDN, 577, 586), ev(GC, 1903, 1907), ev(GC, 2819, 2839)],
        },
    },
]

root_docs = [d for d in documents if d["node"] == "root"]
totals = {k: sum(d["counts"][k] for d in root_docs) for k in ("storage", "view", "dtype_override")}
direct = sum(o["kind"] == "direct_registered_value_type" for o in overrides)
assert totals == {"storage": 82, "view": 88, "dtype_override": 24}
assert direct == 14 and len(overrides) - direct == 10
assert sha(CORPUS_PATH) == INPUTS["parent_corpus"]["sha256"]

output = {
    "coverage": {
        "format": "kernelschema.view_review.semantics.v1",
        "author": "view_semantics_review",
        "current_rules": rule_hashes,
        "excluded_rules_snapshot": "schema_discussion/phase_grammar/sources/rules (pre-rename; not used as current normative input)",
        "corpus": {"path": CORPUS_PATH, "sha256": sha(CORPUS_PATH)},
        "selected_documents": documents,
        "root_totals": totals,
        "root_dtype_override_classification": {"direct_registered_value_match": direct, "local_alias_aggregate_or_encoding": len(overrides) - direct},
        "source_hashes": sources,
        "new_source_hashes": [],
        "source_hash_count": len(sources),
        "semantic_source_file_count": len(READ_RANGES),
        "branch_scope": "Four approved branches only; no review_candidate/needs_authoring results were promoted to evidence.",
        "source_limitations": [
            "MLA source has a compiled MODE1 FP16 alternative, but the frozen root selects _PV_MODE=0; MODE1 is a scope-qualified contrast, not root dtype evidence.",
            "Current-rule BF16 S/P example is illustrative, not claimed as an observed six-root BF16 P alias.",
            "GDN mini-chunk stage-0 race, MLA descriptor/row-address discrepancies, Sparse tc_packed erroneous protocol aliases and FMHA xpose empty-path reuse gap remain source limitations; no GPU validation or fix is claimed.",
        ],
        "directed_recheck": "MLA S/P no-part example was independently falsified by its per-warp ordering and synchronization scopes. Final report/data keep the current P part and row-address limitation; only dtype deletion/common slot-map hoisting remain in that rewrite.",
        "execution": "Read-only analysis of source materials; no GPU; generated report data and bounded audit script only.",
    },
    "findings": findings,
    "proposals": proposals,
}

dest = REVIEW / "data/01_semantics.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"output": str(dest.relative_to(ROOT)), "root_totals": totals,
                  "source_hashes": len(sources), "findings": len(findings),
                  "dtype_direct_matches": direct}, ensure_ascii=False))
