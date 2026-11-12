"""Reproduce direct-name dtype candidates; this is not a binding validator."""

from pathlib import Path
from collections import Counter
import hashlib
import json

import yaml


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "schema_discussion/phase_grammar/view_review"


def main():
    corpus = json.loads((BASE.parent / "corpus.json").read_text())
    rows = []
    overrides = []
    total = Counter()
    common_candidates = []
    for operator, record in corpus["operators"].items():
        path = ROOT / record["root"]
        root_meta = next(n for n in record["nodes"] if n["node"] == "root")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == root_meta["sha256"]
        document = yaml.safe_load(path.read_text())
        catalog = {}

        def register(name, definition, source):
            if isinstance(definition, dict) and "dtype" in definition:
                catalog.setdefault(name.split("(", 1)[0], []).append({
                    "dtype": str(definition["dtype"]), "source": source})

        contract = document.get("contract", {}).get("algorithm_requirement", {})
        for kind in ("input", "output"):
            for name, value in contract.get(kind, {}).items():
                register(name, value, "contract." + kind)
        for step in document["design"]["math_reshape"].get("rewrite", []):
            for name, value in step.get("symbol", {}).items():
                register(name, value, "math.symbol")

        counts = Counter()
        for phase_name, phase in document["design"]["tile_pipeline"]["phase"].items():
            storages = phase.get("storage", phase.get("resource", {}))
            counts["storage_entries"] += len(storages)
            for storage_name, storage in storages.items():
                views = storage.get("view", {})
                formulas = Counter(str(v["slot_index"]) for v in views.values() if "slot_index" in v)
                if formulas and max(formulas.values()) > 1:
                    common_candidates.append({
                        "operator": operator, "phase": phase_name, "storage": storage_name,
                        "formula_occurrences": dict(formulas),
                        "note": "Identical text only. No parent-template binding semantics are inferred."})
                for view_name, view in views.items():
                    counts["view_entries"] += 1
                    counts["empty_views"] += not bool(view)
                    for field in ("slot_index", "part", "dtype"):
                        counts["view_" + field] += field in view
                    if "dtype" not in view:
                        continue
                    matches = catalog.get(view_name.split("(", 1)[0], [])
                    types = {item["dtype"] for item in matches}
                    same = types == {str(view["dtype"])}
                    counts["direct_same_name_dtype_match"] += same
                    overrides.append({
                        "operator": operator, "phase": phase_name, "storage": storage_name,
                        "view": view_name, "parent_dtype": storage["dtype"],
                        "view_dtype": view["dtype"], "typed_names": matches,
                        "direct_same_name_match": same, "source": str(path.relative_to(ROOT)),
                    })
        rows.append({"operator": operator, "counts": dict(counts)})
        total.update(counts)
    assert total["view_entries"] == 88
    assert total["view_dtype"] == 24
    assert total["direct_same_name_dtype_match"] == 14
    result = {
        "method": "Six root field occurrences only, including aliases at each use. Dtype candidates use exact base-name matches in current Design symbols and Contract input/output definitions; reports independently check actual binding. This does not infer format or footprint for unmatched local objects.",
        "roots": rows, "total": dict(total), "dtype_overrides": overrides,
        "common_slot_text_candidates": common_candidates,
    }
    (BASE / "data/census.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"total": dict(total), "shared_formula_candidates": len(common_candidates)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
