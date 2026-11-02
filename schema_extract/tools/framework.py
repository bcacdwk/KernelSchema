"""Optional framework proposals and the maintainer-owned issue directory."""

import json
import re

from .records import StateError, read_json


CATEGORIES = ("schema_docs", "schema_template", "kernel_wiki", "extract_workflow", "extract_tools")
OUTCOMES = ("declined", "resolved", "partially_adopted")
TEXT = {"type": "string"}
FIELDS = {"id": TEXT, "category": {"type": "string", "enum": list(CATEGORIES)},
          "problem": TEXT, "evidence": {"type": "array", "items": TEXT}, "proposal": TEXT, "necessity": TEXT}
CANDIDATES = {"type": "array", "items": {"type": "object", "properties": FIELDS,
              "additionalProperties": False, "required": list(FIELDS)}}
GUIDANCE = (
    "framework_candidates normally stays []. Only report a necessary change to shared schema_docs rules, "
    "schema_template checks, kernel_wiki materials, extraction workflow or extraction tools. "
    "Identify the problem with a stable lowercase ASCII slug independent of the run/operator fingerprint, "
    "cite reproducible file/line or run evidence, and explain why existing rules leave an error, omission or repeated "
    "work. Propose the smallest change that addresses it. A single demonstrated counterexample can suffice. "
    "Use ordinary task issues for local source mismatches, missing proofs and extraction fixes. "
    "Do not investigate just to find proposals, create ledger files, decide proposals, or change the task verdict for one."
)


def valid_candidate(item):
    return (isinstance(item, dict) and set(item) == set(FIELDS)
            and all(isinstance(item[key], str) and item[key].strip() for key in FIELDS if key != "evidence")
            and re.fullmatch(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", item["id"])
            and item["category"] in CATEGORIES and isinstance(item["evidence"], list)
            and item["evidence"] and all(isinstance(s, str) and s.strip() for s in item["evidence"]))


def candidates_from(value):
    """Malformed/legacy optional data does not reject a task receipt or a tick."""
    items = value.get("framework_candidates") if isinstance(value, dict) else None
    return [item for item in items if valid_candidate(item)] if isinstance(items, list) else []


def registry(repo):
    """Read and check single-file identities across pending categories and decisions."""
    root = repo / "schema_extract/issues"
    if not root.is_dir():
        raise StateError(f"framework issue directory missing: {root}; select the maintained repository with --repo")
    index = {}
    for path in sorted(root.rglob("*.json")):
        record = read_json(path)
        directories = path.parent.relative_to(root).parts
        archived = directories[:1] == ("archive",)
        if not isinstance(record, dict):
            raise StateError(f"invalid framework record: {path}")
        proposal = {k: v for k, v in record.items() if k != "decision"}
        if not valid_candidate(proposal) or proposal["id"] != path.stem or path.stem in index:
            raise StateError(f"invalid or duplicate framework identity: {path}")
        category = proposal["category"]
        expected = ("archive", category) if archived else (category,)
        if directories != expected or (not archived and "decision" in record):
            raise StateError(f"expected issue under {root.joinpath(*expected)}: {path}")
        if archived:
            decision = record.get("decision")
            if (not isinstance(decision, dict) or decision.get("outcome") not in OUTCOMES
                    or not isinstance(decision.get("reason"), str) or not decision["reason"].strip()):
                raise StateError(f"archive requires an outcome and its reason: {path}")
            if decision["outcome"] == "partially_adopted" and not all(
                    isinstance(decision.get(k), str) and decision[k].strip() for k in ("adopted", "remaining")):
                raise StateError(f"partial decision requires adopted and remaining scope: {path}")
        index[path.stem] = {"path": str(path.relative_to(repo)), "category": proposal["category"],
                            "problem": proposal["problem"],
                            "status": record["decision"]["outcome"] if archived else "pending"}
    return index


def candidate_report(store, plan):
    """Project accepted reports only. The same task may belong to multiple plans."""
    groups = {}
    registered = registry(store.repo)

    def include(items, source):
        for item in items:
            versions = groups.setdefault(item["id"], {})
            report = versions.setdefault(json.dumps(item, sort_keys=True), {"candidate": item, "sources": []})
            if source not in report["sources"]:
                report["sources"].append(source)

    for key in plan["tasks"]:
        state = store.load(key)
        for checkpoint in state.get("history", []):
            path = store.task_dir(key) / "checkpoints" / checkpoint / "checkpoint.json"
            record = read_json(path)
            # Seeds preserve the original receipt for audit, not as a new report.
            receipt = record.get("worker_receipt", {})
            if (record["id"] == checkpoint and record["fingerprint"] == state["fingerprint"]
                    and receipt.get("task_key") == key):
                include(candidates_from(receipt), {"role": "worker", "record": str(path.relative_to(store.root)),
                                                   "checkpoint": checkpoint})
        for report in state.get("review_candidate_reports", []):
            include(report["candidates"], report["source"])

    supervisor = store.root / "runs" / plan["run_id"] / "supervisor/state.json"
    if supervisor.exists():
        state = read_json(supervisor)
        if state["run_id"] != plan["run_id"]:
            raise StateError("supervisor state belongs to another run")
        for report in state.get("candidate_reports", []):
            include(report["candidates"], report["source"])
    return [{"id": key, "registered": registered.get(key), "reports": list(groups[key].values())}
            for key in sorted(groups)]
