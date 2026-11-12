"""Check two event-notation arguments, not full kernel correctness."""

from collections import deque
from pathlib import Path
import hashlib
import json

import yaml


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "schema_discussion/phase_grammar"
SOURCE = ROOT / "schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/double_verify_root.yaml"


def main():
    document = yaml.safe_load(SOURCE.read_text())
    flow = document["design"]["work_schedule"]["flow"]
    phases = document["design"]["tile_pipeline"]["phase"]
    handoff = flow["handoff"]["Clean"]
    edges = {}

    def edge(before, after):
        edges.setdefault(before, set()).add(after)

    for launch in flow["launch"]:
        # The selected case has one phase per launch; no composition is inferred.
        assert len(launch["phase"]) == 1
        phase = launch["phase"][0]
        start = phase + ".START"
        edge(start, phase + ".launch_ready")
        edge(start, phase + ".kernel_done")
        predecessors = launch.get("start_after", [])
        if isinstance(predecessors, str):
            predecessors = [predecessors]
        for predecessor in predecessors:
            edge(predecessor, start)

    def path_to(start, target):
        queue = deque([(start, [start])])
        seen = {start}
        while queue:
            node, path = queue.popleft()
            if node == target:
                return path
            for child in sorted(edges.get(node, ())):
                if child not in seen:
                    seen.add(child)
                    queue.append((child, path + [child]))
        return None

    assert handoff["ready"] == "verify.kernel_done"
    aliases = handoff["wait_at"]
    assert isinstance(aliases, list) and len(aliases) == 7
    targets = []
    for reference in aliases:
        phase, name = reference.split(".", 1)
        assert phase in handoff["consumer"]
        events = phases[phase]["milestone"][name]
        for event in events:
            qualified = phase + "::" + event
            edge(phase + ".START", qualified)
            proof = path_to(handoff["ready"], qualified)
            assert proof, f"Not implied by startup constraints: {qualified}"
            targets.append({"alias": reference, "event": qualified, "startup_path": proof})
    assert len(targets) == 19

    ready, first, second = 10, 5, 15
    counterexample = {
        "ready": ready, "consumer_events": [first, second],
        "group_completion": max(first, second),
        "aggregate_target_passes": ready <= max(first, second),
        "all_individual_targets_pass": all(ready <= t for t in (first, second)),
    }
    assert counterexample["aggregate_target_passes"]
    assert not counterexample["all_individual_targets_pass"]
    result = {
        "status": "pass",
        "scope": "Two declaration-level arguments only; does not validate complete schemas, CUDA scheduling, visibility, guard implementation, or source correctness.",
        "input": str(SOURCE.relative_to(ROOT)),
        "input_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "assumptions": [
            "start_after constrains the start of the whole target launch",
            "launch_ready cannot occur before its source launch starts",
            "each existing local action event occurs after its own launch starts",
            "the selected explicit startup edges remain unchanged",
        ],
        "redundant_wait_at": {"aliases": 7, "individual_targets": 19,
                              "handoff_edges_used_in_proof": False, "proofs": targets},
        "consumer_group_counterexample": counterexample,
    }
    (BASE / "data/event_reasoning_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "pass", "startup_implied_targets": len(targets),
                      "removable_aliases_in_this_case": len(aliases),
                      "aggregate_target_counterexample": counterexample}, ensure_ascii=False))


if __name__ == "__main__":
    main()
