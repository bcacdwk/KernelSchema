"""Inventory flow fields without treating source text as current valid syntax."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "schema_discussion/flow_review"


def main():
    corpus = json.loads((OUT / "corpus.json").read_text())
    rows, patterns, errors = [], {}, []
    field_count, order_count, protocol_count, storage_count = (Counter() for _ in range(4))
    category_fields = {}
    for opid, operator in corpus["operators"].items():
        nodes = {n["node"]: n for n in operator["nodes"] if n.get("snapshot")}
        docs = {}
        for node_id, meta in nodes.items():
            try:
                docs[node_id] = yaml.load((ROOT / meta["snapshot"]).read_text(), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
            except yaml.YAMLError as exc:
                errors.append({"operator": opid, "node": node_id, "error": str(exc)})

        def resolve(node_id, seen=()):
            if node_id in seen:
                raise ValueError(f"inheritance cycle: {seen}, {node_id}")
            doc = docs[node_id]
            work = (doc.get("design") or {}).get("work_schedule")
            if isinstance(work, dict) and "from_parent" in work:
                return resolve(work["from_parent"], (*seen, node_id))
            return work, node_id

        for node_id, meta in nodes.items():
            if node_id not in docs:
                continue
            try:
                work, owner = resolve(node_id)
            except (KeyError, ValueError) as exc:
                errors.append({"operator": opid, "node": node_id, "error": str(exc)})
                continue
            flow = work.get("flow") if isinstance(work, dict) else None
            row = {"operator": opid, "node": node_id, "category": meta["category"],
                   "semantic_eligible": meta.get("semantic_eligible", False),
                   "path": meta["snapshot"], "work_schedule_owner": owner,
                   "has_flow": isinstance(flow, dict)}
            if isinstance(flow, dict):
                canonical = json.dumps(flow, sort_keys=True, ensure_ascii=False)
                pattern_id = opid + ":" + hashlib.sha256(canonical.encode()).hexdigest()[:12]
                row["pattern"] = pattern_id
                row["phase_count"] = len(work.get("phase", {}))
                row["launch_count"] = len(flow.get("launch", []))
                row["handoff_count"] = len(flow.get("handoff", {}))
                line = next((i for i, s in enumerate((ROOT / nodes[owner]["snapshot"]).read_text().splitlines(), 1)
                             if s.strip() == "flow:"), None)
                row["flow_path"] = nodes[owner]["snapshot"]
                row["flow_line"] = line
                pattern = patterns.setdefault(pattern_id, {"operator": opid,
                    "representative": nodes[owner]["snapshot"], "line": line,
                    "flow": flow, "occurrences": []})
                pattern["occurrences"].append({"node": node_id, "category": meta["category"], "local": owner == node_id})
                if owner == node_id and meta.get("semantic_eligible"):
                    category_counter = category_fields.setdefault(meta["category"], Counter())
                    for launch in flow.get("launch", []):
                        field_count.update("launch." + k for k in launch)
                        category_counter.update("launch." + k for k in launch)
                        order_count[launch.get("order", "(single phase)")] += 1
                        if "protocol" in launch:
                            protocol_count[str(launch["protocol"])] += 1
                    for handoff in flow.get("handoff", {}).values():
                        field_count.update("handoff." + k for k in handoff)
                        category_counter.update("handoff." + k for k in handoff)
                        storage = str(handoff.get("storage"))
                        storage_count["global" if storage == "global" else "on-chip reference"] += 1
            rows.append(row)
    result = {"scope": "Counts of local, semantic-eligible work_schedule fields; repeated branches are not independent semantic evidence.",
              "rows": rows, "patterns": patterns, "errors": errors,
              "counts": {"files_parsed": len(rows), "by_category": dict(Counter(r["category"] for r in rows)),
                         "with_flow": sum(r["has_flow"] for r in rows),
                         "local_flow": sum(r["has_flow"] and r["node"] == r["work_schedule_owner"] for r in rows),
                         "field_instances": dict(field_count), "order": dict(order_count),
                         "protocol": dict(protocol_count), "storage": dict(storage_count),
                         "field_instances_by_category": {k: dict(v) for k, v in category_fields.items()}}}
    (OUT / "data/census.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["# Flow 机械盘点", "", "统计固定输入中的字段使用，不等于当前规则的语义批准；候选稿与已批准稿保留各自状态。", "",
             "## 十个 root", "", "| 算子 | 状态 | phase | launch | handoff | 位置 |", "| --- | --- | ---: | ---: | ---: | --- |"]
    for row in rows:
        if row["node"] == "root" and row["has_flow"]:
            lines.append(f"| {row['operator']} | {row['category']} | {row['phase_count']} | {row['launch_count']} | {row['handoff_count']} | [{row['flow_line']}]({ROOT / row['flow_path']}:{row['flow_line']}) |")
    lines += ["", "## 分支形态", "", "同一算子按解析后 flow 的完整内容去重，ID 不同仍算不同，不能当作语义拓扑数量。", "",
              "| 算子 | 有 flow 的文件 | 本地 flow | 不同 flow 内容 |", "| --- | ---: | ---: | ---: |"]
    for opid in corpus["operators"]:
        selected = [r for r in rows if r["operator"] == opid and r["has_flow"]]
        lines.append(f"| {opid} | {len(selected)} | {sum(r['node'] == r['work_schedule_owner'] for r in selected)} | {len({r['pattern'] for r in selected})} |")
    lines += ["", "## 本地字段计数", "", "只数 semantic_eligible 的本地 work_schedule；YAML alias 按字段实例计数，from_parent 不重复计入。", "", "```json", json.dumps(result["counts"], ensure_ascii=False, indent=2), "```", ""]
    (OUT / "data/census.md").write_text("\n".join(lines))
    print(json.dumps({"counts": result["counts"], "errors": errors}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
