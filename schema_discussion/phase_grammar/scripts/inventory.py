"""Read-only structural census of the fixed six-operator corpus.

Counts are field occurrences, not distinct allocations or independent designs.
Whole-area from_parent references are resolved without running legacy validators.
"""

from collections import Counter
from pathlib import Path
import hashlib
import json

import yaml


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "schema_discussion/phase_grammar"
AREAS = ("math_reshape", "work_schedule", "tile_pipeline")


def mapping(value):
    return value if isinstance(value, dict) else {}


def count_design(design):
    result = Counter()
    pipeline = mapping(mapping(design).get("tile_pipeline"))
    schedule = mapping(mapping(design).get("work_schedule"))
    phases = mapping(pipeline.get("phase"))
    result["phases"] = len(phases)
    for phase in phases.values():
        phase = mapping(phase)
        roles = mapping(phase.get("role"))
        resources = mapping(phase.get("resource"))
        domains = mapping(phase.get("domain"))
        milestones = mapping(phase.get("milestone"))
        result["roles"] += len(roles)
        result["resources"] += len(resources)
        result["domains"] += len(domains)
        result["milestones"] += len(milestones)
        result["phases_with_resources"] += bool(resources)
        result["phases_with_domains"] += bool(domains)
        result["phases_with_milestones"] += bool(milestones)
        result["single_role_phases_with_domains"] += len(roles) == 1 and bool(domains)
        for resource in resources.values():
            resource = mapping(resource)
            views = mapping(resource.get("view"))
            result["views"] += len(views)
            result["resources_with_views"] += bool(views)
            result["resources_with_member"] += "member" in resource
            result["resources_with_total_size"] += "total_size" in resource
            location = resource.get("location", resource.get("space", "missing"))
            result["location:" + str(location)] += 1
            for view in views.values():
                view = mapping(view)
                for key in ("slot_index", "part", "dtype"):
                    result["views_with_" + key] += key in view
        for domain in domains.values():
            if isinstance(domain, list) and len(domain) == 2:
                kind = "range"
            elif domain == "worktile":
                kind = "worktile"
            elif isinstance(domain, dict) and "flatten" in domain:
                kind = "flatten"
            else:
                kind = "other"
            result["domain:" + kind] += 1
        for milestone in milestones.values():
            result["milestone_events"] += len(milestone) if isinstance(milestone, list) else 0
        for role in roles.values():
            role = mapping(role)
            wait = mapping(role.get("waiting", role.get("waits_for")))
            result["roles_with_waiting"] += bool(wait)
            for entries in wait.values():
                result["waiting_rules"] += len(entries) if isinstance(entries, (dict, list)) else 0
    flow = mapping(schedule.get("flow"))
    handoff = mapping(flow.get("handoff"))
    result["handoff_values"] = len(handoff)
    for item in handoff.values():
        item = mapping(item)
        for key in ("ready", "wait_at", "initialize"):
            result["handoff_with_" + key] += key in item
        result["handoff_global"] += item.get("storage") == "global"
    launches = flow.get("launch", [])
    if not isinstance(launches, list):
        launches = []
    result["launches"] = len(launches)
    for launch in launches:
        launch = mapping(launch)
        for key in ("start_after", "protocol", "order"):
            result["launches_with_" + key] += key in launch
        if "order" in launch:
            result["order:" + str(launch["order"])] += 1
    return dict(result)


def value_at(node, keys):
    for key in keys:
        if not isinstance(node, yaml.MappingNode):
            return None
        node = next((v for k, v in node.value if k.value == key), None)
        if node is None:
            return None
    return node


def main():
    corpus = json.loads((BASE / "corpus.json").read_text())
    rows = []
    roots = []
    errors = []
    aggregates = {}
    root_fields = {}
    material_count = 0
    for operator, info in corpus["operators"].items():
        nodes = {node["node"]: node for node in info["nodes"]}
        documents = {}
        for name, meta in nodes.items():
            if not meta.get("snapshot"):
                continue
            path = ROOT / meta["snapshot"]
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != meta["sha256"]:
                raise ValueError(f"Input changed: {path}")
            documents[name] = yaml.safe_load(path.read_text())
            material_count += 1

        def resolve(name, area, stack=()):
            key = (name, area)
            if key in stack:
                raise ValueError(f"area reference cycle: {stack + (key,)}")
            if name not in documents:
                raise ValueError(f"missing parent material: {name}")
            value = mapping(mapping(documents[name]).get("design")).get(area)
            if isinstance(value, dict) and "from_parent" in value:
                if set(value) != {"from_parent"}:
                    raise ValueError(f"mixed reference: {name}.{area}")
                return resolve(value["from_parent"], area, stack + (key,))
            return value, name

        aggregate = Counter()
        categories = Counter()
        for name, meta in nodes.items():
            categories[meta["category"]] += 1
            if name not in documents or not meta.get("semantic_eligible"):
                continue
            try:
                values = {area: resolve(name, area) for area in AREAS}
                effective = {area: value[0] for area, value in values.items()}
                providers = {area: value[1] for area, value in values.items()}
            except ValueError as exc:
                errors.append({"operator": operator, "node": name, "error": str(exc)})
                continue
            counts = count_design(effective)
            raw_counts = count_design(mapping(documents[name]).get("design"))
            row = {"operator": operator, "node": name, "category": meta["category"],
                   "source": meta["snapshot"], "providers": providers,
                   "raw": raw_counts, "effective": counts}
            rows.append(row)
            aggregate.update(counts)
            if name == "root":
                roots.append(row)
                raw_text = (ROOT / meta["snapshot"]).read_text()
                ast = yaml.compose(raw_text)
                fields = []
                phases = mapping(mapping(effective.get("tile_pipeline")).get("phase"))
                for phase_name, phase in phases.items():
                    for field in ("resource", "domain", "milestone"):
                        if field not in phase:
                            continue
                        path_keys = ["design", "tile_pipeline", "phase", phase_name, field]
                        element = value_at(ast, path_keys)
                        fields.append({"phase": phase_name, "field": field,
                                       "path": meta["snapshot"],
                                       "start_line": element.start_mark.line + 1,
                                       "end_line": element.end_mark.line + 1,
                                       "value": phase[field]})
                root_fields[operator] = {"fields": fields,
                                         "flow": mapping(effective.get("work_schedule")).get("flow")}
        aggregates[operator] = {"categories": dict(categories),
                                "counted_designs": sum(r["operator"] == operator for r in rows),
                                "effective_field_occurrences": dict(aggregate)}
    result = {
        "method": "Field occurrences per effective design. YAML aliases and inherited areas contribute at each use; counts are not unique physical allocations, independent mechanisms, or authored prose volume.",
        "scope": "All semantic_eligible material from the prior pause-selected corpus. This includes drafts; approved and candidate status remain separate metadata.",
        "material_files": material_count, "counted_designs": len(rows),
        "errors": errors, "roots": roots, "operators": aggregates, "nodes": rows,
    }
    (BASE / "data/inventory.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (BASE / "data/root_fields.json").write_text(json.dumps(root_fields, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 六 root 的结构登记量", "", "按字段出现位置计数；不表示不同物理分配数或全树独立设计数。", "",
             "| 算子 | phase | role | resource | view | domain | milestone | handoff值 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    total = Counter()
    for row in roots:
        c = row["effective"]
        total.update(c)
        keys = ("phases", "roles", "resources", "views", "domains", "milestones", "handoff_values")
        lines.append("| " + row["operator"].replace("extract_", "").upper() + " | " + " | ".join(str(c.get(k, 0)) for k in keys) + " |")
    lines.append("| 合计 | " + " | ".join(str(total.get(k, 0)) for k in keys) + " |")
    lines += ["", f"全量语料：{material_count} 份文件；其中 {len(rows)} 份有设计材料纳入 raw/effective 统计；解析失败 {len(errors)}。", "",
              "root 是主审样本；全量字段扫描不能替代对全部分支的语义审阅。"]
    (BASE / "data/census.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
