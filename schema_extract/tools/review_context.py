"""Mechanical, checkpoint-bound reading context for index work and review.

The packet is a projection of task.json and its immutable checkpoint. It is
never loaded to decide progress, and carries no substitute semantic verdict.
"""

from copy import deepcopy
from decimal import Decimal
from pathlib import Path

from .records import StateError, artifact_manifest, digest, parser_for, read_json, review_current


def field_excerpt(path: Path, fields: tuple[str, ...], *, include_text=False, parser_root=None) -> dict:
    """Locate an exact source field; optionally project its YAML value."""
    import yaml

    text = path.read_text()
    node = yaml.compose(text, Loader=yaml.SafeLoader)

    def entry_for(mapping, field, active):
        if not isinstance(mapping, yaml.MappingNode) or id(mapping) in active:
            raise StateError(f"cannot locate {path.name}:{'.'.join(fields)}")
        explicit = next(((k, v) for k, v in mapping.value
                         if k.tag != "tag:yaml.org,2002:merge" and k.value == field), None)
        if explicit is not None:
            return explicit
        # Match the fixed parser: explicit keys override merges; earlier
        # mappings in a merge sequence take precedence over later mappings.
        active = active | {id(mapping)}
        for key, value in mapping.value:
            if key.tag == "tag:yaml.org,2002:merge":
                merged = value.value if isinstance(value, yaml.SequenceNode) else [value]
                for candidate in merged:
                    found = entry_for(candidate, field, active)
                    if found is not None:
                        return found
        return None

    key = None
    for field in fields:
        entry = entry_for(node, field, set())
        if entry is None:
            raise StateError(f"cannot locate {path.name}:{'.'.join(fields)}")
        key, node = entry
    # Aliases may refer backwards; locate their actual value rather than
    # accidentally including the intervening Contract or Design text.
    start = min(key.start_mark.index, node.start_mark.index)
    fragment = text[start:node.end_mark.index].rstrip()
    first = text.count("\n", 0, start) + 1
    last = first + fragment.count("\n")
    result = {"path": path.name, "field": ".".join(fields),
              "line_start": first, "line_end": last,
              "quote": (text[key.start_mark.index:key.end_mark.index]
                        if key.start_mark.index <= node.start_mark.index else fragment.split()[0])}
    if include_text:
        # AST tags above are used only to locate source spans. The fixed
        # checker's parser is the sole authority for scalar and merge values.
        parser = parser_for(parser_root)
        value = parser.load(path, path.stem).value
        for field in fields:
            value = value[field]

        class ProjectionDumper(yaml.SafeDumper):
            pass

        # Quote every string: the dumper's YAML 1.1 implicit resolver must not
        # turn core strings such as "0o12" or "1e3" into numbers on read-back.
        ProjectionDumper.add_representer(str, lambda dumper, item: dumper.represent_scalar(
            "tag:yaml.org,2002:str", item, style='"'))
        ProjectionDumper.add_representer(Decimal, lambda dumper, item: dumper.represent_scalar(
            "tag:yaml.org,2002:float", str(item)))
        projected = {fields[-1]: value}
        rendered = yaml.dump(projected, Dumper=ProjectionDumper, sort_keys=False, allow_unicode=True)
        try:
            restored = parser.loads(rendered, "review_context", path=f"{path}:{'.'.join(fields)} projection").value
        except parser.ReadError as error:
            raise StateError(f"review context projection is not readable by the fixed parser: {error}") from error
        if digest(restored) != digest(projected):
            raise StateError("review context projection changed the fixed parser's field value")
        result["yaml"] = rendered.rstrip()
    return result


def build_review_context(store, state, baseline, work: Path, role: str) -> dict | None:
    """Return focused context only when every independent node review applies."""
    if (state.get("receipt_correction") or not state.get("dispatches")
            or not ((role == "worker" and state["stage"] == "index")
                    or (role == "reviewer" and state["review_scope"] == "final"))):
        return None
    checkpoint = store.verify_checkpoint(state)
    if checkpoint is None:
        return None
    record = read_json(checkpoint / "checkpoint.json")
    semantics = record["semantics"]
    order = ["root", *(item["id"] for item in state["dispatches"])]
    if (set(order) != set(semantics["nodes"])
            or not all(review_current(state, scope, semantics) for scope in ["contract", *order])):
        return None
    if artifact_manifest(work) != record["files"]:
        raise StateError("index context requires the current accepted checkpoint artifacts")
    artifacts = checkpoint / "artifacts"
    nodes, source_locations = [], {}
    for node in order:
        review = state["reviews"][node]
        nodes.append({"node": node, "metadata": field_excerpt(artifacts / (node + ".yaml"),
                                                                ("meta_data",), include_text=True,
                                                                parser_root=baseline["snapshot"]),
                      "independent_review": {"checkpoint": review["checkpoint"],
                                             "summary": review["receipt"]["summary"]}})
        for anchor in review["receipt"].get("source_evidence", []):
            location = tuple(anchor[key] for key in ("path", "line_start", "line_end"))
            source_locations.setdefault(location, []).append(node)
    # Source locators are navigation, not a claim that every anchor proves a
    # host selector. The reviewer still reads the relevant actual host source.
    sources = [{"path": path, "line_start": first, "line_end": last,
                "reviewed_nodes": sorted(set(nodes))}
               for (path, first, last), nodes in source_locations.items()]
    return {"kind": "index", "fingerprint": state["fingerprint"],
            "checkpoint": state["checkpoint"], "contract_digest": semantics["contract"],
            "contract_location": field_excerpt(artifacts / "root.yaml", ("contract", "algorithm_requirement")),
            "nodes": nodes, "dispatches": deepcopy(state["dispatches"]),
            "source_locations": sources,
            "host_source_entry": "kernel_wiki/" + baseline["identity"]["entry"]["paths"]["entrypoint"],
            "index": {"path": "dispatch.md", "sha256": record["files"].get("dispatch.md")}}
