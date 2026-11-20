"""Model-free role projection, with full immutable Rules left intact."""
from copy import deepcopy

import pytest

from kernelschema import api
from kernelschema.artifacts import Bundle, Publication, durable_write
from kernelschema.contracts import PROTOCOL, ProtocolError, jcs, sha256
from kernelschema.resources import freeze_rules, resource
from kernelschema.runtime import prepare_role_input, material_text, RoleDelivery
from kernelschema.materials import validate_request, private_evaluator_hashes


def context(make_bundle, tmp_path, *, extra_rules=(), materials=()):
    f = make_bundle()
    p = Publication(f["root"])
    rules_ref = freeze_rules(p)
    p.commit([rules_ref])
    rules = Bundle(f["root"]).document(rules_ref)
    p = Publication(f["root"])
    rules["files"] = sorted([*rules["files"], *extra_rules], key=lambda r: r["path"])
    rules_ref = p.put_json("rules.json", rules)
    mat = Bundle(f["root"]).document(f["request"]["materials"])
    mat["entries"] = sorted([*mat["entries"], *materials], key=lambda e: e["id"])
    request = {**f["request"], "rules": rules_ref, "materials": p.put_json("materials.json", mat)}
    ref = p.put_json("request.json", request); p.commit([ref])
    durable_write(f["root"] / "request.json", jcs(request), replace=True)
    return api.prepare_run("implement", request=f["root"] / "request.json", out=tmp_path / "run", backend_identity={"synthetic": True})


def test_known_host_implementations_stay_in_rules_and_all_semantics_resolve(make_bundle, tmp_path):
    c = context(make_bundle, tmp_path)
    p = Publication(c.root)
    tool = next(t for t in c.objects["rules"]["tools"] if t["name"] == "template")
    unknown = p.put_bytes("unknown/rule.py", b"# mandatory semantic rule, despite extension\n")
    rule = resource("schema_docs/0.common.md").read_bytes()
    duplicate = p.put_bytes("explicit-duplicate.md", rule)
    p.commit([unknown, duplicate])
    before_rules = Bundle(c.root).closure([c.request["rules"]])
    payload = prepare_role_input(c, "coder", source=c.objects["task"]["starter_source"], deliveries=(
        RoleDelivery("duplicate", duplicate, ("coder",)), RoleDelivery("explicit_checker_reference", tool["source"], ("coder",)),
        RoleDelivery("unknown_py", unknown, ("coder",))))
    host = [i for i in payload["materials"] if "host_implementation" in i]
    assert len(host) == 29 and sum(i["size"] for i in host) == 409961
    assert all("text" not in i and "text_alias_of" not in i and i["ref"] for i in host)
    for item in payload["materials"]:
        if "host_implementation" not in item:
            assert sha256(material_text(payload, item["id"]).encode()) == item["sha256"]
    assert material_text(payload, "delivery/explicit_checker_reference") == c.bundle.read(tool["source"]).decode()
    assert material_text(payload, "delivery/unknown_py").startswith("# mandatory semantic")
    alias = next(i for i in payload["materials"] if i["id"] == "delivery/duplicate")
    assert alias["text_alias_of"].startswith("rules/") and alias["ref"] == duplicate and alias["authority"] == "text"
    assert Bundle(c.root).closure([c.request["rules"]]) == before_rules
    broken = deepcopy(payload); alias_item = next(i for i in broken["materials"] if i["id"] == "delivery/duplicate")
    alias_item["text_alias_of"] = "missing"
    with pytest.raises(ProtocolError, match="alias"):
        material_text(broken, alias_item["id"])


def test_unknown_python_rule_not_removed_by_extension(make_bundle, tmp_path):
    f = make_bundle()
    p = Publication(f["root"])
    raw = p.put_bytes("rules/unknown.py", b"# unknown mandatory rule\n")
    rules = p.put_json("rules.json", {"protocol": PROTOCOL, "kind": "rules", "files": [raw], "tools": []})
    request = {**f["request"], "rules": rules}
    ref = p.put_json("request.json", request); p.commit([ref])
    durable_write(f["root"] / "request.json", jcs(request), replace=True)
    c = api.prepare_run("implement", request=f["root"] / "request.json", out=tmp_path / "run", backend_identity={"synthetic": True})
    item = next(i for i in prepare_role_input(c, "coder")["materials"] if i["id"] == "rules/0")
    assert item["text"] == "# unknown mandatory rule\n" and "host_implementation" not in item


def test_permission_precedes_same_hash_dedup_and_source_interface_preserved(make_bundle, tmp_path):
    c = context(make_bundle, tmp_path)
    source = c.objects["task"]["starter_source"]
    p = Publication(c.root)
    copied = p.put_bytes("duplicate-source.py", c.bundle.source(source)); p.commit([copied])
    payload = prepare_role_input(c, "coder", source=source, deliveries=(RoleDelivery("copy", copied, ("coder",)),))
    original = next(i for i in payload["materials"] if i["id"] == "current_source")
    assert original["text"] == c.bundle.source(source).decode()
    assert material_text(payload, "delivery/copy") == original["text"]
    private = c.objects["evaluator"]["files"][0]
    p = Publication(c.root); hidden = p.put_bytes("alias-private.md", c.bundle.read(private)); p.commit([hidden])
    with pytest.raises(ProtocolError, match="private evaluator"):
        prepare_role_input(c, "coder", source=source, deliveries=(RoleDelivery("bad", hidden, ("coder",)),))


def test_evaluator_inputs_direct_alias_and_pre_dedup_are_private(make_bundle, tmp_path):
    c = context(make_bundle, tmp_path)
    inputs = c.objects["evaluator"]["inputs"]
    assert inputs["sha256"] in private_evaluator_hashes(c.objects["evaluator"])
    p = Publication(c.root)
    alias = p.put_bytes("renamed-public-description.md", c.bundle.read(inputs))
    wrapped = p.put_source("wrapped-inputs", c.bundle.read(inputs))
    p.commit([alias, wrapped])
    for ref in (inputs, alias):
        p = Publication(c.root)
        materials = deepcopy(c.objects["materials"])
        materials["entries"].append({"id": "private_input", "file": ref, "roles": ["coder"], "origin": "synthetic negative", "license": "none"})
        materials["entries"].sort(key=lambda i: i["id"])
        mat = p.put_json("bad-materials.json", materials); p.commit([mat])
        with pytest.raises(ProtocolError, match="private evaluator"):
            validate_request(c.bundle, {**c.request, "materials": mat})
        with pytest.raises(ProtocolError, match="private evaluator"):
            prepare_role_input(c, "coder", deliveries=(RoleDelivery("input", ref, ("coder",)), RoleDelivery("same-input", alias, ("coder",))))
    with pytest.raises(ProtocolError, match="private evaluator"):
        prepare_role_input(c, "coder", source=wrapped)
    public = prepare_role_input(c, "coder", deliveries=tuple(RoleDelivery(name, c.objects["evaluator"][name], ("coder",)) for name in ("timing", "environment")))
    for name in ("timing", "environment"):
        assert material_text(public, "delivery/" + name) == c.bundle.read(c.objects["evaluator"][name]).decode()


@pytest.mark.parametrize("private_kind", ["inputs", "entrypoint", "implementation"])
def test_caller_baseline_label_cannot_grant_private_payload(make_bundle, tmp_path, private_kind):
    c = context(make_bundle, tmp_path)
    ref = c.objects["evaluator"]["inputs" if private_kind == "inputs" else "entrypoint"]
    if private_kind == "implementation":
        ref = next(r for r in c.objects["evaluator"]["files"] if r != c.objects["evaluator"]["entrypoint"])
    p = Publication(c.root)
    disguised = p.put_source("fake-public-baseline", c.bundle.read(ref))
    task = {**c.objects["task"], "baseline_source": disguised}
    task_ref = p.put_json("fake-task.json", task)
    schema = {**c.objects["schema"], "task_sha256": task_ref["sha256"]}
    req = {**c.request, "task": task_ref, "schema": p.put_json("fake-schema.json", schema), "seed": disguised}
    request = p.put_json("fake-request.json", req); p.commit([request])
    path = c.root / "fake-request.json"; durable_write(path, c.bundle.read(request))
    fake = api.prepare_run("implement", request=path, out=tmp_path / "fake-run", backend_identity={"synthetic": True})
    with pytest.raises(ProtocolError, match="private evaluator"):
        prepare_role_input(fake, "coder", source=disguised)
