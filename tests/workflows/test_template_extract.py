from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys

import pytest

from kernelschema.artifacts import Bundle, Publication, import_bundle
from kernelschema.contracts import PROTOCOL, ProtocolError, sha256
from kernelschema.extract_bridge import consume_raw, implement_request
from kernelschema.resources import resource
from kernelschema.template_api import check_directory, effective_document, normalize_directory, resolve_bundle, tools


def test_original_template_parser_normalizer_and_private_namespace(make_bundle):
    f = make_bundle(); bundle = Bundle(f["root"])
    manifest = bundle.document(f["schema"])
    resolved = resolve_bundle(bundle, manifest, validate_structure=True)
    assert "CONTRACT" in resolved.effective_document("root")
    assert resolved.source_hashes["root"] == manifest["raw"][0]["file"]["sha256"].split(":", 1)[1]
    directory = resource("examples/workflows/row_softmax/schema/root.yaml").parent
    before = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    check_directory(directory)
    plan = normalize_directory(directory)
    assert plan.files["root"].parent == directory
    assert {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()} == before
    assert tools().__name__ == "kernelschema._template_tools.schema"
    assert "tools" not in sys.modules


def test_nonroot_effective_contains_root_and_missing_from_source_rejected(make_bundle):
    import yaml
    f = make_bundle(); p = Publication(f["root"])
    base = resource("schema_template/tools/tests/fixtures/minimal/root.yaml").read_bytes()
    raw = yaml.safe_load(base)
    child = deepcopy(raw); child.pop("CONTRACT")
    child["META_DATA"]["Dispatch"] = {"id": "branch", "when": "All scalar inputs"}
    child["DESIGN"] = {region: {"from": "root"} for region in ("Math_reshape", "Work_schedule", "Tile_pipeline")}
    root_ref = p.put_bytes("raw/root.yaml", base)
    child_ref = p.put_bytes("raw/branch.yaml", yaml.safe_dump(child, sort_keys=False).encode())
    index = resource("schema_template/tools/tests/fixtures/minimal/dispatch.md").read_bytes()
    dispatch = p.put_bytes("raw/dispatch.md", index + b"| [branch](branch.yaml) | scalar | root |\n")
    manifest = {"protocol": PROTOCOL, "kind": "schema", "task_sha256": f["task"]["sha256"], "raw": [{"node": "branch", "file": child_ref}, {"node": "root", "file": root_ref}], "dispatch": dispatch, "design_revision": 0, "parent": None, "change": "new_task", "change_report": None}
    r = p.put_json("schema.json", manifest)
    p.commit([r], subjects=[f["task"]])
    view = effective_document(Bundle(f["root"]), manifest, "branch")
    assert view["CONTRACT"] == raw["CONTRACT"]
    assert view["DESIGN"] == raw["DESIGN"]
    (f["root"] / root_ref["path"]).unlink()
    with pytest.raises(ProtocolError): Bundle(f["root"]).closure([r], subjects=[f["task"]])


def test_original_extract_passthrough_and_exit_codes(tmp_path):
    script = str(resource("schema_extract/extract_yaml.py"))
    cli = str(Path(sys.executable).parent / "kernelschema")
    for arguments in (["--help"], ["--not-an-extract-option"]):
        original = subprocess.run([sys.executable, "-B", script, *arguments], cwd=tmp_path, capture_output=True)
        wrapped = subprocess.run([cli, "extract", "--", *arguments], cwd=tmp_path, capture_output=True)
        assert original.returncode == (0 if arguments == ["--help"] else 2), original.stderr.decode()
        assert (wrapped.returncode, wrapped.stdout, wrapped.stderr) == (original.returncode, original.stdout, original.stderr)


@pytest.mark.parametrize("origin", ["handwritten", "curated", "design", "extract"])
def test_consume_raw_without_universal_p6_gate(make_bundle, tmp_path, origin):
    f = make_bundle(); p = Publication(f["root"])
    directory = resource("examples/workflows/row_softmax/schema/root.yaml").parent
    before = {x.name: x.read_bytes() for x in directory.iterdir() if x.is_file()}
    consumed = consume_raw(p, directory, task=f["task"], origin=origin)
    p.commit([consumed["schema"]], subjects=[f["task"]])
    assert consumed["receipt"] is None and consumed["historical_usage"] is None
    assert {x.name: x.read_bytes() for x in directory.iterdir() if x.is_file()} == before
    request = implement_request(request_id="consume", **{k: f["request"][k] for k in ("task", "rules", "materials", "scope", "budget", "backend")}, schema=consumed["schema"], seed=f["starter"], mode="verify_only")
    assert request["seed"] == f["starter"] and request["mode"] == "verify_only"


def test_native_receipt_dependency_closure(make_bundle, tmp_path):
    f = make_bundle(); native = tmp_path / "native"; native.mkdir()
    for name in ("root.yaml", "dispatch.md"):
        (native / name).write_bytes(resource("examples/workflows/row_softmax/schema/" + name).read_bytes())
    (native / "notes.md").write_bytes(b"Opaque prose missing/path is not a Ref.\r\n")
    receipt = {"format": "schema_extract.delivery.v2", "files": {p.name: sha256(p.read_bytes()).split(":", 1)[1] for p in native.iterdir()}, "run_id": "synthetic-not-real-extract", "native_statistic": 0.125}
    (native / ".extraction").mkdir()
    receipt_path = native / ".extraction/receipt.json"
    receipt_path.write_bytes(json.dumps(receipt, indent=2).encode() + b"\n")
    before = receipt_path.read_bytes()
    p = Publication(f["root"])
    c = consume_raw(p, native, task=f["task"], origin="extract", receipt=receipt_path)
    p.commit([c["schema"], c["receipt"]], subjects=[f["task"]])
    target = tmp_path / "imported"
    import_bundle(f["root"], target, [c["receipt"]])
    assert Bundle(target).read(c["receipt"]) == before
    assert receipt_path.read_bytes() == before
    notes = target / Path(c["receipt"]["path"]).parent.parent / "notes.md"
    assert notes.read_bytes().endswith(b"\r\n")


def test_rules_freeze_actual_implementation_bytes(tmp_path):
    from kernelschema.resources import freeze_rules
    p = Publication(tmp_path / "rules")
    reference = freeze_rules(p)
    p.commit([reference])
    bundle = Bundle(p.root)
    rules = bundle.document(reference, "rules")
    tool = rules["tools"][0]
    assert tool["version"] == sha256(resource("schema_template/tools/schema.py").read_bytes())
    assert bundle.read(tool["source"]) == resource("schema_template/tools/schema.py").read_bytes()
    assert any(r["path"].endswith("yaml_source.py") for r in rules["files"])


def test_amended_raw_rejects_old_receipt_and_accepts_curated_without_receipt(make_bundle, tmp_path):
    """A current manifest is provenance; it cannot reapprove changed P6 bytes."""
    f = make_bundle()
    native = tmp_path / "amended"
    native.mkdir()
    for name in ("root.yaml", "dispatch.md"):
        (native / name).write_bytes(resource("examples/workflows/row_softmax/schema/" + name).read_bytes())
    proof = {"format": "schema_extract.delivery.v2", "files": {
        name: sha256((native / name).read_bytes()).split(":", 1)[1]
        for name in ("root.yaml", "dispatch.md")}}
    receipt = native / "receipt.json"
    receipt.write_text(json.dumps(proof))
    original_receipt = receipt.read_bytes()
    with (native / "root.yaml").open("a") as stream:
        stream.write("\n# Synthetic byte revision after the historical receipt.\n")
    with pytest.raises(ProtocolError, match="native receipt dependency mismatch: root.yaml"):
        consume_raw(Publication(tmp_path / "rejected"), native, task=f["task"], origin="extract", receipt=receipt)
    publication = Publication(f["root"])
    curated = consume_raw(publication, native, task=f["task"], origin="curated", receipt=None)
    publication.commit([curated["schema"]], subjects=[f["task"]])
    manifest = Bundle(f["root"]).document(curated["schema"])
    assert curated["receipt"] is None
    assert manifest["extensions"]["information"]["origin"] == "curated"
    assert Bundle(f["root"]).read(manifest["raw"][0]["file"]) == (native / "root.yaml").read_bytes()
    current = native / "current.json"
    current.write_text(json.dumps({"format": "schema_wiki.current_schema_manifest.v1",
                                   "acceptance": {"runner_p6_receipt": False}}))
    with pytest.raises(ProtocolError, match="expected original schema_extract.delivery.v2 receipt"):
        consume_raw(Publication(tmp_path / "not-a-receipt"), native, task=f["task"], origin="curated", receipt=current)
    assert receipt.read_bytes() == original_receipt
