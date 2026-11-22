import json
from copy import deepcopy
from pathlib import Path

import pytest

from kernelschema.contracts import PROTOCOL, ProtocolError, check, digest, jcs, loads, validate
from kernelschema.materials import for_role, validate_request
from kernelschema.artifacts import Bundle


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1.0}', b'{"x":1e0}', b'{"x":9007199254740992}', b'{"x":"\\ud800"}', b'\xef\xbb\xbf{}'])
def test_bad_json_rejected(raw):
    with pytest.raises(ProtocolError): loads(raw)


def test_rfc8785_shared_byte_vectors():
    fixtures = Path(__file__).parent / "fixtures/v1/row_softmax/jcs_vectors.json"
    for vector in json.loads(fixtures.read_bytes()):
        actual = jcs(loads(vector["input"]))
        assert actual == bytes.fromhex(vector["canonical_hex"])
        assert digest(loads(vector["input"])) == vector["sha256"]


@pytest.mark.parametrize("operation", ["design", "optimize", "implement"])
def test_strict_request_materials(make_bundle, operation):
    fixture = make_bundle(operation)
    request = fixture["request"]
    validate_request(Bundle(fixture["root"]), request)
    for mutate in (lambda r: r.pop("protocol"), lambda r: r.update(protocol="ks.workflow.v2"), lambda r: r.update(typo=True), lambda r: r["budget"]["limits"].update(active_ms=-1), lambda r: r["backend"].update(token="secret")):
        broken = deepcopy(request); mutate(broken)
        with pytest.raises(ProtocolError): validate(broken)
    visible = for_role(Bundle(fixture["root"]), Bundle(fixture["root"]).document(request["materials"]), "coder")
    assert visible == {}  # planner/reviewer-only objective does not leak via hash/path visibility.


def test_check_requires_facts():
    with pytest.raises(ProtocolError): check({"status": "pass", "receipt": None, "reason": "said so"})
    with pytest.raises(ProtocolError): check({"status": "unknown", "receipt": None, "reason": ""})
    check({"status": "unknown", "receipt": None, "reason": "no original terminal"})


def test_verify_requires_explicit_seed(make_bundle):
    request = make_bundle()["request"]
    request["mode"] = "verify_only"
    with pytest.raises(ProtocolError, match="explicit"): validate(request)


def test_set_arrays_are_sorted_and_unique(make_bundle):
    fixture = make_bundle()
    scope = Bundle(fixture["root"]).document(fixture["request"]["scope"])
    scope["coverage_ids"].reverse()
    with pytest.raises(ProtocolError, match="sorted"): validate(scope)
