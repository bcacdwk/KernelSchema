import importlib.util
import json
from pathlib import Path
import sys

import pytest

from kernelschema.resources import resource


@pytest.fixture(scope="module")
def modules():
    pytest.importorskip("torch")
    result = {}
    for name in ("input", "golden", "check", "timing", "entry", "baseline"):
        path = resource("examples/workflows/row_softmax/evaluator/" + name + ".py")
        spec = importlib.util.spec_from_file_location("test_evaluation_" + name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result[name] = module
    result["spec"] = json.loads(resource("examples/workflows/row_softmax/task/inputs.json").read_bytes())
    result["input"].torch.set_num_threads(1)
    return result


def test_correct_full_suite_and_reuse(modules):
    m = modules
    frozen = m["input"].freeze_inputs(m["spec"])
    result = m["check"].evaluate(m["baseline"].Model, m["spec"], frozen, inputs=m["input"], golden=m["golden"],
                                device=m["input"].torch.device("cpu"), synchronize=lambda: None)
    assert result["passed"]
    assert len(result["basic_calls"]) == 72
    offset_calls = [c for c in result["basic_calls"] if "s23" in c["case_id"]]
    assert len(offset_calls) == 8
    assert all((c["x_before"]["storage_offset"], c["out_before"]["storage_offset"]) == (3, 5) for c in offset_calls)
    trace = result["state_reuse"][0]["calls"]
    assert trace[0]["x_before"] == trace[1]["x_before"]
    assert trace[0]["out_before"] == trace[1]["out_before"]
    assert trace[0]["input_sha256"] != trace[1]["input_sha256"]


@pytest.mark.parametrize("body,failed", [
    ("return torch.softmax(x, -1)", "return_identity"),
    ("out.copy_(torch.softmax(x,-1)); return out.clone()", "return_identity"),
    ("out.set_(torch.softmax(x,-1)); return out", "out_storage"),
    ("out[0].copy_(torch.softmax(x[0],-1)); return out", "out_fully_written"),
    ("out[:,:-1].copy_(torch.softmax(x,-1)[:,:-1]); return out", "out_fully_written"),
    ("x.zero_(); out.copy_(torch.softmax(x,-1)); return out", "x_bytes"),
    ("out.fill_(0.5); return out", "elementwise_tolerance"),
    ("out.copy_(torch.softmax(x,-1)); out.as_strided((out.storage_offset(),),(1,),0).zero_(); return out", "out_prefix_guard"),
    ("out.copy_(torch.softmax(x,-1)); out.as_strided((1,),(1,),out.storage_offset()+out.numel()).zero_(); return out", "out_suffix_guard"),
])
def test_independent_bad_candidate_checks(modules, body, failed):
    m = modules
    namespace = {"torch": m["input"].torch}
    exec("class Model(torch.nn.Module):\n def forward(self,x,out):\n  " + body, namespace)
    case = next(c for c in m["spec"]["cases"] if c["case_id"] == "m32_n1024_random_s23")
    buffers = m["input"].allocate(case, m["input"].generate(case), "cpu")
    record = m["check"].call(namespace["Model"](), buffers, case, "nan", inputs=m["input"], golden=m["golden"],
                              synchronize=lambda: None, device_type="cpu")
    assert record["checks"][failed] is False
    assert record["passed"] is False
    if failed == "out_storage":
        assert record["checks"]["return_identity"] is True
        assert record["checks"]["out_fully_written"] is False
    if failed == "x_bytes":
        assert record["checks"]["elementwise_tolerance"] is False


def test_changed_input_reuse_detects_cache(modules):
    m = modules
    class Cached(m["input"].torch.nn.Module):
        def forward(self, x, out):
            if not hasattr(self, "cached"):
                self.cached = m["input"].torch.softmax(x, -1)
            out.copy_(self.cached)
            return out
    result = m["check"].evaluate(Cached, m["spec"], m["input"].freeze_inputs(m["spec"]), inputs=m["input"], golden=m["golden"],
                                device=m["input"].torch.device("cpu"), synchronize=lambda: None)
    assert all(c["passed"] for c in result["basic_calls"])
    assert result["state_reuse"][0]["calls"][1]["checks"]["elementwise_tolerance"] is False


def test_domain_rejected_before_candidate(modules):
    torch = modules["input"].torch
    x = torch.zeros((1,128))
    with pytest.raises(ValueError, match="shared"):
        modules["input"].admit(x,x,device_type="cpu")
    with pytest.raises(ValueError, match="selected device"):
        modules["input"].admit(x,x.clone())
    with pytest.raises(ValueError, match="shape"):
        modules["input"].admit(torch.zeros((1,129)),torch.zeros((1,129)),device_type="cpu")


def test_timing_sequence_and_soft_target(modules):
    m = modules
    calls, ticks = [], [0]
    class A:
        def forward(self, x, out): calls.append("A"); ticks[0] += 2; return out
    class B:
        def forward(self, x, out): calls.append("B"); ticks[0] += 1; return out
    r = m["timing"].measure(A,B,m["spec"],m["input"].freeze_inputs(m["spec"]), inputs=m["input"],
                              device=m["input"].torch.device("cpu"), synchronize=lambda: None, clock=lambda: ticks[0])
    assert r["complete"] and len(r["batches"]) == 80
    assert len(calls) == 4 * (2 * 21 + 20 * 100)
    for i in range(0,80,20):
        assert "".join(r["batches"][j]["model"] for j in range(i,i+20)) == "ABBAABBAABBAABBAABBA"
    assert all(s["calls"] == 100 and s["elapsed_ns"] == int(s["end_ns"]) - int(s["start_ns"]) for s in r["batches"])
    assert r["soft_goal"] == "unmet" and r["complete"]
    assert float(r["geomean_baseline_over_candidate"]) == 0.5
    assert m["timing"].summarize([1,2,3,4])["iqr_ns"] == "1.5"
