import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from kernelschema.artifacts import Bundle, Publication
from kernelschema.contracts import ProtocolError, UnsupportedError, jcs, sha256
from kernelschema.evaluation import LocalExecutor
from kernelschema.evaluation.agate import AgateExecutor
from kernelschema.evaluation.task import RowSoftmaxTask, _when, stage, verify_stage
from kernelschema.resources import resource

fixture_path = Path(__file__).parent / "fixtures/evaluation/build.py"
spec = importlib.util.spec_from_file_location("evaluation_fixture", fixture_path)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def wait(executor, operation):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        result = executor.query(operation)
        if result["state"] == "terminal":
            return result
        time.sleep(.1)
    pytest.fail("original local job did not finish: " + str(executor.root))


def test_staging_exact_bytes_and_no_mutable_import_cache(tmp_path):
    f, s = fixture.staged(tmp_path)
    assert (Path(s["root"]) / "kernel.py").read_bytes() == f["bundle"].source(f["candidate"])
    verify_stage(s)
    (Path(s["root"]) / "kernel.py").write_bytes(b"# different source\n")
    with pytest.raises(ProtocolError, match="staged bytes changed"):
        verify_stage(s)


def test_scope_when_is_finite_and_never_guessed():
    assert _when(None,[1,128],root=True) is True
    assert _when("m in {1,32}; n in {128,1024}",[32,1024],root=True) is True
    assert _when("m=1; n=128",[32,128],root=True) is False
    assert _when("all shapes probably fast",[1,128],root=True) is None
    assert _when(None,[1,128],root=False) is None


def test_scope_uses_actual_template_and_rejects_wrong_partition(tmp_path):
    f = fixture.build(tmp_path / "bundle")
    adapter = RowSoftmaxTask(f["bundle"], f["task"])
    def freeze(when=None, variant=None, ids=None, hardware=None):
        p = Publication(f["bundle"].root)
        data = resource("examples/workflows/row_softmax/schema/root.yaml").read_bytes()
        if when:
            data = data.replace(b"  Version: 1\n", ("  Version: 1\n  Dispatch: {when: '" + when + "'}\n").encode())
        if hardware is not None:
            import yaml
            document = yaml.safe_load(data)
            document["CONTRACT"]["Hardware_specification"] = hardware
            data = yaml.safe_dump(document, sort_keys=False).encode()
        raw = p.put_bytes("schema/root.yaml",data)
        schema = p.put_json("schema.json",{"protocol":"ks.workflow.v1","kind":"schema","task_sha256":f["task"]["sha256"],
            "raw":[{"node":"root","file":raw}],"dispatch":None,"parent":None,"change_report":None,"design_revision":0,"change":"new_task"})
        ids = ids or sorted(adapter.partitions)
        scope=p.put_json("scope.json",{"protocol":"ks.workflow.v1","kind":"scope","coverage_ids":ids,
            "bindings":[{"coverage_ids":ids,"node":"root","variant":variant}]})
        p.commit([schema,scope],subjects=[f["task"]])
        return schema,scope
    schema,scope=freeze()
    assert adapter.validate_scope(schema,scope)["status"] == "pass"
    assert adapter.validate_scope(schema,scope,require_hardware_binding=True)["status"] == "unknown"
    # Concrete-looking documentation bindings test only the static check, not a
    # real GPU/profile. The fixture declares its synthetic provenance explicitly.
    hardware = {"Device": "NVIDIA H20 (synthetic documentation fixture)", "Architecture": "SM90",
                "Stack": {"Language": "Triton 3.5.1", "Toolchain": "CUDA 12.8"},
                "Wiki": {"Hardware": "<synthetic-docs@fixture-v1:H20.md>"}}
    bound_schema,bound_scope = freeze(hardware=hardware)
    assert adapter.validate_scope(bound_schema,bound_scope,require_hardware_binding=True)["status"] == "pass"
    assert adapter.validate_scope(*freeze(hardware={**hardware,"Device":"unknown"}),require_hardware_binding=True)["status"] == "unknown"
    assert adapter.validate_scope(*freeze(hardware={}),require_hardware_binding=True)["status"] == "unknown"
    assert adapter.validate_scope(*freeze("m=1"))["status"] == "fail"
    assert adapter.validate_scope(*freeze("some other dispatch"))["status"] == "unknown"
    assert adapter.validate_scope(*freeze(variant="uninterpreted"))["status"] == "unknown"
    with pytest.raises(ProtocolError,match="partitions"):
        adapter.validate_scope(*freeze(ids=["m01_n0128_random_s17"]))
    with pytest.raises(UnsupportedError,match="full Task scope only"):
        adapter.validate_scope(*freeze(ids=["shape_m1_n128"]))


def test_subset_evaluation_is_rejected_before_staging(tmp_path):
    f = fixture.build(tmp_path / "bundle")
    adapter = RowSoftmaxTask(f["bundle"], f["task"])
    p = Publication(f["bundle"].root)
    def scope(ids, name):
        return p.put_json(name,{"protocol":"ks.workflow.v1","kind":"scope","coverage_ids":ids,
            "bindings":[{"coverage_ids":ids,"node":"root","variant":None}]})
    full = scope(sorted(adapter.partitions),"full.json")
    partial = scope(["shape_m1_n128"],"partial.json")
    p.commit([full,partial])
    capabilities = adapter.require_evaluation_scope(full)
    assert capabilities == adapter.evaluation_capabilities()
    assert capabilities["scope_mode"] == "full_task_only" and capabilities["subset_only"] is False
    assert len(capabilities["coverage_ids"]) == 4
    with pytest.raises(UnsupportedError,match="subset-only"):
        adapter.require_evaluation_scope(partial)
    with pytest.raises(UnsupportedError,match="subset-only"):
        stage(f["bundle"],f["task"],f["candidate"],tmp_path/"rejected-stage",scope=partial)
    assert not (tmp_path/"rejected-stage").exists()
    full_stage=stage(f["bundle"],f["task"],f["candidate"],tmp_path/"full-stage",scope=full)
    verify_stage(full_stage)


def test_capabilities_do_not_admit_cuda(tmp_path):
    assert LocalExecutor(tmp_path / "jobs").capabilities()["CUDA"] is False
    assert not AgateExecutor(tmp_path / "remote", None).capabilities()["service_qualified"]


def test_core_import_does_not_load_torch():
    r = subprocess.run([sys.executable, "-c", "import sys; import kernelschema.evaluation; assert 'torch' not in sys.modules"], capture_output=True)
    assert r.returncode == 0, r.stderr


def test_real_child_readonly_full_suite_receipts_and_replay(tmp_path):
    pytest.importorskip("torch")
    f, s = fixture.staged(tmp_path)
    executor = LocalExecutor(tmp_path / "jobs")
    ready = executor.prepare(s)
    assert len(json.loads((Path(s["root"]) / "frozen_inputs.json").read_bytes())["cases"]) == 36
    reservation = {"owner": "parent", "evaluator_jobs": 1, "active_ms": 60000}
    args = dict(operation_id="one", submission_key="stable-one", owner="parent", reservation=reservation)
    executor.submit(s, **args)
    facts = wait(executor, "one")
    assert facts["terminal"]["exit_code"] == 0, (Path(facts["native_locator"]["root"]) / "stderr.raw").read_text()
    worker = facts["worker"]
    assert worker["result"]["correctness"]["passed"]
    assert len(worker["result"]["timing"]["batches"]) == 80
    assert worker["loaded"]["kernel.py"] == sha256(f["bundle"].source(f["candidate"]))
    assert worker["permissions"]["seccomp_tsync"]
    assert all(p["write_open_denied"] for p in worker["permissions"]["probes"])
    original_controller = (executor.root / "one/controller.json").read_bytes()
    assert executor.submit(s, **args)["receipt_id"] == facts["receipt_id"]
    assert (executor.root / "one/controller.json").read_bytes() == original_controller
    publication = Publication(f["bundle"].root)
    settled = executor.settle("one", publication)
    assert settled["new_receipt"]
    report = Bundle(f["bundle"].root).document(settled["report"], "evaluation")
    assert report["correctness"]["status"] == "unknown"
    assert report["performance"]["status"] == "unknown"
    assert report["extensions"]["information"]["synthetic"]
    assert executor.settle("one", Publication(f["bundle"].root))["new_receipt"] is False
    assert settled["usage_row"]["evaluator_jobs"] == 1


def test_submit_ack_loss_unknown_fence_then_original_late_terminal(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    f, s = fixture.staged(tmp_path)
    executor = LocalExecutor(tmp_path / "jobs")
    executor.prepare(s, include_timing=False)
    original = subprocess.Popen
    calls = []
    def disconnected(*args, **kwargs):
        child = original(*args, **kwargs)
        calls.append(child.pid)
        raise ConnectionError("ack lost after actual controller spawn")
    monkeypatch.setattr(subprocess, "Popen", disconnected)
    args = dict(operation_id="lost", submission_key="stable-lost", owner="same-parent",
                reservation={"owner": "same-parent", "evaluator_jobs": 1, "active_ms": 60000})
    assert executor.submit(s, **args)["state"] == "unknown"
    assert executor.submit(s, **args)["state"] == "unknown"
    with pytest.raises(ProtocolError, match="fences"):
        executor.submit(s, **{**args,"operation_id":"replacement","submission_key":"new"})
    assert len(calls) == 1
    facts = wait(executor,"lost")
    assert facts["worker"]["result"]["correctness"]["passed"]
    assert len(calls) == 1
    assert executor.settle("lost", Publication(f["bundle"].root))["new_receipt"]
    assert not executor.settle("lost", Publication(f["bundle"].root))["new_receipt"]


def test_pre_spawn_loss_never_retries_even_if_no_pid(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    _, s = fixture.staged(tmp_path)
    executor = LocalExecutor(tmp_path / "jobs")
    executor.prepare(s, include_timing=False)
    calls = []
    def crash(*a, **k):
        calls.append(1)
        raise OSError("submission boundary interrupted")
    monkeypatch.setattr(subprocess,"Popen",crash)
    args = dict(operation_id="unknown", submission_key="fixed", owner="owner",reservation={"owner":"owner","evaluator_jobs":1,"active_ms":60000})
    assert executor.submit(s,**args)["state"] == "unknown"
    assert executor.submit(s,**args)["recovery"] == "settle_only"
    assert len(calls) == 1


def test_candidate_cannot_write_evaluator(tmp_path):
    pytest.importorskip("torch")
    code = b'import torch\nfrom pathlib import Path\nclass Model(torch.nn.Module):\n def forward(self,x,out):\n  Path(__file__).with_name("evaluator").joinpath("golden.py").write_text("tampered")\n  return out\n'
    f,s = fixture.staged(tmp_path,code)
    executor = LocalExecutor(tmp_path / "jobs")
    executor.prepare(s,include_timing=False)
    executor.submit(s,operation_id="write",submission_key="write-key",owner="owner",reservation={"owner":"owner","evaluator_jobs":1,"active_ms":60000})
    facts=wait(executor,"write")
    assert facts["worker"]["result"]["build"] == "fail"
    assert all(r["exception"]["type"] == "PermissionError" for r in facts["worker"]["result"]["correctness"]["basic_calls"])
    verify_stage(s)
    # A forged old-source receipt remains wrong even with internally matching raw hashes.
    job = executor.root / "write"
    (job / "stdout.before-forgery.raw").write_bytes((job / "stdout.raw").read_bytes())
    (job / "terminal.before-forgery.raw").write_bytes((job / "native-terminal.json").read_bytes())
    worker = json.loads((job / "stdout.raw").read_bytes())
    worker["loaded"]["kernel.py"] = sha256(b"old cached candidate")
    rewritten = jcs(worker)
    (job / "stdout.raw").write_bytes(rewritten)
    terminal = json.loads((job / "native-terminal.json").read_bytes())
    terminal["outputs"]["stdout.raw"] = sha256(rewritten)
    (job / "native-terminal.json").write_bytes(jcs(terminal))
    with pytest.raises(ProtocolError,match="actual loaded candidate"):
        executor.query("write")
