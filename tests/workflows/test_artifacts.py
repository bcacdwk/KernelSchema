from copy import deepcopy
from pathlib import Path
import os
import stat

import pytest

from kernelschema import artifacts
from kernelschema.artifacts import Bundle, Publication, durable_write, import_bundle, install_source, recover_import
from kernelschema.contracts import PROTOCOL, ProtocolError, digest, jcs, sha256


def snapshot(root):
    return {str(p.relative_to(root)): ("dir", stat.S_IMODE(p.stat().st_mode)) if p.is_dir() else (p.read_bytes(), stat.S_IMODE(p.stat().st_mode)) for p in root.rglob("*")}


@pytest.mark.parametrize("data", [b"pass\n", b"# CRLF\r\npass\r\n", b"pass", "# 非 ASCII\npass".encode()])
def test_exact_source_round_trip(tmp_path, data):
    p = Publication(tmp_path / "bundle")
    source = p.put_source("source", data)
    p.commit([source])
    install_source(Bundle(p.root), source, tmp_path / "work")
    assert (tmp_path / "work/kernel.py").read_bytes() == data
    assert stat.S_IMODE((tmp_path / "work/kernel.py").stat().st_mode) == 0o644
    with pytest.raises(ProtocolError): install_source(Bundle(p.root), source, tmp_path / "work")
    clone = tmp_path / "clone"
    import_bundle(p.root, clone, [source])
    assert Bundle(clone).source(source) == data
    assert not os.path.samefile(clone / source["path"], p.root / source["path"])


@pytest.mark.parametrize("data", [b"\xef\xbb\xbfpass", b"\xff"])
def test_non_utf8_source_rejected_without_publication(tmp_path, data):
    p = Publication(tmp_path / "bundle")
    with pytest.raises(ProtocolError): p.put_source("source", data)
    assert not (p.root / "artifacts").exists()


@pytest.mark.parametrize("path", ["/etc/passwd", "../escape", "a/../b", "a//b", "a/./b", "a\\b", "a\nb", "request.json", ".runtime/run.json"])
def test_unsafe_ref_rejected(tmp_path, path):
    with pytest.raises(ProtocolError): Bundle(tmp_path).read({"path": path, "sha256": sha256(b"")})


def test_symlinks_and_nonregular_files_rejected(tmp_path):
    root = tmp_path / "bundle"; root.mkdir()
    prefix = root / "artifacts" / ("1" * 32); prefix.mkdir(parents=True)
    outside = tmp_path / "outside"; outside.write_bytes(b"secret")
    (prefix / "link").symlink_to(outside)
    (prefix / "dir").symlink_to(tmp_path, target_is_directory=True)
    os.mkfifo(prefix / "pipe")
    for name in ("link", "dir/outside", "pipe"):
        with pytest.raises(ProtocolError): Bundle(root).read({"path": str((prefix / name).relative_to(root)), "sha256": sha256(b"secret")})


def test_same_path_wrong_mode_and_missing_closure_are_transactional(make_bundle, tmp_path):
    f = make_bundle()
    target = tmp_path / "target"
    import_bundle(f["root"], target, [f["starter"]])
    before = snapshot(target)
    source_path = f["root"] / Path(f["starter"]["path"]).parent / "files/kernel.py"
    old = source_path.read_bytes()
    source_path.write_bytes(old + b"# changed\n")
    with pytest.raises(ProtocolError, match="mismatch"): import_bundle(f["root"], target, [f["request_ref"]])
    assert snapshot(target) == before
    source_path.write_bytes(old)
    installed = target / f["starter"]["path"]
    installed.chmod(0o600)
    before = snapshot(target)
    with pytest.raises(ProtocolError, match="mode"): import_bundle(f["root"], target, [f["request_ref"]])
    assert snapshot(target) == before
    # Missing unrelated history is allowed when only C is selected.
    (f["root"] / f["request"]["rules"]["path"]).unlink()
    independent = tmp_path / "independent"
    import_bundle(f["root"], independent, [f["baseline"]])
    assert Bundle(independent).source(f["baseline"]).startswith(b'"""')


def test_import_preflights_conflicts_before_adding_any_new_file(make_bundle, tmp_path):
    f = make_bundle(); target = tmp_path / "target"
    import_bundle(f["root"], target, [f["starter"]])
    (target / f["starter"]["path"]).write_bytes(b"conflicting bytes")
    before = snapshot(target)
    with pytest.raises(ProtocolError): import_bundle(f["root"], target, [f["request_ref"]])
    assert snapshot(target) == before


def test_interrupted_merge_rolls_back_and_replay_is_idempotent(make_bundle, tmp_path, monkeypatch):
    f = make_bundle(); target = tmp_path / "target"; target.mkdir()
    (target / "unrelated.txt").write_bytes(b"do not scan or change")
    before = snapshot(target)
    original = artifacts._rename_new
    count = 0
    def fail_after_first(source, destination):
        nonlocal count
        if destination.is_relative_to(target / "artifacts"):
            count += 1
            if count == 2: raise OSError("injected interrupted merge")
        return original(source, destination)
    monkeypatch.setattr(artifacts, "_rename_new", fail_after_first)
    with pytest.raises(OSError, match="injected"): import_bundle(f["root"], target, [f["starter"]])
    assert snapshot(target) == before
    monkeypatch.setattr(artifacts, "_rename_new", original)
    import_bundle(f["root"], target, [f["starter"]])
    complete = snapshot(target)
    import_bundle(f["root"], target, [f["starter"]])
    assert snapshot(target) == complete


def test_source_snapshot_rejects_extra_file_and_mode(make_bundle):
    f = make_bundle(); bundle = Bundle(f["root"])
    directory = f["root"] / Path(f["starter"]["path"]).parent / "files"
    (directory / "solution.json").write_bytes(b"{}")
    with pytest.raises(ProtocolError, match="extra"): bundle.source(f["starter"])
    (directory / "solution.json").unlink()
    (directory / "kernel.py").chmod(0o755)
    with pytest.raises(ProtocolError, match="mode"): bundle.source(f["starter"])


def evaluation(f):
    not_run = {"status": "not_run", "receipt": None, "reason": "synthetic, no execution"}
    return {"protocol": PROTOCOL, "kind": "evaluation", "operation_id": "synthetic-eval",
            "subject": {"task_sha256": f["task"]["sha256"], "source_sha256": f["starter"]["sha256"], "evaluator_sha256": f["evaluator"]["sha256"],
                        **{name + "_sha256": f["originals"]["task/" + file]["sha256"] for name, file in (("inputs", "inputs.json"), ("timing", "timing.md"), ("environment", "environment.md"))}},
            "build": not_run, "correctness": not_run, "performance": not_run, "case_ids": [], "samples": None,
            "raw_receipts": [], "usage_receipts": [], "environment": f["originals"]["task/environment.md"], "extensions": {"information": {"synthetic": True}}}


def test_evidence_requires_actual_subject_locators_and_preserves_native_bytes(make_bundle, tmp_path):
    f = make_bundle(); p = Publication(f["root"])
    native = p.put_bytes("native.json", b'{"elapsed":0.125,"message":"ordinary path missing/log.txt"}\r\n')
    report = evaluation(f); report["raw_receipts"] = [native]
    report_ref = p.put_json("report.json", report)
    with pytest.raises(ProtocolError, match="no locator"): p.commit([report_ref])
    p.commit([report_ref], subjects=[f["request_ref"]])
    target = tmp_path / "export"
    import_bundle(f["root"], target, [report_ref], subjects=[f["request_ref"]])
    assert Bundle(target).read(native).endswith(b"\r\n")
    assert not (target / "missing/log.txt").exists()
    # A naked hash with no real selected subject cannot be patched into evidence.
    report["subject"]["source_sha256"] = sha256(b"unknown C")
    p = Publication(f["root"])
    broken = p.put_json("broken.json", report)
    with pytest.raises(ProtocolError, match="no locator"): p.commit([broken], subjects=[f["request_ref"]])


def test_publication_reopen_reuses_id_but_never_overwrites_bytes(tmp_path):
    p = Publication(tmp_path / "bundle")
    source = p.put_source("source", b"pass")
    reopened = Publication(p.root, p.publication_id)
    assert reopened.put_source("source", b"pass") == source
    reopened.commit([source])
    with pytest.raises(ProtocolError, match="immutable"): reopened.put_source("source", b"pass\n")


def test_process_death_leaves_fence_until_durable_rollback(make_bundle, tmp_path):
    f = make_bundle(); target = tmp_path / "target"; target.mkdir()
    (target / "keep.txt").write_bytes(b"existing data")
    before = snapshot(target)
    already_open = Bundle(target)
    pid = os.fork()
    if pid == 0:
        original = artifacts._rename_new
        count = 0
        def crash(source, destination):
            nonlocal count
            if destination.is_relative_to(target / "artifacts"):
                count += 1
                if count == 2: os._exit(99)
            return original(source, destination)
        artifacts._rename_new = crash
        try:
            import_bundle(f["root"], target, [f["starter"]])
        finally:
            os._exit(98)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 99
    with pytest.raises(ProtocolError, match="interrupted import"): Bundle(target)
    with pytest.raises(ProtocolError, match="interrupted import"): already_open.read(f["starter"])
    recover_import(target)
    assert snapshot(target) == before
    import_bundle(f["root"], target, [f["starter"]])
    assert Bundle(target).source(f["starter"])
