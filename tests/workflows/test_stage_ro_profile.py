"""Synthetic wire checks plus real local namespace/mount/permission checks.

The local checks establish no CUDA/Agate qualification. No SDK/network is used.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from kernelschema.artifacts import Publication
from kernelschema.budget import use_authority
from kernelschema.contracts import ProtocolError, UnsupportedError, digest, jcs, loads, sha256
from kernelschema.evaluation import _stage_ro
from kernelschema.evaluation.agate import STAGE_RO_PROFILE, AgateDevExecutor, _unframe, interpret_terminal, validate_prepared_request
from kernelschema.evaluation.channel import AgateChannel, _remote_profile, qualify, read_qualification, selection
from kernelschema.evaluation.task import stage
from test_agate_channel import Wire, context, frame


def remote(channel=STAGE_RO_PROFILE):
    return {"channel": channel, "gpu": "synthetic-gpu",
            **{n: "/synthetic/" + n for n in _stage_ro.TOOLS},
            "tool_sha256": {n: sha256(("synthetic " + n).encode()) for n in _stage_ro.TOOLS}}


def permissions(files, profile):
    hashes = {k.removeprefix("_ks_stage/"): sha256(v.encode()) for k, v in files.items()}
    proof = {"profile": profile["channel"], "scope": "complete_evaluator_stage", "whole_root_readonly": False,
        "stage_files": hashes, "tool_sha256": profile["tool_sha256"],
        "namespaces_before": {n: n + ":[100]" for n in _stage_ro.NAMESPACES},
        "namespaces_after": {n: n + ":[200]" for n in _stage_ro.NAMESPACES},
        "private_mounts": True, "fresh_proc_pid": 1,
        "caps": {**{n: "0000000000000000" for n in _stage_ro.CAPS}, "NoNewPrivs": "1"},
        "inherited_fds": [], "cwd": "/tmp/ks-compile-synthetic", "scratch": "/tmp/ks-compile-synthetic",
        "scratch_writable": True, "mount_aliases": [],
        "probes": [{"path": n, "write_open_denied": True, "errno": 30} for n in [*hashes, ".ks-stage-write-probe"]]}
    if profile["channel"] == _stage_ro.INHERITED_PROFILE:
        del proof["fresh_proc_pid"]
        proof["proc"] = {"mode": "inherited-restricted", "private_pid_view": False, "mounts_unchanged": True,
            "namespace_pid": 1, "proc_self_pid": 103, "outer": {"pid": 101, "directory_fd": 3, "file_fd": 4, "cwd": "/tmp"},
            "helper": {"pid": 102, "namespaces": {**proof["namespaces_after"], "pid": proof["namespaces_before"]["pid"]},
                "permitted_caps": "000001ffffffffff", "cwd": proof["scratch"], "inherited_fds": [], "pidfds": [{"fd": 3, "target": "anon_inode:[pidfd]", "pid": 102}]},
            "mounts": [{"path": "/proc", "type": "proc", "propagation": []}],
            "denied": [{"target": name, "errno": 13} for name in ("outer.root", "outer.cwd", "outer.directory_fd", "outer.file_fd",
                "helper.root", "helper.cwd", "helper.fd0", "helper.fd1", "helper.fd2", "helper.fd3")],
            "self_probes": [{"via": via, "errno": 30} for via in ("root", "cwd", "fd")]}
    return proof


class StageWire(Wire):
    def request(self, method, path, payload=None):
        response = super().request(method, path, payload)
        if method == "POST":
            native = self.native["synthetic_" + str(len(self.posts))]
            execution = _unframe(native["result"]["stdout"])
            files = payload["files"]
            settings = loads(files["_ks_stage/remote.json"])
            manifest = loads(files["_ks_stage/stage.json"])
            worker = loads(execution["stdout"])
            worker.update({"profile": settings["channel"], "permissions": permissions(files, settings),
                "loaded": manifest["files"], "source": manifest["source"], "evaluator": manifest["evaluator"]})
            execution.pop("bwrap_sha256")
            execution.update(profile=settings["channel"], tool_sha256=settings["tool_sha256"], stdout=jcs(worker).decode())
            native["result"]["stdout"] = frame(execution)
        return response


def _qualified(tmp_path_factory, channel):
    demo, ctx, target, selected = context(tmp_path_factory.mktemp("stage-ro-wire"), remote=remote(channel))
    wire = StageWire(target); wire.unknown = True
    with use_authority(demo.owner.path):
        unknown = qualify(ctx, remote=selected, transport=wire)
        assert qualify(ctx, remote=selected, transport=wire)["recovery"] == "settle_only"
    assert unknown["recovery"] == "settle_only" and len(wire.posts) == 1
    wire.unknown = False
    with use_authority(demo.owner.path):
        result = qualify(ctx, remote=selected, transport=wire)
        replay = qualify(ctx, remote=selected, transport=wire)
    assert result["status"] == replay["status"] == "completed"
    assert result["usage"]["evaluator_jobs"] == len(wire.posts) == 6
    return demo, ctx, selected, wire, result


@pytest.fixture(scope="module")
def profile_qualifications(tmp_path_factory):
    return {channel: _qualified(tmp_path_factory, channel) for channel in _stage_ro.PROFILES}


@pytest.fixture(scope="module", params=_stage_ro.PROFILES)
def qualified(profile_qualifications, request):
    return profile_qualifications[request.param]


def test_new_profile_six_jobs_selection_and_original_recovery(qualified, monkeypatch):
    demo, ctx, selected, wire, result = qualified
    read_qualification(result["qualification"], context=ctx, remote=selected, transport=wire.identity())
    assert len(demo.owner.operations()) == 6
    assert all(i["native_locator"]["profile"] == selected["channel"] for i in ctx.intents())
    assert all(loads(p["files"]["_ks_stage/remote.json"])["tool_sha256"] == selected["tool_sha256"] for p in wire.posts)
    profile = ctx.bundle.document(demo.config["reference_profile"])
    profile["parameters"]["evaluation"] = selection(remote=selected, qualification=result["qualification"])
    channel = AgateChannel(profile, ctx.root / "next", transport=wire)
    # Static hardware binding is still required; this wire cannot grant it.
    assert channel.admission(ctx)["status"] == "unknown"
    from kernelschema import template_api
    from kernelschema.evaluation import RowSoftmaxTask
    from kernelschema.budget import cost
    monkeypatch.setattr(RowSoftmaxTask, "validate_scope", lambda *a, **k: {"status": "pass"})
    effective = template_api.effective_document
    def bound(*args, **kwargs):
        document = effective(*args, **kwargs)
        document["CONTRACT"]["Hardware_specification"] = wire.target["hardware_specification"]
        return document
    monkeypatch.setattr(template_api, "effective_document", bound)
    assert channel.admission(ctx)["status"] == "pass"
    opid = ctx.run_id + "_wrapper"
    reservation = demo.owner.reserve(ctx.run_id, opid, role="evaluator", purpose="synthetic stage-ro wrapper",
                                     amount=cost(evaluator_jobs=1, active_ms=1000))
    op = {"id": opid, "source": ctx.objects["task"]["baseline_source"], "reservation": reservation}
    op["stage"] = stage(ctx.bundle, ctx.request["task"], op["source"], ctx.root / "wrapper-stage", scope=ctx.request["scope"])
    op["locator"] = channel.locator(opid)
    assert op["locator"]["profile"] == selected["channel"]
    channel.prepare(ctx, op)
    channel.submit(ctx, op, authorize=lambda: demo.owner.authorize(reservation))
    settled = channel.settle(ctx, op, Publication(ctx.root))
    demo.owner.absorb(ctx.run_id, settled["usage_row"], terminated=True)
    assert ctx.bundle.document(settled["report"])["correctness"]["status"] == "pass"
    channel.validate(ctx, op)
    channel.submit(ctx, op, authorize=lambda: pytest.fail("original key cannot relaunch"))
    assert len(wire.posts) == 7
    # An old bwrap selection cannot reinterpret the new profile's records.
    old = {"channel": "agate-dev-bwrap-v1", "gpu": selected["gpu"], "python": selected["python"],
           "bwrap": "/synthetic/bwrap", "bwrap_sha256": sha256(b"synthetic-bwrap")}
    with pytest.raises(ProtocolError, match="profile changed"):
        read_qualification(result["qualification"], context=ctx, remote=old, transport=wire.identity())
    other = {**selected, "channel": next(p for p in _stage_ro.PROFILES if p != selected["channel"])}
    with pytest.raises(ProtocolError, match="profile changed"):
        read_qualification(result["qualification"], context=ctx, remote=other, transport=wire.identity())
    with pytest.raises(ProtocolError):  # FA4's original validator remains bwrap-only.
        _remote_profile(selected)


@pytest.mark.parametrize("change", ["missing", "root", "namespace", "proc", "caps", "fd", "alias", "scratch", "errno", "metadata", "tool", "loaded", "truncated"])
def test_forged_or_incomplete_boundary_cannot_become_a_report(qualified, change):
    demo, ctx, selected, wire, result = qualified
    record = result["qualification"]["execute"]
    intent, native = loads(record["intent.json"]), loads(record["terminal.raw"])
    execution = _unframe(native["result"]["stdout"]); worker = loads(execution["stdout"])
    p = worker["permissions"]
    if change == "missing": del p["proc" if selected["channel"] == _stage_ro.INHERITED_PROFILE else "fresh_proc_pid"]
    elif change == "root": p["whole_root_readonly"] = True
    elif change == "namespace": p["namespaces_after"]["mnt"] = p["namespaces_before"]["mnt"]
    elif change == "proc": p["fresh_proc_pid"] = 123
    elif change == "caps": p["caps"]["CapBnd"] = "1"
    elif change == "fd": p["inherited_fds"] = [{"fd": "3", "target": "/writable/stage"}]
    elif change == "alias": p["mount_aliases"] = ["/old-stage"]
    elif change == "scratch": p["scratch_writable"] = False
    elif change == "errno": p["probes"][0]["errno"] = 13
    elif change == "metadata": del p["stage_files"]["execution.json"]
    elif change == "tool": execution["tool_sha256"]["mount"] = sha256(b"different")
    elif change == "loaded": worker["loaded"]["_stage_ro.py"] = sha256(b"different")
    elif change == "truncated": native["result"]["output_truncated"] = True
    execution["stdout"] = jcs(worker).decode(); native["result"]["stdout"] = frame(execution)
    observed, diagnostic = interpret_terminal(intent, native)
    assert observed is None and diagnostic


@pytest.mark.parametrize("channel", _stage_ro.PROFILES)
def test_profile_missing_hashes_upload_change_and_no_bwrap_fallback(tmp_path, channel):
    demo, ctx, target, selected = context(tmp_path, remote=remote(channel))
    wire = StageWire(target)
    for name in [*_stage_ro.TOOLS, "tool_sha256"]:
        bad = deepcopy(selected); del bad[name]
        with pytest.raises(ProtocolError): selection(remote=bad)
    bad = deepcopy(selected); del bad["tool_sha256"]["setpriv"]
    with pytest.raises(ProtocolError): selection(remote=bad)
    staged = stage(ctx.bundle, ctx.request["task"], ctx.objects["task"]["baseline_source"], tmp_path / "stage")
    e = AgateDevExecutor(tmp_path / "jobs", wire)
    e.prepare(staged, profile=selected, operation_id="one", submission_key="original-key", owner="owner",
              reservation={"owner": "owner", "evaluator_jobs": 1, "active_ms": 10000}, action="probe")
    original = loads((e.root / "one/request.json").read_bytes())
    intent = loads((e.root / "one/intent.json").read_bytes())
    for name in ("_ks_stage/_stage_ro.py", "_ks_stage/remote.json", "_ks_stage/stage.json"):
        modified = deepcopy(original); modified["files"][name] += " "
        (e.root / "one/request.json").write_bytes(jcs(modified))
        with pytest.raises(ProtocolError): e.submit("one")
        with pytest.raises(ProtocolError, match="uploaded"):
            validate_prepared_request({**intent, "request_sha256": digest(modified)}, modified)
    assert not wire.posts
    (e.root / "one/request.json").write_bytes(jcs(original))
    wire.lost_ack = True
    assert e.submit("one")["recovery"] == e.submit("one")["recovery"] == "settle_only"
    assert len(wire.posts) == 1


def local_stage(tmp_path, channel=STAGE_RO_PROFILE, worker=None):
    root = tmp_path / "stage"; root.mkdir()
    paths = {"python": sys.executable, **{n: shutil.which(n) for n in ("unshare", "mount", "setpriv")}}
    if any(p is None for p in paths.values()): pytest.skip("local util-linux unavailable")
    profile = {"channel": channel, **paths, "tool_sha256": {n: sha256(Path(p).read_bytes()) for n, p in paths.items()}}
    helper = Path(_stage_ro.__file__).read_bytes()
    # A probe worker invokes the production permission verifier; no fake ro flags.
    worker = worker or b'import json,runpy,sys\nfrom pathlib import Path\nr=Path(sys.argv[1])\nh=runpy.run_path(str(r/"_stage_ro.py"))\nprint(json.dumps(h["prove"](r,json.loads(sys.argv[3]))))\n'
    payload = {"_stage_ro.py": helper, "_worker.py": worker}
    for name, data in payload.items(): (root / name).write_bytes(data)
    (root / "stage.json").write_bytes(jcs({"files": {n: sha256(b) for n, b in payload.items()}}))
    (root / "remote.json").write_bytes(jcs(profile))
    return root, profile


@pytest.mark.parametrize("channel", _stage_ro.PROFILES)
def test_actual_local_namespace_ro_scratch_and_parent_unchanged(tmp_path, channel):
    root, profile = local_stage(tmp_path, channel)
    before = _stage_ro.snapshot(root)
    result = _stage_ro.run(root, profile, "probe", 20)
    if result["exit_code"] and ("unshare: unshare failed: Operation not permitted" in result["stderr"]
                               or "mount: /proc: permission denied" in result["stderr"]):
        assert _stage_ro.snapshot(root) == before
        pytest.skip("local kernel denies namespace/fresh-proc setup; failed closed, no readonly claim")
    assert result["exit_code"] == 0, result["stderr"]
    proof = loads(result["stdout"])
    assert proof["stage_files"] == before and all(p["errno"] == 30 for p in proof["probes"])
    assert proof["whole_root_readonly"] is False and proof["scratch_writable"] is True
    if channel == _stage_ro.INHERITED_PROFILE:
        from kernelschema.evaluation.agate import validate_inherited_proc
        validate_inherited_proc(proof, proof["namespaces_before"], proof["namespaces_after"])
        assert "fresh_proc_pid" not in proof
        assert all(p["errno"] == 13 for p in proof["proc"]["denied"])
        assert all(p["errno"] == 30 for p in proof["proc"]["self_probes"])
    assert not Path(proof["scratch"]).exists() and _stage_ro.snapshot(root) == before
    with (root / "_worker.py").open("ab") as output: output.write(b"# parent still writable\n")


def test_namespace_failure_precedes_every_mount_and_real_missing_tool(tmp_path, monkeypatch):
    root, profile = local_stage(tmp_path)
    state = {"namespaces_before": _stage_ro.namespaces(), "stage_files": _stage_ro.snapshot(root), "scratch": str(tmp_path)}
    monkeypatch.setattr(_stage_ro.subprocess, "run", lambda *a, **k: pytest.fail("cannot mutate a mount before namespace proof"))
    with pytest.raises(RuntimeError, match="independent namespaces absent"):
        _stage_ro.inner(root, tmp_path, state, "probe")
    changed = deepcopy(profile); changed["tool_sha256"]["mount"] = sha256(b"changed executable")
    with pytest.raises(RuntimeError, match="actual tool hashes differ"):
        _stage_ro.run(root, changed, "probe", 1)
    profile["unshare"] = str(tmp_path / "no-tool")
    with pytest.raises(RuntimeError, match="existing executable"):
        _stage_ro.run(root, profile, "probe", 1)


def test_stage_hardlink_symlink_and_mount_aliases_refused(tmp_path):
    root, _ = local_stage(tmp_path)
    os.link(root / "_worker.py", tmp_path / "outside-alias")
    with pytest.raises(RuntimeError, match="alias forbidden"): _stage_ro.snapshot(root)
    (tmp_path / "outside-alias").unlink()
    (root / "link").symlink_to(root / "_worker.py")
    with pytest.raises(RuntimeError, match="alias forbidden"): _stage_ro.snapshot(root)
    rows = [{"device": "0:1", "root": Path("/"), "path": Path("/"), "type": "overlay", "options": ["rw"]},
            {"device": "0:1", "root": root, "path": Path("/alias"), "type": "overlay", "options": ["rw"]}]
    with pytest.raises(RuntimeError, match="mount alias"): _stage_ro.stage_mounts(root, rows)


@pytest.mark.parametrize("change", ["private_pid_view", "mounts", "helper_namespace", "helper_caps", "helper_fd",
                                  "outer_fd", "outer_cwd", "pid", "pidfd", "denial_missing", "denial_errno", "self_missing", "self_errno"])
def test_inherited_proc_missing_or_forged_proof_refuses_qualification(profile_qualifications, change):
    _, ctx, selected, wire, result = profile_qualifications[_stage_ro.INHERITED_PROFILE]
    corrupted = deepcopy(result["qualification"])
    record = corrupted["execute"]
    native = loads(record["terminal.raw"])
    execution = _unframe(native["result"]["stdout"]); worker = loads(execution["stdout"])
    p = worker["permissions"]["proc"]
    if change == "private_pid_view": p["private_pid_view"] = True
    elif change == "mounts": p["mounts"] = []
    elif change == "helper_namespace": p["helper"]["namespaces"]["user"] = worker["permissions"]["namespaces_before"]["user"]
    elif change == "helper_caps": p["helper"]["permitted_caps"] = "0000000000000000"
    elif change == "helper_fd": p["helper"]["inherited_fds"] = [3]
    elif change == "outer_fd": p["outer"]["file_fd"] = p["outer"]["directory_fd"]
    elif change == "outer_cwd": del p["outer"]["cwd"]
    elif change == "pid": p["proc_self_pid"] = 1
    elif change == "pidfd": p["helper"]["pidfds"] = [{"fd": 3, "target": "anon_inode:[pidfd]", "pid": p["outer"]["pid"]}]
    elif change == "denial_missing": p["denied"].pop()
    elif change == "denial_errno": p["denied"][0]["errno"] = 2
    elif change == "self_missing": p["self_probes"].pop()
    elif change == "self_errno": p["self_probes"][0]["errno"] = 13
    execution["stdout"] = jcs(worker).decode(); native["result"]["stdout"] = frame(execution)
    record["terminal.raw"] = jcs(native).decode()
    with pytest.raises(UnsupportedError, match="incomplete"):
        read_qualification(corrupted, context=ctx, remote=selected, transport=wire.identity())


def test_actual_old_view_fd_in_unshare_helper_is_rejected(tmp_path, monkeypatch):
    root, profile = local_stage(tmp_path, _stage_ro.INHERITED_PROFILE)
    fd = os.open(root / "_worker.py", os.O_RDWR)
    popen = subprocess.Popen
    def leak(*args, **kwargs):
        kwargs["pass_fds"] = (fd,)
        return popen(*args, **kwargs)
    try:
        monkeypatch.setattr(subprocess, "Popen", leak)
        result = _stage_ro.run(root, profile, "probe", 20)
    finally:
        os.close(fd)
    if "unshare: unshare failed: Operation not permitted" in result["stderr"]:
        pytest.skip("local kernel denies namespace setup")
    assert result["exit_code"] != 0 and "helper retained an original stage FD" in result["stderr"]


@pytest.mark.parametrize("target", ["self", "parent"])
def test_actual_helper_pidfd_must_name_itself(tmp_path, target):
    if not hasattr(os, "pidfd_open"):
        pytest.skip("local Python/kernel does not provide pidfds")
    # Only our own controlled child and this test parent are involved.
    script = "import os,sys; fd=os.pidfd_open(os.getpid() if sys.argv[1]=='self' else os.getppid()); print('ready',flush=True); sys.stdin.readline()"
    with subprocess.Popen([sys.executable, "-I", "-B", "-c", script, target], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True, text=True) as child:
        try:
            assert child.stdout.readline() == "ready\n"
            if target == "self":
                proof = _stage_ro.helper_pidfds(child.pid, str(tmp_path))
                assert len(proof) == 1 and proof[0] == {"fd": 3, "target": "anon_inode:[pidfd]", "pid": child.pid}
            else:
                with pytest.raises(RuntimeError, match="not its own process"):
                    _stage_ro.helper_pidfds(child.pid, str(tmp_path))
        finally:
            child.communicate("done\n", timeout=10)


def test_actual_nested_user_namespace_cannot_unlock_stage(tmp_path):
    if not hasattr(os, "unshare"):
        pytest.skip("local Python has no os.unshare for descendant capability test")
    # Direct syscalls preserve regained descendant capabilities across the test;
    # exec of mount would lose them under the already-empty bounding set.
    worker = b'''import ctypes, errno, json, os, runpy, sys
from pathlib import Path
r=Path(sys.argv[1]);s=json.loads(sys.argv[3]);h=runpy.run_path(str(r/'_stage_ro.py'))
p=h['prove'](r,s); checks={}
for name,action in [('link',lambda:os.link(r/'_worker.py',Path(s['scratch'])/'link')),('chmod',lambda:os.chmod(r/'_worker.py',0o666))]:
    try:action()
    except OSError as e:checks[name]=e.errno
    else:raise RuntimeError('stage mutation allowed: '+name)
before=h['namespaces']()
os.unshare(os.CLONE_NEWUSER|os.CLONE_NEWNS)
after=h['namespaces']()
assert before['user']!=after['user'] and before['mnt']!=after['mnt']
assert int(h['status']()['CapEff'],16)
libc=ctypes.CDLL(None,use_errno=True);alias=Path(s['scratch'])/'alias';alias.mkdir()
for name,action in [('remount',lambda:libc.mount(None,os.fsencode(r),None,4096|32,None)),('unmount',lambda:libc.umount2(os.fsencode(r),2)),('bind_parent',lambda:libc.mount(os.fsencode(r.parent),os.fsencode(alias),None,4096,None))]:
    ctypes.set_errno(0)
    result=action();checks[name]=ctypes.get_errno()
    if result==0:raise RuntimeError('stage mount unlocked: '+name)
checks['outer_fd']=h['write_denied'](Path('/proc')/str(s['outer']['pid'])/'fd'/str(s['outer']['file_fd']),(errno.EACCES,errno.EPERM))
checks['stage']=h['write_denied'](r/'_worker.py',(errno.EROFS,))
print(json.dumps(checks))
'''
    root, profile = local_stage(tmp_path, _stage_ro.INHERITED_PROFILE, worker=worker)
    before = _stage_ro.snapshot(root)
    result = _stage_ro.run(root, profile, "probe", 20)
    if "unshare: unshare failed: Operation not permitted" in result["stderr"]:
        pytest.skip("local kernel denies namespace setup")
    assert result["exit_code"] == 0, result["stderr"]
    assert loads(result["stdout"]) == {"link": 18, "chmod": 30, "remount": 1, "unmount": 22,
                                      "bind_parent": 22, "outer_fd": 13, "stage": 30}
    assert _stage_ro.snapshot(root) == before
