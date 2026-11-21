"""Fixed util-linux launcher for one complete row-softmax evaluator stage.

This protects stage files, not the whole root or Python's address space. Only
the private mount namespace changes: no installation or host chmod/remount.
The same file is frozen with the Task and used by the remote and worker.
"""
import errno
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import stat
import subprocess
import sys
import tempfile

PROFILE = "agate-dev-stage-ro-v1"
INHERITED_PROFILE = "agate-dev-stage-ro-inherited-proc-v1"
PROFILES = (PROFILE, INHERITED_PROFILE)
TOOLS = ("python", "unshare", "mount", "setpriv")
NAMESPACES = ("user", "mnt", "pid", "net", "uts", "ipc")
CAPS = ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")


def sha(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def tool_hashes(profile):
    result = {}
    for name in TOOLS:
        path = Path(profile[name])
        if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError("existing executable required: " + name)
        result[name] = sha(path.read_bytes())
    if result != profile["tool_sha256"]:
        raise RuntimeError("actual tool hashes differ from fixed stage-ro profile")
    if sha(Path("/proc/self/exe").read_bytes()) != result["python"]:
        raise RuntimeError("actual loaded Python differs from fixed stage-ro profile")
    return result


def namespaces(pid="self"):
    return {name: os.readlink("/proc/" + str(pid) + "/ns/" + name) for name in NAMESPACES}


def status(pid="self"):
    return dict(line.split(":", 1) for line in Path("/proc/" + str(pid) + "/status").read_text().splitlines())


def snapshot(root):
    """Reject links/special files; the uploaded tree is the only loading tree."""
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("stage must be an ordinary directory")
    files = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError("stage symlink/hardlink/special-file alias forbidden")
        files[str(path.relative_to(root))] = sha(path.read_bytes())
    manifest = json.loads((root / "stage.json").read_bytes())
    extra = {"stage.json", "remote.json"}
    if (root / "execution.json").exists():
        extra |= {"execution.json", "frozen_inputs.json"}
    if set(files) != set(manifest["files"]) | extra or any(files[k] != v for k, v in manifest["files"].items()):
        raise RuntimeError("complete stage differs from manifest")
    return files


def mounts():
    rows = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields = line.split(); split = fields.index("-")
        decode = lambda value: re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)
        rows.append({"device": fields[2], "root": Path(decode(fields[3])), "path": Path(decode(fields[4])),
                     "options": fields[5].split(","), "propagation": fields[6:split], "type": fields[split + 1],
                     "super_options": fields[split + 3].split(",")})
    return rows


def proc_mounts():
    # Include inherited masks/submounts: this profile never alters /proc.
    return [{**row, "root": str(row["root"]), "path": str(row["path"])} for row in mounts()
            if row["path"].is_relative_to(Path("/proc"))]


def helper_pidfds(pid, root):
    """util-linux 2.39 may retain one self pidfd for --kill-child."""
    fds = {p.name: os.readlink(p) for p in Path("/proc/" + str(pid) + "/fd").iterdir()}
    if not {"0", "1", "2"}.issubset(fds) or any(v == root or v.startswith(root + "/") for v in fds.values()):
        raise RuntimeError("unshare helper retained an original stage FD")
    pidfds = []
    for fd, target in fds.items():
        if int(fd) <= 2:
            continue
        if target != "anon_inode:[pidfd]" or pidfds:
            raise RuntimeError("unshare helper retained an original stage FD or unknown descriptor")
        info = dict(line.split(":", 1) for line in Path("/proc/" + str(pid) + "/fdinfo/" + fd).read_text().splitlines())
        if info.get("Pid", "").strip() != str(pid):
            raise RuntimeError("unshare helper pidfd is not its own process")
        pidfds.append({"fd": int(fd), "target": target, "pid": pid})
    return pidfds


def inherited_helper(state, after):
    """Only our known unshare parent; never enumerate unrelated processes."""
    pid = int(status()["PPid"])
    parent = status(pid)
    ns = namespaces(pid)
    if (int(parent["PPid"]) != state["outer"]["pid"] or ns != {**after, "pid": state["namespaces_before"]["pid"]}
            or os.readlink("/proc/" + str(pid) + "/cwd") != state["scratch"]
            or not int(parent["CapPrm"], 16)):
        raise RuntimeError("unexpected unshare helper identity/capabilities/cwd")
    return {"pid": pid, "namespaces": ns, "permitted_caps": parent["CapPrm"].strip(),
            "cwd": state["scratch"], "inherited_fds": [], "pidfds": helper_pidfds(pid, state["root"])}


def write_denied(path, expected):
    try:
        fd = os.open(path, os.O_WRONLY)
    except OSError as error:
        if error.errno not in expected:
            raise
        return error.errno
    else:
        os.close(fd)
        raise RuntimeError("stage alias write unexpectedly allowed: " + str(path))


def prove_inherited_proc(root, state, after):
    """Cross-userns commoncap checks protect old views; own views remain RO."""
    outer, helper = state["outer"], state["helper"]
    proc_pid = int(os.readlink("/proc/self"))
    if (os.getpid() != 1 or proc_pid == 1 or int(status()["PPid"]) != helper["pid"]
            or int(status(helper["pid"])["PPid"]) != outer["pid"]
            or status(helper["pid"])["CapPrm"].strip() != helper["permitted_caps"]
            or helper["namespaces"] != {**after, "pid": state["namespaces_before"]["pid"]}
            or proc_mounts() != state["proc_mounts"]):
        raise RuntimeError("inherited proc/helper identity or original mounts changed")
    denied = []
    for target, path, readlink in [
        ("outer.root", Path("/proc") / str(outer["pid"]) / "root" / str(root).lstrip("/") / "_worker.py", False),
        ("outer.cwd", Path("/proc") / str(outer["pid"]) / "cwd" / os.path.relpath(root, outer["cwd"]) / "_worker.py", False),
        ("outer.directory_fd", Path("/proc") / str(outer["pid"]) / "fd" / str(outer["directory_fd"]) / "_worker.py", False),
        ("outer.file_fd", Path("/proc") / str(outer["pid"]) / "fd" / str(outer["file_fd"]), False),
        ("helper.root", Path("/proc") / str(helper["pid"]) / "root" / str(root).lstrip("/") / "_worker.py", False),
        ("helper.cwd", Path("/proc") / str(helper["pid"]) / "cwd" / os.path.relpath(root, state["scratch"]) / "_worker.py", False),
        *[("helper.fd" + str(fd), Path("/proc") / str(helper["pid"]) / "fd" / str(fd), True)
          for fd in [0, 1, 2, *[item["fd"] for item in helper["pidfds"]]]],
    ]:
        if readlink:
            try:
                os.readlink(path)
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EPERM):
                    raise
                code = error.errno
            else:
                raise RuntimeError("outside proc alias unexpectedly accessible: " + target)
        else:
            code = write_denied(path, (errno.EACCES, errno.EPERM))
        denied.append({"target": target, "errno": code})
    self_probes = []
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for via, path in [("root", Path("/proc/self/root") / str(root).lstrip("/") / "_worker.py"),
                          ("cwd", Path("/proc/self/cwd") / os.path.relpath(root, state["scratch"]) / "_worker.py"),
                          ("fd", Path("/proc/self/fd") / str(fd) / "_worker.py")]:
            self_probes.append({"via": via, "errno": write_denied(path, (errno.EROFS,))})
    finally:
        os.close(fd)
    return {"mode": "inherited-restricted", "private_pid_view": False, "mounts_unchanged": True,
            "namespace_pid": 1, "proc_self_pid": proc_pid, "mounts": state["proc_mounts"],
            "outer": outer, "helper": helper, "denied": denied, "self_probes": self_probes}


def stage_mounts(root, rows, *, readonly=False):
    """A second mount of the backing subtree would expose a writable alias."""
    containing = max((r for r in rows if root.is_relative_to(r["path"])), key=lambda r: len(r["path"].parts))
    backing = containing["root"] / root.relative_to(containing["path"])
    for row in rows:
        if row["path"] != root and row["path"].is_relative_to(root):
            raise RuntimeError("stage submount forbidden")
        if row["device"] == containing["device"] and backing.is_relative_to(row["root"]):
            alias = row["path"] / backing.relative_to(row["root"])
            if alias != root:
                raise RuntimeError("stage backing has another mount alias")
        if row["type"] == "proc" and not row["path"].is_relative_to(Path("/proc")):
            raise RuntimeError("alternate proc mount forbidden")
    if readonly and (containing["path"] != root or "ro" not in containing["options"]):
        raise RuntimeError("stage mount is not read-only")


def independent(before):
    after = namespaces()
    if set(before) != set(NAMESPACES) or any(before[n] == after[n] for n in NAMESPACES):
        raise RuntimeError("independent namespaces absent; refusing any mount operation")
    return after


def inner(root, scratch, state, action):
    # This guard precedes every mount mutation, including propagation changes.
    after = independent(state["namespaces_before"])
    if os.getpid() != 1 or Path.cwd() != scratch or root.is_relative_to(scratch) or scratch.is_relative_to(root):
        raise RuntimeError("unexpected PID/cwd/stage/scratch layout")
    profile = json.loads((root / "remote.json").read_bytes())
    tool_hashes(profile)
    if snapshot(root) != state["stage_files"]:
        raise RuntimeError("stage changed before namespace entry")
    stage_mounts(root, mounts())
    def mount(*arguments):
        subprocess.run([profile["mount"], *arguments], check=True, stdin=subprocess.DEVNULL, close_fds=True)
    mount("--make-rprivate", "/")
    if any(r["propagation"] for r in mounts()):
        raise RuntimeError("mount propagation is not private")
    if profile["channel"] == INHERITED_PROFILE:
        state["helper"] = inherited_helper(state, after)
        state["proc_mounts"] = proc_mounts()
        if not any(r["path"] == "/proc" and r["type"] == "proc" for r in state["proc_mounts"]):
            raise RuntimeError("inherited proc mount absent")
    elif profile["channel"] == PROFILE:
        mount("-t", "proc", "-o", "nosuid,nodev,noexec", "proc", "/proc")
        if os.readlink("/proc/self") != "1" or namespaces()["pid"] != os.readlink("/proc/1/ns/pid"):
            raise RuntimeError("fresh proc does not belong to the new PID namespace")
    else:
        raise RuntimeError("unknown stage-ro profile")
    mount("--bind", str(root), str(root))
    mount("-o", "remount,bind,ro,nosuid,nodev", str(root))
    stage_mounts(root, mounts(), readonly=True)
    # exec preserves only standard IO, no descriptor/cwd into the old mount.
    os.closerange(3, resource.getrlimit(resource.RLIMIT_NOFILE)[0])
    os.execv(profile["setpriv"], [profile["setpriv"], "--bounding-set=-all", "--inh-caps=-all",
        "--ambient-caps=-all", "--no-new-privs", profile["python"], "-I", "-B", str(root / "_worker.py"),
        str(root), "prepare-cuda" if action == "probe" else "execute", json.dumps(state, separators=(",", ":"))])


def prove(root, state):
    """Run after setpriv, before torch/evaluator/candidate import."""
    profile = json.loads((root / "remote.json").read_bytes())
    actual_tools = tool_hashes(profile)
    after = independent(state["namespaces_before"])
    rows = mounts(); stage_mounts(root, rows, readonly=True)
    if any(r["propagation"] for r in rows):
        raise RuntimeError("mount propagation is not private")
    actual_status = status()
    caps = {name: actual_status[name].strip() for name in CAPS + ("NoNewPrivs",)}
    if any(int(caps[name], 16) for name in CAPS) or caps["NoNewPrivs"] != "1":
        raise RuntimeError("capabilities/NNP not restricted")
    if profile["channel"] == INHERITED_PROFILE:
        proc = {"proc": prove_inherited_proc(root, state, after)}
    elif profile["channel"] == PROFILE:
        if os.getpid() != 1 or os.readlink("/proc/self") != "1" or os.readlink("/proc/1/ns/pid") != after["pid"]:
            raise RuntimeError("fresh PID namespace proc absent")
        proc = {"fresh_proc_pid": 1}
    else:
        raise RuntimeError("unknown stage-ro profile")
    inherited = []
    for fd in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:  # iterdir's own directory descriptor is closed
            continue
        if int(fd.name) > 2 or target == str(root) or target.startswith(str(root) + "/"):
            inherited.append({"fd": fd.name, "target": target})
    if inherited or str(Path.cwd()) != state["scratch"]:
        raise RuntimeError("unexpected inherited FD or original stage cwd")
    if snapshot(root) != state["stage_files"]:
        raise RuntimeError("protected stage bytes changed")
    probes = []
    for name in [*state["stage_files"], ".ks-stage-write-probe"]:
        flags = os.O_WRONLY | (os.O_CREAT | os.O_EXCL if name == ".ks-stage-write-probe" else 0)
        try:
            fd = os.open(root / name, flags, 0o600)
        except OSError as error:
            if error.errno != errno.EROFS:
                raise
            probes.append({"path": name, "write_open_denied": True, "errno": error.errno})
        else:
            os.close(fd)
            raise RuntimeError("stage write unexpectedly allowed")
    with tempfile.TemporaryFile(dir=state["scratch"]) as control:
        control.write(b"writable scratch"); control.flush()
    return {"profile": profile["channel"], "scope": "complete_evaluator_stage", "whole_root_readonly": False,
            "stage_files": state["stage_files"], "tool_sha256": actual_tools,
            "namespaces_before": state["namespaces_before"], "namespaces_after": after,
            "private_mounts": True, **proc, "caps": caps,
            "inherited_fds": inherited, "cwd": state["scratch"], "scratch": state["scratch"],
            "scratch_writable": True, "mount_aliases": [], "probes": probes}


def run(root, profile, action, timeout):
    """One bounded process group; unshare kills PID 1 if its parent dies."""
    tool_hashes(profile)
    files = snapshot(root)
    with contextlib.ExitStack() as cleanup, tempfile.TemporaryDirectory(prefix="ks-compile-", dir="/tmp") as directory:
        scratch = str(Path(directory).resolve())
        state = {"namespaces_before": namespaces(), "stage_files": files, "scratch": scratch}
        if profile["channel"] == INHERITED_PROFILE:
            # Keep real old-view canaries alive only in the trusted outer parent.
            # close_fds prevents inheritance; no process changes the stage bytes.
            directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            cleanup.callback(os.close, directory_fd)
            file_fd = os.open(root / "_worker.py", os.O_RDWR)
            cleanup.callback(os.close, file_fd)
            state.update(root=str(root), outer={"pid": int(os.readlink("/proc/self")),
                "directory_fd": directory_fd, "file_fd": file_fd, "cwd": str(Path.cwd())})
        argv = [profile["unshare"], "--user", "--map-root-user", "--mount", "--pid", "--net", "--uts", "--ipc",
                "--fork", "--kill-child=SIGKILL", "--propagation", "unchanged", profile["python"], "-I", "-B",
                str(root / "_stage_ro.py"), str(root), scratch, json.dumps(state, separators=(",", ":")), action]
        env = {**os.environ, "HOME": scratch, "TMPDIR": scratch, "TRITON_CACHE_DIR": scratch + "/triton",
               "XDG_CACHE_HOME": scratch + "/cache", "CUDA_CACHE_PATH": scratch + "/cuda"}
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            def limits():
                resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 * 1024,) * 2)
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            child = subprocess.Popen(argv, cwd=scratch, env=env, stdin=subprocess.DEVNULL, stdout=stdout,
                stderr=stderr, close_fds=True, start_new_session=True, preexec_fn=limits)
            try:
                code = child.wait(timeout=timeout)
            except BaseException:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
                raise
            stdout.seek(0); stderr.seek(0)
            if snapshot(root) != files:
                raise RuntimeError("outer stage bytes changed during isolated execution")
            return {"action": action, "profile": profile["channel"], "tool_sha256": profile["tool_sha256"], "exit_code": code,
                    "stdout": stdout.read().decode(), "stderr": stderr.read().decode()}


if __name__ == "__main__":
    inner(Path(sys.argv[1]), Path(sys.argv[2]), json.loads(sys.argv[3]), sys.argv[4])
