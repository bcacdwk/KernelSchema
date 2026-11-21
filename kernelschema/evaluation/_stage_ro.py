"""Fixed util-linux launcher for one complete row-softmax evaluator stage.

This protects stage files, not the whole root or Python's address space. Only
the private mount namespace changes: no installation or host chmod/remount.
The same file is frozen with the Task and used by the remote and worker.
"""
import errno
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


def namespaces():
    return {name: os.readlink("/proc/self/ns/" + name) for name in NAMESPACES}


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
                     "options": fields[5].split(","), "propagation": fields[6:split], "type": fields[split + 1]})
    return rows


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
    independent(state["namespaces_before"])
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
    mount("-t", "proc", "-o", "nosuid,nodev,noexec", "proc", "/proc")
    if os.readlink("/proc/self") != "1" or namespaces()["pid"] != os.readlink("/proc/1/ns/pid"):
        raise RuntimeError("fresh proc does not belong to the new PID namespace")
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
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())
    caps = {name: status[name].strip() for name in CAPS + ("NoNewPrivs",)}
    if any(int(caps[name], 16) for name in CAPS) or caps["NoNewPrivs"] != "1":
        raise RuntimeError("capabilities/NNP not restricted")
    if os.getpid() != 1 or os.readlink("/proc/self") != "1" or os.readlink("/proc/1/ns/pid") != after["pid"]:
        raise RuntimeError("fresh PID namespace proc absent")
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
    return {"profile": PROFILE, "scope": "complete_evaluator_stage", "whole_root_readonly": False,
            "stage_files": state["stage_files"], "tool_sha256": actual_tools,
            "namespaces_before": state["namespaces_before"], "namespaces_after": after,
            "private_mounts": True, "fresh_proc_pid": 1, "caps": caps,
            "inherited_fds": inherited, "cwd": state["scratch"], "scratch": state["scratch"],
            "scratch_writable": True, "mount_aliases": [], "probes": probes}


def run(root, profile, action, timeout):
    """One bounded process group; unshare kills PID 1 if its parent dies."""
    tool_hashes(profile)
    files = snapshot(root)
    with tempfile.TemporaryDirectory(prefix="ks-compile-", dir="/tmp") as directory:
        scratch = str(Path(directory).resolve())
        state = {"namespaces_before": namespaces(), "stage_files": files, "scratch": scratch}
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
            return {"action": action, "profile": PROFILE, "tool_sha256": profile["tool_sha256"], "exit_code": code,
                    "stdout": stdout.read().decode(), "stderr": stderr.read().decode()}


if __name__ == "__main__":
    inner(Path(sys.argv[1]), Path(sys.argv[2]), json.loads(sys.argv[3]), sys.argv[4])
