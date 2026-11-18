"""Invocation-owned process identities, gated exec and confirmed termination."""

from __future__ import annotations

import errno
import ctypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import UUID

if __package__:
    from .common import ExtractError, digest, read_json, write_json
else:  # The gate is launched by absolute filename, including from frozen rules.
    from common import ExtractError, digest, read_json, write_json


GATE_PROTOCOL = "schema_extract.pipe_gate.v1"


def _boot_id() -> str:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except UnicodeError as exc:
        raise OSError("cannot read Linux boot identity") from exc
    if not value:
        raise OSError("cannot establish Linux boot identity")
    return value


def _linux_process(pid: int, boot_id: str) -> dict | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = raw[raw.rindex(")") + 2:].split()
        return {"pid": pid, "ppid": int(fields[1]), "pgid": int(fields[2]), "stat": fields[0],
                "start_token": f"linux:{boot_id}:{int(fields[19])}"}
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (IndexError, ValueError, UnicodeError) as exc:
        raise OSError(f"cannot read process identity for {pid}") from exc


def process_identity(pid: int) -> dict | None:
    """Read one Linux boot/start identity; only an absent PID returns None."""
    if type(pid) is not int or pid < 1:
        raise ValueError("process PID must be a positive integer")
    return _linux_process(pid, _boot_id())


def process_table() -> dict[int, dict]:
    """Take a complete observable /proc snapshot or report observation failure."""
    result, boot_id = {}, _boot_id()
    for path in Path("/proc").iterdir():
        if path.name.isdecimal():
            item = _linux_process(int(path.name), boot_id)
            if item is not None:
                result[item["pid"]] = item
    return result


def process_identity_matches(identity: dict, current: dict | None) -> bool | None:
    """Match Linux boot/start identities; unavailable proof is None."""
    if (not isinstance(identity, dict) or type(identity.get("pid")) is not int or identity["pid"] <= 1 or
            not isinstance(identity.get("start_token"), str) or not identity["start_token"].startswith("linux:")):
        return None
    if current is None:
        return False
    if current.get("pid") != identity["pid"]:
        return False
    if not current.get("start_token"):
        return None
    return identity["start_token"] == current["start_token"]


def process_alive(identity: dict, table: dict[int, dict] | None) -> bool | None:
    """Read-only identity/liveness check shared by transport and recovery."""
    if table is None or not isinstance(identity, dict):
        return None
    current = table.get(identity.get("pid"))
    matched = process_identity_matches(identity, current)
    return matched and not current["stat"].startswith("Z") if matched is not None else None


def identity_key(identity: dict) -> tuple:
    return identity["pid"], identity.get("start_token")


def _owned_snapshot(identity: dict, known, table: dict[int, dict]) -> tuple[list[dict], list[dict], list[str]]:
    selected, unknown, reasons = set(), {}, []
    for expected in (identity, *known):
        current = table.get(expected.get("pid"))
        matched = process_identity_matches(expected, current)
        if matched:
            # Zombies still pin their identities/groups until their parent reaps.
            selected.add(expected["pid"])
        elif matched is None:
            reasons.append("process identity cannot be verified")
            if current and not current["stat"].startswith("Z"):
                unknown[current["pid"]] = current
    group = {pid for pid, item in table.items() if item["pgid"] == identity["pid"]}
    # An escaped descendant is not an ownership anchor for a reused original
    # group. At least one matching identity must still belong to that group.
    if selected & group:
        selected.update(group)
    else:
        unowned_group = {pid: table[pid] for pid in group if not table[pid]["stat"].startswith("Z")}
        if unowned_group:
            unknown.update(unowned_group)
            reasons.append("invocation group exists but ownership cannot be confirmed")
    while True:
        children = {pid for pid, item in table.items() if item["ppid"] in selected}
        if children <= selected:
            break
        selected.update(children)
    alive = [table[pid] for pid in sorted(selected) if not table[pid]["stat"].startswith("Z")]
    return alive, [unknown[pid] for pid in sorted(unknown)], reasons


def owned_processes(identity: dict, known=(), *, table: dict[int, dict] | None = None) -> list[dict]:
    return _owned_snapshot(identity, known, process_table() if table is None else table)[0]


def signal_process(identity: dict, sig: signal.Signals = signal.SIGKILL) -> bool:
    """Signal one verified identity; True records delivery, never exit."""
    descriptor = None
    try:
        if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
            try:
                descriptor = os.pidfd_open(identity["pid"])
            except OSError as exc:
                if exc.errno not in {errno.ENOSYS, errno.EINVAL}:
                    raise
        matched = process_identity_matches(identity, process_identity(identity["pid"]))
        if matched is None:
            raise OSError(f"cannot verify process {identity['pid']} before signalling")
        if not matched:
            return False
        if descriptor is not None:
            signal.pidfd_send_signal(descriptor, sig)
        else:
            os.kill(identity["pid"], sig)
        return True
    except ProcessLookupError:
        return False
    finally:
        if descriptor is not None:
            os.close(descriptor)


def observe_owned(identity: dict, known=(), *, table: dict[int, dict] | None = None) -> dict:
    """Observe owned liveness without signalling or waiting for exit."""
    alive, unknown, reasons = _owned_snapshot(identity, known, process_table() if table is None else table)
    descendants = list({identity_key(item): item for item in [*known, *alive]}.values())
    survivors = list({identity_key(item): item for item in [*alive, *unknown]}.values())
    result = {"confirmed": not survivors and not reasons, "actions": [], "survivors": survivors,
              "descendants": descendants}
    if reasons:
        result["reason"] = "; ".join(dict.fromkeys(reasons))
    return result


def terminate_owned(identity: dict, known=(), *, include_anchor: bool = False) -> dict:
    """Send one round of verified KILLs, then observe; invocation gates stay alive."""
    table, actions, errors = process_table(), [], []
    alive, _, _ = _owned_snapshot(identity, known, table)
    seen = list({identity_key(item): item for item in [*known, *alive]}.values())
    for item in reversed(alive):
        if item["pid"] == identity["pid"] and not include_anchor:
            continue
        try:
            if signal_process(item):
                actions.append({"pid": item["pid"], "start_token": item["start_token"], "signal": "SIGKILL"})
        except OSError as exc:
            errors.append(str(exc))
    try:
        result = observe_owned(identity, seen)
    except OSError as exc:
        result = {"confirmed": False, "survivors": alive, "descendants": seen, "reason": str(exc)}
    result["actions"] = actions
    if errors:
        result.update(confirmed=False, reason="; ".join([*errors, *([result["reason"]] if result.get("reason") else [])]))
    return result


def child_exited(process: subprocess.Popen) -> bool:
    """Observe exit without reaping the leader that still pins its process group."""
    if process.returncode is not None:
        return True
    return os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def mark_not_started(path: Path, binding: dict, reason: str) -> None:
    """Publish positive no-exec evidence only for this exact gate/attempt."""
    launch = read_json(path)
    keys = ("protocol", "launch_id", "attempt_id", "attempt_sha256")
    if any(launch.get(key) != binding.get(key) for key in keys):
        raise ExtractError("launch gate binding changed")
    write_json(path, {**launch, "status": "not_started", "reason": reason, "closed_at": time.time()})


def launch_not_started(attempt: dict, *, table: dict[int, dict] | None = None) -> bool:
    """Confirm no model launch and no remaining setup child for this attempt.

    `released` only means exec was authorized, never that it exited. `gated`
    is positive no-exec proof only after the creator has died and a second read
    confirms it did not authorize setup before dying. A wrapper's explicit
    `not_started` marker also covers death between release persistence and send.
    """
    directory = Path(attempt["directory"])
    try:
        launch, invocation = (read_json(directory / name) for name in ("launch.json", "invocation.json"))
        if (launch.get("protocol") != GATE_PROTOCOL or
                str(UUID(launch["launch_id"])) != launch["launch_id"] or
                any(record.get("attempt_id") != attempt["attempt_id"] or
                    record.get("run_id") != attempt.get("run_id") or
                    record.get("attempt_sha256") != digest(attempt) for record in (launch, invocation)) or
                invocation.get("launch_id") != launch["launch_id"]):
            return False
        if (directory / "process.json").exists():
            process = read_json(directory / "process.json")
            if process.get("launch_id") != launch["launch_id"]:
                return False
        if launch.get("status") == "not_started":
            return launch.get("reason") in {"gate_closed", "exec_failed", "parent_aborted", "spawn_failed"}
        if launch.get("status") != "gated":
            return False
        if table is not None and process_alive(launch.get("creator"), table) is not False:
            return False
        # A caller's snapshot may predate the creator. Confirm death with a
        # fresh observation taken after reading the gated marker.
        if process_alive(launch.get("creator"), process_table()) is not False:
            return False
        return read_json(directory / "launch.json") == launch
    except (OSError, KeyError, TypeError, ValueError, AttributeError):
        return False


def gate_reaped(attempt: dict) -> bool:
    """Require the gate to have actually waited for every adopted child."""
    directory = Path(attempt["directory"])
    try:
        ended, launch = read_json(directory / "cli-exit.json"), read_json(directory / "launch.json")
        return (ended.get("descendants_reaped") is True and type(ended.get("exit_code")) is int and
                ended.get("protocol") == launch.get("protocol") == GATE_PROTOCOL and
                ended.get("attempt_id") == attempt["attempt_id"] and ended.get("attempt_sha256") == digest(attempt) and
                ended.get("launch_id") == launch.get("launch_id"))
    except (ExtractError, KeyError, TypeError):
        return False


def _gate_main(argv: list[str]) -> int:
    if len(argv) < 8 or argv[0] != "--launch-gate":
        return 125
    descriptor = int(argv[1])
    path, launch_id, attempt_id, attempt_sha256, cwd = argv[2:7]
    directory = Path(path).parent
    binding = {"protocol": GATE_PROTOCOL, "launch_id": launch_id,
               "attempt_id": attempt_id, "attempt_sha256": attempt_sha256}
    model_started = False
    try:
        if os.read(descriptor, 1) != b"S":
            mark_not_started(Path(path), binding, "gate_closed")
            return 125
        os.chdir(cwd)
        # Keep double-forked/setsid descendants under this invocation's gate.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            raise OSError(ctypes.get_errno(), "cannot establish invocation child ownership")
        version_path = directory / "cli-tmp/version.txt"
        with version_path.open("wb") as output, (directory / "cli-tmp/version-stderr.txt").open("wb") as errors:
            version = subprocess.Popen([argv[7], "--version"], stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=errors, close_fds=True)
        version_code = version.wait()
        while True:
            try:
                os.wait()
            except ChildProcessError:
                break
        write_json(directory / "cli-tmp/version.json", {**binding, "exit_code": version_code,
                   "cli_version": version_path.read_text(encoding="utf-8").strip()})
        if os.read(descriptor, 1) != b"1":
            mark_not_started(Path(path), binding, "gate_closed")
            return 125
        child = subprocess.Popen(argv[7:], close_fds=True)
        model_started = True
        code = child.wait()
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    write_json(directory / "cli-exit.json", {**binding, "exit_code": code, "descendants_reaped": False})
                    os.wait()
            except ChildProcessError:
                write_json(directory / "cli-exit.json", {**binding, "exit_code": code, "descendants_reaped": True})
                return code if code >= 0 else 128 - code
    except (OSError, UnicodeError) as exc:
        if not model_started:
            # A failed setup still owns every child until wait confirms closure.
            while True:
                try:
                    os.wait()
                except ChildProcessError:
                    break
            mark_not_started(Path(path), binding, "exec_failed")
            print(str(exc), file=sys.stderr)
            return 125
        raise
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(_gate_main(sys.argv[1:]))
