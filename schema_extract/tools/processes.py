"""Invocation-owned process identities, gated exec and confirmed termination."""

from __future__ import annotations

import errno
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from uuid import UUID

if __package__:
    from .common import ExtractError, digest, read_json, write_json
else:  # The gate is launched by absolute filename, including from frozen rules.
    from common import ExtractError, digest, read_json, write_json


GATE_PROTOCOL = "schema_extract.pipe_gate.v1"


def _boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip() or None
    except OSError:
        return None


def _linux_process(pid: int, boot_id: str) -> dict | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = raw[raw.rindex(")") + 2:].split()
        return {"pid": pid, "ppid": int(fields[1]), "pgid": int(fields[2]), "stat": fields[0],
                "start_token": f"linux:{boot_id}:{int(fields[19])}"}
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (IndexError, ValueError) as exc:
        raise OSError(f"cannot read process identity for {pid}") from exc


def process_table(*, cwd: Path | None = None) -> dict[int, dict]:
    # Keep ps lstart for persisted pre-token identities. Never inherit frozen
    # rules as the working directory of this diagnostic subprocess.
    output = subprocess.check_output(["ps", "-axo", "pid=,ppid=,pgid=,lstart=,stat="], text=True,
                                     cwd=cwd if cwd is not None else tempfile.gettempdir(), timeout=10)
    result, boot_id = {}, _boot_id()
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 9:
            pid = int(parts[0])
            item = {"pid": pid, "ppid": int(parts[1]), "pgid": int(parts[2]),
                    "started": " ".join(parts[3:8]), "stat": parts[8]}
            if boot_id:
                try:
                    native = _linux_process(pid, boot_id)
                except PermissionError:
                    native = None
                else:
                    if native is None:
                        continue  # It exited after ps; do not manufacture an unknown live row.
                if native is not None:
                    item.update(native)
            result[pid] = item
    return result


def process_identity_matches(identity: dict, current: dict | None) -> bool | None:
    """Match new boot/start tokens or old ps lstart; unavailable proof is None."""
    if (not isinstance(identity, dict) or type(identity.get("pid")) is not int or identity["pid"] <= 1 or
            not isinstance(identity.get("started"), str) or
            identity["started"] in {"", "unobserved", "already_exited"}):
        return None
    if current is None:
        return False
    if current.get("pid") != identity["pid"]:
        return False
    if identity.get("start_token"):
        if not current.get("start_token"):
            return None
        return identity["start_token"] == current["start_token"]
    return identity["started"] == current.get("started")


def process_alive(identity: dict, table: dict[int, dict] | None) -> bool | None:
    """Read-only identity/liveness check shared by transport and recovery."""
    if table is None or not isinstance(identity, dict):
        return None
    current = table.get(identity.get("pid"))
    matched = process_identity_matches(identity, current)
    return matched and not current["stat"].startswith("Z") if matched is not None else None


def identity_key(identity: dict) -> tuple:
    return identity["pid"], identity.get("start_token") or identity["started"]


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


def owned_processes(identity: dict, known=(), *, cwd: Path | None = None) -> list[dict]:
    return _owned_snapshot(identity, known, process_table(cwd=cwd))[0]


def _signal_process(identity: dict, sig: signal.Signals, *, cwd: Path | None) -> bool:
    descriptor = None
    try:
        if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
            try:
                descriptor = os.pidfd_open(identity["pid"])
            except OSError as exc:
                if exc.errno not in {errno.ENOSYS, errno.EINVAL}:
                    raise
        boot_id = _boot_id() if identity.get("start_token") else None
        current = (_linux_process(identity["pid"], boot_id) if boot_id else
                   process_table(cwd=cwd).get(identity["pid"]))
        matched = process_identity_matches(identity, current)
        if matched is None:
            raise OSError(f"cannot verify process {identity['pid']} before signalling")
        if not matched:
            return False
        if descriptor is not None:
            # The descriptor pins the target across the verification/signal gap.
            signal.pidfd_send_signal(descriptor, sig)
        else:
            os.kill(identity["pid"], sig)
        return True
    except ProcessLookupError:
        return False
    finally:
        if descriptor is not None:
            os.close(descriptor)


def terminate_owned(identity: dict, known=(), grace: float = 1.0, *, cwd: Path | None = None) -> dict:
    seen, actions = list(known), []

    def observe():
        nonlocal seen
        alive, unknown, reasons = _owned_snapshot(identity, seen, process_table(cwd=cwd))
        seen = list({identity_key(item): item for item in [*seen, *alive]}.values())
        return alive, unknown, reasons

    def result(alive, unknown, reasons):
        survivors = list({identity_key(item): item for item in [*alive, *unknown]}.values())
        value = {"confirmed": not survivors and not reasons, "actions": actions, "survivors": survivors}
        if reasons:
            value["reason"] = "; ".join(dict.fromkeys(reasons))
        return value

    for sig in (signal.SIGTERM, signal.SIGKILL):
        alive, unknown, reasons = observe()
        if not alive:
            return result(alive, unknown, reasons)
        for item in reversed(alive):
            if _signal_process(item, sig, cwd=cwd):
                actions.append({"pid": item["pid"], "signal": sig.name})
        until = time.monotonic() + max(0, grace)
        while True:
            # Recheck unknown groups on every observation, including after the
            # leader exits. An empty owned list alone never confirms termination.
            alive, unknown, reasons = observe()
            if not alive:
                return result(alive, unknown, reasons)
            if time.monotonic() >= until:
                break
            time.sleep(0.05)
    return result(alive, unknown, reasons)


def child_exited(process: subprocess.Popen) -> bool:
    """Observe exit without reaping the leader that still pins its process group."""
    if process.returncode is not None:
        return True
    if hasattr(os, "waitid") and hasattr(os, "WNOWAIT"):
        return os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
    # Older POSIX hosts retain conservative unknown-group fencing after poll.
    return process.poll() is not None


def mark_not_started(path: Path, binding: dict, reason: str) -> None:
    """Publish positive no-exec evidence only for this exact gate/attempt."""
    launch = read_json(path)
    keys = ("protocol", "launch_id", "attempt_id", "attempt_sha256")
    if any(launch.get(key) != binding.get(key) for key in keys):
        raise ExtractError("launch gate binding changed")
    write_json(path, {**launch, "status": "not_started", "reason": reason, "closed_at": time.time()})


def launch_not_started(attempt: dict, *, table: dict[int, dict] | None = None) -> bool:
    """Confirm this attempt's gate never exec'd; old/mismatched markers fence.

    `released` only means exec was authorized, never that it exited. `gated`
    is positive no-exec proof only after the creator has died and a second read
    confirms it did not persist a release before dying. A wrapper's explicit
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
        if process_alive(launch.get("creator"), process_table(cwd=directory)) is not False:
            return False
        return read_json(directory / "launch.json") == launch
    except (ExtractError, OSError, subprocess.SubprocessError, KeyError, TypeError, ValueError, AttributeError):
        return False


def _gate_main(argv: list[str]) -> int:
    if len(argv) < 8 or argv[0] != "--launch-gate":
        return 125
    descriptor = int(argv[1])
    path, launch_id, attempt_id, attempt_sha256, cwd = argv[2:7]
    binding = {"protocol": GATE_PROTOCOL, "launch_id": launch_id,
               "attempt_id": attempt_id, "attempt_sha256": attempt_sha256}
    try:
        permitted = os.read(descriptor, 1)
    finally:
        os.close(descriptor)
    if permitted != b"1":
        mark_not_started(Path(path), binding, "gate_closed")
        return 125
    try:
        os.chdir(cwd)
        os.execvpe(argv[7], argv[7:], os.environ)
    except OSError:
        mark_not_started(Path(path), binding, "exec_failed")
        raise


if __name__ == "__main__":
    raise SystemExit(_gate_main(sys.argv[1:]))
