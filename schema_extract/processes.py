"""Invocation-owned process identities and confirmed termination, without argv/env capture."""

from __future__ import annotations

import os
import signal
import subprocess
import time


def process_table() -> dict[int, dict]:
    output = subprocess.check_output(["ps", "-axo", "pid=,ppid=,pgid=,lstart=,stat="], text=True)
    result = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 9:
            result[int(parts[0])] = {"pid": int(parts[0]), "ppid": int(parts[1]), "pgid": int(parts[2]),
                                     "started": " ".join(parts[3:8]), "stat": parts[8]}
    return result


def owned_processes(identity: dict, known=()) -> list[dict]:
    table = process_table()
    current = table.get(identity["pid"])
    matched = current and current["started"] == identity["started"]
    selected = {item["pid"] for item in known if item["pid"] in table and
                table[item["pid"]]["started"] == item["started"]}
    if matched:
        selected.add(identity["pid"])
    if selected:
        selected.update(pid for pid, item in table.items() if item["pgid"] == identity["pid"])
    while True:
        children = {pid for pid, item in table.items() if item["ppid"] in selected}
        if children <= selected:
            break
        selected.update(children)
    return [table[pid] for pid in sorted(selected) if not table[pid]["stat"].startswith("Z")]


def terminate_owned(identity: dict, known=(), grace: float = 1.0) -> dict:
    seen, actions = list(known), []
    table = process_table()
    current = table.get(identity["pid"])
    matched = current and current["started"] == identity["started"]
    matched = matched or any(item["pid"] in table and table[item["pid"]]["started"] == item["started"] for item in known)
    group = [item for item in table.values() if item["pgid"] == identity["pid"] and not item["stat"].startswith("Z")]
    if group and not matched:
        return {"confirmed": False, "actions": [], "survivors": group,
                "reason": "invocation group exists but ownership cannot be confirmed"}
    for sig in (signal.SIGTERM, signal.SIGKILL):
        alive = owned_processes(identity, seen)
        seen = list({(item["pid"], item["started"]): item for item in [*seen, *alive]}.values())
        for item in reversed(alive):
            try:
                os.kill(item["pid"], sig)
                actions.append({"pid": item["pid"], "signal": sig.name})
            except ProcessLookupError:
                pass
        until = time.monotonic() + grace
        while time.monotonic() < until and owned_processes(identity, seen):
            time.sleep(0.05)
        if not owned_processes(identity, seen):
            return {"confirmed": True, "actions": actions, "survivors": []}
    return {"confirmed": False, "actions": actions, "survivors": owned_processes(identity, seen)}
