"""Finite invocation allowances and explicit, journaled Supervisor additions."""

from __future__ import annotations

import time

from .common import ExtractError
from .stages import scope_for


CAPABILITY = "budget_extension.v1"
BUDGET_STOPS = {"budget_exhausted", "checker_budget_exhausted", "continuation_budget_exhausted"}
RECOVERY = "显式 resume {run_id} --add-seconds N 追加时间，或用 --add-calls N 追加调用；按所缺额度选择 N，已用计数和一次性决定不重置。"


def addition(seconds=0, calls=0) -> dict:
    if any(type(value) is not int or value < 0 for value in (seconds, calls)):
        raise ExtractError("budget additions must be nonnegative integers")
    return {"seconds": seconds, "calls": calls}


def call_limit(state: dict) -> int:
    return state["config"]["max_calls"] + sum(item["calls"] for item in state.get("budget_extensions", []))


def review_seconds(config: dict) -> float:
    ordinary = min(config["stage_seconds"], config["operator_seconds"]) - config["adjudication_seconds"]
    return min(config["call_seconds"], max(0.0, ordinary / 2))


def reserved_calls(state: dict) -> int:
    mode, scope = state.get("mode", "normal"), scope_for(state)
    adjudication = int(not scope["adjudications"])
    signoff = int(state["stage"] == "P6" and not scope["signoffs"])
    review = int(state.get("role") == "worker")
    return adjudication + signoff + review if mode == "normal" else signoff if mode == "adjudicate" else 0


def _reserve_seconds(state: dict) -> float:
    review = int(state.get("mode", "normal") == "normal" and state.get("role") == "worker")
    return ((reserved_calls(state) - review) * state["config"]["adjudication_seconds"]
            + review * review_seconds(state["config"]))


def _remaining(state: dict, now: float) -> float:
    return min(state["deadline"], scope_for(state)["deadline"] or now + state["config"]["stage_seconds"]) - now


def call_budget(state: dict) -> tuple[float, str | None]:
    scope, config, mode = scope_for(state), state["config"], state.get("mode", "normal")
    if mode in {"adjudicate", "signoff"} and scope["adjudications" if mode == "adjudicate" else "signoffs"]:
        return 0, "本范围的一次裁决/签收已使用；追加预算或重开不能重置该决定。"
    if scope["calls"] >= call_limit(state) - reserved_calls(state):
        return 0, "本阶段/节点调用余额不足；后续审阅、裁决与签收的调用仍保留。"
    remaining = _remaining(state, time.time()) - _reserve_seconds(state)
    if remaining <= 0:
        return 0, "算子或本阶段/节点时间余额不足；后续审阅、裁决与签收的时间仍保留。"
    return min(config["call_seconds"] if mode == "normal" else config["adjudication_seconds"], remaining), None


def check_timeout(state: dict) -> float:
    return max(0.0, min(120.0, _remaining(state, time.time()) - _reserve_seconds(state)))


def extend(state: dict, request: dict, *, identity: str, origin: str) -> bool:
    """Apply a persisted launch/continuation request once; never modify config or usage."""
    request = addition(request.get("seconds", 0), request.get("calls", 0))
    if not any(request.values()):
        return False
    previous = next((item for item in state.get("budget_extensions", []) if item["id"] == identity), None)
    if previous:
        if any(previous[key] != value for key, value in request.items()):
            raise ExtractError("budget authorization identity was reused with different amounts")
        return False
    now = time.time()
    record = {"id": identity, "origin": origin, "run_id": state["run_id"], "at": now, **request,
              "previous_deadline": state["deadline"], "scope_deadlines": {}}
    seconds = request["seconds"]
    if seconds:
        state["deadline"] = max(now, state["deadline"]) + seconds
        scopes = {**state["stages"], **{"P5:" + key: node["stage"] for key, node in state["nodes"].items() if "stage" in node}}
        for key, scope in scopes.items():
            if scope["deadline"] is not None:
                previous_deadline = scope["deadline"]
                scope["deadline"] = max(now, previous_deadline) + seconds
                record["scope_deadlines"][key] = {"before": previous_deadline, "after": scope["deadline"]}
    record["deadline"] = state["deadline"]
    state.setdefault("budget_extensions", []).append(record)
    return True
