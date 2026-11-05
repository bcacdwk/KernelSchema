"""Invocation-boundary semantic requests in the existing supervisor ledger.

Requests fence dispatch and delivery as soon as their control transaction is
committed. Only the run writer projects a request into state; healthy calls keep
their registered context until their normal accounting boundary.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
from uuid import UUID, uuid4

from .common import ExtractError, RunnerBusy, digest, read_json, write_json


CAPABILITY = "semantic_revision.v1"
PENDING = frozenset({"requested", "draining", "drained", "prepared"})


def _request(ledger, run_id, request_id=None):
    value = ledger.get("semantic_requests", {}).get(run_id)
    if request_id is not None and (not value or value["id"] != request_id):
        raise ExtractError("semantic request is missing or has been superseded")
    return value


def revision_barrier(store, state, ledger=None):
    ledger = store.read_supervisor() if ledger is None else ledger
    request = _request(ledger, state["run_id"])
    return bool(request and request.get("status") in PENDING)


def maintenance_active(state):
    maintenance = state.get("maintenance") or {}
    return bool(maintenance and maintenance.get("status") not in {"completed", "complete", "cancelled"})


def publication_barrier(store, state, ledger=None):
    return revision_barrier(store, state, ledger) or maintenance_active(state)


def _unsettled(state):
    return any(state.get(name) for name in ("active_attempts", "unconfirmed_attempts", "unsettled_attempts"))


def _eligible(state, ledger):
    if (state.get("published") or state["status"] == "ready" or state.get("exports") or
            ledger.get("commits", {}).get(state["run_id"])):
        raise ExtractError("semantic revision only accepts unfinished runs without export or committed publication")
    if state["run_id"] in ledger.get("continuations", {}):
        raise ExtractError("this run already has a reserved full-restart successor")
    if maintenance_active(state) and not (state.get("status") in {"paused", "cancelled"} and not _unsettled(state)):
        raise ExtractError("finish or safely pause the active semantic maintenance before requesting another revision")


def _target(request):
    from .snapshots import verify_snapshot
    lock = read_json(Path(request["target_snapshot"]) / "inputs.lock.json")
    if lock["fingerprint"] != request["target_fingerprint"] or lock["rules"]["id"] != request["target_rules_id"]:
        raise ExtractError("semantic request target differs from its frozen snapshot")
    verify_snapshot(lock)
    return lock


def request_revision(store, run_id, *, reason, target_repo=None, request_id=None):
    """Freeze the intention, without binding it to a still-moving frontier."""
    from .revisions import freeze_target
    if not isinstance(reason, str) or not reason.strip():
        raise ExtractError("a semantic revision requires a nonempty reason")
    if request_id is not None:
        try:
            if str(UUID(request_id)) != request_id:
                raise ValueError()
        except (ValueError, TypeError, AttributeError) as exc:
            raise ExtractError("request ID must be a canonical UUID") from exc
    state = store.load(run_id)
    # An old active executor cannot observe this new fence. Installing a
    # reviewed capable runtime at its existing safe boundary comes first.
    if CAPABILITY not in state.get("executor", {}).get("capabilities", []):
        raise ExtractError("frozen executor lacks semantic_revision.v1; settle it and adopt a reviewed capable runtime with upgrade-runtime first")
    target = freeze_target(store, state, target_repo)
    with store.supervisor() as ledger:
        current = store.load(run_id)
        prior = _request(ledger, run_id)
        history = ([*prior.get("history", []), {key: value for key, value in prior.items() if key != "history"}]
                   if prior else [])
        same = next((item for item in reversed(history) if
                     item.get("reason") == reason.strip() and item.get("target_fingerprint") == target["fingerprint"] and
                     (request_id is None or item["id"] == request_id)), None)
        if same and (same["id"] == (prior or {}).get("id") or request_id is not None):
            return {"status": same["status"], "request": deepcopy(same), "reused": True, "delivered": False}
        if request_id and any(item["id"] == request_id for item in history):
            raise ExtractError("request ID is already bound to a different target or reason")
        if prior and prior["status"] in PENDING:
            raise ExtractError("another semantic revision request is already pending")
        _eligible(current, ledger)
        if current.get("semantic_revision") != state.get("semantic_revision"):
            raise ExtractError("semantic context changed while freezing the target; retry the request")
        start = ledger["starts"].get(run_id, {})
        latest = ledger["controls"].get(run_id, {})
        body = {"run_id": run_id, "reason": reason.strip(), "source_semantic_revision": current.get("semantic_revision"),
                "target_fingerprint": target["fingerprint"], "target_rules_id": target["rules"]["id"]}
        request = {**body, "id": request_id or str(uuid4()), "status": "requested", "created_at": time.time(),
                   "eligible_at_request": True, "target_snapshot": target["snapshot"], "request_sha256": digest(body),
                   "prior_status": current["status"], "control_revision": latest.get("revision", 0),
                   "request_ids": list(current.get("control", {}).get("requests", [])),
                   "run_intent": bool(current["status"] == "running" or
                                      current["status"] == "queued" and start.get("authorized")),
                   "history": history}
        ledger.setdefault("semantic_requests", {})[run_id] = request
        return {"status": "requested", "request": deepcopy(request), "reused": False, "delivered": False}


def consume_revision_requests(store, state):
    """Called by the existing run writer after ordinary controls are consumed."""
    with store.supervisor() as ledger:
        request = _request(ledger, state["run_id"])
        if not request or request["status"] not in PENDING:
            return state
        if request["status"] == "requested":
            request.update(status="draining", observed_at=time.time())
        projection = {key: request[key] for key in
                      ("id", "status", "created_at", "reason", "target_fingerprint", "source_semantic_revision")}
    if state.get("revision_request") != projection:
        state["revision_request"] = projection
        store.save(state)
    return state


def mark_revision_drained(store, state):
    """The caller also proves its in-memory running/pending/deferred sets empty."""
    if _unsettled(state):
        raise ExtractError("semantic drain still has active, unsettled or unknown-termination calls")
    with store.supervisor() as ledger:
        request = _request(ledger, state["run_id"])
        if not request or request["status"] not in PENDING:
            return state
        if request["status"] in {"requested", "draining"}:
            request.update(status="drained", drained_at=time.time())
        projection = {key: request[key] for key in
                      ("id", "status", "created_at", "reason", "target_fingerprint", "source_semantic_revision")}
    if state.get("revision_request") != projection:
        state["revision_request"] = projection
        store.save(state)
    return state


def revision_status(store, run_id):
    state = store.load(run_id)
    ledger = store.read_supervisor()
    request = _request(ledger, run_id)
    return {"run_id": run_id, "status": request["status"] if request else "no_request",
            "request": deepcopy(request), "semantic_revision": state.get("semantic_revision"),
            "maintenance": deepcopy(state.get("maintenance")), "run_status": state["status"],
            "dispatch_blocked": revision_barrier(store, state, ledger),
            "publication_blocked": publication_barrier(store, state, ledger),
            "unsettled": _unsettled(state), "delivered": bool(state.get("published"))}


def _drain_progress(store, state, ledger, repo):
    """A dead old writer may be replaced only to settle already-started work."""
    from .monitor import _alive, _processes, start_background
    result = {**revision_status(store, state["run_id"]), "status": "draining"}
    launch = ledger["starts"].get(state["run_id"], {}).get("launch", {})
    identities = []
    if launch.get("status") in {"launching", "claimed", "termination_unknown"}:
        identities.append(launch.get("identity") or launch.get("launcher"))
    if state.get("runner"):
        identities.append(state["runner"].get("identity"))
    table, error = _processes()
    living = [_alive(identity, table) for identity in identities]
    result["fence"] = {"writer": store.writer_status(state["run_id"]), "old_owners": living,
                       "process_observation_error": error,
                       "active_attempts": deepcopy(state.get("active_attempts", {})),
                       "unconfirmed_attempts": deepcopy(state.get("unconfirmed_attempts", {})),
                       "unsettled_attempts": deepcopy(state.get("unsettled_attempts", {}))}
    request = _request(ledger, state["run_id"])
    latest = ledger["controls"].get(state["run_id"], {})
    controls = store.control_requests(state["run_id"])
    if (repo is not None and request and request.get("run_intent") and _unsettled(state) and
            result["fence"]["writer"] in {"free", "absent"} and all(value is False for value in living) and
            not error and not ledger["active_hold"] and state["status"] not in {"paused", "cancelled"} and
            not state.get("control", {}).get("action") and latest.get("action") not in {"pause", "cancel"} and
            not any(item["id"] not in state["control"]["requests"] for item in controls)):
        result["launch"] = start_background(store, state["run_id"], repo, mode="revision_drain")
    result["next_step"] = "settle existing calls under their original context and deadlines; prepare again after confirmed drain"
    return result


def prepare_revision(store, run_id, *, proposal=None, request_id=None, repo=None):
    """Bind scope identities only after drain; repeated preparation can refresh."""
    from .revisions import normalize_plan
    from .runtime import _quiescent
    observed = store.load(run_id)
    observed_ledger = store.read_supervisor()
    request = _request(observed_ledger, run_id, request_id)
    if request and request["status"] in PENDING and (_unsettled(observed) or store.writer_status(run_id) == "held"):
        return _drain_progress(store, observed, observed_ledger, repo)
    try:
        with store.writer(run_id):
            state = store.load(run_id)
            ledger = store.read_supervisor()
            request = _request(ledger, run_id, request_id)
            if not request:
                raise ExtractError("register a semantic revision request before preparing a plan")
            if request["status"] == "applied":
                return {**revision_status(store, run_id), "reused": True}
            try:
                _quiescent(store, state, ledger)
            except ExtractError as exc:
                if _unsettled(state) or "live or unconfirmed" in str(exc):
                    # Already holding the writer: report the fence here. A
                    # subsequent prepare may launch reconciliation after exit.
                    return {**revision_status(store, run_id), "status": "draining", "reason": str(exc)}
                raise
            state = mark_revision_drained(store, state)
            target = _target(request)
            if isinstance(proposal, dict) and "plan" in proposal:
                proposal = proposal.get("proposal") or proposal["plan"].get("proposal")
            elif isinstance(proposal, dict) and proposal.get("format") == "schema_extract.revision_plan.v1":
                proposal = proposal.get("proposal")
            plan = normalize_plan(store, state, target, proposal)
            assessed = bool(proposal is not None and plan.get("reviewed"))
            wrapper = {"format": "schema_extract.revision_preparation.v1", "run_id": run_id,
                       "request_id": request["id"], "target_fingerprint": target["fingerprint"],
                       "proposal": deepcopy(proposal), "plan": plan}
            identity = digest(wrapper)
            path = store.run_dir(run_id) / "revision_requests" / request["id"] / (identity + ".json")
            if path.exists():
                if read_json(path) != wrapper:
                    raise ExtractError("prepared revision content changed")
            else:
                write_json(path, wrapper)
            with store.supervisor() as ledger:
                current = _request(ledger, run_id, request["id"])
                current.update(status="prepared" if assessed else "drained",
                               plan_status="prepared" if assessed else "awaiting_evaluation",
                               plan_path=str(path), plan_sha256=identity, prepared_at=time.time())
            return {"run_id": run_id, "request_id": request["id"],
                    "status": "prepared" if assessed else "awaiting_evaluation",
                    "plan": plan, "plan_path": str(path), "delivered": False}
    except RunnerBusy:
        return {**revision_status(store, run_id), "status": "draining",
                "next_step": "finish the registered calls and their accounting, then prepare again"}


def _resume_applied(store, run_id, request_id, repo):
    from .monitor import start_background
    state = store.load(run_id)
    requests = store.control_requests(run_id)
    with store.supervisor() as ledger:
        request = _request(ledger, run_id, request_id)
        latest = ledger["controls"].get(run_id, {})
        if request.get("launch"):
            return deepcopy(request["launch"])
        if (not request.get("run_intent") or ledger["active_hold"] or
                state["status"] in {"paused", "cancelled", "ready"} or
                state.get("control", {}).get("action") or latest.get("action") in {"pause", "cancel"} or
                latest.get("revision", 0) != request["control_revision"] or
                any(item["id"] not in request["request_ids"] for item in requests)):
            return {"status": "preserved", "reason": "run intent or a newer pause/cancel/hold requires preserving control"}
    result = start_background(store, run_id, repo, mode="revision")
    with store.supervisor() as ledger:
        _request(ledger, run_id, request_id)["launch"] = deepcopy(result)
    return result


def apply_prepared_revision(store, run_id, *, request_id=None, repo=None):
    from .revisions import apply_revision
    request = _request(store.read_supervisor(), run_id, request_id)
    if not request or not request.get("plan_path"):
        raise ExtractError("prepare an independently evaluated revision plan before applying it")
    wrapper = read_json(Path(request["plan_path"]))
    if digest(wrapper) != request["plan_sha256"] or wrapper["request_id"] != request["id"]:
        raise ExtractError("prepared revision differs from its registered plan")
    if not wrapper.get("proposal") or not wrapper["plan"].get("reviewed"):
        return {"run_id": run_id, "status": "awaiting_evaluation", "plan": wrapper["plan"],
                "plan_path": request["plan_path"], "delivered": False}
    result = apply_revision(store, run_id, wrapper["plan"], _target(request), request_id=request["id"])
    # The sealed record/state pointer is authority. A crash before this control
    # acknowledgement is recovered by apply_revision's request-ID idempotence.
    with store.supervisor() as ledger:
        current = _request(ledger, run_id, request["id"])
        current.update(status="applied", semantic_revision=result["revision"], applied_at=current.get("applied_at", time.time()))
    result.update(status="applied", request_id=request["id"], delivered=False)
    if repo is not None:
        result["launch"] = _resume_applied(store, run_id, request["id"], repo)
    return result
