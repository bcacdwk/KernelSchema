"""One durable owner for serial workflow requests, including external callers.

An account descriptor is not a balance. Parent requests must explicitly connect
to the existing private authority; neither a missing file nor a new Q mints one.
All counters are cumulative per operation, and all active time has one clock.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
import fcntl
import os
from pathlib import Path
import time
from uuid import uuid4

from .artifacts import Publication, durable_write, read_regular
from .contracts import PROTOCOL, ROLES, ProtocolError, digest, jcs, loads, limits, validate

_authority = ContextVar("ks_budget_authority", default=None)
_METRICS = ("calls", "evaluator_jobs", "tokens", "gpu_ms")


class BudgetDenied(ProtocolError):
    """A new charge cannot be admitted under the original limits."""


@contextmanager
def use_authority(path):
    """Explicit private connection; external processes use KS_BUDGET_AUTHORITY."""
    token = _authority.set(str(Path(path).absolute()))
    try:
        yield
    finally:
        _authority.reset(token)


def _now():
    return time.time_ns() // 1_000_000


def _boot():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def _process_identity():
    fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": os.getpid(), "start_ticks": fields[19], "boot_id": _boot()}


def _deadline(value):
    return None if value is None else int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def _flatten(value):
    return {**{r: value["calls"][r] for r in ROLES},
            **{k: value.get(k) for k in ("evaluator_jobs", "tokens", "model_calls", "active_ms")}}


def cost(*, role=None, calls=0, evaluator_jobs=0, tokens=0, active_ms=0):
    """A reservation/floor vector, independent of public cumulative receipts."""
    if role is not None and role not in (*ROLES, "evaluator"):
        raise ProtocolError("invalid budget role")
    values = {r: calls if r == role else 0 for r in ROLES}
    values.update(evaluator_jobs=evaluator_jobs, tokens=tokens, active_ms=active_ms,
                  model_calls=calls if role in ROLES else 0)
    if any(type(n) is not int or n < 0 for n in values.values()):
        raise ProtocolError("reservation quantities must be nonnegative integers")
    return values


class OwnerAccount:
    VERSION = "ks.owner.v1"

    def __init__(self, path, *, clock=_now):
        self.path = Path(path).absolute()
        self.clock = clock
        # Opening is deliberately read-only with respect to missing authority.
        self._read()

    @classmethod
    def create(cls, path, *, owner, limits, account_id=None, deadline_utc=None, clock=_now):
        descriptor = validate({"protocol": PROTOCOL, "kind": "budget_account", "id": account_id or uuid4().hex,
                               "owner": owner, "limits": limits, "deadline_utc": deadline_utc})
        durable_write(Path(path).absolute(), jcs({"version": cls.VERSION, "descriptor": descriptor,
            "requests": {}, "operations": {}, "paused": False, "active_ms": 0, "active_ns": "0",
            "active_run": None, "active_since": None, "auto_activity": False, "last_clock": clock(),
            "clock_boot_id": _boot() if clock is _now else "injected",
            "monotonic_ns": str(time.monotonic_ns()) if clock is _now else None}), mode=0o600)
        return cls(path, clock=clock)

    def _read(self):
        state = loads(read_regular(self.path.parent, self.path.name)[0])
        if not isinstance(state, dict) or state.get("version") != self.VERSION:
            raise ProtocolError("missing/corrupt owner authority; cannot reset balance")
        validate(state["descriptor"])
        for key in ("requests", "operations", "paused", "active_ms", "active_run", "active_since", "last_clock"):
            if key not in state:
                raise ProtocolError("incomplete owner authority")
        return state

    @contextmanager
    def _transaction(self):
        # O_NOFOLLOW also applies to the lock; the authority itself is verified
        # via read_regular. All writers, including external adapters, use this API.
        fd = os.open(str(self.path) + ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            state = self._read()
            self._tick(state)
            yield state
            durable_write(self.path, jcs(state), mode=0o600, replace=True)
        finally:
            os.close(fd)

    def _tick(self, state):
        now = max(self.clock(), state["last_clock"])
        monotonic = time.monotonic_ns() if self.clock is _now else None
        boot = _boot() if self.clock is _now else "injected"
        elapsed_ns = max(0, now - state["last_clock"]) * 1_000_000
        if monotonic is not None and state.get("clock_boot_id") == boot:
            elapsed_ns = max(0, monotonic - int(state["monotonic_ns"]))
        elif state.get("clock_boot_id") != boot and state["active_since"] is not None:
            # A reboot plus a reversed wall clock cannot establish a free gap.
            if self.clock() < state["last_clock"]:
                elapsed_ns = max(elapsed_ns, state["descriptor"]["limits"]["active_ms"] * 1_000_000)
        if state["active_since"] is not None:
            state["active_ns"] = str(int(state["active_ns"]) + elapsed_ns)
            state["active_ms"] = int(state["active_ns"]) // 1_000_000
            request = state["requests"][state["active_run"]]
            request["active_ns"] = str(int(request["active_ns"]) + elapsed_ns)
            request["active_ms"] = int(request["active_ns"]) // 1_000_000
            state["active_since"] = now
        state["last_clock"] = now
        state["clock_boot_id"] = boot
        state["monotonic_ns"] = str(monotonic) if monotonic is not None else None

    @property
    def descriptor(self):
        return self._read()["descriptor"]

    @classmethod
    def for_context(cls, context, *, create=False, authority=None, clock=_now):
        binding_file = context.root / ".runtime/state/budget.json"
        if binding_file.exists():
            binding = context.load_state("budget")
            account = cls(binding["authority"], clock=clock)
            if digest(account.descriptor) != binding["descriptor_sha256"]:
                raise ProtocolError("owner descriptor changed")
            if context.run_id not in account._read()["requests"]:
                raise ProtocolError("original request contribution missing; cannot reset it")
        elif not create:
            raise ProtocolError("owner binding missing; continue cannot allocate a balance")
        elif context.request["budget"]["parent"] is not None:
            selected = authority or _authority.get() or os.environ.get("KS_BUDGET_AUTHORITY")
            if not selected:
                raise ProtocolError("parent needs explicit existing private budget authority")
            account = cls(selected, clock=clock)
            if account.descriptor != context.objects["budget_account"]:
                raise ProtocolError("parent descriptor differs from actual owner authority")
        else:
            path = context.root / ".runtime/owner-account.json"
            # A crash after create but before binding may recover that exact file.
            if path.exists():
                account = cls(path, clock=clock)
            else:
                account = cls.create(path, owner=context.owner["id"], limits=context.request["budget"]["limits"],
                                     account_id=context.run_id, clock=clock)
        if account.descriptor["owner"] != context.owner["id"]:
            raise ProtocolError("run and authority owner mismatch")
        if context.request["budget"]["parent"] is None and (account.descriptor["id"] != context.run_id or
                account.descriptor["limits"] != context.request["budget"]["limits"]):
            raise ProtocolError("top-level original authority differs from fixed request")
        account.register(context.run_id, context.request_sha256, context.request["budget"]["limits"],
                         root=str(context.root))
        binding = {"authority": str(account.path), "descriptor_sha256": digest(account.descriptor)}
        if not binding_file.exists():
            context.save_state("budget", binding)
        return account

    def register(self, run_id, request_sha256, request_limits, *, root):
        limits(request_limits)
        with self._transaction() as s:
            fixed = {"request_sha256": request_sha256, "limits": request_limits, "root": root}
            old = s["requests"].get(run_id)
            if old is not None:
                if any(old[k] != v for k, v in fixed.items()):
                    raise ProtocolError("registered request identity/limits changed")
                return
            if self._fences(s):
                raise BudgetDenied("unsettled owner fences a new request")
            s["requests"][run_id] = {**fixed, "active_ms": 0, "active_ns": "0"}

    def begin(self, run_id):
        with self._transaction() as s:
            if run_id not in s["requests"]:
                raise ProtocolError("request is not registered")
            if s["active_run"] not in (None, run_id):
                raise BudgetDenied("only serial requests share this owner")
            s["active_run"] = run_id
            s["auto_activity"] = False
            if s["active_since"] is None:
                s["active_since"] = s["last_clock"]

    def end(self, run_id):
        """Stop only when all original operations are proven quiescent."""
        with self._transaction() as s:
            if s["active_run"] == run_id and not any(not o["terminated"] and not o["released"] for o in s["operations"].values()):
                s["active_run"] = s["active_since"] = None
                s["auto_activity"] = False

    def _auto_begin(self, s, run_id):
        if s["active_run"] is None:
            s["active_run"], s["active_since"], s["auto_activity"] = run_id, s["last_clock"], True

    def _auto_end(self, s):
        if s.get("auto_activity") and not any(not o["terminated"] and not o["released"] for o in s["operations"].values()):
            s["active_run"] = s["active_since"] = None
            s["auto_activity"] = False

    def pause(self, paused=True):
        with self._transaction() as s:
            s["paused"] = bool(paused)
        # In-flight clocks deliberately continue. end() needs proven quiescence.

    def _totals(self, s, run_id=None, *, held=True):
        total = cost()
        unknown = set()
        for o in s["operations"].values():
            if run_id is not None and o["run_id"] != run_id:
                continue
            row = o["latest"]
            if row:
                for metric in ("calls", "evaluator_jobs", "tokens"):
                    key = o["role"] if metric == "calls" else metric
                    if key == "evaluator":
                        continue
                    total[key] += o["lower"][metric]
                    if row[metric] is None:
                        unknown.add(key)
                if o["role"] in ROLES:
                    total["model_calls"] += o["lower"]["calls"]
                    if row["calls"] is None:
                        unknown.add("model_calls")
            if held and not o["terminated"] and not o["released"]:
                for k, n in o["cost"].items():
                    if k != "active_ms":
                        known = o["lower"]["calls" if k in (*ROLES, "model_calls") else k] if row and k in (o["role"], "model_calls", "tokens", "evaluator_jobs") else 0
                        total[k] += max(0, n - known)
            if not held and o["launched"] and not o["terminated"]:
                unknown.update(("tokens", "evaluator_jobs" if o["role"] == "evaluator" else o["role"]))
        total["active_ms"] = s["active_ms"] if run_id is None else s["requests"][run_id]["active_ms"]
        return total, unknown

    def _fences(self, s):
        result = []
        for key, o in s["operations"].items():
            if o["released"]:
                continue
            if not o["terminated"]:
                result.append(key)
                continue
            if o["latest"] is None:
                result.append(key)
                continue
            owner_limits = _flatten(s["descriptor"]["limits"])
            run_limits = _flatten(s["requests"][o["run_id"]]["limits"])
            for metric in ("calls", "evaluator_jobs", "tokens"):
                k = o["role"] if metric == "calls" else metric
                constrained = k != "evaluator" and any(x.get(k) is not None for x in (owner_limits, run_limits))
                if metric == "calls" and o["role"] in ROLES:
                    constrained |= any(x.get("model_calls") is not None for x in (owner_limits, run_limits))
                if constrained and o["latest"][metric] is None:
                    result.append(key)
                    break
        return sorted(set(result))

    def fences(self):
        with self._transaction() as s:
            return self._fences(s)

    def _check(self, s, run_id, requested, floor, native_limits, *, group=None, active_operations=()):
        if s["paused"]:
            raise BudgetDenied("owner paused")
        fences = self._fences(s)
        if group is not None:
            fences = [key for key in fences if s["operations"][key].get("group") != group or s["operations"][key].get("uncertain")]
        if active_operations:
            fences = [key for key in fences if key not in active_operations or s["operations"][key].get("uncertain") or s["operations"][key]["terminated"]]
        if fences:
            raise BudgetDenied("original operations/limited balances remain unresolved")
        if s["active_run"] not in (None, run_id):
            raise BudgetDenied("another serial request is active")
        configurations = [(s["descriptor"]["limits"], None), (s["requests"][run_id]["limits"], run_id)]
        if native_limits:
            limits(native_limits)
            configurations.append((native_limits, run_id))
        deadline = _deadline(s["descriptor"]["deadline_utc"])
        if deadline is not None and s["last_clock"] >= deadline:
            raise BudgetDenied("owner deadline exhausted")
        available_active = []
        for spec, selected in configurations:
            maximum = _flatten(spec)
            used, unknown = self._totals(s, selected)
            deadline = _deadline(spec.get("deadline_utc"))
            if deadline is not None:
                available_active.append(deadline - s["last_clock"])
            for key, cap in maximum.items():
                if cap is None:
                    continue
                if key in unknown or used[key] + requested[key] + floor[key] > cap:
                    raise BudgetDenied("insufficient or unknown " + key + " including minimum finish reserve")
            available_active.append(maximum["active_ms"] - used["active_ms"] - floor["active_ms"])
        if _deadline(s["descriptor"]["deadline_utc"]) is not None:
            available_active.append(_deadline(s["descriptor"]["deadline_utc"]) - s["last_clock"])
        if min(available_active) < requested["active_ms"]:
            raise BudgetDenied("deadline cannot cover minimum active interval")
        return min(available_active)

    def reserve(self, run_id, operation_id, *, role, purpose, amount, floor=None, native_limits=None):
        """Durably reserve before input preparation/submission; repeat is idempotent."""
        empty = cost()
        if role not in (*ROLES, "evaluator") or not isinstance(operation_id, str) or not operation_id or not isinstance(purpose, str) or not purpose:
            raise ProtocolError("reservation needs a role, original operation ID and purpose")
        if set(amount) != set(empty) or any(type(n) is not int or n < 0 for n in amount.values()):
            raise ProtocolError("invalid reservation vector")
        floor = floor or empty
        if set(floor) != set(empty) or any(type(n) is not int or n < 0 for n in floor.values()):
            raise ProtocolError("invalid finish reserve")
        with self._transaction() as s:
            fixed = {"run_id": run_id, "role": role, "purpose": purpose, "cost": amount,
                     "floor": floor, "native_limits": native_limits}
            if operation_id in s["operations"]:
                op = s["operations"][operation_id]
                if any(op[k] != v for k, v in fixed.items()):
                    raise ProtocolError("reservation replay conflicts")
                return op["reservation"]
            interval = self._check(s, run_id, amount, floor, native_limits)
            self._auto_begin(s, run_id)
            reservation = {"owner": s["descriptor"]["owner"], "account_id": s["descriptor"]["id"],
                           "run_id": run_id, "operation_id": operation_id, "authorization": uuid4().hex,
                           "calls": amount.get(role, 0), "evaluator_jobs": amount["evaluator_jobs"],
                           "tokens": amount["tokens"], "active_ms": interval,
                           "deadline_utc_ms": s["last_clock"] + interval}
            if s["monotonic_ns"] is not None:
                reservation.update(deadline_monotonic_ns=str(int(s["monotonic_ns"]) + interval * 1_000_000),
                                   clock_boot_id=s["clock_boot_id"])
            s["operations"][operation_id] = {**fixed, "reservation": reservation, "launched": False,
                "terminated": False, "released": False, "producer": None, "history": [], "latest": None,
                "lower": {k: 0 for k in _METRICS}}
            return reservation

    def reserve_group(self, run_id, group_id, operations, *, floor=None, native_limits=None):
        """Atomically preauthorize a finite native parallel group in one process.

        This is not a scheduler or child account. Every member has its own exact
        charge/receipt. Recovery in another process may settle members, but may
        not launch an unstarted sibling alongside an ambiguous old operation.
        """
        if not isinstance(group_id, str) or not group_id or not operations:
            raise ProtocolError("finite group needs an identity and operation list")
        total = cost()
        floor = floor or cost()
        if set(floor) != set(total) or any(type(n) is not int or n < 0 for n in floor.values()):
            raise ProtocolError("invalid group finish reserve")
        ids = []
        for spec in operations:
            if set(spec) != {"operation_id", "role", "purpose", "amount"}:
                raise ProtocolError("invalid group operation")
            if spec["role"] not in (*ROLES, "evaluator") or set(spec["amount"]) != set(total):
                raise ProtocolError("invalid group role/cost")
            if any(type(n) is not int or n < 0 for n in spec["amount"].values()):
                raise ProtocolError("invalid group amount")
            ids.append(spec["operation_id"])
            for key, value in spec["amount"].items():
                total[key] = max(total[key], value) if key == "active_ms" else total[key] + value
        if len(set(ids)) != len(ids):
            raise ProtocolError("duplicate group operation")
        with self._transaction() as s:
            groups = s.setdefault("groups", {})
            fixed = {"run_id": run_id, "operations": operations, "floor": floor, "native_limits": native_limits}
            if group_id in groups:
                if groups[group_id]["fixed"] != fixed:
                    raise ProtocolError("group reservation replay conflicts")
                return {key: s["operations"][key]["reservation"] for key in ids}
            if any(key in s["operations"] for key in ids):
                raise ProtocolError("group collides with an original operation")
            interval = self._check(s, run_id, total, floor, native_limits)
            self._auto_begin(s, run_id)
            groups[group_id] = {"fixed": fixed, "process": _process_identity()}
            for spec in operations:
                op_id, amount = spec["operation_id"], spec["amount"]
                reservation = {"owner": s["descriptor"]["owner"], "account_id": s["descriptor"]["id"],
                    "run_id": run_id, "operation_id": op_id, "authorization": uuid4().hex,
                    "calls": amount.get(spec["role"], 0), "evaluator_jobs": amount["evaluator_jobs"],
                    "tokens": amount["tokens"], "active_ms": interval, "deadline_utc_ms": s["last_clock"] + interval,
                    "group_id": group_id}
                if s["monotonic_ns"] is not None:
                    reservation.update(deadline_monotonic_ns=str(int(s["monotonic_ns"]) + interval * 1_000_000),
                                       clock_boot_id=s["clock_boot_id"])
                s["operations"][op_id] = {"run_id": run_id, "role": spec["role"], "purpose": spec["purpose"],
                    "cost": amount, "floor": floor, "native_limits": native_limits, "reservation": reservation,
                    "group": group_id, "launched": False, "terminated": False, "released": False,
                    "producer": None, "history": [], "latest": None, "lower": {k: 0 for k in _METRICS}}
            return {key: s["operations"][key]["reservation"] for key in ids}

    def authorize(self, reservation):
        """Launch boundary check. Marks may-have-launched BEFORE external exec."""
        with self._transaction() as s:
            op = s["operations"].get(reservation.get("operation_id"))
            if op is None or jcs(op["reservation"]) != jcs(reservation):
                raise ProtocolError("unknown/mismatched original authorization")
            if op["launched"] or op["terminated"] or op["released"]:
                return False
            if s["last_clock"] >= reservation["deadline_utc_ms"]:
                return False
            if "deadline_monotonic_ns" in reservation and (s["clock_boot_id"] != reservation["clock_boot_id"] or
                    int(s["monotonic_ns"]) >= int(reservation["deadline_monotonic_ns"])):
                return False
            group = op.get("group")
            if group is not None and s["groups"][group]["process"] != _process_identity():
                return False
            # Exclude this unlaunched reservation, retain every other fence.
            op["released"] = True
            try:
                self._check(s, op["run_id"], op["cost"], op["floor"], op["native_limits"], group=group)
            except BudgetDenied:
                return False
            finally:
                op["released"] = False
            op["launched"] = True
            return True

    def check_active(self, reservation) -> bool:
        """Check continued execution of an original launched operation.

        This ticks the sole clock without admitting, charging, extending or
        releasing anything. False means stop bounded work and settle the same
        native operation; it is never proof of termination or a refund.
        Normal in-flight members of the same approved group are not unknown.
        Native identity/acknowledgment loss must be recorded with mark_unknown.
        """
        try:
            with self._transaction() as s:
                if not isinstance(reservation, dict):
                    return False
                operation_id = reservation.get("operation_id")
                op = s["operations"].get(operation_id)
                if op is None or jcs(op["reservation"]) != jcs(reservation):
                    return False
                run_id = op["run_id"]
                if (reservation["owner"] != s["descriptor"]["owner"] or
                    reservation["account_id"] != s["descriptor"]["id"] or
                    reservation["run_id"] != run_id or run_id not in s["requests"] or
                    s["active_run"] != run_id or s["active_since"] is None or
                    op["launched"] is not True or op["released"] is not False or op["terminated"] is not False):
                    return False
                request = s["requests"][run_id]
                if (not isinstance(request["request_sha256"], str) or not request["request_sha256"].startswith("sha256:") or
                    not isinstance(request["root"], str) or not request["root"]):
                    return False
                if s["last_clock"] >= reservation["deadline_utc_ms"]:
                    return False
                if "deadline_monotonic_ns" in reservation or "clock_boot_id" in reservation:
                    if (reservation["clock_boot_id"] != s["clock_boot_id"] or s["monotonic_ns"] is None or
                        int(s["monotonic_ns"]) >= int(reservation["deadline_monotonic_ns"])):
                        return False
                allowed = {operation_id}
                group_id = op.get("group")
                if group_id is not None:
                    group = s["groups"][group_id]
                    process = group["process"]
                    if (reservation.get("group_id") != group_id or group["fixed"]["run_id"] != run_id or
                        type(process["pid"]) is not int or process["pid"] <= 0 or
                        not isinstance(process["start_ticks"], str) or not process["start_ticks"].isdigit() or
                        not isinstance(process["boot_id"], str) or not process["boot_id"]):
                        return False
                    if s["clock_boot_id"] != "injected" and process["boot_id"] != s["clock_boot_id"]:
                        return False
                    members = set()
                    for spec in group["fixed"]["operations"]:
                        key = spec["operation_id"]
                        member = s["operations"][key]
                        if (key in members or member.get("group") != group_id or member["run_id"] != run_id or
                            member["role"] != spec["role"] or member["purpose"] != spec["purpose"] or
                            jcs(member["cost"]) != jcs(spec["amount"]) or
                            jcs(member["floor"]) != jcs(group["fixed"]["floor"]) or
                            jcs(member["native_limits"]) != jcs(group["fixed"]["native_limits"]) or
                            member["reservation"].get("group_id") != group_id or
                            member["reservation"]["operation_id"] != key or member["reservation"]["run_id"] != run_id or
                            member["reservation"]["owner"] != reservation["owner"] or
                            member["reservation"]["account_id"] != reservation["account_id"]):
                            return False
                        members.add(key)
                    if operation_id not in members:
                        return False
                    allowed.update(members)
                elif "group_id" in reservation:
                    return False
                # Do not mask an explicitly lost identity, even inside a group.
                # Older native-constrained unknown usage also remains a fence.
                for other in s["operations"].values():
                    if other.get("uncertain"):
                        return False
                    native = other["native_limits"]
                    if native is not None:
                        limits(native)
                        used, unknown = self._totals(s, other["run_id"])
                        if any(key in unknown and cap is not None for key, cap in _flatten(native).items()):
                            return False
                # Existing reservations are already in _totals(held=True).
                # Add no new cost, retain the original minimum finish floor.
                try:
                    return self._check(s, run_id, cost(), op["floor"], op["native_limits"],
                                       active_operations=allowed) > 0
                except BudgetDenied:
                    return False  # persist the tick even while pause/limits deny progress
        except (KeyError, TypeError, ValueError, OSError, OverflowError, AttributeError):
            # Corrupt/missing original state must stop a bounded caller, never
            # mint replacement authority. This also covers ProtocolError.
            return False

    def claim_group(self, group_id, reservations):
        """Transfer an entirely unstarted group once to its external launcher.

        No public token or new balance: the caller opens the same private
        authority and presents every original reservation. A half-launched or
        previously claimed group can never be reclaimed by a recovery process.
        """
        with self._transaction() as s:
            group = s.get("groups", {}).get(group_id)
            if group is None:
                raise ProtocolError("unknown original reservation group")
            ids = [o["operation_id"] for o in group["fixed"]["operations"]]
            expected = {key: s["operations"][key]["reservation"] for key in ids}
            if jcs(reservations) != jcs(expected):
                raise ProtocolError("group claim changes original authorization")
            process = _process_identity()
            if group.get("claimed") and group["process"] == process:
                return
            if group.get("claimed") or any(s["operations"][key]["launched"] or s["operations"][key]["released"] or
                                           s["operations"][key]["terminated"] or s["operations"][key].get("uncertain") for key in ids):
                raise BudgetDenied("cannot transfer a launched, released or previously claimed group")
            group["process"] = process
            group["claimed"] = True

    def release_unstarted(self, operation_id):
        with self._transaction() as s:
            op = s["operations"][operation_id]
            if op["launched"] or op["latest"]:
                raise BudgetDenied("cannot refund a possibly started operation")
            op["released"] = op["terminated"] = True
            self._auto_end(s)

    def mark_unknown(self, operation_id, reason):
        """A lost acknowledgment/identity fences even preauthorized siblings."""
        if not isinstance(reason, str) or not reason:
            raise ProtocolError("unknown operation needs a reason")
        with self._transaction() as s:
            op = s["operations"][operation_id]
            if not op["terminated"] and not op["released"]:
                op["uncertain"] = reason

    def absorb(self, run_id, row, *, terminated, lower_bounds=None):
        """Absorb cumulative native facts once; preserve original receipt history."""
        validate({"protocol": PROTOCOL, "kind": "usage_index", "entries": [row]})
        lower_bounds = lower_bounds or {}
        if not set(lower_bounds) <= set(_METRICS) or any(type(n) is not int or n < 0 for n in lower_bounds.values()):
            raise ProtocolError("invalid usage lower bounds")
        with self._transaction() as s:
            op = s["operations"].get(row["operation_id"])
            if op is None or op["run_id"] != run_id or op["role"] != row["role"]:
                raise ProtocolError("receipt has no matching original owner reservation")
            if op["producer"] not in (None, row["producer"]):
                raise ProtocolError("receipt producer changed")
            for old in op["history"]:
                if old["sequence"] == row["sequence"]:
                    if old != row:
                        raise ProtocolError("receipt identity/sequence conflicts")
                    if terminated and not op["terminated"]:
                        raise ProtocolError("replayed receipt changes termination fact")
                    if any(n > op["lower"][k] for k, n in lower_bounds.items()):
                        raise ProtocolError("replayed receipt changes known lower bounds")
                    return False
            previous = op["latest"]
            if previous and row["sequence"] <= previous["sequence"]:
                raise ProtocolError("out-of-order unknown cumulative receipt")
            if op["terminated"] and not terminated:
                raise ProtocolError("terminal operation cannot become in-flight")
            for k in _METRICS:
                bound = lower_bounds.get(k, 0)
                if row[k] is not None and row[k] < bound:
                    raise ProtocolError("counter contradicts known lower bound")
                op["lower"][k] = max(op["lower"][k], bound)
                if row[k] is not None:
                    if row[k] < op["lower"][k]:
                        raise ProtocolError("cumulative native counter regressed")
                    op["lower"][k] = row[k]
            op["producer"] = row["producer"]
            op["latest"] = row
            op["history"].append(row)
            op["terminated"] = bool(terminated)
            op["uncertain"] = False if terminated else "native operation remains unknown"
            self._auto_end(s)
            return True

    def operations(self, run_id=None):
        with self._transaction() as s:
            return {k: v for k, v in s["operations"].items() if run_id is None or v["run_id"] == run_id}

    def usage(self, run_id, publication: Publication):
        with self._transaction() as s:
            rows = [r for o in s["operations"].values() if o["run_id"] == run_id for r in o["history"]]
            for op_id, o in s["operations"].items():
                if o["run_id"] == run_id and not o["history"] and o["launched"]:
                    rows.append({"producer": "ks-owner:" + run_id, "operation_id": op_id, "role": o["role"],
                        "purpose": o["purpose"], "receipt_id": op_id + ":pending", "sequence": 0, "receipt": None,
                        "state": "unknown", "calls": 0 if o["role"] == "evaluator" else None,
                        "evaluator_jobs": None if o["role"] == "evaluator" else 0, "tokens": None, "gpu_ms": None})
            totals, unknown = self._totals(s, run_id, held=False)
            index = publication.put_json("usage.json", {"protocol": PROTOCOL, "kind": "usage_index", "entries": rows})
            publication.commit([index])
            return {"scope": "request_cumulative", "receipts": index,
                    "calls": {r: None if r in unknown else totals[r] for r in ROLES},
                    "evaluator_jobs": None if "evaluator_jobs" in unknown else totals["evaluator_jobs"],
                    "tokens": None if "tokens" in unknown else totals["tokens"],
                    "active_ms": totals["active_ms"], "unsettled": self._fences(s)}
