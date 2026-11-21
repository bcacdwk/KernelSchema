from copy import deepcopy
import subprocess
import sys
import json

import pytest

from kernelschema.artifacts import Publication, durable_write
from kernelschema.budget import BudgetDenied, OwnerAccount, cost
from kernelschema.contracts import ProtocolError, jcs


def setup_account(tmp_path, *, tokens=None):
    clock = [1000]
    limits = {"calls": {"coder": 3, "reviewer": 2, "planner": 1, "extract": 0}, "evaluator_jobs": 2,
              "active_ms": 1000, "tokens": tokens, "model_calls": 4}
    account = OwnerAccount.create(tmp_path / "authority.json", owner="owner", limits=limits, clock=lambda: clock[0])
    account.register("run1", "sha256:" + "1" * 64, limits, root=str(tmp_path / "run1"))
    return account, clock, limits


def row(op="op", **changes):
    return {"producer": "native:run1", "operation_id": op, "role": "coder", "purpose": "code", "receipt_id": op + ":1",
            "sequence": 1, "receipt": None, "state": "settled", "calls": 1, "evaluator_jobs": 0, "tokens": 10, "gpu_ms": None, **changes}


def reserve(account, op="op", run="run1", **kw):
    return account.reserve(run, op, role="coder", purpose="code", amount=cost(role="coder", calls=1, active_ms=1), **kw)


def test_cumulative_receipts_replay_conflict_and_increment(tmp_path):
    a, _, _ = setup_account(tmp_path)
    assert a.authorize(reserve(a))
    assert a.absorb("run1", row(), terminated=True)
    assert not a.absorb("run1", row(), terminated=True)
    with pytest.raises(ProtocolError, match="conflict"):
        a.absorb("run1", row(tokens=11), terminated=True)
    assert a.absorb("run1", row(sequence=2, tokens=15), terminated=True)
    usage = a.usage("run1", Publication(tmp_path / "bundle"))
    assert usage["calls"]["coder"] == 1 and usage["tokens"] == 15
    assert not usage["unsettled"]


def test_append_unknown_token_erratum_preserves_original_charge_and_history(tmp_path):
    a, clock, limits = setup_account(tmp_path)
    a.begin("run1")
    a.authorize(reserve(a))
    clock[0] += 57
    original = row(tokens=0)
    a.absorb("run1", original, terminated=True)
    a.end("run1")
    before = a.operations()["op"]
    descriptor = a.descriptor
    clock[0] += 100
    correction = {**original, "sequence": 2, "receipt_id": "op:usage-erratum", "tokens": None}
    assert a.absorb("run1", correction, terminated=True)
    assert not a.absorb("run1", correction, terminated=True)
    # Replaying the old observation neither removes the erratum nor recharges.
    assert not a.absorb("run1", original, terminated=True)
    after = a.operations()["op"]
    assert after["history"] == [original, correction] and after["latest"] == correction
    assert after["reservation"] == before["reservation"] and after["lower"] == before["lower"]
    assert after["terminated"] and a.descriptor == descriptor and descriptor["limits"] == limits
    assert len(a.operations()) == 1
    usage = a.usage("run1", Publication(tmp_path / "erratum"))
    assert usage["calls"]["coder"] == 1 and usage["evaluator_jobs"] == 0
    assert usage["active_ms"] == 57 and usage["tokens"] is None and not usage["unsettled"]
    state = json.loads(a.path.read_bytes())
    assert state["active_run"] is None and state["active_since"] is None


def test_authorize_rejects_python_equal_but_different_jcs_reservation(tmp_path):
    account, _, _ = setup_account(tmp_path)
    original = reserve(account)
    forged = {**original, "calls": True}
    assert forged == original and jcs(forged) != jcs(original)
    with pytest.raises(ProtocolError, match="authorization"):
        account.authorize(forged)
    assert not account.operations()["op"]["launched"]
    assert account.authorize(original)


def test_unknown_statistics_distinct_from_termination_and_constrained_fence(tmp_path):
    a, clock, _ = setup_account(tmp_path)
    a.begin("run1")
    a.authorize(reserve(a))
    a.absorb("run1", row(tokens=None), terminated=True)
    a.end("run1")
    clock[0] += 100
    assert not a.fences()  # no token limit; terminal unknown tokens are quiescent
    assert a.usage("run1", Publication(tmp_path / "out"))["active_ms"] == 0
    b, _, _ = setup_account(tmp_path / "limited", tokens=20)
    b.authorize(reserve(b))
    b.absorb("run1", row(tokens=None), terminated=True)
    assert b.fences() == ["op"]
    with pytest.raises(BudgetDenied):
        reserve(b, "next")


def test_pause_and_parent_wait_share_clock_crash_interval_counts(tmp_path):
    a, clock, _ = setup_account(tmp_path)
    a.begin("run1")
    clock[0] += 100
    a.begin("run1")  # parent/child waiting starts no second clock
    a.authorize(reserve(a))
    a.pause()
    clock[0] += 200
    a.end("run1")  # still in flight: must not stop
    restored = OwnerAccount(a.path, clock=lambda: clock[0])
    clock[0] += 100
    assert restored.usage("run1", Publication(tmp_path / "out"))["active_ms"] == 400
    restored.absorb("run1", row(), terminated=True)
    restored.end("run1")
    clock[0] += 200
    assert restored.usage("run1", Publication(tmp_path / "out2"))["active_ms"] == 400


def test_new_runs_same_q_charge_and_parent_does_not_reset(tmp_path):
    a, _, lim = setup_account(tmp_path)
    a.authorize(reserve(a))
    a.absorb("run1", row(), terminated=True)
    a.register("run2", "sha256:" + "1" * 64, lim, root=str(tmp_path / "run2"))
    a.authorize(reserve(a, "op2", "run2"))
    a.absorb("run2", row("op2", producer="native:run2"), terminated=True)
    assert a.usage("run2", Publication(tmp_path / "out"))["calls"]["coder"] == 1
    assert len(a.operations()) == 2
    smaller = deepcopy(lim)
    smaller["calls"]["coder"] = 0
    a.register("run3", "sha256:" + "3" * 64, smaller, root="run3")
    with pytest.raises(BudgetDenied):
        reserve(a, "op3", "run3")


def test_finish_floor_deadline_pause_and_unknown_do_not_refund(tmp_path):
    a, clock, _ = setup_account(tmp_path)
    with pytest.raises(BudgetDenied):
        reserve(a, floor=cost(role="reviewer", calls=3))
    reservation = reserve(a)
    a.pause()
    assert not a.authorize(reservation)
    a.pause(False)
    assert a.authorize(reservation)
    with pytest.raises(BudgetDenied):
        a.release_unstarted("op")
    clock[0] += 1000
    assert not a.authorize(reservation)
    assert a.fences()


def test_missing_authority_never_created(tmp_path):
    with pytest.raises(ProtocolError):
        OwnerAccount(tmp_path / "absent.json")
    assert not (tmp_path / "absent.json").exists()


def test_native_parallel_group_is_reserved_atomically_and_has_one_clock(tmp_path):
    a, clock, _ = setup_account(tmp_path)
    a.begin("run1")
    specs = [{"operation_id": "v0", "role": "evaluator", "purpose": "V0", "amount": cost(evaluator_jobs=1, active_ms=1)},
             {"operation_id": "review", "role": "reviewer", "purpose": "prefetch", "amount": cost(role="reviewer", calls=1, active_ms=1)}]
    reservations = a.reserve_group("run1", "g", specs)
    assert a.authorize(reservations["v0"])
    assert a.authorize(reservations["review"])
    assert not a.authorize(reservations["review"])
    clock[0] += 20
    assert a.usage("run1", Publication(tmp_path / "out"))["active_ms"] == 20
    with pytest.raises(BudgetDenied):
        reserve(a, "unplanned")
    a.absorb("run1", row("v0", role="evaluator", calls=0, evaluator_jobs=1, tokens=0), terminated=True)
    a.absorb("run1", row("review", role="reviewer"), terminated=True)
    a.end("run1")
    assert not a.fences()


def test_external_process_opens_same_authority_and_cannot_resume_half_launched_group(tmp_path):
    a, _, _ = setup_account(tmp_path)
    specs = [{"operation_id": "v0", "role": "evaluator", "purpose": "V0", "amount": cost(evaluator_jobs=1)},
             {"operation_id": "review", "role": "reviewer", "purpose": "prefetch", "amount": cost(role="reviewer", calls=1)}]
    reservations = a.reserve_group("run1", "g", specs)
    a.authorize(reservations["v0"])
    script = "import json,sys; from kernelschema.budget import OwnerAccount; a=OwnerAccount(sys.argv[1],clock=lambda:1000); print(a.authorize(json.loads(sys.argv[2])))"
    result = subprocess.run([sys.executable, "-B", "-c", script, str(a.path), json.dumps(reservations["review"])], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"
    assert not a.operations()["review"]["launched"]


def test_native_and_parent_limits_and_lower_bounds(tmp_path):
    a, _, lim = setup_account(tmp_path)
    tight = deepcopy(lim)
    tight["model_calls"] = 0
    with pytest.raises(BudgetDenied):
        reserve(a, native_limits=tight)
    a.authorize(reserve(a))
    a.absorb("run1", row(calls=None), terminated=True, lower_bounds={"calls": 1})
    assert a.operations()["op"]["lower"]["calls"] == 1
    assert a.fences() == ["op"]


def test_unknown_member_and_pause_block_prepaid_sibling(tmp_path):
    a, _, _ = setup_account(tmp_path)
    specs = [{"operation_id": "v0", "role": "evaluator", "purpose": "V0", "amount": cost(evaluator_jobs=1)},
             {"operation_id": "review", "role": "reviewer", "purpose": "prefetch", "amount": cost(role="reviewer", calls=1)}]
    reservations = a.reserve_group("run1", "g", specs)
    a.pause()
    assert not a.authorize(reservations["v0"])
    a.pause(False)
    assert a.authorize(reservations["v0"])
    a.mark_unknown("v0", "lost original submission acknowledgment")
    assert not a.authorize(reservations["review"])
    a.release_unstarted("review")
    with pytest.raises(BudgetDenied):
        a.release_unstarted("v0")


def test_group_denial_is_atomic(tmp_path):
    a, _, _ = setup_account(tmp_path)
    specs = [{"operation_id": "v0", "role": "evaluator", "purpose": "V0", "amount": cost(evaluator_jobs=1)},
             {"operation_id": "review", "role": "reviewer", "purpose": "prefetch", "amount": cost(role="reviewer", calls=3)}]
    with pytest.raises(BudgetDenied):
        a.reserve_group("run1", "g", specs)
    assert not a.operations()


def test_unstarted_group_can_be_claimed_once_by_external_launcher(tmp_path):
    a, _, _ = setup_account(tmp_path)
    specs = [{"operation_id": "v0", "role": "evaluator", "purpose": "V0", "amount": cost(evaluator_jobs=1)},
             {"operation_id": "review", "role": "reviewer", "purpose": "prefetch", "amount": cost(role="reviewer", calls=1)}]
    reservations = a.reserve_group("run1", "g", specs)
    forged = deepcopy(reservations)
    forged["review"]["calls"] = True
    with pytest.raises(ProtocolError, match="original authorization"):
        a.claim_group("g", forged)
    script = "import json,sys; from kernelschema.budget import OwnerAccount; a=OwnerAccount(sys.argv[1],clock=lambda:1000); r=json.loads(sys.argv[2]); a.claim_group('g',r); print(a.authorize(r['v0']),a.authorize(r['review']))"
    result = subprocess.run([sys.executable, "-B", "-c", script, str(a.path), json.dumps(reservations)], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "True True"
    with pytest.raises(BudgetDenied):
        a.claim_group("g", reservations)


def test_reservation_automatically_accounts_external_activity_until_quiescence(tmp_path):
    a, clock, _ = setup_account(tmp_path)
    reservation = reserve(a)  # external caller need not remember a second clock API
    clock[0] += 10
    a.authorize(reservation)
    clock[0] += 20
    a.absorb("run1", row(), terminated=True)
    clock[0] += 100
    assert a.usage("run1", Publication(tmp_path / "out"))["active_ms"] == 30


def _rewrite_authority(account, change):
    state = account._read()
    change(state)
    durable_write(account.path, jcs(state), mode=0o600, replace=True)


def _active_group(account, *, native_limits=None):
    specs = [{"operation_id": "coder", "role": "coder", "purpose": "native code", "amount": cost(role="coder", calls=1)},
             {"operation_id": "review", "role": "reviewer", "purpose": "prefetch", "amount": cost(role="reviewer", calls=1)}]
    reservations = account.reserve_group("run1", "active", specs, native_limits=native_limits)
    for reservation in reservations.values():
        assert account.authorize(reservation)
    return reservations


def test_check_active_ticks_only_original_clock_without_readmitting_or_charging(tmp_path):
    a, clock, _ = setup_account(tmp_path)
    reservation = reserve(a)
    assert not a.check_active(reservation)
    assert a.authorize(reservation)
    original = a.operations()
    clock[0] += 20
    assert a.check_active(reservation)
    assert a.check_active(deepcopy(reservation))
    assert a.operations() == original  # no counters, history, reservations or launch facts changed
    assert a._read()["active_ms"] == a._read()["requests"]["run1"]["active_ms"] == 20
    assert not a.authorize(reservation)
    a.pause()
    clock[0] += 10
    assert not a.check_active(reservation)
    assert a._read()["active_ms"] == 30  # refusal/pause does not stop an in-flight clock
    a.pause(False)
    assert a.check_active(reservation)


@pytest.mark.parametrize("change", [lambda r: r.update(owner="other"), lambda r: r.update(account_id="other"),
    lambda r: r.update(run_id="other"), lambda r: r.update(operation_id="missing"),
    lambda r: r.update(authorization="forged"), lambda r: r.update(calls=True), lambda r: r.update(deadline_utc_ms=999999)])
def test_check_active_rejects_different_or_forged_reservation(tmp_path, change):
    a, _, _ = setup_account(tmp_path)
    reservation = reserve(a)
    a.authorize(reservation)
    wrong = deepcopy(reservation)
    change(wrong)
    assert not a.check_active(wrong)
    assert a.check_active(reservation)


@pytest.mark.parametrize("terminal", ["released", "terminated", "unknown"])
def test_check_active_rejects_closed_or_unknown_operation(tmp_path, terminal):
    a, _, _ = setup_account(tmp_path)
    reservation = reserve(a)
    if terminal == "released":
        a.release_unstarted("op")
    else:
        a.authorize(reservation)
        if terminal == "terminated":
            a.absorb("run1", row(), terminated=True)
        else:
            a.mark_unknown("op", "original native identity lost")
    assert not a.check_active(reservation)


def test_check_active_allows_normal_group_but_not_unknown_member(tmp_path):
    a, clock, _ = setup_account(tmp_path)
    reservations = _active_group(a)
    clock[0] += 40
    assert all(a.check_active(r) for r in reservations.values())
    assert a._read()["active_ms"] == 40  # no parent/member double counting
    a.mark_unknown("review", "lost acknowledgment")
    assert not a.check_active(reservations["coder"])
    assert not a.check_active(reservations["review"])


@pytest.mark.parametrize("scope", ["owner", "request", "native"])
def test_check_active_honors_tightest_original_counter_limit(tmp_path, scope):
    _, clock, base = setup_account(tmp_path / "base")
    parent, request, native = deepcopy(base), deepcopy(base), deepcopy(base)
    {"owner": parent, "request": request, "native": native}[scope]["model_calls"] = 2
    a = OwnerAccount.create(tmp_path / "authority.json", owner="owner", limits=parent, clock=lambda: clock[0])
    a.register("run1", "sha256:" + "1" * 64, request, root="run1")
    reservations = _active_group(a, native_limits=native)
    assert a.check_active(reservations["coder"])  # both admitted calls fill cap, no new call is requested
    a.absorb("run1", row("review", role="reviewer", calls=2), terminated=True)
    assert not a.check_active(reservations["coder"])


@pytest.mark.parametrize("limited", [False, True])
def test_check_active_distinguishes_unrestricted_unknown_tokens(tmp_path, limited):
    a, _, lim = setup_account(tmp_path)
    native = deepcopy(lim)
    native["tokens"] = 100 if limited else None
    reservations = _active_group(a, native_limits=native)
    a.absorb("run1", row("review", role="reviewer", tokens=None), terminated=True)
    assert a.check_active(reservations["coder"]) is (not limited)


def test_check_active_preserves_original_finish_reserve_and_utc_expiry(tmp_path):
    a, clock, _ = setup_account(tmp_path)
    reservation = reserve(a, floor=cost(role="reviewer", calls=1, active_ms=100))
    a.authorize(reservation)
    clock[0] += 899
    assert a.check_active(reservation)
    clock[0] += 1
    assert not a.check_active(reservation)
    assert a._read()["active_ms"] == 900  # 100 ms remains reserved for finish, not reissued to this op


@pytest.mark.parametrize("change", [lambda s: s["requests"].clear(), lambda s: s["descriptor"].update(owner="other"),
    lambda s: s["requests"]["run1"].pop("request_sha256"),
    lambda s: s["operations"].clear(), lambda s: s.update(active_run=None, active_since=None)])
def test_check_active_missing_original_identity_does_not_reset_state(tmp_path, change):
    a, _, _ = setup_account(tmp_path)
    reservation = reserve(a)
    a.authorize(reservation)
    _rewrite_authority(a, change)
    assert not a.check_active(reservation)
    assert not a._read()["operations"] or len(a._read()["operations"]) == 1


def test_check_active_missing_authority_is_false_without_recreation(tmp_path):
    a, _, _ = setup_account(tmp_path)
    reservation = reserve(a)
    a.authorize(reservation)
    a.path.unlink()
    assert not a.check_active(reservation)
    assert not a.path.exists()


@pytest.mark.parametrize("change", [lambda s: s["groups"].clear(), lambda s: s["groups"]["active"]["process"].clear(),
    lambda s: s["groups"]["active"]["fixed"].update(run_id="different"), lambda s: s["operations"].pop("review")])
def test_check_active_missing_group_identity_is_false(tmp_path, change):
    a, _, _ = setup_account(tmp_path)
    reservations = _active_group(a)
    _rewrite_authority(a, change)
    assert not a.check_active(reservations["coder"])


@pytest.mark.parametrize("failure", ["monotonic_expired", "boot_changed"])
def test_check_active_uses_original_monotonic_deadline_and_boot(tmp_path, monkeypatch, failure):
    from kernelschema import budget
    _, _, lim = setup_account(tmp_path / "base")
    wall, mono, boot = [1000], [5000000000], ["boot-a"]
    monkeypatch.setattr(budget.time, "time_ns", lambda: wall[0] * 1000000)
    monkeypatch.setattr(budget.time, "monotonic_ns", lambda: mono[0])
    monkeypatch.setattr(budget, "_boot", lambda: boot[0])
    a = OwnerAccount.create(tmp_path / "authority.json", owner="owner", limits=lim)
    a.register("run1", "sha256:" + "1" * 64, lim, root="run1")
    reservation = reserve(a)
    a.authorize(reservation)
    assert a.check_active(reservation)
    if failure == "monotonic_expired":
        wall[0] -= 100  # reversed wall clock cannot extend the original monotonic window
        mono[0] = int(reservation["deadline_monotonic_ns"])
    else:
        boot[0] = "boot-b"
    assert not a.check_active(reservation)
