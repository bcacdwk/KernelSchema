from copy import deepcopy
import subprocess
import sys
import json

import pytest

from kernelschema.artifacts import Publication
from kernelschema.budget import BudgetDenied, OwnerAccount, cost
from kernelschema.contracts import ProtocolError


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
