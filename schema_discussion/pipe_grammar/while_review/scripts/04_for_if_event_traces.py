#!/usr/bin/env python3
"""Small CPU witnesses for the 04 for/if review; no kernel or GPU execution.

This is an explicit serial issue/done trace model, not a pipe parser or a GPU
simulator. Python state updates below implement the named compute action's
contract; they do not propose assignment nodes for the schema language.
"""

import json
from dataclasses import dataclass, field


@dataclass
class Trace:
    events: list[str] = field(default_factory=list)
    predicates: list[dict] = field(default_factory=list)
    result: dict = field(default_factory=dict)

    def action(self, name, update=None):
        self.events.append("issue " + name)
        if update is not None:
            update()
        self.events.append("done " + name)

    def predicate(self, r, value):
        self.predicates.append({"round": r, "value": value})
        return value

    def as_dict(self):
        return {
            "events": self.events,
            "predicates": self.predicates,
            "predicate_visits": len(self.predicates),
            "result": self.result,
        }


def state_loop(required, cap=None, *, use_for=False, blind_success=False):
    """State/Continue are ready, local, and changed only by compute:State.

    cap in a while run is an ORIGINAL MaxIter, not an invented safety budget.
    cap=None models an original loop with no MaxIter. blind_success deliberately
    models the bad transformation that truncates and still reports success.
    """
    trace = Trace()
    state = {"completed": 0, "continue": required > 0}
    trace.action("load:InitialState")

    def body(r):
        def update():
            state["completed"] += 1
            state["continue"] = state["completed"] < required

        trace.action(f"compute:State({r})", update)
        trace.action(f"signal:Progress({r})")

    if use_for:
        assert cap is not None
        for r in range(cap):
            if trace.predicate(r, state["continue"]):
                body(r)
    else:
        r = 0
        while trace.predicate(
            r, (cap is None or r < cap) and state["continue"]
        ):
            body(r)
            r += 1

    trace.result = {
        "completed": state["completed"],
        "status": "success"
        if blind_success or not state["continue"]
        else "exhausted",
        "postcondition_holds": not state["continue"],
    }
    trace.action("store:Outcome")
    trace.action("release:Workspace")
    return trace


def nonabsorbing_flags(use_for):
    """All flag values already exist; no hidden read or predicate mutation."""
    flags = [True, False, True, False]
    trace = Trace()
    trace.action("load:Flags")
    if use_for:
        for r in range(3):
            if trace.predicate(r, flags[r]):
                trace.action(f"compute:State({r})")
    else:
        r = 0
        while trace.predicate(r, flags[r]):
            trace.action(f"compute:State({r})")
            r += 1
    trace.action("store:Outcome")
    return trace


def undefined_future_condition(use_for):
    """Only executing body(r) creates Continue(r+1)."""
    versions = {0: True}
    trace = Trace()
    trace.action("load:InitialState")

    def predicate(r):
        if r not in versions:
            trace.predicates.append({"round": r, "value": "UNDEFINED"})
            raise KeyError(r)
        return trace.predicate(r, versions[r])

    def body(r):
        trace.action(
            f"compute:State({r})", lambda: versions.update({r + 1: False})
        )

    try:
        if use_for:
            for r in range(3):
                if predicate(r):
                    body(r)
        else:
            r = 0
            while predicate(r):
                body(r)
                r += 1
        trace.action("store:Outcome")
        trace.result = {"status": "success"}
    except KeyError as error:
        trace.result = {
            "status": "undefined_predicate",
            "missing_round": error.args[0],
        }
    return trace


def duplicate_final_bad():
    """Two sibling if nodes are not an if/else or a once-only exit hook."""
    trace = Trace()
    state = {"continue": True}
    trace.action("load:InitialState")
    for r in range(3):
        if trace.predicate(r, state["continue"]):
            trace.action(
                f"compute:State({r})",
                lambda: state.update({"continue": False}),
            )
        if trace.predicate(r, not state["continue"]):
            trace.action("store:Outcome")
            trace.action("release:Workspace")
    return trace


def study():
    bounded_cases = 0
    guaranteed_bound_cases = 0
    for cap in range(9):
        for required in range(cap + 4):
            source = state_loop(required, cap)
            target = state_loop(required, cap, use_for=True)
            assert source.events == target.events
            assert source.result == target.result
            bounded_cases += 1
            if required <= cap:
                natural_source = state_loop(required)
                assert natural_source.events == target.events
                assert natural_source.result == target.result
                guaranteed_bound_cases += 1

    flags_source, flags_target = nonabsorbing_flags(False), nonabsorbing_flags(True)
    assert flags_source.events != flags_target.events
    assert "issue compute:State(2)" not in flags_source.events
    assert "issue compute:State(2)" in flags_target.events

    versions_source = undefined_future_condition(False)
    versions_target = undefined_future_condition(True)
    assert versions_source.result["status"] == "success"
    assert versions_target.result == {
        "status": "undefined_predicate",
        "missing_round": 2,
    }

    cap_source = state_loop(3)
    cap_target = state_loop(3, 2, use_for=True, blind_success=True)
    assert cap_source.result["postcondition_holds"]
    assert not cap_target.result["postcondition_holds"]
    assert cap_target.result["status"] == "success"

    duplicate = duplicate_final_bad()
    assert duplicate.events.count("issue release:Workspace") == 3

    return {
        "model": "synthetic_serial_cpu_issue_done_traces",
        "limitations": [
            "No GPU, compiler, concurrency, or timing claims.",
            "Finite examples supplement the report's conditional proof.",
            "Predicates are ready pure values; updates belong to named actions.",
            "The wrong rewrites are intentional semantic counterexamples.",
        ],
        "bounded_equivalence": {
            "original_maxiter_cases": bounded_cases,
            "guaranteed_bound_cases": guaranteed_bound_cases,
            "all_action_traces_and_outcomes_match": True,
            "early_exit_example": {
                "bound": 5,
                "successful_iterations": 1,
                "source": state_loop(1, 5).as_dict(),
                "target": state_loop(1, 5, use_for=True).as_dict(),
            },
        },
        "counterexamples": {
            "false_then_true": {
                "source": flags_source.as_dict(),
                "bad_target": flags_target.as_dict(),
            },
            "undefined_future_condition": {
                "source": versions_source.as_dict(),
                "bad_target": versions_target.as_dict(),
            },
            "fabricated_cap_reports_success": {
                "source": cap_source.as_dict(),
                "bad_target": cap_target.as_dict(),
            },
            "duplicate_final_after_early_stop": {
                "bad_target": duplicate.as_dict(),
                "release_issue_count": 3,
            },
        },
    }


if __name__ == "__main__":
    print(json.dumps(study(), ensure_ascii=False, indent=2))
