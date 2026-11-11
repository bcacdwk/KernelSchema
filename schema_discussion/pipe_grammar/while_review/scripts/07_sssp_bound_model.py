#!/usr/bin/env python3
"""CPU model for reports/07_sssp_bound.md; never executes Galois or GPU code.

A frontier scan has separate guard, atomicMin, and classification source reads.
Every edge task completes before the next wave. Ordinary reads are coherent;
each atomicMin is linearized. Queues are multisets, with dedup only at the outer
boundary. Task interleavings are explored within this abstract model, not all CUDA executions.
The exact search is finite small-input evidence, not the general bound proof.
"""

import argparse
import itertools
import json
import random
from collections import Counter
from functools import lru_cache

INF = 2**31 - 1


def shortest(n, edges):
    d = [0] + [INF] * (n - 1)
    for _ in range(n - 1):
        for u, v, w in edges:
            if d[u] != INF:
                d[v] = min(d[v], d[u] + w)
    return tuple(d)


def jobs_for(edges, frontier):
    return tuple(sorted((u, v, w, 0) for x in frontier
                        for u, v, w in edges if u == x))


def step(state, index, delta, canonical_queues=True):
    """One coherent read/operation stage; source is deliberately read anew."""
    d, jobs, retry, deferred = state
    u, v, w, pc = jobs[index]
    assert d[u] != INF and d[u] + w <= INF, "model arithmetic contract"
    c = d[u] + w
    rest = list(jobs)
    rest.pop(index)
    new_d, new_retry, new_deferred = d, retry, deferred
    if pc == 0:
        if d[v] > c:
            rest.append((u, v, w, 1))
    elif pc == 1:
        a = list(d)
        a[v] = min(a[v], c)
        new_d = tuple(a)
        rest.append((u, v, w, 2))
    else:
        if c <= delta:
            new_retry = retry + (v,)
            if canonical_queues:
                new_retry = tuple(sorted(new_retry))
        else:
            new_deferred = deferred + (v,)
            if canonical_queues:
                new_deferred = tuple(sorted(new_deferred))
    event = {"edge": [u, v, w], "stage": ["guard", "atomic", "classify"][pc],
             "candidate": c, "destination_before": d[v],
             "destination_after": new_d[v]}
    if pc == 2:
        event["queue"] = "re" if c <= delta else "out"
    return (new_d, tuple(sorted(rest)), new_retry, new_deferred), event


def assert_clean(n, edges, d, frontier, delta=None):
    active = set(frontier)
    for u, v, w in edges:
        if u not in active and d[u] != INF:
            if delta is None or d[u] + w <= delta:
                assert d[v] <= d[u] + w, ("unclean inactive source", d, frontier,
                                          delta, (u, v, w))


def exact_case(n, edges, delta_step):
    explored_microstates = 0
    max_pushes = 0

    @lru_cache(maxsize=None)
    def wave(d, frontier, delta):
        nonlocal explored_microstates, max_pushes
        initial = (d, jobs_for(edges, frontier), (), ())
        todo, seen, outputs = [initial], {initial}, set()
        while todo:
            state = todo.pop()
            ds, jobs, retry, deferred = state
            if not jobs:
                outputs.add((ds, retry, deferred))
                max_pushes = max(max_pushes, len(retry), len(deferred))
                continue
            previous = None
            for j, job in enumerate(jobs):
                if job == previous:
                    continue
                previous = job
                successor, _ = step(state, j, delta)
                if successor not in seen:
                    seen.add(successor)
                    todo.append(successor)
        explored_microstates += len(seen)
        return tuple(outputs)

    initial = ((0,) + (INF,) * (n - 1), (0,), (), 0, 0)
    todo, seen, terminals = [initial], {initial}, set()
    max_inner = max_outer = 0
    h = (n - 1) * max((w for _, _, w in edges), default=0)
    while todo:
        d, frontier, deferred, outer, inner = todo.pop()
        delta = outer * delta_step
        assert frontier and inner < n and outer < n, "V bound violated"
        assert_clean(n, edges, d, frontier, delta)
        if inner == 0:
            assert_clean(n, edges, d, frontier)
        for ds, retry, appended in wave(d, frontier, delta):
            assert all(x == INF or 0 <= x <= h for x in ds)
            nr = inner + 1
            max_inner = max(max_inner, nr)
            od = tuple(sorted(deferred + appended))
            if retry:
                successor = (ds, retry, od, outer, nr)
            else:
                nb = outer + 1
                max_outer = max(max_outer, nb)
                if not od:
                    assert ds == shortest(n, edges)
                    assert nb <= min(n, 1 + (h + delta_step - 1) // delta_step)
                    terminals.add((ds, nb, nb * delta_step))
                    continue
                successor = (ds, tuple(sorted(set(od))), (), nb, 0)
            if successor not in seen:
                seen.add(successor)
                todo.append(successor)
    return {"max_inner": max_inner, "max_outer": max_outer,
            "macro_states": len(seen), "micro_states": explored_microstates,
            "terminal_states": len(terminals), "max_single_wave_pushes": max_pushes}


def sampled_wave(edges, d, frontier, delta, rng, schedule=None):
    state = (d, jobs_for(edges, frontier), (), ())
    events = []
    for choice in schedule or ():
        index = state[1].index(choice)
        state, event = step(state, index, delta, canonical_queues=False)
        events.append(event)
    while state[1]:
        state, event = step(state, rng.randrange(len(state[1])), delta,
                            canonical_queues=False)
        events.append(event)
    return (state[0], state[2], state[3]), events


def sampled_run(n, edges, delta_step, seed, control="while"):
    """Compare condition-load and queue/counter protocol traces.

    This models the outer protocol explicitly, including the three physical
    queues and two counter slots. It does not model per-CTA barrier internals.
    """
    rng = random.Random(seed)
    d = (0,) + (INF,) * (n - 1)
    roles, slots = [0, 1, 2], [0, 0, 0]  # in, out, re
    counters, payload = [[1, 0], [0, 0], [0, 0]], [[0], [], []]
    i = delta = 0
    waves = []
    events = []
    reads = Counter()
    h = min(INF - 1, (n - 1) * max((w for _, _, w in edges), default=0))

    def record(name, *detail):
        events.append((name, detail))

    def queue(role):
        q = roles[role]
        return tuple(payload[q][:counters[q][slots[q]]])

    def append(role, items):
        q = roles[role]
        length = counters[q][slots[q]]
        end = length + len(items)
        if end > len(payload[q]):
            payload[q].extend([None] * (end - len(payload[q])))
        payload[q][length:end] = items  # counter resets do not clear old payload
        counters[q][slots[q]] += len(items)

    def load(kind):
        q = roles[0]
        count = counters[q][slots[q]]
        record("load_" + kind, q, slots[q], count)
        reads[kind] += 1
        return count

    def reset():
        q = roles[0]
        counters[q][1 ^ slots[q]] = 0
        record("reset_next_slot", q, 1 ^ slots[q])

    def flip():
        q = roles[0]
        slots[q] ^= 1
        record("swap_slots", q, slots[q])

    def swap(a, b, name):
        roles[a], roles[b] = roles[b], roles[a]
        record(name, tuple(roles))

    def body(b, r):
        nonlocal d
        frontier = queue(0)
        assert_clean(n, edges, d, frontier, delta)
        reset()
        (d, retry, deferred), micro = sampled_wave(edges, d, frontier, delta, rng)
        assert all(x == INF or 0 <= x <= h for x in d)
        assert all(e["candidate"] <= h for e in micro if e["stage"] == "classify")
        append(1, deferred)
        append(2, retry)
        record("relax", b, r, frontier, tuple(tuple(sorted(e.items())) for e in micro),
               d, retry, deferred)
        flip()
        record("gb_relax", b, r)
        swap(0, 2, "retry2")
        return load("inner")

    def epoch(b):
        nonlocal i, delta
        assert_clean(n, edges, d, queue(0))
        inner_count = load("inner")
        r = 0
        if control == "while":
            while inner_count:
                assert r < n, "inner V bound violated"
                inner_count = body(b, r)
                r += 1
        else:
            for rr in range(n):
                if inner_count:
                    inner_count = body(b, rr)
                    r += 1
        assert not inner_count, "exhausting a bound is not normal exit"
        waves.append(r)
        record("gb_retry_drain", b)
        swap(0, 1, "advance2_deferred")
        reset()
        source = queue(0)
        marks = {v: j for j, v in enumerate(source)}
        record("mark_representatives", source, tuple(sorted(marks.items())))
        record("gb_marks", b)
        unique = tuple(v for j, v in enumerate(source) if marks[v] == j)
        append(1, unique)
        record("publish_unique", unique)
        flip()
        record("gb_unique", b)
        swap(0, 1, "advance2_unique")
        i += 1
        delta += delta_step
        record("advance_delta", i, delta)
        return load("outer")

    outer_count = load("outer")
    if control == "while":
        while outer_count:
            assert i < n, "outer V bound violated"
            outer_count = epoch(i)
    else:
        for b in range(n):
            if outer_count:
                outer_count = epoch(b)
    assert not outer_count, "exhausting a bound is not normal exit"
    assert i <= min(n, 1 + (h + delta_step - 1) // delta_step)
    record("gb_final")
    record("store_cl_curdelta_cl_i", delta, i)
    final = {"distances": d, "i": i, "curdelta": delta,
             "roles": tuple(roles), "slots": tuple(slots),
             "counters": tuple(tuple(c) for c in counters),
             "payload": tuple(tuple(p) for p in payload)}
    assert d == shortest(n, edges)
    return {"final": final, "waves": waves, "events": events, "reads": dict(reads)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--random-cases", type=int, default=1000)
    args = parser.parse_args()
    result = {"gpu_run": False, "model": "coherent three-stage edge tasks",
              "limitations": ["CPU model, not GPU memory-model verification",
                              "finite enumeration does not prove a universal bound",
                              "unbounded queue storage in algorithm model",
                              "Only the specified three-stage model is enumerated; no coverage claim for all CUDA executions"]}
    witnesses = [
        ("zero_edge_requires_two_inner_waves", 2, ((0, 1, 0),), 1),
        ("positive_edge_requires_two_outer_epochs", 2, ((0, 1, 1),), 1),
        ("huge_weight_does_not_wait_for_threshold", 2, ((0, 1, 10**9),), 1),
        ("parallel_zero_edges_can_duplicate_notification", 2,
         ((0, 1, 0), (0, 1, 0)), 1),
        ("classification_reread_graph", 4,
         ((0, 1, 2), (0, 2, 1), (2, 1, 0), (1, 3, 0)), 1),
    ]
    result["witnesses"] = []
    for name, n, edges, delta_step in witnesses:
        a = sampled_run(n, edges, delta_step, 42)
        b = sampled_run(n, edges, delta_step, 42, "for")
        assert a == b
        result["witnesses"].append({"id": name, "vertices": n, "edges": edges,
                                    "delta": delta_step, "waves": a["waves"],
                                    "final": a["final"], "reads": a["reads"]})
    # A real three-read hazard: the atomic writes 2 but classification reads 1.
    graph = witnesses[-1][2]
    schedule = [(1, 3, 0, 0), (1, 3, 0, 1),
                (2, 1, 0, 0), (2, 1, 0, 1), (2, 1, 0, 2),
                (1, 3, 0, 2)]
    output, trace = sampled_wave(graph, (0, 2, 1, INF), (1, 2), 1,
                                 random.Random(1), schedule)
    assert output == ((0, 1, 1, 2), (1, 3), ())
    result["above_threshold_retry_witness"] = {"delta": 1, "entry": [0, 2, 1, INF],
                                                 "frontier": [1, 2],
                                                 "output": output, "trace": trace}
    parallel = ((0, 1, 0), (0, 1, 0))
    schedule = [(0, 1, 0, 0)] * 2 + [(0, 1, 0, 1)] * 2 + [(0, 1, 0, 2)] * 2
    output, trace = sampled_wave(parallel, (0, INF), (0,), 0,
                                 random.Random(1), schedule)
    assert output == ((0, 0), (1, 1), ())
    result["failed_atomic_still_enqueues_witness"] = {"output": output, "trace": trace}
    result["targeted_exact_interleavings"] = {
        "classification_reread_graph": exact_case(4, graph, 1),
        "parallel_zero_edges": exact_case(2, parallel, 1)}

    rng = random.Random(20261111)
    max_outer = max_inner = 0
    for case in range(args.random_cases):
        n = rng.randrange(2, 13)
        edges = tuple((u, v, rng.randrange(10)) for u in range(n) for v in range(n)
                      if u != v and rng.random() < 0.25)
        if not edges:
            edges = ((0, 1, 0),)
        delta_step = rng.randrange(1, 11)
        seed = rng.randrange(2**32)
        a = sampled_run(n, edges, delta_step, seed)
        b = sampled_run(n, edges, delta_step, seed, "for")
        assert a == b, "while/for protocol trace mismatch"
        max_outer = max(max_outer, len(a["waves"]))
        max_inner = max(max_inner, max(a["waves"]))
    result["random_protocol_comparison"] = {"cases": args.random_cases,
                                            "max_vertices": 12,
                                            "seed": 20261111,
                                            "max_observed_outer": max_outer,
                                            "max_observed_inner": max_inner,
                                            "full_modeled_trace_and_state_equal": True}
    if args.exact:
        totals = Counter()
        maxima = Counter()
        pairs = tuple((u, v) for u in range(3) for v in range(3) if u != v)
        for choices in itertools.product((None, 0, 1), repeat=len(pairs)):
            edges = tuple((u, v, w) for (u, v), w in zip(pairs, choices) if w is not None)
            if not edges:  # source's allocation 2E cannot hold the initial item
                continue
            for delta_step in (1, 2):
                case = exact_case(3, edges, delta_step)
                totals["graph_parameter_cases"] += 1
                for key in ("macro_states", "micro_states", "terminal_states"):
                    totals[key] += case[key]
                for key in ("max_inner", "max_outer", "max_single_wave_pushes"):
                    maxima[key] = max(maxima[key], case[key])
        result["exhaustive_three_vertex_search"] = {
            "graphs": 728, "weights": [0, 1], "deltas": [1, 2],
            "self_edges": False, "parallel_edges": False,
            "all_model_microstage_interleavings": True, "complete": True,
            **dict(totals), **dict(maxima)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
