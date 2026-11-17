"""Native accounting scopes, bounded Codex evidence, and attempt deduplication."""

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from schema_extract.tools import observation
from schema_extract.tools.audit import (_Evidence, _verify_attempt, parse_events,
                                        result_usage, usage_summary, usage_totals)
from schema_extract.tools.common import ExtractError, digest, json_bytes, write_json


def tokens(inputs, cached, outputs):
    return dict(input_tokens=inputs, cached_input_tokens=cached, output_tokens=outputs)


def event(kind, **body):
    return dict(type="event_msg", payload=dict(type=kind, **body))


def lines(events):
    return b"".join(json.dumps(item).encode() + b"\n" for item in events)


def native_context(root, session_id):
    directory = root / "attempt"
    workspace = directory / "workspace"
    workspace.mkdir(parents=True)
    attempt = dict(backend="codex", run_id="run", attempt_id="attempt", session_id=session_id,
                   input_fingerprint="fixed", directory=str(directory), workspace=str(workspace),
                   session_directory=str(root / "role-session"),
                   session_scope=dict(stage="P1", node="root", role="worker", operation="normal"))
    invocation = {**{key: attempt[key] for key in ("backend", "run_id", "attempt_id", "session_directory", "session_scope")},
                  "launch_id": "launch", "started_at": 1704067200, "attempt_sha256": digest(attempt),
                  "cli_version": "codex 0.166.0", "delegation_policy": "forbidden", "cwd": str(Path(attempt["session_directory"]) / "cwd")}
    return attempt, invocation, Path(attempt["session_directory"]) / "home/.codex"


class UsageObservationTests(unittest.TestCase):
    @patch.object(observation.time, "time", return_value=1704067202)
    def test_bounded_native_delegation_preserves_root_delta_and_audit_binding(self, _time):
        session = "11111111-1111-4111-8111-111111111111"
        child = "22222222-2222-4222-8222-222222222222"
        parent = "33333333-3333-4333-8333-333333333333"
        private = "SECRET_BODY https://private.invalid/error?secret=value"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt, invocation, home = native_context(root, session)
            directory = Path(attempt["directory"])
            native = home / "sessions/2024/01/01" / ("rollout-" + session + ".jsonl")
            native.parent.mkdir(parents=True)
            baseline = lines([dict(type="session_meta", payload=dict(id=session, cwd=invocation["cwd"],
                cli_version="0.166.0", parent_thread_id=parent, source={"private": private})),
                event("token_count", info=dict(total_token_usage=tokens(100, 20, 10), last_token_usage=tokens(100, 20, 10)))])
            count = event("token_count", info=dict(total_token_usage=tokens(160, 30, 16), last_token_usage=tokens(60, 10, 6)))
            markers = [dict(type="response_item", payload=dict(type="function_call", name="spawn_agent", arguments=private)),
                       event("item_completed", item=dict(type="SubAgentActivity", agent_thread_id=child,
                                                         agent_path=private, model=private, kind=private)),
                       event("item_completed", item=dict(type="CollabAgentToolCall", tool="wait", status="completed",
                           sender_thread_id=session, receiver_thread_ids=[child], receiver_agents=private, agents_states=private))]
            (directory / "events.jsonl").write_bytes(lines([dict(type="thread.started", thread_id=session),
                dict(type="turn.completed", usage=tokens(180, 35, 20))]))
            parsed = parse_events(directory / "events.jsonl", backend="codex")
            result = dict(**invocation, session_id=session, completed_at=1704067202,
                          outcome="completed", termination={"confirmed": True})
            good = None
            for mode in ("observed", "absent", "bad_uuid", "unsupported_tool", "open"):
                with self.subTest(mode=mode):
                    native.write_bytes(baseline)
                    start = observation.begin_usage_observation(attempt, invocation, home=home)
                    selected = deepcopy(markers) if mode != "absent" else []
                    if mode == "bad_uuid":
                        selected[1]["payload"]["item"]["agent_thread_id"] = private
                    if mode == "unsupported_tool":
                        selected[2]["payload"]["item"]["tool"] = private
                    records = [event("task_started", turn_id="turn"), count, count, *selected,
                        event("token_count", info=dict(total_token_usage=tokens(180, 35, 20), last_token_usage=tokens(20, 5, 4)))]
                    if mode != "open":
                        records.append(event("task_complete", turn_id="turn"))
                    native.write_bytes(baseline + lines(records))
                    observed = observation.finish_usage_observation(start, parsed, result)
                    self.assertEqual(observed["additive_tokens"], tokens(80, 15, 10))
                    self.assertEqual(observed["delta_count"], 2)
                    self.assertEqual(observed["root_status"], "partial" if mode == "open" else "exact")
                    expected = "observed" if mode == "observed" else "not_observed_in_range" if mode == "absent" else "unavailable"
                    self.assertEqual(observed["delegation"]["status"], expected)
                    self.assertNotIn(private, json.dumps(observed))
                    self.assertEqual(observed["native"]["metadata"]["parent_thread_id"], parent)
                    if mode == "observed":
                        good = observed
                        self.assertEqual(observed["delegation"]["records"][1]["agent_thread_id"], child)
                        self.assertEqual(observed["delegation"]["records"][2]["receiver_thread_ids"], [child])
                        summary = observation.observation_summary(observed)
                        self.assertNotIn("records", summary["delegation"])
                        self.assertEqual(summary["delegation"]["record_count"], 3)
                        self.assertEqual(summary["delegation"]["related_thread_count"], 1)
                        self.assertNotIn(child, json.dumps(summary))
                    total = observation.observation_totals([dict(origin_run_id="run", attempt_id="attempt",
                                                               usage_observation=observation.observation_summary(observed))])
                    self.assertIsNone(total["whole_invocation_tokens"])
                    self.assertIsNone(total["cost"])
            # This projection test supplies the separately tested native result check;
            # audit still binds observation bytes to their invocation and event stream.
            (directory / "prompt.txt").write_text("fixture")
            write_json(directory / "attempt.json", attempt)
            bound, launch = deepcopy(good), deepcopy(invocation)
            closed = {**result, **launch, "trace": parsed["trace"],
                      "usage": result_usage(parsed, outcome="completed", termination=result["termination"]),
                      "usage_observation_sha256": digest(bound)}
            write_json(directory / "invocation.json", launch)
            write_json(directory / "result.json", closed)
            write_json(directory / "usage-observation.json", bound)
            row = dict(origin_run_id="run", attempt_id="attempt", directory=str(directory),
                       usage=usage_summary(closed["usage"]), usage_observation=observation.observation_summary(bound))
            files = _Evidence(root, False)
            try:
                with patch("schema_extract.tools.recovery.verified_result", return_value=closed):
                    _verify_attempt(dict(input_fingerprint="fixed"), row, files, [])
            finally:
                files.close()
            write_json(directory / "invocation.json", {**launch, "delegation_policy": "unknown"})
            files = _Evidence(root, False)
            try:
                with self.assertRaisesRegex(ExtractError, "usage observation differs"):
                    _verify_attempt(dict(input_fingerprint="fixed"), row, files, [])
            finally:
                files.close()

    def test_real_codex_home_indexes_new_and_resumed_root_counters(self):
        session = "11111111-1111-4111-8111-111111111111"
        with tempfile.TemporaryDirectory() as temporary:
            attempt, invocation, home = native_context(Path(temporary), None)
            relative = Path("sessions/2024/01/01") / ("rollout-" + session + ".jsonl")
            native = home / relative
            native.parent.mkdir(parents=True)
            with sqlite3.connect(home / "state_5.sqlite") as db:
                db.execute("CREATE TABLE threads (id TEXT, rollout_path TEXT)")
                db.execute("INSERT INTO threads VALUES (?, ?)", (session, str(home / relative)))
            previous = tokens(0, 0, 0)
            for turn, cumulative, delta in (("first", tokens(100, 20, 10), tokens(100, 20, 10)),
                                            ("resumed", tokens(160, 30, 16), tokens(60, 10, 6))):
                with self.subTest(turn=turn):
                    if turn == "resumed":
                        attempt.update(attempt_id="resumed", session_id=session)
                        invocation.update(attempt_id="resumed", launch_id="resumed", attempt_sha256=digest(attempt))
                    start = observation.begin_usage_observation(attempt, invocation, home=home)
                    records = ([] if native.exists() else [dict(type="session_meta", payload=dict(id=session, cwd=invocation["cwd"]))])
                    records += [event("task_started", turn_id=turn),
                                event("token_count", info=dict(total_token_usage=cumulative, last_token_usage=delta)),
                                event("task_complete", turn_id=turn)]
                    with native.open("ab") as stream:
                        stream.write(lines(records))
                    events = Path(attempt["directory"]) / "events.jsonl"
                    events.write_bytes(lines([dict(type="thread.started", thread_id=session),
                                             dict(type="turn.completed", usage=cumulative)]))
                    result = {**invocation, "session_id": session, "outcome": "completed", "termination": {"confirmed": True}}
                    observed = observation.finish_usage_observation(start, parse_events(events, backend="codex"), result)
                    self.assertEqual((observed["root_status"], observed["status"], observed["cursor_valid"]), ("exact", "partial", True))
                    self.assertEqual(observed["baseline_tokens"], previous)
                    self.assertEqual(observed["additive_tokens"], delta)
                    self.assertEqual(observed["native"]["path"], str(native))
                    total = observation.observation_totals([dict(run_id="run", attempt_id=turn,
                                                               usage_observation=observation.observation_summary(observed))])
                    self.assertIsNone(total["whole_invocation_tokens"])
                    self.assertIsNone(total["cost"])
                    previous = cumulative

    def test_native_index_rejects_paths_identities_and_other_scopes(self):
        session = "11111111-1111-4111-8111-111111111111"
        other = "22222222-2222-4222-8222-222222222222"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt, invocation, home = native_context(root, None)
            relative = Path("sessions/2024/01/01") / ("rollout-" + session + ".jsonl")
            native = home / relative
            native.parent.mkdir(parents=True)
            counts = tokens(100, 20, 10)
            contents = lines([dict(type="session_meta", payload=dict(id=session, cwd=invocation["cwd"])),
                              event("task_started", turn_id="first"),
                              event("token_count", info=dict(total_token_usage=counts, last_token_usage=counts)),
                              event("task_complete", turn_id="first")])
            native.write_bytes(contents)
            indexed = home / relative
            with sqlite3.connect(home / "state_5.sqlite") as db:
                db.execute("CREATE TABLE threads (id TEXT, rollout_path TEXT)")
                db.execute("INSERT INTO threads VALUES (?, ?)", (session, str(indexed)))
            events = Path(attempt["directory"]) / "events.jsonl"
            events.write_bytes(lines([dict(type="thread.started", thread_id=session), dict(type="turn.completed", usage=counts)]))
            parsed = parse_events(events, backend="codex")
            result = {**invocation, "session_id": session, "outcome": "completed", "termination": {"confirmed": True}}
            for path in (root / relative, indexed.parent / ".." / indexed.name,
                         indexed.with_name("rollout-" + other + ".jsonl"), Path("/home/other/.codex") / relative):
                with self.subTest(index_path=path):
                    with sqlite3.connect(home / "state_5.sqlite") as db:
                        db.execute("UPDATE threads SET rollout_path=?", (str(path),))
                    start = observation.begin_usage_observation(attempt, invocation, home=home)
                    observed = observation.finish_usage_observation(start, parsed, result)
                    self.assertIn("native_index_identity_conflict", observed["issues"])
                    self.assertIsNone(observed["additive_tokens"]["input_tokens"])
            with sqlite3.connect(home / "state_5.sqlite") as db:
                db.execute("UPDATE threads SET rollout_path=?", (str(indexed),))
            for mode in ("metadata", "symlink", "home", "scope", "result_scope"):
                with self.subTest(mode=mode):
                    if mode == "metadata":
                        native.write_bytes(contents.replace(session.encode(), other.encode(), 1))
                    if mode == "symlink":
                        outside = root / "outside.jsonl"
                        outside.write_bytes(contents)
                        native.unlink()
                        native.symlink_to(outside)
                    scoped = {**invocation, "session_scope": {**attempt["session_scope"], "role": "reviewer"}}
                    start = observation.begin_usage_observation(attempt, scoped if mode == "scope" else invocation,
                                home=root if mode == "home" else home)
                    observed = observation.finish_usage_observation(start, parsed,
                                {**result, "session_scope": scoped["session_scope"]} if mode == "result_scope" else result)
                    expected = ("native_session_identity_conflict" if mode == "metadata" else
                                "native_path_contains_symlink" if mode == "symlink" else "native_scope_identity_conflict")
                    self.assertIn(expected, observed["issues"])
                    self.assertEqual(observed["root_status"], "unavailable")
                    self.assertIsNone(observed["additive_tokens"]["input_tokens"])
                    if native.is_symlink():
                        native.unlink()
                    native.write_bytes(contents)

    def test_totals_deduplicate_disjoint_codex_observations(self):
        old = dict(format=observation.FORMAT, backend="codex", identity={"attempt_id": "old"}, session_id="session", turn_id="turn",
                   status="partial", root_status="exact", scope="root", source="native", cursor_valid=True,
                   additive_tokens=tokens(10, 2, 3), baseline_tokens=tokens(0, 0, 0), final_tokens=tokens(10, 2, 3),
                   issues=["child_session_coverage_unobserved"], observed_at=1)
        expected = {key: value for key, value in old.items() if key != "observed_at"}
        expected["sha256"] = digest(old)
        before = json_bytes(expected)
        self.assertEqual(json_bytes(observation.observation_summary(old)), before)
        self.assertNotIn("delegation", observation.observation_summary(old))
        exact = dict(origin_run_id="origin", attempt_id="first", usage_observation=observation.observation_summary(old))
        partial = deepcopy(exact)
        partial.update(attempt_id="second")
        partial["usage_observation"].update(root_status="partial", baseline_tokens=tokens(10, 2, 3),
                                            final_tokens=tokens(20, 4, 6))
        unknown = dict(origin_run_id="origin", attempt_id="unknown")
        duplicate = deepcopy(exact)
        total = observation.observation_totals([exact, partial, unknown, duplicate])
        self.assertEqual(total["additive_tokens"], tokens(20, 4, 6))
        self.assertEqual((total["root_status"], total["exact_attempts"], total["partial_attempts"],
                          total["unknown_attempts"], total["known_attempts"], total["duplicate_attempts"]),
                         ("partial", 1, 1, 1, 2, 1))
        self.assertEqual(total["delegation"]["attempt_status"]["unavailable"], 3)
        self.assertEqual(total["delegation"]["policy_counts"]["unknown"], 3)
        self.assertEqual(observation.observation_totals([exact, duplicate])["root_status"], "exact")
        overlap = {**deepcopy(exact), "attempt_id": "overlap"}
        rejected = observation.observation_totals([exact, overlap])
        self.assertEqual((rejected["known_attempts"], rejected["unknown_attempts"], rejected["conflicting_attempts"]), (0, 2, 2))
        for rows in ([], [unknown], [exact], [exact, partial, duplicate]):
            value = observation.observation_totals(rows)
            self.assertIsNone(value["whole_invocation_tokens"])
            self.assertIsNone(value["cost"])
        self.assertEqual(json_bytes(observation.observation_summary(old)), before)

    def test_claude_resume_uses_native_delta_and_never_sums_session_totals(self):
        session = "11111111-1111-4111-8111-111111111111"
        root_usage = dict(input_tokens=4, cache_read_input_tokens=12, cache_creation_input_tokens=8, output_tokens=3)
        delta = {"reported-model": dict(inputTokens=4, cacheReadInputTokens=12,
                                       cacheCreationInputTokens=8, outputTokens=3, costUSD=0.003)}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            def usage(*, cumulative, incremental=True, missing=False):
                terminal = dict(type="result", subtype="success", is_error=False, session_id=session,
                                structured_output={}, modelUsage={"reported-model": {"inputTokens": cumulative}},
                                total_cost_usd=cumulative, result_index=0)
                if not missing:
                    terminal["usage"] = root_usage
                if incremental:
                    terminal["model_usage_since_last_result"] = delta
                path.write_bytes(lines([dict(type="system", subtype="init", session_id=session), terminal]))
                return parse_events(path, backend="claude")["usage"]
            first, resumed = usage(cumulative=4), usage(cumulative=400)
            self.assertFalse(first["cached_input_is_input_subset"])
            self.assertNotIn("cached_input_exceeds_input", first["issues"])
            self.assertEqual(resumed["session_reported_cost_usd"], 400)
            self.assertEqual(resumed["reported_cost_usd"], 0.003)
            only_root, unknown = usage(cumulative=500, incremental=False), usage(cumulative=600, incremental=False, missing=True)
            self.assertIsNone(only_root["reported_cost_usd"])
            self.assertIsNone(unknown["additive_tokens"]["input_tokens"])
            rows = [dict(run_id="run", attempt_id=str(index), usage=usage_summary(item))
                    for index, item in enumerate((first, resumed, only_root, unknown))]
            total = usage_totals([*rows, deepcopy(rows[0])])
            self.assertEqual(total["duplicate_attempts"], 1)
            groups = {item["counter_scope"]: item for item in total["by_backend_and_scope"]}
            query = groups["query_pipeline_invocation_delta"]
            self.assertEqual(query["tokens"]["input_tokens"]["known_subtotal"], 8)
            self.assertEqual(query["tokens"]["cached_input_tokens"]["known_subtotal"], 24)
            self.assertEqual(query["tokens"]["cache_creation_input_tokens"]["known_subtotal"], 16)
            self.assertEqual(query["reported_cost_usd_subtotal"], 0.006)
            self.assertIsNone(query["billing_cost_usd"])
            self.assertEqual(groups["main_loop_invocation"]["tokens"]["input_tokens"]["unknown_attempts"], 1)
            conflicting = deepcopy(rows[0])
            conflicting["usage"]["additive_tokens"]["input_tokens"] = 100
            rejected = usage_totals([*rows, conflicting])
            self.assertEqual(rejected["conflicting_attempts"], 1)
            query = next(item for item in rejected["by_backend_and_scope"] if item["counter_scope"] == "query_pipeline_invocation_delta")
            self.assertEqual(query["tokens"]["input_tokens"]["known_subtotal"], 4)


if __name__ == "__main__":
    unittest.main()
