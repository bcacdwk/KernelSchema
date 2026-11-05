"""Two bounded fixtures for root completeness, delegation and legacy bindings."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from schema_extract.tools import observation
from schema_extract.tools.audit import (_Evidence, _verify_attempt, parse_events,
                                        result_usage, usage_summary)
from schema_extract.tools.common import ExtractError, digest, json_bytes, write_json


def tokens(inputs, cached, outputs):
    return dict(input_tokens=inputs, cached_input_tokens=cached, output_tokens=outputs)


def event(kind, **body):
    return dict(type="event_msg", payload=dict(type=kind, **body))


def lines(events):
    return b"".join(json.dumps(item).encode() + b"\n" for item in events)


class UsageObservationTests(unittest.TestCase):
    @patch.object(observation.time, "time", return_value=1704067202)
    def test_bounded_native_delegation_preserves_root_delta_and_audit_binding(self, _time):
        session = "11111111-1111-4111-8111-111111111111"
        child = "22222222-2222-4222-8222-222222222222"
        parent = "33333333-3333-4333-8333-333333333333"
        private = "SECRET_BODY https://private.invalid/error?secret=value"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "attempt"
            workspace = directory / "workspace"
            workspace.mkdir(parents=True)
            attempt = dict(run_id="run", attempt_id="attempt", session_id=session,
                           input_fingerprint="fixed", directory=str(directory), workspace=str(workspace))
            invocation = dict(run_id="run", attempt_id="attempt", launch_id="launch", started_at=1704067200,
                              attempt_sha256=digest(attempt), cli_version="codex 0.166.0", delegation_policy="forbidden")
            native = root / "native" / "sessions" / "2024" / "01" / "01" / ("rollout-" + session + ".jsonl")
            native.parent.mkdir(parents=True)
            baseline = lines([dict(type="session_meta", payload=dict(id=session, cwd=str(workspace),
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
            parsed = parse_events(directory / "events.jsonl")
            result = dict(**invocation, session_id=session, completed_at=1704067202,
                          outcome="completed", termination={"confirmed": True})
            good = None
            for mode in ("observed", "absent", "bad_uuid", "unsupported_tool", "open"):
                with self.subTest(mode=mode):
                    native.write_bytes(baseline)
                    start = observation.begin_usage_observation(attempt, invocation, home=root / "native")
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
            # Audit reads only the persisted projection, not native history. Both
            # old and new optional-field layouts retain their original bindings.
            (directory / "prompt.txt").write_text("fixture")
            write_json(directory / "attempt.json", attempt)
            for legacy in (False, True):
                bound, launch = deepcopy(good), deepcopy(invocation)
                if legacy:
                    bound.pop("delegation")
                    launch.pop("delegation_policy")
                closed = {**result, **launch, "trace": parsed["trace"],
                          "usage": result_usage(parsed, outcome="completed", termination=result["termination"]),
                          "usage_observation_sha256": digest(bound)}
                if legacy:
                    closed.pop("delegation_policy")
                write_json(directory / "invocation.json", launch)
                write_json(directory / "result.json", closed)
                write_json(directory / "usage-observation.json", bound)
                row = dict(origin_run_id="run", attempt_id="attempt", directory=str(directory),
                           usage=usage_summary(closed["usage"]), usage_observation=observation.observation_summary(bound))
                files = _Evidence(root, False)
                try:
                    _verify_attempt(None, dict(input_fingerprint="fixed"), row, files, [])
                finally:
                    files.close()
                if not legacy:
                    write_json(directory / "invocation.json", {**launch, "delegation_policy": "unknown"})
                    files = _Evidence(root, False)
                    try:
                        with self.assertRaisesRegex(ExtractError, "usage observation differs"):
                            _verify_attempt(None, dict(input_fingerprint="fixed"), row, files, [])
                    finally:
                        files.close()

    def test_totals_classify_disjoint_origins_and_preserve_legacy_summary(self):
        old = dict(format=observation.FORMAT, identity={"attempt_id": "old"}, session_id="session", turn_id="turn",
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
        unknown = dict(origin_run_id="origin", attempt_id="historical")
        inherited = {**deepcopy(exact), "inherited": True}
        total = observation.observation_totals([exact, partial, unknown, inherited])
        self.assertEqual(total["additive_tokens"], tokens(20, 4, 6))
        self.assertEqual((total["root_status"], total["exact_attempts"], total["partial_attempts"],
                          total["unknown_attempts"], total["known_attempts"], total["duplicate_attempts"]),
                         ("partial", 1, 1, 1, 2, 1))
        self.assertEqual(total["delegation"]["attempt_status"]["unavailable"], 3)
        self.assertEqual(total["delegation"]["policy_counts"]["unknown"], 3)
        self.assertEqual(observation.observation_totals([exact, inherited])["root_status"], "exact")
        overlap = {**deepcopy(exact), "attempt_id": "overlap"}
        rejected = observation.observation_totals([exact, overlap])
        self.assertEqual((rejected["known_attempts"], rejected["unknown_attempts"], rejected["conflicting_attempts"]), (0, 2, 2))
        for rows in ([], [unknown], [exact], [exact, partial, inherited]):
            value = observation.observation_totals(rows)
            self.assertIsNone(value["whole_invocation_tokens"])
            self.assertIsNone(value["cost"])
        self.assertEqual(json_bytes(observation.observation_summary(old)), before)


if __name__ == "__main__":
    unittest.main()
