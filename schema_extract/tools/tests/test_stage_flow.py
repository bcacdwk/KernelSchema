"""Six-stage orchestration with real Template gates and a deterministic backend."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from jsonschema import validate

from schema_extract.tools.artifacts import (DESIGN_REGIONS, load_yaml, resolve_workspace,
                                            source_context, write_yaml)
from schema_extract.tools.common import ExtractError, StaleResult, manifest, read_json, rules_root, write_json
from schema_extract.tools.inputs import SourceTree
from schema_extract.tools.prompts import build_prompt
from schema_extract.tools.receipts import REOPEN_TARGETS, proposal_schema
from schema_extract.tools.runner import DEFAULT_CONFIG, _interrupted, _recover_call, accept_result, run
from schema_extract.tools.snapshots import freeze_inputs
from schema_extract.tools.stages import (affected_nodes, effective_nodes, next_calls, required_conclusions,
                                        scope_key, selection_signature, settle)
from schema_extract.tools.store import Store


REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "schema_template/tools/tests/fixtures/minimal"


class FixtureBackend:
    def __init__(self, lock, selected, writing):
        self.lock = lock
        self.rules = rules_root(lock)
        self.selected = deepcopy(selected)
        self.writing = deepcopy(writing)
        self.sessions = {}
        self.calls = []
        self.base = load_yaml(FIXTURE, "root.yaml", parser_root=REPO)
        starter = load_yaml(self.rules / "schema_template", "starter.yaml", parser_root=self.rules)
        execution = self.base["CONTRACT"]["Hardware_specification"]["Execution"]
        self.base["CONTRACT"]["Hardware_specification"] = deepcopy(starter["CONTRACT"]["Hardware_specification"])
        self.base["CONTRACT"]["Hardware_specification"]["Execution"] = execution
        prefix = lock["source"]["repository_id"] + "@" + lock["source"]["ref"] + ":kernel.py::"
        self.implementation = prefix + "copy"
        self.reference = prefix + "reference"
        self.base["CONTRACT"]["Algorithm_requirement"]["Golden_reference"]["Source"] = self.reference

    def index(self, workspace, *, final=False):
        rows = ["# 标量复制", "", "## 范围", "", "临时机器行为验证材料", "", "## 完整设计", "",
                "| 设计 | 具体机制与保留理由 | 关系 |", "| --- | --- | --- |"]
        for entry in self.selected:
            node = entry["id"]
            label = f"[{node}]({node}.yaml)" if final or node == "root" else f"`{node}.yaml`（仅计划）"
            rows.append(f"| {label} | 合成验证实例 {node} | 以实际整区引用为准 |")
        (workspace / "dispatch.md").write_text("\n".join(rows) + "\n")

    def write(self, attempt):
        directory, stage = Path(attempt["workspace"]), attempt["stage"]
        if stage == "P6":
            self.index(directory, final=True)
            return
        name = attempt["node"] + ".yaml"
        value = load_yaml(directory, name, parser_root=self.rules)
        if stage == "P1":
            value["META_DATA"].update(self.base["META_DATA"])
            value["META_DATA"]["Source"] = self.implementation
            value["META_DATA"]["Describe"] += "."
            value["CONTRACT"] = deepcopy(self.base["CONTRACT"])
        elif stage == "P2":
            value["META_DATA"].update(Version=2, Dispatch={"when": "All scalar inputs"})
            self.index(directory)
        elif stage == "P3":
            value["DESIGN"] = deepcopy(self.base["DESIGN"])
            value["META_DATA"]["Version"] = 3
        elif stage == "P4":
            self.index(directory)
            return
        elif stage == "P5":
            value["DESIGN"] = deepcopy(self.base["DESIGN"])
            value["META_DATA"].update(Source=self.implementation)
            value["META_DATA"]["Dispatch"]["when"] = "All scalar inputs"
            if attempt["node"] == "math":
                value["DESIGN"]["Work_schedule"] = {"from": "hidden"}
            elif attempt["node"] == "consumer" and {"math", "schedule"} <= {entry["id"] for entry in self.selected}:
                value["DESIGN"] = {"Math_reshape": {"from": "math"},
                                   "Work_schedule": {"from": "schedule"}, "Tile_pipeline": {"from": "root"}}
        write_yaml(directory / name, value, parser_root=self.rules)
        if stage == "P1":
            path = directory / name
            path.write_text(path.read_text().replace('"META_DATA":', '"meta_data":', 1))

    @staticmethod
    def anchor(path, line=1):
        return {"path": path, "line_start": line, "line_end": line, "quote_offset": 0, "note": "Current raw evidence"}

    def proposal(self, attempt):
        workspace = Path(attempt["workspace"])
        source = [{"category": category, **self.anchor("kernel.py", 3 if category == "reference" else 1)} for category in
                  ("implementation", "host", "io", "input", "reference", "acceptance", "scope")]
        artifacts = [self.anchor(name) for name in manifest(workspace)]
        if attempt["stage"] == "P5":
            resolved = resolve_workspace(workspace, parser_root=self.rules, nodes=[attempt["node"]])
            for node in resolved.raw_task:
                for region in DESIGN_REGIONS:
                    line = resolved.documents[node].positions[node + ".DESIGN." + region][0]
                    artifacts.append(self.anchor(node + ".yaml", line))
        decision = "complete" if attempt["role"] == "worker" else "approve"
        revise_directory = (attempt["stage"] == "P6" and attempt["role"] == "reviewer" and
                            "仅计划" in (workspace / "dispatch.md").read_text())
        if revise_directory:
            decision = "revise"
        value = {"format": "schema_extract.proposal.v1", "attempt_id": attempt["attempt_id"],
                 "decision": decision, "summary": "Deterministic fixture review", "source_evidence": source,
                 "artifact_evidence": artifacts,
                 "conclusions": [{"subject": subject, "text": "Checked the current fixture subject", "evidence": [0]}
                                 for subject in required_conclusions(attempt)],
                 "checks": [], "checks_note": "Runner executes the real frozen Template checks",
                 "issues": [], "framework_candidates": [], "handoff": "Keep the concrete fixture instance and actual sources"}
        if attempt["stage"] in REOPEN_TARGETS:
            value["reopen"] = None
        if attempt["stage"] == "P1":
            value["comparisons"] = []
        if attempt["stage"] in {"P2", "P4"}:
            value["plan"] = deepcopy(self.selected if attempt["stage"] == "P2" else self.writing)
        if attempt["stage"] == "P6":
            value.update(affected_nodes=[], revision_scope=attempt["revision_scope"])
            if revise_directory:
                value["revision_scope"] = [{"path": "dispatch.md", "fields": ["*"]}]
                value["issues"] = [{"kind": "schema_conformance", "scope": "dispatch.md",
                                    "detail": "Final directory must register all now completed YAML files",
                                    "evidence": [0], "recovery_condition": "Update the explicit directory scope"}]
        return value

    def run(self, *, attempt, prompt, schema, timeout, cancel_event=None):
        self.calls.append(deepcopy(attempt))
        if attempt["role"] == "worker":
            self.write(attempt)
        response = self.proposal(attempt)
        validate(response, schema)
        key = tuple(sorted(attempt["session_scope"].items()))
        session = self.sessions.setdefault(key, str(uuid4()))
        if attempt["session_id"] not in {None, session}:
            raise AssertionError("scope tried to resume another session")
        return {"backend": attempt["backend"], "outcome": "completed", "exit_code": 0,
                "termination": {"confirmed": True}, "native_status": {"success": True},
                "session_status": "ready", "session_id": session,
                "response_validation": {"schema_valid": True, "nonce_valid": True, "native_output_matches": True},
                "response": response}


class StageFlowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        source = self.directory / "source"
        source.mkdir()
        (source / "kernel.py").write_text("def copy(x): return x\n\ndef reference(x): return x\n")
        self.source = SourceTree(source)
        locator = {"path": "kernel.py", "symbol": "copy"}
        self.bindings = {"format": "schema_extract.source.v2", "id": "scalar_copy", "directory": ".",
                         **{key: deepcopy(locator) for key in ("entrypoint", "host", "input")},
                         "reference": {"path": "kernel.py", "symbol": "reference"}, "io": [deepcopy(locator)],
                         "acceptance": [deepcopy(locator)], "scope": [deepcopy(locator)]}

    def create(self, nodes=("root",), *, backend="claude", writing=None):
        config = {**DEFAULT_CONFIG, "backend": backend, "agent_binary": "/usr/bin/true",
                  "adjudication_seconds": 30, "retry_limit": 2}
        if backend == "codex":
            config.update(model="fixture-codex", worker_effort="high", reviewer_effort="high")
        self.store = Store(self.directory / ("state-" + backend))
        self.store.initialize()
        self.lock = freeze_inputs(self.store.root, self.source, self.bindings, ["kernel.py"], config)
        self.state = self.store.create(self.lock, self.directory / ("output-" + backend))
        selected = [{"id": node, "instance": "kernel.py::copy(instance=" + node + ")"} for node in nodes]
        writing = writing if writing is not None else [{"id": node, "wait_for": []} for node in nodes if node != "root"]
        self.backend = FixtureBackend(self.lock, selected, writing)

    def prepare(self, *, node=None, stage=None, role=None):
        call = next(item for item in next_calls(self.state) if (node is None or item["node"] == node) and
                    (stage is None or item["stage"] == stage) and (role is None or item["role"] == role))
        return self.store.prepare_attempt(self.state, call["session_id"], call=call, timeout_seconds=60)

    def result(self, attempt):
        value = self.backend.run(attempt=attempt, prompt=build_prompt(self.state, self.lock, attempt),
                                 schema=proposal_schema(attempt, source_files=self.lock["source"]["files"]), timeout=60)
        write_json(Path(attempt["directory"]) / "result.json", value)
        return value

    def accept(self, attempt, result=None):
        self.state = accept_result(self.store, self.state, self.lock, attempt, result or self.result(attempt))
        self.state["unsettled_attempts"].pop(attempt["attempt_id"], None)
        self.store.save(self.state)

    def step(self, **scope):
        attempt = self.prepare(**scope)
        self.accept(attempt)
        return attempt

    def through_root(self):
        while self.state["stage"] in {"P1", "P2", "P3", "P4"}:
            self.step()

    def test_single_root_uses_independent_sessions_and_still_runs_p4_p6(self):
        for backend in ("claude", "codex"):
            with self.subTest(backend=backend):
                self.create(backend=backend)
                with patch("schema_extract.tools.publication.publish", side_effect=lambda store, state, output: state):
                    self.state = run(self.store, self.state["run_id"], transport=self.backend)
                self.assertEqual(self.state["status"], "ready", self.state.get("reason"))
                calls = self.backend.calls
                self.assertEqual([(c["stage"], c["role"]) for c in calls],
                                 [(stage, role) for stage in ("P1", "P2", "P3", "P4") for role in ("worker", "reviewer")] +
                                 [("P6", "reviewer")])
                self.assertTrue(all(c["session_id"] is None for c in calls))
                self.assertEqual(len({c["session_directory"] for c in calls}), len(calls))
                self.assertEqual(self.state["stages"]["P5"]["status"], "skipped")
                self.assertEqual(self.state["writing_plan"], [])
                self.assertNotIn("plan", self.state)
                raw = read_json(Path(calls[0]["directory"]) / "receipt.json")
                checked = read_json(Path(calls[0]["directory"]) / "validated.json")
                self.assertNotEqual(raw["artifacts"][0]["sha256"], checked["artifact_digests"]["root.yaml"])
                final, _ = self.store.verify_checkpoint(self.state)
                final_hashes = manifest(final)
                self.store.verify_approval(self.state, self.state["stages"]["P6"]["approval"])
                self.assertEqual(manifest(final), final_hashes)

    def test_multiple_sources_and_late_unrelated_sibling_do_not_expire_review(self):
        nodes = ("root", "hidden", "math", "schedule", "consumer", "sibling")
        writing = [{"id": "hidden", "wait_for": []}, {"id": "math", "wait_for": ["hidden"]},
                   {"id": "schedule", "wait_for": ["root"]}, {"id": "consumer", "wait_for": ["math", "schedule"]},
                   {"id": "sibling", "wait_for": ["math"]}]
        self.create(nodes, writing=writing)
        self.through_root()
        merged, _ = self.store.verify_checkpoint(self.state, self.state["merged_checkpoint"])
        self.assertEqual(set(manifest(merged)), {"root.yaml", "dispatch.md"})
        self.assertEqual({call["node"] for call in next_calls(self.state)}, {"hidden", "schedule"})
        for node in ("hidden", "schedule", "math"):
            first = self.prepare(node=node, role="worker")
            draft = load_yaml(Path(first["workspace"]), node + ".yaml", parser_root=self.backend.rules)
            self.assertNotIn("CONTRACT", draft)
            self.assertTrue(all("from" not in body for body in draft["DESIGN"].values()))
            self.assertIsNone(draft["DESIGN"]["Math_reshape"]["Thesis"])
            self.accept(first)
            self.step(node=node, role="reviewer")
        self.step(node="consumer", role="worker")
        reviewing = self.prepare(node="consumer", role="reviewer")
        result = self.result(reviewing)
        self.step(node="sibling", role="worker")
        self.step(node="sibling", role="reviewer")
        self.accept(reviewing, result)
        self.assertEqual(self.state["stage"], "P6")
        consumer = self.state["nodes"]["consumer"]["source_context"]
        self.assertEqual(set(consumer["source_manifest"]), {"root.yaml", "consumer.yaml", "math.yaml", "schedule.yaml"})
        self.assertIn("hidden", consumer["resolution_nodes"])
        self.assertNotIn("sibling.yaml", consumer["source_manifest"])
        self.assertNotIn("consumer", affected_nodes(self.state, ["sibling"]))
        self.assertEqual(self.state["nodes"]["consumer"]["approved_sha256"], reviewing["dependency_digests"]["consumer.yaml"])
        first_review = self.step(stage="P6", role="reviewer")
        records = first_review["materials"]["records"]
        consumer_receipt = next(record for record in records if record.get("attempt_id") == reviewing["attempt_id"])
        self.assertEqual(consumer_receipt["purpose"], "current_decision")
        self.assertEqual(read_json(Path(consumer_receipt["path"]))["handoff"], result["response"]["handoff"])
        self.assertTrue({(stage, role) for stage in ("P2", "P4", "P5") for role in ("worker", "reviewer")} <=
                        {(record["stage"], record["role"]) for record in records if record["kind"] == "receipt"})
        self.assertTrue(all(record["read_after"] == "current_artifacts" for record in records))
        self.assertEqual(self.state["role"], "worker")
        self.step(stage="P6", role="worker")
        last_review = self.step(stage="P6", role="reviewer")
        self.assertEqual(last_review["session_id"], self.backend.sessions[tuple(sorted(first_review["session_scope"].items()))])
        self.assertEqual(self.state["status"], "ready")
        self.assertEqual(effective_nodes(self.state), set(nodes))

    def test_last_control_gate_denies_calls_prepared_before_pause_or_cancel(self):
        for stage, actions in (("P5", ("pause",)), ("P1", ("cancel", "pause"))):
            with self.subTest(stage=stage, actions=actions):
                self.create(("root", "branch") if stage == "P5" else ("root",))
                if stage == "P5":
                    self.through_root()
                node = "branch" if stage == "P5" else "root"
                scope = self.state["nodes"][node]["stage"] if stage == "P5" else self.state["stages"][stage]
                used_calls, completed = scope["calls"], deepcopy(self.state["accepted_receipts"])
                invoked = len(self.backend.calls)
                prepared = self.store.prepare_attempt
                def request_after_preparation(*args, **kwargs):
                    attempt = prepared(*args, **kwargs)
                    for action in actions:
                        self.store.request_control(self.state["run_id"], action)
                    return attempt
                with patch.object(self.store, "prepare_attempt", side_effect=request_after_preparation):
                    self.state = run(self.store, self.state["run_id"], transport=self.backend)
                self.assertEqual(len(self.backend.calls), invoked)
                self.assertEqual(self.state["status"], "cancelled" if "cancel" in actions else "paused")
                self.assertEqual(self.state["accepted_receipts"], completed)
                scope = self.state["nodes"][node]["stage"] if stage == "P5" else self.state["stages"][stage]
                self.assertEqual(scope["calls"], used_calls)
                self.assertEqual((scope["adjudications"], scope["signoffs"]), (0, 0))
                self.assertEqual(self.state["active_attempts"], {})
                self.assertEqual(self.state["unsettled_attempts"], {})
                finished = self.store.run_dir(self.state["run_id"]) / "attempts" / self.state["attempts"][-1] / "finished.json"
                self.assertEqual(read_json(finished)["status"], "not_dispatched")
                if stage == "P5":
                    self.assertNotIn("writing_started", self.state["nodes"][node])

    def test_p2_review_cannot_change_the_worker_selection(self):
        self.create()
        self.step(stage="P1", role="worker")
        self.step(stage="P1", role="reviewer")
        self.step(stage="P2", role="worker")
        reviewer = self.prepare(stage="P2", role="reviewer")
        result = self.result(reviewer)
        result["response"]["plan"][0]["instance"] = "kernel.py::copy(other-instance)"
        with self.assertRaisesRegex(ExtractError, "reviewer plan differs"):
            self.accept(reviewer, result)
        self.assertIsNone(self.store.load(self.state["run_id"])["selected_designs"])

    def test_last_control_gate_preserves_pending_recovery_handoff(self):
        from schema_extract.tools.agent import _session

        self.create()
        interrupted = self.prepare(stage="P1", role="worker")
        self.backend.write(interrupted)
        previous_session = str(uuid4())
        with _session(interrupted) as (path, record):
            record.update(status="running", attempt_id=interrupted["attempt_id"], session_id=previous_session)
            write_json(path, record)
        self.state = _interrupted(self.store, self.state, self.lock, interrupted, None,
                                  {"confirmed": True}, "Recover this interrupted draft")
        self.state["unsettled_attempts"].pop(interrupted["attempt_id"], None)
        self.store.save(self.state)
        checkpoint = self.state["checkpoint"]
        used_calls = self.state["stages"]["P1"]["calls"]
        prepared = self.store.prepare_attempt

        def pause_after_preparation(*args, **kwargs):
            attempt = prepared(*args, **kwargs)
            self.assertEqual(attempt["recovery_handoff"]["attempt_id"], interrupted["attempt_id"])
            self.store.request_control(self.state["run_id"], "pause")
            return attempt

        with patch.object(self.store, "prepare_attempt", side_effect=pause_after_preparation):
            self.state = run(self.store, self.state["run_id"], transport=self.backend)
        self.assertEqual(self.state["status"], "paused")
        self.assertEqual(self.backend.calls, [])
        self.assertEqual(self.state["checkpoint"], checkpoint)
        self.assertEqual(self.state["stages"]["P1"]["calls"], used_calls)
        self.assertIsNone(self.state["session_handoffs"][0]["next_attempt_id"])

        invoke = self.backend.run

        def pause_started_call(**kwargs):
            self.store.request_control(self.state["run_id"], "pause")
            return invoke(**kwargs)

        with patch.object(self.backend, "run", side_effect=pause_started_call):
            self.state = run(self.store, self.state["run_id"], transport=self.backend, resume=True)
        self.assertEqual(len(self.backend.calls), 1)
        resumed = self.backend.calls[0]
        self.assertEqual(resumed["recovery_handoff"]["attempt_id"], interrupted["attempt_id"])
        handoff = self.state["session_handoffs"][0]
        self.assertEqual(handoff["next_attempt_id"], resumed["attempt_id"])
        self.assertIsNotNone(handoff["next_session_id"])
        self.assertNotEqual(handoff["next_session_id"], previous_session)
        self.assertEqual(self.state["stages"]["P1"]["calls"], used_calls + 1)

    def test_writing_order_does_not_block_started_local_review(self):
        from schema_extract.tools.stages import dependencies_ready, invalidate_node
        self.create(("root", "provider", "consumer"), writing=[
            {"id": "provider", "wait_for": []}, {"id": "consumer", "wait_for": ["provider"]}])
        self.through_root()
        self.assertFalse(dependencies_ready(self.state, "consumer"))
        self.step(node="provider", role="worker")
        self.step(node="provider", role="reviewer")
        self.step(node="consumer", role="worker")
        invalidate_node(self.state["nodes"]["provider"], require_worker=True)
        self.assertTrue(dependencies_ready(self.state, "consumer"))
        consumer = self.state["nodes"]["consumer"]
        origin = consumer["source_context"]["origins"]["Math_reshape"]
        chain = origin["chain"]
        origin["chain"] = ["consumer", "provider"]
        self.assertFalse(dependencies_ready(self.state, "consumer"))
        origin["chain"] = chain
        instance = consumer["instance"]
        consumer["instance"] += ", corrected"
        self.assertFalse(dependencies_ready(self.state, "consumer"))
        consumer["instance"] = instance
        self.state["writing_plan"][1]["wait_for"].append("root")
        self.assertFalse(dependencies_ready(self.state, "consumer"))
        self.state["writing_plan"][1]["wait_for"].remove("root")
        self.store.save(self.state)
        self.step(node="consumer", role="reviewer")
        self.assertEqual(self.state["nodes"]["consumer"]["status"], "approved")
        self.assertEqual({call["node"] for call in next_calls(self.state)}, {"provider"})

    def test_p4_cannot_extend_p2_or_change_current_selection_identity(self):
        self.create(("root", "local"))
        while not (self.state["stage"] == "P4" and self.state["role"] == "reviewer"):
            self.step()
        reviewer = self.prepare()
        result = self.result(reviewer)
        result["response"]["plan"].append({"id": "extra", "wait_for": []})
        with self.assertRaisesRegex(ExtractError, "exactly once"):
            self.accept(reviewer, result)
        self.state["selected_designs"][1]["instance"] = "kernel.py::copy(changed)"
        self.state["selection_signature"] = selection_signature(self.state["selected_designs"])
        self.store.save(self.state)
        with self.assertRaisesRegex(StaleResult, "selection changed"):
            self.store.validate_attempt(self.state, reviewer)

    def test_newly_approved_source_was_not_available_to_earlier_invocation(self):
        self.create(("root", "provider", "consumer"))
        self.through_root()
        consumer = self.prepare(node="consumer", role="worker")
        self.step(node="provider", role="worker")
        self.step(node="provider", role="reviewer")
        merged, _ = self.store.verify_checkpoint(self.state, self.state["merged_checkpoint"])
        candidate = self.directory / "late-source"
        import shutil
        shutil.copytree(merged, candidate)
        for path in candidate.iterdir():
            path.chmod(0o644)
        candidate.chmod(0o755)
        root = load_yaml(candidate, "root.yaml", parser_root=self.backend.rules)
        root.pop("CONTRACT")
        root["META_DATA"]["Dispatch"] = {"id": "consumer", "when": "All scalar inputs"}
        root["DESIGN"]["Math_reshape"] = {"from": "provider"}
        write_yaml(candidate / "consumer.yaml", root, parser_root=self.backend.rules)
        with self.assertRaisesRegex(StaleResult, "not provided"):
            self.store.validate_sources(self.state, consumer, candidate)
        self.assertNotIn("provider", consumer["source_identities"])

    def test_current_source_approval_loss_or_identity_drift_fences_adoption(self):
        self.create(("root", "provider", "consumer"), writing=[{"id": "provider", "wait_for": []},
                                                                {"id": "consumer", "wait_for": ["provider"]}])
        self.through_root()
        self.step(node="provider", role="worker")
        self.step(node="provider", role="reviewer")
        consumer = self.prepare(node="consumer", role="worker")
        self.backend.write(consumer)
        workspace = Path(consumer["workspace"])
        document = load_yaml(workspace, "consumer.yaml", parser_root=self.backend.rules)
        document["DESIGN"]["Math_reshape"] = {"from": "provider"}
        write_yaml(workspace / "consumer.yaml", document, parser_root=self.backend.rules)
        self.store.validate_sources(self.state, consumer, workspace)
        provider = self.state["nodes"]["provider"]
        provider["status"] = "reviewing"
        with self.assertRaisesRegex(StaleResult, "no longer has current independent approval"):
            self.store.validate_sources(self.state, consumer, workspace)
        provider["status"] = "approved"
        provider["approval"]["effective_identity"] = "changed-approval-identity"
        with self.assertRaisesRegex(StaleResult, "identity changed"):
            self.store.validate_sources(self.state, consumer, workspace)

    def test_extra_yaml_is_unread_in_normal_failed_and_interrupted_adoption(self):
        self.create(("root", "target", "extra"))
        self.through_root()
        attempt = self.prepare(node="target", role="worker")
        result = self.result(attempt)
        workspace = Path(attempt["workspace"])
        (workspace / "extra.yaml").write_text("unprovided_content: true\n")
        baseline = self.state["checkpoint"]
        read_bytes = Path.read_bytes
        def guarded_read(path):
            if path.parent == workspace:
                self.fail("candidate content read before exact inventory rejection")
            return read_bytes(path)
        with patch.object(Path, "read_bytes", guarded_read):
            with self.assertRaisesRegex(ExtractError, "unexpected extra.yaml"):
                accept_result(self.store, self.state, self.lock, attempt, result)
            self.assertFalse((Path(attempt["directory"]) / "original").exists())
            with self.assertRaisesRegex(ExtractError, "unexpected extra.yaml"):
                _recover_call(self.store, self.state, self.lock, attempt, result,
                              kind="cli_retry", reason="Offline failed-call draft")
            with self.assertRaisesRegex(ExtractError, "unexpected extra.yaml"):
                self.store.checkpoint(self.state, workspace, attempt_id=attempt["attempt_id"],
                                      expected_files=attempt["dependency_digests"])
            recovered = _interrupted(self.store, self.state, self.lock, attempt, result,
                                     {"confirmed": True}, "Offline interrupted draft")
        self.assertEqual(recovered["checkpoint"], baseline)
        self.assertIn("unexpected extra.yaml", recovered["recovered_attempts"][attempt["attempt_id"]]["discarded_draft"])

    def test_completed_label_does_not_replace_native_proof(self):
        self.create()
        attempt = self.prepare()
        result = self.result(attempt)
        original = self.store.load(self.state["run_id"])
        for key, value in (("exit_code", 1), ("native_status", {"success": False}),
                           ("response_validation", {"schema_valid": True}),
                           ("session_status", "retired")):
            with self.subTest(key=key), self.assertRaisesRegex(ExtractError, "verified successful"):
                self.accept(attempt, {**result, key: value})
            self.assertEqual(self.store.load(self.state["run_id"]), original)
        workspace = Path(attempt["workspace"])
        value = deepcopy(self.backend.base)
        value["DESIGN"] = load_yaml(self.backend.rules / "schema_template", "starter.yaml", parser_root=self.backend.rules)["DESIGN"]
        value["META_DATA"]["Source"] = "/source/kernel.py::copy"
        write_yaml(workspace / "root.yaml", value, parser_root=self.backend.rules)
        with self.assertRaisesRegex(ExtractError, "fixed repository"):
            self.accept(attempt, {**result, "response": self.backend.proposal(attempt)})
        self.assertEqual(self.store.load(self.state["run_id"]), original)


if __name__ == "__main__":
    unittest.main()
