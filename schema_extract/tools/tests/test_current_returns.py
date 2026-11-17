"""Current-run returns, actual-use applicability and bounded review decisions."""

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from schema_extract.tools import maintenance
from schema_extract.tools.artifacts import load_yaml, resolve_workspace, write_yaml
from schema_extract.tools.budget import call_budget, check_timeout, reserved_calls
from schema_extract.tools.common import ExtractError, StaleResult, manifest, read_json, rules_root, write_json
from schema_extract.tools.prompts import build_prompt
from schema_extract.tools.receipts import proposal_schema
from schema_extract.tools.runner import _interrupted, accept_result, run
from schema_extract.tools.stages import (approval_is_current, effective_nodes, next_calls,
                                        install_selection, scope_for, selection_impact, selection_signature, settle)
from schema_extract.tools.tests import test_stage_flow as stage_fixture


class CurrentReturnsTests(unittest.TestCase):
    def setUp(self):
        self.flow = stage_fixture.StageFlowTests(methodName="runTest")
        self.flow.setUp()
        self.addCleanup(self.flow.doCleanups)

    def start(self, nodes=("root",), writing=None):
        self.flow.create(nodes=nodes, writing=writing)
        self.flow.through_root()
        return self.flow

    def approve_nodes(self, *nodes):
        for node in nodes:
            self.flow.step(node=node, role="worker")
            self.flow.step(node=node, role="reviewer")

    def result(self, attempt, *, decision=None, issues=None, reopen=None, write=None):
        f = self.flow
        def run_fixture():
            if attempt["mode"] == "normal" and attempt.get("purpose") != "maintenance":
                return f.result(attempt)
            result = f.backend.run(attempt=attempt, prompt="Fixture semantic response", schema={"type": "object"}, timeout=30)
            if "reopen" not in proposal_schema(attempt)["properties"]:
                result["response"].pop("reopen", None)
            if attempt.get("purpose") == "maintenance":
                result["response"]["revision_scope"] = (maintenance.task_for(f.state, attempt)["fields"] if attempt["role"] == "worker" else [])
            write_json(Path(attempt["directory"]) / "result.json", result)
            return result
        if write is None:
            result = run_fixture()
        else:
            with patch.object(f.backend, "write", side_effect=write):
                result = run_fixture()
        proposal = result["response"]
        if decision is not None:
            proposal["decision"] = decision
        if issues is not None:
            proposal["issues"] = issues
        if reopen is not None:
            proposal["reopen"] = reopen
        return result

    def return_to(self, attempt, target, issue_id, *, scope="root.yaml", region=None):
        result = self.result(attempt, write=lambda call: None)
        proposal = result["response"]
        proposal.update(decision="reopen", reopen={"issue_id": issue_id, "target": target,
            "reason": "The previously read fixed implementation was misinterpreted", "evidence": [0]},
            issues=[{"kind": "source_misread", "scope": scope, "detail": "Correct the concrete previously read definition",
                     "evidence": [0], "recovery_condition": "Repair the owning responsibility and review actual affected uses"}])
        if region:
            resolved = resolve_workspace(Path(attempt["workspace"]), parser_root=rules_root(self.flow.lock))
            node = scope.split(".yaml", 1)[0]
            line = resolved.documents[node].positions[node + ".DESIGN." + region][0]
            proposal["artifact_evidence"].append(self.flow.backend.anchor(node + ".yaml", line))
        return result

    def update_node(self, attempt, region, text):
        directory = Path(attempt["workspace"])
        name = attempt["node"] + ".yaml"
        value = load_yaml(directory, name, parser_root=rules_root(self.flow.lock))
        value["DESIGN"][region]["Thesis"] = text
        value["META_DATA"]["Version"] += 1
        write_yaml(directory / name, value, parser_root=rules_root(self.flow.lock))

    def correct_metadata(self, attempt):
        directory = Path(attempt["workspace"])
        value = load_yaml(directory, "root.yaml", parser_root=rules_root(self.flow.lock))
        value["META_DATA"].update(Name="scalar_copy_corrected", Describe="Copy one scalar without changing its value.")
        value["META_DATA"]["Version"] += 1
        write_yaml(directory / "root.yaml", value, source=directory / "root.yaml", parser_root=rules_root(self.flow.lock))

    def test_p1_shared_metadata_keeps_unmerged_drafts_and_requires_current_body_reviews(self):
        f = self.start(("root", "approved", "complete", "unfinished", "requester"))
        self.approve_nodes("approved")
        f.step(node="complete", role="worker")
        attempt = f.prepare(node="unfinished", role="worker")
        f.accept(attempt, self.result(attempt, decision="continue", write=lambda call: None))
        saved = deepcopy(f.state)
        draft_hashes = {}
        for node in ("approved", "complete", "unfinished"):
            _, info = f.store.verify_checkpoint(f.state, f.state["nodes"][node]["draft_checkpoint"])
            draft_hashes[node] = info["files"][node + ".yaml"]
        request = f.prepare(node="requester", role="worker")
        f.accept(request, self.return_to(request, "P1", "same-task-metadata-correction"))
        writer = f.prepare(stage="P1", role="worker")
        self.assertEqual(writer["writable_files"], ["root.yaml"])
        for node, checksum in draft_hashes.items():
            self.assertEqual(writer["dependency_digests"][node + ".yaml"], checksum)
        f.accept(writer, self.result(writer, write=self.correct_metadata))
        validation = read_json(Path(writer["directory"]) / "validated.json")
        self.assertEqual(validation["original_changed_files"], ["root.yaml"])
        self.assertEqual(set(validation["shared_metadata_sync"]["nodes"]),
                         {node + ".yaml" for node in ("approved", "complete", "unfinished", "requester")})
        reviewer = f.prepare(stage="P1", role="reviewer")
        reviewed = manifest(Path(reviewer["workspace"]))
        f.accept(reviewer)
        final, info = f.store.verify_checkpoint(f.state, f.state["merged_checkpoint"])
        self.assertEqual(info["files"], reviewed)
        self.assertEqual(f.state["stage"], "P3")
        for stage in ("P2", "P4"):
            self.assertEqual(f.state["stages"][stage]["approval"], saved["stages"][stage]["approval"])
        for node in ("root", "approved"):
            self.assertIsNone(f.state["nodes"][node]["approval"])
            self.assertEqual(f.state["nodes"][node]["last_approval"], saved["nodes"][node]["approval"])
            f.store.verify_approval_origin(f.state, saved["nodes"][node]["approval"])
        self.assertEqual(f.state["nodes"]["complete"]["role"], "reviewer")
        self.assertEqual(f.state["nodes"]["unfinished"]["role"], "worker")
        self.assertIsNone(f.state["nodes"]["unfinished"]["approval"])
        self.assertIsNone(load_yaml(final, "unfinished.yaml", parser_root=f.backend.rules)["DESIGN"]["Math_reshape"]["Thesis"])
        self.assertEqual(load_yaml(final, "unfinished.yaml", parser_root=f.backend.rules)["META_DATA"]["Version"], 2)
        f.step(stage="P3", role="reviewer")
        for node in ("approved", "complete"):
            reviewed_node = f.prepare(node=node, role="reviewer")
            self.assertEqual(manifest(Path(reviewed_node["workspace"]))[node + ".yaml"], reviewed[node + ".yaml"])
            f.accept(reviewed_node)
            f.store.verify_node(f.state, node)
        self.assertEqual({call["node"] for call in next_calls(f.state)}, {"unfinished", "requester"})
        self.assertEqual(f.state["nodes"]["approved"]["stage"]["calls"], saved["nodes"]["approved"]["stage"]["calls"] + 1)
        self.assertNotEqual(f.store.source_identity(f.state, "approved"), saved["nodes"]["approved"]["approval"]["effective_identity"])
        continuing = f.prepare(node="unfinished", role="worker")
        self.assertFalse(continuing["materials"]["first_draft"])

    def test_process_facts_survive_metadata_return_and_removed_instances(self):
        f = self.start(("root", "record"))
        writer = f.prepare(node="record", role="worker")
        result = self.result(writer)
        fact = "Parameter fact: preserve the scalar tail variant in dispatch at P6"
        result["response"]["handoff"] = fact
        f.accept(writer, result)
        f.step(node="record", role="reviewer")
        original = read_json(Path(writer["directory"]) / "receipt.json")
        def provided(call, relation):
            record = next(item for item in call["materials"]["records"] if item.get("attempt_id") == writer["attempt_id"])
            self.assertEqual(record["purpose"], "knowledge_destination")
            self.assertEqual(record["generation"], 0)
            self.assertEqual(record["source_id"], f.lock["source"]["id"])
            self.assertEqual(record["receipt_sha256"], f.state["accepted_receipts"][writer["attempt_id"]])
            self.assertEqual(record["instance_relation"]["record"], relation)
            copied = Path(record["path"])
            self.assertEqual(read_json(copied), original)
            self.assertIn(fact, copied.read_text())
            self.assertEqual(record["read_after"], "current_artifacts")
            self.assertNotIn(fact, build_prompt(f.state, f.lock, call))
        final = f.prepare(stage="P6", role="reviewer")
        f.accept(final, self.return_to(final, "P1", "metadata-keeps-ordinary-facts"))
        p1 = f.prepare(stage="P1", role="worker")
        f.accept(p1, self.result(p1, write=self.correct_metadata))
        f.step(stage="P1", role="reviewer")
        f.step(stage="P3", role="reviewer")
        review = f.prepare(node="record", role="reviewer")
        self.assertEqual(review["generation"], 1)
        provided(review, "current_instance")
        f.accept(review)
        final = f.prepare(stage="P6", role="reviewer")
        provided(final, "current_instance")
        f.accept(final, self.return_to(final, "P2", "retire-record-keeps-ordinary-facts"))
        f.backend.selected = [entry for entry in f.backend.selected if entry["id"] == "root"]
        f.backend.writing = []
        p2 = f.prepare(stage="P2", role="worker")
        f.accept(p2, self.result(p2, write=lambda call: f.backend.index(Path(call["workspace"]))))
        f.step(stage="P2", role="reviewer")
        f.step(stage="P4", role="worker")
        f.step(stage="P4", role="reviewer")
        final = f.prepare(stage="P6", role="reviewer")
        provided(final, "removed")
        self.assertNotIn("record", final["source_identities"])
        self.assertNotIn("record.yaml", final["dependency_digests"])
        self.assertEqual(read_json(Path(writer["directory"]) / "receipt.json"), original)

    def test_p1_source_change_opens_only_the_invalidated_selection_projection(self):
        f = self.start()
        previous = deepcopy(f.state["stages"]["P2"])
        writing = deepcopy(f.state["stages"]["P4"]["approval"])
        selected = deepcopy(f.state["selected_designs"])
        final = f.prepare(stage="P6", role="reviewer")
        f.accept(final, self.return_to(final, "P1", "same-source-locator-correction"))
        def correct_source(call):
            directory = Path(call["workspace"])
            root = load_yaml(directory, "root.yaml", parser_root=f.backend.rules)
            root["META_DATA"]["Source"] = root["META_DATA"]["Source"].split("::")[0]
            root["META_DATA"]["Version"] += 1
            write_yaml(directory / "root.yaml", root, source=directory / "root.yaml", parser_root=f.backend.rules)
        p1 = f.prepare(stage="P1", role="worker")
        f.accept(p1, self.result(p1, write=correct_source))
        approval = f.step(stage="P1", role="reviewer")
        self.assertEqual((f.state["stage"], f.state["role"]), ("P2", "reviewer"))
        self.assertEqual(f.state["plan_draft"], selected)
        self.assertEqual(f.state["selected_designs"], selected)
        current = f.state["stages"]["P2"]
        self.assertEqual(current["last_approval"], previous["approval"])
        self.assertIsNone(current["approval"])
        self.assertEqual(current["generation"], previous["generation"] + 1)
        for key in ("calls", "elapsed_seconds", "deadline", "adjudications", "signoffs", "revisions"):
            self.assertEqual(current[key], previous[key])
        f.store.verify_approval_origin(f.state, previous["approval"])
        validation = read_json(Path(approval["directory"]) / "validated.json")
        self.assertEqual(set(validation["invalidated_projections"]), {"P2"})
        f.step(stage="P2", role="reviewer")
        f.step(stage="P3", role="reviewer")
        self.assertEqual(f.state["stages"]["P4"]["approval"], writing)
        f.step(stage="P6", role="reviewer")
        self.assertEqual(f.state["status"], "ready")
        self.assertEqual(f.state["stages"]["P2"]["calls"], previous["calls"] + 1)

    def test_existing_source_misread_returns_only_the_selection_and_preserves_other_work(self):
        f = self.start(("root", "left", "right"))
        self.approve_nodes("left", "right")
        saved = deepcopy(f.state)
        attempt = f.prepare(stage="P6")
        f.accept(attempt, self.return_to(attempt, "P2", "known-instance-misread"))
        self.assertEqual(f.state["stage"], "P2")
        for node in ("root", "left", "right"):
            self.assertEqual(f.state["nodes"][node]["generation"], saved["nodes"][node]["generation"])
            self.assertEqual(f.state["nodes"][node]["approval"], saved["nodes"][node]["approval"])
        f.backend.selected[1]["instance"] += ", corrected"
        def selection_only(call):
            f.backend.index(Path(call["workspace"]))
        attempt = f.prepare(stage="P2", role="worker")
        f.accept(attempt, self.result(attempt, write=selection_only))
        f.step(stage="P2", role="reviewer")
        self.assertEqual(f.state["stage"], "P4")
        self.assertEqual(f.state["nodes"]["right"]["approval"], saved["nodes"]["right"]["approval"])
        self.assertEqual(f.state["nodes"]["right"]["stage"]["calls"], saved["nodes"]["right"]["stage"]["calls"])
        self.assertEqual(f.state["nodes"]["left"]["role"], "worker")
        self.assertEqual(f.state["nodes"]["left"]["generation"], saved["nodes"]["left"]["generation"] + 1)
        self.assertTrue(approval_is_current(f.state, f.state["nodes"]["right"]["approval"]))
        f.step(stage="P4", role="worker")
        f.step(stage="P4", role="reviewer")
        f.store.verify_node(f.state, "right")
        self.assertEqual({call["node"] for call in next_calls(f.state)}, {"left"})

    def test_records_and_localization_do_not_reopen_p2_or_p4(self):
        f = self.start(("root", "record"))
        before = {stage: deepcopy(f.state["stages"][stage]) for stage in ("P2", "P4")}
        attempt = f.prepare(node="record", role="worker")
        result = self.result(attempt)
        result["response"]["handoff"] = "Keep this ordinary parameter record for P6; this node uses local complete regions"
        f.accept(attempt, result)
        f.step(node="record", role="reviewer")
        self.assertEqual(f.state["stage"], "P6")
        for stage in before:
            self.assertEqual(before[stage], f.state["stages"][stage])
        self.assertFalse(f.state["reopen_requests"])

    def test_real_provider_waits_and_unused_region_change_keeps_consumer_approval(self):
        f = self.start(("root", "hidden", "math", "schedule", "consumer"))
        self.approve_nodes("hidden", "math", "schedule", "consumer")
        consumer = deepcopy(f.state["nodes"]["consumer"])
        attempt = f.prepare(stage="P6")
        self.assertIn("P5:math", attempt["provider_reopen_targets"])
        f.accept(attempt, self.return_to(attempt, "P5:math", "provider-known-misread", scope="math.yaml DESIGN.Tile_pipeline", region="Tile_pipeline"))
        self.assertNotIn("consumer", effective_nodes(f.state))
        self.assertEqual(f.state["nodes"]["consumer"]["generation"], consumer["generation"])
        self.assertEqual({call["node"] for call in next_calls(f.state)}, {"math"})
        attempt = f.prepare(node="math", role="worker")
        f.accept(attempt, self.result(attempt, write=lambda call: self.update_node(call, "Tile_pipeline", "Copy one scalar with one serial execution role")))
        self.assertNotIn("consumer", {call["node"] for call in next_calls(f.state)})
        f.step(node="math", role="reviewer")
        current = f.state["nodes"]["consumer"]
        self.assertEqual(current["approval"], consumer["approval"])
        self.assertEqual(current["generation"], consumer["generation"])
        self.assertNotEqual(f.store.source_identity(f.state, "consumer"), consumer["approval"]["effective_identity"])
        self.assertIn("applicability", current)
        f.store.verify_node(f.state, "consumer")
        self.assertEqual(f.state["nodes"]["schedule"]["stage"]["calls"], 2)
        sibling_approval = deepcopy(f.state["nodes"]["schedule"]["approval"])
        attempt = f.prepare(stage="P6")
        f.accept(attempt, self.return_to(attempt, "P5:math", "provider-used-math-misread", scope="math.yaml DESIGN.Math_reshape", region="Math_reshape"))
        attempt = f.prepare(node="math", role="worker")
        f.accept(attempt, self.result(attempt, write=lambda call: self.update_node(call, "Math_reshape", "Copy the scalar directly from its declared input")))
        f.step(node="math", role="reviewer")
        self.assertIsNone(f.state["nodes"]["consumer"]["approval"])
        self.assertEqual(f.state["nodes"]["consumer"]["role"], "reviewer")
        self.assertEqual(f.state["nodes"]["consumer"]["generation"], consumer["generation"] + 1)
        self.assertEqual(f.state["nodes"]["schedule"]["approval"], sibling_approval)
        self.assertEqual({call["node"] for call in next_calls(f.state)}, {"consumer"})

    def test_actual_definition_change_retires_late_review_and_rechecks_only_users(self):
        f = self.start(("root", "provider", "consumer", "sibling"))
        self.approve_nodes("provider")
        attempt = f.prepare(node="consumer", role="worker")
        original_write = f.backend.write
        def write_consumer(call):
            original_write(call)
            directory = Path(call["workspace"])
            value = load_yaml(directory, "consumer.yaml", parser_root=rules_root(f.lock))
            value["DESIGN"]["Math_reshape"] = {"from": "provider"}
            write_yaml(directory / "consumer.yaml", value, parser_root=rules_root(f.lock))
        f.accept(attempt, self.result(attempt, write=write_consumer))
        late = f.prepare(node="consumer", role="reviewer")
        late_result = self.result(late)
        request = f.prepare(node="sibling", role="worker")
        # A second actual consumer can identify the same provider while the
        # first independent reviewer still holds its original immutable input.
        def write_sibling(call):
            original_write(call)
            directory = Path(call["workspace"])
            value = load_yaml(directory, "sibling.yaml", parser_root=rules_root(f.lock))
            value["DESIGN"]["Math_reshape"] = {"from": "provider"}
            write_yaml(directory / "sibling.yaml", value, parser_root=rules_root(f.lock))
        with patch.object(f.backend, "write", side_effect=write_sibling):
            result = f.result(request)
        result["response"].update(decision="reopen", reopen={"issue_id": "actual-provider-math", "target": "P5:provider", "reason": "Correct a known source interpretation", "evidence": [0]},
            issues=[{"kind": "source_misread", "scope": "provider.yaml DESIGN.Math_reshape", "detail": "The provider definition needs correction", "evidence": [0], "recovery_condition": "Repair provider then recheck actual users"}])
        wrong_binding = deepcopy(result)
        wrong_binding["response"]["issues"][0]["scope"] = "sibling.yaml DESIGN.Math_reshape"
        with self.assertRaisesRegex(ExtractError, "raw definition-region anchor"):
            accept_result(f.store, f.state, f.lock, request, wrong_binding)
        f.accept(request, result)
        with self.assertRaises(StaleResult):
            accept_result(f.store, f.state, f.lock, late, late_result)
        from schema_extract.tools.runner import _retire_stale
        f.state = _retire_stale(f.store, f.state, late, late_result, "actual definition invalidated")
        f.state["unsettled_attempts"].pop(late["attempt_id"], None)
        f.store.save(f.state)
        attempt = f.prepare(node="provider", role="worker")
        f.accept(attempt, self.result(attempt, write=lambda call: self.update_node(call, "Math_reshape", "Copy the scalar input directly into its output")))
        f.step(node="provider", role="reviewer")
        self.assertEqual(f.state["nodes"]["consumer"]["role"], "reviewer")
        self.assertIsNone(f.state["nodes"]["consumer"]["approval"])
        self.assertEqual(f.state["nodes"]["provider"]["stage"]["calls"], 4)
        self.assertFalse(f.state["nodes"]["provider"].get("provider_reopen"))

    def test_removed_provider_stays_historical_and_consumer_must_change_from(self):
        f = self.start(("root", "provider", "consumer"))
        self.approve_nodes("provider")
        original_write = f.backend.write
        def consumer(call):
            original_write(call)
            directory = Path(call["workspace"])
            value = load_yaml(directory, "consumer.yaml", parser_root=rules_root(f.lock))
            value["DESIGN"]["Math_reshape"] = {"from": "provider"}
            write_yaml(directory / "consumer.yaml", value, parser_root=rules_root(f.lock))
        attempt = f.prepare(node="consumer", role="worker")
        f.accept(attempt, self.result(attempt, write=consumer))
        f.step(node="consumer", role="reviewer")
        old = f.state["nodes"]["consumer"]["approved_sha256"]
        attempt = f.prepare(stage="P6")
        f.accept(attempt, self.return_to(attempt, "P2", "remove-one-representative"))
        f.backend.selected = [entry for entry in f.backend.selected if entry["id"] != "provider"]
        f.backend.writing = [entry for entry in f.backend.writing if entry["id"] != "provider"]
        attempt = f.prepare(stage="P2", role="worker")
        f.accept(attempt, self.result(attempt, write=lambda call: f.backend.index(Path(call["workspace"]))))
        f.step(stage="P2", role="reviewer")
        # Shared P1 edits must work while Design still awaits localization.
        attempt = f.prepare(stage="P4", role="worker")
        f.accept(attempt, self.return_to(attempt, "P1", "metadata-after-provider-retirement"))
        attempt = f.prepare(stage="P1", role="worker")
        historical = f.state["nodes"]["consumer"]["baseline_context_checkpoint"]
        historical_files = f.store.verify_checkpoint(f.state, historical)[1]["files"]
        def correct_shared(call):
            self.correct_metadata(call)
            directory = Path(call["workspace"])
            root = load_yaml(directory, "root.yaml", parser_root=f.backend.rules)
            root["CONTRACT"]["Algorithm_requirement"]["Input"]["x"]["meaning"] = "Scalar input value to copy."
            write_yaml(directory / "root.yaml", root, source=directory / "root.yaml", parser_root=f.backend.rules)
        f.accept(attempt, self.result(attempt, write=correct_shared))
        normalization = read_json(Path(attempt["directory"]) / "normalization.json")
        self.assertEqual(set(normalization["source_hashes"]), {"root.yaml"})
        self.assertTrue(any(item["message"].startswith("period:") for item in normalization["corrections"]))
        f.step(stage="P1", role="reviewer")
        f.step(stage="P3", role="reviewer")
        f.step(stage="P4", role="worker")
        f.step(stage="P4", role="reviewer")
        current, _ = f.store.verify_checkpoint(f.state, f.state["merged_checkpoint"])
        self.assertNotIn("provider.yaml", manifest(current))
        synchronized = load_yaml(current, "consumer.yaml", parser_root=f.backend.rules)
        self.assertEqual(synchronized["META_DATA"]["Name"], "scalar_copy_corrected")
        self.assertNotEqual(manifest(current)["consumer.yaml"], old)
        self.assertEqual(synchronized["DESIGN"]["Math_reshape"], {"from": "provider"})
        self.assertNotIn("provider", effective_nodes(f.state))
        self.assertEqual(f.store.verify_checkpoint(f.state, historical)[1]["files"], historical_files)
        self.assertEqual(load_yaml(current, "root.yaml", parser_root=f.backend.rules)["CONTRACT"]["Algorithm_requirement"]["Input"]["x"]["meaning"],
                         "Scalar input value to copy")
        attempt = f.prepare(node="consumer", role="worker")
        self.assertNotIn("provider", attempt["source_identities"])
        self.assertNotIn("provider.yaml", manifest(Path(attempt["workspace"])))
        self.assertIn("baseline_context_checkpoint", attempt)
        def localize(call):
            directory = Path(call["workspace"])
            value = load_yaml(directory, "consumer.yaml", parser_root=rules_root(f.lock))
            value["DESIGN"]["Math_reshape"] = deepcopy(f.backend.base["DESIGN"]["Math_reshape"])
            value["META_DATA"]["Version"] += 1
            write_yaml(directory / "consumer.yaml", value, parser_root=rules_root(f.lock))
        f.accept(attempt, self.result(attempt, write=localize))
        f.step(node="consumer", role="reviewer")
        f.store.verify_node(f.state, "consumer")
        counts = deepcopy(f.state["nodes"]["provider"]["stage"])
        restored = [*f.state["selected_designs"], {"id": "provider", "instance": f.state["nodes"]["provider"]["instance"]}]
        current_approvals = {node: deepcopy(f.state["nodes"][node]["approval"]) for node in ("root", "consumer")}
        attempt = f.prepare(stage="P6")
        f.accept(attempt, self.return_to(attempt, "P2", "restore-representative-after-metadata-correction"))
        f.backend.selected = restored
        f.backend.writing = [{"id": entry["id"], "wait_for": []} for entry in restored if entry["id"] != "root"]
        attempt = f.prepare(stage="P2", role="worker")
        f.accept(attempt, self.result(attempt, write=lambda call: f.backend.index(Path(call["workspace"]))))
        f.step(stage="P2", role="reviewer")
        self.assertEqual(f.state["stage"], "P4")
        self.assertEqual(f.state["nodes"]["provider"]["role"], "reviewer")
        self.assertEqual(f.state["nodes"]["provider"]["status"], "reviewing")
        for key in ("calls", "adjudications", "signoffs", "started_at", "deadline"):
            self.assertEqual(f.state["nodes"]["provider"]["stage"][key], counts[key])
        for node in current_approvals:
            self.assertEqual(f.state["nodes"][node]["approval"], current_approvals[node])
        f.step(stage="P4", role="worker")
        f.step(stage="P4", role="reviewer")
        restored_review = f.prepare(node="provider", role="reviewer")
        prepared = restored_review["preparation"]
        self.assertEqual(prepared["kind"], "shared_metadata_initialization")
        self.assertEqual(prepared["p1_approval"]["attempt_id"], f.state["stages"]["P1"]["approval"]["attempt_id"])
        self.assertEqual(set(prepared["synchronization"]["nodes"]), {"provider.yaml"})
        original, _ = f.store.verify_checkpoint(f.state, prepared["original_checkpoint"])
        original_provider = load_yaml(original, "provider.yaml", parser_root=f.backend.rules)
        self.assertEqual(load_yaml(Path(restored_review["workspace"]), "provider.yaml", parser_root=f.backend.rules)["META_DATA"]["Name"],
                         "scalar_copy_corrected")
        self.assertNotEqual(original_provider["META_DATA"]["Name"], "scalar_copy_corrected")
        self.assertEqual(load_yaml(Path(restored_review["workspace"]), "provider.yaml", parser_root=f.backend.rules)["DESIGN"],
                         original_provider["DESIGN"])
        f.accept(restored_review)
        f.store.verify_node(f.state, "provider")

    def test_all_stages_reserve_signoff_and_p2_patch_binds_final_plan(self):
        f = self.flow
        f.create(nodes=("root", "branch"))
        f.step(stage="P1", role="worker")
        f.step(stage="P1", role="reviewer")
        attempt = f.prepare(stage="P2", role="worker")
        self.assertEqual(reserved_calls({**f.state, "role": "worker"}), 3)
        f.accept(attempt)
        reviewer = f.prepare(stage="P2", role="reviewer")
        revised = self.result(reviewer, decision="revise", issues=[{"kind": "schema_conformance", "scope": "dispatch.md", "detail": "Clarify the representative reason", "evidence": [0], "recovery_condition": "Use one explicit scoped patch"}])
        # Revisions already consumed in this same scope trigger the one
        # available decision without creating a second accounting scope.
        f.state["stages"]["P2"]["revisions"] = f.state["config"]["revision_limit"] - 1
        f.store.save(f.state)
        f.accept(reviewer, revised)
        attempt = f.prepare(stage="P2", role="reviewer")
        self.assertEqual(attempt["mode"], "adjudicate")
        self.assertTrue(attempt["adjudication_candidates"])
        result = f.backend.run(attempt=attempt, prompt="", schema={"type": "object"}, timeout=30)
        value = result["response"]
        value.pop("reopen", None)
        value.update(decision="approve", adjudication={"action": "patch", "checkpoint": attempt["base_checkpoint"],
            "patches": [{"path": "dispatch.md", "old": "合成验证实例 branch", "new": "合成验证实例 branch，补足独立机制依据"}]})
        f.accept(attempt, result)
        self.assertEqual(f.state["mode"], "signoff")
        self.assertIsNone(f.state["stages"]["P2"]["approval"])
        self.assertEqual(f.state["plan_draft"], f.backend.selected)
        signoff = f.prepare(stage="P2", role="reviewer")
        self.assertIsNone(signoff["session_id"])
        result = self.result(signoff)
        result["response"].pop("reopen", None)
        result["response"]["plan"][1]["instance"] += ", unseen"
        with self.assertRaisesRegex(ExtractError, "plan differs"):
            accept_result(f.store, f.state, f.lock, signoff, result)
        result["response"]["plan"] = deepcopy(f.backend.selected)
        f.accept(signoff, result)
        scope = f.state["stages"]["P2"]
        self.assertEqual((scope["adjudications"], scope["signoffs"]), (1, 1))
        self.assertEqual(scope["approval"]["attempt_id"], signoff["attempt_id"])
        self.assertGreater(scope["signoff_reserved_seconds"], 0)
        f.store.verify_approval(f.state, scope["approval"])

    def test_unaffected_inflight_review_survives_selection_and_directory_changes(self):
        f = self.start(("root", "left", "late"))
        f.step(node="left", role="worker")
        f.step(node="late", role="worker")
        late = f.prepare(node="late", role="reviewer")
        late_result = self.result(late)
        request = f.prepare(node="left", role="reviewer")
        f.accept(request, self.return_to(request, "P2", "left-instance-only"))
        f.backend.selected[1]["instance"] += ", corrected"
        attempt = f.prepare(stage="P2", role="worker")
        f.accept(attempt, self.result(attempt, write=lambda call: f.backend.index(Path(call["workspace"]))))
        f.step(stage="P2", role="reviewer")
        f.step(stage="P4", role="worker")
        f.step(stage="P4", role="reviewer")
        self.assertNotEqual(late["selection_signature"], f.state["selection_signature"])
        f.accept(late, late_result)
        self.assertEqual(f.state["nodes"]["late"]["generation"], late["generation"])
        self.assertEqual(f.state["nodes"]["late"]["approval"]["selection_signature"], late["selection_signature"])
        f.store.verify_node(f.state, "late")

    def test_select_only_reads_prepared_canonical_bytes_and_never_edits_after_approval(self):
        f = self.flow
        f.create()
        while f.state["stage"] != "P3":
            f.step()
        first = f.step(stage="P3", role="worker")
        first_checkpoint = f.state["checkpoint"]
        issues = [{"kind": "schema_conformance", "scope": "root.yaml DESIGN.Math_reshape", "detail": "Review this complete alternative wording",
                   "evidence": [0], "recovery_condition": "Choose a completed explicitly read candidate"}]
        attempt = f.prepare(stage="P3", role="reviewer")
        f.accept(attempt, self.result(attempt, decision="revise", issues=issues))
        attempt = f.prepare(stage="P3", role="worker")
        f.accept(attempt, self.result(attempt, write=lambda call: self.update_node(call, "Math_reshape", "Copy the scalar without a separate intermediate buffer")))
        attempt = f.prepare(stage="P3", role="reviewer")
        result = self.result(attempt, decision="revise", issues=issues)
        f.state["stages"]["P3"]["revisions"] = f.state["config"]["revision_limit"] - 1
        f.store.save(f.state)
        f.accept(attempt, result)
        attempt = f.prepare(stage="P3", role="reviewer")
        chosen = next(entry for entry in attempt["adjudication_candidates"] if entry["original_checkpoint"] == first_checkpoint)
        material = Path(attempt["directory"]) / "materials" / "candidates" / chosen["checkpoint"]
        value = load_yaml(material, "root.yaml", parser_root=rules_root(f.lock))
        self.assertEqual(value["META_DATA"]["Version"], 5)
        result = self.result(attempt)
        result["response"]["adjudication"] = {"action": "select", "checkpoint": chosen["checkpoint"], "patches": []}
        f.accept(attempt, result)
        scope = f.state["stages"]["P3"]
        self.assertEqual(scope["signoffs"], 0)
        self.assertEqual(scope["approval"]["attempt_id"], attempt["attempt_id"])
        approved, _ = f.store.verify_checkpoint(f.state, scope["approval"]["checkpoint"])
        self.assertEqual((approved / "root.yaml").read_bytes(), (material / "root.yaml").read_bytes())
        f.store.verify_approval(f.state, scope["approval"])

    def test_p4_signoff_interruption_retires_session_without_new_decision_credit(self):
        from uuid import uuid4
        from schema_extract.tools.agent import _session
        from schema_extract.tools.budget import extend
        from schema_extract.tools.runner import _resume_failed_scopes
        f = self.flow
        f.create(nodes=("root", "branch"))
        while f.state["stage"] != "P4":
            f.step()
        f.step(stage="P4", role="worker")
        attempt = f.prepare(stage="P4", role="reviewer")
        result = self.result(attempt, decision="revise", issues=[{"kind": "schema_conformance", "scope": "dispatch.md",
            "detail": "Clarify this writing handoff", "evidence": [0], "recovery_condition": "Patch once and sign off"}])
        f.state["stages"]["P4"]["revisions"] = f.state["config"]["revision_limit"] - 1
        f.store.save(f.state)
        f.accept(attempt, result)
        attempt = f.prepare(stage="P4", role="reviewer")
        result = self.result(attempt)
        result["response"]["adjudication"] = {"action": "patch", "checkpoint": attempt["base_checkpoint"],
            "patches": [{"path": "dispatch.md", "old": "临时机器行为验证材料", "new": "临时机器行为验证材料，保留已定写作安排"}]}
        f.accept(attempt, result)
        signoff = f.prepare(stage="P4", role="reviewer")
        wrong = self.result(signoff)
        wrong["response"]["plan"][0]["wait_for"] = ["root"]
        with self.assertRaisesRegex(ExtractError, "plan differs"):
            accept_result(f.store, f.state, f.lock, signoff, wrong)
        session_id = str(uuid4())
        with _session(signoff) as (path, record):
            record.update(status="running", attempt_id=signoff["attempt_id"], session_id=session_id)
            write_json(path, record)
        calls = f.state["stages"]["P4"]["calls"]
        f.state = _interrupted(f.store, f.state, f.lock, signoff, None, {"confirmed": True}, "Interrupted bounded final review")
        scope = f.state["stages"]["P4"]
        self.assertEqual((scope["adjudications"], scope["signoffs"], scope["calls"]), (1, 1, calls))
        self.assertEqual(scope["status"], "blocked")
        retired = read_json(Path(signoff["session_directory"]) / "session.json")
        self.assertEqual(retired["status"], "retired")
        self.assertIn(session_id, retired["retired_sessions"])
        extend(f.state, {"seconds": 1000, "calls": 10}, identity="explicit-fixture-extension", origin="resume")
        _resume_failed_scopes(f.state)
        self.assertEqual(f.state["stages"]["P4"]["status"], "blocked")
        self.assertIsNotNone(call_budget(f.state)[1])

    def test_checker_time_after_normalization_cannot_spend_reserved_review_time(self):
        f = self.flow
        f.create()
        attempt = f.prepare(stage="P1", role="worker")
        result = self.result(attempt)
        with patch("schema_extract.tools.runner._check_timeout", side_effect=[30.0, 30.0, 0.0]):
            f.accept(attempt, result)
        self.assertEqual(f.state["status"], "budget_exhausted")
        self.assertIsNone(f.state["stages"]["P1"]["approval"])
        self.assertEqual(f.state["stage"], "P1")
        self.assertIsNotNone(f.state["recoverable_checkpoint"])
        self.assertEqual((f.state["stages"]["P1"]["adjudications"], f.state["stages"]["P1"]["signoffs"]), (0, 0))

    def test_current_maintenance_uses_existing_scope_and_keeps_control_fence(self):
        f = self.start(("root", "branch"))
        self.approve_nodes("branch")
        before = deepcopy(f.state["nodes"]["branch"]["stage"])
        request = maintenance.submit_request(f.store, f.state["run_id"], issue_id="review-local-definition", tasks=[
            {"stage": "P5", "node": "branch", "action": "review", "fields": [{"path": "branch.yaml", "fields": ["DESIGN.Math_reshape"]}], "reason": "Independently recheck this precise local definition"}])
        self.assertTrue(maintenance.barrier(f.store, f.state))
        f.state = maintenance.consume_requests(f.store, f.state)
        f.state = maintenance.mark_drained(f.store, f.state)
        self.assertTrue(maintenance.publication_barrier(f.store, f.state))
        self.assertEqual(f.state["nodes"]["branch"]["stage"]["calls"], before["calls"])
        attempt = f.prepare(node="branch", role="reviewer")
        self.assertEqual(attempt["purpose"], "maintenance")
        result = self.result(attempt)
        result["response"]["revision_scope"] = []
        f.accept(attempt, result)
        self.assertEqual(f.state["maintenance"]["status"], "complete")
        self.assertEqual(f.state["nodes"]["branch"]["stage"]["calls"], before["calls"] + 1)
        repeated = maintenance.submit_request(f.store, f.state["run_id"], issue_id="review-local-definition", tasks=request["request"]["tasks"])
        self.assertTrue(repeated["reused"])
        self.assertEqual(f.state["stage"], "P6")
        from schema_extract.tools.stages import invalidate_node
        invalidate_node(f.state["nodes"]["branch"], require_worker=True)
        f.state.update(stage="P5", node="root", role=None, mode="normal", status="running")
        f.store.save(f.state)
        generation = f.state["nodes"]["branch"]["generation"]
        maintenance.submit_request(f.store, f.state["run_id"], issue_id="unfinished-current-draft", tasks=request["request"]["tasks"])
        f.state = maintenance.consume_requests(f.store, f.state)
        f.state = maintenance.mark_drained(f.store, f.state)
        self.assertEqual(f.state["nodes"]["branch"]["generation"], generation)
        self.assertTrue(f.state["nodes"]["branch"]["stage"]["last_approval"])
        call = next(call for call in next_calls(f.state) if call["node"] == "branch")
        self.assertEqual(call["role"], "worker")
        self.assertNotIn("purpose", call)
        maintenance.submit_request(f.store, f.state["run_id"], issue_id="scope-removed-during-drain", tasks=request["request"]["tasks"])
        selected = [entry for entry in f.state["selected_designs"] if entry["id"] == "root"]
        install_selection(f.state, selected, selection_impact(f.state, selected))
        f.store.save(f.state)
        f.state = maintenance.consume_requests(f.store, f.state)
        f.state = maintenance.mark_drained(f.store, f.state)
        self.assertFalse(maintenance.barrier(f.store, f.state))
        self.assertEqual(f.state["reason"]["code"], "maintenance_scope_changed")
        self.assertEqual(f.store.read_supervisor()["maintenance_requests"][f.state["run_id"]]["status"], "superseded")


if __name__ == "__main__":
    unittest.main()
