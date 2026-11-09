"""Provider repair and reparenting preserve real raw drafts and independent approval."""

from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import yaml

from schema_extract.tools import maintenance
from schema_extract.tools.artifacts import (DESIGN_REGIONS, index_prefix, load_yaml,
    parser_for, resolve_workspace, schema_for, validate_artifacts)
from schema_extract.tools.common import ExtractError, StaleResult, manifest, rules_root
from schema_extract.tools.prompts import build_prompt
from schema_extract.tools.receipts import (normalize_response, proposal_schema, provider_request_details,
    required_artifact_evidence)
from schema_extract.tools.revisions import approval_is_current
from schema_extract.tools.runner import (_merge_check, _retire_stale, accept_result)
from schema_extract.tools.runtime import _same_execution_semantics
from schema_extract.tools.stages import (_approve, effective_nodes, final_eligible, install_plan,
    invalidate_node, new_node, next_calls, plan_impact)
from schema_extract.tools.tests import test_semantic_revision


REPO = Path(__file__).resolve().parents[3]


class DerivationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.f = test_semantic_revision.SemanticRevisionTests(methodName="runTest")
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        f = self.f
        original = f.document
        f.document = yaml.safe_load((REPO / "schema_template/tools/tests/fixtures/minimal/root.yaml").read_text())
        f.document["meta_data"].update(original["meta_data"])
        f.document["contract"]["hardware_specification"] = original["contract"]["hardware_specification"]
        f.document["contract"]["algorithm_requirement"]["golden_reference"]["source"] = original["contract"]["algorithm_requirement"]["golden_reference"]["source"]
        f.put("root.yaml", f.document)
        f.origin("P1")
        f.document["meta_data"]["dispatch"] = {"id": "root", "when": "X < 0"}
        f.put("root.yaml", f.document)
        (f.work / "dispatch.md").write_text("Root [root](root.yaml).\n")
        f.origin("P2")
        f.origin("P3")
        names = [("root", ""), ("provider", "root"), ("middle", "provider"), ("child", "middle"),
                 ("sibling", "root"), ("cousin", "provider")]
        plan = [{"id": node, "file": node + ".yaml", "parent": parent, "workload": "Scalar task.",
                 "difference": "Implementation choice " + node, "selection": "kernel.py::kernel", "evidence": [0]}
                for node, parent in names]
        f.state["plan_draft"] = plan
        (f.work / "dispatch.md").write_text(index_prefix(plan) + "No known gaps.\n")
        for node, parent in names[1:]:
            value = deepcopy(f.document)
            value.pop("contract")
            value["meta_data"]["dispatch"] = {"id": node, "parent": parent, "when": "X >= 0"}
            if node in {"middle", "child"}:
                value["design"] = {region: {"from_parent": parent} for region in DESIGN_REGIONS}
            f.put(node + ".yaml", value)
        f.origin("P4")
        for node, _ in names[1:]:
            f.state["nodes"][node]["authored"] = True
            f.origin("P5", node)
        f.state.update(stage="P5", node="root", role=None, status="running")
        f.save()
        self.rules = rules_root(f.lock)

    def prepare(self, node):
        f = self.f
        call = next(item for item in next_calls(f.state) if item["node"] == node)
        attempt = f.store.prepare_attempt(f.state, call.get("session_id"), call=call, timeout_seconds=60)
        return attempt

    def proposal(self, attempt, decision):
        value = self.f.proposal(attempt, decision=decision)
        if attempt["stage"] == "P5":
            resolved = resolve_workspace(Path(attempt["workspace"]), parser_root=self.rules)
            for node in resolved.raw_task:
                for region in DESIGN_REGIONS:
                    position = resolved.documents[node].positions[f"{node}.design.{region}"]
                    value["artifact_evidence"].append({"path": resolved.source_files[node].name,
                        "line_start": position[0], "line_end": position[0], "quote_offset": 0,
                        "note": "Current raw region definition or direct-parent reference."})
        return value

    def accept(self, attempt, proposal):
        f = self.f
        result = f.records(attempt, proposal)
        f.state = accept_result(f.store, f.state, f.lock, attempt, result)
        f.state["unsettled_attempts"].pop(attempt["attempt_id"], None)
        f.save()
        return result

    def request(self, attempt, target="P5:provider"):
        proposal = self.proposal(attempt, "reopen")
        proposal["reopen"] = {"issue_id": "provider-definition", "target": target,
            "reason": "Correct the provider's documented scalar definition against fixed source.", "evidence": [0]}
        proposal["issues"] = [{"kind": "schema_conformance", "scope": "provider.yaml:design.math_reshape.thesis",
            "detail": "schema_docs/3.design.math_reshape.md requires the local thesis to agree with fixed scalar source.",
            "evidence": [0], "recovery_condition": "Provider worker checks the definition and an independent reviewer approves."}]
        return proposal

    def test_p5_prompt_keeps_current_ancestors_and_full_design_source_chains(self):
        f = self.f
        invalidate_node(f.state["nodes"]["child"])
        f.save()
        current = self.prepare("child")
        f.state["pending_rechecks"] = ["child", "sibling"]
        for role in ("worker", "reviewer"):
            with self.subTest(role=role):
                attempt = {**current, "role": role, "writable_files": ["child.yaml"] if role == "worker" else []}
                before = deepcopy((f.state, f.lock, attempt))
                prompt = build_prompt(f.state, f.lock, attempt)
                self.assertEqual((f.state, f.lock, attempt), before)
                value = json.loads(prompt.split("<invocation>\n", 1)[1].split("\n</invocation>", 1)[0])
                self.assertNotIn("plan", value)
                self.assertNotIn("reopening", value)
                self.assertEqual(value["node_entry"], next(entry for entry in f.state["plan"] if entry["id"] == "child"))
                self.assertEqual([entry["id"] for entry in value["ancestor_entries"]], ["middle", "provider", "root"])
                self.assertEqual(value["pending_rechecks"], ["child"])
                self.assertEqual(value["dependency_digests"], attempt["dependency_digests"])
                self.assertEqual(set(value["dependency_digests"]), {"root.yaml", "provider.yaml", "middle.yaml", "child.yaml"})
                self.assertEqual(value["writable_files"], attempt["writable_files"])
                graph = value["design_resolution"]["nodes"]
                self.assertEqual(set(graph), {"root", "provider", "middle", "child"})
                for region in DESIGN_REGIONS:
                    self.assertEqual(graph["child"]["origins"][region],
                                     {"provider": "provider", "chain": ["child", "middle", "provider"]})
                    self.assertEqual(graph["middle"]["origins"][region],
                                     {"provider": "provider", "chain": ["middle", "provider"]})
                self.assertEqual(set(graph["child"]["source_manifest"]),
                                 {"root.yaml", "child.yaml", "middle.yaml", "provider.yaml"})
                self.assertIn("schema_extract/phases/03_root.md", value["read_specs"])
                if role == "worker":
                    self.assertEqual(set(value["prior_approvals"]), {"P4"})
                    self.assertEqual(set(value["node_approvals"]), {"root", "provider", "middle"})
                    self.assertEqual(set(value["reviewed_artifacts"]), {"root", "provider", "middle"})
                    receipt = json.loads(Path(value["plan_receipt"]).read_text())
                    self.assertEqual(value["node_evidence"], [receipt["source_evidence"][0]])
                else:
                    self.assertEqual(set(value["required_artifact_evidence"]), set(attempt["dependency_digests"]))
                    for key in ("prior_approvals", "node_approvals", "reviewed_artifacts", "plan_receipt", "node_evidence"):
                        self.assertNotIn(key, value)

    def test_provider_repair_fences_descendants_and_closes_only_after_new_approval(self):
        f = self.f
        invalidate_node(f.state["nodes"]["child"])
        f.state["nodes"]["child"].update(role="worker", status="draft")
        invalidate_node(f.state["nodes"]["sibling"])
        invalidate_node(f.state["nodes"]["cousin"])
        f.state["stages"]["P6"].update(calls=5, revisions=2, adjudications=1, signoffs=1)
        f.state["p6_starts"] = 3
        f.save()
        sibling = self.prepare("sibling")
        cousin = self.prepare("cousin")
        child = self.prepare("child")
        document = load_yaml(Path(child["workspace"]), "child.yaml", parser_root=self.rules)
        document["meta_data"]["version"] += 1
        document["meta_data"]["dispatch"]["when"] = "X >= 1"
        (Path(child["workspace"]) / "child.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
        upstream = deepcopy({key: f.state["stages"][key] for key in ("P1", "P2", "P3", "P4")})
        old_approval = deepcopy(f.state["nodes"]["provider"]["approval"])
        budget = deepcopy(f.state["nodes"]["provider"]["stage"])
        child_bytes = (Path(child["workspace"]) / "child.yaml").read_bytes()
        self.assertEqual(set(child["provider_reopen_targets"]), {"P5:provider"})
        self.assertIn("P5:provider", proposal_schema(child)["properties"]["reopen"]["properties"]["target"]["enum"])
        self.accept(child, self.request(child))
        state = f.state
        self.assertIsNone(state["reopening"])
        self.assertEqual(state["stage"], "P5")
        self.assertEqual(upstream, {key: state["stages"][key] for key in upstream})
        self.assertEqual(state["active_attempts"], {"P5:sibling": sibling["attempt_id"], "P5:cousin": cousin["attempt_id"]})
        with self.assertRaises(StaleResult):
            f.store.validate_attempt(state, cousin)
        _retire_stale(f.store, state, cousin, {"termination": {"confirmed": True}}, "provider generation changed")
        f.state = f.store.load(f.run_id)
        self.assertEqual(state["nodes"]["provider"]["role"], "worker")
        self.assertEqual(state["nodes"]["middle"]["role"], "reviewer")
        self.assertFalse(approval_is_current(state, old_approval))
        self.assertFalse(final_eligible(state))
        self.assertEqual((state["p6_starts"], state["stages"]["P6"]["calls"], state["stages"]["P6"]["adjudications"]), (3, 5, 1))
        for field in ("calls", "revisions", "adjudications", "signoffs", "elapsed_seconds"):
            self.assertEqual(state["nodes"]["provider"]["stage"][field], budget[field])
        request_record = next(record for record in state["reopen_requests"].values() if record.get("provider"))
        saved, _ = f.store.verify_checkpoint(state, request_record["request_checkpoint"])
        self.assertEqual((saved / "child.yaml").read_bytes(), child_bytes)
        # An independent sibling approval still merges while the provider is pending.
        self.accept(sibling, self.proposal(sibling, "approve"))
        self.assertIn("sibling", effective_nodes(f.state))
        provider = self.prepare("provider")
        prompt = build_prompt(f.state, f.lock, provider)
        self.assertIn("provider_repair", prompt)
        self.assertIn("provider-definition", prompt)
        before = (Path(provider["workspace"]) / "provider.yaml").read_bytes()
        self.accept(provider, self.proposal(provider, "complete"))
        self.assertEqual(next_calls(f.state)[0]["role"], "reviewer")
        review = self.prepare("provider")
        self.assertEqual((Path(review["workspace"]) / "provider.yaml").read_bytes(), before)
        self.accept(review, self.proposal(review, "approve"))
        record = f.state["reopen_requests"][request_record["key"]]
        self.assertEqual(record["status"], "closed")
        self.assertNotEqual(record["closing_approval"]["attempt_id"], old_approval["attempt_id"])
        self.assertEqual(record["closed_generation"], old_approval["generation"] + 1)
        self.assertEqual(record["closing_approval"]["effective_identity"],
                         resolve_workspace(Path(review["workspace"]), parser_root=self.rules).effective_identity("provider"))
        for node in ("middle", "child", "cousin"):
            review = self.prepare(node)
            if review["role"] == "worker":
                self.accept(review, self.proposal(review, "complete"))
                review = self.prepare(node)
            self.assertEqual(review["role"], "reviewer")
            self.accept(review, self.proposal(review, "approve"))
        self.assertTrue(final_eligible(f.state))
        # A changed note or newly approved provider bytes do not create a new key.
        invalidate_node(f.state["nodes"]["child"])
        f.state.update(stage="P5", role=None, node="root", status="running")
        f.save()
        repeat = self.prepare("child")
        self.accept(repeat, self.request(repeat))
        self.assertEqual(f.state["nodes"]["child"]["reason"]["code"], "reopen_already_used")
        self.assertEqual(f.state["nodes"]["provider"]["generation"], 1)

    def test_submitted_graph_and_real_region_evidence_control_provider_eligibility(self):
        f = self.f
        invalidate_node(f.state["nodes"]["child"])
        f.save()
        attempt = self.prepare("child")
        proposal = self.request(attempt)
        receipt = normalize_response(proposal, f.lock, attempt, Path(attempt["workspace"]))
        details = provider_request_details(receipt, f.state, attempt, Path(attempt["workspace"]), parser_root=self.rules)
        self.assertEqual(details["regions"], {"math_reshape": ["child", "middle", "provider"]})
        self.assertEqual(required_artifact_evidence(f.state, attempt, attempt["dependency_digests"]),
                         {"root.yaml", "provider.yaml", "middle.yaml", "child.yaml"})
        for target in ("P5:middle", "P5:sibling", "P5:root"):
            invalid = deepcopy(receipt)
            invalid["reopen"]["target"] = target
            with self.assertRaisesRegex(ExtractError, "current approved strict ancestor"):
                provider_request_details(invalid, f.state, attempt, Path(attempt["workspace"]), parser_root=self.rules)
        invalid = deepcopy(receipt)
        invalid["artifact_evidence"] = [item for item in invalid["artifact_evidence"]
                                         if item["path"] != "provider.yaml" or item["line_start"] == 1]
        with self.assertRaisesRegex(ExtractError, "definition-region anchor"):
            provider_request_details(invalid, f.state, attempt, Path(attempt["workspace"]), parser_root=self.rules)
        invalid = deepcopy(receipt)
        invalid["issues"][0]["scope"] = "child.yaml:design.math_reshape"
        with self.assertRaisesRegex(ExtractError, "definition-region anchor"):
            provider_request_details(invalid, f.state, attempt, Path(attempt["workspace"]), parser_root=self.rules)
        for call in ({**attempt, "stage": "P6"}, {**attempt, "purpose": "maintenance"}, {**attempt, "mode": "adjudicate"}):
            with self.assertRaisesRegex(ExtractError, "ordinary P5"):
                provider_request_details(receipt, f.state, call, Path(attempt["workspace"]), parser_root=self.rules)
            if "reopen" in proposal_schema(call)["properties"]:
                self.assertNotIn("P5:provider", proposal_schema(call)["properties"]["reopen"]["properties"]["target"]["enum"])
        # The machine enum from invocation time cannot authorize a landed local region.
        resolved = resolve_workspace(Path(attempt["workspace"]), parser_root=self.rules)
        child = resolved.raw_task["child"]
        child["design"] = resolved.effective_task["child"]["design"]
        (Path(attempt["workspace"]) / "child.yaml").write_text(yaml.safe_dump(child, sort_keys=False))
        with self.assertRaisesRegex(ExtractError, "current approved strict ancestor"):
            provider_request_details(receipt, f.state, attempt, Path(attempt["workspace"]), parser_root=self.rules)

    def test_reparent_preserves_old_raw_fences_attempt_and_keeps_invalid_children_out_of_merge_check(self):
        f = self.f
        invalidate_node(f.state["nodes"]["child"])
        f.save()
        old_attempt = self.prepare("child")
        old_raw = (Path(old_attempt["workspace"]) / "child.yaml").read_bytes()
        old_checkpoint = f.state["nodes"]["child"]["draft_checkpoint"]
        new_plan = deepcopy(f.state["plan"])
        next(entry for entry in new_plan if entry["id"] == "child")["parent"] = "sibling"
        install_plan(f.state, new_plan, plan_impact(f.state, new_plan))
        migration = f.state["nodes"]["child"]["identity_migration"]
        self.assertEqual(migration["old_checkpoint"], old_checkpoint)
        self.assertEqual((migration["old_parent"], migration["new_parent"]), ("middle", "sibling"))
        self.assertEqual(f.state["nodes"]["child"]["role"], "worker")
        f.save()
        with self.assertRaises(StaleResult):
            f.store.validate_attempt(f.state, old_attempt)
        _retire_stale(f.store, f.state, old_attempt, {"termination": {"confirmed": True}}, "parent changed")
        f.state = f.store.load(f.run_id)
        worker = self.prepare("child")
        workspace = Path(worker["workspace"])
        self.assertEqual((workspace / "child.yaml").read_bytes(), old_raw)
        self.assertEqual(set(manifest(workspace)), {"root.yaml", "sibling.yaml", "child.yaml"})
        self.assertIn("diagnostic", worker["design_resolution"])
        prompt = build_prompt(f.state, f.lock, worker)
        self.assertIn("old_artifacts", prompt)
        self.assertIn(old_checkpoint, prompt)
        self.assertIn("retired_workspaces", prompt)
        merged, _ = f.store.verify_checkpoint(f.state, f.state["merged_checkpoint"])
        # The complete saved tree remains raw and still contains the old identity.
        self.assertEqual(load_yaml(merged, "child.yaml", parser_root=self.rules)["meta_data"]["dispatch"]["parent"], "middle")
        result = _merge_check(f.store, f.state, merged, node_id="sibling", parser_root=self.rules, timeout=30)
        self.assertEqual(result["exit_code"], 0, result)
        self.assertNotIn("child", result["checked_nodes"])
        self.assertFalse(final_eligible(f.state))
        changed = load_yaml(workspace, "child.yaml", parser_root=self.rules)
        changed["meta_data"]["dispatch"]["parent"] = "sibling"
        changed["meta_data"]["version"] += 1
        changed["design"] = {region: {"from_parent": "sibling"} for region in DESIGN_REGIONS}
        (workspace / "child.yaml").write_text(yaml.safe_dump(changed, sort_keys=False))
        with self.assertRaisesRegex(ExtractError, "outside its evidence-bound scope"):
            before, _ = f.store.verify_checkpoint(f.state, worker["base_checkpoint"])
            validate_artifacts("P5", "worker", before, workspace, self.rules / "schema_template/starter.yaml",
                decision="complete", node_entry=next(item for item in new_plan if item["id"] == "child"),
                maintenance_fields=[{"path": "child.yaml", "fields": ["design"]}], identity_migration=migration)
        self.accept(worker, self.proposal(worker, "complete"))
        self.assertIn("identity_migration", f.state["nodes"]["child"])
        reviewer = self.prepare("child")
        self.accept(reviewer, self.proposal(reviewer, "approve"))
        self.assertNotIn("identity_migration", f.state["nodes"]["child"])
        self.assertEqual(f.state["nodes"]["child"]["identity_migration_history"][-1]["status"], "complete")

    def test_first_seed_references_parent_but_existing_local_body_is_preserved(self):
        f = self.f
        entry = {"id": "fresh", "file": "fresh.yaml", "parent": "provider", "workload": "new actual path",
                 "difference": "local implementation choice", "selection": "kernel", "evidence": [0]}
        f.state["plan"].append(entry)
        f.state["nodes"]["fresh"] = new_node(entry)
        f.store.start_node(f.state, "fresh")
        directory, _ = f.store.verify_checkpoint(f.state, f.state["nodes"]["fresh"]["draft_checkpoint"])
        seeded = load_yaml(directory, "fresh.yaml", parser_root=self.rules)
        self.assertNotIn("contract", seeded)
        self.assertNotIn("realization", seeded)
        self.assertEqual(seeded["design"], {region: {"from_parent": "provider"} for region in DESIGN_REGIONS})
        self.assertEqual(seeded["meta_data"]["dispatch"], {"id": "fresh", "parent": "provider", "when": ""})
        self.assertEqual(seeded["meta_data"]["version"], 1)
        old, _ = f.store.verify_checkpoint(f.state, f.state["nodes"]["provider"]["draft_checkpoint"])
        raw = (old / "provider.yaml").read_bytes()
        f.store.start_node(f.state, "provider")
        refreshed, _ = f.store.verify_checkpoint(f.state, f.state["nodes"]["provider"]["draft_checkpoint"])
        self.assertEqual((refreshed / "provider.yaml").read_bytes(), raw)

    def test_frozen_adapter_and_semantic_runtime_boundary(self):
        self.assertTrue(schema_for(self.rules).__name__.startswith(parser_for(str(self.rules)).__package__ + "."))
        original = resolve_workspace(self.f.work, parser_root=self.rules)
        with patch("schema_template.tools.schema.resolve_directory", side_effect=AssertionError("installed rules bypass")):
            resolved = resolve_workspace(self.f.work, parser_root=self.rules)
            self.assertEqual(resolved.effective_identity("child"), original.effective_identity("child"))
        with patch("schema_extract.tools.runtime._executor", side_effect=lambda lock: lock):
            rules = {"files": {"schema_extract/tools/runtime.py": "a" * 64}}
            old = {"rules": rules, "capabilities": ["semantic_revision.v1"]}
            new = {"rules": rules, "capabilities": ["semantic_revision.v1", "design_region_derivation.v1"]}
            _same_execution_semantics(old, old)
            _same_execution_semantics(new, new)
            with self.assertRaisesRegex(ExtractError, "require semantic revision"):
                _same_execution_semantics(old, new)

    def test_maintenance_keep_cannot_skip_repair_and_plan_removal_retires_record(self):
        f = self.f
        old_approval = deepcopy(f.state["nodes"]["provider"]["approval"])
        old_receipt, _ = f.store.verify_approval(f.state, old_approval)
        invalidate_node(f.state["nodes"]["child"])
        f.save()
        attempt = self.prepare("child")
        self.accept(attempt, self.request(attempt))
        key = f.state["nodes"]["provider"]["provider_reopen"]
        with self.assertRaisesRegex(ExtractError, "current-generation worker"):
            _approve(f.state, old_receipt, old_approval["checkpoint"], old_approval["receipt_sha256"], {})
        task = {"scope": "P5:provider", "stage": "P5", "node": "provider", "action": "keep",
                "status": "done", "complete_candidate": True, "fields": [], "basis": {}, "history": []}
        f.state["maintenance"] = {"status": "active", "tasks": {"repair": task}, "completed": ["repair"],
                                  "normal_cursor": {"stage": "P5", "role": None}}
        maintenance._initialize(f.state)
        self.assertEqual(task["status"], "blocked")
        self.assertEqual(task["reason"]["code"], "maintenance_patch_scope_missing")
        self.assertEqual(task["fields"], [])
        task.update(action="patch", status="done", history=[{"decision": "approve"}],
                    fields=[{"path": "provider.yaml", "fields": ["design.math_reshape"]}])
        maintenance._initialize(f.state)
        self.assertEqual(task["reason"]["code"], "maintenance_dependency_changed_after_review")
        plan = [deepcopy(item) for item in f.state["plan"] if item["id"] != "provider"]
        for item in plan:
            if item["parent"] == "provider":
                item["parent"] = "sibling"
        install_plan(f.state, plan, plan_impact(f.state, plan))
        maintenance._initialize(f.state)
        self.assertEqual(task["status"], "removed")
        self.assertEqual(f.state["reopen_requests"][key]["status"], "retired")
        self.assertEqual(f.state["reopen_requests"][key]["previous_approval"], old_approval)
        self.assertNotIn("provider_reopen", f.state["nodes"]["provider"])

    def test_changed_provider_merge_checks_candidate_before_old_consumer_bindings(self):
        f = self.f
        child = load_yaml(f.work, "child.yaml", parser_root=self.rules)
        child["design"]["work_schedule"] = deepcopy(f.document["design"]["work_schedule"])
        f.put("child.yaml", child)
        f.origin("P5", "child")
        f.save()
        provider = load_yaml(f.work, "provider.yaml", parser_root=self.rules)
        provider["design"]["work_schedule"]["phase"]["copy"]["mapping"]["worker"]["unit_count"] = 2
        provider["design"]["tile_pipeline"]["phase"]["copy"]["role"]["Main"]["info"]["member"] = "CTA[0:2]"
        provider["meta_data"]["version"] += 1
        f.put("provider.yaml", provider)
        with self.assertRaisesRegex(schema_for(self.rules).SchemaError, "member index exceeds"):
            schema_for(self.rules).validate_resolved_file(f.work / "provider.yaml", skip_hardware=True)
        checked = _merge_check(f.store, f.state, f.work, node_id="provider", parser_root=self.rules, timeout=30)
        self.assertEqual(checked["exit_code"], 0, checked)
        self.assertEqual(checked["checked_nodes"], ["provider", "root", "sibling"])
        self.assertEqual(checked["superseded_dependents"], ["child", "cousin", "middle"])
        self.assertIn("child.yaml", manifest(f.work))


if __name__ == "__main__":
    unittest.main()
