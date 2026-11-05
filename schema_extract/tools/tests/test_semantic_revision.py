"""Five combined integration scenarios, using isolated stores and local evidence.

Historical approvals are synthetic accepted records. New maintenance goes through
the real artifact/receipt gates and state machine; only full schema completeness
uses a local checker substitute where the small fixture omits domain detail.
"""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import time
import unittest
from unittest.mock import patch
from uuid import uuid4

import yaml

from schema_extract.tools import maintenance
from schema_extract.tools.artifacts import check_complete, index_prefix, load_yaml
from schema_extract.tools.audit import audit_run
from schema_extract.tools.recovery import account
from schema_extract.tools.common import ExtractError, digest, manifest, read_json, rules_root, write_json
from schema_extract.tools.compatibility import PRESERVED, build_report, verify_report
from schema_extract.tools.prompts import build_prompt
from schema_extract.tools.publication import publish
from schema_extract.tools.receipts import normalize_response, proposal_schema
from schema_extract.tools.revision_control import (apply_prepared_revision,
    mark_revision_drained, prepare_revision, publication_barrier, request_revision)
from schema_extract.tools.revisions import resolve_context, verify_revisions
from schema_extract.tools.runner import accept_result, _reconcile_orphans, _resume_failed_scopes
from schema_extract.tools.runtime import upgrade_runtime, verify_execution
from schema_extract.tools.stages import (STAGES, _approve, new_node, next_calls, node_signature,
    required_conclusions, scope_for, session_group)
from schema_extract.tools.tests import test_cli_recovery


class SemanticRevisionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_cli_recovery.CLIRecoveryTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.store, self.root = self.fixture.store, self.fixture.root
        self.run_id = self.fixture.run_id
        self.lock = read_json(self.store.run_dir(self.run_id) / "inputs.lock.json")
        self.state = self.store.load(self.run_id)
        self.state["status"] = "running"
        self.work = self.root / "artifacts"
        self.work.mkdir()
        self.document = yaml.safe_load((rules_root(self.lock) / "schema_template/starter.yaml").read_text())
        source = self.lock["source"]
        prefix = source["repository_id"] + "@" + source["ref"] + ":kernel.py::"
        self.document["meta_data"].update(name="fixture", describe="Retained scalar fixture.",
            source=prefix + "kernel", version=1)
        self.document["contract"]["algorithm_requirement"] = {
            "golden_reference": {"source": prefix + "reference"}, "retained_fact": "original Contract"}
        self.put("root.yaml", self.document)
        self.state["checkpoint"] = self.checkpoint()
        self.state["nodes"]["root"]["draft_checkpoint"] = self.state["checkpoint"]
        self.save()

    def put(self, name, value):
        (self.work / name).write_text(yaml.safe_dump(value, sort_keys=False))

    def checkpoint(self):
        return self.store.checkpoint(self.state, self.work, attempt_id=None)

    def save(self):
        self.store.save(self.state)

    def reload(self):
        self.state = self.store.load(self.run_id)
        return self.state

    def proposal(self, attempt, *, decision="approve"):
        workspace = Path(attempt["workspace"])
        def anchor(path, **extra):
            return {"path": path, "line_start": 1, "line_end": 1, "quote_offset": 0,
                    "note": "Isolated evidence for the current scope.", **extra}
        evidence = []
        for category in STAGES[attempt["stage"]]["evidence"]:
            binding = self.lock["source"]["bindings"]["entrypoint" if category == "implementation" else category]
            entry = binding[0] if isinstance(binding, list) else binding
            evidence.append(anchor(entry["path"], category=category))
        value = {"format": "schema_extract.proposal.v1", "attempt_id": attempt["attempt_id"],
                 "decision": decision, "summary": "The retained fixture satisfies this review.",
                 "source_evidence": evidence,
                 "artifact_evidence": [anchor(name) for name in manifest(workspace)],
                 "conclusions": [{"subject": subject, "text": "The current full stage content was reviewed.",
                                  "evidence": [0]} for subject in required_conclusions(attempt)],
                 "checks": [], "checks_note": "Local integration fixture; Runner gates remain authoritative.",
                 "issues": [], "framework_candidates": [], "handoff": "Continue with the retained candidate.",
                 "reopen": None}
        if attempt["stage"] == "P1":
            value["comparisons"] = []
        if attempt["stage"] == "P4":
            value["plan"] = self.state["plan_draft"]
        if attempt["stage"] == "P6":
            value.update(affected_nodes=[entry["id"] for entry in self.state["plan"]], revision_scope=[])
        if attempt.get("purpose") == "maintenance":
            value["revision_scope"] = []
        return {key: item for key, item in value.items() if key in proposal_schema(attempt)["properties"]}

    def records(self, attempt, proposal):
        directory = Path(attempt["directory"])
        session_id = attempt.get("session_id") or str(uuid4())
        result = {"outcome": "completed", "termination": {"confirmed": True}, "session_id": session_id,
                  "response": proposal, "elapsed_seconds": 1.0}
        write_json(directory / "response.json", proposal)
        write_json(directory / "result.json", result)
        (directory / "events.jsonl").write_text(json.dumps({"type": "thread.started", "thread_id": session_id}) + "\n")
        (directory / "prompt.txt").write_text("Isolated integration fixture.\n")
        return result

    def origin(self, stage, node="root"):
        """Install a complete old approval, including unchanged raw evidence."""
        checkpoint = self.checkpoint()
        artifacts, info = self.store.verify_checkpoint(self.state, checkpoint)
        identity = str(uuid4())
        directory = self.store.run_dir(self.run_id) / "attempts" / identity
        directory.mkdir(parents=True)
        shutil.copytree(artifacts, directory / "workspace")
        entry = next((entry for entry in self.state.get("plan") or [] if entry["id"] == node), None)
        call = {"stage": stage, "node": node, "role": "reviewer", "mode": "normal"}
        attempt = {"run_id": self.run_id, "attempt_id": identity, "input_fingerprint": self.state["input_fingerprint"],
            **call, "plan_revision": self.state["plan_revision"], "generation": 0,
            "base_checkpoint": checkpoint, "dependency_digests": info["files"], "session_id": None,
            "session_group": session_group(self.state, call), "writable_files": [], "revision_scope": [],
            "response_format": "schema_extract.proposal.v1", "created_at": time.time(),
            "node_signature": node_signature(entry) if stage == "P5" else "",
            "directory": str(directory), "workspace": str(directory / "workspace")}
        write_json(directory / "attempt.json", attempt)
        self.state["attempts"].append(identity)
        proposal = self.proposal(attempt)
        result = self.records(attempt, proposal)
        receipt = normalize_response(proposal, self.lock, attempt, directory / "workspace")
        receipt_hash = digest(receipt)
        validation = {"artifact_digests": info["files"], "node_signature": attempt["node_signature"],
                      "dependency_digests": {name: value for name, value in info["files"].items()
                                              if name != self.state["nodes"][node]["file"]},
                      "check": {"kind": "synthetic_origin", "exit_code": 0}}
        write_json(directory / "receipt.json", receipt)
        write_json(directory / "validated.json", {"receipt_sha256": receipt_hash, **validation})
        self.state["accepted_receipts"][identity] = receipt_hash
        _approve(self.state, receipt, checkpoint, receipt_hash, validation)
        scope_for(self.state, attempt).update(calls=2, elapsed_seconds=9.0)
        account(self.state, attempt, 9.0, result=result, final=True)
        self.state["checkpoint"] = checkpoint
        self.state["nodes"][node]["draft_checkpoint"] = checkpoint
        return deepcopy(scope_for(self.state, attempt)["approval"])

    def upstream(self, through="P3", *, branches=False):
        origins = {"P1": self.origin("P1")}
        self.document["meta_data"]["dispatch"] = {"id": "root", "status": "draft"}
        self.put("root.yaml", self.document)
        (self.work / "dispatch.md").write_text("Retained dispatch draft links [root](root.yaml).\n")
        origins["P2"] = self.origin("P2")
        if through == "P2":
            self.save()
            return origins
        self.document["meta_data"]["dispatch"]["when"] = "x >= 0"
        self.document["design"] = {"own": "root design"}
        self.put("root.yaml", self.document)
        origins["P3"] = self.origin("P3")
        names = [("root", ""), ("a", "root"), ("child", "a"), ("b", "root")] if branches else [("root", "")]
        plan = [{"id": name, "file": name + ".yaml", "parent": parent, "workload": "Scalar inputs.",
                 "difference": "Selected " + name, "selection": "kernel.py::kernel", "evidence": [0]}
                for name, parent in names]
        self.state["plan_draft"] = plan
        (self.work / "dispatch.md").write_text(index_prefix(plan) + "No known coverage gaps.\n")
        if branches:
            for name, parent in names[1:]:
                own = deepcopy(self.document)
                own.pop("contract")
                own["meta_data"]["dispatch"] = {"id": name, "parent": parent, "when": "x >= 0"}
                own["design"] = {"own": name + " private design"}
                self.put(name + ".yaml", own)
        self.state["checkpoint"] = self.checkpoint()
        if through == "P4":
            origins["P4"] = self.origin("P4")
            for name, _ in names[1:]:
                self.state["nodes"][name]["authored"] = True
                origins["P5:" + name] = self.origin("P5", name)
            self.state.update(stage="P5", node="root", role=None, status="running")
        else:
            self.state.update(stage="P4", node="root", role="reviewer", status="running")
            self.state["stages"]["P4"].update(status="reviewing", calls=1, elapsed_seconds=5.0)
        self.save()
        return origins

    def revise(self, actions, *, target=None, reason="isolated semantic issue", prepare_only=False):
        request = request_revision(self.store, self.run_id, reason=reason, target_repo=target)
        first = prepare_revision(self.store, self.run_id)
        covered = [item["path"] for item in first["plan"]["report"]["changes"]]
        scopes = []
        for item in first["plan"]["scopes"]:
            supplied = {"scope": item["scope"], "action": "keep", "reason": "Unchanged complete evidence remains applicable."}
            if item["basis"]["approval"]:
                supplied["assessment"] = {"decision": "adopt", "unchanged_content": True,
                    "requirements_covered": True, "dependencies_compatible": True, "reason": "Reviewed all target changes.",
                    "evidence": ["isolated complete comparison"], "covered_changes": covered, "covered_issues": ["fixture"]}
            supplied.update(actions.get(item["scope"], {}))
            scopes.append(supplied)
        proposal = {"reviewer": {"role": "Supervisor", "identity": "independent-fixture",
                      "record": "isolated plan review", "independent_review": True},
                    "reason": reason, "issues": ["fixture"], "evidence": ["isolated plan basis"], "scopes": scopes}
        prepared = prepare_revision(self.store, self.run_id, proposal=proposal)
        self.assertEqual(prepared["status"], "prepared")
        if prepare_only:
            return prepared
        result = apply_prepared_revision(self.store, self.run_id)
        self.reload()
        maintenance.settle(self.state)
        self.save()
        return result

    def accept(self, *, decision=None, mutate=None):
        call = next_calls(self.state)[0]
        attempt = self.store.prepare_attempt(self.state, call.get("session_id"), call=call, timeout_seconds=60)
        account(self.state, attempt, 0)
        self.save()
        if mutate:
            mutate(Path(attempt["workspace"]))
        proposal = self.proposal(attempt, decision=decision or ("approve" if call["role"] == "reviewer" else "complete"))
        result = self.records(attempt, proposal)
        with patch("schema_extract.tools.runner.check_complete", return_value={"exit_code": 0, "kind": "local_fixture"}):
            self.state = accept_result(self.store, self.state, self.lock, attempt, result)
        self.state["unsettled_attempts"].pop(attempt["attempt_id"], None)
        account(self.state, attempt, 1.0, result=result, final=True)
        self.save()
        return attempt

    def assert_audit(self):
        self.state["status"] = "paused"
        self.save()
        report = audit_run(self.store, self.run_id, verify=True)
        self.assertEqual(report["verification"]["errors"], [], report["verification"])

    def test_partial_p1_p2_preserve_drafts_roles_and_budget(self):
        original = self.state["checkpoint"]
        self.state["stages"]["P1"].update(calls=3, revisions=1, adjudications=1, elapsed_seconds=11)
        self.save()
        self.revise({"P1": {"action": "review"}})
        self.assertEqual((self.state["stage"], self.state["role"], self.state["checkpoint"]), ("P1", "worker", original))
        self.assertIsNone(self.state["stages"]["P1"]["approval"])
        self.assertEqual(self.state["stages"]["P1"]["adjudications"], 1)
        call = next_calls(self.state)[0]
        attempt = self.store.prepare_attempt(self.state, None, call=call)
        self.assertEqual((Path(attempt["workspace"]) / "root.yaml").read_bytes(), (self.work / "root.yaml").read_bytes())
        # A separate partial P2 shares this one scenario, without replaying P1.
        self.setUp()
        p1 = self.origin("P1")
        self.document["meta_data"]["dispatch"] = {"id": "root", "status": "draft"}
        self.put("root.yaml", self.document)
        (self.work / "dispatch.md").write_text("Unfinished retained route [root](root.yaml).\n")
        self.state.update(stage="P2", role="worker", checkpoint=self.checkpoint())
        self.state["stages"]["P2"].update(calls=3, elapsed_seconds=17)
        self.save()
        before = self.state["checkpoint"]
        self.revise({"P2": {"action": "review"}})
        self.assertEqual((self.state["stage"], self.state["role"], self.state["checkpoint"]), ("P2", "worker", before))
        self.assertIsNone(self.state["stages"]["P2"]["approval"])
        self.assertEqual(self.state["stages"]["P1"]["approval"], p1)
        self.store.verify_effective_approval(self.state, p1)
        self.assertEqual(self.state["stages"]["P2"]["calls"], 3)

    def test_p4_reviewer_first_preserves_prior_stage_material_and_audit(self):
        prior = self.upstream("P4", branches=True)
        prior = {key: value for key, value in prior.items() if key in {"P1", "P2", "P3"}}
        self.state.update(stage="P4", role="reviewer")
        self.save()
        result = self.fixture.cli("continue", self.run_id)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "awaiting_evaluation")
        self.assertFalse(json.loads(result.stdout)["delivered"])
        self.assertEqual(next(item for item in json.loads(result.stdout)["plan"]["scopes"] if item["scope"] == "P1")["action"], "review")
        malformed = self.fixture.cli("continue", self.run_id, "--full-restart", "--checkpoint", self.state["checkpoint"])
        self.assertEqual(malformed.returncode, 2)
        reason = self.store.read_supervisor()["semantic_requests"][self.run_id]["reason"]
        self.revise({"P4": {"action": "review"}}, reason=reason)
        self.assertEqual([(call["stage"], call["role"]) for call in next_calls(self.state)], [("P4", "reviewer")])
        attempt = self.accept()
        self.assertEqual(attempt["role"], "reviewer")
        self.assertEqual(set(manifest(Path(attempt["workspace"]))), {"root.yaml", "dispatch.md"})
        merged, _ = self.store.verify_checkpoint(self.state, self.state["merged_checkpoint"])
        self.assertEqual((merged / "b.yaml").read_bytes(), (self.work / "b.yaml").read_bytes())
        self.assertEqual(self.state["stage"], "P6")
        for stage, approval in prior.items():
            self.assertEqual(self.state["stages"][stage]["approval"], approval)
            self.store.verify_approval_origin(self.state, approval)
            self.store.verify_effective_approval(self.state, approval)
        self.store.verify_effective_approval(self.state, self.state["stages"]["P4"]["approval"])
        self.assert_audit()

    def test_p5_patch_preserves_sibling_and_child_while_refreshing_parent(self):
        prior = self.upstream("P4", branches=True)
        siblings = {name: (self.work / (name + ".yaml")).read_bytes() for name in ("b", "child")}
        self.revise({"P5:a": {"action": "patch", "reason": "Fix one copied design fact.",
            "evidence": ["a.yaml design.own"], "fields": [{"path": "a.yaml", "fields": ["design.own", "meta_data.version"]}]}})
        self.assertEqual((next_calls(self.state)[0]["node"], next_calls(self.state)[0]["role"]), ("a", "worker"))
        def mutate(work):
            path = work / "a.yaml"
            value = yaml.safe_load(path.read_text())
            value["design"]["own"] = "patched a design"
            value["meta_data"]["version"] += 1
            path.write_text(yaml.safe_dump(value, sort_keys=False))
        self.accept(decision="continue", mutate=mutate)
        self.assertEqual(next_calls(self.state)[0]["maintenance_step"], "patch")
        self.assertEqual(self.state["nodes"]["a"]["stage"]["continues"], 1)
        task = self.state["maintenance"]["tasks"][next_calls(self.state)[0]["maintenance_task_id"]]
        # A budget stop between successful authoring calls can resume this
        # same logical patch; a failed verification cannot acquire a new pass.
        task["status"], self.state["status"] = "budget_exhausted", "budget_exhausted"
        _resume_failed_scopes(self.state)
        self.assertEqual(task["status"], "pending")
        self.save()
        self.accept()
        self.assertEqual(next_calls(self.state)[0]["maintenance_step"], "verify")
        self.accept()
        self.assertEqual((next_calls(self.state)[0]["node"], next_calls(self.state)[0]["role"]), ("child", "reviewer"))
        child = self.accept()
        work = Path(child["workspace"])
        self.assertEqual((work / "child.yaml").read_bytes(), siblings["child"])
        self.assertIn("patched a design", (work / "a.yaml").read_text())
        merged, info = self.store.verify_checkpoint(self.state, self.state["merged_checkpoint"])
        self.assertEqual((merged / "b.yaml").read_bytes(), siblings["b"])
        self.assertEqual(self.state["nodes"]["b"]["approval"], prior["P5:b"])
        self.store.verify_effective_approval(self.state, prior["P5:b"])
        self.assert_audit()

    def test_new_rules_are_read_and_origins_survive_runtime_interleaving(self):
        prior = self.upstream()
        target = self.root / "target"
        shutil.copytree(rules_root(self.lock), target)
        page = target / "schema_extract/phases/04_dispatch_plan.md"
        page.chmod(0o644)
        page.write_text(page.read_text() + "\nSEMANTIC_TARGET_MARKER: review the retained candidate.\n")
        checker = target / "schema_template/check_yaml.py"
        checker.chmod(0o644)
        checker.write_text("print('SEMANTIC_TARGET_CHECKER')\nraise SystemExit(1)\n")
        self.revise({"P4": {"action": "review"}}, target=target)
        call = next_calls(self.state)[0]
        attempt = self.store.prepare_attempt(self.state, None, call=call)
        context = resolve_context(self.store, self.state, attempt)
        prompt = build_prompt(self.state, self.lock, attempt, rules_lock=context["rules_lock"], executor_lock=context["executor_lock"])
        self.assertIn("SEMANTIC_TARGET_MARKER", prompt)
        checked = check_complete(Path(attempt["workspace"]), rules_root(context["rules_lock"]))
        self.assertEqual((checked["exit_code"], checked["stdout"].strip()), (1, "SEMANTIC_TARGET_CHECKER"))
        # Accept the prepared read-only P4 through its ordinary stage gate.
        response = self.proposal(attempt)
        result = self.records(attempt, response)
        self.state = accept_result(self.store, self.state, self.lock, attempt, result)
        self.state["unsettled_attempts"].pop(attempt["attempt_id"], None)
        account(self.state, attempt, 1, result=result, final=True)
        self.state["status"] = "paused"
        self.save()
        for approval in prior.values():
            self.store.verify_approval_origin(self.state, approval)
            self.store.verify_effective_approval(self.state, approval)
        runtime_target = self.root / "runtime-target"
        shutil.copytree(rules_root(context["rules_lock"]), runtime_target)
        agent = runtime_target / "schema_extract/tools/agent.py"
        agent.chmod(0o644)
        agent.write_text(agent.read_text() + "\n# Isolated compatible runtime fixture.\n")
        report = build_report(self.store, self.run_id, self.state["checkpoint"], runtime_target)
        self.assertEqual([item["path"] for item in report["changes"]], ["schema_extract/tools/agent.py"])
        review = {"format": "schema_extract.compatibility_review.v1", "report_id": report["id"],
            "supervisor": {"role": "Supervisor", "identity": "independent-fixture", "independent_review": True,
                           "record": "isolated full diff review"}, "decision": "compatible_runtime",
            "reason": "Only a comment changes.", "preserves": dict.fromkeys(PRESERVED, True),
            "changes": [{key: item[key] for key in ("path", "before_sha256", "after_sha256")} |
                        {"classification": "runtime_fix", "reason": "Comment only.", "evidence": ["Full diff."]}
                        for item in report["changes"]]}
        upgrade_runtime(self.store, self.run_id, report, review, target_repo=runtime_target)
        self.reload()
        verify_execution(self.store, self.state, attempts=[attempt])
        self.assertEqual(read_json(self.store.run_dir(self.run_id) / "runtime" / self.state["execution_binding"] / "report.json"), report)
        self.assertEqual(resolve_context(self.store, self.state, attempt)["rules_lock"], context["rules_lock"])
        self.assert_audit()

    def test_request_drain_retry_pause_deadline_and_old_p6_publication_fence(self):
        self.upstream("P4")
        self.state.update(stage="P6", role="reviewer", status="running")
        self.state["stages"]["P6"].update(status="reviewing", adjudications=1, signoffs=1)
        self.save()
        attempt = self.store.prepare_attempt(self.state, None, call=next_calls(self.state)[0], timeout_seconds=60)
        request = request_revision(self.store, self.run_id, reason="late final semantic issue")
        again = request_revision(self.store, self.run_id, reason="late final semantic issue")
        self.assertEqual(request["request"]["id"], again["request"]["id"])
        self.assertTrue(again["reused"])
        self.assertEqual(prepare_revision(self.store, self.run_id)["status"], "draining")
        self.assertTrue(publication_barrier(self.store, self.state))
        with self.assertRaises(ExtractError):
            mark_revision_drained(self.store, self.state)
        deadline = self.state["deadline"]
        # The same saved deadline still bounds the in-flight call.
        observed = []
        with patch("schema_extract.tools.runner.termination_observed", return_value=False), \
             patch("schema_extract.tools.runner._reconcile", side_effect=lambda store, state, lock, **kw: (observed.append(kw["scopes"]) or state, [])), \
             patch("schema_extract.tools.runner.time.time", return_value=deadline + 1):
            _reconcile_orphans(self.store, self.state, self.lock, {}, set())
        self.assertEqual(observed, [{"P6"}])
        response = self.proposal(attempt)
        result = self.records(attempt, response)
        with patch("schema_extract.tools.runner.check_complete", return_value={"exit_code": 0}):
            self.state = accept_result(self.store, self.state, self.lock, attempt, result)
        self.state["unsettled_attempts"].pop(attempt["attempt_id"], None)
        account(self.state, attempt, 1, result=result, final=True)
        self.save()
        self.assertEqual(self.state["status"], "ready")
        self.assertTrue(publication_barrier(self.store, self.state))
        with self.assertRaisesRegex(ExtractError, "prevents publication"):
            publish(self.store, self.state, self.root / "forbidden-delivery")
        self.store.request_control(self.run_id, "pause")
        self.revise({}, reason="late final semantic issue", prepare_only=True)
        with self.store.supervisor() as ledger:
            ledger["active_hold"] = {"id": str(uuid4())}
        with patch("schema_extract.tools.monitor.start_background", side_effect=AssertionError("pause/hold must preserve launch")):
            first = apply_prepared_revision(self.store, self.run_id, repo=test_cli_recovery.REPO)
            self.assertEqual(first["launch"]["status"], "preserved")
            # Model a crash after the state pointer commits but before the
            # control acknowledgement, then retry the registered apply.
            with self.store.supervisor() as ledger:
                ledger["semantic_requests"][self.run_id]["status"] = "prepared"
            second = apply_prepared_revision(self.store, self.run_id, repo=test_cli_recovery.REPO)
        self.assertEqual(first["revision"], second["revision"])
        self.assertTrue(second["reused"])
        self.reload()
        self.assertEqual(self.state["deadline"], deadline)
        self.assertEqual(self.state["p6_starts"], 1)
        self.assertEqual(self.state["stages"]["P6"]["signoffs"], 1)
        self.assertEqual(len(verify_revisions(self.store, self.state)), 1)
        self.assertTrue(publication_barrier(self.store, self.state))


if __name__ == "__main__":
    unittest.main()
