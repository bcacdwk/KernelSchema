"""A frozen executor repair preserves real checkpoints, approval origins and controls."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

from schema_extract.tools import executor_repairs
from schema_extract.tools.artifacts import load_yaml, write_yaml
from schema_extract.tools.common import ExtractError, read_json, rules_root, write_json
from schema_extract.tools.runtime import verify_execution
from schema_extract.tools.snapshots import freeze_executor_patch, frozen_invocation
from schema_extract.tools.tests import test_stage_flow

REPO = test_stage_flow.REPO


class ExecutorRepairTests(unittest.TestCase):
    def setUp(self):
        self.f = test_stage_flow.StageFlowTests(methodName="runTest")
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.f.create(("root", "branch", "sibling"))

    def frontier(self, *, decision="complete", save="accepted"):
        f = self.f
        while f.state["stage"] != "P6":
            f.step()
        old_write, old_proposal = f.backend.write, f.backend.proposal
        self.p6_node = "branch"

        def write(attempt):
            old_write(attempt)
            if attempt["stage"] == "P6":
                workspace = Path(attempt["workspace"])
                value = load_yaml(workspace, self.p6_node + ".yaml", parser_root=f.backend.rules)
                value["DESIGN"]["Tile_pipeline"]["Thesis"] += " Current P6 candidate."
                value["META_DATA"]["Version"] += 1
                write_yaml(workspace / (self.p6_node + ".yaml"), value, parser_root=f.backend.rules)

        def proposal(attempt):
            value = old_proposal(attempt)
            if attempt["stage"] == "P6":
                value["affected_nodes"] = [self.p6_node]
                if attempt["role"] == "worker":
                    value["decision"] = decision
                if value["decision"] == "revise":
                    value["revision_scope"].append({"path": self.p6_node + ".yaml", "fields": ["DESIGN.Tile_pipeline.Thesis"]})
                elif value["decision"] == "approve":
                    value["revision_scope"] = []
            return value

        f.backend.write, f.backend.proposal = write, proposal
        f.step()  # Real accepted reviewer scope.
        if save == "accepted":
            f.step()  # Still no final approval.
        else:
            from schema_extract.tools.recovery import account
            from schema_extract.tools.runner import _interrupted, _receive
            attempt = f.prepare()
            result = f.result(attempt)
            if save == "receipt_repair":
                del result["response"]["handoff"]
                write_json(Path(attempt["directory"]) / "result.json", result)
            account(f.state, attempt, 0, result=result)
            f.store.save(f.state)
            if save == "receipt_repair":
                f.state = _receive(f.store, f.state, f.lock, attempt, result, None)
            else:
                f.state = _interrupted(f.store, f.state, f.lock, attempt, result,
                                       {"confirmed": True}, "Confirmed fixture interruption")
            f.state["unsettled_attempts"].pop(attempt["attempt_id"], None)
            account(f.state, attempt, 0, result=result, final=True)
            f.store.save(f.state)
            self.assertNotIn(attempt["attempt_id"], f.state["accepted_receipts"])
        # Reproduce the old executor's missing node-draft handoff, in this fixture only.
        f.state["nodes"]["branch"]["draft_checkpoint"] = f.state["nodes"]["branch"]["approval"]["checkpoint"]
        f.state["status"] = "invalid_receipt"
        f.store.save(f.state)
        return deepcopy(f.state)

    def target(self):
        f = self.f
        tree = f.directory / "target"
        shutil.copytree(rules_root(f.lock), tree)
        for path in tree.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)
        tree.chmod(0o755)
        with (tree / "schema_extract/tools/executor_repairs.py").open("a") as stream:
            stream.write("\n# Distinct frozen repair target for the isolated identity test.\n")
        return tree

    def prepared(self):
        return executor_repairs.prepare(self.f.store, self.f.state["run_id"], tree=self.target(),
                                        issue_id="fixture-executor-repair", reason="Exercise a real frozen repair boundary")

    def review(self, plan):
        state = self.f.state
        aid = state["attempts"][-1] if state["attempts"] else None
        stage = read_json(self.f.store.run_dir(state["run_id"]) / "attempts" / aid / "attempt.json")["stage"] if aid else state["stage"]
        return {"plan_id": plan["id"], "decision": "continue_with_audit_marker", "authority": "user_instruction",
                "scope": "extract_tools_only", "summary": "Continue this fixture under the user instruction with an audit marker",
                "audit_points": [{"stage": stage, "attempt_id": aid, "detail": "Fixture tool incident; null means no invocation started"}]}

    def command(self, target, args):
        command, cwd, env = frozen_invocation(target, args)
        return subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, timeout=60)

    def apply_prepared(self, prepared):
        target = read_json(Path(prepared["directory"]) / "target.lock.json")
        path = self.f.directory / "authorization.json"
        write_json(path, self.review(prepared["plan"]))
        result = self.command(target, ["repair-runtime", "apply", self.f.state["run_id"], "--plan", prepared["plan"]["id"],
                                      "--authorization", str(path), "--state-root", str(self.f.store.root)])
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        return self.f.store.load(self.f.state["run_id"])

    def test_initial_frontier_diagnostics_and_unstarted_audit_point(self):
        f = self.f
        before = deepcopy(f.state)
        prepared = self.prepared()
        self.assertEqual(prepared["plan"]["preflight"]["completion_exit_code"], 1)
        self.assertTrue(prepared["plan"]["preflight"]["diagnostics"])
        updated = self.apply_prepared(prepared)
        for key in ("checkpoint", "stage", "role", "stages", "nodes", "attempts", "accepted_receipts"):
            self.assertEqual(updated[key], before[key])
        records = executor_repairs.verify_bindings(f.store, updated, f.lock)
        self.assertIsNone(records[0]["authorization"]["audit_points"][0]["attempt_id"])
        bad = self.review(prepared["plan"])
        bad["audit_points"][0]["stage"] = "P6"
        with self.assertRaisesRegex(ExtractError, "current stage"):
            executor_repairs._authorization(f.store, prepared["plan"], before, bad)

    def test_p5_retains_all_node_drafts_with_healthy_sibling_approval(self):
        f = self.f
        f.through_root()
        f.step(node="branch", role="worker")
        f.step(node="branch", role="reviewer")
        attempt = f.prepare(node="sibling", role="worker")
        result = f.result(attempt)
        result["response"]["decision"] = "continue"
        f.accept(attempt, result)
        before = deepcopy(f.state)
        self.assertEqual(before["node"], "root")
        prepared = self.prepared()
        updated = self.apply_prepared(prepared)
        self.assertEqual(updated["nodes"], before["nodes"])
        self.assertEqual(updated["stages"], before["stages"])
        candidate = f.store.verify_checkpoint(updated, updated["nodes"]["sibling"]["draft_checkpoint"])[0]
        path = candidate / "sibling.yaml"
        path.chmod(0o644)
        path.write_text(path.read_text() + "# changed after save\n")
        with self.assertRaises(ExtractError):
            executor_repairs.probe(f.store, updated)

    def test_p6_continue_and_safe_repair_or_interruption_preserve_drafts(self):
        for save in ("accepted", "receipt_repair", "interrupted"):
            with self.subTest(save=save):
                if save != "accepted":
                    self.setUp()
                before = self.frontier(decision="continue", save=save)
                prepared = self.prepared()
                self.assertEqual(prepared["plan"]["preflight"]["drafts"]["branch"]["after"], before["checkpoint"])
                updated = self.apply_prepared(prepared)
                for key in ("checkpoint", "stage", "role", "stages", "accepted_receipts", "attempt_usage", "reason", "status"):
                    self.assertEqual(updated[key], before[key], key)
                self.assertEqual(updated["nodes"]["branch"]["draft_checkpoint"], before["checkpoint"])
                self.assertEqual(updated["nodes"]["branch"]["approval"], before["nodes"]["branch"]["approval"])
                if save != "accepted":
                    name = "repair.json" if save == "receipt_repair" else "recovery.json"
                    proof = self.f.store.run_dir(updated["run_id"])/"attempts"/updated["attempts"][-1]/name
                    value = read_json(proof)
                    write_json(proof, {**value, "altered": True})
                    with self.assertRaisesRegex(ExtractError, "save evidence"):
                        executor_repairs.probe(self.f.store, updated)

    def test_audit_stage_and_commit_control_race_reject_without_changing_state(self):
        f = self.f
        before = self.frontier()
        prepared = self.prepared()
        bad = self.review(prepared["plan"])
        bad["audit_points"][0]["stage"] = "P1"
        with self.assertRaisesRegex(ExtractError, "registered invocation"):
            executor_repairs._authorization(f.store, prepared["plan"], before, bad)
        authorization = self.review(prepared["plan"])
        seal = executor_repairs._seal_objects
        def pause_after_seal(*args, **kwargs):
            seal(*args, **kwargs)
            f.store.request_control(before["run_id"], "pause")
        with patch.object(executor_repairs, "verify_executor"), patch.object(executor_repairs, "_seal_objects", side_effect=pause_after_seal):
            with self.assertRaisesRegex(ExtractError, "control authority changed"):
                executor_repairs.apply(f.store, before["run_id"], plan_id=prepared["plan"]["id"], review=authorization)
        self.assertEqual(f.store.load(before["run_id"]), before)

    def test_two_bindings_keep_original_attempts_and_scope_new_calls(self):
        f = self.f
        f.step()
        original = deepcopy(f.state)
        original_attempt = read_json(f.store.run_dir(original["run_id"]) / "attempts" / original["attempts"][0] / "attempt.json")
        first = self.prepared()
        self.apply_prepared(first)
        target = read_json(Path(first["directory"]) / "target.lock.json")
        script = f.directory / "one_call.py"
        script.write_text(r'''
import json,sys
from pathlib import Path
import schema_extract.tools
schema_extract.tools.__path__.append(str(Path(sys.argv[1])/"schema_extract/tools"))
from schema_extract.tools.tests import test_stage_flow
from schema_extract.tools.store import Store
from schema_extract.tools.common import read_json
f=test_stage_flow.StageFlowTests(methodName="runTest")
f.store=Store(Path(sys.argv[2]));f.state=f.store.load(sys.argv[3])
f.lock=read_json(f.store.run_dir(sys.argv[3])/"inputs.lock.json")
selected=[{"id":n,"instance":"kernel.py::copy(instance="+n+")"} for n in ("root","branch","sibling")]
f.backend=test_stage_flow.FixtureBackend(f.lock,selected,[{"id":n,"wait_for":[]} for n in ("branch","sibling")])
f.step()
print(json.dumps({"attempt_id":f.state["attempts"][-1]}))
''')
        def one_call(lock):
            env = {**__import__("os").environ, "PYTHONPATH": str(rules_root(lock)), "PYTHONDONTWRITEBYTECODE": "1"}
            result = subprocess.run([sys.executable, "-B", str(script), str(REPO), str(f.store.root), original["run_id"]],
                                    cwd=rules_root(lock), env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        one_call(target)
        f.state = f.store.load(original["run_id"])
        first_binding = f.state["execution_binding"]
        tree = f.directory / "second-target"
        shutil.copytree(rules_root(target), tree)
        for path in (tree, *tree.rglob("*")):
            path.chmod(0o755 if path.is_dir() else 0o644)
        path = tree / "schema_extract/tools/executor_repairs.py"
        path.write_text(path.read_text() + "\n# Second isolated tool repair.\n")
        result = self.command(target, ["repair-runtime", "prepare", original["run_id"], "--executor-tree", str(tree),
                                      "--issue-id", "second-fixture-repair", "--reason", "Second independent tool correction",
                                      "--state-root", str(f.store.root)])
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        second = json.loads(result.stdout)
        self.apply_prepared(second)
        target2 = read_json(Path(second["directory"]) / "target.lock.json")
        one_call(target2)
        state = f.store.load(original["run_id"])
        self.assertEqual(len(state["execution_bindings"]), 2)
        attempts = [read_json(f.store.run_dir(state["run_id"])/"attempts"/aid/"attempt.json") for aid in state["attempts"]]
        self.assertEqual(attempts[0], original_attempt)
        self.assertEqual([a.get("execution_binding") for a in attempts], [None, first_binding, state["execution_binding"]])
        for attempt in attempts:
            verify_execution(f.store, state, attempt=attempt)
        self.assertEqual(state["stages"]["P1"]["calls"], 2)
        self.assertEqual(state["stages"]["P2"]["calls"], 1)
        self.assertTrue(all(a["session_id"] is None for a in attempts[1:]))

    def test_two_p6_edits_restore_both_nodes_before_node_review(self):
        f = self.f
        self.frontier()
        first = f.state["checkpoint"]
        self.p6_node = "sibling"
        old_proposal = f.backend.proposal
        def revise(attempt):
            value = old_proposal(attempt)
            if attempt["stage"] == "P6" and attempt["role"] == "reviewer":
                value.update(decision="revise", affected_nodes=["sibling"],
                    revision_scope=[{"path": "sibling.yaml", "fields": ["DESIGN.Tile_pipeline.Thesis"]}],
                    issues=[{"kind": "schema_conformance", "scope": "sibling.yaml", "detail": "Revise the second node",
                             "evidence": [0], "recovery_condition": "Apply and independently review this scoped change"}])
            return value
        f.backend.proposal = revise
        f.state["status"] = "running"
        f.store.save(f.state)
        f.step()
        f.step()
        for node_id in ("branch", "sibling"):
            f.state["nodes"][node_id]["draft_checkpoint"] = f.state["nodes"][node_id]["approval"]["checkpoint"]
        f.state["status"] = "invalid_receipt"
        f.store.save(f.state)
        current = f.state["checkpoint"]
        self.assertNotEqual(first, current)
        prepared = self.prepared()
        self.assertEqual(set(prepared["plan"]["preflight"]["drafts"]), {"branch", "sibling"})
        updated = self.apply_prepared(prepared)
        for node_id in ("branch", "sibling"):
            self.assertEqual(updated["nodes"][node_id]["draft_checkpoint"], current)
            f.store.start_node(updated, node_id)
            latest = f.store.verify_checkpoint(updated, updated["nodes"][node_id]["draft_checkpoint"])[1]
            self.assertEqual(latest["files"][node_id+".yaml"], f.store.verify_checkpoint(updated, current)[1]["files"][node_id+".yaml"])

    def test_p6_adjudication_edit_preserves_draft_and_requires_signoff(self):
        f = self.f
        self.frontier()
        old = f.backend.proposal
        def revise(attempt):
            value = old(attempt)
            value.update(decision="revise", revision_scope=[{"path":"branch.yaml","fields":["DESIGN.Tile_pipeline.Thesis"]}],
                issues=[{"kind":"schema_conformance", "scope":"branch.yaml", "detail":"One bounded final correction",
                         "evidence":[0], "recovery_condition":"Patch then independently sign off"}])
            return value
        f.backend.proposal = revise
        f.state["status"] = "running"
        f.state["stages"]["P6"]["revisions"] = f.state["config"]["revision_limit"] - 1
        f.store.save(f.state)
        f.step()
        f.backend.proposal = old
        attempt = f.prepare()
        self.assertEqual(attempt["mode"], "adjudicate")
        result = f.backend.run(attempt=attempt, prompt="", schema={"type":"object"})
        result["response"].pop("reopen", None)
        text = (Path(attempt["workspace"])/"branch.yaml").read_text()
        line = next(line for line in text.splitlines() if line.lstrip().startswith(("Version:", '"Version":')))
        candidate = load_yaml(Path(attempt["workspace"]), "branch.yaml", parser_root=f.backend.rules)
        version = candidate["META_DATA"]["Version"]
        thesis = candidate["DESIGN"]["Tile_pipeline"]["Thesis"]
        self.assertEqual(text.count(thesis), 1)
        result["response"].update(decision="approve", adjudication={"action":"patch", "checkpoint":attempt["base_checkpoint"],
            "patches":[{"path":"branch.yaml", "old":thesis, "new":thesis + " with adjudicated detail"},
                       {"path":"branch.yaml", "old":line, "new":line.replace(str(version), str(version+1))}]})
        f.accept(attempt, result)
        self.assertEqual(f.state["mode"], "signoff")
        self.assertIsNone(f.state["stages"]["P6"]["approval"])
        self.assertEqual(f.state["nodes"]["branch"]["draft_checkpoint"], f.state["checkpoint"])
        prepared = self.prepared()
        updated = self.apply_prepared(prepared)
        self.assertEqual(updated["mode"], "signoff")
        self.assertEqual(updated["stages"]["P6"]["adjudications"], 1)
        self.assertIsNone(updated["stages"]["P6"]["approval"])

    def test_saved_scope_violation_is_not_an_incomplete_diagnostic(self):
        f = self.f
        attempt = f.step()
        directory, info = f.store.verify_checkpoint(f.state)
        unsafe = f.directory / "unsafe-save"
        shutil.copytree(directory, unsafe)
        for path in (unsafe, *unsafe.iterdir()):
            path.chmod(0o755 if path.is_dir() else 0o644)
        value = load_yaml(unsafe, "root.yaml", parser_root=f.backend.rules)
        value["DESIGN"]["Tile_pipeline"]["Thesis"] = "Outside the P1 definition."
        value["META_DATA"]["Version"] += 1
        write_yaml(unsafe / "root.yaml", value, parser_root=f.backend.rules)
        checkpoint = f.store.checkpoint(f.state, unsafe, attempt_id=attempt["attempt_id"], stage="P1", node="root",
                                        parent=attempt["base_checkpoint"], expected_files=info["files"])
        proof_path = Path(attempt["directory"])/"validated.json"
        proof = read_json(proof_path)
        proof["artifact_digests"] = f.store.verify_checkpoint(f.state, checkpoint)[1]["files"]
        write_json(proof_path, proof)
        f.state["checkpoint"] = checkpoint
        f.state["nodes"]["root"]["draft_checkpoint"] = checkpoint
        with self.assertRaisesRegex(ExtractError, "protected fields"):
            executor_repairs.probe(f.store, f.state)

    def test_apply_keeps_blocked_decision_and_exhausted_call_budget(self):
        from schema_extract.tools.stages import next_calls
        f = self.f
        f.state.update(status="blocked", reason={"code":"missing_input", "summary":"A real unmet prerequisite"})
        f.state["stages"]["P1"].update(status="blocked", calls=f.state["config"]["max_calls"])
        f.store.save(f.state)
        before = deepcopy(f.state)
        updated = self.apply_prepared(self.prepared())
        for key in ("status", "reason", "stages", "checkpoint", "accepted_receipts"):
            self.assertEqual(updated[key], before[key])
        self.assertEqual(next_calls(updated), [])
        self.assertFalse(updated["published"])

    def test_preserved_frontier_real_target_execution_and_bounded_maintenance(self):
        f = self.f
        before = self.frontier()
        prepared = self.prepared()
        plan = prepared["plan"]
        self.assertEqual(plan["preflight"]["drafts"]["branch"]["after"], before["checkpoint"])
        target = read_json(Path(prepared["directory"]) / "target.lock.json")
        review_path = f.directory / "review.json"
        write_json(review_path, self.review(plan))
        fault = f.directory / "commit_fault.py"
        fault.write_text('''
import sys
from pathlib import Path
from unittest.mock import patch
from schema_extract.tools.store import Store
from schema_extract.tools.common import read_json
from schema_extract.tools.executor_repairs import apply
store=Store(Path(sys.argv[1]));before=store.load(sys.argv[2]);save=store.save
def fail(state):
 if sys.argv[5]=="after":save(state)
 raise OSError("simulated commit interruption")
try:
 with patch.object(store,"save",side_effect=fail):
  apply(store,sys.argv[2],plan_id=sys.argv[3],review=read_json(Path(sys.argv[4])))
except OSError:pass
else:raise AssertionError("fault was not exercised")
assert (store.load(sys.argv[2])==before)==(sys.argv[5]=="before")
''')
        env = {**__import__("os").environ, "PYTHONPATH": str(rules_root(target)), "PYTHONDONTWRITEBYTECODE": "1"}
        for phase in ("before", "after"):
            failed = subprocess.run([sys.executable, "-B", str(fault), str(f.store.root), before["run_id"],
                                     plan["id"], str(review_path), phase], cwd=rules_root(target), env=env,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(failed.returncode, 0, failed.stderr + failed.stdout)
        args = ["repair-runtime", "apply", before["run_id"], "--plan", plan["id"], "--authorization", str(review_path),
                "--state-root", str(f.store.root)]
        result = self.command(target, args)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        state = f.store.load(before["run_id"])
        self.assertEqual(state["revision_scope"], before["revision_scope"])
        for key in ("checkpoint", "merged_checkpoint", "attempts", "accepted_receipts", "stages", "sessions", "p6_starts"):
            self.assertEqual(state[key], before[key], key)
        self.assertEqual(state["nodes"]["branch"]["draft_checkpoint"], before["checkpoint"])
        self.assertEqual(state["nodes"]["branch"]["approval"], before["nodes"]["branch"]["approval"])
        verify_execution(f.store, state)
        with self.assertRaisesRegex(ExtractError, "executor differs"):
            verify_execution(f.store, state, executor=True)
        old = read_json(f.store.run_dir(state["run_id"]) / "attempts" / before["attempts"][0] / "attempt.json")
        verify_execution(f.store, state, attempt=old)
        self.assertEqual(self.command(target, args).returncode, 0)
        self.assertEqual(f.store.load(state["run_id"]), state)

        script = r'''
import json,sys
from pathlib import Path
import schema_extract.tools
schema_extract.tools.__path__.append(str(Path(sys.argv[1]) / "schema_extract/tools"))
from schema_extract.tools.tests.test_stage_flow import FixtureBackend
from schema_extract.tools.store import Store
from schema_extract.tools.common import read_json,sha256
from schema_extract.tools.artifacts import load_yaml,write_yaml
from schema_extract.tools.maintenance import submit_request
from schema_extract.tools.runner import run
store=Store(Path(sys.argv[2]));state=store.load(sys.argv[3]);before=state
lock=read_json(store.run_dir(state["run_id"])/"inputs.lock.json")
candidate_branch_sha=sha256((store.verify_checkpoint(state)[0]/"branch.yaml").read_bytes())
assert candidate_branch_sha!=state["nodes"]["branch"]["approved_sha256"]
class Backend(FixtureBackend):
 asked_again=False
 def write(self,a):
  if a.get("purpose")=="maintenance":
   p=Path(a["workspace"]);v=load_yaml(p,"root.yaml",parser_root=self.rules)
   v["DESIGN"]["Tile_pipeline"]["Thesis"]+=" Narrow root clarification."
   v["META_DATA"]["Version"]+=1;write_yaml(p/"root.yaml",v,parser_root=self.rules)
  else:
   super().write(a)
   if a["stage"]=="P6":
    p=Path(a["workspace"]);v=load_yaml(p,"branch.yaml",parser_root=self.rules)
    v["DESIGN"]["Tile_pipeline"]["Thesis"]+=" A later independent final revision."
    v["META_DATA"]["Version"]+=1;write_yaml(p/"branch.yaml",v,parser_root=self.rules)
 def proposal(self,a):
  v=super().proposal(a)
  if a.get("purpose")=="maintenance":
   v.pop("reopen",None);v["revision_scope"]=[{"path":"root.yaml","fields":["DESIGN.Tile_pipeline.Thesis"]}] if a["role"]=="worker" else []
  if a["stage"]=="P6" and a["role"]=="reviewer":
   v["revision_scope"]=[];v["affected_nodes"]=["branch"]
   if not self.asked_again:
    self.asked_again=True;v["decision"]="revise"
    v["revision_scope"]=[{"path":"branch.yaml","fields":["DESIGN.Tile_pipeline.Thesis"]}]
    v["issues"]=[{"kind":"schema_conformance","scope":"branch.yaml DESIGN.Tile_pipeline.Thesis","detail":"Exercise a second bounded final edit after executor repair","evidence":[0],"recovery_condition":"Apply the explicit field change and independently review again"}]
  elif a["stage"]=="P6":v["affected_nodes"]=["branch"]
  return v
tasks=[{"stage":"P3","node":"root","action":"patch","fields":[{"path":"root.yaml","fields":["DESIGN.Tile_pipeline.Thesis"]}],"reason":"Narrow only the root design prose"},
       {"stage":"P5","node":"branch","action":"review","fields":[],"reason":"Review the latest accepted P6 branch candidate"}]
submit_request(store,state["run_id"],issue_id="fixture-content-recheck",tasks=tasks)
backend=Backend(lock,state["selected_designs"],state["writing_plan"])
state=run(store,state["run_id"],transport=backend)
assert state["published"],state.get("reason")
assert state["stages"]["P1"]["approval"]==before["stages"]["P1"]["approval"]
assert state["stages"]["P2"]["approval"]==before["stages"]["P2"]["approval"]
assert state["stages"]["P4"]["approval"]==before["stages"]["P4"]["approval"]
assert state["nodes"]["sibling"]["approval"]==before["nodes"]["sibling"]["approval"]
calls=[a for a in backend.calls]
assert [(a["stage"],a["role"]) for a in calls]==[("P3","worker"),("P3","reviewer"),("P5","reviewer"),("P6","reviewer"),("P6","worker"),("P6","reviewer")],calls
for a in calls:
 assert a["execution_binding"]==state["execution_binding"]
 assert ":executor:" in a["session_scope"]["operation"]
for a in calls[:5]:assert a["session_id"] is None
reviewed_branch=Path(next(a for a in calls if a["stage"]=="P5")["workspace"])/"branch.yaml"
assert sha256(reviewed_branch.read_bytes())==candidate_branch_sha
assert sha256((Path(state["output"])/"branch.yaml").read_bytes())!=candidate_branch_sha
assert len(state["execution_bindings"])==1
proof=read_json(Path(state["output"])/".extraction/receipt.json")
assert proof["execution_binding"]==state["execution_binding"]
print(json.dumps({"published":True,"calls":len(calls)}))
'''
        code = f.directory / "target_check.py"
        code.write_text(script)
        env = {**__import__("os").environ, "PYTHONPATH": str(rules_root(target)), "PYTHONDONTWRITEBYTECODE": "1"}
        checked = subprocess.run([sys.executable, "-B", str(code), str(REPO), str(f.store.root), state["run_id"]],
                                 cwd=rules_root(target), env=env, capture_output=True, text=True, timeout=180)
        self.assertEqual(checked.returncode, 0, checked.stderr + checked.stdout)

    def test_semantic_changes_controls_and_changed_frontier_are_rejected(self):
        f = self.f
        before = deepcopy(f.state)
        tree = self.target()
        semantic = tree / "schema_extract/prompt/phases/06_final.md"
        semantic.write_text(semantic.read_text() + "\nChanged rule\n")
        with self.assertRaisesRegex(ExtractError, "semantic rules"):
            freeze_executor_patch(f.store.root, f.lock, tree)
        semantic.write_bytes((rules_root(f.lock) / "schema_extract/prompt/phases/06_final.md").read_bytes())
        f.store.request_control(before["run_id"], "pause")
        with self.assertRaisesRegex(ExtractError, "barrier"):
            executor_repairs.prepare(f.store, before["run_id"], tree=tree, issue_id="held", reason="Cannot pass pause")
        self.assertEqual(f.store.load(before["run_id"]), before)

    def test_user_instruction_continues_p6_and_delivers_deferred_audit_marker(self):
        f = self.f
        before = self.frontier()
        prepared = self.prepared()
        plan = prepared["plan"]
        target = read_json(Path(prepared["directory"]) / "target.lock.json")
        authorization = {"plan_id": plan["id"], "decision": "continue_with_audit_marker",
            "authority": "user_instruction", "scope": "extract_tools_only",
            "summary": "Complete under existing premises after the tool repair; the user decides later audits.",
            "audit_points": [{"stage": "P6", "attempt_id": before["attempts"][-1],
                              "detail": "Fixture tool gate incident and possible later independent content audit."}]}
        auth = f.directory / "authorization.json"
        write_json(auth, authorization)
        applied = self.command(target, ["repair-runtime", "apply", before["run_id"], "--plan", plan["id"],
                                        "--authorization", str(auth), "--state-root", str(f.store.root)])
        self.assertEqual(applied.returncode, 0, applied.stderr + applied.stdout)
        script = f.directory / "continue_only.py"
        script.write_text(r'''
import sys
from pathlib import Path
import schema_extract.tools
schema_extract.tools.__path__.append(str(Path(sys.argv[1])/"schema_extract/tools"))
from schema_extract.tools.tests.test_stage_flow import FixtureBackend
from schema_extract.tools.store import Store
from schema_extract.tools.common import read_json
from schema_extract.tools.runner import run
store=Store(Path(sys.argv[2]));before=store.load(sys.argv[3])
lock=read_json(store.run_dir(before["run_id"])/"inputs.lock.json")
class Backend(FixtureBackend):
 def proposal(self,a):
  value=super().proposal(a)
  if a["stage"]=="P6":value.update(revision_scope=[],affected_nodes=["branch"])
  return value
backend=Backend(lock,before["selected_designs"],before["writing_plan"])
state=run(store,before["run_id"],resume=True,transport=backend)
assert state["published"],state.get("reason")
assert [(a["stage"],a["role"]) for a in backend.calls]==[("P6","reviewer")]
for phase in ("P1","P2","P3","P4"):assert state["stages"][phase]==before["stages"][phase]
assert state["checkpoint"]==before["checkpoint"]
proof=read_json(Path(state["output"])/".extraction/receipt.json")
notice=proof["continuation_audit"][0]
assert notice["status"]=="pending_user_decision"
assert notice["retained_progress"]["checkpoint"]==before["checkpoint"]
assert notice["audit_points"][0]["attempt_id"]==before["attempts"][-1]
assert proof["executor_history"][0]["authorization"]=="continue_with_audit_marker"
''')
        env = {**__import__("os").environ, "PYTHONPATH": str(rules_root(target)), "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run([sys.executable, "-B", str(script), str(REPO), str(f.store.root), before["run_id"]],
                                cwd=rules_root(target), env=env, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
