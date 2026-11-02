"""Checkpoint-bound review reuse, focused context, and revision workflows."""

from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
from textwrap import indent
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from ..codex_cli import write_json
from ..framework import candidate_report
from ..records import StateError, artifact_manifest, digest, parser_for, read_json, review_current, schema_parts
from ..prompts import prompt_for
from ..review_context import build_review_context, field_excerpt
from ..execution import Runner, bound_review, eligible
from . import support as fixtures


class ReviewTests(unittest.TestCase):
    def setUp(self):
        # Reuse the existing selected-source fixture, not its test methods or
        # any earlier extraction artifacts, schemas, or model conversations.
        self.f = fixtures.ExtractionCase()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.dispatches = [{"id": "positive", "parent": "root", "reason": "Positive scalar path."}]

    def receipt(self, state):
        value = self.f.receipt(state)
        value["dispatches"] = deepcopy(state["dispatches"] or self.dispatches)
        return value

    def add_node(self, work, name="positive", parent="root"):
        import yaml
        node = schema_parts(work, self.f.baseline["snapshot"])["root"]
        node.pop("contract")
        node["meta_data"]["version"] = 1
        node["meta_data"]["dispatch"] = {"id": name, "parent": parent, "when": "Select exactly when X > 0."}
        (work / (name + ".yaml")).write_text(yaml.safe_dump(node, sort_keys=False))

    def edit_node(self, work, node, mutate):
        import yaml
        value = schema_parts(work, self.f.baseline["snapshot"])[node]
        mutate(value)
        value["meta_data"]["version"] += 1
        (work / (node + ".yaml")).write_text(yaml.safe_dump(value, sort_keys=False))

    def reviewed(self, *, include_node=True, include_final=False):
        work = self.f.base / "review-artifacts"
        self.f.draft(work)
        root = work / "root.yaml"
        root.write_text(root.read_text().replace("  version: 1", "  version: 1\n  dispatch: {when: Select exactly when X <= 0.}"))
        if include_node:
            self.add_node(work)
        state = self.f.store.commit_checkpoint(self.f.state, work, self.receipt(self.f.state), {"exit_code": 0})
        for scope in ["contract", "root", *(["positive"] if include_node else []), *(["final"] if include_final else [])]:
            self.add_review(state, work, scope)
        state.update(phase="worker", stage="index", node="root", review_scope="final")
        self.f.store.save(state)
        return state, work

    def add_review(self, state, work, scope):
        state.update(review_scope=scope, stage="contract" if scope == "contract" else "node",
                     node="root" if scope in {"contract", "final"} else scope)
        receipt = self.f.review_receipt(state, work)
        receipt["dispatches"] = deepcopy(state["dispatches"])
        receipt["summary"] = "Independent " + scope + " semantic review is complete."
        state["reviews"][scope] = bound_review(state, receipt, self.f.store.checkpoint(state))

    def test_context_is_a_small_current_projection_and_missing_review_refuses_reuse(self):
        state, work = self.reviewed(include_final=True)
        files = artifact_manifest(work)
        context = build_review_context(self.f.store, state, self.f.baseline, work, "worker")
        self.assertEqual(context["checkpoint"], state["checkpoint"])
        self.assertEqual(context["fingerprint"], state["fingerprint"])
        self.assertEqual([node["node"] for node in context["nodes"]], ["root", "positive"])
        text = json.dumps(context)
        self.assertNotIn("hardware_specification", text)
        self.assertNotIn("math_reshape", text)
        self.assertNotIn("source_evidence", text)
        self.assertNotIn("yaml", context["contract_location"])
        self.assertEqual(len(context["source_locations"]), len(self.f.entry["paths"]))
        for item in [context["contract_location"], *(node["metadata"] for node in context["nodes"])]:
            lines = (work / item["path"]).read_text().splitlines()
            self.assertIn(item["quote"], "\n".join(lines[item["line_start"] - 1:item["line_end"]]))
        self.assertEqual(artifact_manifest(work), files)
        state.update(phase="reviewer", review_scope="final")
        self.assertIsNotNone(build_review_context(self.f.store, state, self.f.baseline, work, "reviewer"))
        state["reviews"].pop("positive")
        self.assertIsNone(build_review_context(self.f.store, state, self.f.baseline, work, "reviewer"))
        self.assertFalse(eligible(self.f.store, state))

    def test_projection_rejects_unaccepted_artifact_changes(self):
        state, work = self.reviewed()
        (work / "dispatch.md").write_text("unaccepted index\n")
        with self.assertRaisesRegex(StateError, "accepted checkpoint"):
            build_review_context(self.f.store, state, self.f.baseline, work, "worker")

    def test_single_node_review_always_keeps_full_scope(self):
        state = self.f.checkpoint()
        work = self.f.store.checkpoint(state) / "artifacts"
        state.update(phase="reviewer", stage="node", review_scope="final")
        self.add_review(state, work, "contract")
        self.add_review(state, work, "final")
        self.assertIsNone(build_review_context(self.f.store, state, self.f.baseline, work, "reviewer"))
        self.assertTrue(eligible(self.f.store, state))

    def test_alias_and_flow_metadata_excerpts_do_not_include_intervening_design(self):
        path = self.f.base / "excerpt.yaml"
        parser = parser_for(self.f.baseline["snapshot"])
        examples = [
            "shared: &metadata {name: scalar, version: 1}\nnoise: {design: SECRET_DESIGN}\nmeta_data: *metadata\ndesign: {body: OTHER_SECRET}\n",
            "shared: &metadata {name: scalar, version: 1}\nnoise: {design: SECRET_DESIGN}\nmeta_data: {<<: *metadata, describe: Local description}\ndesign: {body: OTHER_SECRET}\n",
            "{meta_data: {name: scalar, version: 1}, design: {body: SECRET_DESIGN}}\n",
            "meta_data:\n  name: scalar\n  version: 1\ndesign:\n  body: SECRET_DESIGN\n",
            "<<: [{meta_data: {name: scalar, version: 1}}, {meta_data: {name: wrong, version: 1}}]\ndesign: {body: SECRET_DESIGN}\n",
        ]
        for text in examples:
            with self.subTest(text=text):
                path.write_text(text)
                excerpt = field_excerpt(path, ("meta_data",), include_text=True, parser_root=self.f.baseline["snapshot"])
                restored = parser.loads(excerpt["yaml"], "projection").value["meta_data"]
                self.assertEqual(restored["name"], "scalar")
                self.assertEqual(digest(restored), digest(parser.load(path, "source").value["meta_data"]))
                self.assertNotIn("SECRET", excerpt["yaml"])
                self.assertIn(excerpt["quote"], "\n".join(text.splitlines()[excerpt["line_start"] - 1:excerpt["line_end"]]))
        path.write_text("contract:\n  algorithm_requirement:\n    tolerance: 1.000000000000000000000001\n  hardware_specification: SECRET\ndesign: SECRET\n")
        exact = field_excerpt(path, ("contract", "algorithm_requirement"), include_text=True)
        self.assertIn("1.000000000000000000000001", exact["yaml"])
        self.assertNotIn("SECRET", exact["yaml"])
        path.write_text("contract:\n  <<: {algorithm_requirement: {tolerance: 1.000000000000000000000001}}\n  hardware_specification: SECRET\ndesign: SECRET\n")
        inherited = field_excerpt(path, ("contract", "algorithm_requirement"), include_text=True)
        self.assertIn("1.000000000000000000000001", inherited["yaml"])
        self.assertNotIn("SECRET", inherited["yaml"])

    def test_metadata_projection_preserves_fixed_core_scalars_and_explicit_tags(self):
        path = self.f.base / "scalar-metadata.yaml"
        parser = parser_for(self.f.baseline["snapshot"])
        metadata = ("name: off\n"
                    "on: on\n"
                    "date: 2026-11-02\n"
                    "version: 01\n"
                    "leading_zero: 080\n"
                    "explicit_string: !!str 080\n"
                    "octal_string: !!str 0o12\n"
                    "octal_number: 0o12\n"
                    "exponent_string: !!str 1e3\n"
                    "exponent_number: 1e3\n"
                    "quoted_number: '081'\n"
                    "precise: 1.000000000000000000000001\n"
                    "negative_zero: -0.000000000000000000000000\n"
                    "tiny: 1e-1000\n"
                    "enabled: true\n"
                    "absent: null\n"
                    "nested: [off, 'off', 081, '081', 2026-11-02, false]\n")
        examples = ["meta_data:\n" + indent(metadata, "  "),
                    "defaults: &settings\n" + indent(metadata, "  ") + "meta_data: *settings\n",
                    "defaults: &settings\n" + indent(metadata, "  ")
                    + "meta_data: {<<: [*settings, {name: wrong, leading_zero: wrong}], describe: Local}\n"]
        for source in examples:
            with self.subTest(source=source):
                path.write_text(source + "design: SECRET_DESIGN\n")
                expected = parser.load(path, "source").value["meta_data"]
                self.assertEqual(expected["on"], "on")
                self.assertIs(type(expected["date"]), str)
                self.assertEqual(expected["leading_zero"], 80)
                self.assertEqual(expected["explicit_string"], "080")
                self.assertEqual(expected["octal_string"], "0o12")
                self.assertEqual(expected["exponent_string"], "1e3")
                self.assertEqual(expected["precise"], Decimal("1.000000000000000000000001"))
                context = field_excerpt(path, ("meta_data",), include_text=True, parser_root=self.f.baseline["snapshot"])
                actual = parser.loads(context["yaml"], "projection").value["meta_data"]
                # The digest compares types and exact Decimal strings too,
                # unlike equality where True == 1 and Decimal(1) == 1.
                self.assertEqual(digest(actual), digest(expected))
                self.assertNotIn("SECRET_DESIGN", context["yaml"])

    def test_metadata_projection_refuses_readback_semantic_drift(self):
        path = self.f.base / "metadata.yaml"
        path.write_text("meta_data: {name: !!str 080, version: 1}\n")
        parser = parser_for(self.f.baseline["snapshot"])
        original = parser.loads

        def drift(text, name, **kwargs):
            document = original(text, name, **kwargs)
            if name == "review_context":
                document.value["meta_data"]["name"] = 80
            return document

        with patch.object(parser, "loads", side_effect=drift):
            with self.assertRaisesRegex(StateError, "changed the fixed parser"):
                field_excerpt(path, ("meta_data",), include_text=True, parser_root=self.f.baseline["snapshot"])

    def test_reason_only_revision_preserves_node_reviews_but_requires_new_final(self):
        state, work = self.reviewed(include_final=True)
        before = deepcopy(state["reviews"])
        files = artifact_manifest(work)
        receipt = self.receipt(state)
        receipt["dispatches"][0]["reason"] = "More concise index wording."
        state = self.f.store.commit_checkpoint(state, work, receipt, {"exit_code": 0})
        self.assertEqual(set(state["reviews"]), {"contract", "root", "positive"})
        self.assertEqual(state["reviews"]["root"], before["root"])
        self.assertEqual(state["reviews"]["positive"], before["positive"])
        record = read_json(self.f.store.checkpoint(state) / "checkpoint.json")
        self.assertTrue(review_current(state, "root", record["semantics"]))
        self.assertTrue(review_current(state, "positive", record["semantics"]))
        self.assertEqual(record["dispatches"], receipt["dispatches"])
        self.assertEqual(state["dispatches"], receipt["dispatches"])
        self.assertEqual(artifact_manifest(work), files)
        self.assertFalse(eligible(self.f.store, state))
        self.assertIsNotNone(build_review_context(self.f.store, state, self.f.baseline, work, "worker"))

    def test_addition_preserves_review_but_later_condition_change_revokes_earlier_root(self):
        state, work = self.reviewed(include_node=False)
        original_root = deepcopy(state["reviews"]["root"])
        self.add_node(work)
        state = self.f.store.commit_checkpoint(state, work, self.receipt(state), {"exit_code": 0})
        self.assertEqual(state["reviews"]["root"], original_root)
        self.add_review(state, work, "positive")
        original_positive = deepcopy(state["reviews"]["positive"])
        self.edit_node(work, "positive", lambda node: node["meta_data"]["dispatch"].update(when="Select exactly when X >= 1."))
        state = self.f.store.commit_checkpoint(state, work, self.receipt(state), {"exit_code": 0})
        self.assertEqual(set(state["reviews"]), {"contract"})
        # Even if stale entries survived an interrupted/older scheduler, the
        # shared binding test cannot reuse the pre-addition root conclusion.
        state["reviews"].update(root=original_root, positive=original_positive)
        record = read_json(self.f.store.checkpoint(state) / "checkpoint.json")
        self.assertFalse(review_current(state, "root", record["semantics"]))
        self.assertFalse(review_current(state, "positive", record["semantics"]))
        self.assertTrue(Runner(self.f.store, self.f.plan).pending_review(state))
        self.assertEqual((state["phase"], state["review_scope"]), ("reviewer", "root"))

    def test_parent_declaration_change_revokes_existing_selection_context(self):
        state, work = self.reviewed()
        self.add_node(work, "alternate")
        receipt = self.receipt(state)
        receipt["dispatches"].append({"id": "alternate", "parent": "root", "reason": "Another declared path."})
        state = self.f.store.commit_checkpoint(state, work, receipt, {"exit_code": 0})
        self.add_review(state, work, "alternate")
        old = deepcopy(state["reviews"])
        receipt = self.receipt(state)
        receipt["dispatches"][0]["parent"] = "alternate"
        self.edit_node(work, "positive", lambda node: node["meta_data"]["dispatch"].update(parent="alternate"))
        state = self.f.store.commit_checkpoint(state, work, receipt, {"exit_code": 0})
        self.assertEqual(set(state["reviews"]), {"contract"})
        state["reviews"].update(old)
        semantics = read_json(self.f.store.checkpoint(state) / "checkpoint.json")["semantics"]
        self.assertTrue(review_current(state, "contract", semantics))
        for node in ("root", "positive", "alternate"):
            self.assertFalse(review_current(state, node, semantics))

    def test_node_change_rejects_stale_binding_without_reopening_unchanged_sibling(self):
        state, work = self.reviewed()
        old = deepcopy(state["reviews"])
        self.edit_node(work, "positive", lambda node: node["design"]["math_reshape"].update(thesis="An actual corrected node description."))
        state = self.f.store.commit_checkpoint(state, work, self.receipt(state), {"exit_code": 0})
        self.assertEqual(set(state["reviews"]), {"contract", "root"})
        state["reviews"]["positive"] = old["positive"]
        state.update(stage="index", phase="reviewer", review_scope="final", node="root")
        self.assertIsNone(build_review_context(self.f.store, state, self.f.baseline, work, "reviewer"))
        self.assertTrue(Runner(self.f.store, self.f.plan).pending_review(state))
        self.assertEqual((state["phase"], state["review_scope"]), ("reviewer", "positive"))
        self.assertEqual(state["reviews"]["root"], old["root"])

    def test_dispatch_receipt_and_checkpoint_pointer_are_committed_atomically(self):
        state, work = self.reviewed()
        before = self.f.store.load(state["key"])
        receipt = self.receipt(state)
        receipt["dispatches"][0]["reason"] = "An unaccepted replacement."
        with patch.object(self.f.store, "save", side_effect=OSError("injected task save failure")):
            with self.assertRaises(OSError):
                self.f.store.commit_checkpoint(state, work, receipt, {"exit_code": 0})
        restored = self.f.store.load(state["key"])
        self.assertEqual(restored["checkpoint"], before["checkpoint"])
        self.assertEqual(restored["dispatches"], before["dispatches"])
        self.f.store.verify_checkpoint(restored)
        restored["dispatches"][0]["reason"] = "A split-authority task mutation."
        with self.assertRaisesRegex(StateError, "dispatch declarations"):
            self.f.store.verify_checkpoint(restored)

    def flow_cli(self, *, impact="index", pause=False, change_node_in_index=False):
        owner = self

        class CLI:
            def __init__(self):
                self.calls = []
                self.final_count = 0
                self.changed_index_node = False

            def run(self, **kwargs):
                state = owner.f.store.load(owner.f.state["key"])
                work, records, role = kwargs["cwd"], kwargs["records"], kwargs["role"]
                records.mkdir(parents=True)
                self.calls.append({"role": role, "stage": state["stage"], "node": state["node"],
                                   "scope": state["review_scope"], "checkpoint": state["checkpoint"],
                                   "thread": kwargs["thread_id"], "work": str(work), "records": str(records),
                                   "reviews": deepcopy(state["reviews"])})
                response = owner.receipt(state)
                if role == "worker":
                    if state["stage"] == "contract":
                        owner.f.draft(work)
                        root = work / "root.yaml"
                        root.write_text(root.read_text().replace("  version: 1", "  version: 1\n  dispatch: {when: Select exactly when X <= 0.}"))
                    elif state["stage"] == "index":
                        (work / "dispatch.md").write_text("[root](root.yaml)\n- [positive](positive.yaml): "
                                                          + ("correct label" if self.final_count else "old label") + ".\n")
                        if change_node_in_index and self.final_count and not self.changed_index_node:
                            owner.edit_node(work, "positive", lambda node: node["design"]["math_reshape"].update(thesis="A changed node requires its own new semantic review."))
                            self.changed_index_node = True
                    elif state["node"] == "positive":
                        if (work / "positive.yaml").exists():
                            owner.edit_node(work, "positive", lambda node: node["design"]["math_reshape"].update(thesis="The corrected scalar-copy explanation."))
                        else:
                            owner.add_node(work)
                else:
                    owner.assertIsNone(kwargs["thread_id"], "substantive reviewers must be independent")
                    response = owner.f.review_receipt(state, work)
                    response["dispatches"] = deepcopy(owner.dispatches)
                    if state["review_scope"] == "final":
                        self.final_count += 1
                        owner.assertIn("FINAL INDEX REVIEW", kwargs["prompt"])
                        owner.assertNotIn("/schema_docs/3.design.math_reshape.md", kwargs["prompt"])
                        if self.final_count == 1:
                            response.update(status="revise", impact=impact, revision_node="positive" if impact == "node" else "")
                            response["framework_candidates"] = [fixtures.PROPOSAL]
                            response["issues"] = [{"id": "specific_fix", "kind": "extraction_error", "disposition": "fix",
                                                   "scope": "dispatch.md" if impact == "index" else "positive.yaml:design.math_reshape",
                                                   "detail": "Correct the concrete synthetic label or node description.",
                                                   "evidence": ["Synthetic independent source anchor."], "restart_condition": "", "blocks": ["final"]}]
                            if pause:
                                control = owner.f.store.root / "runs/test/control.json"
                                control.parent.mkdir(parents=True, exist_ok=True)
                                write_json(control, {"mode": "pause"})
                        else:
                            response["resolved_issue_ids"] = ["specific_fix"]
                    elif state["review_scope"] == "positive" and self.final_count:
                        response["resolved_issue_ids"] = ["specific_fix"]
                return {"outcome": "completed", "response": response,
                        "thread_id": kwargs["thread_id"] or str(uuid4()), "elapsed_seconds": 0.1,
                        "termination": {"confirmed": True}, "usage": {"input_tokens": 10, "output_tokens": 10}}

        return CLI()

    def test_index_only_fix_returns_directly_to_final_and_all_nodes_stay_independent(self):
        cli = self.flow_cli()
        state = Runner(self.f.store, self.f.plan, cli).work(self.f.state["key"])
        self.assertEqual(state["result"], "ready", state.get("last_error"))
        self.assertTrue(eligible(self.f.store, state))
        scopes = [call["scope"] for call in cli.calls if call["role"] == "reviewer"]
        self.assertEqual(scopes, ["contract", "root", "positive", "final", "final"])
        finals = [call for call in cli.calls if call["role"] == "reviewer" and call["scope"] == "final"]
        reports = candidate_report(self.f.store, self.f.plan)[0]["reports"]
        self.assertEqual(len(reports), 1)
        source, = reports[0]["sources"]
        self.assertEqual(source["checkpoint"], finals[0]["checkpoint"])
        self.assertNotEqual(source["checkpoint"], finals[1]["checkpoint"])
        self.assertEqual(finals[0]["reviews"]["root"], finals[1]["reviews"]["root"])
        self.assertEqual(finals[0]["reviews"]["positive"], finals[1]["reviews"]["positive"])
        for call in cli.calls:
            if call["stage"] == "index":
                context = read_json(Path(call["records"]) / "review_context.json")
                self.assertEqual(context["checkpoint"], call["checkpoint"])
        self.assertEqual(sum(call["role"] == "worker" and call["stage"] == "index" for call in cli.calls), 2)

    def test_final_concrete_node_fix_goes_to_author_then_independent_node_review(self):
        cli = self.flow_cli(impact="node")
        state = Runner(self.f.store, self.f.plan, cli).work(self.f.state["key"])
        self.assertEqual(state["result"], "ready", state.get("last_error"))
        self.assertTrue(eligible(self.f.store, state))
        self.assertEqual([call["scope"] for call in cli.calls if call["role"] == "reviewer"],
                         ["contract", "root", "positive", "final", "positive", "final"])
        self.assertEqual(sum(call["role"] == "worker" and call["node"] == "positive" for call in cli.calls), 2)
        self.assertFalse(any(issue["state"] == "open" and issue["disposition"] == "fix" for issue in state["issues"]))

    def test_index_author_node_edit_cannot_bypass_independent_semantic_review(self):
        cli = self.flow_cli(change_node_in_index=True)
        state = Runner(self.f.store, self.f.plan, cli).work(self.f.state["key"])
        self.assertEqual(state["result"], "ready", state.get("last_error"))
        self.assertTrue(eligible(self.f.store, state))
        self.assertEqual([call["scope"] for call in cli.calls if call["role"] == "reviewer"],
                         ["contract", "root", "positive", "final", "positive", "final"])
        self.assertEqual(sum(call["role"] == "worker" and call["stage"] == "index" for call in cli.calls), 2)

    def test_pause_recovery_rebuilds_context_from_checkpoint_and_uses_fresh_author(self):
        cli = self.flow_cli(pause=True)
        runner = Runner(self.f.store, self.f.plan, cli)
        state = runner.work(self.f.state["key"])
        self.assertEqual((state["status"], state["phase"], state["stage"]), ("stopped", "worker", "index"))
        accepted = state["checkpoint"]
        old_calls = len(cli.calls)
        # Audit projections are disposable: corrupting one cannot authorize a
        # review or affect the context reconstructed on explicit recovery.
        audit = Path(cli.calls[-1]["records"]) / "review_context.json"
        write_json(audit, {"kind": "stale", "checkpoint": "unrelated"})
        runner.control.unlink()
        fresh = Runner(self.f.store, self.f.plan, cli)
        fresh.recover(state)
        complete = fresh.work(state["key"])
        self.assertEqual(complete["result"], "ready", complete.get("last_error"))
        first = cli.calls[old_calls]
        self.assertEqual((first["role"], first["stage"], first["checkpoint"]), ("worker", "index", accepted))
        self.assertIsNone(first["thread"])
        self.assertNotEqual(first["work"], cli.calls[0]["work"])
        self.assertEqual(read_json(Path(first["records"]) / "review_context.json")["checkpoint"], accepted)
        self.assertEqual([call["scope"] for call in cli.calls[old_calls:] if call["role"] == "reviewer"], ["final"])
        self.assertNotIn("review_context", self.f.store.load(state["key"]))

    def test_failure_preparing_context_or_prompt_registers_no_call_and_recovers(self):
        state, _ = self.reviewed()
        state.update(phase="reviewer", stage="index", review_scope="final")
        self.f.store.save(state)
        accepted = state["checkpoint"]
        cli = Mock()
        for target in ("build_review_context", "prompt_for"):
            with self.subTest(target=target):
                runner = Runner(self.f.store, self.f.plan, cli)
                with patch("schema_extract.tools.execution." + target, side_effect=StateError("injected preparation failure")), patch("builtins.print"):
                    runner.run()
                stopped = self.f.store.load(state["key"])
                self.assertEqual((stopped["status"], stopped["reason"]), ("stopped", "runner_failure"))
                self.assertIn("injected preparation failure", stopped["last_error"])
                cli.run.assert_not_called()
                self.assertEqual(stopped["attempts"][-1]["calls"], [])
                self.assertEqual(stopped["usage"]["calls"], 0)
                self.assertEqual(stopped["checkpoint"], accepted)
                runner.recover(stopped)
                self.assertEqual(self.f.store.load(state["key"])["status"], "queued")
        synthetic = self.flow_cli()
        complete = Runner(self.f.store, self.f.plan, synthetic).work(state["key"])
        self.assertEqual(complete["result"], "ready", complete.get("last_error"))
        self.assertEqual(synthetic.calls[0]["checkpoint"], accepted)

    def test_control_request_during_prompt_preparation_registers_no_call(self):
        state, _ = self.reviewed()
        state.update(phase="reviewer", stage="index", review_scope="final")
        self.f.store.save(state)
        accepted = state["checkpoint"]
        for mode in ("pause", "cancel"):
            with self.subTest(mode=mode):
                cli = Mock()
                runner = Runner(self.f.store, self.f.plan, cli)
                runner.control.parent.mkdir(parents=True, exist_ok=True)

                def prepare(*args, **kwargs):
                    prompt = prompt_for(*args, **kwargs)
                    self.assertEqual(kwargs["review_context"]["checkpoint"], accepted)
                    write_json(runner.control, {"mode": mode})
                    return prompt

                with patch("schema_extract.tools.execution.prompt_for", side_effect=prepare):
                    stopped = runner.work(state["key"])
                cli.run.assert_not_called()
                self.assertEqual((stopped["status"], stopped["reason"]), ("stopped", mode))
                self.assertEqual(stopped["attempts"][-1]["calls"], [])
                self.assertEqual(stopped["usage"]["calls"], 0)
                self.assertEqual(stopped["checkpoint"], accepted)
                runner.control.unlink()
                runner.recover(stopped)
                self.assertEqual(self.f.store.load(state["key"])["status"], "queued")


if __name__ == "__main__":
    unittest.main()
