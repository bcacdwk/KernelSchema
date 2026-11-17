"""Current stage decisions, actual sources and atomic normalization boundaries."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from schema_extract.tools.artifacts import (
    CandidateRejected, align_source_identity, approval_projection, check_stage_snapshot, design_context,
    load_root, load_yaml, normalize_candidate, schema_for, source_context, validate_artifacts,
    validate_index, validate_plan, validate_revision_scope, write_yaml,
)
from schema_extract.tools.common import ExtractError, manifest
from schema_extract.tools.comparators import validate_comparisons
from schema_extract.tools.receipts import _anchor, normalize_response, proposal_schema


REPO = Path(__file__).resolve().parents[3]
STARTER = REPO / "schema_template/starter.yaml"
FIXTURE = REPO / "schema_template/tools/tests/fixtures/minimal"


def selection(*nodes):
    return [{"id": node, "instance": "kernel.py::copy(instance=" + node + ")"} for node in nodes]


class ArtifactTransactions(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.starter = load_root(STARTER.parent, STARTER.name, parser_root=REPO)
        self.base = load_root(FIXTURE, parser_root=REPO)
        execution = self.base["CONTRACT"]["Hardware_specification"]["Execution"]
        self.base["CONTRACT"]["Hardware_specification"] = deepcopy(self.starter["CONTRACT"]["Hardware_specification"])
        self.base["CONTRACT"]["Hardware_specification"]["Execution"] = execution
        self.base["META_DATA"].update(Source="testing.invalid/kernel@fixture-v1:kernel.py::copy",
                                      Dispatch={"when": "All scalar inputs"})

    def task(self, name, nodes=None, *, planned=()):
        directory = self.directory / name
        directory.mkdir()
        nodes = nodes or {"root": self.base}
        for node, value in nodes.items():
            write_yaml(directory / (node + ".yaml"), value, parser_root=REPO)
        self.index(directory, set(nodes) | set(planned), planned=planned)
        normalized = schema_for(REPO).normalize_directory(directory, skip_hardware=True)
        for node, text in normalized.candidates.items():
            (directory / (node + ".yaml")).write_text(text)
        return directory

    def index(self, directory, nodes, *, planned=()):
        rows = ["# 标量复制", "", "## 范围", "", "合成验证材料", "", "## 完整设计", "",
                "| 设计 | 具体机制与保留理由 | 关系 |", "| --- | --- | --- |"]
        for node in sorted(nodes, key=lambda node: (node != "root", node)):
            label = f"`{node}.yaml`（仅计划）" if node in planned else f"[{node}]({node}.yaml)"
            rows.append(f"| {label} | 合成完整方案，用于行为检查 | 明确登记 |")
        (directory / "dispatch.md").write_text("\n".join(rows) + "\n")

    def branch(self, node):
        value = deepcopy(self.base)
        value.pop("CONTRACT")
        value["META_DATA"]["Dispatch"] = {"id": node, "when": "All scalar inputs"}
        return value

    def copy(self, source, name):
        destination = self.directory / name
        shutil.copytree(source, destination)
        return destination

    def test_selection_and_waiting_are_distinct_bounded_decisions(self):
        selected = selection("root", "tile", "split")
        signature = validate_plan("P2", selected, parser_root=REPO)
        self.assertEqual(signature, validate_plan("P2", list(reversed(selected)), parser_root=REPO))
        plan = [{"id": "tile", "wait_for": []}, {"id": "split", "wait_for": ["root", "tile"]}]
        validate_plan("P4", plan, parser_root=REPO, selected_designs=selected)
        validate_plan("P4", [], parser_root=REPO, selected_designs=selection("root"))
        invalid = [selection(*["root", *["n" + str(i) for i in range(10)]]), selected + [selected[1]],
                   [{"id": "root", "file": "root.yaml", "parent": ""}],
                   [{"id": "root", "instance": "same"}, {"id": "tile", "instance": "same"}]]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ExtractError):
                validate_plan("P2", value, parser_root=REPO)
        for value in (plan[:1], [{"id": "tile", "wait_for": ["split"]}, {"id": "split", "wait_for": ["tile"]}],
                      [{"id": "tile", "wait_for": ["missing"]}, plan[1]],
                      [{"id": "tile", "wait_for": ["tile"]}, plan[1]],
                      [{"id": "tile", "wait_for": [], "instance": "changed"}, plan[1]]):
            with self.subTest(value=value), self.assertRaises(ExtractError):
                validate_plan("P4", value, parser_root=REPO, selected_designs=selected)
        changed = deepcopy(selected)
        changed[1]["instance"] = "kernel.py::changed"
        with self.assertRaisesRegex(ExtractError, "instances changed"):
            validate_plan("P2", changed, parser_root=REPO, selected_designs=selected)
        attempt = {"run_id": "r", "attempt_id": "n", "input_fingerprint": "i", "stage": "P2", "role": "worker",
                   "node": "root", "plan_revision": 0, "base_checkpoint": "c", "dependency_digests": {}, "writable_files": []}
        self.assertEqual(set(proposal_schema(attempt)["properties"]["plan"]["items"]["properties"]), {"id", "instance"})
        attempt["stage"] = "P4"
        self.assertEqual(set(proposal_schema(attempt)["properties"]["plan"]["items"]["properties"]), {"id", "wait_for"})

    def test_current_field_scopes_and_root_when(self):
        before = self.task("before")
        after = self.copy(before, "after")
        root = deepcopy(self.base)
        root["META_DATA"]["Version"] = 2
        root["META_DATA"]["Source"] = "testing.invalid/kernel@fixture-v2:kernel.py::copy"
        write_yaml(after / "root.yaml", root, parser_root=REPO)
        validate_artifacts("P2", "worker", before, after, STARTER, decision="complete")
        root["CONTRACT"]["Algorithm_requirement"]["Golden_reference"]["Tolerance"]["y"] = "Changed criterion"
        write_yaml(after / "root.yaml", root, parser_root=REPO)
        with self.assertRaisesRegex(ExtractError, "protected fields"):
            validate_artifacts("P2", "worker", before, after, STARTER, decision="complete")
        root = deepcopy(self.base)
        root["META_DATA"].pop("Dispatch")
        write_yaml(before / "root.yaml", root, parser_root=REPO)
        root["META_DATA"].update(Dispatch={"when": "Subset with a scalar input"}, Version=2)
        write_yaml(after / "root.yaml", root, parser_root=REPO)
        validate_artifacts("P3", "worker", before, after, STARTER, decision="complete")
        with self.assertRaisesRegex(ExtractError, "outside stage scope"):
            validate_artifacts("P3", "reviewer", before, after, STARTER, decision="approve")
        with self.assertRaises(ExtractError):
            validate_revision_scope([{"path": "root.yaml", "fields": ["DESIGN"]}], selection("root"))
        with self.assertRaises(ExtractError):
            validate_revision_scope([{"path": "branch.yaml", "fields": ["META_DATA.Dispatch.id"]}], selection("root", "branch"))

    def test_p1_normalizes_before_gates_and_preserves_original_evidence(self):
        before = self.task("before", {"root": self.starter})
        original = self.copy(before, "original")
        root = deepcopy(self.starter)
        root["META_DATA"] = deepcopy(self.base["META_DATA"])
        root["META_DATA"].pop("Dispatch")
        root["META_DATA"]["Describe"] += "."
        root["CONTRACT"] = deepcopy(self.base["CONTRACT"])
        write_yaml(original / "root.yaml", root, parser_root=REPO)
        path = original / "root.yaml"
        path.write_text(path.read_text().replace('"META_DATA":', '"meta_data":', 1))
        original_hashes = manifest(original)
        source = self.directory / "source"
        source.mkdir()
        (source / "kernel.py").write_text("def copy(x): return x\n")
        attempt = {"run_id": "r", "attempt_id": "nonce", "input_fingerprint": "i", "stage": "P1", "role": "worker",
                   "node": "root", "plan_revision": 0, "base_checkpoint": "base", "dependency_digests": manifest(before),
                   "writable_files": ["root.yaml"]}
        proposal = {"format": "schema_extract.proposal.v1", "attempt_id": "nonce", "decision": "complete", "summary": "Source contract",
                    "source_evidence": [], "artifact_evidence": [{"path": "root.yaml", "line_start": 1, "line_end": 3,
                        "quote_offset": 0, "note": "Original draft"}], "conclusions": [], "checks": [], "checks_note": "Runner checks",
                    "issues": [], "framework_candidates": [], "handoff": "", "comparisons": []}
        receipt = normalize_response(proposal, {"snapshot": str(self.directory), "source": {"id": "s", "files": manifest(source)}}, attempt, original)
        destination = self.directory / "landed"
        result = normalize_candidate(before, original, destination, STARTER, stage="P1", role="worker", decision="complete")
        self.assertEqual(result["check"]["exit_code"], 0)
        self.assertTrue(result["corrections"])
        self.assertEqual(manifest(original), original_hashes)
        self.assertNotEqual(result["artifact_digests"]["root.yaml"], original_hashes["root.yaml"])
        self.assertEqual(load_root(destination, parser_root=REPO)["META_DATA"]["Version"], 1)
        anchor = receipt["artifact_evidence"][0]
        self.assertEqual(anchor["sha256"], original_hashes["root.yaml"])
        _anchor(anchor, original, original_hashes)
        with self.assertRaisesRegex(ExtractError, "digest mismatch"):
            _anchor(anchor, destination, manifest(destination))
        with self.assertRaisesRegex(CandidateRejected, "reviewer artifacts are read-only"):
            normalize_candidate(destination, destination, self.directory / "review-write", STARTER,
                                stage="P1", role="reviewer", decision="approve")

    def test_p1_execution_is_the_only_hardware_edit(self):
        before = self.task("before", {"root": self.starter})
        original = self.copy(before, "original")
        root = deepcopy(self.starter)
        root["META_DATA"] = deepcopy(self.base["META_DATA"])
        root["META_DATA"].pop("Dispatch")
        root["CONTRACT"] = deepcopy(self.base["CONTRACT"])
        root["CONTRACT"]["Hardware_specification"]["Architecture"] = "Unapproved hardware"
        write_yaml(original / "root.yaml", root, parser_root=REPO)
        frozen = manifest(original)
        with self.assertRaisesRegex(CandidateRejected, "hardware outside Execution"):
            normalize_candidate(before, original, self.directory / "landed", STARTER, stage="P1", role="worker", decision="complete")
        self.assertEqual(manifest(original), frozen)
        self.assertFalse((self.directory / "landed").exists())

    def test_p1_shared_metadata_is_a_recorded_derivation_without_worker_node_permission(self):
        before = self.task("before", {"root": self.base, "branch": self.branch("branch")})
        previous = source_context(before, parser_root=REPO)["nodes"]["branch"]
        original = self.copy(before, "original")
        root = load_root(original, parser_root=REPO)
        root["META_DATA"].update(Name="scalar_copy_corrected", Describe="Copy one integer exactly.", Version=2)
        write_yaml(original / "root.yaml", root, parser_root=REPO)
        original_hashes = manifest(original)
        final = self.directory / "final"
        result = normalize_candidate(before, original, final, STARTER, stage="P1", role="worker", decision="complete",
                                     selected_designs=selection("root", "branch"))
        self.assertEqual(manifest(original), original_hashes)
        self.assertEqual(result["original_changed_files"], ["root.yaml"])
        self.assertEqual(result["changed_files"], ["branch.yaml", "root.yaml"])
        synchronized = result["shared_metadata_sync"]["nodes"]["branch.yaml"]
        self.assertEqual(synchronized["before_sha256"], original_hashes["branch.yaml"])
        self.assertEqual(synchronized["final_sha256"], result["artifact_digests"]["branch.yaml"])
        current = load_yaml(final, "branch.yaml", parser_root=REPO)
        self.assertEqual(current["META_DATA"]["Name"], "scalar_copy_corrected")
        self.assertEqual(current["META_DATA"]["Describe"], "Copy one integer exactly")
        self.assertEqual(current["META_DATA"]["Version"], 2)
        old = load_yaml(before, "branch.yaml", parser_root=REPO)
        self.assertEqual(current["DESIGN"], old["DESIGN"])
        self.assertEqual(current["META_DATA"]["Source"], old["META_DATA"]["Source"])
        self.assertEqual(current["META_DATA"]["Dispatch"], old["META_DATA"]["Dispatch"])
        check = check_stage_snapshot(final, parser_root=REPO, stage="P5", selected_designs=selection("root", "branch"), node="branch")
        proof = align_source_identity(previous, check["design_resolution"]["nodes"]["branch"])
        self.assertFalse(proof["equivalent"])
        self.assertNotEqual(proof["previous_identity"], proof["current_identity"])
        validate_artifacts("P1", "reviewer", final, final, STARTER, decision="approve")
        with self.assertRaisesRegex(ExtractError, "outside stage scope"):
            validate_artifacts("P1", "worker", before, final, STARTER, decision="complete")
        p5_before = self.copy(final, "p5-before")
        shutil.copyfile(before / "branch.yaml", p5_before / "branch.yaml")
        with self.assertRaisesRegex(ExtractError, "protected task or node identity"):
            validate_artifacts("P5", "worker", p5_before, final, STARTER, decision="complete", node_entry=selection("branch")[0])
        root["META_DATA"]["Version"] = 1
        write_yaml(original / "root.yaml", root, parser_root=REPO)
        with self.assertRaisesRegex(CandidateRejected, "without increasing Version"):
            normalize_candidate(before, original, self.directory / "bad-version", STARTER,
                                stage="P1", role="worker", decision="complete")
        root["META_DATA"]["Version"] = 2
        write_yaml(original / "root.yaml", root, parser_root=REPO)
        current["DESIGN"]["Math_reshape"]["Thesis"] = "Unauthorized node rewrite"
        write_yaml(original / "branch.yaml", current, parser_root=REPO)
        with self.assertRaisesRegex(CandidateRejected, "outside stage scope"):
            normalize_candidate(before, original, self.directory / "bad-node", STARTER,
                                stage="P1", role="worker", decision="complete")
        shutil.copyfile(before / "branch.yaml", original / "branch.yaml")
        root["META_DATA"].pop("Describe")
        write_yaml(original / "root.yaml", root, parser_root=REPO)
        with self.assertRaisesRegex(CandidateRejected, "Describe"):
            normalize_candidate(before, original, self.directory / "missing-field", STARTER,
                                stage="P1", role="worker", decision="complete")
        root["META_DATA"] = None
        write_yaml(original / "root.yaml", root, parser_root=REPO)
        with self.assertRaisesRegex(CandidateRejected, "META_DATA mapping"):
            normalize_candidate(before, original, self.directory / "missing-mapping", STARTER,
                                stage="P1", role="worker", decision="complete")

    def test_atomic_replacement_is_accepted_but_every_residual_entry_is_rejected(self):
        before = self.task("before")
        original = self.copy(before, "original")
        temporary = original / "root.yaml.tmp.12.fixture"
        temporary.write_bytes((original / "root.yaml").read_bytes())
        os.replace(temporary, original / "root.yaml")
        result = normalize_candidate(before, original, self.directory / "good", STARTER,
                                     stage="P1", role="worker", decision="complete")
        self.assertEqual(result["artifact_digests"], manifest(before))
        for kind in ("temporary", "yaml", "directory", "symlink", "hardlink", "protected", "deleted"):
            with self.subTest(kind=kind):
                candidate = self.copy(before, "bad-" + kind)
                if kind == "temporary":
                    (candidate / "root.yaml.tmp.1.leftover").write_text("residual")
                elif kind == "yaml":
                    (candidate / "extra.yaml").write_text("unexpected: file")
                elif kind == "directory":
                    (candidate / "empty").mkdir()
                elif kind == "symlink":
                    (candidate / "root.yaml").unlink()
                    (candidate / "root.yaml").symlink_to(before / "root.yaml")
                elif kind == "hardlink":
                    (candidate / "root.yaml").unlink()
                    external = self.directory / "hardlink-source"
                    shutil.copyfile(before / "root.yaml", external)
                    os.link(external, candidate / "root.yaml")
                elif kind == "protected":
                    (candidate / "dispatch.md").write_text("unpermitted content")
                else:
                    (candidate / "dispatch.md").unlink()
                try:
                    read_bytes = Path.read_bytes
                    def guarded_read(path):
                        if kind in {"temporary", "yaml", "directory", "symlink", "hardlink", "deleted"} and path.parent == candidate:
                            raise AssertionError("unsafe artifact collection was read before preflight completed")
                        return read_bytes(path)
                    with patch.object(Path, "read_bytes", guarded_read):
                        with self.assertRaises(CandidateRejected):
                            normalize_candidate(before, candidate, self.directory / ("rejected-" + kind), STARTER,
                                                stage="P1", role="worker", decision="continue")
                        if kind == "yaml":
                            checked = check_stage_snapshot(candidate, parser_root=REPO, stage="P6", selected_designs=selection("root"))
                            self.assertNotEqual(checked["exit_code"], 0)
                finally:
                    if kind == "hardlink":
                        (candidate / "root.yaml").unlink()
                self.assertFalse((self.directory / ("rejected-" + kind)).exists())

    def graph_task(self, name, *, planned=()):
        provider, helper, consumer = [self.branch(node) for node in ("provider", "helper", "consumer")]
        provider["DESIGN"]["Tile_pipeline"] = {"from": "helper"}
        consumer["DESIGN"]["Math_reshape"] = {"from": "provider"}
        return self.task(name, {"root": self.base, "provider": provider, "helper": helper, "consumer": consumer}, planned=planned)

    def test_parsing_closure_is_larger_than_same_region_manifest(self):
        directory = self.graph_task("task", planned={"helper", "provider", "consumer", "future"})
        (directory / "future.yaml").write_text("not: [a valid completed draft]\n")
        selected = selection("root", "provider", "helper", "consumer", "future")
        graph = source_context(directory, parser_root=REPO, nodes=["consumer"])
        consumer = graph["nodes"]["consumer"]
        self.assertEqual(set(consumer["source_manifest"]), {"root.yaml", "provider.yaml", "consumer.yaml"})
        self.assertEqual(set(consumer["resolution_manifest"]), {"root.yaml", "provider.yaml", "consumer.yaml", "helper.yaml"})
        checked = check_stage_snapshot(directory, parser_root=REPO, stage="P5", selected_designs=selected, node="consumer")
        self.assertEqual(checked["exit_code"], 0, checked)
        self.assertEqual(checked["excluded_nodes"], ["future"])
        self.assertEqual(checked["design_resolution"]["nodes"]["consumer"]["validation"], "strict")
        identities = {node: value["effective_identity"] for node, value in graph["nodes"].items()}
        state = {"nodes": {node: {"status": "approved", "approval": {"effective_identity": identity}}
                           for node, identity in identities.items()}}
        attempt = {"stage": "P5", "node": "consumer", "source_identities": identities, "dependency_digests": manifest(directory)}
        context = design_context(state, attempt, directory, parser_root=REPO)
        self.assertEqual(context["provider_reopen_targets"]["P5:provider"]["regions"], {"Math_reshape": ["consumer", "provider"]})
        attempt["source_identities"] = {}
        self.assertEqual(design_context(state, attempt, directory, parser_root=REPO)["provider_reopen_targets"], {})
        final = check_stage_snapshot(directory, parser_root=REPO, stage="P6", selected_designs=selected)
        self.assertNotEqual(final["exit_code"], 0)
        self.index(directory, {item["id"] for item in selected})
        final = check_stage_snapshot(directory, parser_root=REPO, stage="P6", selected_designs=selected)
        self.assertNotEqual(final["exit_code"], 0)
        (directory / "future.yaml").unlink()
        selected = selected[:-1]
        self.index(directory, {item["id"] for item in selected})
        self.assertEqual(check_stage_snapshot(directory, parser_root=REPO, stage="P6", selected_designs=selected)["exit_code"], 0)
        helper = load_yaml(directory, "helper.yaml", parser_root=REPO)
        helper["DESIGN"]["Work_schedule"] = {"from": "consumer"}
        write_yaml(directory / "helper.yaml", helper, parser_root=REPO)
        with self.assertRaisesRegex(ExtractError, "cross-region source cycle"):
            source_context(directory, parser_root=REPO, nodes=["consumer"])

    def test_source_identity_drift_requires_current_provenance_alignment(self):
        directory = self.graph_task("task")
        selected = selection("root", "provider", "helper", "consumer")
        old = check_stage_snapshot(directory, parser_root=REPO, stage="P5", selected_designs=selected, node="consumer")["design_resolution"]["nodes"]["consumer"]
        provider = load_yaml(directory, "provider.yaml", parser_root=REPO)
        provider["META_DATA"]["Version"] = 2
        provider["DESIGN"]["Work_schedule"]["Thesis"] = "Produce the scalar with the same worker assignment"
        write_yaml(directory / "provider.yaml", provider, parser_root=REPO)
        result = check_stage_snapshot(directory, parser_root=REPO, stage="P5", selected_designs=selected, node="consumer")
        self.assertEqual(result["exit_code"], 0, result)
        current = result["design_resolution"]["nodes"]["consumer"]
        aligned = align_source_identity(old, current)
        self.assertTrue(aligned["equivalent"])
        self.assertNotEqual(aligned["previous_identity"], aligned["current_identity"])
        self.assertEqual(aligned["changed_sources"], ["provider.yaml"])
        provider["DESIGN"]["Math_reshape"]["Thesis"] = "A changed used definition"
        write_yaml(directory / "provider.yaml", provider, parser_root=REPO)
        current = check_stage_snapshot(directory, parser_root=REPO, stage="P5", selected_designs=selected, node="consumer")["design_resolution"]["nodes"]["consumer"]
        self.assertFalse(align_source_identity(old, current)["equivalent"])
        with self.assertRaisesRegex(ExtractError, "strictly validated"):
            align_source_identity(old, source_context(directory, parser_root=REPO, nodes=["consumer"])["nodes"]["consumer"])
        consumer = load_yaml(directory, "consumer.yaml", parser_root=REPO)
        consumer["DESIGN"]["Math_reshape"] = deepcopy(provider["DESIGN"]["Math_reshape"])
        write_yaml(directory / "consumer.yaml", consumer, parser_root=REPO)
        localized = source_context(directory, parser_root=REPO, nodes=["consumer"])["nodes"]["consumer"]
        self.assertEqual(set(localized["source_manifest"]), {"root.yaml", "consumer.yaml"})
        self.assertEqual(localized["origins"]["Math_reshape"]["provider"], "consumer")

    def test_normalization_does_not_rewrite_read_only_provider(self):
        before = self.graph_task("before")
        provider = load_yaml(before, "provider.yaml", parser_root=REPO)
        provider["DESIGN"]["Math_reshape"]["Thesis"] += "."
        write_yaml(before / "provider.yaml", provider, parser_root=REPO)
        original = self.copy(before, "original")
        consumer = load_yaml(original, "consumer.yaml", parser_root=REPO)
        consumer["META_DATA"]["Version"] = 2
        consumer["DESIGN"]["Work_schedule"]["Thesis"] = "The consumer assigns its worker explicitly."
        write_yaml(original / "consumer.yaml", consumer, parser_root=REPO)
        hashes = manifest(original)
        destination = self.directory / "landed"
        with self.assertRaisesRegex(CandidateRejected, "outside stage scope") as caught:
            normalize_candidate(before, original, destination, STARTER, stage="P5", role="worker", decision="complete",
                                node_entry=selection("consumer")[0], selected_designs=selection("root", "provider", "helper", "consumer"))
        self.assertIn("provider.yaml", caught.exception.validation["normalized_files"])
        self.assertTrue(caught.exception.validation["corrections"])
        self.assertFalse(destination.exists())
        self.assertEqual(manifest(original), hashes)

        contract_original = self.copy(before, "contract-original")
        root = load_root(contract_original, parser_root=REPO)
        inputs = root["CONTRACT"]["Algorithm_requirement"]["Input"]
        inputs["X"] = inputs.pop("x")
        root["META_DATA"]["Version"] += 1
        write_yaml(contract_original / "root.yaml", root, parser_root=REPO)
        with self.assertRaisesRegex(CandidateRejected, "outside stage scope") as caught:
            normalize_candidate(before, contract_original, self.directory / "contract-landed", STARTER,
                                stage="P1", role="worker", decision="complete")
        self.assertEqual(set(caught.exception.validation["source_hashes"]), set(hashes) - {"dispatch.md"})
        self.assertIn("provider.yaml", caught.exception.validation["normalized_files"])

    def test_directory_future_entries_never_gain_fake_links(self):
        directory = self.task("task", planned={"future"})
        selected = selection("root", "future")
        validate_index(directory, selected, parser_root=REPO)
        for stage in ("P2", "P3", "P4"):
            checked = check_stage_snapshot(directory, parser_root=REPO, stage=stage, selected_designs=selected)
            self.assertEqual(checked["exit_code"], 0, checked)
        self.index(directory, {"root", "future"})
        with self.assertRaises(ExtractError):
            validate_index(directory, selected, parser_root=REPO)
        (directory / "future.yaml").write_text("placeholder: forbidden\n")
        with self.assertRaises(ExtractError):
            validate_index(directory, selection("root"), parser_root=REPO)

    def test_p6_changes_are_limited_to_the_reviewed_fields(self):
        before = self.task("before", {"root": self.base, "branch": self.branch("branch")})
        original = self.copy(before, "original")
        branch = load_yaml(original, "branch.yaml", parser_root=REPO)
        branch["META_DATA"]["Version"] = 2
        branch["DESIGN"]["Math_reshape"]["Thesis"] = "A clarified scalar rewrite"
        write_yaml(original / "branch.yaml", branch, parser_root=REPO)
        scope = [{"path": "branch.yaml", "fields": ["DESIGN.Math_reshape"]}]
        validate_artifacts("P6", "worker", before, original, STARTER, decision="complete", revision_scope=scope)
        branch["META_DATA"]["Dispatch"]["when"] = "An unrelated change to scope"
        write_yaml(original / "branch.yaml", branch, parser_root=REPO)
        with self.assertRaisesRegex(ExtractError, "outside the explicit revision scope"):
            validate_artifacts("P6", "worker", before, original, STARTER, decision="complete", revision_scope=scope)

    def test_adjudication_format_patch_uses_same_transaction_and_keeps_authorship(self):
        before = self.task("before")
        original = self.copy(before, "original")
        path = original / "root.yaml"
        old = "Retain the scalar copy without intermediate values"
        path.write_text(path.read_text().replace(old, old + "."))
        adjudication = {"format": "schema_extract.receipt.v2", "identity": {"role": "reviewer", "mode": "adjudicate", "stage": "P3"},
                        "decision": "approve", "adjudication": {"action": "patch", "checkpoint": "selected-candidate",
                        "patches": [{"path": "root.yaml", "old": old, "new": old + "."}]}}
        destination = self.directory / "landed"
        result = normalize_candidate(before, original, destination, STARTER, stage="P3", role="runner", decision="approve",
                                     selected_designs=selection("root"), adjudication=adjudication)
        self.assertEqual(result["origin"], "runner_adjudication")
        self.assertEqual(manifest(destination), manifest(before))
        self.assertEqual(load_root(destination, parser_root=REPO)["META_DATA"]["Version"], 1)
        self.assertTrue(result["corrections"])
        path.write_text(path.read_text().replace("Copy one scalar", "Unrecorded change"))
        with self.assertRaisesRegex(CandidateRejected, "exact adjudication patch"):
            normalize_candidate(before, original, self.directory / "bad-landed", STARTER, stage="P3", role="runner", decision="approve",
                                selected_designs=selection("root"), adjudication=adjudication)

    def test_stage_approval_projection_preserves_prior_responsibilities(self):
        directory = self.task("task")
        p1 = approval_projection("P1", directory, parser_root=REPO)
        root = deepcopy(self.base)
        root["META_DATA"]["Source"] = "testing.invalid/kernel@fixture-v2:kernel.py::copy"
        root["META_DATA"]["Version"] = 2
        write_yaml(directory / "root.yaml", root, parser_root=REPO)
        after = approval_projection("P1", directory, parser_root=REPO)
        self.assertEqual(p1["subject_sha256"], after["subject_sha256"])
        self.assertNotEqual(p1["raw_manifest"], after["raw_manifest"])

    def test_current_contract_comparator_keeps_output_and_operand_evidence(self):
        directory = self.task("task")
        source = self.directory / "source"
        source.mkdir()
        text = json.dumps({"outputs": {"y": {"tolerance": {"relative_to": "candidate", "atol": 0.01, "rtol": 0.05}}}})
        (source / "acceptance.json").write_text(text + "\n")
        lock = {"snapshot": str(self.directory), "source": {"bindings": {"acceptance": [{"path": "acceptance.json"}]}}}
        receipt = {"comparisons": [{"output": "y", "source_output": "y", "evidence": [0]}],
                   "source_evidence": [{"category": "acceptance", "path": "acceptance.json", "line_start": 1, "line_end": 1, "quote": text}]}
        root = deepcopy(self.base)
        tolerance = root["CONTRACT"]["Algorithm_requirement"]["Golden_reference"]["Tolerance"]
        tolerance["y"] = "relative_to=candidate; atol=0.01; rtol=0.05"
        write_yaml(directory / "root.yaml", root, parser_root=REPO)
        validate_comparisons(receipt, lock, directory, parser_root=REPO)
        tolerance["y"] = "relative_to=reference; atol=0.01; rtol=0.05"
        write_yaml(directory / "root.yaml", root, parser_root=REPO)
        with self.assertRaisesRegex(ExtractError, "relative_to=candidate"):
            validate_comparisons(receipt, lock, directory, parser_root=REPO)


if __name__ == "__main__":
    unittest.main()
