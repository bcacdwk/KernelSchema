"""Whole-region derivation uses consumer context and preserves real provenance."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from decimal import Decimal
import hashlib
from io import StringIO
import json
from unittest.mock import patch

import check_yaml
from .. import design_resolution, schema, yaml_source
from ..diagnostics import Diagnostic, encode_value, format_diagnostics
from ..errors import CapacityError
from ..pipeline import output
from ..pipeline_graph import main as graph_main
from .support import PHASE, ROLE, TaskCase, add_child, put, tiny_png


class ResolutionTests(TaskCase):
    def derivation(self):
        task = deepcopy(self.base)
        add_child(task, "provider")
        add_child(task, "middle", "provider")
        add_child(task, "child", "middle")
        for node, parent in (("middle", "provider"), ("child", "middle")):
            task[node]["design"] = {region: {"from_parent": parent} for region in design_resolution.REGIONS}
        task["child"]["design"]["work_schedule"] = deepcopy(task["root"]["design"]["work_schedule"])
        task["provider"]["design"]["math_reshape"]["rewrite"][0]["symbol"] = {
            "WorkerCount": {"meaning": "The current node's worker count.",
                            "binding": "bind:work_schedule.phase.copy.mapping.worker_count"}}
        task["child"]["design"]["work_schedule"]["phase"]["copy"]["mapping"]["worker_count"] = 2
        return task

    def test_mixed_transitive_regions_keep_local_context_and_raw_sources(self):
        task = self.derivation()
        task["provider"]["realization"] = {"require": [{"item": "produce:Y", "choice": "local", "reason": "Provider choice."}]}
        self.write_task(task)
        (self.directory / "provider.yaml").rename(self.directory / "provider.yml")
        raw_hashes = {}
        directory, resolved = schema.validate_resolved_file(self.directory / "child.yaml", source_hashes=raw_hashes)
        child = resolved.effective_task["child"]
        self.assertEqual(directory, self.directory)
        self.assertEqual(task, resolved.raw_task)
        self.assertEqual(task["provider"]["design"]["math_reshape"], child["design"]["math_reshape"])
        self.assertEqual(task["child"]["design"]["work_schedule"], child["design"]["work_schedule"])
        self.assertEqual(task["child"]["meta_data"], child["meta_data"])
        self.assertNotIn("realization", child)
        self.assertNotIn("contract", child)
        self.assertEqual(resolved.origins["child"]["math_reshape"],
                         schema.RegionOrigin("provider", ("child", "middle", "provider")))
        self.assertEqual(resolved.origins["child"]["work_schedule"].chain, ("child",))
        self.assertEqual(resolved.dependency_nodes["child"], {"root", "provider", "middle", "child"})
        self.assertEqual(set(resolved.source_manifest("child")), {"root.yaml", "provider.yml", "middle.yaml", "child.yaml"})
        self.assertEqual(raw_hashes, resolved.source_hashes)
        self.assertEqual(schema.validate_file(self.directory / "child.yaml")[1], resolved.effective_task)
        self.assertEqual(yaml_source.load(self.directory / "child.yaml", "child").value, task["child"])

    def test_child_context_failure_points_to_provider_value_and_reference_use(self):
        task = self.derivation()
        put(task, ("provider",) + PHASE[1:] + ("mapping", "worker", "unit_count"), 2)
        put(task, ("provider",) + ROLE[1:] + ("info", "member"), "CTA[0:2]")
        self.write_task(task)
        resolved = schema.resolve_directory(self.directory)
        field = "design.tile_pipeline.phase.copy.role.Main.info.member"
        provider_position = resolved.documents["provider"].positions["provider." + field]
        use_position = resolved.documents["child"].positions["child.design.tile_pipeline.from_parent"]
        with self.assertRaises(schema.SchemaError) as caught:
            schema.validate_file(self.directory / "child.yaml")
        message = str(caught.exception)
        self.assertIn("member index exceeds worker.unit_count", message)
        self.assertIn(f"provider.yaml:{provider_position[0]}:{provider_position[1]}", message)
        self.assertIn(f"child.yaml:{use_position[0]}:{use_position[1]}", message)
        self.assertIn("Consumer: child." + field, message)
        self.assertIn("Source chain: child -> middle -> provider", message)
        self.assertIn('Actual: "CTA[0:2]"', message)

        # A missing effective field has no invented line in either document.
        missing = format_diagnostics([Diagnostic("child.design.tile_pipeline.absent", "required", missing=True)],
                                     resolved.source_files, resolved.documents, resolved=resolved)
        self.assertTrue(missing.startswith(str(self.directory / "provider.yaml") + "\n"))
        self.assertNotIn("Actual:", missing)
        self.assertIn("Inherited at:", missing)

    def test_inherited_pipeline_must_match_child_local_phase_names(self):
        task = self.derivation()
        schedule = task["child"]["design"]["work_schedule"]
        schedule["phase"]["renamed"] = schedule["phase"].pop("copy")
        schedule["flow"]["launch"] = [{"phase": ["renamed"]}]
        self.assert_task_invalid(task, "must describe the same phase IDs", "Consumer: child.design.tile_pipeline.phase",
                                 "Source chain: child -> middle -> provider")

    def test_views_do_not_share_mutable_regions_or_mutate_parser_documents(self):
        self.write_task(self.derivation())
        resolved = schema.resolve_directory(self.directory)
        raw = deepcopy(resolved.raw_task)
        provider = deepcopy(resolved.effective_task["provider"])
        resolved.effective_task["child"]["design"]["math_reshape"]["rewrite"].clear()
        self.assertEqual(resolved.raw_task, raw)
        self.assertEqual(resolved.effective_task["provider"], provider)
        self.assertEqual(resolved.effective_task["middle"]["design"]["math_reshape"], provider["design"]["math_reshape"])
        view = resolved.effective_document("middle")
        view["contract"].clear()
        view["design"]["math_reshape"].clear()
        self.assertEqual(resolved.documents["provider"].value, raw["provider"])
        self.assertEqual(resolved.effective_task["root"]["contract"], raw["root"]["contract"])

    def test_source_identity_changes_without_child_edits_and_excludes_unused_ancestors(self):
        task = self.derivation()
        add_child(task, "local", "middle")
        self.write_task(task)
        first = schema.resolve_directory(self.directory)
        before_child = (self.directory / "child.yaml").read_bytes()
        source = self.directory / "provider.yaml"
        source.write_bytes(source.read_bytes() + b"\n")
        second = schema.resolve_directory(self.directory)
        self.assertEqual(first.effective_task, second.effective_task)
        self.assertNotEqual(first.effective_identity("child"), second.effective_identity("child"))
        self.assertEqual(first.effective_identity("local"), second.effective_identity("local"))
        self.assertEqual(second.dependency_nodes["local"], {"root", "local"})
        self.assertEqual(before_child, (self.directory / "child.yaml").read_bytes())
        source.rename(self.directory / "provider.yml")
        third = schema.resolve_directory(self.directory)
        self.assertNotEqual(second.effective_identity("child"), third.effective_identity("child"))
        self.assertEqual(third.source_hashes["provider"], hashlib.sha256(source.with_suffix(".yml").read_bytes()).hexdigest())

    def test_invalid_reference_shapes_targets_identities_and_cycles_are_invalid_input(self):
        cases = (
            ("child", "math_reshape", {"from_parent": "middle", "thesis": "override"}, "only field"),
            ("child", "math_reshape", {"from_parent": "provider"}, "direct parent"),
            ("child", "math_reshape", {"from_parent": "../middle"}, "lowercase node ID"),
            ("child", "math_reshape", {"from_parent": ["middle"]}, "lowercase node ID"),
            ("root", "math_reshape", {"from_parent": "root"}, "root Design regions must be local"),
        )
        for node, region, value, expected in cases:
            with self.subTest(value=value):
                task = self.derivation()
                task[node]["design"][region] = value
                self.write_task(task)
                for entry in (schema.resolve_directory, schema.validate_directory):
                    with self.assertRaisesRegex(schema.SchemaError, expected):
                        entry(self.directory)
        for field, value, expected in (("id", "another", "match the YAML filename"),
                                       ("parent", "absent", "unknown parent")):
            task = self.derivation()
            task["provider"]["meta_data"]["dispatch"][field] = value
            self.write_task(task)
            with self.assertRaisesRegex(schema.SchemaError, expected):
                schema.resolve_directory(self.directory)
        task = self.derivation()
        task["provider"]["meta_data"]["dispatch"]["parent"] = "child"
        self.write_task(task)
        with self.assertRaisesRegex(schema.SchemaError, "cycle"):
            schema.resolve_directory(self.directory)

    def test_show_effective_preserves_draft_fields_decimal_and_scalar_types_without_validation(self):
        task = self.derivation()
        task["child"]["meta_data"]["dispatch"]["when"] = ""
        task["child"]["design"]["work_schedule"] = {"thesis": "1e3", "draft": [True, False, None, "false", 7]}
        task["root"]["contract"]["algorithm_requirement"]["constant"] = {
            "Precise": {"value": Decimal("0.123456789012345678901234567890123456789"), "meaning": "Exact decimal."}}
        self.write_task(task)
        before, stdout, stderr = self.snapshot(), StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = check_yaml.main(["--show-effective", str(self.directory / "child.yaml")])
        self.assertEqual(code, 0, stderr.getvalue())
        envelope = json.loads(stdout.getvalue())
        self.assertEqual(envelope["format"], "kernel_schema.effective_view.v1")
        self.assertEqual(envelope["validation"], "not_run")
        self.assertNotIn("Valid:", stdout.getvalue())
        view = yaml_source.loads(envelope["effective_yaml"], "child").value
        self.assertEqual(view["contract"], task["root"]["contract"])
        self.assertIsInstance(view["contract"]["algorithm_requirement"]["constant"]["Precise"]["value"], Decimal)
        self.assertEqual(view["design"]["work_schedule"], task["child"]["design"]["work_schedule"])
        self.assertEqual(view["meta_data"]["dispatch"]["when"], "")
        self.assertEqual(set(envelope["source_files"]), {"root.yaml", "provider.yaml", "middle.yaml", "child.yaml"})
        self.assertEqual(before, self.snapshot())
        with self.assertRaises(schema.SchemaError):
            schema.validate_file(self.directory / "child.yaml")

    def test_default_root_parent_and_standalone_rejects_residual_regions(self):
        task = deepcopy(self.base)
        add_child(task)
        task["small"]["design"]["math_reshape"] = {"from_parent": "root"}
        self.write_task(task)
        _, resolved = schema.validate_resolved_file(self.directory / "small.yaml")
        self.assertEqual(resolved.origins["small"]["math_reshape"].chain, ("small", "root"))
        task["small"]["contract"] = deepcopy(task["root"]["contract"])
        (self.directory / "small.yaml").write_text(encode_value(task["small"]))
        with self.assertRaisesRegex(schema.SchemaError, "from_parent"):
            schema.validate_standalone_file(self.directory / "small.yaml")

    def test_skipping_hardware_changes_only_the_checking_copy(self):
        task = self.derivation()
        task["root"]["contract"]["hardware_specification"] = {"unfinished": True}
        self.write_task(task)
        with self.assertRaises(schema.SchemaError):
            schema.validate_resolved_file(self.directory / "child.yaml")
        _, resolved = schema.validate_resolved_file(self.directory / "child.yaml", skip_hardware=True)
        self.assertEqual(resolved.raw_task, task)
        self.assertEqual(resolved.effective_task["root"]["contract"], task["root"]["contract"])
        self.assertEqual(resolved.effective_document("child")["contract"], task["root"]["contract"])

    def test_deep_parent_and_reference_chains_are_iterative(self):
        parent = "root"
        for i in range(1100):
            node = f"node_{i:04d}"
            plan = {"meta_data": {"dispatch": {"id": node, "parent": parent, "when": ""}},
                    "design": {region: {"from_parent": parent} for region in design_resolution.REGIONS}}
            (self.directory / (node + ".yaml")).write_text(encode_value(plan))
            parent = node
        resolved = schema.resolve_directory(self.directory)
        self.assertEqual(len(resolved.origins[parent]["tile_pipeline"].chain), 1101)
        self.assertEqual(len(resolved.dependency_nodes[parent]), 1101)
        self.assertEqual(resolved.effective_task[parent]["design"], self.base["root"]["design"])

    def test_total_capacity_counts_fanout_copies_and_provenance(self):
        task = deepcopy(self.base)
        task["root"]["design"]["math_reshape"] = {"draft_values": list(range(400))}
        for i in range(20):
            node = f"child_{i}"
            task[node] = {"meta_data": {"dispatch": {"id": node}},
                          "design": {region: {"from_parent": "root"} for region in design_resolution.REGIONS}}
        self.write_task(task)
        before = self.snapshot()
        with patch.object(design_resolution, "MAX_RESOLVED_UNITS", 3000):
            with self.assertRaises(CapacityError):
                schema.resolve_directory(self.directory)
            stdout, stderr = StringIO(), StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = check_yaml.main(["--show-effective", str(self.directory / "child_19.yaml")])
            self.assertEqual(code, 2)
            self.assertIn("capacity", stderr.getvalue())
        self.assertEqual(before, self.snapshot())
        self.assertEqual(len(schema.resolve_directory(self.directory).effective_task), 21)

        # Even empty local bodies accumulate quadratic provenance along a
        # chain. Charging only the copied Design would miss this capacity.
        task = {"root": {"meta_data": {}, "design": {region: {} for region in design_resolution.REGIONS}}}
        parent = "root"
        for i in range(50):
            node = f"node_{i}"
            task[node] = {"meta_data": {"dispatch": {"id": node, "parent": parent}},
                          "design": {region: {"from_parent": parent} for region in design_resolution.REGIONS}}
            parent = node
        self.write_task(task)
        with patch.object(design_resolution, "MAX_RESOLVED_UNITS", 3000):
            with self.assertRaises(CapacityError):
                schema.resolve_directory(self.directory)

    def test_graph_errors_locate_inherited_raw_pipeline_fields(self):
        task = self.derivation()
        role = task["provider"]["design"]["tile_pipeline"]["phase"]["copy"]["role"]["Main"]
        role["pipe"] = {"steady": ["produce:Y(0)"]}
        role["waits_for"] = {"data": {"produce:Y(0)": ["produce:Y(99)"]}}
        self.write_task(task)
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = graph_main([str(self.directory / "child.yaml"), "--check-only"])
        self.assertEqual(code, 1, stderr.getvalue())
        self.assertIn("provider.yaml:1:", stderr.getvalue())
        self.assertIn("Consumer: child.design.tile_pipeline", stderr.getvalue())
        self.assertIn("Source chain: child -> middle -> provider", stderr.getvalue())

    def test_graph_metadata_and_all_dependency_drift_checks(self):
        task = self.derivation()
        self.write_task(task)
        child = self.directory / "child.yaml"

        def run(renderer):
            stdout, stderr = StringIO(), StringIO()
            with patch.object(output, "graphviz", side_effect=renderer), redirect_stdout(stdout), redirect_stderr(stderr):
                code = graph_main([str(child)])
            return code, stderr.getvalue()

        code, error = run(lambda dot: tiny_png())
        self.assertEqual(code, 0, error)
        info = output.metadata(child.with_suffix(".png"))
        resolved = schema.resolve_directory(self.directory)
        self.assertEqual(info["source"], "child.yaml")
        self.assertEqual(info["source_sha256"], resolved.source_hashes["child"])
        self.assertEqual(info["source_manifest"], resolved.source_manifest("child"))
        self.assertEqual(info["effective_identity"], resolved.effective_identity("child"))
        self.assertEqual(info["origins"], resolved.origins_dict("child"))
        for dependency in ("root", "provider", "middle"):
            with self.subTest(dependency=dependency):
                def change_during_render(dot):
                    source = self.directory / (dependency + ".yaml")
                    source.write_bytes(source.read_bytes() + b"\n")
                    return tiny_png()
                code, error = run(change_during_render)
                self.assertEqual(code, 2, error)
                self.assertIn(dependency + ".yaml: source changed after checking", error)
                self.assertFalse(child.with_suffix(".png").exists())

        resolved = schema.resolve_directory(self.directory)
        (self.directory / "middle.yaml").unlink()
        with patch.object(output, "graphviz") as render:
            with self.assertRaisesRegex(output.RenderError, "source changed after checking"):
                output.GraphOutput().write(child, [], {}, [], source_sha256=resolved.source_hashes["child"],
                                           source_manifest=resolved.source_manifest("child"),
                                           effective_identity=resolved.effective_identity("child"),
                                           origins=resolved.origins_dict("child"))
            render.assert_not_called()
