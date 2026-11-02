"""Shared synthetic operator, receipts and checkpoints for extraction tests."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest

from ..codex_cli import write_json
from ..planning import DEFAULTS, freeze_inputs, new_task
from ..protocol import RECEIPT
from ..records import Store


REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "schema_template/tools/tests/fixtures/minimal/root.yaml"
PROPOSAL = {"id": "selector-diagnostics", "category": "extract_tools", "problem": "Synthetic ambiguous diagnostic.",
            "evidence": ["schema_template/check_yaml.py:1 (synthetic example)"], "proposal": "Identify the selector.",
            "necessity": "Otherwise any selected task can be sent to fix the wrong node."}


class ExtractionCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="schema-extract-test-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.repo = self.base / "repository"
        self.repo.mkdir()
        for name in ("schema_docs", "schema_extract"):
            shutil.copytree(REPO / name, self.repo / name, ignore=shutil.ignore_patterns("tests", "__pycache__"))
        shutil.copytree(REPO / "schema_template/tools", self.repo / "schema_template/tools",
                        ignore=shutil.ignore_patterns("tests", "__pycache__"))
        for name in ("README.md", "EXTRACT_SCHEMA.md", "schema_template/README.md", "schema_template/starter.yaml", "schema_template/check_yaml.py"):
            destination = self.repo / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / name, destination)
        self.entry = {"id": "synthetic", "title": "Synthetic scalar copy", "paths": {}}
        operator = self.repo / "kernel_wiki/test/operators/synthetic"
        operator.mkdir(parents=True)
        for name in ("readme", "entrypoint", "input", "reference", "acceptance", "metadata"):
            relative = "test/operators/synthetic/" + name + ".txt"
            self.entry["paths"][name] = relative
            (self.repo / "kernel_wiki" / relative).write_text(f"{name}: synthetic scalar copy evidence\n")
        write_json(self.repo / "kernel_wiki/catalog.json", {"operators": [self.entry]})
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(["git", "add", "."], cwd=self.repo, check=True)
        subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
                        "-c", "user.name=Test", "-c", "user.email=test@localhost", "commit", "-qm", "fixture"], cwd=self.repo, check=True)
        self.store = Store(self.repo, self.base / "records")
        self.config = {**DEFAULTS, "jobs": 2}
        self.baseline = freeze_inputs(self.store, self.entry, [], self.config)
        self.state = new_task(self.store, self.baseline, self.base / "output/synthetic")
        self.plan = {"run_id": "test", "tasks": [self.state["key"]], "config": self.config,
                     "work_root": str(self.base / "work")}

    def draft(self, work, version=1, *, gap=False):
        work.mkdir(parents=True, exist_ok=True)
        prefix = self.baseline["repository_id"] + "@" + self.baseline["snapshot_commit"] + ":kernel_wiki/"
        text = FIXTURE.read_text().replace("  version: 1", f"  source: {prefix}{self.entry['paths']['entrypoint']}\n  version: {version}")
        text = text.replace("testing.invalid/kernel@fixture-v1:reference.py::copy", prefix + self.entry["paths"]["reference"] + "::copy")
        if not gap:
            import re
            text = re.sub(r'"<testing.invalid/[^\n]+?>"', '"' + prefix + self.entry["paths"]["metadata"] + '"', text)
        (work / "root.yaml").write_text(text)
        (work / "dispatch.md").write_text("[root](root.yaml): synthetic scalar copy.\n")

    def receipt(self, state=None, *, role="worker", status="complete", gap=False):
        state = state or self.state
        value = {key: [] if field["type"] == "array" else "" for key, field in RECEIPT["properties"].items()}
        value.update(task_key=state["key"], fingerprint=state["fingerprint"], base_checkpoint=state.get("checkpoint") or "none",
                     stage=state["stage"], node=state["node"], status=status, summary="Synthetic test receipt.", impact="none", correction="none",
                     continuation="not_assessed")
        if role == "reviewer" and state["review_scope"] == "contract":
            value["continuation"] = "needs_input" if status == "needs_input" else "proceed"
            value["continuation_reason"] = "Synthetic fundamental ambiguity remains." if status == "needs_input" else "Input, reference and entrypoint identify scalar-copy IO and computation; source-fact extraction can continue."
        if gap:
            value["issues"] = [{"id": "missing_material", "kind": "evidence_gap", "scope": "external numeric proof", "disposition": "note",
                                "detail": "Synthetic required material is absent.", "evidence": ["synthetic fixture"],
                                "restart_condition": "Supply the missing material and create a new plan.", "blocks": ["final"]}]
            if role == "reviewer" and status == "needs_input":
                value["issues"][0].update(disposition="needs_input", scope="fundamental task definition")
        return value

    def review_receipt(self, state, work, *, status="complete", gap=False):
        value = self.receipt(state, role="reviewer", status=status, gap=gap)
        baseline = self.store.baseline(state["key"])
        value["source_evidence"] = [{"path": "kernel_wiki/" + relative, "line_start": 1, "line_end": 1,
                                     "quote": (Path(baseline["snapshot"]) / "kernel_wiki" / relative).read_text().strip(),
                                     "note": "Independently read synthetic source."}
                                    for relative in self.entry["paths"].values()]
        value["artifact_evidence"] = [{"path": path.name, "line_start": 1, "line_end": 1,
                                       "quote": path.read_text().splitlines()[0],
                                       "note": "Independently read complete synthetic artifact."}
                                      for path in work.iterdir() if path.is_file()]
        return value

    def checkpoint(self, state=None, *, gap=False, candidates=()):
        state = state or self.state
        work = self.base / ("artifact-" + str(time.monotonic_ns()))
        self.draft(work, gap=gap)
        receipt = self.receipt(state)
        receipt["framework_candidates"] = list(candidates)
        return self.store.commit_checkpoint(state, work, receipt, {"exit_code": 0, "policy": "skip-hardware"})
