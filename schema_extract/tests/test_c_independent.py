"""Independent C acceptance through executable CLI output and the real checker.

The controlled CLI does not import the extractor. It follows the public prompt
and receipt schema, writes a small scalar-copy artifact, and records what each
role actually received. Assertions exercise the Runner, not a second state
machine implemented in the tests.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from schema_extract.agent import CodexCLI
from schema_extract.common import manifest
from schema_extract.runner import create_run, run


REPO = Path(__file__).resolve().parents[2]


CONTROLLED_CLI = r'''#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import re
import sys
from uuid import uuid4
import yaml

if '--version' in sys.argv:
    print('codex-cli independent-C-test')
    raise SystemExit(0)

base = Path(__file__).parent
scenario = json.loads((base / 'scenario.json').read_text())
prompt = sys.stdin.read()
context = json.loads(re.search(r'<invocation>\n(.*?)\n</invocation>', prompt, re.S).group(1))
identity = context['identity']
stage, node, role = identity['stage'], identity['node'], identity['role']
mode = identity.get('mode', context.get('mode', 'normal'))
log_path = base / 'calls.jsonl'
previous = [json.loads(line) for line in log_path.read_text().splitlines()] if log_path.exists() else []
number = 1 + sum((c['stage'], c['node'], c['role'], c['mode']) == (stage, node, role, mode)
                 for c in previous)
key = '.'.join((stage, node, role, mode))
actions = scenario.get('actions', {})
action = actions.get(key + '.' + str(number), actions.get(key + '.*', 'normal'))
resumed = sys.argv[sys.argv.index('resume') + 1] if 'resume' in sys.argv else None
session_id = resumed or str(uuid4())
if action == 'reuse_first_session':
    session_id = previous[0]['session_id']
if action == 'wrong_session':
    session_id = str(uuid4())
print(json.dumps({'type': 'thread.started', 'thread_id': session_id}), flush=True)
workspace = Path.cwd()
before_text = {p.name: p.read_text() for p in workspace.iterdir() if p.is_file()}
record = {'stage': stage, 'node': node, 'role': role, 'mode': mode, 'number': number,
          'action': action, 'session_id': session_id, 'resumed': resumed,
          'argv': sys.argv[1:], 'context': context,
          'before': {name: yaml.safe_load(text) for name, text in before_text.items() if name.endswith('.yaml')},
          'before_hashes': {name: hashlib.sha256(text.encode()).hexdigest() for name, text in before_text.items()}}
with log_path.open('a') as stream:
    stream.write(json.dumps(record) + '\n')
if action == 'cli_failure':
    print('controlled CLI failure after publishing its session ID', file=sys.stderr)
    raise SystemExit(23)

complete = yaml.safe_load((base / 'complete.yaml').read_text())
plan = scenario['plan']
filename = 'root.yaml' if stage != 'P5' else node + '.yaml'
if role == 'worker':
    current = yaml.safe_load((workspace / filename).read_text())
    if stage == 'P1':
        first = current['meta_data'].get('source') is None
        current['meta_data'].update({'name': complete['meta_data']['name'],
                                    'describe': complete['meta_data']['describe'],
                                    'source': context['implementation_source']})
        current['contract']['algorithm_requirement'] = complete['contract']['algorithm_requirement']
        current['contract']['algorithm_requirement']['golden_reference']['source'] = context['reference_source']
        current['meta_data']['version'] = 1 if first else current['meta_data']['version'] + 1
    elif stage == 'P2':
        current['meta_data']['dispatch'] = {'when': 'Select when X is nonnegative.' if len(plan) > 1
                                                     else 'Select for every scalar in the contract domain.'}
        current['meta_data']['version'] += 1
        (workspace / 'dispatch.md').write_text('# 路径草案\n\n[root](root.yaml) 覆盖代表调用。\n')
    elif stage == 'P3':
        current['design'] = complete['design']
        current['design']['math_reshape']['thesis'] = 'Copy scalar X without intermediates; root revision ' + str(number) + '.'
        current['meta_data']['version'] += 1
        if action == 'bad_design':
            current['design']['math_reshape']['thesis'] = None
    elif stage == 'P4':
        text = '# 路径清单\n\n<!-- independent-index -->\n\n'
        text += '\n'.join('- [' + item['id'] + '](' + item['file'] + ')；直接父：' + (item['parent'] or '无')
                          + '；' + item['workload'] + '；' + item['difference'] for item in plan)
        (workspace / 'dispatch.md').write_text(text + '\n')
    elif stage == 'P5':
        current['meta_data']['dispatch']['when'] = 'Select exactly for the independent host path named ' + node + '.'
        current['design']['math_reshape']['thesis'] = 'Copy scalar X without intermediates; node ' + node + ', revision ' + str(number) + '.'
        if number > 1:
            current['meta_data']['version'] += 1
        if action == 'bad_design':
            current['design']['math_reshape']['thesis'] = None
        if action == 'ancestor_write':
            with (workspace / 'root.yaml').open('a') as stream:
                stream.write('# unauthorized ancestor edit\n')
        if action == 'foreign_source':
            current['meta_data']['source'] = 'not-the-frozen-repository@' + '0' * 40 + ':missing.py::kernel'
    elif stage == 'P6':
        final_edit = scenario.get('final_edit_node')
        if final_edit:
            final_path = workspace / (final_edit + '.yaml')
            edited = yaml.safe_load(final_path.read_text())
            edited['design']['math_reshape']['thesis'] += ' Final source review correction.'
            edited['meta_data']['version'] += 1
            final_path.write_text(yaml.safe_dump(edited, sort_keys=False))
        else:
            with (workspace / 'dispatch.md').open('a') as stream:
                stream.write('<!-- P6 worker correction ' + str(number) + ' -->\n')
    if action == 'contract_write':
        current['contract']['algorithm_requirement']['compute'][0]['math'] = 'Y = 42.'
    if stage not in ('P4', 'P6'):
        (workspace / filename).write_text(yaml.safe_dump(current, sort_keys=False, allow_unicode=True))
if action == 'reviewer_write':
    with (workspace / filename).open('a') as stream:
        stream.write('# unauthorized reviewer edit\n')

def anchor(path, relative):
    lines = path.read_text().splitlines()
    index = next(i for i, line in enumerate(lines) if line.strip())
    return {'path': relative, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'line_start': index + 1, 'line_end': index + 1, 'quote': lines[index][:200],
            'note': 'Controlled scalar-copy evidence, read from the fixed bytes.'}

source_evidence = []
for category in ('implementation', 'host', 'io', 'input', 'reference', 'acceptance', 'scope'):
    binding = context['inputs']['entrypoint' if category == 'implementation' else category]
    entry = binding[0] if isinstance(binding, list) else binding
    source_evidence.append({'category': category, 'source_id': context['source_id'],
                            **anchor(Path(context['source_root']) / entry['path'], entry['path'])})
after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in workspace.iterdir() if p.is_file()}
before = record['before_hashes']
decision = 'complete' if role == 'worker' else 'approve'
if action in ('continue', 'revise', 'blocked'):
    decision = action
subjects = context['required_conclusions']
receipt = {'format': 'schema_extract.receipt.v2', 'identity': identity, 'decision': decision,
           'summary': 'Independent controlled ' + stage + ' ' + role + ' ' + mode + ': ' + action,
           'changed_files': sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name)),
           'artifacts': [{'path': name, 'sha256': value} for name, value in sorted(after.items())],
           'source_evidence': source_evidence,
           'artifact_evidence': [anchor(workspace / name, name) for name in sorted(after)] if role == 'reviewer' else [],
           'conclusions': [{'subject': subject, 'text': 'The scalar-copy fixture and host evidence support this conclusion.',
                            'evidence': [0, 1]} for subject in subjects],
           'checks': [], 'checks_note': 'This controlled model returns no checker claim; Runner must run its frozen checker.',
           'issues': [], 'framework_candidates': [],
           'handoff': '已核对 scalar copy 的固定源码。\n保持共享 Contract 与选定条件。\n按已审直接父核实下一节点。'
                       if stage in ('P2', 'P3', 'P4') and role == 'reviewer' else ''}
if decision in ('revise', 'blocked'):
    issue_scope = (scenario.get('final_edit_node') + '.yaml' if scenario.get('final_edit_node') else 'dispatch.md') if stage == 'P6' else filename
    receipt['issues'] = [{'kind': 'design', 'scope': issue_scope,
                          'detail': 'Make the recorded scalar-copy explanation match the cited host path.',
                          'recovery_condition': 'The current author must resolve this precise issue using the fixed host evidence.',
                          'evidence': [0, 1]}]
if stage == 'P4':
    receipt['plan'] = plan
if stage == 'P5':
    receipt['artifacts'] = [item for item in receipt['artifacts'] if item['path'] == filename]
if stage == 'P6':
    final_edit = scenario.get('final_edit_node')
    requested_scope = {'path': final_edit + '.yaml', 'fields': ['design.math_reshape.thesis']} if final_edit else {'path': 'dispatch.md', 'fields': ['*']}
    receipt['revision_scope'] = [requested_scope] if decision == 'revise' or role == 'worker' or mode == 'adjudicate' else []
    receipt['affected_nodes'] = sorted(set(context['pending_rechecks']) | ({final_edit or 'root'} if receipt['revision_scope'] else set()))
    if scenario.get('omit_final_descendants') and role == 'reviewer' and decision == 'approve' and mode == 'normal':
        receipt['affected_nodes'] = [final_edit]
if mode == 'adjudicate':
    action_name = 'blocked' if decision == 'blocked' else 'select'
    patches = []
    if action in ('patch', 'patch_bad_old', 'patch_bad_design'):
        action_name = 'patch'
        if stage == 'P6':
            patches = [{'path': 'dispatch.md', 'old': '<!-- independent-index -->',
                        'new': '<!-- independently-adjudicated-final-index -->'}]
        else:
            current = yaml.safe_load((workspace / filename).read_text())
            old = current['design']['math_reshape']['thesis']
            patches = [{'path': filename, 'old': old,
                        'new': 'A source-checked scalar-copy explanation selected by adjudication.'}]
        if action == 'patch_bad_old':
            patches[0]['old'] = 'THIS TEXT DOES NOT EXIST IN ANY ARTIFACT'
        if action == 'patch_bad_design':
            version = yaml.safe_load((workspace / filename).read_text())['meta_data']['version']
            patches = [{'path': filename, 'old': 'execution_mode: serial', 'new': 'execution_mode: invalid'},
                       {'path': filename, 'old': 'version: ' + str(version), 'new': 'version: ' + str(version + 1)}]
    selected_checkpoint = identity['base_checkpoint']
    if action == 'select_old':
        selected_checkpoint = next(c['context']['identity']['base_checkpoint'] for c in previous
                                   if c['stage'] == stage and c['node'] == node and c['role'] == 'reviewer'
                                   and c['mode'] == 'normal')
    receipt['adjudication'] = {'action': action_name, 'checkpoint': selected_checkpoint, 'patches': patches}
if action == 'bad_receipt':
    receipt.pop('summary')
if action == 'false_hash':
    receipt['artifacts'][0]['sha256'] = '0' * 64
if stage != 'P1' and mode == 'normal':
    receipt['reopen'] = None
output = Path(sys.argv[sys.argv.index('--output-last-message') + 1])
output.write_text(json.dumps(receipt))
print(json.dumps({'type': 'turn.completed', 'usage': {'input_tokens': 1, 'output_tokens': 1}}), flush=True)
'''


def plan_for(pairs=()):
    return [
        {"id": node, "file": node + ".yaml", "parent": parent,
         "workload": "Scalar-copy path " + node,
         "difference": "The fixed host selects the " + node + " scalar-copy path.",
         "selection": "Use the fixed kernel host conditions.", "evidence": [1]}
        for node, parent in [("root", ""), *pairs]
    ]


class IndependentFixture:
    def __init__(self, *, actions=None, pairs=(), plan=None, final_edit_node=None, omit_final_descendants=False):
        self.temporary = tempfile.TemporaryDirectory(prefix="ks-independent-c-")
        self.base = Path(self.temporary.name)
        self.source = self.base / "source"
        self.source.mkdir()
        sources = {
            "kernel.py": "def kernel(x):\n    if x < -1:\n        return nested(x)\n    if x < 0:\n        return negative(x)\n    return x\n\ndef negative(x):\n    return x\n\ndef nested(x):\n    return x\n",
            "reference.py": "def reference(x):\n    return x\n",
            "input.py": "def make_input():\n    return 3.0\n",
            "io.md": "X is one fp32 scalar; Y is the copied fp32 scalar.\n",
            "acceptance.md": "Every copied output must equal the reference exactly.\n",
            "scope.md": "The valid domain is every finite fp32 scalar.\n",
        }
        for name, contents in sources.items():
            (self.source / name).write_text(contents, encoding="utf-8")
        descriptor = {"format": "schema_extract.source.v2", "id": "independent_scalar_copy",
                      "entrypoint": "kernel.py::kernel", "host": "kernel.py::kernel",
                      "reference": "reference.py::reference", "input": "input.py::make_input",
                      "io": ["io.md"], "acceptance": ["acceptance.md"], "scope": ["scope.md"]}
        (self.source / "schema-inputs.json").write_text(json.dumps(descriptor), encoding="utf-8")
        self.fake = self.base / "codex"
        self.fake.write_text(CONTROLLED_CLI, encoding="utf-8")
        self.fake.chmod(0o755)
        self.scenario = {"actions": actions or {}, "plan": plan if plan is not None else plan_for(pairs),
                         "final_edit_node": final_edit_node, "omit_final_descendants": omit_final_descendants}
        (self.base / "scenario.json").write_text(json.dumps(self.scenario), encoding="utf-8")
        minimal = REPO / "schema_template/tools/tests/fixtures/minimal/root.yaml"
        (self.base / "complete.yaml").write_bytes(minimal.read_bytes())
        self.store = self.state = self.lock = None

    def create(self, **config):
        self.store, self.state, self.lock = create_run(
            REPO, self.base / "state", self.base / "output", source=str(self.source),
            config={"codex_binary": str(self.fake), "call_seconds": 10, "max_calls": 80, **config},
        )
        return self.state

    def run(self, **config):
        self.create(**config)
        self.state = run(self.store, self.state["run_id"])
        return self.state

    def calls(self, *, stage=None, node=None, role=None, mode=None):
        path = self.base / "calls.jsonl"
        values = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        return [value for value in values if all(expected is None or value[key] == expected
                                                for key, expected in (("stage", stage), ("node", node),
                                                                      ("role", role), ("mode", mode)))]

    def cleanup(self):
        for path in self.base.rglob("*"):
            if path.is_dir():
                path.chmod(0o755)
        self.temporary.cleanup()


class IndependentCTests(unittest.TestCase):
    def fixture(self, **kwargs):
        fixture = IndependentFixture(**kwargs)
        self.addCleanup(fixture.cleanup)
        return fixture

    def assert_ready(self, fixture, state):
        self.assertEqual(state["status"], "ready", state.get("reason"))
        artifacts, info = fixture.store.verify_checkpoint(state)
        checked = subprocess.run(
            [sys.executable, "-B", str(Path(fixture.lock["snapshot"]) / "rules/schema_template/check_yaml.py"),
             "--skip-hardware", str(artifacts / "root.yaml")],
            capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertEqual(info["files"], manifest(artifacts))
        scopes = [*state["stages"].values(), *state["nodes"].values()]
        for scope in scopes:
            approval = scope.get("approval")
            if not approval:
                continue
            approved_files, _ = fixture.store.verify_checkpoint(state, approval["checkpoint"])
            self.assertEqual(approval["input_fingerprint"], state["input_fingerprint"])
            for name, expected in approval["artifact_digests"].items():
                self.assertEqual(hashlib.sha256((approved_files / name).read_bytes()).hexdigest(), expected,
                                 "approval is not bound to its actual checkpoint content")
        for node in state["nodes"].values():
            self.assertEqual(node["approved_sha256"], hashlib.sha256((artifacts / node["file"]).read_bytes()).hexdigest())
        for call in fixture.calls():
            identity = call["context"]["identity"]
            dependency_files, _ = fixture.store.verify_checkpoint(state, identity["base_checkpoint"])
            actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dependency_files.iterdir() if p.is_file()}
            self.assertEqual(identity["dependency_digests"], actual,
                             "invocation is not bound to its actual fixed dependencies")
        return artifacts

    def test_single_root_starts_new_roles_and_finishes_with_reviewer_only(self):
        fixture = self.fixture()
        state = fixture.run(worker_effort="high", reviewer_effort="xhigh")
        artifacts = self.assert_ready(fixture, state)
        calls = fixture.calls()
        self.assertEqual([(c["stage"], c["role"]) for c in calls],
                         [("P1", "worker"), ("P1", "reviewer"), ("P2", "worker"), ("P2", "reviewer"),
                          ("P3", "worker"), ("P3", "reviewer"), ("P4", "worker"), ("P4", "reviewer"),
                          ("P6", "reviewer")])
        self.assertEqual(calls[2]["resumed"], calls[0]["session_id"])
        self.assertEqual(calls[3]["resumed"], calls[1]["session_id"])
        fresh = [c for c in calls if c["stage"] != "P2"]
        self.assertTrue(all(c["resumed"] is None for c in fresh))
        self.assertEqual(len({c["session_id"] for c in fresh}), len(fresh))
        self.assertEqual(set(manifest(artifacts)), {"root.yaml", "dispatch.md"})
        for c in calls:
            argv = c["argv"]
            expected_sandbox = "workspace-write" if c["role"] == "worker" else "read-only"
            self.assertEqual(argv[argv.index("-s") + 1], expected_sandbox)
            self.assertNotIn("--ignore-user-config", argv)
            self.assertFalse(any("model_provider" in item for item in argv))
            expected_effort = 'model_reasoning_effort="high"' if c["role"] == "worker" else 'model_reasoning_effort="xhigh"'
            self.assertIn(expected_effort, argv)

    def test_multilevel_seed_copies_the_reviewed_direct_parent(self):
        fixture = self.fixture(pairs=(("negative", "root"), ("nested", "negative"), ("other", "root")),
                               actions={"P5.negative.reviewer.normal.1": "revise"})
        state = fixture.run()
        self.assert_ready(fixture, state)
        calls = fixture.calls()
        first_worker = {node: fixture.calls(stage="P5", node=node, role="worker")[0]
                        for node in ("negative", "nested", "other")}
        for child, parent in (("negative", "root"), ("nested", "negative"), ("other", "root")):
            with self.subTest(child=child):
                call = first_worker[child]
                seed = call["before"][child + ".yaml"]
                self.assertNotIn("contract", seed)
                self.assertEqual(seed["meta_data"]["dispatch"]["id"], child)
                self.assertEqual(seed["meta_data"]["dispatch"]["parent"], parent)
                self.assertEqual(seed["meta_data"]["version"], 1)
                self.assertEqual(seed["design"], call["before"][parent + ".yaml"]["design"])
                parent_reviews = [c for c in calls[:calls.index(call)] if c["role"] == "reviewer" and
                                  (c["stage"] == "P3" if parent == "root" else c["stage"] == "P5" and c["node"] == parent)]
                self.assertTrue(parent_reviews, "child started before its parent was reviewed")
                self.assertEqual(call["before_hashes"][parent + ".yaml"],
                                 parent_reviews[-1]["before_hashes"][parent + ".yaml"])
        nested_seed = first_worker["nested"]["before"]["nested.yaml"]
        self.assertIn("negative, revision 2", nested_seed["design"]["math_reshape"]["thesis"])
        first_pairs = []
        for node in first_worker:
            first_pairs.extend(fixture.calls(stage="P5", node=node, role=role)[0] for role in ("worker", "reviewer"))
        self.assertTrue(all(c["resumed"] is None for c in first_pairs))
        self.assertEqual(len({c["session_id"] for c in first_pairs}), len(first_pairs))
        parent_calls = fixture.calls(stage="P5", node="negative")
        self.assertEqual(parent_calls[2]["resumed"], parent_calls[0]["session_id"])
        self.assertEqual(parent_calls[3]["resumed"], parent_calls[1]["session_id"])

    def test_p3_cannot_reuse_a_session_from_p1(self):
        fixture = self.fixture(actions={"P3.root.worker.normal.1": "reuse_first_session"})
        state = fixture.run()
        self.assertNotEqual(state["status"], "ready")
        self.assertFalse(fixture.calls(stage="P4"))
        self.assertIsNone(state["stages"]["P3"]["approval"])

    def test_plan_rejects_ambiguous_roots_missing_parents_and_cycles(self):
        root, child = plan_for((("negative", "root"),))
        cases = {
            "missing root": [child],
            "duplicate ID": [root, child, deepcopy(child)],
            "second root": [root, {**child, "parent": ""}],
            "missing parent": [root, {**child, "parent": "missing"}],
            "unreachable cycle": [root, {**child, "parent": "nested"},
                                  {**child, "id": "nested", "file": "nested.yaml", "parent": "negative"}],
            "filename identity": [root, {**child, "file": "different.yaml"}],
            "path escape": [root, {**child, "file": "../negative.yaml"}],
        }
        for name, plan in cases.items():
            with self.subTest(case=name):
                fixture = self.fixture(plan=plan)
                state = fixture.run(max_calls=16)
                self.assertNotEqual(state["status"], "ready")
                self.assertFalse(fixture.calls(stage="P5"))
                self.assertFalse(fixture.calls(stage="P6"))
                self.assertIsNone(state["stages"]["P4"]["approval"])

    def test_checker_one_returns_to_same_author_without_a_substantive_revision(self):
        fixture = self.fixture(actions={"P3.root.worker.normal.1": "bad_design"})
        state = fixture.run()
        self.assert_ready(fixture, state)
        workers = fixture.calls(stage="P3", role="worker")
        self.assertEqual(len(workers), 2)
        self.assertEqual(workers[1]["resumed"], workers[0]["session_id"])
        self.assertEqual(len(fixture.calls(stage="P3", role="reviewer")), 1)
        scope = state["stages"]["P3"]
        self.assertEqual(scope["check_failures"], 1)
        self.assertEqual(scope["revisions"], 0)
        self.assertEqual(scope["tool_failures"], 0)
        invalid = subprocess.run(
            [sys.executable, "-B", str(Path(fixture.lock["snapshot"]) / "rules/schema_template/check_yaml.py"),
             "--skip-hardware", str(Path(workers[0]["context"]["workspace"]) / "root.yaml")],
            capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(invalid.returncode, 1, invalid.stderr)

    def test_checker_two_is_persisted_as_a_tool_failure_without_review(self):
        fixture = self.fixture()
        fixture.create()
        original_run = subprocess.run
        observed = []

        def checker_invocation_error(command, *args, **kwargs):
            if isinstance(command, list) and any(str(part).endswith("schema_template/check_yaml.py") for part in command):
                result = original_run([*command, "--independent-invalid-option"], *args, **kwargs)
                observed.append(result.returncode)
                return result
            return original_run(command, *args, **kwargs)

        with patch("schema_extract.artifacts.subprocess.run", side_effect=checker_invocation_error):
            state = run(fixture.store, fixture.state["run_id"])
        self.assertEqual(observed, [2])
        self.assertNotEqual(state["status"], "ready")
        self.assertFalse(fixture.calls(stage="P3", role="reviewer"))
        scope = state["stages"]["P3"]
        self.assertEqual(scope["tool_failures"], 1)
        self.assertEqual(scope["check_failures"], 0)
        self.assertEqual(scope["revisions"], 0)
        self.assertTrue(state["reason"])

    def test_continue_receipt_repair_and_cli_failure_have_separate_persistent_counts(self):
        fixture = self.fixture(actions={"P3.root.worker.normal.1": "continue",
                                        "P3.root.worker.normal.2": "bad_receipt",
                                        "P3.root.worker.normal.3": "cli_failure"})
        state = fixture.run()
        self.assert_ready(fixture, state)
        workers = fixture.calls(stage="P3", role="worker")
        self.assertEqual(len(workers), 4)
        self.assertTrue(all(c["session_id"] == workers[0]["session_id"] for c in workers))
        self.assertTrue(all(c["resumed"] == workers[0]["session_id"] for c in workers[1:]))
        scope = state["stages"]["P3"]
        self.assertEqual(scope["continues"], 1)
        self.assertEqual(scope["receipt_repairs"], 1)
        self.assertEqual(scope["cli_failures"], 1)
        self.assertEqual(scope["revisions"], 0)
        self.assertEqual(fixture.store.load(state["run_id"])["stages"]["P3"], scope)
        repaired_artifact = Path(workers[1]["context"]["workspace"]) / "root.yaml"
        self.assertEqual(workers[2]["before_hashes"]["root.yaml"], hashlib.sha256(repaired_artifact.read_bytes()).hexdigest(),
                         "receipt repair discarded the author's valid draft")

    def test_normal_calls_reserve_adjudication_time_and_reentry_does_not_reset_budget(self):
        for draft_only in (True, False):
            with self.subTest(draft_only=draft_only):
                fixture = self.fixture(actions={"P3.root.worker.normal.*": "continue"} if draft_only else {})
                fixture.create(stage_seconds=20, operator_seconds=300, call_seconds=8)
                reserve = fixture.lock["config"]["adjudication_seconds"]
                self.assertGreater(reserve, 0)
                current = [fixture.state["created_at"]]
                observed, checks = [], []
                cli, original_run = CodexCLI(str(fixture.fake)), subprocess.run

                class AdvancingClockCLI:
                    def run(self, *, attempt, timeout, **kwargs):
                        saved = fixture.store.load(attempt["run_id"])
                        scope = saved["stages"][attempt["stage"]]
                        observed.append((attempt["mode"], timeout,
                                         min(saved["deadline"], scope["deadline"]) - current[0]))
                        result = cli.run(attempt=attempt, timeout=timeout, **kwargs)
                        if attempt["stage"] == "P3":
                            current[0] += timeout
                        return result

                def observe_checker(command, *args, **kwargs):
                    if isinstance(command, list) and any(str(item).endswith("schema_template/check_yaml.py") for item in command):
                        saved = fixture.store.load(fixture.state["run_id"])
                        remaining = min(saved["deadline"], saved["stages"][saved["stage"]]["deadline"]) - current[0]
                        checks.append((kwargs["timeout"], remaining))
                    return original_run(command, *args, **kwargs)

                with patch("schema_extract.runner.time.time", side_effect=lambda: current[0]), \
                        patch("schema_extract.artifacts.subprocess.run", side_effect=observe_checker):
                    state = run(fixture.store, fixture.state["run_id"], transport=AdvancingClockCLI())
                    self.assertEqual(state["status"], "budget_exhausted", state.get("reason"))
                    self.assertEqual(state["stage"], "P3")
                    self.assertEqual(state["stages"]["P3"]["revisions"], 0)
                    preserved, count = deepcopy(state), len(observed)
                    repeated = run(fixture.store, state["run_id"], transport=AdvancingClockCLI())
                self.assertEqual(len(observed), count)
                self.assertEqual(repeated, preserved)
                for mode, timeout, remaining in observed:
                    if mode == "normal":
                        self.assertLessEqual(timeout, remaining - reserve + 1e-6,
                                             "a normal CLI call consumed the adjudicator's reserved time")
                for timeout, remaining in checks:
                    self.assertLessEqual(timeout, remaining - reserve + 1e-6,
                                         "a normal checker call consumed the adjudicator's reserved time")
                if draft_only:
                    self.assertGreater(state["stages"]["P3"]["continues"], 0)
                else:
                    self.assertTrue(checks, "the valid Design was never checked")
                self.assertEqual(state["deadline"], fixture.state["deadline"])
                self.assertIsNotNone(state["stages"]["P3"]["deadline"])

    def test_new_stage_and_node_permissions_reject_foreign_changes_or_sources(self):
        cases = (("P3.root.worker.normal.1", "contract_write", "P3"),
                 ("P3.root.reviewer.normal.1", "reviewer_write", "P3"),
                 ("P5.negative.worker.normal.1", "ancestor_write", "P5"),
                 ("P5.negative.worker.normal.1", "foreign_source", "P5"))
        for key, action, stage in cases:
            with self.subTest(action=action):
                fixture = self.fixture(pairs=(("negative", "root"),), actions={key: action})
                state = fixture.run(max_calls=20)
                self.assertNotEqual(state["status"], "ready")
                self.assertEqual(state["stage"], stage)
                self.assertFalse(fixture.calls(stage="P6"))

    def test_fifth_substantive_revision_uses_original_reviewer_once(self):
        actions = {"P3.root.reviewer.normal.*": "revise"}
        fixture = self.fixture(actions=actions)
        state = fixture.run()
        self.assert_ready(fixture, state)
        reviews = fixture.calls(stage="P3", role="reviewer", mode="normal")
        adjudications = fixture.calls(stage="P3", role="reviewer", mode="adjudicate")
        self.assertEqual(len(reviews), 5)
        self.assertEqual(len(adjudications), 1)
        self.assertEqual(adjudications[0]["resumed"], reviews[0]["session_id"])
        self.assertEqual(state["stages"]["P3"]["revisions"], 5)
        self.assertEqual(len(fixture.calls(stage="P3", role="worker")), 5)

    def test_selecting_an_old_reviewed_draft_rechecks_it_without_rewinding_version_or_budget(self):
        fixture = self.fixture(actions={"P3.root.reviewer.normal.*": "revise",
                                        "P3.root.reviewer.adjudicate.1": "select_old"})
        state = fixture.run()
        artifacts = self.assert_ready(fixture, state)
        root = yaml.safe_load((artifacts / "root.yaml").read_text())
        adjudication = fixture.calls(stage="P3", mode="adjudicate")[0]
        self.assertIn("root revision 1", root["design"]["math_reshape"]["thesis"])
        self.assertGreater(root["meta_data"]["version"], adjudication["before"]["root.yaml"]["meta_data"]["version"])
        self.assertEqual(state["stages"]["P3"]["revisions"], 5)
        self.assertEqual(state["deadline"], state["created_at"] + state["config"]["operator_seconds"])
        self.assertEqual(len(fixture.calls(stage="P3", mode="adjudicate")), 1)

    def test_adjudication_external_gap_or_failed_patch_blocks_without_retry(self):
        for action in ("blocked", "patch_bad_old", "patch_bad_design"):
            with self.subTest(action=action):
                fixture = self.fixture(actions={"P3.root.reviewer.normal.*": "revise",
                                                "P3.root.reviewer.adjudicate.1": action})
                state = fixture.run()
                self.assertEqual(state["status"], "blocked", state.get("reason"))
                self.assertEqual(state["stages"]["P3"]["revisions"], 5)
                self.assertEqual(len(fixture.calls(mode="adjudicate")), 1)
                self.assertFalse(fixture.calls(stage="P4"))
                self.assertIsNotNone(state["checkpoint"])
                self.assertTrue(state["reason"])
                if action == "patch_bad_design":
                    self.assertEqual(state["stages"]["P3"]["check_failures"], 1)

    def test_p6_adjudicator_patch_gets_exactly_one_fresh_read_only_signoff(self):
        fixture = self.fixture(actions={"P6.root.reviewer.normal.*": "revise",
                                        "P6.root.reviewer.adjudicate.1": "patch"})
        state = fixture.run()
        artifacts = self.assert_ready(fixture, state)
        calls = fixture.calls(stage="P6")
        self.assertEqual(calls[0]["role"], "reviewer")
        adjudications = fixture.calls(stage="P6", mode="adjudicate")
        signoffs = fixture.calls(stage="P6", mode="signoff")
        self.assertEqual(len(adjudications), 1)
        self.assertEqual(len(signoffs), 1)
        self.assertEqual(adjudications[0]["resumed"], calls[0]["session_id"])
        self.assertIsNone(signoffs[0]["resumed"])
        self.assertNotIn(signoffs[0]["session_id"], {c["session_id"] for c in calls if c["mode"] != "signoff"})
        self.assertIn("independently-adjudicated-final-index", (artifacts / "dispatch.md").read_text())
        self.assertEqual(signoffs[0]["before_hashes"], manifest(artifacts))
        for c in [*adjudications, *signoffs]:
            self.assertEqual(c["argv"][c["argv"].index("-s") + 1], "read-only")

    def test_p6_parent_revision_requires_descendant_rechecks_before_ready(self):
        for omit_descendants in (True, False):
            with self.subTest(omit_descendants=omit_descendants):
                fixture = self.fixture(pairs=(("negative", "root"), ("nested", "negative")),
                                       actions={"P6.root.reviewer.normal.1": "revise"},
                                       final_edit_node="negative", omit_final_descendants=omit_descendants)
                state = fixture.run()
                final_review = fixture.calls(stage="P6", role="reviewer")[-1]
                self.assertEqual(set(final_review["context"]["pending_rechecks"]), {"negative", "nested"})
                if omit_descendants:
                    self.assertNotEqual(state["status"], "ready")
                    self.assertIsNone(state["stages"]["P6"]["approval"])
                else:
                    artifacts = self.assert_ready(fixture, state)
                    nested_review = state["nodes"]["nested"]["approval"]
                    self.assertEqual(nested_review["attempt_id"], state["stages"]["P6"]["approval"]["attempt_id"])
                    self.assertEqual(nested_review["artifact_digests"]["negative.yaml"],
                                     hashlib.sha256((artifacts / "negative.yaml").read_bytes()).hexdigest())

    def test_signoff_cannot_open_another_revision_or_transport_retry_loop(self):
        for action in ("revise", "cli_failure"):
            with self.subTest(action=action):
                fixture = self.fixture(actions={"P6.root.reviewer.normal.*": "revise",
                                                "P6.root.reviewer.adjudicate.1": "patch",
                                                "P6.root.reviewer.signoff.1": action})
                state = fixture.run()
                self.assertEqual(state["status"], "blocked", state.get("reason"))
                self.assertEqual(len(fixture.calls(mode="signoff")), 1)
                self.assertEqual(fixture.calls()[-1]["mode"], "signoff")
                self.assertEqual(state["stages"]["P6"]["revisions"], 5)
                adjudication = fixture.calls(mode="adjudicate")[0]
                scope = state["stages"]["P6"]
                self.assertEqual(scope["adjudication"]["attempt_id"], adjudication["context"]["identity"]["attempt_id"])
                self.assertEqual(scope["adjudication"]["decision"], "approve")
                self.assertEqual(scope["signoff"]["decision"], "blocked")


if __name__ == "__main__":
    unittest.main()
