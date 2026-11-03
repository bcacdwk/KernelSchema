"""Current frozen phase page, fixed handoffs, and one invocation's locating context."""

from __future__ import annotations

import json
from pathlib import Path
import shlex

from .artifacts import _same, load_root
from .receipts import call_identity, required_artifact_evidence
from .stages import STAGES, owner_for, scope_for
from .common import read_json


def build_prompt(state: dict, lock: dict, attempt: dict) -> str:
    definition = STAGES[attempt["stage"]]
    root = Path(lock["snapshot"])
    page_path = root / "rules/schema_extract/phases" / definition["page"]
    bindings = lock["source"]["bindings"]
    def source_reference(key):
        entry = bindings[key]
        symbol = "::" + entry["symbol"] if entry["symbol"] else ""
        return f"{lock['source']['repository_id']}@{lock['source']['ref']}:{entry['path']}{symbol}"
    node_entry = next((entry for entry in state.get("plan") or [] if entry["id"] == attempt["node"]), None)
    run_dir = Path(attempt["directory"]).parent.parent
    prior_stage = {"P3": "P2", "P4": "P3", "P5": "P4"}.get(attempt["stage"])
    handoff_path = run_dir / "stages" / (prior_stage or "") / "handoff.md"
    handoff = handoff_path.read_text(encoding="utf-8") if prior_stage and handoff_path.is_file() else ""
    scope = scope_for(state, attempt)
    plan_receipt, node_evidence = None, []
    if attempt["stage"] == "P5":
        approved_plan = state["stages"]["P4"]["approval"]
        plan_receipt = run_dir / "attempts" / approved_plan["attempt_id"] / "receipt.json"
        approved_evidence = read_json(plan_receipt)["source_evidence"]
        node_evidence = [approved_evidence[index] for index in node_entry["evidence"]]
    first_draft = attempt["stage"] == "P5" and not state["nodes"][attempt["node"]].get("authored", False)
    if attempt["stage"] == "P1":
        rules = root / "rules"
        baseline = run_dir / "checkpoints" / attempt["base_checkpoint"] / "artifacts"
        first_draft = _same(load_root(baseline, parser_root=rules),
                            load_root(rules / "schema_template", "starter.yaml", parser_root=rules))
    locating = {"identity": call_identity(attempt), "mode": attempt.get("mode", "normal"),
                "source_id": lock["source"]["id"], "source_ref": lock["source"]["ref"],
                "source_root": str(root / "source"), "rules_id": lock["rules"]["id"],
                "rules_root": str(root / "rules"), "phase_page": str(page_path), "workspace": attempt["workspace"],
                "inputs": {key: bindings[key] for key in ("entrypoint", "host", "io", "input", "reference", "acceptance", "scope")},
                "implementation_source": source_reference("entrypoint"), "reference_source": source_reference("reference"),
                "writable_files": attempt["writable_files"], "revision_scope": state["revision_scope"],
                "read_specs": definition["reads"], "feedback": owner_for(state, attempt)["feedback"], "handoff": handoff,
                "required_conclusions": definition["conclusions"], "required_evidence": definition["evidence"],
                "required_artifact_evidence": sorted(required_artifact_evidence(state, attempt, attempt["dependency_digests"])),
                "node_entry": node_entry, "plan": state.get("plan") or state.get("plan_draft"),
                "plan_receipt": str(plan_receipt) if plan_receipt else None, "node_evidence": node_evidence,
                "reopening": state.get("reopening"),
                "recovery_handoff": attempt.get("recovery_handoff"),
                "imported_draft": state.get("imported_draft"),
                "first_draft": first_draft,
                "pending_rechecks": state["pending_rechecks"],
                "prior_approvals": {key: value["approval"] for key, value in state["stages"].items() if value["approval"]},
                "node_approvals": {key: value["approval"] for key, value in state["nodes"].items() if value.get("approval")},
                "reviewed_artifacts": {key: str(run_dir / "checkpoints" / value["approval"]["checkpoint"] / "artifacts" / value["file"])
                                       for key, value in state["nodes"].items() if value.get("approval")},
                "budget": {key: scope[key] for key in ("revisions", "continues", "receipt_repairs", "cli_failures", "calls", "deadline")}}
    if attempt.get("mode") == "adjudicate":
        locating["candidates"] = [{**entry, "artifacts": str(run_dir / "checkpoints" / entry["checkpoint"] / "artifacts")}
                                  for entry in scope["candidates"]]
    reader = root / "rules/schema_extract/reading.py"
    example = f"python3 -B {shlex.quote(str(reader))} --root {shlex.quote(str(root / 'source'))} --path {shlex.quote(bindings['host']['path'])} --start-line 1"
    checker = root / "rules/schema_template/check_yaml.py"
    target = node_entry["file"] if attempt["stage"] == "P5" else "root.yaml"
    check_command = f"python3 -B {shlex.quote(str(checker))} --skip-hardware {shlex.quote(str(Path(attempt['workspace']) / target))}"
    stage_requirements = (f"{attempt['stage']} complete/approve 必须引用 {', '.join(definition['evidence'])}；"
                          f"conclusions 包含 {', '.join(definition['conclusions'])}。")
    if attempt["stage"] == "P1":
        stage_requirements += " first_draft按起始checkpoint与starter的实际内容判断；true时本次提交的root初版version必须为1，同一调用内先复制再填写不算第二版。false时沿起始checkpoint版本修订，内容改变则递增。"
    if attempt["stage"] in {"P2", "P3", "P4"} and attempt["role"] == "reviewer":
        stage_requirements += " approve 的 handoff 最多20个非空行、1200字符。"
    if attempt["stage"] == "P3" and state.get("reopening"):
        stage_requirements += " 本轮由Runner明确重开P3，可复核并修订root的design/realization/meta_data.dispatch.when，内容变更仍提高version；Contract、source和dispatch id/parent继续只读，改选代表路径需回P2。"
    if attempt["stage"] == "P4":
        stage_requirements += " plan 每项为 id/file/parent/workload/difference/selection/evidence；root 为 root/root.yaml/空parent。派生file=<id>.yaml，evidence引用本回执的host来源索引。两角色complete/approve返回相同节点语义清单，source_evidence顺序与evidence编号各自独立。dispatch.md链接集合必须与plan一致。"
    if attempt["stage"] == "P5":
        stage_requirements += " artifacts 只列本节点YAML；artifact_evidence可引用只读root/直接父与本节点。node_entry.evidence只属于plan_receipt，已解析的P4来源锚点在node_evidence；本次证据独立编号。first_draft=true时首次稿version=1；已有ID重开或返工沿本次baseline版本递增，不回1。节点自己的when不能为空，共享内容也要核实。"
    if attempt["stage"] == "P6":
        stage_requirements += " P6回执含revision_scope:[{path,fields:[点分字段路径]}]和affected_nodes:[节点ID]。需退修必须列明限域；dispatch.md用fields:[\"*\"]，派生YAML可指定design/realization/meta_data.source/meta_data.dispatch.when下的具体字段。root/Contract/树变更需通过reopen请求上游复核。worker原样保留授权revision_scope；最终approve覆盖pending_rechecks全部节点。"
    mode = attempt.get("mode", "normal")
    mode_instructions = "worker只能complete/continue/blocked；reviewer只能approve/revise/blocked。continue保存草稿，不能算放行。"
    mode_instructions += " worker的blocked是请求独立审阅：合法草稿及具体缺项交本阶段reviewer判断approve/revise/blocked，不自动结束任务。reviewer独立判断已知事实、缺项的实际影响和可继续的工作；未完成草稿可revise退回撰写，complete/approve仍须全部阶段证据和必需检查通过，真正基础缺项仍blocked。"
    if attempt["stage"] != "P1" and mode == "normal":
        mode_instructions += " 如固定源码新发现推翻上游决定，两角色均可decision=reopen并提供reopen:{issue_id,target,reason,evidence:[source_evidence索引]}与issues具体恢复条件。普通回执reopen=null。target仅指上游：P2→P1；P3→P2/P1；P4→P3/P2/P1；P5/P6→P4/P3/P2/P1。树/选父变更回P4，代表路径改选回P2，现有root Design/when回P3，共享Contract回P1。外部任务要求改变使用target=external，等待任务方明确确认，禁止擅改。新源码锚点可触发重开；已读源行的转录错误用issues.kind=transcription、scope定位产物，并以artifact_evidence精确引用当前错误字段。相同issue_id与固定源码范围只重开一次，换note/短引文不能重新计次，原预算不重置。"
    if mode == "adjudicate":
        mode_instructions = """本轮为原reviewer的一次裁决，只有approve或blocked两个出口；不得直接改workspace。
额外adjudication={action:select|patch|blocked,checkpoint:候选checkpoint,patches:[{path,old,new}]}。
select选择已列出的草稿；patch选择当前checkpoint并给出限于当前阶段/已授权字段的唯一匹配文本替换，old必须非空且仅出现一次。
Runner在隔离副本应用、按需提升恢复旧稿的版本并运行门禁；缺事实、补丁失败或检查未通过都blocked，不能请求再裁决。
artifacts/changed_files/artifact_evidence仍绑定你实际读取的原workspace（changed_files=[]），补丁后的摘要由Runner另存。
P6仅当你亲自patch最终交付时另开独立reviewer签收一次；不自行请求额外返工。"""
    elif mode == "signoff":
        mode_instructions = "本轮为P6裁决补丁后的全新独立reviewer签收。只允许approve或blocked，检查最终改动及全部pending_rechecks；本次不再返工、裁决或重试。"
    checking = ("P1/P2/P4按本阶段产物门禁，不要求尚未完成或未生成的Design通过完整检查。"
                if attempt["stage"] in {"P1", "P2", "P4"} else
                f"完成的YAML必须运行冻结检查器：{check_command}\n退出0才满足完成门槛；1交当前修改者，2报工具/环境问题。Runner还会独立运行，不采信声明代替检查。")
    return f"""你是独立的 {attempt['role']}，只执行 {attempt['stage']} / {attempt['node']} / {mode}。
Runner是运行状态的唯一写入者。只允许修改writable_files中本阶段允许的字段；reviewer只读。
全部材料已固定。只读source_root/rules_root、本次workspace、列出的plan_receipt/reviewed_artifacts、imported_draft.artifacts旧稿及明确给出的candidate副本；不要访问原工作树、修改运行记录、开子代理或GPU测试。
每次先读本页，再按页指定范围读取规范和源码；本页明确引用的方法章节可以读取。不要读取PLAN。
长文件按行读取；有next_cursor时用相同范围和--cursor继续，不丢弃未读的行：
{example}
所有摘要是完整文件的SHA-256；计算产物摘要可用sha256sum。source_evidence.source_id等于调用source_id。
来源/产物证据给出相对path、sha256、1起始行号、最多12行范围、不超过320字符的精确短引文；note说明支持的字段/决定。
source_evidence.category使用implementation/host/io/input/reference/acceptance/scope。证据索引从0开始；每类至少引用对应绑定入口，调用链固定文件可补充。
{stage_requirements}
artifacts是交付文件摘要清单，changed_files是实际变更清单，artifact_evidence是支持审阅结论的文件/行号/原文锚点。
reviewer的artifact_evidence必须覆盖调用required_artifact_evidence列出的每个具体文件，不只引用可写文件。
最后按output-schema返回JSON，不写额外回执文件。原样回传identity；changed_files只列实际改动文件（字典序）。
artifacts绑定本次完整交付视图：除P5只列本节点YAML外，必须列workspace中全部交付文件及其SHA-256，包括只读、未修改的root.yaml、dispatch.md和已审节点；不能只列改动文件。
{mode_instructions}
若有recovery_handoff，本次旧调用已经确认终止；以本页、当前checkpoint和固定来源建立新上下文，不沿用已中断会话。保留原退修、调用及截止预算。
若有imported_draft，旧产物和来源只是只读线索；旧批准无效，必须按当前阶段独立核对。新回执证据仍指向本次source_root。
缺项和退修给出issues范围、证据及恢复条件。checks只记真实执行命令；未执行用[]并填写checks_note。framework_candidates没有建议时为[]。
framework_candidates的id使用与运行无关的小写字母数字及-或_稳定短名，scope定位共享规则或工具，summary说明必要的最小改动；算子自身缺口写issues。不要直接写台账或为可选候选改变任务结论。
{checking}

<invocation>
{json.dumps(locating, ensure_ascii=False, indent=2)}
</invocation>

<current-phase-page path={json.dumps(str(page_path))}>
{page_path.read_text(encoding='utf-8')}
</current-phase-page>
"""
