"""Current frozen phase page, fixed handoffs, and one invocation's locating context."""

from __future__ import annotations

import json
from pathlib import Path
import shlex

from .artifacts import _same, load_root
from .budget import call_limit
from .receipts import REOPEN_TARGETS, required_artifact_evidence
from .stages import STAGES, owner_for, read_specs, required_conclusions, scope_for
from .common import read_json, rules_root, source_root
from .snapshots import frozen_reader


def build_prompt(state: dict, lock: dict, attempt: dict) -> str:
    definition = STAGES[attempt["stage"]]
    rules, source = rules_root(lock), source_root(lock)
    page_path = rules / "schema_extract/phases" / definition["page"]
    bindings = lock["source"]["bindings"]
    def source_reference(key):
        entry = bindings[key]
        symbol = "::" + entry["symbol"] if entry["symbol"] else ""
        return f"{lock['source']['repository_id']}@{lock['source']['ref']}:{entry['path']}{symbol}"
    node_entry = next((entry for entry in state.get("plan") or [] if entry["id"] == attempt["node"]), None)
    run_dir = Path(attempt["directory"]).parent.parent
    prior_stage = {"P3": "P2", "P4": "P3", "P5": "P4"}.get(attempt["stage"])
    handoff_path = run_dir / "stages" / (prior_stage or "") / "handoff.md"
    handoff = (handoff_path.read_text(encoding="utf-8")
               if attempt["role"] == "worker" and prior_stage and handoff_path.is_file() else "")
    scope = scope_for(state, attempt)
    plan_receipt, node_evidence = None, []
    relevant_nodes = set(state["nodes"])
    ancestors = []
    if attempt["stage"] == "P5":
        if attempt["role"] == "worker":
            approved_plan = state["stages"]["P4"]["approval"]
            plan_receipt = run_dir / "attempts" / approved_plan["attempt_id"] / "receipt.json"
            approved_evidence = read_json(plan_receipt)["source_evidence"]
            node_evidence = [approved_evidence[index] for index in node_entry["evidence"]]
        relevant_nodes = {attempt["node"]}
        parent = node_entry["parent"]
        while parent:
            relevant_nodes.add(parent)
            ancestors.append(next(entry for entry in state["plan"] if entry["id"] == parent))
            parent = state["nodes"][parent]["parent"]
    def approval_reference(value):
        return {"checkpoint": value["checkpoint"], "receipt_sha256": value["receipt_sha256"],
                "receipt": str(run_dir / "attempts" / value["attempt_id"] / "receipt.json")}
    first_draft = attempt["stage"] == "P5" and not state["nodes"][attempt["node"]].get("authored", False)
    if attempt["stage"] == "P1":
        baseline = run_dir / "checkpoints" / attempt["base_checkpoint"] / "artifacts"
        first_draft = _same(load_root(baseline, parser_root=rules),
                            load_root(rules / "schema_template", "starter.yaml", parser_root=rules))
    locating = {"attempt_id": attempt["attempt_id"], "stage": attempt["stage"], "role": attempt["role"],
                "node": attempt["node"], "mode": attempt.get("mode", "normal"),
                "base_checkpoint": attempt["base_checkpoint"],
                "source_id": lock["source"]["id"], "source_ref": lock["source"]["ref"],
                "source_root": str(source), "rules_id": lock["rules"]["id"],
                "rules_root": str(rules), "phase_page": str(page_path), "workspace": attempt["workspace"],
                "inputs": {key: bindings[key] for key in ("entrypoint", "host", "io", "input", "reference", "acceptance", "scope")},
                "implementation_source": source_reference("entrypoint"), "reference_source": source_reference("reference"),
                "writable_files": attempt["writable_files"], "revision_scope": state["revision_scope"],
                "read_specs": read_specs(attempt), "feedback": owner_for(state, attempt)["feedback"], "handoff": handoff,
                "required_conclusions": required_conclusions(attempt), "required_evidence": definition["evidence"],
                "required_artifact_evidence": sorted(required_artifact_evidence(state, attempt, attempt["dependency_digests"])),
                "node_entry": node_entry, "plan": state.get("plan") or state.get("plan_draft"),
                "plan_receipt": str(plan_receipt) if plan_receipt else None, "node_evidence": node_evidence,
                "reopening": state.get("reopening"),
                "recovery_handoff": attempt.get("recovery_handoff"),
                "imported_draft": state.get("imported_draft"),
                "first_draft": first_draft,
                "pending_rechecks": state["pending_rechecks"],
                "prior_approvals": {key: approval_reference(value["approval"]) for key, value in state["stages"].items()
                                    if value["approval"] and (attempt["stage"] != "P5" or key == "P4")},
                "node_approvals": {key: approval_reference(value["approval"]) for key, value in state["nodes"].items()
                                   if key in relevant_nodes and value.get("approval")},
                "reviewed_artifacts": {key: str(run_dir / "checkpoints" / value["approval"]["checkpoint"] / "artifacts" / value["file"])
                                       for key, value in state["nodes"].items() if key in relevant_nodes and value.get("approval")},
                "budget": {**{key: scope[key] for key in ("revisions", "continues", "receipt_repairs", "cli_failures", "calls", "deadline")},
                           "call_seconds": attempt.get("timeout_seconds"), "max_calls": call_limit(state),
                           "revision_limit": state["config"]["revision_limit"]}}
    if attempt["stage"] == "P5":
        locating.pop("plan")
        locating["ancestor_entries"] = ancestors
        locating["revision_scope"] = [item for item in state["revision_scope"] if item["path"] == node_entry["file"]]
        locating["pending_rechecks"] = [node for node in state["pending_rechecks"] if node in relevant_nodes]
        # P4 has already handed off the relevant source/plan decision. Its old
        # whole-tree history and an imported legacy tree are not branch input.
        locating.pop("reopening")
        locating.pop("imported_draft")
    continuation = state.get("continuation")
    if continuation and attempt["role"] == "worker" and attempt["stage"] != "P6":
        old = Path(continuation["evidence"]) / "artifacts"
        names = ({state["nodes"][node]["file"] for node in relevant_nodes} if attempt["stage"] == "P5" else
                 {"root.yaml"} if attempt["stage"] == "P1" else {"root.yaml", "dispatch.md"})
        locating["continued_draft"] = {"predecessor": continuation["predecessor"],
            "artifacts": {name: str(old / name) for name in sorted(names) if (old / name).is_file()}}
    if attempt.get("mode") == "adjudicate":
        locating["candidates"] = [{**entry, "artifacts": str(run_dir / "checkpoints" / entry["checkpoint"] / "artifacts")}
                                  for entry in scope["candidates"]]
    if attempt["stage"] == "P4":
        locating["current_node_artifacts"] = dict(locating["reviewed_artifacts"])
    review_instructions = ""
    if attempt["role"] == "reviewer":
        for key in ("feedback", "handoff", "plan_receipt", "node_evidence", "prior_approvals",
                    "node_approvals", "reviewed_artifacts", "imported_draft",
                    "continued_draft", "recovery_handoff", "reopening"):
            locating.pop(key, None)
        locating["review_scope"] = {
            "files": locating["required_artifact_evidence"],
            "method": str(rules / "schema_extract/REVIEW.md"),
            "order": ["current_specs", "current_artifacts", "fixed_source"],
        }
        review_instructions = (rules / "schema_extract/REVIEW.md").read_text(encoding="utf-8")
    reader = frozen_reader(lock)
    example = f"python3 -B {shlex.quote(str(reader))} --root {shlex.quote(str(source))} --path {shlex.quote(bindings['host']['path'])} --start-line 1 --end-line 80"
    checker = rules / "schema_template/check_yaml.py"
    target = node_entry["file"] if attempt["stage"] == "P5" else "root.yaml"
    check_command = f"python3 -B {shlex.quote(str(checker))} --skip-hardware {shlex.quote(str(Path(attempt['workspace']) / target))}"
    mode = attempt.get("mode", "normal")
    mode_instructions = """worker返回complete/continue/blocked；continue保存草稿，blocked将合法草稿和缺项交独立reviewer判断。
reviewer返回approve/revise/blocked：源缺陷先按共用方法分流；本阶段应完成的内容完整、来源限制忠实、且符合本次冻结schema_docs的写法和语义才可approve并保留issues，可继续提取不等于可批准残稿。revise只列有证据的错误、遗漏、规范违背和修改要求。blocked必须说明缺哪个事实或哪项规范/工具能力、哪项当前计算/数值/设计关系因此无法合法表达、为何现有事实、规范写法和明确限制不足以确定设计。
普通退修沿用原角色；达到budget.revision_limit（默认5）后由原reviewer仅裁决一次。格式修复不算实质退修，所有调用与恢复继续原预算。"""
    if attempt["stage"] in REOPEN_TARGETS and mode == "normal":
        locating["reopen_targets"] = sorted(REOPEN_TARGETS[attempt["stage"]]) + ["external"]
        mode_instructions += """
固定来源推翻上游时可reopen，填写稳定issue_id、target、reason和本次来源证据索引；普通提案reopen=null。
树/选父回P4，代表路径回P2，现有root Design/when回P3，共享Contract回P1；仅能使用reopen_targets。
外部任务要求改变用external，等待任务方明确确认。已读源行的转录错误用issues.kind=transcription；当前产物的规范表达错误用schema_conformance。两者均由scope定位错误产物，以artifact_evidence引用该字段；规范错误还须在detail指出对应条款。其他重开需新源码发现。同一issue_id与固定源码范围只重开一次，换note/引文不重计。"""
    if mode == "adjudicate":
        mode_instructions = """本轮为原reviewer的一次裁决，只有approve或blocked两个出口；不得直接改workspace。
adjudication的select只能选已列候选；patch基于当前checkpoint，patches为当前阶段/授权字段内的{path,old,new}，old非空且唯一匹配。
Runner在隔离副本应用、核对版本并检查；关键事实缺失、补丁或检查失败即blocked，不再裁决或重试。证据仍指你实际读取的原workspace，补丁后的文件SHA-256另存。
P6亲自patch最终稿后，由全新独立reviewer签收一次；选择已有稿不追加签收。"""
    elif mode == "signoff":
        mode_instructions = "本轮为P6裁决补丁后的全新独立reviewer签收。只允许approve或blocked，检查最终改动及全部pending_rechecks；本次不再返工、裁决或重试。"
    checking = ("本阶段使用阶段产物门禁；未完成或未生成的Design不要求完整检查。"
                if attempt["stage"] in {"P1", "P2", "P4"} else
                f"Runner接收完成/批准时自动运行冻结检查器，退出0才放行；1的诊断交当前修改者，2报工具/环境问题。\n需要提前自检或定位诊断时可运行：{check_command}；不要求worker/reviewer各自重复执行。")
    return f"""你是独立的 {attempt['role']}，只执行 {attempt['stage']} / {attempt['node']} / {mode}。
目标是按本次冻结schema_docs把固定kernel压缩为忠实、足够简洁的流水线设计。reviewer独立对照规范、源码与checkpoint，按当前阶段核对数学/数值、工作组织、依赖、寿命、复用及适用范围；检查器退出码不代替语义审阅。
填写或审阅一段前，先读rules_root/schema_docs中对应的字段、分层和语义规则，优先采用已有合法写法。不自创固定字段、封闭取值或记法，不用正文取消或重新解释已声明的提交、成员和事件；规范明确允许的具名对象、开放取值和自定义动作按其规则填写。旧稿、源码写法或检查器接受都不授予扩展schema语义的权限。
发现规范与实际内容不适配、缺失或冲突时，先在framework_candidates的scope/summary指出schema_docs文件/小节、具体反例、现有写法为何不足及最小建议，evidence仍引用本次已读源码；已有产物的问题用issues及artifact_evidence定位，交Supervisor处理。不能边提候选边采用未定义的规则；Supervisor通过正式接续采用维护后的冻结规则前，继续按现行规范处理，能合法表达的部分继续推进。reviewer对依靠自造规则才能读通的产物不得approve。
符合规范、语义正确且必要信息齐备就直接approve，不要求找问题、增加篇幅或重复已有决定。删除细节会使读者读错或漏掉设计时才保留；不扩张schema字段义务。源实现缺陷按rules_root/schema_extract/REVIEW.md“源实现缺陷与设计分流”处理：保留缺陷事实和证据，不要求YAML复现坏布局；不借修复无依据地新增逻辑依赖、资源策略、fallback或外部保证。
P1到P6均按缺口对当前阶段的实际影响判断，不要求P1提前完成Design。当前所需的计算与设计事实有依据时，外部验收器的公式/相对基准/归一化/检查路由未提供，可在tolerance写明确的来源记录并携带source_gap继续；不采用默认简写解释，不省略或改写已知判据。仅影响外部测试通过边界不构成提取阻塞。计算、精度转换、输出副作用或依赖必须靠猜测才能写清时，指出具体关系后blocked。后续审阅核对缺口未被用于设计保证，不把未补齐外部验收器再次当作停止理由。
Runner唯一写运行状态。worker只改writable_files及当前页授权字段，reviewer只读；不开子代理，不做GPU测试。
只读固定source_root/rules_root、当前workspace、列明的回执/已审产物/candidates及旧稿线索，不访问原工作树或旧状态。先按嵌入的当前页工作，再沿当前问题读规范与源码；可读页内引用的方法。
先定位当前问题需要的源码范围；不因文件较长而通读。有next_cursor表示所选范围尚未读完，需要续读该范围时保留起止行并用--cursor：
{example}
最后按output-schema仅返回schema_extract.proposal.v1 JSON，原样回传本次attempt_id nonce；缺失或陈旧nonce硬拒绝。summary由你填写本轮简短、非空的语义结论。identity、文件SHA-256、文件清单、实际差异由Runner从已登记调用与真实字节生成，不手填、不另写回执文件。
证据由你亲读并选择：相对path、1起始line_start/line_end（最多12行）、quote_offset（范围原文中从0开始的字符偏移，通常0）和说明语义的note；来源另选category。用reader --evidence --end-line N [--quote-offset N]预览该范围生成的最多320字符精确引文，不会替你寻找依据。不要回传sha256/source_id/quote。
complete/approve覆盖required_evidence、required_conclusions；每类至少选对应绑定入口，调用链可补充。conclusions/issues等的evidence从0索引本次source_evidence。reviewer的artifact_evidence覆盖required_artifact_evidence每个文件，不能用父节点批准代替本节点核对。
revise/blocked至少有一条具体且有依据的issue。仅对当前产物的提交格式或规范表达问题，kind可用submission_format或schema_conformance，evidence可为[]；scope须独立写全该产物的相对文件名（可接:字段/行号），并有引用该文件实际内容的artifact_evidence。其他文件、同名片段或旧目录别名不能代替。kind与引文是否支持所述文档事实仍须独立判断，不把源码缺口、源码缺陷、转录、计算、数值或选择问题改称文档问题来省略源码依据。blocked仍要恢复条件；reopen仍遵循原源码依据与重开规则，不适用这一豁免。
reviewer批准另须填写四项document_*结论，针对当前已完成内容说明判断，以本次artifact_evidence支持，来源evidence可为[]；这是文档验收，不能用历史批准或检查器退出码代替。worker提交前也按相同标准自查。
P6的tree仅陈述当前产物的文件集合、父关系、链接和拓扑时，可由完整artifact_evidence和Runner全树检查支持，evidence可为[]；选择、覆盖等源码语义结论仍须有效源码索引，不为纯产物事实附上不相干的源码引用。
{mode_instructions}
有recovery_handoff时以本页、当前checkpoint和固定来源重建上下文，旧会话已确认中断并退休。imported_draft/continued_draft仅为列明的只读旧稿线索；旧state/批准不授予本次写权限或放行，证据仍指当前source_root/workspace。
checks仅记亲自执行的命令/退出码；交Runner自动检查时填[]，checks_note说明由Runner执行。issues写范围、证据、影响和恢复条件。framework_candidates无必要建议为[]；有则给跨运行稳定的小写数字及-_短ID、规则/工具scope与最小改动，不写台账，不因可选候选扣住正确产物。
P2/P3/P4 reviewer批准时handoff只写已定事项、理由、关键入口和缺口，最多20个非空行/1200字符；其他阶段无交接则用空串。
{checking}

<document-review-method>
{review_instructions}
</document-review-method>

<invocation>
{json.dumps(locating, ensure_ascii=False, indent=2)}
</invocation>

<current-phase-page path={json.dumps(str(page_path))}>
{page_path.read_text(encoding='utf-8')}
</current-phase-page>
"""
