"""Deterministic graph layout and PNG rendering; DOT exists only in memory."""
from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
from graphlib import CycleError, TopologicalSorter
import textwrap

from .model import PhaseGraph

COLORS = ["#e5f0fa", "#e6f2eb", "#fcebdc", "#f6e4ed", "#eee7f8", "#e3f1f1"]
PHASE_COLORS = [("#365b7d", "#f0f5fa"), ("#3e6967", "#eff7f5")]
STYLES = {
    "execute": ("#617b91", "solid"), "program": ("#9aa8ba", "dashed"),
    "completion": ("#b3c3d6", "dotted"), "data": ("#315a7d", "solid"),
    "reuse": ("#b97923", "dashed"), "sync": ("#8761b2", "dashed"),
}


class RenderError(RuntimeError):
    """The checked graph cannot be published as a current PNG (exit 2)."""


def quote(value) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def display_members(graph: PhaseGraph) -> set[str]:
    """Fold only identical member neighborhoods, and only if the quotient is a DAG.

    The checked event graph always keeps every member. Asymmetric inter-member
    handoffs remain separate columns so the picture cannot conceal a cycle.
    """
    normalize = lambda event: re.sub(r"\[\d+\]/", "[*]/", event)
    signatures = {}
    for op in graph.operations.values():
        signatures.setdefault(op.role, {}).setdefault(op.member, set()).add((op.action, op.execution_mode))
    for a, b, kind in graph.edges:
        source = graph.operations[a.rsplit("@", 1)[0]]
        target = graph.operations[b.rsplit("@", 1)[0]]
        signatures[source.role][source.member].add(("out", normalize(a), normalize(b), kind))
        signatures[target.role][target.member].add(("in", normalize(a), normalize(b), kind))
    folded = {role for role, members in signatures.items()
              if len(members) > 1 and all(s == next(iter(members.values())) for s in members.values())}
    predecessors = {}
    for a, b, _ in graph.edges:
        if graph.operations[a.rsplit("@", 1)[0]].role in folded: a = normalize(a)
        if graph.operations[b.rsplit("@", 1)[0]].role in folded: b = normalize(b)
        predecessors.setdefault(b, set()).add(a)
    try:
        tuple(TopologicalSorter(predecessors).static_order())
    except CycleError:
        return set()
    return folded


def dot_text(title: str, graphs: list[PhaseGraph], samples: dict, notes=()) -> str:
    status = "DEPENDENCY CYCLE" if any(graph.cycle for graph in graphs) else "Acyclic sample"
    label = title + "\n" + status + " · Representative event graph · NO TIME SCALE\n"
    label += "Blue: data / completion   Gray dashed: program   Dotted: ordered completion\n"
    label += "Orange: reuse   Purple: sync   Red: dependency cycle\nPhases checked independently · identical members folded for display"
    label += "\nIllustration parameters; no cross-phase or resource-safety proof."
    for note in notes:
        label += "\n" + textwrap.fill(note, 100)
    lines = ["digraph Pipeline {",
             'graph [rankdir=TB, compound=true, newrank=true,',
             f' bgcolor="white", pad=0.3, nodesep=0.16, ranksep=0.27, label={quote(label)},',
             ' labelloc=t, fontname="Helvetica", fontsize=15];',
             'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10,',
             ' margin="0.09,0.06", color="#b6c7d5"];',
             'edge [arrowsize=0.55];']
    for pi, graph in enumerate(graphs):
        sample = samples[graph.phase]
        description = ", ".join(f"{k}={v}" for k, v in sorted(sample["parameters"].items()))
        border, background = PHASE_COLORS[pi % len(PHASE_COLORS)]
        phase_title = html.escape(f"PHASE {pi + 1} / {len(graphs)} · {graph.phase}")
        sample_label = html.escape(f"worktiles={sample['worktiles']} | {description}")
        phase_label = (
            '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
            f'<TR><TD BGCOLOR="{border}" ALIGN="LEFT" CELLPADDING="10">'
            f'<FONT COLOR="white" POINT-SIZE="18"><B>{phase_title}</B></FONT></TD></TR>'
            f'<TR><TD ALIGN="LEFT" CELLPADDING="8"><FONT COLOR="#526477" POINT-SIZE="11">'
            f'{sample_label}</FONT></TD></TR></TABLE>'
        )
        lines.append(f'subgraph cluster_phase_{pi} {{ label=<{phase_label}>; labeljust=l; labelloc=t; '
                     f'color={quote(border)}; fillcolor={quote(background)}; '
                     'style="rounded,filled"; penwidth=2.4; margin=24;')
        folded = display_members(graph)
        # A synchronous action can span another role's work. Its issue and
        # done must stay distinct whenever a dependency uses that interval.
        # Only ordinary incoming-to-issue / outgoing-from-done edges permit
        # contraction without introducing a cycle or hiding a real one.
        separate = {op.id for op in graph.operations.values() if op.execution_mode != "serial"}
        for source, target, kind in graph.edges:
            if kind != "execute":
                if source.endswith("@issue"):
                    separate.add(source.rsplit("@", 1)[0])
                if target.endswith("@done"):
                    separate.add(target.rsplit("@", 1)[0])
        separate.update(event.rsplit("@", 1)[0] for event in graph.cycle)
        roles = list(dict.fromkeys((op.role, "*" if op.role in folded else op.member) for op in graph.operations.values()))
        visual = {}
        shown = set()
        anchors = []
        for li, (role, member) in enumerate(roles):
            color = COLORS[list(dict.fromkeys(x[0] for x in roles)).index(role) % len(COLORS)]
            member_label = ",".join(str(m) for m in sorted({o.member for o in graph.operations.values() if o.role == role})) if member == "*" else str(member)
            lines.append(f"subgraph cluster_{pi}_{li} {{ label={quote(f'{role}[{member_label}]')}; "
                         'labeljust=c; color="#d2dce6"; fillcolor="white"; style="rounded,filled"; '
                         'penwidth=0.8; fontcolor="#334155"; fontsize=12; margin=12;')
            anchor = f"anchor_{pi}_{li}"
            anchors.append(anchor)
            lines.append(f'{anchor} [shape=point, width=0, style=invis];')
            first = True
            for op in graph.operations.values():
                if op.role != role or member != "*" and op.member != member:
                    continue
                event_id = lambda event: re.sub(r"\[\d+\]/", "[*]/", op.event(event)) if member == "*" else op.event(event)
                split = op.id in separate
                visual[op.event("issue")] = event_id("issue" if split else "done")
                visual[op.event("done")] = event_id("done")
                if event_id("done") in shown:
                    continue
                shown.add(event_id("done"))
                events = ("issue", "done") if split else ("done",)
                for event in events:
                    node = event_id(event)
                    suffix = "\n" + ("ISSUE" if event == "issue" else "DONE") if split else ""
                    tip = op.path + "\n" + json.dumps(op.context, sort_keys=True)
                    lines.append(f"{quote(node)} [label={quote(op.action + suffix)}, fillcolor={quote(color)}, tooltip={quote(tip)}];")
                if first:
                    lines.append(f'{anchor} -> {quote(visual[op.event("issue")])} [style=invis, weight=100];')
                    first = False
            lines.append("}")
        lines.append("{rank=same; " + "; ".join(anchors) + ";}")
        for left, right in zip(anchors, anchors[1:]):
            lines.append(f'{left} -> {right} [style=invis, weight=100];')
        cycle_edges = set(zip(graph.cycle, graph.cycle[1:]))
        displayed = set()
        for (a, b, kind), reasons in sorted(graph.edges.items()):
            src, dst = visual[a], visual[b]
            if src == dst and a != b:
                continue
            key = (src, dst, kind)
            if key in displayed:
                continue
            displayed.add(key)
            color, style = STYLES[kind]
            width = 1.1
            if (a, b) in cycle_edges:
                color, width = "#c83232", 2.4
            tip = kind + "\n" + "\n".join(sorted(reasons))
            lines.append(f"{quote(src)} -> {quote(dst)} [color={quote(color)}, style={quote(style)}, penwidth={width}, tooltip={quote(tip)}];")
        # Invisible edges arrange panels and columns only; they never enter the
        # audited event graph or imply a cross-phase synchronization.
        end = f"phase_end_{pi}"
        lines.append(f'{end} [shape=point, width=0, style=invis];')
        outgoing = {visual[a] for a, b, _ in graph.edges if visual[a] != visual[b]}
        for node in sorted(set(visual.values()) - outgoing):
            lines.append(f'{quote(node)} -> {end} [style=invis, weight=0];')
        lines.append("}")
        if pi and anchors:
            lines.append(f'phase_end_{pi-1} -> {anchors[0]} [style=invis, minlen=5];')
    lines.append("}")
    return "\n".join(lines) + "\n"


def graphviz(dot: str) -> bytes:
    executable = shutil.which("dot")
    if executable is None:
        raise RenderError("Graphviz 'dot' is required for PNG; install Graphviz or use --check-only")
    try:
        result = subprocess.run([executable, "-Tpng"], input=dot.encode("utf-8"),
                                capture_output=True, timeout=60)
    except subprocess.TimeoutExpired as error:
        raise RenderError("Graphviz exceeded 60 seconds; reduce the sample size") from error
    if result.returncode:
        raise RenderError("Graphviz failed: " + result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout
