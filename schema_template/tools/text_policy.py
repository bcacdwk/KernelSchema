"""Raw-field text budgets and conservative sentence-period normalization."""
from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import re

from .diagnostics import Diagnostic, field_path
from .errors import bounded
from .yaml_source import Document, plain_scalar_allowed, yaml_backend


# None selects a named mapping member; int selects a list entry. These are
# structural paths, never matches against arbitrary keys called name/meaning.
A = ("CONTRACT", "Algorithm_requirement")
H = ("CONTRACT", "Hardware_specification")
M = ("DESIGN", "Math_reshape")
W = ("DESIGN", "Work_schedule")
P = ("DESIGN", "Tile_pipeline")
WS = W + ("Step", None)
PS = P + ("Step", None)
INFO = PS + ("Role", None, "Info")


@dataclass(frozen=True)
class Budget:
    path: tuple
    characters: int | None
    lines: int
    prose: bool = False
    prefix: str | None = None
    string_only: bool = True


BUDGETS = (
    Budget(("META_DATA", "Dispatch", "when"), 500, 5, prose=True),
    Budget(("META_DATA", "Name"), 50, 1),
    Budget(("META_DATA", "Describe"), 150, 1, prose=True),
    Budget(WS + ("Describe",), 150, 1, prose=True),
    Budget(INFO + ("Describe",), 150, 1, prose=True),
    Budget(INFO + ("Using_verb", None, None, "describe"), 150, 1, prose=True),
    *(Budget(prefix + (int, "Name"), 50, 1)
      for prefix in (A + ("Compute",), M + ("Rewrite",))),
    *(Budget(prefix + (int, "Math"), 1000, 10)
      for prefix in (A + ("Compute",), M + ("Rewrite",))),
    *(Budget(prefix + (None, "meaning"), 50, 1, prose=True) for prefix in (
        A + ("Constant",), A + ("Input",), A + ("Output",),
        A + ("Compute", int, "Symbol"), M + ("Rewrite", int, "Symbol"),
        WS + ("Decomposition", "Partition"),
        WS + ("Decomposition", "Worktile", "Invariant"),
        WS + ("Decomposition", "Worktile", "Iteration"),
        WS + ("Decomposition", "Target", "Single_tile"),
        PS + ("Index",), PS + ("Storage",),
    )),
    *(Budget(A + ("Input", None, field), 150, 1, string_only=False)
      for field in ("range", "default")),
    Budget(A + ("Input", None, "range", int), 150, 1, string_only=False),
    *(Budget(region + ("Thesis",), 300, 2, prose=True) for region in (M, W, P)),
    Budget(A + ("Golden_reference", "Tolerance", None), 100, 1),
    Budget(A + ("Golden_reference", "Gap"), 100, 1, prose=True),
    *(Budget(H + (field,), 50, 1) for field in ("Device", "Architecture")),
    *(Budget(H + ("Stack", field), 150, 1) for field in ("Language", "Template", "Toolchain")),
    *(Budget(H + ("Wiki", field), None, 1) for field in ("Hardware", "Microbench", "Profile")),
    *(Budget(H + ("Wiki", "Programming", field), None, 1)
      for field in ("Language", "Template", "Virtual_ISA", "Machine_ISA")),
    *(Budget(H + ("Wiki", "Sota", field), None, 1) for field in ("Kernel", "Schema")),
    *(Budget(H + (field, None), 100, 1, prose=True) for field in ("Execution", "Feature")),
    *(Budget(M + ("Rewrite", int, "Arithmetic", field), 150, 1, prose=True)
      for field in ("Approx", "Accumulate")),
    *(Budget(M + ("Rewrite", int, "Ledger", field), 150, 1, prose=True)
      for field in ("Baseline", "Gain", "Cost")),
    # Arithmetic.Approx already has the same budget. Only these Symbol
    # properties additionally admit unknown:<reason>; do not widen its scope.
    *(Budget(M + ("Rewrite", int, "Symbol", None, field), 150, 1,
             prose=True, prefix="unknown:", string_only=False) for field in ("shape", "dtype")),
    Budget(M + ("Rewrite", int, "Symbol", None, "retention"), 150, 1),
    *(Budget(WS + (part, "Strategy"), 150, 1, prose=True)
      for part in ("Decomposition", "Mapping", "Schedule")),
    Budget(WS + ("Schedule", "Worker_assignment", "Refill", None), 150, 1, prose=True),
    *(Budget(prefix, 300, 2, string_only=False) for prefix in (
        WS + ("Schedule", "Worker_assignment", "Exit"),
        WS + ("Decomposition", "Target", "Tile_reduction", "input"),
        WS + ("Decomposition", "Target", "Tile_reduction", "output"),
        W + ("Flow", "Handoff", None, "initialize"),
    )),
    *(Budget(prefix + (int,), 300, 2) for prefix in (
        WS + ("Schedule", "Worker_assignment", "Exit"),
        WS + ("Decomposition", "Target", "Tile_reduction", "input"),
        WS + ("Decomposition", "Target", "Tile_reduction", "output"),
    )),
    Budget(WS + ("Decomposition", "Target", "Tile_reduction", "method"), 300, 2),
    *(Budget(("REALIZATION", section, int, field), 300, 2, prose=field != "Item",
             string_only=field != "Item")
      for section in ("Require", "Prefer", "Forbid") for field in ("Item", "Choice", "Reason")),
    *(Budget(("REALIZATION", section, int, "Item", int), 300, 2)
      for section in ("Require", "Prefer", "Forbid", "Feedback")),
    Budget(("REALIZATION", "Feedback", int, "Item"), 300, 2, string_only=False),
    Budget(("REALIZATION", "Feedback", int, "Choice"), 150, 1, prose=True),
)


def _matches(parts, pattern):
    return len(parts) == len(pattern) and all(
        isinstance(part, str) if rule is None else
        type(part) is int if rule is int else part == rule
        for part, rule in zip(parts, pattern)
    )


def expects_text(parts):
    """A small, definite subset for parser hints, not another schema checker."""
    return any(rule.string_only and _matches(parts, rule.path) for rule in BUDGETS)


def _select(value, pattern, path):
    if not pattern:
        yield path, value
        return
    head, rest = pattern[0], pattern[1:]
    if head is None and isinstance(value, dict):
        for key, child in value.items():
            yield from _select(child, rest, field_path(path, key))
    elif head is int and isinstance(value, list):
        for index, child in enumerate(value):
            yield from _select(child, rest, f"{path}[{index}]")
    elif isinstance(value, dict) and head in value:
        yield from _select(value[head], rest, field_path(path, head))


def text_size(value: str) -> tuple[int, int]:
    """Unicode code points and logical lines, retaining interior blank lines."""
    return len(value), len(value.splitlines())


def budget_message(value: str, characters: int | None, lines: int) -> str | None:
    """Share the same accounting for raw fields, parsed names and expressions."""
    count, line_count = text_size(value)
    if (characters is not None and count > characters) or line_count > lines:
        limit = str(characters) if characters is not None else "unlimited"
        return (f"text has {count} Unicode characters / {line_count} logical lines; "
                f"limits are {limit} characters / {lines} logical lines")
    return None


def numeric_form_message(value: str, *, integer=False, allow_empty_set=False) -> str | None:
    """Reject missing operands without imposing a mathematical expression DSL."""
    if not value.strip():
        return "expected a nonempty value or expression"
    if integer and ("∞" in value or re.fullmatch(r"[+\-−]?\s*(?:inf|infinity)", value.strip(), re.I)):
        return "an integer expression cannot use infinity"
    if (not any(char.isalnum() for char in value)
            and not (not integer and "∞" in value)
            and not (allow_empty_set and "∅" in value)):
        return "expected a value or expression, not only punctuation or operators"
    return None


def single_names_message(value: str) -> str | None:
    """Bound names in recognizable expressions; opaque formula text stays open.

    Attribute chains may be actual external code names. Leave them intact and
    inspect the arguments of calls separately, without inventing declarations.
    """
    value = value.strip()
    try:
        tree = ast.parse(value, mode="eval")
    except (SyntaxError, ValueError, RecursionError):
        return None
    pending = [tree]
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Attribute):
            continue
        if isinstance(node, (ast.Name, ast.arg)):
            name = ast.get_source_segment(value, node) or (node.id if isinstance(node, ast.Name) else node.arg)
            if message := budget_message(name, 50, 1):
                return "parsed name: " + message
        elif isinstance(node, ast.keyword) and node.arg is not None:
            if message := budget_message(node.arg, 50, 1):
                return "parsed parameter name: " + message
        pending.extend(ast.iter_child_nodes(node))
    return None


def _sentence_stop(value):
    # Called only for explicitly descriptive fields. Prefer missing an ending
    # over deleting a possible abbreviation, quotation, reference or syntax.
    text = value.strip()
    if len(text) < 3 or text[-1] not in ".。" or text[-2] in ".。…":
        return False
    # Only words, whitespace and ordinary prose separators supply enough
    # evidence; other punctuation may be quotation or mathematical syntax.
    if re.search(r"[^\w\s.,，、;；:：'。-]|_", text):
        return False
    if re.search(r"(?<![A-Za-z])'|'(?![A-Za-z])", text) or re.search(r"\b(?:ref|bind):", text, re.I):
        return False
    body = text[:-1]
    if re.fullmatch(r"[\w-]+(?:\.[\w-]+)+", body):
        return False
    chinese = len(re.findall(r"[\u3400-\u9fff]", body)) >= 2
    if chinese and (text.endswith("。") or re.search(r"[\u3400-\u9fff]$", body)):
        return True
    words = body.split()
    if len(words) < 2 or not re.fullmatch(r"[a-z][a-z'-]{2,}", words[-1]):
        return False
    abbreviations = {
        "approx", "assoc", "asst", "attr", "avg", "cfg", "chap", "config", "cont", "corp",
        "dept", "descr", "diag", "diff", "dist", "doc", "docs", "dup", "eds", "elev", "eng",
        "esp", "est", "etc", "fig", "figs", "freq", "gen", "gov", "inc", "incl", "info",
        "init", "int", "intl", "lang", "lib", "ltd", "max", "min", "misc", "msg", "num", "obj",
        "orig", "param", "params", "phys", "prof", "prop", "ptr", "ref", "refs", "repr",
        "resp", "rev", "sec", "secs", "spec", "std", "str", "sync", "temp", "vol", "vols",
    }
    return words[-1] not in abbreviations and (chinese or bool(re.search(r"\b[A-Za-z]{2,}\b", " ".join(words[:-1]))))


SENTENCE_PERIOD_MESSAGE = "ordinary explanatory text should omit its final sentence period"


@dataclass(frozen=True)
class PeriodChange:
    node: object
    value: str
    paths: tuple[str, ...]
    safe: bool


@bounded
def sentence_period_changes(document: Document, *, skip_hardware=False):
    """Plan exact scalar edits without changing source nodes or decoded values.

    Expand source-node uses at their real field paths, including collection
    aliases and merge contexts. Any key, non-prose or excluded hardware use
    prevents editing that shared scalar. Overridden merge values are treated
    conservatively as uses too, so no raw source text is silently changed.
    """
    yaml = yaml_backend()
    uses = {}

    def visit(node, parts=(), *, key=False):
        if isinstance(node, yaml.ScalarNode):
            uses.setdefault(id(node), (node, set()))[1].add((parts, key))
        elif isinstance(node, yaml.SequenceNode):
            for index, child in enumerate(node.value):
                visit(child, parts + (index,))
        elif isinstance(node, yaml.MappingNode):
            for field, child in node.value:
                visit(field, parts + (field.value,), key=True)
                if field.tag == "tag:yaml.org,2002:merge":
                    entries = child.value if isinstance(child, yaml.SequenceNode) else [child]
                    for entry in entries:
                        visit(entry, parts)
                else:
                    visit(child, parts + (field.value,))

    visit(document.node)
    changes = []
    for node, contexts in uses.values():
        if node.tag != "tag:yaml.org,2002:str":
            continue
        paths = []
        for parts, key in contexts:
            if key or skip_hardware and parts[:len(H)] == H:
                continue
            if any(rule.prose and _matches(parts, rule.path) and
                   (rule.prefix is None or node.value.startswith(rule.prefix)) for rule in BUDGETS):
                actual = document.value
                try:
                    for part in parts:
                        actual = actual[part]
                except (KeyError, IndexError, TypeError):
                    continue
                if actual != node.value:
                    continue
                location = document.name
                for part in parts:
                    location = f"{location}[{part}]" if type(part) is int else field_path(location, part)
                paths.append(location)
        if paths and _sentence_stop(node.value):
            stop = len(node.value.rstrip()) - 1
            changes.append(PeriodChange(node, node.value[:stop] + node.value[stop + 1:],
                                        tuple(sorted(paths)), len(paths) == len(contexts)))
    return tuple(changes)


@bounded
def check_values(value, path="root", *, scope="document", skip_hardware=False):
    """Check raw values without I/O, schema inference, or expression execution.

    scope='algorithm' accepts an Algorithm_requirement mapping and the path
    supplied by rules.validate_algorithm. Type/required-field checks stay in
    that caller. Each list item/string range endpoint has its own budget;
    nullable Ledger entries consume no text budget. Nothing is trimmed.
    """
    if scope not in {"document", "algorithm"}:
        raise ValueError("text policy scope must be 'document' or 'algorithm'")
    diagnostics = []
    for rule in BUDGETS:
        if skip_hardware and rule.path[:len(H)] == H:
            continue
        pattern = rule.path
        if scope == "algorithm":
            if pattern[:len(A)] != A:
                continue
            pattern = pattern[len(A):]
        for location, text in _select(value, pattern, path):
            if not isinstance(text, str) or rule.prefix is not None and not text.startswith(rule.prefix):
                continue
            if message := budget_message(text, rule.characters, rule.lines):
                diagnostics.append(Diagnostic(
                    location, message,
                    hint="Shorten repeated or unnecessary wording while preserving the required meaning",
                ))
            if rule.prose and _sentence_stop(text):
                diagnostics.append(Diagnostic(
                    location, SENTENCE_PERIOD_MESSAGE,
                    hint="Default formatting removes this terminal period where all reused contexts are safe; read-only checks preserve the source",
                    severity="warning",
                ))
    return diagnostics


@bounded
def check_document(document: Document, *, skip_hardware=False):
    """Check one raw definition once, including its actual source styles.

    Call on raw documents, never effective from-expanded copies. Value aliases
    are checked at each distinct field context; wrapper-quote warnings describe
    each physically written scalar once. Comments remain a document-wide
    preservation constraint even when hardware semantics are skipped.
    """
    blocked = {path for change in sentence_period_changes(document, skip_hardware=skip_hardware)
               if not change.safe for path in change.paths}
    diagnostics = [replace(item, hint="Period retained: this scalar is shared with a key, non-prose or excluded hardware context; review its uses")
                   if item.path in blocked and item.message == SENTENCE_PERIOD_MESSAGE else item
                   for item in check_values(document.value, document.name, skip_hardware=skip_hardware)]
    yaml = yaml_backend()
    for use in document.occurrences:
        if use.alias is not None or skip_hardware and use.parts[:len(H)] == H:
            continue
        node = use.node
        if (isinstance(node, yaml.ScalarNode) and node.tag == "tag:yaml.org,2002:str"
                and node.style in {"'", '"'} and plain_scalar_allowed(node, flow=use.flow, key=use.key)):
            diagnostics.append(Diagnostic(
                use.path, f"unnecessary YAML wrapper quotes (line {use.line}, column {use.column}); "
                          "this text can be written plain in its current context",
                hint="Remove wrappers only where YAML syntax and string type remain unchanged",
                severity="warning",
            ))
    for comment in document.comments:
        diagnostics.append(Diagnostic(
            comment.path,
            f"YAML comments are not allowed by DOC0 (line {comment.line}, column {comment.column})",
            hint="Move design facts into their defined fields; formatting will preserve the file by refusing to discard comments",
        ))
    return diagnostics
