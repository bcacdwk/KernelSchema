"""Format raw nodes with explicit spelling plans and verified prose edits."""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from decimal import Decimal
from io import StringIO
import re

from .diagnostics import Diagnostic, field_path
from .errors import SchemaError, ToolError, bounded
from .text_policy import sentence_period_changes
from .yaml_source import (Document, ReadError, ScalarContext, loads, plain_scalar_allowed,
                          scalar_contexts, scalar_resolver, yaml_backend)


A = ("CONTRACT", "Algorithm_requirement")
H = ("CONTRACT", "Hardware_specification")
M = ("DESIGN", "Math_reshape")
W = ("DESIGN", "Work_schedule")
P = ("DESIGN", "Tile_pipeline")
WS = W + ("Step", None)
PS = P + ("Step", None)
INFO = PS + ("Role", None, "Info")
ASSIGNMENT = WS + ("Schedule", "Worker_assignment")
STORAGE = PS + ("Storage", None)
# Match the decoded logical separators counted by str.splitlines().
LINE_BREAKS = "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029"

SCHEDULE_RECORDS = (
    (WS + ("Mapping", "Worker"), ("unit", "unit_count")),
    (WS + ("Mapping", "Placement"), ("unit", "unit_count")),
    (WS + ("Schedule", "Worktile_order", int), ("by", "group", "direction")),
    (ASSIGNMENT + ("Preassigned",), ("start", "stride", "count")),
)

STORAGE_FIRST_LINE = ("meaning", "depth", "location", "shape")
STORAGE_SECOND_LINE = ("dtype", "total_size", "member", "view")
STORAGE_RECORDS = ((STORAGE, STORAGE_FIRST_LINE + STORAGE_SECOND_LINE),)

# Only these fixed records are single-line entries. None means a named member
# and int a list position; matching never depends on the spelling of that name.
RECORDS = (
    (A + ("Constant", None), ("value", "meaning")),
    (A + ("Input", None), ("meaning", "shape", "dtype", "nullable", "range", "default")),
    (A + ("Output", None), ("meaning", "shape", "dtype", "nullable")),
    (A + ("Compute", int, "Symbol", None), ("meaning", "shape", "dtype")),
    (M + ("Rewrite", int, "Symbol", None), ("meaning", "shape", "dtype", "retention", "binding", "value")),
    (WS + ("Decomposition", "Partition", None), ("meaning", "size", "tile", "range")),
    (WS + ("Decomposition", "Worktile", "Invariant", None), ("meaning", "shape", "dtype", "offset")),
    (WS + ("Decomposition", "Worktile", "Iteration", None), ("meaning", "shape", "dtype", "direction")),
    (WS + ("Decomposition", "Target", "Single_tile", None), ("meaning", "shape", "dtype")),
    (WS + ("Decomposition", "Target", "Tile_reduction"), ("dimension", "input", "output", "method")),
    (W + ("Flow", "Launch", int), ("step", "order", "start_after")),
    (W + ("Flow", "Handoff", None), ("producer", "consumer", "storage", "initialize", "ready", "wait_at")),
    (PS + ("Index", None), ("meaning", "definition")),
    (STORAGE + ("view", None), ("slot_index", "region")),
    (INFO + ("Using_verb", None, None), ("execution_mode", "describe")),
) + SCHEDULE_RECORDS

# These existing attribute shapes can appear inside their one-line record.
# They are not permission to flatten a container at an unrelated path.
INLINE_PROPERTIES = frozenset({
    "shape", "range", "value", "offset", "definition", "region", "member",
    "step", "consumer", "start_after", "ready", "wait_at", "input", "output", "by",
})

ORDERS = RECORDS + STORAGE_RECORDS + (
    ((), ("META_DATA", "CONTRACT", "DESIGN", "REALIZATION")),
    (("META_DATA",), ("name", "describe", "source", "version", "Dispatch")),
    (("META_DATA", "Dispatch"), ("id", "when")),
    (("CONTRACT",), ("Algorithm_requirement", "Hardware_specification")),
    (A, ("Constant", "Input", "Output", "Compute", "Golden_reference")),
    (A + ("Compute", int), ("name", "math", "Symbol")),
    (A + ("Golden_reference",), ("source", "Tolerance", "gap")),
    (H, ("Device", "Architecture", "Execution", "Feature", "Stack", "Wiki")),
    (H + ("Stack",), ("Language", "Template", "Toolchain")),
    (H + ("Wiki",), ("Hardware", "Programming", "Microbench", "Sota", "Profile")),
    (H + ("Wiki", "Programming"), ("Language", "Template", "Virtual_isa", "Machine_isa")),
    (H + ("Wiki", "Sota"), ("Kernel", "Schema")),
    (("DESIGN",), ("Math_reshape", "Work_schedule", "Tile_pipeline")),
    (M, ("Thesis", "Rewrite")),
    (M + ("Rewrite", int), ("name", "math", "Symbol", "Arithmetic", "Ledger")),
    (M + ("Rewrite", int, "Arithmetic"), ("Approx", "Accumulate")),
    (M + ("Rewrite", int, "Ledger"), ("Baseline", "Gain", "Cost")),
    (W, ("Thesis", "Step", "Flow")),
    (WS, ("describe", "Decomposition", "Mapping", "Schedule")),
    (WS + ("Decomposition",), ("strategy", "Partition", "Worktile", "Target")),
    (WS + ("Decomposition", "Worktile"), ("Invariant", "Iteration")),
    (WS + ("Decomposition", "Target"), ("Single_tile", "Tile_reduction")),
    (WS + ("Mapping",), ("strategy", "Worker", "worker_count", "Placement", "member")),
    (WS + ("Schedule",), ("strategy", "Worktile_order", "Worker_assignment")),
    (WS + ("Schedule", "Worker_assignment"), ("Preassigned", "Refill", "exit")),
    (W + ("Flow",), ("Launch", "Handoff")),
    (P, ("Thesis", "Step")),
    (PS, ("Index", "Event", "Storage", "Role")),
    (PS + ("Role", None), ("Info", "Pipe", "Waiting")),
    (INFO, ("describe", "member", "register", "Using_verb")),
    (INFO + ("Using_verb",), ("compute", "matmul", "load", "store", "await", "signal", "acquire", "release")),
    (PS + ("Role", None, "Waiting"), ("Prepared", "Released", "Allowed")),
    (("REALIZATION",), ("Require", "Prefer", "Forbid", "Feedback")),
    *(((("REALIZATION", section, int)), ("item", "choice", "reason"))
      for section in ("Require", "Prefer", "Forbid")),
    (("REALIZATION", "Feedback", int), ("item", "choice", "evidence")),
)


def _matches(parts, pattern):
    return len(parts) == len(pattern) and all(
        part is None or isinstance(part, str) if rule is None else
        part is int or type(part) is int if rule is int else part == rule
        for part, rule in zip(parts, pattern)
    )


def _order(path, table):
    return next((order for pattern, order in table if _matches(path, pattern)), None)


def canonical_parts(parts, fields):
    """Canonical fixed components, retaining every named/list component."""
    result = []
    for part in parts:
        names = _order(tuple(result), fields)
        if isinstance(part, str) and names:
            part = next((name for name in names if part.casefold() == name.casefold()), part)
        result.append(part)
    return tuple(result)


def _layout_tables():
    # Layout owns the chosen record/order policy, while structure owns the
    # unique spelling. Case-only field decisions need no second spelling table.
    from .rules.structure import CANONICAL_FIELDS

    def project(table):
        return tuple((canonical_parts(pattern, CANONICAL_FIELDS), tuple(
            canonical_parts(pattern + (name,), CANONICAL_FIELDS)[-1] for name in names
        )) for pattern, names in table)

    return project(RECORDS + STORAGE_RECORDS), project(SCHEDULE_RECORDS), project(ORDERS)


@dataclass(frozen=True)
class ScalarEdit:
    """Authorize a case-only edit at these original scalar-use contexts.

    All uses of a shared source node must agree before any candidate is made.
    Separate field, declaration and reference planners may supply its uses;
    no planner can silently rename a shared key while retaining a value use.
    """

    node: object
    value: str
    contexts: tuple[ScalarContext, ...]
    main_names: tuple[str, ...] = ()


def _main_identity_edit(before, after, names):
    """The only non-case rename is an explicitly bound single identity to MAIN."""
    tokens = r"([A-Za-z_][A-Za-z0-9_]*)"
    left, right = re.split(tokens, before), re.split(tokens, after)
    return len(left) == len(right) and all(
        a == b if index % 2 == 0 else a.casefold() == b.casefold() or b == "MAIN" and a in names
        for index, (a, b) in enumerate(zip(left, right))
    )


def field_case_edits(document: Document, fields) -> tuple[ScalarEdit, ...]:
    """Match only current fixed field spellings at their structural positions.

    The caller supplies the one canonical internal path table, never aliases
    or a version matrix. Named keys pass through unchanged; their descendants
    can still have fixed fields. Ancestor normalization is considered while
    looking up a child, but each edit records the exact original source path.
    """
    canonical = []
    for pattern, names in fields:
        spellings = {name.casefold(): name for name in names}
        if len(spellings) != len(names):
            raise ToolError(f"ambiguous canonical field spellings at {pattern!r}")
        canonical.append((pattern, tuple(spellings.values())))
    edits = []
    for context in scalar_contexts(document):
        if not context.key or context.node.tag != "tag:yaml.org,2002:str":
            continue
        parts = canonical_parts(context.parts, canonical)
        target = parts[-1]
        if target != context.node.value:
            edits.append(ScalarEdit(context.node, target, (context,)))
    return tuple(edits)


def _edit_plan(document, edits):
    """Validate complete, unambiguous permissions before changing a node."""
    edits = tuple(edits)
    if not edits:
        return {}, {}, set(), {}
    yaml = yaml_backend()
    actual = {}
    for context in scalar_contexts(document):
        actual.setdefault(id(context.node), set()).add(context)
    changes, requested = {}, {}
    for edit in edits:
        identity = id(edit.node)
        if identity not in actual or not isinstance(edit.node, yaml.ScalarNode):
            raise ToolError("normalization edit does not belong to this source document")
        if edit.node.tag != "tag:yaml.org,2002:str" or type(edit.value) is not str:
            raise ToolError("spelling normalization may only edit string scalars")
        if (edit.node.value.casefold() != edit.value.casefold() and
                not _main_identity_edit(edit.node.value, edit.value, edit.main_names)):
            raise ToolError("spelling normalization only permits case changes or an explicitly bound single identity to MAIN")
        contexts = set(edit.contexts)
        if not contexts or not contexts <= actual[identity]:
            raise ToolError("normalization edit must name its original scalar-use contexts")
        if edit.node.value == edit.value:
            continue
        for context in contexts:
            parts = tuple(part.casefold() if isinstance(part, str) else part for part in context.parts)
            if not context.key and (parts == ("meta_data", "dispatch", "id") or
                    len(parts) == 3 and parts[0] == "design" and
                    parts[1] in {"math_reshape", "work_schedule", "tile_pipeline"} and parts[2] == "from"):
                raise SchemaError(f"{context.path}: capitalization normalization must preserve the task/source identity")
        if identity in changes and changes[identity] != edit.value:
            location = sorted(context.path for context in contexts)[0]
            raise SchemaError(f"{location}: normalization has conflicting spellings for one shared scalar; review its uses")
        changes[identity] = edit.value
        requested.setdefault(identity, set()).update(contexts)
    key_changes, value_changes = {}, set()
    for identity, contexts in requested.items():
        missing = actual[identity] - contexts
        if missing:
            locations = ", ".join(sorted({context.path for context in missing}))
            raise SchemaError(f"{document.name}: normalization cannot safely change a shared scalar; "
                              f"these uses have no matching declaration/reference edit: {locations}")
        for context in contexts:
            if context.key:
                previous = key_changes.setdefault(context.path, changes[identity])
                if previous != changes[identity]:
                    raise SchemaError(f"{context.path}: normalization disagrees across YAML merge contexts")
            else:
                value_changes.add((context.path, context.node.value, changes[identity]))

    def check_keys(value, path):
        if isinstance(value, dict):
            destinations = {}
            for name, child in value.items():
                location = field_path(path, name)
                target = key_changes.get(location, name)
                if target in destinations:
                    raise SchemaError(f"{path}: normalization would merge distinct keys "
                                      f"{destinations[target]!r} and {name!r} into {target!r}; keep unambiguous names")
                destinations[target] = name
                check_keys(child, location)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                check_keys(child, f"{path}[{index}]")

    check_keys(document.value, document.name)
    return changes, key_changes, value_changes, requested


def _edited_graph(root, changes):
    """Copy the source graph once, preserving node sharing and all metadata."""
    yaml = yaml_backend()
    copies = {}

    def visit(node):
        if id(node) in copies:
            return copies[id(node)]
        result = copies[id(node)] = copy(node)
        if isinstance(node, yaml.ScalarNode):
            result.value = changes.get(id(node), node.value)
        elif isinstance(node, yaml.SequenceNode):
            result.value = [visit(child) for child in node.value]
        elif isinstance(node, yaml.MappingNode):
            result.value = [(visit(key), visit(child)) for key, child in node.value]
        return result

    return visit(root)


def _single_line(node, yaml, *, keep_block=True):
    if isinstance(node, yaml.ScalarNode):
        return (not keep_block or node.style not in {"|", ">"}) and not any(
            c in node.value for c in LINE_BREAKS)
    if isinstance(node, yaml.SequenceNode):
        return all(_single_line(child, yaml, keep_block=keep_block) for child in node.value)
    if isinstance(node, yaml.MappingNode):
        return all(_single_line(key, yaml, keep_block=keep_block) and
                   _single_line(value, yaml, keep_block=keep_block) for key, value in node.value)
    return False


def _inline_record(node, path, yaml, records, schedule_records):
    order = _order(path, records)
    if order is None or not isinstance(node, yaml.MappingNode):
        return False
    storage_record = _matches(path, STORAGE) or _matches(path, STORAGE + ("view", None))
    keep_block = _order(path, schedule_records) is None and not storage_record
    for key, value in node.value:
        if key.tag == "tag:yaml.org,2002:merge":
            if not _single_line(value, yaml, keep_block=keep_block):
                return False
            # A merge might supply view or another structural child. Check the
            # reused record at this field context without expanding its syntax.
            entries = value.value if isinstance(value, yaml.SequenceNode) else [value]
            if not all(_inline_record(entry, path, yaml, records, schedule_records) for entry in entries):
                return False
            continue
        if key.value not in order or not _single_line(value, yaml, keep_block=keep_block):
            return False
        if _matches(path, STORAGE) and key.value == "view":
            if not _inline_views(value, path + ("view",), yaml, records, schedule_records):
                return False
            continue
        if not isinstance(value, yaml.ScalarNode) and key.value.casefold() not in INLINE_PROPERTIES:
            return False
    return True


def _inline_views(node, path, yaml, records, schedule_records):
    """Only the named view table and its existing attribute records may join Storage."""
    if not isinstance(node, yaml.MappingNode):
        return False
    for key, value in node.value:
        if key.tag == "tag:yaml.org,2002:merge":
            entries = value.value if isinstance(value, yaml.SequenceNode) else [value]
            if not all(_inline_views(entry, path, yaml, records, schedule_records) for entry in entries):
                return False
        elif not _inline_record(value, path + (key.value,), yaml, records, schedule_records):
            return False
    return True


def _flow_style(node, path, yaml, records, schedule_records):
    if isinstance(node, yaml.MappingNode):
        if _inline_record(node, path, yaml, records, schedule_records):
            return True
        if _matches(path, ASSIGNMENT + ("Refill",)):
            return (len(node.value) == 1 and
                    node.value[0][0].tag != "tag:yaml.org,2002:merge" and
                    isinstance(node.value[0][1], yaml.ScalarNode))
        # Single-member named tables may join the parent, but Step, Role,
        # Info, regions and every other unlisted structure stay expanded.
        if len(node.value) == 1 and not _matches(path, PS + ("Storage",)):
            key, value = node.value[0]
            if key.tag != "tag:yaml.org,2002:merge" and any(
                pattern[-1:] == (None,) and _matches(path, pattern[:-1])
                for pattern, _ in records
            ):
                return _inline_record(value, path + (key.value,), yaml, records, schedule_records)
    if isinstance(node, yaml.SequenceNode):
        # Pipe and all nested for/if bodies are always block lists.
        pipe_prefix = PS + ("Role", None, "Pipe")
        if len(path) >= len(pipe_prefix) and _matches(path[:len(pipe_prefix)], pipe_prefix):
            return False
        if _matches(path, ASSIGNMENT + ("Exit",)):
            return 1 <= len(node.value) <= 3 and all(
                isinstance(child, yaml.ScalarNode) and child.tag == "tag:yaml.org,2002:str" and
                len(child.value) <= 50 and _single_line(child, yaml, keep_block=False)
                for child in node.value)
        if len(node.value) == 1:
            child = node.value[0]
            if isinstance(child, yaml.ScalarNode) and child.tag == "tag:yaml.org,2002:str" and _single_line(child, yaml):
                return True
            if (_matches(path, W + ("Flow", "Launch")) or
                    _matches(path, WS + ("Schedule", "Worktile_order"))):
                return _inline_record(child, path + (0,), yaml, records, schedule_records)
    # Known nested attribute shapes also stay inline when a surrounding record
    # must expand for a block scalar or an unsupported structural child.
    if path and isinstance(path[-1], str) and path[-1].casefold() in INLINE_PROPERTIES:
        parent_order = _order(path[:-1], records)
        if parent_order is not None and path[-1] in parent_order:
            return _single_line(node, yaml)
    if _matches(path, WS + ("Mapping", "Member")):
        return _single_line(node, yaml)
    return False


def _render(document, changes=None):
    yaml = yaml_backend()
    resolver = scalar_resolver()
    records, schedule_records, orders = _layout_tables()
    stream = StringIO()

    class Emitter(yaml.emitter.Emitter):
        def increase_indent(self, flow=False, indentless=False):
            # Lists use the same two-space nesting as mappings.
            return super().increase_indent(flow=flow, indentless=False)

        def choose_scalar_style(self):
            if not self.event.style and getattr(self.event, "source_plain", False):
                return ""
            return super().choose_scalar_style()

        def check_simple_key(self):
            if super().check_simple_key():
                return True
            # PyYAML conservatively splits keys at 128 characters. Ordinary
            # plain keys can use YAML's actual 1024-character simple-key limit,
            # so a long identifier does not split an otherwise simple entry.
            if isinstance(self.event, yaml.ScalarEvent) and getattr(self.event, "source_plain", False):
                length = len(self.event.value)
                if self.event.anchor is not None:
                    length += len(self.prepared_anchor) + 2
                if not self.event.implicit[0] and self.prepared_tag:
                    length += len(self.prepared_tag) + 1
                return not self.analysis.multiline and not self.analysis.empty and length <= 1024
            return False

        def write_indicator(self, indicator, need_whitespace, whitespace=False, indention=False):
            super().write_indicator(indicator, need_whitespace, whitespace, indention)
            if indicator == "," and getattr(self.event, "storage_break", False):
                # Only the separator between Storage's two fixed groups.
                self.write_indent()

    # Infinity disables physical word wrapping; layout is decided by the
    # finite path table, never by an arbitrary line-width threshold.
    emitter = Emitter(stream, indent=2, width=float("inf"), allow_unicode=True, line_break="\n")
    seen = set()
    # Canonical field order must never move an alias ahead of its anchor.
    preserve_order = bool(document.anchors)
    root = _edited_graph(document.node, changes) if changes else document.node

    def anchor_fields(pairs):
        """Record each not-yet-emitted anchor's defining Storage field.

        Fixed field order may move independent definitions, but must not turn
        an alias at another field into the definition. Other mappings retain
        source order whenever the document contains anchors.
        """
        visited, definitions = set(seen), {}

        def visit(node, owner):
            if id(node) in visited:
                return
            visited.add(id(node))
            if getattr(node, "anchor", None) is not None:
                definitions[id(node)] = owner
            if isinstance(node, yaml.SequenceNode):
                for child in node.value:
                    visit(child, owner)
            elif isinstance(node, yaml.MappingNode):
                for field, child in node.value:
                    visit(field, owner)
                    visit(child, owner)

        for field, child in pairs:
            visit(field, id(field))
            visit(child, id(field))
        return definitions

    def emit_node(node, path, *, flow=False, key=False, storage_break=False):
        identity = id(node)
        if identity in seen:
            anchor = getattr(node, "anchor", None)
            if anchor is None:
                raise ToolError("raw YAML reuses a node without a source anchor")
            event = yaml.AliasEvent(anchor)
            event.storage_break = storage_break
            emitter.emit(event)
            return
        seen.add(identity)
        anchor = getattr(node, "anchor", None)
        explicit = getattr(node, "explicit_tag", None)
        tag = explicit if explicit is not None else node.tag
        if isinstance(node, yaml.ScalarNode):
            implicit = (node.tag == resolver.resolve(yaml.ScalarNode, node.value, (True, False)),
                        node.tag == resolver.resolve(yaml.ScalarNode, node.value, (False, True)))
            if explicit is not None:
                implicit = (False, False)
            style = node.style if node.style in {"|", ">"} and not flow else None
            if not key and _matches(path, ASSIGNMENT + ("Refill", None)):
                # Each mechanism rule occupies one physical line. Quoted
                # escapes retain any decoded line breaks for budget checking.
                style = '"' if any(c in node.value for c in LINE_BREAKS) else None
            elif style is None and node.tag == "tag:yaml.org,2002:str" and "\n" in node.value and not flow and not key:
                style = "|"
            event = yaml.ScalarEvent(anchor, tag, implicit, node.value, style=style)
            event.source_plain = plain_scalar_allowed(node, flow=flow, key=key)
            event.storage_break = storage_break
            emitter.emit(event)
            return
        inline = flow or _flow_style(node, path, yaml, records, schedule_records)
        if isinstance(node, yaml.SequenceNode):
            emitter.emit(yaml.SequenceStartEvent(anchor, tag, explicit is None, flow_style=inline))
            for index, child in enumerate(node.value):
                emit_node(child, path + (index,), flow=inline)
            emitter.emit(yaml.SequenceEndEvent())
        elif isinstance(node, yaml.MappingNode):
            pairs = node.value
            storage_record = _matches(path, STORAGE)
            order = _order(path, orders) if not preserve_order or storage_record else None
            if order is not None:
                rank = {field: index for index, field in enumerate(order)}
                ordered = sorted(pairs, key=lambda pair: rank.get(pair[0].value, len(rank)))
                if not preserve_order or not storage_record or anchor_fields(pairs) == anchor_fields(ordered):
                    pairs = ordered
            split = None
            if inline and storage_record:
                split = next((index for index, (field, _) in enumerate(pairs)
                              if field.value.casefold() in STORAGE_SECOND_LINE), None)
                if split is not None and any(field.value.casefold() in STORAGE_FIRST_LINE
                                             for field, _ in pairs[split:]):
                    # Source anchor order takes priority when it interleaves
                    # the fixed groups; never reorder a definition after use.
                    inline, split = False, None
            emitter.emit(yaml.MappingStartEvent(anchor, tag, explicit is None, flow_style=inline))
            for index, (field, child) in enumerate(pairs):
                child_path = path + (field.value,)
                emit_node(field, child_path, flow=inline, key=True, storage_break=index == split and index > 0)
                emit_node(child, child_path, flow=inline)
            emitter.emit(yaml.MappingEndEvent())
        else:
            raise ToolError("raw YAML contains an unsupported source node")

    emitter.emit(yaml.StreamStartEvent())
    emitter.emit(yaml.DocumentStartEvent(explicit=False))
    emit_node(root, ())
    emitter.emit(yaml.DocumentEndEvent(explicit=False))
    emitter.emit(yaml.StreamEndEvent())
    return stream.getvalue()


def _same_values(left, right, changes=None, path="root", key_changes=None):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        renamed = {key: (key_changes or {}).get(field_path(path, key), key) for key in left}
        return (len(set(renamed.values())) == len(left) and set(renamed.values()) == right.keys() and all(
            _same_values(left[key], right[target], changes, field_path(path, key), key_changes)
            for key, target in renamed.items()))
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_values(a, b, changes, f"{path}[{index}]", key_changes)
            for index, (a, b) in enumerate(zip(left, right)))
    if isinstance(left, Decimal):
        return left.as_tuple() == right.as_tuple()
    if isinstance(left, str) and changes:
        return left == right or (path, left, right) in changes
    return left == right


def _same_graph(left, right, changes=None):
    """A bijection protects scalar aliases as well as collection/merge reuse."""
    yaml = yaml_backend()
    forward, backward = {}, {}

    def compare(a, b):
        if id(a) in forward or id(b) in backward:
            return forward.get(id(a)) == id(b) and backward.get(id(b)) == id(a)
        forward[id(a)], backward[id(b)] = id(b), id(a)
        if (type(a) is not type(b) or a.tag != b.tag or
                getattr(a, "anchor", None) != getattr(b, "anchor", None) or
                getattr(a, "explicit_tag", None) != getattr(b, "explicit_tag", None)):
            return False
        if isinstance(a, yaml.ScalarNode):
            # Preserve the original decimal lexeme, not just its numeric value.
            return (changes or {}).get(id(a), a.value) == b.value
        if isinstance(a, yaml.SequenceNode):
            return len(a.value) == len(b.value) and all(compare(x, y) for x, y in zip(a.value, b.value))
        if isinstance(a, yaml.MappingNode):
            if len(a.value) != len(b.value):
                return False
            remaining = {}
            for key, value in reversed(b.value):
                remaining.setdefault((key.tag, key.value), []).append((key, value))
            for key, value in a.value:
                matches = remaining.get((key.tag, (changes or {}).get(id(key), key.value)), [])
                if not matches:
                    return False
                other, child = matches.pop()
                if not compare(key, other) or not compare(value, child):
                    return False
            return True
        return False

    return compare(left, right)


def _verified_render(document, node_changes, value_changes, key_changes=None):
    candidate = _render(document, node_changes)
    reread = loads(candidate, document.name, path="<formatted candidate>")
    if (not _same_values(document.value, reread.value, value_changes, document.name, key_changes) or
            not _same_graph(document.node, reread.node, node_changes)):
        raise ToolError(f"{document.name}: formatting would change a scalar, node, anchor or alias/merge relation")
    return candidate, reread


def _final_path(context, document, key_changes):
    original = final = document.name
    for part in context.parts:
        if type(part) is int:
            original, final = f"{original}[{part}]", f"{final}[{part}]"
        else:
            original = field_path(original, part)
            final = field_path(final, key_changes.get(original, part))
    return final


def _style_corrections(before, after, key_changes, node_changes):
    """One quote warning per field and one layout warning per changed document."""
    written = {(use.path, use.key): use for use in after.occurrences if use.alias is None}
    quotes = {}
    yaml = yaml_backend()
    for use in before.occurrences:
        if use.alias is not None or not isinstance(use.node, yaml.ScalarNode):
            continue
        path = _final_path(use, before, key_changes)
        target = written.get((path, use.key))
        if target is None or not isinstance(target.node, yaml.ScalarNode):
            continue
        old = use.node.style in {"'", '"'}
        new = target.node.style in {"'", '"'}
        if old != new:
            quotes[path] = ("quote: added necessary YAML wrapper quotes for this field's formatted context" if new else
                            "quote: removed unnecessary YAML wrapper quotes while preserving the string value")
    changes = [Diagnostic(path, message, severity="warning") for path, message in sorted(quotes.items())]

    def skeleton(document, replacements):
        result, cursor = [], 0
        for use in document.occurrences:
            if use.alias is not None or not isinstance(use.node, yaml.ScalarNode) or use.start < cursor:
                continue
            result.append(document.source[cursor:use.start])
            result.append("<key:" + replacements.get(id(use.node), use.node.value) + ">" if use.key else "<value>")
            cursor = use.end
        result.append(document.source[cursor:])
        return "".join(result)

    if skeleton(before, node_changes) != skeleton(after, {}):
        changes.append(Diagnostic(before.name, "layout: standardized field order and YAML layout", severity="warning"))
    elif not changes and not node_changes and before.source != after.source and before.value == after.value:
        changes.append(Diagnostic(before.name, "layout: standardized YAML scalar layout", severity="warning"))
    return changes


def _shown_change(before, after):
    """Show the first changed Unicode character in the decoded scalar."""
    if max(len(before), len(after)) <= 100:
        return f"{before!r} → {after!r}"
    offset = next((index for index, (left, right) in enumerate(zip(before, after)) if left != right),
                  min(len(before), len(after)))
    start, end = max(0, offset - 24), offset + 72

    def excerpt(value):
        return repr(("…" if start else "") + value[start:end] + ("…" if end < len(value) else ""))

    return f"first change at character {offset + 1}: {excerpt(before)} → {excerpt(after)}"


@bounded
def format_document(document: Document, *, skip_hardware=False, corrections=None, edits=()) -> str:
    """Return a verified canonical candidate; never write or mutate a document.

    Callers supply any proven field/declaration/reference edits and own writes.
    Full schema structure and length errors do not gate this source operation.
    Comments are invalid schema source and raise SchemaError rather than being
    silently discarded. Missing raw source or failed preservation/idempotence
    checks raise ToolError; no partially verified candidate is returned.
    Explicit case-only edits and definite prose periods are the only permitted
    string differences. Append warnings to corrections after verification; the
    caller decides when a candidate is written and a correction is reported.
    """
    if document.node is None or not document.source:
        raise ToolError("formatting needs the original Document returned by yaml_source.load/loads")
    if document.comments:
        positions = ", ".join(f"{item.line}:{item.column}" for item in document.comments)
        raise SchemaError(f"{document.name}: YAML comments are not allowed by DOC0 ({positions}); "
                          "formatting refused to discard them; move design facts into their defined fields")
    yaml = yaml_backend()
    try:
        working = document
        spelling, key_changes, values, requested = _edit_plan(document, edits)
        if spelling:
            _, working = _verified_render(document, spelling, values, key_changes)
        periods = [change for change in sentence_period_changes(working, skip_hardware=skip_hardware) if change.safe]
        node_changes = {id(change.node): change.value for change in periods}
        value_changes = {(path, change.node.value, change.value) for change in periods for path in change.paths}
        candidate, reread = _verified_render(working, node_changes, value_changes)
        further_changes = {id(change.node): change.value
                           for change in sentence_period_changes(reread, skip_hardware=skip_hardware) if change.safe}
        if _render(reread, further_changes) != candidate:
            raise ToolError(f"{document.name}: formatting did not reach an idempotent result")
        if corrections is not None:
            changes = {(_final_path(context, document, key_changes), context.node.value, spelling[identity])
                       for identity, contexts in requested.items() for context in contexts}
            corrections.extend(Diagnostic(path, "identifier+reference: " + _shown_change(before, after),
                                          hint="Review the normalized declaration and reference spellings", severity="warning")
                               for path, before, after in sorted(changes))
            corrections.extend(Diagnostic(path, "period: " + _shown_change(change.node.value, change.value),
                                          hint="Only the terminal period was removed; other text and whitespace were preserved",
                                          severity="warning") for change in periods for path in change.paths)
            if candidate != document.source:
                corrections.extend(_style_corrections(document, reread, key_changes, spelling))
        return candidate
    except (ReadError, yaml.YAMLError) as error:
        raise ToolError(f"{document.name}: could not produce a verified YAML format candidate: {error}") from error
