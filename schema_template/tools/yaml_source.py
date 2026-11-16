"""Read YAML data and real source marks without executing tags or writing files."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import hashlib
from io import StringIO
import json
from pathlib import Path
import re
import shlex
import sys

from .diagnostics import Diagnostic, field_path
from .errors import CapacityError, bounded

MAX_EXPANDED_NODES = 100000


class ReadError(Exception):
    def __init__(self, message, *, quote_repair=None):
        super().__init__(message)
        self.quote_repair = quote_repair


class DependencyError(Exception):
    pass


@dataclass(frozen=True)
class SourceOccurrence:
    """One written node or alias, rather than an expanded copy of its value."""

    node: object
    path: str
    parts: tuple[str | int, ...]
    start: int
    end: int
    line: int
    column: int
    flow: bool
    key: bool
    alias: str | None = None


@dataclass(frozen=True)
class ScalarContext:
    """One scalar use, including uses reached through collection reuse."""

    node: object
    path: str
    parts: tuple[str | int, ...]
    key: bool


@dataclass(frozen=True)
class _QuoteRepair:
    start: int
    end: int
    value: str
    parts: tuple[str | int, ...]


@dataclass(frozen=True)
class Comment:
    start: int
    end: int
    line: int
    column: int
    text: str
    path: str
    parts: tuple[str | int, ...]


@dataclass
class Document:
    value: dict
    positions: dict[str, tuple[int, int]]
    values: dict[str, object]
    source_sha256: str
    name: str = "root"
    source: str = ""
    node: object | None = None
    anchors: dict[str, object] = field(default_factory=dict)
    occurrences: tuple[SourceOccurrence, ...] = ()
    comments: tuple[Comment, ...] = ()


@lru_cache(maxsize=1)
def _backend():
    """Keep one resolver for loading, quote decisions and event emission."""
    try:
        import yaml
    except ModuleNotFoundError as error:
        command = f"{shlex.quote(sys.executable)} -m pip install -r {shlex.quote(str(Path(__file__).with_name('requirements.txt')))}"
        raise DependencyError(f"PyYAML is missing. Install it in this Python environment:\n{command}\nThen rerun the same check command.") from error

    class CoreResolver(yaml.resolver.Resolver):
        pass

    # Dates and on/off are strings. Numbers never pass through binary floats.
    removed = {"tag:yaml.org,2002:" + t for t in ("bool", "timestamp", "int", "float")}
    CoreResolver.yaml_implicit_resolvers = {
        k: [(tag, regex) for tag, regex in entries if tag not in removed]
        for k, entries in CoreResolver.yaml_implicit_resolvers.items()
    }
    CoreResolver.add_implicit_resolver("tag:yaml.org,2002:bool", re.compile(r"^(?:true|false)$", re.I), list("tTfF"))
    CoreResolver.add_implicit_resolver("tag:yaml.org,2002:int", re.compile(r"^[-+]?(?:0[bB][01_]+|0[oO][0-7_]+|0[xX][0-9a-fA-F_]+|[0-9][0-9_]*)$"), list("-+0123456789"))
    CoreResolver.add_implicit_resolver("tag:yaml.org,2002:float", re.compile(r"^[-+]?(?:(?:[0-9][0-9_]*\.[0-9_]*|\.[0-9_]+)(?:[eE][-+]?[0-9]+)?|[0-9][0-9_]*[eE][-+]?[0-9]+|\.(?:inf|nan))$", re.I), list("-+0123456789."))

    class Loader(yaml.SafeLoader):
        yaml_implicit_resolvers = CoreResolver.yaml_implicit_resolvers

        def __init__(self, source):
            super().__init__(source)
            self.source_uses = []
            self.source_attempts = []
            self.source_tokens = []
            self.source_anchors = {}
            self.source_stack = []
            self.error_parts = None

        def get_token(self):
            token = super().get_token()
            if token is not None:
                self.source_tokens.append(token)
            return token

        def compose_node(self, parent, index):
            parts, flow = self.source_stack[-1] if self.source_stack else ((), False)
            is_key = isinstance(parent, yaml.MappingNode) and index is None
            if isinstance(parent, yaml.MappingNode) and index is not None:
                parts += (index.value,)
            elif isinstance(parent, yaml.SequenceNode):
                parts += (index,)
            # Establish the field before asking the scanner for its value: a
            # missing quote can make peek_event itself fail.
            try:
                event = self.peek_event()
                self.source_attempts.append((event, parent, index, parts, flow, is_key))
                self.source_stack.append((parts, flow or bool(getattr(event, "flow_style", False))))
                try:
                    node = super().compose_node(parent, index)
                finally:
                    self.source_stack.pop()
                alias = event.anchor if isinstance(event, yaml.AliasEvent) else None
                if not isinstance(event, yaml.AliasEvent):
                    node.anchor = event.anchor
                    node.explicit_tag = event.tag
                    if event.anchor is not None:
                        self.source_anchors[event.anchor] = node
                if is_key and isinstance(node, yaml.ScalarNode):
                    parts += (node.value,)
                self.source_uses.append((node, event, parent, index, parts, flow, is_key, alias))
                return node
            except yaml.YAMLError:
                if self.error_parts is None:
                    self.error_parts = parts
                raise

    return yaml, Loader, CoreResolver


def yaml_backend():
    """The existing PyYAML dependency; imported lazily for useful CLI errors."""
    return _backend()[0]


def scalar_resolver():
    """A resolver with precisely the rules used by load/loads."""
    return _backend()[2]()


def plain_scalar_allowed(node, *, flow=False, key=False):
    """Whether this node can be unquoted in its actual YAML context.

    Explicit standard tags already establish a scalar's type. Otherwise the
    implicit resolver must recover its current tag. PyYAML's emitter supplies
    the context-specific lexical check, including key and flow restrictions.
    """
    yaml = yaml_backend()
    if not isinstance(node, yaml.ScalarNode):
        return False
    explicit = getattr(node, "explicit_tag", None)
    if explicit is None and scalar_resolver().resolve(yaml.ScalarNode, node.value, (True, False)) != node.tag:
        return False
    analysis = yaml.emitter.Emitter(StringIO(), allow_unicode=True).analyze_scalar(node.value)
    if key and (analysis.empty or analysis.multiline):
        return False
    return analysis.allow_flow_plain if flow else analysis.allow_block_plain


def _field(name, parts):
    for part in parts:
        name = f"{name}[{part}]" if isinstance(part, int) else field_path(name, part)
    return name


@bounded
def scalar_contexts(document: Document) -> tuple[ScalarContext, ...]:
    """Enumerate every field context without losing aliases or merge defaults.

    SourceOccurrence describes physical syntax. Normalizing a reused node
    additionally needs every location that it supplies, including overridden
    merge values. Each node's original path is retained for edit authorization.
    """
    yaml = yaml_backend()
    contexts, active = [], set()
    expanded = 0

    def visit(node, parts=(), *, key=False):
        nonlocal expanded
        if id(node) in active:
            raise ReadError(f"{document.name}: recursive aliases do not describe a finite plan")
        if not key:
            expanded += 1
            if expanded > MAX_EXPANDED_NODES:
                raise CapacityError(f"{document.name}: source uses exceed the parser's capacity ({MAX_EXPANDED_NODES} nodes)")
        active.add(id(node))
        try:
            if isinstance(node, yaml.ScalarNode):
                contexts.append(ScalarContext(node, _field(document.name, parts), parts, key))
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
            else:
                raise ReadError(f"{document.name}: scalar contexts need an ordinary YAML source node")
        finally:
            active.remove(id(node))

    visit(document.node)
    return tuple(contexts)


def _comments(text, tokens, occurrences, name):
    """Find only hashes outside scanner tokens; block headers are not text."""
    yaml = yaml_backend()
    breaks = list(re.finditer(r"\r\n|[\r\n\x85\u2028\u2029]", text))
    line_starts = [0] + [match.end() for match in breaks]
    line_ends = [match.start() for match in breaks] + [len(text)]
    use_starts = [use.start for use in occurrences]
    protected = []
    for token in tokens:
        start, end = token.start_mark.index, token.end_mark.index
        if isinstance(token, yaml.tokens.ScalarToken) and token.style in {"|", ">"}:
            # The scanner includes the header in this token. A header comment
            # is real YAML commentary, whereas every hash in the body is data.
            following = bisect_right(line_starts, start)
            start = min(end, line_starts[following]) if following < len(line_starts) else end
        if end > start:
            protected.append((start, end))
    protected.sort()
    comments, cursor = [], 0
    for match in re.finditer(r"#", text):
        start = match.start()
        while cursor < len(protected) and protected[cursor][1] <= start:
            cursor += 1
        if cursor < len(protected) and protected[cursor][0] <= start < protected[cursor][1]:
            continue
        if comments and start < comments[-1].end:
            continue
        line_index = bisect_right(line_starts, start) - 1
        end = line_ends[line_index]
        line, column = line_index + 1, start - line_starts[line_index] + 1
        preceding = bisect_right(use_starts, start) - 1
        use = occurrences[preceding] if preceding >= 0 else None
        if use is not None and use.line != line:
            use = None
        comments.append(Comment(start, end, line, column, text[start:end],
                                use.path if use else name, use.parts if use else ()))
    return tuple(comments)


@lru_cache(maxsize=8192)
def _normalization_fields(parts):
    """Use the structural validators' spelling/type metadata only on opt-in recovery."""
    from .rules.structure import CANONICAL_FIELDS, LITERAL_TYPES, STRING_FIELDS
    from .yaml_format import _matches, canonical_parts
    normalized = canonical_parts(parts, CANONICAL_FIELDS)
    kinds = {kind for pattern, kind in LITERAL_TYPES if _matches(normalized, pattern)}
    if any(_matches(normalized, pattern) for pattern in STRING_FIELDS):
        kinds.add(str)
    return normalized, kinds.pop() if len(kinds) == 1 else None


def _notation_kind(parts):
    normalized, _ = _normalization_fields(parts)
    from .yaml_format import _matches
    prefix = ("DESIGN", "Tile_pipeline", "Step", None)
    if _matches(normalized, prefix + ("Event", None, int)):
        return "event"
    if _matches(normalized, prefix + ("Role", None, "Waiting", None, int)) and normalized[-2] in {
            "Prepared", "Released", "Allowed"}:
        return "waiting"
    return None


def _complete_notation(value, kind):
    """Recognize wrappers with the existing grammar; spelling binds in a later phase.

    The scratch spelling only lets the syntax parser recognize capitalization.
    The returned source repair always quotes the *original* untouched text.
    """
    from .pipeline_syntax import PipelineSyntaxError, parse_event, parse_waiting, split_top

    def event(text):
        scratch = text.lower()
        scratch = re.sub(r"(\{\s*(?:issue|done)\s+)([a-z_][a-z0-9_]*)(?=\s*(?:\[|::))",
                         lambda match: match[1] + match[2].upper(), scratch)
        parse_event(scratch)
        return scratch

    try:
        if kind == "event":
            event(value)
        elif kind == "waiting":
            parts = split_top(value, "←")
            if parts is None or len(parts) != 2 or not parts[1].strip().startswith("[") or not parts[1].strip().endswith("]"):
                return False
            predecessors = split_top(parts[1].strip()[1:-1], ",")
            if not predecessors:
                return False
            scratch = event(parts[0].strip()) + " ← [" + ", ".join(event(part.strip()) for part in predecessors) + "]"
            parse_waiting(scratch)
        else:
            return False
    except PipelineSyntaxError:
        return False
    return True


def _notation_span(start, parts, flow, text):
    """Bracket grammar proves the complete string boundary, including in flow lists."""
    from .pipeline_syntax import _group_end
    kind = _notation_kind(parts)
    if kind is None or text[start:start + 1] != "{":
        return None
    end = _group_end(text, start)
    if end is None:
        return None
    end += 1
    if kind == "waiting":
        arrow = re.match(r"[ \t]*←[ \t]*\[", text[end:])
        if arrow is None:
            return None
        end = _group_end(text, end + arrow.end() - 1)
        if end is None:
            return None
        end += 1
    value = text[start:end]
    if any(char in value for char in "\r\n#") or not _complete_notation(value, kind):
        return None
    rest = text[end:]
    if flow:
        if not re.match(r"[ \t]*[,\]]", rest):
            return None
    else:
        line_start = max((match.end() for match in re.finditer(r"\r\n|[\r\n\x85\u2028\u2029]", text[:start])), default=0)
        if not re.fullmatch(r" *- +", text[line_start:start]) or not re.match(r"[ \t]*(?:[\r\n\x85\u2028\u2029]|$)", rest):
            return None
    return _QuoteRepair(start, end, value, parts)


def _notation_repair(attempts, text):
    """One scanner-established list value and a complete Event/Waiting grammar."""
    yaml = yaml_backend()
    for event, parent, _, parts, flow, key in attempts:
        if (key or not isinstance(parent, yaml.SequenceNode) or
                getattr(event, "anchor", None) is not None or getattr(event, "tag", None) is not None):
            continue
        repair = _notation_span(event.start_mark.index, parts, flow, text)
        if repair is not None:
            return repair
    return None


def _quote_hint(error, loader, text, name):
    """Add a conditional hint only for a known textual field and clear syntax."""
    mark = getattr(error, "problem_mark", None)
    if loader is None or mark is None:
        return ""
    def expects_text(parts):
        return _normalization_fields(parts)[1] is str
    parts = loader.error_parts
    if parts is None or not expects_text(parts):
        # The scanner may discover the second colon only after returning the
        # first plain scalar. Its already-composed path is still authoritative.
        completed = [(event.end_mark.index, candidate)
                     for _, event, _, _, candidate, _, is_key, alias in loader.source_uses
                     if isinstance(event, yaml_backend().ScalarEvent) and not is_key and alias is None
                     and event.style is None and event.start_mark.line == mark.line
                     and event.end_mark.index <= mark.index and expects_text(candidate)]
        parts = max(completed, key=lambda item: item[0])[1] if completed else None
    if parts is None or not expects_text(parts):
        return ""
    problem = getattr(error, "problem", "") or ""
    if not ("mapping values are not allowed" in problem or "cannot start any token" in problem or
            "undefined alias" in problem):
        return ""
    line = text.splitlines()[mark.line] if mark.line < len(text.splitlines()) else ""
    # Do not diagnose a quoted string, explicit tag or block header as lacking
    # its quotes. This is a hint about the shown fragment, not a repaired parse.
    _, separator, value = line.partition(":")
    if not separator or not value.strip() or value.lstrip().startswith(("'", '"', "|", ">", "!")):
        return ""
    return (f"; if the value at {_field(name, parts)} is intended as this text, "
            "add the necessary YAML quotes or use a legal block scalar")


def _missing_quote_repair(error, loader, text):
    """Recognize a single unquoted block value, without reparsing broken YAML.

    Known text type plus a scanner-established simple key provide the bounds.
    Undefined aliases, existing quotes, properties, comments and flow syntax
    remain errors: they admit other interpretations than missing wrappers.
    """
    mark = getattr(error, "problem_mark", None)
    problem = getattr(error, "problem", "") or ""
    if loader is None or mark is None or not (
        "mapping values are not allowed" in problem or "cannot start any token" in problem
    ):
        return None
    def expects_text(parts):
        return _normalization_fields(parts)[1] is str
    yaml = yaml_backend()
    keys = [(node, event, parts) for node, event, parent, _, parts, flow, key, alias in loader.source_uses
            if key and not flow and alias is None and isinstance(parent, yaml.MappingNode)
            and isinstance(node, yaml.ScalarNode) and node.tag == "tag:yaml.org,2002:str"
            and event.start_mark.line == mark.line and event.end_mark.line == mark.line
            and event.end_mark.index <= mark.index and expects_text(parts)]
    if not keys:
        return None
    _, event, parts = max(keys, key=lambda item: item[1].end_mark.index)
    line_start = max((match.end() for match in re.finditer(r"\r\n|[\r\n\x85\u2028\u2029]", text[:event.start_mark.index])), default=0)
    # Exclude explicit keys and every inline container, even if an earlier
    # scanner event happened to have the same expected field spelling.
    if not re.fullmatch(r" *(?:- +)?", text[line_start:event.start_mark.index]):
        return None
    separator = re.match(r":[ \t]+", text[event.end_mark.index:])
    if separator is None:
        return None
    start = event.end_mark.index + separator.end()
    ending = re.search(r"[\r\n\x85\u2028\u2029]", text[start:])
    end = start + ending.start() if ending else len(text)
    end = len(text[:end].rstrip(" \t"))
    value = text[start:end]
    if not value or "#" in value or value.startswith(("'", '"', "!", "&", "*", "|", ">", "[", "{")):
        return None
    if "mapping values are not allowed" in problem:
        if text[mark.index:mark.index + 1] != ":" or not any(
            isinstance(scalar_event, yaml.ScalarEvent) and not key and alias is None and not flow
            and scalar_event.style is None and scalar_event.anchor is None and scalar_event.tag is None
            and candidate == parts and scalar_event.start_mark.index == start
            and scalar_event.end_mark.index == mark.index
            for _, scalar_event, _, _, candidate, flow, key, alias in loader.source_uses
        ):
            return None
    elif mark.index != start or value[0] not in {"@", "`"}:
        return None
    return _QuoteRepair(start, end, value, parts)


def repair_missing_quotes(text: str, name: str, *, path: Path | str = "<memory>", corrections=None,
                          skip_hardware=False) -> str:
    """Collect independently proven local wrappers, returning only a strict parse.

    This opt-in helper never writes files and never changes load/loads policy.
    Its caller retains the original source bytes/hash until candidate writes;
    the returned text is a separate source snapshot. Each scanner-established
    field/line must independently prove a unique repair. No candidate or
    correction escapes if any remaining syntax needs guessing.
    """
    pending, attempted = [], set()
    while True:
        try:
            document = loads(text, name, path=path)
            break
        except ReadError as error:
            repair = error.quote_repair
            if (repair is None or repair.parts in attempted or
                    skip_hardware and _normalization_fields(repair.parts)[0][:2] == ("CONTRACT", "Hardware_specification")):
                raise
            attempted.add(repair.parts)
            pending.append(repair)
            text = text[:repair.start] + json.dumps(repair.value, ensure_ascii=False) + text[repair.end:]
    # A complete unquoted event can itself be valid YAML flow-map syntax.
    # Its known string slot and the complete notation establish the type;
    # arbitrary mappings or lists in other fields are never converted to text.
    yaml = yaml_backend()
    candidates = []
    for use in document.occurrences:
        if (use.alias is None and not use.key and
                isinstance(use.node, yaml.MappingNode) and _notation_kind(use.parts)):
            if getattr(use.node, "anchor", None) is None and getattr(use.node, "explicit_tag", None) is None:
                repair = _notation_span(use.start, use.parts, use.flow, text)
                if repair is not None:
                    candidates.append(repair)
    for repair in sorted(candidates, key=lambda item: item.start, reverse=True):
        text = text[:repair.start] + json.dumps(repair.value, ensure_ascii=False) + text[repair.end:]
    reread = loads(text, name, path=path) if candidates else document
    for repair in pending + candidates:
        value = reread.value
        try:
            for part in repair.parts:
                value = value[part]
        except (KeyError, IndexError, TypeError) as error:
            raise ReadError(f"{path}: necessary text quotes could not be verified") from error
        if type(value) is not str or value != repair.value:
            raise ReadError(f"{path}: necessary text quotes could not be verified")
    if corrections is not None:
        corrections.extend(Diagnostic(
            _field(name, repair.parts), "quote: added necessary YAML quotes around " +
            ("complete event notation" if _notation_kind(repair.parts) else "a single-line text value"),
            hint="Review the original text; its wrappers were added and the final candidate parsed strictly",
            severity="warning",
        ) for repair in pending + candidates)
    return text


def recover_scalar_quotes(document: Document, *, corrections=None, skip_hardware=False) -> Document:
    """Repair quote wrappers only when all uses require one known scalar type.

    This source-stage authorization is deliberately separate from ScalarEdit:
    ordinary formatting never changes scalar types. Only string-only fields
    and strict integer/boolean leaves qualify; unions, null, explicit tags and
    mixed alias uses retain their original types.
    """
    yaml = yaml_backend()
    tags = {str: "tag:yaml.org,2002:str", int: "tag:yaml.org,2002:int", bool: "tag:yaml.org,2002:bool"}
    uses = {}
    for context in scalar_contexts(document):
        uses.setdefault(id(context.node), []).append(context)
    replacements, expected = [], {}
    for identity, contexts in uses.items():
        node = contexts[0].node
        if (getattr(node, "explicit_tag", None) is not None or
                any(context.key or
                    skip_hardware and _normalization_fields(context.parts)[0][:2] == ("CONTRACT", "Hardware_specification")
                    for context in contexts)):
            continue
        kinds = {_normalization_fields(context.parts)[1] for context in contexts}
        if len(kinds) != 1 or (kind := kinds.pop()) not in tags:
            continue
        if kind is str:
            if node.tag not in {tags[bool], tags[int], "tag:yaml.org,2002:float"} or node.style is not None:
                continue
            token, replacement = node.value, json.dumps(node.value, ensure_ascii=False)
        else:
            if (node.tag != tags[str] or node.style not in {"'", '"'} or
                    node.value != node.value.strip() or
                    scalar_resolver().resolve(yaml.ScalarNode, node.value, (True, False)) != tags[kind]):
                continue
            token, replacement = node.style + node.value + node.style, node.value
        # The scalar node marks exclude an anchor property; patch only its
        # literal token so aliases and all graph sharing remain intact.
        written = next((use for use in document.occurrences if id(use.node) == identity and use.alias is None), None)
        if written is None:
            raise ReadError(f"{document.name}: cannot locate a scalar for quote recovery")
        fragment = document.source[written.start:written.end]
        # For quoted values this also excludes escape sequences and physically
        # folded text: removing just the wrappers must retain the exact lexeme.
        if not fragment.endswith(token) or any(c in token for c in "\r\n"):
            continue
        if kind is not str:
            try:
                literal = loads("value: " + node.value + "\n", document.name, path="<scalar quote recovery>")
            except ReadError:
                continue
            if type(literal.value["value"]) is not kind:
                continue
        replacements.append((written.end - len(token), written.end, replacement))
        expected[identity] = tags[kind]
    if not replacements:
        return document
    candidate = document.source
    for start, end, replacement in sorted(replacements, reverse=True):
        candidate = candidate[:start] + replacement + candidate[end:]
    reread = loads(candidate, document.name, path="<scalar quote recovery>")
    forward, backward = {}, {}

    def same(before, after):
        if id(before) in forward or id(after) in backward:
            return forward.get(id(before)) == id(after) and backward.get(id(after)) == id(before)
        forward[id(before)], backward[id(after)] = id(after), id(before)
        if (type(before) is not type(after) or getattr(before, "anchor", None) != getattr(after, "anchor", None) or
                getattr(before, "explicit_tag", None) != getattr(after, "explicit_tag", None)):
            return False
        tag = expected.get(id(before), before.tag)
        if tag != after.tag:
            return False
        if isinstance(before, yaml.ScalarNode):
            return before.value == after.value
        if isinstance(before, yaml.SequenceNode):
            return len(before.value) == len(after.value) and all(same(a, b) for a, b in zip(before.value, after.value))
        if isinstance(before, yaml.MappingNode):
            return len(before.value) == len(after.value) and all(
                same(a, c) and same(b, d) for (a, b), (c, d) in zip(before.value, after.value))
        return False

    if not same(document.node, reread.node):
        raise ReadError(f"{document.name}: scalar quote recovery changed an unauthorized scalar or YAML reuse relation")
    if corrections is not None:
        for identity, tag in expected.items():
            if tag == tags[str]:
                message = "quote: quoted the original boolean/number lexeme in a string-only field"
                hint = "The original literal text was preserved"
            else:
                label = "integer" if tag == tags[int] else "boolean"
                message = f"quote: removed quotes around the YAML {label} literal {uses[identity][0].node.value!r}"
                hint = "Only the quote wrappers were removed; field value rules are checked separately"
            for context in uses[identity]:
                corrections.append(Diagnostic(context.path, message, hint=hint, severity="warning"))
    return reread


def load(path: Path, name: str) -> Document:
    """Parse a file with the same rules as the in-memory entry point."""
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReadError(f"{path}: YAML must be valid UTF-8") from error
    return loads(text, name, path=path)


def loads(text: str, name: str, *, path: Path | str = "<memory>") -> Document:
    """Parse exact UTF-8 text without temporary files or alternate scalar rules."""
    if not isinstance(text, str):
        raise ReadError(f"{path}: YAML input must be text")
    yaml, Loader, _ = _backend()

    def problem(message, mark=None):
        position = f":{mark.line + 1}:{mark.column + 1}" if mark else ""
        return ReadError(f"{path}{position}: {message}")

    def scalar(node):
        tag = node.tag.removeprefix("tag:yaml.org,2002:")
        value = node.value
        if tag == "str":
            return value
        if tag == "null":
            return None
        if tag == "bool" and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        value = value.replace("_", "")
        try:
            if tag == "int":
                unsigned = value.lstrip("+-").lower()
                return int(value, 0 if unsigned.startswith(("0x", "0o", "0b")) else 10)
            if tag == "float":
                number = Decimal(value)
                if number.is_finite():
                    return number
        except (ValueError, InvalidOperation):
            pass
        raise problem(f"unsupported scalar or non-finite number ({node.tag}); use ordinary YAML values", node.start_mark)

    positions, values, active = {}, {}, set()
    expanded = 0

    def convert(node, field, mark=None):
        nonlocal expanded
        if id(node) in active:
            raise problem("recursive aliases do not describe a finite plan", node.start_mark)
        expanded += 1
        if expanded > MAX_EXPANDED_NODES:
            raise CapacityError(f"{path}: YAML expansion exceeds the parser's capacity ({MAX_EXPANDED_NODES} nodes)")
        active.add(id(node))
        try:
            if isinstance(node, yaml.ScalarNode):
                result = scalar(node)
            elif isinstance(node, yaml.SequenceNode) and node.tag == "tag:yaml.org,2002:seq":
                result = [convert(item, f"{field}[{i}]", use_marks.get((id(node), i)))
                          for i, item in enumerate(node.value)]
            elif isinstance(node, yaml.MappingNode) and node.tag == "tag:yaml.org,2002:map":
                result, explicit, merges, seen = {}, [], [], set()
                saw_merge = False
                for key, value in node.value:
                    if key.tag == "tag:yaml.org,2002:merge":
                        if saw_merge:
                            raise problem("duplicate mapping key '<<'; use one merge with an ordered list", key.start_mark)
                        saw_merge = True
                        if isinstance(value, yaml.SequenceNode) and value.tag != "tag:yaml.org,2002:seq":
                            raise problem("use an ordinary sequence for a YAML merge; custom or mismatched tags are not supported", value.start_mark)
                        entries = value.value if isinstance(value, yaml.SequenceNode) else [value]
                        if not all(isinstance(item, yaml.MappingNode) for item in entries):
                            raise problem("a YAML merge needs mappings", value.start_mark)
                        merges.extend(entries)
                        continue
                    if not isinstance(key, yaml.ScalarNode) or key.tag != "tag:yaml.org,2002:str":
                        raise problem("mapping keys must be strings; if this key is literal text, "
                                      "add the necessary YAML quotes", key.start_mark)
                    if key.value in seen:
                        raise problem(f"duplicate mapping key {key.value!r}; keep one definition", key.start_mark)
                    seen.add(key.value)
                    explicit.append((key, value))
                # YAML merges are local value reuse. Explicit fields override
                # merged defaults; explicit duplicate keys still fail above.
                for merged in reversed(merges):
                    result.update(convert(merged, field))
                for key, value in explicit:
                    location = use_marks.get((id(node), id(key)))
                    result[key.value] = convert(value, field_path(field, key.value), location)
            else:
                raise problem("use ordinary mappings, sequences and scalar values; custom tags are not supported", node.start_mark)
            where = mark or node.start_mark
            positions[field] = (where.line + 1, where.column + 1)
            values[field] = result
            return result
        finally:
            active.remove(id(node))

    loader = None
    try:
        source_bytes = text.encode("utf-8")
        loader = Loader(text)
        node = loader.get_node() if loader.check_node() else None
        if node is None or not isinstance(node, yaml.MappingNode) or loader.check_node():
            raise problem("expected exactly one YAML mapping document per node", node.start_mark if node else None)
        use_marks, occurrences = {}, []
        for item, event, parent, index, parts, flow, is_key, alias in loader.source_uses:
            end = event.end_mark if alias is not None or isinstance(item, yaml.ScalarNode) else item.end_mark
            occurrences.append(SourceOccurrence(item, _field(name, parts), parts,
                                                event.start_mark.index, end.index,
                                                event.start_mark.line + 1, event.start_mark.column + 1,
                                                flow, is_key, alias))
            if not is_key:
                edge = index if isinstance(parent, yaml.SequenceNode) else id(index)
                use_marks[(id(parent), edge)] = event.start_mark
        occurrences.sort(key=lambda use: (use.start, not use.key))
        value = convert(node, name)
        return Document(value, positions, values, hashlib.sha256(source_bytes).hexdigest(),
                        name, text, node, dict(loader.source_anchors), tuple(occurrences),
                        _comments(text, loader.source_tokens, occurrences, name))
    except yaml.YAMLError as error:
        message = (getattr(error, "problem", None) or str(error)) + _quote_hint(error, loader, text, name)
        failure = problem(message, getattr(error, "problem_mark", None))
        failure.quote_repair = _missing_quote_repair(error, loader, text)
        if failure.quote_repair is None and "undefined alias" not in message:
            failure.quote_repair = _notation_repair(loader.source_attempts, text) if loader is not None else None
        raise failure from error
    except ReadError as error:
        if error.quote_repair is None and loader is not None:
            error.quote_repair = _notation_repair(loader.source_attempts, text)
        raise
    except UnicodeError as error:
        raise problem("YAML must be valid UTF-8") from error
    except RecursionError as error:
        raise CapacityError(f"{path}: YAML nesting exceeds the parser's capacity") from error
    finally:
        if loader is not None:
            loader.dispose()
