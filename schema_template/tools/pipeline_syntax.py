"""Shared pipe, action and event notation, without evaluation or schema I/O.

``parse_action``, ``parse_event``, ``parse_waiting`` and ``parse_pipe`` return
small records and raise ``PipelineSyntaxError`` with the source field path.
An Action exposes ``family == (verb, noun)`` and ``signature`` including each
operand's arity. Event role/member omissions remain None; callers supply the
current context, or require both fields for an event alias.

``parse_pipe(value, scope=IndexScope(definitions))`` accepts a list, where
definitions maps index names directly to their domain definitions. Its ordered
items are ActionDeclaration, ForLoop and IfBlock. ``actions()`` visits every
declaration, including empty loops and false branches. ForLoop.scope/bound
describe the header's outer context; its body has the extended scope/bindings.
The body's bindings include restored flatten coordinates: finite expansion
must compare already-bound coordinates, never overwrite or discard conflicts.

Declaration existence, fixed family signatures, expression name resolution,
domain validity and event membership belong to consumers. Mathematical text is
opaque to the parser. The optional name helpers only inspect familiar syntax;
an expression they cannot understand is still valid parser input.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Iterator

from .diagnostics import field_path
from .text_policy import budget_message, numeric_form_message, single_names_message

VERBS = frozenset({"compute", "matmul", "load", "store", "await", "signal", "acquire", "release"})
DOMAIN_FUNCTIONS = frozenset({"count", "last"})
LOWER_NAME = re.compile(r"[a-z_][a-z0-9_]*\Z")
ROLE_NAME = re.compile(r"[A-Z_][A-Z0-9_]*\Z")
_PAIRS = {"(": ")", "[": "]", "{": "}", "⌊": "⌋", "⌈": "⌉", "⟨": "⟩"}


class PipelineSyntaxError(ValueError):
    def __init__(self, path: str, message: str):
        self.path, self.message = path, message
        super().__init__(f"{path}: {message}" if path else message)


def literal_integer(value: Any) -> int | None:
    """Recognize an integer literal without evaluating mathematical expressions."""
    if type(value) is int:
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value)
    return None


@dataclass(frozen=True)
class IndexScope:
    definitions: dict[str, Any]

    def extend(self, name: str, definition: Any) -> IndexScope:
        return IndexScope({**self.definitions, name: definition})

    def coordinates(self, name: str) -> set[str]:
        """Include recursively restored coordinates; consumers validate domains."""
        result, pending = set(), [name]
        while pending:
            current = pending.pop()
            if current in result:
                continue
            result.add(current)
            definition = self.definitions.get(current)
            if isinstance(definition, dict):
                pending.extend(definition.get("flatten", ()))
        return result


def same_domain(first: Any, second: Any) -> bool:
    if isinstance(first, list) and isinstance(second, list):
        def endpoint(value):
            literal = literal_integer(value)
            return value if literal is None else literal
        return list(map(endpoint, first)) == list(map(endpoint, second))
    return first == second


def domain_choices(scopes: list[IndexScope], name: str) -> list[Any]:
    values = []
    for scope in scopes:
        if name in scope.definitions and not any(same_domain(scope.definitions[name], value) for value in values):
            values.append(scope.definitions[name])
    return values


def split_top(text: str, delimiter: str) -> list[str] | None:
    """Split at unnested delimiters; return None for unbalanced brackets.

    Delimiters may contain multiple characters. Braces and common mathematical
    brackets count as groups, so event lists and nested arguments share this
    scanner without parsing an expression language.
    """
    if not delimiter:
        raise ValueError("delimiter must not be empty")
    stack, result, start, index = [], [], 0, 0
    while index < len(text):
        if not stack and text.startswith(delimiter, index):
            result.append(text[start:index])
            index += len(delimiter)
            start = index
            continue
        char = text[index]
        if char in _PAIRS:
            stack.append(_PAIRS[char])
        elif char in _PAIRS.values():
            if not stack or stack.pop() != char:
                return None
        index += 1
    return None if stack else result + [text[start:]]


def _group_end(text: str, start: int) -> int | None:
    stack = []
    for index in range(start, len(text)):
        char = text[index]
        if char in _PAIRS:
            stack.append(_PAIRS[char])
        elif char in _PAIRS.values():
            if not stack or stack.pop() != char:
                return None
            if not stack:
                return index
    return None


def _text(value: Any, path: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PipelineSyntaxError(path, f"{label} must be a nonempty string")
    return value.strip()


def _split(text: str, delimiter: str, path: str) -> list[str]:
    parts = split_top(text, delimiter)
    if parts is None:
        raise PipelineSyntaxError(path, "unbalanced brackets in notation")
    return [part.strip() for part in parts]


def _inside(text: str, opening: str, path: str, label: str) -> str:
    if not text.startswith(opening) or _group_end(text, 0) != len(text) - 1:
        raise PipelineSyntaxError(path, f"{label} must be enclosed in {opening}{_PAIRS[opening]}")
    return text[1:-1].strip()


def _name_budget(name: str, path: str) -> None:
    if message := budget_message(name, 50, 1):
        raise PipelineSyntaxError(path, "name: " + message)


def _integer_text(value: str, path: str) -> None:
    for message in (budget_message(value, 100, 1), numeric_form_message(value, integer=True),
                    single_names_message(value)):
        if message:
            raise PipelineSyntaxError(path, message)


def parse_members(value: Any) -> tuple[tuple[str, str, str | None], ...] | None:
    """Return (unit, lower, upper) paths, or None for invalid member notation.

    upper=None selects one member. Unit names are hardware-defined; endpoint
    expressions remain text, and may contain nested calls or indexed values.
    Numeric bounds, parameter definitions and hardware membership are separate
    checks. This helper does not parse event role/member qualifications.
    """
    if not isinstance(value, str) or not value.strip() or any(c in value for c in "\r\n"):
        return None
    result, remaining = [], value.strip()
    while remaining:
        opening = remaining.find("[")
        if opening < 1 or not remaining[:opening].strip() or "]" in remaining[:opening]:
            return None
        end = _group_end(remaining, opening)
        if end is None:
            return None
        bounds = split_top(remaining[opening + 1:end], ":")
        if bounds is None or len(bounds) not in (1, 2) or not all(part.strip() for part in bounds):
            return None
        result.append((remaining[:opening].strip(), bounds[0].strip(),
                       bounds[1].strip() if len(bounds) == 2 else None))
        remaining = remaining[end + 1:].strip()
        if remaining:
            if not remaining.startswith(".") or not remaining[1:].strip():
                return None
            remaining = remaining[1:].strip()
    return tuple(result)


@dataclass(frozen=True)
class Operand:
    name: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class Action:
    text: str
    path: str
    verb: str
    operands: tuple[Operand, ...]

    @property
    def noun(self) -> str:
        return "@".join(operand.name for operand in self.operands)

    @property
    def family(self) -> tuple[str, str]:
        return self.verb, self.noun

    @property
    def signature(self) -> tuple[str, tuple[tuple[str, int], ...]]:
        return self.verb, tuple((operand.name, len(operand.args)) for operand in self.operands)


def parse_action(value: Any, *, path: str = "action") -> Action:
    text = _text(value, path, "action")
    parts = _split(text, ":", path)
    if len(parts) != 2 or parts[0] not in VERBS:
        raise PipelineSyntaxError(path, "action must be verb:noun(arguments), using one of: " + ", ".join(sorted(VERBS)))
    verb, noun = parts
    operands = _split(noun, "@", path)
    if len(operands) not in ((1, 2) if verb == "matmul" else (1,)):
        raise PipelineSyntaxError(path, "only matmul may combine exactly two nouns with @")
    parsed = []
    for operand in operands:
        name, opening, arguments = operand.partition("(")
        name = name.strip()
        if not LOWER_NAME.fullmatch(name):
            raise PipelineSyntaxError(path, "noun bases must be lowercase identifiers")
        _name_budget(name, path)
        args = []
        if opening:
            contents = _inside("(" + arguments, "(", path, f"arguments of {name}")
            args = _split(contents, ",", path)
            if not all(args):
                raise PipelineSyntaxError(path, f"arguments of {name} must be nonempty; omit parentheses for no arguments")
            for argument in args:
                _integer_text(argument, path)
        parsed.append(Operand(name, tuple(args)))
    return Action(value, path, verb, tuple(parsed))


@dataclass(frozen=True)
class EventRef:
    text: str
    path: str
    time: str
    role: str | None
    member: str | None
    action: Action


def parse_event(value: Any, *, path: str = "event") -> EventRef:
    """Read explicit issue/done without filling omitted role or member fields."""
    text = _text(value, path, "event")
    contents = _inside(text, "{", path, "event")
    match = re.fullmatch(r"(issue|done)\s+(.+)", contents, re.DOTALL)
    if match is None:
        raise PipelineSyntaxError(path, "event must explicitly name issue or done: {issue reference} or {done reference}")
    time, reference = match.groups()
    parts = _split(reference, "::", path)
    role = member = None
    if len(parts) == 2:
        qualifier, reference = parts
        role, opening, selection = qualifier.partition("[")
        role = role.strip()
        if not ROLE_NAME.fullmatch(role):
            raise PipelineSyntaxError(path, "event role must be an uppercase identifier")
        _name_budget(role, path)
        if opening:
            member = _inside("[" + selection, "[", path, "event member")
            if not member or len(_split(member, ",", path)) != 1 or len(_split(member, ":", path)) != 1:
                raise PipelineSyntaxError(path, "event member must select one member or *")
            if member != "*":
                _integer_text(member, path)
            number = literal_integer(member)
            if number is not None and number < 0:
                raise PipelineSyntaxError(path, "event member must be nonnegative")
    elif len(parts) != 1:
        raise PipelineSyntaxError(path, "event may qualify one role with ROLE[member]::")
    return EventRef(value, path, time, role, member, parse_action(reference, path=path))


@dataclass(frozen=True)
class WaitingRule:
    text: str
    path: str
    target: EventRef
    predecessors: tuple[EventRef, ...]


def parse_waiting(value: Any, *, path: str = "waiting") -> WaitingRule:
    """Read one target and its nonempty predecessor list, retaining both times."""
    text = _text(value, path, "waiting rule")
    parts = _split(text, "←", path)
    if len(parts) != 2:
        raise PipelineSyntaxError(path, "waiting rule must be {target} ← [{predecessor}, ...] with one ←")
    contents = _inside(parts[1], "[", path, "waiting predecessor list")
    predecessors = _split(contents, ",", path)
    if not all(predecessors):
        raise PipelineSyntaxError(path, "waiting predecessor list must be nonempty, without empty entries")

    def event(text: str, label: str) -> EventRef:
        try:
            return parse_event(text, path=path)
        except PipelineSyntaxError as error:
            raise PipelineSyntaxError(path, f"{label}: {error.message}") from error

    target = event(parts[0], "waiting target")
    refs = tuple(event(part, f"waiting predecessor {index + 1}") for index, part in enumerate(predecessors))
    return WaitingRule(value, path, target, refs)


@dataclass(frozen=True)
class Slice:
    lower: str | None
    upper: str | None


@dataclass(frozen=True)
class ForSpec:
    text: str
    name: str
    selectors: tuple[str | Slice, ...] = ()
    bounds: tuple[str, str] | None = None


def _control(value: Any, path: str) -> tuple[str, str]:
    text = _text(value, path, "control key")
    match = re.match(r"(for|if)\s*(\()", text)
    if match is None:
        raise PipelineSyntaxError(path, "control key must be for(...) or if(...)")
    contents = _inside(text[match.start(2):], "(", path, f"{match[1]} control")
    if not contents:
        raise PipelineSyntaxError(path, f"{match[1]} control needs a nonempty expression")
    if match[1] == "if":
        for message in (numeric_form_message(contents, allow_empty_set=True), single_names_message(contents)):
            if message:
                raise PipelineSyntaxError(path, message)
    return match[1], contents


def parse_for(value: Any, *, path: str = "for") -> ForSpec:
    """Parse a complete for(...) key; range and selector expressions stay text."""
    kind, contents = _control(value, path)
    if kind != "for":
        raise PipelineSyntaxError(path, "expected a for(...) control")
    assignment = split_top(contents, "=")
    # _control has checked the enclosing group's balance. Keep the raw pieces
    # here so a later '=' inside a range endpoint does not alter its whitespace.
    assert assignment is not None
    if len(assignment) > 1:
        name = assignment[0].strip()
        bounds = _split("=".join(assignment[1:]), ":", path)
        if not LOWER_NAME.fullmatch(name) or len(bounds) != 2 or not all(bounds):
            raise PipelineSyntaxError(path, "local range must be for(name=lower:upper), with a lowercase index name")
        _name_budget(name, path)
        for endpoint in bounds:
            _integer_text(endpoint, path)
        return ForSpec(value, name, bounds=(bounds[0], bounds[1]))
    name, opening, selection = contents.partition("[")
    name = name.strip()
    if not LOWER_NAME.fullmatch(name):
        raise PipelineSyntaxError(path, "for domain must be a lowercase index name, optionally followed by a slice or window")
    _name_budget(name, path)
    selectors = []
    if opening:
        selection = _inside("[" + selection, "[", path, "for selection")
        for part in _split(selection, ",", path):
            if not part:
                raise PipelineSyntaxError(path, "for selection must not contain an empty coordinate")
            bounds = _split(part, ":", path)
            if len(bounds) == 1:
                _integer_text(part, path)
                selectors.append(part)
            elif len(bounds) == 2:
                for endpoint in bounds:
                    if endpoint:
                        _integer_text(endpoint, path)
                selectors.append(Slice(bounds[0] or None, bounds[1] or None))
            else:
                raise PipelineSyntaxError(path, "for slices have lower:upper bounds, without a step")
    return ForSpec(value, name, tuple(selectors))


@dataclass(frozen=True)
class ActionDeclaration:
    action: Action
    scope: IndexScope
    bound: frozenset[str]

    @property
    def text(self) -> str:
        return self.action.text

    @property
    def path(self) -> str:
        return self.action.path


@dataclass(frozen=True)
class ForLoop:
    spec: ForSpec
    body: ScopedPipe
    path: str
    scope: IndexScope
    bound: frozenset[str]


@dataclass(frozen=True)
class IfBlock:
    condition: str
    body: ScopedPipe
    path: str


@dataclass(frozen=True)
class ScopedPipe:
    items: tuple[ActionDeclaration | ForLoop | IfBlock, ...]
    path: str
    scope: IndexScope
    bound: frozenset[str]

    def walk(self) -> Iterator[ActionDeclaration | ForLoop | IfBlock]:
        """Visit nodes in declaration order, including all control bodies."""
        pending = list(reversed(self.items))
        while pending:
            item = pending.pop()
            yield item
            if isinstance(item, (ForLoop, IfBlock)):
                pending.extend(reversed(item.body.items))

    def actions(self) -> Iterator[ActionDeclaration]:
        return (item for item in self.walk() if isinstance(item, ActionDeclaration))

    def controls(self) -> Iterator[ForLoop | IfBlock]:
        return (item for item in self.walk() if isinstance(item, (ForLoop, IfBlock)))


def parse_pipe(value: Any, *, path: str = "Pipe", scope: IndexScope | None = None,
               bound: frozenset[str] = frozenset()) -> ScopedPipe:
    """Read ordered controls and lexical bindings; do not resolve expressions.

    With no scope, domain names are still parsed and loop indices are bound.
    Pass the step's definitions to expose recovered flatten coordinates. Missing
    declarations are checked by the consumer, so syntax-only use needs no schema.
    """
    def read(body, location, current, active):
        if not isinstance(body, list) or not body:
            raise PipelineSyntaxError(location, "pipe and control bodies must be nonempty lists")
        items = []
        for index, item in enumerate(body):
            item_path = f"{location}[{index}]"
            if isinstance(item, str):
                items.append(ActionDeclaration(parse_action(item, path=item_path), current, active))
                continue
            if not isinstance(item, dict) or len(item) != 1:
                raise PipelineSyntaxError(item_path, "pipe item must be an action string or a single for(...)/if(...) mapping")
            key, content = next(iter(item.items()))
            control_path = field_path(item_path, key) if isinstance(key, str) else item_path
            kind, expression = _control(key, control_path)
            if kind == "if":
                child = read(content, control_path, current, active)
                items.append(IfBlock(expression, child, control_path))
                continue
            spec = parse_for(key, path=control_path)
            if spec.name in active:
                raise PipelineSyntaxError(control_path, f"for cannot rebind active outer index {spec.name}")
            inner = current.extend(spec.name, list(spec.bounds)) if spec.bounds is not None else current
            coordinates = frozenset(inner.coordinates(spec.name))
            child = read(content, control_path, inner, active | coordinates)
            items.append(ForLoop(spec, child, control_path, current, active))
        return ScopedPipe(tuple(items), location, current, active)

    return read(value, path, scope if scope is not None else IndexScope({}), frozenset(bound))


def expression_names(expression: str) -> set[str]:
    """Conservative free-name hint, never a mathematical syntax validator.

    Familiar expressions can establish an unbound index. Other mathematical
    notation returns no hint and remains for review. Called functions, indexed
    objects, and count/last domain names are not current loop-variable uses.
    """
    try:
        node = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, RecursionError):
        return set()
    names, pending = set(), [node]
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in DOMAIN_FUNCTIONS:
                continue
            pending.extend(node.args)
            pending.extend(keyword.value for keyword in node.keywords)
        elif isinstance(node, ast.Subscript):
            pending.append(node.slice)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        else:
            pending.extend(ast.iter_child_nodes(node))
    return names


def free_action_indices(action: Action | str) -> set[str]:
    parsed = parse_action(action) if isinstance(action, str) else action
    return set().union(*(expression_names(arg) for operand in parsed.operands for arg in operand.args))


def domain_queries(expression: str) -> set[tuple[str, str]]:
    """Recognize count(name)/last(name) even inside otherwise opaque math."""
    return set(re.findall(r"(?<![\w.])(count|last)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", expression))
