"""Shared action notation and lexical pipe scopes; no validation or expansion I/O."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

from .diagnostics import IDENTIFIER

DOMAIN_FUNCTIONS = {"count", "last"}
REGIONS = ("prologue", "steady", "epilogue")


@dataclass(frozen=True)
class IndexScope:
    definitions: dict[str, Any]
    parent: IndexScope | None = None

    def extend(self, name: str, definition: Any) -> IndexScope:
        return IndexScope({**self.definitions, name: definition}, self)

    def coordinates(self, name: str) -> set[str]:
        result, pending = set(), [name]
        while pending:
            current = pending.pop()
            if current in result:
                continue
            result.add(current)
            definition = self.definitions.get(current)
            if isinstance(definition, dict):
                pending.extend(definition["flatten"])
        return result


def domain_choices(scopes: list[IndexScope], name: str) -> list[Any]:
    values = []
    for scope in scopes:
        if name in scope.definitions and scope.definitions[name] not in values:
            values.append(scope.definitions[name])
    return values


def common_domains(scopes: list[IndexScope]) -> dict[str, Any]:
    names = dict.fromkeys(name for scope in scopes for name in scope.definitions)
    return {name: choices[0] for name in names if len(choices := domain_choices(scopes, name)) == 1}


@dataclass(frozen=True)
class ActionDeclaration:
    text: str
    path: str
    scope: IndexScope
    bound: frozenset[str]


@dataclass
class ScopedPipe:
    body: dict
    path: str
    scope: IndexScope
    repeat: str | None
    regions: dict[str, list[ScopedPipe | ActionDeclaration]]

    @classmethod
    def read(cls, body: dict, scope: IndexScope, path: str, bound=frozenset()) -> ScopedPipe:
        repeat = body.get("repeat")
        if isinstance(repeat, dict):
            name, definition = next(iter(repeat.items()))
            scope = scope.extend(name, definition)
            repeat = name
        steady_bound = bound | scope.coordinates(repeat.split("[", 1)[0]) if repeat else bound
        regions = {}
        for region in REGIONS:
            current = steady_bound if region == "steady" else bound
            items = []
            for index, item in enumerate(body.get(region, [])):
                location = f"{path}.{region}[{index}]"
                items.append(cls.read(item, scope, location, current) if isinstance(item, dict)
                             else ActionDeclaration(item, location, scope, frozenset(current)))
            regions[region] = items
        return cls(body, path, scope, repeat, regions)

    def blocks(self):
        pending = [self]
        while pending:
            block = pending.pop()
            yield block
            pending.extend(reversed([item for region in REGIONS for item in block.regions[region]
                                     if isinstance(item, ScopedPipe)]))

    def actions(self):
        pending = [self]
        while pending:
            item = pending.pop()
            if isinstance(item, ActionDeclaration):
                yield item
            else:
                pending.extend(reversed([child for region in REGIONS for child in item.regions[region]]))


def split_top(text: str, delimiter: str) -> list[str] | None:
    pairs, stack, result, start = {"(": ")", "[": "]"}, [], [], 0
    for i, char in enumerate(text):
        if char in pairs:
            stack.append(pairs[char])
        elif char in ")]":
            if not stack or stack.pop() != char:
                return None
        elif char == delimiter and not stack:
            result.append(text[start:i])
            start = i + 1
    return None if stack else result + [text[start:]]


def action_parts(text: str) -> tuple[str, list[str]] | None:
    verb, colon, rest = text.partition(":")
    if not colon or not IDENTIFIER.fullmatch(verb):
        return None
    operands = split_top(rest, "@")
    if operands is None or len(operands) != (2 if verb == "mma" else 1):
        return None
    return verb, operands


def parse_action(text: str) -> tuple[str, tuple[tuple[str, int], ...]] | None:
    parts = action_parts(text)
    if parts is None:
        return None
    verb, operands = parts
    identities = []
    for operand in operands:
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(?:\((.+)\))?", operand)
        if not match:
            return None
        args = split_top(match[2], ",") if match[2] is not None else []
        if args is None or any(not a.strip() for a in args):
            return None
        identities.append((match[1], len(args)))
    return verb, tuple(identities)


def parse_event(text: str) -> tuple[str | None, str | None, str]:
    text = re.sub(r"@(issue|done)$", "", text)
    match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)(?:\[(\*|[0-9]+)\])?/", text)
    if match:
        return match[1], match[2], text[match.end():]
    return None, None, text


def free_action_indices(text: str) -> set[str]:
    """Check scopes when indices use familiar syntax; other math stays for review."""
    parts, names = action_parts(text), set()
    if parts is None:
        return names

    def walk(node: ast.AST) -> None:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in DOMAIN_FUNCTIONS:
            return
        if isinstance(node, ast.Call):
            for argument in node.args:
                walk(argument)
            return
        if isinstance(node, ast.Subscript):
            walk(node.slice)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        else:
            for child in ast.iter_child_nodes(node):
                walk(child)

    for operand in parts[1]:
        if "(" in operand and operand.endswith(")"):
            try:
                walk(ast.parse(operand.split("(", 1)[1][:-1], mode="eval"))
            except (SyntaxError, ValueError, RecursionError):
                pass  # General mathematical index interpretation is not a proof obligation here.
    return names

