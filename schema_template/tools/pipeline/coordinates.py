"""Recover target coordinates from declared action arguments, without source guesses."""
from __future__ import annotations

import ast
from fractions import Fraction

from .expressions import Domains, referenced_domains, variables
from .model import ExpansionError


def affine(node: ast.AST, domains: Domains, env: dict) -> tuple[dict[str, Fraction], Fraction] | None:
    """Return coefficients and a constant, or None for an unsupported inverse."""
    if isinstance(node, ast.Name) and node.id in domains.definitions:
        return {node.id: Fraction(1)}, Fraction(0)
    if not variables(node) & domains.definitions.keys():
        queries = referenced_domains(node) & domains.definitions.keys()
        if any(domains.dependencies(name) for name in queries):
            return None
        return {}, domains.evaluate(node, env)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = affine(node.operand, domains, env)
        if value is not None:
            sign = -1 if isinstance(node.op, ast.USub) else 1
            return {name: sign * coefficient for name, coefficient in value[0].items()}, sign * value[1]
    if isinstance(node, ast.BinOp):
        left, right = affine(node.left, domains, env), affine(node.right, domains, env)
        if left is None or right is None:
            return None
        a, b = left[0], right[0]
        if isinstance(node.op, (ast.Add, ast.Sub)):
            sign = -1 if isinstance(node.op, ast.Sub) else 1
            coefficients = {name: a.get(name, 0) + sign * b.get(name, 0) for name in a.keys() | b.keys()}
            return {name: value for name, value in coefficients.items() if value}, left[1] + sign * right[1]
        if isinstance(node.op, ast.Mult) and (not a or not b):
            linear, scalar = (right, left[1]) if not a else (left, right[1])
            return {name: value * scalar for name, value in linear[0].items()}, linear[1] * scalar
        if isinstance(node.op, ast.Div) and not b and right[1]:
            return {name: value / right[1] for name, value in a.items()}, left[1] / right[1]
    return None


def recover(arguments: list[tuple[ast.AST, int]], domains: Domains, env: dict,
            needed: set[str], *, partial=False) -> dict[str, int] | None:
    """Solve the affine constraints exactly; unresolved inverses are tool limits.

    An inconsistent system is a missing action, not evidence of a boundary.
    Only genuinely external coordinates may be inherited from the source env.
    Partial results leave unresolved coordinates to the caller's declaration
    projection; they never fill those coordinates from the source environment.
    """
    names = sorted(set().union(*(variables(node) & domains.definitions.keys() for node, _ in arguments)))
    if len(names) > 64:
        raise ExpansionError("action coordinate system exceeds the tool limit (64)")
    rows = []
    for node, target in arguments:
        value = affine(node, domains, env)
        if value is not None:
            coefficients, constant = value
            rows.append([coefficients.get(name, Fraction(0)) for name in names] + [Fraction(target) - constant])
    pivots, rank = {}, 0
    for column in range(len(names)):
        pivot = next((i for i in range(rank, len(rows)) if rows[i][column]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        divisor = rows[rank][column]
        rows[rank] = [value / divisor for value in rows[rank]]
        for i, row in enumerate(rows):
            if i != rank and row[column]:
                factor = row[column]
                rows[i] = [a - factor * b for a, b in zip(row, rows[rank])]
        pivots[column] = rank
        rank += 1
    if any(not any(row[:-1]) and row[-1] for row in rows):
        return None
    represented = set().union(*(domains.coordinates(name) for name in names))
    context = {name: value for name, value in env.items() if name not in represented | needed}
    for column, row_index in pivots.items():
        row = rows[row_index]
        if all(not value for i, value in enumerate(row[:-1]) if i != column):
            if row[-1].denominator != 1:
                raise ExpansionError(f"cannot recover an integer action coordinate {names[column]}")
            context[names[column]] = int(row[-1])
    for name in names:
        if name in context and isinstance(domains.definitions[name], dict):
            entry = next((entry for entry in domains.entries(name, context) if entry[name] == context[name]), None)
            if entry is not None:
                if any(key in context and context[key] != value for key, value in entry.items()):
                    return None
                context.update(entry)
    missing = needed - context.keys()
    if missing and not partial:
        raise ExpansionError(f"cannot uniquely recover action-index coordinates: {', '.join(sorted(missing))}")
    return context
