"""Declared action-index windows, independent of generated action coverage."""
from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from ..pipeline_syntax import parse_action
from .actions import operands
from .expressions import Domains, LIMIT, tree, variables
from .coordinates import recover
from .model import ExpansionError


class RangeExpression(ast.NodeTransformer):
    """A flat coordinate or last(d) refers to d's whole declared window."""
    def __init__(self, domains):
        self.domains = domains

    def visit_Subscript(self, node):
        if (isinstance(node.value, ast.Name)
                and isinstance(self.domains.definitions.get(node.value.id), dict)):
            return ast.Name(id=node.value.id, ctx=ast.Load())
        return self.generic_visit(node)

    def visit_Call(self, node):
        if (isinstance(node.func, ast.Name) and node.func.id == "last" and len(node.args) == 1
                and isinstance(node.args[0], ast.Name)):
            return ast.Name(id=node.args[0].id, ctx=ast.Load())
        return self.generic_visit(node)


@dataclass
class Window:
    expression: ast.AST
    domains: Domains
    position: tuple[int, int]
    other_arguments: list[tuple[tuple[int, int], ast.AST]]


class WindowMatch(Enum):
    REQUIRED = "an unshifted reference still requires an actual action"


class IndexWindows:
    def __init__(self):
        self.pipe = defaultdict(list)
        self.reference = defaultdict(list)
        self.cache = {}
        self.steps = 0

    def step(self):
        self.steps += 1
        if self.steps > 100000:
            raise ExpansionError("combined index-window expansion limit exceeded (100000); use a smaller sample")

    def register(self, role, action, domains, *, reference=False):
        try:
            arguments = {(oi, ai): tree(value) for oi, (_, args) in enumerate(operands(action)[1])
                         for ai, value in enumerate(args)}
        except ExpansionError:
            # An inactive pipe branch need not be evaluated for this sample.
            # Active actions and references still fail at integer substitution.
            return
        for position, argument in arguments.items():
            # Bare references connect a literal/unrolled action to a declared
            # domain. A shifted reference alone cannot establish its own window.
            if reference and not isinstance(argument, ast.Name):
                continue
            expression = RangeExpression(domains).visit(tree(ast.unparse(argument)))
            if not variables(expression) & domains.definitions.keys():
                continue
            key = (role, parse_action(action), position)
            collection = self.reference if reference else self.pipe
            collection[key].append(Window(expression, domains, position,
                [(pos, node) for pos, node in arguments.items() if pos != position]))

    def target_contexts(self, window, concrete, env, needed, require_context_shift):
        domains = window.domains
        arguments = [(window.expression, int(concrete[window.position]))]
        for position, expression in window.other_arguments:
            normalized = RangeExpression(domains).visit(tree(ast.unparse(expression)))
            arguments.append((normalized, int(concrete[position])))
        if not needed:
            if require_context_shift:
                yield WindowMatch.REQUIRED
                return
            for position, expression in window.other_arguments:
                if not variables(expression) & domains.definitions.keys():
                    if domains.integer(ast.unparse(expression), env) != int(concrete[position]):
                        return
            yield dict(env)
            return
        context = recover(arguments, domains, env, needed, partial=True)
        if context is None:
            return
        represented = set().union(*(domains.coordinates(name) for expression, _ in arguments
                                    for name in variables(expression) & domains.definitions.keys()))
        missing = needed - context.keys()
        if missing & represented:
            raise ExpansionError("cannot uniquely recover action-index coordinates: "
                                 + ", ".join(sorted(missing & represented)))
        if require_context_shift and not any(name in env and name in context and context[name] != env[name]
                                             for name in needed):
            yield WindowMatch.REQUIRED
            return
        # Coordinates omitted from the action identity quantify declaration
        # contexts. Enumerate their complete domains, never the emitted actions.
        if not self.outer_exists(domains, needed & context.keys(), context):
            yield None
            return
        found = False
        for i, candidate in enumerate(domains.contexts(missing, context)):
            self.step()
            if i >= LIMIT:
                raise ExpansionError("action-index declaration contexts exceed expansion limit")
            found = True
            yield candidate
        if missing and not found:
            yield None

    def outer_exists(self, domains, needed, context):
        pending, checked = set(needed), set()
        while pending:
            ready = [name for name in pending if domains.dependencies(name) & needed <= checked]
            if not ready:
                raise ExpansionError("cannot order action-index outer coordinates")
            name = min(ready)
            self.step()
            if not any(entry[name] == context[name] for entry in domains.entries(name, context)):
                return False
            checked.add(name)
            pending.remove(name)
        return True

    def bounds(self, window, concrete, env, *, require_context_shift=False):
        domains = window.domains
        names = variables(window.expression) & domains.definitions.keys()
        dependencies = set().union(*(domains.dependencies(name) for name in names))
        varying = names - dependencies
        unbound = set().union(*(domains.coordinates(name) for name in varying))
        needed = dependencies - unbound
        bounds = []
        for context in self.target_contexts(window, concrete, env, needed, require_context_shift):
            if context is WindowMatch.REQUIRED:
                return WindowMatch.REQUIRED
            if context is None or not self.outer_exists(domains, needed, context):
                bounds.append((0, -1))
                continue
            seed = {name: context[name] for name in needed}
            key = (id(window), tuple(sorted(seed.items())))
            if key not in self.cache:
                values = []
                for i, bound in enumerate(domains.contexts(varying, seed)):
                    self.step()
                    if i >= LIMIT:
                        raise ExpansionError("action index window exceeds expansion limit; use a smaller sample")
                    values.append(domains.integer(ast.unparse(window.expression), bound))
                self.cache[key] = (min(values), max(values)) if values else (0, -1)
            bounds.append(self.cache[key])
        return (min(bound[0] for bound in bounds), max(bound[1] for bound in bounds)) if bounds else None

    def outside(self, role, template, concrete, domains, env):
        values = {(oi, ai): value for oi, (_, args) in enumerate(operands(concrete)[1])
                  for ai, value in enumerate(args)}
        unknown, incomplete = [], []
        for oi, (_, args) in enumerate(operands(template)[1]):
            for ai, argument in enumerate(args):
                node = tree(argument)
                if not variables(node) & domains.definitions.keys():
                    continue  # Fixed missing references are always errors.
                unshifted = isinstance(node, ast.Name)
                key = (role, parse_action(template), (oi, ai))
                sources = self.pipe[key] or self.reference[key]
                if not sources:
                    if not unshifted:
                        unknown.append(argument)
                    continue
                bounds, uncertain, required = [], False, False
                for source in sources:
                    try:
                        bound = self.bounds(source, values, env, require_context_shift=unshifted)
                    except ExpansionError as error:
                        incomplete.append(error)
                        uncertain = True
                        continue
                    if bound is WindowMatch.REQUIRED:
                        required = True
                    elif bound is not None:
                        bounds.append(bound)
                if bounds and not uncertain and not required:
                    lo, hi = min(b[0] for b in bounds), max(b[1] for b in bounds)
                    if not lo <= int(values[(oi, ai)]) <= hi:
                        return True
        if incomplete:
            raise incomplete[0]
        if unknown:
            raise ExpansionError(f"cannot determine declared index window for {role}/{template}: "
                                 f"{', '.join(unknown)}; no unshifted domain binding is available")
        return False
