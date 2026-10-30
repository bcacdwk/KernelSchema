"""Declared action-index windows, independent of generated action coverage."""
from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass

from ..rules.pipeline import parse_action
from .actions import operands
from .expressions import Domains, LIMIT, tree, variables
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

    def outer_context(self, window, concrete, env, needed):
        domains = window.domains
        context = dict(env)
        bound = set()
        if isinstance(window.expression, ast.Name):
            context[window.expression.id] = int(concrete[window.position])
            bound.add(window.expression.id)
        for position, expression in window.other_arguments:
            if isinstance(expression, ast.Name) and expression.id in domains.definitions:
                context[expression.id] = int(concrete[position])
                bound.add(expression.id)
        pending = [(position, expression) for position, expression in window.other_arguments
                   if variables(expression) & (needed - bound)]
        targets = set().union(*(variables(expression) & needed for _, expression in pending))
        while pending:
            progress = False
            for position, expression in list(pending):
                unbound = variables(expression) & (domains.definitions.keys() - bound)
                if not unbound:
                    if domains.integer(ast.unparse(expression), context) != int(concrete[position]):
                        return None
                elif len(unbound) == 1:
                    name = next(iter(unbound))
                    dependencies = domains.dependencies(name)
                    if dependencies & (targets - bound):
                        continue
                    if not dependencies <= context.keys():
                        raise ExpansionError(f"cannot bind outer coordinate {name} for an action-index window")
                    seed = {key: context[key] for key in dependencies}
                    matches = []
                    for entry in domains.entries(name, seed):
                        self.step()
                        if any(key in bound and context[key] != value for key, value in entry.items()):
                            continue
                        if domains.integer(ast.unparse(expression), dict(context, **entry)) == int(concrete[position]):
                            matches.append(entry)
                            if len(matches) > 1:
                                raise ExpansionError(f"ambiguous outer coordinate in action-index window: {ast.unparse(expression)}")
                    if not matches:
                        return None
                    context.update(matches[0])
                    bound.update(matches[0])
                else:
                    continue
                pending.remove((position, expression))
                progress = True
            if not progress:
                raise ExpansionError("cannot bind coupled outer coordinates for an action-index window")
        for position, expression in window.other_arguments:
            if not variables(expression) & domains.definitions.keys():
                if domains.integer(ast.unparse(expression), context) != int(concrete[position]):
                    return None
        return context

    def bounds(self, window, concrete, env, *, require_context_shift=False):
        domains = window.domains
        names = variables(window.expression) & domains.definitions.keys()
        dependencies = set().union(*(domains.dependencies(name) for name in names))
        # An outer coordinate used both in an expression and an inner bound is
        # fixed for that instance. A flat index, however, spans its full domain.
        varying = names - dependencies
        unbound = set().union(*(domains.coordinates(name) for name in varying))
        needed = dependencies - unbound
        context = self.outer_context(window, concrete, env, needed)
        if context is None:
            return None
        if require_context_shift and not any(name in env and context.get(name) != env[name] for name in needed):
            return None
        if not needed <= context.keys():
            raise ExpansionError(f"cannot bind action-index window context: {', '.join(sorted(needed - context.keys()))}")
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
        return self.cache[key]

    def outside(self, role, template, concrete, domains, env):
        values = {(oi, ai): value for oi, (_, args) in enumerate(operands(concrete)[1])
                  for ai, value in enumerate(args)}
        unknown = []
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
                bounds = [bound for source in sources
                          if (bound := self.bounds(source, values, env,
                                                   require_context_shift=unshifted)) is not None]
                if bounds:
                    lo, hi = min(b[0] for b in bounds), max(b[1] for b in bounds)
                    if not lo <= int(values[(oi, ai)]) <= hi:
                        return True
        if unknown:
            raise ExpansionError(f"cannot determine declared index window for {role}/{template}: "
                                 f"{', '.join(unknown)}; no unshifted domain binding is available")
        return False
