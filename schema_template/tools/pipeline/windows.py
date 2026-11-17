"""Exact declared action windows, independent of emitted action coverage.

The bounded projection enumerates complete declared index domains, ignoring
for selections and if conditions. It never takes a min/max of emitted actions
or intersects producer/consumer logs. Missing in-window predecessors remain
errors; only a recoverable variable displacement beyond a domain may be skipped.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass

from ..pipeline_syntax import ActionDeclaration, IfBlock
from .actions import arguments, parsed
from .coordinates import recover
from .expressions import Boundary, Domains, LIMIT, shifted, tree, variables
from .model import ExpansionError, GraphError


@dataclass
class Window:
    declaration: ActionDeclaration
    domains: Domains
    conditions: tuple[tuple[IfBlock, Domains], ...]


class IndexWindows:
    def __init__(self):
        self.declarations = defaultdict(list)
        self.cache = {}
        self.steps = 0

    def register(self, role: str, pipe, domains_for, member: int, conditions=()):
        for item in pipe.items:
            if isinstance(item, ActionDeclaration):
                self.declarations[(role, member, item.action.family)].append(
                    Window(item, domains_for(item.scope), conditions))
            else:
                inner = conditions
                if isinstance(item, IfBlock):
                    inner += ((item, domains_for(item.body.scope)),)
                self.register(role, item.body, domains_for, member, inner)

    def step(self):
        self.steps += 1
        if self.steps > 100000:
            raise ExpansionError("declared index-window expansion limit exceeded (100000); use a smaller sample")

    def values(self, window: Window, positions=None):
        expressions = dict(arguments(window.declaration.action))
        positions = tuple(expressions) if positions is None else positions
        key = (id(window), positions)
        if key not in self.cache:
            expressions = [expressions[position] for position in positions]
            values = set()
            for env in window.domains.contexts(set(window.declaration.bound)):
                self.step()
                values.add(tuple(window.domains.integers(expressions, env)))
                if len(values) > LIMIT:
                    raise ExpansionError(f"declared action window exceeds {LIMIT} entries")
            self.cache[key] = values
        return self.cache[key]

    def validate_fixed(self, role: str, action, domains: Domains, env: dict, member: int):
        """A displaced coordinate cannot hide invalid fixed arguments."""
        action = parsed(action)
        fixed = []
        for position, expression in arguments(action):
            if not variables(tree(expression)) & domains.definitions.keys():
                fixed.append((position, domains.integer(expression, env)))
        if not fixed:
            return
        positions, expected = zip(*fixed)
        populated = matched = False
        for window in self.declarations[(role, member, action.family)]:
            values = self.values(window, positions)
            populated |= bool(values)
            matched |= expected in values
        if populated and not matched:
            label = f"index {expected[0]}" if len(expected) == 1 else f"indices {expected}"
            raise GraphError(f"fixed {label} does not identify a declared {role}/{action.verb}:{action.noun} action")

    def empty_target(self, role: str, action, coordinate, member: int) -> bool:
        """Prove the target's declaration has no item at this outer coordinate.

        An empty queried index alone does not remove an unrelated action. Each
        declaration must either have a false outer guard, or bind that index
        (directly or via flatten) with an empty domain in this work's context.
        """
        for window in self.declarations[(role, member, action.family)]:
            inactive = False
            for condition, local in window.conditions:
                if not condition.body.bound <= coordinate.context.keys():
                    break
                env = {name: coordinate.context[name] for name in condition.body.bound}
                if not local.predicate(condition.condition, env):
                    inactive = True
                    break
            if inactive:
                continue
            if coordinate.name not in window.declaration.bound:
                return False
            if window.domains.entries(coordinate.name, coordinate.context):
                return False
        return True

    def missing(self, role: str, action, concrete: str, domains: Domains, env: dict, member: int, *, target=False):
        """Return only for an inactive target; raise for missing predecessors.

        The caller has already checked family/signature/member and the actual
        event lookup. A concrete value in this full declaration window but not
        in the pipe's occurrence set can be skipped only on the target side.
        """
        action = parsed(action)
        windows = self.declarations[(role, member, action.family)]
        given = {position: int(value) for position, value in arguments(concrete)}
        any_context = False
        for window in windows:
            values = self.values(window)
            any_context |= bool(values)
            if tuple(given.values()) in values:
                if target:
                    return
                raise GraphError(f"missing predecessor inside its declared index window: {role}/{concrete}")
        if target and not any_context:
            return  # The target's declared loop itself has no items.
        displaced = {position for position, value in arguments(action) if shifted(tree(value), domains)}
        if not displaced:
            raise GraphError(f"fixed or unshifted event does not exist: {role}/{concrete}")
        self.validate_fixed(role, action, domains, env, member)
        uncertain, boundaries = [], []
        for window in windows:
            declaration, local = window.declaration, window.domains
            equations = [(tree(expression), given[position])
                         for position, expression in arguments(declaration.action)]
            needed = set(declaration.bound)
            try:
                recovered = recover(equations, local)
            except ExpansionError as error:
                uncertain.append(error)
                continue
            if recovered is None:
                continue
            # A full flatten address returns its continuous domain ordinal.
            # An ordinal outside that domain needs no inverse for its children.
            flat_boundaries = []
            for position, expression in arguments(declaration.action):
                node = tree(expression)
                if position not in displaced or not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
                    continue
                name = node.value.id
                definition = local.definitions.get(name)
                parts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
                if (isinstance(definition, dict) and len(parts) == len(definition["flatten"])
                        and not any(isinstance(part, ast.Slice) for part in parts)
                        and not local.dependencies(name)):
                    self.step()
                    value = given[position]
                    if not any(item[name] == value for item in local.entries(name, {})):
                        flat_boundaries.append(f"{name}={value} outside its declared domain")
            if flat_boundaries:
                boundaries.append("; ".join(flat_boundaries))
                continue
            # Test solved coordinates before their children. An absent outer
            # coordinate may make an inner domain undefined, but does not hide
            # a separate fixed-index error (validated above).
            pending = set(recovered) & local.definitions.keys()
            outside, unavailable = [], set()
            while pending:
                ready = [name for name in pending if local.dependencies(name) <= recovered.keys()
                         and not (local.dependencies(name) & pending)]
                if not ready:
                    if any(local.dependencies(name) & unavailable for name in pending):
                        break
                    uncertain.append(ExpansionError("cannot recover the outer coordinates of " + concrete))
                    break
                name = min(ready, key=lambda item: (not isinstance(local.definitions.get(item), dict), item))
                pending.remove(name)
                if local.dependencies(name) & unavailable:
                    unavailable.add(name)
                    continue
                self.step()
                entries = local.entries(name, recovered)
                if not any(item[name] == recovered[name] for item in entries):
                    positions = {position for position, expression in arguments(declaration.action)
                                 if any(name in local.coordinates(variable)
                                        for variable in variables(tree(expression)) & local.definitions.keys())}
                    if not positions or not positions <= displaced:
                        outside = []
                        break
                    outside.append(f"{name}={recovered[name]} outside its declared domain")
                    unavailable.update(local.coordinates(name))
            if outside:
                boundaries.append("; ".join(outside))
            elif needed - recovered.keys():
                uncertain.append(ExpansionError("cannot uniquely recover action-index coordinates: "
                                                + ", ".join(sorted(needed - recovered.keys()))))
        if uncertain:
            raise uncertain[0]
        if boundaries:
            raise Boundary(f"{role}/{concrete}: {boundaries[0]}")
        raise GraphError(f"event does not identify an action in its declared window: {role}/{concrete}")
