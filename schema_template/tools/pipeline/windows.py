"""Exact declared action windows, independent of emitted action coverage.

The bounded projection enumerates complete declared index domains, ignoring
for selections and if conditions. It never takes a min/max of emitted actions
or intersects producer/consumer logs. Missing in-window predecessors remain
errors; only a recoverable variable displacement beyond a domain may be skipped.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ..pipeline_syntax import ActionDeclaration
from .actions import arguments, instantiate, parsed
from .coordinates import recover
from .expressions import Boundary, Domains, LIMIT, shifted, tree, variables
from .model import ExpansionError, GraphError


@dataclass
class Window:
    declaration: ActionDeclaration
    domains: Domains


class IndexWindows:
    def __init__(self):
        self.declarations = defaultdict(list)
        self.cache = {}
        self.steps = 0

    def register(self, role: str, declaration: ActionDeclaration, domains: Domains):
        self.declarations[(role, declaration.action.family)].append(Window(declaration, domains))

    def step(self):
        self.steps += 1
        if self.steps > 100000:
            raise ExpansionError("declared index-window expansion limit exceeded (100000); use a smaller sample")

    def values(self, window: Window):
        key = id(window)
        if key not in self.cache:
            values = {}
            for env in window.domains.contexts(set(window.declaration.bound)):
                self.step()
                action = instantiate(window.declaration.action, window.domains, env)
                values.setdefault(action, []).append(env)
                if len(values) > LIMIT:
                    raise ExpansionError(f"declared action window exceeds {LIMIT} entries")
            self.cache[key] = values
        return self.cache[key]

    def validate_fixed(self, role: str, action, domains: Domains, env: dict):
        """A displaced coordinate cannot hide an invalid independent constant."""
        action = parsed(action)
        fixed = []
        for position, expression in arguments(action):
            if not variables(tree(expression)) & domains.definitions.keys():
                fixed.append((position, domains.integer(expression, env)))
        if not fixed:
            return
        windows = self.declarations[(role, action.family)]
        possibilities = defaultdict(set)
        for window in windows:
            for concrete in self.values(window):
                for position, value in arguments(concrete):
                    possibilities[position].add(int(value))
        for position, value in fixed:
            if possibilities[position] and value not in possibilities[position]:
                raise GraphError(f"fixed index {value} does not identify a declared {role}/{action.verb}:{action.noun} action")

    def empty_target(self, role: str, action, coordinate) -> bool:
        """Prove the target's declaration has no item at this outer coordinate.

        An empty queried index alone does not remove an unrelated action. Each
        applicable declaration must bind that index (directly or via flatten),
        and its own domain must be empty in the referenced work's context.
        """
        for window in self.declarations[(role, action.family)]:
            if coordinate.name not in window.declaration.bound:
                return False
            if window.domains.entries(coordinate.name, coordinate.context):
                return False
        return True

    def missing(self, role: str, action, concrete: str, domains: Domains, env: dict, *, target=False):
        """Return only for an inactive target; raise for missing predecessors.

        The caller has already checked family/signature/member and the actual
        event lookup. A concrete value in this full declaration window but not
        in the pipe's occurrence set can be skipped only on the target side.
        """
        action = parsed(action)
        windows = self.declarations[(role, action.family)]
        any_context = False
        for window in windows:
            values = self.values(window)
            any_context |= bool(values)
            if concrete in values:
                if target:
                    return
                raise GraphError(f"missing predecessor inside its declared index window: {role}/{concrete}")
        if target and not any_context:
            return  # The target's declared loop itself has no items.
        given = dict(arguments(concrete))
        displaced = {position for position, value in arguments(action) if shifted(tree(value), domains)}
        if not displaced:
            raise GraphError(f"fixed or unshifted event does not exist: {role}/{concrete}")
        self.validate_fixed(role, action, domains, env)
        uncertain, boundaries = [], []
        for window in windows:
            declaration, local = window.declaration, window.domains
            equations = [(tree(expression), int(given[position]))
                         for position, expression in arguments(declaration.action)]
            needed = set(declaration.bound)
            try:
                recovered = recover(equations, local, {}, needed, partial=True)
            except ExpansionError as error:
                uncertain.append(error)
                continue
            if recovered is None:
                continue
            # Test solved coordinates before their children. An absent outer
            # coordinate may make an inner domain undefined, but does not hide
            # a separate fixed-index error (validated above).
            pending = set(recovered) & local.definitions.keys()
            checked, outside, unavailable = set(), [], set()
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
                checked.add(name)
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
