"""A bounded integer evaluator and declared index domains. Never uses eval."""
from __future__ import annotations

import ast
from fractions import Fraction
from math import ceil, floor

from .model import ExpansionError, GraphError

LIMIT = 20000
DEPENDENCY_DEPTH = 128


class Boundary(GraphError):
    """A variable coordinate stepped beyond its declared domain."""


def tree(text: str) -> ast.AST:
    try:
        result = ast.parse(str(text).strip(), mode="eval").body
        if len(list(ast.walk(result))) > 100:
            raise ExpansionError(f"expression too large to expand: {text}")
        return result
    except (SyntaxError, RecursionError) as error:
        raise ExpansionError(f"unsupported integer expression: {text}") from error


def variables(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in {"count", "last"}:
            return set()
        return set().union(*(variables(a) for a in node.args))
    if isinstance(node, ast.Subscript):
        return variables(node.slice)
    return set().union(*(variables(c) for c in ast.iter_child_nodes(node)))


def referenced_domains(node: ast.AST) -> set[str]:
    """Names whose definitions are needed, even when they bind no variable."""
    result = variables(node)
    for part in ast.walk(node):
        if (isinstance(part, ast.Call) and isinstance(part.func, ast.Name)
                and part.func.id in {"count", "last"} and len(part.args) == 1
                and isinstance(part.args[0], ast.Name)):
            result.add(part.args[0].id)
        if isinstance(part, ast.Subscript) and isinstance(part.value, ast.Name):
            result.add(part.value.id)
    return result


class Domains:
    def __init__(self, definitions: dict, parameters: dict[str, int], worktiles: int):
        self.definitions = definitions
        self.parameters = parameters
        self.worktiles = worktiles
        self._active: set[str] = set()
        self._dependencies: dict[str, set[str]] = {}

    def extended(self, name: str, definition) -> Domains:
        return Domains(dict(self.definitions, **{name: definition}), self.parameters, self.worktiles)

    def coordinates(self, name: str, visited=None) -> set[str]:
        visited = set() if visited is None else visited
        if len(visited) >= DEPENDENCY_DEPTH:
            raise ExpansionError(f"index-domain dependency depth exceeds {DEPENDENCY_DEPTH}")
        if name in visited:
            raise GraphError(f"recursive index domain: {name}")
        definition = self.definitions.get(name)
        result = {name}
        if isinstance(definition, dict):
            for child in definition["flatten"]:
                result.update(self.coordinates(child, visited | {name}))
        return result

    def dependencies(self, name: str, visited=None) -> set[str]:
        """Coordinates needed before this domain can be enumerated."""
        if name in self._dependencies:
            return self._dependencies[name]
        visited = set() if visited is None else visited
        if len(visited) >= DEPENDENCY_DEPTH:
            raise ExpansionError(f"index-domain dependency depth exceeds {DEPENDENCY_DEPTH}")
        if name in visited:
            raise ExpansionError(f"cannot order dependent index domains involving {name}")
        if name not in self.definitions:
            raise GraphError(f"undefined index domain: {name}")
        definition = self.definitions[name]
        needed = set()
        if isinstance(definition, list):
            for expression in definition:
                node = tree(str(expression))
                if ast.unparse(node).replace(" ", "") in self.parameters:
                    continue
                needed.update(variables(node) & self.definitions.keys())
                for query in ast.walk(node):
                    if (isinstance(query, ast.Call) and isinstance(query.func, ast.Name)
                            and query.func.id in {"count", "last"} and len(query.args) == 1
                            and isinstance(query.args[0], ast.Name)):
                        if ast.unparse(query).replace(" ", "") not in self.parameters:
                            needed.update(self.dependencies(query.args[0].id, visited | {name}))
        elif isinstance(definition, dict):
            bound = set()
            children = set().union(*(self.coordinates(child) for child in definition["flatten"]))
            for child in definition["flatten"]:
                required = self.dependencies(child, visited | {name}) - bound
                if required & children:
                    raise ExpansionError(f"flatten {name} needs {', '.join(sorted(required & children))} before {child}")
                needed.update(required)
                bound.update(self.coordinates(child))
        for dependency in tuple(needed):
            needed.update(self.dependencies(dependency, visited | {name}))
        self._dependencies[name] = needed
        return needed

    def integer(self, expression, env: dict[str, int]) -> int:
        value = self.evaluate(tree(str(expression)), env)
        if value.denominator != 1:
            raise GraphError(f"noninteger index/bound: {expression} = {value}")
        return int(value)

    def integers(self, expressions: list[str], env: dict[str, int]) -> list[int]:
        values, boundary = [], None
        for expression in expressions:
            try:
                values.append(self.integer(expression, env))
            except Boundary as error:
                boundary = boundary or error
        if boundary is not None:
            raise boundary
        return values

    def evaluate_all(self, nodes: list[ast.AST], env: dict[str, int]) -> list[Fraction]:
        """A boundary in one operand does not hide an invalid independent operand."""
        values, boundary = [], None
        for node in nodes:
            try:
                values.append(self.evaluate(node, env))
            except Boundary as error:
                boundary = boundary or error
        if boundary is not None:
            raise boundary
        return values

    def evaluate(self, node: ast.AST, env: dict[str, int]) -> Fraction:
        value = self._evaluate(node, env)
        if abs(value) > 10**12:
            raise ExpansionError("integer expression magnitude exceeds the tool limit (10^12)")
        return Fraction(value)

    def _evaluate(self, node: ast.AST, env: dict[str, int]):
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        canonical = ast.unparse(node).replace(" ", "")
        if canonical in self.parameters:
            return self.parameters[canonical]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self.evaluate(node.operand, env)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            a, b = self.evaluate_all([node.left, node.right], env)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and b == 0:
                raise GraphError(f"division by zero: {canonical}")
            if isinstance(node.op, ast.Div): return a / b
            if isinstance(node.op, ast.FloorDiv): return a // b
            if isinstance(node.op, ast.Mod): return a % b
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            if name in {"count", "last"} and len(node.args) == 1 and isinstance(node.args[0], ast.Name):
                domain = node.args[0].id
                values = self.entries(domain, env)
                if name == "count": return len(values)
                if not values: raise GraphError(f"last({domain}) on an empty domain")
                return values[-1][domain]
            values = self.evaluate_all(node.args, env)
            if name == "min" and values: return min(values)
            if name == "max" and values: return max(values)
            if name == "floor" and len(values) == 1: return floor(values[0])
            if name == "ceil" and len(values) == 1: return ceil(values[0])
            if name == "ceil_div" and len(values) == 2 and values[1]: return ceil(values[0] / values[1])
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            name = node.value.id
            definition = self.definitions.get(name)
            if not isinstance(definition, dict) or "flatten" not in definition:
                raise ExpansionError(f"unsupported indexed expression {canonical!r}; "
                                     "only declared flatten coordinates can be expanded")
            parts = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
            children = definition["flatten"]
            if len(parts) != len(children) or any(isinstance(p, ast.Slice) for p in parts):
                raise GraphError(f"an action needs full flatten coordinates: {canonical}")
            coordinates, unavailable, boundary = {}, set(), None
            for child, part in zip(children, parts):
                try:
                    coordinate = self.integer(ast.unparse(part), env)
                except Boundary as error:
                    boundary = boundary or error
                    unavailable.update(self.coordinates(child))
                    continue
                # A dependent inner domain has no range at an absent outer
                # coordinate. Independent coordinates still require validation.
                if unavailable and self.dependencies(child) & unavailable:
                    unavailable.update(self.coordinates(child))
                    continue
                selected = next((item for item in self.entries(child, dict(env, **coordinates))
                                 if item[child] == coordinate), None)
                if selected is None:
                    message = f"{canonical}: {child}={coordinate} outside its declared domain"
                    if not variables(part) & self.definitions.keys():
                        raise GraphError(message)
                    boundary = boundary or Boundary(message)
                    unavailable.update(self.coordinates(child))
                else:
                    coordinates.update(selected)
            if boundary is not None:
                raise boundary
            for entry in self.entries(name, {}):
                if all(entry[c] == v for c, v in coordinates.items()):
                    return entry[name]
            raise GraphError(f"no coordinate in {name}: {coordinates}")
        raise ExpansionError(f"unbound or unsupported expression {canonical!r}; supply it in the sample or --set")

    def entries(self, name: str, env: dict[str, int]) -> list[dict[str, int]]:
        if name not in self.definitions:
            raise GraphError(f"undefined index domain: {name}")
        if name in self._active:
            raise GraphError(f"recursive index domain: {name}")
        self._active.add(name)
        try:
            definition = self.definitions[name]
            if definition == "worktile":
                return [{name: i} for i in range(self.worktiles)]
            if isinstance(definition, list):
                lo, hi = (self.integer(e, env) for e in definition)
                if hi < lo:
                    raise GraphError(f"reversed domain {name}: [{lo}, {hi})")
                if hi - lo > LIMIT:
                    raise ExpansionError(f"oversized domain {name}: [{lo}, {hi}); use a smaller sample")
                return [{name: i} for i in range(lo, hi)]
            result = [{}]
            for child in definition["flatten"]:
                expanded = []
                for prefix in result:
                    for item in self.entries(child, dict(env, **prefix)):
                        if len(expanded) >= LIMIT:
                            raise ExpansionError(f"flatten domain {name} exceeds {LIMIT} entries; use a smaller sample")
                        expanded.append(dict(prefix, **item))
                result = expanded
            return [dict(item, **{name: i}) for i, item in enumerate(result)]
        finally:
            self._active.remove(name)

    def contexts(self, names: set[str], env: dict[str, int] | None = None):
        env = {} if env is None else env
        names = set(names)
        for name in tuple(names):
            if name not in env:
                names.update(self.dependencies(name))
        remaining = names - env.keys()
        if not remaining:
            yield env
            return
        # Binding a flat index also binds its coordinates; do not form an
        # independent Cartesian product of n, t and i.
        ready = [name for name in remaining if self.dependencies(name) <= env.keys()]
        if not ready:
            raise ExpansionError(f"cannot order dependent index domains: {', '.join(sorted(remaining))}")
        name = min(ready, key=lambda n: (not isinstance(self.definitions.get(n), dict), n))
        for item in self.entries(name, env):
            if all(k not in env or env[k] == value for k, value in item.items()):
                yield from self.contexts(names, dict(env, **item))

    def select(self, repeat: str, env: dict[str, int]) -> list[dict[str, int]]:
        node = tree(repeat)
        if isinstance(node, ast.Name):
            return self.entries(node.id, env)
        if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
            raise GraphError(f"invalid repeat: {repeat}")
        name = node.value.id
        parts = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        definition = self.definitions.get(name)
        axes = definition["flatten"] if isinstance(definition, dict) else [name]
        if len(parts) > len(axes):
            raise GraphError(f"too many coordinates in repeat: {repeat}")
        result = []
        for item in self.entries(name, env):
            match = True
            for axis, part in zip(axes, parts):
                if isinstance(part, ast.Slice):
                    if part.step is not None:
                        raise GraphError("repeat slices do not support a step")
                    lo = self.integer(ast.unparse(part.lower), env) if part.lower else None
                    hi = self.integer(ast.unparse(part.upper), env) if part.upper else None
                    match &= (lo is None or item[axis] >= lo) and (hi is None or item[axis] < hi)
                else:
                    match &= item[axis] == self.integer(ast.unparse(part), env)
            if match:
                result.append(item)
        return result
