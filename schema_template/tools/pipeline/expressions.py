"""Bounded, exact sample evaluation and declared index domains; never eval.

Unsupported mathematical notation is an ExpansionError, not a schema syntax
error. The evaluator only knows explicit sample parameters and lexical indices.
"""
from __future__ import annotations

import ast
from fractions import Fraction
from math import ceil, floor

from ..pipeline_syntax import ForSpec, IndexScope, Slice, literal_integer
from .model import ExpansionError, GraphError

LIMIT = 20000
DEPENDENCY_DEPTH = 128


class Boundary(GraphError):
    """A variable shift passed a declared domain boundary."""


class EmptyCoordinate(GraphError):
    """A fixed flatten coordinate addresses an empty declared child domain."""

    def __init__(self, message, name=None, context=None):
        super().__init__(message)
        self.name, self.context = name, dict(context or {})


def tree(text) -> ast.AST:
    try:
        literal = literal_integer(text)
        node = ast.Constant(value=literal) if literal is not None else ast.parse(str(text).strip(), mode="eval").body
    except (SyntaxError, ValueError, RecursionError) as error:
        raise ExpansionError(f"unsupported sample expression: {text}") from error
    if sum(1 for _ in ast.walk(node)) > 100:
        raise ExpansionError(f"expression too large to expand: {text}")
    return node


def variables(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return set() if node.id in {"true", "false"} else {node.id}
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in {"count", "last"}:
            return set()
        return set().union(*(variables(arg) for arg in node.args))
    if isinstance(node, ast.Subscript):
        return variables(node.slice)
    return set().union(*(variables(child) for child in ast.iter_child_nodes(node)))


def referenced_domains(node: ast.AST) -> set[str]:
    result = variables(node)
    for part in ast.walk(node):
        if (isinstance(part, ast.Call) and isinstance(part.func, ast.Name)
                and part.func.id in {"count", "last"} and len(part.args) == 1
                and isinstance(part.args[0], ast.Name)):
            result.add(part.args[0].id)
        if isinstance(part, ast.Subscript) and isinstance(part.value, ast.Name):
            result.add(part.value.id)
    return result


def merge_coordinates(env: dict[str, int], values: dict[str, int]) -> dict[str, int]:
    """Restore flatten coordinates without silently rebinding or filtering."""
    conflicts = [name for name in values if name in env and env[name] != values[name]]
    if conflicts:
        detail = ", ".join(f"{name}: {env[name]} != {values[name]}" for name in sorted(conflicts))
        raise GraphError("flatten coordinates conflict with bound outer indices: " + detail)
    return {**env, **values}


class Domains:
    def __init__(self, definitions: dict | IndexScope, parameters: dict[str, int], worktiles: int,
                 *, dynamic_worktiles=False):
        self.scope = definitions if isinstance(definitions, IndexScope) else IndexScope(definitions)
        self.definitions = self.scope.definitions
        self.parameters, self.worktiles = parameters, worktiles
        self.dynamic_worktiles = dynamic_worktiles
        self._active: set[str] = set()
        self._dependencies: dict[str, set[str]] = {}

    def coordinates(self, name: str, visited=frozenset()) -> set[str]:
        if len(visited) >= DEPENDENCY_DEPTH:
            raise ExpansionError(f"index-domain dependency depth exceeds {DEPENDENCY_DEPTH}")
        if name in visited:
            raise GraphError(f"recursive index domain: {name}")
        result = {name}
        definition = self.definitions.get(name)
        if isinstance(definition, dict):
            for child in definition["flatten"]:
                child_coordinates = self.coordinates(child, visited | {name})
                if result & child_coordinates:
                    raise GraphError(f"flatten {name} restores repeated coordinates")
                result.update(child_coordinates)
        return result

    def dependencies(self, name: str, visited=frozenset()) -> set[str]:
        if name in self._dependencies:
            return self._dependencies[name]
        if len(visited) >= DEPENDENCY_DEPTH:
            raise ExpansionError(f"index-domain dependency depth exceeds {DEPENDENCY_DEPTH}")
        if name in visited:
            raise GraphError(f"recursive index domain: {name}")
        if name not in self.definitions:
            raise GraphError(f"undefined index domain: {name}")
        definition, needed = self.definitions[name], set()
        if isinstance(definition, list):
            for expression in definition:
                node = tree(expression)
                if not variables(node) & self.definitions.keys() and ast.unparse(node).replace(" ", "") in self.parameters:
                    continue
                needed.update(variables(node) & self.definitions.keys())
                for query in ast.walk(node):
                    if (isinstance(query, ast.Call) and isinstance(query.func, ast.Name)
                            and query.func.id in {"count", "last"} and len(query.args) == 1
                            and isinstance(query.args[0], ast.Name)):
                        needed.update(self.dependencies(query.args[0].id, visited | {name}))
        elif isinstance(definition, dict):
            bound = set()
            children = set().union(*(self.coordinates(child) for child in definition["flatten"]))
            for child in definition["flatten"]:
                required = self.dependencies(child, visited | {name}) - bound
                if required & children:
                    raise GraphError(f"flatten {name} needs {', '.join(sorted(required & children))} before {child}")
                needed.update(required)
                bound.update(self.coordinates(child))
        elif definition != "worktile":
            raise GraphError(f"invalid index domain: {name}")
        for dependency in tuple(needed):
            needed.update(self.dependencies(dependency, visited | {name}))
        self._dependencies[name] = needed
        return needed

    def has_worktiles(self, name: str, seen=frozenset()) -> bool:
        if name in seen or name not in self.definitions:
            return False
        definition = self.definitions[name]
        if definition == "worktile":
            return True
        if isinstance(definition, dict):
            names = definition["flatten"]
        else:
            names = set().union(*(referenced_domains(tree(value)) for value in definition)) & self.definitions.keys()
        return any(self.has_worktiles(child, seen | {name}) for child in names)

    def integer(self, expression, env: dict[str, int]) -> int:
        value = self.evaluate(tree(expression), env)
        if type(value) is bool or value.denominator != 1:
            raise GraphError(f"noninteger index/bound: {expression} = {value}")
        return int(value)

    def predicate(self, expression: str, env: dict[str, int]) -> bool:
        value = self.evaluate(tree(expression), env)
        if type(value) is bool:
            return value
        if value in (0, 1):
            return bool(value)
        raise GraphError(f"condition must yield a boolean or 0/1: {expression} = {value}")

    def integers(self, expressions, env: dict[str, int]) -> list[int]:
        values, boundary = [], None
        for expression in expressions:
            try:
                values.append(self.integer(expression, env))
            except Boundary as error:
                boundary = boundary or error
        if boundary is not None:
            raise boundary
        return values

    def evaluate_all(self, nodes, env):
        values, boundary = [], None
        for node in nodes:
            try:
                values.append(self.evaluate(node, env))
            except Boundary as error:
                boundary = boundary or error
        if boundary is not None:
            raise boundary
        return values

    def evaluate(self, node: ast.AST, env: dict[str, int]) -> Fraction | bool:
        value = self._evaluate(node, env)
        if type(value) is bool:
            return value
        if abs(value) > 10**12:
            raise ExpansionError("sample expression magnitude exceeds the tool limit (10^12)")
        return Fraction(value)

    def _parameter(self, node, env):
        """An indexed/called sample may be supplied for its concrete arguments."""
        canonical = ast.unparse(node).replace(" ", "")
        if canonical in self.parameters:
            return self.parameters[canonical]
        if isinstance(node, (ast.Call, ast.Subscript)):
            name = node.func if isinstance(node, ast.Call) else node.value
            if isinstance(name, ast.Name):
                if isinstance(node, ast.Call):
                    if node.keywords:
                        return None
                    args, brackets = node.args, ("(", ")")
                else:
                    args = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
                    brackets = ("[", "]")
                if any(isinstance(arg, ast.Slice) for arg in args):
                    return None
                values = self.evaluate_all(args, env)
                if any(type(value) is bool or value.denominator != 1 for value in values):
                    raise GraphError("sample lookup arguments must be integers: " + canonical)
                concrete = name.id + brackets[0] + ",".join(str(int(value)) for value in values) + brackets[1]
                return self.parameters.get(concrete)
        return None

    def _evaluate(self, node, env):
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            if node.id in {"true", "false"}:
                return node.id == "true"
        if isinstance(node, ast.Constant) and type(node.value) in (int, bool):
            return node.value
        canonical = ast.unparse(node).replace(" ", "")
        # Domain queries and flatten coordinates always follow declarations;
        # a sample value must not turn a dynamic worktile count into a constant.
        is_query = (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in {"count", "last"})
        is_flat = (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                   and isinstance(self.definitions.get(node.value.id), dict))
        if not is_query and not is_flat and canonical in self.parameters:
            return self.parameters[canonical]
        if isinstance(node, ast.UnaryOp):
            value = self.evaluate(node.operand, env)
            if isinstance(node.op, ast.Not):
                if type(value) is not bool and value not in (0, 1):
                    raise GraphError("not requires a boolean or 0/1: " + canonical)
                return not value
            if type(value) is not bool and isinstance(node.op, (ast.UAdd, ast.USub)):
                return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            for child in node.values:
                value = self.evaluate(child, env)
                if type(value) is not bool and value not in (0, 1):
                    raise GraphError("and/or require booleans or 0/1: " + canonical)
                if isinstance(node.op, ast.And) and not value:
                    return False
                if isinstance(node.op, ast.Or) and value:
                    return True
            return isinstance(node.op, ast.And)
        if isinstance(node, ast.Compare):
            values = self.evaluate_all([node.left, *node.comparators], env)
            result = True
            for a, operator, b in zip(values, node.ops, values[1:]):
                if isinstance(operator, ast.Eq): result &= a == b
                elif isinstance(operator, ast.NotEq): result &= a != b
                elif isinstance(operator, ast.Lt): result &= a < b
                elif isinstance(operator, ast.LtE): result &= a <= b
                elif isinstance(operator, ast.Gt): result &= a > b
                elif isinstance(operator, ast.GtE): result &= a >= b
                else: raise ExpansionError("unsupported comparison: " + canonical)
            return result
        if isinstance(node, ast.BinOp):
            a, b = self.evaluate_all([node.left, node.right], env)
            if type(a) is bool or type(b) is bool:
                raise GraphError("arithmetic operands must be numeric: " + canonical)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and b == 0:
                raise GraphError("division by zero: " + canonical)
            if isinstance(node.op, ast.Div): return a / b
            if isinstance(node.op, ast.FloorDiv): return a // b
            if isinstance(node.op, ast.Mod): return a % b
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            if name in {"count", "last"} and len(node.args) == 1 and isinstance(node.args[0], ast.Name):
                domain = node.args[0].id
                if self.dynamic_worktiles and self.has_worktiles(domain):
                    raise GraphError("count/last on a dynamically refilled worktile domain is undefined")
                values = self.entries(domain, env)
                if name == "count": return len(values)
                if not values: raise GraphError(f"last({domain}) on an empty domain")
                return values[-1][domain]
            values = self.evaluate_all(node.args, env)
            if any(type(value) is bool for value in values):
                raise GraphError("numeric function requires numeric operands: " + canonical)
            if name == "min" and values: return min(values)
            if name == "max" and values: return max(values)
            if name == "floor" and len(values) == 1: return floor(values[0])
            if name == "ceil" and len(values) == 1: return ceil(values[0])
            if name == "ceil_div" and len(values) == 2:
                if not values[1]: raise GraphError("division by zero: " + canonical)
                return ceil(values[0] / values[1])
        if is_flat:
            name = node.value.id
            children = self.definitions[name]["flatten"]
            parts = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
            if len(parts) != len(children) or any(isinstance(part, ast.Slice) for part in parts):
                raise GraphError("an action needs full flatten coordinates: " + canonical)
            coordinates, unavailable, boundary = {}, set(), None
            for child, part in zip(children, parts):
                try:
                    coordinate = self.integer(ast.unparse(part), env)
                except Boundary as error:
                    boundary = boundary or error
                    unavailable.update(self.coordinates(child))
                    continue
                if self.dependencies(child) & unavailable:
                    unavailable.update(self.coordinates(child))
                    continue
                candidate_env = {**env, **coordinates}
                entries = self.entries(child, candidate_env)
                entry = next((item for item in entries if item[child] == coordinate), None)
                if entry is None:
                    message = f"{canonical}: {child}={coordinate} outside its declared domain"
                    if not shifted(part, self):
                        if not entries:
                            raise EmptyCoordinate(message, child, candidate_env)
                        raise GraphError(message)
                    boundary = boundary or Boundary(message)
                    unavailable.update(self.coordinates(child))
                else:
                    coordinates = merge_coordinates(coordinates, entry)
            if boundary is not None:
                raise boundary
            external = {key: value for key, value in env.items() if key not in self.coordinates(name)}
            for entry in self.entries(name, external):
                if all(entry[key] == value for key, value in coordinates.items()):
                    return entry[name]
            raise GraphError(f"no flatten coordinate in {name}: {coordinates}")
        parameter = self._parameter(node, env)
        if parameter is not None:
            return parameter
        raise ExpansionError(f"unbound or unsupported expression {canonical!r}; supply an explicit sample value")

    def entries(self, name: str, env: dict[str, int]) -> list[dict[str, int]]:
        if name not in self.definitions:
            raise GraphError(f"undefined index domain: {name}")
        if name in self._active:
            raise GraphError(f"recursive index domain: {name}")
        if len(self._active) >= DEPENDENCY_DEPTH:
            raise ExpansionError(f"index-domain dependency depth exceeds {DEPENDENCY_DEPTH}")
        self._active.add(name)
        try:
            definition = self.definitions[name]
            if definition == "worktile":
                return [{name: value} for value in range(self.worktiles)]
            if isinstance(definition, list):
                lo, hi = self.integers(definition, env)
                if hi < lo:
                    raise GraphError(f"reversed domain {name}: [{lo}, {hi})")
                if hi - lo > LIMIT:
                    raise ExpansionError(f"domain {name} exceeds {LIMIT} entries; use a smaller sample")
                return [{name: value} for value in range(lo, hi)]
            if not isinstance(definition, dict) or "flatten" not in definition:
                raise GraphError(f"invalid index domain: {name}")
            result = [{}]
            for child in definition["flatten"]:
                expanded = []
                for prefix in result:
                    for item in self.entries(child, {**env, **prefix}):
                        if len(expanded) >= LIMIT:
                            raise ExpansionError(f"flatten domain {name} exceeds {LIMIT} entries; use a smaller sample")
                        expanded.append(merge_coordinates(prefix, item))
                result = expanded
            return [{**item, name: value} for value, item in enumerate(result)]
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
            yield dict(env)
            return
        ready = [name for name in remaining if self.dependencies(name) <= env.keys()]
        if not ready:
            raise ExpansionError("cannot order dependent index domains: " + ", ".join(sorted(remaining)))
        name = min(ready, key=lambda key: (not isinstance(self.definitions.get(key), dict), key))
        for item in self.entries(name, env):
            yield from self.contexts(names, merge_coordinates(env, item))

    def select(self, spec: ForSpec, env: dict[str, int]) -> list[dict[str, int]]:
        definition = self.definitions.get(spec.name)
        axes = definition["flatten"] if isinstance(definition, dict) else [spec.name]
        if len(spec.selectors) > len(axes):
            raise GraphError("too many coordinates in " + spec.text)
        selectors = []
        for selector in spec.selectors:
            if isinstance(selector, Slice):
                lo = self.integer(selector.lower, env) if selector.lower is not None else None
                hi = self.integer(selector.upper, env) if selector.upper is not None else None
                if lo is not None and hi is not None and hi < lo:
                    raise GraphError("reversed slice in " + spec.text)
                selectors.append((lo, hi, None))
            else:
                selectors.append((None, None, self.integer(selector, env)))
        result = []
        for item in self.entries(spec.name, env):
            if all((lo is None or item[axis] >= lo) and (hi is None or item[axis] < hi)
                   and (exact is None or item[axis] == exact)
                   for axis, (lo, hi, exact) in zip(axes, selectors)):
                # Selection is explicit. Once selected, restoring conflicting
                # coordinates is an error, including a bare nested for(n).
                merge_coordinates(env, item)
                result.append(item)
        return result


def shifted(node: ast.AST, domains: Domains) -> bool:
    """Recognize a variable displacement, never classify a bare index as shifted."""
    if isinstance(node, ast.Subscript):
        parts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        return any(shifted(part, domains) for part in parts)
    return (isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub))
            and bool(variables(node) & domains.definitions.keys()))
