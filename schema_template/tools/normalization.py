"""One in-memory normalization plan for a complete task source snapshot.

Structure owns field spelling and scalar types. This module owns binding:
declarations, recognizable uses, lexical pipe/formal scopes, and all consumers
of a shared ``from`` region. It never writes files or interprets prose as code.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

from .diagnostics import Diagnostic, field_path
from .errors import SchemaError, ToolError, bounded
from .pipeline_syntax import DOMAIN_FUNCTIONS, VERBS, _group_end, parse_members, split_top
from .yaml_format import (A, H, M, W, P, WS, PS, INFO, ScalarEdit,
                          canonical_parts, field_case_edits, format_document)
from .yaml_source import (Document, loads, recover_scalar_quotes,
                          repair_missing_quotes, scalar_contexts, ScalarContext)

NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
TOKEN = re.compile(r"(?<![\w.])[A-Za-z_][A-Za-z0-9_]*(?!\w)")
REGIONS = ("Math_reshape", "Work_schedule", "Tile_pipeline")
AREAS = (A, H, M, W, P)


def _matches(parts, pattern):
    """Internal semantic paths follow casing-only decisions in the field table."""
    return len(parts) == len(pattern) and all(
        isinstance(part, str) if rule is None else type(part) is int if rule is int else
        isinstance(part, str) and part.casefold() == rule.casefold()
        for part, rule in zip(parts, pattern)
    )


@dataclass(frozen=True)
class LoadedDocument:
    document: Document
    source_sha256: str
    corrections: tuple[Diagnostic, ...] = ()


@dataclass(frozen=True)
class NormalizedTask:
    files: dict[str, Path]
    documents: dict[str, Document]
    candidates: dict[str, str]
    source_hashes: dict[str, str]
    corrections: tuple[Diagnostic, ...]


def loads_for_normalization(text: str, name: str, *, path="<memory>", skip_hardware=False) -> LoadedDocument:
    """Opt-in source recovery, retaining the hash of the actual original bytes."""
    from .yaml_source import ReadError
    if not isinstance(text, str):
        raise ReadError(f"{path}: YAML input must be text")
    try:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    except UnicodeError as error:
        raise ReadError(f"{path}: YAML must be valid UTF-8") from error
    corrections = []
    candidate = repair_missing_quotes(text, name, path=path, corrections=corrections, skip_hardware=skip_hardware)
    document = recover_scalar_quotes(loads(candidate, name, path=path), corrections=corrections,
                                     skip_hardware=skip_hardware)
    return LoadedDocument(document, digest, tuple(corrections))


def load_for_normalization(path: Path | str, name: str, *, skip_hardware=False) -> LoadedDocument:
    from .yaml_source import ReadError
    path = Path(path)
    try:
        text = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReadError(f"{path}: YAML must be valid UTF-8") from error
    return loads_for_normalization(text, name, path=path, skip_hardware=skip_hardware)


def _location(name, parts):
    for part in parts:
        name = f"{name}[{part}]" if type(part) is int else field_path(name, part)
    return name


def _get(value, *parts):
    for part in parts:
        if not isinstance(value, (dict, list)):
            return None
        try:
            value = value[part]
        except KeyError:
            # This is a post-field-normalization view. Casing-only decisions
            # in the canonical table do not create another reader spelling.
            key = next((key for key in value if isinstance(key, str) and isinstance(part, str)
                        and key.casefold() == part.casefold()), None)
            if key is None:
                return None
            value = value[key]
        except (IndexError, TypeError):
            return None
    return value


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _sequence(value):
    return value if isinstance(value, list) else []


class _Names:
    """One namespace; the stored original spelling makes collisions explicit."""
    def __init__(self, previous=None):
        self.entries = dict(previous.entries) if previous is not None else {}

    def add(self, name, path, *, target=None, related=False):
        if not isinstance(name, str):
            return
        target = name.lower() if target is None else target
        previous = self.entries.get(name.casefold())
        if previous and (previous[1] != target or previous[0] != name and not related):
            raise SchemaError(f"{path}: name normalization is ambiguous: {name!r} and "
                              f"{previous[0]!r} at {previous[2]} would share an identity; use unambiguous declarations")
        self.entries[name.casefold()] = (name, target, path)

    def add_all(self, names, *, related=False):
        for name, target, path in names.entries.values():
            self.add(name, path, target=target, related=related)

    def find(self, name):
        entry = self.entries.get(name.casefold())
        return entry[1] if entry else name


def _tokens(value, names):
    """Change bound roots/tokens, preserving strings and attribute members."""
    protected = [(match.start(), match.end()) for match in re.finditer(
        r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"", value)]
    protected.extend((match.start(1), match.end(1)) for match in re.finditer(
        r"\.\s*([A-Za-z_][A-Za-z0-9_]*)", value))

    def replace(match):
        if any(start <= match.start() < end for start, end in protected):
            return match[0]
        return names.find(match[0])

    return TOKEN.sub(replace, value)


def _name_text(value, names):
    match = re.fullmatch(r"(\s*)(.*?)(\s*)", value, re.DOTALL)
    return match[1] + names.find(match[2]) + match[3]


def _formal(value, path):
    match = re.fullmatch(r"\s*([A-Za-z_][A-Za-z0-9_]*)(?:\s*\((.*?)\))?\s*", value, re.DOTALL)
    if match is None:
        return None, ()
    arguments = tuple(part.strip() for part in match[2].split(",")) if match[2] is not None else ()
    if not all(NAME.fullmatch(argument) for argument in arguments):
        return None, ()
    if len({argument.casefold() for argument in arguments}) != len(arguments):
        raise SchemaError(f"{path}: formal parameters collide after lowercase normalization")
    return match[1], arguments


def _formula_scopes(value, names, established, path):
    """Bind simple definitions across operator continuations, not arbitrary prose."""
    lines = value.splitlines(keepends=True)
    definitions = [re.match(
        r"^[ \t]*([A-Za-z_][A-Za-z0-9_]*)(?:[ \t]*([([])([^\n]*?)[)\]])?[ \t]*=(?!=)", line)
        for line in lines]
    quantifiers = [list(re.finditer(r"\bfor\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\b", line)) for line in lines]
    operators, local = tuple("+-*/%@^·×÷−"), {}
    for index, match in enumerate(definitions):
        if match is None:
            continue
        names.add(match[1], path, related=match[1].casefold() in established)
        if not match[2]:
            continue
        arguments = [parameter.strip() for parameter in match[3].split(",")]
        if (match[2] == "(" and all(NAME.fullmatch(parameter) for parameter in arguments) and
                len({parameter.casefold() for parameter in arguments}) != len(arguments)):
            raise SchemaError(f"{path}: function parameters collide after normalization")
        parameters = _Names()
        for parameter in arguments:
            if NAME.fullmatch(parameter):
                parameters.add(parameter, path, related=True)
        if not parameters.entries:
            continue
        stop = next((line for line in range(index + 1, len(lines))
                     if definitions[line] or not lines[line].strip()), len(lines))
        fragment = "".join(lines[index:stop])

        def ambiguous(line):
            if _tokens(fragment, parameters) != fragment:
                raise SchemaError(f"{path}: cannot uniquely bind a multiline formal definition "
                                  f"of {match[1]!r} near Math line {line + 1}; "
                                  "make its parameter scope explicit before normalizing names")

        end, rhs = index + 1, lines[index][match.end():]
        if not rhs.strip() or split_top(rhs, ",") is None:
            ambiguous(index)
        else:
            while end < len(lines) and (lines[end - 1].rstrip().endswith(operators) or
                                       lines[end].lstrip().startswith(operators)):
                if end >= stop or split_top(lines[end], ",") is None:
                    ambiguous(end)
                    break
                end += 1
            if lines[end - 1].rstrip().endswith(operators):
                ambiguous(end - 1)
            # A bare following expression could still belong to this definition.
            # Do not publish half a rename when its scope cannot be established.
            for line in range(end, stop):
                unbound = _Names(parameters)
                for quantifier in quantifiers[line]:
                    unbound.entries.pop(quantifier[1].casefold(), None)
                if _tokens(lines[line], unbound) != lines[line]:
                    ambiguous(line)
        for line in range(index, end):
            local.setdefault(line, _Names()).add_all(parameters, related=True)
    result, offset = {}, 0
    for index, line in enumerate(lines):
        for match in quantifiers[index]:
            local.setdefault(index, _Names()).add(match[1], path, related=True)
        if index in local:
            result[offset, offset + len(line)] = local[index]
        offset += len(line)
    return result


def _origins(raw):
    """Resolve only source identity, before strict field/semantic validation."""
    origins, edges = {}, {name: [] for name in raw}
    for name, value in raw.items():
        for region in REGIONS:
            body = _get(value, "DESIGN", region)
            if isinstance(body, dict) and any(key.casefold() == "from" for key in body):
                if len(body) != 1:
                    raise SchemaError(f"{name}.DESIGN.{region}: from must be the only field")
                target = _get(body, "from")
                if not isinstance(target, str) or target not in raw:
                    raise SchemaError(f"{name}.DESIGN.{region}.from: unknown source {target!r}; supply the complete task snapshot")
                edges[name].append(target)
    active, done = [], set()

    def visit(name):
        if name in active:
            raise SchemaError("from normalization requires an acyclic task: " + " -> ".join(active + [name]))
        if name in done:
            return
        active.append(name)
        for target in edges[name]:
            visit(target)
        active.pop()
        done.add(name)
        for region in REGIONS:
            body = _get(raw[name], "DESIGN", region)
            target = _get(body, "from")
            origins[name, region] = origins[target, region] if target is not None else name

    for name in raw:
        visit(name)
    return origins


LOWER_KEYS = (
    *(prefix + (None,) for prefix in (A + ("Constant",), A + ("Input",), A + ("Output",),
                                      A + ("Compute", int, "Symbol"), M + ("Rewrite", int, "Symbol"),
                                      A + ("Golden_reference", "Tolerance"),
                                      WS + ("Decomposition", "Partition"),
                                      WS + ("Decomposition", "Worktile", "Invariant"),
                                      WS + ("Decomposition", "Worktile", "Iteration"),
                                      WS + ("Decomposition", "Target", "Single_tile"),
                                      WS + ("Schedule", "Worker_assignment", "Refill"),
                                      W + ("Flow", "Handoff"), PS + ("Index",), PS + ("Storage",),
                                      H + ("Execution",), H + ("Feature",))),
    WS + ("Mapping", "Member", int, None),
    PS + ("Storage", None, "view", None, "region", None),
)


class _Scope:
    """Bindings visible to one effective node, including its imported regions."""
    def __init__(self, name, value, fields, *, skip_hardware=False):
        self.name, self.value, self.fields, self.skip_hardware = name, value, fields, skip_hardware
        self.algorithm, self.math = _Names(), _Names()
        self.hardware, self.steps, self.titles = _Names(), _Names(), {A: [], M: []}
        self.work, self.indices, self.roles, self.events = {}, {}, {}, {}
        self.formals, self.pipe_scopes, self.family_scopes, self.formulas = {}, {}, {}, {}
        self.objects = {}
        self._collect()

    def path(self, parts):
        return _location(self.name, parts)

    def _table(self, parts, names, *, related=False, target=None):
        for key in _mapping(_get(self.value, *parts)):
            names.add(key, self.path(parts + (key,)), related=related,
                      target=target(key) if target is not None else None)

    def _axes(self, parts, names):
        def visit(value, path):
            if isinstance(value, dict):
                fields = next((keys for pattern, keys in self.fields if _matches(path, pattern)), ())
                if "shape" in fields:
                    for index, axis in enumerate(_sequence(value.get("shape"))):
                        for key in _mapping(axis):
                            names.add(key, self.path(path + ("shape", index, key)), related=True)
                for key, child in value.items():
                    visit(child, path + (key,))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, path + (index,))
        visit(_get(self.value, *parts), parts)

    def _collect(self):
        for group in ("Constant", "Input", "Output"):
            self._table(A + (group,), self.algorithm)
            for key in _mapping(_get(self.value, *A, group)):
                self.objects[(A, key.casefold())] = A + (group, key)
        for index, entry in enumerate(_sequence(_get(self.value, *A, "Compute"))):
            prefix = A + ("Compute", index)
            self._table(prefix + ("Symbol",), self.algorithm)
            for key in _mapping(_get(entry, "Symbol")):
                self.objects[(A, key.casefold())] = prefix + ("Symbol", key)
            title = _get(entry, "Name")
            if isinstance(title, str):
                self.titles[A].append((title, self.path(prefix + ("Name",))))
        # Input axes define runtime extents. Repeated identical extents share
        # one declaration; an extent never silently merges into a value object.
        for key, entry in _mapping(_get(self.value, *A, "Input")).items():
            for index, axis in enumerate(_sequence(_get(entry, "shape"))):
                for dim, extent in _mapping(axis).items():
                    if isinstance(extent, str) and NAME.fullmatch(extent):
                        self.algorithm.add(extent, self.path(A + ("Input", key, "shape", index, dim)), related=True)
        self._axes(A, self.algorithm)
        self.math.add_all(self.algorithm)
        for index, entry in enumerate(_sequence(_get(self.value, *M, "Rewrite"))):
            prefix = M + ("Rewrite", index)
            self._table(prefix + ("Symbol",), self.math)
            for key in _mapping(_get(entry, "Symbol")):
                self.objects[(M, key.casefold())] = prefix + ("Symbol", key)
            title = _get(entry, "Name")
            if isinstance(title, str):
                self.titles[M].append((title, self.path(prefix + ("Name",))))
        self._axes(M, self.math)
        for area, collection, names in ((A, "Compute", self.algorithm), (M, "Rewrite", self.math)):
            established = set(names.entries)
            for index, entry in enumerate(_sequence(_get(self.value, *area, collection))):
                formula = _get(entry, "Math")
                if not isinstance(formula, str):
                    continue
                parts = area + (collection, index, "Math")
                self.formulas[parts] = _formula_scopes(formula, names, established, self.path(parts))
        # Contract formulas are visible to later regions as mathematical names.
        self.math.add_all(self.algorithm, related=True)
        if not self.skip_hardware:
            for group in ("Execution", "Feature"):
                self._table(H + (group,), self.hardware, related=True)
        for area in (W, P):
            table = _mapping(_get(self.value, *area, "Step"))
            single = all(len(entries) == 1 for entries in (
                _mapping(_get(self.value, *region, "Step")) for region in (W, P)) if entries)
            for step in table:
                self.steps.add(step, self.path(area + ("Step", step)),
                               target=("MAIN" if single else step.upper()) if NAME.fullmatch(step) else step, related=True)
        for step, body in _mapping(_get(self.value, *W, "Step")).items():
            names = self.work[step.casefold()] = _Names(self.math)
            for builtin in ("worker_count", "worker_id", "worktile_count"):
                names.add(builtin, self.path(W + ("Step", step, "Mapping")), related=True)
            for suffix in (("Partition",), ("Worktile", "Invariant"), ("Worktile", "Iteration"), ("Target", "Single_tile")):
                self._table(W + ("Step", step, "Decomposition") + suffix, names, related=True)
            self._axes(W + ("Step", step), names)
        for step, body in _mapping(_get(self.value, *P, "Step")).items():
            prefix = P + ("Step", step)
            names = self.indices[step.casefold()] = _Names(self.work.get(step.casefold(), self.math))
            for function in DOMAIN_FUNCTIONS:
                names.add(function, self.path(prefix + ("Index",)), related=True)
            self._table(prefix + ("Index",), names)
            self._axes(prefix, names)
            events = self.events[step.casefold()] = _Names()
            for builtin in ("kernel_done", "launch_ready"):
                events.add(builtin, self.path(prefix + ("Event",)))
            roles = self.roles[step.casefold()] = _Names()
            role_table = _mapping(_get(body, "Role"))
            for role in role_table:
                roles.add(role, self.path(prefix + ("Role", role)),
                          target=("MAIN" if len(role_table) == 1 else role.upper()) if NAME.fullmatch(role) else role)
            for key in _mapping(_get(body, "Event")):
                ep = prefix + ("Event", key)
                base, parameters = _formal(key, self.path(ep))
                if base is not None:
                    events.add(base, self.path(ep))
                    local = self.formals[ep] = _Names(names)
                    for parameter in parameters:
                        local.add(parameter, self.path(ep), related=True)
            for storage, block in _mapping(_get(body, "Storage")).items():
                for key in _mapping(_get(block, "view")):
                    vp = prefix + ("Storage", storage, "view", key)
                    base, parameters = _formal(key, self.path(vp))
                    if base is not None:
                        local = self.formals[vp] = _Names(names)
                        for parameter in parameters:
                            local.add(parameter, self.path(vp), related=True)
            for role, definition in role_table.items():
                self._pipe(_get(definition, "Pipe"), prefix + ("Role", role, "Pipe"),
                           _Names(names), set(), step, role)

    def _pipe(self, value, parts, names, active, step, role):
        self.pipe_scopes[parts] = names
        for index, entry in enumerate(_sequence(value)):
            path = parts + (index,)
            self.pipe_scopes[path] = names
            if isinstance(entry, str):
                family = _action_family(entry)
                if family:
                    self.family_scopes.setdefault((step.casefold(), role.casefold(), family), []).append(names)
                continue
            if not isinstance(entry, dict) or len(entry) != 1:
                continue
            control, body = next(iter(entry.items()))
            if not isinstance(control, str):
                continue
            child, bound = _Names(names), set(active)
            match = re.fullmatch(r"\s*for\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*=.*\)\s*", control, re.I | re.S)
            if match:
                parameter = match[1]
                if parameter.casefold() in bound:
                    raise SchemaError(f"{self.path(path + (control,))}: local loop indices collide in one active scope")
                # A local for binding can shadow an outer *unbound* parameter.
                child.entries.pop(parameter.casefold(), None)
                child.add(parameter, self.path(path + (control,)))
                bound.add(parameter.casefold())
            else:
                match = re.fullmatch(r"\s*for\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)(?:\[.*\])?\s*\)\s*", control, re.I | re.S)
                if match:
                    bound.add(match[1].casefold())
            self._pipe(body, path + (control,), child, bound, step, role)

    def names(self, parts):
        if parts[:len(A)] == A:
            names = self.algorithm
        else:
            names = self.math
        if len(parts) >= 4 and parts[:3] == W + ("Step",):
            names = self.work.get(parts[3].casefold(), names)
        if len(parts) >= 4 and parts[:3] == P + ("Step",):
            names = self.indices.get(parts[3].casefold(), names)
        for length in range(len(parts), 0, -1):
            prefix = parts[:length]
            if prefix in self.pipe_scopes:
                return self.pipe_scopes[prefix]
            if prefix in self.formals:
                return self.formals[prefix]
        return names

    def formula(self, value, parts):
        names, result, cursor = self.names(parts), [], 0
        for (start, end), parameters in sorted(self.formulas.get(parts, {}).items()):
            result.append(_tokens(value[cursor:start], names))
            local = _Names(names)
            local.entries.update(parameters.entries)
            result.append(_tokens(value[start:end], local))
            cursor = end
        result.append(_tokens(value[cursor:], names))
        return "".join(result)

    def key(self, parts):
        value = parts[-1]
        if _matches(parts, W + ("Step", None)) or _matches(parts, P + ("Step", None)):
            return self.steps.find(value)
        if _matches(parts, P + ("Step", None, "Role", None)):
            return self.roles[parts[3].casefold()].find(value)
        if any(_matches(parts, pattern) for pattern in LOWER_KEYS):
            return value.lower()
        if (_matches(parts, PS + ("Event", None)) or
                _matches(parts, PS + ("Storage", None, "view", None))):
            base, parameters = _formal(value, self.path(parts))
            return value.lower() if base is not None else value
        if _matches(parts, INFO + ("Using_verb", None, None)):
            return value.lower() if all(NAME.fullmatch(part.strip()) for part in value.split("@")) else value
        if len(parts) >= 3 and parts[-3] == "shape" and type(parts[-2]) is int:
            fields = next((keys for pattern, keys in self.fields if _matches(parts[:-3], pattern)), ())
            if "shape" in fields:
                return value.lower()
        pipe = PS + ("Role", None, "Pipe")
        if len(parts) > len(pipe) and _matches(parts[:len(pipe)], pipe):
            return _control(value, self.names(parts[:-1]), self.path(parts))
        return value


def _action_family(value):
    parts = split_top(value, ":")
    if parts is None or len(parts) != 2 or parts[0].strip().lower() not in VERBS:
        return None
    operands = split_top(parts[1], "@")
    if operands is None:
        return None
    bases = []
    for operand in operands:
        match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)", operand)
        if match is None:
            return None
        bases.append(match[1].lower())
    return parts[0].strip().lower(), "@".join(bases)


def _action(value, names):
    parts = split_top(value, ":")
    if parts is None or len(parts) != 2 or parts[0].strip().lower() not in VERBS:
        return value
    operands = split_top(parts[1], "@")
    if operands is None:
        return value
    changed = []
    for operand in operands:
        match = re.match(r"(\s*)([A-Za-z_][A-Za-z0-9_]*)(.*)\Z", operand, re.S)
        if match is None:
            return value
        tail = match[3]
        if tail.strip() and (not tail.strip().startswith("(") or _group_end(tail.strip(), 0) != len(tail.strip()) - 1):
            return value
        changed.append(match[1] + match[2].lower() + _tokens(tail, names))
    return parts[0].lower() + ":" + "@".join(changed)


def _control(value, names, path):
    match = re.fullmatch(r"(\s*)(for|if)(\s*\()(.*)(\)\s*)", value, re.I | re.S)
    if match is None or split_top(match[4], ",") is None:
        return value
    expression = match[4]
    if match[2].lower() == "for":
        local = re.match(r"(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*=)", expression)
        if local:
            expression = local[1] + local[2].lower() + local[3] + _tokens(expression[local.end():], names)
        else:
            expression = _tokens(expression, names)
    else:
        expression = _tokens(expression, names)
    return match[1] + match[2].lower() + match[3] + expression + match[5]


def _event(value, scope, parts, *, waiting=False):
    match = re.fullmatch(r"(\s*\{\s*)(issue|done)(\s+)(.*)(\}\s*)", value, re.I | re.S)
    if match is None:
        return value
    segments = split_top(match[4], "::")
    if segments is None or len(segments) not in (1, 2):
        return value
    names, step = scope.names(parts), parts[3]
    role = parts[5] if len(parts) > 5 and parts[4] == "Role" else None
    qualifier = ""
    if len(segments) == 2:
        qualified = re.fullmatch(r"(\s*)([A-Za-z_][A-Za-z0-9_]*)(.*)", segments[0], re.S)
        if qualified is None:
            return value
        role = qualified[2]
        tail = qualified[3]
        if tail.strip() and (not tail.strip().startswith("[") or _group_end(tail.strip(), 0) != len(tail.strip()) - 1):
            return value
        qualifier = qualified[1] + scope.roles.get(step.casefold(), _Names()).find(role) + _tokens(tail, names) + "::"
    if waiting and role is not None and (family := _action_family(segments[-1])):
        local = _Names(names)
        seen = {}
        for candidate in scope.family_scopes.get((step.casefold(), role.casefold(), family), []):
            for fold, (original, target, location) in candidate.entries.items():
                if fold in seen and seen[fold][0] != original and re.search(r"\b" + re.escape(original) + r"\b", value, re.I):
                    raise SchemaError(f"{scope.path(parts)}: Waiting index {original!r} is ambiguous between "
                                      f"{seen[fold][1]} and {location} after normalization")
                seen[fold] = original, location
                local.entries[fold] = (original, target, location)
        names = local
    return match[1] + match[2].lower() + match[3] + qualifier + _action(segments[-1], names) + match[5]


def _waiting(value, scope, parts):
    halves = split_top(value, "←")
    if halves is None or len(halves) != 2:
        return value
    right = re.fullmatch(r"(\s*\[)(.*)(\]\s*)", halves[1], re.S)
    predecessors = split_top(right[2], ",") if right else None
    if predecessors is None:
        return value
    return (_event(halves[0], scope, parts, waiting=True) + "←" + right[1] +
            ",".join(_event(part, scope, parts, waiting=True) for part in predecessors) + right[3])


def _member(value, scope, parts):
    if parse_members(value) is None:
        return value
    result, cursor, names = [], 0, scope.names(parts)
    while cursor < len(value):
        opening = value.find("[", cursor)
        end = _group_end(value, opening)
        result.append(_name_text(value[cursor:opening], scope.hardware))
        result.append("[" + _tokens(value[opening + 1:end], names) + "]")
        cursor = end + 1
        separator = re.match(r"\s*\.\s*", value[cursor:])
        if separator:
            result.append(separator[0])
            cursor += separator.end()
        elif value[cursor:].strip() == "":
            result.append(value[cursor:])
            break
    return "".join(result)


def _tree_reference(text, value, parts, scope):
    if not isinstance(value, dict) or not text:
        return text
    candidates = []
    for key in value:
        if not isinstance(key, str):
            continue
        prefix = text[:len(key)]
        rest = text[len(key):]
        if prefix.casefold() == key.casefold() and (not rest or rest.startswith(".") or rest.startswith("::")):
            candidates.append(key)
    if not candidates:
        if parts in (W, P):
            return _tree_reference(text, _get(value, "Step"), parts + ("Step",), scope)
        if _matches(parts, PS):
            return _tree_reference(text, _get(value, "Role"), parts + ("Role",), scope)
        return text
    key = max(candidates, key=len)
    target = scope.key(parts + (key,))
    rest = text[len(key):]
    if not rest:
        return target
    delimiter = "::" if rest.startswith("::") else "."
    return target + delimiter + _tree_reference(rest[len(delimiter):], value[key], parts + (key,), scope)


def _title_target(text, scope, areas, path):
    matching = [(title, location) for area in areas for title, location in scope.titles[area]
                if title.casefold() == text.casefold()]
    if any(title == text for title, _ in matching):
        return text
    targets = {title for title, _ in matching}
    if len(targets) > 1:
        raise SchemaError(f"{path}: reference capitalization is ambiguous between " +
                          ", ".join(f"{title!r} at {location}" for title, location in matching))
    return next(iter(targets), None)


def _area_target(text, area, scope):
    if area in (A, M):
        title = _title_target(text, scope, (area,), scope.path(area))
        exact_object = any(owner == area and text == path[-1] for (owner, _), path in scope.objects.items())
        if title is not None and not exact_object:
            return title
        for (owner, folded), path in scope.objects.items():
            if owner != area:
                continue
            key = path[-1]
            rest = text[len(key):]
            if text[:len(key)].casefold() == folded and (not rest or rest.startswith(".")):
                return key.lower() + ("." + _tree_reference(rest[1:], _get(scope.value, *path), path, scope) if rest else "")
    rewritten = _tree_reference(text, _get(scope.value, *area), area, scope)
    return _tokens(rewritten, scope.algorithm if area == A else scope.math) if rewritten == text and area in (A, M) else rewritten


def _reference(value, scope, parts):
    match = re.fullmatch(r"(\s*)((?:(?i:ref|bind):)?)(.*?)(\s*)", value, re.S)
    lead, marker, text, trailing = match.groups()
    # Source functions, paths and revision spellings are external identities.
    if re.match(r"<?[^@<>\s]+@[^:<>\s]+:", text) or re.match(r"[A-Za-z][A-Za-z0-9+.-]*://", text):
        return value
    if "::" in text:
        owner, target = text.split("::", 1)
        for area in AREAS:
            area_name = ".".join(area)
            if owner.casefold() == area_name.casefold():
                if area == H and scope.skip_hardware:
                    return value
                text = area_name + "::" + _area_target(target, area, scope)
                break
            if owner[:len(area_name) + 1].casefold() == area_name.casefold() + ".":
                if area == H and scope.skip_hardware:
                    return value
                suffix = owner[len(area_name) + 1:] + "::" + target
                text = area_name + "." + _area_target(suffix, area, scope)
                break
    else:
        names = _Names(scope.names(parts))
        title = _title_target(text, scope, (A, M), scope.path(parts))
        registered = names.entries.get(text.casefold())
        if title is not None and (registered is None or registered[0] != text):
            if registered is not None and title != text and registered[1] != title:
                raise SchemaError(f"{scope.path(parts)}: reference {text!r} has ambiguous mathematical and Name targets")
            text = title
        elif registered is not None:
            text = names.find(text)
        elif marker:
            for area in ((A,) if parts[:len(A)] == A else (M, A)):
                changed = _area_target(text, area, scope)
                if changed != text:
                    text = changed
                    break
            text = _tokens(text, names)
        else:
            text = _tokens(text, names)
    return lead + marker.lower() + text + trailing


def _flow_event(value, scope, parts):
    match = re.fullmatch(r"(\s*)([A-Za-z_][A-Za-z0-9_]*)(\.)(.*?)(\s*)", value, re.S)
    if match is None:
        return value
    step = match[2]
    tail = match[4]
    event = re.match(r"([A-Za-z_][A-Za-z0-9_]*)(.*)\Z", tail, re.S)
    if event is None:
        return value
    names = scope.indices.get(step.casefold(), scope.math)
    base = scope.events.get(step.casefold(), _Names()).find(event[1])
    return match[1] + scope.steps.find(step) + match[3] + base + _tokens(event[2], names) + match[5]


def _retention(value, scope, parts):
    match = re.match(r"(carry_forward|keep_value)(\s+)", value, re.I)
    if match is None:
        return value
    names = _Names(scope.names(parts))
    iteration = re.search(r";\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", value)
    if iteration:
        names.entries.pop(iteration[1].casefold(), None)
        names.add(iteration[1], scope.path(parts))
    rest = _tokens(value[match.end():], names)
    rest = re.sub(r"^invocation\b", "invocation", rest, flags=re.I)
    return match[1].lower() + match[2] + rest


def _expression_field(parts, fields):
    # These positions carry expressions, not arbitrary descriptive strings.
    if any(_matches(parts, pattern) for pattern in (
        A + ("Constant", None, "value"), A + ("Input", None, "range"),
        A + ("Input", None, "range", int), A + ("Input", None, "default"),
        A + ("Compute", int, "Math"), M + ("Rewrite", int, "Math"),
        M + ("Rewrite", int, "Symbol", None, "retention"),
        WS + ("Mapping", "Worker_count"), WS + ("Mapping", "Worker", "unit_count"),
        WS + ("Mapping", "Placement", "unit_count"), WS + ("Mapping", "Member", int, None, int),
        WS + ("Decomposition", "Partition", None, "size"),
        WS + ("Decomposition", "Partition", None, "tile"),
        WS + ("Decomposition", "Partition", None, "range", int),
        WS + ("Schedule", "Worktile_order", int, "group"),
        WS + ("Schedule", "Worker_assignment", "Preassigned", None),
        PS + ("Storage", None, "depth"), PS + ("Storage", None, "total_size"),
        PS + ("Storage", None, "view", None, "slot_index"),
        PS + ("Storage", None, "view", None, "region", None, int),
        PS + ("Index", None, "definition", int), PS + ("Index", None, "definition", "flatten", int),
        WS + ("Decomposition", "Worktile", "Invariant", None, "offset", int),
    )):
        return True
    if len(parts) >= 3 and parts[-3] == "shape" and type(parts[-2]) is int:
        return any(_matches(parts[:-3], pattern) and "shape" in keys for pattern, keys in fields)
    prefix = M + ("Rewrite", int, "Symbol", None, "value")
    return len(parts) >= len(prefix) and _matches(parts[:len(prefix)], prefix) and all(type(part) is int for part in parts[len(prefix):])


def _value(context, scope):
    value, parts = context.node.value, context.parts
    if context.key:
        return scope.key(parts)
    if _matches(parts, ("META_DATA", "Dispatch", "id")) or any(_matches(parts, ("DESIGN", region, "from")) for region in REGIONS):
        return value
    if any(_matches(parts, pattern) for pattern in (
        WS + ("Mapping", "Worker", "unit"), WS + ("Mapping", "Placement", "unit"),
        PS + ("Storage", None, "location"),
    )):
        return _name_text(value, scope.hardware)
    if _matches(parts, INFO + ("Member",)):
        return _member(value, scope, parts)
    if (_matches(parts, W + ("Flow", "Launch", int, "step", int)) or
            _matches(parts, W + ("Flow", "Handoff", None, "producer")) or
            _matches(parts, W + ("Flow", "Handoff", None, "consumer", int))):
        return _name_text(value, scope.steps)
    flow = W + ("Flow",)
    if parts[:len(flow)] == flow and any(part.casefold() in {"start_after", "ready", "wait_at"} for part in parts if isinstance(part, str)):
        return _flow_event(value, scope, parts)
    if _matches(parts, W + ("Flow", "Handoff", None, "storage")):
        if value.lower() == "global":
            return value.lower()
        match = re.match(r"(?i:Tile_pipeline)\.", value)
        return "Tile_pipeline." + _tree_reference(value[match.end():], _get(scope.value, *P), P, scope) if match else value
    if _matches(parts, PS + ("Event", None, int)):
        return _event(value, scope, parts)
    if _matches(parts, PS + ("Role", None, "Waiting", None, int)):
        return _waiting(value, scope, parts)
    pipe = PS + ("Role", None, "Pipe")
    if len(parts) > len(pipe) and _matches(parts[:len(pipe)], pipe):
        return _action(value, scope.names(parts))
    if (_matches(parts, WS + ("Schedule", "Worktile_order", int, "by")) or
            _matches(parts, WS + ("Schedule", "Worktile_order", int, "by", int)) or
            _matches(parts, WS + ("Decomposition", "Target", "Tile_reduction", "dimension"))):
        return _name_text(value, scope.names(parts))
    reference_slot = (
        len(parts) >= 4 and parts[0] == "REALIZATION" and parts[3] == "Item" or
        len(parts) >= 7 and _matches(parts[:6], WS + ("Schedule", "Worker_assignment")) and isinstance(parts[6], str) and parts[6].casefold() == "exit" or
        len(parts) >= 8 and _matches(parts[:7], WS + ("Decomposition", "Target", "Tile_reduction")) and isinstance(parts[7], str) and parts[7].casefold() in {"input", "output", "method"} or
        _matches(parts, W + ("Flow", "Handoff", None, "initialize"))
    )
    if reference_slot or parts[-1:] in (("binding",), ("shape",), ("dtype",)) and re.match(r"(?i:ref|bind):", value):
        return _reference(value, scope, parts)
    if parts[-1:] in (("shape",), ("dtype",)):
        if value.casefold() == "not_applicable":
            return value.lower()
        if re.match(r"(?i:unknown):", value):
            return "unknown:" + value[len("unknown:"):]
    if _matches(parts, M + ("Rewrite", int, "Symbol", None, "retention")):
        return _retention(value, scope, parts)
    if parts in scope.formulas:
        return scope.formula(value, parts)
    if _expression_field(parts, scope.fields):
        return _tokens(value, scope.names(parts))
    if parts == ("META_DATA", "Dispatch", "when"):
        try:
            ast.parse(value, mode="eval")
        except (SyntaxError, ValueError, RecursionError):
            return value
        return _tokens(value, scope.algorithm)
    return value


def _renamed_paths(document, edits):
    keys = {context.path: edit.value for edit in edits for context in edit.contexts if context.key}
    paths = {}
    for context in scalar_contexts(document):
        before = after = document.name
        for part in context.parts:
            if type(part) is int:
                before, after = f"{before}[{part}]", f"{after}[{part}]"
            else:
                before = field_path(before, part)
                after = field_path(after, keys.get(before, part))
            paths[before] = after
    return paths


def _relocate(diagnostics, paths):
    return [Diagnostic(paths.get(item.path, item.path), item.message, item.missing, item.hint, item.severity)
            for item in diagnostics]


def _value_case_edits(document, fields, values):
    for context in scalar_contexts(document):
        if context.key or context.node.tag != "tag:yaml.org,2002:str":
            continue
        parts = canonical_parts(context.parts, fields)
        for pattern, choices in values:
            if not _matches(parts, pattern):
                continue
            target = next((choice for choice in choices if choice.casefold() == context.node.value.casefold()), context.node.value)
            if target != context.node.value:
                yield ScalarEdit(context.node, target, (context,))


def _field_view(value, fields, name, *, parts=(), skip_hardware=False):
    """A read-only binding view, never an alternative source/formatter document.

    Actual source keys remain untouched until the final *combined* edit plan.
    This lets a shared scalar's field and named uses authorize the same change
    together, instead of prematurely rejecting either half of that plan.
    """
    if skip_hardware and parts == H:
        return value
    if isinstance(value, dict):
        result, originals = {}, {}
        for key, child in value.items():
            target = canonical_parts(parts + (key,), fields)[-1]
            if target in result:
                raise SchemaError(f"{_location(name, parts)}: fixed-field normalization would merge "
                                  f"{originals[target]!r} and {key!r} into {target!r}")
            originals[target] = key
            result[target] = _field_view(child, fields, name, parts=parts + (target,), skip_hardware=skip_hardware)
        return result
    if isinstance(value, list):
        return [_field_view(child, fields, name, parts=parts + (index,), skip_hardware=skip_hardware)
                for index, child in enumerate(value)]
    return value


@bounded
def normalize_documents(files, documents, *, canonical_fields=None, skip_hardware=False) -> NormalizedTask:
    """Plan all task files together; no partially normalized source set is returned.

    ``documents`` accepts strict Documents or LoadedDocuments from opt-in read
    recovery. ``candidates`` contains only changed files, while ``documents``
    contains the complete final snapshot. Original source hashes are retained
    independently for the caller's compare-before-write check. Invalid or
    ambiguous bindings raise SchemaError; ordinary semantic validation follows.
    """
    from .rules.structure import CANONICAL_FIELDS, CANONICAL_VALUES
    fields = CANONICAL_FIELDS if canonical_fields is None else tuple(canonical_fields)
    if files.keys() != documents.keys() or not documents:
        raise ToolError("normalization needs matching, nonempty task file/document mappings")
    working, hashes, warnings, field_plans, value_targets = {}, {}, {}, {}, {}
    for name, source in documents.items():
        if isinstance(source, LoadedDocument):
            document, digest, corrections = source.document, source.source_sha256, list(source.corrections)
        else:
            document, digest, corrections = source, source.source_sha256, []
        if document.name != name:
            raise ToolError("normalization document names must match the task snapshot keys")
        document = recover_scalar_quotes(document, corrections=corrections, skip_hardware=skip_hardware)
        hashes[name] = digest
        edits = tuple(edit for edit in (*field_case_edits(document, fields), *_value_case_edits(document, fields, CANONICAL_VALUES))
                      if not skip_hardware or all(len(context.parts) <= 2 or canonical_parts(context.parts, fields)[:2] != H
                                                  for context in edit.contexts))
        field_plans[name] = edits
        value_targets[name] = {context: edit.value for edit in edits for context in edit.contexts if not context.key}
        warnings[name] = corrections
        working[name] = document
    raw = {name: _field_view(document.value, fields, name, skip_hardware=skip_hardware)
           for name, document in working.items()}
    if "root" not in raw and len(raw) > 1:
        raise SchemaError("multi-file normalization needs root to bind the shared Contract; supply the complete task snapshot")
    origins = _origins(raw)
    root = "root" if "root" in raw else next(iter(raw))
    scopes = {}
    for name, value in raw.items():
        effective = dict(value)
        effective["CONTRACT"] = raw[root].get("CONTRACT", {})
        effective["DESIGN"] = {region: _get(raw[origins[name, region]], "DESIGN", region) for region in REGIONS}
        scopes[name] = _Scope(name, effective, fields, skip_hardware=skip_hardware)
    final, candidates = {}, {}
    for name, document in working.items():
        edits = []
        for context in scalar_contexts(document):
            parts = canonical_parts(context.parts, fields)
            if (context.node.tag != "tag:yaml.org,2002:str" or
                    skip_hardware and len(parts) > len(H) and parts[:len(H)] == H):
                continue
            use = ScalarContext(context.node, _location(name, parts), parts, context.key)
            consumers = [name]
            if len(parts) >= 2 and parts[0] == "DESIGN" and parts[1] in REGIONS:
                consumers = [consumer for consumer in raw if origins[consumer, parts[1]] == name]
            elif parts[:1] == ("CONTRACT",) and name == root:
                consumers = list(raw)
            # A raw {from: id} wrapper has no content consumers of its own;
            # its fixed keys still belong to this file and need normalization.
            consumers = consumers or [name]
            proposals = [(consumer, _value(use, scopes[consumer])) for consumer in consumers]
            targets = {target for _, target in proposals}
            if len(targets) > 1:
                uses = ", ".join(f"{consumer}: {target!r}" for consumer, target in proposals)
                raise SchemaError(f"{context.path}: shared source has incompatible normalization bindings ({uses})")
            target = next(iter(targets), context.node.value)
            if target == context.node.value:
                target = value_targets[name].get(context, target)
            if target != context.node.value:
                main_names = tuple(token[0] for token in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", context.node.value)
                                   if any(scope.steps.find(token[0]) == "MAIN" or
                                          any(roles.find(token[0]) == "MAIN" for roles in scope.roles.values())
                                          for scope in (scopes[consumer] for consumer in consumers)))
                edits.append(ScalarEdit(context.node, target, (context,), main_names))
        paths = _renamed_paths(document, edits)
        warnings[name] = _relocate(warnings[name], paths)
        stage = []
        text = format_document(document, edits=edits, skip_hardware=skip_hardware, corrections=stage)
        field_locations = {paths.get(context.path, context.path) for edit in field_plans[name] for context in edit.contexts}
        warnings[name].extend(Diagnostic(item.path,
                                         item.message.replace("identifier+reference:", "field:", 1)
                                         if item.path in field_locations else item.message,
                                         item.missing, item.hint, item.severity) for item in stage)
        final[name] = loads(text, name, path=files[name])
        if final[name].source_sha256 != hashes[name]:
            candidates[name] = text
    corrections = []
    seen = set()
    for name in candidates:
        for diagnostic in warnings[name]:
            identity = diagnostic.path, diagnostic.message
            if identity not in seen:
                corrections.append(diagnostic)
                seen.add(identity)
    return NormalizedTask(dict(files), final, candidates, hashes, tuple(corrections))
