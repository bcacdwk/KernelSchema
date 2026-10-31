"""Reusable field/value checks; mathematical expressions and external names remain open."""
from __future__ import annotations

from decimal import Decimal
import re

from ..diagnostics import Diagnostic, field_path
from ..pipeline_syntax import literal_integer


def reject(errors, path, message, *, missing=False, hint=""):
    errors.append(Diagnostic(path, message, missing, hint))


def predicate(test, description):
    def check(value, path, errors):
        if not test(value):
            reject(errors, path, "expected " + description)
    return check


text = predicate(lambda v: isinstance(v, str) and bool(v.strip()) and v.strip() != "-", "nonempty text")
identifier = predicate(lambda v: isinstance(v, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", v), "an identifier (letters, digits and underscores; no leading digit)")
node_id = predicate(lambda v: isinstance(v, str) and re.fullmatch(r"[a-z_][a-z0-9_]*", v), "a lowercase node ID matching its filename")
number = predicate(lambda v: type(v) in (int, Decimal, float), "a number")


def record(required, optional=None, *, empty=False):
    optional = optional or {}
    def check(value, path, errors):
        if not isinstance(value, dict):
            reject(errors, path, "expected a mapping")
            return
        if not value and not required and not empty:
            reject(errors, path, "empty optional mapping; omit it when there is no content")
        for key in required:
            if key not in value:
                reject(errors, field_path(path, key), "field is required but not present", missing=True)
        for key, item in value.items():
            rule = required.get(key, optional.get(key))
            if rule is None:
                reject(errors, field_path(path, key), "field not allowed; check its spelling and owning section")
            else:
                rule(item, field_path(path, key), errors)
    return check


def named(rule, *, key=identifier, empty=False):
    def check(value, path, errors):
        if not isinstance(value, dict):
            reject(errors, path, "expected a named mapping")
            return
        if not value and not empty:
            reject(errors, path, "expected at least one named entry")
        for name, item in value.items():
            key(name, field_path(path, name), errors)
            rule(item, field_path(path, name), errors)
    return check


def sequence(rule, *, empty=False, unique=False):
    def check(value, path, errors):
        if not isinstance(value, list):
            reject(errors, path, "expected a list")
            return
        if not value and not empty:
            reject(errors, path, "expected a nonempty list")
        for i, item in enumerate(value):
            rule(item, f"{path}[{i}]", errors)
            if unique and item in value[:i]:
                reject(errors, f"{path}[{i}]", "duplicate entry")
    return check


def one_or_many(rule, *, unique=False):
    """Use only for fields whose documentation permits both forms."""
    many = sequence(rule, unique=unique)
    def check(value, path, errors):
        (many if isinstance(value, list) else rule)(value, path, errors)
    return check


def enum(*choices):
    return predicate(lambda v: v in choices, "one of " + ", ".join(repr(c) for c in choices))


def integer(minimum=None, *, expression=True):
    def check(value, path, errors):
        literal = value
        if expression and isinstance(value, str):
            text(value, path, errors)
            if value.startswith(("unknown:", "ref:", "bind:", "<")):
                reject(errors, path, "supply a defined integer expression, not a missing-value marker")
            literal = literal_integer(value)
            if literal is None:
                return  # Expressions are read, never compiled, evaluated or restricted to Python syntax.
        if type(literal) is not int:
            reject(errors, path, "expected an integer" + (" or a defined integer expression" if expression else ""))
        elif minimum is not None and literal < minimum:
            reject(errors, path, f"literal must be >= {minimum}")
    return check


def pair(rule, *, nullable_upper=False):
    def check(value, path, errors):
        if not isinstance(value, list) or len(value) != 2:
            reject(errors, path, "expected a two-endpoint interval")
            return
        rule(value[0], path + "[0]", errors)
        if value[1] is not None or not nullable_upper:
            rule(value[1], path + "[1]", errors)
    return check


def known_text(value, path, errors):
    text(value, path, errors)
    if isinstance(value, str) and (value.startswith(("unknown:", "ref:", "bind:")) or
                                   value.startswith("<") and value.endswith(">")):
        reject(errors, path, "a placeholder is not a known value")


def scalar(value, path, errors):
    if isinstance(value, str):
        known_text(value, path, errors)
    elif type(value) not in (int, Decimal, float, bool):
        reject(errors, path, "expected a known scalar value or expression")


def literal(value, path, errors):
    if isinstance(value, list):
        sequence(literal, empty=True)(value, path, errors)
    else:
        scalar(value, path, errors)


def constant(value, path, errors):
    (known_text if isinstance(value, str) else number)(value, path, errors)


def repository(value, path, errors, *, entry=False, pending=False):
    if not isinstance(value, str) or not re.fullmatch(r"[^@<>\s]+@[^:<>\s]+:[^<>\r\n]+", value):
        reject(errors, path, "expected repository@ref:path" + ("::entry" if entry else ""))
    elif not pending and "@pending:" in value:
        reject(errors, path, "a bound reference needs a real revision, not pending")
    elif entry and ("::" not in value or not value.rsplit("::", 1)[1].strip()):
        reject(errors, path, "reference must identify an entry using ::entry")


def wiki(value, path, errors):
    if isinstance(value, str) and value.startswith("<") and value.endswith(">"):
        repository(value[1:-1], path, errors, pending=True)
    else:
        repository(value, path, errors)


def source(value, path, errors):
    known_text(value, path, errors)
    if isinstance(value, str) and "@pending:" in value:
        reject(errors, path, "source must identify actual material, not a pending reference")


def reference(value, path, errors):
    text(value, path, errors)
    if isinstance(value, str) and value.startswith(("unknown:", "<")):
        reject(errors, path, "reference an existing upstream object")


def reference_list(value, path, errors):
    (sequence(reference, unique=True) if isinstance(value, list) else reference)(value, path, errors)


def shape(value, path, errors):
    if not isinstance(value, list):
        reject(errors, path, "expected ordered single-axis mappings; use [] for a scalar")
        return
    for i, axis in enumerate(value):
        if not isinstance(axis, dict) or len(axis) != 1:
            reject(errors, f"{path}[{i}]", "each shape axis must be one name: size mapping")
        else:
            named(integer(0))(axis, f"{path}[{i}]", errors)


def attribute(attribute_name, *, context="data"):
    def check(value, path, errors):
        if attribute_name == "shape" and isinstance(value, list):
            shape(value, path, errors)
            return
        text(value, path, errors)
        if not isinstance(value, str) or not value.strip():
            return
        if value == "not_applicable":
            if context != "contract_symbol":
                reject(errors, path, "not_applicable is only used for non-tensor Contract symbol properties")
        elif value.startswith("unknown:") and context == "math":
            if not value[8:].strip():
                reject(errors, path, "unknown: must explain the missing evidence")
        elif value.startswith(("ref:", "bind:")):
            target = value.split(":", 1)[1]
            if context == "io":
                reject(errors, path, "Contract input/output dtype must state its format directly")
            if value.startswith("bind:") and context != "math":
                reject(errors, path, "this property must already be defined upstream")
            if not target.strip() or not target.endswith("." + attribute_name):
                reject(errors, path, f"reference the same .{attribute_name} property")
            elif context == "contract_symbol" and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\." + attribute_name, target):
                reject(errors, path, "Contract property references use a local object name")
        elif attribute_name == "shape":
            reject(errors, path, "expected a shape list or an allowed property reference")
        else:
            known_text(value, path, errors)  # Dtypes are open names, including qualified/vendor spellings.
    return check


def binding(value, path, errors):
    if not isinstance(value, str) or not re.fullmatch(r"(?:ref|bind):\S.*", value):
        reject(errors, path, "expected ref:target or bind:target")


def input_range(value, path, errors):
    def endpoint(v, p, e):
        (text if isinstance(v, str) else number)(v, p, e)
    (text if isinstance(value, str) else pair(endpoint))(value, path, errors)
