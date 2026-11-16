"""Small value checks; mathematical expressions and external names stay open."""
from __future__ import annotations

from decimal import Decimal
from functools import partial
import math
from pathlib import Path, PurePosixPath
import re
import subprocess

from ..diagnostics import Diagnostic, field_path
from ..errors import ToolError
from ..pipeline_syntax import LOWER_NAME, ROLE_NAME, _group_end, literal_integer, split_top
from ..text_policy import budget_message, numeric_form_message, single_names_message


def reject(errors, path, message, *, missing=False, hint=""):
    errors.append(Diagnostic(path, message, missing, hint))


def predicate(test, description):
    def check(value, path, errors):
        if not test(value):
            reject(errors, path, "expected " + description)
    return check


def text(value, path, errors):
    if not isinstance(value, str):
        reject(errors, path, "expected a string; if this was intended as text, add the necessary YAML quotes")
    elif not value.strip() or value.strip() == "-":
        reject(errors, path, "expected nonempty text")


def text_budget(value, path, errors, characters, lines=1):
    if isinstance(value, str) and (message := budget_message(value, characters, lines)):
        reject(errors, path, message)


def expression_text(value, path, errors, *, integer=False, allow_empty_set=False):
    known_text(value, path, errors)
    if isinstance(value, str):
        for message in (numeric_form_message(value, integer=integer, allow_empty_set=allow_empty_set),
                        single_names_message(value)):
            if message:
                reject(errors, path, message)


def _name(pattern, description):
    def check(value, path, errors):
        if not isinstance(value, str):
            text(value, path, errors)
        elif not pattern.fullmatch(value):
            reject(errors, path, description)
        else:
            text_budget(value, path, errors, 50)
    check.string_paths = ((),)
    return check


identifier = _name(LOWER_NAME, "expected a lowercase identifier (letters, digits and underscores; no leading digit)")
step_id = _name(ROLE_NAME, "expected an uppercase step or role identifier (no leading digit)")
node_id = _name(LOWER_NAME, "expected a lowercase node ID matching its filename")


def number(value, path, errors):
    if type(value) not in (int, Decimal, float):
        reject(errors, path, "expected a number")
    elif (isinstance(value, Decimal) and not value.is_finite() or
          isinstance(value, float) and not math.isfinite(value)):
        reject(errors, path, "expected a finite number")


def _string_paths(rule):
    while isinstance(rule, partial):
        rule = rule.func
    return getattr(rule, "string_paths", ())


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
            location = field_path(path, key)
            rule = required.get(key, optional.get(key))
            if rule is None:
                reject(errors, location, "field not allowed; check its spelling, capitalization and owning section")
            else:
                rule(item, location, errors)
    # The normalizer consumes this same structural definition. A path names
    # a mapping with fixed fields; named data keys are represented separately.
    children = {**required, **optional}
    check.field_paths = (((), tuple(children)),) + tuple(
        ((key,) + parent, fields)
        for key, rule in children.items()
        for parent, fields in getattr(rule, "field_paths", ())
    )
    check.string_paths = tuple((key,) + leaf
                               for key, rule in children.items() for leaf in _string_paths(rule))
    check.literal_types = tuple(((key,) + leaf, kind)
                                for key, rule in children.items()
                                for leaf, kind in getattr(rule, "literal_types", ()))
    check.value_cases = tuple(((key,) + leaf, choices)
                              for key, rule in children.items()
                              for leaf, choices in getattr(rule, "value_cases", ()))
    return check


def named(rule, *, key=identifier, empty=False):
    def check(value, path, errors):
        if not isinstance(value, dict):
            reject(errors, path, "expected a named mapping")
            return
        if not value and not empty:
            reject(errors, path, "expected at least one named entry")
        for name, item in value.items():
            location = field_path(path, name)
            key(name, location, errors)
            rule(item, location, errors)
    check.field_paths = tuple(((None,) + parent, fields)
                             for parent, fields in getattr(rule, "field_paths", ()))
    check.string_paths = tuple((None,) + leaf for leaf in _string_paths(rule))
    check.literal_types = tuple(((None,) + leaf, kind)
                                for leaf, kind in getattr(rule, "literal_types", ()))
    check.value_cases = tuple(((None,) + leaf, choices)
                              for leaf, choices in getattr(rule, "value_cases", ()))
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
    check.field_paths = tuple(((int,) + parent, fields)
                             for parent, fields in getattr(rule, "field_paths", ()))
    check.string_paths = tuple((int,) + leaf for leaf in _string_paths(rule))
    check.literal_types = tuple(((int,) + leaf, kind)
                                for leaf, kind in getattr(rule, "literal_types", ()))
    check.value_cases = tuple(((int,) + leaf, choices)
                              for leaf, choices in getattr(rule, "value_cases", ()))
    return check


def one_or_many(rule, *, unique=False):
    """Use only where the field permits both scalar and list forms."""
    many = sequence(rule, unique=unique)

    def check(value, path, errors):
        (many if isinstance(value, list) else rule)(value, path, errors)
    check.field_paths = getattr(rule, "field_paths", ()) + many.field_paths
    check.string_paths = _string_paths(rule) + many.string_paths
    check.literal_types = getattr(rule, "literal_types", ()) + many.literal_types
    check.value_cases = getattr(rule, "value_cases", ()) + many.value_cases
    return check


def enum(*choices):
    def check(value, path, errors):
        if not isinstance(value, str):
            text(value, path, errors)
        elif value not in choices:
            reject(errors, path, "expected one of " + ", ".join(repr(c) for c in choices))
    check.string_paths = ((),)
    check.value_cases = (((), choices),)
    return check


def integer(minimum=None, *, expression=True):
    def check(value, path, errors):
        literal = value
        if expression and isinstance(value, str):
            expression_text(value, path, errors, integer=True)
            text_budget(value, path, errors, 100)
            if value.startswith(("unknown:", "ref:", "bind:", "<")):
                reject(errors, path, "supply a defined integer expression, not a missing-value marker")
            literal = literal_integer(value)
            if literal is None:
                return
        if type(literal) is not int:
            reject(errors, path, "expected an integer" + (" or a defined integer expression" if expression else ""))
        elif minimum is not None and literal < minimum:
            reject(errors, path, f"literal must be >= {minimum}")
    # Quote recovery must not guess within integer-or-expression unions.
    check.literal_types = (((), int),) if not expression else ()
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
        expression_text(value, path, errors, allow_empty_set=True)
    elif type(value) is not bool:
        if type(value) in (int, Decimal, float):
            number(value, path, errors)
        else:
            reject(errors, path, "expected a known scalar value or expression; YAML null does not denote Python None")


def literal(value, path, errors):
    if isinstance(value, list):
        sequence(literal, empty=True)(value, path, errors)
    else:
        scalar(value, path, errors)
        text_budget(value, path, errors, 100)


def constant(value, path, errors):
    if isinstance(value, str):
        expression_text(value, path, errors, allow_empty_set=True)
        text_budget(value, path, errors, 100)
    else:
        number(value, path, errors)


_REPOSITORY = re.compile(r"([^@<>\s]+)@([^:<>\s]+):([^<>\r\n]*)")


def is_repository(value):
    return isinstance(value, str) and re.match(r"[^@<>\s]+@[^:<>\s]+:", value) is not None


def repository(value, path, errors, *, entry=False, pending=False, fixed=False):
    if not isinstance(value, str):
        text(value, path, errors)
        return
    match = _REPOSITORY.fullmatch(value)
    if not match:
        reject(errors, path, "expected repository@ref:path" + ("::entry" if entry else ""))
        return
    revision, target = match[2], match[3]
    filename, separator, symbol = target.partition("::")
    if not filename.strip() or any(char in filename for char in "*?\n\r"):
        reject(errors, path, "reference must locate one repository path without wildcards")
    elif filename != filename.strip():
        reject(errors, path, "repository paths must not have leading or trailing whitespace")
    elif (PurePosixPath(filename).is_absolute() or ".." in PurePosixPath(filename).parts
          or re.match(r"(?:[A-Za-z]:[\\/]|~[\\/]|\.\.[\\/])", filename)):
        reject(errors, path, "reference must use a repository-relative path without parent traversal")
    if re.search(r"(?:\s|[;,])[^@<>\s]+@[^:<>\s]+:", target):
        reject(errors, path, "reference must locate one repository target per string")
    if not pending and revision == "pending":
        reject(errors, path, "a bound reference needs a real revision, not pending")
    if fixed and revision.lower() in {"main", "master", "head", "latest", "develop", "development"}:
        reject(errors, path, "reference must use a fixed commit or immutable tag")
    if (entry and not separator) or separator and not symbol.strip():
        reject(errors, path, "reference must identify an entry using ::entry")
    elif separator and symbol != symbol.strip():
        reject(errors, path, "repository entries must not have leading or trailing whitespace")


def wiki(value, path, errors):
    if isinstance(value, str) and value.startswith("<") and value.endswith(">"):
        repository(value[1:-1], path, errors, pending=True)
    else:
        repository(value, path, errors)


def source(value, path, errors):
    known_text(value, path, errors)
    if isinstance(value, str):
        if is_repository(value):
            repository(value, path, errors, fixed=True)
        elif "pending" == value.strip().lower():
            reject(errors, path, "source must identify actual material, not a pending reference")
        elif not re.fullmatch(r"(?:https?|ftp)://\S+", value):
            text_budget(value, path, errors, 300, 2)


def reference(value, path, errors):
    text(value, path, errors)
    if isinstance(value, str) and value.startswith(("unknown:", "<")):
        reject(errors, path, "reference an existing upstream object")


def reference_list(value, path, errors):
    one_or_many(reference, unique=True)(value, path, errors)


def semantic_reference(value, path, errors, *, known=()):
    """Prefer registered full names; otherwise accept only reference-shaped text.

    Unfamiliar argument mathematics stays open. This checks the outer name,
    qualification and indexing, without resolving prose-defined objects.
    """
    if value in known:
        return
    target = value.split(":", 1)[1] if value.startswith(("ref:", "bind:")) else value
    if target in known:
        return
    if is_repository(target):
        repository(target, path, errors, fixed=True)
        return
    qualified = split_top(target, "::")
    if qualified is None or len(qualified) > 2:
        reject(errors, path, "expected one object name, indexed object or qualified reference")
        return
    if len(qualified) == 2:
        target = qualified[1].strip()
        if target in known:
            return
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", qualified[0].strip()):
            reject(errors, path, "reference owner must be a qualified name")
            return
    parts = split_top(target, ".")
    if parts is not None:
        valid = True
        for part in parts:
            match = re.match(r"\s*([^\W\d]\w*)", part)
            if match is None:
                valid = False
                break
            text_budget(match[1], path, errors, 50)
            tail = part[match.end():].strip()
            while tail:
                end = _group_end(tail, 0) if tail[0] in "([" else None
                if end is None:
                    valid = False
                    break
                contents = tail[1:end]
                if contents.strip() != ":" and numeric_form_message(contents):
                    valid = False
                    break
                if message := single_names_message(contents):
                    reject(errors, path, message)
                tail = tail[end + 1:].strip()
            if not valid:
                break
        if valid:
            return
    reject(errors, path, "expected one object name, indexed object or qualified reference; put explanations in a descriptive field")


def identity(value, path, errors, *, characters=100):
    """Names may contain genuine spaces, parentheses, dots or slashes."""
    known_text(value, path, errors)
    if characters is not None:
        text_budget(value, path, errors, characters)
    if isinstance(value, str):
        separated = split_top(value, ";")
        if not any(char.isalnum() for char in value) or separated is not None and len(separated) > 1:
            reject(errors, path, "expected one type or hardware name; put explanations in a descriptive field")


def hardware_identity(value, path, errors):
    """Registered hardware identities are lowercase; external type names are not."""
    identity(value, path, errors)
    if isinstance(value, str) and value != value.lower():
        reject(errors, path, "expected a lowercase registered hardware identity")


def shape(value, path, errors):
    if not isinstance(value, list):
        reject(errors, path, "expected ordered single-axis mappings; use [] for a scalar")
        return
    seen = set()
    for i, axis in enumerate(value):
        location = f"{path}[{i}]"
        if not isinstance(axis, dict) or len(axis) != 1:
            reject(errors, location, "each shape axis must be one name: size mapping; quote expressions containing YAML delimiters")
        else:
            named(integer(0))(axis, location, errors)
            name = next(iter(axis))
            if name in seen:
                reject(errors, location, "axis names must be unique within one shape")
            seen.add(name)


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
            if context in {"io", "storage"}:
                reject(errors, path, "this dtype must state its defined format directly")
            if value.startswith("bind:") and context != "math":
                reject(errors, path, "this property must already be defined upstream")
            if not target.strip() or not target.endswith("." + attribute_name):
                reject(errors, path, f"reference the same .{attribute_name} property")
            elif context == "contract_symbol" and not re.fullmatch(r"[a-z_][a-z0-9_]*\." + attribute_name, target):
                reject(errors, path, "Contract property references use a local lowercase object name")
        elif attribute_name == "shape":
            reject(errors, path, "expected a shape list or an allowed property reference")
        else:
            identity(value, path, errors)
            if value in {"None", "none"} or re.search(r"(?:\bor\b|\|)\s*(?:None|python_none)\b", value):
                reject(errors, path, "dtype names the present object's type; use nullable for an additional None branch")
    check.string_paths = ((),) if attribute_name == "dtype" else ()
    return check


def binding(value, path, errors):
    if not isinstance(value, str):
        text(value, path, errors)
    elif not re.fullmatch(r"(?:ref|bind):\S.*", value, re.DOTALL):
        reject(errors, path, "expected ref:target or bind:target")
    elif is_repository(value.split(":", 1)[1]):
        repository(value.split(":", 1)[1], path, errors, fixed=True)
    else:
        text_budget(value, path, errors, 300, 2)


def input_range(value, path, errors):
    def endpoint(item, location, diagnostics):
        (expression_text if isinstance(item, str) else number)(item, location, diagnostics)
    (expression_text if isinstance(value, str) else pair(endpoint))(value, path, errors)


def _repository_fields(plan, path, *, skip_hardware):
    """Known locator slots only; never interpret source prose or formula bodies."""
    yield plan.get("META_DATA", {}).get("Source"), path + ".META_DATA.Source"
    contract = plan.get("CONTRACT", {})
    algorithm = contract.get("Algorithm_requirement", {})
    yield algorithm.get("Golden_reference", {}).get("Source"), path + ".CONTRACT.Algorithm_requirement.Golden_reference.Source"

    def leaves(value, location):
        if isinstance(value, dict):
            for key, child in value.items():
                yield from leaves(child, field_path(location, key))
        else:
            yield value, location

    if not skip_hardware:
        yield from leaves(contract.get("Hardware_specification", {}).get("Wiki", {}),
                          path + ".CONTRACT.Hardware_specification.Wiki")
    for index, rewrite in enumerate(plan.get("DESIGN", {}).get("Math_reshape", {}).get("Rewrite", [])):
        for name, symbol in rewrite.get("Symbol", {}).items():
            location = field_path(f"{path}.DESIGN.Math_reshape.Rewrite[{index}].Symbol", name)
            for field in ("binding", "shape", "dtype"):
                value = symbol.get(field)
                if isinstance(value, str) and value.startswith(("ref:", "bind:")):
                    target = value.split(":", 1)[1]
                    if field != "binding":
                        target = target.removesuffix("." + field)
                    yield target, location + "." + field
    for index, feedback in enumerate(plan.get("REALIZATION", {}).get("Feedback", [])):
        yield feedback.get("Evidence"), f"{path}.REALIZATION.Feedback[{index}].Evidence"


def local_repository_errors(files, documents, *, skip_hardware=False):
    """Check available commit trees at file entry points, without network access.

    Only the Git repository containing an explicit input file is considered.
    Its directory name or exact origin identifies it; unknown repositories and
    unavailable revisions remain for source review. Entry semantics cannot be
    proved by a generic source-symbol parser and are deliberately not guessed.
    """
    def git(root, *arguments, missing=False):
        try:
            result = subprocess.run(["git", "--literal-pathspecs", "-C", str(root), *arguments],
                                    capture_output=True, text=True, encoding="utf-8", timeout=10)
        except (OSError, UnicodeError, subprocess.TimeoutExpired) as error:
            raise ToolError(f"{root}: cannot inspect the local repository: {error}") from error
        if result.returncode and not missing:
            raise ToolError(f"{root}: cannot inspect the local repository: {result.stderr.strip()[:200]}")
        return result

    def identity(value):
        value = re.sub(r"^(?:https?|ssh)://(?:git@)?", "", value).rstrip("/").removesuffix(".git")
        return value.removeprefix("git@").replace(":", "/", 1) if value.startswith("git@") else value

    errors, roots, commits, checked = [], {}, {}, set()
    for node, filename in files.items():
        locators = []
        for value, location in _repository_fields(documents[node].value, node, skip_hardware=skip_hardware):
            if isinstance(value, str) and (match := _REPOSITORY.fullmatch(value)):
                locators.append((match, location))
        if not locators:
            continue
        try:
            parent = Path(filename).resolve().parent
            root = next((candidate for candidate in (parent, *parent.parents) if (candidate / ".git").exists()), None)
        except OSError as error:
            raise ToolError(f"{filename}: cannot inspect the local repository context: {error}") from error
        if root is None:
            continue
        if root not in roots:
            origin = git(root, "config", "--get", "remote.origin.url", missing=True)
            if origin.returncode not in (0, 1):
                raise ToolError(f"{root}: cannot read the local repository origin")
            roots[root] = {root.name, identity(origin.stdout.strip())} - {""}
        for match, location in locators:
            if identity(match[1]) not in roots[root]:
                continue
            key = (root, match[2])
            if key not in commits:
                revision = git(root, "rev-parse", "--verify", "--end-of-options", match[2] + "^{commit}", missing=True)
                if revision.returncode and not any(message in revision.stderr for message in (
                        "Needed a single revision", "unknown revision", "bad revision", "not a valid object name")):
                    raise ToolError(f"{root}: cannot resolve the local revision: {revision.stderr.strip()[:200]}")
                commits[key] = revision.stdout.strip() if revision.returncode == 0 else None
            if commits[key] is None:
                continue
            target = match[3].partition("::")[0]
            if not target or target != target.strip() or PurePosixPath(target).is_absolute() or ".." in PurePosixPath(target).parts:
                continue  # The pure field validator owns malformed paths.
            target = str(PurePosixPath(target))
            if target == ".":
                continue
            query = (root, commits[key], target, location)
            if query in checked:
                continue
            checked.add(query)
            tree = git(root, "ls-tree", "--full-tree", "-z", commits[key], "--", target)
            entries = [record.partition("\t")[2] for record in tree.stdout.split("\0") if record]
            if target not in entries:
                reject(errors, location, "repository path is absent from the locally available referenced commit")
    return errors


# Scalar positions whose non-null value is necessarily text are exposed for
# source-type recovery. Callers preserve null; numeric/shape unions stay unmarked.
for _text_rule in (text, known_text, expression_text, repository, wiki, source,
                   reference, identity, hardware_identity, binding):
    _text_rule.string_paths = ((),)
reference_list.string_paths = one_or_many(reference).string_paths
