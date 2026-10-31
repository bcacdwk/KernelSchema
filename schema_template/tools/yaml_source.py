"""Read YAML data and real source marks without executing tags or writing files."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
from pathlib import Path
import re
import shlex
import sys

from .diagnostics import field_path
from .errors import CapacityError

MAX_EXPANDED_NODES = 100000


class ReadError(Exception):
    pass


class DependencyError(Exception):
    pass


@dataclass
class Document:
    value: dict
    positions: dict[str, tuple[int, int]]
    values: dict[str, object]
    source_sha256: str


def load(path: Path, name: str) -> Document:
    try:
        import yaml
    except ModuleNotFoundError as error:
        command = f"{shlex.quote(sys.executable)} -m pip install -r {shlex.quote(str(Path(__file__).with_name('requirements.txt')))}"
        raise DependencyError(f"PyYAML is missing. Install it in this Python environment:\n{command}\nThen rerun the same check command.") from error

    class Loader(yaml.SafeLoader):
        pass

    # YAML core scalars: do not turn names such as on/off or dates into booleans
    # or datetime objects. Decimal is constructed from source text, never float.
    removed = {"tag:yaml.org,2002:" + t for t in ("bool", "timestamp", "int", "float")}
    Loader.yaml_implicit_resolvers = {k: [(tag, regex) for tag, regex in entries if tag not in removed]
                                      for k, entries in Loader.yaml_implicit_resolvers.items()}
    Loader.add_implicit_resolver("tag:yaml.org,2002:bool", re.compile(r"^(?:true|false)$", re.I), list("tTfF"))
    Loader.add_implicit_resolver("tag:yaml.org,2002:int", re.compile(r"^[-+]?(?:0[bB][01_]+|0[oO][0-7_]+|0[xX][0-9a-fA-F_]+|[0-9][0-9_]*)$"), list("-+0123456789"))
    Loader.add_implicit_resolver("tag:yaml.org,2002:float", re.compile(r"^[-+]?(?:(?:[0-9][0-9_]*\.[0-9_]*|\.[0-9_]+)(?:[eE][-+]?[0-9]+)?|[0-9][0-9_]*[eE][-+]?[0-9]+|\.(?:inf|nan))$", re.I), list("-+0123456789."))

    def problem(message, mark=None):
        position = f":{mark.line + 1}:{mark.column + 1}" if mark else ""
        return ReadError(f"{path}{position}: {message}")

    def scalar(node):
        tag = node.tag.removeprefix("tag:yaml.org,2002:")
        value = node.value
        if tag == "str":
            return value
        if tag == "null":
            return None
        if tag == "bool" and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        value = value.replace("_", "")
        try:
            if tag == "int":
                unsigned = value.lstrip("+-").lower()
                return int(value, 0 if unsigned.startswith(("0x", "0o", "0b")) else 10)
            if tag == "float":
                number = Decimal(value)
                if number.is_finite():
                    return number
        except (ValueError, InvalidOperation):
            pass
        raise problem(f"unsupported scalar or non-finite number ({node.tag}); use ordinary YAML values", node.start_mark)

    positions, values, active = {}, {}, set()
    expanded = 0

    def convert(node, field, mark=None):
        nonlocal expanded
        if id(node) in active:
            raise problem("recursive aliases do not describe a finite plan", node.start_mark)
        expanded += 1
        if expanded > MAX_EXPANDED_NODES:
            raise CapacityError(f"{path}: YAML expansion exceeds the parser's capacity ({MAX_EXPANDED_NODES} nodes)")
        active.add(id(node))
        try:
            if isinstance(node, yaml.ScalarNode):
                result = scalar(node)
            elif isinstance(node, yaml.SequenceNode) and node.tag == "tag:yaml.org,2002:seq":
                result = [convert(item, f"{field}[{i}]") for i, item in enumerate(node.value)]
            elif isinstance(node, yaml.MappingNode) and node.tag == "tag:yaml.org,2002:map":
                result, explicit, merges, seen = {}, [], [], set()
                for key, value in node.value:
                    if key.tag == "tag:yaml.org,2002:merge":
                        entries = value.value if isinstance(value, yaml.SequenceNode) else [value]
                        if not all(isinstance(item, yaml.MappingNode) for item in entries):
                            raise problem("a YAML merge needs mappings", value.start_mark)
                        merges.extend(entries)
                        continue
                    if not isinstance(key, yaml.ScalarNode) or key.tag != "tag:yaml.org,2002:str":
                        raise problem("mapping keys must be strings", key.start_mark)
                    if key.value in seen:
                        raise problem(f"duplicate mapping key {key.value!r}; keep one definition", key.start_mark)
                    seen.add(key.value)
                    explicit.append((key, value))
                # YAML merges are local value reuse. Explicit fields override
                # merged defaults; explicit duplicate keys still fail above.
                for merged in reversed(merges):
                    result.update(convert(merged, field))
                for key, value in explicit:
                    location = key.start_mark if value.start_mark.index < key.start_mark.index else None
                    result[key.value] = convert(value, field_path(field, key.value), location)
            else:
                raise problem("use ordinary mappings, sequences and scalar values; custom tags are not supported", node.start_mark)
            where = mark or node.start_mark
            positions[field] = (where.line + 1, where.column + 1)
            values[field] = result
            return result
        finally:
            active.remove(id(node))

    loader = None
    try:
        source_bytes = path.read_bytes()
        loader = Loader(source_bytes.decode("utf-8"))
        node = loader.get_node() if loader.check_node() else None
        if node is None or not isinstance(node, yaml.MappingNode) or loader.check_node():
            raise problem("expected exactly one YAML mapping document per node", node.start_mark if node else None)
        value = convert(node, name)
        return Document(value, positions, values, hashlib.sha256(source_bytes).hexdigest())
    except yaml.YAMLError as error:
        raise problem(getattr(error, "problem", None) or str(error), getattr(error, "problem_mark", None)) from error
    except UnicodeDecodeError as error:
        raise problem("YAML must be valid UTF-8") from error
    except RecursionError as error:
        raise CapacityError(f"{path}: YAML nesting exceeds the parser's capacity") from error
    finally:
        if loader is not None:
            loader.dispose()
