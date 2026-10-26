"""Readable YAML for concrete JSON values; presentation never changes schema.

Keep small records and shape/range/reference lists in flow style, while large
structures and multiline prose stay in block style. No YAML parsing or schema
inference happens here. The caller verifies a round trip through CUE.
"""
from __future__ import annotations

import json
import re
from typing import Any

import json_value

_PLAIN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_RESERVED = {"null", "true", "false", "yes", "no", "on", "off", "y", "n"}
_SPECIAL_BREAKS = re.compile(r"[\r\x85\u2028\u2029]")
_CONTROLS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _scalar(value: Any) -> str:
    if isinstance(value, str):
        if _PLAIN.fullmatch(value) and value.lower() not in _RESERVED:
            return value
        return json.dumps(value, ensure_ascii=bool(_SPECIAL_BREAKS.search(value)))
    return json_value.dumps(value)


def _flow(value: Any) -> str:
    if isinstance(value, dict):
        return "{" + ", ".join(_scalar(k) + ": " + _flow(v) for k, v in value.items()) + "}"
    if isinstance(value, list):
        items = []
        for item in value:
            # YAML permits [batch: B, token: T] for a list of singleton maps.
            if isinstance(item, dict) and len(item) == 1:
                key, child = next(iter(item.items()))
                items.append(_scalar(key) + ": " + _flow(child))
            else:
                items.append(_flow(item))
        return "[" + ", ".join(items) + "]"
    return _scalar(value)


def _simple(value: Any) -> bool:
    if isinstance(value, dict):
        return len(value) == 1 and all(not isinstance(v, (dict, list)) and _simple(v) for v in value.values())
    if isinstance(value, list):
        return all(_simple(item) for item in value)
    return not isinstance(value, str) or "\n" not in value


def _inline(value: Any, indent: int, width: int) -> str | None:
    if isinstance(value, dict):
        simple = all(not isinstance(v, dict) and _simple(v) for v in value.values())
    else:
        simple = _simple(value)
    if simple:
        result = _flow(value)
        if not isinstance(value, (dict, list)) or not value or len(result) + indent <= width:
            return result
    return None


def dumps(value: Any, *, width: int = 280) -> str:
    """The width is a soft presentation preference, never a validation rule."""
    def render(node: Any, indent: int, *, block: bool = False) -> list[str]:
        prefix = " " * indent
        inline = _inline(node, indent, width)
        if inline is not None and (not block or not isinstance(node, (dict, list))):
            return [prefix + inline]
        if isinstance(node, str):
            if _SPECIAL_BREAKS.search(node) or _CONTROLS.search(node):
                return [prefix + _scalar(node)]
            trailing = len(node) - len(node.rstrip("\n"))
            lines = node.split("\n")
            if trailing:
                lines.pop()
            first = next((line for line in lines if line), "")
            explicit = "2" if first[:1].isspace() else ""
            chomp = "-" if trailing == 0 else ("+" if trailing > 1 or not node.strip("\n") else "")
            indicator = "|" + explicit + chomp
            return [prefix + indicator, *(" " * (indent + 2) + line for line in lines)]
        if isinstance(node, dict):
            lines = []
            for key, child in node.items():
                label = prefix + _scalar(key) + ":"
                inline = _inline(child, len(label) + 1, width)
                if inline is not None:
                    lines.append(label + " " + inline)
                else:
                    nested = render(child, indent + 2, block=True)
                    if isinstance(child, str):
                        lines.append(label + " " + nested[0].strip())
                        # A block scalar's explicit indentation is relative to
                        # its mapping key, not the provisional child indent.
                        lines.extend(line[2:] for line in nested[1:])
                    else:
                        lines.append(label)
                        lines.extend(nested)
            return lines
        if isinstance(node, list):
            lines = []
            for child in node:
                nested = render(child, indent + 2)
                lines.append(prefix + "- " + nested[0][indent + 2:])
                if isinstance(child, str):
                    lines.extend(line[2:] for line in nested[1:])
                else:
                    lines.extend(nested[1:])
            return lines
        raise TypeError(f"not a JSON value: {type(node).__name__}")

    return "\n".join(render(value, 0)) + "\n"
