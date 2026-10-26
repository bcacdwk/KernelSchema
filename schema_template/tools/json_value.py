"""Exact JSON values between CUE evaluations, without binary64 rounding.

The standard decoder supports Decimal; this small encoder preserves it as a
JSON number. It accepts only JSON value types and never uses eval or repr.
"""
from __future__ import annotations

from decimal import Decimal
import json
from typing import Any


def loads(text: str) -> Any:
    return json.loads(text, parse_float=Decimal)


def dumps(value: Any, *, sort_keys: bool = False) -> str:
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("JSON numbers must be finite")
        return str(value)
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        keys = sorted(value) if sort_keys else value
        return "{" + ",".join(json.dumps(key, ensure_ascii=False) + ":" +
                              dumps(value[key], sort_keys=sort_keys) for key in keys) + "}"
    if isinstance(value, list):
        return "[" + ",".join(dumps(item, sort_keys=sort_keys) for item in value) + "]"
    if value is None or type(value) in (str, int, float, bool):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    raise TypeError(f"not a JSON value: {type(value).__name__}")
