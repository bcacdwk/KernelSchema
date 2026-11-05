"""Bind explicit acceptance declarations to each actual Contract criterion."""

from __future__ import annotations

from bisect import bisect_right
from decimal import Decimal
import json
from pathlib import Path
import re

from .artifacts import load_root
from .common import ExtractError, regular_file, rules_root, source_root


RELATIVE_TO = re.compile(r'"relative_to"\s*:\s*("(?:[^"\\]|\\.)*")')
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
FORMULA = re.compile(
    r"abs\(\s*(candidate|reference)\s*-\s*(candidate|reference)\s*\)\s*<=\s*"
    rf"(?:({NUMBER}|atol)\s*\+\s*)?({NUMBER}|rtol)\s*\*\s*abs\(\s*(candidate|reference)\s*\)")


def _rules(value, name=None):
    """JSON member order matches the source order of tolerance objects."""
    if isinstance(value, dict):
        name = value.get("name", name)
        for key, child in value.items():
            if key == "tolerance" and isinstance(child, dict):
                yield name, child
            if key == "outputs" and isinstance(child, dict):
                for output, entry in child.items():
                    yield from _rules(entry, output)
            else:
                yield from _rules(child, name)
    elif isinstance(value, list):
        for child in value:
            yield from _rules(child, name)


def declarations(lock: dict) -> list[dict]:
    """Keep output identities exact; group only the known input.state family."""
    result = {}
    for binding in lock["source"]["bindings"]["acceptance"]:
        path = binding["path"]
        text = regular_file(source_root(lock), path).read_text(encoding="utf-8")
        if not RELATIVE_TO.search(text):
            continue
        decoder = json.JSONDecoder(parse_float=Decimal)
        try:
            document = json.loads(text, parse_float=Decimal)
        except ValueError as exc:
            raise ExtractError(f"explicit relative_to declarations require readable acceptance JSON: {path}") from exc
        fields = iter(re.finditer(r'"tolerance"\s*:\s*(?=\{)', text))
        newlines = [match.start() for match in re.finditer("\n", text)]
        for name, rule in _rules(document):
            field = next(fields, None)
            if field is None:
                raise ExtractError(f"cannot locate acceptance tolerance declaration: {path}")
            located, end = decoder.raw_decode(text, field.end())
            if located != rule:
                raise ExtractError(f"ambiguous acceptance tolerance declaration: {path}")
            operand = rule.get("relative_to")
            if not isinstance(operand, str):
                continue
            if not isinstance(name, str) or not name.strip():
                raise ExtractError(f"explicit relative_to needs a named acceptance output: {path}")
            match = next((item for item in RELATIVE_TO.finditer(text, field.end(), end)
                          if json.loads(item[1]) == operand), None)
            if match is None:
                raise ExtractError(f"cannot locate relative_to for acceptance output {name}: {path}")
            # Tuple return positions can have different Contract outputs even
            # with identical thresholds. Only this known state family groups.
            source_output = "input.state[*]" if re.fullmatch(r"input\.state\[\d+\]", name) else name
            thresholds = {key: str(rule[key]) for key in ("atol", "rtol")
                          if type(rule.get(key)) in (int, Decimal)}
            key = (path, source_output, operand, tuple(thresholds.items()))
            first = bisect_right(newlines, match.start()) + 1
            location = {"line_start": first, "line_end": first + match[0].count("\n"), "quote": match[0]}
            result.setdefault(key, {"path": path, "source_output": source_output, "relative_to": operand,
                                    **thresholds, "locations": []})["locations"].append(location)
    return list(result.values())


def _criterion(text: str, declaration: dict, output: str) -> None:
    location = "contract.algorithm_requirement.golden_reference.tolerance." + output
    operands = re.findall(r"\brelative_to\s*[:=]\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)", text)
    if not operands or set(operands) != {declaration["relative_to"]}:
        raise ExtractError(f"{location} must explicitly preserve relative_to={declaration['relative_to']}")
    # These conventional forms have mechanical operands. Other mathematical
    # prose remains subject to the independent review, not an NLP proof here.
    for match in FORMULA.finditer(text):
        left, right, atol, rtol, basis = match.groups()
        if left == right or basis != declaration["relative_to"]:
            raise ExtractError(f"{location} formula contradicts its declared comparison operands")
        for key, token in (("atol", atol or "0"), ("rtol", rtol)):
            if token == key:
                assigned = re.search(rf"\b{key}\s*=\s*({NUMBER})", text)
                token = assigned[1] if assigned else None
            if key in declaration and token is not None and Decimal(token) != Decimal(declaration[key]):
                raise ExtractError(f"{location} formula changes the declared {key}={declaration[key]}")
    for key in ("atol", "rtol"):
        for match in re.finditer(rf"\b{key}\s*=\s*({NUMBER})", text):
            if key in declaration and Decimal(match[1]) != Decimal(declaration[key]):
                raise ExtractError(f"{location} changes the declared {key}={declaration[key]}")


def validate_comparisons(receipt: dict, lock: dict, artifacts: Path, *, parser_root: Path | None = None) -> None:
    expected = declarations(lock)
    root = load_root(artifacts, parser_root=parser_root or rules_root(lock))
    algorithm = root["contract"]["algorithm_requirement"]
    criteria = algorithm["golden_reference"]["tolerance"]
    covered, seen = set(), set()
    for item in receipt["comparisons"]:
        output, source_output = item["output"], item["source_output"]
        if output not in criteria or (output, source_output) in seen:
            raise ExtractError("comparison bindings need a declared output and unique output/source_output pair")
        seen.add((output, source_output))
        if source_output in criteria and output != source_output:
            raise ExtractError("an identically named acceptance output must bind its own Contract criterion")
        indices = item["evidence"]
        if not indices or any(index < 0 or index >= len(receipt["source_evidence"]) for index in indices):
            raise ExtractError("comparison binding requires valid source evidence indices")
        anchors = [receipt["source_evidence"][index] for index in indices]
        matched = False
        for index, rule in enumerate(expected):
            if rule["source_output"] != source_output:
                continue
            if not any(anchor["category"] == "acceptance" and anchor["path"] == rule["path"] and
                       any(anchor["line_start"] <= loc["line_start"] <= loc["line_end"] <= anchor["line_end"]
                           and loc["quote"] in anchor["quote"] for loc in rule["locations"])
                       for anchor in anchors):
                continue
            _criterion(criteria[output], rule, output)
            covered.add(index)
            matched = True
        if not matched:
            raise ExtractError(f"comparison {source_output} -> {output} must cite its relative_to declaration line")
    for index, rule in enumerate(expected):
        if index not in covered:
            first = rule["locations"][0]
            raise ExtractError(f"missing output comparison binding and declaration anchor: {rule['source_output']} "
                               f"at {rule['path']}:{first['line_start']}")
