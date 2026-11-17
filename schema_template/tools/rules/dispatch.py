"""Small, read-only checks for common's task index; prose remains review work."""
from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from ..diagnostics import Diagnostic

_HEADING = re.compile(r"^ {0,3}(#{1,6})\s+(.+?)(?:\s+#+)?\s*$")
_DEFINITION = re.compile(r'^ {0,3}\[([^\]]+)\]:\s*(<[^>]+>|\S+)(?:\s+.*)?$')
_INLINE = re.compile(r'''(?<!!)\[([^\]\n]+)\]\(\s*(<[^>\n]+>|(?:\\.|[^\s)])+)(?:\s+(?:"[^"]*"|'[^']*'|\([^)]*\)))?\s*\)''')
_REFERENCE = re.compile(r"(?<!!)\[([^\]\n]+)\](?:\[([^\]\n]*)\])?(?![\[(])")
_DESIGN_COLUMNS = ["设计", "具体机制与保留理由", "关系"]


def _visible(text):
    fence = None
    for number, line in enumerate(text.splitlines(), 1):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
            continue
        if marker:
            fence = marker[1]
            continue
        yield number, line


def _label(value):
    return " ".join(value.split()).casefold()


def _links(lines):
    definitions = {}
    for _, line in lines:
        match = _DEFINITION.fullmatch(line)
        if match:
            definitions[_label(match[1])] = match[2].strip("<>")
    for number, line in lines:
        if _DEFINITION.fullmatch(line):
            continue
        # A link may use a code-styled label; only a link wholly inside a
        # code sample is ignored. Keep offsets while masking inline links.
        code = [match.span() for match in re.finditer(r"(`+).*?\1", line)]
        for match in _INLINE.finditer(line):
            if not any(start <= match.start() < end for start, end in code):
                yield number, match[2].strip("<>")
        line = _INLINE.sub(lambda match: " " * len(match[0]), line)
        for match in _REFERENCE.finditer(line):
            if any(start <= match.start() < end for start, end in code):
                continue
            target = definitions.get(_label(match[2] or match[1]))
            if target is not None:
                yield number, target


def _cells(line):
    if "|" not in line:
        return []
    cells = re.split(r"(?<!\\)\|", line.strip())
    if not cells[0]:
        cells.pop(0)
    if cells and not cells[-1]:
        cells.pop()
    return [cell.strip() for cell in cells]


def _design_rows(lines):
    """Read the minimum table and its design cells, without interpreting prose."""
    found, rows, in_table = False, [], False
    for index, (number, line) in enumerate(lines):
        cells = _cells(line)
        if (index and cells and len(cells) == len(_DESIGN_COLUMNS)
                and all(re.fullmatch(r":?-+:?", cell) for cell in cells)
                and lines[index - 1][0] == number - 1
                and _cells(lines[index - 1][1]) == _DESIGN_COLUMNS):
            found = in_table = True
        elif in_table:
            if cells and not _HEADING.fullmatch(line) and lines[index - 1][0] == number - 1:
                rows.append((number, cells[0]))
            else:
                in_table = False
    return found, rows


def validate(text: str, files: dict[str, Path]) -> list[Diagnostic]:
    """Check required sections and direct links to the current complete designs.

    No files are read here. In particular, an index never supplies a patch or
    missing YAML content. Existing-object meaning and design value need review.
    """
    errors = []

    def fail(line, message):
        errors.append(Diagnostic(f"dispatch.md:{line}", message))

    if len(files) > 10:
        fail(1, "at most 10 complete YAML designs, including root, are allowed")
    lines = list(_visible(text))
    headings = [(number, len(match[1]), match[2]) for number, line in lines
                if (match := _HEADING.fullmatch(line))]
    sections = {}
    for index, (number, level, title) in enumerate(headings):
        if level != 2:
            continue
        end = next((later for later, depth, _ in headings[index + 1:] if depth <= 2), len(text.splitlines()) + 1)
        if title in {"范围", "完整设计", "变体与组合"}:
            if title in sections:
                fail(number, f"duplicate section {title}")
            sections[title] = (number, end)
    for title in ("范围", "完整设计"):
        if title not in sections:
            fail(1, f"required section: ## {title}")
        else:
            start, end = sections[title]
            if not any(line.strip() and not _DEFINITION.fullmatch(line)
                       for number, line in lines if start < number < end):
                fail(start, f"section {title} needs content")

    current = {path.name: node for node, path in files.items()}
    registrations = set()
    if "完整设计" in sections:
        start, end = sections["完整设计"]
        found, rows = _design_rows([(number, line) for number, line in lines if start < number < end])
        if not found:
            fail(start, "complete-design section needs the table: 设计 | 具体机制与保留理由 | 关系")
        definitions = [(number, line) for number, line in lines if _DEFINITION.fullmatch(line)]
        registrations = set(_links(definitions + rows))
    links = []
    registered = set()
    for number, target in _links(lines):
        try:
            parsed = urlsplit(target)
        except ValueError:
            continue  # Non-file URLs are outside this small checker.
        if parsed.scheme or parsed.netloc:
            continue
        path = unquote(parsed.path)
        if Path(path).suffix not in {".yaml", ".yml"}:
            continue
        normalized = str(Path(path))
        if normalized not in current:
            fail(number, f"YAML link does not name a current task file: {target}")
        else:
            links.append((number, current[normalized]))
            if (number, target) in registrations:
                registered.add(current[normalized])

    if "完整设计" in sections:
        start, end = sections["完整设计"]
        for node in sorted(files.keys() - registered):
            fail(start, f"complete-design table is missing a link to {files[node].name} in the design column")
    if "变体与组合" in sections:
        start, end = sections["变体与组合"]
        entries = []
        for number, level, title in headings:
            if level == 3 and start < number < end:
                entries.append(number)
                if not re.fullmatch(r"\S(?:.*\S)?\s*（(?:参数|局部|组合)）", title):
                    fail(number, "variant heading must name its type: 参数, 局部 or 组合")
        if not entries and any(line.strip() for number, line in lines if start < number < end):
            fail(start, "variant section needs entries headed ### <name>（参数|局部|组合）")
        for begin, stop in zip(entries, entries[1:] + [end]):
            if not any(begin < number < stop for number, _ in links):
                fail(begin, "variant or combination needs a direct baseline link to a current complete YAML")
    return errors
