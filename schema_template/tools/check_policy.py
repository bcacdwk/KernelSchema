"""Explicit repository-local exemptions; never a schema-version fallback."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

WIKI_ROOT = Path(__file__).resolve().parents[2] / "schema_wiki"


class PolicyError(ValueError):
    """A check policy could not be read or has an unsupported setting."""


@dataclass(frozen=True)
class CheckPolicy:
    allow_missing_hardware: bool = False
    source: Path | None = None

    def notice(self) -> str:
        return ("Hardware exemption: missing specifications and hardware-dependent references "
                f"are deferred by {self.source}; all other checks remain enabled.")


def for_directory(directory: Path) -> CheckPolicy:
    root = WIKI_ROOT.resolve()
    try:
        relative = directory.resolve().relative_to(root)
    except ValueError:
        return CheckPolicy()
    if "archived" in relative.parts:
        return CheckPolicy()
    source = root / "check_policy.json"
    if not source.exists():
        return CheckPolicy()

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate setting {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=unique)
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        raise PolicyError(f"{source}: {error}") from error
    if (not isinstance(value, dict) or set(value) != {"allow_missing_hardware"}
            or type(value["allow_missing_hardware"]) is not bool):
        raise PolicyError(f"{source}: expected only allow_missing_hardware: true or false")
    return CheckPolicy(value["allow_missing_hardware"], source)
