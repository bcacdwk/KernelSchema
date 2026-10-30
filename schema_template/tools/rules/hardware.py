"""The same hardware fields, with an explicit missing-information exemption."""
from __future__ import annotations

import re

from .values import known_text, named, record, text, wiki


def hardware_rule(*, allow_missing=False):
    def field(rule):
        if not allow_missing:
            return rule

        def check(value, path, errors):
            missing = value is None or isinstance(value, dict) and not value
            if isinstance(value, str):
                stripped = value.strip()
                missing |= (not stripped or stripped.startswith("unknown:")
                            or re.fullmatch(r"<[^<>]+@pending:[^<>]+>", stripped) is not None)
            if not missing:
                rule(value, path, errors)
        return check

    def fields(required, optional=None):
        optional = optional or {}
        if allow_missing:
            required, optional = {}, {**required, **optional}
        return field(record(required, optional, empty=allow_missing))

    stack = fields({"language": field(known_text), "toolchain": field(known_text)},
                   {"template": field(known_text)})
    programming = fields({"language": field(wiki)},
                         {name: field(wiki) for name in ("template", "virtual_isa", "machine_isa")})
    references = fields({
        "hardware": field(wiki), "microbench": field(wiki), "profile": field(wiki),
        "sota": fields({"kernel": field(wiki), "schema": field(wiki)}),
        "programming": programming,
    })
    return fields({
        "device": field(known_text), "arch": field(known_text),
        "execution": field(named(field(known_text), key=text, empty=allow_missing)),
        "feature": field(named(field(known_text), key=text, empty=allow_missing)),
        "stack": stack, "wiki": references,
    })
