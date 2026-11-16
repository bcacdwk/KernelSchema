"""Hardware field structure; only an explicit caller request skips this region."""
from __future__ import annotations

from functools import partial

from .values import hardware_identity, identity, known_text, named, record, wiki


def hardware_rule():
    # Device/Architecture retain their existing raw-field 50/1 budgets.
    device_name = partial(identity, characters=None)
    stack = record({"Language": known_text, "Toolchain": known_text}, {"Template": known_text})
    programming = record({"Language": wiki},
                         {name: wiki for name in ("Template", "Virtual_ISA", "Machine_ISA")})
    references = record({
        "Hardware": wiki, "Microbench": wiki, "Profile": wiki,
        "Sota": record({"Kernel": wiki, "Schema": wiki}), "Programming": programming,
    })
    return record({
        "Device": device_name, "Architecture": device_name,
        "Execution": named(known_text, key=hardware_identity), "Feature": named(known_text, key=hardware_identity),
        "Stack": stack, "Wiki": references,
    })
