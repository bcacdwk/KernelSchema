"""Integer substitution of the shared Action record; no second grammar."""
from __future__ import annotations

from ..pipeline_syntax import Action, parse_action
from .expressions import Domains, tree, variables


def parsed(action: Action | str) -> Action:
    return parse_action(action) if isinstance(action, str) else action


def operands(action: Action | str):
    value = parsed(action)
    return value.verb, [(operand.name, operand.args) for operand in value.operands]


def arguments(action: Action | str):
    return [((oi, ai), argument) for oi, operand in enumerate(parsed(action).operands)
            for ai, argument in enumerate(operand.args)]


def instantiate(action: Action | str, domains: Domains, env: dict) -> str:
    action = parsed(action)
    indices = iter(domains.integers([argument for _, argument in arguments(action)], env))
    return action.verb + ":" + "@".join(
        operand.name + ("(" + ",".join(str(next(indices)) for _ in operand.args) + ")" if operand.args else "")
        for operand in action.operands)


def action_variables(action: Action | str, domains: Domains) -> set[str]:
    return set().union(*(variables(tree(argument)) & domains.definitions.keys()
                         for _, argument in arguments(action)))
