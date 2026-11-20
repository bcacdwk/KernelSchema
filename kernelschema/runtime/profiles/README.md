# Public Claude CLI profile and explanatory summary

This small directory is suitable for source submission. It contains only the
necessary nonsecret CLI configuration, four observed call outcomes/model/usage
facts, and original hashes. It does **not** contain authentication status,
credentials, private run locators, raw streams, or a private runtime directory.
It is an explanation with provenance hashes, **not a complete importable
original-evidence bundle**. Hashes do not stand in for the missing originals.

[The profile](claude-code-2.1.316-qwen3.8-max-no-tools.v1.json) preserves the exact
observed final CLI configuration, with SHA-256
`76fd98e0094f4a3e07b849c8f7a9207752689dd236105bd82919d904b405603e`.
It pins Claude Code CLI 2.1.316's executable bytes, explicitly requests
`qwen3.8-max[1m]`, uses effort xhigh, and allows only the two observed builtin
plugins. Model tools, MCP and skills are disabled. It uses the official CLI's
existing authentication (`existing_cli_auth`); the public file contains no
authentication value or authentication-status snapshot.

[The summary](qualification-summary.v1.json) retains the initial rejected
planner outcome and the final-profile coder/reviewer/planner successes. Four
calls reported 15,291 tokens and USD 0.232672750000000010 in total. USD is the
CLI's reported estimate, not independently verified billing. The native USD
0.50 threshold is not a host-guaranteed hard ceiling; token hard enforcement is
unsupported. These observations qualify CLI transport, permissions, model
identity, terminal collection and original-operation readback. They do not
qualify a kernel, evaluator, fidelity conclusion or real writing behavior.

K4/K5 can load the profile from a source checkout/editable install, independently
of cwd and without any private capture:

```python
import json
from importlib.resources import files
from kernelschema.runtime import ClaudeProfile, ClaudeRuntime

data = json.loads(files("kernelschema.runtime").joinpath(
    "profiles/claude-code-2.1.316-qwen3.8-max-no-tools.v1.json"
).read_text(encoding="utf-8"))
data["reported_models"] = tuple(data["reported_models"])
data["builtin_plugins"] = tuple(data["builtin_plugins"])
profile = ClaudeProfile(**data)
runtime = ClaudeRuntime(profile)
# Freeze profile.backend_profile(environment_ref) into the containing request.
```

Loading this file does not query authentication or invoke a model. The pinned
executable must be available and unchanged before a new call. A different CLI
installation/model/configuration requires its own explicit compatible profile;
loading this historical profile does not qualify arbitrary installations.
The K4/K5 owner still owns admission, pause/fences, the only budget and business
validation/publication, as described in [the runtime interface](../INTERFACE.md).

The summary separates two later CPU-only changes: request-aware writing/review
authority and `workflow_objective` grants; and preserving caller HOME in the
unqualified `environment_auth` mode while isolating only `CLAUDE_CONFIG_DIR`.
The `existing_cli_auth` branch and observed CLI transport/terminal mechanisms
remain unchanged. No model calls were made for these increments. Actual writing
semantics still need K5.real. The full historical captures retain their original
source identities and bytes in local private storage; they must not be submitted
recursively as runtime source or treated as required package resources.
