#!/usr/bin/env python3
"""Synthetic CLI only. Executed as a real subprocess by runtime tests."""
import json
from pathlib import Path
import sys
import time

if "--version" in sys.argv:
    print("synthetic-claude-1")
    raise SystemExit
if "--help" in sys.argv:
    print("--tools --safe-mode --strict-mcp-config --setting-sources --session-id --max-budget-usd --permission-prompts --no-session-persistence")
    raise SystemExit

def arg(name):
    return sys.argv[sys.argv.index(name) + 1]

root = Path(__file__).parent
scenario = json.loads((root / "scenario.json").read_text())
with (root / "invocations.jsonl").open("a") as f:
    f.write(json.dumps({"argv": sys.argv[1:], "cwd": str(Path.cwd())}) + "\n")
assert arg("--tools") == "" and arg("--permission-mode") == "dontAsk"
assert "--safe-mode" in sys.argv and "--strict-mcp-config" in sys.argv
assert arg("--mcp-config") == '{"mcpServers":{}}'
assert arg("--setting-sources") == "" and "--bare" in sys.argv
data = json.load(sys.stdin)
mode = scenario.get("mode", "success")
session = arg("--session-id")
model = "wrong-model" if mode == "wrong_model" else arg("--model")
initial = {"type": "system", "subtype": "init", "session_id": session, "model": model,
           "tools": ["Read"] if mode == "permissions" else [], "mcp_servers": [], "skills": [], "plugins": [], "permissionMode":"dontAsk", "cwd":str(Path.cwd())}
if mode == "user_plugin": initial["plugins"] = [{"name":"cc-plugin-agents-md", "path":"/dummy-user-plugin"}]
if mode == "unexpected_builtin": initial["plugins"] = [{"name":"unexpected", "path":"builtin"}]
if mode == "wrong_cwd": initial["cwd"] = "/dummy-wrong-cwd"
if mode == "wrong_permission_mode": initial["permissionMode"] = "bypassPermissions"
print(json.dumps(initial), flush=True)
if mode == "sleep": time.sleep(scenario.get("seconds", 1))
if mode == "missing_terminal": raise SystemExit
if mode == "truncated":
    print('{"type":"result",', end="", flush=True)
    raise SystemExit
if mode == "tool_use":
    print(json.dumps({"type":"assistant", "message":{"content":[{"type":"tool_use", "name":"Read", "input":{"path":"dummy-canary"}}]}}), flush=True)
role = data["role"]
content = {"synthetic": True, "allowed_ids": [x["id"] for x in data["materials"]]}
if role == "coder":
    source = next((x["text"] for x in data["materials"] if x["id"] == "current_source"), "# empty start\n")
    content = source + "# synthetic CLI continuation\n"
output = {"protocol": "ks.claude.output.v1", "role": role, "assignment_sha256": data["assignment_sha256"],
          "status": "ok", "kind": {"coder":"candidate_source", "planner":"writing_draft", "reviewer":"review"}[role],
          "content": content, "reason": "synthetic CPU fixture"}
if mode == "forged_output": output["reported_model"] = "trusted-model"
terminal = {"type":"result", "subtype":"success", "is_error":False, "session_id":session,
            "result":json.dumps(output), "num_turns":1, "total_cost_usd":0.001,
            "usage":{"input_tokens":11,"output_tokens":7,"cache_creation_input_tokens":3,"cache_read_input_tokens":5},
            "modelUsage":{model:{"inputTokens":11,"outputTokens":7}}, "permission_denials":[]}
if mode == "missing_usage": del terminal["usage"]
if mode == "missing_cost": del terminal["total_cost_usd"]
if mode == "error_terminal": terminal.update(subtype="error_during_execution", is_error=True)
if mode == "wrong_session": terminal["session_id"] = "other"
line = json.dumps(terminal)
print(line, flush=True)
if mode == "replay": print(line, flush=True)
if mode == "conflict":
    terminal["total_cost_usd"] = 0.002
    print(json.dumps(terminal), flush=True)
if mode == "post_terminal_sleep": time.sleep(scenario.get("seconds", 1))
print("synthetic stderr original", file=sys.stderr, flush=True)
raise SystemExit(7 if mode == "nonzero" else 0)
