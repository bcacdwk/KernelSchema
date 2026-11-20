#!/usr/bin/env python3
"""Real subprocess, synthetic outputs only; production never imports this file."""
import json
from pathlib import Path
import sys

if "--version" in sys.argv:
    print("synthetic-writing-1")
    raise SystemExit
if "--help" in sys.argv:
    print("--tools --safe-mode --strict-mcp-config --setting-sources --session-id --max-budget-usd --permission-prompts --no-session-persistence")
    raise SystemExit

def arg(key):
    return sys.argv[sys.argv.index(key) + 1]

root = Path(__file__).parent
scenario = json.loads((root / "scenario.json").read_text())
request = json.load(sys.stdin)
with (root / "invocations.jsonl").open("a") as f:
    f.write(json.dumps(request) + "\n")
role = request["role"]
model = arg("--model")
session = arg("--session-id")
print(json.dumps({"type": "system", "subtype": "init", "session_id": session, "model": model, "tools": [], "mcp_servers": [], "skills": [], "plugins": [], "permissionMode": "dontAsk", "cwd": str(Path.cwd())}), flush=True)
if scenario.get("failure") == role:
    raise SystemExit(7)
if role == "planner":
    content = scenario["draft"]
else:
    supplied = next(x for x in request["materials"] if x["id"] == "delivery/review_subject")
    review = json.loads(supplied["text"])
    content = {"format": "ks.writing.review.v1", "subject": review["subject"], "items": [
        {"id": item["id"], "location": item["location"], "quote": item["quote"], "status": scenario.get("review", "pass"), "reason": "SYNTHETIC scripted review only; no semantic/model/GPU qualification"}
        for item in review["obligations"]]}
    if scenario.get("bad_subject"):
        content["subject"]["schema_sha256"] = "sha256:" + "0" * 64
output = {"protocol": "ks.claude.output.v1", "role": role, "assignment_sha256": request["assignment_sha256"], "status": "ok", "kind": "writing_draft" if role == "planner" else "review", "content": content, "reason": "SYNTHETIC writing CPU fixture"}
print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "session_id": session, "result": json.dumps(output), "num_turns": 1, "total_cost_usd": 0.001, "usage": {"input_tokens": 11, "output_tokens": 7, "cache_creation_input_tokens": 3, "cache_read_input_tokens": 5}, "modelUsage": {model: {"inputTokens": 11, "outputTokens": 7}}, "permission_denials": []}), flush=True)
