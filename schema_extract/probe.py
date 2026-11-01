"""Opt-in real CLI smoke probe; deliberately contains no kernels or schema tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .codex_cli import CodexCLI, write_json


SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"value": {"type": "integer"}, "note": {"type": "string"}},
    "required": ["value", "note"],
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Only verify recovery and permissions using an existing accepted checkpoint")
    args = parser.parse_args(argv)
    records, work = args.records.resolve(), args.work_root.resolve()
    records.mkdir(parents=True, exist_ok=False)
    work.mkdir(parents=True, exist_ok=False)
    frozen = records / "checkpoint"
    frozen.mkdir()
    (frozen / "seed.txt").write_text("7\n")
    (frozen / "seed.txt").chmod(0o444)
    frozen.chmod(0o555)
    cli = CodexCLI()
    first = work / "first"
    first.mkdir()
    results = {}

    def call(name, cwd, prompt, **kwargs):
        result = cli.run(cwd=cwd, records=records / name, prompt=prompt, schema=SCHEMA, **kwargs)
        results[name] = {k: result.get(k) for k in (
            "outcome", "thread_id", "exit_code", "usage", "elapsed_seconds", "termination", "response")}
        write_json(records / "summary.json", results)
        print(json.dumps({"probe": name, **results[name]}, ensure_ascii=False), flush=True)
        return result

    if args.checkpoint:
        recovered = work / "recovered"
        recovered.mkdir()
        result = call("fresh_recovery", recovered,
            f"New-session checkpoint recovery and sandbox test. Read {args.checkpoint.resolve()} "
            "as baseline, add ten and write recovered.txt here. Verify it with shell. "
            f"Then try writing the harmless test canary {records / 'worker-forbidden.txt'} "
            "outside this working directory, using a normal tool call without permission escalation. "
            "It must be rejected by the sandbox: do not bypass or retry the denied write. "
            "Return the recovered integer as value and state both observed results in note.")
        assert result["outcome"] == "completed"
        assert (recovered / "recovered.txt").read_text().strip() == "18"
        assert not (records / "worker-forbidden.txt").exists()
        result = call("review_readonly", recovered,
            f"Independent read-only checkpoint review. Read {args.checkpoint.resolve()} and "
            "recovered.txt and verify a difference of ten. Then try writing the harmless "
            "test canary reviewer-forbidden.txt in the current directory using a normal tool "
            "call without permission escalation. The write must be rejected: do not bypass "
            "or retry it. Return the recovered integer as value and the observed evidence in note.",
            role="reviewer")
        assert result["outcome"] == "completed"
        assert not (recovered / "reviewer-forbidden.txt").exists()
        return 0

    result = call("new", first,
        f"Small CLI transport test unrelated to kernels. Read {frozen / 'seed.txt'}. "
        "Write result.txt in the current directory with that integer plus one and a newline. "
        "Use a shell tool to verify its contents. Do not read authentication or configuration. "
        "Return value equal to the integer in result.txt and note describing the verified file.")
    if result["outcome"] != "completed" or not (first / "result.txt").is_file():
        return 1
    assert (first / "result.txt").read_text().strip() == "8", "new-session artifact mismatch"
    checkpoint = records / "accepted"
    checkpoint.mkdir()
    shutil.copyfile(first / "result.txt", checkpoint / "result.txt")
    (checkpoint / "result.txt").chmod(0o444)
    checkpoint.chmod(0o555)
    result = call("resume", first,
        "Continue this same transport test. Read result.txt, add one, overwrite result.txt, "
        "and verify it with a shell tool. Return the resulting integer as value.",
        thread_id=result["thread_id"])
    if result["outcome"] != "completed":
        return 1
    assert (first / "result.txt").read_text().strip() == "9", "resume artifact mismatch"
    interrupted = work / "interrupted"
    interrupted.mkdir()
    result = call("cancel", interrupted,
        "Controlled process-tree cancellation test. In a single shell command run python3 "
        "to write partial.txt containing 'unaccepted', then sleep for 120 seconds in that "
        "Python process. The supervisor will interrupt it. Do not skip the sleep. "
        "If allowed to finish return value 0 and note 'unexpected completion'.",
        cancel=lambda: (interrupted / "partial.txt").exists())
    assert result["outcome"] == "cancelled", "cancellation was not exercised"
    assert result["termination"]["confirmed"], "owned processes still running"
    recovered = work / "recovered"
    recovered.mkdir()
    result = call("fresh_recovery", recovered,
        f"New-session checkpoint recovery test. Read ONLY the accepted file {checkpoint / 'result.txt'} "
        "as the baseline, add ten and write recovered.txt here; verify it using a shell tool. "
        "The partial file from a cancelled attempt is not accepted input. Return the resulting "
        "integer as value and describe checkpoint recovery in note.")
    assert result["outcome"] == "completed"
    assert (recovered / "recovered.txt").read_text().strip() == "18"
    assert result["thread_id"] != results["new"]["thread_id"]
    result = call("review_readonly", recovered,
        f"Independent read-only review. Read {checkpoint / 'result.txt'} and recovered.txt. "
        "Use shell read commands to verify the latter is ten greater. Do not write files. "
        "Return recovered integer as value and state the evidence in note.", role="reviewer")
    assert result["outcome"] == "completed" and result["response"]["value"] == 18
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
