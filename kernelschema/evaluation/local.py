"""Concrete local CPU executor and exactly-once receipt absorption interface.

The caller owns admission, reservation, parent accounting and active time. This
module stores operation facts, never a balance. CUDA is explicitly unsupported.
"""
from __future__ import annotations
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys

from ..artifacts import Publication, durable_mkdir, durable_write, read_regular
from ..contracts import PROTOCOL, ProtocolError, UnsupportedError, digest, jcs, loads, sha256, validate
from .task import verify_stage

PROFILE = "local-cpu-logic-seccomp-v1"


class LocalExecutor:
    def __init__(self, root, *, python=sys.executable):
        self.root = Path(root).absolute()
        self.python = str(Path(python).absolute())
        durable_mkdir(self.root)

    def capabilities(self):
        return {"profiles": [PROFILE], "CUDA": False, "Task_qualification": False,
                "recovery": "original local controller terminal only; unknown is settle_only",
                "permission_boundary": "Linux x86_64 seccomp TSYNC; filesystem writes denied"}

    def identity(self):
        return {"profile": PROFILE, "python": self.python,
                "implementation": {name: sha256(Path(__file__).with_name(name).read_bytes())
                                   for name in ("local.py", "_controller.py", "_worker.py", "_sandbox.py")}}

    def prepare(self, staged, *, include_timing=True):
        """Freeze exact CPU RNG bytes/environment before importing any candidate."""
        verify_stage(staged)
        root = Path(staged["root"])
        if (root / "execution.json").exists():
            ready = loads(read_regular(root, "execution.json")[0])
            if ready["include_timing"] != include_timing or ready["python"] != self.python:
                raise ProtocolError("prepared execution cannot change profile/timing/interpreter")
            return ready
        result = subprocess.run([self.python, "-I", "-B", str(root / "_worker.py"), str(root), "prepare"],
                                stdin=subprocess.DEVNULL, capture_output=True, timeout=60)
        if result.returncode:
            raise UnsupportedError("CPU preparation failed: " + result.stderr.decode(errors="replace")[-2000:])
        facts = loads(result.stdout)
        frozen = jcs(facts["inputs"])
        durable_write(root / "frozen_inputs.json", frozen)
        ready = {"profile": PROFILE, "python": self.python, "include_timing": include_timing,
                 "environment": facts["environment"], "frozen_inputs_sha256": sha256(frozen)}
        durable_write(root / "execution.json", jcs(ready))
        return ready

    def _job(self, operation_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", operation_id):
            raise ProtocolError("operation_id must be a safe stable local identifier")
        return self.root / operation_id

    def locator(self, operation_id):
        """Available to the caller's K2 RunContext intent before submission."""
        return {"root": str(self._job(operation_id)), "profile": PROFILE}

    def submit(self, staged, *, operation_id, submission_key, owner, reservation, profile=PROFILE):
        """Persist original key/locator/authorization BEFORE launching the controller.

        Repeated calls for the same intent only query it. An unknown operation
        fences new submissions in this executor root; no new key/channel retry.
        """
        if profile != PROFILE:
            raise UnsupportedError("no qualified CUDA/local profile; only CPU logic is implemented")
        if not owner or not submission_key or reservation.get("owner") != owner:
            raise ProtocolError("explicit matching caller owner/reservation/key required")
        if reservation.get("evaluator_jobs") != 1 or type(reservation.get("active_ms")) is not int or not 0 < reservation["active_ms"] <= 1200000:
            raise ProtocolError("one job and a positive <= 20 minute caller reservation required")
        verify_stage(staged)
        ready_data = read_regular(Path(staged["root"]), "execution.json")[0]
        ready = loads(ready_data)
        if ready["python"] != self.python or ready["profile"] != profile:
            raise ProtocolError("prepared interpreter/profile differs")
        job = self._job(operation_id)
        intent = {"operation_id": operation_id, "submission_key": submission_key, "owner": owner,
                  "reservation": reservation, "profile": profile, "stage": staged, "python": self.python,
                  "execution_sha256": sha256(ready_data), "native_locator": {"root": str(job)},
                  "producer": "kernelschema.local:" + str(self.root), "identity": self.identity()}
        with (self.root / ".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (job / "intent.json").exists():
                if loads(read_regular(job, "intent.json")[0]) != intent:
                    raise ProtocolError("operation replay differs from original intent")
                return self.query(operation_id)
            for old in self.root.iterdir():
                if old.is_dir() and (old / "intent.json").exists() and not (old / "native-terminal.json").exists():
                    raise ProtocolError("unsettled original operation fences new admission")
            durable_mkdir(job)
            durable_write(job / "intent.json", jcs(intent))
            # Intent existence means may-have-submitted, even if spawn acknowledgment
            # is lost. A second submit never starts a replacement controller.
            try:
                child = subprocess.Popen([self.python, "-I", "-B", str(Path(staged["root"]) / "_controller.py"), str(job)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    close_fds=True, start_new_session=True)
                durable_write(job / "controller.json", jcs({"pid": child.pid}))
            except Exception as exc:
                durable_write(job / "submission-error.json", jcs({"type": type(exc).__name__, "message": str(exc)}))
        return self.query(operation_id)

    def query(self, operation_id):
        job = self._job(operation_id)
        intent = loads(read_regular(job, "intent.json")[0])
        if intent["operation_id"] != operation_id or intent["native_locator"]["root"] != str(job):
            raise ProtocolError("original job locator/operation mismatch")
        if not (job / "native-terminal.json").exists():
            return {"state": "unknown", "recovery": "settle_only", "operation_id": operation_id,
                    "submission_key": intent["submission_key"], "native_locator": intent["native_locator"]}
        raw = read_regular(job, "native-terminal.json")[0]
        terminal = loads(raw)
        for name in ("operation_id", "submission_key", "owner", "profile", "execution_sha256"):
            if terminal[name] != intent[name]:
                raise ProtocolError("terminal subject differs from original " + name)
        if terminal["stage_sha256"] != intent["stage"]["sha256"]:
            raise ProtocolError("terminal stage subject mismatch")
        outputs = {name: read_regular(job, name)[0] for name in ("stdout.raw", "stderr.raw")}
        if any(sha256(data) != terminal["outputs"][name] for name, data in outputs.items()):
            raise ProtocolError("original terminal output bytes changed")
        worker = None
        if terminal["exit_code"] == 0:
            worker = loads(outputs["stdout.raw"])
            manifest = intent["stage"]["manifest"]
            if worker["loaded"] != manifest["files"] or worker["source"] != manifest["source"] or worker["evaluator"] != manifest["evaluator"]:
                raise ProtocolError("actual loaded candidate/evaluator differs from original subject")
            ready = loads(read_regular(Path(intent["stage"]["root"]), "execution.json")[0])
            if digest(ready) != intent["execution_sha256"] or worker["environment"] != ready["environment"] or worker["frozen_inputs_sha256"] != ready["frozen_inputs_sha256"]:
                raise ProtocolError("execution input/environment differs from preparation")
        return {"state": "terminal", "operation_id": operation_id, "terminal": terminal, "worker": worker,
                "receipt_id": sha256(raw), "sequence": terminal["sequence"], "native_locator": intent["native_locator"]}

    def settle(self, operation_id, publication: Publication):
        """Publish original receipts once. Returned usage row is cumulative per op.

        Caller absorbs producer+operation_id/receipt_id/sequence, never sums replay.
        No active_ms total is manufactured; owner accounts its single clock.
        """
        job = self._job(operation_id)
        facts = self.query(operation_id)
        if facts["state"] != "terminal":
            return facts
        intent = loads(read_regular(job, "intent.json")[0])
        binding = job / "absorption.json"
        publication_binding = job / "publication.json"
        if publication_binding.exists():
            selected = loads(read_regular(job, "publication.json")[0])
            if selected["root"] != str(publication.root):
                raise ProtocolError("receipt publication must use its original bundle root")
            publication = Publication(selected["root"], selected["publication_id"])
        else:
            durable_write(publication_binding, jcs({"root": str(publication.root), "publication_id": publication.publication_id}))
        previous = loads(read_regular(job, "absorption.json")[0]) if binding.exists() else None
        if previous:
            if previous["receipt_id"] != facts["receipt_id"] or previous["sequence"] != facts["sequence"]:
                raise ProtocolError("conflicting original operation receipt")
            from ..artifacts import Bundle
            Bundle(publication.root).document(previous["report"], "evaluation")
            return {**previous, "new_receipt": False}
        prefix = "evaluation/" + operation_id
        raw_refs = [publication.put_bytes(prefix + "/" + name, read_regular(job, name)[0]) for name in ("native-terminal.json", "stdout.raw", "stderr.raw")]
        worker = facts["worker"]
        ready = loads(read_regular(Path(intent["stage"]["root"]), "execution.json")[0])
        environment = publication.put_bytes(prefix + "/environment.json", jcs(ready["environment"]))
        frozen = publication.put_bytes(prefix + "/frozen-inputs.json", read_regular(Path(intent["stage"]["root"]), "frozen_inputs.json")[0])
        manifest = intent["stage"]["manifest"]
        row = {"producer": intent["producer"], "operation_id": operation_id, "role": "evaluator", "purpose": "CPU evaluator logic validation",
               "receipt_id": facts["receipt_id"], "sequence": facts["sequence"], "receipt": raw_refs[0], "state": "settled", **facts["terminal"]["usage"]}
        usage = publication.put_json(prefix + "/usage.json", {"protocol": PROTOCOL, "kind": "usage_index", "entries": [row]})
        result = None if worker is None else worker["result"]
        samples = publication.put_bytes(prefix + "/samples.json", jcs(result["timing"])) if result and result["timing"] else None
        check = lambda status, reason: {"status": status, "reason": reason, "receipt": raw_refs[1] if status in ("pass", "fail") else None}
        # CPU can prove import/build facts and reject a bad logical candidate, but
        # cannot issue a CUDA correctness/performance pass.
        build = check(result["build"], "actual CPU import/construction/calls; no CUDA build qualification") if result else check("unknown", "worker did not return complete structured execution facts")
        correctness = check("fail", "CPU logic exposed an independent Task postcondition failure") if result and result.get("correctness") and not result["correctness"]["passed"] else check("unknown", "CPU logic only; Task requires CUDA")
        report = {"protocol": PROTOCOL, "kind": "evaluation", "operation_id": operation_id,
                  "subject": {"task_sha256": manifest["task"]["sha256"], "source_sha256": manifest["source"]["sha256"],
                              "evaluator_sha256": manifest["evaluator"]["sha256"], "inputs_sha256": manifest["inputs"]["sha256"],
                              "timing_sha256": manifest["timing"]["sha256"], "environment_sha256": environment["sha256"]},
                  "build": build, "correctness": correctness, "performance": check("unknown", "CPU timing is logic evidence only; no CUDA timing qualification"),
                  "case_ids": sorted({r["case_id"] for r in result["correctness"]["basic_calls"]}) if result and result.get("correctness") else [],
                  "samples": samples, "environment": environment, "raw_receipts": raw_refs + [frozen], "usage_receipts": [usage],
                  "extensions": {"information": {"synthetic": True, "qualification": "none", "profile": PROFILE}}}
        report_ref = publication.put_json(prefix + "/report.json", report)
        publication.commit([report_ref], subjects=[manifest["task"], manifest["source"]])
        settled = {"state": "terminal", "report": report_ref, "usage_row": row, "receipt_id": facts["receipt_id"],
                   "sequence": facts["sequence"], "new_receipt": True}
        durable_write(binding, jcs(settled))
        return settled
