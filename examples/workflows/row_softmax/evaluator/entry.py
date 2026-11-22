"""One evaluator shared by explicit profiles; no wrapper repairs candidate out."""
import platform
import hashlib
from pathlib import Path
import subprocess
import sys
import time
import torch


def environment(mode):
    result = {"mode": mode, "python": sys.version, "executable": sys.executable,
              "os": platform.platform(), "machine": platform.machine(), "torch": str(torch.__version__),
              "cuda_build": torch.version.cuda, "device": "cpu", "qualification": "none"}
    result["python_binary_sha256"] = "sha256:" + hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
    result["torch_init_sha256"] = "sha256:" + hashlib.sha256(Path(torch.__file__).read_bytes()).hexdigest()
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA profile unavailable; no fallback permitted")
        prop = torch.cuda.get_device_properties(torch.cuda.current_device())
        import triton
        result.update(device=str(torch.cuda.current_device()), name=prop.name, architecture=list(torch.cuda.get_device_capability()),
                      total_memory=prop.total_memory, triton=triton.__version__)
        observed = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=10, check=True)
        result["nvidia_smi"] = observed.stdout.strip()
    return result


def run(candidate, baseline, spec, frozen, modules, mode, *, include_timing=True):
    if frozen["torch_version"] != str(torch.__version__) or sys.byteorder != "little":
        raise ValueError("frozen input runtime/byte order differs")
    device = torch.device("cuda" if mode == "cuda" else "cpu")
    sync = (lambda: torch.cuda.synchronize(device)) if mode == "cuda" else (lambda: None)
    started = time.perf_counter_ns()
    try:
        model = candidate()
        if not isinstance(model, torch.nn.Module):
            raise TypeError("candidate Model must inherit torch.nn.Module")
    except Exception as exc:
        return {"build": "fail", "error": {"type": type(exc).__name__, "message": str(exc)},
                "correctness": None, "timing": None, "preparation_ns": time.perf_counter_ns() - started}
    preparation = time.perf_counter_ns() - started
    correctness = modules["check"].evaluate(candidate, spec, frozen, inputs=modules["input"], golden=modules["golden"], device=device, synchronize=sync)
    timing, timing_error = None, None
    if correctness["passed"] and include_timing:
        try:
            timing = modules["timing"].measure(candidate, baseline, spec, frozen, inputs=modules["input"], device=device, synchronize=sync)
        except Exception as exc:
            timing_error = {"type": type(exc).__name__, "message": str(exc)}
    return {"build": "pass" if all(r["exception"] is None for r in correctness["basic_calls"]) else "fail",
            "correctness": correctness, "timing": timing, "timing_error": timing_error,
            "preparation_ns": preparation, "preparation_note": "constructor probe and all correctness calls precede timing; compilation may occur there"}
