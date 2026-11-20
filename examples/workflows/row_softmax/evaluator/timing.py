"""The Task's fixed complete-forward clock, with lossless raw ns samples."""
import math
import statistics
import time
import torch


def summarize(samples):
    ordered = sorted(samples)
    def quantile(p):
        rank = (len(ordered) - 1) * p
        low = math.floor(rank)
        return ordered[low] + (ordered[math.ceil(rank)] - ordered[low]) * (rank - low)
    return {"median_ns": str(statistics.median(ordered)), "iqr_ns": str(quantile(.75) - quantile(.25)),
            "min_ns": str(min(ordered)), "max_ns": str(max(ordered)), "sample_count": len(ordered)}


def measure(candidate, baseline, spec, frozen, *, inputs, device, synchronize, clock=time.perf_counter_ns):
    batches, preparation, summaries = [], [], []
    for case in (c for c in spec["cases"] if c["generator"] == "uniform" and c["seed"] == 17):
        cpu = inputs.thaw(case, frozen)
        buffers = {label: inputs.allocate(case, cpu, device) for label in ("A", "B")}
        models = {}
        for label, factory in (("A", candidate), ("B", baseline)):
            start = clock()
            models[label] = factory()
            construction = clock() - start
            args = {k: buffers[label][k] for k in ("x", "out")}
            inputs.admit(**args, device_type=device.type)
            synchronize()
            start = clock()
            with torch.inference_mode():
                models[label].forward(**args)
                synchronize()
                first = clock() - start
                start = clock()
                for _ in range(20):
                    models[label].forward(**args)
                synchronize()
                warmup = clock() - start
            preparation.append({"case_id": case["case_id"], "model": label, "construction_ns": construction,
                                "first_invocation_ns": first, "warmup_calls": 20, "warmup_ns": warmup})
        with torch.inference_mode():
            for pair in range(10):
                order = "AB" if pair % 2 == 0 else "BA"
                for position, label in enumerate(order):
                    model, b = models[label], buffers[label]
                    synchronize()
                    start = clock()
                    for _ in range(100):
                        model.forward(x=b["x"], out=b["out"])
                    synchronize()
                    end = clock()
                    if end <= start:
                        raise ValueError("unusable nonpositive timing sample")
                    batches.append({"case_id": case["case_id"], "pair": pair, "order": order,
                                    "position": position, "model": label, "start_ns": str(start), "end_ns": str(end),
                                    "elapsed_ns": end - start, "calls": 100,
                                    "per_forward_ns": str((end - start) / 100)})
        stats = {label: summarize([r["elapsed_ns"] / 100 for r in batches if r["case_id"] == case["case_id"] and r["model"] == label]) for label in ("A", "B")}
        summaries.append({"case_id": case["case_id"], "models": stats})
    ratio = math.exp(sum(math.log(float(s["models"]["B"]["median_ns"]) / float(s["models"]["A"]["median_ns"])) for s in summaries) / 4)
    complete = len(batches) == 80 and len(summaries) == 4 and all(s["models"][m]["sample_count"] == 10 for s in summaries for m in ("A", "B"))
    return {"complete": complete, "preparation": preparation, "batches": batches, "summaries": summaries,
            "geomean_baseline_over_candidate": str(ratio), "soft_goal": ("met" if ratio >= 1 else "unmet") if complete else "unknown",
            "qualification": "CUDA Task timing" if device.type == "cuda" else "CPU logic only; no Task performance qualification"}
