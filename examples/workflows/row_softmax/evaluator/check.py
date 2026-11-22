"""Independent original-storage, byte, view, poison and golden checks."""
import base64
import torch


def signature(tensor):
    return {"object_id": id(tensor), "storage_id": tensor.untyped_storage()._cdata,
            "storage_offset": tensor.storage_offset(), "shape": list(tensor.shape),
            "stride": list(tensor.stride()), "dtype": str(tensor.dtype), "device": str(tensor.device),
            "requires_grad": tensor.requires_grad}


def call(model, buffers, case, poison, *, inputs, golden, synchronize, device_type):
    x, out = buffers["x"], buffers["out"]
    inputs.admit(x, out, device_type=device_type)
    out.fill_(float("nan") if poison == "nan" else -123.0)
    synchronize()
    # Keep original storages and independent views alive even if set_ mutates out.
    x_storage, out_storage = x.untyped_storage(), out.untyped_storage()
    xb, ob = buffers["x_backing"], buffers["out_backing"]
    xs, os = signature(x), signature(out)
    x_before = x.detach().cpu().clone()
    xb_before, ob_before = inputs.tensor_bytes(xb), inputs.tensor_bytes(ob)
    expected = torch.tensor(golden.softmax_fp64(x_before.tolist()), dtype=torch.float64)
    error, returned = None, None
    try:
        with torch.inference_mode():
            returned = model.forward(x=x, out=out)
        synchronize()
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        # A poisoned device after a fault is handled by the outer worker.
        synchronize()
    xb_after, ob_after = inputs.tensor_bytes(xb), inputs.tensor_bytes(ob)
    start = os["storage_offset"] * 4
    end = start + 4 * case["shape"][0] * case["shape"][1]
    actual_bytes = ob_after[start:end]
    actual = torch.frombuffer(bytearray(actual_bytes), dtype=torch.float32).clone().reshape(case["shape"]).to(torch.float64)
    finite = bool(torch.isfinite(actual).all())
    within = torch.abs(actual - expected) <= 2e-6 + 2e-5 * torch.abs(expected)
    numerical = finite and bool(within.all())
    written = bool(torch.isfinite(actual).all()) if poison == "nan" else bool((actual != -123.0).all())
    checks = {"execution": error is None, "return_identity": returned is out,
              "out_object": id(out) == os["object_id"], "out_storage": out.untyped_storage()._cdata == out_storage._cdata,
              "out_view": signature(out) == os, "out_fully_written": written,
              "x_object": id(x) == xs["object_id"], "x_storage": x.untyped_storage()._cdata == x_storage._cdata,
              "x_view": signature(x) == xs, "x_bytes": xb_before == xb_after,
              "out_prefix_guard": ob_before[:start] == ob_after[:start],
              "out_suffix_guard": ob_before[end:] == ob_after[end:], "finite_output": finite,
              "elementwise_tolerance": numerical}
    return {"case_id": case["case_id"], "poison": poison, "checks": checks, "passed": all(checks.values()),
            "exception": error, "input_sha256": inputs.digest(inputs.tensor_bytes(x_before)),
            "x_before": xs, "x_after": signature(x), "out_before": os, "out_after": signature(out),
            "x_backing_object": id(xb), "out_backing_object": id(ob),
            "x_backing_before_sha256": inputs.digest(xb_before), "x_backing_after_sha256": inputs.digest(xb_after),
            "out_prefix_before_hex": ob_before[:start].hex(), "out_prefix_after_hex": ob_after[:start].hex(),
            "out_suffix_before_hex": ob_before[end:].hex(), "out_suffix_after_hex": ob_after[end:].hex(),
            "original_out_fp32_le_base64": base64.b64encode(actual_bytes).decode("ascii"),
            "golden_fp64_le_base64": base64.b64encode(inputs.tensor_bytes(expected)).decode("ascii"),
            "failing_elements": int((~within).sum()), "max_abs_error": str(float(torch.abs(actual - expected).max()))}


def evaluate(factory, spec, frozen, *, inputs, golden, device, synchronize):
    records, traces = [], []
    for case in spec["cases"]:
        for poison in case["out_initializers"]:
            # Each basic call has a fresh Model. Stateful recomputation is checked separately.
            buffers = inputs.allocate(case, inputs.thaw(case, frozen), device)
            records.append(call(factory(), buffers, case, poison, inputs=inputs, golden=golden,
                                synchronize=synchronize, device_type=device.type))
    cases = {c["case_id"]: c for c in spec["cases"]}
    for trace in spec["state_reuse_traces"]:
        model = factory()
        first = cases[trace["basic_case_ids"][0]]
        buffers = inputs.allocate(first, inputs.thaw(first, frozen), device)
        calls = []
        for case_id, poison in zip(trace["basic_case_ids"], trace["out_initializers"], strict=True):
            case = cases[case_id]
            buffers["x"].copy_(inputs.thaw(case, frozen))
            calls.append(call(model, buffers, case, poison, inputs=inputs, golden=golden,
                              synchronize=synchronize, device_type=device.type))
        traces.append({"trace_id": trace["trace_id"], "model_object": id(model), "calls": calls})
    return {"basic_calls": records, "state_reuse": traces, "basic_call_count": len(records),
            "passed": len(records) == 72 and all(r["passed"] for r in records) and
                      sum(len(t["calls"]) for t in traces) == 2 and all(r["passed"] for t in traces for r in t["calls"])}
