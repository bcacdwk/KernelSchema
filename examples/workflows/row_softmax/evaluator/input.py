"""Exact Task inputs; imported by the selected worker, never core import."""
import base64
import hashlib
import torch


def tensor_bytes(tensor):
    # A byte view preserves signed zero and NaN payloads; no numeric equality.
    return bytes(tensor.detach().cpu().contiguous().view(torch.uint8).flatten().tolist())


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def generate(case):
    m, n = case["shape"]
    kind = case["generator"]
    i = torch.arange(m, dtype=torch.int64).reshape(m, 1)
    j = torch.arange(n, dtype=torch.int64).reshape(1, n)
    if kind == "uniform":
        g = torch.Generator(device="cpu").manual_seed(case["seed"])
        value = -20 + 40 * torch.rand((m, n), generator=g, dtype=torch.float32, device="cpu")
    elif kind == "constant":
        value = torch.full((m, n), float(case["value"]), dtype=torch.float32)
    elif kind == "alternating":
        value = torch.where((i + j) % 2 == 0, 20, -20).to(torch.float32)
    elif kind == "single_peak":
        value = torch.where(j == (n - 1 - 17 * i) % n, 20, -20).to(torch.float32)
    elif kind == "near_equal":
        value = ((1 - 2 * (i % 2)) * 19).to(torch.float32) + ((j % 5) - 2).to(torch.float32) * (2 ** -12)
    elif kind == "permuted_ramp":
        value = ((j * (2 * i + 1) + i) % n).to(torch.float32)
        value = value / float(n - 1) * 40.0 + (-20.0)
    else:
        raise ValueError("unsupported Task generator " + kind)
    return value.contiguous()


def freeze_inputs(spec):
    cases = []
    for case in spec["cases"]:
        data = tensor_bytes(generate(case))
        cases.append({"case_id": case["case_id"], "sha256": digest(data),
                      "fp32_le_base64": base64.b64encode(data).decode("ascii")})
    return {"torch_version": str(torch.__version__), "byteorder": "little", "cases": cases}


def thaw(case, frozen):
    record = next(c for c in frozen["cases"] if c["case_id"] == case["case_id"])
    data = base64.b64decode(record["fp32_le_base64"], validate=True)
    if digest(data) != record["sha256"] or len(data) != 4 * case["shape"][0] * case["shape"][1]:
        raise ValueError("frozen FP32 input identity mismatch")
    return torch.frombuffer(bytearray(data), dtype=torch.float32).clone().reshape(case["shape"])


def allocate(case, cpu_input, device):
    m, n = case["shape"]
    layout = case["layout"]
    result = {}
    for name, bits in (("x", 0x4b123456), ("out", 0xcb123456)):
        offset = layout[name + "_storage_offset"]
        signed = bits if bits < 2 ** 31 else bits - 2 ** 32
        backing = torch.full((offset + m * n + layout["suffix_guard_elements"],), signed,
                             dtype=torch.int32, device=device).view(torch.float32)
        result[name + "_backing"] = backing
        result[name] = backing[offset:offset + m * n].view(m, n)
    result["x"].copy_(cpu_input)
    return result


def admit(x, out, *, device_type="cuda"):
    for tensor in (x, out):
        if tensor.dtype != torch.float32 or tensor.device.type != device_type or tensor.requires_grad:
            raise ValueError("Task requires non-gradient FP32 on the selected device")
        if tensor.ndim != 2 or tuple(tensor.shape) not in ((1, 128), (1, 1024), (32, 128), (32, 1024)):
            raise ValueError("out-of-domain Task shape")
        if tuple(tensor.stride()) != (tensor.shape[1], 1):
            raise ValueError("out-of-domain Task stride")
    if x.shape != out.shape or x.device != out.device:
        raise ValueError("Task requires equal shapes and devices")
    if x.untyped_storage()._cdata == out.untyped_storage()._cdata:
        raise ValueError("Task prohibits shared x/out storage")
    if not bool(torch.isfinite(x).all()) or not bool(((x >= -20) & (x <= 20)).all()):
        raise ValueError("out-of-domain Task values")
