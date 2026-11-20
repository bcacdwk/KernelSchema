"""Finite, self-contained public target materials; never device qualification."""
from copy import deepcopy
import re
from ..contracts import ProtocolError, UnsupportedError, jcs, loads, sha256

FORMAT = "ks.row-softmax.target.v1"


def read_target(data):
    """Read explicit target declarations with inline original source documents.

    Original document text is retained inside the environment object so Task's
    ordinary environment edge carries the complete closure without a new codec.
    Runtime expectations are compared separately with actual K1 CUDA capture.
    """
    try:
        value = loads(data)
    except ProtocolError as error:
        raise UnsupportedError("unrecognized row-softmax environment; use ks.row-softmax.target.v1") from error
    if not isinstance(value, dict) or set(value) != {"format", "hardware_specification", "runtime", "sources"} or value["format"] != FORMAT:
        raise UnsupportedError("unrecognized row-softmax environment; use ks.row-softmax.target.v1")
    hardware = value["hardware_specification"]
    required = {"Device", "Architecture", "Execution", "Feature", "Stack", "Wiki"}
    if not isinstance(hardware, dict) or set(hardware) != required:
        raise ProtocolError("target hardware requires Device/Architecture/Execution/Feature/Stack/Wiki")
    def strings(obj):
        if isinstance(obj, dict) and obj:
            for key, val in obj.items():
                if not isinstance(key, str) or not key: raise ProtocolError("invalid hardware key")
                strings(val)
        elif not isinstance(obj, str) or not obj.strip() or re.search(r"@pending|unselected|unknown", obj, re.I):
            raise ProtocolError("target hardware leaves must be explicit nonempty strings")
    strings(hardware)
    runtime = value["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != {"name", "architecture", "torch", "triton", "cuda_build"}:
        raise ProtocolError("target runtime requires name/architecture/torch/triton/cuda_build")
    if any(not isinstance(runtime[k], str) or not runtime[k].strip() for k in ("name", "torch", "triton", "cuda_build")):
        raise ProtocolError("target runtime versions/device name must be explicit")
    if not isinstance(runtime["architecture"], list) or len(runtime["architecture"]) != 2 or any(type(n) is not int or n < 0 for n in runtime["architecture"]):
        raise ProtocolError("target architecture must be [major, minor]")
    sources = value["sources"]
    if not isinstance(sources, list) or not sources:
        raise ProtocolError("target needs original public source documents")
    locators = set()
    for source in sources:
        if not isinstance(source, dict) or set(source) != {"locator", "sha256", "text", "license"}:
            raise ProtocolError("target source requires locator/hash/text/license")
        if any(not isinstance(source[k], str) or not source[k].strip() for k in ("locator", "text", "license")):
            raise ProtocolError("target source fields must be nonempty")
        if not re.fullmatch(r"[^\s<>@]+@[0-9a-f]{40}:[^\s<>]+", source["locator"]):
            raise ProtocolError("target source locator must use an immutable repository@commit:path")
        if source["sha256"] != sha256(source["text"].encode()) or source["locator"] in locators:
            raise ProtocolError("target source bytes/hash or duplicate locator")
        locators.add(source["locator"])
    def leaves(obj):
        return [v for child in obj.values() for v in leaves(child)] if isinstance(obj, dict) else [obj]
    for locator in leaves(hardware["Wiki"]):
        if locator.strip("<>") not in locators:
            raise ProtocolError("every target Wiki locator must have original source bytes")
    return deepcopy(value)


def target_environment(*, hardware_specification, runtime, sources):
    """Construct an explicit declaration from caller-selected facts/documents."""
    value = {"format": FORMAT, "hardware_specification": hardware_specification,
             "runtime": runtime, "sources": sources}
    return read_target(jcs(value))


def match_target(data, observed):
    target = read_target(data)
    if observed.get("mode") != "cuda" or any(observed.get(k) != v for k, v in target["runtime"].items()):
        raise UnsupportedError("actual CUDA capture differs from the fixed Task target runtime")
    return target
