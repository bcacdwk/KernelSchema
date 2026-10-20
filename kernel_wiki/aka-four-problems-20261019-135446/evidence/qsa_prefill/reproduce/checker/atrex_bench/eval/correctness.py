"""Stage 1: Correctness verification against the eager reference baseline."""

from __future__ import annotations

import inspect
import json
import math
import os
import traceback
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path

import torch

from atrex_bench.eval._runtime import (
    ShapeSpec,
    StorageSnapshot,
    clone_model_inputs,
    clone_output_values,
    deterministic_input_seed,
    flatten_outputs,
    get_device,
    instantiate_model_pair,
    load_reference_inputs,
    load_shape_call_inputs,
    load_shape_spec,
    normalize_model_inputs,
    output_aliases,
    seed_all_input_rngs,
    sync_device,
    write_input_artifact,  # noqa: F401 - backward-compatible module export
)
from atrex_bench.eval._timeout import CandidateTimeoutError, candidate_timeout
from atrex_bench.eval.reward_hack import (
    RewardHackDetected,
    check_plain_tensor_outputs,
)

_DEFAULT_CANDIDATE_TIMEOUT_S = 60
CORRECTNESS_MAX_REL_L2_ENV = "ATREX_CORRECTNESS_MAX_REL_L2"


@dataclass(frozen=True)
class OutputDiff:
    """Per-output comparison result for one correctness case.

    Fields match the data schema spec, Section 7 outputs entry exactly:
    ``name`` / ``passed`` / ``max_elementwise_abs_diff`` /
    ``max_elementwise_rel_diff`` / ``error``. dtype / shape are intentionally
    not recorded — they are derivable from metadata.json.output_dtypes and
    do not have a real consumer.
    """

    name: str
    passed: bool
    max_elementwise_abs_diff: float | None = None
    max_elementwise_rel_diff: float | None = None
    relative_l2: float | None = None
    error: str | None = None
    mismatched_elements: int | None = None
    total_elements: int | None = None
    allowed_mismatched_elements: int | None = None
    tolerance: dict[str, object] | None = None


@dataclass(frozen=True)
class OutputTolerance:
    """Elementwise tolerance for one logical output tensor path."""

    atol: float
    rtol: float
    max_mismatch_fraction: float = 0.0
    max_rel_l2: float | None = None


@dataclass(frozen=True)
class CorrectnessCase:
    """One correctness case: a single random input draw.

    ``input_artifact`` is the only per-case input information persisted in
    eval_result.json — the actual random tensor values live in the .pt file.
    Everything else (init_kwargs, input_kwargs) is derivable from
    shapes.json + input.py, so it is not duplicated here.
    """

    input_artifact: dict[str, str] | None
    outputs: list[OutputDiff] = field(default_factory=list)
    mutated_inputs: list[OutputDiff] = field(default_factory=list)
    unexpected_mutations: list[OutputDiff] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class CorrectnessShapeResult:
    """Per-shape correctness result returned by ``check_correctness``.

    ``status`` ∈ ``{"passed", "failed", "skipped"}`` is what bubbles up to
    ``eval_result.json.passed.correctness.<shape_id>``; ``cases`` populate
    ``eval_result.json.correctness.shapes.<shape_id>.cases``.
    ``check_correctness`` itself only returns ``passed`` or ``failed``;
    ``skipped`` is set by run_eval when the stage was not run because of an
    earlier-stage failure.
    """

    status: str
    reason: str | None = None
    cases: list[CorrectnessCase] = field(default_factory=list)


def _is_leaf_output(value: object) -> bool:
    """Return whether a value is a leaf in the output tree (tensor or numeric scalar)."""
    return value is None or isinstance(value, (torch.Tensor, bool, int, float))


def _describe_output_structure(value: object) -> str:
    """Return a short structural label for one output node, used in error messages."""
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={list(value.shape)}, dtype={value.dtype})"
    if isinstance(value, dict):
        return f"dict(keys={sorted(value.keys())!r})"
    if isinstance(value, tuple):
        return f"tuple(len={len(value)})"
    if isinstance(value, list):
        return f"list(len={len(value)})"
    return type(value).__name__


def _validate_output_structures_match(
    reference: object,
    candidate: object,
    *,
    path: str = "output",
    strict_types: bool = False,
) -> None:
    """Raise ValueError if reference / candidate output structures don't match."""
    if (strict_types and type(reference) is not type(candidate)) or (
        (reference is None) != (candidate is None)
    ):
        raise ValueError(
            f"Output structure mismatch at {path}: reference={type(reference).__name__}, "
            f"candidate={type(candidate).__name__}"
        )
    ref_is_dict = isinstance(reference, dict)
    cand_is_dict = isinstance(candidate, dict)
    if ref_is_dict != cand_is_dict:
        raise ValueError(
            f"Output structure mismatch at {path}: "
            f"reference is {_describe_output_structure(reference)}, "
            f"candidate is {_describe_output_structure(candidate)}. "
            "When the reference Model.forward returns dict[str, Tensor], the "
            "candidate must also return a dict with matching keys."
        )
    if ref_is_dict:
        ref_keys = set(reference.keys())
        cand_keys = set(candidate.keys())
        if ref_keys != cand_keys:
            missing_in_candidate = sorted(ref_keys - cand_keys)
            extra_in_candidate = sorted(cand_keys - ref_keys)
            raise ValueError(
                f"Output structure mismatch at {path}: dict keys differ. "
                f"reference keys={sorted(ref_keys)}, "
                f"candidate keys={sorted(cand_keys)}, "
                f"missing in candidate={missing_in_candidate}, "
                f"extra in candidate={extra_in_candidate}."
            )
        for key in sorted(ref_keys):
            _validate_output_structures_match(
                reference[key], candidate[key], path=f"{path}.{key}", strict_types=strict_types
            )
        return

    ref_is_seq = isinstance(reference, (tuple, list))
    cand_is_seq = isinstance(candidate, (tuple, list))
    if ref_is_seq != cand_is_seq:
        raise ValueError(
            f"Output structure mismatch at {path}: "
            f"reference is {_describe_output_structure(reference)}, "
            f"candidate is {_describe_output_structure(candidate)}."
        )
    if ref_is_seq:
        if len(reference) != len(candidate):
            raise ValueError(
                f"Output structure mismatch at {path}: sequence length differs. "
                f"reference={len(reference)}, candidate={len(candidate)}."
            )
        for index, (ref_item, cand_item) in enumerate(zip(reference, candidate)):
            _validate_output_structures_match(
                ref_item, cand_item, path=f"{path}[{index}]", strict_types=strict_types
            )
        return

    if not (_is_leaf_output(reference) and _is_leaf_output(candidate)):
        raise ValueError(
            f"Output structure mismatch at {path}: "
            f"reference is {_describe_output_structure(reference)}, "
            f"candidate is {_describe_output_structure(candidate)}."
        )


def _flatten_output_name(prefix_path: str) -> str:
    """Convert flatten_outputs() prefix paths to the schema's tensor name.

    flatten_outputs emits "output" for single tensors, "output.<key>" for dict
    branches, "output[i]" for tuple branches. The schema records just the
    tensor name from metadata.json.output_dtypes:
      "output"        -> "out"           (single-tensor convention)
      "output.<key>"  -> "<key>"         (dict tensor name)
      "output[i]"     -> kept as-is      (legacy tuple path; not reachable
                                          under the current schema, but
                                          tolerated for backward compat)
    """
    if prefix_path == "output":
        return "out"
    if prefix_path.startswith("output."):
        return prefix_path[len("output.") :]
    return prefix_path


def configured_max_rel_l2(explicit: float | None = None) -> float | None:
    if explicit is not None:
        value = float(explicit)
    else:
        raw = os.environ.get(CORRECTNESS_MAX_REL_L2_ENV)
        if raw is None or not raw.strip():
            return None
        value = float(raw)
    if value < 0:
        raise ValueError("correctness max_rel_l2 must be non-negative")
    return value


def _compare_output_tensors(
    reference_tensor: torch.Tensor,
    candidate_tensor: torch.Tensor,
    *,
    name: str,
    atol: float,
    rtol: float,
    max_rel_l2: float | None = None,
    strict_dtype: bool = False,
    max_mismatch_fraction: float = 0.0,
    max_rel_l2_bound: float | None = None,
) -> OutputDiff:
    """Compare a pair of output tensors and return the per-output diff record."""
    if strict_dtype and reference_tensor.dtype != candidate_tensor.dtype:
        return OutputDiff(
            name=name,
            passed=False,
            error=f"Output dtype mismatch: {reference_tensor.dtype} != {candidate_tensor.dtype}",
        )
    if reference_tensor.shape != candidate_tensor.shape:
        return OutputDiff(
            name=name,
            passed=False,
            error=(
                "Output shape mismatch: "
                f"reference={list(reference_tensor.shape)}, "
                f"candidate={list(candidate_tensor.shape)}"
            ),
        )

    error: str | None = None
    if torch.is_floating_point(reference_tensor) or torch.is_floating_point(candidate_tensor):
        reference_float = reference_tensor.detach().to(torch.float64)
        candidate_float = candidate_tensor.detach().to(torch.float64)
        ref_is_finite = bool(torch.isfinite(reference_float).all().item())
        cand_is_finite = bool(torch.isfinite(candidate_float).all().item())
        if not ref_is_finite or not cand_is_finite:
            return OutputDiff(
                name=name,
                passed=False,
                max_elementwise_abs_diff=0.0,
                max_elementwise_rel_diff=0.0,
                relative_l2=0.0,
                error=(
                    "Non-finite output detected: "
                    f"reference_finite={ref_is_finite}, candidate_finite={cand_is_finite}"
                ),
            )

        reference_norm = torch.linalg.vector_norm(reference_float)
        candidate_norm = torch.linalg.vector_norm(candidate_float)
        candidate_is_zero = (
            float(reference_norm.item()) > 0.0 and float(candidate_norm.item()) == 0.0
        )

        abs_diff = (reference_float - candidate_float).abs()
        max_elementwise_abs_diff = float(abs_diff.max().item()) if abs_diff.numel() else 0.0
        denominator = reference_float.abs().clamp_min(max(atol, 1e-12))
        max_elementwise_rel_diff = (
            float((abs_diff / denominator).max().item()) if abs_diff.numel() else 0.0
        )
        diff_l2 = torch.linalg.vector_norm(reference_float - candidate_float)
        relative_l2 = float((diff_l2 / reference_norm.clamp_min(1e-12)).item())
        # Preserve the historical allclose(ref, candidate) argument direction.
        mismatched = int((abs_diff > atol + rtol * candidate_float.abs()).sum().item())
        fraction = Fraction(str(max_mismatch_fraction))
        allowed = reference_float.numel() * fraction.numerator // fraction.denominator
        passed = relative_l2 <= max_rel_l2 if max_rel_l2 is not None else mismatched <= allowed
        if max_rel_l2_bound is not None:
            # This metadata bound is conjunctive, unlike the legacy CLI L2 mode.
            l2_passed = (
                max_elementwise_abs_diff == 0.0
                if float(reference_norm.item()) == 0.0
                else relative_l2 <= max_rel_l2_bound
            )
            passed = passed and l2_passed
        if candidate_is_zero and not passed:
            error = "Candidate output is all zero while reference output is non-zero"
    else:
        passed = bool(torch.equal(reference_tensor, candidate_tensor))
        max_elementwise_abs_diff = 0.0 if passed else 1.0
        max_elementwise_rel_diff = 0.0 if passed else 1.0
        relative_l2 = None
        mismatched = int((reference_tensor != candidate_tensor).sum().item())
        allowed = 0

    return OutputDiff(
        name=name,
        passed=passed,
        max_elementwise_abs_diff=max_elementwise_abs_diff,
        max_elementwise_rel_diff=max_elementwise_rel_diff,
        relative_l2=relative_l2,
        error=error,
        mismatched_elements=mismatched,
        total_elements=reference_tensor.numel(),
        allowed_mismatched_elements=allowed,
        tolerance={
            "atol": atol,
            "rtol": rtol,
            "max_mismatch_fraction": max_mismatch_fraction,
            "max_rel_l2": max_rel_l2_bound,
            "relative_to": "candidate",
            "mode": "exact" if relative_l2 is None else (
                "legacy_relative_l2" if max_rel_l2 is not None else "elementwise_and_l2"
            ),
        },
    )


def _compare_value_trees(
    reference,
    candidate,
    *,
    name,
    atol,
    rtol,
    max_rel_l2=None,
    strict_types=False,
    output_tolerances=None,
    tolerance_name=None,
    matched_tolerance_paths=None,
):
    """Compare return values, with exact types for explicit mutation contracts."""
    try:
        _validate_output_structures_match(
            reference, candidate, path=name, strict_types=strict_types
        )
    except ValueError as error:
        return [OutputDiff(name=name, passed=False, error=str(error))]
    logical_name = name if tolerance_name is None else tolerance_name
    if isinstance(reference, dict):
        return [
            diff
            for key in reference
            for diff in _compare_value_trees(
                reference[key],
                candidate[key],
                name=f"{name}.{key}",
                atol=atol,
                rtol=rtol,
                max_rel_l2=max_rel_l2,
                strict_types=strict_types,
                output_tolerances=output_tolerances,
                tolerance_name=f"{logical_name}.{key}",
                matched_tolerance_paths=matched_tolerance_paths,
            )
        ]
    if isinstance(reference, (list, tuple)):
        return [
            diff
            for index, (left, right) in enumerate(zip(reference, candidate))
            for diff in _compare_value_trees(
                left,
                right,
                name=f"{name}[{index}]",
                atol=atol,
                rtol=rtol,
                max_rel_l2=max_rel_l2,
                strict_types=strict_types,
                output_tolerances=output_tolerances,
                tolerance_name=f"{logical_name}[{index}]",
                matched_tolerance_paths=matched_tolerance_paths,
            )
        ]
    if reference is None:
        return [OutputDiff(name=name, passed=True)]
    # Retain legacy scalar conversion and tolerance semantics as well as
    # list/tuple interoperability for benchmarks without an explicit contract.
    reference_tensor = flatten_outputs(reference)[0][1]
    candidate_tensor = flatten_outputs(candidate)[0][1]
    policies = output_tolerances or {}
    tolerance = policies.get(logical_name, policies.get("default"))
    if logical_name in policies and matched_tolerance_paths is not None:
        matched_tolerance_paths.add(logical_name)
    return [
        _compare_output_tensors(
            reference_tensor,
            candidate_tensor,
            name=name,
            atol=tolerance.atol if tolerance is not None else atol,
            rtol=tolerance.rtol if tolerance is not None else rtol,
            max_rel_l2=max_rel_l2,
            strict_dtype=strict_types,
            max_mismatch_fraction=tolerance.max_mismatch_fraction if tolerance else 0.0,
            max_rel_l2_bound=tolerance.max_rel_l2 if tolerance else None,
        )
    ]


def _load_benchmark_contract(reference_path: Path) -> tuple[Path, dict[str, object]]:
    """Load the evaluator-only benchmark contract next to a reference."""

    metadata_path = reference_path.parent / "metadata.json"
    if not metadata_path.is_file():
        return metadata_path, {}
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    contract = payload.get("benchmark_contract") or {}
    if not isinstance(contract, dict):
        raise TypeError(f"{metadata_path}.benchmark_contract must be an object")
    return metadata_path, contract


def _load_output_tolerance_contract(reference_path: Path) -> dict[str, OutputTolerance]:
    """Load optional per-output allclose tolerances from metadata."""

    metadata_path, contract = _load_benchmark_contract(reference_path)
    raw_tolerances = contract.get("correctness_tolerances")
    if raw_tolerances is None:
        return {}
    if not isinstance(raw_tolerances, dict) or not raw_tolerances:
        raise TypeError(
            f"{metadata_path}.benchmark_contract.correctness_tolerances "
            "must be a non-empty object"
        )

    tolerances: dict[str, OutputTolerance] = {}
    for raw_path, raw_policy in raw_tolerances.items():
        is_output_path = isinstance(raw_path, str) and (
            raw_path in ("default", "output")
            or raw_path.startswith("output[")
            or raw_path.startswith("output.")
        )
        is_mutation_path = isinstance(raw_path, str) and raw_path.startswith(
            "mutated_inputs."
        )
        if not (is_output_path or is_mutation_path):
            raise ValueError(
                "correctness_tolerances keys must be logical output or "
                f"declared-mutation paths: {raw_path!r}"
            )
        if not isinstance(raw_policy, dict):
            raise TypeError(f"correctness_tolerances[{raw_path!r}] must be an object")
        unknown = sorted(set(raw_policy) - {"atol", "rtol", "max_mismatch_fraction", "max_rel_l2"})
        if unknown:
            raise ValueError(
                f"correctness_tolerances[{raw_path!r}] has unknown fields: {unknown}"
            )
        if "atol" not in raw_policy or "rtol" not in raw_policy:
            raise ValueError(f"correctness_tolerances[{raw_path!r}] requires atol and rtol")
        tensor_atol = float(raw_policy["atol"])
        tensor_rtol = float(raw_policy["rtol"])
        if (
            not math.isfinite(tensor_atol)
            or not math.isfinite(tensor_rtol)
            or tensor_atol < 0
            or tensor_rtol < 0
        ):
            raise ValueError(
                f"correctness_tolerances[{raw_path!r}] must contain finite, "
                "non-negative atol and rtol"
            )
        fraction = float(raw_policy.get("max_mismatch_fraction", 0.0))
        l2 = raw_policy.get("max_rel_l2")
        l2 = float(l2) if l2 is not None else None
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("max_mismatch_fraction must be finite and between 0 and 1")
        if l2 is not None and (not math.isfinite(l2) or l2 < 0.0):
            raise ValueError("max_rel_l2 must be finite and non-negative")
        tolerances[raw_path] = OutputTolerance(tensor_atol, tensor_rtol, fraction, l2)
    return tolerances


def metadata_owns_correctness(reference_path: Path) -> bool:
    """Return whether metadata declares authoritative per-output tolerances."""

    return bool(_load_output_tolerance_contract(reference_path))


def load_minimum_correctness_cases(reference_path: Path) -> int:
    """Return the optional metadata-owned minimum random coverage per shape."""

    metadata_path, contract = _load_benchmark_contract(reference_path)
    raw_value = contract.get("correctness_min_cases", 1)
    if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value < 1:
        raise ValueError(
            f"{metadata_path}.benchmark_contract.correctness_min_cases "
            "must be a positive integer"
        )
    return raw_value


def _check_unchanged(before, after, *, name):
    """Check full backing bytes (including pool tails) without float tolerances."""
    if type(before) is not type(after):
        return [OutputDiff(name=name, passed=False, error="Input type changed")]
    if isinstance(before, torch.Tensor):
        same = (
            before.dtype == after.dtype
            and before.shape == after.shape
            and before.stride() == after.stride()
            and before.storage_offset() == after.storage_offset()
        )

        def raw(tensor):
            storage = tensor.untyped_storage()
            return torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
                storage, 0, (storage.nbytes(),), (1,)
            )

        same = same and torch.equal(raw(before), raw(after))
        return [
            OutputDiff(
                name=name, passed=bool(same), error=None if same else "Undeclared input mutation"
            )
        ]
    if isinstance(before, dict):
        if before.keys() != after.keys():
            return [OutputDiff(name=name, passed=False, error="Input keys changed")]
        return [
            diff
            for key in before
            for diff in _check_unchanged(before[key], after[key], name=f"{name}.{key}")
        ]
    if isinstance(before, (tuple, list)):
        if len(before) != len(after):
            return [OutputDiff(name=name, passed=False, error="Input length changed")]
        return [
            diff
            for index, (left, right) in enumerate(zip(before, after))
            for diff in _check_unchanged(left, right, name=f"{name}[{index}]")
        ]
    same = before == after
    return [OutputDiff(name=name, passed=bool(same), error=None if same else "Input value changed")]


def check_correctness(
    reference_path: Path,
    candidate_path: Path,
    *,
    shape_id: str = "0",
    atol: float = 1e-2,
    rtol: float = 0.05,
    num_correctness_cases: int = 1,
    device: str = "auto",
    artifact_dir: Path | None = None,
    artifact_root: Path | None = None,
    candidate_timeout_s: int | float | None = _DEFAULT_CANDIDATE_TIMEOUT_S,
    max_rel_l2: float | None = None,
    untrusted_mode: bool = False,
) -> CorrectnessShapeResult:
    """Compare candidate outputs against the eager reference baseline for one shape.

    ``shape_id`` selects the entry from ``shapes.json`` next to ``reference_path``.
    Synthetic inline references without a sibling shapes.json fall back to the
    legacy ``get_inputs()`` / ``get_init_inputs()`` path; ``shape_id`` is then
    informational only.
    """
    if num_correctness_cases < 1:
        return CorrectnessShapeResult(
            status="failed",
            reason="num_correctness_cases must be at least 1",
        )
    try:
        # The CLI resolves this floor before sizing worker budgets and writing
        # runner_config. Keep the same guard here for direct library callers.
        effective_num_correctness_cases = max(
            num_correctness_cases,
            load_minimum_correctness_cases(reference_path),
        )
        effective_max_rel_l2 = configured_max_rel_l2(max_rel_l2)
    except (OSError, TypeError, ValueError) as error:
        return CorrectnessShapeResult(status="failed", reason=str(error))

    try:
        resolved_device = get_device(device)
        loaded_models = instantiate_model_pair(
            reference_path,
            candidate_path,
            resolved_device,
            module_prefix="atrex_correctness",
            shape_id=shape_id,
            candidate_timeout_s=candidate_timeout_s,
        )
        shape: ShapeSpec | None
        if (reference_path.parent / "shapes.json").is_file():
            shape = load_shape_spec(reference_path, shape_id)
        else:
            shape = None
        _, contract = _load_benchmark_contract(reference_path)
        mutations = contract.get("mutates_inputs", [])
        preserve_storage = contract.get('preserve_input_storage', False)
        scratch_inputs = contract.get("scratch_inputs", [])
        strict_types = "mutates_inputs" in contract
        if not isinstance(mutations, list) or not all(isinstance(x, str) for x in mutations):
            raise ValueError("benchmark_contract.mutates_inputs must be a list of input names")
        if not isinstance(scratch_inputs, list) or not all(
            isinstance(x, str) for x in scratch_inputs
        ):
            raise ValueError("benchmark_contract.scratch_inputs must be a list of input names")
        if set(mutations) & set(scratch_inputs):
            raise ValueError(
                "benchmark_contract mutates_inputs and scratch_inputs must not overlap"
            )
        output_tolerances = _load_output_tolerance_contract(reference_path)
        if output_tolerances:
            # Per-output metadata is authoritative and must not be replaced by
            # a legacy process-wide relative-L2 threshold.
            effective_max_rel_l2 = None
        prepare = getattr(loaded_models.input_module, 'prepare_model', None)
        signature = inspect.signature(loaded_models.reference_model.forward) if not callable(prepare) else None
    except Exception:
        return CorrectnessShapeResult(
            status="failed",
            reason=traceback.format_exc(),
        )

    case_records: list[CorrectnessCase] = []
    failed_cases = 0
    # Surfaced into CorrectnessShapeResult.reason for the deterministic-failure
    # paths, so the shape-level summary (which is what bubbles up to the
    # eval_result.json passed.correctness.<id>.reason field) says *why* the
    # shape failed, not just "X/N cases failed".
    early_abort_reason: str | None = None

    def _abort_remaining_cases(after_case_index: int, short_reason: str) -> None:
        """Append 'skipped' CorrectnessCase entries for every case after this one.

        Used for *non-accuracy* failures only — timeouts, exceptions during the
        candidate call, structural / shape / count mismatches. All of these are
        deterministic w.r.t. the candidate (same code + same inputs class -> same
        failure), so re-running with a fresh random draw of inputs is pure
        wasted wall-clock. Accuracy diffs (atol/rtol failures) deliberately do
        NOT trigger early-abort: those CAN be input-dependent and we want full
        per-case coverage to surface the diff distribution.

        ``short_reason`` is ALSO bubbled up into the shape-level
        ``CorrectnessShapeResult.reason`` so the eval_result.json
        ``passed.correctness.<id>.reason`` field tells you *why* the shape
        failed instead of just "X/N cases failed".
        """
        nonlocal failed_cases, early_abort_reason
        early_abort_reason = short_reason
        skipped = effective_num_correctness_cases - (after_case_index + 1)
        if skipped <= 0:
            return
        failed_cases += skipped
        reason = f"skipped after case {after_case_index} failed deterministically: {short_reason}"
        for _ in range(skipped):
            case_records.append(CorrectnessCase(input_artifact=None, error=reason))

    for case_index in range(effective_num_correctness_cases):
        # Seed every RNG just before generating inputs so the random tensors
        # are reproducible from the recorded seed alone (no .pt files needed).
        seed = deterministic_input_seed("correctness", shape_id, case_index)
        seeds = contract.get('correctness_seeds')
        if seeds is not None:
            if len(seeds) < effective_num_correctness_cases or not all(type(s) is int for s in seeds):
                raise ValueError('correctness_seeds must supply every configured case')
            seed = seeds[case_index]
        seed_all_input_rngs(seed)
        if shape is not None:
            inputs = load_shape_call_inputs(loaded_models.input_module, shape, resolved_device)
        else:
            inputs = load_reference_inputs(loaded_models.input_module, resolved_device)
        artifact = {"seed": seed, "format": "manual_seed"}

        try:
            reference_call_inputs = inputs if preserve_storage else clone_model_inputs(inputs)
            candidate_call_inputs = inputs if preserve_storage else clone_model_inputs(inputs)

            reference_model = loaded_models.reference_model
            candidate_model = loaded_models.candidate_model
            reference_arguments = reference_call_inputs
            candidate_arguments = candidate_call_inputs
            if callable(prepare):
                reference_model, arguments = prepare(loaded_models.reference_module.Model,
                    reference_call_inputs.kwargs, shape.init_kwargs, shape.input_kwargs)
                reference_arguments = normalize_model_inputs(arguments)
                with candidate_timeout(candidate_timeout_s):
                    candidate_model, arguments = prepare(loaded_models.candidate_module.Model,
                        candidate_call_inputs.kwargs, shape.init_kwargs, shape.input_kwargs)
                    candidate_arguments = normalize_model_inputs(arguments)

            def named(call):
                return signature.bind(*call.args, **call.kwargs).arguments if signature is not None else call.kwargs

            original_named = named(inputs)
            snapshot = None
            if preserve_storage:
                storage_views = getattr(loaded_models.input_module, 'storage_views', None)
                writable_views = getattr(loaded_models.input_module, 'writable_views', None)
                values = (lambda: storage_views(inputs.kwargs)) if callable(storage_views) else inputs.kwargs
                writable = writable_views(inputs.kwargs) if callable(writable_views) else {k: original_named[k] for k in mutations + scratch_inputs}
                snapshot = StorageSnapshot(values, writable)
            if set(mutations) - original_named.keys():
                raise ValueError(
                    f"Unknown mutated input names: {set(mutations) - original_named.keys()}"
                )
            if set(scratch_inputs) - original_named.keys():
                raise ValueError(
                    f"Unknown scratch input names: {set(scratch_inputs) - original_named.keys()}"
                )
            with torch.inference_mode():
                # Reference is the golden implementation; we trust it and
                # never time it out. The candidate is the AI-generated code
                # and may hang in JIT compile or run a pathological kernel,
                # so we wrap its call (plus the trailing GPU sync so the
                # alarm fires while we are still inside the scope).
                reference_output = (reference_model.forward if callable(prepare) else reference_model)(
                    *reference_arguments.args,
                    **reference_arguments.kwargs,
                )
                reference_aliases = output_aliases(reference_output, reference_call_inputs.kwargs) if preserve_storage else None
                if snapshot is not None:
                    snapshot.check()
                    observe = getattr(loaded_models.input_module, 'observable_inputs', None)
                    observed = observe(inputs.kwargs) if callable(observe) else original_named
                    reference_state = {key: clone_output_values(observed[key]) for key in mutations}
                    reference_output = clone_output_values(reference_output)
                    snapshot.restore()
                try:
                    with candidate_timeout(candidate_timeout_s):
                        candidate_output = (candidate_model.forward if callable(prepare) else candidate_model)(
                            *candidate_arguments.args,
                            **candidate_arguments.kwargs,
                        )
                        sync_device(resolved_device)
                        if snapshot is not None:
                            snapshot.check()
                except CandidateTimeoutError as timeout_error:
                    failed_cases += 1
                    case_records.append(
                        CorrectnessCase(
                            input_artifact=artifact,
                            error=str(timeout_error),
                        )
                    )
                    _abort_remaining_cases(case_index, f"{candidate_timeout_s}s timeout")
                    break
                if untrusted_mode:
                    try:
                        check_plain_tensor_outputs(candidate_output)
                    except RewardHackDetected as reward_hack:
                        failed_cases += 1
                        case_records.append(
                            CorrectnessCase(
                                input_artifact=artifact,
                                error=str(reward_hack),
                            )
                        )
                        _abort_remaining_cases(case_index, "untrusted output guard failed")
                        break

            try:
                _validate_output_structures_match(
                    reference_output, candidate_output, strict_types=strict_types
                )
            except ValueError as structure_error:
                failed_cases += 1
                case_records.append(
                    CorrectnessCase(
                        input_artifact=artifact,
                        error=str(structure_error),
                    )
                )
                _abort_remaining_cases(case_index, "output structure mismatch")
                break

            matched_tolerance_paths: set[str] = set()
            return_tolerances = dict(output_tolerances)
            # A returned caller-owned tensor is the same logical output as the
            # declared mutation. Reuse its policy for exact identity aliases;
            # explicit return-path policies take precedence. Views do not inherit.
            for path, aliases in (reference_aliases or {}).items():
                inherited = {
                    output_tolerances["mutated_inputs" + key]
                    for key, identical, _ in aliases
                    if identical and "mutated_inputs" + key in output_tolerances
                }
                if len(inherited) == 1:
                    return_tolerances.setdefault("output" + path, inherited.pop())
                elif len(inherited) > 1:
                    raise ValueError("Conflicting tolerance policies for identical input aliases")
            output_diffs = _compare_value_trees(
                reference_output,
                candidate_output,
                name="output",
                atol=atol,
                rtol=rtol,
                max_rel_l2=effective_max_rel_l2,
                strict_types=strict_types,
                output_tolerances=return_tolerances,
                matched_tolerance_paths=matched_tolerance_paths,
            )
            output_diffs = [
                replace(diff, name=_flatten_output_name(diff.name)) for diff in output_diffs
            ]
            reference_named = reference_state if snapshot is not None else named(reference_call_inputs)
            candidate_named = named(candidate_call_inputs)
            if snapshot is not None and callable(observe):
                candidate_named = observe(candidate_call_inputs.kwargs)
            mutation_diffs = []
            unexpected_diffs = []
            if preserve_storage:
                alias_match = reference_aliases == output_aliases(candidate_output, candidate_call_inputs.kwargs)
                unexpected_diffs = [OutputDiff(name='return_aliases', passed=alias_match,
                    error=None if alias_match else 'Returned-to-input alias mismatch')]
            for key, before in original_named.items():
                if key in mutations:
                    mutation_diffs.extend(
                        _compare_value_trees(
                            reference_named[key],
                            candidate_named[key],
                            name=f"input.{key}",
                            atol=atol,
                            rtol=rtol,
                            max_rel_l2=effective_max_rel_l2,
                            strict_types=True,
                            output_tolerances=output_tolerances,
                            tolerance_name=f"mutated_inputs.{key}",
                            matched_tolerance_paths=matched_tolerance_paths,
                        )
                    )
                elif key in scratch_inputs:
                    continue
                elif snapshot is not None:
                    # The snapshot checks full backing bytes, including aliases
                    # and holes in declared writable tensor views.
                    continue
                else:
                    for role, state in (
                        ("reference", reference_named),
                        ("candidate", candidate_named),
                    ):
                        unexpected_diffs.extend(
                            _check_unchanged(before, state[key], name=f"{role}.input.{key}")
                        )
            unmatched_tolerance_paths = sorted(
                set(output_tolerances) - {"default"} - matched_tolerance_paths
            )
            if unmatched_tolerance_paths:
                raise ValueError(
                    "correctness_tolerances paths did not match compared values: "
                    + ", ".join(unmatched_tolerance_paths)
                )
            all_diffs = output_diffs + mutation_diffs + unexpected_diffs
            case_passed = all(diff.passed for diff in all_diffs)
            has_structural_failure = any(diff.error is not None for diff in all_diffs)

            if not case_passed:
                failed_cases += 1

            case_records.append(
                CorrectnessCase(
                    input_artifact=artifact,
                    outputs=output_diffs,
                    mutated_inputs=mutation_diffs,
                    unexpected_mutations=unexpected_diffs,
                )
            )

            if has_structural_failure:
                _abort_remaining_cases(case_index, "per-tensor shape/structural mismatch")
                break
        except Exception:
            # Any other exception in the case body (e.g. candidate raised an
            # OOM, kernel launch error, AttributeError on .Model, etc.) is
            # also deterministic w.r.t. the candidate. Record this case as
            # failed AND abort the remaining cases.
            failed_cases += 1
            tb = traceback.format_exc()
            case_records.append(
                CorrectnessCase(
                    input_artifact=artifact,
                    error=tb,
                )
            )
            _abort_remaining_cases(case_index, "candidate raised exception")
            break
        finally:
            # Do not retain the previous case's large native allocations while
            # constructing the next case. Recorded results contain scalars only.
            inputs = reference_call_inputs = candidate_call_inputs = None
            reference_arguments = candidate_arguments = None
            reference_output = candidate_output = None
            original_named = reference_named = candidate_named = None
            reference_state = snapshot = values = writable = observed = None
            reference_model = candidate_model = None

    if failed_cases == 0:
        status = "passed"
        reason: str | None = None
    else:
        status = "failed"
        base = f"{failed_cases}/{effective_num_correctness_cases} correctness cases failed"
        if early_abort_reason is not None:
            reason = f"{base}: {early_abort_reason}"
        else:
            reason = base

    return CorrectnessShapeResult(
        status=status,
        reason=reason,
        cases=case_records,
    )
