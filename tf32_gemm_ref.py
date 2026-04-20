"""Reference GEMM implementations modeling NVIDIA TF32 and AMD XF32.

This module provides three CPU references for an FP32 matmul C = A @ B:

  * gemm_fp64(A, B): the ground-truth FP64 reference.
  * gemm_ideal_nvidia_tf32(A, B): operands rounded to E8M10 with
    round-to-nearest-even, then accumulated in FP32. This matches NVIDIA's
    published TF32 spec (Ampere onward).
  * gemm_ideal_amd_xf32(A, B): operands rounded to E8M10 with round-down
    (toward -inf), then accumulated in FP32. This is the simplest model
    of AMD CDNA3's XF32 path that captures the asymmetric / negatively
    biased rounding documented in arXiv:2511.10909 §IV-F, §VI-C.

The AMD reference is *not* a bit-accurate FDRDA simulator — that would
require reproducing the hardware's exact alignment / accumulator width /
chained-FDRDA structure (CoFDRDA for v_mfma_f32_16x16x8_xf32). It is a
faithful first-order model: same rounding mode applied at the operand
quantization step, which is the dominant error source for moderate K.

If the analysis turns up a test where MI300 hipBLASLt produces different
results than this reference predicts even after accounting for shape /
chunking, the gap will be filed as a hipBLASLt accuracy issue and the
reference can be upgraded to a full bit-accurate FDRDA simulator.
"""

from __future__ import annotations

import torch

from e8m10 import e8m10_round_rd, e8m10_round_rne


def _ensure_cpu_float32(name: str, x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.float32:
        raise TypeError(f"{name}: expected float32, got {x.dtype}")
    return x.detach().to("cpu").contiguous()


def gemm_fp64(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    A = _ensure_cpu_float32("A", A)
    B = _ensure_cpu_float32("B", B)
    return (A.double() @ B.double()).float()


def gemm_ideal_nvidia_tf32(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """A and B rounded to E8M10 (RNE), then FP32 matmul on CPU."""
    A = _ensure_cpu_float32("A", A)
    B = _ensure_cpu_float32("B", B)
    Ar = e8m10_round_rne(A)
    Br = e8m10_round_rne(B)
    return Ar @ Br


def gemm_ideal_amd_xf32(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """A and B rounded to E8M10 (RD = toward -inf), then FP32 matmul on CPU."""
    A = _ensure_cpu_float32("A", A)
    B = _ensure_cpu_float32("B", B)
    Ar = e8m10_round_rd(A)
    Br = e8m10_round_rd(B)
    return Ar @ Br


# ---------------------------------------------------------------------------
# Error metric helpers used by the per-test harness.
# ---------------------------------------------------------------------------


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to("cpu").float().reshape(-1)


def error_stats(actual: torch.Tensor, ref: torch.Tensor) -> dict:
    """Return a dict of max/rms/mean abs/rel error stats vs a reference."""
    a = _flat(actual)
    r = _flat(ref)
    if a.shape != r.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {r.shape}")
    diff = a - r
    abs_diff = diff.abs()
    denom = r.abs().clamp_min(1e-30)
    rel = abs_diff / denom
    return {
        "max_abs": abs_diff.max().item(),
        "rms_abs": (diff.double().pow(2).mean().sqrt()).item(),
        "mean_signed": diff.mean().item(),
        "max_rel": rel.max().item(),
        "ref_max_abs": r.abs().max().item(),
        "actual_max_abs": a.abs().max().item(),
        "shape": tuple(actual.shape),
    }


def theoretical_e8m10_floor(A: torch.Tensor, B: torch.Tensor) -> dict:
    """Two simple per-element error estimates for an MxNxK FP32 GEMM in E8M10:

      symmetric_floor = sqrt(K) * 2^-10 * |A|_inf * |B|_inf
        — the standard random-walk bound for symmetric rounding.
      rd_bias_estimate = K * 2^-11 * |A|_inf * |B|_inf
        — a back-of-envelope estimate of the systematic mean error
          introduced by RD truncation per multiply-accumulate.

    These are *order of magnitude* references for the report, not tight
    bounds. They are computed without hitting the GPU.
    """
    A = _ensure_cpu_float32("A", A)
    B = _ensure_cpu_float32("B", B)
    K = A.shape[-1]
    a_inf = A.abs().max().item()
    b_inf = B.abs().max().item()
    sym = (K ** 0.5) * (2.0 ** -10) * a_inf * b_inf
    rd = K * (2.0 ** -11) * a_inf * b_inf
    return {"symmetric_floor": sym, "rd_bias_estimate": rd, "K": int(K)}
