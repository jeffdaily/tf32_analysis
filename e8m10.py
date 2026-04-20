"""E8M10 rounding primitives for the AMD-XF32 vs NVIDIA-TF32 analysis.

Both vendors' "TF32" matrix instructions use the E8M10 storage format
(1 sign + 8 exponent + 10 mantissa bits in a 32-bit container with
13 trailing mantissa bits dropped from FP32). They differ in HOW
those 13 bits are rounded:

  * NVIDIA Tensor Cores: round-to-nearest-even (RNE) — symmetric error.
  * AMD CDNA3 Matrix Cores (gfx942): the multiply stage uses input
    operands as full FP32 (per the AMD matrix-instruction-calculator,
    xf32 = "FP32 (IEEE binary32 floating point)"), but the FDRDA
    accumulation step rounds toward -inf (round-down, RD) — asymmetric,
    biased toward negative.

Reference: arXiv:2511.10909 "MMA-Sim: Bit-Accurate Reference Model of
Tensor Cores and Matrix Cores", esp. Sections IV-F (FDRDA), IV-G (CoFDRDA),
VI-C (RD-bias example). Per Table IV the CDNA3 instructions
v_mfma_f32_32x32x4_xf32 and v_mfma_f32_16x16x8_xf32 use FDRDA / CoFDRDA.

This module provides scalar-style bit manipulation on FP32 tensors. It is
not a bit-accurate FDRDA simulator (that is in fdrda.py); it provides the
input-rounding primitive that the simulator uses, and a simpler "input
round + FP32 accumulate" reference that matches the standard TF32 textbook
spec (used by gemm_ideal_nvidia_tf32 in tf32_gemm_ref.py).

The 13 dropped bits are the LSBs of the 23-bit FP32 mantissa.
"""

from __future__ import annotations

import torch


# Number of mantissa bits dropped when going from FP32 (23) to E8M10 (10).
_DROPPED_BITS = 13
_DROPPED_MASK = (1 << _DROPPED_BITS) - 1            # 0x1FFF
_KEEP_MASK = (~_DROPPED_MASK) & 0xFFFFFFFF          # 0xFFFFE000
_ROUND_BIT_POS = _DROPPED_BITS - 1                  # 12
_LSB_POS = _DROPPED_BITS                            # 13
_SIGN_MASK = 0x80000000

# FP32 special-value masks.
_EXP_MASK = 0x7F800000
_MANT_MASK = 0x007FFFFF


def _as_int32(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.float32:
        raise TypeError(f"expected float32, got {x.dtype}")
    return x.view(torch.int32)


def _is_nan_or_inf(int_bits: torch.Tensor) -> torch.Tensor:
    """Boolean mask: True where the FP32 bit pattern is NaN or +/-Inf."""
    return (int_bits & _EXP_MASK) == _EXP_MASK


def e8m10_round_rne(x: torch.Tensor) -> torch.Tensor:
    """Round FP32 -> E8M10 with round-to-nearest, ties-to-even (NVIDIA TF32).

    Pure bit manipulation on int32 view. Preserves NaN / Inf bit patterns
    (no rounding applied to specials).
    """
    bits = _as_int32(x).clone()
    specials = _is_nan_or_inf(bits)

    # Extract the dropped low 13 bits and the LSB of the kept mantissa.
    low = bits & _DROPPED_MASK
    truncated = bits & ~torch.tensor(_DROPPED_MASK, dtype=torch.int32)

    round_bit = (low >> _ROUND_BIT_POS) & 1
    sticky = (low & ((1 << _ROUND_BIT_POS) - 1)) != 0
    lsb = (bits >> _LSB_POS) & 1

    # RNE: round up iff round_bit AND (sticky OR lsb).
    round_up = (round_bit.to(torch.bool) & (sticky | lsb.to(torch.bool))).to(torch.int32)

    rounded = truncated + (round_up << _LSB_POS)

    # For NaN/Inf, leave the original bits untouched.
    rounded = torch.where(specials, bits, rounded)
    return rounded.view(torch.float32)


def e8m10_round_rd(x: torch.Tensor) -> torch.Tensor:
    """Round FP32 -> E8M10 with round-toward-minus-infinity (AMD XF32 / RD).

    For positive normals this is truncation (round toward zero == round
    toward -inf). For negative normals this rounds *away* from zero
    (the magnitude grows by one mantissa ULP if any of the dropped bits
    are non-zero).

    NaN / Inf are preserved bit-exact.
    """
    bits = _as_int32(x).clone()
    specials = _is_nan_or_inf(bits)

    low = bits & _DROPPED_MASK
    truncated = bits & ~torch.tensor(_DROPPED_MASK, dtype=torch.int32)

    sign = ((bits >> 31) & 1).to(torch.bool)
    has_low = low != 0

    # If negative and any dropped bits were non-zero, increment the kept
    # mantissa by one ULP (which moves the value toward -inf in IEEE 754
    # sign-magnitude encoding).
    bump = (sign & has_low).to(torch.int32)
    rounded = truncated + (bump << _LSB_POS)

    rounded = torch.where(specials, bits, rounded)
    return rounded.view(torch.float32)


def fp32_bits(x: float) -> int:
    """Convenience: bit pattern of a Python float as a uint32 (for tests)."""
    t = torch.tensor([x], dtype=torch.float32)
    return int(t.view(torch.int32).item()) & 0xFFFFFFFF


def from_fp32_bits(b: int) -> float:
    """Inverse of fp32_bits."""
    t = torch.tensor([b & 0xFFFFFFFF], dtype=torch.int64).to(torch.int32)
    return float(t.view(torch.float32).item())
