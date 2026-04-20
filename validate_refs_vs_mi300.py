"""Validate the ideal-TF32 references against MI300 hipBLASLt hardware.

Three checks:

  1. The MMA-Sim pathological dot-product example (arXiv:2511.10909 §VI-C):
     a = [2048, 2048], b = [2048, -2048], c = -1e-6.
     Paper says hardware returns -0.25, simulator should match.

  2. Random GEMM (M=N=K=64) — confirm:
       * MI300 FP32 (TF32 off) is essentially exact vs FP64.
       * MI300 TF32 (FAST_TF32 on) error vs FP64 is in the same order of
         magnitude as ideal-NVIDIA-TF32 vs FP64 and ideal-AMD-XF32 vs FP64,
         AND has a NEGATIVE mean signed error (the RD bias signature).

  3. Random GEMM at several K values to sketch the scaling of error with K.

Run:
    cd /tmp && HIP_VISIBLE_DEVICES=7 python /var/lib/jenkins/pytorch/agent_space/tf32_analysis/validate_refs_vs_mi300.py
"""

import os
import sys

import torch

sys.path.insert(0, "/var/lib/jenkins/pytorch/agent_space/tf32_analysis")

from e8m10 import e8m10_round_rd, e8m10_round_rne
from tf32_gemm_ref import (
    error_stats,
    gemm_fp64,
    gemm_ideal_amd_xf32,
    gemm_ideal_nvidia_tf32,
    theoretical_e8m10_floor,
)


def banner(s: str) -> None:
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def gpu_gemm(A: torch.Tensor, B: torch.Tensor, *, allow_tf32: bool) -> torch.Tensor:
    """Run A @ B on the current CUDA device with the requested TF32 setting."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    try:
        out = (A.to("cuda") @ B.to("cuda")).to("cpu")
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return out


def case_pathological() -> None:
    banner("Case 1: MMA-Sim pathological dot product (paper §VI-C)")
    # 1xK x Kx1 dot product: a = [2048, 2048], b = [2048, -2048].
    # True result a @ b = 0 exactly (the two products cancel). Adding
    # c = -1e-6 gives -1e-6. The paper claims AMD MFMA returns -0.25 due
    # to round-down accumulation aligning -1e-6 to a high-magnitude
    # accumulator and rounding it down to -0.25.
    A = torch.tensor([[2048.0, 2048.0]], dtype=torch.float32)
    B = torch.tensor([[2048.0], [-2048.0]], dtype=torch.float32)
    C = torch.tensor([[-1e-6]], dtype=torch.float32)

    # FP64 ground truth: A @ B + C = -1e-6
    true_val = (A.double() @ B.double() + C.double()).item()
    print(f"  true (FP64)          = {true_val:+.6e}")

    # MI300 FP32 (no TF32): should be essentially exact.
    fp32 = gpu_gemm(A, B, allow_tf32=False) + C
    print(f"  MI300 FP32           = {fp32.item():+.6e}")

    # MI300 TF32 (FAST_TF32): paper says -0.25 from HW MFMA. Note: this is
    # for the *fused* MFMA op; PyTorch's matmul calls hipBLASLt which may
    # or may not exhibit the same behavior depending on tiling. Report
    # the raw value and let the analysis interpret.
    tf32 = gpu_gemm(A, B, allow_tf32=True) + C
    print(f"  MI300 TF32           = {tf32.item():+.6e}")

    # Ideal references (input rounding only, FP32 accumulate).
    ref_n = gemm_ideal_nvidia_tf32(A, B) + C
    ref_a = gemm_ideal_amd_xf32(A, B) + C
    print(f"  ideal-NVIDIA-TF32    = {ref_n.item():+.6e}")
    print(f"  ideal-AMD-XF32       = {ref_a.item():+.6e}")

    # 2048 is exactly representable in E8M10 (2^11 with zero mantissa),
    # so input rounding doesn't change A or B. The simple "input-round +
    # FP32 accumulate" model thus produces the exact answer -1e-6 for both
    # ideals — it does NOT capture the FDRDA accumulator bias. That's a
    # known limitation: this case requires a bit-accurate FDRDA simulator
    # to model. We document the gap here.
    print()
    print("  NOTE: This case requires a bit-accurate FDRDA simulator to fully")
    print("        reproduce the -0.25 result. The simple 'input-round + FP32")
    print("        accumulate' reference understates the AMD bias for cases")
    print("        where the AB sum cancels and C is far smaller in magnitude.")


def case_random_gemm(M: int, N: int, K: int, seed: int = 0) -> None:
    banner(f"Case 2: random GEMM M={M}, N={N}, K={K}")
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(M, K, generator=g, dtype=torch.float32)
    B = torch.randn(K, N, generator=g, dtype=torch.float32)

    truth = gemm_fp64(A, B)
    nv = gemm_ideal_nvidia_tf32(A, B)
    amd = gemm_ideal_amd_xf32(A, B)
    mi_fp32 = gpu_gemm(A, B, allow_tf32=False)
    mi_tf32 = gpu_gemm(A, B, allow_tf32=True)

    floors = theoretical_e8m10_floor(A, B)
    print(f"  K={K}, |A|_inf={A.abs().max().item():.3f}, "
          f"|B|_inf={B.abs().max().item():.3f}")
    print(f"  symmetric_floor = sqrt(K)*2^-10*|A|*|B| = {floors['symmetric_floor']:.4e}")
    print(f"  rd_bias_estimate = K*2^-11*|A|*|B|     = {floors['rd_bias_estimate']:.4e}")
    print()
    for label, x in [
        ("MI300 FP32  vs FP64", error_stats(mi_fp32, truth)),
        ("MI300 TF32  vs FP64", error_stats(mi_tf32, truth)),
        ("ideal-NV-TF32 vs FP64", error_stats(nv, truth)),
        ("ideal-AMD-XF32 vs FP64", error_stats(amd, truth)),
        ("MI300 TF32  vs ideal-NV-TF32", error_stats(mi_tf32, nv)),
        ("MI300 TF32  vs ideal-AMD-XF32", error_stats(mi_tf32, amd)),
    ]:
        print(f"  {label:32s}  "
              f"max_abs={x['max_abs']:.4e}  "
              f"rms={x['rms_abs']:.4e}  "
              f"mean(signed)={x['mean_signed']:+.4e}")


def case_k_scan() -> None:
    banner("Case 3: K-scan to show error scaling")
    print(f"  {'K':>6}  {'mi_tf32_max':>14} {'mi_tf32_mean':>14} "
          f"{'nv_max':>14} {'nv_mean':>14} "
          f"{'amd_max':>14} {'amd_mean':>14}")
    for K in [4, 16, 64, 256, 1024, 4096]:
        g = torch.Generator(device="cpu").manual_seed(K)
        A = torch.randn(128, K, generator=g, dtype=torch.float32)
        B = torch.randn(K, 128, generator=g, dtype=torch.float32)
        truth = gemm_fp64(A, B)
        nv = gemm_ideal_nvidia_tf32(A, B)
        amd = gemm_ideal_amd_xf32(A, B)
        mi = gpu_gemm(A, B, allow_tf32=True)
        e_mi = error_stats(mi, truth)
        e_nv = error_stats(nv, truth)
        e_amd = error_stats(amd, truth)
        print(f"  {K:>6}  "
              f"{e_mi['max_abs']:14.4e} {e_mi['mean_signed']:+14.4e} "
              f"{e_nv['max_abs']:14.4e} {e_nv['mean_signed']:+14.4e} "
              f"{e_amd['max_abs']:14.4e} {e_amd['mean_signed']:+14.4e}")


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA/HIP not available", file=sys.stderr)
        sys.exit(1)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"HIP   : {getattr(torch.version, 'hip', None)}")
    print(f"HIP_VISIBLE_DEVICES = {os.environ.get('HIP_VISIBLE_DEVICES', '<unset>')}")

    case_pathological()
    case_random_gemm(64, 64, 64)
    case_random_gemm(256, 256, 1024)
    case_k_scan()


if __name__ == "__main__":
    main()
