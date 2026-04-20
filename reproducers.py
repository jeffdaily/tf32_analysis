"""Per-test reproducers for the MI300 TF32 numerical analysis.

Each reproducer is a function `repro_<short_name>()` that:
  * builds the same inputs the original test uses (same shapes, same seed
    where possible),
  * runs the underlying op three ways (MI300 TF32, MI300 FP32, CPU FP64
    plus the two ideal-TF32 references when applicable),
  * returns a dict with the upstream test's tolerance, the underlying op,
    and the error stats vs each reference.

The dict schema is:
  {
    "test_id": "test/foo.py::TestClass::test_method",
    "issue": "https://github.com/pytorch/pytorch/issues/NNNNN" or None,
    "op": "matmul" | "addmm" | "conv2d_k1" | "cholesky" | "cdist" |
          "tensordot" | "affine_grid_sample" | "lstm" | "encoder",
    "tol_atol": float,                  # the @tf32_on_and_off arg
    "tol_origin": "atol_override (max)"|"explicit"|"max(self.precision, explicit atol)",
    "shapes": {...},                    # op-shape summary for the report
    "K": int or None,                   # dominant inner reduction
    "errors": {
        "mi300_tf32_vs_fp64":  {max_abs, rms_abs, mean_signed, ...},
        "mi300_fp32_vs_fp64":  {...},
        "ideal_nv_tf32_vs_fp64": {...},
        "ideal_amd_xf32_vs_fp64": {...},
        "mi300_tf32_vs_ideal_nv":  {...},
        "mi300_tf32_vs_ideal_amd": {...},
    },
    "floor": {symmetric_floor, rd_bias_estimate, K},
    "verdict_hint": str,                # short narrative used by the report
  }

Run all reproducers via run_all.py.
"""

from __future__ import annotations

import math
import sys
from contextlib import contextmanager
from typing import Any

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


# ---------------------------------------------------------------------------
# TF32 toggling helper (mirrors common_cuda.tf32_on / tf32_off but standalone).
# ---------------------------------------------------------------------------


@contextmanager
def tf32_mode(enabled: bool):
    """Temporarily toggle torch.backends.cuda.matmul.allow_tf32 and cuDNN."""
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = enabled
        with torch.backends.cudnn.flags(
            enabled=None, benchmark=None, deterministic=None, allow_tf32=enabled
        ):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul


def _seed(s: int = 0) -> torch.Generator:
    g = torch.Generator(device="cpu").manual_seed(s)
    return g


def _make_pair(
    A_shape, B_shape, *, seed=0, dtype=torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    g = _seed(seed)
    A = torch.randn(*A_shape, generator=g, dtype=dtype)
    B = torch.randn(*B_shape, generator=g, dtype=dtype)
    return A, B


# ---------------------------------------------------------------------------
# Common GEMM-style reproducer driver.
# ---------------------------------------------------------------------------


def _run_gemm_op(A: torch.Tensor, B: torch.Tensor) -> dict:
    """Run A @ B six ways and return error stats."""
    truth = gemm_fp64(A, B)
    nv = gemm_ideal_nvidia_tf32(A, B)
    amd = gemm_ideal_amd_xf32(A, B)
    A_d, B_d = A.to("cuda"), B.to("cuda")
    with tf32_mode(False):
        mi_fp32 = (A_d @ B_d).to("cpu")
    with tf32_mode(True):
        mi_tf32 = (A_d @ B_d).to("cpu")
    return {
        "mi300_fp32_vs_fp64": error_stats(mi_fp32, truth),
        "mi300_tf32_vs_fp64": error_stats(mi_tf32, truth),
        "ideal_nv_tf32_vs_fp64": error_stats(nv, truth),
        "ideal_amd_xf32_vs_fp64": error_stats(amd, truth),
        "mi300_tf32_vs_ideal_nv": error_stats(mi_tf32, nv),
        "mi300_tf32_vs_ideal_amd": error_stats(mi_tf32, amd),
    }


def _classify(errors: dict, tol_atol: float) -> str:
    """Heuristic verdict to seed the per-test report section."""
    tf32_err = errors["mi300_tf32_vs_fp64"]["max_abs"]
    nv_err = errors["ideal_nv_tf32_vs_fp64"]["max_abs"]
    amd_err = errors["ideal_amd_xf32_vs_fp64"]["max_abs"]
    if nv_err > tol_atol and amd_err > tol_atol:
        return "tolerance_below_e8m10_floor"
    if tf32_err > 1.5 * amd_err:
        return "mi300_exceeds_amd_xf32_envelope"
    if tf32_err > tol_atol and nv_err <= tol_atol:
        return "mi300_matches_amd_xf32_exceeds_nvidia_tf32"
    if tf32_err <= tol_atol:
        return "passes"
    return "needs_review"


# ===========================================================================
# Reproducers
# ===========================================================================


def repro_addmm_sizes() -> dict:
    """test_linalg.py::test_addmm_sizes — sweeps m,n,k in {0,1,25}x{0,1,10}x{0,1,8}.
    The largest non-trivial shape is m=25,n=10,k=8 -> mm(10x8, 8x25). We run that.
    """
    g = _seed(7)
    n, m, k = 10, 25, 8
    M_bias = torch.randn(n, m, generator=g, dtype=torch.float32)
    A = torch.randn(n, k, generator=g, dtype=torch.float32)
    B = torch.randn(k, m, generator=g, dtype=torch.float32)
    # addmm(M, A, B) = M + A @ B; the GEMM is A @ B.
    errors = _run_gemm_op(A, B)
    floor = theoretical_e8m10_floor(A, B)
    return {
        "test_id": "test/test_linalg.py::TestLinalg::test_addmm_sizes",
        "issue": None,
        "op": "addmm (mm part)",
        "tol_atol": 0.005,
        "tol_origin": "tf32_on_and_off(0.005) + reduced_f32_on_and_off(0.005)",
        "shapes": {"A": list(A.shape), "B": list(B.shape)},
        "K": int(k),
        "errors": errors,
        "floor": floor,
        "verdict_hint": _classify(errors, 0.005),
    }


def repro_compile_kernel_advanced() -> dict:
    """test_cuda.py::test_compile_kernel_advanced — 64x32 @ 32x48 matmul.
    The test compiles a NAIVE CUDA matmul kernel that does NOT use tensor cores
    and compares its output against torch.matmul (which DOES use TF32 when
    allow_tf32=True). So the reference is the naive kernel and the unit-under-
    test is torch.matmul. Both run on GPU.
    """
    g = _seed(11)
    M, K, N = 64, 32, 48
    A = torch.rand(M, K, generator=g, dtype=torch.float32)
    B = torch.rand(K, N, generator=g, dtype=torch.float32)
    errors = _run_gemm_op(A, B)
    floor = theoretical_e8m10_floor(A, B)
    return {
        "test_id": "test/test_cuda.py::TestCompileKernel::test_compile_kernel_advanced",
        "issue": None,
        "op": "matmul (vs naive CUDA kernel)",
        "tol_atol": 0.005,
        "tol_origin": "tf32_on_and_off(0.005)",
        "shapes": {"A": [M, K], "B": [K, N]},
        "K": int(K),
        "errors": errors,
        "floor": floor,
        "verdict_hint": _classify(errors, 0.005),
    }


def repro_broadcast_batched_matmul() -> dict:
    """test_linalg.py::test_broadcast_batched_matmul — random small batched matmul.
    Test uses random.randint(1, 8) for n,m,p dims. Pick a representative case.
    """
    g = _seed(13)
    # Representative n=8, m=8, p=8 with batch dims [3, 2].
    A = torch.randn(3, 2, 8, 8, generator=g, dtype=torch.float32)
    B = torch.randn(3, 2, 8, 8, generator=g, dtype=torch.float32)
    A_d, B_d = A.to("cuda"), B.to("cuda")
    truth = (A.double() @ B.double()).float()
    nv_a = torch.empty_like(truth)
    amd_a = torch.empty_like(truth)
    for i in range(3):
        for j in range(2):
            nv_a[i, j] = gemm_ideal_nvidia_tf32(A[i, j], B[i, j])
            amd_a[i, j] = gemm_ideal_amd_xf32(A[i, j], B[i, j])
    with tf32_mode(False):
        mi_fp32 = (A_d @ B_d).to("cpu")
    with tf32_mode(True):
        mi_tf32 = (A_d @ B_d).to("cpu")
    errors = {
        "mi300_fp32_vs_fp64": error_stats(mi_fp32, truth),
        "mi300_tf32_vs_fp64": error_stats(mi_tf32, truth),
        "ideal_nv_tf32_vs_fp64": error_stats(nv_a, truth),
        "ideal_amd_xf32_vs_fp64": error_stats(amd_a, truth),
        "mi300_tf32_vs_ideal_nv": error_stats(mi_tf32, nv_a),
        "mi300_tf32_vs_ideal_amd": error_stats(mi_tf32, amd_a),
    }
    floor = theoretical_e8m10_floor(A.reshape(-1, 8), B.reshape(-1, 8))
    return {
        "test_id": "test/test_linalg.py::TestLinalg::test_broadcast_batched_matmul",
        "issue": None,
        "op": "batched matmul",
        "tol_atol": 0.001,
        "tol_origin": "tf32_on_and_off(0.001) + reduced_f32_on_and_off(0.001)",
        "shapes": {"A": list(A.shape), "B": list(B.shape)},
        "K": 8,
        "errors": errors,
        "floor": floor,
        "verdict_hint": _classify(errors, 0.001),
    }


def repro_linear_no_bias_common_nn() -> dict:
    """common_nn.py Linear-no-bias module test (issue #155216).
    Module: torch.nn.Linear(10, 8, bias=False); input (4, 10). Reference: CPU.
    """
    g = _seed(155216)
    inp = torch.randn(4, 10, generator=g, dtype=torch.float32)
    weight = torch.randn(8, 10, generator=g, dtype=torch.float32)  # out, in
    # Linear: input @ weight.T
    A = inp
    B = weight.t().contiguous()
    errors = _run_gemm_op(A, B)
    floor = theoretical_e8m10_floor(A, B)
    return {
        "test_id": "common_nn.py:: Linear no-bias module test",
        "issue": "https://github.com/pytorch/pytorch/issues/155216",
        "op": "linear (no bias) = matmul(input, W.T)",
        "tol_atol": 0.005,
        "tol_origin": "tf32_precision=0.005 in module test dict",
        "shapes": {"input": [4, 10], "weight": [8, 10]},
        "K": 10,
        "errors": errors,
        "floor": floor,
        "verdict_hint": _classify(errors, 0.005),
    }


def _slow_conv2d_k1_forward_rounded(x, weight, bias, round_fn):
    """Model slow_conv2d_forward for k=1 conv with per-operand rounding.

    Mirrors aten/src/ATen/native/cuda/ConvolutionMM2d.cu: per-batch GEMM
    with alpha=1, beta=1 accumulating into (initially bias-copied) output.
    Rounds the GEMM operands (input and weight) to E8M10 via round_fn
    before each per-batch GEMM, then accumulates in FP32.
    """
    B, C_in, H, W = x.shape
    C_out = weight.shape[0]
    W_flat = weight.reshape(C_out, C_in)
    y = torch.empty(B, C_out, H, W, dtype=torch.float32)
    if bias is not None:
        y[:] = bias.reshape(1, C_out, 1, 1)
    else:
        y.zero_()
    W_flat_r = round_fn(W_flat)
    for b in range(B):
        x_b = x[b].reshape(C_in, H * W)
        x_b_r = round_fn(x_b)
        y[b] += (W_flat_r @ x_b_r).reshape(C_out, H, W)
    return y


def _slow_conv2d_k1_grad_weight_rounded(grad_output, x, round_fn):
    """Model slow_conv2d_grad_weight for k=1 conv with per-operand rounding.

    Mirrors the CUDA path: per-batch GEMM (t,n) with K=outputH*outputW,
    alpha=1, beta=1 so results accumulate into grad_weight in FP32 across
    batches. Rounds grad_output and input to E8M10 within each per-batch
    GEMM; cross-batch accumulation is plain FP32.
    """
    B, C_out, H, W = grad_output.shape
    C_in = x.shape[1]
    grad_w = torch.zeros(C_out, C_in, dtype=torch.float32)
    for b in range(B):
        x_b = x[b].reshape(C_in, H * W)
        g_b = grad_output[b].reshape(C_out, H * W)
        x_b_r = round_fn(x_b)
        g_b_r = round_fn(g_b)
        grad_w += g_b_r @ x_b_r.T
    return grad_w.reshape(C_out, C_in, 1, 1)


def repro_conv2d_k1() -> dict:
    """test_convolution.py::test_Conv2d_size_1_kernel.
    Forward: Conv2d(3,3,k=1) on (2,3,5,5). With cudnn.flags(enabled=False),
    CUDA conv2d falls back to a non-cuDNN path that uses GEMM (im2col).
    Reference: CPU conv. Tolerance: 0.005 abs.
    """
    torch.manual_seed(20251201)
    x_cpu = torch.randn(2, 3, 5, 5, dtype=torch.float32)
    conv_cpu = torch.nn.Conv2d(3, 3, kernel_size=1)
    y_cpu = conv_cpu(x_cpu)
    grad_y = torch.rand_like(y_cpu)
    y_cpu.backward(grad_y)
    bias_grad_cpu = conv_cpu.bias.grad.data.clone()
    weight_grad_cpu = conv_cpu.weight.grad.data.clone()

    weight_cpu = conv_cpu.weight.data.detach().clone()
    bias_cpu = conv_cpu.bias.data.detach().clone()

    def gpu_run(allow_tf32: bool, dtype_promote_input_double: bool):
        with tf32_mode(allow_tf32):
            with torch.backends.cudnn.flags(enabled=False):
                conv_cuda = torch.nn.Conv2d(3, 3, kernel_size=1).to("cuda")
                conv_cuda.bias.data.copy_(conv_cpu.bias.data)
                conv_cuda.weight.data.copy_(conv_cpu.weight.data)
                if dtype_promote_input_double:
                    x = x_cpu.to("cuda").double()
                    conv_cuda = conv_cuda.double()
                else:
                    x = x_cpu.to("cuda")
                y = conv_cuda(x)
                y.backward(grad_y.to("cuda").to(y.dtype))
        return (
            y.detach().to("cpu").float(),
            conv_cuda.weight.grad.detach().to("cpu").float(),
            conv_cuda.bias.grad.detach().to("cpu").float(),
        )

    y_fp32, wg_fp32, bg_fp32 = gpu_run(False, False)
    y_tf32, wg_tf32, bg_tf32 = gpu_run(True, False)
    y_ref, wg_ref, bg_ref = gpu_run(False, True)  # FP64 ground truth on GPU

    # Ideal NV-TF32 / AMD-XF32 references modelling the per-batch GEMM
    # structure of slow_conv2d_forward / slow_conv2d_grad_weight.
    y_nv = _slow_conv2d_k1_forward_rounded(x_cpu, weight_cpu, bias_cpu, e8m10_round_rne)
    y_amd = _slow_conv2d_k1_forward_rounded(x_cpu, weight_cpu, bias_cpu, e8m10_round_rd)
    wg_nv = _slow_conv2d_k1_grad_weight_rounded(grad_y, x_cpu, e8m10_round_rne)
    wg_amd = _slow_conv2d_k1_grad_weight_rounded(grad_y, x_cpu, e8m10_round_rd)

    return {
        "test_id": "test/nn/test_convolution.py::TestConvolutionNNDeviceType::test_Conv2d_size_1_kernel",
        "issue": None,
        "op": "conv2d k=1 (forward + backward, cudnn disabled -> GEMM path)",
        "tol_atol": 0.005,
        "tol_origin": "tf32_on_and_off(0.005); explicit assertEqual atol=1e-5 dominated by self.precision via max()",
        "shapes": {"input": [2, 3, 5, 5], "weight": [3, 3, 1, 1]},
        "K": 3,  # forward: C_in=3; weight grad: per-batch H*W=25, batch-accumulated in FP32
        "errors": {
            "forward_mi300_fp32_vs_fp64": error_stats(y_fp32, y_ref),
            "forward_mi300_tf32_vs_fp64": error_stats(y_tf32, y_ref),
            "forward_ideal_nv_tf32_vs_fp64": error_stats(y_nv, y_ref),
            "forward_ideal_amd_xf32_vs_fp64": error_stats(y_amd, y_ref),
            "forward_mi300_tf32_vs_ideal_amd": error_stats(y_tf32, y_amd),
            "weight_grad_mi300_fp32_vs_fp64": error_stats(wg_fp32, wg_ref),
            "weight_grad_mi300_tf32_vs_fp64": error_stats(wg_tf32, wg_ref),
            "weight_grad_ideal_nv_tf32_vs_fp64": error_stats(wg_nv, wg_ref),
            "weight_grad_ideal_amd_xf32_vs_fp64": error_stats(wg_amd, wg_ref),
            "weight_grad_mi300_tf32_vs_ideal_amd": error_stats(wg_tf32, wg_amd),
            "bias_grad_mi300_fp32_vs_fp64": error_stats(bg_fp32, bg_ref),
            "bias_grad_mi300_tf32_vs_fp64": error_stats(bg_tf32, bg_ref),
            "forward_mi300_tf32_vs_cpu": error_stats(y_tf32, y_cpu),
            "weight_grad_mi300_tf32_vs_cpu": error_stats(wg_tf32, weight_grad_cpu),
        },
        "floor": {"symmetric_floor": math.sqrt(3) * 2 ** -10, "rd_bias_estimate": 3 * 2 ** -11, "K": 3},
        "verdict_hint": "see per-component errors",
    }


def repro_cdist_large() -> dict:
    """test_torch.py::test_cdist_large — cdist on (1000, 10) tensors.
    Three compute modes; the matmul-based modes can produce very different
    numerics from the brute-force mode. Tolerance: 0.005 abs.
    """
    g = _seed(20)
    N, D = 1000, 10
    x = torch.randn(N, D, generator=g, dtype=torch.float32)
    y = torch.randn(N, D, generator=g, dtype=torch.float32)
    x_d, y_d = x.to("cuda"), y.to("cuda")

    truth = torch.cdist(x.double(), y.double(), p=2,
                        compute_mode="donot_use_mm_for_euclid_dist").float()
    out = {}
    for mode in ("use_mm_for_euclid_dist_if_necessary",
                 "use_mm_for_euclid_dist",
                 "donot_use_mm_for_euclid_dist"):
        with tf32_mode(False):
            v_fp32 = torch.cdist(x_d, y_d, p=2, compute_mode=mode).to("cpu")
        with tf32_mode(True):
            v_tf32 = torch.cdist(x_d, y_d, p=2, compute_mode=mode).to("cpu")
        out[f"{mode}_fp32_vs_fp64"] = error_stats(v_fp32, truth)
        out[f"{mode}_tf32_vs_fp64"] = error_stats(v_tf32, truth)
    floor = {"symmetric_floor": math.sqrt(D) * 2 ** -10 * 4, "K": D}
    return {
        "test_id": "test/test_torch.py::TestTorchDeviceType::test_cdist_large",
        "issue": None,
        "op": "cdist p=2 (three compute modes)",
        "tol_atol": 0.005,
        "tol_origin": "tf32_on_and_off(0.005) + reduced_f32_on_and_off(0.08)",
        "shapes": {"x": [N, D], "y": [N, D]},
        "K": D,
        "errors": out,
        "floor": floor,
        "verdict_hint": "see per-mode errors; use_mm modes use TF32 GEMM",
    }


def repro_tensordot() -> dict:
    """test_linalg.py::test_tensordot — two main contractions.
    First: arange-based, deterministic, small.
    Second: random (2,3,4,5) x (4,5,6,7), dims=2 -> contract last 2 of A with first 2 of B.
    Tolerance: 0.005 abs.
    """
    import numpy as np

    out = {}
    # Deterministic arange case
    a = torch.arange(60.0, dtype=torch.float32).reshape(3, 4, 5)
    b = torch.arange(24.0, dtype=torch.float32).reshape(4, 3, 2)
    a_d, b_d = a.to("cuda"), b.to("cuda")
    truth = torch.from_numpy(np.tensordot(a.numpy(), b.numpy(), axes=([1, 0], [0, 1])))
    with tf32_mode(False):
        v_fp32 = torch.tensordot(a_d, b_d, dims=([1, 0], [0, 1])).to("cpu")
    with tf32_mode(True):
        v_tf32 = torch.tensordot(a_d, b_d, dims=([1, 0], [0, 1])).to("cpu")
    out["arange_fp32_vs_fp64"] = error_stats(v_fp32, truth)
    out["arange_tf32_vs_fp64"] = error_stats(v_tf32, truth)

    # Random case: contracted dim K = 4*5 = 20
    g = _seed(31)
    a = torch.randn(2, 3, 4, 5, generator=g, dtype=torch.float32)
    b = torch.randn(4, 5, 6, 7, generator=g, dtype=torch.float32)
    a_d, b_d = a.to("cuda"), b.to("cuda")
    truth_rand = torch.from_numpy(np.tensordot(a.double().numpy(),
                                                b.double().numpy(),
                                                axes=2)).float()
    with tf32_mode(False):
        v_fp32 = torch.tensordot(a_d, b_d, dims=2).to("cpu")
    with tf32_mode(True):
        v_tf32 = torch.tensordot(a_d, b_d, dims=2).to("cpu")
    out["random_K20_fp32_vs_fp64"] = error_stats(v_fp32, truth_rand)
    out["random_K20_tf32_vs_fp64"] = error_stats(v_tf32, truth_rand)

    return {
        "test_id": "test/test_linalg.py::TestLinalg::test_tensordot",
        "issue": None,
        "op": "tensordot (matmul reduction)",
        "tol_atol": 0.005,
        "tol_origin": "tf32_on_and_off(0.005)",
        "shapes": {"arange": "60 -> (3,4,5) tensordot (4,3,2) -> (5,2)",
                   "random": "(2,3,4,5) tensordot (4,5,6,7) dims=2 -> (2,3,6,7)"},
        "K": 20,
        "errors": out,
        "floor": theoretical_e8m10_floor(a.reshape(-1, 20), b.reshape(20, -1)),
        "verdict_hint": "K=20 contraction; arange has very small values",
    }


def repro_old_cholesky() -> dict:
    """test_linalg.py::test_old_cholesky — A is 10x10 Hermitian PD, dtype=float32.
    Compute C = cholesky(A); check ||A - C @ C.T||. Tolerance: max(0.01, 1e-14)
    = 0.01 in the TF32 branch (assertEqual takes max).
    """
    from torch.testing._internal.common_utils import random_hermitian_pd_matrix

    g = _seed(42)
    torch.manual_seed(42)
    A = random_hermitian_pd_matrix(10, dtype=torch.float32, device="cpu")
    A_d = A.to("cuda")

    out = {}
    with tf32_mode(False):
        C_fp32 = torch.linalg.cholesky(A_d)
        recon_fp32 = (C_fp32 @ C_fp32.transpose(-2, -1).conj()).to("cpu")
    with tf32_mode(True):
        C_tf32 = torch.linalg.cholesky(A_d)
        recon_tf32 = (C_tf32 @ C_tf32.transpose(-2, -1).conj()).to("cpu")
    out["recon_fp32_vs_input"] = error_stats(recon_fp32, A)
    out["recon_tf32_vs_input"] = error_stats(recon_tf32, A)

    # Decompose into "TF32 hits the cholesky kernel itself" vs "TF32 hits the
    # mm reconstruct". Cholesky decomp is done in MAGMA/cuSOLVER and likely
    # does not use TF32 GEMM; the mm reconstruct DOES.
    C_fp32_cpu = C_fp32.to("cpu")
    nv_recon = gemm_ideal_nvidia_tf32(C_fp32_cpu, C_fp32_cpu.t())
    amd_recon = gemm_ideal_amd_xf32(C_fp32_cpu, C_fp32_cpu.t())
    out["recon_ideal_nv_vs_input"] = error_stats(nv_recon, A)
    out["recon_ideal_amd_vs_input"] = error_stats(amd_recon, A)

    cond = torch.linalg.cond(A.double()).item()
    floor = {"symmetric_floor": math.sqrt(10) * 2 ** -10 * cond,
             "rd_bias_estimate": 10 * 2 ** -11 * cond, "K": 10, "cond": cond}
    return {
        "test_id": "test/test_linalg.py::TestLinalg::test_old_cholesky [float32]",
        "issue": None,
        "op": "cholesky reconstruct = mm(C, C.T)",
        "tol_atol": 0.01,
        "tol_origin": "max(self.precision=0.01 from tf32_on_and_off, explicit atol=1e-14)",
        "shapes": {"A": [10, 10]},
        "K": 10,
        "errors": out,
        "floor": floor,
        "verdict_hint": f"recon error scales as cond(A) ~= {cond:.2e}",
    }


def repro_affine_2d_rotate_random() -> dict:
    """test_nn.py::test_affine_2d_rotateRandom — affine_grid + grid_sample.
    The TF32 path enters via the 2x3 affine matrix multiply that builds the
    sampling grid; for output (H, W), the grid construction does
    base_grid (H*W, 3) @ theta.T (3, 2). K=3 is tiny.
    Tolerance: 0.005 abs.
    """
    g = _seed(2025)
    angle = 0.7
    H = W = 64
    theta = torch.tensor([[math.cos(angle), -math.sin(angle), 0.0],
                          [math.sin(angle),  math.cos(angle), 0.0]],
                         dtype=torch.float32).unsqueeze(0)
    inp = torch.randn(1, 1, H, W, generator=g, dtype=torch.float32)

    inp_d = inp.to("cuda")
    theta_d = theta.to("cuda")
    with tf32_mode(False):
        grid_fp32 = torch.nn.functional.affine_grid(theta_d, inp_d.shape, align_corners=True)
        out_fp32 = torch.nn.functional.grid_sample(inp_d, grid_fp32, align_corners=True).to("cpu")
    with tf32_mode(True):
        grid_tf32 = torch.nn.functional.affine_grid(theta_d, inp_d.shape, align_corners=True)
        out_tf32 = torch.nn.functional.grid_sample(inp_d, grid_tf32, align_corners=True).to("cpu")
    grid_ref = torch.nn.functional.affine_grid(
        theta.double().to("cuda"), inp_d.shape, align_corners=True
    )
    out_ref = torch.nn.functional.grid_sample(
        inp_d.double(), grid_ref, align_corners=True
    ).to("cpu").float()

    return {
        "test_id": "test/test_nn.py::TestNNDeviceType::test_affine_2d_rotateRandom",
        "issue": None,
        "op": "affine_grid + grid_sample (rotation 64x64)",
        "tol_atol": 0.005,
        "tol_origin": "tf32_on_and_off(0.005) + reduced_f32_on_and_off(0.005)",
        "shapes": {"input": list(inp.shape), "theta": list(theta.shape)},
        "K": 3,
        "errors": {
            "out_mi300_fp32_vs_fp64": error_stats(out_fp32, out_ref),
            "out_mi300_tf32_vs_fp64": error_stats(out_tf32, out_ref),
            "grid_mi300_fp32_vs_fp64": error_stats(grid_fp32.to("cpu"), grid_ref.to("cpu").float()),
            "grid_mi300_tf32_vs_fp64": error_stats(grid_tf32.to("cpu"), grid_ref.to("cpu").float()),
        },
        "floor": {"symmetric_floor": math.sqrt(3) * 2 ** -10, "K": 3},
        "verdict_hint": "K=3 affine GEMM is tiny; bilinear interp dominates",
    }


def repro_lstm_short() -> dict:
    """test_nn.py::test_variable_sequence — LSTM forward + backward.
    Uses LSTM(input_size=3, hidden_size=4, num_layers=1). Input (seq=7, batch=2, 3).
    Each timestep has K = input_size + hidden_size = 7 GEMMs feeding gates.
    Tolerance: 0.005 abs.
    """
    torch.manual_seed(123)
    seq, batch, inp_sz, hid_sz = 7, 2, 3, 4
    inp = torch.randn(seq, batch, inp_sz, dtype=torch.float32)

    def run(allow_tf32: bool, dtype_to_double: bool):
        with tf32_mode(allow_tf32):
            torch.manual_seed(456)
            lstm = torch.nn.LSTM(inp_sz, hid_sz, num_layers=1).to("cuda")
            if dtype_to_double:
                lstm = lstm.double()
                x = inp.to("cuda").double()
            else:
                x = inp.to("cuda")
            out, (h, c) = lstm(x)
        return out.detach().to("cpu").float(), h.detach().to("cpu").float()

    o_fp32, h_fp32 = run(False, False)
    o_tf32, h_tf32 = run(True, False)
    o_ref, h_ref = run(False, True)

    return {
        "test_id": "test/test_nn.py::TestNNDeviceType::test_variable_sequence",
        "issue": None,
        "op": "LSTM (forward, seq=7)",
        "tol_atol": 0.005,
        "tol_origin": "tf32_on_and_off(0.005)",
        "shapes": {"input": [seq, batch, inp_sz], "hidden": hid_sz},
        "K": inp_sz + hid_sz,
        "errors": {
            "out_mi300_fp32_vs_fp64": error_stats(o_fp32, o_ref),
            "out_mi300_tf32_vs_fp64": error_stats(o_tf32, o_ref),
            "h_mi300_fp32_vs_fp64": error_stats(h_fp32, h_ref),
            "h_mi300_tf32_vs_fp64": error_stats(h_tf32, h_ref),
        },
        "floor": {"symmetric_floor": math.sqrt(7) * 2 ** -10, "K": 7,
                  "note": "compounds across seq=7 timesteps"},
        "verdict_hint": "compounding error across timesteps; K=7 per gate GEMM",
    }


def repro_transformer_encoder_fastpath() -> dict:
    """test_transformers.py::test_transformerencoder_fastpath.
    Compare fastpath (BetterTransformer fused kernel) vs slowpath (Python
    forward) for a small encoder. Build the encoder ONCE, then pin the
    same weights and same input across all four runs. Tolerance: 0.001 abs.
    """
    torch.manual_seed(7777)
    d_model = 12
    nhead = 4
    bsz = 3
    seq_len = 5

    # Build the encoder once on CPU with deterministic weights.
    layer_proto = torch.nn.TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=24, batch_first=True,
        dropout=0.0,
    )
    enc_proto = torch.nn.TransformerEncoder(layer_proto, num_layers=1)
    enc_proto.eval()
    state = {k: v.detach().clone() for k, v in enc_proto.state_dict().items()}
    inp_cpu = torch.randn(bsz, seq_len, d_model, dtype=torch.float32)

    def make_enc(double=False, eval_mode=True):
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=24, batch_first=True,
            dropout=0.0,
        )
        e = torch.nn.TransformerEncoder(layer, num_layers=1)
        e.load_state_dict(state)
        e = e.to("cuda")
        if double:
            e = e.double()
        if eval_mode:
            e.eval()
        else:
            e.train()
        return e

    def run(allow_tf32, fastpath, double=False):
        e = make_enc(double=double, eval_mode=fastpath)
        with tf32_mode(allow_tf32):
            x = inp_cpu.to("cuda")
            if double:
                x = x.double()
            with torch.no_grad():
                y = e(x)
        return y.detach().to("cpu").float()

    y_fp32_slow = run(False, fastpath=False)
    y_tf32_slow = run(True, fastpath=False)
    y_fp32_fast = run(False, fastpath=True)
    y_tf32_fast = run(True, fastpath=True)
    y_ref = run(False, fastpath=False, double=True)

    return {
        "test_id": "test/test_transformers.py::TestTransformers::test_transformerencoder_fastpath",
        "issue": None,
        "op": "TransformerEncoder (fastpath vs slowpath; same weights/input)",
        "tol_atol": 0.001,
        "tol_origin": "tf32_on_and_off(0.001)",
        "shapes": {"input": [bsz, seq_len, d_model], "nhead": nhead},
        "K": d_model,
        "errors": {
            "slow_fp32_vs_fp64": error_stats(y_fp32_slow, y_ref),
            "slow_tf32_vs_fp64": error_stats(y_tf32_slow, y_ref),
            "fast_fp32_vs_fp64": error_stats(y_fp32_fast, y_ref),
            "fast_tf32_vs_fp64": error_stats(y_tf32_fast, y_ref),
            "fast_vs_slow_tf32": error_stats(y_tf32_fast, y_tf32_slow),
            "fast_vs_slow_fp32": error_stats(y_fp32_fast, y_fp32_slow),
            "tf32_vs_fp32_slow": error_stats(y_tf32_slow, y_fp32_slow),
            "tf32_vs_fp32_fast": error_stats(y_tf32_fast, y_fp32_fast),
        },
        "floor": {"symmetric_floor": math.sqrt(d_model) * 2 ** -10, "K": d_model},
        "verdict_hint": "compares fastpath vs slowpath under same weights",
    }


def repro_inductor_padding_like() -> dict:
    """test/inductor/test_padding.py — these tests force highest precision on HIP.
    To characterize: run a representative GEMM at 'high' (TF32) precision and
    measure error vs 'highest' (FP32 IEEE). Use shapes the tests typically
    cover (padded conv-style mm), e.g. 1024 x 1024 x 1024 or close.
    """
    g = _seed(8888)
    M = N = K = 1024
    A = torch.randn(M, K, generator=g, dtype=torch.float32)
    B = torch.randn(K, N, generator=g, dtype=torch.float32)
    A_d, B_d = A.to("cuda"), B.to("cuda")

    truth = (A.double() @ B.double()).float()

    prev_prec = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        c_highest = (A_d @ B_d).to("cpu")
        torch.set_float32_matmul_precision("high")
        c_high = (A_d @ B_d).to("cpu")
    finally:
        torch.set_float32_matmul_precision(prev_prec)

    return {
        "test_id": "test/inductor/test_padding.py (representative 1024 GEMM)",
        "issue": None,
        "op": "matmul under highest vs high precision",
        "tol_atol": 1e-3,  # inductor tests use various; ~1e-3 is common
        "tol_origin": "varies; class forces highest on HIP",
        "shapes": {"A": [M, K], "B": [K, N]},
        "K": K,
        "errors": {
            "highest_vs_fp64": error_stats(c_highest, truth),
            "high_vs_fp64":    error_stats(c_high,    truth),
            "high_vs_highest": error_stats(c_high, c_highest),
        },
        "floor": theoretical_e8m10_floor(A, B),
        "verdict_hint": "K=1024; 'high' = TF32 has notably higher error vs 'highest'",
    }


# ---------------------------------------------------------------------------
# Registry of reproducers exposed to run_all.py
# ---------------------------------------------------------------------------

ALL_REPRODUCERS: dict[str, Any] = {
    "addmm_sizes": repro_addmm_sizes,
    "compile_kernel_advanced": repro_compile_kernel_advanced,
    "broadcast_batched_matmul": repro_broadcast_batched_matmul,
    "linear_no_bias_module": repro_linear_no_bias_common_nn,
    "conv2d_k1": repro_conv2d_k1,
    "cdist_large": repro_cdist_large,
    "tensordot": repro_tensordot,
    "old_cholesky": repro_old_cholesky,
    "affine_2d_rotate_random": repro_affine_2d_rotate_random,
    "lstm_short": repro_lstm_short,
    "transformer_encoder_fastpath": repro_transformer_encoder_fastpath,
    "inductor_padding_like": repro_inductor_padding_like,
}
