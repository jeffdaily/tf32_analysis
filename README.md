# tf32_analysis

Numerical analysis of PyTorch TF32 test failures on AMD MI300 (CDNA3).

This repo captures the measurement harness, CPU bit-accurate reference
models, and the resulting dataset + write-up for a batch of PyTorch unit
tests that fail on MI300 under `@tf32_on_and_off`. The question being
answered: **are the MI300 failures a hipBLASLt accuracy bug, or expected
behavior of the CDNA3 XF32 matrix instructions?**

Short answer: every measured failure is consistent with the documented
round-down semantics of the `v_mfma_f32_*_xf32` instructions (arXiv
2511.10909). The appropriate remediation is tolerance adjustment or a
`"highest"` precision fallback on MI300, not a hipBLASLt fix. See
[REPORT.md](REPORT.md) for the full analysis, numbers, and per-test
verdicts.

## Contents

- `REPORT.md` — full write-up: TL;DR, rounding-semantics background,
  per-test table, and recommended remediations.
- `e8m10.py` — bit-accurate CPU models for NVIDIA TF32 (RNE) and AMD
  XF32 (round-toward-zero / round-down) E8M10 rounding.
- `tf32_gemm_ref.py` — ideal-TF32 and ideal-XF32 GEMM references built
  on top of `e8m10.py`.
- `reproducers.py` — one reproducer per failing PyTorch test; each
  runs MI300 TF32, MI300 FP32, CPU FP64, and the two ideal references,
  and emits max/rms/signed error stats.
- `run_all.py` — driver that runs every reproducer and dumps
  `results.json`.
- `test_e8m10.py` — unit tests for the E8M10 rounding models.
- `validate_refs_vs_mi300.py` — sanity check: how well each CPU
  reference predicts what MI300 actually produces.
- `results.json` — output of the full `run_all.py` sweep on MI300X.
- `repros/compile_kernel_advanced/repro.cpp` — standalone HIP / hipBLASLt
  reproducer for the headline case (no PyTorch dependency), showing the
  same negative-bias signature directly from `hipblasLtMatmul` with
  `HIPBLAS_COMPUTE_32F_FAST_TF32`.

## Running

```bash
# full sweep
python run_all.py

# single reproducer
python run_all.py --only compile_kernel_advanced

# E8M10 model unit tests
python test_e8m10.py

# standalone HIP reproducer
cd repros/compile_kernel_advanced
hipcc -O2 -lhipblaslt repro.cpp -o repro && ./repro
```

Hardware used for the recorded `results.json`: AMD Instinct MI300X HF,
ROCm 7.2.53211, gfx942, PyTorch 2.13.0a0+git665a875 (HIP build).
