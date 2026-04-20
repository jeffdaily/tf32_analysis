"""Unit tests for the E8M10 rounding primitives.

Run from any directory other than /var/lib/jenkins/pytorch:
    cd /tmp && python /var/lib/jenkins/pytorch/agent_space/tf32_analysis/test_e8m10.py
"""

import sys
import torch

sys.path.insert(0, "/var/lib/jenkins/pytorch/agent_space/tf32_analysis")
from e8m10 import e8m10_round_rne, e8m10_round_rd, fp32_bits


def _bits(x: torch.Tensor) -> str:
    return format(x.view(torch.int32).item() & 0xFFFFFFFF, "032b")


def _show(label, x):
    print(f"  {label:30s} value={x.item():>20.10g}  bits={_bits(x)}")


def check(name, predicate, *details):
    status = "PASS" if predicate else "FAIL"
    line = f"[{status}] {name}"
    if details:
        line += " :: " + " ".join(str(d) for d in details)
    print(line)
    return predicate


def main():
    failures = 0

    # ------------------------------------------------------------------
    # Test 1: 1.0 + 2^-13 — the round bit is 1 but no sticky/lsb -> tie.
    # RNE rule: round to even. The kept LSB is 0 (1.0 has all-zero
    # mantissa), so RNE rounds DOWN to 1.0. RD on a positive value
    # is also truncation = 1.0.
    # ------------------------------------------------------------------
    x = torch.tensor([1.0 + 2.0 ** -13], dtype=torch.float32)
    rne = e8m10_round_rne(x)
    rd = e8m10_round_rd(x)
    print("Case A: 1.0 + 2^-13 (positive tie at LSB=0 -> round to even)")
    _show("input", x[0])
    _show("RNE", rne[0])
    _show("RD", rd[0])
    failures += not check("RNE: 1.0 + 2^-13 ties down to 1.0", rne.item() == 1.0)
    failures += not check("RD : 1.0 + 2^-13 truncates to 1.0", rd.item() == 1.0)

    # ------------------------------------------------------------------
    # Test 2: 1.0 + 2^-12 — for x in [1, 2) the FP32 mantissa LSB is 2^-23.
    # Dropping 13 bits leaves a kept LSB of 2^-10. Bits with values 2^-23
    # ... 2^-11 are dropped (2^-12 IS in the dropped range, at mantissa bit
    # position 11 < 13). So 1.0 + 2^-12 has only sticky bits set; round bit
    # (position 12) is zero; both RNE and RD truncate to 1.0.
    # ------------------------------------------------------------------
    x = torch.tensor([1.0 + 2.0 ** -12], dtype=torch.float32)
    rne = e8m10_round_rne(x)
    rd = e8m10_round_rd(x)
    print("\nCase B: 1.0 + 2^-12 (sticky-only -> truncate)")
    _show("input", x[0])
    _show("RNE", rne[0])
    _show("RD", rd[0])
    failures += not check("RNE: round bit clear -> truncate to 1.0",
                          rne.item() == 1.0)
    failures += not check("RD : positive truncate to 1.0",
                          rd.item() == 1.0)

    # 2b: 1.0 + 2^-10 IS exactly representable (bit 13 is the kept LSB).
    x = torch.tensor([1.0 + 2.0 ** -10], dtype=torch.float32)
    rne = e8m10_round_rne(x)
    rd = e8m10_round_rd(x)
    print("\nCase B': 1.0 + 2^-10 (exact in E8M10 -> preserved)")
    _show("input", x[0])
    _show("RNE", rne[0])
    _show("RD", rd[0])
    failures += not check("RNE: exact representation preserved",
                          rne.item() == 1.0 + 2.0 ** -10)
    failures += not check("RD : exact representation preserved",
                          rd.item() == 1.0 + 2.0 ** -10)

    # ------------------------------------------------------------------
    # Test 3: 1.0 + 2^-11 — round bit (mantissa pos 12) is set; sticky=0;
    # kept LSB (pos 13) = 0 (even). RNE tie-to-even -> round DOWN to 1.0.
    # RD positive -> truncate to 1.0.
    # ------------------------------------------------------------------
    x = torch.tensor([1.0 + 2.0 ** -11], dtype=torch.float32)
    rne = e8m10_round_rne(x)
    rd = e8m10_round_rd(x)
    print("\nCase C: 1.0 + 2^-11 (exact tie at LSB=0 -> RNE rounds down)")
    _show("input", x[0])
    _show("RNE", rne[0])
    _show("RD", rd[0])
    failures += not check("RNE: tie-to-even -> 1.0", rne.item() == 1.0)
    failures += not check("RD : truncates to 1.0", rd.item() == 1.0)

    # 3b: 1.0 + 2^-10 + 2^-11 — kept LSB=1 (odd), tie. RNE rounds UP
    # (away from zero) to 1.0 + 2^-10 + 2^-10 = 1.0 + 2^-9.
    x = torch.tensor([1.0 + 2.0 ** -10 + 2.0 ** -11], dtype=torch.float32)
    rne = e8m10_round_rne(x)
    rd = e8m10_round_rd(x)
    print("\nCase C': 1.0 + 2^-10 + 2^-11 (tie at LSB=1 -> RNE rounds up)")
    _show("input", x[0])
    _show("RNE", rne[0])
    _show("RD", rd[0])
    failures += not check("RNE: rounds up to next E8M10",
                          rne.item() == 1.0 + 2.0 ** -9)
    failures += not check("RD : positive truncates to 1.0 + 2^-10",
                          rd.item() == 1.0 + 2.0 ** -10)

    # ------------------------------------------------------------------
    # Test 4: -1.0 - 2^-13 — negative value with non-zero dropped bits.
    # RD rounds toward -inf, so the magnitude INCREASES. The kept LSB
    # of E8M10 at this magnitude is 2^-10. So RD result = -(1.0 + 2^-10).
    # RNE: tie (round bit set, sticky 0, LSB even=0) -> round to even
    # which means truncate the magnitude back to -1.0.
    # ------------------------------------------------------------------
    x = torch.tensor([-(1.0 + 2.0 ** -13)], dtype=torch.float32)
    rne = e8m10_round_rne(x)
    rd = e8m10_round_rd(x)
    print("\nCase D: -(1.0 + 2^-13)  (negative, RD bumps magnitude)")
    _show("input", x[0])
    _show("RNE", rne[0])
    _show("RD", rd[0])
    failures += not check("RNE: -1.0 - 2^-13 ties to even -> -1.0",
                          rne.item() == -1.0)
    expected_rd = -(1.0 + 2.0 ** -10)
    failures += not check("RD : -1.0 - 2^-13 rounds toward -inf -> -(1+2^-10)",
                          abs(rd.item() - expected_rd) < 1e-12,
                          f"got {rd.item()}, expected {expected_rd}")

    # ------------------------------------------------------------------
    # Test 5: A value with the round bit clear -> truncation either way.
    # ------------------------------------------------------------------
    x = torch.tensor([1.0 + 2.0 ** -10 + 2.0 ** -14], dtype=torch.float32)
    rne = e8m10_round_rne(x)
    rd = e8m10_round_rd(x)
    print("\nCase E: 1.0 + 2^-10 + 2^-14  (round bit clear -> truncate)")
    _show("input", x[0])
    _show("RNE", rne[0])
    _show("RD", rd[0])
    expected = 1.0 + 2.0 ** -10
    failures += not check("RNE: truncates", abs(rne.item() - expected) < 1e-12)
    failures += not check("RD : truncates", abs(rd.item() - expected) < 1e-12)

    # ------------------------------------------------------------------
    # Test 6: NaN and Inf preservation.
    # ------------------------------------------------------------------
    x = torch.tensor([float("nan"), float("inf"), float("-inf")], dtype=torch.float32)
    rne = e8m10_round_rne(x)
    rd = e8m10_round_rd(x)
    print("\nCase F: NaN/Inf preservation")
    failures += not check("RNE: NaN preserved", torch.isnan(rne[0]).item())
    failures += not check("RNE: +Inf preserved", rne[1].item() == float("inf"))
    failures += not check("RNE: -Inf preserved", rne[2].item() == float("-inf"))
    failures += not check("RD : NaN preserved", torch.isnan(rd[0]).item())
    failures += not check("RD : +Inf preserved", rd[1].item() == float("inf"))
    failures += not check("RD : -Inf preserved", rd[2].item() == float("-inf"))

    # ------------------------------------------------------------------
    # Test 7: Zero handling.
    # ------------------------------------------------------------------
    x = torch.tensor([0.0, -0.0], dtype=torch.float32)
    rne = e8m10_round_rne(x)
    rd = e8m10_round_rd(x)
    print("\nCase G: zero handling")
    failures += not check("RNE: +0.0 preserved", rne[0].item() == 0.0)
    failures += not check("RD : +0.0 preserved", rd[0].item() == 0.0)
    # -0.0 has sign bit set but mantissa zero -> low13 == 0 -> no bump.
    failures += not check("RD : -0.0 stays -0.0",
                          fp32_bits(rd[1].item()) == fp32_bits(-0.0))

    # ------------------------------------------------------------------
    # Test 8: Bias direction over many random samples.
    # RNE mean error should be ~0; RD mean error should be < 0
    # (everything tilts toward -inf).
    # ------------------------------------------------------------------
    torch.manual_seed(0)
    x = torch.randn(100_000, dtype=torch.float32) * 100  # spread out
    err_rne = (e8m10_round_rne(x) - x).double().mean().item()
    err_rd = (e8m10_round_rd(x) - x).double().mean().item()
    print(f"\nCase H: signed mean error over 100k random samples")
    print(f"  RNE mean error: {err_rne:+.4e}  (expect ~0)")
    print(f"  RD  mean error: {err_rd:+.4e}  (expect < 0, i.e. negative bias)")
    failures += not check("RNE: |mean error| small", abs(err_rne) < 1e-3)
    failures += not check("RD : mean error is negative", err_rd < 0)

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {failures} assertion(s)")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
