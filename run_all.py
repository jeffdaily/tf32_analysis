"""Run every reproducer in reproducers.ALL_REPRODUCERS, dump JSON + a table.

Usage:
    cd /tmp && HIP_VISIBLE_DEVICES=7 python /var/lib/jenkins/pytorch/agent_space/tf32_analysis/run_all.py [--only NAME]
"""

import argparse
import json
import os
import sys
import traceback

import torch

sys.path.insert(0, "/var/lib/jenkins/pytorch/agent_space/tf32_analysis")
from reproducers import ALL_REPRODUCERS

OUT_DIR = "/var/lib/jenkins/pytorch/agent_space/tf32_analysis"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="Run only these reproducer names (default: all)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA/HIP not available", file=sys.stderr)
        sys.exit(1)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"HIP   : {getattr(torch.version, 'hip', None)}")
    print(f"HIP_VISIBLE_DEVICES = {os.environ.get('HIP_VISIBLE_DEVICES', '<unset>')}")
    print()

    names = args.only if args.only else list(ALL_REPRODUCERS.keys())
    results = {}
    failures = []

    for name in names:
        if name not in ALL_REPRODUCERS:
            print(f"Unknown reproducer: {name}", file=sys.stderr)
            failures.append(name)
            continue
        print(f"--- {name} ---")
        try:
            res = ALL_REPRODUCERS[name]()
            results[name] = res
            _print_short(res)
        except Exception as exc:
            print(f"FAILED: {exc}")
            traceback.print_exc()
            failures.append(name)
        print()

    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True, default=str)
    print(f"\nResults written to {out_path}")
    print(f"Successful: {len(results)} / {len(names)}")
    if failures:
        print(f"Failed: {failures}")


def _print_short(res: dict) -> None:
    print(f"  test_id: {res['test_id']}")
    print(f"  op: {res['op']}  K={res.get('K')}  tol_atol={res['tol_atol']}")
    if "verdict_hint" in res:
        print(f"  verdict_hint: {res['verdict_hint']}")
    errors = res["errors"]
    # Show all top-level error keys
    for key, stats in errors.items():
        if isinstance(stats, dict) and "max_abs" in stats:
            print(f"    {key:42s}  max_abs={stats['max_abs']:.4e}  "
                  f"mean(signed)={stats.get('mean_signed', float('nan')):+.4e}")


if __name__ == "__main__":
    main()
