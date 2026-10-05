#!/usr/bin/env python3
"""Compare the patient metrics of a new run with the paper's.

For every results/generated/Evaluations/**/Metric_TRAIN.json with a counterpart in
results/raw/Evaluations, print the largest difference of each metric over the patients present
in both: relative for MAE, PSNR, SSIM, d_SAM, LPIPS and the uncertainty, absolute for the Dice
(all labels together). Exit status 1 if a relative difference exceeds --tol.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generated", type=Path, default=ROOT / "results/generated/Evaluations")
    parser.add_argument("--reference", type=Path, default=ROOT / "results/raw/Evaluations")
    parser.add_argument("--tol", type=float, default=0.01, help="relative tolerance (default 1%%)")
    args = parser.parse_args()

    worst, worst_dice = 0.0, 0.0
    for new_file in sorted(args.generated.rglob("Metric_TRAIN.json")):
        branch = new_file.parent.relative_to(args.generated)
        old_file = args.reference / branch / "Metric_TRAIN.json"
        if not old_file.is_file() and branch.name == "ood":  # Ext-T2 d_SAM of Task 1 is under CSIRO
            old_file = args.reference / branch.parent / "CSIRO/Metric_TRAIN.json"
        if not old_file.is_file():
            continue
        new, old = (json.loads(path.read_text())["case"] for path in (new_file, old_file))
        differences: dict[str, list[tuple[float, float]]] = {}
        for key in sorted(set(new) & set(old)):
            name = "Dice" if "Dice" in key.split(":") else key.split(":")[-1]
            for case in set(new[key]) & set(old[key]):
                mine, reference = new[key][case], old[key][case]
                if mine is None or reference is None or mine != mine or reference != reference:
                    continue  # a label absent for this patient
                differences.setdefault(name, []).append((abs(mine - reference), abs(reference)))
        for name, values in differences.items():
            absolute = max(value[0] for value in values)
            if name == "Dice":
                worst_dice = max(worst_dice, absolute)
                print(f"{str(branch):58s} {name:14s} n={len(values):5d}  max|diff|={absolute:10.4f}")
                continue
            relative = max(value[0] / max(value[1], 1e-12) for value in values)
            worst = max(worst, relative)
            print(f"{str(branch):58s} {name:14s} n={len(values):5d}  max|diff|={absolute:10.4f}  max rel={relative:8.3%}")
    print(f"\nlargest relative difference: {worst:.3%}; largest Dice difference: {worst_dice:.3f}")
    raise SystemExit(worst > args.tol)


if __name__ == "__main__":
    main()
