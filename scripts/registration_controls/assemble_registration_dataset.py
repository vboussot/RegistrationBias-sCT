#!/usr/bin/env python3
"""Assemble the KonfAI dataset of the Table 9 CT-only registration controls.

Reads the baselines written by
  evaluate_ct_elx_impact_perturbations.py --samples-per-level 0 --save-images
(`<input-root>/Task_t/<region>/<case>/CT_deformed_baseline.mha`) and writes
  <output-root>/Task_t/<region>/<case>/{CT,CT_deformed,MASK}.mha
with CT and MASK copied from data/processed and CT_deformed left as resampled, as in the paper: outside
MASK it keeps the B-spline ripple of the warped background (about -1022 HU instead of -1024), which
the unmasked d_SAM and LPIPS see (masking it lowers them by 15-30 %; MAE, PSNR and SSIM are masked).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[2]
YEAR = {"AB": "synthrad2025", "HN": "synthrad2025", "TH": "synthrad2025",
        "brain": "synthrad2023", "pelvis": "synthrad2023"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-root", type=Path,
                        default=ROOT / "results/generated/Registration/CT_ELX_IMPACT_baseline")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "results/generated/Registration/CT_ELX_IMPACT_to_CT/dataset")
    parser.add_argument("--processed-root", type=Path, default=ROOT / "data/processed")
    args = parser.parse_args()

    baselines = sorted(args.input_root.glob("Task_*/*/*/CT_deformed_baseline.mha"))
    if not baselines:
        raise SystemExit(f"No CT_deformed_baseline.mha under {args.input_root}")
    for baseline in baselines:
        task, region, case = baseline.parts[-4:-1]
        source = args.processed_root / YEAR[region] / task / region / case
        target = args.output_root / task / region / case
        target.mkdir(parents=True, exist_ok=True)
        for name in ("CT.mha", "MASK.mha"):
            if not ((target / name).exists() and (target / name).samefile(source / name)):  # may be a hardlink
                shutil.copy2(source / name, target / name)
        sitk.WriteImage(sitk.ReadImage(str(baseline)), str(target / "CT_deformed.mha"), True)
    print(f"Wrote {len(baselines)} cases to {args.output_root}")


if __name__ == "__main__":
    main()
