#!/usr/bin/env python3
"""Compute the SynthRAD2025 organizer (ELX) deformable registrations.

Same call as stage2.py of https://github.com/SynthRAD2025/preprocessing
(commit 8a5b125), on the stage-2 images of the public Zenodo release: the CT
(moving) is registered to the MR/CBCT (fixed) with the patient mask as fixed
mask and the task/region parameter file.

Output: data/transforms/synthrad2025-elx-registration/Task_{task}/{region}/{case}.txt,
read by prepare_pairs.py. Requires SimpleITK-SimpleElastix (ElastixImageFilter).
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[2]


def register(case_dir: Path, parameter_file: Path, task: int) -> sitk.ParameterMap:
    fixed = sitk.ReadImage(str(case_dir / ("mr.mha" if task == 1 else "cbct.mha")))
    moving = sitk.ReadImage(str(case_dir / "ct.mha"))
    mask = sitk.Cast(sitk.ReadImage(str(case_dir / "mask.mha")), sitk.sitkUInt8)
    elastix = sitk.ElastixImageFilter()
    elastix.SetParameterMap(sitk.ReadParameterFile(str(parameter_file)))
    elastix.SetFixedImage(fixed)
    elastix.SetMovingImage(moving)
    elastix.SetFixedMask(mask)
    elastix.LogToConsoleOff()
    elastix.LogToFileOff()
    elastix.SetNumberOfThreads(16)  # as in stage2.py
    with tempfile.TemporaryDirectory() as work:
        previous = os.getcwd()
        os.chdir(work)  # elastix writes intermediate files to the working directory
        try:
            elastix.Execute()
        finally:
            os.chdir(previous)
    return elastix.GetTransformParameterMap()[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", type=int, choices=(1, 2), required=True)
    parser.add_argument("--region", choices=("AB", "HN", "TH"), required=True)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/raw/synthrad2025/Train")
    parser.add_argument("--parameters", type=Path, default=ROOT / "data/transforms/synthrad2025-elx-parameters")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "data/transforms/synthrad2025-elx-registration")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not hasattr(sitk, "ElastixImageFilter"):
        raise SystemExit("SimpleITK-SimpleElastix is required (ElastixImageFilter missing).")

    modality = "mr" if args.task == 1 else "cbct"
    parameter_file = args.parameters / f"param_def_{modality}_{args.region}.txt"
    cases = args.raw_root / f"Task_{args.task}" / args.region
    output = args.output_root / f"Task_{args.task}" / args.region
    if not parameter_file.is_file() or not cases.is_dir():
        raise SystemExit(f"Missing {parameter_file} or {cases}; run scripts/download.py first.")
    output.mkdir(parents=True, exist_ok=True)
    for case_dir in sorted(path for path in cases.iterdir() if path.is_dir()):
        target = output / f"{case_dir.name}.txt"
        if target.is_file() and not args.overwrite:
            continue
        sitk.WriteParameterFile(register(case_dir, parameter_file, args.task), str(target))
        print(f"registered {case_dir.name}")


if __name__ == "__main__":
    main()
