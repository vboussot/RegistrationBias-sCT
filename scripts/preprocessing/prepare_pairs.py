#!/usr/bin/env python3
"""Prepare ELX- and IMPACT-convention pairs from the SynthRAD data.

``CT`` is the planning CT, ``CT_ELX`` is the CT warped with the organizer (ELX)
transform (computed by register_elx.py for SynthRAD2025), and
``MR_IMPACT``/``CBCT_IMPACT`` is the source modality resampled with the
released IMPACT transform. It requires a SimpleITK-SimpleElastix build.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import SimpleITK as sitk


def require_simpleelastix() -> None:
    missing = [
        name
        for name in ("ReadParameterFile", "TransformixImageFilter")
        if not hasattr(sitk, name)
    ]
    if missing:
        raise SystemExit(
            "This preprocessing step requires SimpleITK-SimpleElastix; "
            f"missing API: {', '.join(missing)}. See docs/ENVIRONMENT.md."
        )


def resample(image: sitk.Image, parameters) -> sitk.Image:
    extrema = sitk.MinimumMaximumImageFilter()
    extrema.Execute(image)
    low, high = extrema.GetMinimum(), extrema.GetMaximum()
    if image.GetPixelID() == sitk.sitkUInt8:
        parameters["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
        parameters["ResultImagePixelType"] = ["unsigned char"]
    else:
        parameters["ResampleInterpolator"] = ["FinalBSplineInterpolator"]
        parameters["ResultImagePixelType"] = ["short"]
    parameters["DefaultPixelValue"] = [str(low)]
    transformix = sitk.TransformixImageFilter()
    transformix.SetTransformParameterMap(parameters)
    transformix.SetMovingImage(image)
    transformix.Execute()
    result = sitk.Clamp(transformix.GetResultImage(), lowerBound=low, upperBound=high)
    return sitk.Cast(result, image.GetPixelID())


def snap_to(image: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """Parameter files store the output grid with float32 rounding (e.g. -152.699997
    for -152.700012), which ITK rejects when masking. Snap sub-micron differences."""
    same_grid = image.GetSize() == reference.GetSize() and all(
        abs(a - b) < 1e-3
        for a, b in zip(image.GetOrigin() + image.GetSpacing(), reference.GetOrigin() + reference.GetSpacing())
    )
    if not same_grid:
        raise ValueError("resampled image does not match the reference grid")
    image.CopyInformation(reference)
    return image


def input_file(case_dir: Path, stem: str, year: int) -> Path:
    return case_dir / f"{stem}{'.mha' if year == 2025 else '.nii.gz'}"


def prepare_case(
    case_dir: Path,
    transform_path: Path,
    elx_path: Path | None,
    output_dir: Path,
    task: int,
    year: int,
    overwrite: bool,
) -> None:
    modality = "MR" if task == 1 else "CBCT"
    with_elx = year == 2023 or elx_path is not None
    names = ("CT.mha", "MASK.mha", f"{modality}.mha", f"{modality}_IMPACT.mha") + (("CT_ELX.mha",) if with_elx else ())
    expected = [output_dir / name for name in names]
    if not overwrite and all(path.exists() for path in expected):
        print(f"skip {output_dir.name}")
        return

    ct = sitk.Cast(sitk.ReadImage(str(input_file(case_dir, "ct", year))), sitk.sitkInt16)
    source_stem = "mr" if task == 1 else "cbct"
    source = sitk.Cast(sitk.ReadImage(str(input_file(case_dir, source_stem, year))), sitk.sitkInt16)
    mask = sitk.Cast(sitk.ReadImage(str(input_file(case_dir, "mask", year))), sitk.sitkUInt8)
    mask.CopyInformation(ct)
    output_dir.mkdir(parents=True, exist_ok=True)

    sitk.WriteImage(sitk.Mask(ct, mask, -1024), str(output_dir / "CT.mha"))
    sitk.WriteImage(sitk.Mask(source, mask, 0 if task == 1 else -1024), str(output_dir / f"{modality}.mha"))
    sitk.WriteImage(mask, str(output_dir / "MASK.mha"))

    if year == 2025 and elx_path is not None:
        ct_elx = snap_to(resample(ct, sitk.ReadParameterFile(str(elx_path))), mask)
        sitk.WriteImage(sitk.Mask(ct_elx, mask, -1024), str(output_dir / "CT_ELX.mha"))
    elif year == 2023:
        sitk.WriteImage(sitk.Mask(ct, mask, -1024), str(output_dir / "CT_ELX.mha"))

    source_impact = snap_to(resample(source, sitk.ReadParameterFile(str(transform_path))), mask)
    source_impact = sitk.Mask(source_impact, mask, 0 if task == 1 else -1024)
    sitk.WriteImage(source_impact, str(output_dir / f"{modality}_IMPACT.mha"))
    print(f"prepared {output_dir.name}")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, choices=(2023, 2025), required=True)
    parser.add_argument("--task", type=int, choices=(1, 2), required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--raw-root", type=Path, default=root / "data/raw")
    parser.add_argument("--transforms-root", type=Path, default=root / "data/transforms")
    parser.add_argument("--output-root", type=Path, default=root / "data/processed")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--without-elx", action="store_true",
                        help="2025: skip CT_ELX (enough to train or predict in the IMPACT convention)")
    args = parser.parse_args()
    require_simpleelastix()

    cases = args.raw_root / f"synthrad{args.year}" / "Train" / f"Task_{args.task}" / args.region
    transforms = args.transforms_root / f"synthrad{args.year}-impact-registration" / f"Task_{args.task}" / args.region
    output = args.output_root / f"synthrad{args.year}" / f"Task_{args.task}" / args.region
    elx = args.transforms_root / "synthrad2025-elx-registration" / f"Task_{args.task}" / args.region
    if not cases.is_dir() or not transforms.is_dir():
        raise SystemExit(f"Missing input directory: {cases} or {transforms}")
    for transform in sorted(transforms.glob("*.txt")):
        case_dir = cases / transform.stem
        if not case_dir.is_dir():
            print(f"skip missing raw case {case_dir}")
            continue
        elx_path = elx / transform.name if args.year == 2025 and not args.without_elx else None
        if elx_path is not None and not elx_path.is_file():
            print(f"skip {transform.stem}: no ELX transform yet (run register_elx.py)")
            continue
        prepare_case(case_dir, transform, elx_path, output / transform.stem, args.task, args.year, args.overwrite)


if __name__ == "__main__":
    main()
