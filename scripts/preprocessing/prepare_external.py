#!/usr/bin/env python3
"""Prepare the two independent test sets.

Sim-CBCT (Task 2): CBCT simulated from the held-out SynthRAD2025 CTs by cbct_synthesis.py, one
``{case}.mha`` (or ``{case}_CT.mha``) per case; download.py puts those of the paper in data/raw/sim_cbct. The simulated image is both the ELX and the
IMPACT input, and the CT both references (aligned by construction):
    data/processed/external/Task_2/{region}_ood/{case}_ood/{CT,CT_ELX,CBCT,CBCT_IMPACT,MASK}.mha
(``--sim-suffix`` changes the ``ood`` suffix, to keep a second set of simulations apart).

Ext-T2 (Task 1): expert-aligned T2 MRI/CT pairs (Dowling et al.), one folder per case
with CT.nii.gz, MR.nii.gz and the body mask C_ext.nii.gz, cropped to the mask's slice range
minus 3 slices at each end:
    data/processed/external/Task_1/CSIRO/{case}/{CT,CT_ELX,MR,MR_IMPACT,MASK}.mha
"""

import argparse
from pathlib import Path

import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[2]


def write(images: dict[str, sitk.Image], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, image in images.items():
        sitk.WriteImage(image, str(output / f"{name}.mha"), True)


def prepare_sim_cbct(sim_dir: Path, raw_root: Path, output_root: Path, suffix: str) -> None:
    for path in sorted(sim_dir.glob("*.mha")):
        case = path.name.removesuffix(".mha").removesuffix("_CT")
        region = case[1:3]
        raw = raw_root / f"Task_2/{region}/{case}"
        if not raw.is_dir():
            print(f"skip {case}: no raw SynthRAD2025 case")
            continue
        ct = sitk.Cast(sitk.ReadImage(str(raw / "ct.mha")), sitk.sitkInt16)
        mask = sitk.Cast(sitk.ReadImage(str(raw / "mask.mha")), sitk.sitkUInt8)
        mask.CopyInformation(ct)
        cbct = sitk.Mask(sitk.Cast(sitk.ReadImage(str(path)), sitk.sitkInt16), mask, -1024)
        ct = sitk.Mask(ct, mask, -1024)
        write({"CT": ct, "CT_ELX": ct, "CBCT": cbct, "CBCT_IMPACT": cbct, "MASK": mask},
              output_root / f"Task_2/{region}_{suffix}/{case}_{suffix}")
        print(f"prepared {case}_{suffix}")


def crop_slices(images: list[sitk.Image], mask: sitk.Image, margin: int = 3) -> list[sitk.Image]:
    """Keep the body-mask slice range minus `margin` slices at each end, as evaluated in the paper."""
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask > 0)
    _, _, z, _, _, depth = stats.GetBoundingBox(1)
    size = [*mask.GetSize()[:2], depth - 2 * margin]
    return [sitk.RegionOfInterest(image, size=size, index=[0, 0, z + margin]) for image in images]


def prepare_ext_t2(csiro_dir: Path, output_root: Path) -> None:
    for case_dir in sorted(path for path in csiro_dir.iterdir() if path.is_dir()):
        images = [sitk.ReadImage(str(case_dir / name)) for name in ("C_ext.nii.gz", "CT.nii.gz", "MR.nii.gz")]
        mask, ct, mr = crop_slices(images, images[0])
        mask = sitk.Cast(mask, sitk.sitkUInt8)
        mr = sitk.Cast(mr, sitk.sitkInt16)
        ct.CopyInformation(mask)
        mr.CopyInformation(mask)
        ct = sitk.Cast(sitk.Clamp(sitk.Mask(ct, mask, -1024), lowerBound=-1024, upperBound=3071), sitk.sitkInt16)
        mr = sitk.Mask(mr, mask, 0)
        write({"CT": ct, "CT_ELX": ct, "MR": mr, "MR_IMPACT": mr, "MASK": mask},
              output_root / f"Task_1/CSIRO/{case_dir.name}")
        print(f"prepared {case_dir.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim-cbct", type=Path, help="folder of simulated CBCT volumes")
    parser.add_argument("--sim-suffix", default="ood",
                        help="suffix of the region and case names (default: ood)")
    parser.add_argument("--ext-t2", type=Path, help="folder of Ext-T2 cases")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/raw/synthrad2025/Train")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/processed/external")
    args = parser.parse_args()
    if args.sim_cbct:
        prepare_sim_cbct(args.sim_cbct, args.raw_root, args.output_root, args.sim_suffix)
    if args.ext_t2:
        prepare_ext_t2(args.ext_t2, args.output_root)
    if not (args.sim_cbct or args.ext_t2):
        parser.error("give --sim-cbct and/or --ext-t2")


if __name__ == "__main__":
    main()
