#!/usr/bin/env python3
"""Score the Table 9 CT-only registration controls (CT_deformed vs CT) with KonfAI.

Renders configs/evaluation/registration_controls/<kind>.yml for every task/region of the
dataset written by assemble_registration_dataset.py, with the train names of
results/raw/Evaluations:

  image  Registration_ELX_IMPACT/Task_t/<r>               MAE, PSNR, SSIM
  sam    Registration_ELX_IMPACT_Perceptual/Task_t/<r>1   calibrated d_SAM
  lpips  Registration_ELX_IMPACT_Perceptual/Task_t/<r>    LPIPS
  seg    Registration_ELX_IMPACT_SEG/Task_t/<r>           TotalSegmentator Dice per label

Metrics go to results/generated/Evaluations; existing ones are skipped unless --force.
"""

import argparse
import sys
from pathlib import Path
from string import Template

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _konfai import ROOT, device_args, resolve, run

OUTPUT = "results/generated/Evaluations"
TEMPLATES = ROOT / "configs/evaluation/registration_controls"
KINDS = {
    "image": "Registration_ELX_IMPACT/Task_$task/$region",
    "sam": "Registration_ELX_IMPACT_Perceptual/Task_$task/${region}1",
    "lpips": "Registration_ELX_IMPACT_Perceptual/Task_$task/$region",
    "seg": "Registration_ELX_IMPACT_SEG/Task_$task/$region",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=int, choices=(1, 2), nargs="+", default=[1, 2])
    parser.add_argument("--regions", nargs="+", default=["AB", "HN", "TH", "brain", "pelvis"])
    parser.add_argument("--kind", nargs="+", choices=list(KINDS), default=list(KINDS))
    parser.add_argument("--dataset", type=Path,
                        default=Path("results/generated/Registration/CT_ELX_IMPACT_to_CT/dataset"))
    parser.add_argument("--gpu", type=int, nargs="+")
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--force", action="store_true", help="recompute existing metrics")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_root = resolve(args.dataset)
    for task in args.task:
        for region in args.regions:
            dataset = dataset_root / f"Task_{task}" / region
            if not dataset.is_dir():
                print(f"skip missing {dataset}")
                continue
            for kind in args.kind:
                name = Template(KINDS[kind]).substitute(task=task, region=region)
                if (ROOT / OUTPUT / name / "Metric_TRAIN.json").is_file() and not args.force:
                    continue
                # Written fresh each run: KonfAI rewrites the configs it reads.
                config = Template((TEMPLATES / f"{kind}.yml").read_text()).substitute(dataset=dataset, name=name)
                path = ROOT / "results/generated/configs" / f"{name.replace('/', '_')}.yml"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(config)
                run(["konfai", "EVALUATION", "-y", "--config", str(path), *device_args(args.gpu, args.cpu),
                     "--evaluations-dir", OUTPUT], args.dry_run)


if __name__ == "__main__":
    main()
