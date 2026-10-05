#!/usr/bin/env python3
"""Evaluate every prediction produced by scripts/predict.py.

Each metric has its KonfAI configuration in configs/evaluation/predictions. Results go to
results/generated/Evaluations with the same tree as results/raw/Evaluations, so the
table scripts can read either (IMPACTSYNTH_EVALUATIONS=results/generated/Evaluations).

Branch suffix -> metric:
  (none)         MAE, PSNR, SSIM            every prediction
  _SEG           TotalSegmentator Dice      5-fold ensembles
  _PERCEPTUAL    calibrated d_SAM           ELX-trained MAE/SAM, ensembles and folds
  _PERCEPTUAL1   LPIPS                      same
  _Uncertainty   variance of the 15         5-fold ensembles (MAE, SAM, VGG_IMPACT)
                 predictions, mean

A criterion containing ``_IMPACT`` is scored against ``CT`` (IMPACT geometry), otherwise
against ``CT_ELX``.
"""

import argparse
import shutil
import sys
from pathlib import Path
from string import Template

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _konfai import ROOT, device_args, run
from predict import REGIONS as DATASETS
from predict import UNCERTAINTY

OUTPUT = "results/generated/Evaluations"
TEMPLATES = ROOT / "configs/evaluation/predictions"
# region label of the metric tree -> prepared dataset
REGIONS = {task: {"ood" if dataset.endswith("CSIRO") else dataset.rsplit("/", 1)[1]: dataset for dataset in datasets}
           for task, datasets in DATASETS.items()}
# branch suffix -> (configuration, selector(task, mode, fold, criterion))
KINDS = {
    "": ("image", lambda t, m, f, c: True),
    "_SEG": ("seg", lambda t, m, f, c: f == "CV"),
    "_PERCEPTUAL": ("sam", lambda t, m, f, c: m == "ELX" and c in ("MAE", "SAM")),
    "_PERCEPTUAL1": ("lpips", lambda t, m, f, c: m == "ELX" and c in ("MAE", "SAM")),
    "_Uncertainty": ("uncertainty", lambda t, m, f, c: f == "CV" and c in UNCERTAINTY),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=int, choices=(1, 2), nargs="+", default=[1, 2])
    parser.add_argument("--mode", choices=("ELX", "IMPACT"), nargs="+", default=["ELX", "IMPACT"])
    parser.add_argument("--loss", choices=("MAE", "SAM", "VGG"), nargs="+", default=["MAE", "SAM", "VGG"])
    parser.add_argument("--fold", nargs="+", help="e.g. CV CV_0 (default: all)")
    parser.add_argument("--kind", nargs="+", choices=[k.strip("_") or "image" for k in KINDS],
                        default=[k.strip("_") or "image" for k in KINDS])
    parser.add_argument("--regions", nargs="+", default=["synthrad2025", "synthrad2023", "external"],
                        help="restrict to prepared datasets containing these names")
    parser.add_argument("--check", action="store_true",
                        help="quick check: 10 held-out cases per table row (data/splits/task_{t}/Check.txt)")
    parser.add_argument("--gpu", type=int, nargs="+")
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--force", action="store_true", help="recompute existing metrics")
    parser.add_argument("--remove-predictions", action="store_true",
                        help="delete each prediction once all its metrics exist")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for task in args.task:
        base = ROOT / f"predictions/Task_{task}/Supervised"
        outputs = sorted(base.glob("*/*/*/Output")) if base.is_dir() else []
        for output in outputs:
            mode, fold, criterion = output.relative_to(base).parts[:3]
            if mode not in args.mode or criterion.split("_")[0] not in args.loss or (args.fold and fold not in args.fold):
                continue
            for suffix, (template, selected) in KINDS.items():
                if (suffix.strip("_") or "image") not in args.kind or not selected(task, mode, fold, criterion):
                    continue
                target = "CT" if "_IMPACT" in criterion else "CT_ELX"
                for label, dataset in REGIONS[task].items():
                    if not (ROOT / "data/processed" / dataset).is_dir() or not any(r in dataset for r in args.regions):
                        continue
                    cases = (ROOT / "data/processed" / dataset).iterdir()
                    if not any((output / case.name).is_dir() for case in cases):
                        continue  # nothing predicted for this dataset in this branch (e.g. prepared later)
                    name = f"Task_{task}/Supervised/{mode}/{fold}{suffix}/{criterion}/{label}"
                    if (ROOT / OUTPUT / name / "Metric_TRAIN.json").is_file() and not args.force:
                        continue
                    subset = f"./data/splits/task_{task}/Check.txt" if args.check else "None"
                    config = Template((TEMPLATES / f"{template}.yml").read_text()).substitute(
                        target=target, subset=subset, dataset=f"./data/processed/{dataset}",
                        predictions=f"./{output.relative_to(ROOT)}", name=name)
                    path = ROOT / "results/generated/configs" / f"{name.replace('/', '_')}.yml"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(config)
                    run(["konfai", "EVALUATION", "-y", "--config", str(path), *device_args(args.gpu, args.cpu),
                         "--evaluations-dir", OUTPUT], args.dry_run)
            if args.remove_predictions and not args.dry_run:  # run() raises on any failed evaluation
                shutil.rmtree(output)
                print(f"removed {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
