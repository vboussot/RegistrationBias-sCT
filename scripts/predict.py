#!/usr/bin/env python3
"""Predict the held-out cases with the checkpoints of the paper.

An ensemble (``CV``) runs the five fold checkpoints, each on the original image plus two
flips, and averages the 15 predictions. A single fold (``CV_i``) runs its checkpoint on
the original image only, as in the paper. The output tree is the one scripts/evaluate.py
reads:

    predictions/Task_{t}/Supervised/{MODE}/{FOLD}/{CRITERION}/Output/{case}/sCT.mha

MODE is the training convention (ELX or IMPACT). CRITERION is the loss, suffixed with
``_IMPACT`` when the model reads the IMPACT-registered input (MR_IMPACT/CBCT_IMPACT)
instead of the native MR/CBCT of the ELX convention.

The ensembles behind the uncertainty analysis (Table 4) also get ``Uncertainty.mha``, the
voxel-wise variance of their 15 predictions. Datasets are predicted one after the other:
what is already there is skipped, so an interrupted run resumes where it stopped.
"""

import argparse
import sys
from pathlib import Path

import SimpleITK as sitk
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _konfai import ROOT, device_args, run

REGIONS = {
    1: ["synthrad2025/Task_1/AB", "synthrad2025/Task_1/HN", "synthrad2025/Task_1/TH",
        "synthrad2023/Task_1/brain", "synthrad2023/Task_1/pelvis", "external/Task_1/CSIRO"],
    2: ["synthrad2025/Task_2/AB", "synthrad2025/Task_2/HN", "synthrad2025/Task_2/TH",
        "synthrad2023/Task_2/brain", "synthrad2023/Task_2/pelvis",
        "external/Task_2/AB_ood", "external/Task_2/HN_ood", "external/Task_2/TH_ood",
        "external/Task_2/AB_sim", "external/Task_2/HN_sim", "external/Task_2/TH_sim"],
}
UNCERTAINTY = ("MAE", "SAM", "VGG_IMPACT")  # ensemble criteria scored by evaluate.py --kind Uncertainty


def plan():
    """(task, mode, loss, input convention, fold) read by the table scripts."""
    for task in (1, 2):
        for mode in ("ELX", "IMPACT"):
            for convention in ("ELX", "IMPACT"):
                yield task, mode, "MAE", convention, "CV"  # Tables 2-4, 14, 15
        yield task, "ELX", "SAM", "ELX", "CV"  # Tables 6, 18, 19
        yield task, "IMPACT", "SAM", "IMPACT", "CV"  # Tables 5, 16, 17
        yield task, "IMPACT", "VGG", "IMPACT", "CV"
        for loss in ("MAE", "SAM"):
            for fold in range(5):
                yield task, "ELX", loss, "ELX", f"CV_{fold}"  # Mean CV, Tables 6, 18, 19


def finish(output: Path, keep_uncertainty: bool) -> None:
    """Replace each 15-member stack (1 to 2 GB) by its variance, or drop it; compress the sCT (lossless)."""
    for stack in sorted(output.glob("*/InferenceStack.mha")):
        if keep_uncertainty:
            image = sitk.ReadImage(str(stack))
            members = torch.from_numpy(sitk.GetArrayFromImage(image))
            if image.GetNumberOfComponentsPerPixel() > 1:
                members = members.movedim(-1, 0)
            variance = sitk.GetImageFromArray(torch.var(members.float(), 0).numpy())  # as KonfAI's Variance
            variance.CopyInformation(sitk.ReadImage(str(stack.with_name("sCT.mha"))))
            sitk.WriteImage(variance, str(stack.with_name("Uncertainty.mha")), True)
        stack.unlink()
    for path in sorted(output.glob("*/sCT.mha")):
        with path.open("rb") as handle:
            if b"CompressedData = True" not in handle.read(1024):
                sitk.WriteImage(sitk.ReadImage(str(path)), str(path), True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=int, choices=(1, 2), nargs="+", default=[1, 2])
    parser.add_argument("--mode", choices=("ELX", "IMPACT"), nargs="+", default=["ELX", "IMPACT"])
    parser.add_argument("--loss", choices=("MAE", "SAM", "VGG"), nargs="+", default=["MAE", "SAM", "VGG"])
    parser.add_argument("--fold", nargs="+", help="e.g. CV CV_0 (default: all)")
    parser.add_argument("--convention", choices=("ELX", "IMPACT"), nargs="+", default=["ELX", "IMPACT"],
                        help="input convention: native MR/CBCT (ELX) or IMPACT-registered")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "models/checkpoints",
                        help="root with Task_{t}/{MODE}/{LOSS}/CV_{i}.pt")
    parser.add_argument("--regions", nargs="+", default=["synthrad2025", "synthrad2023", "external"],
                        help="restrict to prepared datasets containing these names")
    parser.add_argument("--check", action="store_true",
                        help="quick check: 10 held-out cases per table row (data/splits/task_{t}/Check.txt)")
    parser.add_argument("--batch-size", type=int,
                        help="patches per forward; 0 lets KonfAI measure what the GPU holds "
                             "(default: 0 for an ensemble, 1 for a single fold)")
    parser.add_argument("--gpu", type=int, nargs="+")
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--force", action="store_true", help="recompute existing predictions")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for task, mode, loss, convention, fold in plan():
        if task not in args.task or mode not in args.mode or loss not in args.loss:
            continue
        if (args.fold and fold not in args.fold) or convention not in args.convention:
            continue
        criterion = loss + ("_IMPACT" if convention == "IMPACT" else "")
        name = f"Task_{task}/Supervised/{mode}/{fold}/{criterion}"
        output = ROOT / "predictions" / name / "Output"
        folds = range(5) if fold == "CV" else [int(fold.split("_")[1])]
        models = [args.checkpoints / f"Task_{task}/{mode}/{loss}/CV_{index}.pt" for index in folds]
        missing = [str(path) for path in models if not path.is_file()]
        if missing:
            print(f"miss  {name}: " + ", ".join(missing))
            continue
        subset = f"./data/splits/task_{task}/{'Check' if args.check else 'Validation'}.txt"
        held_out = set((ROOT / subset).read_text().split())
        for region in REGIONS[task]:
            dataset = ROOT / "data/processed" / region
            if not dataset.is_dir() or not any(r in region for r in args.regions):
                continue
            cases = [case.name for case in dataset.iterdir() if case.name in held_out]
            if not cases or (not args.force and all((output / case / "sCT.mha").is_file() for case in cases)):
                print(f"skip  {name} {region}")
                continue

            config = yaml.safe_load((ROOT / f"configs/inference/task_{task}_{convention.lower()}_tta.yaml").read_text())
            predictor = config["Predictor"]
            predictor["Dataset"]["dataset_filenames"] = [f"./data/processed/{region}:a:mha"]
            predictor["Dataset"]["subset"] = subset
            predictor["Dataset"]["batch_size"] = args.batch_size if args.batch_size is not None else int(fold != "CV")
            if fold != "CV":  # single folds are predicted without test-time flips
                predictor["Dataset"]["augmentations"] = "None"
                predictor["Dataset"]["num_workers"] = 1
            predictor["train_name"] = name
            stack = predictor["outputs_dataset"]["Head:Tanh"]["OutputDataset"]["after_reduction_transforms"]
            stack["InferenceStack"]["dataset"] = f"predictions/{name}/Output:mha"
            # KonfAI rewrites the config it reads, so a fresh copy is generated for every run.
            path = ROOT / "predictions/configs" / (f"{name}/{region}".replace("/", "_") + ".yaml")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            run(["konfai", "PREDICTION", "-y", "--config", str(path), *device_args(args.gpu, args.cpu),
                 "--models", *map(str, models), "--predictions-dir", "predictions"], args.dry_run)
            if not args.dry_run:
                finish(output, fold == "CV" and criterion in UNCERTAINTY)


if __name__ == "__main__":
    main()
