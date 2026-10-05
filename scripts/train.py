#!/usr/bin/env python3
"""Train the models of the paper: 2 tasks x (ELX, IMPACT) x (MAE, SAM) + IMPACT/VGG, 5 folds.

Optional: predict.py starts from the released checkpoints. Needs the full training sets (download.py --scope all), the
prepared pairs, and for SAM models/external/sam2.1_hiera_small.pt. KonfAI writes every
checkpoint under artifacts/checkpoints/<train_name>; the paper keeps, for each fold,
the checkpoint with the lowest validation MAE. Copy it to
models/<root>/Task_{t}/{MODE}/{LOSS}/CV_{i}.pt and pass --checkpoints to predict.py.
Hardware requirements are in docs/ENVIRONMENT.md.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _konfai import ROOT, device_args, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=int, choices=(1, 2), nargs="+", default=[1, 2])
    parser.add_argument("--mode", choices=("elx", "impact"), nargs="+", default=["elx", "impact"])
    parser.add_argument("--loss", choices=("mae", "sam", "vgg"), nargs="+", default=["mae", "sam", "vgg"])
    parser.add_argument("--fold", type=int, choices=range(5), nargs="+", default=list(range(5)))
    parser.add_argument("--gpu", type=int, nargs="+")
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for task in args.task:
        for mode in args.mode:
            for loss in args.loss:
                for fold in args.fold:
                    config = ROOT / f"configs/training/task_{task}/{mode}/{loss}/fold_{fold}.yaml"
                    if not config.is_file():  # VGG exists for IMPACT only
                        continue
                    # KonfAI rewrites the config it reads: train from a copy, keep the tracked file intact.
                    copy = ROOT / f"artifacts/configs/task_{task}_{mode}_{loss}_fold_{fold}.yaml"
                    copy.parent.mkdir(parents=True, exist_ok=True)
                    copy.write_text(config.read_text())
                    run(["konfai", "TRAIN", "-y", "--config", str(copy), *device_args(args.gpu, args.cpu),
                         "--checkpoints-dir", "artifacts/checkpoints", "--statistics-dir", "artifacts/statistics"],
                        args.dry_run)


if __name__ == "__main__":
    main()
