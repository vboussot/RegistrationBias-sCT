#!/usr/bin/env python3
"""Generate a Task 2 ELX table comparing mean CV folds against CV for MAE, SAM, and LPIPS."""

from __future__ import annotations

import argparse
import json
import os
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# The paper's metrics by default; IMPACTSYNTH_EVALUATIONS=results/generated/Evaluations for those of a new run.
EVALUATIONS = ROOT / os.environ.get("IMPACTSYNTH_EVALUATIONS", "results/raw/Evaluations")
EVALUATIONS_DIR = EVALUATIONS / "Task_2" / "Supervised" / "ELX"
OUTPUT_PATH = ROOT / os.environ.get("IMPACTSYNTH_TABLES", "tables/from_metrics") / "table_19_task2_perceptual.tex"

FOLD_INDICES = tuple(range(5))
CRITERIA = ("MAE", "SAM")
METRICS = ("MAE", "SAM", "LPIPS")
METRIC_KEYS = {
    "MAE": "sCT:CT;MASK:MAE",
    "SAM": "sCT:CT:SAM_Perceptual",
    "LPIPS": "sCT:CT:LPIPS",
}
METRIC_FOLD_DIRS = {
    "MAE": tuple(f"CV_{index}" for index in FOLD_INDICES),
    "SAM": tuple(f"CV_{index}_PERCEPTUAL" for index in FOLD_INDICES),
    "LPIPS": tuple(f"CV_{index}_PERCEPTUAL1" for index in FOLD_INDICES),
}
METRIC_CV_DIRS = {
    "MAE": "CV",
    "SAM": "CV_PERCEPTUAL",
    "LPIPS": "CV_PERCEPTUAL1",
}
REGION_ROWS = (
    ("AB", ("AB",)),
    ("HN", ("HN",)),
    ("TH", ("TH",)),
    ("AB/HN/TH", ("AB", "HN", "TH")),
    ("Sim-CBCT", ("AB_ood", "HN_ood", "TH_ood")),
)


def metric_path(fold: str, criterion: str, region: str) -> Path:
    return EVALUATIONS_DIR / fold / criterion / region / "Metric_TRAIN.json"


def read_case_dict(fold: str, criterion: str, region: str, metric: str) -> dict[str, float]:
    path = metric_path(fold, criterion, region)
    if not path.exists():
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    key = METRIC_KEYS[metric]
    return {patient: float(value) for patient, value in data["case"][key].items()}


def read_patient_means(
    fold: str,
    criterion: str,
    regions: tuple[str, ...],
    metric: str,
) -> dict[str, float]:
    patient_values: dict[str, list[float]] = {}
    for region in regions:
        cases = read_case_dict(fold, criterion, region, metric)
        for patient, value in cases.items():
            patient_values.setdefault(patient, []).append(value)

    return {
        patient: sum(values) / len(values)
        for patient, values in patient_values.items()
    }


def read_mean_fold_patient_means(
    metric: str,
    criterion: str,
    regions: tuple[str, ...],
) -> dict[str, float]:
    patient_values: dict[str, list[float]] = {}
    for fold_dir in METRIC_FOLD_DIRS[metric]:
        cases = read_patient_means(fold_dir, criterion, regions, metric)
        for patient, value in cases.items():
            patient_values.setdefault(patient, []).append(value)

    return {
        patient: sum(values) / len(values)
        for patient, values in patient_values.items()
    }


def aggregate(values: list[float]) -> tuple[float, float] | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def format_value(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.2f}"


def display_value(metric: str, value: float | None) -> float | None:
    if value is None:
        return None
    return value * 100.0 if metric in {"SAM", "LPIPS"} else value


def format_stats(metric: str, stats: tuple[float, float] | None, bold: bool, stars: str = "") -> str:
    if stats is None:
        return "--"
    mean, std = stats
    command = r"\bmstd" if bold else r"\mstd"
    return rf"{command}{{{format_value(display_value(metric, mean))}}}{{{format_value(display_value(metric, std))}}}{{{stars}}}"


def best_index(metric: str, values: list[float | None]) -> int | None:
    rounded = [
        (index, round(display_value(metric, value), 2))
        for index, value in enumerate(values)
        if value is not None
    ]
    if not rounded:
        return None
    return min(rounded, key=lambda item: (item[1], item[0]))[0]


def read_patient_means_for_experiment(
    experiment: str,
    criterion: str,
    regions: tuple[str, ...],
    metric: str,
) -> dict[str, float]:
    if experiment == "MEAN_CV":
        return read_mean_fold_patient_means(metric, criterion, regions)
    return read_patient_means(METRIC_CV_DIRS[metric], criterion, regions, metric)


def read_stats_for_experiment(
    experiment: str,
    criterion: str,
    regions: tuple[str, ...],
    metric: str,
) -> tuple[float, float] | None:
    return aggregate(list(read_patient_means_for_experiment(experiment, criterion, regions, metric).values()))


def read_mean_for_experiment(
    experiment: str,
    criterion: str,
    regions: tuple[str, ...],
    metric: str,
) -> float | None:
    stats = read_stats_for_experiment(experiment, criterion, regions, metric)
    if stats is None:
        return None
    return stats[0]


def render_header() -> list[str]:
    split_headers = [
        r"\multicolumn{6}{c}{Mean CV}",
        r"\multicolumn{6}{c}{CV}",
    ]
    criterion_headers = [r"\multicolumn{3}{c}{MAE}", r"\multicolumn{3}{c}{SAM}"] * 2
    metric_headers = list(METRICS) * len(CRITERIA) * 2
    cmidrules = []
    start = 2
    for _ in range(2):
        cmidrules.append(rf"\cmidrule(lr){{{start}-{start + 5}}}")
        start += 6
    inner_cmidrules = []
    start = 2
    for _ in range(4):
        inner_cmidrules.append(rf"\cmidrule(lr){{{start}-{start + 2}}}")
        start += 3
    return [
        r"\begin{tabular}{l" + "c" * (2 * len(CRITERIA) * len(METRICS)) + "}",
        r"\toprule",
        "Region & " + " & ".join(split_headers) + r" \\",
        " ".join(cmidrules),
        " & " + " & ".join(criterion_headers) + r" \\",
        " ".join(inner_cmidrules),
        " & " + " & ".join(metric_headers) + r" \\",
        r"\midrule",
    ]


def render_rows() -> list[str]:
    rows: list[str] = []
    for label, regions in REGION_ROWS:
        if label == "ood":
            rows.extend(
                [
                    r"\addlinespace[0.25em]",
                    r"\hline",
                    r"\hline",
                    r"\addlinespace[0.25em]",
                ]
            )

        values_by_metric = {
            metric: [
                read_mean_for_experiment(experiment, criterion, regions, metric)
                for experiment in ("MEAN_CV", "CV")
                for criterion in CRITERIA
            ]
            for metric in METRICS
        }
        best_indices = {metric: best_index(metric, values) for metric, values in values_by_metric.items()}

        row = [label]
        metric_offsets = {metric: 0 for metric in METRICS}
        for experiment in ("MEAN_CV", "CV"):
            for criterion in CRITERIA:
                for metric in METRICS:
                    stats = read_stats_for_experiment(experiment, criterion, regions, metric)
                    row.append(
                        format_stats(
                            metric,
                            stats,
                            metric_offsets[metric] == best_indices[metric],
                        )
                    )
                    metric_offsets[metric] += 1

        rows.append(" & ".join(row) + r" \\")

    return rows


def render_table() -> str:
    lines = [
        r"\providecommand{\mstd}[3]{\shortstack{#1$^{#3}$\\[0.4ex]{\scriptsize$\pm$#2}}}",
        r"\providecommand{\bmstd}[3]{\shortstack{\textbf{#1}$^{#3}$\\[0.4ex]{\scriptsize\textbf{$\pm$#2}}}}",
        "",
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.2}",
        *render_header(),
        *render_rows(),
        r"\bottomrule",
        r"\end{tabular}",
        (
            r"\caption{Task 2 metrics for ELX, comparing the mean across CV folds (CV\_0 to CV\_4) "
            r"against CV for models trained with MAE or SAM. MAE is read from the standard CV evaluations, "
            r"while SAM and LPIPS are read from the perceptual CV evaluations. Rows report mean $\pm$ "
            r"standard deviation for matched patients. Best mean values per metric are shown in bold with "
            r"a single winner per row.}"
        ),
        r"\label{tab:task2_elx_fold_vs_cv_mae_median_perceptual}",
        r"\end{table}",
        r"\renewcommand{\arraystretch}{1}",
    ]
    return "\n".join(lines) + "\n"


def missing_files() -> list[Path]:
    missing: list[Path] = []
    for _, regions in REGION_ROWS:
        for region in regions:
            for metric in METRICS:
                for fold_dir in METRIC_FOLD_DIRS[metric]:
                    for criterion in CRITERIA:
                        path = metric_path(fold_dir, criterion, region)
                        if not path.exists():
                            missing.append(path)
                for criterion in CRITERIA:
                    path = metric_path(METRIC_CV_DIRS[metric], criterion, region)
                    if not path.exists():
                        missing.append(path)
    return missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Task 2 ELX mean-CV-vs-CV table for MAE, SAM, and LPIPS."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any required Metric_TRAIN.json file is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = missing_files()
    if missing:
        print(f"Missing {len(missing)} Metric_TRAIN.json file(s):")
        for path in missing:
            print(f"- {path}")
        if args.strict:
            raise SystemExit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render_table(), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
