#!/usr/bin/env python3
"""Generate Task 1/Task 2 combined CV tables for IMPACT MAE_IMPACT vs SAM_IMPACT."""

from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import json
import os
import math
from pathlib import Path

from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
# The paper's metrics by default; IMPACTSYNTH_EVALUATIONS=results/generated/Evaluations for those of a new run.
EVALUATIONS = ROOT / os.environ.get("IMPACTSYNTH_EVALUATIONS", "results/raw/Evaluations")
MODE = "IMPACT"
CRITERIA = ("MAE_IMPACT", "VGG_IMPACT", "SAM_IMPACT")
CV_FOLD = "CV"
CV_SEG_FOLD = "CV_SEG"
IMAGE_METRICS = ("SSIM",)
ALL_METRICS = ("Dice", "SSIM")
REFERENCE_CRITERION = "SAM_IMPACT"
MIN_SHARED_DICE = 0.1

TASK_SPECS = (
    {
        "task_title": "Task 1",
        "evaluations_dir": EVALUATIONS / "Task_1" / "Supervised",
        "output": ROOT / os.environ.get("IMPACTSYNTH_TABLES", "tables/from_metrics") / "table_16_task1_sam.tex",
        "in_regions": ("AB", "HN", "TH", "brain", "pelvis"),
        "ood_regions": ("ood",),
        "summary_regions": (
            ("AB/HN/TH", ("AB", "HN", "TH")),
            ("brain/pelvis", ("brain", "pelvis")),
        ),
        "label_prefix": "task1_cv_impact_mae_vs_sam_combined",
        "min_shared_dice": MIN_SHARED_DICE,
    },
    {
        "task_title": "Task 2",
        "evaluations_dir": EVALUATIONS / "Task_2" / "Supervised",
        "output": ROOT / os.environ.get("IMPACTSYNTH_TABLES", "tables/from_metrics") / "table_17_task2_sam.tex",
        "in_regions": ("AB", "HN", "TH", "brain", "pelvis"),
        "ood_regions": ("AB_ood", "HN_ood", "TH_ood"),
        "summary_regions": (
            ("AB/HN/TH", ("AB", "HN", "TH")),
            ("brain/pelvis", ("brain", "pelvis")),
        ),
        "label_prefix": "task2_cv_impact_mae_vs_sam_combined",
        "min_shared_dice": None,
    },
)


def region_members(region: str | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(region, tuple):
        return region
    return (region,)


def image_metric_path(evaluations_dir: Path, criterion: str, region: str) -> Path:
    return evaluations_dir / MODE / CV_FOLD / criterion / region / "Metric_TRAIN.json"


def seg_metric_path(evaluations_dir: Path, criterion: str, region: str) -> Path:
    return evaluations_dir / MODE / CV_SEG_FOLD / criterion / region / "Metric_TRAIN.json"


@lru_cache(maxsize=None)
def read_metric_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_value(metric: str, value: float | None) -> str:
    if value is None:
        return "--"
    if metric in {"Dice", "SSIM"}:
        return f"{value:.3f}"
    return f"{value:.2f}"


def latex_label(text: str) -> str:
    return text.replace("_", r"\_")


def significance_stars(p_value: float | None) -> str:
    if p_value is None:
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def rounded_value(metric: str, value: float) -> float:
    return round(value, 3 if metric in {"Dice", "SSIM"} else 2)


def is_best(metric: str, value: float | None, values: list[float | None]) -> bool:
    valid_values = [rounded_value(metric, candidate) for candidate in values if candidate is not None]
    if value is None or not valid_values:
        return False
    if metric == "MAE":
        return rounded_value(metric, value) == min(valid_values)
    return rounded_value(metric, value) == max(valid_values)


def format_stats(metric: str, stats: tuple[float, float] | None, bold: bool, stars: str = "") -> str:
    if stats is None:
        return "--"
    mean, std = stats
    command = r"\bmstd" if bold else r"\mstd"
    return rf"{command}{{{format_value(metric, mean)}}}{{{format_value(metric, std)}}}{{{stars}}}"


def aggregate(values: list[float]) -> tuple[float, float] | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def is_nan_value(value: object) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


@lru_cache(maxsize=None)
def label_values_for_seg_file(
    evaluations_dir: Path,
    criterion: str,
    region: str,
) -> dict[str, dict[str, float]]:
    path = seg_metric_path(evaluations_dir, criterion, region)
    if not path.exists():
        return {}
    data = read_metric_file(path)
    result: dict[str, dict[str, float]] = defaultdict(dict)
    dice_prefixes = ("sCT_Seg:CT_Seg;MASK:Dice:", "MR_Seg:CT_Seg;MASK:Dice:")
    for key, patients in data.get("case", {}).items():
        matched_prefix = next((prefix for prefix in dice_prefixes if key.startswith(prefix)), None)
        if matched_prefix is None:
            continue
        label_name = key.removeprefix(matched_prefix)
        for patient, value in patients.items():
            if value is None or is_nan_value(value):  # a label absent for this patient (null in JSON)
                continue
            result[patient][label_name] = float(value)
    return {patient: labels for patient, labels in result.items()}


@lru_cache(maxsize=None)
def shared_seg_labels_for_region(
    evaluations_dir: Path,
    region: str,
    min_shared_dice: float | None,
) -> dict[str, frozenset[str]]:
    criterion_patient_labels = {
        criterion: label_values_for_seg_file(evaluations_dir, criterion, region)
        for criterion in CRITERIA
    }
    shared_labels: dict[str, set[str]] | None = None
    for criterion in CRITERIA:
        patient_labels = criterion_patient_labels[criterion]
        if not patient_labels:
            continue
        file_shared = {patient: set(labels) for patient, labels in patient_labels.items()}
        if shared_labels is None:
            shared_labels = file_shared
        else:
            next_shared: dict[str, set[str]] = {}
            for patient in set(shared_labels) & set(file_shared):
                common = shared_labels[patient] & file_shared[patient]
                if common:
                    next_shared[patient] = common
            shared_labels = next_shared
    if shared_labels is None:
        return {}
    filtered_shared: dict[str, frozenset[str]] = {}
    for patient, labels in shared_labels.items():
        valid_labels = []
        for label in labels:
            if all(
                patient in criterion_patient_labels[criterion]
                and label in criterion_patient_labels[criterion][patient]
                and (
                    min_shared_dice is None
                    or criterion_patient_labels[criterion][patient][label] > min_shared_dice
                )
                for criterion in CRITERIA
            ):
                valid_labels.append(label)
        if valid_labels:
            filtered_shared[patient] = frozenset(valid_labels)
    return filtered_shared


def read_image_case_dict(
    evaluations_dir: Path,
    criterion: str,
    region: str,
    metric: str,
) -> dict[str, float]:
    path = image_metric_path(evaluations_dir, criterion, region)
    if not path.exists():
        return {}
    data = read_metric_file(path)
    key = f"sCT:CT;MASK:{metric}"
    return {patient: float(value) for patient, value in data["case"][key].items()}


def read_seg_case_dict(
    evaluations_dir: Path,
    criterion: str,
    region: str,
    min_shared_dice: float | None,
) -> dict[str, float]:
    path = seg_metric_path(evaluations_dir, criterion, region)
    if not path.exists():
        return {}
    patient_labels = label_values_for_seg_file(evaluations_dir, criterion, region)
    shared_labels = shared_seg_labels_for_region(evaluations_dir, region, min_shared_dice)
    result: dict[str, float] = {}
    for patient, labels in shared_labels.items():
        values = [
            patient_labels[patient][label]
            for label in labels
            if patient in patient_labels and label in patient_labels[patient]
        ]
        if values:
            result[patient] = sum(values) / len(values)
    return result


def read_case_dict(
    evaluations_dir: Path,
    criterion: str,
    region: str,
    metric: str,
    min_shared_dice: float | None,
) -> dict[str, float]:
    if metric == "Dice":
        return read_seg_case_dict(evaluations_dir, criterion, region, min_shared_dice)
    return read_image_case_dict(evaluations_dir, criterion, region, metric)


def read_patient_means(
    evaluations_dir: Path,
    criterion: str,
    region: str | tuple[str, ...],
    metric: str,
    min_shared_dice: float | None,
) -> dict[str, float]:
    patient_values: dict[str, list[float]] = {}
    for member in region_members(region):
        cases = read_case_dict(evaluations_dir, criterion, member, metric, min_shared_dice)
        for patient, value in cases.items():
            patient_values.setdefault(patient, []).append(value)
    return {patient: sum(values) / len(values) for patient, values in patient_values.items()}


def read_case_values(
    evaluations_dir: Path,
    criterion: str,
    region: str | tuple[str, ...],
    metric: str,
    min_shared_dice: float | None,
) -> list[float]:
    return list(read_patient_means(evaluations_dir, criterion, region, metric, min_shared_dice).values())


def read_stats(
    evaluations_dir: Path,
    criterion: str,
    region: str | tuple[str, ...],
    metric: str,
    min_shared_dice: float | None,
) -> tuple[float, float] | None:
    return aggregate(read_case_values(evaluations_dir, criterion, region, metric, min_shared_dice))


def read_mean(
    evaluations_dir: Path,
    criterion: str,
    region: str | tuple[str, ...],
    metric: str,
    min_shared_dice: float | None,
) -> float | None:
    stats = read_stats(evaluations_dir, criterion, region, metric, min_shared_dice)
    if stats is None:
        return None
    return stats[0]


def patient_count(
    evaluations_dir: Path,
    region: str | tuple[str, ...],
    min_shared_dice: float | None,
) -> int | None:
    cases = read_patient_means(evaluations_dir, REFERENCE_CRITERION, region, "SSIM", min_shared_dice)
    if cases:
        return len(cases)
    return None


def region_label(
    evaluations_dir: Path,
    region: str | tuple[str, ...],
    min_shared_dice: float | None,
    display_name: str | None = None,
) -> str:
    label = display_name if display_name is not None else "/".join(region_members(region))
    count = patient_count(evaluations_dir, region, min_shared_dice)
    if count is None:
        return label
    return rf"{label} ($n={count}$)"


def paired_wilcoxon(
    evaluations_dir: Path,
    reference_criterion: str,
    candidate_criterion: str,
    region: str | tuple[str, ...],
    metric: str,
    min_shared_dice: float | None,
) -> float | None:
    reference = read_patient_means(
        evaluations_dir, reference_criterion, region, metric, min_shared_dice
    )
    candidate = read_patient_means(
        evaluations_dir, candidate_criterion, region, metric, min_shared_dice
    )
    paired_patients = sorted(set(reference) & set(candidate))
    if len(paired_patients) < 2:
        return None
    reference_values = [reference[p] for p in paired_patients]
    candidate_values = [candidate[p] for p in paired_patients]
    if all(abs(left - right) < 1e-12 for left, right in zip(reference_values, candidate_values)):
        return 1.0
    result = wilcoxon(reference_values, candidate_values)
    return float(result.pvalue)


def render_header() -> list[str]:
    group_columns = len(ALL_METRICS)
    alignment = "l" + ("c" * (len(CRITERIA) * group_columns))
    header_groups = " & ".join(
        rf"\multicolumn{{{group_columns}}}{{c}}{{{latex_label(criterion)}}}"
        for criterion in CRITERIA
    )
    cmidrules = " ".join(
        rf"\cmidrule(lr){{{start}-{start + group_columns - 1}}}"
        for start in range(2, 2 + len(CRITERIA) * group_columns, group_columns)
    )
    metric_headers = " & ".join(ALL_METRICS)
    return [
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        f" & {header_groups} \\\\",
        cmidrules,
        rf"Region & {' & '.join(metric_headers for _ in CRITERIA)} \\",
        r"\midrule",
    ]


def render_row(
    evaluations_dir: Path,
    region: str | tuple[str, ...],
    min_shared_dice: float | None,
    display_name: str | None = None,
) -> str:
    row = [region_label(evaluations_dir, region, min_shared_dice, display_name)]
    row_values = {
        metric: [
            read_mean(evaluations_dir, criterion, region, metric, min_shared_dice)
            for criterion in CRITERIA
        ]
        for metric in ALL_METRICS
    }
    for criterion in CRITERIA:
        for metric in ALL_METRICS:
            stats = read_stats(evaluations_dir, criterion, region, metric, min_shared_dice)
            stars = ""
            if criterion != REFERENCE_CRITERION:
                stars = significance_stars(
                    paired_wilcoxon(
                        evaluations_dir,
                        REFERENCE_CRITERION,
                        criterion,
                        region,
                        metric,
                        min_shared_dice,
                    )
                )
            value = None if stats is None else stats[0]
            row.append(
                format_stats(
                    metric,
                    stats,
                    is_best(metric, value, row_values[metric]),
                    stars,
                )
            )
    return " & ".join(row) + r" \\"


def render_combined_table(spec: dict[str, object]) -> str:
    evaluations_dir = Path(spec["evaluations_dir"])
    min_shared_dice = spec["min_shared_dice"]
    rows = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.2}",
        *render_header(),
    ]
    summary_regions = tuple(spec["summary_regions"])
    for region in tuple(spec["in_regions"]):
        if region == "brain":
            rows.extend(
                [
                    r"\addlinespace[0.25em]",
                    r"\hline",
                    r"\hline",
                    r"\addlinespace[0.25em]",
                ]
            )
        rows.append(render_row(evaluations_dir, region, min_shared_dice))
        if region == "TH":
            rows.append(
                render_row(
                    evaluations_dir, summary_regions[0][1], min_shared_dice, summary_regions[0][0]
                )
            )
        if region == "pelvis":
            rows.append(
                render_row(
                    evaluations_dir, summary_regions[1][1], min_shared_dice, summary_regions[1][0]
                )
            )

    if tuple(spec["ood_regions"]):
        rows.extend(
            [
                r"\addlinespace[0.25em]",
                r"\hline",
                r"\hline",
                r"\addlinespace[0.25em]",
            ]
        )
        for region in tuple(spec["ood_regions"]):
            rows.append(render_row(evaluations_dir, region, min_shared_dice))

    rows.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                rf"\caption{{Combined CV metrics by region for {spec['task_title']} "
                rf"supervised IMPACT experiments, comparing "
                rf"{', '.join(latex_label(criterion) for criterion in CRITERIA[:-1])} "
                rf"and {latex_label(CRITERIA[-1])}. "
                r"Dice is read from CV\_SEG and averaged per patient over labels shared across "
                r"the compared criteria; Task 1 also requires Dice $> 0.1$ for all compared criteria, "
                r"while Task 2 keeps the shared-label rule without that threshold. SSIM is read from CV. "
                r"Values report mean $\pm$ standard deviation. Best mean values per metric are "
                r"shown in bold. Stars indicate paired Wilcoxon tests against "
                rf"{latex_label(REFERENCE_CRITERION)} on matched patients "
                r"(* $p<0.05$, ** $p<0.01$, *** $p<0.001$).}"
            ),
            rf"\label{{tab:{spec['label_prefix']}}}",
            r"\end{table}",
            r"\renewcommand{\arraystretch}{1}",
        ]
    )
    return "\n".join(rows) + "\n"


def render_task_tables(spec: dict[str, object]) -> str:
    macros = "\n".join(
        [
            r"\providecommand{\mstd}[3]{\shortstack{#1$^{#3}$\\[0.4ex]{\scriptsize$\pm$#2}}}",
            r"\providecommand{\bmstd}[3]{\shortstack{\textbf{#1}$^{#3}$\\[0.4ex]{\scriptsize\textbf{$\pm$#2}}}}",
            "",
        ]
    )
    return macros + render_combined_table(spec)


def missing_files(spec: dict[str, object]) -> list[Path]:
    evaluations_dir = Path(spec["evaluations_dir"])
    missing: list[Path] = []
    regions = tuple(spec["in_regions"]) + tuple(spec["ood_regions"])
    for region in regions:
        for criterion in CRITERIA:
            for path_fn in (image_metric_path, seg_metric_path):
                path = path_fn(evaluations_dir, criterion, region)
                if not path.exists():
                    missing.append(path)
    return missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Task 1/Task 2 combined CV IMPACT MAE vs SAM tables."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any required Metric_TRAIN.json file is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for spec in TASK_SPECS:
        missing = missing_files(spec)
        if missing:
            print(f"Missing {len(missing)} Metric_TRAIN.json file(s) for {spec['task_title']}:")
            for path in missing:
                print(f"- {path}")
            if args.strict:
                raise SystemExit(1)
        output = Path(spec["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_task_tables(spec), encoding="utf-8")
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
