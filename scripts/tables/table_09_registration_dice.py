#!/usr/bin/env python3
"""Generate the LaTeX table for Registration_ELX_IMPACT_SEG (the Dice of the Table 9 Reg. rows)."""

from __future__ import annotations

import json
import os
import math
from collections import defaultdict
from pathlib import Path

import table_16_17_sam as combined_seg_logic

ROOT = Path(__file__).resolve().parents[2]
# The paper's metrics by default; IMPACTSYNTH_EVALUATIONS=results/generated/Evaluations for those of a new run.
EVALUATIONS = ROOT / os.environ.get("IMPACTSYNTH_EVALUATIONS", "results/raw/Evaluations")
EVAL_DIR = EVALUATIONS / "Registration_ELX_IMPACT_SEG"
OUTPUT_PATH = ROOT / os.environ.get("IMPACTSYNTH_TABLES", "tables/from_metrics") / "table_09_registration_dice.tex"

REGION_GROUPS = (
    ("Task 1: AB / HN / TH", "Task_1", ("AB", "HN", "TH"), "AB/HN/TH overall"),
    ("Task 1: brain / pelvis", "Task_1", ("brain", "pelvis"), "brain/pelvis overall"),
    ("Task 2: AB / HN / TH", "Task_2", ("AB", "HN", "TH"), "AB/HN/TH overall"),
    ("Task 2: brain / pelvis", "Task_2", ("brain", "pelvis"), "brain/pelvis overall"),
)
TASK_CONFIG = {
    "Task_1": {
        "supervised_evaluations_dir": EVALUATIONS / "Task_1" / "Supervised",
        "min_shared_dice": 0.1,
    },
    "Task_2": {
        "supervised_evaluations_dir": EVALUATIONS / "Task_2" / "Supervised",
        "min_shared_dice": None,
    },
}


def is_nan_value(value: object) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def read_metric(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def patient_means_from_metric(
    path: Path,
    shared_labels: dict[str, frozenset[str]],
) -> dict[str, float]:
    data = read_metric(path)
    per_patient_labels: dict[str, dict[str, float]] = defaultdict(dict)
    prefix = "CT_deformed_Seg:CT_Seg:Dice:"

    for key, patients in data.get("case", {}).items():
        if not key.startswith(prefix):
            continue
        label = key.removeprefix(prefix)
        for patient, value in patients.items():
            if value is None or is_nan_value(value):  # a label absent for this patient (null in JSON)
                continue
            per_patient_labels[patient][label] = float(value)

    result: dict[str, float] = {}
    for patient, labels in shared_labels.items():
        values = [
            per_patient_labels[patient][label]
            for label in labels
            if patient in per_patient_labels and label in per_patient_labels[patient]
        ]
        if values:
            result[patient] = sum(values) / len(values)
    return result


def aggregate(values: list[float]) -> tuple[float, float, int] | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance), len(values)


def shared_labels_for_task_region(task_name: str, region: str) -> dict[str, frozenset[str]]:
    task_cfg = TASK_CONFIG[task_name]
    return combined_seg_logic.shared_seg_labels_for_region(
        task_cfg["supervised_evaluations_dir"],
        region,
        task_cfg["min_shared_dice"],
    )


def region_stats(task_name: str, task_dir: Path, region: str) -> tuple[float, float, int] | None:
    metric_path = task_dir / region / "Metric_TRAIN.json"
    if not metric_path.exists():
        return None
    shared_labels = shared_labels_for_task_region(task_name, region)
    case_values = list(patient_means_from_metric(metric_path, shared_labels).values())
    return aggregate(case_values)


def fmt_stats(stats: tuple[float, float, int] | None) -> tuple[str, str]:
    if stats is None:
        return "--", "--"
    mean, std, _ = stats
    return f"{mean:.3f}", f"{std:.3f}"


def render_table() -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Registration\_ELX\_IMPACT segmentation results on the train split. Dice is computed per patient as the mean over the same retained labels as in the combined CV IMPACT MAE/VGG/SAM comparison: Task 1 uses labels shared across criteria with Dice $> 0.1$ in all compared criteria, while Task 2 uses only labels shared across criteria. Values are then summarized as mean $\pm$ std across patients.}",
        r"\label{tab:registration-elx-impact-seg}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Task & Region & $N$ & Dice $\uparrow$ \\",
        r"\midrule",
    ]

    for section_title, task_name, regions, overall_label in REGION_GROUPS:
        task_dir = EVAL_DIR / task_name
        lines.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{section_title}}}}} \\")
        section_values: list[float] = []
        section_count = 0

        for region in regions:
            stats = region_stats(task_name, task_dir, region)
            mean_str, std_str = fmt_stats(stats)
            count = 0 if stats is None else stats[2]
            pretty_task = task_name.replace("_", " ")
            lines.append(rf"{pretty_task} & {region} & {count} & ${mean_str} \pm {std_str}$ \\")
            if stats is not None:
                metric_path = task_dir / region / "Metric_TRAIN.json"
                shared_labels = shared_labels_for_task_region(task_name, region)
                section_values.extend(patient_means_from_metric(metric_path, shared_labels).values())
                section_count += count

        overall = aggregate(section_values)
        overall_mean, overall_std = fmt_stats(overall)
        pretty_task = task_name.replace("_", " ")
        lines.append(rf"{pretty_task} & {overall_label} & {section_count} & ${overall_mean} \pm {overall_std}$ \\")
        if (section_title, task_name, regions, overall_label) != REGION_GROUPS[-1]:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    table = render_table()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(table, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
