#!/usr/bin/env python3
"""Generate a single LaTeX table with ELX vs IMPACT percentage deltas for Task 1 and Task 2."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
# The paper's metrics by default; IMPACTSYNTH_EVALUATIONS=results/generated/Evaluations for those of a new run.
EVALUATIONS = ROOT / os.environ.get("IMPACTSYNTH_EVALUATIONS", "results/raw/Evaluations")
METRIC_KEY = "Uncertainty:None:Uncertainty"
CV_FOLD = "CV_Uncertainty"
CRITERION = "MAE"
MODES = ("IMPACT", "ELX")
OUTPUT = ROOT / os.environ.get("IMPACTSYNTH_TABLES", "tables/from_metrics") / "table_04_uncertainty.tex"

TASK_SPECS = {
    "Task 1": {
        "evaluations_dir": EVALUATIONS / "Task_1" / "Supervised",
        "rows": (
            ("AB", ("AB",)),
            ("HN", ("HN",)),
            ("TH", ("TH",)),
            ("AB/HN/TH", ("AB", "HN", "TH")),
            ("brain", ("brain",)),
            ("pelvis", ("pelvis",)),
            ("brain/pelvis", ("brain", "pelvis")),
            ("ood", ("ood",)),
        ),
    },
    "Task 2": {
        "evaluations_dir": EVALUATIONS / "Task_2" / "Supervised",
        "rows": (
            ("AB", ("AB",)),
            ("HN", ("HN",)),
            ("TH", ("TH",)),
            ("AB/HN/TH", ("AB", "HN", "TH")),
            ("brain", ("brain",)),
            ("pelvis", ("pelvis",)),
            ("brain/pelvis", ("brain", "pelvis")),
            ("ood", ("AB_ood", "HN_ood", "TH_ood")),
        ),
    },
}

ROW_ORDER = (
    "AB",
    "HN",
    "TH",
    "AB/HN/TH",
    "brain",
    "pelvis",
    "brain/pelvis",
    "ood",
)


def metric_path(evaluations_dir: Path, mode: str, region: str) -> Path:
    return evaluations_dir / mode / CV_FOLD / CRITERION / region / "Metric_TRAIN.json"


def read_case_dict(evaluations_dir: Path, mode: str, region: str) -> dict[str, float]:
    path = metric_path(evaluations_dir, mode, region)
    if not path.is_file():  # e.g. the external sets, which are not public
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {patient: float(value) for patient, value in data["case"][METRIC_KEY].items()}


def read_patient_means(
    evaluations_dir: Path,
    mode: str,
    regions: tuple[str, ...],
) -> dict[str, float]:
    patient_values: dict[str, list[float]] = {}
    for region in regions:
        cases = read_case_dict(evaluations_dir, mode, region)
        for patient, value in cases.items():
            patient_values.setdefault(patient, []).append(value)

    return {
        patient: sum(values) / len(values)
        for patient, values in patient_values.items()
    }


def read_mean(evaluations_dir: Path, mode: str, regions: tuple[str, ...]) -> float | None:
    patient_means = list(read_patient_means(evaluations_dir, mode, regions).values())
    if not patient_means:
        return None
    return sum(patient_means) / len(patient_means)


def percent_change(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None or abs(reference) < 1e-12:
        return None
    return ((candidate - reference) / reference) * 100.0


def format_percent(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:+.1f}\\%"


def build_task_values(task_name: str) -> dict[str, float | None]:
    spec = TASK_SPECS[task_name]
    evaluations_dir = Path(spec["evaluations_dir"])
    values: dict[str, float | None] = {}
    for label, regions in spec["rows"]:
        impact_mean = read_mean(evaluations_dir, "IMPACT", regions)
        elx_mean = read_mean(evaluations_dir, "ELX", regions)
        values[label] = percent_change(impact_mean, elx_mean)
    return values


def render_table() -> str:
    task1_values = build_task_values("Task 1")
    task2_values = build_task_values("Task 2")

    rows = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Region & Task 1 & Task 2 \\",
        r"\midrule",
    ]

    for label in ROW_ORDER:
        if label == "brain":
            rows.extend(
                [
                    r"\addlinespace[0.25em]",
                    r"\hline",
                    r"\hline",
                    r"\addlinespace[0.25em]",
                ]
            )
        rows.append(
            f"{label} & {format_percent(task1_values[label])} & {format_percent(task2_values[label])} \\\\"
        )

    rows.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Relative change of ELX versus IMPACT for Uncertainty with the MAE criterion, computed as $(ELX-IMPACT)/IMPACT \times 100$. Positive values mean ELX is higher; negative values mean ELX is lower. The \texttt{ood} row is included in the same table; for Task 2 it aggregates AB\_ood, HN\_ood, and TH\_ood.}",
            r"\label{tab:cv_uncertainty_mae_percentage}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(rows)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render_table(), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
