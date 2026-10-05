#!/usr/bin/env python3
"""Generate the Task 2 per-region LaTeX tables for CV evaluations."""

from __future__ import annotations

import argparse
import json
import os
import math
from pathlib import Path

from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
# The paper's metrics by default; IMPACTSYNTH_EVALUATIONS=results/generated/Evaluations for those of a new run.
EVALUATIONS = ROOT / os.environ.get("IMPACTSYNTH_EVALUATIONS", "results/raw/Evaluations")
DEFAULT_EVALUATIONS_DIR = EVALUATIONS / "Task_2" / "Supervised"
DEFAULT_OUTPUT = ROOT / os.environ.get("IMPACTSYNTH_TABLES", "tables/from_metrics") / "table_15_task2_registration.tex"

IN_DISTRIBUTION_REGIONS = ("AB", "HN", "TH", "brain", "pelvis")
OUT_OF_DISTRIBUTION_REGIONS = ("AB_ood", "HN_ood", "TH_ood")
REGIONS = IN_DISTRIBUTION_REGIONS + OUT_OF_DISTRIBUTION_REGIONS
SUMMARY_REGIONS = (
    ("AB/HN/TH", ("AB", "HN", "TH")),
    ("brain/pelvis", ("brain", "pelvis")),
)
MODES = ("IMPACT", "ELX")
METRICS = ("MAE", "PSNR", "SSIM")
LOWER_IS_BETTER = {"MAE"}
OOD_REGIONS = {"AB_ood", "HN_ood", "TH_ood"}
CV_FOLD = "CV"

TABLE_SPECS = (
    {
        "name": "cv",
        "title": "CV",
        "output_label_suffix": "cv",
        "reference_columns": (
            ("IMPACT", "MAE_IMPACT"),
            ("ELX", "MAE"),
        ),
        "reference_mode": "IMPACT",
        "reference_criterion": "MAE_IMPACT",
    },
)


def region_members(region: str | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(region, tuple):
        return region
    return (region,)


def patient_count(
    evaluations_dir: Path,
    region: str | tuple[str, ...],
    reference_mode: str,
    reference_criterion: str,
) -> int | None:
    cases = read_patient_means(
        evaluations_dir,
        reference_mode,
        reference_criterion,
        region,
        METRICS[0],
    )
    if cases:
        return len(cases)
    return None


def region_label(
    evaluations_dir: Path,
    region: str | tuple[str, ...],
    reference_mode: str,
    reference_criterion: str,
    display_name: str | None = None,
) -> str:
    count = patient_count(evaluations_dir, region, reference_mode, reference_criterion)
    label = display_name if display_name is not None else "/".join(region_members(region))
    if count is None:
        return label
    return rf"{label} ($n={count}$)"


def metric_path(
    evaluations_dir: Path,
    mode: str,
    criterion: str,
    region: str,
) -> Path:
    return evaluations_dir / mode / CV_FOLD / criterion / region / "Metric_TRAIN.json"


def read_case_values(
    evaluations_dir: Path,
    mode: str,
    criterion: str,
    region: str | tuple[str, ...],
    metric: str,
) -> list[float]:
    return list(read_patient_means(evaluations_dir, mode, criterion, region, metric).values())


def read_case_dict(
    evaluations_dir: Path,
    mode: str,
    criterion: str,
    region: str,
    metric: str,
) -> dict[str, float]:
    path = metric_path(evaluations_dir, mode, criterion, region)
    if not path.exists():
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    key = f"sCT:CT;MASK:{metric}"
    return {patient: float(value) for patient, value in data["case"][key].items()}


def read_patient_means(
    evaluations_dir: Path,
    mode: str,
    criterion: str,
    region: str | tuple[str, ...],
    metric: str,
) -> dict[str, float]:
    patient_values: dict[str, list[float]] = {}
    for member in region_members(region):
        cases = read_case_dict(
            evaluations_dir,
            mode,
            criterion,
            member,
            metric,
        )
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


def read_stats(
    evaluations_dir: Path,
    mode: str,
    criterion: str,
    region: str | tuple[str, ...],
    metric: str,
) -> tuple[float, float] | None:
    return aggregate(read_case_values(evaluations_dir, mode, criterion, region, metric))


def read_mean(
    evaluations_dir: Path,
    mode: str,
    criterion: str,
    region: str | tuple[str, ...],
    metric: str,
) -> float | None:
    stats = read_stats(evaluations_dir, mode, criterion, region, metric)
    if stats is None:
        return None
    return stats[0]


def format_value(metric: str, value: float | None) -> str:
    if value is None:
        return "--"
    if metric == "SSIM":
        return f"{value:.3f}"
    return f"{value:.2f}"


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


def format_stats(
    metric: str,
    stats: tuple[float, float] | None,
    bold: bool,
    stars: str = "",
) -> str:
    if stats is None:
        return "--"
    mean, std = stats
    command = r"\bmstd" if bold else r"\mstd"
    return rf"{command}{{{format_value(metric, mean)}}}{{{format_value(metric, std)}}}{{{stars}}}"


def rounded_value(metric: str, value: float) -> float:
    return round(value, 3 if metric == "SSIM" else 2)


def is_best(metric: str, value: float | None, values: list[float | None]) -> bool:
    valid_values = [
        rounded_value(metric, candidate)
        for candidate in values
        if candidate is not None
    ]
    if value is None or not valid_values:
        return False

    best = min(valid_values) if metric in LOWER_IS_BETTER else max(valid_values)
    return rounded_value(metric, value) == best


def format_row_value(
    metric: str,
    stats: tuple[float, float] | None,
    values: list[float | None],
    stars: str = "",
) -> str:
    value = None if stats is None else stats[0]
    return format_stats(metric, stats, is_best(metric, value, values), stars)


def read_ood_mode_stats(
    evaluations_dir: Path,
    mode: str,
    region: str,
    metric: str,
    reference_columns: tuple[tuple[str, str], ...],
) -> tuple[float, float] | None:
    criterion_by_mode = dict(reference_columns)
    criterion = criterion_by_mode[mode]
    return read_stats(evaluations_dir, mode, criterion, region, metric)


def render_ood_mode_cell(
    evaluations_dir: Path,
    mode: str,
    region: str,
    row_values: dict[str, list[float | None]],
    reference_columns: tuple[tuple[str, str], ...],
) -> str:
    parts = []
    for metric in METRICS:
        stats = read_ood_mode_stats(
            evaluations_dir,
            mode,
            region,
            metric,
            reference_columns,
        )
        parts.append(f"{metric}: {format_row_value(metric, stats, row_values[metric])}")
    return "; ".join(parts)


def paired_wilcoxon(
    evaluations_dir: Path,
    region: str | tuple[str, ...],
    metric: str,
    mode: str,
    criterion: str,
    reference_mode: str,
    reference_criterion: str,
) -> tuple[int, float | None, float | None]:
    reference = read_patient_means(
        evaluations_dir,
        reference_mode,
        reference_criterion,
        region,
        metric,
    )
    candidate = read_patient_means(
        evaluations_dir,
        mode,
        criterion,
        region,
        metric,
    )

    reference_values: list[float] = []
    candidate_values: list[float] = []
    for patient in sorted(set(reference) & set(candidate)):
        reference_values.append(reference[patient])
        candidate_values.append(candidate[patient])

    if len(reference_values) < 2:
        return len(reference_values), None, None

    if all(abs(left - right) < 1e-12 for left, right in zip(reference_values, candidate_values)):
        return len(reference_values), 0.0, 1.0

    result = wilcoxon(reference_values, candidate_values)
    return len(reference_values), float(result.statistic), float(result.pvalue)


def comparison_stars(
    evaluations_dir: Path,
    region: str | tuple[str, ...],
    metric: str,
    mode: str,
    criterion: str,
    reference_mode: str,
    reference_criterion: str,
) -> str:
    if mode == reference_mode and criterion == reference_criterion:
        return ""
    _, _, p_value = paired_wilcoxon(
        evaluations_dir,
        region,
        metric,
        mode,
        criterion,
        reference_mode,
        reference_criterion,
    )
    return significance_stars(p_value)


def render_in_distribution_table(
    evaluations_dir: Path,
    regions: tuple[str, ...],
    caption: str,
    label: str,
    reference_columns: tuple[tuple[str, str], ...],
    reference_mode: str,
    reference_criterion: str,
) -> str:
    rows = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{1.25}",
        r"\begin{tabular}{lcccccccccccc}",
        r"\toprule",
        r" & \multicolumn{6}{c}{IMPACT} & \multicolumn{6}{c}{ELX} \\",
        r"\cmidrule(lr){2-7} \cmidrule(lr){8-13}",
        (
            r"Region & \multicolumn{3}{c}{IMPACT} "
            r"& \multicolumn{3}{c}{ELX} "
            r"& \multicolumn{3}{c}{IMPACT} "
            r"& \multicolumn{3}{c}{ELX} \\"
        ),
        (
            r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} "
            r"\cmidrule(lr){8-10} \cmidrule(lr){11-13}"
        ),
        (
            r" & MAE & PSNR & SSIM & MAE & PSNR & SSIM "
            r"& MAE & PSNR & SSIM & MAE & PSNR & SSIM \\"
        ),
        r"\midrule",
    ]

    for region in regions:
        if region in OOD_REGIONS:
            row_values = {
                metric: [
                    None
                    if (stats := read_ood_mode_stats(
                        evaluations_dir,
                        mode,
                        region,
                        metric,
                        reference_columns,
                    )) is None
                    else stats[0]
                    for mode in MODES
                ]
                for metric in METRICS
            }
            impact_cell = render_ood_mode_cell(
                evaluations_dir,
                "IMPACT",
                region,
                row_values,
                reference_columns,
            )
            elx_cell = render_ood_mode_cell(
                evaluations_dir,
                "ELX",
                region,
                row_values,
                reference_columns,
            )
            rows.append(
                f"{region_label(evaluations_dir, region, reference_mode, reference_criterion)} & "
                rf"\multicolumn{{6}}{{c}}{{{impact_cell}}} & "
                rf"\multicolumn{{6}}{{c}}{{{elx_cell}}} \\"
            )
            continue

        if region == "brain":
            rows.extend(
                [
                    r"\addlinespace[0.25em]",
                    r"\hline",
                    r"\hline",
                    r"\addlinespace[0.25em]",
                ]
            )

        row = [region_label(evaluations_dir, region, reference_mode, reference_criterion)]
        row_values = {
            metric: [
                read_mean(evaluations_dir, mode, criterion, region, metric)
                for mode in MODES
                for _, criterion in reference_columns
            ]
            for metric in METRICS
        }
        for mode in MODES:
            for _, criterion in reference_columns:
                for metric in METRICS:
                    stats = read_stats(evaluations_dir, mode, criterion, region, metric)
                    stars = comparison_stars(
                        evaluations_dir,
                        region,
                        metric,
                        mode,
                        criterion,
                        reference_mode,
                        reference_criterion,
                    )
                    row.append(format_row_value(metric, stats, row_values[metric], stars))
        rows.append(" & ".join(row) + r" \\")

        if region == "TH":
            rows.extend(
                render_summary_rows(
                    evaluations_dir,
                    (SUMMARY_REGIONS[0],),
                    reference_columns,
                    reference_mode,
                    reference_criterion,
                )
            )

        if region == "pelvis":
            rows.extend(
                render_summary_rows(
                    evaluations_dir,
                    (SUMMARY_REGIONS[1],),
                    reference_columns,
                    reference_mode,
                    reference_criterion,
                )
            )

    rows.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
            r"\renewcommand{\arraystretch}{1}",
        ]
    )
    return "\n".join(rows) + "\n"


def render_summary_rows(
    evaluations_dir: Path,
    summaries: tuple[tuple[str, tuple[str, ...]], ...],
    reference_columns: tuple[tuple[str, str], ...],
    reference_mode: str,
    reference_criterion: str,
) -> list[str]:
    rows: list[str] = []
    for label, members in summaries:
        row = [region_label(evaluations_dir, members, reference_mode, reference_criterion, label)]
        row_values = {
            metric: [
                read_mean(evaluations_dir, mode, criterion, members, metric)
                for mode in MODES
                for _, criterion in reference_columns
            ]
            for metric in METRICS
        }
        for mode in MODES:
            for _, criterion in reference_columns:
                for metric in METRICS:
                    stats = read_stats(evaluations_dir, mode, criterion, members, metric)
                    stars = comparison_stars(
                        evaluations_dir,
                        members,
                        metric,
                        mode,
                        criterion,
                        reference_mode,
                        reference_criterion,
                    )
                    row.append(format_row_value(metric, stats, row_values[metric], stars))
        rows.append(" & ".join(row) + r" \\")
    return rows


def render_out_of_distribution_table(
    evaluations_dir: Path,
    regions: tuple[str, ...],
    caption: str,
    label: str,
    reference_columns: tuple[tuple[str, str], ...],
    reference_mode: str,
    reference_criterion: str,
) -> str:
    rows = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{IMPACT} & \multicolumn{3}{c}{ELX} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"Region & MAE & PSNR & SSIM & MAE & PSNR & SSIM \\",
        r"\midrule",
    ]

    for region in regions:
        row = [region_label(evaluations_dir, region, reference_mode, reference_criterion)]
        row_values = {
            metric: [
                None
                if (stats := read_ood_mode_stats(
                    evaluations_dir,
                    mode,
                    region,
                    metric,
                    reference_columns,
                )) is None
                else stats[0]
                for mode in MODES
            ]
            for metric in METRICS
        }
        for mode in MODES:
            for metric in METRICS:
                stats = read_ood_mode_stats(
                    evaluations_dir, mode, region, metric, reference_columns
                )
                stars = ""
                if mode != reference_mode:
                    stars = comparison_stars(
                        evaluations_dir,
                        region,
                        metric,
                        mode,
                        dict(reference_columns)[mode],
                        reference_mode,
                        reference_criterion,
                    )
                row.append(format_row_value(metric, stats, row_values[metric], stars))
        rows.append(" & ".join(row) + r" \\")

    rows.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
        ]
    )
    return "\n".join(rows) + "\n"


def render_table(evaluations_dir: Path) -> str:
    macros = "\n".join(
        [
            r"\providecommand{\mstd}[3]{\shortstack{#1$^{#3}$\\[0.4ex]{\scriptsize$\pm$#2}}}",
            r"\providecommand{\bmstd}[3]{\shortstack{\textbf{#1}$^{#3}$\\[0.4ex]{\scriptsize\textbf{$\pm$#2}}}}",
            "",
        ]
    )
    tables: list[str] = []
    for spec in TABLE_SPECS:
        shared_caption_suffix = (
            r" Values report mean $\pm$ standard deviation. Column groups indicate "
            r"the registration method used to train the images (IMPACT or ELX), "
            r"and subgroups indicate the registration method used for evaluation. "
            r"Best mean values per row are shown in bold. Stars indicate paired "
            rf"Wilcoxon tests against {spec['reference_mode']}/{spec['reference_criterion']} "
            r"on matched patients (* $p<0.05$, ** $p<0.01$, *** $p<0.001$)."
        )
        in_distribution = render_in_distribution_table(
            evaluations_dir,
            IN_DISTRIBUTION_REGIONS,
            (
                rf"Mean evaluation metrics by in-distribution region for Task 2 supervised {spec['title']} "
                r"experiments."
                + shared_caption_suffix
            ),
            f"tab:task2_{spec['output_label_suffix']}_region_metrics_in_distribution",
            spec["reference_columns"],
            spec["reference_mode"],
            spec["reference_criterion"],
        )
        out_of_distribution = render_out_of_distribution_table(
            evaluations_dir,
            OUT_OF_DISTRIBUTION_REGIONS,
            (
                rf"Mean evaluation metrics by out-of-distribution region for Task 2 supervised {spec['title']} "
                r"experiments. Values report mean $\pm$ standard deviation. Column groups "
                r"indicate the registration method (IMPACT or ELX). Best mean values per row "
                r"are shown in bold. Stars indicate paired Wilcoxon tests against IMPACT on "
                r"matched patients (* $p<0.05$, ** $p<0.01$, *** $p<0.001$)."
            ),
            f"tab:task2_{spec['output_label_suffix']}_region_metrics_ood",
            spec["reference_columns"],
            spec["reference_mode"],
            spec["reference_criterion"],
        )
        tables.append(in_distribution + "\n" + out_of_distribution)
    return macros + "\n\n".join(tables)


def missing_files(evaluations_dir: Path) -> list[Path]:
    missing: list[Path] = []
    for spec in TABLE_SPECS:
        reference_columns = spec["reference_columns"]
        for region in REGIONS:
            for mode in MODES:
                for _, criterion in reference_columns:
                    path = metric_path(evaluations_dir, mode, criterion, region)
                    if not path.exists():
                        missing.append(path)
    return missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Task 2 per-region LaTeX table from CV Metric_TRAIN.json files."
    )
    parser.add_argument(
        "--evaluations-dir",
        type=Path,
        default=DEFAULT_EVALUATIONS_DIR,
        help=f"Base directory containing Task 2 evaluation outputs (default: {DEFAULT_EVALUATIONS_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination .tex file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any required Metric_TRAIN.json file is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = missing_files(args.evaluations_dir)
    if missing:
        print(f"Missing {len(missing)} Metric_TRAIN.json file(s):")
        for path in missing:
            print(f"- {path}")
        if args.strict:
            raise SystemExit(1)

    table = render_table(args.evaluations_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
