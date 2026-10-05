#!/usr/bin/env python3
"""Evaluate CT_ELX warped by IMPACT plus small coarse perturbations.

This script:
1. reads `CT.mha`, `CT_ELX.mha`, and `MASK.mha`,
2. materializes the IMPACT dense deformation field on the CT grid,
3. optionally adds small coarse random perturbations to that field,
4. resamples `CT_ELX` with the perturbed field,
5. computes masked MAE / PSNR / SSIM against `CT`.

It is meant to study robustness when the IMPACT registration is not a perfect
ground truth. The default perturbations are intentionally small and fast to
compute, using a coarse random displacement lattice interpolated to the full
grid.

With `--samples-per-level 0 --save-images` only the baseline is computed: CT_ELX
warped by IMPACT, the `CT_deformed` of the Table 9 controls;
assemble_registration_dataset.py turns it into a KonfAI dataset.

Important:
- run this script with the separate preprocessing environment, because it needs
  `ReadParameterFile` and `TransformixImageFilter`.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parents[2]


DATA_RANGE = 4095.0
DEFAULT_OUTPUT_DIR = ROOT / "results" / "generated" / "Registration" / "CT_ELX_IMPACT_Perturbations"
# the held-out cases
TASKS_TO_PATIENTS = {task: set((ROOT / f"data/splits/task_{task}/Validation.txt").read_text().split())
                     for task in (1, 2)}
REGION_SPECS = (
    (1, "2025", "AB", ROOT / "data" / "processed" / "synthrad2025" / "Task_1" / "AB"),
    (1, "2025", "HN", ROOT / "data" / "processed" / "synthrad2025" / "Task_1" / "HN"),
    (1, "2025", "TH", ROOT / "data" / "processed" / "synthrad2025" / "Task_1" / "TH"),
    (1, "2023", "brain", ROOT / "data" / "processed" / "synthrad2023" / "Task_1" / "brain"),
    (1, "2023", "pelvis", ROOT / "data" / "processed" / "synthrad2023" / "Task_1" / "pelvis"),
    (2, "2025", "AB", ROOT / "data" / "processed" / "synthrad2025" / "Task_2" / "AB"),
    (2, "2025", "HN", ROOT / "data" / "processed" / "synthrad2025" / "Task_2" / "HN"),
    (2, "2025", "TH", ROOT / "data" / "processed" / "synthrad2025" / "Task_2" / "TH"),
    (2, "2023", "brain", ROOT / "data" / "processed" / "synthrad2023" / "Task_2" / "brain"),
    (2, "2023", "pelvis", ROOT / "data" / "processed" / "synthrad2023" / "Task_2" / "pelvis"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CT_ELX warped by IMPACT plus small smooth deformation perturbations."
    )
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="+",
        choices=(1, 2),
        default=(1,),
        help="Tasks to include (default: 1).",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        choices=("AB", "HN", "TH", "brain", "pelvis"),
        default=("HN",),
        help="Regions to include (default: HN).",
    )
    parser.add_argument(
        "--patients",
        nargs="*",
        default=None,
        help="Optional patient subset. Default: the provided Task 1 / Task 2 lists.",
    )
    parser.add_argument(
        "--samples-per-level",
        type=int,
        default=1,
        help="Number of random perturbed samples per patient and per level, excluding baseline (default: 1).",
    )
    parser.add_argument(
        "--perturbation-levels-mm",
        type=float,
        nargs="+",
        default=(0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00),
        help=(
            "Perturbation amplitudes in mm. Defaults to a denser progression around IMPACT up to 5 mm: "
            "0.10 0.25 0.50 0.75 1.00 1.50 2.00 3.00 4.00 5.00"
        ),
    )
    parser.add_argument(
        "--grid-spacing-mm",
        type=float,
        default=40.0,
        help="Spacing of the coarse random displacement lattice in mm (default: 40.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base random seed (default: 1234).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save baseline and perturbed warped volumes.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logs.",
    )
    return parser.parse_args()


def require_simpleitk_transformix() -> None:
    required = ("ReadParameterFile", "TransformixImageFilter")
    missing = [name for name in required if not hasattr(sitk, name)]
    if missing:
        raise RuntimeError(
            "SimpleITK elastix/transformix API unavailable. "
            f"Missing: {', '.join(missing)}. Install requirements-preprocessing.txt."
        )


def format_numbers(values: list[float | int]) -> str:
    formatted: list[str] = []
    for value in values:
        if isinstance(value, (int, np.integer)) or float(value).is_integer():
            formatted.append(str(int(round(float(value)))))
        else:
            formatted.append(f"{float(value):.10f}".rstrip("0").rstrip("."))
    return " ".join(formatted)


def patch_transform_geometry(transform_path: Path, reference_image: sitk.Image, output_path: Path) -> None:
    replacements = {
        "Size": list(reference_image.GetSize()),
        "Spacing": list(reference_image.GetSpacing()),
        "Origin": list(reference_image.GetOrigin()),
        "Direction": list(reference_image.GetDirection()),
        "Index": [0, 0, 0],
    }

    lines = transform_path.read_text(encoding="utf-8").splitlines()
    found = {key: False for key in replacements}
    patched: list[str] = []

    for line in lines:
        stripped = line.strip()
        replaced = False
        for key, values in replacements.items():
            if stripped.startswith(f"({key} "):
                patched.append(f"({key} {format_numbers(values)})")
                found[key] = True
                replaced = True
                break
        if not replaced:
            patched.append(line)

    for key, values in replacements.items():
        if not found[key]:
            patched.append(f"({key} {format_numbers(values)})")

    output_path.write_text("\n".join(patched) + "\n", encoding="utf-8")


def load_parameter_map(transform_path: Path, reference_image: sitk.Image) -> object:
    with tempfile.TemporaryDirectory(prefix="impact_perturb_eval_") as tmp_root:
        temp_path = Path(tmp_root) / transform_path.name
        patch_transform_geometry(transform_path, reference_image, temp_path)
        return sitk.ReadParameterFile(str(temp_path))


def parameter_map_to_displacement_field(reference_image: sitk.Image, parameter_map: object) -> sitk.Image:
    transformix = sitk.TransformixImageFilter()
    transformix.ComputeDeformationFieldOn()
    transformix.SetMovingImage(reference_image)
    transformix.SetTransformParameterMap(parameter_map)
    transformix.Execute()
    field = transformix.GetDeformationField()
    field.CopyInformation(reference_image)
    return sitk.Cast(field, sitk.sitkVectorFloat64)


def resample_with_displacement_field(moving_image: sitk.Image, reference_image: sitk.Image, field: sitk.Image) -> sitk.Image:
    displacement_transform = sitk.DisplacementFieldTransform(sitk.Cast(field, sitk.sitkVectorFloat64))
    min_max = sitk.MinimumMaximumImageFilter()
    min_max.Execute(moving_image)
    default_value = float(min_max.GetMinimum())
    result = sitk.Resample(
        moving_image,
        reference_image,
        displacement_transform,
        sitk.sitkBSpline,
        default_value,
        moving_image.GetPixelID(),
    )
    return sitk.Cast(
        sitk.Clamp(result, lowerBound=min_max.GetMinimum(), upperBound=min_max.GetMaximum()),
        moving_image.GetPixelID(),
    )


def gaussian_sigma_in_voxels(reference_image: sitk.Image, sigma_mm: float) -> tuple[float, float, float]:
    spacing = reference_image.GetSpacing()
    return tuple(float(sigma_mm / axis_spacing) for axis_spacing in spacing)


def build_coarse_random_field(
    reference_image: sitk.Image,
    mask_image: sitk.Image,
    rng: np.random.Generator,
    max_displacement_mm: float,
    grid_spacing_mm: float,
) -> sitk.Image:
    spacing_xyz = reference_image.GetSpacing()
    origin_xyz = reference_image.GetOrigin()
    direction_xyz = reference_image.GetDirection()
    size_xyz = reference_image.GetSize()

    coarse_spacing_xyz = tuple(float(grid_spacing_mm) for _ in range(3))
    coarse_size_xyz = tuple(
        max(3, int(np.ceil(((size_xyz[axis] - 1) * spacing_xyz[axis]) / coarse_spacing_xyz[axis])) + 1)
        for axis in range(3)
    )
    coarse_size_zyx = (coarse_size_xyz[2], coarse_size_xyz[1], coarse_size_xyz[0])

    components = []
    for _ in range(3):
        coarse_noise = rng.uniform(-max_displacement_mm, max_displacement_mm, size=coarse_size_zyx).astype(np.float32)
        coarse_image = sitk.GetImageFromArray(coarse_noise)
        coarse_image.SetSpacing(coarse_spacing_xyz)
        coarse_image.SetOrigin(origin_xyz)
        coarse_image.SetDirection(direction_xyz)
        full_image = sitk.Resample(
            coarse_image,
            reference_image,
            sitk.Transform(),
            sitk.sitkLinear,
            0.0,
            sitk.sitkFloat32,
        )
        full_array = sitk.GetArrayFromImage(full_image).astype(np.float64)
        full_array -= float(full_array.mean())
        components.append(full_array)

    vector_array = np.stack(components, axis=-1)
    mask_array = sitk.GetArrayFromImage(mask_image) > 0
    magnitude = np.linalg.norm(vector_array, axis=-1)
    valid = magnitude[mask_array] if mask_array.any() else magnitude.ravel()

    if valid.size > 0:
        p95 = float(np.percentile(valid, 95))
        if p95 > 1e-8:
            vector_array *= float(max_displacement_mm / p95)

    magnitude = np.linalg.norm(vector_array, axis=-1)
    hard_cap = 2.0 * float(max_displacement_mm)
    over = magnitude > hard_cap
    if np.any(over):
        scale = np.ones_like(magnitude, dtype=np.float64)
        scale[over] = hard_cap / magnitude[over]
        vector_array *= scale[..., None]

    vector_image = sitk.GetImageFromArray(vector_array, isVector=True)
    vector_image.CopyInformation(reference_image)
    return sitk.Cast(vector_image, sitk.sitkVectorFloat64)


def displacement_stats(field: sitk.Image, mask_image: sitk.Image) -> dict[str, float]:
    field_array = sitk.GetArrayFromImage(field)
    magnitude = np.linalg.norm(field_array, axis=-1)
    mask_array = sitk.GetArrayFromImage(mask_image) > 0
    valid = magnitude[mask_array] if mask_array.any() else magnitude.ravel()
    if valid.size == 0:
        return {
            "perturbation_mean_mm": float("nan"),
            "perturbation_p95_mm": float("nan"),
            "perturbation_max_mm_realized": float("nan"),
        }
    return {
        "perturbation_mean_mm": float(valid.mean()),
        "perturbation_p95_mm": float(np.percentile(valid, 95)),
        "perturbation_max_mm_realized": float(valid.max()),
    }


def masked_metrics(reference: sitk.Image, candidate: sitk.Image, mask: sitk.Image) -> dict[str, float]:
    reference_array = sitk.GetArrayFromImage(sitk.Cast(reference, sitk.sitkFloat32))
    candidate_array = sitk.GetArrayFromImage(sitk.Cast(candidate, sitk.sitkFloat32))
    mask_array = sitk.GetArrayFromImage(mask) > 0

    if not mask_array.any():
        return {"mae": float("nan"), "psnr": float("nan"), "ssim": float("nan")}

    reference_masked = reference_array[mask_array]
    candidate_masked = candidate_array[mask_array]

    mae = float(np.mean(np.abs(candidate_masked - reference_masked)))
    psnr = float(peak_signal_noise_ratio(reference_masked, candidate_masked, data_range=DATA_RANGE))

    reference_for_ssim = np.where(mask_array, reference_array, 0.0)
    candidate_for_ssim = np.where(mask_array, candidate_array, 0.0)
    ssim = float(structural_similarity(reference_for_ssim, candidate_for_ssim, data_range=DATA_RANGE))
    return {"mae": mae, "psnr": psnr, "ssim": ssim}


def build_patient_index(
    tasks: tuple[int, ...],
    regions: tuple[str, ...],
    explicit_patients: set[str] | None,
) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for task, version, region, region_dir in REGION_SPECS:
        if task not in tasks or region not in regions or not region_dir.exists():
            continue
        transform_root = ROOT / "data" / "transforms" / f"synthrad{version}-impact-registration" / f"Task_{task}" / region
        for patient_dir in sorted(path for path in region_dir.iterdir() if path.is_dir()):
            patient = patient_dir.name
            if patient not in TASKS_TO_PATIENTS[task]:
                continue
            if explicit_patients is not None and patient not in explicit_patients:
                continue
            jobs.append(
                {
                    "task": task,
                    "version": version,
                    "region": region,
                    "patient": patient,
                    "dataset_dir": patient_dir,
                    "transform_path": transform_root / f"{patient}.txt",
                }
            )
    return jobs


def case_seed(base_seed: int, patient: str, level_mm: float, sample_index: int) -> int:
    # A CRC rather than hash(), so that the seed is the same in every process.
    return zlib.crc32(f"{patient}:{level_mm:.2f}:{sample_index}".encode()) + base_seed


def process_patient(
    job: dict[str, object],
    samples_per_level: int,
    perturbation_levels_mm: tuple[float, ...],
    grid_spacing_mm: float,
    base_seed: int,
    output_dir: Path,
    save_images: bool,
) -> list[dict[str, object]]:
    require_simpleitk_transformix()

    task = int(job["task"])
    version = str(job["version"])
    region = str(job["region"])
    patient = str(job["patient"])
    dataset_dir = Path(str(job["dataset_dir"]))
    transform_path = Path(str(job["transform_path"]))

    ct_path = dataset_dir / "CT.mha"
    ct_elx_path = dataset_dir / "CT_ELX.mha"
    mask_path = dataset_dir / "MASK.mha"
    required = (ct_path, ct_elx_path, mask_path, transform_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files for {patient}: {missing}")

    ct_image = sitk.ReadImage(str(ct_path))
    ct_elx_image = sitk.ReadImage(str(ct_elx_path))
    mask_image = sitk.Cast(sitk.ReadImage(str(mask_path)), sitk.sitkUInt8)
    parameter_map = load_parameter_map(transform_path, ct_image)
    impact_field = parameter_map_to_displacement_field(ct_image, parameter_map)

    rows: list[dict[str, object]] = []
    patient_output_dir = output_dir / f"Task_{task}" / region / patient
    if save_images:
        patient_output_dir.mkdir(parents=True, exist_ok=True)

    baseline_image = resample_with_displacement_field(ct_elx_image, ct_image, impact_field)
    baseline_metrics = masked_metrics(ct_image, baseline_image, mask_image)
    baseline_row = {
        "task": task,
        "version": version,
        "region": region,
        "patient": patient,
        "sample_kind": "baseline",
        "sample": "baseline",
        "perturbation_max_mm": 0.0,
        "grid_spacing_mm": grid_spacing_mm,
        "perturbation_mean_mm": 0.0,
        "perturbation_p95_mm": 0.0,
        "perturbation_max_mm_realized": 0.0,
        **baseline_metrics,
    }
    if save_images:
        baseline_path = patient_output_dir / "CT_deformed_baseline.mha"
        sitk.WriteImage(baseline_image, str(baseline_path))
        baseline_row["image_path"] = str(baseline_path)
    rows.append(baseline_row)

    for level_mm in perturbation_levels_mm:
        for sample_index in range(samples_per_level):
            seed = case_seed(base_seed, patient, level_mm, sample_index)
            rng = np.random.default_rng(seed)
            perturbation = build_coarse_random_field(ct_image, mask_image, rng, level_mm, grid_spacing_mm)
            perturbed_field = sitk.Cast(impact_field + perturbation, sitk.sitkVectorFloat64)
            deformed_image = resample_with_displacement_field(ct_elx_image, ct_image, perturbed_field)
            metrics = masked_metrics(ct_image, deformed_image, mask_image)
            perturbation_info = displacement_stats(perturbation, mask_image)

            row = {
                "task": task,
                "version": version,
                "region": region,
                "patient": patient,
                "sample_kind": "perturbed",
                "sample": f"{level_mm:.2f}mm_{sample_index:02d}",
                "perturbation_max_mm": float(level_mm),
                "grid_spacing_mm": grid_spacing_mm,
                **perturbation_info,
                **metrics,
            }
            if save_images:
                image_path = patient_output_dir / f"CT_deformed_{level_mm:.2f}mm_{sample_index:02d}.mha"
                sitk.WriteImage(deformed_image, str(image_path))
                row["image_path"] = str(image_path)
            rows.append(row)

    return rows


def summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "count": int(array.size),
    }


def plot_metric_curves(output_dir: Path, rows: list[dict[str, object]], tasks: tuple[int, ...]) -> list[str]:
    curve_dir = output_dir / "curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    metrics = ("mae", "psnr", "ssim")
    task_to_regions: dict[int, set[str]] = defaultdict(set)
    grouped: dict[tuple[int, str, float], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        task = int(row["task"])
        region = str(row["region"])
        level = float(row["perturbation_max_mm"])
        task_to_regions[task].add(region)
        for metric in metrics:
            grouped[(task, region, level)][metric].append(float(row[metric]))

    for task in tasks:
        regions = sorted(task_to_regions.get(task, set()))
        if not regions:
            continue
        for metric in metrics:
            plt.figure(figsize=(7, 4.5))
            for region in regions:
                levels = sorted(
                    {
                        level
                        for grouped_task, grouped_region, level in grouped
                        if grouped_task == task and grouped_region == region
                    }
                )
                if not levels:
                    continue
                means = []
                stds = []
                for level in levels:
                    values = grouped[(task, region, level)][metric]
                    means.append(float(np.mean(values)))
                    stds.append(float(np.std(values)))
                means_array = np.asarray(means, dtype=np.float64)
                stds_array = np.asarray(stds, dtype=np.float64)
                levels_array = np.asarray(levels, dtype=np.float64)
                plt.plot(levels_array, means_array, marker="o", label=region)
                plt.fill_between(levels_array, means_array - stds_array, means_array + stds_array, alpha=0.18)

            plt.xlabel("Perturbation Max (mm)")
            plt.ylabel(metric.upper())
            plt.title(f"Task {task} {metric.upper()} vs IMPACT Perturbation")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

            figure_path = curve_dir / f"task_{task}_{metric}_by_region.png"
            plt.savefig(figure_path, dpi=180)
            plt.close()
            created.append(str(figure_path))

    return created


def main() -> None:
    args = parse_args()
    explicit_patients = set(args.patients) if args.patients else None
    jobs = build_patient_index(tuple(args.tasks), tuple(args.regions), explicit_patients)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {
            executor.submit(
                process_patient,
                job,
                args.samples_per_level,
                tuple(float(value) for value in args.perturbation_levels_mm),
                args.grid_spacing_mm,
                args.seed,
                args.output_dir,
                args.save_images,
            ): job
            for job in jobs
        }
        for index, future in enumerate(as_completed(future_to_job), start=1):
            job = future_to_job[future]
            patient = str(job["patient"])
            if not args.quiet:
                print(
                    f"[{index}/{len(jobs)}] Task {job['task']} {job['region']} {patient}",
                    flush=True,
                )
            try:
                all_rows.extend(future.result())
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "task": int(job["task"]),
                        "region": str(job["region"]),
                        "patient": patient,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )

    all_rows.sort(key=lambda row: (int(row["task"]), str(row["region"]), str(row["patient"]), str(row["sample"])))

    csv_path = args.output_dir / "metrics_per_case.csv"
    fieldnames = [
        "task",
        "version",
        "region",
        "patient",
        "sample",
        "sample_kind",
        "perturbation_max_mm",
        "grid_spacing_mm",
        "perturbation_mean_mm",
        "perturbation_p95_mm",
        "perturbation_max_mm_realized",
        "mae",
        "psnr",
        "ssim",
        "image_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in all_rows:
        level = float(row["perturbation_max_mm"])
        group = f"Task_{row['task']}/{row['region']}/{level:.2f}mm"
        grouped[group]["mae"].append(float(row["mae"]))
        grouped[group]["psnr"].append(float(row["psnr"]))
        grouped[group]["ssim"].append(float(row["ssim"]))

    figure_paths = plot_metric_curves(args.output_dir, all_rows, tuple(args.tasks))

    summary = {
        "output_dir": str(args.output_dir),
        "samples_per_level": int(args.samples_per_level),
        "perturbation_levels_mm": [float(value) for value in args.perturbation_levels_mm],
        "grid_spacing_mm": float(args.grid_spacing_mm),
        "case_count_processed": len(all_rows),
        "patient_count_processed": len({(row["task"], row["patient"]) for row in all_rows}),
        "failure_count": len(failures),
        "metrics_by_group": {
            group: {metric: summarize(values) for metric, values in metrics.items()}
            for group, metrics in grouped.items()
        },
        "curve_paths": figure_paths,
        "failures": failures,
    }

    json_path = args.output_dir / "metrics_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "case_count_processed": len(all_rows),
                "patient_count_processed": len({(row["task"], row["patient"]) for row in all_rows}),
                "failure_count": len(failures),
            },
            indent=2,
        )
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
