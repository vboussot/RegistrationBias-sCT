#!/usr/bin/env python3
"""Generate the final 4-row AB/TH comparison figure used for presentation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "figures" / "generated" / "figure_3"
DATASET_DIR = ROOT / "data" / "processed" / "synthrad2025" / "Task_1"
PREDICTIONS_DIR = ROOT / "predictions" / "Task_1" / "Supervised"
TARGET_CROP_SHAPE = (288, 400)


@dataclass(frozen=True)
class PredictionSpec:
    title: str
    mode: str
    fold: str
    criterion: str


@dataclass(frozen=True)
class FigureRow:
    group: str
    region: str
    patient: str
    slice_index: int
    reference_name: str
    predictions: tuple[PredictionSpec, PredictionSpec]
    window: tuple[float, float]
    filename_suffix: str
    reference_title: str


ROWS = (
    FigureRow(
        group="CTELX_AB_1ABA098_z66",
        region="AB",
        patient="1ABA098",
        slice_index=66,
        reference_name="CT",
        predictions=(
            PredictionSpec("MSE", "IMPACT", "CV_0", "MSE_IMPACT"),
            PredictionSpec("MAE", "IMPACT", "CV", "MAE_IMPACT"),
            PredictionSpec("VGG", "IMPACT", "CV", "VGG_IMPACT"),
            PredictionSpec("SAM", "IMPACT", "CV", "SAM_IMPACT"),
        ),
        window=(-800.0, 500.0),
        filename_suffix="",
        reference_title="CT",
    ),
    FigureRow(
        group="CTELX_TH_1THB017_z62",
        region="TH",
        patient="1THB017",
        slice_index=62,
        reference_name="CT",
        predictions=(
            PredictionSpec("MSE", "IMPACT", "CV_0", "MSE_IMPACT"),
            PredictionSpec("MAE", "IMPACT", "CV", "MAE_IMPACT"),
            PredictionSpec("VGG", "IMPACT", "CV", "VGG_IMPACT"),
            PredictionSpec("SAM", "IMPACT", "CV", "SAM_IMPACT"),
        ),
        window=(-1000.0, 400.0),
        filename_suffix="_thorax_window",
        reference_title="CT",
    ),
    FigureRow(
        group="CT_AB_1ABB113_z37",
        region="AB",
        patient="1ABB113",
        slice_index=37,
        reference_name="CT",
        predictions=(
            PredictionSpec("MSE", "IMPACT", "CV_0", "MSE_IMPACT"),
            PredictionSpec("MAE", "IMPACT", "CV", "MAE_IMPACT"),
            PredictionSpec("VGG", "IMPACT", "CV", "VGG_IMPACT"),
            PredictionSpec("SAM", "IMPACT", "CV", "SAM_IMPACT"),
        ),
        window=(-800.0, 500.0),
        filename_suffix="",
        reference_title="CT",
    ),
    FigureRow(
        group="CT_TH_1THB121_z40",
        region="TH",
        patient="1THB121",
        slice_index=40,
        reference_name="CT",
        predictions=(
            PredictionSpec("MSE", "IMPACT", "CV_0", "MSE_IMPACT"),
            PredictionSpec("MAE", "IMPACT", "CV", "MAE_IMPACT"),
            PredictionSpec("VGG", "IMPACT", "CV", "VGG_IMPACT"),
            PredictionSpec("SAM", "IMPACT", "CV", "SAM_IMPACT"),
        ),
        window=(-1000.0, 400.0),
        filename_suffix="_thorax_window",
        reference_title="CT",
    ),
)


def load_volume(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def window_image(image: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def normalize_mr_image(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, (1.0, 99.0))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def bbox_from_mask(
    mask: np.ndarray,
    *,
    pad_fraction: float,
    min_pad: int = 4,
) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, mask.shape[0], 0, mask.shape[1]

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    pady = max(min_pad, int((y1 - y0) * pad_fraction))
    padx = max(min_pad, int((x1 - x0) * pad_fraction))
    return (
        max(0, y0 - pady),
        min(mask.shape[0], y1 + pady),
        max(0, x0 - padx),
        min(mask.shape[1], x1 + padx),
    )


def fit_bbox_to_shape(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    y0, y1, x0, x1 = bbox
    image_h, image_w = image_shape
    target_h, target_w = target_shape

    cy = (y0 + y1) // 2
    cx = (x0 + x1) // 2

    y0 = cy - target_h // 2
    x0 = cx - target_w // 2
    y1 = y0 + target_h
    x1 = x0 + target_w

    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y1 > image_h:
        y0 -= y1 - image_h
        y1 = image_h
    if x1 > image_w:
        x0 -= x1 - image_w
        x1 = image_w

    y0 = max(0, y0)
    x0 = max(0, x0)
    y1 = min(image_h, y1)
    x1 = min(image_w, x1)
    return y0, y1, x0, x1


def save_panel(path: Path, image: np.ndarray) -> None:
    plt.imsave(path, image, cmap="gray", vmin=0, vmax=1)


def figure_size(num_rows: int, num_cols: int) -> tuple[float, float]:
    return 2.8 * num_cols, 2.45 * num_rows


def missing_panel(shape: tuple[int, int]) -> np.ndarray:
    return np.full(shape, 0.5, dtype=np.float32)


def add_panel_labels(
    ax: plt.Axes,
    title: str,
    metric_label: str | None = None,
) -> None:
    if metric_label is None:
        ax.text(
            0.5,
            1.04,
            title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=16,
        )
        return

    ax.text(
        0.5,
        1.12,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=16,
    )
    ax.text(
        0.5,
        1.00,
        metric_label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=16,
    )


def add_metric_label(ax: plt.Axes, metric_label: str) -> None:
    ax.text(
        0.5,
        1.02,
        metric_label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=16,
    )


def pad_to_shape(
    image: np.ndarray,
    target_shape: tuple[int, int],
    fill_value: float = 1.0,
) -> np.ndarray:
    target_h, target_w = target_shape
    image_h, image_w = image.shape
    if (image_h, image_w) == (target_h, target_w):
        return image

    padded = np.full((target_h, target_w), fill_value, dtype=np.float32)
    y0 = (target_h - image_h) // 2
    x0 = (target_w - image_w) // 2
    padded[y0:y0 + image_h, x0:x0 + image_w] = image
    return padded


def render_row(
    row: FigureRow,
    output_dir: Path,
) -> tuple[list[np.ndarray], list[str], list[float | None]]:
    patient_dir = DATASET_DIR / row.region / row.patient
    mr_volume = load_volume(patient_dir / "MR_IMPACT.mha")
    reference_volume = load_volume(patient_dir / f"{row.reference_name}.mha")
    mask_volume = load_volume(patient_dir / "MASK.mha")

    mr_slice = mr_volume[row.slice_index]
    reference_slice = reference_volume[row.slice_index]
    mask_slice = mask_volume[row.slice_index] > 0
    pad_fraction = 0.025 if row.region == "TH" else 0.035
    y0, y1, x0, x1 = fit_bbox_to_shape(
        bbox_from_mask(mask_slice, pad_fraction=pad_fraction),
        reference_slice.shape,
        TARGET_CROP_SHAPE,
    )

    lo, hi = row.window
    mr_crop = normalize_mr_image(mr_slice)[y0:y1, x0:x1]
    reference_crop = window_image(reference_slice, lo, hi)[y0:y1, x0:x1]

    row_output_dir = output_dir / row.group
    row_output_dir.mkdir(parents=True, exist_ok=True)

    panel_images: list[np.ndarray] = []
    panel_titles: list[str] = []
    metric_values: list[float | None] = []

    save_panel(row_output_dir / f"MR{row.filename_suffix}.png", mr_crop)
    panel_images.append(mr_crop)
    panel_titles.append("MR")
    metric_values.append(None)

    for prediction in row.predictions:
        prediction_path = (
            PREDICTIONS_DIR
            / prediction.mode
            / prediction.fold
            / prediction.criterion
            / "Output"
            / row.patient
            / "sCT.mha"
        )
        if prediction_path.exists():
            prediction_volume = load_volume(prediction_path)
            prediction_slice = prediction_volume[row.slice_index]
            prediction_crop = window_image(prediction_slice, lo, hi)[y0:y1, x0:x1]
            mae_value = float(np.mean(np.abs(prediction_slice[mask_slice] - reference_slice[mask_slice])))
            save_panel(row_output_dir / f"{prediction.title}{row.filename_suffix}.png", prediction_crop)
        else:
            prediction_crop = missing_panel(reference_crop.shape)
            mae_value = None

        panel_images.append(prediction_crop)
        panel_titles.append(prediction.title)
        metric_values.append(mae_value)

    reference_panel_name = (
        "CT_ELX_reference" if row.reference_name == "CT_ELX" else "CT_reference"
    )
    save_panel(row_output_dir / f"{reference_panel_name}{row.filename_suffix}.png", reference_crop)

    panel_images.append(reference_crop)
    panel_titles.append(row.reference_title)
    metric_values.append(None)

    num_cols = len(panel_images)
    fig, axes = plt.subplots(1, num_cols, figsize=(2.9 * num_cols, 3.6), facecolor="white")
    for index, ax in enumerate(axes):
        ax.imshow(panel_images[index], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        if metric_values[index] is not None:
            metric_label = (
                f"MAE={metric_values[index]:.1f}"
                if metric_values[index] is not None
                else "missing"
            )
            add_panel_labels(ax, panel_titles[index], metric_label)
        else:
            add_panel_labels(ax, panel_titles[index])
    fig.tight_layout(w_pad=1.3)
    fig.savefig(
        row_output_dir / f"comparison{row.filename_suffix}.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    return panel_images, panel_titles, metric_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the final 4-row AB/TH comparison figure."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for the figure and exported panels (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered_rows: list[tuple[FigureRow, list[np.ndarray], list[str], list[float | None]]] = []
    summary_lines: list[str] = []

    for row in ROWS:
        panel_images, panel_titles, metric_values = render_row(row, output_dir)
        rendered_rows.append((row, panel_images, panel_titles, metric_values))
        prediction_summary = ", ".join(
            f"{prediction.title}={value:.1f}" if value is not None else f"{prediction.title}=missing"
            for prediction, value in zip(row.predictions, metric_values)
        )
        summary_lines.append(
            (
                f"{row.region} {row.patient} z={row.slice_index} ref={row.reference_name}: "
                f"{prediction_summary}, "
                f"window=({row.window[0]:.0f},{row.window[1]:.0f})"
            )
        )

    num_rows = len(rendered_rows[0][1])
    num_cols = len(rendered_rows)
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=figure_size(num_rows, num_cols),
        facecolor="white",
    )
    if num_rows == 1:
        axes = np.array([axes])
    elif num_cols == 1:
        axes = np.array([[ax] for ax in axes])

    target_height = max(image.shape[0] for _, panel_images, _, _ in rendered_rows for image in panel_images)
    target_width = max(image.shape[1] for _, panel_images, _, _ in rendered_rows for image in panel_images)
    target_shape = (target_height, target_width)

    for col_index, (row, panel_images, panel_titles, metric_values) in enumerate(rendered_rows):
        for row_index, ax in enumerate(axes[:, col_index]):
            display_image = pad_to_shape(panel_images[row_index], target_shape)
            ax.imshow(display_image, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if metric_values[row_index] is not None:
                metric_label = (
                    f"MAE={metric_values[row_index]:.1f}"
                    if metric_values[row_index] is not None
                    else "missing"
                )
                add_metric_label(ax, metric_label)

    for row_index, label in enumerate(panel_titles):
        axes[row_index, 0].text(
            -0.06,
            0.5,
            label,
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=15,
        )

    fig.subplots_adjust(left=0.055, right=0.994, top=0.995, bottom=0.005, hspace=-0.26, wspace=0.05)
    fig.savefig(output_dir / "comparison_4rows.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    (output_dir / "selection_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
