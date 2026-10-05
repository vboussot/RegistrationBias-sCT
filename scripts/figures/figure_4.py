#!/usr/bin/env python3
"""Generate qualitative registration comparison figures for MR/CT pairs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "data" / "processed" / "synthrad2025" / "Task_1"
DEFAULT_OUTPUT_DIR = ROOT / "figures" / "generated" / "figure_4"


@dataclass(frozen=True)
class RegistrationPair:
    moving_name: str
    fixed_name: str
    moving_title: str
    fixed_title: str


@dataclass(frozen=True)
class CaseSpec:
    region: str
    patient: str
    slice_index: int
    ct_window: tuple[float, float]


PAIRS = (
    RegistrationPair("MR", "CT_ELX", "MR", "CT_ELX"),
    RegistrationPair("MR_IMPACT", "CT", "MR_IMPACT", "CT"),
)

CASES = (
    CaseSpec("AB", "1ABA062", 46, (-160.0, 240.0)),
    CaseSpec("HN", "1HNC109", 19, (-300.0, 500.0)),
    CaseSpec("TH", "1THA224", 25, (-1000.0, 400.0)),
)


def load_volume(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def normalize_mr(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, 1), np.percentile(image, 99)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def normalize_ct(image: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def bbox_from_mask(mask: np.ndarray, pad_fraction: float = 0.03) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, mask.shape[0], 0, mask.shape[1]

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    pady = max(4, int((y1 - y0) * pad_fraction))
    padx = max(4, int((x1 - x0) * pad_fraction))
    return (
        max(0, y0 - pady),
        min(mask.shape[0], y1 + pady),
        max(0, x0 - padx),
        min(mask.shape[1], x1 + padx),
    )


def color_overlay(moving: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    overlay = np.zeros(moving.shape + (3,), dtype=np.float32)
    overlay[..., 1] = moving
    overlay[..., 0] = fixed
    overlay[..., 2] = fixed
    return np.clip(overlay, 0.0, 1.0)


def checkerboard(moving: np.ndarray, fixed: np.ndarray, tiles: int = 8) -> np.ndarray:
    height, width = moving.shape
    yy, xx = np.indices((height, width))
    tile_h = max(1, height // tiles)
    tile_w = max(1, width // tiles)
    mask = ((yy // tile_h) + (xx // tile_w)) % 2 == 0
    return np.where(mask, moving, fixed)


def render_case(case: CaseSpec, output_dir: Path) -> Path:
    patient_dir = DATASET_DIR / case.region / case.patient
    mask_volume = load_volume(patient_dir / "MASK.mha")
    mask_slice = mask_volume[case.slice_index] > 0
    y0, y1, x0, x1 = bbox_from_mask(mask_slice)

    fig, axes = plt.subplots(len(PAIRS), 4, figsize=(10, 5.6), facecolor="white")
    if len(PAIRS) == 1:
        axes = np.array([axes])

    for row_index, pair in enumerate(PAIRS):
        moving_volume = load_volume(patient_dir / f"{pair.moving_name}.mha")
        fixed_volume = load_volume(patient_dir / f"{pair.fixed_name}.mha")

        moving_slice = normalize_mr(moving_volume[case.slice_index])[y0:y1, x0:x1]
        fixed_slice = normalize_ct(fixed_volume[case.slice_index], *case.ct_window)[y0:y1, x0:x1]

        panels = (
            moving_slice,
            fixed_slice,
            color_overlay(moving_slice, fixed_slice),
            checkerboard(moving_slice, fixed_slice),
        )
        titles = (pair.moving_title, pair.fixed_title, "Overlay", "Checkerboard")

        for column_index, (panel, title) in enumerate(zip(panels, titles)):
            ax = axes[row_index, column_index]
            if panel.ndim == 3:
                ax.imshow(panel)
            else:
                ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            ax.set_title(title, fontsize=12)

    fig.suptitle(f"{case.region} | {case.patient} | z={case.slice_index}", fontsize=15)
    fig.tight_layout()

    output_path = output_dir / f"{case.region}_{case.patient}_z{case.slice_index}.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def render_thesis_figure(output_dir: Path, *, reverse_blocks: bool = False) -> Path:
    fig, axes = plt.subplots(
        len(CASES),
        9,
        figsize=(14, 18),
        facecolor="white",
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.12, 1, 1, 1, 1]},
    )
    if len(CASES) == 1:
        axes = np.array([axes])

    left_titles = ("MR", "CT_ELX", "Overlay", "Checkerboard")
    right_titles = ("MR_IMPACT", "CT", "Overlay", "Checkerboard")
    left_block_name = "ELX"
    right_block_name = "IMPACT"
    if reverse_blocks:
        left_titles, right_titles = right_titles, left_titles
        left_block_name, right_block_name = right_block_name, left_block_name
    for column_index, title in enumerate(left_titles):
        axes[0, column_index].set_title(title, fontsize=13, pad=10)
    for offset, title in enumerate(right_titles, start=5):
        axes[0, offset].set_title(title, fontsize=13, pad=10)

    fig.text(0.27, 0.985, left_block_name, ha="center", va="top", fontsize=18, fontweight="bold")
    fig.text(0.74, 0.985, right_block_name, ha="center", va="top", fontsize=18, fontweight="bold")

    for row_index, case in enumerate(CASES):
        patient_dir = DATASET_DIR / case.region / case.patient
        mask_volume = load_volume(patient_dir / "MASK.mha")
        mask_slice = mask_volume[case.slice_index] > 0
        y0, y1, x0, x1 = bbox_from_mask(mask_slice)

        mr = normalize_mr(load_volume(patient_dir / "MR.mha")[case.slice_index])[y0:y1, x0:x1]
        ct_elx = normalize_ct(
            load_volume(patient_dir / "CT_ELX.mha")[case.slice_index],
            *case.ct_window,
        )[y0:y1, x0:x1]
        mr_impact = normalize_mr(load_volume(patient_dir / "MR_IMPACT.mha")[case.slice_index])[y0:y1, x0:x1]
        ct = normalize_ct(
            load_volume(patient_dir / "CT.mha")[case.slice_index],
            *case.ct_window,
        )[y0:y1, x0:x1]

        elx_panels = (
            mr,
            ct_elx,
            color_overlay(mr, ct_elx),
            checkerboard(mr, ct_elx),
        )
        impact_panels = (
            mr_impact,
            ct,
            color_overlay(mr_impact, ct),
            checkerboard(mr_impact, ct),
        )
        left_panels = impact_panels if reverse_blocks else elx_panels
        right_panels = elx_panels if reverse_blocks else impact_panels
        panels = (*left_panels, None, *right_panels)

        for column_index, panel in enumerate(panels):
            ax = axes[row_index, column_index]
            if panel is None:
                ax.axis("off")
                continue
            if panel.ndim == 3:
                ax.imshow(panel)
            else:
                ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")

        axes[row_index, 0].set_ylabel(case.region, fontsize=16, fontweight="bold", rotation=0, labelpad=28, va="center")

    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.2, w_pad=0.35)
    filename = (
        "registration_visual_comparison_3patients_reverse.png"
        if reverse_blocks
        else "registration_visual_comparison_3patients.png"
    )
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def render_thesis_compact_figure(output_dir: Path) -> Path:
    fig, axes = plt.subplots(4, len(CASES), figsize=(10, 14), facecolor="white")
    if len(CASES) == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]], [axes[3]]])

    for column_index, case in enumerate(CASES):
        axes[0, column_index].set_title(case.region, fontsize=16, fontweight="bold", pad=10)

    row_labels = ("ELX\nOverlay", "ELX\nCheckerboard", "IMPACT\nOverlay", "IMPACT\nCheckerboard")

    for column_index, case in enumerate(CASES):
        patient_dir = DATASET_DIR / case.region / case.patient
        mask_volume = load_volume(patient_dir / "MASK.mha")
        mask_slice = mask_volume[case.slice_index] > 0
        y0, y1, x0, x1 = bbox_from_mask(mask_slice)

        mr = normalize_mr(load_volume(patient_dir / "MR.mha")[case.slice_index])[y0:y1, x0:x1]
        ct_elx = normalize_ct(
            load_volume(patient_dir / "CT_ELX.mha")[case.slice_index],
            *case.ct_window,
        )[y0:y1, x0:x1]
        mr_impact = normalize_mr(load_volume(patient_dir / "MR_IMPACT.mha")[case.slice_index])[y0:y1, x0:x1]
        ct = normalize_ct(
            load_volume(patient_dir / "CT.mha")[case.slice_index],
            *case.ct_window,
        )[y0:y1, x0:x1]

        panels = (
            color_overlay(mr, ct_elx),
            checkerboard(mr, ct_elx),
            color_overlay(mr_impact, ct),
            checkerboard(mr_impact, ct),
        )

        for row_index, panel in enumerate(panels):
            ax = axes[row_index, column_index]
            if panel.ndim == 3:
                ax.imshow(panel)
            else:
                ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if column_index == 0:
                ax.set_ylabel(row_labels[row_index], fontsize=13, fontweight="bold", rotation=0, labelpad=35, va="center")

    fig.tight_layout(h_pad=1.0, w_pad=0.5)
    output_path = output_dir / "registration_visual_comparison_3patients_compact.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def render_thesis_rowwise_figure(output_dir: Path) -> Path:
    fig, axes = plt.subplots(2 * len(CASES), 4, figsize=(11.5, 18), facecolor="white")
    if 2 * len(CASES) == 1:
        axes = np.array([axes])

    column_titles = ("MR", "CT", "Overlay", "Checkerboard")
    for column_index, title in enumerate(column_titles):
        axes[0, column_index].set_title(title, fontsize=13, pad=10)

    for case_index, case in enumerate(CASES):
        patient_dir = DATASET_DIR / case.region / case.patient
        mask_volume = load_volume(patient_dir / "MASK.mha")
        mask_slice = mask_volume[case.slice_index] > 0
        y0, y1, x0, x1 = bbox_from_mask(mask_slice)

        mr = normalize_mr(load_volume(patient_dir / "MR.mha")[case.slice_index])[y0:y1, x0:x1]
        ct_elx = normalize_ct(
            load_volume(patient_dir / "CT_ELX.mha")[case.slice_index],
            *case.ct_window,
        )[y0:y1, x0:x1]
        mr_impact = normalize_mr(load_volume(patient_dir / "MR_IMPACT.mha")[case.slice_index])[y0:y1, x0:x1]
        ct = normalize_ct(
            load_volume(patient_dir / "CT.mha")[case.slice_index],
            *case.ct_window,
        )[y0:y1, x0:x1]

        row_specs = (
            (2 * case_index, f"{case.region} IMPACT", mr_impact, ct),
            (2 * case_index + 1, f"{case.region} ELX", mr, ct_elx),
        )

        for row_index, row_label, moving, fixed in row_specs:
            panels = (
                moving,
                fixed,
                color_overlay(moving, fixed),
                checkerboard(moving, fixed),
            )
            for column_index, panel in enumerate(panels):
                ax = axes[row_index, column_index]
                if panel.ndim == 3:
                    ax.imshow(panel)
                else:
                    ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
            axes[row_index, 0].text(
                0.03,
                0.97,
                row_label,
                transform=axes[row_index, 0].transAxes,
                ha="left",
                va="top",
                fontsize=13,
                fontweight="bold",
                color="white",
                bbox={
                    "facecolor": "black",
                    "alpha": 0.7,
                    "pad": 4,
                    "edgecolor": "none",
                },
            )

    fig.tight_layout(rect=(0.02, 0.02, 1, 0.98), h_pad=1.0, w_pad=0.5)
    output_path = output_dir / "registration_visual_comparison_3patients_rowwise.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate qualitative registration comparison figures."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = [render_case(case, output_dir) for case in CASES]
    thesis_figure = render_thesis_figure(output_dir)
    thesis_figure_reverse = render_thesis_figure(output_dir, reverse_blocks=True)
    thesis_figure_compact = render_thesis_compact_figure(output_dir)
    thesis_figure_rowwise = render_thesis_rowwise_figure(output_dir)
    summary = "\n".join(
        str(path)
        for path in [
            *output_paths,
            thesis_figure,
            thesis_figure_reverse,
            thesis_figure_compact,
            thesis_figure_rowwise,
        ]
    ) + "\n"
    (output_dir / "generated_files.txt").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
