#!/usr/bin/env python3
"""Best-effort pipeline to generate a simulated CBCT from a planning CT using RTK.

The purpose of this script is not exact scanner replication. Instead, it aims
to generate CBCT-like volumes that preserve the CT anatomy while reproducing
enough center-specific acquisition characteristics, noise, and scatter to test
whether a synthesis model trained on real CBCT data introduces anatomical bias.

High-level pipeline:
1) read the input CT,
2) resample it to isotropic spacing,
3) recenter the volume around the simulated isocenter,
4) resample the volume so the RTK rotation matches the desired acquisition
   plane,
5) convert HU to linear attenuation coefficients,
6) generate an RTK circular geometry,
7) compute forward projections,
8) add Poisson noise, heuristic scatter, and detector saturation,
9) reconstruct with RTK FDK,
10) convert the reconstruction back to a HU-like scale.

Usage:
    python cbct_synthesis.py --input CT.mha

Python dependencies:
    pip install SimpleITK numpy scipy

System requirements:
    rtkforwardprojections, rtksimulatedgeometry, rtkfdk available in PATH
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
try:
    import SimpleITK as sitk
except ModuleNotFoundError:
    sitk = None

try:
    from scipy.ndimage import gaussian_filter
except ModuleNotFoundError:
    gaussian_filter = None


# ----------------------------
# Utility helpers
# ----------------------------

def run(cmd: str) -> None:
    print(f"\n[CMD] {cmd}\n", flush=True)
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Command failed (code={exc.returncode}): {cmd}") from exc
  


def candidate_rtk_bin_dirs():
    repo_root = Path(__file__).resolve().parent
    return [
        repo_root / "RTK" / "build" / "bin",
        repo_root.parent / "RTK" / "build" / "bin",
    ]


def resolve_binary(name: str, search_dirs=None) -> str:
    if search_dirs is None:
        search_dirs = []

    for directory in search_dirs:
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())

    resolved = shutil.which(name)
    if resolved is not None:
        return resolved

    searched = [str(Path(d).resolve()) for d in search_dirs if Path(d).exists()]
    search_hint = ""
    if searched:
        search_hint = "\nSearched extra directories:\n  - " + "\n  - ".join(searched)
    raise RuntimeError(
        f"Binary not found: {name}\n"
        f"Make sure RTK is installed and available in PATH, or provide --rtk-bin-dir.{search_hint}"
    )


def ensure_python_dependencies() -> None:
    missing = []
    if sitk is None:
        missing.append("SimpleITK")
    if gaussian_filter is None:
        missing.append("scipy")
    if missing:
        raise RuntimeError(
            "Missing Python dependencies: "
            + ", ".join(missing)
            + ". Install them before running the simulation."
        )


def read_image(path: Path) -> sitk.Image:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return sitk.ReadImage(str(path))


def write_image(img: sitk.Image, path: Path) -> None:
    sitk.WriteImage(img, str(path))


def image_from_array_like(array: np.ndarray, reference_img: sitk.Image, dtype=None) -> sitk.Image:
    arr = np.asarray(array)
    if dtype is not None:
        arr = arr.astype(dtype)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(reference_img)
    return out


def image_center_physical(img: sitk.Image):
    size = np.array(list(img.GetSize()), dtype=np.float64)
    spacing = np.array(list(img.GetSpacing()), dtype=np.float64)
    origin = np.array(list(img.GetOrigin()), dtype=np.float64)
    direction = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)

    # Volume center in physical coordinates.
    center_index = (size - 1.0) / 2.0
    center_phys = origin + direction @ (center_index * spacing)
    return center_phys


def centered_image_origin(
    center_phys,
    size_xyz,
    spacing_xyz,
    direction_matrix,
):
    """
    Compute the image origin that places the image center at center_phys.

    The returned origin is expressed in physical coordinates and accounts for
    image direction.
    """
    center_phys = np.array(center_phys, dtype=np.float64)
    size_xyz = np.array(size_xyz, dtype=np.float64)
    spacing_xyz = np.array(spacing_xyz, dtype=np.float64)
    direction_matrix = np.array(direction_matrix, dtype=np.float64).reshape(3, 3)
    half_extent = ((size_xyz - 1.0) * spacing_xyz) / 2.0
    return center_phys - direction_matrix @ half_extent


def centered_axis_aligned_origin(size_xyz, spacing_xyz):
    """
    Compute the origin of an axis-aligned image centered at the isocenter.

    This matches the RTK convention where the fixed coordinate system is
    centered on the isocenter.
    """
    size_xyz = np.array(size_xyz, dtype=np.float64)
    spacing_xyz = np.array(spacing_xyz, dtype=np.float64)
    return (-0.5 * (size_xyz - 1.0) * spacing_xyz).tolist()


def resample_isotropic(img: sitk.Image, iso_spacing: float, default_value: float = -1024.0) -> sitk.Image:
    orig_spacing = np.array(img.GetSpacing(), dtype=np.float64)
    orig_size = np.array(img.GetSize(), dtype=np.int32)
    direction = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)
    center = image_center_physical(img)

    new_spacing = np.array([iso_spacing, iso_spacing, iso_spacing], dtype=np.float64)
    new_size = np.round(orig_size * (orig_spacing / new_spacing)).astype(np.int32)
    new_size = np.maximum(new_size, 1)
    new_origin = centered_image_origin(
        center_phys=center,
        size_xyz=new_size,
        spacing_xyz=new_spacing,
        direction_matrix=direction,
    )

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing(tuple(new_spacing.tolist()))
    resampler.SetSize([int(v) for v in new_size.tolist()])
    resampler.SetOutputOrigin(tuple(new_origin.tolist()))
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetDefaultPixelValue(default_value)
    return resampler.Execute(img)


def recenter_image(img: sitk.Image, target_center_phys=(0.0, 0.0, 0.0)) -> sitk.Image:
    current_center = image_center_physical(img)
    target_center = np.array(target_center_phys, dtype=np.float64)
    shift = target_center - current_center

    new_img = sitk.Image(img)
    new_origin = np.array(img.GetOrigin(), dtype=np.float64) + shift
    new_img.SetOrigin(tuple(new_origin.tolist()))
    return new_img


def shift_image_origin(img: sitk.Image, shift_xyz) -> sitk.Image:
    """
    Shift image physical coordinates by updating the origin only.

    This is used to undo the temporary recentering applied before simulation.
    """
    shifted_img = sitk.Image(img)
    new_origin = np.array(img.GetOrigin(), dtype=np.float64) + np.array(shift_xyz, dtype=np.float64)
    shifted_img.SetOrigin(tuple(new_origin.tolist()))
    return shifted_img


def resample_swap_yz(
    img: sitk.Image,
    interpolator=None,
    default_value: float = -1024.0,
) -> sitk.Image:
    """
    Resample the image while physically swapping Y and Z around the image
    center.

    This acts on the voxel sampling itself rather than only changing metadata.
    The transform is involutive, so the same function can be used after FDK to
    map the reconstructed CBCT back to the original patient axis convention.
    """
    size = np.array(img.GetSize(), dtype=np.int32)
    spacing = np.array(img.GetSpacing(), dtype=np.float64)
    direction = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)
    center = image_center_physical(img)

    if interpolator is None:
        interpolator = sitk.sitkLinear

    out_size = size[[0, 2, 1]]
    out_spacing = spacing[[0, 2, 1]]
    out_origin = centered_image_origin(
        center_phys=center,
        size_xyz=out_size,
        spacing_xyz=out_spacing,
        direction_matrix=direction,
    )

    swap = sitk.AffineTransform(3)
    swap.SetCenter(tuple(center.tolist()))
    swap.SetMatrix((1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0,
                    0.0, 1.0, 0.0))

    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(swap)
    resampler.SetInterpolator(interpolator)
    resampler.SetSize([int(v) for v in out_size.tolist()])
    resampler.SetOutputSpacing(tuple(out_spacing.tolist()))
    resampler.SetOutputOrigin(tuple(out_origin.tolist()))
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetDefaultPixelValue(default_value)
    return resampler.Execute(img)


def resample_to_reference(
    img: sitk.Image,
    reference: sitk.Image,
    interpolator,
    default_value: float,
) -> sitk.Image:
    """
    Resample an image onto the exact geometry of a reference image.

    The output inherits the reference size, spacing, origin, and direction.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    return resampler.Execute(img)


def format_csv(values) -> str:
    return ",".join(str(v) for v in values)


def parse_csv_numbers(text: str, caster=float, expected_length: int | None = None, name: str = "value"):
    values = [v.strip() for v in text.split(",") if v.strip()]
    if expected_length is not None and len(values) != expected_length:
        raise ValueError(
            f"Invalid {name}: expected {expected_length} comma-separated values, got {len(values)} in '{text}'."
        )

    try:
        return [caster(v) for v in values]
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: could not parse '{text}'.") from exc


def candidate_label_schema_paths():
    repo_root = Path(__file__).resolve().parent
    return [
        repo_root / "Dataset_IMPACTSeg" / "task_2maps_labels.json",
        repo_root / "Code" / "preprocessing" / "task_2maps_labels.json",
    ]


def resolve_label_schema_path(path: str | None) -> Path | None:
    if path is not None:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Label schema not found: {resolved}")
        return resolved

    for candidate in candidate_label_schema_paths():
        if candidate.exists():
            return candidate.resolve()
    return None


def load_label_schema_maps(schema_path: Path):
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    fine_name_to_id = {
        meta["name"]: int(label_id)
        for label_id, meta in schema["fine_map"]["labels"].items()
    }
    broad_name_to_id = {
        meta["name"]: int(label_id)
        for label_id, meta in schema["map_559"]["labels"].items()
    }
    return {
        "fine_name_to_id": fine_name_to_id,
        "broad_name_to_id": broad_name_to_id,
    }


def resolve_auxiliary_mask_path(explicit_path: str | None, input_path: Path, filename: str) -> Path | None:
    if explicit_path is not None:
        resolved = Path(explicit_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Mask not found: {resolved}")
        return resolved

    sibling = input_path.parent / filename
    if sibling.exists():
        return sibling.resolve()
    return None


def load_mask_in_centered_space(mask_path: Path, ct_iso: sitk.Image) -> sitk.Image:
    mask = read_image(mask_path)
    mask_iso = resample_to_reference(
        mask,
        reference=ct_iso,
        interpolator=sitk.sitkNearestNeighbor,
        default_value=0,
    )
    return recenter_image(mask_iso, target_center_phys=(0.0, 0.0, 0.0))


def infer_anatomy_from_masks(fine_mask_img: sitk.Image | None, broad_mask_img: sitk.Image | None, label_maps) -> str | None:
    fine_arr = sitk.GetArrayFromImage(fine_mask_img) if fine_mask_img is not None else None
    broad_arr = sitk.GetArrayFromImage(broad_mask_img) if broad_mask_img is not None else None
    fine_ids = label_maps["fine_name_to_id"]
    broad_ids = label_maps["broad_name_to_id"]

    lung_ids = [
        fine_ids.get("left_lung_upper_lobe"),
        fine_ids.get("left_lung_lower_lobe"),
        fine_ids.get("right_lung_upper_lobe"),
        fine_ids.get("right_lung_middle_lobe"),
        fine_ids.get("right_lung_lower_lobe"),
    ]
    lung_ids = [v for v in lung_ids if v is not None]

    abdominal_ids = [
        fine_ids.get("liver"),
        fine_ids.get("stomach"),
        fine_ids.get("pancreas"),
        fine_ids.get("small_intestine"),
        fine_ids.get("duodenum"),
        fine_ids.get("colon"),
    ]
    abdominal_ids = [v for v in abdominal_ids if v is not None]

    thorax_score = 0
    abdomen_score = 0
    if fine_arr is not None:
        if lung_ids:
            thorax_score += int(np.isin(fine_arr, lung_ids).sum())
        if abdominal_ids:
            abdomen_score += int(np.isin(fine_arr, abdominal_ids).sum())
    if broad_arr is not None:
        if broad_ids.get("thoracic_cavity") is not None:
            thorax_score += int((broad_arr == broad_ids["thoracic_cavity"]).sum())
        if broad_ids.get("abdominal_cavity") is not None:
            abdomen_score += int((broad_arr == broad_ids["abdominal_cavity"]).sum())

    if max(thorax_score, abdomen_score) > 0:
        return "TH" if thorax_score >= abdomen_score else "AB"

    if fine_arr is not None and fine_ids.get("mandible") is not None and np.any(fine_arr == fine_ids["mandible"]):
        return "HN"
    return None


def translate_image(
    img: sitk.Image,
    shift_xyz_mm,
    interpolator,
    default_value: float,
) -> sitk.Image:
    transform = sitk.TranslationTransform(3)
    transform.SetOffset(tuple((-np.array(shift_xyz_mm, dtype=np.float64)).tolist()))

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    return resampler.Execute(img)


def smooth_mask(mask_arr: np.ndarray, sigma_zyx=(2.0, 2.0, 2.0)) -> np.ndarray:
    if not np.any(mask_arr):
        return np.zeros(mask_arr.shape, dtype=np.float32)
    weight = gaussian_filter(mask_arr.astype(np.float32), sigma=sigma_zyx)
    max_value = float(weight.max())
    if max_value > 0:
        weight /= max_value
    return weight.astype(np.float32)


def axis_weight_volume(mask_arr: np.ndarray, axis: int, prefer_high_end: bool) -> np.ndarray:
    coord = np.linspace(0.0, 1.0, mask_arr.shape[axis], dtype=np.float32)
    if not prefer_high_end:
        coord = coord[::-1]
    shape = [1, 1, 1]
    shape[axis] = mask_arr.shape[axis]
    return coord.reshape(shape)


def compute_inferior_is_high_index(broad_arr: np.ndarray | None) -> bool:
    if broad_arr is None:
        return True
    thorax = broad_arr == 4
    abdomen = broad_arr == 3
    if np.any(thorax) and np.any(abdomen):
        thorax_mean = float(np.argwhere(thorax)[:, 0].mean())
        abdomen_mean = float(np.argwhere(abdomen)[:, 0].mean())
        return abdomen_mean > thorax_mean
    return True


def build_motion_model(
    ct_img: sitk.Image,
    fine_mask_img: sitk.Image | None,
    broad_mask_img: sitk.Image | None,
    label_maps,
    anatomy: str,
    motion_strength: float,
    motion_cycles: float,
):
    if fine_mask_img is None:
        return None

    fine_arr = sitk.GetArrayFromImage(fine_mask_img)
    broad_arr = sitk.GetArrayFromImage(broad_mask_img) if broad_mask_img is not None else None
    ct_arr = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    fine_ids = label_maps["fine_name_to_id"]
    broad_ids = label_maps["broad_name_to_id"]
    inferior_is_high = compute_inferior_is_high_index(broad_arr)
    z_weight_inferior = axis_weight_volume(fine_arr, axis=0, prefer_high_end=inferior_is_high)
    z_weight_superior = 1.0 - z_weight_inferior

    components = []
    if anatomy == "TH":
        lung_ids = [fine_ids[name] for name in (
            "left_lung_upper_lobe",
            "left_lung_lower_lobe",
            "right_lung_upper_lobe",
            "right_lung_middle_lobe",
            "right_lung_lower_lobe",
        ) if name in fine_ids]
        heart_ids = [fine_ids[name] for name in (
            "myocardium",
            "heart_atrium_right",
            "heart_atrium_left",
            "heart_ventricle_right",
            "heart_ventricle_left",
        ) if name in fine_ids]
        lung_mask = np.isin(fine_arr, lung_ids)
        heart_mask = np.isin(fine_arr, heart_ids)
        thorax_mask = broad_arr == broad_ids["thoracic_cavity"] if broad_arr is not None and "thoracic_cavity" in broad_ids else lung_mask
        gas_mask = np.logical_and(ct_arr < -700.0, thorax_mask)

        lower_lung = smooth_mask(lung_mask * (0.2 + 0.8 * z_weight_inferior))
        upper_lung = smooth_mask(lung_mask * (0.3 + 0.7 * z_weight_superior))
        heart_weight = smooth_mask(heart_mask, sigma_zyx=(2.5, 2.5, 2.5))
        gas_weight = smooth_mask(gas_mask, sigma_zyx=(1.5, 1.5, 1.5))
        components.extend([
            {"weight": lower_lung, "scale": 1.0},
            {"weight": upper_lung, "scale": 0.45},
            {"weight": heart_weight, "scale": 0.35},
            {"weight": gas_weight, "scale": 1.1},
        ])
        shift_xyz_mm = np.array([0.0, 2.5, 8.0], dtype=np.float32) * float(motion_strength)
        cycles = motion_cycles if motion_cycles > 0 else 12.0
    elif anatomy == "AB":
        upper_ids = [fine_ids[name] for name in (
            "liver",
            "spleen",
            "stomach",
            "pancreas",
            "duodenum",
        ) if name in fine_ids]
        bowel_ids = [fine_ids[name] for name in (
            "small_intestine",
            "colon",
        ) if name in fine_ids]
        upper_mask = np.isin(fine_arr, upper_ids)
        bowel_mask = np.isin(fine_arr, bowel_ids)
        abdomen_mask = broad_arr == broad_ids["abdominal_cavity"] if broad_arr is not None and "abdominal_cavity" in broad_ids else np.logical_or(upper_mask, bowel_mask)
        gas_mask = np.logical_and(ct_arr < -650.0, abdomen_mask)

        upper_weight = smooth_mask(upper_mask * (0.35 + 0.65 * z_weight_superior))
        bowel_weight = smooth_mask(bowel_mask * (0.5 + 0.5 * z_weight_superior))
        gas_weight = smooth_mask(gas_mask * (0.2 + 0.8 * z_weight_superior), sigma_zyx=(1.4, 1.4, 1.4))
        components.extend([
            {"weight": upper_weight, "scale": 1.0},
            {"weight": bowel_weight, "scale": 0.65},
            {"weight": gas_weight, "scale": 1.15},
        ])
        shift_xyz_mm = np.array([0.0, 3.5, 11.0], dtype=np.float32) * float(motion_strength)
        cycles = motion_cycles if motion_cycles > 0 else 9.0
    else:
        return None

    components = [component for component in components if np.any(component["weight"] > 1e-3)]
    if not components:
        return None

    return {
        "components": components,
        "shift_xyz_mm": shift_xyz_mm,
        "cycles": float(cycles),
        "phase_values": np.linspace(-1.0, 1.0, 5, dtype=np.float32),
    }


def apply_motion_phase(base_img: sitk.Image, motion_model, phase_value: float, default_value: float = -1024.0) -> sitk.Image:
    base_arr = sitk.GetArrayFromImage(base_img).astype(np.float32)
    out_arr = base_arr.copy()
    for component in motion_model["components"]:
        shift_xyz_mm = motion_model["shift_xyz_mm"] * float(phase_value) * float(component["scale"])
        moved_img = translate_image(
            base_img,
            shift_xyz_mm=shift_xyz_mm,
            interpolator=sitk.sitkLinear,
            default_value=default_value,
        )
        moved_arr = sitk.GetArrayFromImage(moved_img).astype(np.float32)
        weight = component["weight"]
        out_arr = out_arr * (1.0 - weight) + moved_arr * weight
    return image_from_array_like(out_arr, base_img, np.float32)


def blend_motion_projection_stacks(phase_projection_images, motion_model, seed: int) -> sitk.Image:
    phase_values = np.asarray(motion_model["phase_values"], dtype=np.float32)
    stacks = [sitk.GetArrayFromImage(img).astype(np.float32) for img in phase_projection_images]
    out = np.empty_like(stacks[0], dtype=np.float32)

    rng = np.random.default_rng(seed)
    phase_offset = rng.uniform(0.0, 2.0 * np.pi)
    nproj = out.shape[0]
    denom = max(nproj - 1, 1)

    for k in range(nproj):
        phi = phase_offset + 2.0 * np.pi * motion_model["cycles"] * (k / denom)
        resp_state = 0.85 * np.sin(phi) + 0.15 * np.sin(2.0 * phi + 0.7)
        if resp_state <= phase_values[0]:
            out[k] = stacks[0][k]
            continue
        if resp_state >= phase_values[-1]:
            out[k] = stacks[-1][k]
            continue
        hi = int(np.searchsorted(phase_values, resp_state, side="right"))
        lo = hi - 1
        alpha = (resp_state - phase_values[lo]) / max(phase_values[hi] - phase_values[lo], 1e-6)
        out[k] = (1.0 - alpha) * stacks[lo][k] + alpha * stacks[hi][k]

    out_img = sitk.GetImageFromArray(out)
    out_img.CopyInformation(phase_projection_images[0])
    return out_img


def draw_ellipsoid(mask_arr: np.ndarray, center_zyx, radii_zyx) -> None:
    center = np.array(center_zyx, dtype=np.float32)
    radii = np.maximum(np.array(radii_zyx, dtype=np.float32), 1.0)

    bounds = []
    for dim, radius in zip(mask_arr.shape, radii):
        lower = max(int(np.floor(center[len(bounds)] - radius - 1)), 0)
        upper = min(int(np.ceil(center[len(bounds)] + radius + 2)), dim)
        bounds.append((lower, upper))

    z0, z1 = bounds[0]
    y0, y1 = bounds[1]
    x0, x1 = bounds[2]
    zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
    dist = (
        ((zz - center[0]) / radii[0]) ** 2
        + ((yy - center[1]) / radii[1]) ** 2
        + ((xx - center[2]) / radii[2]) ** 2
    )
    mask_arr[z0:z1, y0:y1, x0:x1][dist <= 1.0] = 1.0


def draw_capsule(mask_arr: np.ndarray, start_zyx, end_zyx, radii_zyx) -> None:
    start = np.array(start_zyx, dtype=np.float32)
    end = np.array(end_zyx, dtype=np.float32)
    radii = np.maximum(np.array(radii_zyx, dtype=np.float32), 1.0)
    num_steps = max(int(np.linalg.norm(end - start) / max(np.min(radii), 1.0)) * 2, 2)
    for alpha in np.linspace(0.0, 1.0, num_steps):
        center = (1.0 - alpha) * start + alpha * end
        draw_ellipsoid(mask_arr, center, radii)


def representative_point(mask_arr: np.ndarray, q: float | None = None):
    coords = np.argwhere(mask_arr)
    if coords.size == 0:
        return None
    if q is None:
        return np.median(coords, axis=0)

    target = float(np.quantile(coords[:, 0], q))
    local = coords[np.abs(coords[:, 0] - target) <= 1.5]
    if local.size == 0:
        local = coords
    return np.median(local, axis=0)


def create_dental_metal_mask(fine_arr: np.ndarray, fine_ids, spacing_xyz, rng) -> np.ndarray | None:
    mandible_id = fine_ids.get("mandible")
    if mandible_id is None:
        return None
    mandible = fine_arr == mandible_id
    if not np.any(mandible):
        return None

    coords = np.argwhere(mandible)
    z_cut = np.quantile(coords[:, 0], 0.45)
    candidate = np.zeros_like(mandible, dtype=bool)
    candidate_coords = coords[coords[:, 0] <= z_cut]
    if candidate_coords.size > 0:
        candidate[candidate_coords[:, 0], candidate_coords[:, 1], candidate_coords[:, 2]] = True
    if not np.any(candidate):
        candidate = mandible

    mask = np.zeros_like(fine_arr, dtype=np.float32)
    radii_mm = np.array([1.5, 2.0, 2.0], dtype=np.float32)
    radii_zyx = radii_mm / np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]], dtype=np.float32)
    candidate_coords = np.argwhere(candidate)
    num_fillings = int(rng.integers(3, 6))
    for _ in range(num_fillings):
        center = candidate_coords[int(rng.integers(0, len(candidate_coords)))]
        draw_ellipsoid(mask, center, radii_zyx)
    return mask


def create_sternum_wire_mask(fine_arr: np.ndarray, fine_ids, spacing_xyz) -> np.ndarray | None:
    sternum_id = fine_ids.get("sternum")
    if sternum_id is None:
        return None
    sternum = fine_arr == sternum_id
    if not np.any(sternum):
        return None

    mask = np.zeros_like(fine_arr, dtype=np.float32)
    radii_mm = np.array([1.2, 1.5, 3.0], dtype=np.float32)
    radii_zyx = radii_mm / np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]], dtype=np.float32)
    coords = np.argwhere(sternum)
    for q in np.linspace(0.18, 0.82, 5):
        z_target = np.quantile(coords[:, 0], q)
        slab = coords[np.abs(coords[:, 0] - z_target) <= 1.5]
        if slab.size == 0:
            continue
        center = np.median(slab, axis=0)
        x_lo = np.quantile(slab[:, 2], 0.35)
        x_hi = np.quantile(slab[:, 2], 0.65)
        start = np.array([center[0], center[1], x_lo], dtype=np.float32)
        end = np.array([center[0], center[1], x_hi], dtype=np.float32)
        draw_capsule(mask, start, end, radii_zyx)
    return mask if np.any(mask) else None


def create_hip_prosthesis_mask(fine_arr: np.ndarray, fine_ids, spacing_xyz) -> np.ndarray | None:
    side_configs = [
        ("hip_right", "femur_right"),
        ("hip_left", "femur_left"),
    ]
    selected = None
    for hip_name, femur_name in side_configs:
        hip_id = fine_ids.get(hip_name)
        femur_id = fine_ids.get(femur_name)
        hip_mask = fine_arr == hip_id if hip_id is not None else None
        femur_mask = fine_arr == femur_id if femur_id is not None else None
        if hip_mask is not None and np.any(hip_mask):
            selected = (hip_mask, femur_mask if femur_mask is not None and np.any(femur_mask) else None)
            break
    if selected is None:
        return None

    hip_mask, femur_mask = selected
    hip_center = representative_point(hip_mask)
    if hip_center is None:
        return None
    if femur_mask is not None and np.any(femur_mask):
        femur_coords = np.argwhere(femur_mask)
        femur_center = np.median(femur_coords, axis=0)
    else:
        femur_center = hip_center + np.array([12.0, 0.0, 0.0], dtype=np.float32)

    mask = np.zeros_like(fine_arr, dtype=np.float32)
    head_radii_mm = np.array([6.0, 6.0, 6.0], dtype=np.float32)
    stem_radii_mm = np.array([4.0, 3.5, 3.5], dtype=np.float32)
    spacing_zyx = np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]], dtype=np.float32)
    draw_ellipsoid(mask, hip_center, head_radii_mm / spacing_zyx)
    draw_capsule(mask, hip_center, femur_center, stem_radii_mm / spacing_zyx)
    return mask if np.any(mask) else None


def create_spine_hardware_mask(fine_arr: np.ndarray, spacing_xyz) -> np.ndarray | None:
    vertebra_ids = list(range(25, 42))
    selected_masks = [fine_arr == label_id for label_id in vertebra_ids if np.any(fine_arr == label_id)]
    if len(selected_masks) < 2:
        return None

    mask = np.zeros_like(fine_arr, dtype=np.float32)
    spacing_zyx = np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]], dtype=np.float32)
    screw_radii = np.array([3.0, 2.0, 2.0], dtype=np.float32) / spacing_zyx
    rod_radii = np.array([2.0, 1.6, 1.6], dtype=np.float32) / spacing_zyx

    right_points = []
    left_points = []
    for vertebra_mask in selected_masks[:4]:
        coords = np.argwhere(vertebra_mask)
        center = np.median(coords, axis=0)
        x_lo = np.quantile(coords[:, 2], 0.3)
        x_hi = np.quantile(coords[:, 2], 0.7)
        right = np.array([center[0], center[1], x_lo], dtype=np.float32)
        left = np.array([center[0], center[1], x_hi], dtype=np.float32)
        right_points.append(right)
        left_points.append(left)
        draw_ellipsoid(mask, right, screw_radii)
        draw_ellipsoid(mask, left, screw_radii)

    for points in (right_points, left_points):
        for start, end in zip(points[:-1], points[1:]):
            draw_capsule(mask, start, end, rod_radii)
    return mask if np.any(mask) else None


def build_metal_model(
    ct_img: sitk.Image,
    fine_mask_img: sitk.Image | None,
    label_maps,
    anatomy: str | None,
    metal_template: str,
    metal_strength: float,
    seed: int,
):
    if fine_mask_img is None:
        return None

    fine_arr = sitk.GetArrayFromImage(fine_mask_img)
    fine_ids = label_maps["fine_name_to_id"]
    spacing_xyz = np.array(ct_img.GetSpacing(), dtype=np.float32)
    rng = np.random.default_rng(seed)

    candidates = []
    if metal_template != "auto":
        candidates = [metal_template]
    elif anatomy == "HN":
        candidates = ["dental", "spine"]
    elif anatomy == "TH":
        candidates = ["sternum", "spine"]
    elif anatomy == "AB":
        candidates = ["hip", "spine"]
    else:
        candidates = ["sternum", "hip", "dental", "spine"]

    builders = {
        "dental": lambda: create_dental_metal_mask(fine_arr, fine_ids, spacing_xyz, rng),
        "sternum": lambda: create_sternum_wire_mask(fine_arr, fine_ids, spacing_xyz),
        "hip": lambda: create_hip_prosthesis_mask(fine_arr, fine_ids, spacing_xyz),
        "spine": lambda: create_spine_hardware_mask(fine_arr, spacing_xyz),
    }
    descriptions = {
        "dental": "mandible-based dental fillings",
        "sternum": "sternal wire-like clips",
        "hip": "hip prosthesis",
        "spine": "posterior spinal hardware",
    }

    for template_name in candidates:
        metal_mask = builders[template_name]()
        if metal_mask is not None and np.any(metal_mask):
            metal_hu_arr = np.zeros_like(metal_mask, dtype=np.float32)
            metal_hu_arr[metal_mask > 0] = 3071.0
            metal_mu_boost_arr = (metal_mask > 0).astype(np.float32) * (0.22 * float(metal_strength))
            return {
                "template": template_name,
                "description": descriptions[template_name],
                "metal_hu_img": image_from_array_like(metal_hu_arr, ct_img, np.float32),
                "metal_mu_boost_img": image_from_array_like(metal_mu_boost_arr, ct_img, np.float32),
            }
    return None


def add_images(img_a: sitk.Image, img_b: sitk.Image) -> sitk.Image:
    arr = sitk.GetArrayFromImage(img_a).astype(np.float32) + sitk.GetArrayFromImage(img_b).astype(np.float32)
    return image_from_array_like(arr, img_a, np.float32)


def combine_hu_with_metal(base_hu_img: sitk.Image, metal_hu_img: sitk.Image | None) -> sitk.Image:
    if metal_hu_img is None:
        return base_hu_img
    base_arr = sitk.GetArrayFromImage(base_hu_img).astype(np.float32)
    metal_arr = sitk.GetArrayFromImage(metal_hu_img).astype(np.float32)
    out = np.where(metal_arr > 0.0, metal_arr, base_arr)
    return image_from_array_like(out, base_hu_img, np.float32)


def run_forward_projection_command(
    rtk_forward_binary: str,
    geometry_path: Path,
    input_volume_path: Path,
    output_path: Path,
    detector_spacing_mm: float,
    detector_dim_text: str,
) -> None:
    cmd_proj = (
        f'"{rtk_forward_binary}" '
        f'-g "{geometry_path}" '
        f'-i "{input_volume_path}" '
        f'-o "{output_path}" '
        f'--spacing {detector_spacing_mm},{detector_spacing_mm} '
        f'--size {detector_dim_text}'
    )
    run(cmd_proj)


def compute_volume_fov_requirements(volume_img: sitk.Image, threshold: float = 1e-6, safety_margin: float = 1.05):
    """
    Estimate the minimum detector field of view required at isocenter.

    The image is assumed to be in the RTK simulation space, i.e. after the
    axis swap used before projection. For a circular orbit around the RTK
    rotation axis:
    - the detector width must cover the maximal cylindrical extent in the
      rotation plane,
    - the detector height must cover the object extent along the rotation axis.

    The support is extracted from positive attenuation values rather than from a
    separate body mask.
    """
    support = sitk.GetArrayFromImage(volume_img) > threshold
    if not np.any(support):
        return {"required_width_mm": 0.0, "required_height_mm": 0.0}

    spacing = np.array(volume_img.GetSpacing(), dtype=np.float64)
    origin = np.array(volume_img.GetOrigin(), dtype=np.float64)
    direction = np.array(volume_img.GetDirection(), dtype=np.float64).reshape(3, 3)

    k_idx, j_idx, i_idx = np.nonzero(support)
    indices_xyz = np.column_stack((i_idx, j_idx, k_idx)).astype(np.float64)
    physical_points = origin + (indices_xyz * spacing) @ direction.T

    x = physical_points[:, 0]
    y = physical_points[:, 1]
    z = physical_points[:, 2]

    required_width_mm = 2.0 * np.max(np.sqrt(x**2 + z**2)) * safety_margin
    required_height_mm = (np.max(y) - np.min(y)) * safety_margin

    return {
        "required_width_mm": float(required_width_mm),
        "required_height_mm": float(required_height_mm),
    }


def auto_fit_detector_geometry(
    volume_img: sitk.Image,
    detector_dim_text: str,
    detector_spacing_mm: float,
    sid_mm: float,
    sdd_mm: float,
    safety_margin: float = 1.05,
):
    """
    Adjust detector spacing and SDD so the body remains inside the field of
    view at isocenter.

    The detector dimensions stay fixed. The function first enlarges detector
    pixel spacing if the physical detector is too small, then reduces SDD if
    necessary to enlarge the field of view at isocenter while keeping SID
    fixed. Because RTK uses a divergent geometry with SDD > SID, the detector
    spacing is slightly inflated when needed so the final FOV at isocenter is
    still guaranteed to cover the requested support.
    """
    detector_dim = parse_csv_numbers(
        detector_dim_text,
        int,
        expected_length=2,
        name="detector-dim",
    )
    det_cols = int(detector_dim[0])
    det_rows = int(detector_dim[1])

    requirements = compute_volume_fov_requirements(volume_img, threshold=1e-6, safety_margin=safety_margin)
    required_width_mm = requirements["required_width_mm"]
    required_height_mm = requirements["required_height_mm"]
    min_valid_sdd_mm = sid_mm + 1.0

    min_spacing_for_isocenter_coverage = max(
        detector_spacing_mm,
        required_width_mm * min_valid_sdd_mm / (sid_mm * max(det_cols, 1)),
        required_height_mm * min_valid_sdd_mm / (sid_mm * max(det_rows, 1)),
    )

    detector_spacing_mm = float(min_spacing_for_isocenter_coverage)
    detector_width_mm = det_cols * detector_spacing_mm
    detector_height_mm = det_rows * detector_spacing_mm

    ratio_needed = max(
        required_width_mm / max(detector_width_mm, 1e-6),
        required_height_mm / max(detector_height_mm, 1e-6),
        1e-6,
    )
    target_sdd_mm = sid_mm / ratio_needed
    fitted_sdd_mm = max(min(sdd_mm, target_sdd_mm), min_valid_sdd_mm)

    fov_width_mm = detector_width_mm * sid_mm / fitted_sdd_mm
    fov_height_mm = detector_height_mm * sid_mm / fitted_sdd_mm

    return {
        "detector_spacing_mm": detector_spacing_mm,
        "sdd_mm": fitted_sdd_mm,
        "required_width_mm": required_width_mm,
        "required_height_mm": required_height_mm,
        "fov_width_mm": fov_width_mm,
        "fov_height_mm": fov_height_mm,
        "detector_width_mm": detector_width_mm,
        "detector_height_mm": detector_height_mm,
    }


def compute_reconstruction_dimensions(
    reference_img: sitk.Image,
    recon_spacing_xyz,
    safety_margin: float = 1.05,
):
    """
    Compute a reconstruction grid size from the physical extent of a reference
    image.

    This keeps the intermediate FDK reconstruction large enough to cover the
    prepared CT volume before the final resampling back to the original CT
    geometry.
    """
    reference_size = np.array(reference_img.GetSize(), dtype=np.float64)
    reference_spacing = np.array(reference_img.GetSpacing(), dtype=np.float64)
    recon_spacing_xyz = np.array(recon_spacing_xyz, dtype=np.float64)

    physical_extent = reference_size * reference_spacing * safety_margin
    recon_dim = np.ceil(physical_extent / recon_spacing_xyz).astype(np.int32)
    recon_dim = np.maximum(recon_dim, 1)
    return [int(v) for v in recon_dim.tolist()]


def validate_runtime_parameters(parser: argparse.ArgumentParser, args) -> None:
    scalar_constraints = {
        "iso-spacing": args.iso_spacing,
        "detector-spacing": args.detector_spacing,
        "sid": args.sid,
        "sdd": args.sdd,
        "nproj": args.nproj,
        "arc": args.arc,
        "fov-margin": args.fov_margin,
        "recon-margin": args.recon_margin,
        "photon-fluence": args.photon_fluence,
        "tube-current": args.tube_current,
        "exposure-ms": args.exposure_ms,
        "saturation": args.saturation,
        "mu-water": args.mu_water,
        "recon-spacing-scale": args.recon_spacing_scale,
    }
    for name, value in scalar_constraints.items():
        if value <= 0:
            parser.error(f"Invalid {name}: expected a strictly positive value, got {value}.")

    if args.scatter_sigma < 0:
        parser.error(f"Invalid scatter-sigma: expected a non-negative value, got {args.scatter_sigma}.")
    if args.detector_blur_sigma < 0:
        parser.error(f"Invalid detector-blur-sigma: expected a non-negative value, got {args.detector_blur_sigma}.")
    if args.shading_strength < 0:
        parser.error(f"Invalid shading-strength: expected a non-negative value, got {args.shading_strength}.")
    if args.spr < 0:
        parser.error(f"Invalid spr: expected a non-negative value, got {args.spr}.")
    if args.motion_strength < 0:
        parser.error(f"Invalid motion-strength: expected a non-negative value, got {args.motion_strength}.")
    if args.metal_strength < 0:
        parser.error(f"Invalid metal-strength: expected a non-negative value, got {args.metal_strength}.")
    if args.motion_phases < 2:
        parser.error(f"Invalid motion-phases: expected at least 2 phases, got {args.motion_phases}.")
    if args.motion_cycles < 0:
        parser.error(f"Invalid motion-cycles: expected a non-negative value, got {args.motion_cycles}.")
    if args.truncation_factor <= 0 or args.truncation_factor > 1.0:
        parser.error(
            f"Invalid truncation-factor: expected a value in (0, 1], got {args.truncation_factor}."
        )


TASK2_CBCT_TABLES = {
    "HN": {
        "label": "Head-and-Neck",
        "simulation": {
            "spr": 1.1,
            "saturation": 1.5,
            "scatter_sigma": 4.5,
        },
        "centers": {
            "A": {"manufacturer": "Elekta", "model": "XVI v5.x", "kVp": "100-120", "detector_spacing": 1.0, "detector_dim": "270,270", "tube_current": 16.0, "exposure_ms": 21.0},
            "B": {"manufacturer": "Elekta", "model": "XVI v5.52", "kVp": "100", "detector_spacing": 1.0, "detector_dim": "270,270", "tube_current": 10.0, "exposure_ms": 10.0},
            "C": {"manufacturer": "Elekta", "model": "XVI v5.x", "kVp": "120", "detector_spacing": 1.0, "detector_dim": "270,270", "tube_current": 15.0, "exposure_ms": 22.0},
            "D": {"manufacturer": "IBA / Elekta", "model": "Proteus P+ / XVI v5.x", "kVp": "100", "detector_spacing": 0.75, "detector_dim": "384,384", "tube_current": 160.0, "exposure_ms": 3225.0},
            "E": {"manufacturer": "Varian", "model": "TrueBeam OBI", "kVp": "100-125", "detector_spacing": 0.7, "detector_dim": "512,512", "tube_current": 15.0, "exposure_ms": 12780.0},
        },
    },
    "TH": {
        "label": "Thorax",
        "simulation": {
            "spr": 1.1,
            "saturation": 1.5,
            "scatter_sigma": 4.5,
        },
        "centers": {
            "A": {"manufacturer": "Elekta", "model": "XVI v5.x", "kVp": "100-120", "detector_spacing": 1.0, "detector_dim": "270,270", "tube_current": 30.0, "exposure_ms": 25.0},
            "B": {"manufacturer": "Elekta", "model": "XVI v5.x", "kVp": "120", "detector_spacing": 1.0, "detector_dim": "410,410", "tube_current": 40.0, "exposure_ms": 40.0},
            "C": {"manufacturer": "Elekta", "model": "XVI v5.x", "kVp": "120", "detector_spacing": 1.5, "detector_dim": "270,270", "tube_current": 25.0, "exposure_ms": 28.0},
            "D": {"manufacturer": "IBA / Elekta", "model": "Proteus P+ / XVI v5.x", "kVp": "110-120", "detector_spacing": 0.73, "detector_dim": "519,519", "tube_current": 168.0, "exposure_ms": 2955.0},
            "E": {"manufacturer": "Varian", "model": "TrueBeam OBI", "kVp": "100-125", "detector_spacing": 0.7, "detector_dim": "512,512", "tube_current": 46.0, "exposure_ms": 9915.0},
        },
    },
    "AB": {
        "label": "Abdomen",
        "simulation": {
            "spr": 1.1,
            "saturation": 1.5,
            "scatter_sigma": 4.5,
        },
        "centers": {
            "A": {"manufacturer": "Elekta", "model": "XVI 5.x", "kVp": "100-120", "detector_spacing": 1.0, "detector_dim": "410,410", "tube_current": 42.0, "exposure_ms": 25.0},
            "B": {"manufacturer": "Elekta", "model": "XVI", "kVp": "120", "detector_spacing": 1.0, "detector_dim": "410,410", "tube_current": 40.0, "exposure_ms": 40.0},
            "C": {"manufacturer": "Elekta", "model": "XVI v5.x", "kVp": "120", "detector_spacing": 1.0, "detector_dim": "410,410", "tube_current": 25.0, "exposure_ms": 30.0},
            "D": {"manufacturer": "Elekta / IBA", "model": "XVI v5.x / Proteus P+", "kVp": "120-125", "detector_spacing": 0.825, "detector_dim": "519,519", "tube_current": 168.0, "exposure_ms": 2955.0},
            "E": {"manufacturer": "Varian", "model": "TrueBeam OBI", "kVp": "125-140", "detector_spacing": 0.7, "detector_dim": "512,512", "tube_current": 57.0, "exposure_ms": 9515.0},
        },
    },
}


# Supplemental random scanner profiles used only for synthetic generation.
# They are not part of the Task 2 A-E mapping and therefore are intentionally
# kept separate from TASK2_CBCT_TABLES.
#
# Selection rationale:
# - HN center A from the provided table is already well covered by the existing
#   Elekta HN presets and is therefore omitted.
# - The provided abdomen table is already closely matched by the existing AB
#   Elekta presets (A/B/C) and is therefore not duplicated here.
# - Two HN profiles are retained because they add materially different
#   acquisition regimes compared with the current pool.
SUPPLEMENTAL_RANDOM_CBCT_PROFILES = {
    "HN": [
        {
            "center": "S1",
            "profile_id": "hn_iba_proteus_plus_hires80",
            "profile_source": "supplemental_sequence_table",
            "manufacturer": "IBA",
            "model": "Proteus Plus",
            "kVp": "80",
            "detector_dim_choices": ["512,512"],
            "detector_spacing_range": (0.51, 0.51),
            "tube_current_range": (50.0, 50.0),
            "exposure_ms_range": (3225.0, 3225.0),
            "slice_thickness_range": (2.5, 2.5),
            "reconstruction_diameter_range": (260.0, 260.0),
        },
        {
            "center": "S2",
            "profile_id": "hn_elekta_xvi_large_fov_high_output",
            "profile_source": "supplemental_sequence_table",
            "manufacturer": "Elekta",
            "model": "XVI",
            "kVp": "120",
            "detector_dim_choices": ["512,512"],
            "detector_spacing_range": (0.61, 1.17),
            "tube_current_range": (239.0, 497.0),
            "exposure_ms_range": (888.0, 2661.0),
            "slice_thickness_range": (1.0, 3.0),
            "reconstruction_diameter_range": (310.0, 600.0),
        },
    ],
    "TH": [],
    "AB": [],
}


MACHINE_GEOMETRY_PRESETS = {
    # Plausible starting geometries by machine family.
    # These are used as priors before the anatomy-driven FOV auto-fit step.
    "elekta_xvi": {
        "sid": 1000.0,
        "sdd": 1536.0,
    },
    "varian_obi": {
        "sid": 1000.0,
        "sdd": 1500.0,
    },
    "iba_proteus": {
        "sid": 1000.0,
        "sdd": 1536.0,
    },
}


def infer_machine_geometry_preset(manufacturer: str, model: str):
    """
    Map manufacturer/model strings to a machine-family geometry preset.

    The returned SID/SDD values are machine-family priors, not exact
    center-specific calibrations. They are intentionally followed by an
    anatomy-driven FOV fitting step to avoid truncation artifacts.
    """
    manufacturer_l = manufacturer.lower()
    model_l = model.lower()
    combined = f"{manufacturer_l} {model_l}"

    if "truebeam" in combined or "obi" in combined or "varian" in combined:
        family = "varian_obi"
    elif "proteus" in combined or "iba" in combined:
        family = "iba_proteus"
    elif "xvi" in combined or "elekta" in combined:
        family = "elekta_xvi"
    else:
        return None

    preset = dict(MACHINE_GEOMETRY_PRESETS[family])
    preset["family"] = family
    return preset


def build_random_scanner_profile_pool(anatomy: str):
    """
    Return the full scanner-profile pool used for random synthetic generation.

    This includes:
    - the static Task 2 A-E scanner presets,
    - optional supplemental profiles that are only used for random generation.
    """
    table_entry = TASK2_CBCT_TABLES[anatomy]
    simulation = dict(table_entry.get("simulation", {}))
    profiles = []

    for center, center_entry in table_entry["centers"].items():
        profile = {
            "anatomy": anatomy,
            "anatomy_label": table_entry["label"],
            "center": center,
            **simulation,
            **dict(center_entry),
        }
        geometry = infer_machine_geometry_preset(
            profile["manufacturer"],
            profile["model"],
        )
        if geometry is not None:
            profile["sid"] = geometry["sid"]
            profile["sdd"] = geometry["sdd"]
            profile["geometry_family"] = geometry["family"]
        profiles.append(profile)

    for extra_entry in SUPPLEMENTAL_RANDOM_CBCT_PROFILES.get(anatomy, []):
        profile = {
            "anatomy": anatomy,
            "anatomy_label": table_entry["label"],
            **simulation,
            **dict(extra_entry),
        }
        geometry = infer_machine_geometry_preset(
            profile["manufacturer"],
            profile["model"],
        )
        if geometry is not None:
            profile["sid"] = geometry["sid"]
            profile["sdd"] = geometry["sdd"]
            profile["geometry_family"] = geometry["family"]
        profiles.append(profile)

    return profiles


def infer_task2_case_from_path(path: Path):
    """
    Parse Task 2 anatomy and center from a case name such as 2ABA001.mha.
    Expected format: 2 + anatomy(HN|TH|AB) + center(A-E) + three digits.
    """
    match = re.match(r"^2(HN|TH|AB)([A-E])\d{3}$", path.stem.upper())
    if match is None:
        return None
    anatomy, center = match.groups()
    return {"anatomy": anatomy, "center": center}


def resolve_task2_scanner_profile(path: Path):
    case_info = infer_task2_case_from_path(path)
    if case_info is None:
        return None

    anatomy = case_info["anatomy"]
    center = case_info["center"]
    table_entry = TASK2_CBCT_TABLES[anatomy]
    center_entry = table_entry["centers"][center]

    resolved = dict(center_entry)
    resolved["anatomy"] = anatomy
    resolved["anatomy_label"] = table_entry["label"]
    resolved.update(table_entry.get("simulation", {}))
    resolved["center"] = center
    resolved["case_id"] = path.stem
    return resolved


def print_task2_scanner_tables() -> None:
    print("Available Task 2 CBCT scanner tables:")
    for anatomy in ("HN", "TH", "AB"):
        table_entry = TASK2_CBCT_TABLES[anatomy]
        simulation = table_entry.get("simulation", {})
        print(
            f"  {anatomy} - {table_entry['label']} | "
            f"SPR={simulation.get('spr', 'n/a')}, "
            f"saturation={simulation.get('saturation', 'n/a')}, "
            f"scatter_sigma={simulation.get('scatter_sigma', 'n/a')}"
        )
        for center in ("A", "B", "C", "D", "E"):
            entry = table_entry["centers"][center]
            print(
                f"    center {center}: {entry['manufacturer']} | {entry['model']} | "
                f"det={entry['detector_dim']} px @ {entry['detector_spacing']} mm | "
                f"{entry['tube_current']} mA | {entry['exposure_ms']} ms | kVp {entry['kVp']}"
            )


STANDARD_ACQUISITION_PRESETS = {
    # These presets intentionally stay below the 600-view ceiling requested for
    # faster synthetic generation while remaining close to plausible 3D-CBCT
    # operating points.
    "quick": {
        "label": "Fast 3D-like CBCT",
        "nproj": 240,
        "motion_phases": 4,
    },
    "balanced": {
        "label": "Balanced 3D-like CBCT",
        "nproj": 360,
        "motion_phases": 6,
    },
    "dense": {
        "label": "Dense 3D-like CBCT",
        "nproj": 540,
        "motion_phases": 8,
    },
    "thorax": {
        "label": "Thorax standard 3D-like CBCT",
        "nproj": 420,
        "motion_phases": 6,
    },
    "abdomen": {
        "label": "Abdomen standard 3D-like CBCT",
        "nproj": 360,
        "motion_phases": 6,
    },
}


def resolve_standard_acquisition_preset_name(
    preset_name: str,
    inferred_profile,
) -> str | None:
    if preset_name == "none":
        return None

    if preset_name != "auto":
        return preset_name

    if inferred_profile is not None:
        anatomy = inferred_profile.get("anatomy")
        if anatomy == "TH":
            return "thorax"
        if anatomy == "AB":
            return "abdomen"
    return "balanced"


def apply_standard_acquisition_preset(args, inferred_profile) -> str | None:
    resolved_name = resolve_standard_acquisition_preset_name(
        args.standard_preset,
        inferred_profile=inferred_profile,
    )
    if resolved_name is None:
        if args.nproj is None:
            args.nproj = 667
        if args.motion_phases is None:
            args.motion_phases = 5
        return None

    preset = STANDARD_ACQUISITION_PRESETS[resolved_name]
    if args.nproj is None:
        args.nproj = preset["nproj"]
    if args.motion_phases is None:
        args.motion_phases = preset["motion_phases"]
    return resolved_name


def print_standard_acquisition_presets() -> None:
    print("Available standard CBCT acquisition presets:")
    print("  Motion phases are only used when motion artifacts are enabled.")
    print("  The presets stay within nproj=200-600 and motion-phases=4-8.")
    for preset_name in ("quick", "balanced", "dense", "thorax", "abdomen"):
        preset = STANDARD_ACQUISITION_PRESETS[preset_name]
        print(
            f"  {preset_name}: {preset['label']} | "
            f"nproj={preset['nproj']} | motion-phases={preset['motion_phases']}"
        )
    print("  auto: thorax -> thorax, abdomen -> abdomen, otherwise balanced")
    print("  none: keep the fallback defaults (nproj=667, motion-phases=5)")


def hu_to_mu(img_hu: sitk.Image, mu_water: float = 0.018) -> sitk.Image:
    arr = sitk.GetArrayFromImage(img_hu).astype(np.float32)
    mu = mu_water * (arr + 1000.0) / 1000.0
    mu = np.clip(mu, 0.0, None)  # Clamp air and negative values for stability.
    out = sitk.GetImageFromArray(mu.astype(np.float32))
    out.CopyInformation(img_hu)
    return out


def make_bowtie_profile(nx: int, ny: int, sigma_ratio_x: float = 0.42, sigma_ratio_y: float = 1.5):
    """
    Simple 2D Gaussian profile used as a crude bow-tie / beam profile model.

    The center is normalized to 1.0 and the edges are attenuated.
    """
    x = np.linspace(-1.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, ny, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    sx = sigma_ratio_x
    sy = sigma_ratio_y
    bp = np.exp(-0.5 * ((xx / sx) ** 2 + (yy / sy) ** 2))
    bp = bp / np.max(bp)
    return bp.astype(np.float32)


def make_smooth_random_field(shape_yx, rng, sigma_yx) -> np.ndarray:
    field = rng.normal(0.0, 1.0, size=shape_yx).astype(np.float32)
    field = gaussian_filter(field, sigma=sigma_yx)
    field -= np.mean(field)
    max_abs = float(np.max(np.abs(field)))
    if max_abs > 0:
        field /= max_abs
    return field.astype(np.float32)


def attenuation_to_noisy_projection(
    proj_attn: np.ndarray,
    detector_spacing_mm: float,
    photon_fluence_per_mm2_mAs: float = 4.16e5,
    tube_current_mA: float = 40.0,
    exposure_ms: float = 40.0,
    spr: float = 1.6,
    saturation_factor: float = 2.0,
    add_scatter_blur_sigma_px: float = 7.0,
    detector_blur_sigma_px: float = 0.0,
    shading_field: np.ndarray | None = None,
    shading_strength: float = 0.0,
    metal_trace: np.ndarray | None = None,
    metal_starvation_strength: float = 0.0,
    metal_streak_strength: float = 0.0,
    seed: int = 1234,
):
    """
    Convert one clean attenuation projection into a noisy CBCT-like projection.

    proj_attn stores P(i, j) = integral(mu dl).

    The noise model is intentionally simple:
    I0 = BP * mAs * Phi * pixel_area
    N_primary = exp(-P) * I0
    N_scatter = SPR * percentile_5(N_primary) * Omega
    N_detected ~ Poisson(N_primary + N_scatter)
    P_noisy = -ln(clamp(csat * N_detected / I0, 1e-6, 1.0))
    """
    rng = np.random.default_rng(seed)

    h, w = proj_attn.shape
    pixel_area = detector_spacing_mm * detector_spacing_mm
    mAs = tube_current_mA * (exposure_ms / 1000.0)
    bowtie = make_bowtie_profile(w, h)  # shape (h,w)
    I0 = bowtie * mAs * photon_fluence_per_mm2_mAs * pixel_area
    I0 = np.clip(I0, 1.0, None)
    if shading_field is not None and shading_strength > 0:
        I0 = I0 * np.exp(shading_strength * np.clip(shading_field.astype(np.float32), -1.0, 1.0))
        I0 = np.clip(I0, 1.0, None)

    metal_strength = np.zeros_like(proj_attn, dtype=np.float32)
    if metal_trace is not None:
        metal_strength = 1.0 - np.exp(-np.clip(metal_trace.astype(np.float32), 0.0, None) / 0.6)
        I0 = I0 * np.exp(-metal_starvation_strength * metal_strength)
        I0 = np.clip(I0, 1.0, None)

    # Primary transmission signal.
    T_primary = np.exp(-np.clip(proj_attn, 0.0, 20.0))
    N_primary = T_primary * I0
    if detector_blur_sigma_px > 0:
        N_primary = gaussian_filter(N_primary, sigma=detector_blur_sigma_px)

    # Crude attenuation mask used to localize scatter.
    omega = (T_primary < 0.5).astype(np.float32)

    # Heuristic low-frequency scatter term.
    p5 = np.percentile(N_primary, 5.0)
    N_scatter = spr * p5 * omega
    if add_scatter_blur_sigma_px > 0:
        N_scatter = gaussian_filter(N_scatter, sigma=add_scatter_blur_sigma_px)
    if metal_trace is not None:
        N_scatter = N_scatter * (1.0 + 0.5 * metal_strength)

    lam = np.clip(N_primary + N_scatter, 0.0, None)
    N_detected = rng.poisson(lam).astype(np.float32)

    ratio = saturation_factor * (N_detected / I0)
    ratio = np.clip(ratio, 1e-6, 1.0)
    P_noisy = -np.log(ratio)
    if metal_trace is not None and metal_streak_strength > 0:
        streak_seed = metal_strength * (0.85 + 0.3 * rng.random(metal_strength.shape, dtype=np.float32))
        streak = gaussian_filter(streak_seed, sigma=(0.8, 10.0))
        streak += 0.45 * gaussian_filter(streak_seed, sigma=(6.0, 1.0))
        P_noisy = P_noisy + metal_streak_strength * streak.astype(np.float32)

    return P_noisy.astype(np.float32)


def process_projection_stack(
    proj_img: sitk.Image,
    detector_spacing_mm: float,
    photon_fluence_per_mm2_mAs: float,
    tube_current_mA: float,
    exposure_ms: float,
    spr: float,
    saturation_factor: float,
    scatter_blur_sigma_px: float,
    detector_blur_sigma_px: float,
    shading_strength: float,
    metal_trace_img: sitk.Image | None,
    metal_starvation_strength: float,
    metal_streak_strength: float,
    seed: int,
) -> sitk.Image:
    """
    Apply the projection degradation model independently to each projection.

    SimpleITK array ordering is z, y, x.
    Here:
      z = projection index
      y, x = detector coordinates
    """
    arr = sitk.GetArrayFromImage(proj_img).astype(np.float32)
    metal_arr = sitk.GetArrayFromImage(metal_trace_img).astype(np.float32) if metal_trace_img is not None else None
    out = np.empty_like(arr, dtype=np.float32)
    shading_base = None
    shading_aux = None
    shading_phase0 = 0.0
    nproj = arr.shape[0]
    denom = max(nproj - 1, 1)

    if shading_strength > 0:
        rng = np.random.default_rng(seed + 100003)
        h, w = arr.shape[1], arr.shape[2]
        shading_base = make_smooth_random_field(
            (h, w),
            rng=rng,
            sigma_yx=(max(h * 0.18, 1.0), max(w * 0.18, 1.0)),
        )
        shading_aux = make_smooth_random_field(
            (h, w),
            rng=rng,
            sigma_yx=(max(h * 0.28, 1.0), max(w * 0.28, 1.0)),
        )
        shading_phase0 = float(rng.uniform(0.0, 2.0 * np.pi))

    for k in range(arr.shape[0]):
        shading_field = None
        if shading_base is not None and shading_aux is not None:
            angle = shading_phase0 + 2.0 * np.pi * (k / denom)
            shading_field = (
                np.cos(angle) * shading_base
                + 0.35 * np.sin(0.7 * angle + 0.9) * shading_aux
            ).astype(np.float32)
            shading_field = np.clip(shading_field, -1.0, 1.0)

        out[k] = attenuation_to_noisy_projection(
            arr[k],
            detector_spacing_mm=detector_spacing_mm,
            photon_fluence_per_mm2_mAs=photon_fluence_per_mm2_mAs,
            tube_current_mA=tube_current_mA,
            exposure_ms=exposure_ms,
            spr=spr,
            saturation_factor=saturation_factor,
            add_scatter_blur_sigma_px=scatter_blur_sigma_px,
            detector_blur_sigma_px=detector_blur_sigma_px,
            shading_field=shading_field,
            shading_strength=shading_strength,
            metal_trace=None if metal_arr is None else metal_arr[k],
            metal_starvation_strength=metal_starvation_strength,
            metal_streak_strength=metal_streak_strength,
            seed=seed + k,
        )

    out_img = sitk.GetImageFromArray(out)
    out_img.CopyInformation(proj_img)
    return out_img


def mu_to_hu_like(img_mu: sitk.Image, mu_water: float = 0.018) -> sitk.Image:
    arr = sitk.GetArrayFromImage(img_mu).astype(np.float32)
    hu = 1000.0 * (arr / mu_water) - 1000.0
    hu = np.clip(hu, -1024.0, 3071.0)
    out = sitk.GetImageFromArray(hu.astype(np.float32))
    out.CopyInformation(img_mu)
    return out


def calibrate_cbct_mu_to_reference_ct(
    cbct_mu_img: sitk.Image,
    reference_ct_img: sitk.Image,
    mu_water: float = 0.018,
    body_threshold_hu: float = -500.0,
    target_contrast_scale: float = 0.90,
    min_body_voxels: int = 1000,
):
    """
    Rescale reconstructed mu values to a plausible HU-like dynamic range.

    RTK reconstruction combined with the heuristic projection corruption can
    occasionally compress the attenuation range too strongly, producing volumes
    that are almost entirely air-like after HU conversion. Because the
    synthetic CBCT is ultimately supervised from the source CT, we can safely
    use the source CT as a weak intensity anchor and apply a body-restricted
    multiplicative mu calibration.

    The calibration remains intentionally conservative:
    - air stays fixed because the mapping is multiplicative in mu,
    - the target contrast is kept slightly below the reference CT via
      target_contrast_scale,
    - the gain is clipped to avoid extreme corrections.
    """
    cbct_mu = sitk.GetArrayFromImage(cbct_mu_img).astype(np.float32)
    reference_ct = sitk.GetArrayFromImage(reference_ct_img).astype(np.float32)

    body_mask = reference_ct > body_threshold_hu
    body_voxels = int(np.count_nonzero(body_mask))
    if body_voxels < min_body_voxels:
        return cbct_mu_img, None

    reference_mu = mu_water * (reference_ct + 1000.0) / 1000.0
    reference_mu = np.clip(reference_mu, 0.0, None)

    cbct_body = cbct_mu[body_mask]
    reference_body = reference_mu[body_mask]
    if cbct_body.size == 0 or reference_body.size == 0:
        return cbct_mu_img, None

    cbct_q = np.percentile(cbct_body, [50.0, 90.0]).astype(np.float32)
    reference_q = np.percentile(reference_body, [50.0, 90.0]).astype(np.float32)
    reference_q *= float(target_contrast_scale)

    gain_candidates = reference_q / np.maximum(cbct_q, 1e-6)
    gain = float(np.median(gain_candidates))
    gain = float(np.clip(gain, 0.75, 4.0))

    calibrated = cbct_mu * gain
    calibrated_img = image_from_array_like(calibrated, cbct_mu_img, dtype=np.float32)
    return calibrated_img, {
        "gain": gain,
        "body_voxels": body_voxels,
        "reference_mu_q50": float(reference_q[0]),
        "reference_mu_q90": float(reference_q[1]),
        "cbct_mu_q50": float(cbct_q[0]),
        "cbct_mu_q90": float(cbct_q[1]),
    }


def cast_hu_image_to_int16(img: sitk.Image) -> sitk.Image:
    """
    Convert a HU-like image to int16 for compact on-disk storage.

    Values are rounded to the nearest integer and clipped to a standard CT-like
    range before casting.
    """
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    arr = np.rint(arr).clip(-1024.0, 3071.0).astype(np.int16)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out


def main():
    table_parser = argparse.ArgumentParser(add_help=False)
    table_parser.add_argument("--list-scanner-tables", action="store_true", help="List Task 2 scanner tables")
    table_parser.add_argument("--list-standard-presets", action="store_true", help="List standard CBCT acquisition presets")
    table_args, remaining_argv = table_parser.parse_known_args()

    if table_args.list_scanner_tables:
        print_task2_scanner_tables()
    if table_args.list_standard_presets:
        if table_args.list_scanner_tables:
            print()
        print_standard_acquisition_presets()
    if table_args.list_scanner_tables or table_args.list_standard_presets:
        return

    ensure_python_dependencies()

    parser = argparse.ArgumentParser(
        description="Generate a simulated CBCT from a planning CT.",
        parents=[table_parser],
    )
    parser.add_argument("--input", required=True, help="Input CT (MHA/MHD/NIfTI)")
    parser.add_argument("--outdir", default="cbct_out", help="Output directory for final CBCT files")
    parser.add_argument("--rtk-bin-dir", default=None, help="Directory containing RTK CLI binaries such as rtkfdk. If omitted, the script first tries local RTK/build/bin folders, then PATH.")
    parser.add_argument("--mask-fine", default=None, help="Fine anatomical mask aligned with the input CT. If omitted, the script tries to use a sibling mask_fine.nii.gz.")
    parser.add_argument("--mask-559", default=None, help="Broad anatomical mask aligned with the input CT. If omitted, the script tries to use a sibling mask_559.nii.gz.")
    parser.add_argument("--label-schema", default=None, help="Path to task_2maps_labels.json. If omitted, the script tries repo-local defaults.")
    parser.add_argument("--iso-spacing", type=float, default=1.0, help="Isotropic CT spacing in mm")
    parser.add_argument("--detector-spacing", type=float, default=0.8, help="Detector pixel spacing in mm [overridden for inferred Task 2 cases and auto-fitted to keep anatomy in FOV]")
    parser.add_argument("--detector-dim", default="512,512", help="Detector dimensions, e.g. 512,512 [overridden for inferred Task 2 cases]")
    parser.add_argument("--sid", type=float, default=1000.0, help="Source to isocenter distance (mm) [overridden by machine-family preset for inferred Task 2 cases]")
    parser.add_argument("--sdd", type=float, default=1536.0, help="Source to detector distance (mm) [overridden by machine-family preset for inferred Task 2 cases, then auto-fitted to avoid truncation]")
    parser.add_argument("--standard-preset", choices=["auto", "quick", "balanced", "dense", "thorax", "abdomen", "none"], default="auto", help="Standard acquisition preset used to fill nproj and motion-phases when they are not given explicitly.")
    parser.add_argument("--nproj", type=int, default=None, help="Number of projections. If omitted, the selected standard preset provides a default.")
    parser.add_argument("--arc", type=float, default=360.0, help="Total scan arc in degrees")
    parser.add_argument("--recon-spacing", default="1,1,3", help="Reconstruction spacing in mm, e.g. 1,1,3")
    parser.add_argument("--recon-dim", default="410,410,66", help="Unused: reconstruction dimensions are auto-sized from the prepared CT extent.")
    parser.add_argument("--fov-margin", type=float, default=1.15, help="Safety margin used to keep anatomy inside the detector field of view")
    parser.add_argument("--recon-margin", type=float, default=1.15, help="Safety margin used to size the intermediate FDK reconstruction volume")
    parser.add_argument("--photon-fluence", type=float, default=4.16e5, help="Photon fluence in photons/(mm^2*mAs)")
    parser.add_argument("--tube-current", type=float, default=40.0, help="Tube current (mA) [overridden for inferred Task 2 cases]")
    parser.add_argument("--exposure-ms", type=float, default=40.0, help="Exposure time per projection (ms) [overridden for inferred Task 2 cases]")
    parser.add_argument("--spr", type=float, default=1.6, help="Heuristic scatter-to-primary ratio [overridden by anatomy-specific Task 2 appearance presets when available]")
    parser.add_argument("--saturation", type=float, default=2.0, help="Detector saturation factor [overridden by anatomy-specific Task 2 appearance presets when available]")
    parser.add_argument("--scatter-sigma", type=float, default=7.0, help="Scatter blur sigma in pixels [overridden by anatomy-specific Task 2 appearance presets when available]")
    parser.add_argument("--detector-blur-sigma", type=float, default=0.0, help="Additional detector/focal-spot blur applied in projection space, in pixels.")
    parser.add_argument("--shading-strength", type=float, default=0.0, help="Low-frequency projection shading strength. 0 disables the effect.")
    parser.add_argument("--motion-artifact", choices=["off", "auto", "on"], default="auto", help="Respiratory motion artifact synthesis. 'auto' enables it only for thorax/abdomen cases with usable masks.")
    parser.add_argument("--motion-strength", type=float, default=1.0, help="Global multiplier for respiratory motion amplitude.")
    parser.add_argument("--motion-phases", type=int, default=None, help="Number of respiratory phases used to synthesize motion-corrupted projections. If omitted, the selected standard preset provides a default.")
    parser.add_argument("--motion-cycles", type=float, default=0.0, help="Respiratory cycles across the CBCT acquisition. 0 selects an anatomy-specific default.")
    parser.add_argument("--metal-artifact", choices=["off", "auto", "on"], default="auto", help="Synthetic metal/prosthesis artifact synthesis.")
    parser.add_argument("--metal-template", choices=["auto", "dental", "sternum", "hip", "spine"], default="auto", help="Anatomical metal template to inject when metal artifacts are enabled.")
    parser.add_argument("--metal-strength", type=float, default=1.0, help="Global multiplier for synthetic metal attenuation and streaking strength.")
    parser.add_argument("--truncation-factor", type=float, default=1.0, help="Scale factor applied to the fitted detector FOV at isocenter. Values below 1 induce controlled truncation.")
    parser.add_argument("--isocenter-shift", default="0,0,0", help="Physical miscentering in mm applied before projection, formatted as x,y,z.")
    parser.add_argument("--recon-spacing-scale", type=float, default=1.0, help="Uniform multiplier applied to reconstruction spacing to generate a coarser FDK reconstruction.")
    parser.add_argument("--mu-water", type=float, default=0.018, help="Water attenuation coefficient in mm^-1")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep intermediate files")
    args = parser.parse_args(remaining_argv)

    input_path = Path(args.input).resolve()
    inferred_profile = resolve_task2_scanner_profile(input_path)
    selected_standard_preset = apply_standard_acquisition_preset(args, inferred_profile)
    try:
        parse_csv_numbers(args.detector_dim, int, expected_length=2, name="detector-dim")
        parse_csv_numbers(args.recon_spacing, float, expected_length=3, name="recon-spacing")
        args.isocenter_shift = parse_csv_numbers(
            args.isocenter_shift,
            float,
            expected_length=3,
            name="isocenter-shift",
        )
    except ValueError as exc:
        parser.error(str(exc))
    validate_runtime_parameters(parser, args)
    inferred_geometry = None
    if inferred_profile is not None:
        args.detector_spacing = inferred_profile["detector_spacing"]
        args.detector_dim = inferred_profile["detector_dim"]
        args.tube_current = inferred_profile["tube_current"]
        args.exposure_ms = inferred_profile["exposure_ms"]
        if "spr" in inferred_profile:
            args.spr = inferred_profile["spr"]
        if "saturation" in inferred_profile:
            args.saturation = inferred_profile["saturation"]
        if "scatter_sigma" in inferred_profile:
            args.scatter_sigma = inferred_profile["scatter_sigma"]
        inferred_geometry = infer_machine_geometry_preset(
            inferred_profile["manufacturer"],
            inferred_profile["model"],
        )
        if inferred_geometry is not None:
            args.sid = inferred_geometry["sid"]
            args.sdd = inferred_geometry["sdd"]

    rtk_search_dirs = []
    if args.rtk_bin_dir is not None:
        rtk_search_dirs.append(Path(args.rtk_bin_dir))
    rtk_search_dirs.extend(candidate_rtk_bin_dirs())

    rtk_binaries = {
        "rtksimulatedgeometry": resolve_binary("rtksimulatedgeometry", search_dirs=rtk_search_dirs),
        "rtkforwardprojections": resolve_binary("rtkforwardprojections", search_dirs=rtk_search_dirs),
        "rtkfdk": resolve_binary("rtkfdk", search_dirs=rtk_search_dirs),
    }

    label_schema_path = resolve_label_schema_path(args.label_schema)
    mask_fine_path = resolve_auxiliary_mask_path(args.mask_fine, input_path, "mask_fine.nii.gz")
    mask_559_path = resolve_auxiliary_mask_path(args.mask_559, input_path, "mask_559.nii.gz")
    label_maps = load_label_schema_maps(label_schema_path) if label_schema_path is not None else None

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    case_id = input_path.stem
    workdir = outdir / f".{case_id}_rtk_tmp"
    workdir.mkdir(parents=True, exist_ok=True)

    if inferred_profile is not None:
        print("[0/9] Scanner settings inferred from Task 2 case name...")
        print("    case    =", inferred_profile["case_id"])
        print("    anatomy =", inferred_profile["anatomy"], f"({inferred_profile['anatomy_label']})")
        print("    center  =", inferred_profile["center"])
        print("    system  =", inferred_profile["manufacturer"], "|", inferred_profile["model"])
        print("    kVp     =", inferred_profile["kVp"])
        print("    det     =", args.detector_dim, "px @", args.detector_spacing, "mm")
        print("    dose    =", args.tube_current, "mA ;", args.exposure_ms, "ms")
        print("    noise   =", f"SPR={args.spr}, saturation={args.saturation}, scatter_sigma={args.scatter_sigma}")
        if inferred_geometry is not None:
            print("    geom    =", inferred_geometry["family"], f"(SID={args.sid} mm, SDD={args.sdd} mm)")
    else:
        print("[0/9] No Task 2 scanner profile inferred from input name.")
        print("    Expected pattern: 2HNA001, 2THB001, 2ABA001, ...")
        print("    Using CLI values for detector and exposure settings.")
    print("    preset   =", selected_standard_preset if selected_standard_preset is not None else "none")
    print("    nproj    =", args.nproj)
    print("    phases   =", args.motion_phases, "(used when motion artifacts are enabled)")
    print("    shift    =", args.isocenter_shift, "mm")
    print("    trunc    =", args.truncation_factor)
    print("    blur     =", args.detector_blur_sigma, "px")
    print("    shading  =", args.shading_strength)
    print("    recon x  =", args.recon_spacing_scale)
    print("    RTK bin =", Path(rtk_binaries["rtkfdk"]).resolve().parent)
    print("    label schema =", label_schema_path if label_schema_path is not None else "not found")
    print("    mask fine =", mask_fine_path if mask_fine_path is not None else "not found")
    print("    mask 559 =", mask_559_path if mask_559_path is not None else "not found")

    # RTK needs a working directory because FDK reads the projection stack from
    # disk. The final deliverable is written directly to outdir/<case_id>.mha.
    ct_iso_path = workdir / "CT_iso.mha"
    ct_centered_path = workdir / "CT_centered.mha"
    ct_rtk_rot_path = workdir / "CT_rtk_rotaxisZ.mha"
    ct_mu_path = workdir / "CT_mu.mha"
    geometry_path = workdir / "geometry.xml"
    proj_clean_path = workdir / "projections_clean.mha"
    proj_noisy_path = workdir / "projections_noisy.mha"
    cbct_mu_path = workdir / "cbct_mu.mha"
    cbct_mu_patient_path = workdir / "cbct_mu_patient_axes.mha"
    cbct_hu_path = outdir / f"{case_id}.mha"
    metal_trace_volume_path = workdir / "metal_mu_trace.mha"
    metal_trace_proj_path = workdir / "projections_metal_trace.mha"

    print("[1/9] Reading CT...")
    ct = read_image(input_path)
    print("    size   =", ct.GetSize())
    print("    spacing=", ct.GetSpacing())
    print("    origin =", ct.GetOrigin())

    print("[2/9] Resampling to isotropic spacing...")
    ct_iso = resample_isotropic(ct, iso_spacing=args.iso_spacing, default_value=-1024.0)
    write_image(ct_iso, ct_iso_path)
    print("    ->", ct_iso_path)
    print("    size   =", ct_iso.GetSize())
    print("    spacing=", ct_iso.GetSpacing())

    print("[3/9] Recentering volume...")
    ct_centered = recenter_image(ct_iso, target_center_phys=tuple(args.isocenter_shift))
    write_image(ct_centered, ct_centered_path)
    print("    ->", ct_centered_path)
    print("    new origin =", ct_centered.GetOrigin())
    centering_shift = np.array(ct_centered.GetOrigin(), dtype=np.float64) - np.array(ct_iso.GetOrigin(), dtype=np.float64)

    fine_mask_centered = None
    broad_mask_centered = None
    if label_maps is not None and mask_fine_path is not None:
        fine_mask_centered = load_mask_in_centered_space(mask_fine_path, ct_iso)
    if label_maps is not None and mask_559_path is not None:
        broad_mask_centered = load_mask_in_centered_space(mask_559_path, ct_iso)

    anatomy_for_effects = inferred_profile["anatomy"] if inferred_profile is not None else None
    if anatomy_for_effects is None and label_maps is not None:
        anatomy_for_effects = infer_anatomy_from_masks(fine_mask_centered, broad_mask_centered, label_maps)

    motion_model = None
    motion_should_try = (
        args.motion_artifact == "on"
        or (args.motion_artifact == "auto" and anatomy_for_effects in {"TH", "AB"})
    )
    if motion_should_try and label_maps is not None and fine_mask_centered is not None and anatomy_for_effects in {"TH", "AB"}:
        motion_model = build_motion_model(
            ct_img=ct_centered,
            fine_mask_img=fine_mask_centered,
            broad_mask_img=broad_mask_centered,
            label_maps=label_maps,
            anatomy=anatomy_for_effects,
            motion_strength=args.motion_strength,
            motion_cycles=args.motion_cycles,
        )
        if motion_model is not None:
            motion_model["phase_values"] = np.linspace(-1.0, 1.0, args.motion_phases, dtype=np.float32)

    metal_model = None
    metal_should_try = (
        args.metal_artifact == "on"
        or (args.metal_artifact == "auto" and label_maps is not None and fine_mask_centered is not None)
    )
    if metal_should_try and label_maps is not None and fine_mask_centered is not None:
        metal_model = build_metal_model(
            ct_img=ct_centered,
            fine_mask_img=fine_mask_centered,
            label_maps=label_maps,
            anatomy=anatomy_for_effects,
            metal_template=args.metal_template,
            metal_strength=args.metal_strength,
            seed=args.seed,
        )

    if motion_should_try and motion_model is None:
        print("    motion = skipped (no suitable thorax/abdomen masks found)")
    elif motion_model is not None:
        print(
            "    motion = enabled",
            f"(anatomy={anatomy_for_effects}, phases={len(motion_model['phase_values'])}, cycles={motion_model['cycles']:.1f})",
        )
    else:
        print("    motion = disabled")

    if metal_should_try and metal_model is None:
        print("    metal  = skipped (no suitable anatomy found for realistic placement)")
    elif metal_model is not None:
        print("    metal  = enabled", f"({metal_model['description']})")
    else:
        print("    metal  = disabled")

    ct_centered_augmented = combine_hu_with_metal(
        ct_centered,
        metal_model["metal_hu_img"] if metal_model is not None else None,
    )

    print("[4/9] Mask-aware augmentation, axis resampling, and HU-to-mu conversion...")
    # Physically swap Y and Z before RTK so the acquisition rotates in the
    # desired plane, then apply the same transform after FDK.
    ct_rtk_rot = resample_swap_yz(
        ct_centered_augmented,
        interpolator=sitk.sitkLinear,
        default_value=-1024.0,
    )
    write_image(ct_rtk_rot, ct_rtk_rot_path)
    print("    ->", ct_rtk_rot_path)
    print("    permuted size =", ct_rtk_rot.GetSize())
    print("    permuted spacing =", ct_rtk_rot.GetSpacing())

    metal_mu_boost_rtk = None
    if metal_model is not None:
        metal_mu_boost_rtk = resample_swap_yz(
            metal_model["metal_mu_boost_img"],
            interpolator=sitk.sitkLinear,
            default_value=0.0,
        )

    ct_mu = hu_to_mu(ct_rtk_rot, mu_water=args.mu_water)
    if metal_mu_boost_rtk is not None:
        ct_mu = add_images(ct_mu, metal_mu_boost_rtk)
    write_image(ct_mu, ct_mu_path)
    print("    ->", ct_mu_path)

    geometry_fit = auto_fit_detector_geometry(
        ct_mu,
        detector_dim_text=args.detector_dim,
        detector_spacing_mm=args.detector_spacing,
        sid_mm=args.sid,
        sdd_mm=args.sdd,
        safety_margin=args.fov_margin,
    )
    args.detector_spacing = geometry_fit["detector_spacing_mm"]
    args.sdd = geometry_fit["sdd_mm"]
    if args.truncation_factor < 1.0:
        args.detector_spacing *= args.truncation_factor
        geometry_fit["fov_width_mm"] *= args.truncation_factor
        geometry_fit["fov_height_mm"] *= args.truncation_factor
        geometry_fit["detector_width_mm"] *= args.truncation_factor
        geometry_fit["detector_height_mm"] *= args.truncation_factor
    print("    fitted detector spacing =", round(args.detector_spacing, 4), "mm")
    print("    fitted SDD =", round(args.sdd, 4), "mm")
    print("    required FOV at isocenter =", f"{geometry_fit['required_width_mm']:.1f} x {geometry_fit['required_height_mm']:.1f} mm")
    print("    available FOV at isocenter =", f"{geometry_fit['fov_width_mm']:.1f} x {geometry_fit['fov_height_mm']:.1f} mm")

    print("[5/9] Building RTK geometry...")
    # Keep the geometry command minimal and robust because RTK CLI options can
    # differ slightly across versions.
    cmd_geom = (
        f'"{rtk_binaries["rtksimulatedgeometry"]}" '
        f'-n {args.nproj} '
        f'--arc {args.arc} '
        f'--sid {args.sid} '
        f'--sdd {args.sdd} '
        f'-o "{geometry_path}"'
    )
    run(cmd_geom)

    print("[6/9] Computing forward projections...")
    if motion_model is None:
        run_forward_projection_command(
            rtk_forward_binary=rtk_binaries["rtkforwardprojections"],
            geometry_path=geometry_path,
            input_volume_path=ct_mu_path,
            output_path=proj_clean_path,
            detector_spacing_mm=args.detector_spacing,
            detector_dim_text=args.detector_dim,
        )
        proj_clean = read_image(proj_clean_path)
    else:
        phase_projection_images = []
        for phase_idx, phase_value in enumerate(motion_model["phase_values"]):
            phase_hu_patient = apply_motion_phase(
                ct_centered_augmented,
                motion_model=motion_model,
                phase_value=float(phase_value),
                default_value=-1024.0,
            )
            phase_hu_rtk = resample_swap_yz(
                phase_hu_patient,
                interpolator=sitk.sitkLinear,
                default_value=-1024.0,
            )
            phase_mu = hu_to_mu(phase_hu_rtk, mu_water=args.mu_water)
            if metal_mu_boost_rtk is not None:
                phase_mu = add_images(phase_mu, metal_mu_boost_rtk)

            phase_mu_path = workdir / f"CT_phase_{phase_idx:02d}_mu.mha"
            phase_proj_path = workdir / f"projections_phase_{phase_idx:02d}.mha"
            write_image(phase_mu, phase_mu_path)
            run_forward_projection_command(
                rtk_forward_binary=rtk_binaries["rtkforwardprojections"],
                geometry_path=geometry_path,
                input_volume_path=phase_mu_path,
                output_path=phase_proj_path,
                detector_spacing_mm=args.detector_spacing,
                detector_dim_text=args.detector_dim,
            )
            phase_projection_images.append(read_image(phase_proj_path))

        proj_clean = blend_motion_projection_stacks(
            phase_projection_images=phase_projection_images,
            motion_model=motion_model,
            seed=args.seed,
        )
        write_image(proj_clean, proj_clean_path)

    metal_trace_img = None
    if metal_mu_boost_rtk is not None:
        write_image(metal_mu_boost_rtk, metal_trace_volume_path)
        run_forward_projection_command(
            rtk_forward_binary=rtk_binaries["rtkforwardprojections"],
            geometry_path=geometry_path,
            input_volume_path=metal_trace_volume_path,
            output_path=metal_trace_proj_path,
            detector_spacing_mm=args.detector_spacing,
            detector_dim_text=args.detector_dim,
        )
        metal_trace_img = read_image(metal_trace_proj_path)

    print("[7/9] Injecting noise and heuristic scatter...")
    proj_noisy = process_projection_stack(
        proj_img=proj_clean,
        detector_spacing_mm=args.detector_spacing,
        photon_fluence_per_mm2_mAs=args.photon_fluence,
        tube_current_mA=args.tube_current,
        exposure_ms=args.exposure_ms,
        spr=args.spr,
        saturation_factor=args.saturation,
        scatter_blur_sigma_px=args.scatter_sigma,
        detector_blur_sigma_px=args.detector_blur_sigma,
        shading_strength=args.shading_strength,
        metal_trace_img=metal_trace_img,
        metal_starvation_strength=2.2 * args.metal_strength if metal_model is not None else 0.0,
        metal_streak_strength=0.35 * args.metal_strength if metal_model is not None else 0.0,
        seed=args.seed,
    )
    write_image(proj_noisy, proj_noisy_path)
    print("    ->", proj_noisy_path)

    print("[8/9] Reconstructing with FDK...")
    recon_spacing_patient = parse_csv_numbers(args.recon_spacing, float)
    recon_spacing_patient = [value * args.recon_spacing_scale for value in recon_spacing_patient]
    recon_spacing_rtk = [recon_spacing_patient[0], recon_spacing_patient[2], recon_spacing_patient[1]]
    recon_dim_xyz = compute_reconstruction_dimensions(
        ct_rtk_rot,
        recon_spacing_xyz=recon_spacing_rtk,
        safety_margin=args.recon_margin,
    )
    recon_origin_rtk = centered_axis_aligned_origin(recon_dim_xyz, recon_spacing_rtk)
    print("    reconstruction dim =", recon_dim_xyz)
    print("    reconstruction spacing (patient xyz) =", recon_spacing_patient)
    print("    reconstruction spacing (RTK xyz) =", recon_spacing_rtk)
    print("    reconstruction origin =", [round(v, 4) for v in recon_origin_rtk])

    cmd_fdk = (
        f'"{rtk_binaries["rtkfdk"]}" '
        f'-p "{workdir}" '
        f'-r "{proj_noisy_path.name}" '
        f'-g "{geometry_path}" '
        f'-o "{cbct_mu_path}" '
        f'--spacing {format_csv(recon_spacing_rtk)} '
        f'--origin {format_csv(recon_origin_rtk)} '
        f'--size {format_csv(recon_dim_xyz)}'
    )
    run(cmd_fdk)

    print("[9/9] Mapping reconstruction back to HU-like values...")
    cbct_mu = read_image(cbct_mu_path)
    cbct_mu_patient = resample_swap_yz(
        cbct_mu,
        interpolator=sitk.sitkLinear,
        default_value=0.0,
    )
    write_image(cbct_mu_patient, cbct_mu_patient_path)

    # Undo the temporary recentering so the simulated anatomy goes back to the
    # physical location of the original CT before the final resampling step.
    cbct_mu_final = shift_image_origin(cbct_mu_patient, -centering_shift)

    # Project the final simulated CBCT back onto the original CT grid so the
    # delivered output matches the input geometry exactly.
    cbct_mu_final = resample_to_reference(
        cbct_mu_final,
        reference=ct,
        interpolator=sitk.sitkLinear,
        default_value=0.0,
    )
    cbct_mu_final, calibration_info = calibrate_cbct_mu_to_reference_ct(
        cbct_mu_final,
        reference_ct_img=ct,
        mu_water=args.mu_water,
    )
    if calibration_info is not None:
        print(
            "    intensity calibration gain =",
            round(calibration_info["gain"], 4),
            f"(body voxels={calibration_info['body_voxels']})",
        )
        print(
            "    mu q50/q90 ref->cbct =",
            f"{calibration_info['reference_mu_q50']:.5f}/{calibration_info['reference_mu_q90']:.5f}",
            "vs",
            f"{calibration_info['cbct_mu_q50']:.5f}/{calibration_info['cbct_mu_q90']:.5f}",
        )
    cbct_hu_final = mu_to_hu_like(cbct_mu_final, mu_water=args.mu_water)
    cbct_hu_final = cast_hu_image_to_int16(cbct_hu_final)
    write_image(cbct_hu_final, cbct_hu_path)

    print("\n=== DONE ===")
    print("Final CBCT:", cbct_hu_path)
    print("Working dir:", workdir)

    if not args.keep_intermediate:
        print("\nCleaning intermediate files...")
        shutil.rmtree(workdir, ignore_errors=True)
        print("Intermediate files cleaned.")
    else:
        print("Intermediate files kept in:", workdir)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
