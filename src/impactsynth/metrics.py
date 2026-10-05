"""Custom image quality metrics for KonfAI evaluation.

These metrics are designed for CT vs CT-like comparisons where we want to
quantify smoothing / blur effects on a prediction with respect to a reference.

Typical KonfAI usage examples:

    metric:GradientMagnitudeMeanRatio: {}
    metric:GradientMagnitudeP95Ratio: {}
    metric:EdgeWidthRatio:
      edge_percentile: 0.95
    metric:HighFrequencyEnergyRatio:
      cutoff_ratio: 0.25
    metric:LaplacianVarianceRatio: {}

They accept:
- `output`: predicted image
- `targets[0]`: reference image
- `targets[1]` (optional): binary mask
"""

# No `from __future__ import annotations`: KonfAI reads these annotations at runtime.
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from konfai.metric.measure import Criterion
from konfai.data.patching import ModelPatch
from konfai.metric.measure import LPIPS as KonfAILPIPS
from konfai.metric.measure import SAM_Perceptual as KonfAISAMPerceptual
from konfai.metric.measure.base import MaskedLoss


EPS = 1e-8


@dataclass
class MetricInputs:
    output: torch.Tensor
    reference: torch.Tensor
    mask: torch.Tensor | None


def _ensure_channel_dim(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x
    if x.ndim == 5:
        return x
    raise ValueError(f"Unsupported tensor shape {tuple(x.shape)}; expected 4D or 5D tensor.")


def _flatten_metric_dims(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    x = _ensure_channel_dim(x)
    is_3d = x.ndim == 5
    if is_3d:
        return x, True
    return x.unsqueeze(2), False


def _restore_metric_dims(x: torch.Tensor, was_3d: bool) -> torch.Tensor:
    if was_3d:
        return x
    return x.squeeze(2)


def _prepare_inputs(output: torch.Tensor, targets: tuple[torch.Tensor, ...]) -> MetricInputs:
    if len(targets) == 0:
        raise ValueError("At least one target tensor is required.")

    reference = targets[0]
    mask = targets[1] if len(targets) > 1 else None

    output = output.to(torch.float32)
    reference = reference.to(torch.float32)
    if mask is not None:
        mask = (mask > 0.5).to(torch.float32)

    return MetricInputs(output=output, reference=reference, mask=mask)


def _apply_mask(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.reshape(values.shape[0], -1)
    mask = mask.expand_as(values) > 0.5
    masked_values = []
    for batch_index in range(values.shape[0]):
        selected = values[batch_index][mask[batch_index]]
        if selected.numel() == 0:
            selected = values[batch_index].reshape(-1)
        masked_values.append(selected)
    max_len = max(v.numel() for v in masked_values)
    padded = values.new_zeros((values.shape[0], max_len))
    valid = torch.zeros((values.shape[0], max_len), dtype=torch.bool, device=values.device)
    for batch_index, selected in enumerate(masked_values):
        padded[batch_index, : selected.numel()] = selected
        valid[batch_index, : selected.numel()] = True
    return padded.masked_fill(~valid, torch.nan)


def _nanmean(x: torch.Tensor) -> torch.Tensor:
    return torch.nanmean(x, dim=1)


def _nanvar(x: torch.Tensor) -> torch.Tensor:
    mean = torch.nanmean(x, dim=1, keepdim=True)
    return torch.nanmean((x - mean) ** 2, dim=1)


def _nanquantile(x: torch.Tensor, q: float) -> torch.Tensor:
    rows = []
    max_samples = 2_000_000
    for batch_index in range(x.shape[0]):
        valid = x[batch_index][~torch.isnan(x[batch_index])]
        if valid.numel() == 0:
            valid = x[batch_index].reshape(-1)
        # torch.quantile can fail on very large tensors; compute on CPU with
        # deterministic sub-sampling and NumPy's quantile implementation.
        valid_cpu = valid.detach().to("cpu", dtype=torch.float32).reshape(-1)
        if valid_cpu.numel() > max_samples:
            step = int(np.ceil(valid_cpu.numel() / max_samples))
            valid_cpu = valid_cpu[::step]
        quantile_value = float(np.quantile(valid_cpu.numpy(), q))
        rows.append(torch.tensor(quantile_value, device=x.device, dtype=x.dtype))
    return torch.stack(rows)


def _gradient_components(x: torch.Tensor) -> list[torch.Tensor]:
    x5d, was_3d = _flatten_metric_dims(x)
    gradients = []

    gx = x5d[..., 1:] - x5d[..., :-1]
    gx = F.pad(gx, (0, 1, 0, 0, 0, 0), mode="replicate")
    gradients.append(gx)

    gy = x5d[..., 1:, :] - x5d[..., :-1, :]
    gy = F.pad(gy, (0, 0, 0, 1, 0, 0), mode="replicate")
    gradients.append(gy)

    gz = x5d[:, :, 1:, :, :] - x5d[:, :, :-1, :, :]
    gz = F.pad(gz, (0, 0, 0, 0, 0, 1), mode="replicate")
    gradients.append(gz)

    if not was_3d:
        gradients[2] = torch.zeros_like(gradients[0])

    return [_restore_metric_dims(g, was_3d) for g in gradients]


def gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    gradients = _gradient_components(x)
    grad_sq = sum(g * g for g in gradients)
    return torch.sqrt(grad_sq + EPS)


def laplacian_response(x: torch.Tensor) -> torch.Tensor:
    x5d, was_3d = _flatten_metric_dims(x)
    center = x5d
    xp = F.pad(x5d[..., 1:], (0, 1, 0, 0, 0, 0), mode="replicate")
    xm = F.pad(x5d[..., :-1], (1, 0, 0, 0, 0, 0), mode="replicate")
    yp = F.pad(x5d[..., 1:, :], (0, 0, 0, 1, 0, 0), mode="replicate")
    ym = F.pad(x5d[..., :-1, :], (0, 0, 1, 0, 0, 0), mode="replicate")
    zp = F.pad(x5d[:, :, 1:, :, :], (0, 0, 0, 0, 0, 1), mode="replicate")
    zm = F.pad(x5d[:, :, :-1, :, :], (0, 0, 0, 0, 1, 0), mode="replicate")
    if not was_3d:
        zp = center
        zm = center
    lap = xp + xm + yp + ym + zp + zm - 6.0 * center
    return _restore_metric_dims(lap, was_3d)


def local_intensity_range(x: torch.Tensor) -> torch.Tensor:
    x5d, was_3d = _flatten_metric_dims(x)
    x_pad = F.pad(x5d, (1, 1, 1, 1, 1, 1), mode="replicate")
    local_max = F.max_pool3d(x_pad, kernel_size=3, stride=1)
    local_min = -F.max_pool3d(-x_pad, kernel_size=3, stride=1)
    response = local_max - local_min
    return _restore_metric_dims(response, was_3d)


def strong_edge_mask(reference: torch.Tensor, mask: torch.Tensor | None, edge_percentile: float) -> torch.Tensor:
    ref_grad = gradient_magnitude(reference)
    if mask is not None:
        masked = _apply_mask(ref_grad, mask)
    else:
        masked = ref_grad.reshape(ref_grad.shape[0], -1)
    threshold = _nanquantile(masked, edge_percentile)
    if ref_grad.ndim == 5:
        threshold = threshold.view(-1, 1, 1, 1, 1)
    else:
        threshold = threshold.view(-1, 1, 1, 1)
    return (ref_grad >= threshold).to(ref_grad.dtype) * (mask if mask is not None else 1.0)


def edge_width_proxy(x: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor | None, edge_percentile: float) -> torch.Tensor:
    edges = strong_edge_mask(reference, mask, edge_percentile=edge_percentile)
    grad = gradient_magnitude(x)
    contrast = local_intensity_range(x)
    width = contrast / (grad + EPS)
    width_values = _apply_mask(width, edges)
    return _nanmean(width_values)


def high_frequency_energy(x: torch.Tensor, mask: torch.Tensor | None, cutoff_ratio: float) -> torch.Tensor:
    x5d, was_3d = _flatten_metric_dims(x)
    if mask is not None:
        mask5d, _ = _flatten_metric_dims(mask)
        x5d = x5d * mask5d

    fft = torch.fft.fftn(x5d, dim=(-3, -2, -1))
    power = torch.abs(fft) ** 2

    depth, height, width = x5d.shape[-3:]
    fz = torch.fft.fftfreq(depth, d=1.0, device=x5d.device).view(depth, 1, 1)
    fy = torch.fft.fftfreq(height, d=1.0, device=x5d.device).view(1, height, 1)
    fx = torch.fft.fftfreq(width, d=1.0, device=x5d.device).view(1, 1, width)
    radius = torch.sqrt(fx**2 + fy**2 + fz**2)
    high_mask = radius >= (cutoff_ratio * radius.max())
    high_mask = high_mask.view(1, 1, depth, height, width)

    energy = power * high_mask
    energy = energy.reshape(energy.shape[0], -1).sum(dim=1)
    return energy


class _BlurMetric(Criterion):
    def _inputs(self, output: torch.Tensor, targets: tuple[torch.Tensor, ...]) -> MetricInputs:
        return _prepare_inputs(output, targets)

    def _reduce_scalar_batch(self, values: torch.Tensor, output: torch.Tensor) -> tuple[torch.Tensor, float]:
        scalar = values.mean().to(output)
        return scalar, scalar.item()


class GradientMagnitudeMean(_BlurMetric):
    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        inputs = self._inputs(output, targets)
        grad = gradient_magnitude(inputs.output)
        masked = _apply_mask(grad, inputs.mask)
        return self._reduce_scalar_batch(_nanmean(masked), output)


class GradientMagnitudeP95(_BlurMetric):
    def __init__(self, quantile: float = 0.95) -> None:
        super().__init__()
        self.quantile = quantile

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        inputs = self._inputs(output, targets)
        grad = gradient_magnitude(inputs.output)
        masked = _apply_mask(grad, inputs.mask)
        return self._reduce_scalar_batch(_nanquantile(masked, self.quantile), output)


class GradientMagnitudeMeanRatio(_BlurMetric):
    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        inputs = self._inputs(output, targets)
        out_values = _nanmean(_apply_mask(gradient_magnitude(inputs.output), inputs.mask))
        ref_values = _nanmean(_apply_mask(gradient_magnitude(inputs.reference), inputs.mask))
        return self._reduce_scalar_batch(out_values / (ref_values + EPS), output)


class GradientMagnitude(GradientMagnitudeMeanRatio):
    """Alias returning the mean gradient-magnitude ratio output/reference."""


class GradientMagnitudeLoss(_BlurMetric):
    """Relative loss of gradient magnitude: 0 means identical, larger means smoother."""

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        inputs = self._inputs(output, targets)
        out_values = _nanmean(_apply_mask(gradient_magnitude(inputs.output), inputs.mask))
        ref_values = _nanmean(_apply_mask(gradient_magnitude(inputs.reference), inputs.mask))
        loss_values = 100.0 * torch.clamp((ref_values - out_values) / (ref_values + EPS), min=0.0)
        return self._reduce_scalar_batch(loss_values, output)


class GradientMagnitudeP95Ratio(_BlurMetric):
    def __init__(self, quantile: float = 0.95) -> None:
        super().__init__()
        self.quantile = quantile

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        inputs = self._inputs(output, targets)
        out_values = _nanquantile(_apply_mask(gradient_magnitude(inputs.output), inputs.mask), self.quantile)
        ref_values = _nanquantile(_apply_mask(gradient_magnitude(inputs.reference), inputs.mask), self.quantile)
        return self._reduce_scalar_batch(out_values / (ref_values + EPS), output)


class EdgeSharpness(_BlurMetric):
    def __init__(self, edge_percentile: float = 0.95) -> None:
        super().__init__()
        self.edge_percentile = edge_percentile

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        inputs = self._inputs(output, targets)
        widths = edge_width_proxy(
            inputs.output,
            reference=inputs.reference,
            mask=inputs.mask,
            edge_percentile=self.edge_percentile,
        )
        return self._reduce_scalar_batch(1.0 / (widths + EPS), output)


class EdgeWidth(_BlurMetric):
    def __init__(self, edge_percentile: float = 0.95) -> None:
        super().__init__()
        self.edge_percentile = edge_percentile

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        inputs = self._inputs(output, targets)
        widths = edge_width_proxy(
            inputs.output,
            reference=inputs.reference,
            mask=inputs.mask,
            edge_percentile=self.edge_percentile,
        )
        return self._reduce_scalar_batch(widths, output)


class EdgeWidthRatio(_BlurMetric):
    def __init__(self, edge_percentile: float = 0.95) -> None:
        super().__init__()
        self.edge_percentile = edge_percentile

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        inputs = self._inputs(output, targets)
        out_width = edge_width_proxy(
            inputs.output,
            reference=inputs.reference,
            mask=inputs.mask,
            edge_percentile=self.edge_percentile,
        )
        ref_width = edge_width_proxy(
            inputs.reference,
            reference=inputs.reference,
            mask=inputs.mask,
            edge_percentile=self.edge_percentile,
        )
        return self._reduce_scalar_batch(out_width / (ref_width + EPS), output)


class HighFrequencyEnergy(_BlurMetric):
    def __init__(self, cutoff_ratio: float = 0.25) -> None:
        super().__init__()
        self.cutoff_ratio = cutoff_ratio

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        inputs = self._inputs(output, targets)
        values = high_frequency_energy(inputs.output, inputs.mask, cutoff_ratio=self.cutoff_ratio)
        return self._reduce_scalar_batch(values, output)


class HighFrequencyEnergyRatio(_BlurMetric):
    def __init__(self, cutoff_ratio: float = 0.25) -> None:
        super().__init__()
        self.cutoff_ratio = cutoff_ratio

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        inputs = self._inputs(output, targets)
        out_values = high_frequency_energy(inputs.output, inputs.mask, cutoff_ratio=self.cutoff_ratio)
        ref_values = high_frequency_energy(inputs.reference, inputs.mask, cutoff_ratio=self.cutoff_ratio)
        return self._reduce_scalar_batch(out_values / (ref_values + EPS), output)


class HighFrequencyLoss(_BlurMetric):
    """Relative change of high-frequency energy in percent: 0 means identical."""

    def __init__(self, cutoff_ratio: float = 0.25) -> None:
        super().__init__()
        self.cutoff_ratio = cutoff_ratio

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        inputs = self._inputs(output, targets)
        out_values = high_frequency_energy(inputs.output, inputs.mask, cutoff_ratio=self.cutoff_ratio)
        ref_values = high_frequency_energy(inputs.reference, inputs.mask, cutoff_ratio=self.cutoff_ratio)
        loss_values = 100.0 * torch.abs(out_values - ref_values) / (ref_values + EPS)
        return self._reduce_scalar_batch(loss_values, output)


class LaplacianVariance(_BlurMetric):
    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        inputs = self._inputs(output, targets)
        lap = laplacian_response(inputs.output)
        values = _nanvar(_apply_mask(lap, inputs.mask))
        return self._reduce_scalar_batch(values, output)


class LaplacianVarianceRatio(_BlurMetric):
    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        inputs = self._inputs(output, targets)
        out_values = _nanvar(_apply_mask(laplacian_response(inputs.output), inputs.mask))
        ref_values = _nanvar(_apply_mask(laplacian_response(inputs.reference), inputs.mask))
        return self._reduce_scalar_batch(out_values / (ref_values + EPS), output)


class LaplacianVarianceLoss(_BlurMetric):
    """Relative loss of Laplacian variance: 0 means identical, larger means smoother."""

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        inputs = self._inputs(output, targets)
        out_values = _nanvar(_apply_mask(laplacian_response(inputs.output), inputs.mask))
        ref_values = _nanvar(_apply_mask(laplacian_response(inputs.reference), inputs.mask))
        loss_values = 100.0 * torch.clamp((ref_values - out_values) / (ref_values + EPS), min=0.0)
        return self._reduce_scalar_batch(loss_values, output)



class SAM_Perceptual(KonfAISAMPerceptual):
    """KonfAI's calibrated d_SAM as computed in the paper: one 512x512 tile per axial slice, the slice
    being zero-padded (0 HU) or cropped at the end of its axes. Both images are normalized with the
    reference's statistics."""

    def forward(self, output, *targets, attributes):
        def pad(tensor: torch.Tensor) -> torch.Tensor:
            height, width = tensor.shape[-2:]
            return F.pad(tensor, (0, 512 - width, 0, 512 - height), value=0)

        # KonfAI >= 1.8.7 passes the attributes of the targets only, which is what the normalization needs.
        assert len(attributes) == len(targets), "SAM_Perceptual needs KonfAI >= 1.8.7"
        return super().forward(pad(output), *(pad(target) for target in targets), attributes=attributes)


class LPIPS(KonfAILPIPS):
    """The official LPIPS (VGG) as computed in the paper: each axial slice centred in one 512x512 frame,
    padded with air (-1 after the [-1, 1] normalization) or cropped symmetrically, without resizing."""

    def __init__(self, model: str = "vgg") -> None:
        import lpips

        MaskedLoss.__init__(self, partial(KonfAILPIPS._loss, lpips.LPIPS(net=model), ModelPatch([1, 512, 512])), True)

    def forward(self, output, *targets):
        def center(tensor: torch.Tensor) -> torch.Tensor:
            height, width = tensor.shape[-2:]
            pad_h, pad_w = 512 - height, 512 - width
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            return F.pad(tensor, padding, value=-1.0)

        return super().forward(center(output), *(center(target) for target in targets))
