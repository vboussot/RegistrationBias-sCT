"""Perceptual losses used in the supervised comparison."""

# No `from __future__ import annotations`: KonfAI reads these annotations at runtime.
import torch
import torchvision
from konfai.metric.measure import Criterion
from sam2.build_sam import build_sam2


class SAMEncoder(torch.nn.Module):
    def __init__(self, model_cfg: str, checkpoint: str) -> None:
        super().__init__()
        self.model = build_sam2(model_cfg, checkpoint, device="cpu").image_encoder.trunk

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.model.patch_embed(x)
        x = x + self.model._get_pos_embed((x.shape[1], x.shape[2]))
        outputs: list[torch.Tensor] = []
        for index, block in enumerate(self.model.blocks):
            x = block(x)
            if index == self.model.stage_ends[-1] or (
                index in self.model.stage_ends and self.model.return_interm_layers
            ):
                outputs.append(x.permute(0, 3, 1, 2))
        return outputs


class Perceptual(Criterion):
    def __init__(
        self,
        model: torch.nn.Module,
        shape: list[int],
        in_channels: int,
        weight: list[float],
    ) -> None:
        super().__init__()
        self.model = model.eval().requires_grad_(False)
        self.shape = shape
        self.in_channels = in_channels
        self.weight = weight
        self.loss_function = torch.nn.L1Loss()

    def preprocessing(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[1] != self.in_channels:
            value = value.repeat(tuple([1, 3] + [1 for _ in self.shape]))
        return value

    def _compute(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        output = self.preprocessing(output)
        target = self.preprocessing(target)
        self.model.to(output.device)
        loss = torch.zeros(1, device=output.device, dtype=torch.float32)
        for out_features, target_features, weight in zip(
            self.model(output), self.model(target), self.weight
        ):
            loss = loss + weight * self.loss_function(out_features, target_features)
        return loss

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        if len(output.shape) == 5 and len(self.shape) == 2:
            values = [
                self._compute(output[:, :, index, ...], targets[0][:, :, index, ...])
                for index in range(output.shape[2])
            ]
            return torch.stack(values).mean().to(output)
        return self._compute(output, targets[0]).to(output)


class SAM(Perceptual):
    """SAM 2.1 small loss using hierarchical weights (0, 1, 1, 0)."""

    def __init__(
        self,
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
        checkpoint: str = "models/external/sam2.1_hiera_small.pt",
    ) -> None:
        super().__init__(SAMEncoder(model_cfg, checkpoint), [512, 512], 3, [0, 1, 1, 0])


class VGGEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        features = torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
        ).features
        self.blocks = torch.nn.ModuleList(
            [features[:4], features[4:9], features[9:16], features[16:23]]
        ).to(torch.float32)

    def forward(self, value: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for block in self.blocks:
            value = block(value)
            outputs.append(value)
        return outputs


class VGG(Perceptual):
    def __init__(self) -> None:
        super().__init__(VGGEncoder(), [512, 512], 3, [1, 1, 1, 1])
