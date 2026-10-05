"""2.5D U-Net++ generator used in the paper."""

# No `from __future__ import annotations`: KonfAI reads these annotations at runtime.
import torch
import segmentation_models_pytorch as smp
from konfai.data.patching import ModelPatch
from konfai.network import network


class Head(network.ModuleArgsDict):
    def __init__(self) -> None:
        super().__init__()
        self.add_module("Tanh", torch.nn.Tanh())


class UNetpp(network.Network):
    """Five-slice 2.5D U-Net++ with a ResNet-34 encoder.

    ``ModelPatch`` is imported because KonfAI resolves it while deserializing
    some checkpoints, even though the default configuration does
    not instantiate it directly.
    """

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default:ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {
            "default": network.TargetCriterionsLoader()
        },
        nb_channel: int = 5,
    ) -> None:
        super().__init__(
            in_channels=nb_channel,
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            dim=2,
        )
        self.add_module(
            "model",
            smp.UnetPlusPlus(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=nb_channel,
                classes=1,
                activation=None,
            ),
        )
        self.add_module("Head", Head())


def parameter_counts() -> tuple[int, int]:
    model = UNetpp()
    return sum(p.numel() for p in model.parameters()), sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
