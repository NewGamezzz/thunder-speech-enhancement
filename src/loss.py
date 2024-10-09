import torch
import torch.nn as nn
from .trainer import OutputData
from src.utils.registry import Registry

LossRegistry = Registry("Loss")


@LossRegistry.register("mse_x0")
class MSECleanSpeech(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, out: OutputData):
        err = out.pred_x0 - out.x0
        losses = torch.square(err.abs())
        loss = 0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1)
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError()
        return loss


@LossRegistry.register("mse_score")
class DenoisingScoreMatching(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, out: OutputData):
        err = out.std * out.pred_score + out.z
        losses = torch.square(err.abs())
        loss = torch.mean(0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss
