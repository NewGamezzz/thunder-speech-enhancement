import torch
import torch.nn as nn
from .diffusion import OutputData
from src.utils.registry import Registry
from src.utils.other import si_sdr_torch

LossRegistry = Registry("Loss")


@LossRegistry.register("mse_x0")
class MSECleanSpeech(nn.Module):
    def __init__(self, reduction="mean", transform=None):
        super().__init__()
        self.reduction = reduction
        self.transform = transform

    def forward(self, out: OutputData):
        if self.transform:
            out.pred_x0 = self.transform(out.pred_x0)
            out.x0 = self.transform(out.x0)
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


# This class is especially for ConvTasNet
@LossRegistry.register("sisdr")
class SISDR(nn.Module):
    def __init__(self, transform=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transform = transform

    def forward(self, out: OutputData):
        if self.transform:
            out.pred_x0 = self.transform(out.pred_x0)
            out.x0 = self.transform(out.x0)

        clean_signal = out.x0
        estimated_signal = out.pred_x0

        loss = -torch.mean(
            torch.stack(
                [
                    si_sdr_torch(clean_signal[i], estimated_signal[i])
                    for i in range(clean_signal.size(0))
                ]
            )
        )
        return loss


@LossRegistry.register("sdsnr")
class ScaleDependentSNR(nn.Module):
    def __init__(self, transform=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transform = transform

    def forward(self, out: OutputData):
        # NOTE: The input is in frequency domain.
        if self.transform:
            out.pred_x0 = self.transform(out.pred_x0)
            out.x0 = self.transform(out.x0)

        estimated_signal = out.pred_x0.flatten(1)
        clean_signal = out.x0.flatten(1)
        alpha = (
            torch.einsum("bh,bh->b", estimated_signal, clean_signal)
            / torch.norm(clean_signal, dim=-1) ** 2
        ).unsqueeze(-1)
        sdsnr = 10 * torch.log10(
            torch.norm(alpha * clean_signal, dim=-1) ** 2
            / torch.norm(clean_signal - estimated_signal, dim=-1) ** 2
        )
        return -torch.mean(sdsnr)  # Return negative scale-dependent signal-to-noise ratio


@LossRegistry.register("mse_time_frequencey")
class MSETimeFrequency(nn.Module):
    def __init__(self, time_loss_weight, transform, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.time_loss_weight = time_loss_weight
        self.transform = transform

    def forward(self, out: OutputData):
        B, C, F, T = out.x0.shape

        losses_tf = (1 / (F * T)) * torch.square(torch.abs(out.pred_x0 - out.x0))
        losses_tf = torch.mean(0.5 * torch.sum(losses_tf.reshape(losses_tf.shape[0], -1), dim=-1))

        time_pred_x0 = self.transform(out.pred_x0)
        time_x0 = self.transform(out.x0)
        losses_time = (1 / time_x0.shape[-1]) * torch.abs(time_pred_x0 - time_x0)
        losses_time = torch.mean(
            0.5 * torch.sum(losses_time.reshape(losses_time.shape[0], -1), dim=-1)
        )

        return losses_tf + self.time_loss_weight * losses_time
