from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class OutputData:
    pred_x0: torch.Tensor
    pred_score: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    z: torch.Tensor  # gaussian noise for calculating denoising score matching loss
    x0: torch.Tensor


class Regression(nn.Module):
    def __init__(self, model, loss_func, transform=None, device="cuda"):
        """
        Initialize diffusion model

        Args:
            model: Deep learning model such as UNet or NCSN++
            sde: Forward and reverse sde to be used
            pred_type: Determine the output of the model
            t_eps: The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical instability

        """
        super().__init__()
        self.model = model
        self.loss_func = loss_func
        self.transform = transform
        self.device = device

    def forward(self, x, *args, **kwargs):
        t = torch.ones(x.shape[0], device=x.device)
        return self.model(x, t)

    def step(self, batch):
        x0, y = batch
        x0, y = x0.to(self.device), y.to(self.device)

        if self.transform:
            x0, y = self.transform(x0), self.transform(y)

        out = self.forward(y)
        out = OutputData(
            pred_x0=out,  # TODO Derive pred_x0 from pred_score
            pred_score=None,
            mean=None,
            std=None,
            z=None,
            x0=x0,
        )
        loss = self.loss_func(out)
        return out, loss
