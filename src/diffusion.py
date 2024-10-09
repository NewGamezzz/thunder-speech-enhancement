import torch
import torch.nn as nn

from .sampling.predictors import PredictorRegistry
from .sampling.correctors import CorrectorRegistry


class Diffusion(nn.Module):
    def __init__(self, model, sde, pred_type="score", t_eps=3e-2):
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
        self.sde = sde
        self.pred_type = pred_type
        self.t_eps = t_eps

    def forward_process(self, x0, t, y, noise=None):
        """
        Sample xt from p_t(xt|x0, y)

        Args:
            x0: Clean speech
            t: Time step
            y: Noisy speech
            noise: Gaussian noise to be add. If set to None, noise is overwrited with gaussian noise. Default to None.

        """
        if noise is None:
            noise = torch.randn_like(x0)
        mean, std = self.sde.marginal_prob(x0, t, y)
        if std.ndim < y.ndim:
            std = std.view(*std.size(), *((1,) * (y.ndim - std.ndim)))
        xt = mean + std * noise
        return xt

    def reverse_process(
        self,
        predictor_name,
        corrector_name,
        y,
        snr=0.1,
        corrector_steps=1,
        denoise=True,
        conditioning=None,
        probability_flow=False,
        N=30,
        **kwargs
    ):
        """
        Perform reverse diffusion process

        Args:
            predictor_name: Type of predictor
            corrector_name: Type of corrector
            y: Noisy speech
            snr: The SNR to use for the corrector. Default to 0.1, and ignored for `NoneCorrector`
            denoise: If `True`, add one-step denoising to the final samples
            N: The number of reverse sampling steps.

        """
        predictor_cls = PredictorRegistry.get_by_name(predictor_name)
        corrector_cls = CorrectorRegistry.get_by_name(corrector_name)
        predictor = predictor_cls(self.sde, self.pred_score, probability_flow=probability_flow)
        corrector = corrector_cls(self.sde, self.pred_score, snr=snr, n_steps=corrector_steps)

        with torch.no_grad():
            xt, _ = self.sde.prior_sampling(y.shape, y)
            xt = xt.to(y.device)
            timesteps = torch.linspace(self.sde.T, self.t_eps, N, device=y.device)
            for i in range(N):
                t = timesteps[i]
                if i != len(timesteps) - 1:
                    stepsize = t - timesteps[i + 1]
                else:
                    stepsize = timesteps[-1]  # from eps to 0
                vec_t = torch.ones(y.shape[0], device=y.device) * t
                xt, xt_mean = corrector.update_fn(xt, vec_t, y)
                xt, xt_mean = predictor.update_fn(xt, vec_t, y, stepsize)
            x_result = xt_mean if (denoise and N) else xt
            ns = self.sde.N * (corrector.n_steps + 1)
        return x_result, ns

    def forward(self, xt, t, y, **kwargs):
        dnn_input = torch.cat([xt, y], dim=1)  # b, 2*d, f, t
        # print("INPUT SHAPE:", dnn_input.shape)
        # print("MODEL", self.model)
        return self.model(dnn_input, t)

    def pred_score(self, xt, t, y, **kwargs):
        if self.pred_type == "score":
            score = self(xt, t, y)
        elif self.pred_type == "x0":
            pred_x0 = self(xt, t, y)
            mean, std = self.sde.marginal_prob(pred_x0, t, y)
            score = -(xt - mean) / std[:, None, None, None] ** 2
        else:
            raise NotImplementedError
        return score
