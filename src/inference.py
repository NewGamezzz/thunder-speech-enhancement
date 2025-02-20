import torch
from .diffusion import Diffusion
from .data_module import SpecsDataModule
from .utils.other import pad_spec
from .utils.registry import Registry

InferenceRegistry = Registry("Inference")


class BaseInference:
    def __init__(self, data_module, transform="mel_spec"):
        assert transform in ["mel_spec", "raw"], "transform parameters must be 'mel_spec' or 'raw'."
        self.data_module = data_module  # Use for transforming time-domain signal
        self.transform = transform

    def sampling(self, y, **kwargs):
        raise NotImplementedError

    def preprocess(self, y: torch.Tensor, norm_factor: float):
        """Normalize and transform time-domain signal into spectrogram"""
        y = y / norm_factor
        if self.transform == "mel_spec":
            Y = torch.unsqueeze(self.data_module.spec_fwd(self.data_module.stft(y)), 0)
            Y = pad_spec(Y)
        elif self.transform == "raw":
            Y = y
        return Y

    def postprocess(self, X: torch.Tensor, T_orig: int, norm_factor: float):
        """Transform spectrogram to time-domain signal and denormalize"""
        if self.transform == "mel_spec":
            x = self.data_module.istft(self.data_module.spec_back(X), T_orig)
        elif self.transform == "raw":
            x = X
        x = x * norm_factor
        return x

    def inference(self, y: torch.Tensor, **kwargs):
        """Inference single utterance"""
        T_orig = y.size(1)
        norm_factor = y.abs().max().item()
        Y = self.preprocess(y, norm_factor)
        X = self.sampling(Y, **kwargs)
        X = X.squeeze()
        x = self.postprocess(X, T_orig, norm_factor).squeeze()
        return x


@InferenceRegistry.register("diffusion")
class DiffusionInference(BaseInference):
    def __init__(
        self,
        model: Diffusion,
        data_module: SpecsDataModule,
        sampler_type="pc",
        predictor="reverse_diffusion",
        corrector="ald",
        N=30,
        corrector_steps=1,
        snr=0.5,
        **kwargs,
    ):
        super().__init__(data_module, **kwargs)
        assert sampler_type == "pc", "Only support the predictor-corrector sampling method."
        self.model = model
        self.sampler_type = sampler_type
        self.predictor = predictor
        self.corrector = corrector
        self.N = N
        self.corrector_steps = corrector_steps
        self.snr = snr

    def sampling(self, y, **kwargs):
        """
        Perform reverse diffusion process.
        """
        sampler_x0 = self.model.reverse_process(
            self.predictor,
            self.corrector,
            y,
            N=self.N,
            corrector_steps=self.corrector_steps,
            snr=self.snr,
            **kwargs,
        )
        return sampler_x0[0]


@InferenceRegistry.register("regression")
class RegressionInference(BaseInference):
    def __init__(self, model: Diffusion, data_module: SpecsDataModule, **kwargs):
        super().__init__(data_module, **kwargs)
        self.model = model

    def sampling(self, y, **kwargs):
        with torch.no_grad():
            t = torch.ones(y.shape[0], device=y.device)
            out = self.model(y, t, y)
        return out


@InferenceRegistry.register("twostage")
class TwoStageInference(BaseInference):
    def __init__(
        self,
        model: Diffusion,
        data_module: SpecsDataModule,
        sampler_type="pc",
        predictor="reverse_diffusion",
        corrector="ald",
        N=30,
        corrector_steps=1,
        snr=0.5,
        interpolate_weight=0.8,
        **kwargs,
    ):
        super().__init__(data_module, **kwargs)
        assert sampler_type == "pc", "Only support the predictor-corrector sampling method."
        self.model = model
        self.sampler_type = sampler_type
        self.predictor = predictor
        self.corrector = corrector
        self.N = N
        self.corrector_steps = corrector_steps
        self.snr = snr
        self.interpolate_weight = interpolate_weight

    def sampling(self, y, **kwargs):
        """
        Perform reverse diffusion process.
        """
        with torch.no_grad():
            t = torch.ones(y.shape[0], device=y.device)
            out = self.model(y, t, y)

        out = self.interpolate_weight * out + (1 - self.interpolate_weight) * y
        sampler_x0 = self.model.reverse_process(
            self.predictor,
            self.corrector,
            out,
            N=self.N,
            corrector_steps=self.corrector_steps,
            snr=self.snr,
            **kwargs,
        )
        return sampler_x0[0]


@InferenceRegistry.register("multistage")
class MultiStageInference(BaseInference):
    def __init__(
        self,
        model: Diffusion,
        data_module: SpecsDataModule,
        sampler_type="pc",
        predictor="reverse_diffusion",
        corrector="ald",
        N=30,
        corrector_steps=1,
        snr=0.5,
        interpolate_weight=0.8,
        **kwargs,
    ):
        super().__init__(data_module, **kwargs)
        assert sampler_type == "pc", "Only support the predictor-corrector sampling method."
        self.model = model
        self.sampler_type = sampler_type
        self.predictor = predictor
        self.corrector = corrector
        self.N = N
        self.corrector_steps = corrector_steps
        self.snr = snr
        self.interpolate_weight = interpolate_weight

    def sampling(self, y, **kwargs):
        """
        Perform reverse diffusion process.
        """
        with torch.no_grad():
            t = torch.ones(y.shape[0], device=y.device)
            out = self.model(y, t, y)

        out = self.interpolate_weight * out + (1 - self.interpolate_weight) * y
        sampler_x0 = self.model.reverse_process(
            self.predictor,
            self.corrector,
            out,
            N=self.N,
            corrector_steps=self.corrector_steps,
            snr=self.snr,
            **kwargs,
        )[0]
        out = (self.interpolate_weight + 0.1) * sampler_x0 + (
            1 - (self.interpolate_weight + 0.1)
        ) * y  # (1 - self.interpolate_weight) * out
        sampler_x0 = self.model.reverse_process(
            self.predictor,
            self.corrector,
            out,
            N=self.N,
            corrector_steps=self.corrector_steps,
            snr=self.snr,
            **kwargs,
        )[0]
        # out = (self.interpolate_weight + 0.2) * sampler_x0 + (
        #     1 - (self.interpolate_weight + 0.2)
        # ) * y  # (1 - self.interpolate_weight) * out
        # sampler_x0 = self.model.reverse_process(
        #     self.predictor,
        #     self.corrector,
        #     out,
        #     N=self.N,
        #     corrector_steps=self.corrector_steps,
        #     snr=self.snr,
        #     **kwargs,
        # )[0]
        # out = (
        #     self.interpolate_weight * sampler_x0 + (1 - self.interpolate_weight) * y
        # )  # (1 - self.interpolate_weight) * out
        # sampler_x0 = self.model.reverse_process(
        #     self.predictor,
        #     self.corrector,
        #     out,
        #     N=self.N,
        #     corrector_steps=self.corrector_steps,
        #     snr=self.snr,
        # **kwargs,
        # )[0]

        return sampler_x0
