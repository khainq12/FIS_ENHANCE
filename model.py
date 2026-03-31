import torch
import torch.nn as nn
from channel import Channel
from model_baseline import ratio2filtersize, _Encoder, _Decoder
from fis_modules import FIS_SpatialPowerController


def power_normalize(z: torch.Tensor, P: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """
    Per-sample power normalization so that mean(z^2) = P.
    z: [B,C,H,W] real-valued (I/Q stacked channels)
    """
    p = z.pow(2).mean(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
    scale = torch.sqrt(torch.tensor(P, device=z.device, dtype=z.dtype) / p)
    return z * scale


class DeepJSCC_FIS(nn.Module):
    """
    Deep-JSCC backbone (same encoder/decoder as baseline) + FIS spatial power controller.

    FIX-1 (model.py): gain = A (amplitude map), NOT sqrt(A).
        The FIS designs A as an *amplitude* scaling map (not a power map).
        Using sqrt(A) halved the effective gain range, making FIS behave
        much more weakly than its rule consequents intended.
        After this fix, A=1.35 gives a 35% amplitude boost as designed.

    FIX-2 (model.py): a_high default aligned to 1.35 (matches fis_modules defaults).
        The original model.py used a_high=2.0 while fis_modules.py defaulted to 1.35,
        creating an inconsistency. We choose 1.35 as the canonical value because it is
        what the signed-consequent FIS was calibrated for. If you want a wider range,
        change it consistently in both files.

    FIX-3 (model.py): snr_max_db default changed from 13.0 to 20.0 to match
        the training script default snr_max=20.0. Without this, the FIS SNR
        normalizer saturated at u=1 for SNRs above 13 dB during training,
        causing the power-allocation layer to lose SNR sensitivity in the
        upper training range.
    """

    def __init__(
        self,
        ratio: float = 1 / 6,
        c: int = None,
        input_size=(3, 32, 32),
        P: float = 1.0,
        channel_type: str = "awgn",
        rician_k: float = 4.0,
        # controller params
        # FIX-2: a_high=1.35 (was 2.0) — consistent with fis_modules defaults
        a_min: float = 0.75,
        a_med: float = 1.00,
        a_high: float = 1.35,
        alpha_linear: float = 1.10,
        # FIX-3: snr_max_db=20.0 (was 13.0) — must match training snr_max
        snr_min_db: float = 0.0,
        snr_max_db: float = 20.0,
        eps: float = 1e-8,
    ):
        super().__init__()

        if c is None:
            if len(input_size) != 3:
                raise ValueError("input_size must be a (C,H,W) tuple")
            dummy = torch.zeros(1, *input_size)
            c = ratio2filtersize(dummy, ratio)
        self.c = int(c)

        self.encoder = _Encoder(c=self.c, P=P, apply_norm=False)
        self.decoder = _Decoder(self.c)
        self.channel = Channel(channel_type=channel_type, P=P, rician_k=rician_k)
        self.P = float(P)

        self.controller = FIS_SpatialPowerController(
            a_min=a_min,
            a_med=a_med,
            a_high=a_high,
            alpha_linear=alpha_linear,
            snr_min_db=snr_min_db,
            snr_max_db=snr_max_db,
            eps=eps,
        )
        self.eps = eps

    def set_channel(self, channel_type: str = None, snr: float = None, rician_k: float = None):
        if channel_type is not None:
            self.channel.channel_type = str(channel_type).lower()
        if snr is not None:
            self.channel.change_snr(snr)
        if rician_k is not None:
            self.channel.change_rician_k(rician_k)

    def forward(
        self,
        x: torch.Tensor,
        snr: float = None,
        budget: float = 1.0,
        mode: str = "full",
        return_info: bool = False,
    ):
        snr = snr if snr is not None else self.controller.snr_min_db

        z = self.encoder(x)  # [B, C, H', W']

        if return_info:
            A, info = self.controller(z, snr_db=snr, budget=budget, mode=mode, return_info=True)
        else:
            A = self.controller(z, snr_db=snr, budget=budget, mode=mode, return_info=False)

        # FIX-1: A is an amplitude map — apply directly, no sqrt.
        # The FIS output A is already in amplitude space (gain on z values).
        # Using sqrt(A) previously was mathematically equivalent to treating A as
        # a power map, which halved the effective redistribution strength.
        gain = A.clamp_min(self.eps)          # was: torch.sqrt(A.clamp_min(self.eps))
        z_g = z * gain.unsqueeze(1)

        # Re-normalize after gating to preserve the same average transmit power as baseline.
        z_tx = power_normalize(z_g, P=self.P, eps=self.eps)

        self.channel.change_snr(snr)
        y = self.channel(z_tx)
        x_hat = self.decoder(y)

        if return_info:
            info.setdefault("A", A)
            return z_tx, x_hat, info
        return z_tx, x_hat


# Backward-compatible exports
from model_baseline import DeepJSCC, ratio2filtersize
