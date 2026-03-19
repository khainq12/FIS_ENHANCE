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

    Outputs:
      - z_tx: gated + power-normalized latent sent over the channel
      - x_hat: reconstructed image
      - optional info dict: importance map, gain map, rule activations
    """
    def __init__(
        self,
        ratio: float = 1 / 12,
        c: int = None,
        input_size=(3, 32, 32),
        P: float = 1.0,
        channel_type: str = "awgn",
        rician_k: float = 4.0,
        # controller params
        a_min: float = 0.70,
        a_med: float = 1.00,
        a_high: float = 2.0,
        alpha_linear: float = 0.8,
        snr_min_db: float = 0.0,
        snr_max_db: float = 13.0,
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
        # Encoder outputs pre-normalization latent; we apply spatial gating then power-normalize once
        snr = snr if snr is not None else self.controller.snr_min_db
        z = self.encoder(x)  # [B,C,H',W']

        if return_info:
            A, info = self.controller(z, snr_db=snr, budget=budget, mode=mode, return_info=True)
        else:
            A = self.controller(z, snr_db=snr, budget=budget, mode=mode, return_info=False)

        # Spatial gating uses sqrt(A) as amplitude gain (A is a power map)
        gain = torch.sqrt(A.clamp_min(self.eps))
        z_g = z * gain.unsqueeze(1)

        # Re-normalize after gating to preserve the same average transmit power as baseline
        z_tx = power_normalize(z_g, P=self.P, eps=self.eps)


        # Channel + decoder
        self.channel.change_snr(snr)
        y = self.channel(z_tx)
        x_hat = self.decoder(y)

        if return_info:
            info.setdefault("A", A)
            return z_tx, x_hat, info
        return z_tx, x_hat


# Backward-compatible exports
from model_baseline import DeepJSCC, ratio2filtersize
