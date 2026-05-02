# -*- coding: utf-8 -*-
"""
DeepJSCC + FIS spatial power controller.

Fix vs repo:
-----------
- Pass num_fading_taps to Channel based on encoder output channels c.
- num_fading_taps = c // 2  ->  per-subcarrier fading
- channel_rel now shape (B, num_taps) -> controller gets richer channel info.
"""
import torch
import torch.nn as nn

from channel import Channel           # ★ use fixed channel
from model_baseline import ratio2filtersize, _Encoder, _Decoder
from fis_modules import FIS_SpatialPowerController  # ★ use fixed FIS


def power_normalize(z: torch.Tensor, P: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample power normalization so that mean(z^2) = P."""
    p = z.pow(2).mean(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
    scale = torch.sqrt(torch.tensor(P, device=z.device, dtype=z.dtype) / p)
    return z * scale


class DeepJSCC_FIS(nn.Module):
    """Deep-JSCC backbone + FIS spatial power controller.

    ★ FIX ISSUE 3: num_fading_taps is set from encoder channel count c.
      When c=6 (CIFAR-10), num_fading_taps=3 -> per-subcarrier fading.
      channel_rel becomes (B, 3) instead of (B,) -> Layer-2 receives
      genuine per-channel channel information.
    """

    def __init__(
        self,
        ratio: float = 1 / 6,
        c: int = None,
        input_size=(3, 32, 32),
        P: float = 1.0,
        channel_type: str = "awgn",
        rician_k: float = 4.0,
        a_min: float = 0.75,
        a_med: float = 1.00,
        a_high: float = 1.35,
        alpha_linear: float = 1.10,
        snr_min_db: float = 0.0,
        snr_max_db: float = 20.0,
        use_channel_context: bool = True,
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

        # ★ FIX ISSUE 3: set num_fading_taps = c // 2 (per-subcarrier)
        # When c=6 -> 3 fading taps -> channel_rel shape (B, 3)
        # When c=8 -> 4 fading taps -> channel_rel shape (B, 4)
        num_fading_taps = self.c // 2

        self.channel = Channel(
            channel_type=channel_type,
            P=P,
            rician_k=rician_k,
            num_fading_taps=num_fading_taps,    # ★ NEW
        )
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
        self.use_channel_context = bool(use_channel_context)
        self.eps = eps

    def set_channel(
        self,
        channel_type: str = None,
        snr: float = None,
        rician_k: float = None,
        rayleigh_equalize: bool = None,
    ):
        if channel_type is not None:
            self.channel.channel_type = str(channel_type).lower()
        if snr is not None:
            self.channel.change_snr(snr)
        if rician_k is not None:
            self.channel.change_rician_k(rician_k)
        if rayleigh_equalize is not None:
            self.channel.enable_rayleigh_equalization(rayleigh_equalize)

    def forward(
        self,
        x: torch.Tensor,
        snr: float = None,
        budget: float = 1.0,
        mode: str = "full",
        return_info: bool = False,
    ):
        snr = snr if snr is not None else self.controller.snr_min_db
        z = self.encoder(x)

        channel_ctx = None
        if self.use_channel_context:
            channel_ctx = self.channel.sample_context(
                batch_size=z.shape[0],
                device=z.device,
                dtype=z.dtype,
            )

        # channel_rel: (B,) for block fading, (B, num_taps) for per-subcarrier
        channel_rel_val = None
        if channel_ctx is not None:
            channel_rel_val = channel_ctx.get(
                "channel_rel", channel_ctx["gamma_eff_norm"]
            )

        if return_info:
            A, info = self.controller(
                z,
                snr_db=snr,
                budget=budget,
                mode=mode,
                channel_rel=channel_rel_val,
                return_info=True,
            )
        else:
            A = self.controller(
                z,
                snr_db=snr,
                budget=budget,
                mode=mode,
                channel_rel=channel_rel_val,
                return_info=False,
            )

        gain = A.clamp_min(self.eps)
        z_g = z * gain.unsqueeze(1)
        z_tx = power_normalize(z_g, P=self.P, eps=self.eps)

        self.channel.change_snr(snr)
        y = self.channel(z_tx)
        x_hat = self.decoder(y)

        if return_info:
            info.setdefault("A", A)
            if channel_ctx is not None:
                info["channel_ctx"] = channel_ctx
            return z_tx, x_hat, info

        return z_tx, x_hat
