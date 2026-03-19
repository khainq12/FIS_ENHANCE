# -*- coding: utf-8 -*-
"""
Channel models used by the paper-simulation scripts.

Supported channels
- AWGN: additive white Gaussian noise.
- Rayleigh: proper complex flat fading, optionally equalized with perfect CSI.
- Rician: proper complex flat fading with LOS + NLOS, optionally equalized with perfect CSI.
- rayleigh_legacy: legacy non-physical variant retained only for reproducibility.

Notation
- Input x is shaped [B, C, H, W], where C is even and represents (I,Q) halves:
  x_I = x[:, :C/2, ...], x_Q = x[:, C/2:, ...].
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class Channel(nn.Module):
    def __init__(
        self,
        channel_type: str = "awgn",
        P: float = 1.0,
        snr_db: float = 13.0,
        eps: float = 1e-8,
        rician_k: float = 4.0,
    ):
        super().__init__()
        self.channel_type = str(channel_type).lower()
        self.P = float(P)
        self.snr_db = float(snr_db)
        self.eps = float(eps)
        self.rician_k = float(rician_k)

        # If True, assume perfect CSI at Rx and apply 1-tap equalization for fading channels.
        self._fading_equalize = False

    def change_snr(self, snr_db: float) -> None:
        self.snr_db = float(snr_db)

    def change_rician_k(self, rician_k: float) -> None:
        self.rician_k = float(rician_k)

    def enable_rayleigh_equalization(self, enable: bool = True) -> None:
        # Backward-compatible name. Works for Rayleigh and Rician.
        self._fading_equalize = bool(enable)

    def enable_fading_equalization(self, enable: bool = True) -> None:
        self._fading_equalize = bool(enable)

    def _sigma(self, device, dtype=torch.float32) -> torch.Tensor:
        snr = 10.0 ** (self.snr_db / 10.0)
        # With unit-power normalization at Tx, E[|x|^2]≈P, so noise variance is P/SNR.
        # Split equally over I/Q -> sigma^2 per real dim = (P/SNR)/2.
        sigma = math.sqrt(self.P / snr / 2.0)
        return torch.tensor(sigma, device=device, dtype=dtype)

    def _awgn(self, x: torch.Tensor) -> torch.Tensor:
        sigma = self._sigma(x.device, x.dtype)
        noise = sigma * torch.randn_like(x)
        return x + noise

    def _split_iq(self, x: torch.Tensor):
        B, C, H, W = x.shape
        if C % 2 != 0:
            raise ValueError("Channel expects even channel dimension (I/Q halves).")
        C2 = C // 2
        return x[:, :C2], x[:, C2:]

    def _apply_complex_fading(self, x: torch.Tensor, hI: torch.Tensor, hQ: torch.Tensor) -> torch.Tensor:
        xI, xQ = self._split_iq(x)

        yI = xI * hI - xQ * hQ
        yQ = xI * hQ + xQ * hI

        sigma = self._sigma(x.device, x.dtype)
        yI = yI + sigma * torch.randn_like(yI)
        yQ = yQ + sigma * torch.randn_like(yQ)

        if self._fading_equalize:
            denom = (hI * hI + hQ * hQ).clamp_min(self.eps)
            xI_hat = (yI * hI + yQ * hQ) / denom
            xQ_hat = (yQ * hI - yI * hQ) / denom
            return torch.cat([xI_hat, xQ_hat], dim=1)

        return torch.cat([yI, yQ], dim=1)

    def _rayleigh(self, x: torch.Tensor) -> torch.Tensor:
        """
        Proper complex flat Rayleigh fading:
          y = h * x + n, with h ~ CN(0,1).
        In real representation:
          y_I = h_I x_I - h_Q x_Q + n_I
          y_Q = h_I x_Q + h_Q x_I + n_Q
        """
        B = x.shape[0]
        hI = torch.randn(B, 1, 1, 1, device=x.device, dtype=x.dtype) / math.sqrt(2.0)
        hQ = torch.randn(B, 1, 1, 1, device=x.device, dtype=x.dtype) / math.sqrt(2.0)
        return self._apply_complex_fading(x, hI, hQ)

    def _rician(self, x: torch.Tensor) -> torch.Tensor:
        """
        Proper complex flat Rician fading with unit average channel power:
          h = sqrt(K/(K+1)) * h_LOS + sqrt(1/(K+1)) * h_NLOS
        where |h_LOS| = 1 and h_NLOS ~ CN(0,1).

        rician_k is the linear K-factor. Example values: 1, 4, 10.
        """
        B = x.shape[0]
        K = max(float(self.rician_k), 0.0)

        # Random LOS phase per sample keeps the LOS component deterministic in magnitude
        # without locking the phase to zero across the full dataset.
        theta = 2.0 * math.pi * torch.rand(B, 1, 1, 1, device=x.device, dtype=x.dtype)
        h_los_I = torch.cos(theta)
        h_los_Q = torch.sin(theta)

        h_nlos_I = torch.randn(B, 1, 1, 1, device=x.device, dtype=x.dtype) / math.sqrt(2.0)
        h_nlos_Q = torch.randn(B, 1, 1, 1, device=x.device, dtype=x.dtype) / math.sqrt(2.0)

        los_scale = math.sqrt(K / (K + 1.0)) if K > 0.0 else 0.0
        nlos_scale = math.sqrt(1.0 / (K + 1.0))

        hI = los_scale * h_los_I + nlos_scale * h_nlos_I
        hQ = los_scale * h_los_Q + nlos_scale * h_nlos_Q
        return self._apply_complex_fading(x, hI, hQ)

    def _rayleigh_legacy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Legacy Rayleigh (for reproducibility with older code):
        - Apply independent real fading to I-half and Q-half.
        - Optional equalization divides by h0/h1 separately.
        This is non-physical for complex baseband, but retained for reproducibility only.
        """
        B, C, H, W = x.shape
        if C % 2 != 0:
            raise ValueError("Channel expects even channel dimension (I/Q halves).")
        C2 = C // 2
        xI, xQ = x[:, :C2], x[:, C2:]

        h0 = torch.randn(B, 1, 1, 1, device=x.device, dtype=x.dtype) / math.sqrt(2.0)
        h1 = torch.randn(B, 1, 1, 1, device=x.device, dtype=x.dtype) / math.sqrt(2.0)

        yI = xI * h0
        yQ = xQ * h1

        sigma = self._sigma(x.device, x.dtype)
        yI = yI + sigma * torch.randn_like(yI)
        yQ = yQ + sigma * torch.randn_like(yQ)

        if self._fading_equalize:
            yI = yI / h0.clamp_min(self.eps)
            yQ = yQ / h1.clamp_min(self.eps)

        return torch.cat([yI, yQ], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ct = self.channel_type
        if ct in ("awgn",):
            return self._awgn(x)
        if ct in ("rayleigh",):
            return self._rayleigh(x)
        if ct in ("rician",):
            return self._rician(x)
        if ct in ("rayleigh_legacy", "rayleighlegacy", "rayleigh-legacy"):
            return self._rayleigh_legacy(x)
        raise ValueError(
            f"Unsupported channel_type='{self.channel_type}'. "
            "Use 'awgn', 'rayleigh', 'rician', or 'rayleigh_legacy'."
        )

