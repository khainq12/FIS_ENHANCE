# -*- coding: utf-8 -*-
"""
Channel models with optional transmitter-side fading context.

This file is a drop-in replacement for the current FIS_ENHANCE channel.py,
but adds a small mechanism that exposes a per-sample channel-reliability
context before transmission. The context is intended for the FIS controller
so that the controller can distinguish nominal-SNR degradation from deep
instantaneous fading.

Key idea
--------
- AWGN: context is derived from the nominal SNR only.
- Rayleigh / Rician: context is derived from the instantaneous fading power
  |h|^2 and the current noise variance.
- The same sampled fading coefficient is reused in the subsequent forward
  pass, so the controller sees the same block-fading realization that is
  applied by the channel.

Supported channels
------------------
- AWGN
- Rayleigh
- Rician
- rayleigh_legacy (retained only for reproducibility)
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Utility: per-batch min-max normalization
# ---------------------------------------------------------------------------

def per_batch_minmax(
    x: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize *x* per-batch element to [0, 1].

    Parameters
    ----------
    x : torch.Tensor
        Tensor of arbitrary shape where the first dimension is the batch axis.
    eps : float
        Small constant to avoid division by zero when batch-min == batch-max.

    Returns
    -------
    torch.Tensor
        Tensor with the same shape as *x*, where each batch element is
        linearly mapped from [min, max] within that batch to [0, 1].

    Example
    -------
    >>> x = torch.tensor([[-20.0, -5.0], [-30.0, 0.0]])  # 2 samples, 2 "spatial" dims
    >>> per_batch_minmax(x)
    tensor([[0.0000, 1.0000],
            [0.0000, 1.0000]])

    Notes
    -----
    For a batch of size 1 the denominator (max - min) collapses to zero;
    the function safely returns 0.5 for every element in that case via *eps*.
    """
    # x: (B, ...) — compute min/max along every axis except batch
    dims = tuple(range(1, x.ndim)) if x.ndim > 1 else None
    if dims:
        x_min = x.amin(dim=dims, keepdim=True)
        x_max = x.amax(dim=dims, keepdim=True)
    else:
        # 1-D case (B,)
        x_min = x.min()
        x_max = x.max()
    x_range = (x_max - x_min).clamp_min(eps)
    return (x - x_min) / x_range


# ---------------------------------------------------------------------------
# Channel module
# ---------------------------------------------------------------------------

class Channel(nn.Module):
    """Wireless channel simulator with per-sample reliability context.

    Supported channel types:
        * ``"awgn"``            — Additive White Gaussian Noise
        * ``"rayleigh"``        — Complex Rayleigh block-fading + AWGN
        * ``"rician"``          — Complex Rician block-fading + AWGN
        * ``"rayleigh_legacy"`` — Legacy real-coefficient fading (repro only)

    Parameters
    ----------
    channel_type : str
        One of the supported types listed above.
    P : float
        Average signal power (default 1.0).
    snr_db : float
        Channel SNR in dB (default 13.0).
    eps : float
        Numerical stability constant (default 1e-8).
    rician_k : float
        Rician K-factor in dB (default 4.0).  Only used when
        *channel_type* is ``"rician"``.
    context_db_min : float
        Lower clipping bound (dB) for context normalization (default -15.0).
    context_db_max : float
        Upper clipping bound (dB) for context normalization (default 20.0).
    """

    _VALID_TYPES = {"awgn", "rayleigh", "rician",
                    "rayleigh_legacy", "rayleighlegacy", "rayleigh-legacy"}

    def __init__(
        self,
        channel_type: str = "awgn",
        P: float = 1.0,
        snr_db: float = 13.0,
        eps: float = 1e-8,
        rician_k: float = 4.0,
        context_db_min: float = -15.0,
        context_db_max: float = 20.0,
    ):
        super().__init__()
        self.channel_type = str(channel_type).lower()
        if self.channel_type not in self._VALID_TYPES:
            raise ValueError(
                f"Unsupported channel_type='{self.channel_type}'. "
                f"Valid types: {sorted(self._VALID_TYPES)}"
            )
        self.P = float(P)
        self.snr_db = float(snr_db)
        self.eps = float(eps)
        self.rician_k = float(rician_k)
        self.context_db_min = float(context_db_min)
        self.context_db_max = float(context_db_max)

        # If True, assume perfect CSI at Rx and apply 1-tap equalization
        # for fading channels.
        self._fading_equalize: bool = False

        # Canonical alias set for legacy channel types.
        self._legacy_alias: set = {
            "rayleigh_legacy", "rayleighlegacy", "rayleigh-legacy"
        }

        # Optional pending fading realization.  If present, forward() uses it
        # once and then clears it.  This keeps the controller context aligned
        # with the actually applied channel realization.
        self._pending_fading: Optional[Tuple[str, torch.Tensor, torch.Tensor]] = None

        # Last sampled context — kept for diagnostics / logging.
        self.last_context: Dict[str, torch.Tensor] = {}

    # ----- public helpers ---------------------------------------------------

    def change_snr(self, snr_db: float) -> None:
        """Update the channel SNR (in dB)."""
        self.snr_db = float(snr_db)

    def change_rician_k(self, rician_k: float) -> None:
        """Update the Rician K-factor (linear, not dB)."""
        self.rician_k = float(rician_k)

    def enable_rayleigh_equalization(self, enable: bool = True) -> None:
        """Enable/disable 1-tap fading equalization (deprecated alias)."""
        self._fading_equalize = bool(enable)

    def enable_fading_equalization(self, enable: bool = True) -> None:
        """Enable/disable 1-tap fading equalization."""
        self._fading_equalize = bool(enable)

    def _is_legacy(self) -> bool:
        return self.channel_type in self._legacy_alias

    def _is_fading(self) -> bool:
        return self.channel_type in ("rayleigh", "rician") or self._is_legacy()

    # ----- private helpers --------------------------------------------------

    def _sigma(self, device, dtype=torch.float32) -> torch.Tensor:
        """Noise standard deviation per real dimension."""
        snr = 10.0 ** (self.snr_db / 10.0)
        sigma = math.sqrt(self.P / snr / 2.0)
        return torch.tensor(sigma, device=device, dtype=dtype)

    def _snr_lin(self, device, dtype=torch.float32) -> torch.Tensor:
        """Nominal SNR in linear scale."""
        return torch.tensor(10.0 ** (self.snr_db / 10.0), device=device, dtype=dtype)

    def _norm_db(self, x_db: torch.Tensor) -> torch.Tensor:
        """Map dB value to [0, 1] using configured clipping bounds."""
        x = (x_db - self.context_db_min) / (
            self.context_db_max - self.context_db_min + self.eps
        )
        return torch.clamp(x, 0.0, 1.0)

    def _split_iq(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split the channel dim into I and Q halves."""
        B, C, H, W = x.shape
        if C % 2 != 0:
            raise ValueError(
                "Channel expects even channel dimension for I/Q representation "
                f"(got C={C})."
            )
        C2 = C // 2
        return x[:, :C2], x[:, C2:]

    # ----- noise-only channel -----------------------------------------------

    def _awgn(self, x: torch.Tensor) -> torch.Tensor:
        sigma = self._sigma(x.device, x.dtype)
        noise = sigma * torch.randn_like(x)
        return x + noise

    # ----- fading helpers ---------------------------------------------------

    def _apply_complex_fading(
        self,
        x: torch.Tensor,
        hI: torch.Tensor,
        hQ: torch.Tensor,
    ) -> torch.Tensor:
        """Apply complex fading h = hI + j*hQ, add AWGN, optionally equalize."""
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

    # ----- fading samplers --------------------------------------------------

    @staticmethod
    def _sample_rayleigh(
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample complex Rayleigh fading coefficients.

        Returns (hI, hQ) each of shape (B, 1, 1, 1) with E[|h|^2] = 1.
        """
        hI = torch.randn(batch_size, 1, 1, 1, device=device, dtype=dtype) / math.sqrt(2.0)
        hQ = torch.randn(batch_size, 1, 1, 1, device=device, dtype=dtype) / math.sqrt(2.0)
        return hI, hQ

    def _sample_rician(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample complex Rician fading coefficients.

        Returns (hI, hQ) each of shape (B, 1, 1, 1).
        The K-factor is taken from ``self.rician_k`` (linear scale).
        """
        K = max(float(self.rician_k), 0.0)
        theta = 2.0 * math.pi * torch.rand(
            batch_size, 1, 1, 1, device=device, dtype=dtype
        )
        h_los_I = torch.cos(theta)
        h_los_Q = torch.sin(theta)

        h_nlos_I = torch.randn(
            batch_size, 1, 1, 1, device=device, dtype=dtype
        ) / math.sqrt(2.0)
        h_nlos_Q = torch.randn(
            batch_size, 1, 1, 1, device=device, dtype=dtype
        ) / math.sqrt(2.0)

        los_scale = math.sqrt(K / (K + 1.0)) if K > 0.0 else 0.0
        nlos_scale = math.sqrt(1.0 / (K + 1.0))

        hI = los_scale * h_los_I + nlos_scale * h_nlos_I
        hQ = los_scale * h_los_Q + nlos_scale * h_nlos_Q
        return hI, hQ

    @staticmethod
    def _sample_legacy_rayleigh(
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample *legacy* real-coefficient Rayleigh fading (repro only).

        Returns (h0, h1) each of shape (B, 1, 1, 1).
        """
        h0 = torch.randn(batch_size, 1, 1, 1, device=device, dtype=dtype) / math.sqrt(2.0)
        h1 = torch.randn(batch_size, 1, 1, 1, device=device, dtype=dtype) / math.sqrt(2.0)
        return h0, h1

    # ----- context sampling -------------------------------------------------

    @torch.no_grad()
    def sample_context(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Dict[str, torch.Tensor]:
        """Sample a per-batch channel reliability context for the transmitter.

        Returns a dictionary containing at least:

        - ``gamma_eff_lin`` : instantaneous effective SNR (linear).
        - ``gamma_eff_db``  : same quantity in dB.
        - ``gamma_eff_norm``: normalized to [0, 1] via per-batch min-max.
        - ``channel_rel``   : explicit reliability for the FIS controller.
        - ``h_abs2``        : instantaneous channel power (AWGN → 1).

        For fading channels the sampled coefficients are cached and reused
        in the next ``forward()`` call so the controller sees the same
        block-fading realization that the channel applies.
        """
        ct = self.channel_type
        sigma = self._sigma(device, dtype)
        noise_var = 2.0 * sigma * sigma
        snr_lin = self._snr_lin(device, dtype)

        if ct == "awgn":
            h_abs2 = torch.ones(
                batch_size, 1, 1, 1, device=device, dtype=dtype
            )
            gamma_eff_lin = snr_lin.view(1, 1, 1, 1).expand_as(h_abs2)
            self._pending_fading = None

        elif ct == "rayleigh":
            hI, hQ = self._sample_rayleigh(batch_size, device, dtype)
            h_abs2 = (hI * hI + hQ * hQ).clamp_min(self.eps)
            gamma_eff_lin = snr_lin.view(1, 1, 1, 1) * h_abs2
            self._pending_fading = ("complex", hI, hQ)

        elif ct == "rician":
            hI, hQ = self._sample_rician(batch_size, device, dtype)
            h_abs2 = (hI * hI + hQ * hQ).clamp_min(self.eps)
            gamma_eff_lin = snr_lin.view(1, 1, 1, 1) * h_abs2
            self._pending_fading = ("complex", hI, hQ)

        elif self._is_legacy():
            h0, h1 = self._sample_legacy_rayleigh(batch_size, device, dtype)
            h_abs2 = 0.5 * (h0 * h0 + h1 * h1).clamp_min(self.eps)
            gamma_eff_lin = snr_lin.view(1, 1, 1, 1) * h_abs2
            self._pending_fading = ("legacy", h0, h1)

        else:
            raise ValueError(
                f"Unsupported channel_type='{self.channel_type}'. "
                "Use 'awgn', 'rayleigh', 'rician', or 'rayleigh_legacy'."
            )

        # Derived scalars
        gamma_eff_db = 10.0 * torch.log10(gamma_eff_lin.clamp_min(self.eps))
        gamma_eff_norm = per_batch_minmax(gamma_eff_db)  # ← per-batch → [0, 1]

        posteq_noise_var = noise_var / h_abs2.clamp_min(self.eps)
        eq_quality = 1.0 / posteq_noise_var.clamp_min(self.eps)

        # Explicit channel reliability for the FIS controller.
        if ct == "awgn":
            channel_rel = torch.ones_like(gamma_eff_norm)
        elif self._is_fading():
            if self._fading_equalize:
                channel_rel = torch.sigmoid(
                    (torch.log1p(eq_quality) - math.log(2.0)) / 1.0
                )
            else:
                channel_rel = torch.sigmoid(
                    (gamma_eff_db - self.snr_db) / 10.0
                )
        else:
            channel_rel = gamma_eff_norm

        channel_rel = channel_rel.clamp(0.0, 1.0)

        # Remove the three trailing singleton dims → (B,)
        squeeze = lambda t: t.squeeze(-1).squeeze(-1).squeeze(-1)

        ctx: Dict[str, torch.Tensor] = {
            "gamma_eff_lin":    squeeze(gamma_eff_lin),
            "gamma_eff_db":     squeeze(gamma_eff_db),
            "gamma_eff_norm":   squeeze(gamma_eff_norm),
            "channel_rel":      squeeze(channel_rel),
            "h_abs2":           squeeze(h_abs2),
            "posteq_noise_var": squeeze(posteq_noise_var),
            "eq_quality":       squeeze(eq_quality),
            "channel_type":     ct,
            "fading_equalize":  torch.tensor(
                float(self._fading_equalize), device=device, dtype=dtype
            ),
        }
        self.last_context = ctx
        return ctx

    # ----- forward pass -----------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the channel to *x* and return the received signal.

        If ``sample_context`` was called immediately before this forward
        (and a fading channel is active), the cached fading coefficients
        are consumed so that the same realization seen by the controller
        is actually applied.
        """
        ct = self.channel_type

        if ct == "awgn":
            return self._awgn(x)

        if ct in ("rayleigh", "rician"):
            if self._pending_fading is not None:
                kind, hI, hQ = self._pending_fading
                self._pending_fading = None
                if kind != "complex":
                    raise RuntimeError(
                        f"Pending fading kind mismatch: expected 'complex', "
                        f"got '{kind}'."
                    )
            else:
                if ct == "rayleigh":
                    hI, hQ = self._sample_rayleigh(x.shape[0], x.device, x.dtype)
                else:
                    hI, hQ = self._sample_rician(x.shape[0], x.device, x.dtype)
            return self._apply_complex_fading(x, hI, hQ)

        if self._is_legacy():
            if self._pending_fading is not None:
                kind, h0, h1 = self._pending_fading
                self._pending_fading = None
                if kind != "legacy":
                    raise RuntimeError(
                        f"Pending fading kind mismatch: expected 'legacy', "
                        f"got '{kind}'."
                    )
            else:
                h0, h1 = self._sample_legacy_rayleigh(
                    x.shape[0], x.device, x.dtype
                )

            B, C, H, W = x.shape
            if C % 2 != 0:
                raise ValueError(
                    "Channel expects even channel dimension for I/Q "
                    f"(got C={C})."
                )
            C2 = C // 2
            xI, xQ = x[:, :C2], x[:, C2:]
            yI = xI * h0
            yQ = xQ * h1

            sigma = self._sigma(x.device, x.dtype)
            yI = yI + sigma * torch.randn_like(yI)
            yQ = yQ + sigma * torch.randn_like(yQ)

            if self._fading_equalize:
                yI = yI / h0.clamp_min(self.eps)
                yQ = yQ / h1.clamp_min(self.eps)

            return torch.cat([yI, yQ], dim=1)

        raise ValueError(
            f"Unsupported channel_type='{self.channel_type}'. "
            "Use 'awgn', 'rayleigh', 'rician', or 'rayleigh_legacy'."
        )
