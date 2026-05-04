# -*- coding: utf-8 -*-
"""
Channel models with optional transmitter-side fading context.

Fixes applied (vs repo feature/fis-channel-aware-controller-r2):
----------------------------------------------------------------
1) Per-subcarrier fading: h shape (B,1,1,1) -> (B,num_taps,1,1)
   -> channel_rel becomes (B, num_taps) instead of (B,)
   -> Layer-2 FIS finally receives spatially-meaningful channel info

2) Legacy bug fix: yQ = yQ * h1 -> yQ = xQ * h1
   Legacy bug fix: xI, xQ = x[:, :C2], x[:, :C2] -> x[:, C2:]

3) _norm_db fixed-window normalization (already in repo, kept)

4) channel_rel = fading ONLY, separated from SNR (already in repo, kept)
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def per_batch_minmax(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    dims = tuple(range(1, x.ndim)) if x.ndim > 1 else None
    if dims:
        x_min = x.amin(dim=dims, keepdim=True)
        x_max = x.amax(dim=dims, keepdim=True)
    else:
        x_min = x.min()
        x_max = x.max()
    x_range = (x_max - x_min).clamp_min(eps)
    return (x - x_min) / x_range


class Channel(nn.Module):
    """Wireless channel simulator with per-sample reliability context.

    Supported channel types:
        * ``"awgn"``           -- Additive White Gaussian Noise
        * ``"rayleigh"``       -- Complex Rayleigh block-fading + AWGN
        * ``"rician"``         -- Complex Rician block-fading + AWGN
        * ``"rayleigh_legacy"``-- Legacy real-coefficient fading (repro only)

    New parameter
    -------------
    num_fading_taps : int
        Number of independent fading coefficients per sample.

        * ``1``  (default) -- Block fading: one h per image.
          channel_rel shape = (B,).  Backward-compatible.
        * ``> 1``          -- Per-subcarrier fading: one h per I/Q subcarrier.
          channel_rel shape = (B, num_fading_taps).
          When set equal to C_half = c // 2, each I/Q pair gets its own
          fading coefficient, giving the FIS controller genuine spatial
          information to act on.
    """

    _VALID_TYPES = {
        "awgn", "rayleigh", "rician",
        "rayleigh_legacy", "rayleighlegacy", "rayleigh-legacy",
    }

    def __init__(
        self,
        channel_type: str = "awgn",
        P: float = 1.0,
        snr_db: float = 13.0,
        eps: float = 1e-8,
        rician_k: float = 4.0,
        context_db_min: float = -15.0,
        context_db_max: float = 20.0,
        # ★ FIX ISSUE 3: NEW parameter
        num_fading_taps: int = 1,
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
        self.h_abs2_min: float = 0.01
        self.h_abs_min: float = math.sqrt(self.h_abs2_min)

        # ★ FIX ISSUE 3: store num_fading_taps
        self.num_fading_taps = max(1, int(num_fading_taps))

        self._fading_equalize: bool = False
        self._legacy_alias: set = {
            "rayleigh_legacy", "rayleighlegacy", "rayleigh-legacy",
        }
        self._pending_fading: Optional[Tuple[str, torch.Tensor, torch.Tensor]] = None
        self.last_context: Dict[str, torch.Tensor] = {}

    # ── public helpers ──────────────────────────────────────────
    def change_snr(self, snr_db: float) -> None:
        self.snr_db = float(snr_db)

    def change_rician_k(self, rician_k: float) -> None:
        self.rician_k = float(rician_k)

    def enable_rayleigh_equalization(self, enable: bool = True) -> None:
        self._fading_equalize = bool(enable)

    def enable_fading_equalization(self, enable: bool = True) -> None:
        self._fading_equalize = bool(enable)

    def _is_legacy(self) -> bool:
        return self.channel_type in self._legacy_alias

    def _is_fading(self) -> bool:
        return self.channel_type in ("rayleigh", "rician") or self._is_legacy()

    # ── private helpers ─────────────────────────────────────────
    def _sigma(self, device, dtype=torch.float32) -> torch.Tensor:
        snr = 10.0 ** (self.snr_db / 10.0)
        sigma = math.sqrt(self.P / snr / 2.0)
        return torch.tensor(sigma, device=device, dtype=dtype)

    def _snr_lin(self, device, dtype=torch.float32) -> torch.Tensor:
        return torch.tensor(10.0 ** (self.snr_db / 10.0), device=device, dtype=dtype)

    def _norm_db(self, x_db: torch.Tensor) -> torch.Tensor:
        """Fixed-window dB normalization (NOT per-batch minmax)."""
        x = (x_db - self.context_db_min) / (
            self.context_db_max - self.context_db_min + self.eps
        )
        return torch.clamp(x, 0.0, 1.0)

    def _split_iq(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        if C % 2 != 0:
            raise ValueError(
                f"Channel expects even channel dimension (got C={C})."
            )
        C2 = C // 2
        return x[:, :C2], x[:, C2:]

    # ── noise-only channel ──────────────────────────────────────
    def _awgn(self, x: torch.Tensor) -> torch.Tensor:
        sigma = self._sigma(x.device, x.dtype)
        return x + sigma * torch.randn_like(x)

    # ── fading helpers ──────────────────────────────────────────
    def _apply_complex_fading(
        self,
        x: torch.Tensor,
        hI: torch.Tensor,
        hQ: torch.Tensor,
    ) -> torch.Tensor:
        xI, xQ = self._split_iq(x)     # (B, C2, H, W)
        C2 = xI.shape[1]
        n = hI.shape[1]                 # num_fading_taps

        # ★ If fading taps != I/Q channels, repeat to match
        if n != C2 and C2 % n == 0:
            repeat = C2 // n
            hI = hI.repeat(1, repeat, 1, 1)
            hQ = hQ.repeat(1, repeat, 1, 1)
        elif n != C2:
            raise ValueError(
                f"fading taps ({n}) incompatible with I/Q channels ({C2}); "
                f"C2 must be a multiple of n."
            )

        yI = xI * hI - xQ * hQ
        yQ = xI * hQ + xQ * hI

        sigma = self._sigma(x.device, x.dtype)
        yI = yI + sigma * torch.randn_like(yI)
        yQ = yQ + sigma * torch.randn_like(yQ)

        if self._fading_equalize:
            denom = (hI * hI + hQ * hQ).clamp_min(self.h_abs2_min)
            xI_hat = (yI * hI + yQ * hQ) / denom
            xQ_hat = (yQ * hI - yI * hQ) / denom
            return torch.cat([xI_hat, xQ_hat], dim=1)
        return torch.cat([yI, yQ], dim=1)

    # ── fading samplers ─────────────────────────────────────────
    # ★ FIX ISSUE 3: all samplers now use self.num_fading_taps
    #    shape changes from (B,1,1,1) to (B,num_fading_taps,1,1)

    @staticmethod
    def _sample_rayleigh(
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        num_taps: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hI = torch.randn(batch_size, num_taps, 1, 1, device=device, dtype=dtype) / math.sqrt(2.0)
        hQ = torch.randn(batch_size, num_taps, 1, 1, device=device, dtype=dtype) / math.sqrt(2.0)
        return hI, hQ

    def _sample_rician(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = self.num_fading_taps
        K = max(float(self.rician_k), 0.0)
        theta = 2.0 * math.pi * torch.rand(batch_size, n, 1, 1, device=device, dtype=dtype)
        h_los_I = torch.cos(theta)
        h_los_Q = torch.sin(theta)
        h_nlos_I = torch.randn(batch_size, n, 1, 1, device=device, dtype=dtype) / math.sqrt(2.0)
        h_nlos_Q = torch.randn(batch_size, n, 1, 1, device=device, dtype=dtype) / math.sqrt(2.0)
        los_scale = math.sqrt(K / (K + 1.0)) if K > 0.0 else 0.0
        nlos_scale = math.sqrt(1.0 / (K + 1.0))
        hI = los_scale * h_los_I + nlos_scale * h_nlos_I
        hQ = los_scale * h_los_Q + nlos_scale * h_nlos_Q
        return hI, hQ

    def _sample_legacy_rayleigh(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = self.num_fading_taps
        h0 = torch.randn(batch_size, n, 1, 1, device=device, dtype=dtype) / math.sqrt(2.0)
        h1 = torch.randn(batch_size, n, 1, 1, device=device, dtype=dtype) / math.sqrt(2.0)
        return h0, h1

    # ── context sampling ────────────────────────────────────────
    @torch.no_grad()
    def sample_context(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Dict[str, torch.Tensor]:
        ct = self.channel_type
        sigma = self._sigma(device, dtype)
        noise_var = 2.0 * sigma * sigma
        snr_lin = self._snr_lin(device, dtype)
        n = self.num_fading_taps          # ★ 1 = block, >1 = per-subcarrier

        if ct == "awgn":
            h_abs2 = torch.ones(batch_size, n, 1, 1, device=device, dtype=dtype)
            gamma_eff_lin = snr_lin.view(1, 1, 1, 1).expand_as(h_abs2)
            self._pending_fading = None

        elif ct == "rayleigh":
            hI, hQ = self._sample_rayleigh(batch_size, device, dtype, num_taps=n)
            h_abs2 = (hI * hI + hQ * hQ).clamp_min(self.eps)       # (B,n,1,1)
            gamma_eff_lin = snr_lin.view(1, 1, 1, 1) * h_abs2
            self._pending_fading = ("complex", hI, hQ)

        elif ct == "rician":
            hI, hQ = self._sample_rician(batch_size, device, dtype)
            h_abs2 = (hI * hI + hQ * hQ).clamp_min(self.eps)       # (B,n,1,1)
            gamma_eff_lin = snr_lin.view(1, 1, 1, 1) * h_abs2
            self._pending_fading = ("complex", hI, hQ)

        elif self._is_legacy():
            h0, h1 = self._sample_legacy_rayleigh(batch_size, device, dtype)
            h_abs2 = 0.5 * (h0 * h0 + h1 * h1).clamp_min(self.eps) # (B,n,1,1)
            gamma_eff_lin = snr_lin.view(1, 1, 1, 1) * h_abs2
            self._pending_fading = ("legacy", h0, h1)

        else:
            raise ValueError(f"Unsupported channel_type='{self.channel_type}'.")

        gamma_eff_db = 10.0 * torch.log10(gamma_eff_lin.clamp_min(self.eps))
        gamma_eff_norm = self._norm_db(gamma_eff_db)

        noise_var_map = noise_var.view(1, 1, 1, 1).expand_as(h_abs2)

        # Equalization-specific diagnostics
        posteq_noise_var = None
        eq_quality = None
        if self._is_fading() and self._fading_equalize:
            posteq_noise_var = noise_var_map / h_abs2.clamp_min(self.h_abs2_min)
            eq_quality = 1.0 / posteq_noise_var.clamp_min(self.eps)

        # ── channel_rel: fading quality ONLY (independent of SNR) ──
        if ct == "awgn":
            # ★ PATCH r3-C: use gamma_eff_norm (SNR-normalised) instead of
            # a constant ones tensor.  With the old constant=1.0 the
            # good-channel attenuation gate in FIS_PowerAllocation was always
            # fully open, self-disabling the controller on every AWGN forward
            # pass.  gamma_eff_norm varies with the configured SNR (low SNR →
            # low value, high SNR → high value) so the controller now receives
            # a meaningful signal even without fading.
            # Shape: (B, n) after squeeze below (n = num_fading_taps).
            channel_rel = gamma_eff_norm
        elif self._is_fading():
            if self._fading_equalize:
                fading_rel = self._norm_db(10.0 * torch.log10(
                    h_abs2.clamp_min(self.eps)
                ))                                              # (B,n,1,1)
                assert posteq_noise_var is not None
                eq_rel = 1.0 / (1.0 + posteq_noise_var)
                channel_rel = 0.6 * fading_rel + 0.4 * eq_rel   # (B,n,1,1)
            else:
                channel_rel = self._norm_db(
                    10.0 * torch.log10(h_abs2.clamp_min(self.eps))
                )                                                # (B,n,1,1)
        else:
            channel_rel = gamma_eff_norm

        channel_rel = channel_rel.clamp(0.0, 1.0)

        # ── squeeze: (B,n,1,1) -> (B,n) if n>1, or (B,1,1,1)->(B,) if n==1 ──
        if n == 1:
            squeeze = lambda t: t.squeeze(-1).squeeze(-1).squeeze(-1)
        else:
            squeeze = lambda t: t.squeeze(-1).squeeze(-1)

        ctx: Dict[str, torch.Tensor] = {
            "gamma_eff_lin":  squeeze(gamma_eff_lin),
            "gamma_eff_db":   squeeze(gamma_eff_db),
            "gamma_eff_norm": squeeze(gamma_eff_norm),
            "channel_rel":    squeeze(channel_rel),
            "h_abs2":         squeeze(h_abs2),
            "noise_var":      squeeze(noise_var_map),
            "num_fading_taps": torch.tensor(n, device=device, dtype=torch.long),
            "channel_type":   ct,
            "fading_equalize": torch.tensor(
                float(self._fading_equalize), device=device, dtype=dtype
            ),
        }
        if posteq_noise_var is not None:
            ctx["posteq_noise_var"] = squeeze(posteq_noise_var)
            ctx["eq_quality"] = squeeze(eq_quality)
        else:
            ctx["rx_noise_var"] = squeeze(noise_var_map)

        self.last_context = ctx
        return ctx

    # ── forward pass ────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ct = self.channel_type
        B = x.shape[0]
        C = x.shape[1]

        if ct == "awgn":
            return self._awgn(x)

        if ct in ("rayleigh", "rician"):
            if self._pending_fading is not None:
                kind, hI, hQ = self._pending_fading
                self._pending_fading = None
                if kind != "complex":
                    raise RuntimeError(
                        f"Pending fading kind mismatch: expected 'complex', got '{kind}'."
                    )
            else:
                if ct == "rayleigh":
                    hI, hQ = self._sample_rayleigh(B, x.device, x.dtype, self.num_fading_taps)
                else:
                    hI, hQ = self._sample_rician(B, x.device, x.dtype)
            return self._apply_complex_fading(x, hI, hQ)

        if self._is_legacy():
            if self._pending_fading is not None:
                kind, h0, h1 = self._pending_fading
                self._pending_fading = None
                if kind != "legacy":
                    raise RuntimeError(
                        f"Pending fading kind mismatch: expected 'legacy', got '{kind}'."
                    )
            else:
                h0, h1 = self._sample_legacy_rayleigh(B, x.device, x.dtype)

            if C % 2 != 0:
                raise ValueError(
                    f"Channel expects even channel dimension (got C={C})."
                )
            C2 = C // 2

            # ★ FIX BUG (legacy): was x[:, :C2] twice; xQ must be x[:, C2:]
            xI, xQ = x[:, :C2], x[:, C2:]

            # ★ FIX BUG (legacy): was yQ = yQ * h1 (yQ undefined); must be xQ
            yI = xI * h0
            yQ = xQ * h1

            sigma = self._sigma(x.device, x.dtype)
            yI = yI + sigma * torch.randn_like(yI)
            yQ = yQ + sigma * torch.randn_like(yQ)

            if self._fading_equalize:
                sign_h0 = torch.where(h0 >= 0, torch.ones_like(h0), -torch.ones_like(h0))
                sign_h1 = torch.where(h1 >= 0, torch.ones_like(h1), -torch.ones_like(h1))
                h0_safe = sign_h0 * h0.abs().clamp_min(self.h_abs_min)
                h1_safe = sign_h1 * h1.abs().clamp_min(self.h_abs_min)
                yI = yI / h0_safe
                yQ = yQ / h1_safe
            return torch.cat([yI, yQ], dim=1)

        raise ValueError(f"Unsupported channel_type='{self.channel_type}'.")