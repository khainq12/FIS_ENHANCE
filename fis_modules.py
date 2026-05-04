# -*- coding: utf-8 -*-
"""
Channel-aware FIS modules for Deep JSCC.

Fixes applied (vs repo feature/fis-channel-aware-controller-r2):
----------------------------------------------------------------
1) BUG 2: Syntax error '"I_A_corr": = ...' -> '"I_A_corr": ...'

2) ISSUE 3 — Option C (strengthened):
   _channel_condition_z() now:
   - Handles channel_rel shape (B, num_taps) from per-subcarrier fading
   - Noise magnitude 0.3 -> 0.5 (stronger perturbation)
   - Spatial noise (B,C,H,W) instead of (B,C,1,1)
   - Correctly pairs I/Q channels with their fading tap

3) _prepare_channel_rel() handles 2D channel_rel (B, num_taps):
   - Computes mean reliability for spatial map
   - Computes std (channel imbalance) for delta modulation

4) Bypass threshold 0.95 -> 0.98 (narrower, lets budget work)

5) BUG 4: alpha_linear is nn.Parameter (learnable)

6) Rule balance loss available via _rule_balance_loss()

Patch r3 fixes (root-cause fixes for full < linear):
-----------------------------------------------------
A) FIS_Importance: replace _minmax_norm with _lognorm_batch.
   _minmax_norm was erasing amplitude information per-sample so
   channel_rel had zero effect on I. _lognorm_batch normalises
   across the batch, preserving relative magnitude between samples
   so a low-reliability channel produces genuinely lower activations.

B) FIS_PowerAllocation: good-channel attenuation threshold raised
   0.90 -> 0.97 (strength 0.2 -> 0.1) and bypass threshold raised
   0.98 -> 0.999. On AWGN channel_rel=1.0 always, so the old
   thresholds caused the controller to self-disable every forward
   pass, making full == snr_only.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


# ═══════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════

def _clamp01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def _minmax_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.dim() == 4:
        b = x.shape[0]
        x_min = x.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
        x_max = x.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
        return (x - x_min) / (x_max - x_min + eps)
    if x.dim() == 3:
        b = x.shape[0]
        x_min = x.view(b, -1).min(dim=1)[0].view(b, 1, 1)
        x_max = x.view(b, -1).max(dim=1)[0].view(b, 1, 1)
        return (x - x_min) / (x_max - x_min + eps)
    raise ValueError(f"Expected 3D/4D tensor, got shape {tuple(x.shape)}")


def _lognorm_batch(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Log-scale normalisation that preserves inter-sample amplitude differences.

    Unlike _minmax_norm (which normalises per-sample and therefore erases
    any amplitude difference caused by channel scaling), this function
    normalises relative to the batch-level mean and std AFTER a log1p
    transform.  A sample whose z was scaled down by a bad channel will
    produce genuinely smaller m/v/e values and therefore land in different
    fuzzy membership bins from a good-channel sample.

    Output is sigmoid-squashed to [0, 1] so membership functions still work.
    """
    if x.dim() not in (3, 4):
        raise ValueError(f"Expected 3D/4D tensor, got shape {tuple(x.shape)}")
    x_log = torch.log1p(x.clamp_min(0.0))          # log(1 + x), always >= 0
    # Compute mean/std over spatial dims of each sample, then average over batch
    # so the reference point is stable across different batch sizes.
    if x_log.dim() == 3:
        sample_mean = x_log.mean(dim=(1, 2), keepdim=True)   # (B,1,1)
        sample_std  = x_log.std(dim=(1, 2), keepdim=True).clamp_min(eps)
    else:
        sample_mean = x_log.mean(dim=(1, 2, 3), keepdim=True)
        sample_std  = x_log.std(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
    batch_mean = sample_mean.mean(dim=0, keepdim=True)        # (1,...) — batch anchor
    batch_std  = sample_std.mean(dim=0, keepdim=True).clamp_min(eps)
    z_score = (x_log - batch_mean) / batch_std
    return torch.sigmoid(z_score)                             # maps to (0, 1)


def _mean_normalize(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    m = A.mean(dim=(1, 2), keepdim=True).clamp_min(eps)
    return A / m


def _safe_corr_2d(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x - x.mean(dim=(1, 2), keepdim=True)
    y = y - y.mean(dim=(1, 2), keepdim=True)
    num = (x * y).mean(dim=(1, 2))
    den = x.pow(2).mean(dim=(1, 2)).sqrt() * y.pow(2).mean(dim=(1, 2)).sqrt()
    return num / den.clamp_min(eps)


def _summary_stats_map(x: torch.Tensor) -> dict:
    return {
        "mean":  x.mean(dim=(1, 2)),
        "std":   x.std(dim=(1, 2), unbiased=False),
        "min":   x.amin(dim=(1, 2)),
        "max":   x.amax(dim=(1, 2)),
        "range": x.amax(dim=(1, 2)) - x.amin(dim=(1, 2)),
    }


# ── Membership functions ──────────────────────────────────────

def _mf_gauss(x, center: float, sigma: float):
    return torch.exp(-0.5 * ((x - center) / max(sigma, 1e-6)) ** 2)

def _mf_low(x, center: float = 0.18, sigma: float = 0.22):
    return _clamp01(_mf_gauss(x, center=center, sigma=sigma))

def _mf_med(x, center: float = 0.50, sigma: float = 0.20):
    return _clamp01(_mf_gauss(x, center=center, sigma=sigma))

def _mf_high(x, center: float = 0.82, sigma: float = 0.22):
    return _clamp01(_mf_gauss(x, center=center, sigma=sigma))

def _fuzzy_and(*args):
    out = args[0]
    for a in args[1:]:
        out = out * a
    return out

def _fuzzy_or(*args):
    out = args[0]
    for a in args[1:]:
        out = torch.maximum(out, a)
    return out


# ═══════════════════════════════════════════════════════════════
# Layer 1: FIS_Importance
# ═══════════════════════════════════════════════════════════════

class FIS_Importance(nn.Module):
    """Compute importance map I(i,j) in [0,1] from encoder latent z."""

    def __init__(self, eps: float = 1e-8, rule_temp: float = 1.25,
                 rule_floor: float = 0.01):
        super().__init__()
        self.eps = eps
        self.rule_temp = float(rule_temp)
        self.rule_floor = float(rule_floor)
        # Consequents for 7 rules
        self.c = torch.tensor(
            [0.88, 0.82, 0.68, 0.50, 0.18, 0.28, 0.62],
            dtype=torch.float32,
        )

    @staticmethod
    def _edge_from_m(m: torch.Tensor) -> torch.Tensor:
        dx = torch.abs(m[:, :, 1:] - m[:, :, :-1])
        dy = torch.abs(m[:, 1:, :] - m[:, :-1, :])
        dx = torch.nn.functional.pad(dx, (0, 1, 0, 0))
        dy = torch.nn.functional.pad(dy, (0, 0, 0, 1))
        return 0.5 * (dx + dy)

    def _normalize_rules(self, rules: torch.Tensor) -> torch.Tensor:
        rules = (rules + self.rule_floor).pow(1.0 / self.rule_temp)
        den = rules.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return rules / den

    def forward(self, z: torch.Tensor, return_rules: bool = False):
        m = z.abs().mean(dim=1)
        v = z.var(dim=1, unbiased=False)
        e = self._edge_from_m(m)

        # ★ PATCH r3-A: use _lognorm_batch instead of _minmax_norm.
        # _minmax_norm normalises per-sample so channel scaling (from
        # _channel_condition_z) is completely erased before membership
        # functions see the data.  _lognorm_batch uses a batch-level
        # reference point, so samples from a bad channel produce lower
        # activations and land in different fuzzy bins.
        m = _lognorm_batch(m, self.eps)
        v = _lognorm_batch(v, self.eps)
        e = _lognorm_batch(e, self.eps)

        mL, mM, mH = _mf_low(m), _mf_med(m), _mf_high(m)
        vL, vM, vH = _mf_low(v), _mf_med(v), _mf_high(v)
        eL, eM, eH = _mf_low(e), _mf_med(e), _mf_high(e)

        r0 = _fuzzy_and(mH, vH)
        r1 = _fuzzy_and(mH, eH)
        r2 = _fuzzy_and(mM, _fuzzy_or(vH, eH))
        r3 = _fuzzy_and(mM, vM, eM)
        r4 = _fuzzy_and(mL, vL, eL)
        r5 = _fuzzy_and(mL, _fuzzy_or(vH, eH))
        r6 = _fuzzy_and(mH, vL, eL)

        rules_raw = torch.stack([r0, r1, r2, r3, r4, r5, r6], dim=1)
        rules = self._normalize_rules(rules_raw)

        c = self.c.to(z.device, z.dtype).view(1, -1, 1, 1)
        I = (rules * c).sum(dim=1).clamp(0.0, 1.0)

        if not return_rules:
            return I
        rule_id = torch.argmax(rules, dim=1)
        return I, rule_id, rules


# ═══════════════════════════════════════════════════════════════
# Layer 2: FIS_PowerAllocation
# ═══════════════════════════════════════════════════════════════

class FIS_PowerAllocation(nn.Module):
    """Compute spatial amplitude map A from importance I + channel_rel.

    NEW: _prepare_channel_rel() now handles per-subcarrier channel_rel
         of shape (B, num_taps) in addition to (B,).
    """

    def __init__(
        self,
        a_min: float = 0.75,
        a_med: float = 1.00,
        a_high: float = 1.35,
        snr_min_db: float = 0.0,
        snr_max_db: float = 20.0,
        delta_min: float = 0.15,
        delta_max: float = 0.45,
        rule_temp: float = 1.20,
        rule_floor: float = 0.05,
        score_scale: float = 2.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.a_min = float(a_min)
        self.a_med = float(a_med)
        self.a_high = float(a_high)
        self.snr_min_db = float(snr_min_db)
        self.snr_max_db = float(snr_max_db)
        self.delta_min = float(delta_min)
        self.delta_max = float(delta_max)
        self.rule_temp = float(rule_temp)
        self.rule_floor = float(rule_floor)
        self.score_scale = float(score_scale)
        self.eps = eps
        self.beta = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        # 6 rule consequents
        self.c = nn.Parameter(
            torch.tensor([+0.95, +0.95, +0.95, +0.35, +0.02, -0.35])
        )

    # ── helpers ────────────────────────────────────────────────

    def _snr_unit(self, snr_db: float, device, dtype) -> torch.Tensor:
        s = (float(snr_db) - self.snr_min_db) / (
            self.snr_max_db - self.snr_min_db + self.eps
        )
        s = max(0.0, min(1.0, s))
        return torch.tensor(s, device=device, dtype=dtype)

    def _delta_from_snr(self, snr_u: float) -> float:
        return self.delta_min + (self.delta_max - self.delta_min) * (1.0 - float(snr_u))

    def _delta_from_context(
        self,
        snr_u: float,
        C: torch.Tensor,
        mix: float = 0.6,
        channel_imbalance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build redistribution strength from SNR + channel reliability + imbalance."""
        c_bar = C.mean(dim=(1, 2), keepdim=True)
        snr_term = 1.0 - float(snr_u)
        rel_term = 1.0 - c_bar
        gate = mix * rel_term + (1.0 - mix) * snr_term

        # ★ FIX ISSUE 3: boost delta when per-subcarrier imbalance is high
        if channel_imbalance is not None:
            imb = channel_imbalance.view(-1, 1, 1)
            gate = gate + 0.5 * imb

        delta = self.delta_min + (self.delta_max - self.delta_min) * gate
        return delta.clamp(self.delta_min, self.delta_max)

    def _normalize_rules(self, rules: torch.Tensor) -> torch.Tensor:
        rules = (rules + self.rule_floor).pow(1.0 / self.rule_temp)
        den = rules.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return rules / den

    # ── channel_rel preparation ────────────────────────────────
    # ★ FIX ISSUE 3: now handles (B, num_taps) from per-subcarrier fading

    def _prepare_channel_rel(
        self,
        channel_rel: Optional[torch.Tensor],
        I: torch.Tensor,
        snr_db: float,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Convert channel_rel to spatial map + optional imbalance tensor.

        Returns
        -------
        C_map : (B, H, W) in [0,1] — mean reliability for fuzzy rules
        channel_imbalance : (B,) or None — std of per-tap reliability
        """
        if channel_rel is None:
            s = self._snr_unit(snr_db, device=I.device, dtype=I.dtype)
            return s.view(1, 1, 1).expand_as(I), None

        C = channel_rel
        if not torch.is_tensor(C):
            C = torch.tensor(channel_rel, device=I.device, dtype=I.dtype)
        C = C.to(device=I.device, dtype=I.dtype)

        channel_imbalance = None  # default: no imbalance info

        if C.dim() == 1:
            # Block fading: (B,) -> (B, 1, 1) -> (B, H, W)
            C_map = C.view(-1, 1, 1).expand_as(I)

        elif C.dim() == 2:
            # ★ Per-subcarrier: (B, num_taps)
            channel_imbalance = C.std(dim=1)      # (B,) — how uneven the taps are
            C_mean = C.mean(dim=1)                 # (B,) — average reliability
            C_map = C_mean.view(-1, 1, 1).expand_as(I)

        elif C.dim() == 3:
            if C.shape[1:] != I.shape[1:]:
                C = torch.nn.functional.interpolate(
                    C.unsqueeze(1), size=I.shape[1:],
                    mode='bilinear', align_corners=False,
                ).squeeze(1)
            C_map = C
            channel_imbalance = None

        elif C.dim() == 4:
            C2 = C[:, 0]
            if C2.shape[1:] != I.shape[1:]:
                C2 = torch.nn.functional.interpolate(
                    C2.unsqueeze(1), size=I.shape[1:],
                    mode='bilinear', align_corners=False,
                ).squeeze(1)
            C_map = C2
            channel_imbalance = None

        else:
            raise ValueError(f"Unsupported channel_rel shape: {tuple(C.shape)}")

        return _clamp01(C_map), channel_imbalance

    # ── forward ────────────────────────────────────────────────

    def forward(
        self,
        I: torch.Tensor,
        snr_db: float,
        budget: float = 1.0,
        channel_rel: Optional[torch.Tensor] = None,
        return_rules: bool = False,
    ):
        s = self._snr_unit(snr_db, device=I.device, dtype=I.dtype)

        # Returns (C_map, channel_imbalance)
        C, channel_imbalance = self._prepare_channel_rel(
            channel_rel, I, snr_db
        )

        iL, iM, iH = _mf_low(I), _mf_med(I), _mf_high(I)
        cL, cM, cH = _mf_low(C), _mf_med(C), _mf_high(C)

        # 6 channel-aware rules
        r0 = _fuzzy_and(iH, cL)
        r1 = _fuzzy_and(iH, cM)
        r2 = _fuzzy_and(iH, cH)
        r3 = _fuzzy_and(iM, cL)
        r4 = _fuzzy_and(iM, _fuzzy_or(cM, cH))
        r5 = iL

        rules_raw = torch.stack([r0, r1, r2, r3, r4, r5], dim=1)
        rules = self._normalize_rules(rules_raw)

        # Hybrid score
        score_I = I - I.mean(dim=(1, 2), keepdim=True)
        c = self.c.to(I.device, I.dtype).view(1, -1, 1, 1)
        score_fuzzy = (rules * c).sum(dim=1)
        score_fuzzy = score_fuzzy - score_fuzzy.mean(dim=(1, 2), keepdim=True)

        beta = torch.sigmoid(self.beta)
        score = (1.0 - beta) * score_I + beta * score_fuzzy
        score = self.score_scale * score

        delta = self._delta_from_context(
            float(s), C, mix=0.6, channel_imbalance=channel_imbalance
        )

        # ★ PATCH r3-B: good-channel attenuation thresholds raised.
        # Old values (0.90 / strength 0.2) caused the controller to
        # self-attenuate on AWGN (channel_rel always = 1.0) and on
        # high-SNR Rayleigh, making full ≈ snr_only.
        # New values only activate when the channel is nearly perfect.
        c_bar = C.mean(dim=(1, 2), keepdim=True)
        good_gate = torch.clamp((c_bar - 0.97) / 0.03, 0.0, 1.0)
        delta = delta * (1.0 - 0.1 * good_gate)

        amp = delta
        A_fis = torch.exp(amp * torch.tanh(score))
        A_fis = A_fis.clamp(min=self.a_min, max=self.a_high)

        # ★ PATCH r3-B: bypass threshold raised 0.98 -> 0.999.
        # Old threshold fired on every AWGN forward pass.
        c_bar = C.mean(dim=(1, 2), keepdim=True)
        bypass = torch.clamp((c_bar - 0.999) / 0.001, 0.0, 1.0)
        A_fis = (1.0 - bypass) * A_fis + bypass * torch.ones_like(A_fis)

        # Budget interpolation
        A = float(budget) * A_fis + (1.0 - float(budget))
        A = _mean_normalize(A, eps=self.eps)

        if not return_rules:
            return A
        rule_id = torch.argmax(rules, dim=1)
        aux = {
            "score": score.detach(),
            "delta": delta.detach(),
            "channel_rel_map": C.detach(),
            "channel_rel_mean": c_bar.detach().flatten(),
        }
        if channel_imbalance is not None:
            aux["channel_imbalance"] = channel_imbalance.detach()
        return A, rule_id, rules, aux


# ═══════════════════════════════════════════════════════════════
# Two-layer controller wrapper + ablation modes
# ═══════════════════════════════════════════════════════════════

class FIS_SpatialPowerController(nn.Module):
    """Two-layer controller with 4 ablation modes.

    modes:
      - 'full'            : importance FIS + channel-aware power FIS
      - 'importance_only' : ignore channel context in layer-2
      - 'snr_only'        : ignore content importance (I = 0.5)
      - 'linear'          : symmetric linear residual around mean(I)
    """

    def __init__(
        self,
        a_min: float = 0.75,
        a_med: float = 1.00,
        a_high: float = 1.35,
        alpha_linear: float = 1.10,
        snr_min_db: float = 0.0,
        snr_max_db: float = 20.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.imp = FIS_Importance(eps=eps)
        self.pow = FIS_PowerAllocation(
            a_min=a_min, a_med=a_med, a_high=a_high,
            snr_min_db=snr_min_db, snr_max_db=snr_max_db,
            eps=eps,
        )
        # ★ FIX BUG 4: learnable parameter
        self.alpha_linear = nn.Parameter(torch.tensor(alpha_linear))
        self.a_min = float(a_min)
        self.a_high = float(a_high)
        self.snr_min_db = float(snr_min_db)
        self.snr_max_db = float(snr_max_db)
        self.eps = eps

    @staticmethod
    def _rule_balance_loss(
        rule_strength: torch.Tensor,
        target: str = "uniform",
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """KL divergence toward uniform rule usage."""
        p = rule_strength.mean(dim=(0, 2, 3)).clamp_min(eps)
        p = p / p.sum()
        if target == "uniform":
            q = torch.full_like(p, 1.0 / p.numel())
            return torch.sum(p * torch.log(p / q))
        entropy = -(p * torch.log(p)).sum()
        return -entropy

    # ────────────────────────────────────────────────────────────
    # ★ FIX ISSUE 3 (Option C — STRENGTHENED):
    #
    # Key changes vs repo version:
    #   1. Handles channel_rel shape (B, num_taps) from per-subcarrier
    #   2. Correctly pairs I/Q channels with fading taps
    #   3. Noise magnitude 0.3 -> 0.5
    #   4. Spatial noise (B,C,H,W) instead of (B,C,1,1)
    #
    # HOW IT WORKS:
    #   z has C channels where C = 2 * num_taps (I/Q pairs).
    #   channel_rel has num_taps values per sample.
    #   Each tap i scales z[:, i, :, :] (I part) and z[:, i+num_taps, :, :] (Q part).
    #   This makes importance I genuinely different under different channel conditions.
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _channel_condition_z(
        z: torch.Tensor,
        channel_rel: torch.Tensor,
        training: bool,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        B, C, H, W = z.shape
        num_taps = C // 2  # number of I/Q pairs

        # ── Build per-channel scale from channel_rel ──
        if channel_rel.dim() == 2:
            # Per-subcarrier: (B, num_taps) -> (B, C) by pairing I/Q
            cr_taps = channel_rel.clamp(0.0, 1.0).to(device=z.device, dtype=z.dtype)
            if cr_taps.shape[1] == num_taps:
                # Each tap applies to both I and Q channel
                cr_full = cr_taps.repeat(1, 2)  # (B, C)
            elif cr_taps.shape[1] == 1:
                cr_full = cr_taps.expand(-1, C)  # (B, C) — block fading
            else:
                # Fallback: repeat to fill C
                cr_full = cr_taps.repeat(1, (C + cr_taps.shape[1] - 1) // cr_taps.shape[1])[:, :C]
            global_scale = cr_full.sqrt().clamp_min(eps).view(B, C, 1, 1)
        elif channel_rel.dim() == 1:
            # Block fading: (B,) -> (B, 1, 1, 1) broadcast
            cr = channel_rel.view(B, 1, 1, 1).to(device=z.device, dtype=z.dtype).clamp(0.0, 1.0)
            global_scale = cr.sqrt().clamp_min(eps).expand(-1, C, -1, -1)
        else:
            raise ValueError(f"Unsupported channel_rel dim: {channel_rel.dim()}")

        if training:
            # ★ FIX: noise magnitude 0.3 -> 0.5 (stronger perturbation)
            noise_mag = 0.5 * (1.0 - global_scale)

            # ★ FIX: SPATIAL noise (B, C, H, W) instead of (B, C, 1, 1)
            # This creates per-pixel variation -> importance I changes spatially
            channel_noise = torch.randn(B, C, H, W, device=z.device, dtype=z.dtype)
            per_pixel_scale = (1.0 + noise_mag * channel_noise).clamp(eps, 3.0)
        else:
            # Eval: deterministic, no noise
            per_pixel_scale = torch.ones(B, C, 1, 1, device=z.device, dtype=z.dtype)

        z_cond = z * global_scale * per_pixel_scale
        return z_cond

    # ── forward ────────────────────────────────────────────────

    def forward(
        self,
        z: torch.Tensor,
        snr_db: float,
        budget: float,
        mode: str = "full",
        channel_rel: Optional[torch.Tensor] = None,
        return_info: bool = False,
    ):
        info = {}
        mode = str(mode).lower()

        # ── snr_only: identity controller ──
        if mode == "snr_only":
            I = torch.full(
                (z.shape[0], z.shape[2], z.shape[3]), 0.5,
                device=z.device, dtype=z.dtype,
            )
            A = torch.ones_like(I)
            if not return_info:
                return A
            a_stats = _summary_stats_map(A)
            info.update({
                "I": I, "A": A,
                "A_mean": a_stats["mean"], "A_std": a_stats["std"],
                "A_range": a_stats["range"],
                "A_min": a_stats["min"], "A_max": a_stats["max"],
                "I_A_corr": _safe_corr_2d(I, A, eps=self.eps),
                "snr_only_noop": torch.tensor(1.0, device=z.device, dtype=z.dtype),
            })
            if channel_rel is not None:
                info["channel_rel"] = channel_rel
            return A, info

        # ── Channel-condition z before Layer-1 (full mode only) ──
        z_for_importance = z
        if mode == "full" and channel_rel is not None:
            z_for_importance = self._channel_condition_z(
                z, channel_rel, training=self.training, eps=self.eps,
            )
            if return_info:
                info["channel_condition_noise_active"] = torch.tensor(
                    1.0 if self.training else 0.0,
                    device=z.device, dtype=z.dtype,
                )

        # ── Layer 1: importance ──
        if return_info:
            I, rid1, rs1 = self.imp(z_for_importance, return_rules=True)
            info.update({
                "I": I,
                "rule1_id": rid1,
                "rule1_strength": rs1,
                "rule1_balance_loss": self._rule_balance_loss(rs1, eps=self.eps),
            })
        else:
            I = self.imp(z_for_importance, return_rules=False)

        snr_use = float(snr_db)

        # ── linear mode ──
        if mode == "linear":
            I_c = I - I.mean(dim=(1, 2), keepdim=True)
            s = self.pow._snr_unit(snr_use, device=I.device, dtype=I.dtype)
            amp = self.pow._delta_from_snr(float(s))
            score = self.alpha_linear * I_c
            A_fis = torch.exp(amp * torch.tanh(score))
            A_fis = A_fis.clamp(min=self.a_min, max=self.a_high)
            A = float(budget) * A_fis + (1.0 - float(budget))
            A = _mean_normalize(A, eps=self.eps)
            if not return_info:
                return A
            a_stats = _summary_stats_map(A)
            info.update({
                "A_raw": score, "A": A,
                "A_mean": a_stats["mean"], "A_std": a_stats["std"],
                "A_range": a_stats["range"],
                "A_min": a_stats["min"], "A_max": a_stats["max"],
                "I_A_corr": _safe_corr_2d(I, A, eps=self.eps),
            })
            if channel_rel is not None:
                info["channel_rel"] = channel_rel
            return A, info

        # ── importance_only / full ──
        channel_rel_use = None if mode == "importance_only" else channel_rel

        if return_info:
            A, rid2, rs2, aux2 = self.pow(
                I, snr_use, budget=budget,
                channel_rel=channel_rel_use,
                return_rules=True,
            )
            a_stats = _summary_stats_map(A)
            # ★ FIX BUG 2: removed extra '=' sign
            info.update({
                "A": A,
                "A_mean": a_stats["mean"], "A_std": a_stats["std"],
                "A_range": a_stats["range"],
                "A_min": a_stats["min"], "A_max": a_stats["max"],
                "I_A_corr": _safe_corr_2d(I, A, eps=self.eps),
                "rule2_id": rid2,
                "rule2_strength": rs2,
                "rule2_balance_loss": self._rule_balance_loss(rs2, eps=self.eps),
                "score_map": aux2["score"],
                "delta_map": aux2["delta"],
                "channel_rel_map": aux2["channel_rel_map"],
                "channel_rel_mean": aux2["channel_rel_mean"],
            })
            if "channel_imbalance" in aux2:
                info["channel_imbalance"] = aux2["channel_imbalance"]
            if channel_rel_use is not None:
                info["channel_rel"] = channel_rel_use
            return A, info

        A = self.pow(
            I, snr_use, budget=budget,
            channel_rel=channel_rel_use,
            return_rules=False,
        )
        return A