import torch
import torch.nn as nn

# ============================================================
# Channel-aware FIS modules for Deep JSCC
#
# Main change relative to the current public repo:
# - Layer-2 no longer relies only on global SNR and latent importance.
# - It can now take one extra channel descriptor: channel_rel in [0,1].
#   In the recommended setup, channel_rel = normalized instantaneous
#   effective channel reliability derived from |h|^2 / sigma_n^2.
#
# Design intent:
# - Keep the controller interpretable.
# - Keep the encoder/decoder backbone unchanged.
# - Improve full-FIS behavior on fading channels without turning the paper
#   into a new end-to-end architecture story.
# ============================================================


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


def _mean_normalize(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Divide A by its spatial mean so mean(A)=1."""
    m = A.mean(dim=(1, 2), keepdim=True).clamp_min(eps)
    return A / m


# ----------------------------
# Membership functions
# ----------------------------
def _mf_gauss(x: torch.Tensor, center: float, sigma: float) -> torch.Tensor:
    return torch.exp(-0.5 * ((x - center) / max(sigma, 1e-6)) ** 2)


def _mf_low(x: torch.Tensor, center: float = 0.18, sigma: float = 0.22) -> torch.Tensor:
    return _clamp01(_mf_gauss(x, center=center, sigma=sigma))


def _mf_med(x: torch.Tensor, center: float = 0.50, sigma: float = 0.20) -> torch.Tensor:
    return _clamp01(_mf_gauss(x, center=center, sigma=sigma))


def _mf_high(x: torch.Tensor, center: float = 0.82, sigma: float = 0.22) -> torch.Tensor:
    return _clamp01(_mf_gauss(x, center=center, sigma=sigma))


def _fuzzy_and(*args: torch.Tensor) -> torch.Tensor:
    out = args[0]
    for a in args[1:]:
        out = out * a
    return out


def _fuzzy_or(*args: torch.Tensor) -> torch.Tensor:
    out = args[0]
    for a in args[1:]:
        out = torch.maximum(out, a)
    return out


class FIS_Importance(nn.Module):
    """
    Layer-1 FIS: compute importance map I(i,j) in [0,1] from encoder latent z.
    """

    def __init__(self, eps: float = 1e-8, rule_temp: float = 1.25, rule_floor: float = 0.01):
        super().__init__()
        self.eps = eps
        self.rule_temp = float(rule_temp)
        self.rule_floor = float(rule_floor)
        self.c = torch.tensor([0.88, 0.82, 0.68, 0.50, 0.18, 0.28, 0.62], dtype=torch.float32)

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

        m = _minmax_norm(m, self.eps)
        v = _minmax_norm(v, self.eps)
        e = _minmax_norm(e, self.eps)

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


class FIS_PowerAllocation(nn.Module):
    """
    Layer-2 FIS: compute a spatial amplitude map A(i,j) from:
      - latent importance I(i,j)
      - nominal SNR (used for global redistribution strength)
      - channel_rel in [0,1] (used for fading-aware rule selection)
      - budget R (used through interpolation between uniform and A_fis)

    Recommended channel_rel:
      channel_rel = normalized instantaneous effective SNR,
      e.g. gamma_eff_norm from |h|^2 / sigma_n^2.

    Design principle:
    - SNR controls how strong redistribution should be globally.
    - channel_rel tells the controller how reliable the current channel block is.
    - I tells the controller which spatial locations deserve protection.
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

        # Consequents for the 6 channel-aware rules below.
        # Positive -> allocate more, negative -> allocate less.
        # Rule order:
        # 0: high-I & low-reliability
        # 1: high-I & medium-reliability
        # 2: high-I & high-reliability
        # 3: medium-I & low-reliability
        # 4: medium-I & medium/high reliability
        # 5: low-I (all reliability levels merged)
        self.c = torch.tensor([+0.95, +0.60, +0.18, +0.35, +0.02, -0.35], dtype=torch.float32)

    def _snr_unit(self, snr_db: float, device, dtype) -> torch.Tensor:
        s = (float(snr_db) - self.snr_min_db) / (self.snr_max_db - self.snr_min_db + self.eps)
        s = max(0.0, min(1.0, s))
        return torch.tensor(s, device=device, dtype=dtype)

    def _delta_from_snr(self, snr_u: float) -> float:
        """Low SNR -> stronger redistribution; high SNR -> softer."""
        return self.delta_min + (self.delta_max - self.delta_min) * (1.0 - float(snr_u))

    def _delta_from_context(self, snr_u: float, C: torch.Tensor, mix: float = 0.6) -> torch.Tensor:
        """
        Build a per-sample redistribution strength from both nominal SNR and
        instantaneous channel reliability.

        - snr_u in [0,1]: operating-point indicator
        - C in [0,1], shape [B,H,W]: effective channel reliability map or scalar

        Lower reliability should increase redistribution strength.
        """
        c_bar = C.mean(dim=(1, 2), keepdim=True)
        snr_term = 1.0 - float(snr_u)
        rel_term = 1.0 - c_bar
        gate = mix * rel_term + (1.0 - mix) * snr_term
        delta = self.delta_min + (self.delta_max - self.delta_min) * gate
        return delta.clamp(self.delta_min, self.delta_max)

    def _normalize_rules(self, rules: torch.Tensor) -> torch.Tensor:
        rules = (rules + self.rule_floor).pow(1.0 / self.rule_temp)
        den = rules.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return rules / den

    def _prepare_channel_rel(self, channel_rel: torch.Tensor | None, I: torch.Tensor, snr_db: float) -> torch.Tensor:
        """
        Convert channel_rel to a [B,H,W] tensor in [0,1].
        If not provided, fall back to nominal-SNR-based reliability.
        """
        if channel_rel is None:
            s = self._snr_unit(snr_db, device=I.device, dtype=I.dtype)
            return s.view(1, 1, 1).expand_as(I)

        C = channel_rel
        if not torch.is_tensor(C):
            C = torch.tensor(channel_rel, device=I.device, dtype=I.dtype)
        C = C.to(device=I.device, dtype=I.dtype)

        if C.dim() == 1:
            C = C.view(-1, 1, 1).expand_as(I)
        elif C.dim() == 2:
            C = C.unsqueeze(-1).expand_as(I)
        elif C.dim() == 3:
            if C.shape[1:] != I.shape[1:]:
                C = torch.nn.functional.interpolate(
                    C.unsqueeze(1), size=I.shape[1:], mode='bilinear', align_corners=False
                ).squeeze(1)
        elif C.dim() == 4:
            C = C[:, 0]
            if C.shape[1:] != I.shape[1:]:
                C = torch.nn.functional.interpolate(
                    C.unsqueeze(1), size=I.shape[1:], mode='bilinear', align_corners=False
                ).squeeze(1)
        else:
            raise ValueError(f"Unsupported channel_rel shape: {tuple(C.shape)}")
        return _clamp01(C)

    def forward(
        self,
        I: torch.Tensor,
        snr_db: float,
        budget: float = 1.0,
        channel_rel: torch.Tensor | None = None,
        return_rules: bool = False,
    ):
        s = self._snr_unit(snr_db, device=I.device, dtype=I.dtype)
        C = self._prepare_channel_rel(channel_rel, I, snr_db)

        iL, iM, iH = _mf_low(I), _mf_med(I), _mf_high(I)
        cL, cM, cH = _mf_low(C), _mf_med(C), _mf_high(C)

        # 6 channel-aware rules.
        r0 = _fuzzy_and(iH, cL)
        r1 = _fuzzy_and(iH, cM)
        r2 = _fuzzy_and(iH, cH)
        r3 = _fuzzy_and(iM, cL)
        r4 = _fuzzy_and(iM, _fuzzy_or(cM, cH))
        r5 = iL

        rules_raw = torch.stack([r0, r1, r2, r3, r4, r5], dim=1)
        rules = self._normalize_rules(rules_raw)

        c = self.c.to(I.device, I.dtype).view(1, -1, 1, 1)
        score = (rules * c).sum(dim=1)
        score = score - score.mean(dim=(1, 2), keepdim=True)
        score = self.score_scale * score

        delta = self._delta_from_context(float(s), C, mix=0.6)

        # Good-channel attenuation:
        # when the effective channel reliability is already very high, strong
        # redistribution is often unnecessary and can hurt AWGN performance.
        c_bar = C.mean(dim=(1, 2), keepdim=True)
        good_gate = torch.clamp((c_bar - 0.85) / 0.15, 0.0, 1.0)
        delta = delta * (1.0 - 0.8 * good_gate)

        amp = delta

        A_fis = torch.exp(amp * torch.tanh(score))
        A_fis = A_fis.clamp(min=self.a_min, max=self.a_high)

        # Budget interpolation: R=0 -> uniform, R=1 -> full redistribution.
        A = float(budget) * A_fis + (1.0 - float(budget))
        A = _mean_normalize(A, eps=self.eps)

        if not return_rules:
            return A
        rule_id = torch.argmax(rules, dim=1)
        return A, rule_id, rules


class FIS_SpatialPowerController(nn.Module):
    """
    Two-layer controller wrapper + ablation modes.

    modes:
      - "full": importance FIS + channel-aware power FIS
      - "importance_only": ignore channel context in layer-2
      - "snr_only": ignore content importance (I = 0.5 constant)
      - "linear": symmetric linear residual baseline around mean(I)
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
            a_min=a_min,
            a_med=a_med,
            a_high=a_high,
            snr_min_db=snr_min_db,
            snr_max_db=snr_max_db,
            eps=eps,
        )
        self.alpha_linear = float(alpha_linear)
        self.a_min = float(a_min)
        self.a_high = float(a_high)
        self.snr_min_db = float(snr_min_db)
        self.snr_max_db = float(snr_max_db)
        self.eps = eps

    @staticmethod
    def _rule_balance_loss(rule_strength: torch.Tensor, target: str = "uniform", eps: float = 1e-8) -> torch.Tensor:
        p = rule_strength.mean(dim=(0, 2, 3)).clamp_min(eps)
        p = p / p.sum()
        if target == "uniform":
            q = torch.full_like(p, 1.0 / p.numel())
            return torch.sum(p * torch.log(p / q))
        entropy = -(p * torch.log(p)).sum()
        return -entropy

    def forward(
        self,
        z: torch.Tensor,
        snr_db: float,
        budget: float,
        mode: str = "full",
        channel_rel: torch.Tensor | None = None,
        return_info: bool = False,
    ):
        info = {}
        mode = str(mode).lower()

        if mode == "snr_only":
            # IMPORTANT:
            # Under the current architecture, the controller can only build a
            # spatial power map A(i,j). However, the available channel context
            # is a block-level reliability scalar per sample. After expansion,
            # that scalar is spatially constant, so any "SNR-only" spatial map
            # would collapse to A = 1 after mean normalization.
            #
            # Therefore, snr_only is intentionally kept as an identity spatial
            # controller. It is useful only as a diagnostic ablation / no-control
            # reference and must not be interpreted as true SNR-aware allocation.
            I = torch.full((z.shape[0], z.shape[2], z.shape[3]), 0.5, device=z.device, dtype=z.dtype)
            A = torch.ones((z.shape[0], z.shape[2], z.shape[3]), device=z.device, dtype=z.dtype)
            if not return_info:
                return A
            info["I"] = I
            info["A"] = A
            info["snr_only_noop"] = torch.tensor(1.0, device=z.device, dtype=z.dtype)
            if channel_rel is not None:
                info["channel_rel"] = channel_rel
            return A, info
        else:
            if return_info:
                I, rid1, rs1 = self.imp(z, return_rules=True)
                info.update({
                    "I": I,
                    "rule1_id": rid1,
                    "rule1_strength": rs1,
                    "rule1_balance_loss": self._rule_balance_loss(rs1, eps=self.eps),
                })
            else:
                I = self.imp(z, return_rules=False)

        snr_use = float(snr_db)

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
            info["A_raw"] = score
            info["A"] = A
            if channel_rel is not None:
                info["channel_rel"] = channel_rel
            return A, info

        # importance_only explicitly ignores the extra channel descriptor.
        channel_rel_use = None if mode == "importance_only" else channel_rel

        if return_info:
            A, rid2, rs2 = self.pow(
                I,
                snr_use,
                budget=budget,
                channel_rel=channel_rel_use,
                return_rules=True,
            )
            info.update({
                "A": A,
                "rule2_id": rid2,
                "rule2_strength": rs2,
                "rule2_balance_loss": self._rule_balance_loss(rs2, eps=self.eps),
            })
            if channel_rel_use is not None:
                info["channel_rel"] = channel_rel_use
            return A, info

        A = self.pow(I, snr_use, budget=budget, channel_rel=channel_rel_use, return_rules=False)
        return A
