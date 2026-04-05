import torch
import torch.nn as nn

# ============================================================
# Patched Torch-based FIS modules
#
# DESIGN NOTE (amplitude vs power map):
#   The output A of FIS_PowerAllocation is an *amplitude* scaling map.
#   It is applied as:  z_gated = z * A   (in model.py, after FIX-1)
#   This means A=1.35 gives a 35% amplitude boost to that spatial location.
#   The consequents in FIS_PowerAllocation.c are calibrated for this convention.
#
#   Do NOT use sqrt(A) in model.py — that was the old bug that halved the gain.
#
# FIX-2 (fis_modules.py): a_high default aligned to 1.35 throughout this file.
#   Previously model.py used a_high=2.0 as its default while this file used 1.35,
#   causing inconsistency. The canonical value is 1.35 here. Change both files
#   together if a wider range is needed.
#
# Changes relative to original:
#   - a_high default in FIS_PowerAllocation: 1.35 (was 1.35, unchanged here)
#   - a_high default in FIS_SpatialPowerController: 1.35 (was 1.35, unchanged here)
#   - snr_max_db default in FIS_SpatialPowerController: 20.0 (was 20.0, unchanged)
#     NOTE: model.py previously overrode this with 13.0 — that bug is fixed in model.py.
#   - Added docstring clarifying amplitude convention throughout.
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
    """Divide A by its spatial mean so mean(A) = 1. Preserves redistribution shape."""
    m = A.mean(dim=(1, 2), keepdim=True).clamp_min(eps)
    return A / m


# ----------------------------
# Smooth membership functions
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
    Uses 7 zero-order Sugeno rules with Gaussian membership functions and
    normalized (softened) rule weighting to reduce single-rule dominance.
    """

    def __init__(self, eps: float = 1e-8, rule_temp: float = 1.25, rule_floor: float = 0.01):
        super().__init__()
        self.eps = eps
        self.rule_temp = float(rule_temp)
        self.rule_floor = float(rule_floor)
        # Consequents: [very_high, very_high, high, medium, very_low, low, mod_high]
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

        # 7 rules — see paper Section IV-A
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
        # if not self.training:
        #     print("[DEBUG][I]",
        #         "min:", I.min().item(),
        #         "max:", I.max().item(),
        #         "mean:", I.mean().item(),
        #         "std:", I.std().item())
        if not return_rules:
            return I
        rule_id = torch.argmax(rules, dim=1)
        return I, rule_id, rules


class FIS_PowerAllocation(nn.Module):
    """
    Layer-2 FIS: compute a spatial *amplitude* map A(i,j) from importance I(i,j),
    SNR, and budget R.

    Output convention: A is an amplitude multiplier.
        z_gated = z * A   →   power at (i,j) scales as A^2.
    The consequents self.c are signed deviations; the final A is computed via
    exp(amp * tanh(score)) and then mean-normalized to preserve average power.

    FIX-2 note: a_high default is 1.35, consistent with FIS_SpatialPowerController.
    FIX-3 note: snr_max_db default is 20.0; must match the training script snr_max.
                FIS_SpatialPowerController passes this down, so changing it there
                is sufficient — no need to edit this class directly.
    """

    def __init__(
        self,
        a_min: float = 0.5,
        a_med: float = 1.00,
        a_high: float = 2,      # FIX-2: canonical value, do not override to 2.0
        snr_min_db: float = 0.0,
        snr_max_db: float = 20.0,  # FIX-3: must match training snr_max (default 20)
        delta_min: float = 0.15,    # 0.1
        delta_max: float = 0.45,    # 0.25
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
        # Signed consequents. Positive → allocate more, negative → less.
        # Calibrated for amplitude-map convention (applied directly, no sqrt in model.py).
        # self.c = torch.tensor([+0.90, +0.55, +0.20, +0.25, 0.00, -0.65], dtype=torch.float32)
        self.c = torch.tensor([+0.80, +0.55, +0.45, +0.35, +0.05, -0.40], dtype=torch.float32)
    def _snr_unit(self, snr_db: float, device, dtype) -> torch.Tensor:
        s = (float(snr_db) - self.snr_min_db) / (self.snr_max_db - self.snr_min_db + self.eps)
        s = max(0.0, min(1.0, s))
        return torch.tensor(s, device=device, dtype=dtype)

    def _delta_from_snr(self, snr_u: float) -> float:
        """Low SNR → stronger redistribution; high SNR → softer."""
        return self.delta_min + (self.delta_max - self.delta_min) * (1.0 - float(snr_u))

    def _normalize_rules(self, rules: torch.Tensor) -> torch.Tensor:
        rules = (rules + self.rule_floor).pow(1.0 / self.rule_temp)
        den = rules.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return rules / den

    def forward(self, I: torch.Tensor, snr_db: float, budget: float = 1.0,
                return_rules: bool = False):
        s = self._snr_unit(snr_db, device=I.device, dtype=I.dtype)
        S = s.view(1, 1, 1).expand_as(I)

        iL, iM, iH = _mf_low(I), _mf_med(I), _mf_high(I)
        sL, sM, sH = _mf_low(S), _mf_med(S), _mf_high(S)

        # 6 rules — see paper Section IV-B
        r0 = _fuzzy_and(iH, sL)
        r1 = _fuzzy_and(iH, sM)
        r2 = _fuzzy_and(iH, sH)
        r3 = _fuzzy_and(iM, sL)
        r4 = _fuzzy_and(iM, _fuzzy_or(sL, sM))
        r5 = _fuzzy_and(iL, _fuzzy_or(sL, sM))

        rules_raw = torch.stack([r0, r1, r2, r3, r4, r5], dim=1)
        rules = self._normalize_rules(rules_raw)
        # if not self.training:
        #     r_mean = rules.mean(dim=(0,2,3))
        #     print("[DEBUG][Rules]", r_mean.detach().cpu().numpy())
        c = self.c.to(I.device, I.dtype).view(1, -1, 1, 1)
        score = (rules * c).sum(dim=1)

        # Remove global bias: controller performs redistribution, not uniform uplift.
        score = score - score.mean(dim=(1, 2), keepdim=True)
        score = self.score_scale * score

        # delta = self._delta_from_snr(float(s))
        # amp = float(budget) * delta

        delta = self._delta_from_snr(float(s))
        amp = delta   # amp KHÔNG nhân budget — budget áp dụng qua interpolation

        A_fis = torch.exp(amp * torch.tanh(score))
        A_fis = A_fis.clamp(min=self.a_min, max=self.a_high)

        # Budget = interpolation giữa uniform(1.0) và FIS(A_fis)
        # R=0 → A=1 (không redistribution)  
        # R=1 → A=A_fis (đầy đủ redistribution)
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
        "full"            — importance FIS + power FIS (I + SNR + budget)
        "importance_only" — ignore SNR in layer-2 (fixed at 10 dB)
        "snr_only"        — ignore content importance (I = 0.5 constant)
        "linear"          — symmetric linear residual baseline around mean(I)

    FIX-2: a_high default = 1.35 (was 1.35 here, but model.py was passing 2.0).
           The fix is in model.py; this class is already correct.
    FIX-3: snr_max_db default = 20.0. This value is passed through to
           FIS_PowerAllocation so the SNR normalizer covers the full training range.
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
    def _rule_balance_loss(rule_strength: torch.Tensor, target: str = "uniform",
                           eps: float = 1e-8) -> torch.Tensor:
        """Optional auxiliary loss to encourage balanced rule usage."""
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
        return_info: bool = False,
    ):
        info = {}
        mode = str(mode).lower()

        if mode == "snr_only":
            I = torch.full(
                (z.shape[0], z.shape[2], z.shape[3]), 0.5,
                device=z.device, dtype=z.dtype,
            )
            if return_info:
                info["I"] = I
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

        snr_use = 10.0 if mode == "importance_only" else float(snr_db)

        if mode == "linear":
            I_c = I - I.mean(dim=(1, 2), keepdim=True)
            s = self.pow._snr_unit(snr_use, device=I.device, dtype=I.dtype)
            amp = self.pow._delta_from_snr(float(s))
            score = self.alpha_linear * I_c
            A_fis = torch.exp(amp * torch.tanh(score))
            A_fis = A_fis.clamp(min=self.a_min, max=self.a_high)

            # Budget interpolation (giống PowerAllocation)
            A = float(budget) * A_fis + (1.0 - float(budget))
            A = _mean_normalize(A, eps=self.eps)
            if not return_info:
                return A
            info["A_raw"] = score
            info["A"] = A
            return A, info

        if return_info:
            A, rid2, rs2 = self.pow(I, snr_use, budget=budget, return_rules=True)
            info.update({
                "A": A,
                "rule2_id": rid2,
                "rule2_strength": rs2,
                "rule2_balance_loss": self._rule_balance_loss(rs2, eps=self.eps),
            })
            return A, info

        A = self.pow(I, snr_use, budget=budget, return_rules=False)
        return A
