import torch
import torch.nn as nn

# ============================================================
# Torch-based FIS modules (fast, vectorized, no scikit-fuzzy)
# These are designed to support the paper simulations:
# - Importance map I(i,j) from encoder latent feature statistics
# - Spatial gain/power map A(i,j) conditioned on I, SNR, and budget R
# - Rule activation logging (argmax rule per pixel)
# ============================================================

def _clamp01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def _minmax_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Normalize per-sample to [0,1]
    # x: [B,H,W] or [B,1,H,W]
    if x.dim() == 4:
        x_ = x
        b = x_.shape[0]
        x_min = x_.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
        x_max = x_.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
        return (x_ - x_min) / (x_max - x_min + eps)

    elif x.dim() == 3:
        b = x.shape[0]
        x_min = x.view(b, -1).min(dim=1)[0].view(b, 1, 1)
        x_max = x.view(b, -1).max(dim=1)[0].view(b, 1, 1)
        return (x - x_min) / (x_max - x_min + eps)

    else:
        raise ValueError(f"Expected 3D/4D tensor, got shape {tuple(x.shape)}")


def _mean_normalize(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Enforce per-sample mean(A)=1 over spatial dimensions.
    # A: [B,H,W]
    m = A.mean(dim=(1, 2), keepdim=True).clamp_min(eps)
    return A / m


def _mf_low(x: torch.Tensor) -> torch.Tensor:
    # triangular-like low on [0,1], peak at 0, zero at 0.5
    # return _clamp01((0.5 - x) / 0.5)
    return torch.clamp((0.5 - x) / 0.5, 0, 1)

def _mf_high(x: torch.Tensor) -> torch.Tensor:
    # peak at 1, zero at 0.5
    return torch.clamp((x - 0.5) / 0.5, 0, 1)
   # return _clamp01((x - 0.5) / 0.5)


def _mf_med(x: torch.Tensor) -> torch.Tensor:
    # peak at 0.5, zero at 0 and 1
    # return _clamp01(1.0 - torch.abs(x - 0.5) / 0.4)
    return torch.clamp(1.0 - torch.abs(x - 0.5) / 0.5, 0, 1)

def _fuzzy_and(*args: torch.Tensor) -> torch.Tensor:
    out = args[0]
    for a in args[1:]:
        out = torch.minimum(out, a)
    return out


def _fuzzy_or(*args: torch.Tensor) -> torch.Tensor:
    out = args[0]
    for a in args[1:]:
        out = torch.maximum(out, a)
    return out


class FIS_Importance(nn.Module):
    """
    Layer-1 FIS: compute an importance map I(i,j) in [0,1] from encoder latent z.
    Inputs (per spatial location):
      - m: mean(|z|) across channels
      - v: var(z) across channels (texture/complexity proxy)
      - e: local spatial gradient magnitude proxy from m
    Rule base: 7 rules (compact but expressive).
    Output uses zero-order Sugeno with fixed consequents for efficiency.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

        # Consequent values for 7 rules (in [0,1])
        # (VL, L, M, H, VH) mapped into numeric values
        # self.c = torch.tensor([0.9, 0.85, 0.75, 0.65, 0.5, 0.4, 0.6])  # 7 consequents
        self.c = nn.Parameter(torch.tensor([0.9, 0.85, 0.75, 0.65, 0.5, 0.4, 0.6]))
    @staticmethod
    def _edge_from_m(m: torch.Tensor) -> torch.Tensor:
        # m: [B,H,W]
        # edge: average abs gradient (simple, fast)
        # print("DEBUG m shape:", m.shape)s
        dx = torch.abs(m[:, :, 1:] - m[:, :, :-1])
        dy = torch.abs(m[:, 1:, :] - m[:, :-1, :])

        # pad to [B,H,W]
        dx = torch.nn.functional.pad(dx, (0, 1, 0, 0))
        dy = torch.nn.functional.pad(dy, (0, 0, 0, 1))

        e = 0.5 * (dx + dy)
        return e

    def forward(self, z: torch.Tensor, return_rules: bool = False):
        """
        z: [B,C,H,W] (real-valued latent feature map, e.g., I/Q stacked channels)
        returns:
          I: [B,H,W] importance map in [0,1]
          (optional) rule_id_map: [B,H,W] argmax rule index 0..6
          (optional) rule_strengths: [B,7,H,W]
        """

        # feature stats
        m = z.abs().mean(dim=1)                # [B,H,W]
        v = z.var(dim=1, unbiased=False)       # [B,H,W]
        e = self._edge_from_m(m)               # [B,H,W]

        # normalize to [0,1] per sample
        m = _minmax_norm(m, self.eps)
        v = _minmax_norm(v, self.eps)
        e = _minmax_norm(e, self.eps)

        # membership degrees
        mL, mM, mH = _mf_low(m), _mf_med(m), _mf_high(m)
        vL, vM, vH = _mf_low(v), _mf_med(v), _mf_high(v)
        eL, eM, eH = _mf_low(e), _mf_med(e), _mf_high(e)

        # 7 compact rules (min for AND, max for OR):
        # R0: IF m is High AND v is High THEN I is VeryHigh
        r0 = _fuzzy_and(mH, vH)

        # R1: IF m is High AND e is High THEN I is VeryHigh
        r1 = _fuzzy_and(mH, eH)

        # R2: IF m is Med AND (v is High OR e is High) THEN I is High
        r2 = _fuzzy_and(mM, _fuzzy_or(vH, eH))

        # R3: IF m is Med AND v is Med AND e is Med THEN I is Med
        r3 = _fuzzy_and(mM, vM, eM)

        # R4: IF m is Low AND v is Low AND e is Low THEN I is VeryLow
        r4 = _fuzzy_and(mL, vL, eL)

        # R5: IF m is Low AND (v is High OR e is High) THEN I is Low
        r5 = _fuzzy_and(mL, _fuzzy_or(vH, eH))

        # R6: IF m is High AND v is Low AND e is Low THEN I is HighMed
        r6 = _fuzzy_and(mH, vL, eL)

        rules = torch.stack([r0, r1, r2, r3, r4, r5, r6], dim=1)  # [B,7,H,W]
        rules = rules + 0.05
        rules = rules / rules.sum(dim=1, keepdim=True)
        # Sugeno defuzzification with fixed consequents
        c = self.c.to(z.device).view(1, -1, 1, 1)  # [1,7,1,1]
        num = (rules * c).sum(dim=1)               # [B,H,W]

        # den = rules.sum(dim=1).clamp_min(self.eps)
        den = rules.sum(dim=1).clamp_min(1e-6)

        I = (num / den).clamp(0.0, 1.0)
        I = (I - 0.5) * 2
        I = torch.tanh(2 * I)
        I = (I + 1) / 2
        if not return_rules:
            return I

        if return_rules:
            print("\n[DEBUG][FIS_Importance]")
            print("I histogram:", torch.histc(I, bins=10))
            print(f"I stats: min={I.min().item():.4f}, max={I.max().item():.4f}, mean={I.mean().item():.4f}, std={I.std().item():.4f}")
            print(f"m std={m.std().item():.4f}, v std={v.std().item():.4f}, e std={e.std().item():.4f}")

            rule_id = torch.argmax(rules, dim=1)
            rule_counts = torch.bincount(rule_id.view(-1), minlength=7).float()
            rule_counts = rule_counts / rule_counts.sum()

            print("Rule usage (%):", (rule_counts * 100).cpu().numpy())

            return I, rule_id, rules


class FIS_PowerAllocation(nn.Module):
    """
    Layer-2 FIS: compute a spatial gain map A(i,j) from importance I(i,j), SNR, and budget R.
    """

    def __init__(
        self,
        a_min: float = 0.70,
        a_med: float = 1.00,
        a_high: float = 2.0,
        snr_min_db: float = 0.0,
        snr_max_db: float = 13.0,
        delta_min: float = 0.2,
        delta_max: float = 0.6,
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
        self.eps = eps
        self.gamma = nn.Parameter(torch.tensor(2.0)) 
        # self.c = torch.tensor([1.40, 1.20, 1.10, 1.10, 1.00, 0.90])
        # self.c = torch.tensor([[2.5, 2.0, 1.5, 1.0, 0.6, 0.3]])
        self.c = nn.Parameter(torch.tensor([[3.0, 2.5, 2.0, 1.0, 0.7, 0.4, 0.2]]))
    def forward(self, I, snr_db, budget=1.0, return_rules=False):

        s = (float(snr_db) - self.snr_min_db) / (
            self.snr_max_db - self.snr_min_db + self.eps
        )
        s = max(0.0, min(1.0, s))
        s = torch.tensor(s, device=I.device, dtype=I.dtype).clamp(0.0, 1.0)

        S = s.view(1, 1, 1).expand_as(I)

        iL, iM, iH = _mf_low(I), _mf_med(I), _mf_high(I)
        sL, sM, sH = _mf_low(S), _mf_med(S), _mf_high(S)

        r0 = _fuzzy_and(iH, sL)
        r1 = _fuzzy_and(iH, sM)
        r2 = _fuzzy_and(iH, sH)
        r3 = _fuzzy_and(iM, sL)
        r4 = _fuzzy_and(iM, sM)
        r5 = _fuzzy_and(iL, sH)  # low importance + high SNR
        r6 = _fuzzy_and(iL, sL)  # low importance + low SNR

        rules = torch.stack([r0, r1, r2, r3, r4, r5, r6], dim=1)

        c = self.c.to(I.device).view(1, -1, 1, 1)
        num = (rules * c).sum(dim=1)
        den = rules.sum(dim=1).clamp_min(self.eps)

        A_raw = num / den
        # A_raw = (A_raw - 1.0) 
        
        snr_u = float(s)
        delta = self.delta_min + (self.delta_max - self.delta_min) * (1.0 - snr_u)

        A_shrink = 1.0 + delta * (A_raw - 1.0)

        R = float(budget)
        A = 1.0 + R * (A_shrink - 1.0)

        # A = A.clamp(min=self.a_min, max=self.a_high)
        # A = 1.0 + 2.5 * (A - 1.0)
        A = _mean_normalize(A, eps=self.eps)
        
       
        if not return_rules:
            return A

        if return_rules:
            print("\n[DEBUG][FIS_Power]")
            print("A std before norm:", A.std().item())
            print("A histogram:", torch.histc(A, bins=10))
            print("snr_db:", snr_db)
            print("snr_norm:", s)
            print("A std:", A.std().item())
            print("A mean:", A.mean().item())
            print(f"A_raw stats: min={A_raw.min().item():.4f}, max={A_raw.max().item():.4f}, std={A_raw.std().item():.4f}")
            print(f"A stats: min={A.min().item():.4f}, max={A.max().item():.4f}, std={A.std().item():.4f}")
            print(f"delta={delta:.4f}, snr_norm={snr_u:.4f}, budget={budget}")

            rule_id = torch.argmax(rules, dim=1)
            rule_counts = torch.bincount(rule_id.view(-1), minlength=7).float()
            rule_counts = rule_counts / rule_counts.sum()

            print("Rule usage (%):", (rule_counts * 100).cpu().numpy())

            return A, rule_id, rules


class FIS_SpatialPowerController(nn.Module):
    """
    Two-layer controller wrapper + ablation modes.
    """

    def __init__(
        self,
        a_min: float = 0.80,
        a_med: float = 1.00,
        a_high: float = 2.0,
        alpha_linear: float = 0.6,
        snr_min_db: float = 0.0,
        snr_max_db: float = 13.0,
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
        self.eps = eps

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
                (z.shape[0], z.shape[2], z.shape[3]),
                0.5,
                device=z.device,
                dtype=z.dtype,
            )
            if return_info:
                info["I"] = I
        else:
            if return_info:
                I, rid1, rs1 = self.imp(z, return_rules=True)
                info.update({"I": I, "rule1_id": rid1, "rule1_strength": rs1})
            else:
                I = self.imp(z, return_rules=False)

        if mode == "importance_only":
            snr_use = 10.0
        else:
            snr_use = float(snr_db)

        if mode == "linear":
            A_raw = 1.0 + self.alpha_linear * (I - 0.5)
            A_raw = A_raw.clamp(min=self.a_min, max=self.a_high)

            s = (float(snr_use) - self.pow.snr_min_db) / (
                self.pow.snr_max_db - self.pow.snr_min_db + self.eps
            )
            s = max(0.0, min(1.0, s))

            R = float(budget)
            R_eff = R * (0.5 + 0.5 * s)

            A = 1.0 + R_eff * (A_raw - 1.0)
            A = A.clamp(min=self.a_min, max=self.a_high)
            A = _mean_normalize(A, eps=self.eps)

            if not return_info:
                return A

            info["A_raw"] = A_raw
            info["A"] = A
            return A, info

        if return_info:
            A, rid2, rs2 = self.pow(I, snr_use, budget=budget, return_rules=True)
            info.update({"A": A, "rule2_id": rid2, "rule2_strength": rs2})
            return A, info

        A = self.pow(I, snr_use, budget=budget, return_rules=False)
        return A