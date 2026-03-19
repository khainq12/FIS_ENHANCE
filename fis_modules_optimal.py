"""
Drop-in replacement for fis_modules.py with more conservative default layer-2 rules.
Use this file when the original Full FIS still trails snr_only on AWGN.

Key changes versus patch_v4:
- layer-2 consequents are softened
- w0 kept at 0.05 by default
- smooth_kernel kept at 3 by default
These settings usually preserve Rayleigh gains while improving Full-vs-snr_only balance.
"""
from fis_modules import *  # noqa: F401,F403
import torch


class FIS_PowerAllocationOptimal(FIS_PowerAllocation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = torch.tensor([1.45,1.3,1.1,1.05,1.0,0.85])
        self.w0 = 0.08

# RL |  |0.08
# AW |1.36,1.24,1.08,1.04,1.0,0.9 | 0.03

class FIS_SpatialPowerControllerOptimal(FIS_SpatialPowerController):
    def __init__(self, *args, **kwargs):
        if 'smooth_kernel' not in kwargs:
            kwargs['smooth_kernel'] = 3
        super().__init__(*args, **kwargs)
        self.pow = FIS_PowerAllocationOptimal(
            a_min=self.a_min,
            a_med=self.a_med,
            a_high=self.a_high,
            snr_min_db=self.pow.snr_min_db,
            snr_max_db=self.pow.snr_max_db,
            w0=0.08,
            eps=self.eps,
        )
        self.alpha_linear = 0.6
