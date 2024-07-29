import numpy as np
import torch
import torch.nn as nn

class TimeEncode(nn.Module):
    def __init__(self, expand_dim, factor=5):
        super().__init__()
        self.time_dim = expand_dim
        assert self.time_dim % 2 == 0
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim // 2))).float())

    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)
        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        ts_cos, ts_sin = ts * self.basis_freq.view(1, 1, -1), ts * self.basis_freq.view(1, 1, -1)
        cos_basis, sin_basis = torch.cos(ts_cos).unsqueeze(-2), torch.sin(ts_cos).unsqueeze(-2)    # [N, M*m, 1, time_dim], [N, M*m, 1, time_dim]
        harmonic = torch.cat([cos_basis, sin_basis], dim=-2).flatten(-2, -1) * (self.time_dim ** -0.5)
        return harmonic


