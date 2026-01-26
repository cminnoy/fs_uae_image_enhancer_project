import torch
import torch.nn as nn
import torch.nn.functional as F

def sinkhorn_projection(logits, num_iters=5):
    M = torch.exp(logits)
    for _ in range(num_iters):
        M = M / (M.sum(dim=1, keepdim=True) + 1e-8)
        M = M / (M.sum(dim=0, keepdim=True) + 1e-8)
    return M

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinkhorn_projection(logits, num_iters=5):
    M = torch.exp(logits)
    for _ in range(num_iters):
        M = M / (M.sum(dim=1, keepdim=True) + 1e-8)
        M = M / (M.sum(dim=0, keepdim=True) + 1e-8)
    return M


class APReLU(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 4)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels * 2)

    def forward(self, x):
        N, C, H, W = x.shape

        s = self.gap(x).view(N, C)
        s = F.relu(self.fc1(s))
        slopes = torch.sigmoid(self.fc2(s))

        alpha = slopes[:, :C].view(N, C, 1, 1)
        beta = slopes[:, C:].view(N, C, 1, 1)

        pos = F.relu(x)
        neg = (x - x.abs()) * 0.5

        return alpha * pos + beta * neg


class MHCBasicBlock(nn.Module):
    def __init__(self, channels=64, n_streams=5):
        super().__init__()
        self.n = n_streams
        self.channels = channels

        # CNN transform using APReLU
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            APReLU(channels)
        )

        # aggregation weights (n → 1)
        self.agg_logits = nn.Parameter(torch.zeros(self.n))

        # expansion weights (1 → n)
        self.exp_logits = nn.Parameter(torch.zeros(self.n))

        # feature mixing matrix (n × n)
        self.mix_logits = nn.Parameter(torch.zeros(self.n, self.n))

    def forward(self, x):
        B, n, C, H, W = x.shape

        # 1. Sinkhorn‑projected mixing matrix
        mix = sinkhorn_projection(self.mix_logits)

        # 2. Aggregation to single feature map
        agg_w = torch.sigmoid(self.agg_logits)
        agg_w = agg_w / agg_w.sum()
        agg = (x * agg_w.view(1, n, 1, 1, 1)).sum(dim=1)

        # 3. CNN transform with APReLU
        h = self.conv(agg)

        # 4. Expansion to n streams
        exp_w = 2.0 * torch.sigmoid(self.exp_logits)
        h_expanded = h.unsqueeze(1) * exp_w.view(1, n, 1, 1, 1)

        # 5. Mix original x across streams
        x_flat = x.view(B, n, -1)
        x_mixed = torch.einsum("ij, bjd -> bid", mix, x_flat)
        x_mixed = x_mixed.view(B, n, C, H, W)

        # 6. Residual output
        return x_mixed + h_expanded

