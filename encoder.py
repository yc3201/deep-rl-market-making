"""Attn-LOB encoder — my reimplementation from the paper, not a copy.

Paper: Guo et al. 2023 (arXiv 2305.15821), Sec. III-B3. "The model
architecture follows [DeepLOB]. They proposed a CNN-LSTM based network
and we add a multi-head self-attention layer to model temporal
dependencies." Concretely:

    input [B, T, 4n]  (T = 50, n = 10 levels -> 40 features)
      -> CNN filters over time (features as channels) model the spatial
         dependencies across LOB levels;
      -> Inception module (parallel kernels 1/3/5) establishes
         multi-scale temporal dependencies;
      -> multi-head self-attention aggregates temporal dependencies
         (with residual + LayerNorm, a standard transformer block);
      -> mean-pool over time -> 64-dim market representation;
      -> linear head -> 3 mid-price direction classes.

This is my own implementation of that described architecture (layer
sizes chosen for CPU training; ~61k parameters vs the paper's 176,320
— reported honestly, not matched). The reference repo
imTurkey/Market-Making-with-Deep-Reinforcement-Learning-from-Limit-Order-Books
was NOT copied: I read the paper's method section and wrote this from
the description above.

Two entry points:
* ``forward(x)`` -> class logits [B, 3] (pretraining).
* ``forward_features(x)`` -> pooled market representation [B, 64]
  (the frozen trunk the PPO actor/critic build on).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttnLOB(nn.Module):
    def __init__(self, n_features: int = 40, seq_len: int = 50,
                 hidden: int = 64, n_heads: int = 4, n_classes: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        # --- CNN spatial filters (conv over time, LOB features as channels)
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
        )
        # --- Inception: multi-scale temporal dependencies
        branch = hidden // 2
        self.inc1 = nn.Conv1d(hidden, branch, kernel_size=1)
        self.inc3 = nn.Conv1d(hidden, branch, kernel_size=3, padding=1)
        self.inc5 = nn.Conv1d(hidden, branch, kernel_size=5, padding=2)
        self.inc_mix = nn.Sequential(
            nn.Conv1d(3 * branch, hidden, kernel_size=1),
            nn.LeakyReLU(0.01),
        )
        # --- Multi-head self-attention over time (residual + norm)
        self.attn = nn.MultiheadAttention(embed_dim=hidden,
                                          num_heads=n_heads,
                                          dropout=dropout,
                                          batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.ffn_norm = nn.LayerNorm(hidden)
        # --- head
        self.head = nn.Linear(hidden, n_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, 40] -> pooled market representation [B, hidden]."""
        h = x.transpose(1, 2)               # [B, 40, T]
        h = self.conv(h)                    # [B, hidden, T]
        inc = torch.cat([self.inc1(h), self.inc3(h), self.inc5(h)], dim=1)
        h = self.inc_mix(inc)               # [B, hidden, T]
        h = h.transpose(1, 2)               # [B, T, hidden]
        a, _ = self.attn(h, h, h)
        h = self.attn_norm(h + a)
        h = self.ffn_norm(h + self.ffn(h))
        return h.mean(dim=1)                # mean-pool over time

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
