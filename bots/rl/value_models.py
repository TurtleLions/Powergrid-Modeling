"""Value networks for search that see every seat at once.

A position is scored from every seat's ego-centric view ([seats, obs]); the
model predicts each seat's chance of winning, as a softmax over seats
("joint" loss: the shares sum to 1) or per seat ("mse"). Architectures:

  mlp    a shared per-seat MLP
  attn   a shared per-seat MLP, then self-attention across seats, so each
         seat's estimate can use the others' views (e.g. their build costs)

Offline experiments (runs/varch, 62k games, held-out games): with TD(lambda)
targets these beat runs/value1/big_aux05.pt (per-seat MLP on final results)
on held-out cross-entropy against the real outcome. Training: bots/rl/value_td.py.

load_value(path) returns a callable for search: obs [seats, obs] -> win
shares [seats] (numpy, summing to 1), for these models and for plain
PolicyValueNet checkpoints (whose per-seat values are normalised, as search
always did).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SeatValueNet(nn.Module):
    def __init__(self, obs, arch="mlp", hidden=512, depth=3, layers=2, loss="joint", d_model=256):
        super().__init__()
        enc, d = [], obs
        for _ in range(depth):
            enc += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        self.enc = nn.Sequential(*enc)
        self.arch, self.loss = arch, loss
        if arch == "attn":
            self.proj = nn.Linear(hidden, d_model)
            layer = nn.TransformerEncoderLayer(d_model, 4, 2 * d_model, dropout=0.0, batch_first=True)
            self.mix = nn.TransformerEncoder(layer, layers)
            d = d_model
        self.value = nn.Linear(d, 1)
        self.aux = nn.Linear(d, 1)               # final margin, a training aid only
        self.cfg = dict(obs=obs, arch=arch, hidden=hidden, depth=depth, layers=layers, loss=loss)

    def forward(self, x):                         # x [B, seats, obs]
        h = self.enc(x)
        if self.arch == "attn":
            h = self.mix(self.proj(h))
        return self.value(h).squeeze(-1), self.aux(h).squeeze(-1)

    def win_prob(self, x):
        v, _ = self(x)
        if self.loss == "joint":
            return torch.softmax(v, -1)
        v = v.clamp(min=1e-3)
        return v / v.sum(-1, keepdim=True)


def save_value(path, model: SeatValueNet, **meta):
    torch.save({"kind": "seat_value", "cfg": model.cfg, "state_dict": model.state_dict(), **meta}, path)


class SeatValue:
    def __init__(self, model):
        self.model = model.eval()
        self.obs_size = model.cfg["obs"]

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> np.ndarray:
        return self.model.win_prob(x[None])[0].numpy().astype(np.float64)

    @torch.no_grad()
    def batch(self, x: torch.Tensor) -> np.ndarray:
        """x [positions, seats, obs] -> win shares [positions, seats]."""
        return self.model.win_prob(x).numpy().astype(np.float64)


class LegacyValue:
    def __init__(self, net):
        self.net = net
        self.obs_size = net.body[0].in_features
        self.feature_version = getattr(net, "feature_version", 1)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> np.ndarray:
        v = self.net.value(self.net.body(x)).squeeze(-1).numpy().astype(np.float64)
        v = np.clip(v, 1e-3, None)
        return v / v.sum()

    @torch.no_grad()
    def batch(self, x: torch.Tensor) -> np.ndarray:
        v = self.net.value(self.net.body(x)).squeeze(-1).numpy().astype(np.float64)
        v = np.clip(v, 1e-3, None)
        return v / v.sum(-1, keepdims=True)


def load_value(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "arch" in ck.get("cfg", {}):               # seat value model (also runs/varch checkpoints)
        m = SeatValueNet(**ck["cfg"])
        m.load_state_dict(ck["state_dict"])
        return SeatValue(m)
    from .model import load
    return LegacyValue(load(path))
