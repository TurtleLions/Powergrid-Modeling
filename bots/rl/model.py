"""Policy/value network and the agent that plays with a trained checkpoint."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn

from ..base import Agent
from .features import FEATURE_VERSION, AbstractActions, Encoder


class PolicyValueNet(nn.Module):
    def __init__(self, obs_size: int, n_actions: int, hidden: int = 512):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)   # expected final share of the win

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        h = self.body(obs)
        logits = self.policy(h).masked_fill(~mask, -1e9)
        return logits, self.value(h).squeeze(-1)


def save(path: str, net: PolicyValueNet, **meta):
    torch.save({"state_dict": net.state_dict(), "obs_size": net.body[0].in_features,
                "n_actions": net.policy.out_features, "hidden": net.body[0].out_features,
                "feature_version": getattr(net, "feature_version", FEATURE_VERSION),
                **meta}, path)


def load(path: str) -> PolicyValueNet:
    """Checkpoint -> network; net.feature_version says which Encoder it expects
    (checkpoints from before versioning are version 1)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = PolicyValueNet(ck["obs_size"], ck["n_actions"], ck["hidden"])
    net.load_state_dict(ck["state_dict"])
    net.feature_version = ck.get("feature_version", 1)
    net.eval()
    return net


@torch.no_grad()
def widen_input(net: PolicyValueNet, new_size: int, version: int) -> PolicyValueNet:
    """Grow the input layer for appended features. The new inputs get zero
    weights, so the widened network plays exactly as before until trained."""
    old = net.body[0]
    if new_size == old.in_features:
        net.feature_version = version
        return net
    assert new_size > old.in_features, "feature versions only append"
    layer = nn.Linear(new_size, old.out_features)
    layer.weight.zero_()
    layer.weight[:, :old.in_features] = old.weight
    layer.bias.copy_(old.bias)
    net.body[0] = layer
    net.feature_version = version
    return net


class RLAgent(Agent):
    """Plays a trained checkpoint. greedy=False samples from the policy."""

    def __init__(self, path: str, greedy: bool = True, net: PolicyValueNet = None):
        self.path = path
        self.greedy = greedy
        self.net = net if net is not None else load(path)
        self.name = "rl"

    def reset(self, rules, seat, rng):
        super().reset(rules, seat, rng)
        if not hasattr(self, "enc") or self.enc.rules is not rules:
            self.enc = Encoder(rules, getattr(self.net, "feature_version", 1))
            self.actions = AbstractActions(rules)
        torch.set_num_threads(1)

    @torch.no_grad()
    def act(self, state) -> int:
        mask, amap = self.actions.mask_and_map(state.legal_actions())
        obs = torch.from_numpy(self.enc.encode(state, self.seat))[None]
        logits, _ = self.net(obs, torch.from_numpy(mask)[None])
        if self.greedy:
            x = int(logits.argmax())
        else:
            x = int(torch.distributions.Categorical(logits=logits).sample())
        return amap[x]
