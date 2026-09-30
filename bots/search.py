"""Decision-time search on top of a trained policy/value network.

    make("mcts:100:runs/league3/champion.pt")   # 100 simulations per searched decision

Monte Carlo tree search in the AlphaZero style, adapted to 2-6 players and
chance: the network's policy is the prior over moves (PUCT), and leaves are
scored by the value head from every seat's point of view, giving a vector of
expected win shares. Each player in the tree picks the move best for itself
(max^n). Plant draws are chance nodes: every visit samples an outcome from
the true probabilities, and each outcome gets its own subtree, so the search
does not plan as if one sampled draw were certain. The engine's cheap state copy (under a
microsecond) makes this affordable.

Search runs only where decisions matter most (auctions, building, choosing
which plant to discard) and only when there is a real choice; routine
decisions (single fuel units, running plants) use the network directly.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional

import numpy as np
import torch

from .base import Agent
from .rl.features import AbstractActions, Encoder
from .rl.model import load

SEARCH_PHASES = ("AUCTION_SELECT", "AUCTION_BID", "AUCTION_DISCARD", "BUILD")


class Node:
    __slots__ = ("state", "mover", "prior", "amap", "children", "n", "w", "expanded")

    def __init__(self, state):
        self.state = state
        self.mover = state.current_player() if not state.is_terminal() else -1
        self.prior: Dict[int, float] = {}
        self.amap: Dict[int, int] = {}
        self.children: Dict[int, "Node"] = {}
        self.n = 0
        self.w: Optional[np.ndarray] = None
        self.expanded = False


class SearchAgent(Agent):
    def __init__(self, path: str, sims: int = 100, c_puct: float = 1.5,
                 phases=SEARCH_PHASES, seed: int = 0):
        self.path = path
        self.sims = sims
        self.c_puct = c_puct
        self.phases = set(phases)
        self.net = load(path)
        self.name = f"mcts{sims}"

    def reset(self, rules, seat, rng):
        super().reset(rules, seat, rng)
        self.n_players = rules.num_players
        self.enc = Encoder(rules, getattr(self.net, "feature_version", 1))
        self.actions = AbstractActions(rules)
        torch.set_num_threads(1)

    # ---- network --------------------------------------------------------
    @torch.no_grad()
    def _evaluate(self, state):
        """Prior over the mover's abstract actions, and a value vector (one
        expected win share per seat, normalised to sum to 1)."""
        v = json.loads(state.to_json())
        n = self.n_players
        obs = np.stack([self.enc._encode(state, v, p) for p in range(n)])
        mask, amap = self.actions.mask_and_map(state.legal_actions())
        masks = np.ones((n, self.actions.n), dtype=np.bool_)
        mover = state.current_player()
        masks[mover] = mask
        logits, values = self.net(torch.from_numpy(obs), torch.from_numpy(masks))
        prior = torch.softmax(logits[mover], dim=0).numpy()
        vals = np.clip(values.numpy().astype(np.float64), 1e-3, None)
        vals /= vals.sum()
        return {x: float(prior[x]) for x in amap}, amap, vals

    def _greedy(self, state) -> int:
        prior, amap, _ = self._evaluate(state)
        return amap[max(prior, key=prior.get)]

    # ---- search ---------------------------------------------------------
    def act(self, state) -> int:
        legal = state.legal_actions()
        if len(legal) == 1:
            return legal[0]
        phase = json.loads(state.to_json())["phase"]
        if phase not in self.phases:
            return self._greedy(state)
        root = Node(state.clone())
        self._expand(root)
        if len(root.prior) == 1:
            return root.amap[next(iter(root.prior))]
        for _ in range(self.sims):
            self._simulate(root)
        best = max(root.children.items(), key=lambda kv: kv[1].n)[0] if root.children else \
            max(root.prior, key=root.prior.get)
        return root.amap[best]

    def _expand(self, node: Node) -> np.ndarray:
        node.prior, node.amap, vals = self._evaluate(node.state)
        node.expanded = True
        return vals

    def _simulate(self, root: Node):
        path = [root]
        node = root
        while True:
            s = node.state
            if s.is_terminal():
                vals = np.asarray(s.returns(), dtype=np.float64)
                break
            if s.is_chance_node():
                outs = s.chance_outcomes()
                o = self.rng.choices([a for a, _ in outs], [p for _, p in outs])[0]
                child = node.children.get(o)
                if child is None:
                    nxt = s.clone()
                    nxt.apply_action(o)
                    child = Node(nxt)
                    node.children[o] = child
                path.append(child)
                node = child
                continue
            if not node.expanded:
                vals = self._expand(node)
                break
            x = self._select(node)
            child = node.children.get(x)
            if child is None:
                nxt = s.clone()
                nxt.apply_action(node.amap[x])
                child = Node(nxt)
                node.children[x] = child
            path.append(child)
            node = child
        for nd in path:
            nd.n += 1
            nd.w = vals.copy() if nd.w is None else nd.w + vals

    def _select(self, node: Node) -> int:
        mover = node.mover
        sqrt_n = math.sqrt(max(node.n, 1))
        parent_q = node.w[mover] / node.n if node.n else 1.0 / self.n_players
        best, best_score = None, -1e9
        for x, p in node.prior.items():
            child = node.children.get(x)
            if child is not None and child.n:
                q = child.w[mover] / child.n
                u = self.c_puct * p * sqrt_n / (1 + child.n)
            else:
                q = parent_q                      # first-play urgency: the parent's value
                u = self.c_puct * p * sqrt_n
            if q + u > best_score:
                best, best_score = x, q + u
        return best
