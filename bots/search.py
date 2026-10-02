"""Decision-time search on top of a trained policy/value network.

    make("mcts:100:runs/league3/champion.pt")   # 100 simulations per searched decision
    make("mcts:100:runs/league3/champion.pt:runs/value1/big_aux05.pt")   # separate value net
    make("mcts:100+norm+c3:runs/league3/champion.pt")   # options after the simulation count

Monte Carlo tree search in the AlphaZero style, adapted to 2-6 players and
chance: the network's policy is the prior over moves (PUCT), and leaves are
scored by the value head from every seat's point of view, giving a vector of
expected win shares. The value can come from a separate network (value_path),
e.g. one fitted by bots/rl/value.py, while the policy network supplies priors. Each player in the tree picks the move best for itself
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
from .rl.value_models import LegacyValue, load_value

SEARCH_PHASES = ("AUCTION_SELECT", "AUCTION_BID", "AUCTION_DISCARD", "BUILD")


class Node:
    __slots__ = ("state", "mover", "prior", "amap", "children", "n", "w", "expanded", "value")

    def __init__(self, state):
        self.state = state
        self.mover = state.current_player() if not state.is_terminal() else -1
        self.prior: Dict[int, float] = {}
        self.amap: Dict[int, int] = {}
        self.children: Dict[int, "Node"] = {}
        self.n = 0
        self.w: Optional[np.ndarray] = None
        self.expanded = False
        self.value: Optional[np.ndarray] = None   # network value at expansion (batched search)


class SearchAgent(Agent):
    def __init__(self, path: str = "", sims: int = 100, c_puct: float = 1.5,
                 phases=SEARCH_PHASES, net=None, root_noise: float = 0.0,
                 noise_alpha: float = 0.3, value_path: str = "", value_net=None,
                 normalize_q: bool = False, batch: int = 1):
        """root_noise > 0 mixes Dirichlet(noise_alpha) noise into the root prior
        (for exploration when generating training data; 0 for normal play).
        value_path / value_net: score leaves with this value network instead
        of the policy network's value head: a PolicyValueNet, or a seat
        value model (bots/rl/value_models.py) that scores all seats jointly.
        normalize_q: rescale the mover's Q values at each node to [0, 1] over
        its children (min-max, as in MuZero). Win shares differ by a few
        hundredths between moves, far less than the exploration term, so
        without this a confident prior is almost never overruled.
        batch > 1: each round walks `batch` paths down the tree, each node on
        a path counting a provisional visit (virtual loss: value 0) so the
        next path goes elsewhere, then scores all new leaves in one network
        call: about 2x cheaper per simulation on an idle core. It explores
        differently and, as implemented, is WEAKER at equal time (ladder
        2026-10-01, exit5 + v4: b16 at 200 sims -93 Elo vs sequential 100;
        b16 at 420 -29 vs sequential 200), probably because virtual loss
        skews the normalised Q. batch = 1, the plain sequential search, is
        the default."""
        self.path = path
        self.sims = sims
        self.c_puct = c_puct
        self.phases = set(phases)
        self.net = net if net is not None else load(path)
        self.normalize_q = normalize_q
        if value_net is not None and isinstance(value_net, torch.nn.Module):
            value_net = LegacyValue(value_net)
        self.value_net = value_net if value_net is not None else (load_value(value_path) if value_path else None)
        self.root_noise = root_noise
        self.noise_alpha = noise_alpha
        self.batch = max(1, batch)
        self.name = f"mcts{sims}"

    def reset(self, rules, seat, rng):
        super().reset(rules, seat, rng)
        self.n_players = rules.num_players
        self.enc = Encoder(rules, getattr(self.net, "feature_version", 1))
        if self.value_net is not None:
            assert self.value_net.obs_size == self.enc.size, "value network expects other features"
        self.actions = AbstractActions(rules)
        torch.set_num_threads(1)

    # ---- network --------------------------------------------------------
    def _features(self, state) -> np.ndarray:
        """Every seat's features [players, size]: the engine's C++ encoder
        (identical values) when this pgcore has it and the version matches."""
        if self.enc.version == 2 and hasattr(state, "features_all"):
            return state.features_all()
        v = json.loads(state.to_json())
        return np.stack([self.enc._encode(state, v, p) for p in range(self.n_players)])

    @torch.no_grad()
    def _evaluate(self, state):
        """Prior over the mover's abstract actions, and a value vector (one
        expected win share per seat, normalised to sum to 1)."""
        n = self.n_players
        obs = self._features(state)
        mask, amap = self.actions.mask_and_map(state.legal_actions())
        masks = np.ones((n, self.actions.n), dtype=np.bool_)
        mover = state.current_player()
        masks[mover] = mask
        x = torch.from_numpy(obs)
        if self.value_net is None:
            logits, values = self.net(x, torch.from_numpy(masks))
            logits = logits[mover]
        else:
            logits, _ = self.net(x[mover:mover + 1], torch.from_numpy(mask)[None])
            logits = logits[0]
        prior = torch.softmax(logits, dim=0).numpy()
        if self.value_net is None:
            vals = np.clip(values.numpy().astype(np.float64), 1e-3, None)
            vals /= vals.sum()
        else:
            vals = self.value_net(x)                  # win shares, summing to 1
        return {x: float(prior[x]) for x in amap}, amap, vals

    def _greedy(self, state) -> int:
        prior, amap, _ = self._evaluate(state)
        return amap[max(prior, key=prior.get)]

    # ---- search ---------------------------------------------------------
    def wants_search(self, state) -> bool:
        return (len(state.legal_actions()) > 1
                and json.loads(state.to_json())["phase"] in self.phases)

    def search(self, state):
        """Run the search from `state`; returns (visit counts per abstract
        action, abstract -> engine action map)."""
        root = Node(state.clone())
        self._expand(root)
        if len(root.prior) > 1 and self.root_noise > 0:
            keys = list(root.prior)
            noise = np.random.default_rng(self.rng.randrange(1 << 30)).dirichlet(
                [self.noise_alpha] * len(keys))
            for k, eta in zip(keys, noise):
                root.prior[k] = (1 - self.root_noise) * root.prior[k] + self.root_noise * eta
        if len(root.prior) > 1:
            if self.batch == 1:
                for _ in range(self.sims):
                    self._simulate(root)
            else:
                done = 0
                while done < self.sims:
                    done += self._simulate_batch(root, min(self.batch, self.sims - done))
        visits = {x: (root.children[x].n if x in root.children else 0) for x in root.prior}
        if not any(visits.values()):
            visits = {x: 1 if x == max(root.prior, key=root.prior.get) else 0 for x in root.prior}
        return visits, root.amap

    def act(self, state) -> int:
        legal = state.legal_actions()
        if len(legal) == 1:
            return legal[0]
        if not self.wants_search(state):
            return self._greedy(state)
        visits, amap = self.search(state)
        return amap[max(visits, key=visits.get)]

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

    @torch.no_grad()
    def _evaluate_batch(self, states):
        """_evaluate for several states in one network call per network."""
        n = self.n_players
        obs = np.stack([self._features(s) for s in states])            # [L, n, F]
        movers = [s.current_player() for s in states]
        maps = [self.actions.mask_and_map(s.legal_actions()) for s in states]
        masks = np.stack([m for m, _ in maps])
        x = torch.from_numpy(obs)
        rows = x[torch.arange(len(states)), torch.tensor(movers)]      # the movers' views
        logits, own_values = self.net(rows, torch.from_numpy(masks))
        priors = torch.softmax(logits, dim=-1).numpy()
        if self.value_net is not None:
            vals = self.value_net.batch(x)
        else:
            full = torch.ones(len(states) * n, self.actions.n, dtype=torch.bool)
            _, v = self.net(x.reshape(len(states) * n, -1), full)
            vals = np.clip(v.numpy().astype(np.float64).reshape(len(states), n), 1e-3, None)
            vals /= vals.sum(-1, keepdims=True)
        return [({a: float(priors[i][a]) for a in amap}, amap, vals[i])
                for i, (_, amap) in enumerate(maps)]

    def _descend(self, root: Node):
        """One path from the root to a leaf, counting a provisional visit on
        every node of it. Returns (path, value or None if the leaf needs the
        network)."""
        path, node = [root], root
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
                    child = node.children[o] = Node(nxt)
                path.append(child)
                node = child
                continue
            if not node.expanded:
                vals = None
                break
            x = self._select(node)
            child = node.children.get(x)
            if child is None:
                nxt = s.clone()
                nxt.apply_action(node.amap[x])
                child = node.children[x] = Node(nxt)
            path.append(child)
            node = child
        for nd in path:                       # virtual loss: a visit worth 0 to everyone
            nd.n += 1
            if nd.w is None:
                nd.w = np.zeros(self.n_players)
        return path, vals

    def _simulate_batch(self, root: Node, k: int) -> int:
        """k simulations whose new leaves are scored in one network call."""
        pending = []                          # (path, leaf) awaiting the network
        for _ in range(k):
            path, vals = self._descend(root)
            if vals is None:
                pending.append(path)
            else:
                for nd in path:
                    nd.w += vals
        leaves = list({id(p[-1]): p[-1] for p in pending}.values())
        if leaves:
            for leaf, (prior, amap, vals) in zip(leaves, self._evaluate_batch([l.state for l in leaves])):
                leaf.prior, leaf.amap, leaf.expanded = prior, amap, True
                leaf.value = vals
        for path in pending:
            vals = path[-1].value
            for nd in path:
                nd.w += vals
        return k

    def _select(self, node: Node) -> int:
        mover = node.mover
        sqrt_n = math.sqrt(max(node.n, 1))
        parent_q = node.w[mover] / node.n if node.n else 1.0 / self.n_players
        if self.normalize_q:
            qs = [parent_q] + [c.w[mover] / c.n for c in node.children.values() if c.n]
            lo, span = min(qs), max(qs) - min(qs)
        best, best_score = None, -1e9
        for x, p in node.prior.items():
            child = node.children.get(x)
            if child is not None and child.n:
                q = child.w[mover] / child.n
                u = self.c_puct * p * sqrt_n / (1 + child.n)
            else:
                q = parent_q                      # first-play urgency: the parent's value
                u = self.c_puct * p * sqrt_n
            if self.normalize_q:
                q = (q - lo) / span if span > 1e-9 else 0.5
            if q + u > best_score:
                best, best_score = x, q + u
        return best
