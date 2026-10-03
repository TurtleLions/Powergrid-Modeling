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

Options (all off by default; agent names take them after the simulation count,
e.g. "mcts:100+norm+c3+reuse+om+np+fuel:policy.pt:value.pt"):
  fuel   also search fuel purchases.
  reuse  keep the subtree of the position actually reached: after our move,
         the moves played since (seen through Agent.observe) lead down the old
         tree; if the state there matches exactly, its statistics are reused.
  om     opponent modelling: identify scripted opponents from their moves
         (each style's choice is compared with what they played; a style needs
         >= 95% posterior against an "unknown player" baseline), and at their
         turns in the tree play the identified style's move instead of
         assuming they search like us. Unidentified opponents are unchanged.
The value network may differ by player count: "...:policy.pt:a.pt@4|b.pt" uses
a.pt in 4-player games and b.pt otherwise.
  np     NumPy forward passes for MLP networks (same values up to float
         rounding; avoids PyTorch's per-call overhead).
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
FUEL_PHASE = "BUY_FUEL"
CHANCE = -1
OM_EPS = 0.05          # a scripted style plays its own move with probability 1 - eps
OM_UNKNOWN = 0.5       # an unidentified player's chance of making any given style's move
OM_MIN_MOVES = 6       # observed decisions before an opponent can be identified
OM_CONFIDENCE = 0.95   # posterior mass needed to treat a predicted move as certain


class OpponentModel:
    """Which scripted style (if any) each opponent plays, from their moves."""

    def __init__(self, rules, me: int, rng):
        import random
        from .heuristic import STYLES, make
        self.styles = list(STYLES)
        self.cands, self.ll, self.unknown, self.count = {}, {}, {}, {}
        for p in range(rules.num_players):
            if p == me:
                continue
            self.cands[p] = {}
            for name in self.styles:
                a = make(name)
                a.reset(rules, p, random.Random(rng.random()))
                self.cands[p][name] = a
            self.ll[p] = {name: 0.0 for name in self.styles}
            self.unknown[p] = 0.0
            self.count[p] = 0

    def _act(self, agent, state):
        agent._plan_key = None                    # the run plan is cached per round
        return agent.act(state)

    def observe(self, state, player, action):
        if player not in self.cands:
            return
        legal = state.legal_actions()
        if len(legal) <= 1:
            return
        miss = math.log(OM_EPS / (len(legal) - 1))
        hit = math.log(1 - OM_EPS)
        best = max(self.ll[player].values())
        for name in list(self.ll[player]):
            if self.ll[player][name] < best - 15:  # ruled out: stop paying for it
                continue
            self.ll[player][name] += hit if self._act(self.cands[player][name], state) == action else miss
        self.unknown[player] += math.log(OM_UNKNOWN)
        self.count[player] += 1

    def posterior(self, player) -> Dict[str, float]:
        if self.count.get(player, 0) < OM_MIN_MOVES:
            return {}
        logits = dict(self.ll[player])
        logits[None] = self.unknown[player]
        top = max(logits.values())
        w = {k: math.exp(v - top) for k, v in logits.items()}
        z = sum(w.values())
        return {k: v / z for k, v in w.items() if k is not None and v / z > 0.01}

    def predict(self, state, player) -> Optional[int]:
        """The engine action this opponent will play, if the model is sure enough."""
        post = self.posterior(player)
        if not post:
            return None
        mass: Dict[int, float] = {}
        for name, pr in post.items():
            a = self._act(self.cands[player][name], state)
            mass[a] = mass.get(a, 0.0) + pr
        a, m = max(mass.items(), key=lambda kv: kv[1])
        return a if m >= OM_CONFIDENCE else None


class _NumpyNets:
    """MLP forward passes in NumPy: the policy net, and an MLP value net if any."""

    def __init__(self, net, value_net):
        def layers(seq):
            return [(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy())
                    for m in seq if isinstance(m, torch.nn.Linear)]
        self.body = layers(net.body)
        self.policy = layers([net.policy])[0]
        self.own_value = layers([net.value])[0]
        self.value, self.joint = None, False
        if isinstance(value_net, LegacyValue):
            self.value = (layers(value_net.net.body), layers([value_net.net.value])[0])
        elif value_net is not None and getattr(value_net.model, "arch", "") == "mlp":
            self.value = (layers(value_net.model.enc), layers([value_net.model.value])[0])
            self.joint = value_net.model.loss == "joint"
        self.ok = value_net is None or self.value is not None

    @staticmethod
    def _mlp(layers, x):
        for W, b in layers:
            x = np.maximum(x @ W + b, 0.0)
        return x

    def evaluate(self, obs, mover, mask, has_value_net):
        h = self._mlp(self.body, obs if not has_value_net else obs[mover:mover + 1])
        logits = (h[mover if not has_value_net else 0] @ self.policy[0] + self.policy[1])
        logits = np.where(mask, logits, -1e9)
        e = np.exp(logits - logits.max())
        prior = e / e.sum()
        if not has_value_net:
            v = (h @ self.own_value[0] + self.own_value[1])[:, 0].astype(np.float64)
            v = np.clip(v, 1e-3, None)
            return prior, v / v.sum()
        enc, head = self.value
        v = (self._mlp(enc, obs) @ head[0] + head[1])[:, 0].astype(np.float64)
        if self.joint:
            e = np.exp(v - v.max())
            return prior, e / e.sum()
        v = np.clip(v, 1e-3, None)
        return prior, v / v.sum()


class Node:
    __slots__ = ("state", "mover", "prior", "amap", "children", "n", "w", "expanded", "value", "forced")

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
        self.forced: Optional[int] = None          # modelled opponent's move (-1: none), lazily


class SearchAgent(Agent):
    def __init__(self, path: str = "", sims: int = 100, c_puct: float = 1.5,
                 phases=SEARCH_PHASES, net=None, root_noise: float = 0.0,
                 noise_alpha: float = 0.3, value_path: str = "", value_net=None,
                 normalize_q: bool = False, batch: int = 1, search_fuel: bool = False,
                 reuse_tree: bool = False, opponent_model: bool = False, numpy_forward: bool = False):
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
        self.phases = set(phases) | ({FUEL_PHASE} if search_fuel else set())
        self.reuse_tree = reuse_tree
        self.opponent_model = opponent_model
        self.numpy_forward = numpy_forward
        self.net = net if net is not None else load(path)
        self.normalize_q = normalize_q
        if value_net is not None and isinstance(value_net, torch.nn.Module):
            value_net = LegacyValue(value_net)
        # value_path may pick a network per player count: "a.pt@4|b.pt" = a in 4-player games, else b
        self._values = {}
        if value_net is None and value_path:
            for part in value_path.split("|"):
                path, _, count = part.partition("@")
                self._values[int(count) if count else None] = load_value(path)
            value_net = self._values.get(None) or next(iter(self._values.values()))
        self.value_net = value_net
        self.root_noise = root_noise
        self.noise_alpha = noise_alpha
        self.batch = max(1, batch)
        self._fast_cache = {}
        self.fast = self._make_fast()
        self.name = f"mcts{sims}"

    def _make_fast(self):
        if not self.numpy_forward:
            return None
        key = id(self.value_net)
        if key not in self._fast_cache:
            fast = _NumpyNets(self.net, self.value_net)
            self._fast_cache[key] = fast if fast.ok else None   # attention value nets stay on PyTorch
        return self._fast_cache[key]

    def reset(self, rules, seat, rng):
        super().reset(rules, seat, rng)
        self.n_players = rules.num_players
        if self._values:                              # the value network for this player count
            self.value_net = self._values.get(rules.num_players, self._values.get(None)) \
                or next(iter(self._values.values()))
            self.fast = self._make_fast()
        self.enc = Encoder(rules, getattr(self.net, "feature_version", 1))
        if self.value_net is not None:
            assert self.value_net.obs_size == self.enc.size, "value network expects other features"
        self.actions = AbstractActions(rules)
        self.om = OpponentModel(rules, seat, rng) if self.opponent_model else None
        self._last: Optional[Node] = None             # tree reuse: node after our last move
        self._chosen: Optional[int] = None
        self._since: List[tuple] = []
        torch.set_num_threads(1)

    def observe(self, state, player, action):
        if self.om is not None and player >= 0 and player != self.seat:
            self.om.observe(state, player, action)
        if self.reuse_tree and self._last is not None:
            if player == self.seat and action == self._chosen and self._chosen is not None:
                self._chosen = None                   # our own move: the tree is already there
            else:
                self._since.append((player, action))

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
        mover = state.current_player()
        if self.fast is not None:
            prior, vals = self.fast.evaluate(obs, mover, mask, self.value_net is not None)
            return {x: float(prior[x]) for x in amap}, amap, vals
        masks = np.ones((n, self.actions.n), dtype=np.bool_)
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

    def search(self, state, root: Optional[Node] = None):
        """Run the search from `state` (or continue an existing tree whose root
        is this state); returns (visit counts per abstract action, abstract ->
        engine action map)."""
        if root is None:
            root = Node(state.clone())
            self._expand(root)
        self._root = root
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
        root = self._reused_root(state)
        self._last, self._since, self._chosen = None, [], None
        self.last_visits = None
        if len(legal) == 1:
            return legal[0]
        if not self.wants_search(state):
            return self._greedy(state)
        visits, amap = self.search(state, root)
        self.last_visits, self.last_amap = visits, amap   # for recording search targets
        x = max(visits, key=visits.get)
        if self.reuse_tree:
            self._last, self._chosen = self._root.children.get(x), amap[x]
        return amap[x]

    def _reused_root(self, state) -> Optional[Node]:
        """The old tree's node for `state`, following the moves seen since our last move."""
        node = self._last
        if not self.reuse_tree or node is None:
            return None
        for player, a in self._since:
            if node.state.is_chance_node():
                key = a
            elif node.forced is not None and node.forced == a:
                key = ("f", a)
            elif node.expanded:
                key = next((x for x, e in node.amap.items() if e == a), None)
            else:
                return None
            node = node.children.get(key)
            if node is None:
                return None
        if not node.expanded or node.state.to_json() != state.to_json():
            return None
        return node

    def _forced(self, node: Node) -> Optional[int]:
        """The modelled opponent's move at this node, if any."""
        if self.om is None or node.mover < 0 or node.mover == self.seat:
            return None
        if node.forced is None:
            a = self.om.predict(node.state, node.mover)
            node.forced = -1 if a is None else a
        return node.forced if node.forced >= 0 else None

    def _step(self, node: Node):
        """The child to descend to from an expanded decision node."""
        f = self._forced(node)
        if f is not None:
            key, a = ("f", f), f
        else:
            key = self._select(node)
            a = node.amap[key]
        child = node.children.get(key)
        if child is None:
            nxt = node.state.clone()
            nxt.apply_action(a)
            child = node.children[key] = Node(nxt)
        return child

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
            child = self._step(node)
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
            child = self._step(node)
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
