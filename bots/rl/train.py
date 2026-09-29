"""PPO with a league of opponents for Power Grid.

    python3 -m bots.rl.train --out runs/first --iterations 200

Each iteration, worker processes play complete games and the learner updates
on them. The seats the learner does not control are filled from a league:
pure self-play (every seat the current policy), the scripted styles, a random
player, and frozen snapshots of earlier versions. Opponent types the learner
does poorly against are drawn more often (prioritised fictitious self-play),
which is what pushes it towards a strategy that beats any kind of opponent
rather than just itself.

Reward: the final share of the win (1 for a sole winner), plus optional
potential-based shaping on min(cities, capacity), which speeds learning
without changing which policies are optimal.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import io
import json
import multiprocessing as mp
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cpp", "build"))
import pgcore  # noqa: E402

from .. import arena  # noqa: E402
from ..heuristic import STYLES, make  # noqa: E402
from .features import AbstractActions, Encoder  # noqa: E402
from .model import PolicyValueNet, RLAgent, load, save  # noqa: E402

LEARNER = "learner"


@dataclasses.dataclass
class Config:
    out: str = "runs/ppo"
    players: int = 4
    map: str = "germany"
    iterations: int = 200
    workers: int = max(1, (os.cpu_count() or 2) - 2)
    games_per_worker: int = 2
    hidden: int = 512
    lr: float = 3e-4
    epochs: int = 4
    minibatch: int = 4096
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lam: float = 0.95
    shaping: float = 0.2          # potential scale; 0 = pure win/loss reward
    p_selfplay: float = 0.3       # games where the learner plays every seat
    p_two_seats: float = 0.3      # otherwise, learner plays two seats instead of one
    snapshot_every: int = 10
    max_snapshots: int = 8
    eval_every: int = 10
    eval_games: int = 400
    seed: int = 0
    init: str = ""                # checkpoint to start from


# ---------------------------------------------------------------------------
# Rollouts (worker processes)
# ---------------------------------------------------------------------------
_CACHE: Dict[str, PolicyValueNet] = {}


def _snapshot(path: str) -> PolicyValueNet:
    if path not in _CACHE:
        _CACHE[path] = load(path)
    return _CACHE[path]


def _potential(v: dict, seat: int, scale: float) -> float:
    p = v["players"][seat]
    cap = sum(pl["power"] for pl in p["plants"])
    return scale * min(len(p["cities"]), cap) / v["rules"]["end_cities"]


def _rollout(job):
    cfg, weights, opponents, seed = job
    torch.set_num_threads(1)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    rules = pgcore.Rules(players=cfg.players, map=cfg.map)
    enc, acts = Encoder(rules), AbstractActions(rules)
    net = PolicyValueNet(enc.size, acts.n, cfg.hidden)
    net.load_state_dict(torch.load(io.BytesIO(weights), weights_only=False))
    net.eval()
    names = [o for o, _ in opponents]
    weights_ = [w for _, w in opponents]
    batch = defaultdict(list)
    results = []
    for _ in range(cfg.games_per_worker):
        n = cfg.players
        if rng.random() < cfg.p_selfplay:
            lineup = [LEARNER] * n
        else:
            k = 2 if (rng.random() < cfg.p_two_seats and n > 2) else 1
            lineup = [LEARNER] * k + rng.choices(names, weights_, k=n - k)
            rng.shuffle(lineup)
        seats = {}
        for seat, who in enumerate(lineup):
            if who == LEARNER:
                continue
            agent = RLAgent(who, net=_snapshot(who)) if who.endswith(".pt") else make(who)
            agent.reset(rules, seat, random.Random(rng.random()))
            seats[seat] = agent
        traj = {s: defaultdict(list) for s, who in enumerate(lineup) if who == LEARNER}
        state = pgcore.State(rules)
        while not state.is_terminal():
            if state.is_chance_node():
                outs = state.chance_outcomes()
                state.apply_action(rng.choices([o for o, _ in outs], [p for _, p in outs])[0])
                continue
            seat = state.current_player()
            if seat in seats:
                state.apply_action(seats[seat].act(state))
                continue
            v = json.loads(state.to_json())
            obs = enc._encode(state, v, seat)
            mask, amap = acts.mask_and_map(state.legal_actions())
            with torch.no_grad():
                logits, value = net(torch.from_numpy(obs)[None], torch.from_numpy(mask)[None])
                dist = torch.distributions.Categorical(logits=logits)
                x = dist.sample()
            t = traj[seat]
            t["obs"].append(obs)
            t["mask"].append(mask)
            t["act"].append(int(x))
            t["logp"].append(float(dist.log_prob(x)))
            t["val"].append(float(value))
            t["phi"].append(_potential(v, seat, cfg.shaping))
            state.apply_action(amap[int(x)])
        returns = state.returns()
        opp_kinds = sorted({w for w in lineup if w != LEARNER}) or ["self"]
        results.append({"lineup": lineup, "returns": returns, "rounds": state.round(),
                        "learner": float(np.mean([returns[s] for s in traj])),
                        "opponents": opp_kinds})
        for seat, t in traj.items():
            m = len(t["act"])
            if m == 0:
                continue
            # reward: shaping phi(s_{t+1}) - phi(s_t), terminal phi = 0, plus the result
            phi = t["phi"] + [0.0]
            rew = np.array([phi[i + 1] - phi[i] for i in range(m)], dtype=np.float32)
            rew[-1] += returns[seat]
            val = np.array(t["val"] + [0.0], dtype=np.float32)
            adv = np.zeros(m, dtype=np.float32)
            gae = 0.0
            for i in reversed(range(m)):            # gamma = 1: episodic
                delta = rew[i] + val[i + 1] - val[i]
                gae = delta + cfg.lam * gae
                adv[i] = gae
            batch["obs"].append(np.stack(t["obs"]))
            batch["mask"].append(np.stack(t["mask"]))
            batch["act"].append(np.array(t["act"], dtype=np.int64))
            batch["logp"].append(np.array(t["logp"], dtype=np.float32))
            batch["adv"].append(adv)
            batch["ret"].append(adv + val[:-1])
    out = {k: np.concatenate(v) for k, v in batch.items()} if batch else {}
    return out, results


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------
def ppo_update(net, opt, data, cfg) -> dict:
    net.train()
    obs = torch.from_numpy(data["obs"])
    mask = torch.from_numpy(data["mask"])
    act = torch.from_numpy(data["act"])
    old_logp = torch.from_numpy(data["logp"])
    adv = torch.from_numpy(data["adv"])
    ret = torch.from_numpy(data["ret"])
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = len(act)
    stats = defaultdict(float)
    steps = 0
    for _ in range(cfg.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, cfg.minibatch):
            idx = perm[i:i + cfg.minibatch]
            logits, value = net(obs[idx], mask[idx])
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(act[idx])
            ratio = torch.exp(logp - old_logp[idx])
            pg = -torch.min(ratio * adv[idx],
                            torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv[idx]).mean()
            vf = 0.5 * ((value - ret[idx]) ** 2).mean()
            ent = dist.entropy().mean()
            loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            stats["policy_loss"] += pg.item()
            stats["value_loss"] += vf.item()
            stats["entropy"] += ent.item()
            stats["clip_frac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
            steps += 1
    net.eval()
    return {k: v / steps for k, v in stats.items()}


def evaluate(path: str, cfg: Config, games: int, seed: int) -> dict:
    """Win rate of the checkpoint (greedy) against fields of scripted bots."""
    focus = "rl:" + path
    out = {}
    res = arena.run(list(STYLES), cfg.players, games, seed, focus=focus, procs=cfg.workers,
                    start="spawn")
    out["vs_scripted_field"] = round(res[focus]["win_rate"], 3)
    res = arena.run(["random"], cfg.players, games // 4, seed + 1, focus=focus, procs=cfg.workers,
                    start="spawn")
    out["vs_random"] = round(res[focus]["win_rate"], 3)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    for f in dataclasses.fields(Config):
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    cfg = Config(**{k.replace("-", "_"): v for k, v in vars(ap.parse_args()).items()})
    os.makedirs(cfg.out, exist_ok=True)
    json.dump(dataclasses.asdict(cfg), open(os.path.join(cfg.out, "config.json"), "w"), indent=1)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    rules = pgcore.Rules(players=cfg.players, map=cfg.map)
    enc, acts = Encoder(rules), AbstractActions(rules)
    net = load(cfg.init) if cfg.init else PolicyValueNet(enc.size, acts.n, cfg.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    print(f"obs {enc.size}, actions {acts.n}, params {sum(p.numel() for p in net.parameters()):,}")

    pool_scripted = list(STYLES) + ["random"]
    win_vs = defaultdict(lambda: 0.5)          # EMA of learner result per opponent kind
    log = open(os.path.join(cfg.out, "log.jsonl"), "a")
    rng = random.Random(cfg.seed)
    ctx = mp.get_context("spawn")   # forking after PyTorch has started can deadlock
    with ctx.Pool(cfg.workers) as pool:
        for it in range(1, cfg.iterations + 1):
            t0 = time.time()
            snaps = sorted(glob.glob(os.path.join(cfg.out, "snap_*.pt")))[-cfg.max_snapshots:]
            league = pool_scripted + snaps
            # prioritise opponents the learner does badly against
            opponents = [(o, (1.05 - win_vs[o]) ** 2 * (0.25 if o == "random" else 1.0))
                         for o in league]
            buf = io.BytesIO()
            torch.save(net.state_dict(), buf)
            jobs = [(cfg, buf.getvalue(), opponents, rng.randrange(1 << 30))
                    for _ in range(cfg.workers)]
            parts, results = [], []
            for data, res in pool.imap_unordered(_rollout, jobs):
                if data:
                    parts.append(data)
                results.extend(res)
            data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            t_roll = time.time() - t0
            stats = ppo_update(net, opt, data, cfg)
            for r in results:
                for o in r["opponents"]:
                    win_vs[o] = 0.95 * win_vs[o] + 0.05 * r["learner"]
            rec = {"iter": it, "samples": int(len(data["act"])), "games": len(results),
                   "rollout_s": round(t_roll, 1), "update_s": round(time.time() - t0 - t_roll, 1),
                   "rounds": round(float(np.mean([r["rounds"] for r in results])), 1),
                   "learner_share": round(float(np.mean([r["learner"] for r in results
                                                         if r["opponents"] != ["self"]] or [0])), 3),
                   **{k: round(v, 4) for k, v in stats.items()},
                   "win_vs": {k: round(v, 3) for k, v in sorted(win_vs.items())
                              if not k.endswith(".pt")}}
            save(os.path.join(cfg.out, "latest.pt"), net, iteration=it, config=dataclasses.asdict(cfg))
            if it % cfg.snapshot_every == 0:
                save(os.path.join(cfg.out, f"snap_{it:05d}.pt"), net, iteration=it)
            if it % cfg.eval_every == 0 or it == cfg.iterations:
                rec["eval"] = evaluate(os.path.join(cfg.out, "latest.pt"), cfg, cfg.eval_games, it)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
