"""Expert iteration: search makes the network stronger, not just its play.

    python3 -m bots.rl.exit --out runs/exit1 --init runs/league3/champion.pt

Each generation:

 1. Play games in which the learner's seats use tree search (bots/search.py)
    with the BEST network so far (so a rejected generation cannot poison the
    data), Dirichlet noise at the root for exploration.
    The other seats come from the league (scripted styles and past
    checkpoints), or every seat is the learner (--p-selfplay).
 2. Train the network to imitate the search -- cross-entropy to the root
    visit distribution at every searched decision -- and to predict the
    final result (win share) at every decision the learner made. Decisions
    that are not searched (fuel, running plants) are anchored to the
    generating network's own policy, so improving auctions and building
    cannot make the shared layers forget them. Training reuses the last
    --buffer-gens generations of data.
 3. Gate: the new network (without search) plays three copies of the best
    network so far; best.pt advances only if it is not worse (>= 1/N). Its
    worst case against the scripted styles is logged for monitoring.

Improvement compounds: a sharper accepted network makes a stronger search,
which gives better targets.
"""
from __future__ import annotations

import argparse
import dataclasses
import io
import json
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cpp", "build"))
import pgcore  # noqa: E402

from ..evaluate import DEFAULT_OPPONENTS, worst_case  # noqa: E402
from ..heuristic import make  # noqa: E402
from ..search import SearchAgent  # noqa: E402
from .features import AbstractActions, Encoder  # noqa: E402
from .model import PolicyValueNet, RLAgent, load, save  # noqa: E402

LEARNER = "learner"


@dataclasses.dataclass
class Config:
    out: str = "runs/exit"
    init: str = "runs/league3/champion.pt"
    players: int = 4
    map: str = "germany"
    generations: int = 12
    workers: int = max(1, (os.cpu_count() or 2) - 2)
    games_per_worker: int = 6
    sims: int = 100
    root_noise: float = 0.15
    sample_rounds: int = 3        # sample moves in proportion to visits up to this round
    learner_seats: int = 2        # searched seats per game (unless self-play)
    p_selfplay: float = 0.25      # games where every seat is the searching learner
    extra_opponents: str = ""     # comma-separated checkpoints for the other seats
    p_scripted: float = 0.5       # other seats: scripted style vs past checkpoint
    buffer_gens: int = 3
    epochs: int = 3
    batch: int = 1024
    lr: float = 1e-4
    value_coef: float = 1.0
    gate_games: int = 400
    eval_games: int = 100         # per scripted opponent type
    seed: int = 0


# ---------------------------------------------------------------------------
# Self-play with search (worker processes)
# ---------------------------------------------------------------------------
def _play(job):
    cfg, weights, meta, scripted, past, seed = job
    torch.set_num_threads(1)
    rng = random.Random(seed)
    np.random.seed(seed % (1 << 31))
    rules = pgcore.Rules(players=cfg.players, map=cfg.map)
    acts = AbstractActions(rules)
    net = PolicyValueNet(meta["obs_size"], meta["n_actions"], meta["hidden"])
    net.load_state_dict(torch.load(io.BytesIO(weights), weights_only=False))
    net.feature_version = meta["feature_version"]
    net.eval()
    enc = Encoder(rules, net.feature_version)
    searcher = SearchAgent(net=net, sims=cfg.sims, root_noise=cfg.root_noise)
    searcher.reset(rules, 0, random.Random(rng.random()))
    pol_obs, pol_mask, pol_pi = [], [], []
    val_obs, val_z = [], []
    results = []
    for _ in range(cfg.games_per_worker):
        n = cfg.players
        if rng.random() < cfg.p_selfplay:
            lineup = [LEARNER] * n
        else:
            k = min(cfg.learner_seats, n - 1)

            def pick():
                group = scripted if (not past or rng.random() < cfg.p_scripted) else past
                return rng.choice(group)
            lineup = [LEARNER] * k + [pick() for _ in range(n - k)]
            rng.shuffle(lineup)
        others = {}
        for seat, who in enumerate(lineup):
            if who != LEARNER:
                agent = make("rl:" + who) if who.endswith(".pt") else make(who)
                agent.reset(rules, seat, random.Random(rng.random()))
                others[seat] = agent
        seat_vals = defaultdict(list)                     # seat -> indices into val_obs
        state = pgcore.State(rules)
        while not state.is_terminal():
            if state.is_chance_node():
                outs = state.chance_outcomes()
                state.apply_action(rng.choices([o for o, _ in outs], [p for _, p in outs])[0])
                continue
            seat = state.current_player()
            if seat in others:
                state.apply_action(others[seat].act(state))
                continue
            v = json.loads(state.to_json())
            obs = enc._encode(state, v, seat)
            seat_vals[seat].append(len(val_obs))
            val_obs.append(obs)
            legal = state.legal_actions()
            if len(legal) == 1:
                state.apply_action(legal[0])
                continue
            prior, amap, _ = searcher._evaluate(state)
            mask, _ = acts.mask_and_map(legal)
            if searcher.wants_search(state):
                visits, amap = searcher.search(state)
                pi = np.zeros(acts.n, dtype=np.float32)
                total = sum(visits.values())
                for x, c in visits.items():
                    pi[x] = c / total
                pol_obs.append(obs)
                pol_mask.append(mask)
                pol_pi.append(pi)
                keys = list(visits)
                if v["round"] <= cfg.sample_rounds:
                    x = rng.choices(keys, [visits[k] + 1e-6 for k in keys])[0]
                else:
                    x = max(keys, key=lambda k: visits[k])
                state.apply_action(amap[x])
            else:
                # anchor: the generating network's own policy is the target
                pi = np.zeros(acts.n, dtype=np.float32)
                for x, p in prior.items():
                    pi[x] = p
                pol_obs.append(obs)
                pol_mask.append(mask)
                pol_pi.append(pi)
                state.apply_action(amap[max(prior, key=prior.get)])
        returns = state.returns()
        for seat, idx in seat_vals.items():
            val_z.extend([(i, returns[seat]) for i in idx])
        learner_seats = [s for s, w in enumerate(lineup) if w == LEARNER]
        results.append({"learner": float(np.mean([returns[s] for s in learner_seats])),
                        "selfplay": len(learner_seats) == n, "rounds": state.round()})
    z = np.zeros(len(val_obs), dtype=np.float32)
    for i, r in val_z:
        z[i] = r
    data = {"pol_obs": np.stack(pol_obs) if pol_obs else np.zeros((0, meta["obs_size"]), np.float32),
            "pol_mask": np.stack(pol_mask) if pol_mask else np.zeros((0, meta["n_actions"]), bool),
            "pol_pi": np.stack(pol_pi) if pol_pi else np.zeros((0, meta["n_actions"]), np.float32),
            "val_obs": np.stack(val_obs), "val_z": z}
    return data, results


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------
def train(net, opt, buffer, cfg) -> dict:
    data = {k: np.concatenate([b[k] for b in buffer]) for k in buffer[0]}
    po = torch.from_numpy(data["pol_obs"])
    pm = torch.from_numpy(data["pol_mask"])
    pp = torch.from_numpy(data["pol_pi"])
    vo = torch.from_numpy(data["val_obs"])
    vz = torch.from_numpy(data["val_z"])
    n_pol, n_val = len(po), len(vo)
    steps = max(1, cfg.epochs * n_pol // cfg.batch)
    full = torch.ones(cfg.batch, net.policy.out_features, dtype=torch.bool)
    net.train()
    stats = defaultdict(float)
    for _ in range(steps):
        i = torch.randint(0, n_pol, (cfg.batch,))
        logits, _ = net(po[i], pm[i])
        pol_loss = -(pp[i] * F.log_softmax(logits, dim=-1)).sum(-1).mean()
        j = torch.randint(0, n_val, (cfg.batch,))
        _, value = net(vo[j], full)
        val_loss = F.mse_loss(value, vz[j])
        loss = pol_loss + cfg.value_coef * val_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        stats["policy_loss"] += pol_loss.item()
        stats["value_loss"] += val_loss.item()
    net.eval()
    out = {k: round(v / steps, 4) for k, v in stats.items()}
    out.update(steps=steps, policy_samples=n_pol, value_samples=n_val)
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
    net = load(cfg.init)
    meta = {"obs_size": net.body[0].in_features, "n_actions": net.policy.out_features,
            "hidden": net.body[0].out_features, "feature_version": net.feature_version}
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    best = os.path.join(cfg.out, "best.pt")
    shutil.copyfile(cfg.init, best)
    scripted = DEFAULT_OPPONENTS
    past = [p for p in cfg.extra_opponents.split(",") if p]
    log = open(os.path.join(cfg.out, "log.jsonl"), "a")
    rng = random.Random(cfg.seed)
    buffer = []
    with mp.get_context("spawn").Pool(cfg.workers) as pool:
        for gen in range(1, cfg.generations + 1):
            t0 = time.time()
            buf = io.BytesIO()
            torch.save(load(best).state_dict(), buf)      # data from the best network
            jobs = [(cfg, buf.getvalue(), meta, scripted, past, rng.randrange(1 << 30))
                    for _ in range(cfg.workers)]
            parts, results = [], []
            for data, res in pool.imap_unordered(_play, jobs):
                parts.append(data)
                results.extend(res)
            gen_data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            buffer = (buffer + [gen_data])[-cfg.buffer_gens:]
            t_play = time.time() - t0
            stats = train(net, opt, buffer, cfg)
            path = os.path.join(cfg.out, f"gen{gen:03d}.pt")
            save(path, net, generation=gen)
            t_train = time.time() - t0 - t_play
            gate = worst_case("rl:" + path, ["rl:" + best], cfg.players, cfg.gate_games,
                              gen, procs=cfg.workers, start="spawn")["worst"]
            scripted_eval = worst_case("rl:" + path, DEFAULT_OPPONENTS, cfg.players,
                                       cfg.eval_games, 1000 + gen, procs=cfg.workers,
                                       start="spawn")
            accepted = gate >= 1.0 / cfg.players
            if accepted:
                shutil.copyfile(path, best)
            vs_league = [r["learner"] for r in results if not r["selfplay"]]
            rec = {"gen": gen, "games": len(results), "play_s": round(t_play),
                   "train_s": round(t_train), **stats,
                   "learner_share_vs_league": round(float(np.mean(vs_league)), 3) if vs_league else None,
                   "vs_best": round(gate, 3), "accepted": accepted,
                   "worst": round(scripted_eval["worst"], 3),
                   "worst_opponent": scripted_eval["worst_opponent"],
                   "mean": round(scripted_eval["mean"], 3)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
