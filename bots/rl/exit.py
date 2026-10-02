"""Expert iteration: search makes the network stronger, not just its play.

    python3 -m bots.rl.exit --out runs/exit4 --init runs/exit3/best.pt \
        --value-init runs/value1/big_aux05.pt --value-replay runs/value1/data.npz

Each generation:

 1. Play games in which the learner's seats use tree search (bots/search.py)
    with the BEST network so far (so a rejected generation cannot poison the
    data). With --value-init, leaves are scored by that separate value
    network (fitted by bots/rl/value.py, held fixed here) and the policy
    network is trained on policy targets only: per-decision value targets
    from a few hundred games are what made exit2's value head memorise games.
    --normalize-q with --c-puct 3 lets Q overrule a confident prior (see
    bots/search.py); --root-noise (0 by default) adds Dirichlet noise, which
    also leaks into the visit targets.
    The other seats come from the league (scripted styles and past
    checkpoints), or every seat is the learner (--p-selfplay).
 2. Train the network to imitate the search -- cross-entropy to the root
    visit distribution at every searched decision -- and to predict the
    final result (win share) at every decision the learner made. Decisions
    that are not searched (fuel, running plants) are anchored to the
    generating network's own policy, so improving auctions and building
    cannot make the shared layers forget them. Training reuses the last
    --buffer-gens generations of data.
 3. Gate: search with the new policy (--gate-sims, same value network) plays
    N-1 copies of search with the best policy; best.pt advances only if it
    is not worse: its share times N is at least 1 (a fair share), averaged
    over the player counts when --player-counts mixes them (e.g. 3,4,5,6;
    each self-play game then draws its count). The final bot searches, and a policy that helps
    search need not play better on its own: runs/exit3's raw-vs-raw gate
    rejected every generation after gen 6 while search with gen 6 was +79 Elo
    over search with its starting policy. The raw result (vs_best_raw) and
    the raw worst case against the scripted styles are logged. After a
    rejection the learner restarts from best.pt (--reset-on-reject) instead
    of drifting further on the same data.
 4. Every --value-every generations the value network is refreshed: fast
    raw-network games with the best policy (bots/rl/value.py gen) plus
    --value-replay, fine-tuned from the current value network; kept only if
    its held-out error beats the current one's. (Only for PolicyValueNet
    value networks; seat value models from bots/rl/value_td.py stay fixed.)

A --policy-holdout share of each generation's games is never trained on: the
log reports the fit on their searched decisions (cross-entropy, KL to the
search target, top-move agreement) for the new network and for best.pt, so
overfitting shows up as training loss falling while held-out KL does not.

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
from .value_models import LegacyValue, load_value  # noqa: E402

LEARNER = "learner"


@dataclasses.dataclass
class Config:
    out: str = "runs/exit"
    init: str = "runs/league3/champion.pt"
    players: int = 4
    player_counts: str = ""       # e.g. "3,4,5,6": each game draws its count (overrides players)
    map: str = "germany"
    generations: int = 12
    workers: int = max(1, (os.cpu_count() or 2) - 2)
    games_per_worker: int = 6
    sims: int = 100
    value_init: str = ""          # separate, fixed value network for search
    normalize_q: int = 1          # 1: min-max normalised Q in search
    c_puct: float = 3.0
    root_noise: float = 0.0
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
    gate_sims: int = 32           # simulations per decision in the search gate
    gate_raw_games: int = 400     # raw-vs-raw result, logged only
    reset_on_reject: int = 1
    policy_holdout: float = 0.1   # share of workers whose games are held out
    value_every: int = 4          # refresh the value network every k generations (0: never)
    value_games: int = 20000
    value_replay: str = ""        # earlier value data mixed into each refresh
    value_epochs: int = 2
    value_lr: float = 1e-4
    eval_games: int = 100         # per scripted opponent type
    seed: int = 0


# ---------------------------------------------------------------------------
# Self-play with search (worker processes)
# ---------------------------------------------------------------------------
def _play(job):
    cfg, weights, value_path, meta, scripted, past, seed = job
    torch.set_num_threads(1)
    rng = random.Random(seed)
    np.random.seed(seed % (1 << 31))
    counts = _counts(cfg)
    by_count = {}
    for k in counts:                                  # regions, step 2, game end per player count
        r = pgcore.Rules(players=k, map=cfg.map)
        by_count[k] = (r, AbstractActions(r))
    rules, acts = by_count[counts[0]]
    net = PolicyValueNet(meta["obs_size"], meta["n_actions"], meta["hidden"])
    net.load_state_dict(torch.load(io.BytesIO(weights), weights_only=False))
    net.feature_version = meta["feature_version"]
    net.eval()
    enc = Encoder(rules, net.feature_version)
    vnet = load_value(value_path) if value_path else None
    searcher = SearchAgent(net=net, sims=cfg.sims, root_noise=cfg.root_noise, value_net=vnet,
                           c_puct=cfg.c_puct, normalize_q=bool(cfg.normalize_q))
    searcher.reset(rules, 0, random.Random(rng.random()))
    pol_obs, pol_mask, pol_pi, pol_searched = [], [], [], []
    val_obs, val_z = [], []
    results = []
    for _ in range(cfg.games_per_worker):
        n = rng.choice(counts)
        rules, acts = by_count[n]
        enc = Encoder(rules, net.feature_version)
        searcher.reset(rules, 0, random.Random(rng.random()))
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
            obs = state.features(seat) if hasattr(state, "features") and enc.version == 2 \
                else enc._encode(state, v, seat)          # identical; C++ when available
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
                pol_searched.append(True)
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
                pol_searched.append(False)
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
            "pol_searched": np.array(pol_searched, dtype=bool),
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
        if cfg.value_init:                    # search uses the separate value network
            val_loss = torch.zeros(())
        else:
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


@torch.no_grad()
def policy_fit(net, data) -> dict:
    """Fit of `net` to the search targets on held-out searched decisions."""
    sel = data["pol_searched"]
    if not sel.any():
        return {}
    obs = torch.from_numpy(data["pol_obs"][sel])
    mask = torch.from_numpy(data["pol_mask"][sel])
    pi = torch.from_numpy(data["pol_pi"][sel])
    logits, _ = net(obs, mask)
    logp = F.log_softmax(logits, dim=-1)
    ce = -(pi * logp).sum(-1)
    ent = -(pi * torch.log(torch.where(pi > 0, pi, torch.ones_like(pi)))).sum(-1)
    agree = (logits.argmax(-1) == pi.argmax(-1)).float()
    return {"ce": round(ce.mean().item(), 4), "kl": round((ce - ent).mean().item(), 4),
            "agree": round(agree.mean().item(), 4), "n": int(sel.sum())}


def _counts(cfg):
    return [int(k) for k in cfg.player_counts.split(",")] if cfg.player_counts else [cfg.players]


def per_count(cfg, focus, opponents, games, seed):
    """worst_case at every player count (games split evenly); returns
    ({count: result}, mean over counts of the worst share x players: 1.0 = a fair share)."""
    counts = _counts(cfg)
    res = {k: worst_case(focus, opponents, k, max(1, games // len(counts)), seed,
                         procs=cfg.workers, start="spawn") for k in counts}
    return res, float(np.mean([r["worst"] * k for k, r in res.items()]))


def search_name(cfg, policy: str, value: str, sims: int) -> str:
    opts = f"{sims}" + ("+norm" if cfg.normalize_q else "") + f"+c{cfg.c_puct:g}"
    return f"mcts:{opts}:{policy}" + (f":{value}" if value else "")


def refresh_value(cfg, gen, best, past, value_path) -> tuple:
    """New value data with the best policy, fine-tuned from the current value
    network; returns (path to use, fit summary)."""
    from . import value as V
    data = os.path.join(cfg.out, f"value_data_{gen:03d}.npz")
    V.gen(V.GenConfig(out=data, games=cfg.value_games, players=cfg.players, map=cfg.map,
                      checkpoints=",".join([best, best, cfg.init] + past),
                      workers=cfg.workers, seed=100 + gen))
    cand = os.path.join(cfg.out, f"value_{gen:03d}.pt")
    files = data + ("," + cfg.value_replay if cfg.value_replay else "")
    summary = V.fit(V.FitConfig(data=files, init=value_path, out=cand, mode="full",
                                epochs=cfg.value_epochs, lr=cfg.value_lr, holdout=0.05,
                                seed=gen))
    os.remove(data)
    return (cand if summary["best"] is not None else value_path), summary


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
    value_path = cfg.value_init
    if value_path:
        assert load_value(value_path).obs_size == meta["obs_size"]
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
            jobs = [(cfg, buf.getvalue(), value_path, meta, scripted, past, rng.randrange(1 << 30))
                    for _ in range(cfg.workers)]
            parts, results = [], []
            for data, res in pool.imap_unordered(_play, jobs):
                parts.append(data)
                results.extend(res)
            n_hold = int(round(len(parts) * cfg.policy_holdout))
            held = {k: np.concatenate([p[k] for p in parts[:n_hold]]) for k in parts[0]} if n_hold else None
            gen_data = {k: np.concatenate([p[k] for p in parts[n_hold:]]) for k in parts[0]}
            buffer = (buffer + [gen_data])[-cfg.buffer_gens:]
            t_play = time.time() - t0
            stats = train(net, opt, buffer, cfg)
            path = os.path.join(cfg.out, f"gen{gen:03d}.pt")
            save(path, net, generation=gen, value=value_path)
            t_train = time.time() - t0 - t_play
            held_fit = {}
            if held is not None:
                held_fit = {"held_new": policy_fit(net, held), "held_best": policy_fit(load(best), held)}
            gate_res, gate = per_count(cfg, search_name(cfg, path, value_path, cfg.gate_sims),
                                       [search_name(cfg, best, value_path, cfg.gate_sims)],
                                       cfg.gate_games, gen)
            _, gate_raw = per_count(cfg, "rl:" + path, ["rl:" + best], cfg.gate_raw_games, gen)
            evals, _ = per_count(cfg, "rl:" + path, DEFAULT_OPPONENTS, cfg.eval_games * len(_counts(cfg)),
                                 1000 + gen)
            worst_k = min(evals, key=lambda k: evals[k]["worst"] * k)
            scripted_eval = evals[worst_k]
            accepted = gate >= 1.0                    # search vs best: at least a fair share
            if accepted:
                shutil.copyfile(path, best)
            elif cfg.reset_on_reject:
                net = load(best)
                opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
            t_gate = time.time() - t0 - t_play - t_train
            value_rec = {}
            if cfg.value_every and value_path and gen % cfg.value_every == 0 \
                    and isinstance(load_value(value_path), LegacyValue):   # value.fit's kind only
                new_path, summary = refresh_value(cfg, gen, best, past, value_path)
                value_rec = {"value_refresh": {"kept": new_path != value_path,
                                               "base_mse": summary["base"]["mse"],
                                               "best_mse": (summary["best"] or {}).get("mse")}}
                value_path = new_path
            vs_league = [r["learner"] for r in results if not r["selfplay"]]
            rec = {"gen": gen, "games": len(results), "play_s": round(t_play),
                   "train_s": round(t_train), "gate_s": round(t_gate), **stats, **held_fit,
                   "learner_share_vs_league": round(float(np.mean(vs_league)), 3) if vs_league else None,
                   "vs_best": round(gate, 3), "vs_best_raw": round(gate_raw, 3),
                   "vs_best_by_count": {k: round(r["worst"], 3) for k, r in gate_res.items()},
                   "worst_by_count": {k: round(r["worst"], 3) for k, r in evals.items()},
                   "accepted": accepted, "value": value_path, **value_rec,
                   "worst": round(scripted_eval["worst"], 3),
                   "worst_opponent": scripted_eval["worst_opponent"],
                   "mean": round(scripted_eval["mean"], 3)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
