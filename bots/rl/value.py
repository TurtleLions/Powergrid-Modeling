"""Value head fitting: generate labelled positions fast, fit, measure held out.

    python3 -m bots.rl.value gen --out runs/value1/data.npz --games 20000
    python3 -m bots.rl.value fit --data runs/value1/data.npz --init runs/league3/champion.pt \
        --out runs/value1/fit_full.pt --mode full

Why: search scores leaves with the value head, and the value heads we have
barely predict the outcome (held-out correlation ~0.33 with the final win
share; exit2's head was worse than a constant). Expert iteration trained the
value on every decision of ~540 games, and positions from one game share one
0/1 label, so the head memorised games instead of learning positions.

gen   plays raw-network and scripted games (much faster than searched games),
      keeps --per-game random decision positions per game and encodes each
      from EVERY seat's point of view (search evaluates all seats), labelled
      with that seat's final win share and its final margin over the best
      rival: powered cities, plus up to half a city for money, which breaks
      ties (an auxiliary, lower-variance target).
fit   trains a value network with a split by game, so the held-out numbers
      measure positions from games never seen in training:
        head   only the value layer (body frozen: the policy is unchanged)
        full   body and value layer (a separate value network for search)
      --out is written only for an epoch that beats --init on the held-out
      games, so fine-tuning never replaces a value network with a worse one.
      --aux-coef weights the margin target (a second output, dropped when the
      network is used as a plain value head).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cpp", "build"))
import pgcore  # noqa: E402

from ..heuristic import RANDOMIZED, STYLES, make  # noqa: E402
from .features import AbstractActions, Encoder  # noqa: E402
from .model import RLAgent, load, save  # noqa: E402

CHECKPOINTS = ["runs/league3/champion.pt", "runs/exit2/best.pt", "runs/league_v2b/best.pt"] + \
    [f"runs/league3/hof/it{i:05d}.pt" for i in (550, 675, 800, 875, 900, 925)]
MARGIN_SCALE = 4.0      # powered cities per unit of the auxiliary target


@dataclasses.dataclass
class GenConfig:
    out: str = "runs/value/data.npz"
    games: int = 20000
    per_game: int = 24
    players: int = 4
    map: str = "germany"
    p_all_rl: float = 0.5         # every seat a checkpoint (else rl and scripted mixed)
    p_sampled: float = 0.25       # an rl seat samples from its policy instead of argmax
    checkpoints: str = ",".join(CHECKPOINTS)
    agents: str = ""              # instead: every seat drawn from these make() names
                                  # (e.g. mcts:..., for positions from searched games)
    workers: int = max(1, (os.cpu_count() or 2) - 2)
    seed: int = 0


def _gen(job):
    cfg, seeds = job
    torch.set_num_threads(1)
    rules = pgcore.Rules(players=cfg.players, map=cfg.map)
    enc = Encoder(rules, 2)
    nets = {} if cfg.agents else {p: load(p) for p in cfg.checkpoints.split(",")}
    scripted = list(STYLES) + [RANDOMIZED]
    n = cfg.players
    obs, win, margin, game, rnd = [], [], [], [], []
    acts = AbstractActions(rules)
    pol_obs, pol_mask, pol_pi, pol_game = [], [], [], []     # search targets (visit distributions)
    for gid in seeds:
        rng = random.Random(gid)
        agents = []
        all_rl = rng.random() < cfg.p_all_rl
        for seat in range(n):
            if cfg.agents:
                agents.append(make(rng.choice(cfg.agents.split(","))))
            elif all_rl or rng.random() < 0.5:
                p = rng.choice(list(nets))
                agents.append(RLAgent(p, greedy=rng.random() >= cfg.p_sampled, net=nets[p]))
            else:
                agents.append(make(rng.choice(scripted)))
            agents[-1].reset(rules, seat, random.Random(rng.random()))
        state = pgcore.State(rules)
        snaps = []                       # (state clone, round) at decision points
        while not state.is_terminal():
            if state.is_chance_node():
                outs = state.chance_outcomes()
                a = rng.choices([o for o, _ in outs], [p for _, p in outs])[0]
                for ag in agents:
                    ag.observe(state, pgcore.CHANCE, a)
                state.apply_action(a)
                continue
            snaps.append(state.clone())
            seat = state.current_player()
            a = agents[seat].act(state)
            visits = getattr(agents[seat], "last_visits", None)
            if visits:                                   # a search agent searched this decision
                mask, _ = acts.mask_and_map(state.legal_actions())
                pi = np.zeros(acts.n, np.float32)
                total = sum(visits.values())
                for x, c in visits.items():
                    pi[x] = c / total
                f = state.features(seat) if hasattr(state, "features") else \
                    enc._encode(state, json.loads(state.to_json()), seat)
                pol_obs.append(f.astype(np.float16))
                pol_mask.append(mask)
                pol_pi.append(pi)
                pol_game.append(gid)
            for ag in agents:                            # opponent modelling and tree reuse
                ag.observe(state, seat, a)
            state.apply_action(a)
        returns = state.returns()
        final = json.loads(state.to_json())["players"]
        score = [(p["powered"], p["money"]) for p in final]     # the engine's ranking
        for s in rng.sample(snaps, min(cfg.per_game, len(snaps))):
            v = json.loads(s.to_json())
            for seat in range(n):
                obs.append(enc._encode(s, v, seat).astype(np.float16))
                win.append(returns[seat])
                rival = max(score[o] for o in range(n) if o != seat)
                tie = 0.5 * np.tanh((score[seat][1] - rival[1]) / 30.0)   # money breaks ties
                margin.append((score[seat][0] - rival[0] + tie) / MARGIN_SCALE)
                game.append(gid)
                rnd.append(v["round"])
    out = {"obs": np.stack(obs), "win": np.array(win, np.float32),
           "margin": np.array(margin, np.float32), "game": np.array(game, np.int64),
           "round": np.array(rnd, np.int16)}
    if pol_obs:
        out.update(pol_obs=np.stack(pol_obs), pol_mask=np.stack(pol_mask), pol_pi=np.stack(pol_pi),
                   pol_game=np.array(pol_game, np.int64))
    return out


def gen(cfg: GenConfig):
    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    t0 = time.time()
    ids = list(range(cfg.seed * 10_000_000, cfg.seed * 10_000_000 + cfg.games))
    chunks = [ids[i::cfg.workers * 8] for i in range(cfg.workers * 8)]
    with mp.get_context("spawn").Pool(cfg.workers) as pool:
        parts = pool.map(_gen, [(cfg, c) for c in chunks if c])
    keys = dict.fromkeys(k for p in parts for k in p)          # policy targets only from search agents
    data = {k: np.concatenate([p[k] for p in parts if k in p]) for k in keys}
    np.savez(cfg.out, players=cfg.players, **data)
    print(f"{cfg.games} games, {len(data['win'])} samples, {time.time() - t0:.0f}s -> {cfg.out}")


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class FitConfig:
    data: str = "runs/value/data.npz"   # comma-separated files (game ids must not overlap)
    init: str = "runs/league3/champion.pt"
    out: str = ""
    mode: str = "full"            # head | full
    aux_coef: float = 0.5
    holdout: float = 0.1          # fraction of GAMES held out
    epochs: int = 20
    batch: int = 1024
    lr: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 0


class ValueFit(nn.Module):
    """The init network's body and value layer, plus a margin output."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.aux = nn.Linear(net.value.in_features, 1)

    def forward(self, obs):
        h = self.net.body(obs)
        return self.net.value(h).squeeze(-1), self.aux(h).squeeze(-1)


@torch.no_grad()
def metrics(model, obs, win, margin, game, rounds, n_players) -> dict:
    model.eval()
    vs, ms = [], []
    for i in range(0, len(obs), 16384):
        v, m = model(torch.from_numpy(obs[i:i + 16384].astype(np.float32)))
        vs.append(v.numpy())
        ms.append(m.numpy())
    v, m = np.concatenate(vs), np.concatenate(ms)
    out = {"mse": float(np.mean((v - win) ** 2)),
           "corr": float(np.corrcoef(v, win)[0, 1]),
           "margin_corr": float(np.corrcoef(m, margin)[0, 1])}
    # does the seat with the highest value win? (samples come in blocks of n seats)
    k = len(v) // n_players * n_players
    vb, wb = v[:k].reshape(-1, n_players), win[:k].reshape(-1, n_players)
    out["pick_winner"] = float(wb[np.arange(len(vb)), vb.argmax(1)].mean())
    rb = rounds[:k].reshape(-1, n_players)[:, 0]
    for lo, hi in ((1, 3), (4, 7), (8, 12), (13, 99)):
        sel = (rb >= lo) & (rb <= hi)
        if sel.any():
            out[f"pick_r{lo}-{hi}"] = round(float(wb[sel][np.arange(sel.sum()), vb[sel].argmax(1)].mean()), 3)
    model.train()
    return {k: round(x, 4) if isinstance(x, float) else x for k, x in out.items()}


def fit(cfg: FitConfig):
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    ds = [np.load(f) for f in cfg.data.split(",")]
    obs, win, margin, game, rounds = (np.concatenate([d[k] for d in ds])
                                      for k in ("obs", "win", "margin", "game", "round"))
    n_players = int(ds[0]["players"])
    assert all(int(d["players"]) == n_players for d in ds)
    games = np.unique(game)
    rng = np.random.default_rng(cfg.seed)
    held = set(rng.choice(games, int(len(games) * cfg.holdout), replace=False).tolist())
    te = np.isin(game, list(held))
    tr = ~te
    net = load(cfg.init)
    model = ValueFit(net)
    aux_state = torch.load(cfg.init, map_location="cpu", weights_only=False).get("aux_state")
    if aux_state is not None:                     # fine-tuning a fitted value network
        model.aux.load_state_dict(aux_state)
    base = metrics(model, obs[te], win[te], margin[te], game[te], rounds[te], n_players)
    print(json.dumps({"epoch": 0, "held_out": base, "const_mse": round(float(np.var(win[te])), 4)}), flush=True)
    if cfg.mode == "head":
        for p in net.parameters():
            p.requires_grad_(False)
        params = list(net.value.parameters()) + list(model.aux.parameters())
        for p in params:
            p.requires_grad_(True)
    else:
        params = list(net.body.parameters()) + list(net.value.parameters()) + list(model.aux.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    tr_idx = np.flatnonzero(tr)
    steps = len(tr_idx) // cfg.batch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs * steps)
    best, best_m = base["mse"], None       # save only an improvement on the init network
    for epoch in range(1, cfg.epochs + 1):
        rng.shuffle(tr_idx)
        tot = np.zeros(2)
        for s in range(steps):
            i = np.sort(tr_idx[s * cfg.batch:(s + 1) * cfg.batch])
            x = torch.from_numpy(obs[i].astype(np.float32))
            v, m = model(x)
            lv = F.mse_loss(v, torch.from_numpy(win[i]))
            lm = F.mse_loss(m, torch.from_numpy(margin[i]))
            loss = lv + cfg.aux_coef * lm
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += (lv.item(), lm.item())
        held_m = metrics(model, obs[te], win[te], margin[te], game[te], rounds[te], n_players)
        rec = {"epoch": epoch, "train_mse": round(tot[0] / steps, 4),
               "train_margin_mse": round(tot[1] / steps, 4), "held_out": held_m}
        print(json.dumps(rec), flush=True)
        if held_m["mse"] < best:
            best, best_m = held_m["mse"], held_m
            if cfg.out:
                save(cfg.out, net, value_fit=dataclasses.asdict(cfg), held_out=held_m, epoch=epoch,
                     aux_state={k: v.clone() for k, v in model.aux.state_dict().items()})
    return {"base": base, "best": best_m}


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    Cfg = {"gen": GenConfig, "fit": FitConfig}.get(cmd)
    if Cfg is None:
        print(__doc__)
        sys.exit(2)
    ap = argparse.ArgumentParser(prog=f"value {cmd}")
    for f in dataclasses.fields(Cfg):
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    cfg = Cfg(**{k.replace("-", "_"): v for k, v in vars(ap.parse_args(sys.argv[2:])).items()})
    (gen if cmd == "gen" else fit)(cfg)


if __name__ == "__main__":
    main()
