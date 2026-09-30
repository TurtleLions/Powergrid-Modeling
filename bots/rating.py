"""Elo-style ladder for multiplayer Power Grid.

    python3 -m bots.rating --agents balanced,builder,rl:runs/league3/champion.pt,... --games 6000
    python3 -m bots.rating --runs runs/league3 --games 6000     # every checkpoint of a run

Model (Plackett-Luce, top-1): each agent has a strength s; at a table T the
chance that agent i wins is exp(s_i) / sum_{j in T} exp(s_j). Shared wins count
fractionally. Strengths are fitted by maximum likelihood over many mixed games
of distinct agents in random seats and reported on the Elo scale
(400 * log10), anchored so `random` = 0 (or the mean = 0 without it).

How to read it: at a 4-player table of equals everyone wins 25%. An agent
200 points above three equal opponents wins exp(200/173.7) / (that + 3) ~ 50%.
Intervals are 95% bootstrap over games. Unlike win rates against a fixed field,
ratings keep separating agents that are all much stronger than the field.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import random
import time
from typing import Dict, List, Sequence

import numpy as np

from .arena import _worker
from .heuristic import RANDOMIZED, STYLES, make

ELO = 400.0 / math.log(10.0)


def play(agents: Sequence[str], players: int, games: int, seed: int, procs: int = None,
         start: str = "spawn") -> List[tuple]:
    """Mixed games of distinct agents; returns [(names, returns)]."""
    rng = random.Random(seed)
    jobs = []
    counts = {a: 0 for a in agents}
    for _ in range(games):
        # favour agents that have played least, so every agent gets enough games
        pool = sorted(agents, key=lambda a: (counts[a], rng.random()))
        names = pool[:players]
        for a in names:
            counts[a] += 1
        rng.shuffle(names)
        jobs.append((names, players, rng.randrange(1 << 30), make))
    out = []
    with mp.get_context(start).Pool(procs or os.cpu_count()) as pool:
        for names, returns, _ in pool.imap_unordered(_worker, jobs, chunksize=4):
            out.append((names, returns))
    return out


def fit(results: List[tuple], agents: Sequence[str], iters: int = 500, l2: float = 1e-3,
        anchor: str = "random") -> np.ndarray:
    """Maximum-likelihood strengths (natural-log scale) by Newton-free gradient ascent."""
    idx = {a: i for i, a in enumerate(agents)}
    tables = np.array([[idx[n] for n in names] for names, _ in results])
    targets = np.array([r for _, r in results], dtype=np.float64)
    s = np.zeros(len(agents))
    lr = 1.0
    for _ in range(iters):
        logits = s[tables]
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        grad = np.zeros(len(agents))
        np.add.at(grad, tables, targets - p)       # d log-lik / d s
        grad = grad / len(results) - l2 * s
        s += lr * grad * len(agents)
    s -= s[idx[anchor]] if anchor in idx else s.mean()
    return s


def ladder(results: List[tuple], agents: Sequence[str], boot: int = 100, seed: int = 0) -> Dict:
    s = fit(results, agents)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(boot):
        pick = rng.integers(0, len(results), len(results))
        samples.append(fit([results[i] for i in pick], agents, iters=300))
    samples = np.array(samples)
    games = {a: 0 for a in agents}
    wins = {a: 0.0 for a in agents}
    for names, returns in results:
        for n, r in zip(names, returns):
            games[n] += 1
            wins[n] += r
    rows = []
    for i, a in enumerate(agents):
        lo, hi = np.percentile(samples[:, i], [2.5, 97.5])
        rows.append({"agent": a, "elo": float(s[i] * ELO), "lo": float(lo * ELO),
                     "hi": float(hi * ELO), "games": games[a],
                     "win_rate": wins[a] / max(games[a], 1)})
    rows.sort(key=lambda r: -r["elo"])
    return {"rows": rows, "games": len(results)}


def run_checkpoints(run: str) -> List[str]:
    paths = []
    for name in ("champion.pt", "best.pt"):
        p = os.path.join(run, name)
        if os.path.exists(p):
            paths.append(p)
    index = os.path.join(run, "hof", "index.json")
    if os.path.exists(index):
        paths += [m["path"] for m in json.load(open(index))]
    return paths


def dedupe(agents: List[str]) -> List[str]:
    """Drop repeats, including checkpoints that are byte-identical copies
    (best.pt and champion.pt are copies of hall-of-fame members)."""
    import hashlib
    seen, out = set(), []
    for a in agents:
        key = a
        if a.startswith("rl:") and os.path.exists(a[3:]):
            key = hashlib.md5(open(a[3:], "rb").read()).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append(a)
    return out


def report(lad: Dict):
    print(f"{'agent':44s} {'elo':>7s} {'95% interval':>17s} {'games':>6s} {'win%':>6s}")
    for r in lad["rows"]:
        print(f"{r['agent']:44s} {r['elo']:7.0f} [{r['lo']:6.0f}, {r['hi']:6.0f}] "
              f"{r['games']:6d} {100 * r['win_rate']:5.1f}")
    print(f"{lad['games']} games")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--agents", default="", help="comma-separated agents (styles, random, rl:<ckpt>)")
    ap.add_argument("--runs", default="", help="comma-separated run dirs: add their checkpoints")
    ap.add_argument("--no-scripted", action="store_true", help="do not add the scripted styles")
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--games", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=None)
    ap.add_argument("--out", default="", help="write the ladder as JSON here")
    args = ap.parse_args()
    agents = [a for a in args.agents.split(",") if a]
    if not args.no_scripted:
        agents += list(STYLES) + [RANDOMIZED, "random"]
    for run in [r for r in args.runs.split(",") if r]:
        agents += ["rl:" + p for p in run_checkpoints(run)]
    agents = dedupe(agents)
    t0 = time.time()
    results = play(agents, args.players, args.games, args.seed, args.procs)
    lad = ladder(results, agents)
    report(lad)
    print(f"{len(agents)} agents, {time.time() - t0:.0f}s")
    if args.out:
        json.dump(lad, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
