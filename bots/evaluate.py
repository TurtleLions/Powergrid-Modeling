"""Worst-case evaluation: can an agent win against every kind of opponent?

    python3 -m bots.evaluate rl:runs/league/best.pt --games 200
    python3 -m bots.evaluate builder --opponents balanced,blocker --games 400

For each opponent type, the agent under test takes one seat and every other
seat is that opponent type (seats shuffled). The headline number is the WORST
win rate across opponent types: a bot that crushes weak styles but loses to
one strong style is not robust. With N players, 1/N is an average result.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import random
import time
from collections import defaultdict
from typing import Dict, List

from .arena import _worker
from .heuristic import RANDOMIZED, STYLES, make

DEFAULT_OPPONENTS = list(STYLES) + [RANDOMIZED]


def worst_case(focus: str, opponents: List[str] = None, players: int = 4, games: int = 200,
               seed: int = 0, procs: int = None, start: str = "fork") -> Dict:
    """Win rate of `focus` against each opponent type; see the module docstring.

    start: multiprocessing start method ("spawn" when called from a process
    that already uses PyTorch).
    """
    opponents = opponents or DEFAULT_OPPONENTS
    rng = random.Random(seed)
    tags, jobs = [], []
    for opp in opponents:
        for _ in range(games):
            names = [focus] + [opp] * (players - 1)
            rng.shuffle(names)
            tags.append(opp)
            jobs.append((names, players, rng.randrange(1 << 30), make))
    acc = defaultdict(lambda: [0, 0.0, 0.0])
    with mp.get_context(start).Pool(procs or os.cpu_count()) as pool:
        for tag, (names, returns, _) in zip(tags, pool.imap(_worker, jobs, chunksize=8)):
            r = returns[names.index(focus)]
            a = acc[tag]
            a[0] += 1
            a[1] += r
            a[2] += r * r
    per = {}
    for opp in opponents:
        n, s, sq = acc[opp]
        mean = s / n
        per[opp] = {"win_rate": mean, "ci95": 1.96 * math.sqrt(max(sq / n - mean * mean, 0) / n)}
    worst = min(per, key=lambda o: per[o]["win_rate"])
    return {"focus": focus, "players": players, "per_opponent": per,
            "worst_opponent": worst, "worst": per[worst]["win_rate"],
            "mean": sum(p["win_rate"] for p in per.values()) / len(per)}


def report(res: Dict):
    n = res["players"]
    print(f"{res['focus']}: 1 seat vs {n - 1} copies of each opponent (1/{n} = {1 / n:.3f})")
    for opp, p in sorted(res["per_opponent"].items(), key=lambda kv: kv[1]["win_rate"]):
        print(f"  vs {opp:12s} {p['win_rate']:.3f} ± {p['ci95']:.3f}")
    print(f"  WORST {res['worst']:.3f} (vs {res['worst_opponent']}), mean {res['mean']:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("focus", help='agent under test: a style, "random", or rl:<checkpoint.pt>')
    ap.add_argument("--opponents", default=",".join(DEFAULT_OPPONENTS),
                    help="comma-separated opponent types (styles or rl:<checkpoint.pt>)")
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--games", type=int, default=200, help="games per opponent type")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    res = worst_case(args.focus, args.opponents.split(","), args.players, args.games, args.seed,
                     args.procs, start="spawn" if args.focus.startswith("rl:") else "fork")
    report(res)
    print(f"{args.games * len(res['per_opponent'])} games in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
