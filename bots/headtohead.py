"""One seat of an agent against (n-1) copies of each opponent, at any player count.

    python3 -m bots.headtohead FOCUS PLAYERS GAMES LOG PROCS [OPPONENTS]
    python3 -m bots.headtohead --multi SPEC.json PROCS   # several at once in one pool

OPPONENTS is comma-separated (default: every scripted style). Every game is
appended to LOG (JSONL) as it finishes, so a rerun resumes; the summary gives
the focus agent's win share per opponent with 95% intervals (a fair share is
1/players) and the worst case.
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cpp", "build"))


def _job(args):
    focus, opp, players, seed = args
    import pgcore
    import torch
    from .arena import play_game
    from .heuristic import make
    torch.set_num_threads(1)
    names = [focus] + [opp] * (players - 1)
    random.Random(seed).shuffle(names)
    returns, rounds = play_game([make(n) for n in names], pgcore.Rules(players=players), seed)
    return opp, seed, returns[names.index(focus)], rounds


def run(focus, players, games, log, procs, opponents=None):
    from .evaluate import DEFAULT_OPPONENTS
    opponents = opponents or DEFAULT_OPPONENTS
    done = set()
    if os.path.exists(log):
        for line in open(log):
            try:
                g = json.loads(line)
                done.add((g["opp"], g["seed"]))
            except json.JSONDecodeError:
                pass
    jobs = [(focus, o, players, 1000 * i + k) for i, o in enumerate(opponents) for k in range(games)
            if (o, 1000 * i + k) not in done]
    if jobs:
        with open(log, "a") as f, mp.get_context("spawn").Pool(procs) as pool:
            for opp, seed, r, rounds in pool.imap_unordered(_job, jobs):
                f.write(json.dumps({"opp": opp, "seed": seed, "r": r, "rounds": rounds}) + "\n")
                f.flush()
    res, rounds = {}, []
    for line in open(log):
        try:
            g = json.loads(line)
        except json.JSONDecodeError:
            continue
        res.setdefault(g["opp"], []).append(g["r"])
        rounds.append(g["rounds"])
    return res, rounds


def run_many(specs, procs):
    """Several head-to-heads in one pool (keeps every worker busy across player counts).
    specs: [(focus, players, games per opponent, log, opponents or None)]."""
    from .evaluate import DEFAULT_OPPONENTS
    jobs, files = [], {}
    for focus, players, games, log, opponents in specs:
        done = set()
        if os.path.exists(log):
            for line in open(log):
                try:
                    g = json.loads(line)
                    done.add((g["opp"], g["seed"]))
                except json.JSONDecodeError:
                    pass
        files[log] = open(log, "a")
        jobs += [((focus, o, players, 1000 * i + k), log) for i, o in enumerate(opponents or DEFAULT_OPPONENTS)
                 for k in range(games) if (o, 1000 * i + k) not in done]
    if jobs:
        with mp.get_context("spawn").Pool(procs) as pool:
            for (opp, seed, r, rounds), log in zip(pool.imap(_job, [j for j, _ in jobs]), [l for _, l in jobs]):
                files[log].write(json.dumps({"opp": opp, "seed": seed, "r": r, "rounds": rounds}) + "\n")
                files[log].flush()
    for f in files.values():
        f.close()


def main():
    if sys.argv[1] == "--multi":                     # --multi SPEC.json PROCS
        run_many([tuple(x) for x in json.load(open(sys.argv[2]))], int(sys.argv[3]))
        return
    focus, players, games, log, procs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
    opponents = sys.argv[6].split(",") if len(sys.argv) > 6 else None
    res, rounds = run(focus, players, games, log, procs, opponents)
    rows = sorted((sum(v) / len(v), o, len(v),
                   1.96 * math.sqrt(max(sum(x * x for x in v) / len(v) - (sum(v) / len(v)) ** 2, 0) / len(v)))
                  for o, v in res.items())
    print(f"{players} players (fair share {1 / players:.3f}), mean game length {sum(rounds) / len(rounds):.1f} rounds")
    for m, o, n, ci in rows:
        print(f"  vs {o:12s} {m:.3f} ± {ci:.3f}  ({n} games)")
    print(f"  WORST {rows[0][0]:.3f} (vs {rows[0][1]}), mean {sum(r[0] for r in rows) / len(rows):.3f}")


if __name__ == "__main__":
    main()
