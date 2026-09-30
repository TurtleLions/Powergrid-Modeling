"""Pick the final bot from a set of checkpoints.

    python3 -m bots.select runs/league3 --games 400

Candidates are the run's best.pt and hall-of-fame members (plus any extra
checkpoints given with --extra). Each is scored twice:

  worst    worst-case win rate over the scripted opponent types, with more
           games than the in-training evaluations (bots/evaluate.py);
  peers    win rate in mixed tables of candidates only, i.e. against the
           other strong versions (1/N is even).

The champion is the candidate with the best worst case among those at least
even with their peers (peers >= 1/N minus the 95% interval), so it is both
robust against every scripted style and not weaker than its siblings. It is
copied to <run>/champion.pt and the table is written to <run>/selection.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

from . import arena
from .evaluate import DEFAULT_OPPONENTS, worst_case


def candidates(run: str, extra: list) -> list:
    paths = []
    best = os.path.join(run, "best.pt")
    if os.path.exists(best):
        paths.append(best)
    index = os.path.join(run, "hof", "index.json")
    if os.path.exists(index):
        paths += [m["path"] for m in json.load(open(index))]
    paths += extra
    seen, out = set(), []
    for p in paths:
        key = os.path.realpath(p)
        if key not in seen and os.path.exists(p):
            seen.add(key)
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run", help="training run directory")
    ap.add_argument("--extra", default="", help="comma-separated extra checkpoints")
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--games", type=int, default=400, help="games per scripted opponent type")
    ap.add_argument("--peer-games", type=int, default=2000, help="mixed games among candidates")
    ap.add_argument("--procs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    t0 = time.time()
    cands = candidates(args.run, [p for p in args.extra.split(",") if p])
    print(f"{len(cands)} candidates")
    rows = []
    for i, path in enumerate(cands):
        res = worst_case("rl:" + path, DEFAULT_OPPONENTS, args.players, args.games,
                         args.seed + i, args.procs, start="spawn")
        rows.append({"path": path, "worst": res["worst"], "worst_opponent": res["worst_opponent"],
                     "mean": res["mean"],
                     "worst_ci95": res["per_opponent"][res["worst_opponent"]]["ci95"]})
        print(f"  {path}: worst {res['worst']:.3f} (vs {res['worst_opponent']}), "
              f"mean {res['mean']:.3f}", flush=True)
    names = ["rl:" + p for p in cands]
    peers = arena.run(names, args.players, args.peer_games, args.seed, procs=args.procs,
                      start="spawn")
    even = 1.0 / args.players
    for row, name in zip(rows, names):
        row["peers"] = peers[name]["win_rate"]
        row["peers_ci95"] = peers[name]["ci95"]
    eligible = [r for r in rows if r["peers"] >= even - r["peers_ci95"]] or rows
    champ = max(eligible, key=lambda r: (r["worst"], r["peers"]))
    print(f"\n{'candidate':42s} {'worst':>7s} {'mean':>6s} {'peers':>6s}")
    for r in sorted(rows, key=lambda r: -r["worst"]):
        mark = "  <- champion" if r is champ else ("" if r in eligible else "  (weaker than peers)")
        print(f"{r['path']:42s} {r['worst']:7.3f} {r['mean']:6.3f} {r['peers']:6.3f}{mark}")
    dest = os.path.join(args.run, "champion.pt")
    shutil.copyfile(champ["path"], dest)
    json.dump({"champion": champ["path"], "rows": rows, "players": args.players,
               "games": args.games, "peer_games": args.peer_games},
              open(os.path.join(args.run, "selection.json"), "w"), indent=1)
    print(f"\nchampion: {champ['path']} -> {dest}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
