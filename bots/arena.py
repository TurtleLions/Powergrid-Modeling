"""Play Power Grid games between bots and measure who wins.

    python3 -m bots.arena --agents balanced,builder,tycoon,miser --games 400
    python3 -m bots.arena --focus tycoon --agents balanced,builder,miser --games 400

Without --focus every game seats `players` agents drawn from --agents (with
repetition, in random seats). With --focus one seat is always the focus agent
and the rest are drawn from --agents: that is the "can it win against anyone"
number. Games are independent and run in parallel.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import defaultdict
from typing import Callable, Dict, List, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cpp", "build"))
import pgcore  # noqa: E402

from .heuristic import make  # noqa: E402

Factory = Callable[[str], object]


def play_game(agents: Sequence, rules, seed: int):
    """One game; returns (returns per seat, rounds played)."""
    rng = random.Random(seed)
    state = pgcore.State(rules)
    for seat, agent in enumerate(agents):
        agent.reset(rules, seat, random.Random(rng.random()))
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            a = rng.choices([o for o, _ in outcomes], [p for _, p in outcomes])[0]
        else:
            seat = state.current_player()
            agent = agents[seat]
            message = agent.speak(state)
            if message:
                for other, listener in enumerate(agents):
                    if other != seat:
                        listener.hear(seat, message)
            a = agent.act(state)
            if a not in state.legal_actions():
                raise ValueError(f"{agent.name} played illegal action {a}")
        state.apply_action(a)
    return state.returns(), state.round()


def _worker(job):
    names, players, seed, factory = job
    rules = pgcore.Rules(players=players)
    agents = [factory(n) for n in names]
    returns, rounds = play_game(agents, rules, seed)
    return names, returns, rounds


def run(pool_names: List[str], players: int, games: int, seed: int, focus: str = None,
        factory: Factory = make, procs: int = None, start: str = "fork") -> Dict[str, dict]:
    """start: multiprocessing start method; use "spawn" from a process that has
    already used PyTorch (forking it can deadlock)."""
    rng = random.Random(seed)
    jobs = []
    for g in range(games):
        if focus:
            names = [focus] + [rng.choice(pool_names) for _ in range(players - 1)]
        else:
            names = [rng.choice(pool_names) for _ in range(players)]
        rng.shuffle(names)
        jobs.append((names, players, rng.randrange(1 << 30), factory))
    stats = defaultdict(lambda: {"seats": 0, "wins": 0.0, "sq": 0.0})
    rounds = []
    with mp.get_context(start).Pool(procs or os.cpu_count()) as pool:
        for names, returns, r in pool.imap_unordered(_worker, jobs, chunksize=4):
            rounds.append(r)
            for name, ret in zip(names, returns):
                s = stats[name]
                s["seats"] += 1
                s["wins"] += ret
                s["sq"] += ret * ret
    out = {}
    for name, s in stats.items():
        n = s["seats"]
        mean = s["wins"] / n
        var = max(s["sq"] / n - mean * mean, 0.0)
        out[name] = {"seats": n, "win_rate": mean, "ci95": 1.96 * math.sqrt(var / n)}
    out["_rounds"] = sum(rounds) / len(rounds)
    return out


def report(results: Dict[str, dict], players: int):
    print(f"{'agent':10s} {'seats':>6s} {'win rate':>9s}   (1/{players} = {1 / players:.3f})")
    for name, s in sorted(((k, v) for k, v in results.items() if not k.startswith("_")),
                          key=lambda kv: -kv[1]["win_rate"]):
        print(f"{name:10s} {s['seats']:6d} {s['win_rate']:9.3f} ± {s['ci95']:.3f}")
    print(f"average game length: {results['_rounds']:.1f} rounds")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--agents", default="balanced,builder,tycoon,miser,random")
    ap.add_argument("--focus", default=None)
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    results = run(args.agents.split(","), args.players, args.games, args.seed, args.focus,
                  procs=args.procs)
    report(results, args.players)
    print(f"{args.games} games in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
