"""A league of search agents: search-vs-search data, a value network trained on
it, a head-to-head gate against the current best, a per-player-count Elo
ladder as the benchmark, and exploiters every few iterations.

    taskset -c 0-23 python3 -m bots.rl.search_league --dir runs/sleague --workers 24

Agents are search agents (bots/search.py): `policy` + `value` + search options.
Each iteration:

 1. League games at 3-6 players. Each seat is drawn from: the current best
    (--w-best), past bests in the pool (--w-pool), scripted styles
    (--w-scripted; kept for robustness, not as the target). Data is labelled
    as in bots/rl/value.py gen.
 2. A new value network (bots/rl/value_td.py: out-of-fold teachers, TD(lambda)
    targets) on all data so far, league games oversampled (--oversample).
 3. Gate: 1 x candidate vs (n-1) x best at every player count; accepted if its
    share x players, pooled over counts, is at least 1 (a fair share). The old
    best then joins the pool.
 4. Benchmark: an Elo ladder (bots/rating.py) per player count over the best,
    the pool and anchors. The scripted worst case is a floor, not the metric.
 5. Every --exploit-every iterations, an exploiter: a value network fine-tuned
    only on games against the frozen best (one exploiter seat, the rest the
    best), twice in a row, so its search learns what wins against that
    particular agent. If it beats the best head to head (lower 95% bound of
    share x players above 1), the weakness is real: its games stay in the
    training data and it joins the pool, so the next value network learns to
    handle those lines.

Everything is resumable: state.json records what is done, data comes in
chunk files, ladders and head-to-heads log every game.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import math
import os
import subprocess
import sys
import time

PY = sys.executable
COUNTS = (3, 4, 5, 6)
SCRIPTED = ("builder", "balanced", "randomized")


@dataclasses.dataclass
class Config:
    dir: str = "runs/sleague"
    workers: int = 24
    policy: str = "runs/exit5/best.pt"
    value: str = "runs/value5/mlp_l0.5.pt"
    options: str = "100+norm+c3+np+om+reuse+fuel"
    pool: str = "runs/value4/mlp_l0.5.pt,runs/value3/mlp_l0.5.pt"  # starting pool (value nets)
    base_from: str = "runs/value5" # always train on this value_td dir's sources, with its split groups
    seed_data: str = ""            # earlier search games to add to the league data
    games_per_count: int = 500     # league games per player count per iteration
    w_best: int = 12
    w_pool: int = 2                # per pool member (at most --pool-size most recent)
    pool_size: int = 3
    w_scripted: int = 1            # per scripted style
    oversample: int = 3
    epochs: int = 4
    gate_games: int = 150          # per player count
    ladder_games: int = 400        # per player count
    exploit_every: int = 2
    exploit_games: int = 300       # per player count per exploiter step
    exploit_steps: int = 2
    iterations: int = 100


def label(name: str) -> str:
    """Short ladder label: options, policy run and value file."""
    if not name.startswith("mcts:"):
        return name
    _, opts, policy, value = name.split(":", 3)
    run = lambda p: p.replace("runs/", "").replace("/mlp_l0.5.pt", "").replace("/value.pt", "").replace("/best.pt", "")
    return f"[{opts}] {run(policy)} {run(value)}"


def agent(cfg, value, policy=None):
    return f"mcts:{cfg.options}:{policy or cfg.policy}:{value}"


class League:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        os.makedirs(cfg.dir, exist_ok=True)
        self.path = os.path.join(cfg.dir, "state.json")
        if os.path.exists(self.path):
            self.s = json.load(open(self.path))
        else:
            self.s = {"iteration": 0, "best": cfg.value, "pool": [p for p in cfg.pool.split(",") if p],
                      "league_data": [p for p in cfg.seed_data.split(",") if p], "done": [],
                      "history": []}
            self.save()

    # ---- bookkeeping ------------------------------------------------------
    def save(self):
        json.dump(self.s, open(self.path + ".tmp", "w"), indent=1)
        os.replace(self.path + ".tmp", self.path)

    def log(self, msg):
        line = f"{time.strftime('%m-%d %H:%M')} {msg}"
        print(line, flush=True)
        with open(os.path.join(self.cfg.dir, "report.md"), "a") as f:
            f.write(line + "\n")

    def step(self, key, fn):
        """Run fn once; `key` marks it done in state.json (resumable)."""
        if key in self.s["done"]:
            return
        fn()
        self.s["done"].append(key)
        self.save()

    def run(self, args, log=None):
        out = open(log, "a") if log else subprocess.DEVNULL
        r = subprocess.run([PY] + args, stdout=out, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError(f"failed ({r.returncode}): {' '.join(args[:4])} ... (see {log})")

    # ---- pieces -----------------------------------------------------------
    def gen(self, out, players, games, agents, seed):
        if os.path.exists(out):
            return
        self.run(["-m", "bots.rl.value", "gen", "--out", out + ".tmp.npz", "--players", str(players),
                  "--games", str(games), "--workers", str(self.cfg.workers), "--seed", str(seed),
                  "--agents", ",".join(agents)], log=out + ".log")
        os.replace(out + ".tmp.npz", out)

    def h2h(self, name, new, old, games):
        """share x players of 1 x new vs (n-1) x old, per count and pooled."""
        res, pooled = {}, []
        for n in COUNTS:
            log = os.path.join(self.cfg.dir, f"h2h_{name}_p{n}.jsonl")
            self.run(["-m", "bots.headtohead", new, str(n), str(games), log, str(self.cfg.workers), old],
                     log=log[:-6] + ".out")
            r = [json.loads(l)["r"] * n for l in open(log) if l.startswith("{")]
            res[n] = sum(r) / len(r)
            pooled += r
        m = sum(pooled) / len(pooled)
        sd = math.sqrt(max(sum(x * x for x in pooled) / len(pooled) - m * m, 0) / len(pooled))
        return res, m, m - 1.96 * sd, m + 1.96 * sd

    def base(self):
        """(sources, split groups) of --base-from: training on them with the same groups keeps
        that run's held-out games (e.g. runs/value5/held_searched_p*.npz) unseen."""
        if not self.cfg.base_from:
            return [], []
        info = json.load(open(os.path.join(self.cfg.base_from, "prep.json")))
        return info["sources"], info.get("split_groups", [info["sources"]])

    def train_value(self, name, sources, oversample_from, init="", teacher="", groups=None):
        """prep -> (teachers) -> TD targets -> student; returns the checkpoint path."""
        d = os.path.join(self.cfg.dir, name)
        ck = os.path.join(d, "value.pt")
        if os.path.exists(ck):
            return ck
        log = os.path.join(self.cfg.dir, f"{name}.log")
        ev = ",".join(sorted(glob.glob("runs/value5/held_searched_p*.npz")))
        groups = ";".join(",".join(g) for g in (groups or [[s] for s in sources]))
        self.run(["-m", "bots.rl.value_td", "prep", "--dir", d, "--sources", ",".join(sources),
                  "--split-groups", groups], log=log)
        common = ["--dir", d, "--search-eval", ev, "--threads", str(self.cfg.workers)]
        if teacher:
            self.run(["-m", "bots.rl.value_td", "targets", "--dir", d, "--lam", "0.5", "--teacher", teacher],
                     log=log)
        else:
            half = str(max(1, self.cfg.workers // 2))
            procs = [subprocess.Popen([PY, "-m", "bots.rl.value_td", "train", *common[:4], "--threads", half,
                                       "--name", f"teach{k}", "--fold", str(k), "--arch", "mlp", "--epochs", "2"],
                                      stdout=open(log, "a"), stderr=subprocess.STDOUT) for k in (0, 1)]
            if any(p.wait() for p in procs):
                raise RuntimeError(f"teachers failed (see {log})")
            self.run(["-m", "bots.rl.value_td", "targets", "--dir", d, "--lam", "0.5"], log=log)
        args = ["-m", "bots.rl.value_td", "train", *common, "--name", "value", "--target", "target_l0.5.npy",
                "--arch", "mlp", "--epochs", str(self.cfg.epochs if not init else 2)]
        if self.cfg.oversample > 1 and oversample_from < len(sources):
            args += ["--oversample", str(self.cfg.oversample), "--oversample-from", str(oversample_from)]
        if init:
            args += ["--init", init, "--lr", "3e-4"]
        self.run(args, log=log)
        return ck

    def ladder(self, it):
        agents = [agent(self.cfg, self.s["best"])] + [agent(self.cfg, v) for v in self.s["pool"][-4:]] \
            + ["mcts:100+norm+c3:runs/exit5/best.pt:runs/value4/mlp_l0.5.pt",   # 2026-10-01's agent
               "mcts:100+norm+c3:runs/exit3/best.pt:runs/value1/big_aux05.pt",  # 2026-09-30's agent
               "rl:runs/league3/champion.pt"]                                   # raw-network anchor
        agents = list(dict.fromkeys(agents))
        rows = {}
        for n in COUNTS:
            if len(agents) < n:
                continue
            log = os.path.join(self.cfg.dir, f"ladder_it{it}_p{n}.jsonl")
            out = os.path.join(self.cfg.dir, f"ladder_it{it}_p{n}.json")
            self.run(["-m", "bots.rating", "--no-scripted", "--agents", ",".join(agents), "--players", str(n),
                      "--games", str(self.cfg.ladder_games), "--procs", str(self.cfg.workers),
                      "--seed", str(100 * it + n), "--games-log", log, "--out", out], log=out[:-5] + ".out")
            rows[n] = {r["agent"]: (round(r["elo"]), round(r["lo"]), round(r["hi"])) for r in json.load(open(out))["rows"]}
        return rows

    # ---- one iteration ------------------------------------------------------
    def iterate(self):
        cfg, it = self.cfg, self.s["iteration"] + 1
        best = self.s["best"]
        pool = self.s["pool"][-cfg.pool_size:]
        seats = [agent(cfg, best)] * cfg.w_best + [agent(cfg, v) for v in pool for _ in range(cfg.w_pool)] \
            + [s for s in SCRIPTED for _ in range(cfg.w_scripted)]
        self.log(f"iteration {it}: best {best}; pool {pool}")

        def data():
            for n in COUNTS:
                f = os.path.join(cfg.dir, "data", f"league_it{it}_p{n}.npz")
                os.makedirs(os.path.dirname(f), exist_ok=True)
                self.gen(f, n, cfg.games_per_count, seats, 7000 + 100 * it + n)
                if f not in self.s["league_data"]:
                    self.s["league_data"].append(f)
                    self.save()
            self.log(f"  league games: {cfg.games_per_count} per count ({len(self.s['league_data'])} league files)")
        self.step(f"it{it}:data", data)

        def value():
            base, groups = self.base()
            league = self.s["league_data"]
            ck = self.train_value(f"value_it{it}", base + league, len(base),
                                  groups=groups + [[f] for f in league])
            self.s[f"candidate_it{it}"] = ck
        self.step(f"it{it}:value", value)
        cand = self.s[f"candidate_it{it}"]

        def gate():
            res, m, lo, hi = self.h2h(f"gate_it{it}", agent(cfg, cand), agent(cfg, best), cfg.gate_games)
            ok = m >= 1.0
            self.log(f"  gate: candidate vs best, share x players {m:.3f} [{lo:.3f}, {hi:.3f}] "
                     + " ".join(f"{n}p {v:.2f}" for n, v in res.items()) + (" -> ACCEPTED" if ok else " -> rejected"))
            if ok:
                self.s["pool"].append(best)
                self.s["best"] = cand
            self.s["history"].append({"iteration": it, "candidate": cand, "gate": round(m, 3),
                                      "lo": round(lo, 3), "accepted": ok})
        self.step(f"it{it}:gate", gate)

        def bench():
            rows = self.ladder(it)
            for n, r in rows.items():
                self.log(f"  ladder {n}p: " + "; ".join(f"{label(a)} "
                                                      f"{e[0]} [{e[1]},{e[2]}]" for a, e in
                                                      sorted(r.items(), key=lambda kv: -kv[1][0])))
        self.step(f"it{it}:ladder", bench)

        if cfg.exploit_every and it % cfg.exploit_every == 0:
            self.step(f"it{it}:exploit", lambda: self.exploit(it))
        self.s["iteration"] = it
        self.save()

    def exploit(self, it):
        cfg, best = self.cfg, self.s["best"]
        x_value, files = best, []
        for j in range(1, cfg.exploit_steps + 1):
            x = agent(cfg, x_value)
            for n in COUNTS:                        # one exploiter seat on average, the rest the best
                seats = [agent(cfg, best)] * (n - 1) + [x]
                f = os.path.join(cfg.dir, "data", f"exploit_it{it}_s{j}_p{n}.npz")
                self.gen(f, n, cfg.exploit_games, seats, 9000 + 100 * it + 10 * j + n)
                files.append(f)
            # fine-tune on games against the best only; the best's value net is the teacher
            x_value = self.train_value(f"exploit_it{it}_s{j}", list(files), len(files), init=x_value, teacher=best)
        res, m, lo, hi = self.h2h(f"exploit_it{it}", agent(cfg, x_value), agent(cfg, best), cfg.gate_games)
        found = lo > 1.0
        self.log(f"  exploiter vs best: share x players {m:.3f} [{lo:.3f}, {hi:.3f}] "
                 + " ".join(f"{n}p {v:.2f}" for n, v in res.items())
                 + (" -> EXPLOIT FOUND: its games join the training data, it joins the pool" if found
                    else " -> no significant exploit"))
        self.s["history"].append({"iteration": it, "exploiter": x_value, "share": round(m, 3),
                                  "lo": round(lo, 3), "found": found})
        if found:
            self.s["league_data"] += [f for f in files if f not in self.s["league_data"]]
            self.s["pool"].append(x_value)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    for f in dataclasses.fields(Config):
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    cfg = Config(**{k.replace("-", "_"): v for k, v in vars(ap.parse_args()).items()})
    league = League(cfg)
    while league.s["iteration"] < cfg.iterations:
        league.iterate()


if __name__ == "__main__":
    main()
