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
 2. A new value network: by default (--finetune 1) the best value net fine-tuned
    on all league games plus the earlier searched-game data (whole split groups,
    so their held-out games stay unseen), TD(lambda) targets from the best net
    (it never saw the new games); league games oversampled (--oversample).
    --finetune 0 retrains from scratch on everything (out-of-fold teachers).
 3. Policy: the best policy fine-tuned on the search targets recorded in the
    league games (bots/rl/policy_distill.py: root visit distributions of every
    search decision, all player counts).
 4. Gate: 1 x candidate vs (n-1) x best at every player count; accepted if its
    share x players, pooled over counts, is at least 1 (a fair share). The
    candidate is new policy + new value, then (if that fails) the best policy
    + new value. The old best joins the pool.
 5. Benchmark: an Elo ladder (bots/rating.py) per player count over the best,
    the pool and anchors. The scripted worst case is a floor, not the metric.
 6. Every --exploit-every iterations, an exploiter: a value network fine-tuned
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

import numpy as np

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
    finetune: int = 1              # 1: fine-tune the best value net on league games + searched-game replay
    finetune_epochs: int = 2
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


def split_spec(spec, default_policy):
    """Pool / best entries are 'value.pt' or 'policy.pt|value.pt'."""
    return spec.split("|", 1) if "|" in spec else (default_policy, spec)


def agent_of(cfg, spec):
    policy, value = split_spec(spec, cfg.policy)
    return agent(cfg, value, policy)


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

    def best_spec(self):
        return f"{self.s.get('best_policy', self.cfg.policy)}|{self.s['best']}"

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
                "--arch", "mlp", "--epochs", str(self.cfg.epochs if not init else self.cfg.finetune_epochs)]
        if self.cfg.oversample > 1 and oversample_from < len(sources):
            args += ["--oversample", str(self.cfg.oversample), "--oversample-from", str(oversample_from)]
        if init:
            args += ["--init", init, "--lr", "3e-4"]
        self.run(args, log=log)
        return ck

    def ladder(self, it):
        agents = [agent_of(self.cfg, self.best_spec())] + [agent_of(self.cfg, v) for v in self.s["pool"][-4:]] \
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
        best_policy = self.s.get("best_policy", cfg.policy)
        pool = self.s["pool"][-cfg.pool_size:]
        seats = [agent(cfg, best, best_policy)] * cfg.w_best \
            + [agent_of(cfg, v) for v in pool for _ in range(cfg.w_pool)] \
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
            if cfg.finetune:
                # replay only the searched-game groups (whole groups: their held-out games stay unseen)
                groups = [g for g in groups if all("searched" in f for f in g)]
                base = [f for g in groups for f in g]
                ck = self.train_value(f"value_it{it}", base + league, len(base), init=best, teacher=best,
                                      groups=groups + [[f] for f in league])
            else:
                ck = self.train_value(f"value_it{it}", base + league, len(base),
                                      groups=groups + [[f] for f in league])
            self.s[f"candidate_it{it}"] = ck
        self.step(f"it{it}:value", value)
        cand = self.s[f"candidate_it{it}"]

        def policy():
            files = [f for f in self.s["league_data"] if os.path.exists(f) and "pol_obs" in np.load(f).files]
            out = os.path.join(cfg.dir, f"policy_it{it}.pt")
            if files and not os.path.exists(out):
                self.run(["-m", "bots.rl.policy_distill", "--init", best_policy, "--data", ",".join(files),
                          "--out", out, "--threads", str(min(32, cfg.workers + 1))],
                         log=os.path.join(cfg.dir, f"policy_it{it}.log"))
            improved = os.path.exists(out) and json.loads(
                open(os.path.join(cfg.dir, f"policy_it{it}.log")).read().strip().splitlines()[-1])["improved"]
            self.s[f"cand_policy_it{it}"] = out if improved else best_policy
            if files:
                last = [json.loads(l) for l in open(os.path.join(cfg.dir, f"policy_it{it}.log"))
                        if l.startswith('{"epoch"')][-1]
                self.log(f"  policy on {len(files)} files with search targets: held-out KL "
                         f"{last['held_init']['kl'] if 'held_init' in last else last['held']['kl']} -> "
                         f"{last['held']['kl']}, top-move agreement -> {last['held']['agree']}"
                         + ("" if improved else " (not improved: policy unchanged)"))
        self.step(f"it{it}:policy", policy)
        cand_policy = self.s.get(f"cand_policy_it{it}", best_policy)

        def gate():
            tries = [(cand_policy, cand)] + ([(best_policy, cand)] if cand_policy != best_policy else [])
            ok = False
            for pol, val in tries:
                tag = "new policy + new value" if pol != best_policy else "new value"
                res, m, lo, hi = self.h2h(f"gate_it{it}{'' if pol != best_policy or len(tries) == 1 else '_v'}",
                                          agent(cfg, val, pol), agent(cfg, best, best_policy), cfg.gate_games)
                ok = m >= 1.0
                self.log(f"  gate ({tag}) vs best: share x players {m:.3f} [{lo:.3f}, {hi:.3f}] "
                         + " ".join(f"{n}p {v:.2f}" for n, v in res.items()) + (" -> ACCEPTED" if ok else " -> rejected"))
                self.s["history"].append({"iteration": it, "candidate": f"{pol}|{val}", "gate": round(m, 3),
                                          "lo": round(lo, 3), "accepted": ok})
                if ok:
                    self.s["pool"].append(f"{best_policy}|{best}")
                    self.s["best"], self.s["best_policy"] = val, pol
                    break
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
        bp = self.s.get("best_policy", cfg.policy)
        x_value, files = best, []
        for j in range(1, cfg.exploit_steps + 1):
            x = agent(cfg, x_value, bp)
            for n in COUNTS:                        # one exploiter seat on average, the rest the best
                seats = [agent(cfg, best, bp)] * (n - 1) + [x]
                f = os.path.join(cfg.dir, "data", f"exploit_it{it}_s{j}_p{n}.npz")
                self.gen(f, n, cfg.exploit_games, seats, 9000 + 100 * it + 10 * j + n)
                files.append(f)
            # fine-tune on games against the best only; the best's value net is the teacher
            x_value = self.train_value(f"exploit_it{it}_s{j}", list(files), len(files), init=x_value, teacher=best)
        res, m, lo, hi = self.h2h(f"exploit_it{it}", agent(cfg, x_value, bp), agent(cfg, best, bp), cfg.gate_games)
        found = lo > 1.0
        self.log(f"  exploiter vs best: share x players {m:.3f} [{lo:.3f}, {hi:.3f}] "
                 + " ".join(f"{n}p {v:.2f}" for n, v in res.items())
                 + (" -> EXPLOIT FOUND: its games join the training data, it joins the pool" if found
                    else " -> no significant exploit"))
        self.s["history"].append({"iteration": it, "exploiter": x_value, "share": round(m, 3),
                                  "lo": round(lo, 3), "found": found})
        if found:
            self.s["league_data"] += [f for f in files if f not in self.s["league_data"]]
            self.s["pool"].append(f"{bp}|{x_value}")


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
