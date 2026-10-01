"""Seat value networks trained on TD(lambda) targets (see value_models.py).

    D=runs/value2
    python3 -m bots.rl.value_td prep    --dir $D --sources runs/value1/data.npz,runs/value1/data2.npz
    python3 -m bots.rl.value_td train   --dir $D --name teach0 --fold 0 --epochs 2 --arch mlp
    python3 -m bots.rl.value_td train   --dir $D --name teach1 --fold 1 --epochs 2 --arch mlp
    python3 -m bots.rl.value_td targets --dir $D --lam 0.5
    python3 -m bots.rl.value_td train   --dir $D --name value --target target_l0.5.npy --arch attn
    python3 -m bots.rl.value_td eval    --dir $D --ckpts $D/value.pt,runs/value1/big_aux05.pt

Data: positions from bots/rl/value.py gen (every seat's view of a position in
consecutive rows), merged into memory-mapped arrays by prep. Held out: 5% of
the games of all --sources (seed 0, as value.fit), never trained on.

Why TD targets: the final result (who won) is a very noisy label for an
early position, and value networks fitted to it memorise games instead of
learning positions. A position's target here is

    (1 - lam) * teacher(next sampled position, in a later round of the game)
        + lam * target(that position)

and the last position's target is the final result (lam = 1: the result
alone). The teacher's predictions are out-of-fold: two teachers each train
on half the training games and label the other half, so a target cannot leak
the result through memorisation. Evaluation is always against the real
result. Offline (62k games), lam = 0.5 cut held-out cross-entropy from 0.986
to 0.966 for the same network, and it stopped overfitting.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

from .value_models import LegacyValue, SeatValueNet, save_value

N_SEATS = 4


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _open_member(path, key):
    """(stream positioned at the data, shape, dtype) of an uncompressed .npz member."""
    f = zipfile.ZipFile(path).open(key + ".npy")
    version = np.lib.format.read_magic(f)
    read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, _, dtype = read(f)
    return f, shape, dtype


def _fill(f, buf):
    chunk = 1 << 26
    for i in range(0, buf.size, chunk):
        n = f.readinto(memoryview(buf[i:i + chunk]))
        assert n == min(chunk, buf.size - i)


def npz_member(path, key, rows=None):
    """The first `rows` rows of an .npz member, without loading the rest."""
    f, shape, dtype = _open_member(path, key)
    rows = shape[0] if rows is None else min(rows, shape[0])
    out = np.empty((rows,) + tuple(shape[1:]), dtype)
    _fill(f, out.reshape(-1).view(np.uint8))
    return out


def prep(a):
    os.makedirs(a.dir, exist_ok=True)
    files = a.sources.split(",")
    sizes = [_open_member(p, "obs")[1][0] for p in files]
    if a.max_rows:
        sizes = [min(s, max(0, a.max_rows - sum(sizes[:i]))) for i, s in enumerate(sizes)]
        sizes = [s - s % (N_SEATS * 24) for s in sizes]      # whole games
    width = _open_member(files[0], "obs")[1][1]
    obs = np.lib.format.open_memmap(f"{a.dir}/obs.npy", "w+", np.float16, (sum(sizes), width))
    i = 0
    for path, n in zip(files, sizes):
        f, _, _ = _open_member(path, "obs")
        for j in range(0, n, 200_000):
            k = min(200_000, n - j)
            _fill(f, obs[i + j:i + j + k].reshape(-1).view(np.uint8))
        i += n
    obs.flush()
    for k in ("win", "margin", "game", "round"):
        np.save(f"{a.dir}/{k}.npy", np.concatenate([npz_member(p, k, n) for p, n in zip(files, sizes)]))
    json.dump({"sources": files, "rows": sizes}, open(f"{a.dir}/prep.json", "w"))
    game = np.load(f"{a.dir}/game.npy")
    assert (game.reshape(-1, N_SEATS) == game.reshape(-1, N_SEATS)[:, :1]).all()
    print("rows", i, "games", len(np.unique(game)), flush=True)


def split(d):
    """(training positions, held-out positions): 5% of all source games, seed 0."""
    files = json.load(open(f"{d}/prep.json"))["sources"]
    allg = np.unique(np.concatenate([npz_member(p, "game") for p in files]))
    held = np.random.default_rng(0).choice(allg, int(len(allg) * 0.05), replace=False)
    pos_te = np.isin(np.load(f"{d}/game.npy"), held)[::N_SEATS]
    return np.flatnonzero(~pos_te), np.flatnonzero(pos_te)


def _rows(pos):
    return (pos[:, None] * N_SEATS + np.arange(N_SEATS)).ravel()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class _LegacyModel(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        from .model import load
        self.inner = LegacyValue(load(path))

    def win_prob(self, x):
        v = self.inner.net.value(self.inner.net.body(x)).squeeze(-1).clamp(min=1e-3)
        return v / v.sum(-1, keepdim=True)


@torch.no_grad()
def metrics(model, obs, win, rnd) -> dict:
    """obs [P, seats, obs], win [P, seats], rnd [P] -> scores against the real result."""
    model.eval()
    p = np.concatenate([model.win_prob(torch.from_numpy(np.asarray(obs[i:i + 8192], np.float32))).numpy()
                        for i in range(0, len(obs), 8192)])
    out = {"mse": float(np.mean((p - win) ** 2)), "corr": float(np.corrcoef(p.ravel(), win.ravel())[0, 1]),
           "xent": float(-(win * np.log(np.clip(p, 1e-6, 1))).sum(1).mean()),
           "pick": float(win[np.arange(len(p)), p.argmax(1)].mean())}
    for lo, hi in ((1, 3), (4, 7), (8, 12), (13, 99)):
        s = (rnd >= lo) & (rnd <= hi)
        if s.any():
            out[f"pick_r{lo}-{hi}"] = float(win[s][np.arange(s.sum()), p[s].argmax(1)].mean())
    model.train()
    return {k: round(v, 4) for k, v in out.items()}


def eval_sets(a):
    obs = np.load(f"{a.dir}/obs.npy", mmap_mode="r")
    win, rnd = np.load(f"{a.dir}/win.npy"), np.load(f"{a.dir}/round.npy")
    _, te = split(a.dir)
    r = _rows(te)
    sets = {"held": (np.asarray(obs[r]).reshape(len(te), N_SEATS, -1), win[r].reshape(-1, N_SEATS),
                     rnd[r][::N_SEATS])}
    if a.search_eval and os.path.exists(a.search_eval):
        s = np.load(a.search_eval)
        sets["search"] = (s["obs"].reshape(-1, N_SEATS, s["obs"].shape[1]), s["win"].reshape(-1, N_SEATS),
                          s["round"][::N_SEATS])
    return sets


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(a):
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    obs = np.load(f"{a.dir}/obs.npy", mmap_mode="r")
    win, margin = np.load(f"{a.dir}/win.npy"), np.load(f"{a.dir}/margin.npy")
    tr, _ = split(a.dir)
    pgame = np.load(f"{a.dir}/game.npy")[::N_SEATS]
    if a.fold >= 0:                                   # teacher: train on the other fold
        other = tr[pgame[tr] % 2 == a.fold]
        tr = tr[pgame[tr] % 2 != a.fold]
    target = np.load(f"{a.dir}/{a.target}") if a.target else None
    sets = eval_sets(a)
    m = SeatValueNet(obs.shape[1], a.arch, a.hidden, a.depth, a.layers, a.loss)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = len(tr) // a.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps * a.epochs, pct_start=0.05)
    rng = np.random.default_rng(a.seed)
    log = open(f"{a.dir}/{a.name}.log", "w")
    best, t0 = None, time.time()
    for epoch in range(1, a.epochs + 1):
        rng.shuffle(tr)
        tot = np.zeros(2)
        for s in range(steps):
            pos = np.sort(tr[s * a.batch:(s + 1) * a.batch])
            r = _rows(pos)
            x = torch.from_numpy(np.asarray(obs[r], np.float32)).view(len(pos), N_SEATS, -1)
            w = torch.from_numpy(win[r]).view(len(pos), N_SEATS) if target is None \
                else torch.from_numpy(target[pos])
            v, mg = m(x)
            lv = -(w * F.log_softmax(v, -1)).sum(-1).mean() if a.loss == "joint" else F.mse_loss(v, w)
            lm = F.mse_loss(mg, torch.from_numpy(margin[r]).view(len(pos), N_SEATS))
            loss = lv + a.aux_coef * lm
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += (lv.item(), lm.item())
            if s % 2000 == 0:
                print(json.dumps({"epoch": epoch, "step": s, "of": steps, "s": round(time.time() - t0)}),
                      flush=True)
        rec = {"epoch": epoch, "train_value": round(tot[0] / steps, 4),
               "train_margin": round(tot[1] / steps, 4), "s": round(time.time() - t0)}
        for k, (o, w_, r_) in sets.items():
            rec[k] = metrics(m, o, w_, r_)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        if best is None or rec["held"]["xent"] < best:
            best = rec["held"]["xent"]
            save_value(f"{a.dir}/{a.name}.pt", m, epoch=epoch, metrics=rec, target=a.target)
    if a.fold >= 0:                                   # out-of-fold predictions
        m.load_state_dict(torch.load(f"{a.dir}/{a.name}.pt", weights_only=False)["state_dict"])
        m.eval()
        oof = np.full((len(pgame), N_SEATS), np.nan, np.float32)
        with torch.no_grad():
            for i in range(0, len(other), 4096):
                pos = other[i:i + 4096]
                x = torch.from_numpy(np.asarray(obs[_rows(pos)], np.float32)).view(len(pos), N_SEATS, -1)
                oof[pos] = m.win_prob(x).numpy()
        np.save(f"{a.dir}/oof_{a.fold}.npy", oof)
        print("oof", int((~np.isnan(oof[:, 0])).sum()), flush=True)


def targets(a):
    """TD(lambda) soft targets from the two teachers' out-of-fold predictions."""
    o0, o1 = np.load(f"{a.dir}/oof_0.npy"), np.load(f"{a.dir}/oof_1.npy")
    teach = np.where(np.isnan(o0), o1, o0)
    win = np.load(f"{a.dir}/win.npy").reshape(-1, N_SEATS)
    game = np.load(f"{a.dir}/game.npy")[::N_SEATS]
    rnd = np.load(f"{a.dir}/round.npy")[::N_SEATS]
    tr, _ = split(a.dir)
    assert not np.isnan(teach[tr]).any()
    out = win.astype(np.float32).copy()               # held-out positions keep the real result
    order = tr[np.lexsort((rnd[tr], game[tr]))]       # by game, then round
    gs = game[order]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    for s0, e0 in zip(starts, np.r_[starts[1:], len(order)]):
        idx = order[s0:e0]
        r = rnd[idx]
        g = np.empty((len(idx), N_SEATS), np.float32)
        nxt = len(idx)                                # first position of a later round
        for i in range(len(idx) - 1, -1, -1):
            if i + 1 < len(idx) and r[i + 1] > r[i]:
                nxt = i + 1
            g[i] = win[idx[i]] if nxt == len(idx) else (1 - a.lam) * teach[idx[nxt]] + a.lam * g[nxt]
        out[idx] = g
    np.save(f"{a.dir}/target_l{a.lam:g}.npy", out)
    print(f"lam {a.lam}: mean |target - result| {np.abs(out[tr] - win[tr]).mean():.4f}", flush=True)


def evaluate(a):
    sets = eval_sets(a)
    for c in a.ckpts.split(","):
        ck = torch.load(c, map_location="cpu", weights_only=False)
        if "arch" in ck.get("cfg", {}):
            m = SeatValueNet(**ck["cfg"])
            m.load_state_dict(ck["state_dict"])
        else:
            m = _LegacyModel(c)
        print(c, json.dumps({k: metrics(m, *v) for k, v in sets.items()}), flush=True)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in ("prep", "train", "targets", "eval"):
        print(__doc__)
        sys.exit(2)
    ap = argparse.ArgumentParser(prog=f"value_td {cmd}")
    ap.add_argument("--dir", default="runs/value2")
    ap.add_argument("--sources", default="runs/value1/data.npz,runs/value1/data2.npz")
    ap.add_argument("--max-rows", type=int, default=0, help="prep: cap the rows (RAM)")
    ap.add_argument("--search-eval", default="runs/varch/search_eval.npz",
                    help="positions from searched games (value.py gen --agents mcts:...)")
    ap.add_argument("--name", default="value")
    ap.add_argument("--arch", default="attn")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--loss", default="joint")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--aux-coef", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--target", default="")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--ckpts", default="")
    a = ap.parse_args(sys.argv[2:])
    {"prep": prep, "train": train, "targets": targets, "eval": evaluate}[cmd](a)


if __name__ == "__main__":
    main()
