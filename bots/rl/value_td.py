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


MAX_SEATS = 6      # positions are padded to this many seats (features.MAX_PLAYERS)


def _players(path) -> int:
    return int(np.load(path)["players"])


def prep(a):
    os.makedirs(a.dir, exist_ok=True)
    files = a.sources.split(",")
    players = [_players(p) for p in files]
    sizes = [_open_member(p, "obs")[1][0] for p in files]
    if a.max_rows:
        sizes = [min(s, max(0, a.max_rows - sum(sizes[:i]))) for i, s in enumerate(sizes)]
        sizes = [s - s % (n * 24) for s, n in zip(sizes, players)]   # whole games
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
    np.save(f"{a.dir}/source.npy", np.concatenate([np.full(n, i, np.int8) for i, n in enumerate(sizes)]))
    # positions: consecutive blocks of `players` rows (one per seat)
    pstart, pn, off = [], [], 0
    for n_rows, k in zip(sizes, players):
        pstart.append(off + np.arange(0, n_rows, k))
        pn.append(np.full(n_rows // k, k, np.int8))
        off += n_rows
    np.save(f"{a.dir}/pstart.npy", np.concatenate(pstart))
    np.save(f"{a.dir}/pn.npy", np.concatenate(pn))
    groups = [g.split(",") for g in a.split_groups.split(";")] if a.split_groups else [files]
    assert sorted(sum(groups, [])) == sorted(files), "--split-groups must partition --sources"
    json.dump({"sources": files, "rows": sizes, "players": players, "split_groups": groups},
              open(f"{a.dir}/prep.json", "w"))
    pos = Positions(a.dir)
    game = np.load(f"{a.dir}/game.npy")
    for k in range(1, MAX_SEATS):                    # every seat of a position is the same game
        sel = pos.n > k
        assert (game[pos.start[sel] + k] == game[pos.start[sel]]).all()
    print("rows", i, "positions", len(pos.start), "games", len(np.unique(game)),
          "by players", {int(k): int((pos.n == k).sum()) for k in np.unique(pos.n)}, flush=True)


class Positions:
    """Where each position's rows are (positions have 3-6 seats), padded to MAX_SEATS."""

    def __init__(self, d):
        if os.path.exists(f"{d}/pstart.npy"):
            self.start, self.n = np.load(f"{d}/pstart.npy"), np.load(f"{d}/pn.npy").astype(np.int64)
        else:                                         # dirs prepared before mixed player counts: 4 seats
            rows = len(np.load(f"{d}/win.npy", mmap_mode="r"))
            self.start, self.n = np.arange(0, rows, 4), np.full(rows // 4, 4)

    def index(self, pos):
        """Row index [B, MAX_SEATS] (padding repeats the first row) and seat mask."""
        mask = np.arange(MAX_SEATS)[None, :] < self.n[pos][:, None]
        idx = np.where(mask, self.start[pos][:, None] + np.arange(MAX_SEATS)[None, :],
                       self.start[pos][:, None])
        return idx, mask


def gather(arr, idx, mask):
    """Padded per-seat values of `arr` ([rows] or [rows, F]); padding seats are 0."""
    x = np.asarray(arr[idx.ravel()]).reshape(idx.shape + arr.shape[1:])
    m = mask.reshape(mask.shape + (1,) * (x.ndim - 2))
    return np.where(m, x, 0)


def split(d):
    """(training positions, held-out positions): 5% of the games of each split
    group (seed 0). One group of runs/value1/data.npz + data2.npz gives the
    held-out games of runs/value1/big_aux05.pt, so adding sources as a new
    group keeps earlier models' held-out games unseen."""
    info = json.load(open(f"{d}/prep.json"))
    held = []
    for group in info.get("split_groups", [info["sources"]]):
        allg = np.unique(np.concatenate([npz_member(p, "game") for p in group]))
        held.append(np.random.default_rng(0).choice(allg, int(len(allg) * 0.05), replace=False))
    pos = Positions(d)
    pos_te = np.isin(np.load(f"{d}/game.npy")[pos.start], np.concatenate(held))
    return np.flatnonzero(~pos_te), np.flatnonzero(pos_te)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class _LegacyModel(torch.nn.Module):
    def __init__(self, path):
        super().__init__()
        from .model import load
        self.inner = LegacyValue(load(path))

    def win_prob(self, x, mask=None):
        v = self.inner.net.value(self.inner.net.body(x)).squeeze(-1).clamp(min=1e-3)
        if mask is not None:
            v = v * mask
        return v / v.sum(-1, keepdim=True)


class EvalSet:
    """Positions to score against the real result: obs [P, MAX_SEATS, F] (float16),
    win [P, MAX_SEATS], mask, round [P], players [P]."""

    def __init__(self, obs, win, mask, rnd, n):
        self.obs, self.win, self.mask, self.rnd, self.n = obs, win, mask, rnd, n

    @classmethod
    def from_dir(cls, d, pos):
        P = Positions(d)
        idx, mask = P.index(pos)
        obs = np.load(f"{d}/obs.npy", mmap_mode="r")
        return cls(gather(obs, idx, mask), gather(np.load(f"{d}/win.npy"), idx, mask), mask,
                   np.load(f"{d}/round.npy")[P.start[pos]], P.n[pos])

    @classmethod
    def from_npz(cls, path):
        """A file from bots/rl/value.py gen (one player count)."""
        s = np.load(path)
        k = int(s["players"]) if "players" in s else 4
        obs, win, rnd = s["obs"], s["win"], s["round"]
        P = len(win) // k
        pad = ((0, 0), (0, MAX_SEATS - k))
        mask = np.zeros((P, MAX_SEATS), bool)
        mask[:, :k] = True
        return cls(np.pad(obs.reshape(P, k, -1), pad + ((0, 0),)), np.pad(win.reshape(P, k), pad), mask,
                   rnd[::k], np.full(P, k))


@torch.no_grad()
def metrics(model, es: EvalSet) -> dict:
    """Scores against the real result, overall and per player count."""
    model.eval()
    p = np.concatenate([model.win_prob(torch.from_numpy(np.asarray(es.obs[i:i + 8192], np.float32)),
                                       torch.from_numpy(es.mask[i:i + 8192])).numpy()
                        for i in range(0, len(es.obs), 8192)])
    model.train()

    def score(sel):
        w, q, m = es.win[sel], p[sel], es.mask[sel]
        out = {"mse": float(((q - w) ** 2)[m].mean()), "corr": float(np.corrcoef(q[m], w[m])[0, 1]),
               "xent": float(-(w * np.log(np.clip(q, 1e-6, 1))).sum(1).mean()),
               "pick": float(w[np.arange(len(q)), np.where(m, q, -1).argmax(1)].mean())}
        return {k: round(v, 4) for k, v in out.items()}

    out = score(np.ones(len(p), bool))
    ns = np.unique(es.n)
    if len(ns) > 1:
        for k in ns:
            out[f"p{k}"] = score(es.n == k)
    for lo, hi in ((1, 3), (4, 7), (8, 12), (13, 99)):
        s = (es.rnd >= lo) & (es.rnd <= hi)
        if s.any():
            q = np.where(es.mask[s], p[s], -1)
            out[f"pick_r{lo}-{hi}"] = round(float(es.win[s][np.arange(s.sum()), q.argmax(1)].mean()), 4)
    return out


def eval_sets(a):
    _, te = split(a.dir)
    sets = {"held": EvalSet.from_dir(a.dir, te)}
    for path in [p for p in a.search_eval.split(",") if p]:
        if os.path.exists(path):
            sets["search" if path == a.search_eval.split(",")[0] else os.path.basename(path)[:-4]] = \
                EvalSet.from_npz(path)
    return sets


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(a):
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    obs = np.load(f"{a.dir}/obs.npy", mmap_mode="r")
    win, margin = np.load(f"{a.dir}/win.npy"), np.load(f"{a.dir}/margin.npy")
    P = Positions(a.dir)
    tr, _ = split(a.dir)
    pgame = np.load(f"{a.dir}/game.npy")[P.start]
    if a.fold >= 0:                                   # teacher: train on the other fold
        other = tr[pgame[tr] % 2 == a.fold]
        tr = tr[pgame[tr] % 2 != a.fold]
    target = np.load(f"{a.dir}/{a.target}") if a.target else None    # [P, MAX_SEATS]
    if a.oversample > 1:                              # repeat positions from later sources
        src = np.load(f"{a.dir}/source.npy")[P.start]
        extra = tr[src[tr] >= a.oversample_from]
        tr = np.concatenate([tr] + [extra] * (a.oversample - 1))
        print("oversampled", len(extra), "positions x", a.oversample, flush=True)
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
            idx, mask = P.index(pos)
            x = torch.from_numpy(gather(obs, idx, mask).astype(np.float32))
            mk = torch.from_numpy(mask)
            w = torch.from_numpy(gather(win, idx, mask) if target is None else target[pos])
            v, mg = m(x, mk)
            if a.loss == "joint":
                lv = -(w * F.log_softmax(v.masked_fill(~mk, -1e9), -1)).sum(-1).mean()
            else:
                lv = ((v - w) ** 2)[mk].mean()
            lm = ((mg - torch.from_numpy(gather(margin, idx, mask))) ** 2)[mk].mean()
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
        for k, es in sets.items():
            rec[k] = metrics(m, es)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        if best is None or rec["held"]["xent"] < best:
            best = rec["held"]["xent"]
            save_value(f"{a.dir}/{a.name}.pt", m, epoch=epoch, metrics=rec, target=a.target)
    if a.fold >= 0:                                   # out-of-fold predictions
        m.load_state_dict(torch.load(f"{a.dir}/{a.name}.pt", weights_only=False)["state_dict"])
        m.eval()
        oof = np.full((len(P.start), MAX_SEATS), np.nan, np.float32)
        with torch.no_grad():
            for i in range(0, len(other), 4096):
                pos = other[i:i + 4096]
                idx, mask = P.index(pos)
                x = torch.from_numpy(gather(obs, idx, mask).astype(np.float32))
                oof[pos] = m.win_prob(x, torch.from_numpy(mask)).numpy()
        np.save(f"{a.dir}/oof_{a.fold}.npy", oof)
        print("oof", int((~np.isnan(oof[:, 0])).sum()), flush=True)


def targets(a):
    """TD(lambda) soft targets from the two teachers' out-of-fold predictions."""
    o0, o1 = np.load(f"{a.dir}/oof_0.npy"), np.load(f"{a.dir}/oof_1.npy")
    teach = np.where(np.isnan(o0), o1, o0)
    P = Positions(a.dir)
    idx, mask = P.index(np.arange(len(P.start)))
    win = gather(np.load(f"{a.dir}/win.npy"), idx, mask).astype(np.float32)   # [P, MAX_SEATS]
    game = np.load(f"{a.dir}/game.npy")[P.start]
    rnd = np.load(f"{a.dir}/round.npy")[P.start]
    tr, _ = split(a.dir)
    assert not np.isnan(teach[tr]).any()
    out = win.copy()                                  # held-out positions keep the real result
    order = tr[np.lexsort((rnd[tr], game[tr]))]       # by game, then round
    gs = game[order]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    for s0, e0 in zip(starts, np.r_[starts[1:], len(order)]):
        idx_g = order[s0:e0]
        r = rnd[idx_g]
        g = np.empty((len(idx_g), MAX_SEATS), np.float32)
        nxt = len(idx_g)                              # first position of a later round
        for i in range(len(idx_g) - 1, -1, -1):
            if i + 1 < len(idx_g) and r[i + 1] > r[i]:
                nxt = i + 1
            g[i] = win[idx_g[i]] if nxt == len(idx_g) else \
                (1 - a.lam) * teach[idx_g[nxt]] + a.lam * g[nxt]
        out[idx_g] = g
    out[~mask] = 0.0
    np.save(f"{a.dir}/target_l{a.lam:g}.npy", out)
    print(f"lam {a.lam}: mean |target - result| {np.abs(out[tr] - win[tr])[mask[tr]].mean():.4f}", flush=True)


def evaluate(a):
    sets = eval_sets(a)
    for c in a.ckpts.split(","):
        ck = torch.load(c, map_location="cpu", weights_only=False)
        if "arch" in ck.get("cfg", {}):
            m = SeatValueNet(**ck["cfg"])
            m.load_state_dict(ck["state_dict"])
        else:
            m = _LegacyModel(c)
        print(c, json.dumps({k: metrics(m, es) for k, es in sets.items()}), flush=True)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in ("prep", "train", "targets", "eval"):
        print(__doc__)
        sys.exit(2)
    ap = argparse.ArgumentParser(prog=f"value_td {cmd}")
    ap.add_argument("--dir", default="runs/value2")
    ap.add_argument("--sources", default="runs/value1/data.npz,runs/value1/data2.npz")
    ap.add_argument("--max-rows", type=int, default=0, help="prep: cap the rows (RAM)")
    ap.add_argument("--split-groups", default="",
                    help="prep: ';'-separated groups of sources, 5%% of each group's games held out")
    ap.add_argument("--search-eval", default="runs/varch/search_eval.npz",
                    help="comma-separated value.py gen files scored every epoch (e.g. searched games)")
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
    ap.add_argument("--oversample", type=int, default=1,
                    help="train: repeat positions of sources >= --oversample-from this many times")
    ap.add_argument("--oversample-from", type=int, default=2)
    a = ap.parse_args(sys.argv[2:])
    {"prep": prep, "train": train, "targets": targets, "eval": evaluate}[cmd](a)


if __name__ == "__main__":
    main()
