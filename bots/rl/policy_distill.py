"""Fine-tune a policy network on search targets recorded in league games.

    python3 -m bots.rl.policy_distill --init runs/exit5/best.pt --data a.npz,b.npz --out p.pt

bots/rl/value.py gen records, at every decision a search agent searched, the
mover's features, the legal-move mask and the root visit distribution
(pol_obs / pol_mask / pol_pi / pol_game). This trains the policy head and body
to imitate those distributions (cross-entropy), as expert iteration does, but
on league games at every player count. 5% of the games are held out (by
game id); the log reports held-out KL to the search targets and top-move
agreement for the new and the initial network, and the best epoch (held-out
KL) is saved. The value head is left as it is (search uses a separate value
network).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from .model import load, save


def load_targets(paths):
    obs, mask, pi, game = [], [], [], []
    for p in paths:
        d = np.load(p)
        if "pol_obs" not in d:
            continue
        obs.append(d["pol_obs"])
        mask.append(d["pol_mask"])
        pi.append(d["pol_pi"])
        game.append(d["pol_game"] + 10**12 * len(game))     # game ids unique across files
    if not obs:
        return None
    return np.concatenate(obs), np.concatenate(mask), np.concatenate(pi), np.concatenate(game)


@torch.no_grad()
def fit(net, obs, mask, pi):
    net.eval()
    ce, kl, agree = [], [], []
    for i in range(0, len(obs), 8192):
        x = torch.from_numpy(obs[i:i + 8192].astype(np.float32))
        m = torch.from_numpy(mask[i:i + 8192])
        t = torch.from_numpy(pi[i:i + 8192])
        logits, _ = net(x, m)
        logp = F.log_softmax(logits, -1)
        c = -(t * logp).sum(-1)
        ent = -(t * torch.log(torch.where(t > 0, t, torch.ones_like(t)))).sum(-1)
        ce.append(c)
        kl.append(c - ent)
        agree.append((logits.argmax(-1) == t.argmax(-1)).float())
    net.train()
    return {"ce": round(torch.cat(ce).mean().item(), 4), "kl": round(torch.cat(kl).mean().item(), 4),
            "agree": round(torch.cat(agree).mean().item(), 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--init", required=True)
    ap.add_argument("--data", required=True, help="comma-separated value.py gen files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    data = load_targets(a.data.split(","))
    if data is None:
        print("no search targets in the data", flush=True)
        sys.exit(2)
    obs, mask, pi, game = data
    games = np.unique(game)
    held = np.random.default_rng(a.seed).choice(games, max(1, len(games) // 20), replace=False)
    te = np.isin(game, held)
    tr = np.flatnonzero(~te)
    init = load(a.init)
    net = load(a.init)
    base = fit(init, obs[te], mask[te], pi[te])
    print(json.dumps({"epoch": 0, "rows": len(obs), "held_rows": int(te.sum()), "held": base}), flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    rng = np.random.default_rng(a.seed)
    best = base["kl"]
    saved = False
    for epoch in range(1, a.epochs + 1):
        rng.shuffle(tr)
        tot = 0.0
        steps = max(1, len(tr) // a.batch)
        for s in range(steps):
            i = np.sort(tr[s * a.batch:(s + 1) * a.batch])
            logits, _ = net(torch.from_numpy(obs[i].astype(np.float32)), torch.from_numpy(mask[i]))
            loss = -(torch.from_numpy(pi[i]) * F.log_softmax(logits, -1)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        held_fit = fit(net, obs[te], mask[te], pi[te])
        print(json.dumps({"epoch": epoch, "train_ce": round(tot / steps, 4), "held": held_fit,
                          "held_init": base}), flush=True)
        if held_fit["kl"] < best:
            best = held_fit["kl"]
            save(a.out, net, distilled_from=a.init, held=held_fit, held_init=base, epoch=epoch)
            saved = True
    if not saved:                                        # no improvement: keep the initial network
        save(a.out, init, distilled_from=a.init, held=base, held_init=base, epoch=0)
    print(json.dumps({"saved": a.out, "improved": saved}), flush=True)


if __name__ == "__main__":
    main()
