"""Record one game between bots and write a replay you can watch in a browser.

    PY=~/.local/opt/python311/bin/python3
    $PY -m tools.replay --agents rl:runs/league3/champion.pt,balanced,builder,tycoon
    $PY -m tools.replay --agents "mcts:100+norm+c3:runs/exit4/best.pt:runs/value1/big_aux05.pt,rl:runs/league3/champion.pt,builder,planner" --seed 7 --out runs/replays/mcts.html

Agents are named as in bots.heuristic.make, one per seat in seat order. The
output is one self-contained HTML file (map, players, markets, move log,
playback). For network agents each decision also records what the network
thought: its top moves with probabilities (search visits for mcts agents) and
its estimate of the final win share.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "cpp", "build"))
sys.path.insert(0, ROOT)
import pgcore  # noqa: E402

import powergrid_core  # noqa: E402
from bots.heuristic import make  # noqa: E402

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay_viewer.html")
DATA_MARK = "/*__REPLAY_DATA__*/null"
TOP_K = 5


def _top(scores: dict, amap: dict, state, total: float):
    best = sorted(scores, key=scores.get, reverse=True)[:TOP_K]
    return [[state.action_to_string(amap[x]), scores[x] / total] for x in best if scores[x] > 0]


def explain_and_act(agent, state):
    """Play the agent's move; for network agents also return what it thought:
    {"kind", "top": [[move, share]], "value": own win share, "values": per seat}."""
    from bots.rl.model import RLAgent
    from bots.search import SearchAgent
    if isinstance(agent, SearchAgent) and len(state.legal_actions()) > 1:
        import torch
        with torch.no_grad():
            prior, amap, vals = agent._evaluate(state)
        info = {"kind": "prior", "values": [float(v) for v in vals],
                "value": float(vals[agent.seat]), "prior": _top(prior, amap, state, 1.0)}
        if agent.wants_search(state):
            visits, amap = agent.search(state)
            info["kind"] = "visits"
            info["sims"] = agent.sims
            info["top"] = _top(visits, amap, state, sum(visits.values()))
            return amap[max(visits, key=visits.get)], info
        info["top"] = info.pop("prior")
        return amap[max(prior, key=prior.get)], info
    if isinstance(agent, RLAgent) and len(state.legal_actions()) > 1:
        import torch
        with torch.no_grad():
            mask, amap = agent.actions.mask_and_map(state.legal_actions())
            obs = torch.from_numpy(agent.enc.encode(state, agent.seat))[None]
            logits, value = agent.net(obs, torch.from_numpy(mask)[None])
            probs = torch.softmax(logits[0], 0).numpy()
        info = {"kind": "policy", "value": float(value[0]),
                "top": _top({x: float(probs[x]) for x in amap}, amap, state, 1.0)}
        return agent.act(state), info
    return agent.act(state), None


def _chance_text(before: dict, after: dict, raw: str) -> str:
    if not before["regions"] and after["regions"]:
        return "Regions in play: " + ", ".join(REGIONS[r] for r in after["regions"])
    new = {p["number"] for p in after["market"]} - {p["number"] for p in before["market"]}
    if new:
        return "Drew plant " + ", ".join(f"#{n}" for n in sorted(new))
    return raw.replace("CHANCE", "Deck:")


REGIONS = powergrid_core.GERMANY_REGIONS


def record(names, seed: int, players: int):
    rules = pgcore.Rules(players=players)
    rng = random.Random(seed)
    agents = [make(n) for n in names]
    for seat, agent in enumerate(agents):
        agent.reset(rules, seat, random.Random(rng.random()))
    state = pgcore.State(rules)
    view = json.loads(state.to_json())
    frames = [{"state": view, "actor": None, "text": "Game start", "think": None}]
    t0 = time.time()
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            a = rng.choices([o for o, _ in outcomes], [p for _, p in outcomes])[0]
            raw, actor, think = state.action_to_string(a), -1, None
        else:
            actor = state.current_player()
            a, think = explain_and_act(agents[actor], state)
            if a not in state.legal_actions():
                raise ValueError(f"{names[actor]} played illegal action {a}")
            raw = state.action_to_string(a)
        state.apply_action(a)
        after = json.loads(state.to_json())
        text = _chance_text(view, after, raw) if actor == -1 else raw
        frames.append({"state": after, "actor": actor, "text": text, "think": think})
        view = after
        if len(frames) % 100 == 0:
            print(f"  {len(frames)} moves, round {state.round()}, {time.time() - t0:.0f}s",
                  file=sys.stderr)
    for f in frames:            # derivable from players[].cities; keeps the file small
        f["state"].pop("occupants", None)
    return rules, frames, state.returns()


def board(rules):
    spec = powergrid_core.germany_map()
    assert list(spec.city_names) == list(rules.city_names), "viewer map differs from the engine's"
    return {"cities": list(spec.city_names), "city_region": list(spec.city_region),
            "regions": list(spec.region_names),
            "edges": [[a, b, w] for a, b, w in spec.edges]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--agents", default="rl:runs/league3/champion.pt,balanced,builder,tycoon",
                    help="comma-separated, one per seat")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default runs/replays/game_<seed>.html")
    args = ap.parse_args()
    names = args.agents.split(",")
    out = args.out or os.path.join(ROOT, "runs", "replays", f"game_{args.seed}.html")
    rules, frames, returns = record(names, args.seed, len(names))
    data = {"agents": names, "seed": args.seed, "returns": list(returns),
            "board": board(rules), "frames": frames}
    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read()
    assert DATA_MARK in html
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html.replace(DATA_MARK, payload))
    winners = [names[i] for i, r in enumerate(returns) if r > 0]
    print(f"{len(frames)} moves, {frames[-1]['state']['round']} rounds; won by "
          f"{', '.join(winners)}; wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
