"""Differential test: the C++ engine must match powergrid_core.py move for move.

Plays random games with the Python engine, replays the same action sequences
through cpp/build/powergrid_trace, and compares the canonical state dump, the
action strings and the event log after every move. Prints the first mismatch.

    make -C cpp && python3 tools/difftest.py [num_games] [seed]
"""
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import powergrid_core as pg  # noqa: E402

TRACE_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cpp", "build",
                         "powergrid_trace")


def fmt(v):
    if isinstance(v, bool):
        return "1" if v else "0"
    if v is None:
        return "-"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(fmt(x) for x in v) + "]"
    return str(v)


def event_str(e):
    return " ".join(f"{k}={fmt(v)}" for k, v in e.items())


def dump(s):
    """Canonical state string; State::Dump() in C++ builds the same thing."""
    ph = int(s.phase)
    cp = s.current_player()
    au = s.auction
    if au is None:
        auction = "-"
    else:
        high = -1 if au["high"] is None else au["high"]
        auction = fmt([au["plant"], au["selector"], au["bid"], high, au["pos"], au["ring"]])
    result = "-" if s.result is None else "[" + ",".join(f"{x:.6f}" for x in s.result) + "]"
    probs = "-"
    if s.phase == pg.Phase.CHANCE:
        probs = "[" + ",".join(f"{w:.9f}" for _, w in s.chance_outcomes()) + "]"
    trust = fmt([sorted(s.trust_plants), s.trust_stored, s.trust_houses, s.trust_due,
                 s.trust_took, s.trust_setup])
    parts = [
        f"ph={ph}", f"cp={cp}", f"rd={s.round}", f"st={s.step}", f"regions={fmt(s.regions)}",
        f"order={fmt(s.order)}", f"money={fmt(s.money)}", f"plants={fmt(s.plants)}",
        f"stored={fmt(s.stored)}", f"cities={fmt([sorted(c) for c in s.cities])}",
        f"occ={fmt(s.occupants)}", f"market={fmt(s.market)}",
        f"dplug={fmt(sorted(s.deck_plug))}", f"dsock={fmt(sorted(s.deck_socket))}",
        f"real={fmt([s.real_plug, s.real_socket])}", f"top={fmt(s.top_pending)}",
        f"setup={s.setup_draws}", f"disc={fmt(s.discount)}",
        f"s3p={fmt(s.step3_pending)}", f"s3due={fmt(s.step3_removal_due)}",
        f"s3starts={fmt(s.step3_starts)}", f"s3next={fmt(s.step3_next_round)}",
        f"cap={fmt(s.market_cap)}", f"bottom={fmt(sorted(s.bottom))}",
        f"removed={fmt(sorted(s.removed))}", f"fuel={fmt(s.fuel_market)}",
        f"ustop={fmt(s.uranium_stopped)}", f"done={fmt(sorted(s.done_auction))}",
        f"bought={fmt(s.bought_any)}", f"auction={auction}",
        f"disc_player={fmt(s.discarder)}", f"queue={fmt(s.queue)}", f"qpos={s.qpos}",
        f"ran={fmt([sorted(r) for r in s.ran])}", f"powered={fmt(s.powered)}",
        f"trust={trust}", f"result={result}", f"legal={fmt(s.legal_actions())}",
        f"probs={probs}",
    ]
    return " ".join(parts)


def random_config(rng):
    """(players, map, play_regions, regions, step2, end, max_rounds, max_bid, money, houses,
    trust); -1 / "-" mean "rulebook value"."""
    n = rng.randint(2, 6)
    houses = rng.choice([-1, -1, -1, 5])
    max_bid = rng.choice([400, 120])
    money = rng.choice([50, 50, 80])
    trust = -1 if n == 2 and rng.random() < 0.8 else (0 if n == 2 else -1)
    if rng.random() < 0.4:     # tiny map: reaches step 3 and the end quickly
        play = rng.choice([3, 3, 2])
        return (n, "tiny", play, "-", 3, 6, 40, max_bid, money, houses, trust)
    map_name = rng.choice(["germany", "germany", "germany-2004"])
    rules = pg.Ruleset(num_players=n, map=pg.MAPS[map_name]())
    regions = "-"
    if rng.random() < 0.3:
        regions = ",".join(map(str, rng.choice(rules.region_sets)))
    return (n, map_name, -1, regions, -1, -1, rng.choice([30, 100]), max_bid, money, houses,
            trust)


def make_rules(cfg):
    n, map_name, play, regions, step2, end, rounds, max_bid, money, houses, trust = cfg
    kw = dict(num_players=n, max_rounds=rounds, max_bid=max_bid, start_money=money,
              map=pg.MAPS[map_name]())
    if play >= 0:
        kw["play_regions"] = play
    if regions != "-":
        kw["regions"] = tuple(int(x) for x in regions.split(","))
    if step2 >= 0:
        kw.update(step2_cities=step2, end_cities=end)
    if houses >= 0:
        kw["houses"] = houses
    if trust >= 0:
        kw["trust"] = bool(trust)
    return pg.Ruleset(**kw)


def play(cfg, rng):
    """Random game; returns (actions, expected trace lines)."""
    s = pg.PowerGridState(make_rules(cfg))
    # vary the policy so some games build a lot and some auctions run long
    eager = rng.random()
    lines = ["D " + dump(s)]
    actions = []
    logged = 0
    while not s.is_terminal():
        if s.current_player() == pg.CHANCE:
            outs = s.chance_outcomes()
            a = rng.choices([o for o, _ in outs], [p for _, p in outs])[0]
        else:
            acts = s.legal_actions()
            real = [x for x in acts if x not in (s.codec.PASS, s.codec.DONE)]
            if real and rng.random() < eager:
                if s.phase == pg.Phase.AUCTION_BID and rng.random() < 0.8:
                    a = min(real)      # small raises keep auctions going
                else:
                    a = rng.choice(real)
            else:
                a = rng.choice(acts)
            lines.append("A " + s.action_to_string(a))
        s.apply_action(a)
        actions.append(a)
        lines += ["L " + event_str(e) for e in s.log[logged:]]
        logged = len(s.log)
        lines.append("D " + dump(s))
    lines.append("END")
    return actions, lines


def main():
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rng = random.Random(seed)
    games = []
    for _ in range(num_games):
        cfg = random_config(rng)
        actions, lines = play(cfg, rng)
        games.append((cfg, actions, lines))

    stdin = "".join("game " + " ".join(map(str, cfg)) + "\n" + " ".join(map(str, acts)) + "\n"
                    for cfg, acts, _ in games)
    proc = subprocess.run([TRACE_BIN], input=stdin, capture_output=True, text=True)
    if proc.returncode != 0:
        print("powergrid_trace failed:\n" + proc.stderr)
        sys.exit(1)
    blocks, cur = [], []
    for line in proc.stdout.splitlines():
        cur.append(line)
        if line == "END":
            blocks.append(cur)
            cur = []
    if len(blocks) != len(games):
        print(f"expected {len(games)} games from powergrid_trace, got {len(blocks)}")
        sys.exit(1)

    moves, lines = 0, 0
    for g, ((cfg, actions, want), got) in enumerate(zip(games, blocks)):
        for i in range(max(len(want), len(got))):
            w = want[i] if i < len(want) else "<missing>"
            c = got[i] if i < len(got) else "<missing>"
            if c != w:
                print(f"MISMATCH in game {g} (cfg={cfg}), trace line {i}:")
                print("  python:", w)
                print("  c++:   ", c)
                print("  preceding python lines:\n    " + "\n    ".join(want[max(0, i - 4):i]))
                sys.exit(1)
        moves += len(actions)
        lines += len(want)
    print(f"OK: {num_games} games, {moves} moves, {lines} trace lines identical")


if __name__ == "__main__":
    main()
