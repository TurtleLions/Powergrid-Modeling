"""Scripted Power Grid bots with distinct playing styles.

They serve three purposes: sparring partners with different styles for the RL
league (a bot that only ever meets itself overfits to itself), baselines for
evaluation, and a sanity check that learning agents beat sensible play, not
just random moves.

Styles differ in strategy, not just in numbers: which plant types they
favour, how much fuel they hoard (denying it to others), how they pick cities
(cheapest, keep room to grow, or crowd opponents), whether they drive up
prices in auctions they do not want to win, and whether they stay small to go
first in the turn order before rushing the end. `randomized` draws a fresh
style every game, for broad coverage in the RL league.

Each decision is recomputed from the public state, so the bots are stateless
except for the plan of which plants to run in the current bureaucracy.
"""
from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .base import Agent, view

FUELS = ("coal", "oil", "garbage", "uranium")
FUEL_IDX = {f: i for i, f in enumerate(FUELS)}
NO_FUEL = 4
EXPENSIVE = 99   # price used for a sold-out fuel


@dataclass(frozen=True)
class Style:
    name: str
    bid_mult: float      # willingness to pay relative to the plant's worth
    power_value: float   # Elektro of worth per city of added capacity
    overbuild: int       # cities built beyond generating capacity
    cash_floor: int      # money kept back when building
    fuel_runs: float     # plant runs of fuel to keep in stock (2 = fill storage)
    upgrade_margin: int  # capacity gain needed to buy a plant when not short
    kind_bonus: Tuple[Tuple[str, float], ...] = ()  # extra worth for plant kinds
    city_mode: str = "cheap"   # cheap | room (keep room to grow) | block (crowd rivals)
    drive_up: float = 0.0      # in auctions it does not want, raise to this x face value
    rush_margin: int = 0       # >0: stay behind the leader until the leader is this
    #                            close to the end, then build as much as possible


STYLES: Dict[str, Style] = {
    "balanced": Style("balanced", 1.0, 6.0, 0, 10, 1.0, 1),
    "builder": Style("builder", 0.9, 5.0, 2, 0, 1.0, 2),
    "tycoon": Style("tycoon", 1.35, 9.0, -1, 20, 1.5, 1),
    "miser": Style("miser", 0.7, 4.0, 0, 30, 1.0, 2),
    "eco": Style("eco", 1.0, 6.0, 1, 10, 1.0, 1, kind_bonus=(("eco", 18.0), ("uranium", -10.0))),
    "nuclear": Style("nuclear", 1.1, 6.0, 1, 10, 1.5, 1, kind_bonus=(("uranium", 14.0),)),
    "hoarder": Style("hoarder", 1.0, 6.0, 1, 5, 2.0, 1),
    "blocker": Style("blocker", 1.0, 6.0, 1, 5, 1.0, 1, city_mode="block"),
    "planner": Style("planner", 1.0, 6.0, 1, 10, 1.0, 1, city_mode="room"),
    "driver": Style("driver", 1.0, 6.0, 1, 10, 1.0, 1, drive_up=0.9),
    "turtle": Style("turtle", 1.1, 7.0, 2, 15, 1.2, 1, rush_margin=4),
}
RANDOMIZED = "randomized"


def random_style(rng) -> Style:
    """A fresh style for one game, drawn across the range the named styles span."""
    bonus = tuple((k, rng.uniform(-10.0, 18.0)) for k in ("eco", "uranium", "hybrid", "garbage")
                  if rng.random() < 0.3)
    return Style(RANDOMIZED, rng.uniform(0.6, 1.5), rng.uniform(3.0, 10.0), rng.randint(-1, 3),
                 rng.randint(0, 35), rng.uniform(1.0, 2.0), rng.randint(1, 3), kind_bonus=bonus,
                 city_mode=rng.choice(["cheap", "cheap", "room", "block"]),
                 drive_up=rng.choice([0.0, 0.0, 0.0, 0.7, 0.9]),
                 rush_margin=rng.choice([0, 0, 0, 3, 5]))


class HeuristicAgent(Agent):
    def __init__(self, style: str = "balanced"):
        self.randomized = style == RANDOMIZED
        self.style = None if self.randomized else STYLES[style]
        self.name = style
        self._plan_key = None
        self._plan: List[int] = []

    def reset(self, rules, seat, rng):
        super().reset(rules, seat, rng)
        self.c = rules.codec
        self._plan_key = None
        if self.randomized:
            self.style = random_style(rng)

    # ---- valuation helpers ------------------------------------------------
    @staticmethod
    def _run_cost(pl: dict, prices: List[int]) -> int:
        """Fuel cost of one run at current market prices."""
        if pl["kind"] == "eco":
            return 0
        if pl["kind"] == "hybrid":
            return pl["need"] * min(prices[0], prices[1])
        return pl["need"] * prices[FUEL_IDX[pl["kind"]]]

    def _worth(self, pl: dict, v: dict) -> float:
        """What this plant is worth to us, in Elektro."""
        me = v["players"][self.seat]
        prices = [p if p is not None else EXPENSIVE for p in v["fuel_price"]]
        powers = [q["power"] for q in me["plants"]]
        if len(powers) < self.rules.plant_limit:
            gain = pl["power"]
        else:
            gain = pl["power"] - min(powers)   # it replaces our smallest plant
        if gain <= 0:
            return 0.0
        worth = pl["number"] + self.style.power_value * gain - self._run_cost(pl, prices)
        worth += dict(self.style.kind_bonus).get(pl["kind"], 0.0)
        if pl["kind"] == "uranium" and v.get("uranium_stopped"):
            worth -= 15                         # no more uranium resupply
        return self.style.bid_mult * worth

    def _wants_plant(self, pl: dict, v: dict) -> bool:
        me = v["players"][self.seat]
        capacity = sum(q["power"] for q in me["plants"])
        short = capacity < len(me["cities"]) + max(self.style.overbuild, 0) + 2
        if len(me["plants"]) < 2 or short:
            return True
        smallest = min(q["power"] for q in me["plants"])
        full = len(me["plants"]) >= self.rules.plant_limit
        gain = pl["power"] - smallest if full else pl["power"]
        return gain >= self.style.upgrade_margin

    # ---- decisions ----------------------------------------------------------
    def act(self, state) -> int:
        v = view(state)
        legal = state.legal_actions()
        handler = {
            "AUCTION_SELECT": self._select,
            "AUCTION_BID": self._bid,
            "AUCTION_DISCARD": self._discard,
            "FUEL_DISCARD": self._return_fuel,
            "BUY_FUEL": self._buy_fuel,
            "BUILD": self._build,
            "BUREAUCRACY": self._run,
        }.get(v["phase"])
        a = handler(state, v, legal) if handler else None
        return a if a in legal else legal[0]

    def _select(self, state, v, legal):
        c, me = self.c, v["players"][self.seat]
        current = [pl for pl in v["market"] if pl["current"]]
        best, best_surplus = None, None
        for k, pl in enumerate(current):
            if c["SELECT0"] + k not in legal:
                continue
            surplus = self._worth(pl, v) - pl["min_bid"]
            if best is None or surplus > best_surplus:
                best, best_surplus = k, surplus
        if best is None:
            return c["PASS"]
        if c["PASS"] not in legal:                   # first round: must buy
            return c["SELECT0"] + best
        if best_surplus > 0 and self._wants_plant(current[best], v):
            return c["SELECT0"] + best
        return c["PASS"]

    def _bid(self, state, v, legal):
        c, au = self.c, v["auction"]
        pl = next(p for p in v["market"] if p["number"] == au["plant"])
        me = v["players"][self.seat]
        limit = min(me["money"], int(self._worth(pl, v)))
        if au["high"] is None:                       # we opened: bid the minimum
            return c["BID0"] + pl["min_bid"]
        nxt = au["bid"] + 1
        if not self._wants_plant(pl, v) and c["PASS"] in legal:
            # price driving: make the others pay, but stop well below face value
            drive = int(self.style.drive_up * (pl["number"] + 5 * pl["power"]))
            if self.style.drive_up and nxt <= min(drive, me["money"]) and c["BID0"] + nxt in legal:
                return c["BID0"] + nxt
            return c["PASS"]
        if nxt <= limit and c["BID0"] + nxt in legal:
            return c["BID0"] + nxt
        return c["PASS"]

    def _discard(self, state, v, legal):
        plants = v["players"][self.seat]["plants"][:-1]   # the new plant is kept
        j = min(range(len(plants)), key=lambda i: (plants[i]["power"], plants[i]["number"]))
        return self.c["DISCARD0"] + j

    def _return_fuel(self, state, v, legal):
        prices = [p if p is not None else EXPENSIVE for p in v["fuel_price"]]
        # give back whichever is cheaper to buy again
        return min(legal, key=lambda a: prices[a - self.c["BUY0"]])

    def _fuel_targets(self, v) -> List[int]:
        me = v["players"][self.seat]
        prices = [p if p is not None else EXPENSIVE for p in v["fuel_price"]]
        want = [0.0] * 4
        for pl in me["plants"]:
            if pl["kind"] == "eco":
                continue
            if pl["kind"] == "hybrid":
                f = 0 if prices[0] <= prices[1] else 1
            else:
                f = FUEL_IDX[pl["kind"]]
            want[f] += pl["need"] * self.style.fuel_runs
        return [int(round(w)) for w in want]

    def _buy_fuel(self, state, v, legal):
        c, me = self.c, v["players"][self.seat]
        want = self._fuel_targets(v)
        options = []
        for f in range(4):
            a = c["BUY0"] + f
            price = v["fuel_price"][f]
            if a in legal and me["stored"][f] < want[f] and me["money"] - price >= self.style.cash_floor // 2:
                options.append((price, a))
        return min(options)[1] if options else c["DONE"]

    def _build(self, state, v, legal):
        c, me = self.c, v["players"][self.seat]
        n_cities = len(me["cities"])
        capacity = sum(pl["power"] for pl in me["plants"])
        target = max(capacity + self.style.overbuild, 1)
        end = v["rules"]["end_cities"]
        leader = max(len(p["cities"]) for i, p in enumerate(v["players"]) if i != self.seat)
        if self.style.rush_margin and leader < end - self.style.rush_margin:
            # stay behind the leader: go early when buying fuel and building
            target = min(target, max(leader - 1, 2))
        if n_cities >= target:
            return c["DONE"]
        if n_cities + 1 >= end and v["round"] < 30:
            # only end the game when we would be the one powering the most cities
            # (after round 30, end it regardless so games cannot stall)
            others = max(sum(pl["power"] for pl in p["plants"])
                         for i, p in enumerate(v["players"]) if i != self.seat)
            if capacity < others:
                return c["DONE"]
        builds = [a - c["BUILD0"] for a in legal if c["BUILD0"] <= a < c["DISCARD0"]]
        if not builds:
            return c["DONE"]
        cost = {city: state.build_cost(self.seat, city) for city in builds}
        affordable = [x for x in builds if me["money"] - cost[x] >= self.style.cash_floor]
        if not affordable:
            return c["DONE"]
        def room(city):   # free cities close by: room to grow
            return sum(1 for o in builds if o != city and state.distance(city, o) <= 12)

        rivals = [c_ for i, p in enumerate(v["players"]) if i != self.seat for c_ in p["cities"]]

        def crowding(city):   # rival cities close by
            return sum(1 for o in rivals if state.distance(city, o) <= 8)

        mode = self.style.city_mode
        if mode == "block" and rivals:
            return c["BUILD0"] + min(affordable, key=lambda x: (cost[x] - 4 * crowding(x), x))
        if n_cities == 0 or mode == "room":
            return c["BUILD0"] + min(affordable, key=lambda x: (cost[x] - 3 * room(x), x))
        return c["BUILD0"] + min(affordable, key=lambda x: (cost[x], x))

    def _run(self, state, v, legal):
        c, me = self.c, v["players"][self.seat]
        key = (v["round"], self.seat)
        if self._plan_key != key:
            self._plan_key = key
            self._plan = self._plan_runs(v)
        while self._plan:
            number = self._plan[0]
            j = next((i for i, pl in enumerate(me["plants"]) if pl["number"] == number), None)
            base = c["RUN0"] + (j if j is not None else 0) * 5
            options = [a for a in legal if base <= a < base + 5] if j is not None else []
            self._plan.pop(0)
            if options:
                return options[-1]   # hybrids: burn as much coal as possible
        return c["DONE"]

    def _plan_runs(self, v) -> List[int]:
        """Plants to run: power as many cities as possible, then burn the least."""
        me = v["players"][self.seat]
        prices = [p if p is not None else EXPENSIVE for p in v["fuel_price"]]
        stored, cities = me["stored"], len(me["cities"])
        best, best_key = [], (0, 0)
        plants = me["plants"]
        for k in range(1, len(plants) + 1):
            for combo in itertools.combinations(plants, k):
                need, hybrid = [0] * 4, 0
                for pl in combo:
                    if pl["kind"] == "hybrid":
                        hybrid += pl["need"]
                    elif pl["kind"] != "eco":
                        need[FUEL_IDX[pl["kind"]]] += pl["need"]
                if any(need[f] > stored[f] for f in range(4)):
                    continue
                if hybrid > stored[0] - need[0] + stored[1] - need[1]:
                    continue
                powered = min(cities, sum(pl["power"] for pl in combo))
                cost = sum(self._run_cost(pl, prices) for pl in combo)
                key_ = (powered, -cost)
                if key_ > best_key:
                    best, best_key = [pl["number"] for pl in combo], key_
        return best


def make(name: str) -> Agent:
    """Agent by registry name: a style name, "randomized", "random",
    "rl:<checkpoint.pt>", or "mcts:<simulations>[+norm][+c<c_puct>][+b<batch>]:<checkpoint.pt>[:<value.pt>]"
    (norm: min-max normalised Q, b: leaves per network call; see SearchAgent;
    e.g. "mcts:100+norm+c3+b8:...")."""
    from .base import RandomAgent
    if name == "random":
        return RandomAgent()
    if name.startswith("rl:"):
        from .rl.model import RLAgent
        return RLAgent(name[3:])
    if name.startswith("mcts:"):
        from .search import SearchAgent
        _, sims, path, *value = name.split(":")
        sims, *opts = sims.split("+")
        c_puct = [float(o[1:]) for o in opts if o.startswith("c")]
        batch = [int(o[1:]) for o in opts if o.startswith("b")]
        return SearchAgent(path, sims=int(sims), value_path=value[0] if value else "",
                           normalize_q="norm" in opts, c_puct=c_puct[0] if c_puct else 1.5,
                           batch=batch[0] if batch else 1)
    return HeuristicAgent(name)
