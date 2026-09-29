"""Scripted Power Grid bots with distinct playing styles.

They serve three purposes: sparring partners with different styles for the RL
league (a bot that only ever meets itself overfits to itself), baselines for
evaluation, and a sanity check that learning agents beat sensible play, not
just random moves.

Each decision is recomputed from the public state, so the bots are stateless
except for the plan of which plants to run in the current bureaucracy.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional

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
    fuel_runs: float     # plant runs of fuel to keep in stock
    upgrade_margin: int  # capacity gain needed to buy a plant when not short


STYLES: Dict[str, Style] = {
    "balanced": Style("balanced", 1.0, 6.0, 0, 10, 1.0, 1),
    "builder": Style("builder", 0.9, 5.0, 2, 0, 1.0, 2),
    "tycoon": Style("tycoon", 1.35, 9.0, -1, 20, 1.5, 1),
    "miser": Style("miser", 0.7, 4.0, 0, 30, 1.0, 2),
}


class HeuristicAgent(Agent):
    def __init__(self, style: str = "balanced"):
        self.style = STYLES[style]
        self.name = style
        self._plan_key = None
        self._plan: List[int] = []

    def reset(self, rules, seat, rng):
        super().reset(rules, seat, rng)
        self.c = rules.codec
        self._plan_key = None

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
        if not self._wants_plant(pl, v) and c["PASS"] in legal:
            return c["PASS"]
        nxt = au["bid"] + 1
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
        if n_cities >= target:
            return c["DONE"]
        end = v["rules"]["end_cities"]
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
        if n_cities == 0:
            # start where many cheap cities are close by
            def room(city):
                return sum(1 for o in builds if o != city and state.distance(city, o) <= 12)
            return c["BUILD0"] + max(affordable, key=lambda x: (room(x), -cost[x], -x))
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
    """Agent by registry name: a style name, "random", or "rl:<checkpoint.pt>"."""
    from .base import RandomAgent
    if name == "random":
        return RandomAgent()
    if name.startswith("rl:"):
        from .rl.model import RLAgent
        return RLAgent(name[3:])
    return HeuristicAgent(name)
