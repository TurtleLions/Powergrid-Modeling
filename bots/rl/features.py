"""State features and the abstract action space for learning agents.

Features are ego-centric (the acting seat first) and padded to MAX_PLAYERS
seats, so one network plays any player count on a given map.

Feature versions: checkpoints record the version they were trained with, and
new versions only APPEND features, so a network can be widened to a newer
version without changing how it plays (see model.widen_input).
  1  market, players, plants, fuel, auction, per-city occupancy and build cost
  2  + end-game awareness: every player's powerable cities right now and
       distance to the end, the margin "if the game ended now", and whether one
       more city would end it

Actions: the engine's flat action space has one action per bid amount (481 in
all on the German map). Learning agents use ABSTRACT actions instead, where a
bid is one of a few raises over the minimum legal bid; every other engine
action maps one to one. `AbstractActions.mask` says which abstract actions are
legal now and `to_engine` turns the chosen one back into an engine action.
"""
from __future__ import annotations

import json
from typing import List, Tuple

import numpy as np

MAX_PLAYERS = 6
FEATURE_VERSION = 2   # current version; see the module docstring
KINDS = ("coal", "oil", "garbage", "uranium", "hybrid", "eco")
PHASES = ("CHANCE", "TRUST_SETUP", "AUCTION_SELECT", "AUCTION_BID", "AUCTION_DISCARD",
          "FUEL_DISCARD", "BUY_FUEL", "BUILD", "BUREAUCRACY", "GAME_OVER")
BID_RAISES = (0, 2, 4, 7, 13, 25)   # added to the minimum legal bid
MARKET_SLOTS = 8
PLANT_FEATS = 1 + len(KINDS) + 2    # number, kind one-hot, need, power


def _plant(pl) -> List[float]:
    v = [0.0] * PLANT_FEATS
    if pl is None:
        return v
    v[0] = pl["number"] / 50.0
    v[1 + KINDS.index(pl["kind"])] = 1.0
    v[1 + len(KINDS)] = pl["need"] / 3.0
    v[2 + len(KINDS)] = pl["power"] / 7.0
    return v


class AbstractActions:
    """The abstract action space for one rule set (map and plant limit fixed)."""

    def __init__(self, rules):
        self.c = rules.codec
        self.num_cities = rules.num_cities
        self.slots = rules.plant_limit + 1
        self.PASS, self.DONE = 0, 1
        self.SELECT0 = 2
        self.BID0 = self.SELECT0 + MARKET_SLOTS
        self.BUY0 = self.BID0 + len(BID_RAISES)
        self.BUILD0 = self.BUY0 + 4
        self.DISCARD0 = self.BUILD0 + self.num_cities
        self.RUN0 = self.DISCARD0 + self.slots
        self.n = self.RUN0 + self.slots * 5

    def mask_and_map(self, legal: List[int]) -> Tuple[np.ndarray, dict]:
        """Legal abstract actions, and abstract -> engine action for each."""
        c = self.c
        mask = np.zeros(self.n, dtype=np.bool_)
        amap = {}
        bids = [a - c["BID0"] for a in legal if c["BID0"] <= a < c["BUY0"]]
        for a in legal:
            if a == c["PASS"]:
                x = self.PASS
            elif a == c["DONE"]:
                x = self.DONE
            elif a < c["BID0"]:
                x = self.SELECT0 + (a - c["SELECT0"])
            elif a < c["BUY0"]:
                continue                              # bids handled below
            elif a < c["BUILD0"]:
                x = self.BUY0 + (a - c["BUY0"])
            elif a < c["DISCARD0"]:
                x = self.BUILD0 + (a - c["BUILD0"])
            elif a < c["RUN0"]:
                x = self.DISCARD0 + (a - c["DISCARD0"])
            else:
                x = self.RUN0 + (a - c["RUN0"])
            mask[x] = True
            amap[x] = a
        if bids:
            lo, hi = min(bids), max(bids)            # legal bids are a contiguous range
            for i, raise_ in enumerate(BID_RAISES):
                if lo + raise_ <= hi:
                    mask[self.BID0 + i] = True
                    amap[self.BID0 + i] = c["BID0"] + lo + raise_
        return mask, amap


class Encoder:
    """Ego-centric feature vector for one seat."""

    def __init__(self, rules, version: int = FEATURE_VERSION):
        if version not in (1, 2):
            raise ValueError(f"unknown feature version {version}")
        self.rules = rules
        self.version = version
        self.num_cities = rules.num_cities
        self.limit_slots = rules.plant_limit + 1
        self.size = len(self.encode_dummy())

    def encode_dummy(self):
        return self._encode(None, None, 0)

    def encode(self, state, seat: int) -> np.ndarray:
        return self._encode(state, json.loads(state.to_json()), seat)

    def _encode(self, state, v, seat) -> np.ndarray:
        f: List[float] = []
        n = v["num_players"] if v else 4
        # --- global
        if v:
            f += [v["round"] / 30.0]
            f += [1.0 if v["step"] == s else 0.0 for s in (1, 2, 3)]
            f += [1.0 if v["phase"] == p else 0.0 for p in PHASES]
            f += [float(v["step3_card_in_market"]), float(v["uranium_stopped"]),
                  v["stack_cards"] / 42.0, float(v["top_is_plug"])]
            f += [m / 24.0 for m in v["fuel_market"][:3]] + [v["fuel_market"][3] / 12.0]
            f += [(p / 16.0 if p is not None else 1.5) for p in v["fuel_price"]]
            f += [n / MAX_PLAYERS]
        else:
            f += [0.0] * (1 + 3 + len(PHASES) + 4 + 4 + 4 + 1)
        # --- plant market
        market = v["market"] if v else []
        for k in range(MARKET_SLOTS):
            pl = market[k] if k < len(market) else None
            f += _plant(pl)
            f += ([float(pl["current"]), float(pl["number"] == v["discount"]), pl["min_bid"] / 50.0]
                  if pl else [0.0, 0.0, 0.0])
        # --- players, acting seat first, padded to MAX_PLAYERS
        for i in range(MAX_PLAYERS):
            if v and i < n:
                p = (seat + i) % n
                pv = v["players"][p]
                cap = sum(pl["power"] for pl in pv["plants"])
                f += [1.0, pv["money"] / 100.0, len(pv["cities"]) / 20.0, cap / 20.0,
                      v["order"].index(p) / MAX_PLAYERS, float(pv["done_auction"])]
                f += [s / 10.0 for s in pv["stored"]]
                for j in range(self.limit_slots):
                    f += _plant(pv["plants"][j] if j < len(pv["plants"]) else None)
            else:
                f += [0.0] * (6 + 4 + self.limit_slots * PLANT_FEATS)
        # --- auction
        au = v["auction"] if v else None
        if au:
            pl = next((m for m in market if m["number"] == au["plant"]), None)
            high = -1 if au["high"] is None else (au["high"] - seat) % n
            f += [1.0] + _plant(pl) + [au["bid"] / 100.0]
            f += [1.0 if high == i else 0.0 for i in range(MAX_PLAYERS)]
            f += [float(seat in au["ring"]), len(au["ring"]) / MAX_PLAYERS]
        else:
            f += [0.0] * (1 + PLANT_FEATS + 1 + MAX_PLAYERS + 2)
        # --- cities: in play, occupancy, and what building there costs us now
        in_play = set(v["regions"] or []) if v else set()
        region = self.rules.city_region
        for c in range(self.num_cities):
            if v:
                occ = v["occupants"][c]
                cost = state.build_cost(seat, c)
                f += [float(region[c] in in_play), len(occ) / 3.0, float(seat in occ),
                      sum(1 for q in occ if q != seat and q < n) / 3.0,
                      cost / 50.0 if cost >= 0 else -1.0]
            else:
                f += [0.0] * 5
        if self.version >= 2:
            f += self._endgame(state, v, seat, n)
        return np.asarray(f, dtype=np.float32)

    def _endgame(self, state, v, seat, n) -> List[float]:
        """Who would win if the game ended now, and how close it is to ending."""
        if not v:
            return [0.0] * (2 * MAX_PLAYERS + 3)
        end = v["rules"]["end_cities"]
        supply = [state.max_supply(p) for p in range(n)]
        f: List[float] = []
        for i in range(MAX_PLAYERS):
            if i < n:
                p = (seat + i) % n
                f += [supply[p] / 20.0, (end - len(v["players"][p]["cities"])) / end]
            else:
                f += [0.0, 0.0]
        me = v["players"][seat]
        best_other = max((supply[p], v["players"][p]["money"]) for p in range(n) if p != seat)
        f += [(supply[seat] - best_other[0]) / 10.0,
              float((supply[seat], me["money"]) > best_other),
              float(len(me["cities"]) + 1 >= end)]
        return f
