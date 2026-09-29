"""
powergrid_core.py -- pure-Python reference engine for the board game Power Grid.

Rules follow the Recharged rulebook (2F-Spiele 2018 / Rio Grande Games;
Power-Grid-Recharged-Rules.pdf in this repo). Comments cite it as [R p.N].
The Recharged German board is not printed in the rulebook: its data comes from
github.com/boardgamers/powergrid (mapRecharged), agrees with every
connection shown in the rulebook's building example [R p.6], and Mannheim-
Saarbrücken = 16 was checked against a physical board. The power plant
cards agree across four independent implementations and every card the
rulebook describes.

Design goals
  * No dependencies. The fast C++ port and OpenSpiel game live in cpp/.
  * Every decision is a small discrete action (see Codec).
  * Chance (regions, turn order, plant draws) is explicit, so MCTS/CFR can enumerate it.
  * Every auction / purchase / build is written to an event log
    (state.log) -- that log is the input to collusion metrics.
  * Clone copies only dynamic fields, but it is still ~0.5 ms here; use the
    C++ engine (~13 ns clone) for search.
  * This file is the *spec*: change rules here first, then port them to
    cpp/powergrid_engine.cc and run tools/difftest.py (see cpp/README.md).

Modelling choices the rulebook leaves open (see also the Trust notes below)
  * Money is public information: payments are open, so it can be tracked.
  * The players "choose" the playing zone [R p.2]; we draw it uniformly among
    all adjacent sets of the right size (or fix it with Ruleset.regions).
  * Seat order (clockwise, used for bidding) is player id order.
  * Fuel is bought one unit at a time; a player may rearrange fuel between
    plants at any time [R p.5], so storage is checked for the player as a whole.
  * Hidden setup removals are drawn lazily: the stack is sampled exactly as a
    shuffled stack with face-down removals, without revealing which were removed.
  * If the Step 3 card comes up in the same phase 5 in which Step 2 started, the
    phase-5 market update of that round is skipped (the card already removed
    the lowest plant "without replacements").
  * max_bid caps bids to keep the action space finite; set it above any
    reachable amount of money. max_rounds is a safety cap, not a rule.
  * The optional "Default Starting Cities" rule for experienced players is
    deliberately not used.

2 players: "Against the Trust" [R p.9-10] (on by default for 2 players)
  * The Trust takes its plant once per round: right after the first plant of
    the round is bought, or after the first player opts out.
  * The Trust buys fuel for its plants in ascending plant order.
  * Trust setup houses must be placed next to (one connection away from) an
    earlier Trust house; if none is free, any empty city in play is allowed.
  * The uranium phase-out is triggered by players buying plant 39, not by the
    Trust taking it.
"""
from __future__ import annotations

import bisect
import copy
import itertools
import math
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Tuple

CHANCE = -1      # matches pyspiel.PlayerId.CHANCE
TERMINAL = -4    # matches pyspiel.PlayerId.TERMINAL


# --------------------------------------------------------------------------
# Static data
# --------------------------------------------------------------------------
class Fuel(IntEnum):
    COAL = 0
    OIL = 1
    GARBAGE = 2
    URANIUM = 3


NUM_FUELS = 4
NO_FUEL = 4  # used by RUN actions for plants that burn nothing

KIND_IDX = {"coal": 0, "oil": 1, "garbage": 2, "uranium": 3, "hybrid": 4, "eco": 5}
FUELS_OF = {
    "coal": (Fuel.COAL,), "oil": (Fuel.OIL,), "garbage": (Fuel.GARBAGE,),
    "uranium": (Fuel.URANIUM,), "hybrid": (Fuel.COAL, Fuel.OIL), "eco": (),
}

# Resource market: coal, oil and garbage 3 tokens on each of spaces 1-8; uranium
# 1 token on each of spaces 1-8, 10, 12, 14, 16 [R p.7 example]. Price of the k-th
# cheapest slot, cheapest first. Tokens always sit on the most expensive slots
# (bought cheapest first, refilled most expensive first [R p.7]), so with `count`
# left the next costs prices[len(prices) - count].
_THREE_PER_LEVEL = tuple(1 + i // 3 for i in range(24))
FUEL_PRICES = {
    Fuel.COAL: _THREE_PER_LEVEL,
    Fuel.OIL: _THREE_PER_LEVEL,
    Fuel.GARBAGE: _THREE_PER_LEVEL,
    Fuel.URANIUM: (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16),
}


@dataclass(frozen=True)
class Plant:
    number: int
    kind: str   # coal | oil | garbage | uranium | hybrid | eco
    need: int   # fuel units burned per run
    power: int  # cities powered per run


# The 42 power plants [R p.2: numbers 03-40, 42, 44, 46, 50].
_P = Plant
PLANTS: Tuple[Plant, ...] = (
    _P(3, "oil", 2, 1), _P(4, "coal", 2, 1), _P(5, "hybrid", 2, 1), _P(6, "garbage", 1, 1),
    _P(7, "oil", 3, 2), _P(8, "coal", 3, 2), _P(9, "oil", 1, 1), _P(10, "coal", 2, 2),
    _P(11, "uranium", 1, 2), _P(12, "hybrid", 2, 2), _P(13, "eco", 0, 1), _P(14, "garbage", 2, 2),
    _P(15, "coal", 2, 3), _P(16, "oil", 2, 3), _P(17, "uranium", 1, 2), _P(18, "eco", 0, 2),
    _P(19, "garbage", 2, 3), _P(20, "coal", 3, 5), _P(21, "hybrid", 2, 4), _P(22, "eco", 0, 2),
    _P(23, "uranium", 1, 3), _P(24, "garbage", 2, 4), _P(25, "coal", 2, 5), _P(26, "oil", 2, 5),
    _P(27, "eco", 0, 3), _P(28, "uranium", 1, 4), _P(29, "hybrid", 1, 4), _P(30, "garbage", 3, 6),
    _P(31, "coal", 3, 6), _P(32, "oil", 3, 6), _P(33, "eco", 0, 4), _P(34, "uranium", 1, 5),
    _P(35, "oil", 1, 5), _P(36, "coal", 3, 7), _P(37, "eco", 0, 4), _P(38, "garbage", 3, 7),
    _P(39, "uranium", 1, 6), _P(40, "oil", 2, 6), _P(42, "coal", 2, 6), _P(44, "eco", 0, 5),
    _P(46, "hybrid", 3, 7), _P(50, "eco", 0, 6),
)
MAX_PLUG = 15              # [R p.3] plants 03-15 have a plug on the back, the rest a socket
NUCLEAR_PHASE_OUT_PLANT = 39   # [R p.7] Germany: no uranium resupply once a player buys it

# Resupply per round: RESUPPLY[players][step-1] = (coal, oil, garbage, uranium).
# The rulebook prints the 5-player card [R p.3, p.7]; the other rows are on the
# refill cards and match the boardgamers.space implementation.
RESUPPLY = {
    2: ((3, 2, 1, 1), (4, 2, 2, 1), (3, 4, 3, 1)),
    3: ((4, 2, 1, 1), (5, 3, 2, 1), (3, 4, 3, 1)),
    4: ((5, 3, 2, 1), (6, 4, 3, 2), (4, 5, 4, 2)),
    5: ((5, 4, 3, 2), (7, 5, 3, 3), (5, 6, 5, 2)),
    6: ((7, 5, 3, 2), (9, 6, 5, 3), (6, 7, 6, 3)),
}
# Per player count: (areas in play [R p.2], plug plants removed, socket plants
# removed [R p.3], plant limit [R p.5], step-2 trigger [R p.7], game end [R p.8]).
PLAYER_COUNT_RULES = {
    2: (3, 1, 5, 3, 7, 18),
    3: (3, 2, 6, 3, 7, 17),
    4: (4, 1, 3, 3, 7, 17),
    5: (5, 0, 0, 3, 7, 15),
    6: (5, 0, 0, 3, 6, 14),
}
MARKET_SIZE = {1: 8, 2: 8, 3: 6}
HOUSES_PER_PLAYER = 22   # [R p.2]
TRUST_HOUSES = 16        # [R p.9]
TRUST_START_HOUSES = 6
# [R p.6] Income by number of cities powered (index = cities powered, capped at 20).
INCOME = (10, 22, 33, 44, 54, 64, 73, 82, 90, 98, 105, 112, 118, 124, 129, 134,
          138, 142, 145, 148, 150)

GERMANY_REGIONS = ("north-west", "north-east", "east", "west", "south-west", "south-east")
# The Recharged German board (default map).
GERMANY_CITIES = (  # (name, region)
    ("Flensburg", "north-west"), ("Kiel", "north-west"), ("Hamburg", "north-west"),
    ("Hannover", "north-west"), ("Bremen", "north-west"), ("Wilhelmshaven", "north-west"),
    ("Cuxhaven", "north-west"), ("Lübeck", "north-east"), ("Rostock", "north-east"),
    ("Schwerin", "north-east"), ("Stralsund", "north-east"), ("Magdeburg", "north-east"),
    ("Berlin", "north-east"), ("Frankfurt (Oder)", "north-east"), ("Halle", "east"),
    ("Leipzig", "east"), ("Dresden", "east"), ("Erfurt", "east"), ("Fulda", "east"),
    ("Würzburg", "east"), ("Nürnberg", "east"), ("Osnabrück", "west"), ("Münster", "west"),
    ("Essen", "west"), ("Duisburg", "west"), ("Dortmund", "west"), ("Düsseldorf", "west"),
    ("Kassel", "west"), ("Aachen", "south-west"), ("Köln", "south-west"),
    ("Trier", "south-west"), ("Mainz", "south-west"), ("Frankfurt (Main)", "south-west"),
    ("Saarbrücken", "south-west"), ("Mannheim", "south-west"), ("Freiburg", "south-east"),
    ("Stuttgart", "south-east"), ("Konstanz", "south-east"), ("Augsburg", "south-east"),
    ("Regensburg", "south-east"), ("München", "south-east"), ("Passau", "south-east"),
)
GERMANY_CONNECTIONS = (  # (city, city, cost)
    ("Flensburg", "Kiel", 4), ("Kiel", "Hamburg", 8), ("Kiel", "Lübeck", 4),
    ("Hamburg", "Hannover", 17), ("Hamburg", "Bremen", 11), ("Hamburg", "Cuxhaven", 11),
    ("Hamburg", "Lübeck", 6), ("Hamburg", "Schwerin", 8), ("Bremen", "Wilhelmshaven", 11),
    ("Bremen", "Hannover", 10), ("Bremen", "Cuxhaven", 8), ("Bremen", "Osnabrück", 11),
    ("Hannover", "Schwerin", 19), ("Hannover", "Magdeburg", 15), ("Hannover", "Erfurt", 19),
    ("Hannover", "Osnabrück", 16), ("Hannover", "Kassel", 15),
    ("Wilhelmshaven", "Osnabrück", 14), ("Lübeck", "Schwerin", 6), ("Schwerin", "Rostock", 6),
    ("Schwerin", "Stralsund", 19), ("Schwerin", "Berlin", 18), ("Schwerin", "Magdeburg", 16),
    ("Stralsund", "Rostock", 19), ("Stralsund", "Berlin", 15), ("Berlin", "Magdeburg", 10),
    ("Berlin", "Frankfurt (Oder)", 6), ("Berlin", "Halle", 17), ("Magdeburg", "Halle", 11),
    ("Frankfurt (Oder)", "Leipzig", 21), ("Frankfurt (Oder)", "Dresden", 16),
    ("Halle", "Leipzig", 0), ("Leipzig", "Dresden", 13), ("Erfurt", "Halle", 6),
    ("Erfurt", "Dresden", 19), ("Erfurt", "Nürnberg", 21), ("Erfurt", "Fulda", 13),
    ("Erfurt", "Kassel", 15), ("Fulda", "Würzburg", 11), ("Fulda", "Kassel", 8),
    ("Fulda", "Frankfurt (Main)", 8), ("Würzburg", "Nürnberg", 8),
    ("Würzburg", "Frankfurt (Main)", 13), ("Würzburg", "Mannheim", 10),
    ("Würzburg", "Stuttgart", 12), ("Würzburg", "Augsburg", 19), ("Nürnberg", "Augsburg", 18),
    ("Nürnberg", "Regensburg", 12), ("Osnabrück", "Münster", 7), ("Osnabrück", "Kassel", 20),
    ("Münster", "Essen", 6), ("Münster", "Dortmund", 2), ("Essen", "Duisburg", 0),
    ("Essen", "Düsseldorf", 2), ("Essen", "Dortmund", 5), ("Dortmund", "Kassel", 18),
    ("Dortmund", "Köln", 10), ("Dortmund", "Frankfurt (Main)", 20),
    ("Kassel", "Frankfurt (Main)", 13), ("Düsseldorf", "Aachen", 9), ("Düsseldorf", "Köln", 4),
    ("Aachen", "Köln", 7), ("Aachen", "Trier", 19), ("Trier", "Köln", 20),
    ("Trier", "Mainz", 18), ("Trier", "Saarbrücken", 11), ("Mainz", "Frankfurt (Main)", 0),
    ("Mainz", "Köln", 21), ("Mainz", "Saarbrücken", 10), ("Mainz", "Mannheim", 11),
    ("Mannheim", "Saarbrücken", 16), ("Mannheim", "Stuttgart", 6),
    ("Stuttgart", "Saarbrücken", 17), ("Stuttgart", "Freiburg", 16),
    ("Stuttgart", "Konstanz", 16), ("Stuttgart", "Augsburg", 15), ("Konstanz", "Freiburg", 14),
    ("Konstanz", "Augsburg", 17), ("Regensburg", "Augsburg", 13),
    ("Regensburg", "München", 10), ("Regensburg", "Passau", 12), ("Augsburg", "München", 6),
    ("Passau", "München", 14),
)
# The 2004 German board, for reference: Torgelow and Wiesbaden instead of Stralsund
# and Mainz, Essen-Dortmund 4 and Mannheim-Saarbrücken 11.
GERMANY_2004_CITIES = (  # (name, region)
    ("Flensburg", "north-west"), ("Kiel", "north-west"), ("Hamburg", "north-west"),
    ("Hannover", "north-west"), ("Bremen", "north-west"), ("Wilhelmshaven", "north-west"),
    ("Cuxhaven", "north-west"), ("Lübeck", "north-east"), ("Rostock", "north-east"),
    ("Schwerin", "north-east"), ("Torgelow", "north-east"), ("Magdeburg", "north-east"),
    ("Berlin", "north-east"), ("Frankfurt (Oder)", "north-east"), ("Halle", "east"),
    ("Leipzig", "east"), ("Dresden", "east"), ("Erfurt", "east"), ("Fulda", "east"),
    ("Würzburg", "east"), ("Nürnberg", "east"), ("Osnabrück", "west"), ("Münster", "west"),
    ("Essen", "west"), ("Duisburg", "west"), ("Dortmund", "west"), ("Düsseldorf", "west"),
    ("Kassel", "west"), ("Aachen", "south-west"), ("Köln", "south-west"),
    ("Trier", "south-west"), ("Wiesbaden", "south-west"), ("Frankfurt (Main)", "south-west"),
    ("Saarbrücken", "south-west"), ("Mannheim", "south-west"), ("Freiburg", "south-east"),
    ("Stuttgart", "south-east"), ("Konstanz", "south-east"), ("Augsburg", "south-east"),
    ("Regensburg", "south-east"), ("München", "south-east"), ("Passau", "south-east"),
)
GERMANY_2004_CONNECTIONS = (  # (city, city, cost)
    ("Flensburg", "Kiel", 4), ("Kiel", "Hamburg", 8), ("Kiel", "Lübeck", 4),
    ("Hamburg", "Hannover", 17), ("Hamburg", "Bremen", 11), ("Hamburg", "Cuxhaven", 11),
    ("Hamburg", "Lübeck", 6), ("Hamburg", "Schwerin", 8), ("Bremen", "Wilhelmshaven", 11),
    ("Bremen", "Hannover", 10), ("Bremen", "Cuxhaven", 8), ("Bremen", "Osnabrück", 11),
    ("Hannover", "Schwerin", 19), ("Hannover", "Magdeburg", 15), ("Hannover", "Erfurt", 19),
    ("Hannover", "Osnabrück", 16), ("Hannover", "Kassel", 15),
    ("Wilhelmshaven", "Osnabrück", 14), ("Lübeck", "Schwerin", 6), ("Schwerin", "Rostock", 6),
    ("Schwerin", "Torgelow", 19), ("Schwerin", "Berlin", 18), ("Schwerin", "Magdeburg", 16),
    ("Torgelow", "Rostock", 19), ("Torgelow", "Berlin", 15), ("Berlin", "Magdeburg", 10),
    ("Berlin", "Frankfurt (Oder)", 6), ("Berlin", "Halle", 17), ("Magdeburg", "Halle", 11),
    ("Frankfurt (Oder)", "Leipzig", 21), ("Frankfurt (Oder)", "Dresden", 16),
    ("Halle", "Leipzig", 0), ("Leipzig", "Dresden", 13), ("Erfurt", "Halle", 6),
    ("Erfurt", "Dresden", 19), ("Erfurt", "Nürnberg", 21), ("Erfurt", "Fulda", 13),
    ("Erfurt", "Kassel", 15), ("Fulda", "Würzburg", 11), ("Fulda", "Kassel", 8),
    ("Fulda", "Frankfurt (Main)", 8), ("Würzburg", "Nürnberg", 8),
    ("Würzburg", "Frankfurt (Main)", 13), ("Würzburg", "Mannheim", 10),
    ("Würzburg", "Stuttgart", 12), ("Würzburg", "Augsburg", 19), ("Nürnberg", "Augsburg", 18),
    ("Nürnberg", "Regensburg", 12), ("Osnabrück", "Münster", 7), ("Osnabrück", "Kassel", 20),
    ("Münster", "Essen", 6), ("Münster", "Dortmund", 2), ("Essen", "Duisburg", 0),
    ("Essen", "Düsseldorf", 2), ("Essen", "Dortmund", 4), ("Dortmund", "Kassel", 18),
    ("Dortmund", "Köln", 10), ("Dortmund", "Frankfurt (Main)", 20),
    ("Kassel", "Frankfurt (Main)", 13), ("Düsseldorf", "Aachen", 9), ("Düsseldorf", "Köln", 4),
    ("Aachen", "Köln", 7), ("Aachen", "Trier", 19), ("Trier", "Köln", 20),
    ("Trier", "Wiesbaden", 18), ("Trier", "Saarbrücken", 11),
    ("Wiesbaden", "Frankfurt (Main)", 0), ("Wiesbaden", "Köln", 21),
    ("Wiesbaden", "Saarbrücken", 10), ("Wiesbaden", "Mannheim", 11),
    ("Mannheim", "Saarbrücken", 11), ("Mannheim", "Stuttgart", 6),
    ("Stuttgart", "Saarbrücken", 17), ("Stuttgart", "Freiburg", 16),
    ("Stuttgart", "Konstanz", 16), ("Stuttgart", "Augsburg", 15), ("Konstanz", "Freiburg", 14),
    ("Konstanz", "Augsburg", 17), ("Regensburg", "Augsburg", 13),
    ("Regensburg", "München", 10), ("Regensburg", "Passau", 12), ("Augsburg", "München", 6),
    ("Passau", "München", 14),
)


class MapSpec:
    """City graph with regions. Building costs use the cheapest path through
    the cities in play, so distances depend on the chosen regions (cached)."""

    def __init__(self, name: str, city_names: Sequence[str], city_region: Sequence[int],
                 region_names: Sequence[str], edges: Sequence[Tuple[int, int, int]],
                 nuclear_phase_out: bool = False):
        self.name = name
        self.city_names = tuple(city_names)
        self.city_region = tuple(city_region)
        self.region_names = tuple(region_names)
        self.num_cities = len(city_names)
        self.num_regions = len(region_names)
        self.edges = tuple(edges)
        self.nuclear_phase_out = nuclear_phase_out
        adj = set()
        self.neighbors: List[List[int]] = [[] for _ in range(self.num_cities)]
        for a, b, _ in self.edges:
            self.neighbors[a].append(b)
            self.neighbors[b].append(a)
            ra, rb = self.city_region[a], self.city_region[b]
            if ra != rb:
                adj.add((min(ra, rb), max(ra, rb)))
        self.region_adjacent = frozenset(adj)
        self._dist: Dict[Tuple[int, ...], List[List[int]]] = {}

    @staticmethod
    def from_names(name, regions, cities, connections, **kw) -> "MapSpec":
        idx = {c: i for i, (c, _) in enumerate(cities)}
        ridx = {r: i for i, r in enumerate(regions)}
        return MapSpec(name, [c for c, _ in cities], [ridx[r] for _, r in cities], regions,
                       [(idx[a], idx[b], w) for a, b, w in connections], **kw)

    def valid_region_sets(self, k: int) -> List[Tuple[int, ...]]:
        """All sets of k regions that form one adjacent block, in lexicographic order."""
        out = []
        for combo in itertools.combinations(range(self.num_regions), k):
            seen, todo = {combo[0]}, [combo[0]]
            while todo:
                r = todo.pop()
                for s in combo:
                    if s not in seen and (min(r, s), max(r, s)) in self.region_adjacent:
                        seen.add(s)
                        todo.append(s)
            if len(seen) == k:
                out.append(combo)
        return out

    def active_cities(self, regions: Tuple[int, ...]) -> List[int]:
        return [c for c in range(self.num_cities) if self.city_region[c] in regions]

    def dist(self, regions: Tuple[int, ...]) -> List[List[int]]:
        """Cheapest connection costs using only cities in `regions` (Floyd-Warshall)
        [R p.6: never use cities or connections outside the playing zone]."""
        if regions in self._dist:
            return self._dist[regions]
        n, inf = self.num_cities, 10 ** 9
        active = set(self.active_cities(regions))
        d = [[inf] * n for _ in range(n)]
        for i in active:
            d[i][i] = 0
        for a, b, w in self.edges:
            if a in active and b in active:
                d[a][b] = min(d[a][b], w)
                d[b][a] = min(d[b][a], w)
        for k in range(n):
            dk = d[k]
            for i in range(n):
                dik = d[i][k]
                if dik == inf:
                    continue
                di = d[i]
                for j in range(n):
                    if dik + dk[j] < di[j]:
                        di[j] = dik + dk[j]
        self._dist[regions] = d
        return d


def germany_map() -> MapSpec:
    return MapSpec.from_names("germany", GERMANY_REGIONS, GERMANY_CITIES, GERMANY_CONNECTIONS,
                              nuclear_phase_out=True)


def germany_2004_map() -> MapSpec:
    return MapSpec.from_names("germany-2004", GERMANY_REGIONS, GERMANY_2004_CITIES,
                              GERMANY_2004_CONNECTIONS, nuclear_phase_out=True)


def tiny_map() -> MapSpec:
    """12-city test map: three regions of four cities on a ring, with chords."""
    n = 12
    edges = [(i, (i + 1) % n, 3 + (i % 4)) for i in range(n)] + [(0, 6, 8), (3, 9, 9), (2, 8, 7)]
    return MapSpec("tiny", [f"T{i}" for i in range(n)], [i // 4 for i in range(n)],
                   ["a", "b", "c"], edges)


MAPS = {"germany": germany_map, "germany-2004": germany_2004_map, "tiny": tiny_map}


@dataclass
class Ruleset:
    num_players: int = 4
    map: MapSpec = field(default_factory=germany_map)
    plants: Tuple[Plant, ...] = PLANTS
    start_money: int = 50
    # None = take the value for this player count from PLAYER_COUNT_RULES
    play_regions: Optional[int] = None
    plug_removed: Optional[int] = None
    socket_removed: Optional[int] = None
    plant_limit: Optional[int] = None
    step2_cities: Optional[int] = None
    end_cities: Optional[int] = None
    trust: Optional[bool] = None               # None = on for 2 players
    regions: Optional[Tuple[int, ...]] = None   # fixed regions in play; None = chance
    max_rounds: int = 100          # safety cap (not a rule) so random play terminates
    max_bid: int = 400
    houses: int = HOUSES_PER_PLAYER
    slot_costs: Tuple[int, ...] = (10, 15, 20)
    income: Tuple[int, ...] = INCOME
    resupply: Dict[int, tuple] = field(default_factory=lambda: dict(RESUPPLY))
    fuel_totals: Tuple[int, ...] = (24, 24, 24, 12)   # pieces in the game
    fuel_init: Tuple[int, ...] = (24, 18, 9, 2)       # [R p.2] coal 1-8, oil 3-8, garbage 6-8, uranium 14-16
    keep_log: bool = True

    def __post_init__(self):
        if self.num_players not in self.resupply:
            raise NotImplementedError("only player counts with a resupply table are supported")
        regions, plugs, sockets, limit, step2, end = PLAYER_COUNT_RULES[self.num_players]
        if self.play_regions is None:
            self.play_regions = min(regions, self.map.num_regions)
        if self.plug_removed is None:
            self.plug_removed = plugs
        if self.socket_removed is None:
            self.socket_removed = sockets
        if self.plant_limit is None:
            self.plant_limit = limit
        if self.step2_cities is None:
            self.step2_cities = step2
        if self.end_cities is None:
            self.end_cities = end
        if self.trust is None:
            self.trust = self.num_players == 2
        if self.trust and self.num_players != 2:
            raise ValueError("the Trust is a 2-player rule")
        self.region_sets = self.map.valid_region_sets(self.play_regions)
        if not self.region_sets:
            raise ValueError("no adjacent set of regions of the requested size")
        if self.regions is not None:
            self.regions = tuple(sorted(self.regions))
            if self.regions not in self.region_sets:
                raise ValueError(f"regions {self.regions} are not an adjacent set of "
                                 f"{self.play_regions} regions")
        self.plant_by_number = {p.number: p for p in self.plants}
        plugs_ = sum(1 for p in self.plants if p.number <= MAX_PLUG)
        sockets_ = len(self.plants) - plugs_
        if plugs_ < 9 + self.plug_removed or sockets_ < self.socket_removed:
            raise ValueError("not enough plants for the setup")


def tiny_ruleset(num_players: int = 4, **kw) -> Ruleset:
    """Small config for tests: the tiny map, all three regions, ends quickly."""
    kw.setdefault("play_regions", 3)
    return Ruleset(num_players=num_players, map=tiny_map(), step2_cities=3, end_cities=6,
                   max_rounds=40, **kw)


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------
class Phase(IntEnum):
    CHANCE = 0
    TRUST_SETUP = 1      # 2 players: place the Trust's six starting houses
    AUCTION_SELECT = 2
    AUCTION_BID = 3
    AUCTION_DISCARD = 4
    FUEL_DISCARD = 5     # after a discard: choose coal or oil to return (hybrid overflow)
    BUY_FUEL = 6
    BUILD = 7
    BUREAUCRACY = 8
    GAME_OVER = 9
    # transient markers: never a resting phase
    FUEL_START = 10
    ROUND_START = 11
    BUREAU_START = 12


NUM_RESTING_PHASES = 10


class Codec:
    """Flat integer action space.

    PASS     : pass the auction round / drop out of a bidding ring
    DONE     : finish buying fuel / building / bureaucracy
    SELECT+k : put purchasable market slot k up for auction
    BID+x    : bid x (absolute amount)
    BUY+f    : buy one unit of fuel f (in FUEL_DISCARD: return one unit of f)
    BUILD+c  : build in city c (in TRUST_SETUP: place a Trust house there)
    DISCARD+j: discard your j-th plant (never the one just bought)
    RUN+j*5+f: run plant j burning fuel f (f == NO_FUEL for eco plants);
               for hybrid plants f is instead the number of coal units burned
               (the rest of the plant's need is paid in oil)
    """

    MAX_SLOTS = 8

    def __init__(self, rules: Ruleset):
        self.rules = rules
        self.PASS = 0
        self.DONE = 1
        self.SELECT0 = 2
        self.BID0 = self.SELECT0 + self.MAX_SLOTS
        self.BUY0 = self.BID0 + rules.max_bid + 1
        self.BUILD0 = self.BUY0 + NUM_FUELS
        self.DISCARD0 = self.BUILD0 + rules.map.num_cities
        self.RUN0 = self.DISCARD0 + (rules.plant_limit + 1)
        self.num_actions = self.RUN0 + (rules.plant_limit + 1) * (NUM_FUELS + 1)

    def describe(self, a: int) -> str:
        if a == self.PASS:
            return "PASS/DROP"
        if a == self.DONE:
            return "DONE"
        if a < self.BID0:
            return f"SELECT slot {a - self.SELECT0}"
        if a < self.BUY0:
            return f"BID {a - self.BID0}"
        if a < self.BUILD0:
            return f"BUY {Fuel(a - self.BUY0).name}"
        if a < self.DISCARD0:
            return f"BUILD city {a - self.BUILD0}"
        if a < self.RUN0:
            return f"DISCARD plant slot {a - self.DISCARD0}"
        j, f = divmod(a - self.RUN0, NUM_FUELS + 1)
        return f"RUN plant slot {j} option {f} (fuel code; coal count for hybrids)"


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
_DYNAMIC = (
    "round", "step", "phase", "chance_kind", "regions", "order", "money", "plants", "stored",
    "cities", "occupants", "market", "deck_plug", "deck_socket", "real_plug", "real_socket",
    "top_pending", "setup_draws", "discount", "step3_pending", "step3_removal_due",
    "step3_starts", "step3_next_round", "market_cap", "bottom", "removed", "fuel_market",
    "uranium_stopped", "done_auction", "bought_any", "auction", "discarder", "queue", "qpos",
    "ran", "powered", "after_draw", "trust_plants", "trust_stored", "trust_houses",
    "trust_due", "trust_took", "trust_setup", "log", "result",
)


class PowerGridState:
    def __init__(self, rules: Ruleset):
        self.rules = rules
        self.codec = Codec(rules)
        self.n = rules.num_players
        n = self.n
        self.round = 1
        self.step = 1
        self.phase = Phase.CHANCE
        # [R p.2] the playing zone is chosen, then player order is drawn at random,
        # then the plant market is dealt [R p.3]
        self.regions: Optional[Tuple[int, ...]] = rules.regions
        if self.regions is None and len(rules.region_sets) == 1:
            self.regions = rules.region_sets[0]
        self.chance_kind = "initial_order" if self.regions is not None else "regions"
        self.order = list(range(n))
        self.money = [rules.start_money] * n
        self.plants: List[List[int]] = [[] for _ in range(n)]
        self.stored = [[0] * NUM_FUELS for _ in range(n)]
        self.cities: List[List[int]] = [[] for _ in range(n)]
        # occupants[c] lists who holds the 10/15/20 spaces; the Trust is player n
        self.occupants: List[List[int]] = [[] for _ in range(rules.map.num_cities)]
        self.market: List[int] = []     # current + future market, kept sorted
        # The stack is drawn lazily. deck_plug / deck_socket are the plants that may
        # still be in it; real_plug / real_socket count how many of them really are
        # (the rest were removed face down at setup [R p.3]).
        nums = sorted(p.number for p in rules.plants)
        self.deck_plug = [x for x in nums if x <= MAX_PLUG]
        self.deck_socket = [x for x in nums if x > MAX_PLUG]
        self.real_plug = 0
        self.real_socket = 0
        self.top_pending = False        # the set-aside plug on top of the stack [R p.3]
        self.setup_draws = MARKET_SIZE[1]
        self.discount: Optional[int] = None    # plant under the discount token [R p.4]
        self.step3_pending = False      # step-3 card drawn in phase 2 sits in the market
        self.step3_removal_due = False  # remove the card and the lowest plant when full
        self.step3_starts: Optional[str] = None   # "now" | "next_round" for that removal
        self.step3_next_round = False   # step 3 begins at the next round [R p.8 case 2]
        self.market_cap: Optional[int] = None  # "do not draw replacements" while refilling
        self.bottom: List[int] = []     # plants placed under the step-3 card
        self.removed: List[int] = []
        self.fuel_market = list(rules.fuel_init)
        self.uranium_stopped = False    # [R p.7] German nuclear power phase-out
        self.done_auction: List[int] = []
        self.bought_any = False
        self.auction: Optional[dict] = None
        self.discarder: Optional[int] = None
        self.queue: List[int] = []
        self.qpos = 0
        self.ran: List[List[int]] = [[] for _ in range(n)]
        self.powered = [0] * n
        self.after_draw = Phase.AUCTION_SELECT
        # 2 players: the Trust [R p.9-10]
        self.trust_plants: List[int] = []
        self.trust_stored = [0] * NUM_FUELS
        self.trust_houses = TRUST_HOUSES - TRUST_START_HOUSES if rules.trust else 0
        self.trust_due = False          # the Trust takes a plant at the next opportunity
        self.trust_took = False         # ... once per round
        self.trust_setup: List[int] = []   # who places the remaining starting houses
        self.log: List[dict] = []
        self.result: Optional[List[float]] = None

    # ---- cloning -------------------------------------------------------
    def clone(self) -> "PowerGridState":
        new = object.__new__(PowerGridState)
        new.rules, new.codec, new.n = self.rules, self.codec, self.n
        for name in _DYNAMIC:
            setattr(new, name, copy.deepcopy(getattr(self, name)))
        return new

    def __deepcopy__(self, memo):
        return self.clone()

    # ---- basic queries -------------------------------------------------
    @property
    def trust_id(self) -> int:
        return self.n

    def is_terminal(self) -> bool:
        return self.phase == Phase.GAME_OVER

    def returns(self) -> List[float]:
        return list(self.result) if self.result is not None else [0.0] * self.n

    def purchasable(self) -> List[int]:
        return self.market[:4] if self.step < 3 else self.market[:6]

    def min_bid(self, plant: int) -> int:
        """[R p.4] the discounted plant may be had for 1 Elektro."""
        return 1 if plant == self.discount else plant

    def _plant(self, number: int) -> Plant:
        return self.rules.plant_by_number[number]

    def _starter(self) -> Optional[int]:
        for p in self.order:
            if p not in self.done_auction:
                return p
        return None

    def current_player(self) -> int:
        ph = self.phase
        if ph == Phase.GAME_OVER:
            return TERMINAL
        if ph == Phase.CHANCE:
            return CHANCE
        if ph == Phase.TRUST_SETUP:
            return self.trust_setup[0]
        if ph == Phase.AUCTION_SELECT:
            return self._starter()
        if ph == Phase.AUCTION_BID:
            return self.auction["ring"][self.auction["pos"]]
        if ph in (Phase.AUCTION_DISCARD, Phase.FUEL_DISCARD):
            return self.discarder
        return self.queue[self.qpos]

    # ---- fuel helpers --------------------------------------------------
    def _price(self, f: int) -> Optional[int]:
        c = self.fuel_market[f]
        if c <= 0:
            return None
        prices = FUEL_PRICES[Fuel(f)]
        return prices[len(prices) - c]

    def _capacity(self, p: int):
        cap = [0] * NUM_FUELS
        hybrid = 0
        for num in self.plants[p]:
            pl = self._plant(num)
            if pl.kind == "hybrid":
                hybrid += 2 * pl.need
            elif pl.kind != "eco":
                cap[int(FUELS_OF[pl.kind][0])] += 2 * pl.need
        return cap, hybrid

    def _fits(self, p: int, stored) -> bool:
        cap, hybrid = self._capacity(p)
        over = max(0, stored[Fuel.COAL] - cap[Fuel.COAL]) + max(0, stored[Fuel.OIL] - cap[Fuel.OIL])
        return (over <= hybrid and stored[Fuel.GARBAGE] <= cap[Fuel.GARBAGE]
                and stored[Fuel.URANIUM] <= cap[Fuel.URANIUM])

    def _trim_options(self, p: int) -> List[int]:
        """Fuels the player may return to get back within storage after a discard.

        [R p.5] fuel that no retained plant can hold goes back to the supply. Only
        coal vs oil competing for hybrid space is a real choice; excess garbage and
        uranium (and excess of a single fuel) is returned automatically.
        """
        cap, hybrid = self._capacity(p)
        s = self.stored[p]
        oc = max(0, s[Fuel.COAL] - cap[Fuel.COAL])
        oo = max(0, s[Fuel.OIL] - cap[Fuel.OIL])
        if oc + oo <= hybrid:
            return []
        return [f for f, over in ((Fuel.COAL, oc), (Fuel.OIL, oo)) if over > 0]

    def _return_fuel(self, p: int, f: int, forced: bool):
        self.stored[p][f] -= 1
        self._log("return_fuel", player=p, fuel=Fuel(f).name, forced=forced)

    def _continue_trim(self):
        """Return excess fuel after a discard; stop when the player must choose."""
        p = self.discarder
        cap, _ = self._capacity(p)
        for f in (Fuel.GARBAGE, Fuel.URANIUM):
            while self.stored[p][f] > cap[f]:
                self._return_fuel(p, f, forced=True)
        while True:
            opts = self._trim_options(p)
            if not opts:
                break
            if len(opts) > 1:
                self.phase = Phase.FUEL_DISCARD
                return
            self._return_fuel(p, opts[0], forced=True)
        self.discarder = None
        self._refill(Phase.AUCTION_SELECT)

    def _build_cost(self, p: int, c: int) -> Optional[int]:
        """[R p.5-6] price of the lowest free space plus the cheapest connection
        (through any cities in play) from one of the player's cities."""
        m = self.rules.map
        if m.city_region[c] not in self.regions or len(self.cities[p]) >= self.rules.houses:
            return None
        occ = self.occupants[c]
        if len(occ) >= self.step or p in occ:
            return None
        conn = 0
        if self.cities[p]:
            dist = m.dist(self.regions)
            conn = min(dist[o][c] for o in self.cities[p])
        return self.rules.slot_costs[len(occ)] + conn

    def _trust_setup_options(self) -> List[int]:
        """[R p.9] the first Trust house goes anywhere in play, the others next to
        an earlier one (one connection away). Falls back to any empty city."""
        m, t = self.rules.map, self.trust_id
        empty = [c for c in m.active_cities(self.regions) if not self.occupants[c]]
        placed = [c for c in range(m.num_cities) if t in self.occupants[c]]
        if not placed:
            return empty
        near = {nb for c in placed for nb in m.neighbors[c]}
        return [c for c in empty if c in near] or empty

    # ---- chance --------------------------------------------------------
    def _cards_left(self) -> bool:
        return self.top_pending or self.real_plug + self.real_socket > 0

    def chance_outcomes(self) -> List[Tuple[int, float]]:
        if self.chance_kind == "regions":
            k = len(self.rules.region_sets)
            return [(i, 1.0 / k) for i in range(k)]
        if self.chance_kind == "initial_order":
            k = math.factorial(self.n)
            return [(i, 1.0 / k) for i in range(k)]
        if self.chance_kind == "setup" or self.top_pending:
            # the market is dealt from the plug plants [R p.3]; the top card is a plug
            k = len(self.deck_plug)
            return [(num, 1.0 / k) for num in self.deck_plug]
        # The next card is a real plug with probability real_plug / real, and then
        # any plug candidate equally likely (by symmetry); likewise for sockets.
        total = self.real_plug + self.real_socket
        out = []
        if self.real_plug:
            w = self.real_plug / total / len(self.deck_plug)
            out += [(num, w) for num in self.deck_plug]
        if self.real_socket:
            w = self.real_socket / total / len(self.deck_socket)
            out += [(num, w) for num in self.deck_socket]
        return sorted(out)

    # ---- legal actions -------------------------------------------------
    def legal_actions(self) -> List[int]:
        ph, c = self.phase, self.codec
        if ph == Phase.GAME_OVER:
            return []
        if ph == Phase.CHANCE:
            return [a for a, _ in self.chance_outcomes()]
        p = self.current_player()
        if ph == Phase.TRUST_SETUP:
            return [c.BUILD0 + city for city in self._trust_setup_options()]
        money = self.money[p]
        if ph == Phase.AUCTION_SELECT:
            acts = [c.SELECT0 + k for k, num in enumerate(self.purchasable())
                    if money >= self.min_bid(num)]
            if self.round > 1 or not acts:   # everyone must buy in round 1 (if they can)
                acts.insert(0, c.PASS)
            return acts
        if ph == Phase.AUCTION_BID:
            au = self.auction
            hi = min(money, self.rules.max_bid)
            if au["high"] is None:
                return [c.BID0 + x for x in range(self.min_bid(au["plant"]), hi + 1)]
            return [c.PASS] + [c.BID0 + x for x in range(au["bid"] + 1, hi + 1)]
        if ph == Phase.AUCTION_DISCARD:
            # [R p.5] "They may not choose to scrap the just bought power plant"
            return [c.DISCARD0 + j for j in range(len(self.plants[p]) - 1)]
        if ph == Phase.FUEL_DISCARD:
            return [c.BUY0 + f for f in self._trim_options(p)]
        if ph == Phase.BUY_FUEL:
            acts = [c.DONE]
            for f in range(NUM_FUELS):
                price = self._price(f)
                if price is None or money < price:
                    continue
                st = list(self.stored[p])
                st[f] += 1
                if self._fits(p, st):
                    acts.append(c.BUY0 + f)
            return acts
        if ph == Phase.BUILD:
            acts = [c.DONE]
            for city in range(self.rules.map.num_cities):
                cost = self._build_cost(p, city)
                if cost is not None and cost <= money:
                    acts.append(c.BUILD0 + city)
            return acts
        if ph == Phase.BUREAUCRACY:
            acts = [c.DONE]
            for j, num in enumerate(self.plants[p]):
                if j in self.ran[p]:
                    continue
                pl = self._plant(num)
                base = c.RUN0 + j * (NUM_FUELS + 1)
                if pl.kind == "eco":
                    acts.append(base + NO_FUEL)
                elif pl.kind == "hybrid":
                    st = self.stored[p]
                    for k in range(pl.need + 1):   # k coal + (need - k) oil
                        if st[Fuel.COAL] >= k and st[Fuel.OIL] >= pl.need - k:
                            acts.append(base + k)
                else:
                    f = FUELS_OF[pl.kind][0]
                    if self.stored[p][f] >= pl.need:
                        acts.append(base + int(f))
            return acts
        raise RuntimeError(f"no legal actions defined for {ph}")

    # ---- applying actions ---------------------------------------------
    def apply_action(self, a: int):
        self._dispatch(a, forced=False)
        self._auto_advance()

    def _auto_advance(self):
        """Apply forced PASS/DONE moves so agents only see real decisions."""
        c = self.codec
        while self.phase not in (Phase.CHANCE, Phase.GAME_OVER):
            acts = self.legal_actions()
            if len(acts) == 1 and acts[0] in (c.PASS, c.DONE):
                self._dispatch(acts[0], forced=True)
            else:
                break

    def _dispatch(self, a: int, forced: bool):
        ph = self.phase
        if ph == Phase.CHANCE:
            self._do_chance(a)
        elif ph == Phase.TRUST_SETUP:
            self._do_trust_setup(a)
        elif ph == Phase.AUCTION_SELECT:
            self._do_select(a, forced)
        elif ph == Phase.AUCTION_BID:
            self._do_bid(a, forced)
        elif ph == Phase.AUCTION_DISCARD:
            self._do_discard(a)
        elif ph == Phase.FUEL_DISCARD:
            self._return_fuel(self.discarder, a - self.codec.BUY0, forced=False)
            self._continue_trim()
        elif ph == Phase.BUY_FUEL:
            self._do_buy(a)
        elif ph == Phase.BUILD:
            self._do_build(a)
        elif ph == Phase.BUREAUCRACY:
            self._do_run(a)
        else:
            raise RuntimeError(f"cannot apply action in {ph}")

    def _log(self, kind: str, **kw):
        if self.rules.keep_log:
            self.log.append(dict(type=kind, round=self.round, step=self.step, **kw))

    # -- chance
    def _do_chance(self, a: int):
        if self.chance_kind == "regions":
            self.regions = self.rules.region_sets[a]
            self._log("regions", regions=list(self.regions))
            self.chance_kind = "initial_order"
            return
        if self.chance_kind == "initial_order":
            perms = list(itertools.permutations(range(self.n)))
            self.order = list(perms[a])
            self.chance_kind = "setup"
            return
        if self.chance_kind == "setup":
            self.deck_plug.remove(a)
            bisect.insort(self.market, a)
            self._log("draw", plant=a)
            self.setup_draws -= 1
            if self.setup_draws == 0:
                self._finish_setup()
            return
        self._take_from_stack(a)
        bisect.insort(self.market, a)
        self._log("draw", plant=a)
        if self.after_draw == Phase.AUCTION_SELECT and self.discount is not None \
                and a < self.discount:
            # [R p.4] the first replacement smaller than the discounted plant is
            # removed together with the discount token
            self.market.remove(a)
            self.removed.append(a)
            self.discount = None
            self._log("discount_lost", plant=a)
        self._refill(self.after_draw)

    def _take_from_stack(self, a: int):
        if a <= MAX_PLUG:
            self.deck_plug.remove(a)
            if not self.top_pending:
                self.real_plug -= 1
        else:
            self.deck_socket.remove(a)
            self.real_socket -= 1
        self.top_pending = False

    def _finish_setup(self):
        """[R p.3] one more plug is set aside for the top of the stack; then plugs
        and sockets are removed face down and the rest shuffled together."""
        r = self.rules
        self.top_pending = True
        self.real_plug = len(self.deck_plug) - 1 - r.plug_removed
        self.real_socket = len(self.deck_socket) - r.socket_removed
        if r.trust:
            # [R p.9] starting player 1, other player 2, starting 2, other 1
            s, o = self.order
            self.trust_setup = [s, o, o, s, s, o]
            self.phase = Phase.TRUST_SETUP
        else:
            self._begin_auction_phase()

    def _market_target(self) -> int:
        # [R p.8] a pending step-3 card occupies the last slot of the future market
        target = MARKET_SIZE[self.step] - (1 if self.step3_pending else 0)
        if self.market_cap is not None:
            target = min(target, self.market_cap)
        if self.step3_next_round:
            # the card removed a plant "without replacements"; step 3 starts next round
            target = min(target, len(self.market))
        return target

    def _refill(self, after: Phase):
        """Draw until the market is full, then move on to `after`.

        Stops at a chance node whenever a hidden card must be drawn; _do_chance
        resumes here.
        """
        self.after_draw = after
        while True:
            short = len(self.market) < self._market_target()
            if short and self._cards_left():
                self.phase = Phase.CHANCE
                self.chance_kind = "plant"
                return
            if (short and self.step < 3 and not self.step3_pending
                    and not self.step3_removal_due and not self.step3_next_round):
                self._step3_card(after)
                continue
            if self.step3_removal_due:
                self._step3_removal()
                continue
            break                        # full, or the stack is exhausted [R p.7]
        self.market_cap = None
        self.phase = after
        self._settle()

    def _step3_card(self, after: Phase):
        """[R p.8] The Step 3 card comes up: shuffle the plants that were put under
        the stack; they are the new stack."""
        self._log("step3_card")
        self.removed.extend(self.deck_plug + self.deck_socket)   # face-down removals
        self.deck_plug = sorted(x for x in self.bottom if x <= MAX_PLUG)
        self.deck_socket = sorted(x for x in self.bottom if x > MAX_PLUG)
        self.real_plug, self.real_socket = len(self.deck_plug), len(self.deck_socket)
        self.bottom = []
        if after == Phase.AUCTION_SELECT:
            # case 1, phase 2: the card waits at the end of the future market and
            # replacements keep coming until the phase ends
            self.step3_pending = True
            return
        # case 2, phase 5 (or the end of phase 2): card and lowest plant leave,
        # no replacements
        self.market_cap = len(self.market)
        self.step3_starts = "now" if after == Phase.FUEL_START else "next_round"
        self._schedule_step3_removal()

    def _schedule_step3_removal(self):
        if self.step == 1:
            # [R p.8] Step 3 before Step 2: first perform the Step 2 changes
            self._start_step2()
        self.step3_removal_due = True

    def _step3_removal(self):
        """Remove the Step 3 card and the lowest plant; no replacements."""
        self.step3_removal_due = False
        self.step3_pending = False
        if self.market:
            self.removed.append(self.market.pop(0))
        self.market_cap = len(self.market)
        if self.step3_starts == "now":
            self.step = 3
            self._log("step3")
        else:
            self.step3_next_round = True
        self.step3_starts = None

    def _start_step2(self):
        """[R p.7] once: remove the lowest plant and replace it."""
        self.step = 2
        if self.market:
            self.removed.append(self.market.pop(0))
        self._log("step2")

    def _settle(self):
        ph = self.phase
        if ph == Phase.AUCTION_SELECT:
            self._advance_auction()
        elif ph == Phase.FUEL_START:
            self._start_fuel()
        elif ph == Phase.BUREAU_START:
            self._start_bureaucracy()
        elif ph == Phase.ROUND_START:
            self._start_round()

    # -- Trust setup
    def _do_trust_setup(self, a: int):
        city = a - self.codec.BUILD0
        p = self.trust_setup.pop(0)
        self.occupants[city].append(self.trust_id)
        self._log("trust_place", player=p, city=city)
        if not self.trust_setup:
            self._begin_auction_phase()

    # -- auction
    def _begin_auction_phase(self):
        self.done_auction = []
        self.bought_any = False
        self.auction = None
        self.trust_due = False
        self.trust_took = False
        # [R p.4] the discount token goes on the smallest plant in the current market
        self.discount = self.market[0] if self.market else None
        if self.discount is not None:
            self._log("discount", plant=self.discount)
        self.phase = Phase.AUCTION_SELECT

    def _advance_auction(self):
        self.phase = Phase.AUCTION_SELECT
        if self.trust_due:
            self._trust_take()
            return
        if self._starter() is None:
            self._end_auction()

    def _trust_take(self):
        """[R p.9] the Trust takes the biggest plant in the current market, for
        free; with 3 plants only if it beats its smallest, which it scraps."""
        self.trust_due = False
        self.trust_took = True
        current = self.purchasable()
        if current:
            best = max(current)
            if len(self.trust_plants) < 3 or best > min(self.trust_plants):
                if len(self.trust_plants) >= 3:
                    small = min(self.trust_plants)
                    self.trust_plants.remove(small)
                    self.removed.append(small)
                    self._log("trust_scrap", plant=small)
                self.market.remove(best)
                self.trust_plants.append(best)
                if best == self.discount:
                    self.discount = None
                self._log("trust_take", plant=best)
                self._refill(Phase.AUCTION_SELECT)
                return
        self._advance_auction()

    def _end_auction(self):
        if self.discount is not None:
            # [R p.4] nobody bought the discounted plant: it leaves and is replaced
            self.market.remove(self.discount)
            self.removed.append(self.discount)
            self._log("discount_scrapped", plant=self.discount)
            self.discount = None
        if self.step3_pending:
            # [R p.8] after phase 2: remove the card and the lowest plant, no
            # replacements; step 3 starts in phase 3
            self.step3_starts = "now"
            self._schedule_step3_removal()
        self._refill(Phase.FUEL_START)

    def _do_select(self, a: int, forced: bool):
        c, p = self.codec, self.current_player()
        if a == c.PASS:
            self.done_auction.append(p)
            self._log("pass_round", player=p, forced=forced)
            if self.rules.trust and not self.trust_took and p == self.order[0]:
                self.trust_due = True   # [R p.9] after the first player opted out
            self._advance_auction()
            return
        plant = self.purchasable()[a - c.SELECT0]
        # bidding goes clockwise in seat order (seat == player id), starting with the selector
        seq = [(p + i) % self.n for i in range(self.n)]
        ring = [q for q in seq if q not in self.done_auction]
        self.auction = dict(plant=plant, selector=p, bid=0, high=None, ring=ring, pos=0)
        self._log("select", player=p, plant=plant, ring=list(ring))
        if len(ring) == 1:   # [R p.5] the last player pays the minimum bid
            self.auction["high"], self.auction["bid"] = p, self.min_bid(plant)
            self._sell()
        else:
            self.phase = Phase.AUCTION_BID

    def _do_bid(self, a: int, forced: bool):
        c, p, au = self.codec, self.current_player(), self.auction
        if a == c.PASS:
            au["ring"].pop(au["pos"])
            self._log("drop", player=p, plant=au["plant"], at_bid=au["bid"], forced=forced)
            if au["pos"] >= len(au["ring"]):
                au["pos"] = 0
            if len(au["ring"]) == 1:
                self._sell()
        else:
            au["bid"], au["high"] = a - c.BID0, p
            au["pos"] = (au["pos"] + 1) % len(au["ring"])
            self._log("bid", player=p, plant=au["plant"], amount=au["bid"])

    def _sell(self):
        au = self.auction
        buyer, price, plant = au["high"], au["bid"], au["plant"]
        self.money[buyer] -= price
        self.plants[buyer].append(plant)
        self.market.remove(plant)
        self.done_auction.append(buyer)
        self.bought_any = True
        self._log("sale", buyer=buyer, plant=plant, price=price, selector=au["selector"],
                  bidders=list(au["ring"]))
        self.auction = None
        if plant == self.discount:
            self.discount = None
        if (plant == NUCLEAR_PHASE_OUT_PLANT and self.rules.map.nuclear_phase_out
                and not self.uranium_stopped):
            self.uranium_stopped = True
            self._log("uranium_stopped")
        if self.rules.trust and not self.trust_took:
            self.trust_due = True        # [R p.9] after the first purchase of the round
        if len(self.plants[buyer]) > self.rules.plant_limit:
            self.phase = Phase.AUCTION_DISCARD
            self.discarder = buyer
        else:
            self._refill(Phase.AUCTION_SELECT)

    def _do_discard(self, a: int):
        p = self.discarder
        num = self.plants[p].pop(a - self.codec.DISCARD0)
        self.removed.append(num)
        self._log("discard", player=p, plant=num)
        self._continue_trim()

    # -- fuel
    def _start_fuel(self):
        if self.round == 1:
            self.order = self._ranked_order()   # round 1: order is re-set once plants are bought
            self._log("reorder", order=list(self.order))
        self.queue, self.qpos = list(reversed(self.order)), 0
        self.phase = Phase.BUY_FUEL

    def _do_buy(self, a: int):
        c, p = self.codec, self.current_player()
        if a == c.DONE:
            self.qpos += 1
            if self.rules.trust and self.qpos == 1:
                self._trust_buy()        # [R p.9] the Trust is always second in order
            if self.qpos >= len(self.queue):
                self._start_build()
            return
        f = a - c.BUY0
        price = self._price(f)
        self.money[p] -= price
        self.fuel_market[f] -= 1
        self.stored[p][f] += 1
        self._log("buy_fuel", player=p, fuel=Fuel(f).name, price=price)

    def _trust_buy(self):
        """[R p.9] The Trust takes (for free) the fuel for one run of each plant,
        as much as is available; hybrids alternate coal and oil, starting with coal."""
        for num in sorted(self.trust_plants):
            pl = self._plant(num)
            for i in range(pl.need):
                if pl.kind == "hybrid":
                    pref = (Fuel.COAL, Fuel.OIL) if i % 2 == 0 else (Fuel.OIL, Fuel.COAL)
                    choices = [f for f in pref if self.fuel_market[f] > 0]
                else:
                    f = FUELS_OF[pl.kind][0]
                    choices = [f] if self.fuel_market[f] > 0 else []
                if not choices:
                    break
                f = choices[0]
                self.fuel_market[f] -= 1
                self.trust_stored[f] += 1
                self._log("trust_fuel", plant=num, fuel=Fuel(f).name)

    # -- build
    def _start_build(self):
        self.queue, self.qpos = list(reversed(self.order)), 0
        self.phase = Phase.BUILD

    def _do_build(self, a: int):
        c, p = self.codec, self.current_player()
        if a == c.DONE:
            self.qpos += 1
            if self.qpos >= len(self.queue):
                self._end_build()
            return
        city = a - c.BUILD0
        cost = self._build_cost(p, city)
        was_empty = not self.occupants[city]
        self.money[p] -= cost
        self.occupants[city].append(p)
        self.cities[p].append(city)
        self._log("build", player=p, city=city, cost=cost, occupants=list(self.occupants[city]))
        if self.rules.trust and was_empty and self.trust_houses > 0:
            # [R p.10] the Trust blocks the 15 space of every newly connected city
            self.occupants[city].append(self.trust_id)
            self.trust_houses -= 1
            self._log("trust_block", city=city)

    def _end_build(self):
        # [R p.8] The game ends immediately after phase 4 once someone has the
        # end-game number of cities: no income is paid.
        if max(len(x) for x in self.cities) >= self.rules.end_cities:
            self._end_game()
            return
        # [R p.7] Step 2 starts at the beginning of phase 5
        if self.step == 1 and max(len(x) for x in self.cities) >= self.rules.step2_cities:
            self._start_step2()
        self._refill(Phase.BUREAU_START)

    # -- bureaucracy
    def _start_bureaucracy(self):
        self.queue, self.qpos = list(self.order), 0
        self.ran = [[] for _ in range(self.n)]
        self.powered = [0] * self.n
        self.phase = Phase.BUREAUCRACY

    def _do_run(self, a: int):
        c, p = self.codec, self.current_player()
        if a == c.DONE:
            cap = sum(self._plant(self.plants[p][j]).power for j in self.ran[p])
            powered = min(len(self.cities[p]), cap)
            income = self.rules.income[min(powered, len(self.rules.income) - 1)]
            self.money[p] += income
            self.powered[p] = powered
            self._log("income", player=p, powered=powered, income=income)
            self.qpos += 1
            if self.qpos >= len(self.queue):
                self._finish_round()
            return
        j, f = divmod(a - c.RUN0, NUM_FUELS + 1)
        pl = self._plant(self.plants[p][j])
        burned = [0] * NUM_FUELS
        if pl.kind == "hybrid":
            burned[Fuel.COAL], burned[Fuel.OIL] = f, pl.need - f
        elif f != NO_FUEL:
            burned[f] = pl.need
        for g in range(NUM_FUELS):
            self.stored[p][g] -= burned[g]
        self.ran[p].append(j)
        self._log("run", player=p, plant=pl.number, burned=burned)

    def _finish_round(self):
        # [R p.10] the Trust burns its fuel; it goes back to the supply
        self.trust_stored = [0] * NUM_FUELS
        if self.round >= self.rules.max_rounds:   # safety cap, not a rule
            self._end_game()
            return
        add = self.rules.resupply[self.n][self.step - 1]
        for f in range(NUM_FUELS):
            if f == Fuel.URANIUM and self.uranium_stopped:
                continue
            held = sum(s[f] for s in self.stored)
            bank = self.rules.fuel_totals[f] - self.fuel_market[f] - held
            room = len(FUEL_PRICES[Fuel(f)]) - self.fuel_market[f]
            self.fuel_market[f] += max(0, min(add[f], bank, room))
        if self.market and not self.step3_next_round:
            if self.step < 3:
                self.bottom.append(self.market.pop())   # highest plant under the step-3 card
            else:
                self.removed.append(self.market.pop(0))  # step 3: lowest plant leaves
        self._refill(Phase.ROUND_START)

    def _ranked_order(self) -> List[int]:
        return sorted(range(self.n),
                      key=lambda p: (-len(self.cities[p]), -max(self.plants[p], default=0)))

    def _start_round(self):
        self.round += 1
        if self.step3_next_round:
            self.step3_next_round = False
            self.step = 3
            self._log("step3")
        self.order = self._ranked_order()
        self._begin_auction_phase()

    def _max_supply(self, p: int) -> int:
        """Most cities p can power with the plants and fuel they have [R p.8]."""
        best = 0
        s = self.stored[p]
        plants = [self._plant(num) for num in self.plants[p]]
        for k in range(len(plants) + 1):
            for combo in itertools.combinations(plants, k):
                need = [0] * NUM_FUELS
                hybrid = 0
                for pl in combo:
                    if pl.kind == "hybrid":
                        hybrid += pl.need
                    elif pl.kind != "eco":
                        need[FUELS_OF[pl.kind][0]] += pl.need
                if any(need[f] > s[f] for f in range(NUM_FUELS)):
                    continue
                if hybrid > (s[Fuel.COAL] - need[Fuel.COAL]) + (s[Fuel.OIL] - need[Fuel.OIL]):
                    continue
                best = max(best, sum(pl.power for pl in combo))
        return min(best, len(self.cities[p]))

    def _end_game(self):
        # [R p.8] most cities suppliable, then most money; remaining ties share the win
        self.powered = [self._max_supply(p) for p in range(self.n)]
        score = [(self.powered[p], self.money[p]) for p in range(self.n)]
        best = max(score)
        winners = [p for p in range(self.n) if score[p] == best]
        self.result = [1.0 / len(winners) if p in winners else 0.0 for p in range(self.n)]
        self.phase = Phase.GAME_OVER
        self._log("game_over", winners=winners, score=score)

    # ---- observations --------------------------------------------------
    def observation(self, viewer: int) -> Dict[str, List[float]]:
        """Perfect-information observation, ego-centric (viewer is player 0).
        With the Trust, occupant index n is the Trust."""
        n, r = self.n, self.rules
        m = n + (1 if r.trust else 0)

        def one_hot(i, size):
            v = [0.0] * size
            if 0 <= i < size:
                v[i] = 1.0
            return v

        def rel(q):
            return q if q >= n else (q - viewer) % n

        obs: Dict[str, List[float]] = {}
        obs["phase"] = one_hot(int(self.phase), NUM_RESTING_PHASES)
        obs["step"] = one_hot(self.step - 1, 3) + [
            1.0 if self.step3_pending else 0.0, 1.0 if self.step3_next_round else 0.0]
        obs["round"] = [self.round / r.max_rounds]
        obs["fuel_market"] = ([self.fuel_market[f] / len(FUEL_PRICES[Fuel(f)])
                               for f in range(NUM_FUELS)]
                              + [1.0 if self.uranium_stopped else 0.0])
        buyable = len(self.purchasable())
        mk: List[float] = []
        for k in range(8):
            if k < len(self.market):
                pl = self._plant(self.market[k])
                mk += ([pl.number / 50.0] + one_hot(KIND_IDX[pl.kind], 6)
                       + [pl.need / 3.0, pl.power / 7.0, 1.0 if k < buyable else 0.0,
                          1.0 if pl.number == self.discount else 0.0])
            else:
                mk += [0.0] * 11
        obs["market"] = mk
        obs["stack"] = [1.0 if self.top_pending else 0.0,
                        (self.real_plug + self.real_socket) / len(r.plants)]
        cities: List[float] = []
        for occ in self.occupants:
            for s in range(3):
                cities += one_hot(rel(occ[s]), m) if s < len(occ) else [0.0] * m
        obs["cities"] = cities
        regions = self.regions or ()
        obs["in_play"] = [1.0 if r.map.city_region[c] in regions else 0.0
                          for c in range(r.map.num_cities)]
        pv: List[float] = []
        for i in range(n):
            p = (viewer + i) % n
            slots = [0.0] * (r.plant_limit + 1)
            for j, num in enumerate(self.plants[p]):
                slots[j] = num / 50.0
            pv += ([self.money[p] / 300.0, len(self.cities[p]) / r.end_cities,
                    self.order.index(p) / n] + [s / 24.0 for s in self.stored[p]] + slots)
        obs["players"] = pv
        if r.trust:
            slots = [0.0] * 3
            for j, num in enumerate(sorted(self.trust_plants)):
                slots[j] = num / 50.0
            obs["trust"] = slots + [self.trust_houses / TRUST_HOUSES]
        au = self.auction
        if au is None:
            obs["auction"] = [0.0, 0.0] + [0.0] * (n + 1) + [0.0]
        else:
            high = n if au["high"] is None else (au["high"] - viewer) % n
            obs["auction"] = ([au["plant"] / 50.0, au["bid"] / 300.0] + one_hot(high, n + 1)
                              + [1.0 if viewer in au["ring"] else 0.0])
        return obs

    def action_to_string(self, a: int) -> str:
        """Like Codec.describe, but resolves plant slots and fuel choices against this state."""
        c = self.codec
        if self.phase == Phase.CHANCE:
            return f"CHANCE {a}"
        p = self.current_player()
        if self.phase == Phase.TRUST_SETUP:
            return f"PLACE TRUST HOUSE in {self.rules.map.city_names[a - c.BUILD0]}"
        if c.SELECT0 <= a < c.BID0:
            num = self.purchasable()[a - c.SELECT0]
            tag = " (discounted: minimum bid 1)" if num == self.discount else ""
            return f"SELECT plant #{num}{tag}"
        if c.DISCARD0 <= a < c.RUN0:
            return f"DISCARD plant #{self.plants[p][a - c.DISCARD0]}"
        if self.phase == Phase.FUEL_DISCARD and c.BUY0 <= a < c.BUILD0:
            return f"RETURN {Fuel(a - c.BUY0).name}"
        if c.BUILD0 <= a < c.DISCARD0:
            city = a - c.BUILD0
            return f"BUILD {self.rules.map.city_names[city]} for {self._build_cost(p, city)}"
        if a >= c.RUN0:
            j, f = divmod(a - c.RUN0, NUM_FUELS + 1)
            pl = self._plant(self.plants[p][j])
            if pl.kind == "eco":
                how = "no fuel"
            elif pl.kind == "hybrid":
                how = f"{f} coal + {pl.need - f} oil"
            else:
                how = f"{pl.need} {Fuel(f).name.lower()}"
            return f"RUN plant #{pl.number} with {how}"
        return c.describe(a)

    def describe(self, viewer: Optional[int] = None) -> str:
        """Plain-text view of the game (also suitable as an LLM observation)."""
        lines = [f"Round {self.round}, step {self.step}, phase {self.phase.name}, "
                 f"to act: {self.current_player()}"]
        m = self.rules.map
        if self.regions is not None:
            lines.append("Regions in play: " + ", ".join(m.region_names[r] for r in self.regions))
        lines.append("Turn order: " + " ".join(f"P{p}" for p in self.order))
        mk = []
        for k, num in enumerate(self.market):
            pl = self._plant(num)
            tag = "" if k < len(self.purchasable()) else " (future)"
            if num == self.discount:
                tag += " (discount: minimum bid 1)"
            mk.append(f"#{num} {pl.kind} burns {pl.need} -> {pl.power} cities{tag}")
        if self.step3_pending:
            mk.append("STEP 3 card (future)")
        lines.append("Market: " + "; ".join(mk))
        top = " (top card is a plug plant, 03-15)" if self.top_pending else ""
        lines.append(f"Plant stack: {self.real_plug + self.real_socket + self.top_pending} cards{top}")
        lines.append("Fuel market (coal/oil/garbage/uranium): " + "/".join(map(str, self.fuel_market))
                     + "  next prices: " + "/".join(str(self._price(f)) for f in range(NUM_FUELS))
                     + ("  (no more uranium resupply)" if self.uranium_stopped else ""))
        for p in range(self.n):
            names = ", ".join(m.city_names[c] for c in sorted(self.cities[p]))
            lines.append(f"P{p}: ${self.money[p]}, {len(self.cities[p])} cities [{names}], "
                         f"plants {self.plants[p]}, fuel {self.stored[p]}")
        if self.rules.trust:
            blocked = [m.city_names[c] for c in range(m.num_cities)
                       if self.trust_id in self.occupants[c]]
            lines.append(f"Trust: plants {sorted(self.trust_plants)}, houses left "
                         f"{self.trust_houses}, in [{', '.join(blocked)}]")
        if self.auction:
            au = self.auction
            lines.append(f"Auction: plant #{au['plant']} bid {au['bid']} high {au['high']} ring {au['ring']}")
        return "\n".join(lines)

    # ---- testing -------------------------------------------------------
    def check_invariants(self):
        r = self.rules
        for p in range(self.n):
            assert self.money[p] >= 0, ("negative money", p, self.money[p])
            if not (self.phase == Phase.AUCTION_DISCARD and p == self.discarder):
                assert len(self.plants[p]) <= r.plant_limit, ("plant limit", p)
            if not (self.phase == Phase.FUEL_DISCARD and p == self.discarder):
                assert self._fits(p, self.stored[p]), ("storage overflow", p, self.stored[p])
            assert len(self.cities[p]) <= r.houses, ("out of houses", p)
            for c in self.cities[p]:
                assert r.map.city_region[c] in self.regions, ("city not in play", p, c)
        for f in range(NUM_FUELS):
            total = self.fuel_market[f] + sum(s[f] for s in self.stored) + self.trust_stored[f]
            assert total <= r.fuel_totals[f], ("fuel created from nothing", f, total)
        for c, occ in enumerate(self.occupants):
            assert len(occ) <= 3 and len(set(occ)) == len(occ), ("bad occupancy", c, occ)
            assert len([q for q in occ if q < self.n]) <= self.step, ("too many houses", c, occ)
        seen = (list(self.market) + self.deck_plug + self.deck_socket + list(self.bottom)
                + list(self.removed) + [x for pl in self.plants for x in pl] + self.trust_plants)
        assert sorted(seen) == sorted(pl.number for pl in r.plants), "plant conservation broken"
        limit = MARKET_SIZE[self.step] - (1 if self.step3_pending else 0)
        assert len(self.market) <= limit, ("market too large", self.market)
        assert self.market == sorted(self.market), "market not sorted"
        assert 0 <= self.real_plug <= len(self.deck_plug) - (1 if self.top_pending else 0)
        assert 0 <= self.real_socket <= len(self.deck_socket)
        assert self.discount is None or self.discount in self.market, "discount on a missing plant"
        assert len(self.trust_plants) <= 3


# --------------------------------------------------------------------------
# Random-play fuzzing
# --------------------------------------------------------------------------
def play_random(rules: Ruleset, rng: random.Random, check: bool = True) -> PowerGridState:
    s = PowerGridState(rules)
    steps = 0
    while not s.is_terminal():
        if s.current_player() == CHANCE:
            outs = s.chance_outcomes()
            a = rng.choices([o for o, _ in outs], [pr for _, pr in outs])[0]
        else:
            a = rng.choice(s.legal_actions())
        s.apply_action(a)
        steps += 1
        if check:
            s.check_invariants()
        if steps > 200000:
            raise RuntimeError("game did not terminate")
    return s


if __name__ == "__main__":
    import collections
    import sys

    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rng = random.Random(0)
    rounds, sales, prices, decisions = [], 0, [], 0
    kinds = collections.Counter()
    for g in range(n_games):
        rules = tiny_ruleset(num_players=2 + g % 5)
        s = play_random(rules, rng)
        rounds.append(s.round)
        for e in s.log:
            kinds[e["type"]] += 1
            if e["type"] == "sale":
                sales += 1
                prices.append(e["price"])
    print(f"{n_games} random games OK (invariants checked every move)")
    print(f"avg rounds {sum(rounds) / len(rounds):.1f}, sales {sales}, "
          f"avg sale price {sum(prices) / max(1, len(prices)):.1f}")
    print("event counts:", dict(kinds))
    demo = PowerGridState(Ruleset(num_players=4))
    while demo.phase == Phase.CHANCE:
        demo.apply_action(demo.chance_outcomes()[0][0])
    print("\n" + demo.describe())
    print("legal actions:", [demo.action_to_string(a) for a in demo.legal_actions()])
