"""Rule-level tests for powergrid_core. Run: python3 -m unittest -v

[R p.N] cites the page of the Recharged rulebook (Power-Grid-Recharged-Rules.pdf)
a test checks.
"""
import random
import unittest

import powergrid_core as pg
from powergrid_core import CHANCE, Fuel, Phase, PowerGridState, tiny_ruleset


def resolve_chance(s):
    """Resolve chance nodes with the first (lowest) outcome."""
    while s.phase == Phase.CHANCE:
        s.apply_action(s.chance_outcomes()[0][0])


def start(n=4, order_perm=0, **kw):
    """Tiny-map game after setup: market 03-10 (lowest plugs dealt), P0 first."""
    s = PowerGridState(tiny_ruleset(n, **kw))
    s.apply_action(order_perm)          # chance: initial turn order
    resolve_chance(s)                   # chance: the 8 market plants
    return s


def buy_uncontested(s):
    """Play out the current auction: the opener bids the minimum, everyone else drops."""
    c = s.codec
    while s.phase == Phase.AUCTION_BID:
        s.apply_action(c.PASS if s.auction["high"] is not None
                       else c.BID0 + s.min_bid(s.auction["plant"]))


def give(s, p, nums):
    """Hand plants to player p straight from the stack (as real cards)."""
    for n in nums:
        if n <= pg.MAX_PLUG:
            s.deck_plug.remove(n)
            s.real_plug -= 1
        else:
            s.deck_socket.remove(n)
            s.real_socket -= 1
        s.plants[p].append(n)


def random_games(n_games, seed=0, make=lambda n: tiny_ruleset(n)):
    rng = random.Random(seed)
    for g in range(n_games):
        s = PowerGridState(make(2 + g % 5))
        yield s
        while not s.is_terminal():
            if s.current_player() == CHANCE:
                outs = s.chance_outcomes()
                a = rng.choices([o for o, _ in outs], [p for _, p in outs])[0]
            else:
                a = rng.choice(s.legal_actions())
            s.apply_action(a)
            s.check_invariants()
            yield s


def city(name):
    return pg.germany_map().city_names.index(name)


class SetupTests(unittest.TestCase):
    def test_player_count_tables(self):
        # areas [R p.2], removed plug/socket plants [R p.3], 3 plants each [R p.5],
        # step 2 [R p.7], game end [R p.8]
        want = {2: (3, 1, 5, 3, 7, 18), 3: (3, 2, 6, 3, 7, 17), 4: (4, 1, 3, 3, 7, 17),
                5: (5, 0, 0, 3, 7, 15), 6: (5, 0, 0, 3, 6, 14)}
        for n, row in want.items():
            r = pg.Ruleset(num_players=n)
            self.assertEqual((r.play_regions, r.plug_removed, r.socket_removed, r.plant_limit,
                              r.step2_cities, r.end_cities), row)
            self.assertEqual(r.trust, n == 2)

    def test_resource_market_and_tables(self):
        # [R p.2] coal 1-8, oil 3-8, garbage 6-8, uranium 14-16
        s = PowerGridState(pg.Ruleset())
        self.assertEqual(s.fuel_market, [24, 18, 9, 2])
        self.assertEqual([s._price(f) for f in range(4)], [1, 3, 6, 14])
        # [R p.3] the 5-player refill card; [R p.6] payment table
        self.assertEqual(pg.RESUPPLY[5], ((5, 4, 3, 2), (7, 5, 3, 3), (5, 6, 5, 2)))
        self.assertEqual(pg.INCOME[6], 73)
        self.assertEqual(pg.INCOME[20], 150)

    def test_market_is_dealt_from_plug_plants(self):
        # [R p.3] 8 plug plants (03-15) face up, then one more set aside for the top
        s = PowerGridState(pg.Ruleset(num_players=4))
        s.apply_action(0)            # regions
        s.apply_action(0)            # turn order
        self.assertEqual(s.chance_kind, "setup")
        self.assertEqual([a for a, _ in s.chance_outcomes()], list(range(3, 16)))
        resolve_chance(s)
        self.assertEqual(s.market, list(range(3, 11)))
        self.assertEqual(s.phase, Phase.AUCTION_SELECT)
        self.assertTrue(s.top_pending)
        # 4 players remove 1 plug and 3 sockets face down: 13 - 8 - 1 - 1 plugs, 29 - 3 sockets
        self.assertEqual((s.real_plug, s.real_socket), (3, 26))

    def test_stack_draw_probabilities(self):
        s = start(4)
        s.apply_action(s.codec.SELECT0)                 # P0 buys #3 unopposed...
        buy_uncontested(s)
        outs = dict(s.chance_outcomes())                # the top card: one of 5 plugs
        self.assertEqual(sorted(outs), [11, 12, 13, 14, 15])
        self.assertAlmostEqual(outs[11], 0.2)
        s.apply_action(11)
        s.apply_action(s.codec.SELECT0)
        buy_uncontested(s)
        outs = dict(s.chance_outcomes())                # 3 real plugs, 26 real sockets
        self.assertAlmostEqual(sum(outs.values()), 1.0)
        self.assertAlmostEqual(outs[12], 3 / 29 / 4)     # 4 plug candidates, 3 real
        self.assertAlmostEqual(outs[20], 26 / 29 / 29)   # 29 socket candidates, 26 real


class GermanyMapTests(unittest.TestCase):
    def test_shape(self):
        # [R p.2] 6 areas with 7 cities each
        m = pg.germany_map()
        self.assertEqual(m.num_cities, 42)
        for r in range(6):
            self.assertEqual(sum(1 for c in range(42) if m.city_region[c] == r), 7)
        self.assertEqual(len(m.edges), 83)
        self.assertIn("Stralsund", m.city_names)
        self.assertIn("Mainz", m.city_names)

    def test_rulebook_map_excerpt(self):
        # [R p.6] every connection shown in the building example
        m = pg.germany_map()
        cost = {frozenset((m.city_names[a], m.city_names[b])): w for a, b, w in m.edges}
        shown = {("Duisburg", "Essen"): 0, ("Essen", "Münster"): 6, ("Münster", "Dortmund"): 2,
                 ("Essen", "Dortmund"): 5, ("Essen", "Düsseldorf"): 2, ("Düsseldorf", "Köln"): 4,
                 ("Düsseldorf", "Aachen"): 9, ("Aachen", "Köln"): 7, ("Köln", "Dortmund"): 10}
        for (a, b), w in shown.items():
            self.assertEqual(cost[frozenset((a, b))], w, (a, b))

    def test_rulebook_building_example(self):
        # [R p.6] Anna has Essen and Münster... she may add Duisburg for 10; Bob has
        # Düsseldorf and Köln.
        s = PowerGridState(pg.Ruleset(num_players=3, regions=(0, 1, 2)))
        s.regions = tuple(range(6))
        s.cities[0] = [city("Essen"), city("Münster")]
        s.cities[1] = [city("Düsseldorf"), city("Köln")]
        for p, cs in enumerate(s.cities):
            for c in cs:
                s.occupants[c].append(p)
        self.assertEqual(s._build_cost(0, city("Duisburg")), 10)
        self.assertEqual(s._build_cost(0, city("Dortmund")), 12)
        self.assertEqual(s._build_cost(0, city("Aachen")), 21)
        s.step = 2
        self.assertEqual(s._build_cost(0, city("Düsseldorf")), 17)
        self.assertEqual(s._build_cost(0, city("Köln")), 21)
        s.cities[0].append(city("Düsseldorf"))
        s.occupants[city("Düsseldorf")].append(0)
        self.assertEqual(s._build_cost(0, city("Köln")), 19)   # 17 + 19 = 36 for both

    def test_regions_are_a_chance_node_over_adjacent_sets(self):
        r = pg.Ruleset(num_players=4)
        s = PowerGridState(r)
        self.assertEqual(s.chance_kind, "regions")
        self.assertEqual(len(s.chance_outcomes()), len(r.region_sets))
        for regions in r.region_sets:
            self.assertEqual(len(regions), 4)
        with self.assertRaises(ValueError):
            pg.Ruleset(num_players=3, regions=(1, 4, 5))  # north-east is not next to the south

    def test_no_paths_through_regions_out_of_play(self):
        # [R p.6] never use cities or connections outside the playing zone
        m = pg.tiny_map()
        full, ab = m.dist((0, 1, 2)), m.dist((0, 1))
        self.assertEqual(ab[0][8], 10 ** 9)
        self.assertEqual(ab[3][4], full[3][4])


class AuctionTests(unittest.TestCase):
    def test_bidding_goes_clockwise_by_seat(self):
        # [R p.4] "Continuing in clockwise order"
        s = start()
        s.order = [2, 0, 3, 1]
        s.apply_action(s.codec.SELECT0 + 1)        # P2 selects
        self.assertEqual(s.auction["ring"], [2, 3, 0, 1])

    def test_round_one_order_reset_after_auction(self):
        # [R p.5] player order is determined again once, after the first auction
        s = start()
        c = s.codec
        while s.phase in (Phase.AUCTION_SELECT, Phase.AUCTION_BID, Phase.CHANCE):
            if s.phase == Phase.AUCTION_SELECT:
                s.apply_action(c.SELECT0)
            buy_uncontested(s)
            resolve_chance(s)
        self.assertEqual(s.phase, Phase.BUY_FUEL)
        self.assertEqual(s.order, [3, 2, 1, 0])    # biggest plant first
        self.assertEqual(s.current_player(), 0)    # fuel is bought in reverse order

    def test_discount_token(self):
        # [R p.4] the smallest current plant costs at least 1 instead of its number
        s = start()
        self.assertEqual(s.discount, 3)
        s.money[0] = 2
        self.assertEqual(s.legal_actions(), [s.codec.SELECT0])   # only the discounted #3
        s.money[0] = 50
        s.apply_action(s.codec.SELECT0)
        self.assertEqual(s.legal_actions()[0], s.codec.BID0 + 1)
        buy_uncontested(s)
        self.assertEqual(s.money[0], 49)
        self.assertIsNone(s.discount)              # bought: the token is set aside

    def test_smaller_replacement_is_removed_with_the_token(self):
        # [R p.4] the first replacement smaller than the discounted plant leaves the
        # game together with the token, and another plant is drawn
        s = start()
        s.round = 2
        s.market.remove(3)                          # market 04-10, stack holds #3 and #11
        s.deck_plug = [3, 11]
        s.real_plug, s.top_pending = 2, False
        s.discount = 4
        s.apply_action(s.codec.SELECT0 + 2)        # P0 buys #6: a replacement is drawn
        buy_uncontested(s)
        self.assertEqual(s.phase, Phase.CHANCE)
        s.apply_action(3)                          # smaller than the discounted #4
        self.assertIsNone(s.discount)
        self.assertIn(3, s.removed)
        self.assertNotIn(3, s.market)
        self.assertEqual(s.phase, Phase.CHANCE)    # ... and another plant is drawn
        s.apply_action(11)
        self.assertEqual(s.market, [4, 5, 7, 8, 9, 10, 11])

    def test_unsold_discounted_plant_leaves_at_end_of_phase_2(self):
        # [R p.4] "If nobody buys the discounted power plant, remove it from the game
        # at the end of Phase 2. Replace it"
        s = start()
        s.round = 2
        for _ in range(4):
            s.apply_action(s.codec.PASS)
        self.assertIn(3, s.removed)
        resolve_chance(s)
        self.assertEqual(len(s.market), 8)
        self.assertNotIn(3, s.market)
        self.assertIsNone(s.discount)

    def test_new_plant_cannot_be_scrapped(self):
        # [R p.5] "They may not choose to scrap the just bought power plant"
        s = start()
        give(s, 0, [20, 21, 25])
        s.round = 2
        s.apply_action(s.codec.SELECT0 + 3)        # P0 buys #6
        buy_uncontested(s)
        self.assertEqual(s.phase, Phase.AUCTION_DISCARD)
        self.assertEqual(s.plants[0], [20, 21, 25, 6])
        self.assertEqual(s.legal_actions(), [s.codec.DISCARD0 + j for j in range(3)])

    def test_unstorable_fuel_is_returned_automatically(self):
        # [R p.5] fuel no retained plant can store goes back to the supply
        s = start()
        give(s, 0, [21, 25, 24])                   # hybrid (holds 4), coal (4), garbage (4)
        s.stored[0] = [6, 2, 2, 0]
        s.round = 2
        s.apply_action(s.codec.SELECT0 + 1)        # buy #4 (coal, holds 4)
        buy_uncontested(s)
        s.apply_action(s.codec.DISCARD0 + 0)       # scrap the hybrid: oil has nowhere to go
        self.assertEqual(s.stored[0], [6, 0, 2, 0])

    def test_player_chooses_coal_or_oil_to_return(self):
        s = start()
        give(s, 0, [21, 29, 24])                   # hybrid space 4 + 2
        s.stored[0] = [3, 3, 0, 0]
        s.round = 2
        s.apply_action(s.codec.SELECT0 + 3)        # buy #6 (garbage)
        buy_uncontested(s)
        s.apply_action(s.codec.DISCARD0 + 0)       # scrap #21: 6 cubes, hybrid space 2
        self.assertEqual(s.phase, Phase.FUEL_DISCARD)
        coal, oil = s.codec.BUY0 + Fuel.COAL, s.codec.BUY0 + Fuel.OIL
        self.assertEqual(s.legal_actions(), [coal, oil])
        self.assertEqual(s.action_to_string(coal), "RETURN COAL")
        for _ in range(3):
            s.apply_action(coal)                   # no coal left: the last oil goes automatically
        self.assertEqual(s.stored[0], [0, 2, 0, 0])
        self.assertEqual(s.phase, Phase.CHANCE)    # next: the replacement for the sold plant
        s.check_invariants()

    def test_last_player_pays_the_minimum(self):
        # [R p.5]
        s = start()
        s.round = 2
        for _ in range(3):
            s.apply_action(s.codec.PASS)
        s.apply_action(s.codec.SELECT0)            # P3 alone: discounted #3 for 1
        self.assertEqual(s.plants[3], [3])
        self.assertEqual(s.money[3], 49)

    def test_uranium_phase_out(self):
        # [R p.7] Germany: after a player buys #39, uranium is no longer resupplied
        s = PowerGridState(pg.Ruleset(num_players=3, regions=(0, 1, 2)))
        resolve_chance(s)
        s.deck_socket.remove(39)
        s.real_socket -= 1
        s.removed.append(6)
        s.market = [3, 4, 5, 39, 7, 8, 9, 10]      # #39 in the current market
        s.money = [100, 100, 100]
        s.apply_action(s.codec.SELECT0 + 3)
        buy_uncontested(s)
        self.assertTrue(s.uranium_stopped)
        s.fuel_market[Fuel.URANIUM] = 0
        s.round = 5
        s._finish_round()
        self.assertEqual(s.fuel_market[Fuel.URANIUM], 0)


class StepTests(unittest.TestCase):
    def test_step2_at_start_of_phase_5(self):
        # [R p.7] step 2 starts at the beginning of phase 5
        s = start()
        s.phase, s.queue, s.qpos = Phase.BUILD, [0, 1, 2, 3], 0
        s.money = [500] * 4
        for city_ in range(s.rules.step2_cities):
            s.apply_action(s.codec.BUILD0 + city_)
        self.assertEqual(s.step, 1)
        for _ in range(4):
            s.apply_action(s.codec.DONE)
        resolve_chance(s)
        self.assertEqual(s.step, 2)
        kinds = [e["type"] for e in s.log]
        self.assertLess(max(i for i, k in enumerate(kinds) if k == "build"), kinds.index("step2"))
        self.assertLess(kinds.index("step2"), kinds.index("income"))

    def _empty_stack(self, s, bottom):
        s.removed += s.deck_plug + s.deck_socket
        s.deck_plug, s.deck_socket = [], []
        s.real_plug = s.real_socket = 0
        s.top_pending = False
        s.bottom = list(bottom)

    def test_step3_card_in_phase2(self):
        # [R p.8] case 1: shuffle now, keep drawing replacements; after phase 2
        # remove the card and the lowest plant, no replacements; step 3 in phase 3
        s = start()
        s.step = 2
        self._empty_stack(s, [40, 42, 44])
        c = s.codec
        s.apply_action(c.SELECT0)
        buy_uncontested(s)
        self.assertTrue(s.step3_pending)
        self.assertEqual(len(s.market), 7)
        self.assertEqual(sorted(s.deck_socket), [40, 42, 44])
        draws = 0
        while s.phase in (Phase.AUCTION_SELECT, Phase.AUCTION_BID, Phase.CHANCE):
            if s.phase == Phase.CHANCE:
                resolve_chance(s)
                draws += 1
            elif s.phase == Phase.AUCTION_SELECT:
                s.apply_action(c.SELECT0)
            else:
                buy_uncontested(s)
        self.assertEqual(draws, 3)
        self.assertEqual(s.step, 3)
        self.assertEqual(len(s.market), 6)
        self.assertEqual(s.phase, Phase.BUY_FUEL)

    def test_step3_card_in_phase5(self):
        # [R p.8] case 2: card and lowest plant leave, no replacements; resupply for
        # step 2 a final time; step 3 starts with the next round
        s = start()
        s.step = 2
        self._empty_stack(s, [44, 46])
        s.round = 3
        s.phase, s.queue, s.qpos = Phase.BUREAUCRACY, [0, 1, 2, 3], 0
        s.fuel_market = [0, 0, 0, 0]
        s._finish_round()
        self.assertEqual(s.step, 3)                # the next round has started
        self.assertEqual(s.fuel_market, list(pg.RESUPPLY[4][1]))   # step 2 values
        self.assertEqual(len(s.market), 6)
        self.assertEqual(sorted(s.deck_socket), [44, 46])
        self.assertEqual(s.purchasable(), s.market)

    def test_step3_before_step2(self):
        # [R p.8] "first perform all changes for Step 2"
        s = start()
        self._empty_stack(s, [44, 46])
        s.round = 3
        s._finish_round()
        resolve_chance(s)
        kinds = [e["type"] for e in s.log]
        self.assertLess(kinds.index("step2"), kinds.index("step3"))
        self.assertEqual(s.step, 3)
        self.assertEqual(len(s.market), 6)

    def test_step3_market_stays_full(self):
        seen = 0
        for s in random_games(150, seed=3):
            if s.step == 3 and s.phase == Phase.AUCTION_SELECT and s._cards_left() \
                    and not s.bought_any and not s.trust_took:
                self.assertEqual(len(s.market), 6)
                self.assertEqual(s.purchasable(), s.market)
                seen += 1
        self.assertGreater(seen, 0)


class PhaseTests(unittest.TestCase):
    def test_hybrid_can_mix_coal_and_oil(self):
        # [R p.3] "2 coal tokens, 2 oil tokens, or 1 coal and 1 oil token"
        s = start()
        s.plants[0] = [12]
        s.stored[0] = [1, 1, 0, 0]
        s.cities[0] = [0, 1]
        s.phase, s.queue, s.qpos = Phase.BUREAUCRACY, [0, 1, 2, 3], 0
        s.ran = [[] for _ in range(4)]
        run = s.codec.RUN0 + 1
        self.assertEqual(s.legal_actions(), [s.codec.DONE, run])
        self.assertEqual(s.action_to_string(run), "RUN plant #12 with 1 coal + 1 oil")
        s.apply_action(run)
        self.assertEqual(s.stored[0], [0, 0, 0, 0])

    def test_house_limit(self):
        # [R p.2] 22 houses per player
        s = start(houses=2)
        s.phase, s.queue, s.qpos = Phase.BUILD, [0, 1, 2, 3], 0
        s.money = [500] * 4
        s.apply_action(s.codec.BUILD0 + 0)
        s.apply_action(s.codec.BUILD0 + 1)
        self.assertEqual(s.current_player(), 1)


class GameEndTests(unittest.TestCase):
    def setUp(self):
        self.s = start(3)
        s = self.s
        s.phase, s.queue, s.qpos = Phase.BUILD, [0, 1, 2], 0
        s.money = [200, 200, 200]

    def _end(self):
        s = self.s
        for c in range(6):
            s.apply_action(s.codec.BUILD0 + c)
        self.money_after_building = list(s.money)
        for _ in range(3):
            s.apply_action(s.codec.DONE)

    def test_ends_after_phase_4_without_income(self):
        # [R p.8] ends after phase 4; no money for powering cities
        s = self.s
        s.plants[0] = [13]
        self._end()
        self.assertTrue(s.is_terminal())
        self.assertEqual(s.money, self.money_after_building)
        self.assertNotIn("income", [e["type"] for e in s.log])

    def test_winner_is_most_cities_suppliable_then_money_only(self):
        # [R p.8] most cities suppliable; tie: most money (no further tie-break)
        s = self.s
        s.plants[0] = [13]                         # powers 1
        s.plants[1], s.stored[1] = [15], [2, 0, 0, 0]
        s.cities[1] = [8, 9, 10]
        for c in (8, 9, 10):
            s.occupants[c].append(1)
        self._end()
        self.assertEqual(s.returns(), [0.0, 1.0, 0.0])

    def test_tie_on_money_is_shared(self):
        # equal supply and money: more cities does not break the tie [R p.8]
        s = start(3)
        s.plants[0], s.plants[1] = [13], [13]      # each powers 1
        s.cities[0], s.cities[1] = [0, 1], [5]
        s.money = [30, 30, 30]
        s._end_game()
        self.assertEqual(s.returns(), [0.5, 0.5, 0.0])

    def test_max_supply_uses_hybrid_flexibly(self):
        s = self.s
        s.cities[0] = list(range(10))
        s.plants[0] = [5, 12, 16]
        s.stored[0] = [3, 3, 0, 0]
        self.assertEqual(s._max_supply(0), 6)
        s.stored[0] = [3, 2, 0, 0]
        self.assertEqual(s._max_supply(0), 5)


class TrustTests(unittest.TestCase):
    """2 players: "Against the Trust" [R p.9-10]."""

    def setup_game(self):
        s = PowerGridState(tiny_ruleset(2))
        s.apply_action(0)                          # P0 starts
        resolve_chance(s)
        return s

    def place_all(self, s):
        while s.phase == Phase.TRUST_SETUP:
            s.apply_action(s.legal_actions()[0])

    def test_setup_placement_order_and_adjacency(self):
        # [R p.9] start player 1 house, other 2, start 2, other 1; adjacent cities
        s = self.setup_game()
        self.assertEqual(s.phase, Phase.TRUST_SETUP)
        self.assertEqual(s.trust_setup, [0, 1, 1, 0, 0, 1])
        self.assertEqual(len(s.legal_actions()), 12)
        s.apply_action(s.codec.BUILD0 + 0)
        self.assertEqual(s.current_player(), 1)
        neighbours = {s.rules.map.neighbors[0][i] for i in range(len(s.rules.map.neighbors[0]))}
        self.assertEqual({a - s.codec.BUILD0 for a in s.legal_actions()}, neighbours)
        self.place_all(s)
        self.assertEqual(sum(1 for occ in s.occupants if s.trust_id in occ), 6)
        self.assertEqual(s.trust_houses, 10)
        self.assertEqual(s.phase, Phase.AUCTION_SELECT)

    def test_trust_cities_blocked_in_step1_open_in_step2(self):
        s = self.setup_game()
        self.place_all(s)
        trust_cities = [c for c, occ in enumerate(s.occupants) if s.trust_id in occ]
        s.phase, s.queue, s.qpos = Phase.BUILD, [0, 1], 0
        builds = {a - s.codec.BUILD0 for a in s.legal_actions() if a >= s.codec.BUILD0}
        self.assertFalse(builds & set(trust_cities))
        s.step = 2
        self.assertEqual(s._build_cost(0, trust_cities[0]), 15)

    def test_trust_blocks_the_15_space_of_new_cities(self):
        # [R p.10]
        s = self.setup_game()
        self.place_all(s)
        free = next(c for c, occ in enumerate(s.occupants) if not occ)
        s.phase, s.queue, s.qpos = Phase.BUILD, [0, 1], 0
        s.apply_action(s.codec.BUILD0 + free)
        self.assertEqual(s.occupants[free], [0, s.trust_id])
        self.assertEqual(s.trust_houses, 9)
        s.step = 2
        self.assertIsNone(s._build_cost(1, free))  # full until step 3
        s.step = 3
        self.assertEqual(s._build_cost(1, free) - 20 >= 0, True)

    def test_trust_takes_biggest_current_plant_after_first_purchase(self):
        # [R p.9]
        s = self.setup_game()
        self.place_all(s)
        s.apply_action(s.codec.SELECT0)            # P0 buys #3
        buy_uncontested(s)
        resolve_chance(s)
        self.assertEqual(len(s.trust_plants), 1)
        self.assertTrue(s.trust_took)
        take = next(e for e in s.log if e["type"] == "trust_take")
        self.assertEqual(take["plant"], 7)         # current market after the replacement: 04-07

    def test_trust_takes_after_first_player_opts_out(self):
        s = self.setup_game()
        self.place_all(s)
        s.round = 2
        s.apply_action(s.codec.PASS)
        self.assertTrue(s.trust_took)
        self.assertEqual(s.trust_plants, [6])

    def test_trust_with_three_plants_only_upgrades(self):
        s = self.setup_game()
        self.place_all(s)
        s.trust_plants = [20, 21, 25]
        for n in (20, 21, 25):
            s.deck_socket.remove(n)
            s.real_socket -= 1
        s.round = 2
        s.apply_action(s.codec.PASS)
        self.assertEqual(s.trust_plants, [20, 21, 25])   # #6 is smaller than 20
        self.assertNotIn("trust_take", [e["type"] for e in s.log])

    def test_trust_buys_fuel_second_and_alternates_hybrid(self):
        # [R p.9] Trust is second in order; hybrids alternate coal and oil, coal first
        s = self.setup_game()
        self.place_all(s)
        s.trust_plants = [5, 8]                    # hybrid 2, coal 3
        s.plants[0] = [10]                         # so P0 has fuel to buy afterwards
        s.phase, s.queue, s.qpos = Phase.BUY_FUEL, [1, 0], 0
        market = list(s.fuel_market)
        s.apply_action(s.codec.DONE)               # P1 (last in order) done
        self.assertEqual(s.trust_stored, [4, 1, 0, 0])
        self.assertEqual(s.fuel_market, [market[0] - 4, market[1] - 1] + market[2:])
        self.assertEqual(s.current_player(), 0)

    def test_two_player_game_ends_at_18(self):
        self.assertEqual(pg.Ruleset(num_players=2).end_cities, 18)


class GermanyGameTests(unittest.TestCase):
    def test_random_games_on_germany(self):
        n = 0
        for s in random_games(15, seed=5, make=lambda k: pg.Ruleset(num_players=k)):
            if s.is_terminal():
                n += 1
        self.assertEqual(n, 15)


if __name__ == "__main__":
    unittest.main()
