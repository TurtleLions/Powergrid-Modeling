"""Tests for the bot layer. Run: python3 -m unittest bots.test_bots"""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cpp", "build"))
import pgcore  # noqa: E402

from bots.arena import play_game  # noqa: E402
from bots.heuristic import STYLES, make  # noqa: E402
from bots.rl.features import BID_RAISES, AbstractActions, Encoder  # noqa: E402


def random_states(players, games, seed=0):
    rng = random.Random(seed)
    rules = pgcore.Rules(players=players)
    for _ in range(games):
        s = pgcore.State(rules)
        while not s.is_terminal():
            if s.is_chance_node():
                o = s.chance_outcomes()
                s.apply_action(rng.choices([a for a, _ in o], [p for _, p in o])[0])
                continue
            yield rules, s
            s.apply_action(rng.choice(s.legal_actions()))


class AbstractActionTests(unittest.TestCase):
    def test_every_abstract_action_maps_to_a_legal_engine_action(self):
        for players in (3, 4, 5):
            acts = None
            for rules, s in random_states(players, 3, seed=players):
                acts = acts or AbstractActions(rules)
                legal = set(s.legal_actions())
                mask, amap = acts.mask_and_map(sorted(legal))
                self.assertTrue(mask.any())
                self.assertEqual(set(amap), set(mask.nonzero()[0]))
                for x, a in amap.items():
                    self.assertIn(a, legal)
                # every non-bid engine action is reachable
                c = rules.codec
                non_bid = {a for a in legal if not (c["BID0"] <= a < c["BUY0"])}
                self.assertTrue(non_bid <= set(amap.values()))

    def test_bids_are_raises_over_the_minimum(self):
        rules = pgcore.Rules(players=4)
        acts = AbstractActions(rules)
        c = rules.codec
        legal = [c["PASS"]] + [c["BID0"] + x for x in range(21, 30)]   # bids 21..29
        mask, amap = acts.mask_and_map(legal)
        bids = sorted(amap[acts.BID0 + i] - c["BID0"] for i, r in enumerate(BID_RAISES)
                      if mask[acts.BID0 + i])
        self.assertEqual(bids, [21 + r for r in BID_RAISES if 21 + r <= 29])


class EncoderTests(unittest.TestCase):
    def test_fixed_size_for_any_player_count(self):
        sizes = set()
        for players in (2, 3, 4, 6):
            rules = pgcore.Rules(players=players)
            enc = Encoder(rules)
            for _, s in random_states(players, 1):
                self.assertEqual(enc.encode(s, s.current_player()).shape, (enc.size,))
            sizes.add(enc.size)
        self.assertEqual(len(sizes), 1)


class FeatureVersionTests(unittest.TestCase):
    def test_v2_appends_to_v1_and_widening_keeps_the_policy(self):
        import numpy as np
        import torch
        from bots.rl.model import PolicyValueNet, widen_input
        rules = pgcore.Rules(players=4)
        e1, e2 = Encoder(rules, 1), Encoder(rules, 2)
        self.assertGreater(e2.size, e1.size)
        acts = AbstractActions(rules)
        torch.manual_seed(0)
        net = PolicyValueNet(e1.size, acts.n, 64)
        ref = PolicyValueNet(e1.size, acts.n, 64)
        ref.load_state_dict(net.state_dict())
        widen_input(net, e2.size, 2)
        self.assertEqual(net.feature_version, 2)
        for _, s in random_states(4, 1, seed=3):
            p = s.current_player()
            x1, x2 = e1.encode(s, p), e2.encode(s, p)
            np.testing.assert_array_equal(x1, x2[:e1.size])
            mask = torch.ones(1, acts.n, dtype=torch.bool)
            with torch.no_grad():
                l1, v1 = ref(torch.from_numpy(x1)[None], mask)
                l2, v2 = net(torch.from_numpy(x2)[None], mask)
            torch.testing.assert_close(l1, l2, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(v1, v2, atol=1e-4, rtol=1e-4)

    def test_max_supply_matches_what_decides_the_winner(self):
        # at the end of a game, the winner has the highest (max_supply, money)
        rules = pgcore.Rules(players=4)
        rng = random.Random(9)
        s = pgcore.State(rules)
        while not s.is_terminal():
            if s.is_chance_node():
                o = s.chance_outcomes()
                s.apply_action(rng.choices([a for a, _ in o], [p for _, p in o])[0])
            else:
                s.apply_action(rng.choice(s.legal_actions()))
        key = [(s.max_supply(p), s.money(p)) for p in range(4)]
        best = max(key)
        self.assertEqual([r > 0 for r in s.returns()], [k == best for k in key])


class ArenaTests(unittest.TestCase):
    def test_scripted_bots_play_legal_games_to_the_end(self):
        for players in (3, 4, 5):
            rules = pgcore.Rules(players=players)
            for seed in range(5):
                names = [list(STYLES)[(seed + i) % len(STYLES)] for i in range(players)]
                returns, rounds = play_game([make(n) for n in names], rules, seed)
                self.assertAlmostEqual(sum(returns), 1.0)
                self.assertLess(rounds, rules.max_rounds)


if __name__ == "__main__":
    unittest.main()
