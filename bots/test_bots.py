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


class RatingTests(unittest.TestCase):
    def test_fit_recovers_known_strengths(self):
        import numpy as np
        from bots.rating import ELO, fit
        rng = random.Random(1)
        agents = [f"a{i}" for i in range(8)] + ["random"]
        true = np.array([rng.uniform(-1, 3) for _ in range(8)] + [0.0])
        results = []
        for _ in range(4000):
            names = rng.sample(agents, 4)
            p = np.exp(true[[agents.index(x) for x in names]])
            w = rng.choices(range(4), p / p.sum())[0]
            results.append((names, [1.0 if i == w else 0.0 for i in range(4)]))
        err = np.abs(fit(results, agents) - true) * ELO
        self.assertLess(err.mean(), 30)


class SearchTests(unittest.TestCase):
    def test_search_agent_plays_legal_games(self):
        import tempfile
        import torch
        from bots.rl.features import FEATURE_VERSION
        from bots.rl.model import PolicyValueNet, save
        from bots.search import SearchAgent
        rules = pgcore.Rules(players=4)
        torch.manual_seed(0)
        net = PolicyValueNet(Encoder(rules).size, AbstractActions(rules).n, 32)
        net.feature_version = FEATURE_VERSION
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            save(f.name, net)
            agents = [SearchAgent(f.name, sims=4)] + [make("balanced") for _ in range(3)]
            returns, _ = play_game(agents, rules, 3)   # play_game rejects illegal moves
            self.assertAlmostEqual(sum(returns), 1.0)

    def test_separate_value_net_scores_leaves(self):
        import tempfile
        import torch
        from bots.rl.features import FEATURE_VERSION
        from bots.rl.model import PolicyValueNet, save
        from bots.search import SearchAgent
        rules = pgcore.Rules(players=4)
        torch.manual_seed(0)
        size, n = Encoder(rules).size, AbstractActions(rules).n
        pnet, vnet = PolicyValueNet(size, n, 32), PolicyValueNet(size, n, 16)
        pnet.feature_version = vnet.feature_version = FEATURE_VERSION
        with tempfile.NamedTemporaryFile(suffix=".pt") as fp, \
                tempfile.NamedTemporaryFile(suffix=".pt") as fv:
            save(fp.name, pnet)
            save(fv.name, vnet)
            both = make(f"mcts:4:{fp.name}:{fv.name}")
            plain = SearchAgent(fp.name, sims=4)
            for a in (both, plain):
                a.reset(rules, 0, random.Random(0))
            state = pgcore.State(rules)
            while state.is_chance_node():
                state.apply_action(state.chance_outcomes()[0][0])
            p1, _, v1 = both._evaluate(state)
            p2, _, v2 = plain._evaluate(state)
            for x in p1:                                  # priors from the policy net
                self.assertAlmostEqual(p1[x], p2[x], places=5)
            self.assertFalse(all(abs(a - b) < 1e-6 for a, b in zip(v1, v2)))   # values from the value net
            norm = make(f"mcts:4+norm+c3:{fp.name}:{fv.name}")
            self.assertTrue(norm.normalize_q and not both.normalize_q)
            self.assertEqual((norm.c_puct, both.c_puct), (3.0, 1.5))
            returns, _ = play_game([both, norm] + [make("balanced") for _ in range(2)], rules, 3)
            self.assertAlmostEqual(sum(returns), 1.0)


    def test_seat_value_model_in_search(self):
        import tempfile
        import torch
        from bots.rl.features import FEATURE_VERSION
        from bots.rl.model import PolicyValueNet, save
        from bots.rl.value_models import SeatValueNet, load_value, save_value
        rules = pgcore.Rules(players=4)
        torch.manual_seed(0)
        size, n = Encoder(rules).size, AbstractActions(rules).n
        pnet = PolicyValueNet(size, n, 32)
        pnet.feature_version = FEATURE_VERSION
        for arch in ("mlp", "attn"):
            vm = SeatValueNet(size, arch=arch, hidden=16, depth=1, layers=1, loss="joint")
            with tempfile.NamedTemporaryFile(suffix=".pt") as fp, \
                    tempfile.NamedTemporaryFile(suffix=".pt") as fv:
                save(fp.name, pnet)
                save_value(fv.name, vm)
                agent = make(f"mcts:4+norm+c3:{fp.name}:{fv.name}")
                agent.reset(rules, 0, random.Random(0))
                state = pgcore.State(rules)
                while state.is_chance_node():
                    state.apply_action(state.chance_outcomes()[0][0])
                _, _, vals = agent._evaluate(state)
                self.assertEqual(len(vals), 4)
                self.assertAlmostEqual(float(sum(vals)), 1.0, places=5)
                obs = torch.stack([torch.from_numpy(agent.enc.encode(state, p)) for p in range(4)])
                want = torch.softmax(vm(obs[None])[0][0], -1).detach().numpy()
                self.assertTrue(all(abs(a - b) < 1e-5 for a, b in zip(vals, want)))
                self.assertEqual(load_value(fv.name).obs_size, size)
                returns, _ = play_game([agent] + [make("balanced") for _ in range(3)], rules, 3)
                self.assertAlmostEqual(sum(returns), 1.0)


class FastPathTests(unittest.TestCase):
    def test_cpp_features_match_python_encoder(self):
        import json
        import numpy as np
        if not hasattr(pgcore.State, "features"):
            self.skipTest("pgcore built without features(); run make -C cpp pgcore")
        for players in (2, 4, 6):
            rules = pgcore.Rules(players=players)
            enc = Encoder(rules, 2)
            self.assertEqual(rules.feature_size, enc.size)
            rng = random.Random(players)
            agents = [make(list(STYLES)[i % len(STYLES)]) for i in range(players)]
            for seat, a in enumerate(agents):
                a.reset(rules, seat, random.Random(seat))
            state = pgcore.State(rules)
            while not state.is_terminal():
                if state.is_chance_node():
                    outs = state.chance_outcomes()
                    state.apply_action(rng.choices([o for o, _ in outs], [p for _, p in outs])[0])
                    continue
                v = json.loads(state.to_json())
                allf = state.features_all()
                for seat in range(players):
                    want = enc._encode(state, v, seat)
                    self.assertTrue(np.array_equal(state.features(seat), want), v["phase"])
                    self.assertTrue(np.array_equal(allf[seat], want))
                state.apply_action(agents[state.current_player()].act(state))

    def test_batched_search_plays_legal_games(self):
        import tempfile
        import torch
        from bots.rl.features import FEATURE_VERSION
        from bots.rl.model import PolicyValueNet, save
        from bots.rl.value_models import SeatValueNet, save_value
        rules = pgcore.Rules(players=4)
        torch.manual_seed(0)
        size, n = Encoder(rules).size, AbstractActions(rules).n
        pnet = PolicyValueNet(size, n, 32)
        pnet.feature_version = FEATURE_VERSION
        with tempfile.NamedTemporaryFile(suffix=".pt") as fp, \
                tempfile.NamedTemporaryFile(suffix=".pt") as fv:
            save(fp.name, pnet)
            save_value(fv.name, SeatValueNet(size, hidden=16, depth=1))
            for name in (f"mcts:12+norm+c3+b4:{fp.name}:{fv.name}", f"mcts:12+b4:{fp.name}"):
                agent = make(name)
                self.assertEqual(agent.batch, 4)
                agent.reset(rules, 0, random.Random(0))
                state = pgcore.State(rules)
                while not (not state.is_chance_node() and agent.wants_search(state)):
                    if state.is_chance_node():
                        state.apply_action(state.chance_outcomes()[0][0])
                    else:
                        state.apply_action(state.legal_actions()[0])
                visits, amap = agent.search(state)
                self.assertEqual(sum(visits.values()), 12)  # every simulation reaches the root
                self.assertTrue(set(amap.values()) <= set(state.legal_actions()))
                returns, _ = play_game([agent] + [make("balanced") for _ in range(3)], rules, 3)
                self.assertAlmostEqual(sum(returns), 1.0)


class SearchOptionTests(unittest.TestCase):
    def _nets(self, tmp):
        import os
        import torch
        from bots.rl.features import FEATURE_VERSION
        from bots.rl.model import PolicyValueNet, save
        from bots.rl.value_models import SeatValueNet, save_value
        rules = pgcore.Rules(players=4)
        torch.manual_seed(0)
        size, n = Encoder(rules).size, AbstractActions(rules).n
        pnet = PolicyValueNet(size, n, 32)
        pnet.feature_version = FEATURE_VERSION
        fp, fv = os.path.join(tmp, "p.pt"), os.path.join(tmp, "v.pt")
        save(fp, pnet)
        save_value(fv, SeatValueNet(size, hidden=16, depth=2))
        return rules, fp, fv

    def _first_search_state(self, agent, rules):
        state = pgcore.State(rules)
        while state.is_chance_node() or not agent.wants_search(state):
            if state.is_chance_node():
                state.apply_action(state.chance_outcomes()[0][0])
            else:
                state.apply_action(state.legal_actions()[0])
        return state

    def test_options_parse_and_play_legal_games(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            rules, fp, fv = self._nets(tmp)
            a = make(f"mcts:8+norm+c3+fuel+reuse+om+np:{fp}:{fv}")
            self.assertTrue(a.reuse_tree and a.opponent_model and a.fast is not None)
            self.assertIn("BUY_FUEL", a.phases)
            plain = make(f"mcts:8+norm+c3:{fp}:{fv}")
            self.assertFalse(plain.reuse_tree or plain.opponent_model or plain.fast is not None)
            returns, _ = play_game([a] + [make("builder") for _ in range(3)], rules, 4)
            self.assertAlmostEqual(sum(returns), 1.0)

    def test_numpy_forward_matches_torch(self):
        import tempfile
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            rules, fp, fv = self._nets(tmp)
            for value in (fv, ""):
                torch_agent = make(f"mcts:8:{fp}" + (f":{value}" if value else ""))
                np_agent = make(f"mcts:8+np:{fp}" + (f":{value}" if value else ""))
                for ag in (torch_agent, np_agent):
                    ag.reset(rules, 0, random.Random(0))
                state = self._first_search_state(torch_agent, rules)
                p1, _, v1 = torch_agent._evaluate(state)
                p2, _, v2 = np_agent._evaluate(state)
                self.assertTrue(np.allclose([p1[x] for x in p1], [p2[x] for x in p1], atol=1e-5))
                self.assertTrue(np.allclose(v1, v2, atol=1e-5))

    def test_tree_reuse_reuses_the_reached_subtree(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            rules, fp, fv = self._nets(tmp)
            a = make(f"mcts:30+norm+c3+reuse:{fp}:{fv}")
            reused = []
            original = a._reused_root

            def spy(state):
                node = original(state)
                reused.append(node is not None)
                if node is not None:            # a reused root is exactly the current state
                    self.assertEqual(node.state.to_json(), state.to_json())
                return node
            a._reused_root = spy
            returns, _ = play_game([a] + [make("builder") for _ in range(3)], rules, 2)
            self.assertAlmostEqual(sum(returns), 1.0)
            self.assertTrue(any(reused))

    def test_opponent_model_identifies_scripted_not_networks(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            rules, fp, fv = self._nets(tmp)
            a = make(f"mcts:4+norm+om:{fp}:{fv}")
            play_game([a, make("builder"), make(f"rl:{fp}"), make("tycoon")], rules, 3)
            top = {p: max(a.om.posterior(p).items(), key=lambda kv: kv[1], default=(None, 0.0))
                   for p in (1, 2, 3)}
            self.assertEqual(top[1][0], "builder")
            self.assertGreater(top[1][1], 0.95)
            self.assertEqual(top[3][0], "tycoon")
            self.assertLess(top[2][1], 0.95)    # the network is not mistaken for a script


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
