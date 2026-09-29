"""End-to-end check of the OpenSpiel build: load "powergrid" through pyspiel,
compare it against powergrid_core.py, and play MCTS against random bots.

    PYTHONPATH=/path/to/open_spiel:/path/to/open_spiel/build/python \
        python3 tools/openspiel_smoke.py
"""
import os
import random
import sys
import time

import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts
from open_spiel.python.bots import uniform_random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import powergrid_core as pg  # noqa: E402

TINY = "powergrid(players={n},map=tiny,play_regions=3,step2_cities=3,end_cities=6,max_rounds=40)"
GERMANY = "powergrid(players={n})"


def check_against_spec(num_games=40):
    """Same random game through pyspiel and the Python spec: legal actions,
    action strings, observation tensors and returns must agree."""
    rng = random.Random(0)
    for g in range(num_games):
        n = 2 + g % 5
        germany = g % 2 == 1
        game = pyspiel.load_game((GERMANY if germany else TINY).format(n=n))
        os_state = game.new_initial_state()
        py_state = pg.PowerGridState(pg.Ruleset(num_players=n) if germany else pg.tiny_ruleset(n))
        while not os_state.is_terminal():
            assert os_state.legal_actions() == py_state.legal_actions()
            if os_state.is_chance_node():
                outs = os_state.chance_outcomes()
                a = rng.choices([o for o, _ in outs], [p for _, p in outs])[0]
            else:
                p = os_state.current_player()
                a = rng.choice(os_state.legal_actions())
                assert os_state.action_to_string(p, a) == py_state.action_to_string(a)
                want = np.concatenate([np.asarray(v, dtype=np.float32)
                                       for v in py_state.observation(p).values()])
                got = np.asarray(os_state.observation_tensor(p), dtype=np.float32)
                np.testing.assert_allclose(got, want, rtol=1e-6)
            os_state.apply_action(a)
            py_state.apply_action(a)
        assert py_state.is_terminal()
        np.testing.assert_allclose(os_state.returns(), py_state.returns())
    print(f"pyspiel matches the Python spec on {num_games} games, Germany and tiny map "
          "(legal actions, action strings, observation tensors, returns)")


def mcts_vs_random(num_games=10, simulations=200):
    game = pyspiel.load_game(GERMANY.format(n=3))
    rng = np.random.RandomState(0)
    evaluator = mcts.RandomRolloutEvaluator(n_rollouts=1, random_state=rng)
    bots = [mcts.MCTSBot(game, uct_c=2, max_simulations=simulations, evaluator=evaluator,
                         random_state=rng)]
    bots += [uniform_random.UniformRandomBot(p, rng) for p in (1, 2)]
    wins = 0.0
    t0 = time.time()
    for g in range(num_games):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                actions, probs = zip(*state.chance_outcomes())
                state.apply_action(rng.choice(actions, p=probs))
            else:
                state.apply_action(bots[state.current_player()].step(state))
        wins += state.returns()[0]
    print(f"MCTS ({simulations} sims) vs 2 random bots: won {wins:.1f}/{num_games} games "
          f"(random baseline ~{num_games / 3:.1f}) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    print("registered:", "powergrid" in pyspiel.registered_names())
    check_against_spec()
    mcts_vs_random()
