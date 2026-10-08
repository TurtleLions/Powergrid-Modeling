# Power Grid: a game engine and search-based AI

A faithful implementation of the board game **Power Grid (Recharged edition)**
and an AI that plays it at 3–6 players, built with tree search, learned value
and policy networks, and a self-play league with exploiters. The trained agent
is included ([`models/`](models)) and you can play against it in your browser.

- **Engine**: a pure-Python rules specification ([`powergrid_core.py`](powergrid_core.py),
  rulebook pages cited throughout) and a C++17 port ([`cpp/`](cpp)) verified
  move-for-move against it (`make -C cpp check`: random games, identical traces).
  The state clones in nanoseconds, which makes search practical. An
  [OpenSpiel](https://github.com/google-deepmind/open_spiel) adapter is included.
- **AI**: Monte Carlo tree search guided by a policy network and a separate,
  all-seats value network, trained in a league of search agents
  ([`bots/`](bots)). The whole history — what worked and what did not — is in
  the commit log.
- **Play and watch**: a browser lobby to take a seat against the agents or watch
  them play, and recorded replays that show what each network was thinking
  ([`tools/`](tools)).

## How strong is it?

All numbers are measured against this project's own agents and eleven scripted
strategies (no human or external benchmark yet). Elo is relative within each
ladder.

**Against scripted strategies**: one seat of the included agent against copies
of each of 11 scripted styles (`builder`, `tycoon`, `blocker`, `hoarder`, ...)
plus a randomised style; "worst" is the lowest win rate over the 12 opponent
types (a fair share is 1/players).

| players | fair share | worst case | mean |
|---|---|---|---|
| 3 | 0.33 | 0.96 (vs `builder`) | 0.997 |
| 4 | 0.25 | 0.98 (vs `builder`) | 0.998 |
| 5 | 0.20 | 1.00 (won all 600 games) | 1.000 |
| 6 | 0.17 | 1.00 (won all 600 games) | 1.000 |

(50 games per opponent type and player count, 2,400 games in all.)

The scripted strategies are no longer a useful yardstick; the ladder of the
project's own agents is.

**Progress over the project**: Elo ladder per player count among eight agents
(mixed tables, 300 per count; about 110–225 games per agent, intervals ±40–100):

| agent | 3p | 4p | 5p | 6p |
|---|---|---|---|---|
| **current best** (league iteration 11, in `models/`) | **252** | **148** | **211** | **251** |
| strongest exploiter of the league | 184 | 140 | 198 | 214 |
| search, 2026-10-01 (value net v4, plain search) | −248 | −43 | −134 | −76 |
| search, 2026-09-30 (first value net, plain search) | −172 | −73 | −295 | −433 |
| policy network alone, no search (PPO) | −391 | −384 | −387 | −362 |

The current best is 190–500 Elo above the agent of two days earlier, depending
on the player count, and the last two exploiters trained specifically against
it failed to beat it significantly (1.12× a fair share, 95% interval [0.96, 1.27]).

**Against an LLM**: Claude (Opus 5.5) played one full 4-player game against three
copies of the current best agent, making every auction, fuel and building
decision itself (only running plants was automated). It finished **3rd of 4**:
the agents powered 18, 17 and 16 cities, Claude 16 (ahead of the fourth on
money). Claude's start in the empty north-east and early plant spending left it
short of cities mid-game; the agents expanded faster and ended richer.
Move log: [`docs/claude_vs_agents_game1.log`](docs/claude_vs_agents_game1.log).

**Speed**: 100 simulations per searched decision; a full game of search takes
about 1.5–2.5 s of thinking per player on one core.

## Play against it

```bash
make -C cpp pgcore                          # build the engine's Python module
python3 -m tools.play_server --port 8765    # lobby at http://localhost:8765/
```

Pick an agent for each seat ("human" is you) or leave every seat to bots to
watch. The strongest agent in the lobby is the included one at 200 simulations.
The server has no login: run it on a private network only.

To record a game and watch it later, with each network's top moves and win
estimate at every decision:

```bash
python3 -m tools.replay --agents "mcts:100+norm+c3+np+om+reuse+fuel:models/policy.pt:models/value.pt,balanced,builder,tycoon" --seed 1
# -> runs/replays/game_1.html
```

## How it works

```
                 ┌──────────── league iteration (~1 h on 32 cores) ─────────────┐
 self-play games │ seats: current best ~57% · past bests/exploiters ~29% ·      │
 (3–6 players)   │ scripted ~14%; search targets recorded at every decision     │
        │        └───────────────────────────────────────────────────────────────┘
        ▼
 value network: fine-tuned on recent league games, TD(λ) targets
 policy network: distilled from the search's visit counts
        ▼
 league gate: candidates + best + pool rated together; highest mean Elo wins
        ▼
 every 3rd iteration: benchmark ladder, then an exploiter trained only
 against the best — if it wins, its games join training and it joins the pool
```

**Search** ([`bots/search.py`](bots/search.py)): PUCT tree search with max^n
backups (each player maximises their own expected win share), plant draws as
chance nodes, and Q values min-max normalised per node — without that, a
confident prior was almost never overruled (98.5% agreement with the raw
policy; normalising was worth +139 Elo). Options:

| option | what it does | measured effect |
|---|---|---|
| `reuse` | keep the subtree of the position actually reached | 1.20–1.35× a fair share at equal simulations |
| `om` | identify scripted opponents from their moves and model them exactly in the tree | worst case vs scripted up to +12 points; never fires on networks |
| `fuel` | also search fuel purchases | 1.30× a fair share (for 1.8× the time) |
| `np` | NumPy forward passes | same moves, ~2.7× faster |

**Value network** ([`bots/rl/value_models.py`](bots/rl/value_models.py),
[`bots/rl/value_td.py`](bots/rl/value_td.py)): scores a position from every
seat's view at once (softmax over seats). Trained on TD(λ) targets — the result
blended with an out-of-fold teacher's prediction for later positions — instead
of the noisy final result, which also stopped it memorising games.

**League** ([`bots/rl/search_league.py`](bots/rl/search_league.py)): the loop
above, resumable, with playout-cap randomisation for cheap training games and
the next iteration's games generated while the current one trains and gates.

### What did not work (and why it is still in the history)

- **Bigger value networks** (3×1024, 6×512): ≤0.003 better held-out
  cross-entropy for 2–4× the search cost — data and targets were the limit.
- **Batched leaf evaluation** (virtual loss): ~2× cheaper per simulation but
  weaker at equal time (−29 to −93 Elo).
- **Long expert-iteration runs**: gains only in the first generations; 400-game
  gates could not see ~20 Elo.
- **Head-to-head gates against the best**: fragile once exploiters joined the
  league (strength is not transitive) — replaced by the league gate.
- **Fine-tuning on all past data**: older games from weaker agents diluted the
  signal; exploiters (trained on recent games only) kept overtaking the main
  line until it, too, trained on a window of recent games.

## Quick start

Requirements: Python 3.11 with NumPy and PyTorch, a C++17 compiler, pybind11.

```bash
make -C cpp pgcore                      # build the engine's Python module
make -C cpp check                       # C++ vs Python spec: identical traces
python3 -m unittest                     # tests

# the included agent vs copies of every scripted style, at any player count
python3 -m bots.headtohead "mcts:100+norm+c3+np+om+reuse+fuel:models/policy.pt:models/value.pt" 4 20 log.jsonl 8

# watch scripted bots, or rate agents on one Elo scale
python3 -m bots.arena --agents balanced,builder,tycoon,miser --games 400
python3 -m bots.rating --agents "builder,balanced,mcts:100+norm+c3+np:models/policy.pt:models/value.pt" --games 400

# the league (resumable; progress in runs/sleague/report.md)
python3 -m bots.rl.search_league --dir runs/sleague --workers 31
```

Agent names: `builder` (a scripted style), `rl:policy.pt` (raw network),
`mcts:<sims>[+norm][+c<c_puct>][+reuse][+om][+fuel][+np]:policy.pt[:value.pt]`
(search; the value network may differ per player count: `v4.pt@4|v5.pt`).

## Repository layout

| path | contents |
|---|---|
| `powergrid_core.py` | rules specification (Python, no dependencies) and its tests |
| `cpp/` | C++ engine, Python bindings (`pgcore`), OpenSpiel adapter, diff test |
| `bots/` | scripted styles, arena, Elo ladder, head-to-head, search agent |
| `bots/rl/` | features, networks, PPO, expert iteration, value/policy training, the league |
| `models/` | the current best agent: `policy.pt` (policy network) and `value.pt` (seat value network) |
| `tools/` | browser play server and replay viewer; data generation and diff test for the C++ engine |
| `docs/` | the move log of Claude's game against the agents |

## Scope and limitations

- 3–6 players; the 2-player variant (with the Trust) is implemented in the
  engine but not trained on.
- The German map only.
- Strength is relative to the project's own agents and scripted strategies.
