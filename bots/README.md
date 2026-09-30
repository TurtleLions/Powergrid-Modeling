# Power Grid bots

Bots play the C++ engine through the `pgcore` Python module (`make -C cpp pgcore`).
The player count is a parameter everywhere; the default is 4.

| File | What it is |
|---|---|
| `base.py` | The `Agent` interface (`act`), plus `speak`/`hear` hooks reserved for LLM agents that talk. |
| `heuristic.py` | Scripted bots in 11 styles that differ in strategy: `balanced`, `builder`, `tycoon`, `miser`, `eco`, `nuclear`, `hoarder` (fuel denial), `blocker` (crowds rivals on the map), `planner` (keeps room to grow), `driver` (bids up plants it doesn't want), `turtle` (stays small, then rushes). Plus `randomized`, a fresh style each game. `make(name)` also accepts `random` and `rl:<checkpoint.pt>`. |
| `arena.py` | Plays many games in parallel and reports win rates with 95% intervals. |
| `evaluate.py` | Worst-case evaluation: one seat against N-1 copies of each opponent type, reporting the minimum across types. |
| `rl/features.py` | Ego-centric state features padded to 6 seats, and the abstract action space (bids as raises). |
| `rl/model.py` | Policy/value network, and `RLAgent`, which plays a checkpoint. |
| `select.py` | Picks the final bot from a run: a larger worst-case evaluation of `best.pt` and the hall of fame, plus mixed games among those candidates; writes `champion.pt`. |
| `rl/train.py` | PPO against a league: self-play, every scripted style, and a hall of fame of the strongest earlier versions (ranked by worst case). Opponents it loses to are drawn more often. `best.pt` is the checkpoint with the best worst case. |

## Usage

    make -C cpp pgcore
    PY=~/.local/opt/python311/bin/python3

    # scripted bots against each other
    $PY -m bots.arena --agents balanced,builder,tycoon,miser,random --games 1000

    # train (checkpoints, snapshots and log.jsonl go to --out)
    $PY -m bots.rl.train --out runs/league --iterations 1000
    $PY -m bots.rl.train --out runs/league2 --init runs/league/best.pt   # continue from a checkpoint

    # pick the final bot from a run (writes runs/league/champion.pt)
    $PY -m bots.select runs/league --games 400

    # can the trained bot win against anyone? Worst case over opponent types
    $PY -m bots.evaluate rl:runs/league/champion.pt --games 400

With 4 players, a win rate of 0.25 means the bot is only as good as the average player at the table.

Run the full test suite with the Python that has NumPy and PyTorch:

    $PY -m unittest
