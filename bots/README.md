# Power Grid bots

Bots play the C++ engine through the `pgcore` Python module (`make -C cpp pgcore`).
The player count is a parameter everywhere; the default is 4.

| File | What it is |
|---|---|
| `base.py` | The `Agent` interface (`act`), plus `speak`/`hear` hooks reserved for LLM agents that talk. |
| `heuristic.py` | Scripted bots in four styles: `balanced`, `builder`, `tycoon`, `miser`. Also `make(name)`, which accepts those, `random`, or `rl:<checkpoint.pt>`. |
| `arena.py` | Plays many games in parallel and reports win rates with 95% intervals. |
| `rl/features.py` | Ego-centric state features padded to 6 seats, and the abstract action space (bids as raises). |
| `rl/model.py` | Policy/value network, and `RLAgent`, which plays a checkpoint. |
| `rl/train.py` | PPO against a league of opponents: self-play, the scripted styles, random, and snapshots of earlier versions. Opponents it loses to are drawn more often. |

## Usage

    make -C cpp pgcore
    PY=~/.local/opt/python311/bin/python3

    # scripted bots against each other
    $PY -m bots.arena --agents balanced,builder,tycoon,miser,random --games 1000

    # train (checkpoints, snapshots and log.jsonl go to --out)
    $PY -m bots.rl.train --out runs/ppo --iterations 300

    # can the trained bot win against anyone? One seat is the bot, the rest are drawn from --agents
    $PY -m bots.arena --focus rl:runs/ppo/latest.pt --agents balanced,builder,tycoon,miser --games 1000

With 4 players, a win rate of 0.25 means the bot is only as good as the average player at the table.
