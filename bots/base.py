"""The interface every Power Grid bot implements.

A bot sees the full engine state (Power Grid has no hidden information apart
from the plant stack, and money is public) and returns one of
state.legal_actions() whenever it is its seat's turn.

Communication hooks: `speak` and `hear` are no-ops for bots that do not talk.
The arena will call them for LLM agents later; defining them now means scripted
and RL bots can sit at the same table as talking agents without changes.
"""
from __future__ import annotations

import json
import random
from typing import Optional


class Agent:
    name = "agent"

    def reset(self, rules, seat: int, rng: random.Random) -> None:
        """Called once before each game."""
        self.rules = rules
        self.seat = seat
        self.rng = rng

    def act(self, state) -> int:
        """Return one of state.legal_actions(); state.current_player() == self.seat."""
        raise NotImplementedError

    # -- communication (for LLM agents later) ---------------------------------
    def speak(self, state) -> Optional[str]:
        """A message to the table before acting, or None."""
        return None

    def hear(self, speaker: int, message: str) -> None:
        """A message another seat sent to the table."""


class RandomAgent(Agent):
    """Uniformly random legal action: the floor any bot must beat."""

    name = "random"

    def act(self, state) -> int:
        return self.rng.choice(state.legal_actions())


def view(state) -> dict:
    """The engine's structured JSON view as a dict."""
    return json.loads(state.to_json())
