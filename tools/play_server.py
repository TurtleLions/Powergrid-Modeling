"""Watch bots play live, or take a seat against them, in a browser.

    PY=~/.local/opt/python311/bin/python3
    $PY -m tools.play_server --port 8765

Open http://<host>:8765/ and pick an agent for each seat ("human" is you);
leave every seat to bots to watch. Recorded replays in runs/replays are listed
too. Everything is served from this one process with the standard library: the
page is tools/replay_viewer.html in live mode, games run in background threads,
and the page polls for new moves.

There is no login: serve it only on a private network (e.g. `tailscale serve`,
never `tailscale funnel`). Network checkpoints are only loaded from runs/ and models/.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import replay
from .replay import ROOT, board, explain_and_act

import pgcore  # noqa: E402  (replay put cpp/build on the path)
from bots.heuristic import STYLES, make  # noqa: E402

HUMAN = "human"
MAX_GAMES = 30
RUNS = os.path.join(ROOT, "runs")
REPLAYS = os.path.join(RUNS, "replays")
BEST = "mcts:200+norm+c3+np+om+reuse+fuel:models/policy.pt:models/value.pt"


def catalog():
    """Agent names offered in the lobby: the strongest first."""
    out = []
    if os.path.exists(os.path.join(ROOT, "models/policy.pt")):
        out += [BEST, BEST.replace("mcts:200", "mcts:100"), "rl:models/policy.pt"]
    for p in ("runs/league3/champion.pt", "runs/exit4/best.pt", "runs/exit3/best.pt", "runs/exit2/best.pt",
              "runs/league3/best.pt", "runs/league_v2b/best.pt", "runs/league/best.pt"):
        if os.path.exists(os.path.join(ROOT, p)):
            out.append("rl:" + p)
    out += list(STYLES) + ["randomized", "random"]
    return out


def check_agent(name: str):
    """Only load checkpoints under runs/ or models/: torch checkpoints are pickles."""
    allowed = [os.path.realpath(d) + os.sep for d in (RUNS, os.path.join(ROOT, "models"))]
    for part in name.replace("|", ":").split(":"):
        part = part.split("@")[0]                    # per-player-count value spec: a.pt@4|b.pt
        if part.endswith(".pt"):
            path = os.path.realpath(os.path.join(ROOT, part))
            if not any(path.startswith(d) for d in allowed) or not os.path.exists(path):
                raise ValueError(f"no checkpoint {part} under runs/ or models/")


class Game:
    def __init__(self, names, seed):
        self.id = uuid.uuid4().hex[:8]
        self.names = names
        self.seed = seed
        self.created = time.time()
        self.rules = pgcore.Rules(players=len(names))
        self.rng = random.Random(seed)
        self.agents = []
        for seat, n in enumerate(names):
            if n == HUMAN:
                self.agents.append(None)
                continue
            check_agent(n)
            try:
                a = make(n)
            except KeyError:
                raise ValueError(f"unknown agent {n!r}") from None
            a.reset(self.rules, seat, random.Random(self.rng.random()))
            self.agents.append(a)
        self.state = pgcore.State(self.rules)
        self.frames = [self._frame(None, "Game start", None)]
        self.cond = threading.Condition()
        self.waiting = None       # {"seat", "legal": [[action, text]]} while a human is to move
        self.pending = None
        self.error = None
        self.closed = False
        self.advisors = {}
        threading.Thread(target=self._run, daemon=True).start()

    def _frame(self, actor, text, think):
        s = json.loads(self.state.to_json())
        s.pop("occupants", None)
        return {"state": s, "actor": actor, "text": text, "think": think}

    def _run(self):
        try:
            st = self.state
            while not st.is_terminal() and not self.closed:
                before = self.frames[-1]["state"]
                if st.is_chance_node():
                    outs = st.chance_outcomes()
                    a = self.rng.choices([o for o, _ in outs], [p for _, p in outs])[0]
                    raw, actor, think = st.action_to_string(a), -1, None
                elif self.agents[st.current_player()] is None:
                    actor, think = st.current_player(), None
                    legal = st.legal_actions()
                    with self.cond:
                        self.waiting = {"seat": actor, "legal": [[x, st.action_to_string(x)] for x in legal]}
                        self.cond.notify_all()
                        self.cond.wait_for(lambda: self.pending is not None or self.closed)
                        if self.closed:
                            return
                        a, self.pending, self.waiting = self.pending, None, None
                    raw = st.action_to_string(a)
                else:
                    actor = st.current_player()
                    a, think = explain_and_act(self.agents[actor], st)
                    raw = st.action_to_string(a)
                for ag in self.agents:          # opponent modelling and tree reuse see every move
                    if ag is not None:
                        ag.observe(st, actor, a)
                st.apply_action(a)
                frame = self._frame(actor, raw, think)
                if actor == -1:
                    frame["text"] = replay._chance_text(before, frame["state"], raw)
                with self.cond:
                    self.frames.append(frame)
                    self.cond.notify_all()
        except Exception:
            self.error = traceback.format_exc()
            print(self.error, file=sys.stderr)

    def move(self, action: int, seen: int):
        with self.cond:
            if self.waiting is None or seen != len(self.frames):
                raise ValueError("it isn't your move in this position any more")
            if action not in [x for x, _ in self.waiting["legal"]]:
                raise ValueError("that move isn't legal here")
            self.pending = action
            self.cond.notify_all()

    def hint(self, advisor: str):
        """What `advisor` would play for the human to move (computed on a copy)."""
        with self.cond:
            if self.waiting is None:
                raise ValueError("no one is waiting for a hint")
            seat = self.waiting["seat"]
            state = self.state.clone()
        key = (advisor, seat)
        if key not in self.advisors:
            check_agent(advisor)
            a = make(advisor)
            a.reset(self.rules, seat, random.Random(seat))
            self.advisors[key] = a
        action, think = explain_and_act(self.advisors[key], state)
        return {"advisor": advisor, "action": state.action_to_string(action), "think": think}

    def summary(self):
        s = self.frames[-1]["state"]
        return {"id": self.id, "agents": self.names, "seed": self.seed, "round": s["round"],
                "moves": len(self.frames) - 1, "over": self.state.is_terminal(),
                "waiting": self.waiting is not None, "created": self.created}

    def view(self, since: int):
        with self.cond:
            over = self.state.is_terminal()
            return {**self.summary(), "frames": self.frames[since:], "total": len(self.frames),
                    "returns": list(self.state.returns()) if over else None,
                    "turn": self.waiting, "error": self.error}


GAMES: dict = {}
GAMES_LOCK = threading.Lock()


def new_game(names, seed):
    if not 2 <= len(names) <= 6:
        raise ValueError("pick 2 to 6 seats")
    g = Game(names, seed)
    with GAMES_LOCK:
        GAMES[g.id] = g
        for old in sorted(GAMES.values(), key=lambda x: x.created)[:-MAX_GAMES]:
            old.closed = True
            with old.cond:
                old.cond.notify_all()
            del GAMES[old.id]
    return g


LIVE = {}


def page():
    """The viewer in live mode, read on every request so page fixes apply without a restart."""
    with open(replay.TEMPLATE, encoding="utf-8") as f:
        html = f.read()
    return html.replace("/*__LIVE__*/null", json.dumps(LIVE, ensure_ascii=False).replace("</", "<\\/"))


class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        if not self.path.startswith("/api/games/") or self.command == "POST":
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _send(self, body, status=200, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, status=200):
        self._send(json.dumps(obj, separators=(",", ":"), ensure_ascii=False), status)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _game(self, gid):
        g = GAMES.get(gid)
        if g is None:
            self._json({"error": "no such game (the server may have restarted)"}, 404)
        return g

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        try:
            if url.path == "/":
                return self._send(page(), ctype="text/html")
            if url.path == "/api/games":
                return self._json(sorted((g.summary() for g in GAMES.values()), key=lambda s: -s["created"]))
            if m := re.fullmatch(r"/api/games/(\w+)", url.path):
                if g := self._game(m[1]):
                    self._json(g.view(int(q.get("since", ["0"])[0])))
                return
            if m := re.fullmatch(r"/api/games/(\w+)/hint", url.path):
                if g := self._game(m[1]):
                    self._json(g.hint(q.get("advisor", [catalog()[0]])[0]))
                return
            if url.path == "/api/replays":
                files = sorted(glob.glob(os.path.join(REPLAYS, "*.html")), key=os.path.getmtime, reverse=True)
                return self._json([os.path.basename(f) for f in files])
            m = re.fullmatch(r"/replays/([\w.-]+\.html)", url.path)
            if m and os.path.exists(os.path.join(REPLAYS, m[1])):
                with open(os.path.join(REPLAYS, m[1]), "rb") as f:
                    return self._send(f.read(), ctype="text/html")
            self._json({"error": "not found"}, 404)
        except ValueError as e:
            self._json({"error": str(e)}, 400)

    def do_POST(self):
        url = urlparse(self.path)
        try:
            body = self._body()
            if url.path == "/api/games":
                seed = body.get("seed")
                seed = random.randrange(1 << 30) if seed in (None, "") else int(seed)
                names = [str(n).strip() for n in body.get("agents", [])]
                return self._json(new_game(names, seed).summary())
            if m := re.fullmatch(r"/api/games/(\w+)/move", url.path):
                if g := self._game(m[1]):
                    g.move(int(body["action"]), int(body["seen"]))
                    self._json({"ok": True})
                return
            self._json({"error": "not found"}, 404)
        except (ValueError, KeyError) as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:      # a bad agent name from make(), a missing file, ...
            self._json({"error": f"{type(e).__name__}: {e}"}, 400)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    rules = pgcore.Rules(players=4)
    LIVE.update(catalog=catalog(), board=board(rules), human=HUMAN)
    page()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"serving on http://{args.host}:{args.port}/", file=sys.stderr)
    srv.serve_forever()


if __name__ == "__main__":
    main()
