// powergrid_bench -- random-play throughput and clone cost for the C++ engine.
// Usage: powergrid_bench [games] [players] [keep_log 0/1] [map: germany|tiny]

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

#include "powergrid_engine.h"

int main(int argc, char** argv) {
  int games = argc > 1 ? std::atoi(argv[1]) : 20000;
  int players = argc > 2 ? std::atoi(argv[2]) : 4;
  bool tiny = argc > 4 && std::string(argv[4]) == "tiny";
  auto rules = std::make_shared<powergrid::Rules>();
  if (tiny) {
    *rules = powergrid::TinyRules(players);
  } else {
    rules->num_players = players;
    rules->Finalize();
  }
  rules->keep_log = argc > 3 ? std::atoi(argv[3]) != 0 : false;

  std::mt19937_64 rng(0);
  long moves = 0;
  auto t0 = std::chrono::steady_clock::now();
  for (int g = 0; g < games; ++g) {
    powergrid::State s(rules);
    while (!s.IsTerminal()) {
      int a;
      if (s.IsChanceNode()) {
        auto outs = s.ChanceOutcomes();  // uniform
        a = outs[rng() % outs.size()].first;
      } else {
        auto acts = s.LegalActions();
        a = acts[rng() % acts.size()];
      }
      s.ApplyAction(a);
      ++moves;
    }
  }
  double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("%s, %d players, log=%d: %.0f games/s, %.2fM moves/s, %.0f moves/game\n",
              rules->map.name.c_str(), players,
              int(rules->keep_log), games / dt, moves / dt / 1e6, double(moves) / games);

  // clone cost around round 10
  powergrid::State s(rules);
  while (!s.IsTerminal() && s.round() < 10) {
    if (s.IsChanceNode()) s.ApplyAction(s.ChanceOutcomes()[0].first);
    else { auto acts = s.LegalActions(); s.ApplyAction(acts[rng() % acts.size()]); }
  }
  const int reps = 1000000;
  long sink = 0;
  t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < reps; ++i) {
    powergrid::State c = s;
    asm volatile("" : : "g"(&c) : "memory");  // keep the copy from being optimized away
    sink += c.money(0);
  }
  dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("clone at round %d (log %zu events): %.0f ns  [%ld]\n", s.round(), s.log().size(),
              dt / reps * 1e9, sink % 7);
  return 0;
}
