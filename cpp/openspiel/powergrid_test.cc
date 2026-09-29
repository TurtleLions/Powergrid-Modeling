#include "open_spiel/games/powergrid/powergrid.h"

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace powergrid {
namespace {

namespace testing = open_spiel::testing;

// Short games: the tiny map reaches step 3 and the end in ~15 rounds.
std::shared_ptr<const Game> TinyGame(int players, bool keep_log = false) {
  return LoadGame("powergrid", {{"players", GameParameter(players)},
                                {"map", GameParameter(std::string("tiny"))},
                                {"play_regions", GameParameter(3)},
                                {"step2_cities", GameParameter(3)},
                                {"end_cities", GameParameter(6)},
                                {"max_rounds", GameParameter(40)},
                                {"keep_log", GameParameter(keep_log)}});
}

void BasicTests() {
  testing::LoadGameTest("powergrid");
  testing::ChanceOutcomesTest(*LoadGame("powergrid"));
  for (int players = 2; players <= 6; ++players) {
    testing::RandomSimTest(*TinyGame(players), 20);
  }
  testing::RandomSimTest(*TinyGame(4, /*keep_log=*/true), 5);
  // The real game: Germany, rulebook settings, random regions.
  for (int players = 2; players <= 6; ++players) {
    testing::RandomSimTest(
        *LoadGame("powergrid", {{"players", GameParameter(players)}}), 5);
  }
  testing::RandomSimTest(*LoadGame("powergrid(regions=north-west+west+east,players=3)"), 3);
  testing::RandomSimTest(*LoadGame("powergrid(regions=0:1:2,players=3)"), 3);
  testing::RandomSimTest(*LoadGame("powergrid(map=tiny,play_regions=3,max_rounds=12)"), 3);
  testing::RandomSimTest(*LoadGame("powergrid(map=germany-2004,players=4)"), 3);
  testing::RandomSimTest(*LoadGame("powergrid(players=2,trust=0)"), 3);  // no Trust
}

void RegionChance() {
  std::shared_ptr<const Game> game = LoadGame("powergrid(players=4)");
  std::unique_ptr<State> state = game->NewInitialState();
  SPIEL_CHECK_TRUE(state->IsChanceNode());
  SPIEL_CHECK_EQ(state->ChanceOutcomes().size(), 12);  // adjacent sets of 4 regions
  std::string first = state->ActionToString(kChancePlayerId, 0);
  SPIEL_CHECK_EQ(first.substr(0, 8), "Regions ");
  state->ApplyAction(0);
  SPIEL_CHECK_EQ(state->ActionToString(kChancePlayerId, 0), "Turn order 0 1 2 3");
}

void SerializationRoundTrip() {
  std::shared_ptr<const Game> game = TinyGame(4);
  std::mt19937 rng(7);
  std::unique_ptr<State> state = game->NewInitialState();
  for (int i = 0; i < 400 && !state->IsTerminal(); ++i) {
    std::vector<Action> acts = state->LegalActions();
    state->ApplyAction(acts[rng() % acts.size()]);
    std::unique_ptr<State> copy = game->DeserializeState(state->Serialize());
    SPIEL_CHECK_EQ(state->ToString(), copy->ToString());
    SPIEL_CHECK_EQ(state->LegalActions(), copy->LegalActions());
  }
}

void FirstRoundEveryoneMustBuy() {
  std::shared_ptr<const Game> game = TinyGame(4);
  std::unique_ptr<State> state = game->NewInitialState();
  state->ApplyAction(0);  // turn order P0 P1 P2 P3
  SPIEL_CHECK_EQ(state->ActionToString(kChancePlayerId, 3), "Deal plant #3");
  while (state->IsChanceNode()) state->ApplyAction(state->LegalActions()[0]);  // deal 03-10
  // P0 selects, cannot pass in round 1; #3 carries the discount token
  std::vector<Action> acts = state->LegalActions();
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_EQ(acts.front(), 2);  // SELECT slot 0, no PASS (0)
  SPIEL_CHECK_EQ(state->ActionToString(0, 2), "SELECT plant #3 (discounted: minimum bid 1)");
}

void ChanceActionStrings() {
  std::shared_ptr<const Game> game = TinyGame(3);
  std::unique_ptr<State> state = game->NewInitialState();
  SPIEL_CHECK_EQ(state->ActionToString(kChancePlayerId, 0), "Turn order 0 1 2");
  SPIEL_CHECK_EQ(state->ActionToString(kChancePlayerId, 5), "Turn order 2 1 0");
}

void EventLogIsExposed() {
  std::shared_ptr<const Game> game = TinyGame(3, /*keep_log=*/true);
  std::unique_ptr<State> state = game->NewInitialState();
  std::mt19937 rng(3);
  while (!state->IsTerminal()) {
    std::vector<Action> acts = state->LegalActions();
    state->ApplyAction(acts[rng() % acts.size()]);
  }
  const auto& engine = down_cast<const PowerGridState&>(*state).engine();
  SPIEL_CHECK_FALSE(engine.log().empty());
  SPIEL_CHECK_EQ(static_cast<int>(engine.log().back().type),
                 static_cast<int>(::powergrid::EventType::kGameOver));
}

}  // namespace
}  // namespace powergrid
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::powergrid::BasicTests();
  open_spiel::powergrid::SerializationRoundTrip();
  open_spiel::powergrid::FirstRoundEveryoneMustBuy();
  open_spiel::powergrid::ChanceActionStrings();
  open_spiel::powergrid::EventLogIsExposed();
  open_spiel::powergrid::RegionChance();
}
