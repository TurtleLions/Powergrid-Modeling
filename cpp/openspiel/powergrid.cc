#include "open_spiel/games/powergrid/powergrid.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/numbers.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace powergrid {
namespace {

const GameType kGameType{
    /*short_name=*/"powergrid",
    /*long_name=*/"Power Grid",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    // Money is public and hidden deck order is modelled as chance draws.
    GameType::Information::kPerfectInformation,
    GameType::Utility::kConstantSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/::powergrid::kMaxPlayers,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {{"players", GameParameter(4)},
     {"map", GameParameter(std::string("germany"))},
     {"regions", GameParameter(std::string(""))},
     {"play_regions", GameParameter(-1)},
     {"step2_cities", GameParameter(-1)},
     {"end_cities", GameParameter(-1)},
     {"houses", GameParameter(-1)},
     {"trust", GameParameter(-1)},
     {"max_rounds", GameParameter(100)},
     {"max_bid", GameParameter(400)},
     {"start_money", GameParameter(50)},
     {"keep_log", GameParameter(false)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new PowerGridGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

RegisterSingleTensorObserver single_tensor(kGameType.short_name);

std::shared_ptr<const ::powergrid::Rules> MakeRules(const GameParameters& params) {
  auto get_int = [&](const std::string& key) {
    auto it = params.find(key);
    return it != params.end() ? it->second.int_value()
                              : kGameType.parameter_specification.at(key).int_value();
  };
  auto get_string = [&](const std::string& key) {
    auto it = params.find(key);
    return it != params.end() ? it->second.string_value()
                              : kGameType.parameter_specification.at(key).string_value();
  };
  auto rules = std::make_shared<::powergrid::Rules>();
  std::string map_name = get_string("map");
  try {
    rules->map = ::powergrid::MapByName(map_name);
  } catch (const std::exception& e) {
    SpielFatalError(absl::StrCat("powergrid: ", e.what()));
  }
  // Region names or indices joined by '+' (e.g. "north-west+east") or ':'
  // ("0:3"); game strings already use ',' between parameters.
  const std::string regions = get_string("regions");
  for (absl::string_view tok : absl::StrSplit(regions, absl::ByAnyChar("+:,"), absl::SkipEmpty())) {
    const auto& names = rules->map.region_names;
    auto it = std::find(names.begin(), names.end(), tok);
    int region = -1;
    if (it != names.end()) {
      region = static_cast<int>(it - names.begin());
    } else if (!absl::SimpleAtoi(tok, &region) || region < 0 ||
               region >= rules->map.num_regions) {
      SpielFatalError(absl::StrCat("powergrid: unknown region '", tok, "' in ", regions));
    }
    rules->regions.push_back(region);
  }
  rules->play_regions = get_int("play_regions");
  rules->houses = get_int("houses");
  rules->trust = get_int("trust");
  rules->num_players = get_int("players");
  rules->step2_cities = get_int("step2_cities");
  rules->end_cities = get_int("end_cities");
  rules->max_rounds = get_int("max_rounds");
  rules->max_bid = get_int("max_bid");
  rules->start_money = get_int("start_money");
  auto log_it = params.find("keep_log");
  rules->keep_log = log_it != params.end() && log_it->second.bool_value();
  try {
    rules->Finalize();
  } catch (const std::exception& e) {
    SpielFatalError(absl::StrCat("powergrid: ", e.what()));
  }
  return rules;
}

}  // namespace

// ---------------------------------------------------------------------------
PowerGridState::PowerGridState(std::shared_ptr<const Game> game,
                               std::shared_ptr<const ::powergrid::Rules> rules)
    : State(std::move(game)), engine_(std::move(rules)) {}

std::vector<Action> PowerGridState::LegalActions() const {
  std::vector<int> acts = engine_.LegalActions();
  return std::vector<Action>(acts.begin(), acts.end());
}

ActionsAndProbs PowerGridState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  ActionsAndProbs out;
  for (const auto& [a, p] : engine_.ChanceOutcomes()) out.emplace_back(a, p);
  return out;
}

std::string PowerGridState::ActionToString(Player player, Action action_id) const {
  if (player == kChancePlayerId) {
    if (engine_.IsRegionChance()) {
      const auto& rules = engine_.rules();
      std::vector<std::string> names;
      for (int r : rules.region_sets[action_id]) names.push_back(rules.map.region_names[r]);
      return absl::StrCat("Regions ", absl::StrJoin(names, ", "));
    }
    if (engine_.IsInitialOrderChance()) {
      std::vector<int> perm(num_players_);
      std::iota(perm.begin(), perm.end(), 0);
      for (Action i = 0; i < action_id; ++i) std::next_permutation(perm.begin(), perm.end());
      return absl::StrCat("Turn order ", absl::StrJoin(perm, " "));
    }
    return absl::StrCat(engine_.IsSetupChance() ? "Deal plant #" : "Draw plant #", action_id);
  }
  return engine_.ActionToString(static_cast<int>(action_id));
}

std::string PowerGridState::InformationStateString(Player player) const {
  // Perfect information: the action history identifies the state.
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string PowerGridState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return absl::StrCat("You are P", player, "\n", engine_.Describe());
}

void PowerGridState::ObservationTensor(Player player, absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  SPIEL_CHECK_EQ(values.size(), ::powergrid::State::ObservationSize(engine_.rules()));
  engine_.ObservationTensor(player, values.data());
}

// ---------------------------------------------------------------------------
PowerGridGame::PowerGridGame(const GameParameters& params)
    : Game(kGameType, params), rules_(MakeRules(params)), codec_(*rules_) {}

int PowerGridGame::MaxChanceOutcomes() const {
  return std::max({::powergrid::State::NumOrders(rules_->num_players),
                   ::powergrid::kMaxPlantNumber + 1,
                   static_cast<int>(rules_->region_sets.size())});
}

int PowerGridGame::MaxGameLength() const {
  // Upper bound on player decisions per round, times the round cap.
  const ::powergrid::Rules& r = *rules_;
  const int n = r.num_players;
  // auctions: at most n sales; each has <= max_bid raises, n drops, a select
  // and a discard; plus n passes
  int auction = n * (r.max_bid + n + 2) + n;
  int fuel = 0;  // every unit can be bought (and returned after a discard), then n DONEs
  for (int f = 0; f < ::powergrid::kNumFuels; ++f) fuel += 2 * r.fuel_totals[f];
  fuel += n;
  int build = ::powergrid::kMaxOccupants * r.map.num_cities + n;
  int run = n * (r.plant_limit + 1);
  return ::powergrid::kTrustStartHouses + r.max_rounds * (auction + fuel + build + run);
}

}  // namespace powergrid
}  // namespace open_spiel
