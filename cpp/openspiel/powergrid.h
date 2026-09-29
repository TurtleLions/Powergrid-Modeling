// OpenSpiel adapter for the Power Grid engine (powergrid_engine.h).
//
// The rules live in the dependency-free engine; this file only maps it onto
// the OpenSpiel State/Game API. Install into an OpenSpiel checkout with
// cpp/openspiel/install_into_openspiel.sh.
//
// Rules: Power Grid Recharged. Game parameters (all optional; -1 means
// "rulebook value for the player count"):
//   players       2..6 (default 4)
//   map           "germany" (Recharged board, default), "germany-2004" (the
//                 original board) or "tiny" (12-city test map)
//   regions       regions in play, by name joined with '+' (Germany: north-west,
//                 north-east, east, west, south-west, south-east), e.g.
//                 "north-west+west+east", or by index joined with ':'; "" = drawn
//                 at random among the adjacent sets (chance node)
//   play_regions  number of regions in play
//   step2_cities  step-2 trigger
//   end_cities    game-end trigger
//   houses        houses per player (22)
//   trust         2-player "Against the Trust" rules: -1 = on for 2 players,
//                 0 = off, 1 = on
//   max_rounds    safety cap, not a rule (default 100)
//   max_bid       largest bid; sets the size of the action space (default 400)
//   start_money   (default 50)
//   keep_log      record the auction/purchase/build event log (default false;
//                 costs clone time, and the log can be rebuilt by replaying
//                 History() with it on)

#ifndef OPEN_SPIEL_GAMES_POWERGRID_H_
#define OPEN_SPIEL_GAMES_POWERGRID_H_

#include <memory>
#include <string>
#include <vector>

#include "open_spiel/games/powergrid/powergrid_engine.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace powergrid {

class PowerGridGame;

class PowerGridState : public State {
 public:
  PowerGridState(std::shared_ptr<const Game> game,
                 std::shared_ptr<const ::powergrid::Rules> rules);
  PowerGridState(const PowerGridState&) = default;

  Player CurrentPlayer() const override { return engine_.CurrentPlayer(); }
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override { return engine_.Describe(); }
  bool IsTerminal() const override { return engine_.IsTerminal(); }
  std::vector<double> Returns() const override { return engine_.Returns(); }
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player, absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override {
    return std::unique_ptr<State>(new PowerGridState(*this));
  }
  ActionsAndProbs ChanceOutcomes() const override;

  // Game-specific access for analysis code (event log, money, ...).
  const ::powergrid::State& engine() const { return engine_; }

 protected:
  void DoApplyAction(Action action_id) override { engine_.ApplyAction(action_id); }

 private:
  ::powergrid::State engine_;
};

class PowerGridGame : public Game {
 public:
  explicit PowerGridGame(const GameParameters& params);

  int NumDistinctActions() const override { return codec_.num_actions; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new PowerGridState(shared_from_this(), rules_));
  }
  int MaxChanceOutcomes() const override;
  int NumPlayers() const override { return rules_->num_players; }
  double MinUtility() const override { return 0; }
  double MaxUtility() const override { return 1; }
  absl::optional<double> UtilitySum() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {::powergrid::State::ObservationSize(*rules_)};
  }
  int MaxGameLength() const override;
  int MaxChanceNodesInHistory() const override {
    return 2 + static_cast<int>(rules_->plants.size());  // regions, order, plant draws
  }

  const ::powergrid::Rules& rules() const { return *rules_; }

 private:
  std::shared_ptr<const ::powergrid::Rules> rules_;
  ::powergrid::Codec codec_;
};

}  // namespace powergrid
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_POWERGRID_H_
