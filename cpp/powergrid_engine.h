// powergrid_engine.h -- C++ port of powergrid_core.py (the reference spec).
//
// Dependency-free C++17. Behaviour must match the Python engine action for
// action; tools/difftest.py checks this by replaying random games through both
// and comparing State::Dump() after every move. When a rule changes, change
// the Python spec first, then port it here.
//
// Layout for fast cloning: the dynamic state is a flat struct of fixed-size
// arrays, and every set of plants (market, stack candidates, removed, ...) and
// each player's set of cities is a 64-bit mask. Copying a State is a memcpy
// plus the event log, which is empty unless Rules::keep_log is set.

#ifndef POWERGRID_ENGINE_H_
#define POWERGRID_ENGINE_H_

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace powergrid {

constexpr int kChancePlayer = -1;    // matches open_spiel::kChancePlayerId
constexpr int kTerminalPlayer = -4;  // matches open_spiel::kTerminalPlayerId

constexpr int kMaxPlayers = 6;
constexpr int kMaxCities = 64;       // cities are bits of a uint64_t
constexpr int kMaxPlantNumber = 63;  // plants are bits of a uint64_t
constexpr int kMaxPlantSlots = 5;    // plant_limit + 1 (while discarding)
constexpr int kMaxOccupants = 3;
constexpr int kMarketSlots = 8;      // Codec::MAX_SLOTS in Python
constexpr int kMaxPlug = 15;         // plants 03-15 have a plug on the back
constexpr int kNuclearPhaseOutPlant = 39;
constexpr int kTrustHouses = 16;
constexpr int kTrustStartHouses = 6;
constexpr int kTrustPlantLimit = 3;

enum Fuel : int { kCoal = 0, kOil = 1, kGarbage = 2, kUranium = 3 };
constexpr int kNumFuels = 4;
constexpr int kNoFuel = 4;  // RUN option for plants that burn nothing
extern const char* const kFuelNames[kNumFuels];

// Order matches KIND_IDX in Python (used in the observation tensor).
enum Kind : int { kKindCoal, kKindOil, kKindGarbage, kKindUranium, kKindHybrid, kKindEco };
constexpr int kNumKinds = 6;

struct Plant {
  int number;
  Kind kind;
  int need;   // fuel units burned per run
  int power;  // cities powered per run
};

// The 42 power plants (generated from the Python spec, see powergrid_data.inc).
const std::vector<Plant>& Plants();
constexpr std::array<int, 4> kMarketSize = {0, 8, 8, 6};  // indexed by step

enum class Phase : int8_t {
  kChance = 0,
  kTrustSetup = 1,  // 2 players: place the Trust's six starting houses
  kAuctionSelect = 2,
  kAuctionBid = 3,
  kAuctionDiscard = 4,
  kFuelDiscard = 5,  // after a discard: choose coal or oil to return (hybrid overflow)
  kBuyFuel = 6,
  kBuild = 7,
  kBureaucracy = 8,
  kGameOver = 9,
  // transient markers: never a resting phase
  kFuelStart = 10,
  kRoundStart = 11,
  kBureauStart = 12,
};
constexpr int kNumRestingPhases = 10;
constexpr int kUnreachable = 1000000000;

// City graph with regions. Connection costs depend on the regions in play
// (no paths through cities out of play), so Rules precomputes one distance
// matrix per possible set of regions.
struct MapSpec {
  std::string name;
  std::vector<std::string> city_names;
  std::vector<int> city_region;
  std::vector<std::string> region_names;
  std::vector<std::array<int, 3>> edges;  // (city, city, cost)
  bool nuclear_phase_out = false;         // Germany: no uranium after #39 is bought
  int num_cities = 0;
  int num_regions = 0;
  std::vector<uint64_t> neighbors;        // per city: cities one connection away

  MapSpec() = default;
  MapSpec(std::string name, std::vector<std::string> city_names, std::vector<int> city_region,
          std::vector<std::string> region_names, std::vector<std::array<int, 3>> edges,
          bool nuclear_phase_out = false);
  // All sets of k regions forming one adjacent block, in lexicographic order.
  std::vector<std::vector<int>> ValidRegionSets(int k) const;
  uint64_t CityMask(const std::vector<int>& regions) const;
  // Cheapest connection costs using only cities in `regions` (Floyd-Warshall).
  std::vector<int> Distances(const std::vector<int>& regions) const;
};
MapSpec GermanyMap();      // the Recharged board (default)
MapSpec Germany2004Map();  // the 2004 board
MapSpec TinyMap();         // 12 test cities: three regions of four on a ring
// "germany", "germany-2004" or "tiny"; throws std::invalid_argument otherwise.
MapSpec MapByName(const std::string& name);

struct Rules {
  int num_players = 4;
  MapSpec map = GermanyMap();
  std::vector<Plant> plants = Plants();
  int start_money = 50;
  // -1 = take the value for this player count from the player-count table.
  int play_regions = -1;
  int plug_removed = -1;
  int socket_removed = -1;
  int plant_limit = -1;
  int step2_cities = -1;
  int end_cities = -1;
  int trust = -1;            // 2-player Trust: -1 = on for 2 players, 0 off, 1 on
  std::vector<int> regions;  // fixed regions in play; empty = chance
  int max_rounds = 100;      // safety cap (not a rule) so random play terminates
  int max_bid = 400;
  int houses = -1;
  std::array<int, 3> slot_costs = {10, 15, 20};
  std::vector<int> income;                    // empty = default table
  std::array<std::array<int, 4>, 3> resupply{};  // filled by Finalize()
  std::array<int, kNumFuels> fuel_totals = {24, 24, 24, 12};
  std::array<int, kNumFuels> fuel_init = {24, 18, 9, 2};
  bool keep_log = true;

  // Derived by Finalize().
  std::array<int, kMaxPlantNumber + 1> plant_index{};  // number -> index in plants, or -1
  uint64_t plug_mask = 0, socket_mask = 0;
  std::vector<std::vector<int>> region_sets;  // chance outcomes for the regions in play
  std::vector<uint64_t> region_city_mask;     // per region set: cities in play
  std::vector<std::vector<int>> region_dist;  // per region set: num_cities^2 distances
  int fixed_region_set = -1;                  // index into region_sets, or -1

  // Fills defaults for this player count and validates. Throws std::invalid_argument.
  void Finalize();
  bool has_trust() const { return trust == 1; }
  const Plant& plant(int number) const { return plants[plant_index[number]]; }
};

// Small config for tests: the tiny map ends quickly. Mirrors tiny_ruleset().
Rules TinyRules(int num_players, int play_regions = 3);

// Flat integer action space; see Codec in powergrid_core.py.
struct Codec {
  int kPass = 0, kDone = 1, kSelect0 = 2;
  int kBid0, kBuy0, kBuild0, kDiscard0, kRun0, num_actions;
  explicit Codec(const Rules& rules);
  std::string Describe(int a) const;
};

enum class EventType : int8_t {
  kDraw, kStep2, kStep3Card, kStep3, kPassRound, kSelect, kDrop, kBid, kSale,
  kDiscard, kReorder, kBuyFuel, kBuild, kIncome, kRun, kGameOver, kRegions,
  kReturnFuel, kDiscount, kDiscountLost, kDiscountScrapped, kUraniumStopped,
  kTrustPlace, kTrustTake, kTrustScrap, kTrustFuel, kTrustBlock,
};

// One entry of the event log (the input to collusion metrics). Field meaning
// depends on the type; ToString() gives the same key=value form as Python.
struct Event {
  EventType type;
  int8_t step;
  int16_t round;
  std::array<int32_t, 5> v{};
  std::vector<int> list;   // ring / bidders / order / occupants / burned / winners / regions
  std::vector<int> list2;  // game_over: flattened (powered, money) per player
  std::string ToString() const;
};

class State {
 public:
  explicit State(std::shared_ptr<const Rules> rules);

  const Rules& rules() const { return *rules_; }
  const Codec& codec() const { return codec_; }
  int num_players() const { return n_; }

  int CurrentPlayer() const;
  bool IsTerminal() const { return d_.phase == Phase::kGameOver; }
  bool IsChanceNode() const { return d_.phase == Phase::kChance; }
  // True at the turn-order chance node.
  bool IsInitialOrderChance() const {
    return IsChanceNode() && d_.chance_kind == ChanceKind::kInitialOrder;
  }
  // True at the opening chance node that picks the regions in play.
  bool IsRegionChance() const {
    return IsChanceNode() && d_.chance_kind == ChanceKind::kRegions;
  }
  // True while the opening market is being dealt.
  bool IsSetupChance() const { return IsChanceNode() && d_.chance_kind == ChanceKind::kSetup; }
  // Index into rules().region_sets, or -1 before the regions are chosen.
  int region_set() const { return d_.region_set; }
  std::vector<double> Returns() const;
  std::vector<std::pair<int, double>> ChanceOutcomes() const;
  std::vector<int> LegalActions() const;
  void ApplyAction(int a);

  std::string ActionToString(int a) const;
  std::string Describe() const;       // plain-text view, also usable as an LLM observation
  std::string Dump() const;           // canonical form compared by tools/difftest.py
  // Structured view of everything public (all of it: money is public) as JSON,
  // for scripted bots and as LLM input.
  std::string ToJson() const;
  void CheckInvariants() const;       // throws std::logic_error on violation

  static int ObservationSize(const Rules& rules);
  void ObservationTensor(int viewer, float* out) const;  // writes ObservationSize floats

  const std::vector<Event>& log() const { return log_; }
  int round() const { return d_.round; }
  int step() const { return d_.step; }
  Phase phase() const { return d_.phase; }
  int money(int p) const { return d_.money[p]; }
  int num_cities_of(int p) const;
  std::vector<int> purchasable() const;
  int MinBid(int plant) const;  // 1 for the discounted plant
  int Price(int f) const;       // next unit of fuel f, -1 when sold out
  int BuildCost(int p, int city) const;  // -1 when p may not build there now
  // Cheapest connection cost between two cities through the cities in play
  // (kUnreachable if either is out of play or regions are not chosen yet).
  int Distance(int a, int b) const;

  // Number of initial-order chance outcomes (n!).
  static int NumOrders(int n);

 private:
  enum class ChanceKind : int8_t { kRegions, kInitialOrder, kSetup, kPlant };
  enum Step3Starts : int8_t { kStep3None = 0, kStep3Now = 1, kStep3NextRound = 2 };

  struct Auction {
    int16_t bid;
    int8_t plant, selector, high, pos, ring_len;
    std::array<int8_t, kMaxPlayers> ring;
  };

  // Everything that changes during play; trivially copyable.
  struct Dyn {
    int16_t round;
    int8_t step;
    Phase phase;
    ChanceKind chance_kind;
    Phase after_draw;
    bool bought_any, auction_active, has_result;
    bool top_pending;        // the set-aside plug is on top of the stack
    bool step3_pending;      // step-3 card drawn in phase 2 sits in the market
    bool step3_removal_due;  // remove the card and the lowest plant when full
    bool step3_next_round;   // step 3 begins at the next round
    bool uranium_stopped;    // German nuclear power phase-out
    bool trust_due, trust_took;
    int8_t step3_starts;     // Step3Starts for the pending removal
    int8_t setup_draws;      // market plants still to deal at setup
    int8_t real_plug, real_socket;  // real cards among the stack candidates
    int8_t discount;         // plant under the discount token, 0 = none
    int8_t discarder;        // -1 = none
    int8_t region_set;       // index into rules.region_sets, -1 = not chosen yet
    int8_t market_cap;       // "do not draw replacements" limit while refilling, -1 = none
    int8_t qpos, qlen;
    int8_t trust_houses;     // Trust houses left in its supply
    int8_t trust_setup_len;  // Trust starting houses still to place
    uint8_t done_auction;    // player bitmask
    uint64_t market, deck_plug, deck_socket, bottom, removed, trust_plants;
    std::array<int8_t, kTrustStartHouses> trust_setup;  // who places them, in order
    std::array<int8_t, kMaxPlayers> order;
    std::array<int8_t, kMaxPlayers> queue;
    std::array<int32_t, kMaxPlayers> money;
    std::array<std::array<int8_t, kMaxPlantSlots>, kMaxPlayers> plants;
    std::array<int8_t, kMaxPlayers> num_plants;
    std::array<std::array<int16_t, kNumFuels>, kMaxPlayers> stored;
    std::array<int16_t, kNumFuels> trust_stored;
    std::array<uint64_t, kMaxPlayers> cities;
    std::array<std::array<int8_t, kMaxOccupants>, kMaxCities> occupants;  // Trust = n
    std::array<int8_t, kMaxCities> num_occupants;
    std::array<int16_t, kNumFuels> fuel_market;
    std::array<uint8_t, kMaxPlayers> ran;  // bitmask of plant slots run this bureaucracy
    std::array<int16_t, kMaxPlayers> powered;
    std::array<double, kMaxPlayers> result;
    Auction auction;
  };

  // helpers
  const Plant& plant(int number) const { return rules_->plant(number); }
  int trust_id() const { return n_; }
  bool Occupies(int c, int who) const;
  int MaxCities() const;
  int Starter() const;
  void Capacity(int p, std::array<int, kNumFuels>* cap, int* hybrid) const;
  bool Fits(int p, const std::array<int16_t, kNumFuels>& stored) const;
  std::vector<int> TrimOptions(int p) const;
  void ReturnFuel(int p, int f, bool forced);
  void ContinueTrim();
  std::vector<int> TrustSetupOptions() const;
  bool CardsLeft() const;
  int MarketTarget() const;
  int MaxSupply(int p) const;
  std::vector<int> RankedOrder() const;
  void SetOrder(const std::vector<int>& order);

  // phase machinery (same names as the Python spec, minus underscores)
  void AutoAdvance();
  void Dispatch(int a, bool forced);
  void DoChance(int a);
  void TakeFromStack(int a);
  void FinishSetup();
  void Refill(Phase after);
  void Step3Card(Phase after);
  void ScheduleStep3Removal();
  void Step3Removal();
  void StartStep2();
  void Settle();
  void DoTrustSetup(int a);
  void BeginAuctionPhase();
  void AdvanceAuction();
  void TrustTake();
  void EndAuction();
  void DoSelect(int a, bool forced);
  void DoBid(int a, bool forced);
  void Sell();
  void DoDiscard(int a);
  void StartFuel();
  void DoBuy(int a);
  void TrustBuy();
  void StartBuild();
  void DoBuild(int a);
  void EndBuild();
  void StartBureaucracy();
  void DoRun(int a);
  void FinishRound();
  void StartRound();
  void EndGame();

  Event& Log(EventType type);  // valid only when keep_log

  std::shared_ptr<const Rules> rules_;
  Codec codec_;  // small and trivially copyable
  int n_;
  Dyn d_;
  std::vector<Event> log_;
};

}  // namespace powergrid

#endif  // POWERGRID_ENGINE_H_
