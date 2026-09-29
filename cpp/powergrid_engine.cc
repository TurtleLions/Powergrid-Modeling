// powergrid_engine.cc -- see powergrid_engine.h. Mirrors powergrid_core.py;
// functions keep the Python names so the two can be read side by side.

#include "powergrid_engine.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace powergrid {

const char* const kFuelNames[kNumFuels] = {"COAL", "OIL", "GARBAGE", "URANIUM"};

namespace {

const char* const kKindNames[kNumKinds] = {"coal", "oil", "garbage", "uranium", "hybrid", "eco"};

// Price of the k-th cheapest space on each fuel track (cheapest first). Units
// always sit on the most expensive spaces, so with `count` units left the next
// unit costs prices[len - count].
constexpr std::array<int, 12> kUraniumPrices = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16};
constexpr int kTrackLen[kNumFuels] = {24, 24, 24, 12};

int TrackPrice(int f, int k) { return f == kUranium ? kUraniumPrices[k] : 1 + k / 3; }

inline int Popcount(uint64_t x) { return __builtin_popcountll(x); }
inline int LowestBit(uint64_t x) { return __builtin_ctzll(x); }
inline int HighestBit(uint64_t x) { return 63 - __builtin_clzll(x); }
inline uint64_t Bit(int i) { return uint64_t{1} << i; }

int PopLowest(uint64_t* mask) {
  int b = LowestBit(*mask);
  *mask &= *mask - 1;
  return b;
}

int PopHighest(uint64_t* mask) {
  int b = HighestBit(*mask);
  *mask &= ~Bit(b);
  return b;
}

// k-th lowest set bit (0-based).
int NthBit(uint64_t mask, int k) {
  for (int i = 0; i < k; ++i) mask &= mask - 1;
  return LowestBit(mask);
}

std::vector<int> Bits(uint64_t mask) {
  std::vector<int> out;
  while (mask) out.push_back(PopLowest(&mask));
  return out;
}

void Require(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(what);
}

void Invariant(bool ok, const std::string& what) {
  if (!ok) throw std::logic_error("invariant violated: " + what);
}

std::string ListStr(const std::vector<int>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) s += ",";
    s += std::to_string(v[i]);
  }
  return s + "]";
}

template <typename It>
std::string RangeStr(It begin, It end) {
  return ListStr(std::vector<int>(begin, end));
}

}  // namespace

// ---------------------------------------------------------------------------
// Static data
// ---------------------------------------------------------------------------
#include "powergrid_data.inc"

MapSpec::MapSpec(std::string name_, std::vector<std::string> city_names_,
                 std::vector<int> city_region_, std::vector<std::string> region_names_,
                 std::vector<std::array<int, 3>> edges_, bool nuclear_phase_out_)
    : name(std::move(name_)),
      city_names(std::move(city_names_)),
      city_region(std::move(city_region_)),
      region_names(std::move(region_names_)),
      edges(std::move(edges_)),
      nuclear_phase_out(nuclear_phase_out_),
      num_cities(static_cast<int>(city_names.size())),
      num_regions(static_cast<int>(region_names.size())) {
  Require(num_cities > 0 && num_cities <= kMaxCities, "num_cities must be in 1..64");
  Require(city_region.size() == city_names.size(), "every city needs a region");
  neighbors.assign(num_cities, 0);
  for (const auto& e : edges) {
    neighbors[e[0]] |= Bit(e[1]);
    neighbors[e[1]] |= Bit(e[0]);
  }
}

std::vector<std::vector<int>> MapSpec::ValidRegionSets(int k) const {
  std::vector<std::array<bool, 16>> adj(num_regions);
  for (auto& row : adj) row.fill(false);
  for (const auto& e : edges) {
    int ra = city_region[e[0]], rb = city_region[e[1]];
    if (ra != rb) adj[ra][rb] = adj[rb][ra] = true;
  }
  std::vector<std::vector<int>> out;
  // combinations in lexicographic order (itertools.combinations)
  std::vector<int> combo(k);
  std::iota(combo.begin(), combo.end(), 0);
  while (k > 0 && k <= num_regions) {
    std::vector<int> seen = {combo[0]}, todo = {combo[0]};
    while (!todo.empty()) {
      int r = todo.back();
      todo.pop_back();
      for (int s : combo)
        if (std::find(seen.begin(), seen.end(), s) == seen.end() && adj[r][s]) {
          seen.push_back(s);
          todo.push_back(s);
        }
    }
    if (static_cast<int>(seen.size()) == k) out.push_back(combo);
    int i = k - 1;
    while (i >= 0 && combo[i] == num_regions - k + i) --i;
    if (i < 0) break;
    ++combo[i];
    for (int j = i + 1; j < k; ++j) combo[j] = combo[j - 1] + 1;
  }
  return out;
}

uint64_t MapSpec::CityMask(const std::vector<int>& regions) const {
  uint64_t mask = 0;
  for (int c = 0; c < num_cities; ++c)
    if (std::find(regions.begin(), regions.end(), city_region[c]) != regions.end())
      mask |= Bit(c);
  return mask;
}

std::vector<int> MapSpec::Distances(const std::vector<int>& regions) const {
  const int n = num_cities;
  const uint64_t active = CityMask(regions);
  std::vector<int> d(n * n, kUnreachable);
  for (int i = 0; i < n; ++i)
    if (active & Bit(i)) d[i * n + i] = 0;
  for (const auto& e : edges) {
    int a = e[0], b = e[1], w = e[2];
    if (!(active & Bit(a)) || !(active & Bit(b))) continue;
    d[a * n + b] = std::min(d[a * n + b], w);
    d[b * n + a] = std::min(d[b * n + a], w);
  }
  for (int k = 0; k < n; ++k)
    for (int i = 0; i < n; ++i) {
      int dik = d[i * n + k];
      if (dik == kUnreachable) continue;
      for (int j = 0; j < n; ++j) d[i * n + j] = std::min(d[i * n + j], dik + d[k * n + j]);
    }
  return d;
}

MapSpec TinyMap() {
  const int n = 12;
  std::vector<std::string> names;
  std::vector<int> regions;
  for (int i = 0; i < n; ++i) {
    names.push_back("T" + std::to_string(i));
    regions.push_back(i / 4);
  }
  std::vector<std::array<int, 3>> edges;
  for (int i = 0; i < n; ++i) edges.push_back({i, (i + 1) % n, 3 + (i % 4)});
  edges.push_back({0, 6, 8});
  edges.push_back({3, 9, 9});
  edges.push_back({2, 8, 7});
  return MapSpec("tiny", names, regions, {"a", "b", "c"}, edges);
}

MapSpec MapByName(const std::string& name) {
  if (name == "germany") return GermanyMap();
  if (name == "germany-2004") return Germany2004Map();
  if (name == "tiny") return TinyMap();
  throw std::invalid_argument("unknown map: " + name);
}

void Rules::Finalize() {
  Require(num_players >= 2 && num_players <= kMaxPlayers, "num_players must be in 2..6");
  const auto& pc = kPlayerCountRules[num_players];
  if (play_regions < 0) play_regions = std::min(pc[0], map.num_regions);
  if (plug_removed < 0) plug_removed = pc[1];
  if (socket_removed < 0) socket_removed = pc[2];
  if (plant_limit < 0) plant_limit = pc[3];
  if (step2_cities < 0) step2_cities = pc[4];
  if (end_cities < 0) end_cities = pc[5];
  if (trust < 0) trust = num_players == 2 ? 1 : 0;
  Require(trust == 0 || num_players == 2, "the Trust is a 2-player rule");
  if (houses < 0) houses = kHousesPerPlayer;
  Require(plant_limit + 1 <= kMaxPlantSlots, "plant_limit too large");
  if (income.empty()) income = kIncome;
  resupply = kResupply[num_players];
  Require(max_bid > 0 && max_bid < 32000, "max_bid out of range");
  Require(map.num_cities > 0, "map has no cities");
  Require(map.num_regions <= 16, "at most 16 regions");
  plant_index.fill(-1);
  plug_mask = socket_mask = 0;
  for (size_t i = 0; i < plants.size(); ++i) {
    Require(plants[i].number > 0 && plants[i].number <= kMaxPlantNumber,
            "plant numbers must be in 1..63");
    Require(plants[i].kind != kKindHybrid || plants[i].need <= kNoFuel,
            "hybrid need must fit the RUN encoding");
    plant_index[plants[i].number] = static_cast<int>(i);
    (plants[i].number <= kMaxPlug ? plug_mask : socket_mask) |= Bit(plants[i].number);
  }
  Require(Popcount(plug_mask) >= kMarketSlots + 1 + plug_removed &&
              Popcount(socket_mask) >= socket_removed,
          "not enough plants for the setup");
  region_sets = map.ValidRegionSets(play_regions);
  Require(!region_sets.empty(), "no adjacent set of regions of the requested size");
  Require(region_sets.size() < 128, "too many region sets");
  region_city_mask.clear();
  region_dist.clear();
  for (const auto& rs : region_sets) {
    region_city_mask.push_back(map.CityMask(rs));
    region_dist.push_back(map.Distances(rs));
  }
  fixed_region_set = -1;
  if (!regions.empty()) {
    std::vector<int> sorted = regions;
    std::sort(sorted.begin(), sorted.end());
    auto it = std::find(region_sets.begin(), region_sets.end(), sorted);
    Require(it != region_sets.end(), "regions are not an adjacent set of play_regions regions");
    fixed_region_set = static_cast<int>(it - region_sets.begin());
  } else if (region_sets.size() == 1) {
    fixed_region_set = 0;
  }
}

Rules TinyRules(int num_players, int play_regions) {
  Rules r;
  r.num_players = num_players;
  r.map = TinyMap();
  r.play_regions = play_regions;
  r.step2_cities = 3;
  r.end_cities = 6;
  r.max_rounds = 40;
  r.Finalize();
  return r;
}

// ---------------------------------------------------------------------------
// Codec
// ---------------------------------------------------------------------------
Codec::Codec(const Rules& rules) {
  kBid0 = kSelect0 + kMarketSlots;
  kBuy0 = kBid0 + rules.max_bid + 1;
  kBuild0 = kBuy0 + kNumFuels;
  kDiscard0 = kBuild0 + rules.map.num_cities;
  kRun0 = kDiscard0 + (rules.plant_limit + 1);
  num_actions = kRun0 + (rules.plant_limit + 1) * (kNumFuels + 1);
}

std::string Codec::Describe(int a) const {
  if (a == kPass) return "PASS/DROP";
  if (a == kDone) return "DONE";
  if (a < kBid0) return "SELECT slot " + std::to_string(a - kSelect0);
  if (a < kBuy0) return "BID " + std::to_string(a - kBid0);
  if (a < kBuild0) return std::string("BUY ") + kFuelNames[a - kBuy0];
  if (a < kDiscard0) return "BUILD city " + std::to_string(a - kBuild0);
  if (a < kRun0) return "DISCARD plant slot " + std::to_string(a - kDiscard0);
  int j = (a - kRun0) / (kNumFuels + 1), f = (a - kRun0) % (kNumFuels + 1);
  return "RUN plant slot " + std::to_string(j) + " option " + std::to_string(f) +
         " (fuel code; coal count for hybrids)";
}

// ---------------------------------------------------------------------------
// Event log
// ---------------------------------------------------------------------------
std::string Event::ToString() const {
  std::string s;
  auto kv = [&s](const char* k, const std::string& val) {
    s += " ";
    s += k;
    s += "=";
    s += val;
  };
  auto i = [](int x) { return std::to_string(x); };
  static const char* const kNames[] = {
      "draw",      "step2",       "step3_card",    "step3",         "pass_round",
      "select",    "drop",        "bid",           "sale",          "discard",
      "reorder",   "buy_fuel",    "build",         "income",        "run",
      "game_over", "regions",     "return_fuel",   "discount",      "discount_lost",
      "discount_scrapped",        "uranium_stopped", "trust_place", "trust_take",
      "trust_scrap", "trust_fuel", "trust_block"};
  s = std::string("type=") + kNames[static_cast<int>(type)];
  kv("round", i(round));
  kv("step", i(step));
  switch (type) {
    case EventType::kDraw:
    case EventType::kDiscount:
    case EventType::kDiscountLost:
    case EventType::kDiscountScrapped:
    case EventType::kTrustTake:
    case EventType::kTrustScrap:
      kv("plant", i(v[0]));
      break;
    case EventType::kStep2:
    case EventType::kStep3Card:
    case EventType::kStep3:
    case EventType::kUraniumStopped:
      break;
    case EventType::kPassRound:
      kv("player", i(v[0]));
      kv("forced", i(v[1]));
      break;
    case EventType::kSelect:
      kv("player", i(v[0]));
      kv("plant", i(v[1]));
      kv("ring", ListStr(list));
      break;
    case EventType::kDrop:
      kv("player", i(v[0]));
      kv("plant", i(v[1]));
      kv("at_bid", i(v[2]));
      kv("forced", i(v[3]));
      break;
    case EventType::kBid:
      kv("player", i(v[0]));
      kv("plant", i(v[1]));
      kv("amount", i(v[2]));
      break;
    case EventType::kSale:
      kv("buyer", i(v[0]));
      kv("plant", i(v[1]));
      kv("price", i(v[2]));
      kv("selector", i(v[3]));
      kv("bidders", ListStr(list));
      break;
    case EventType::kDiscard:
      kv("player", i(v[0]));
      kv("plant", i(v[1]));
      break;
    case EventType::kReorder:
      kv("order", ListStr(list));
      break;
    case EventType::kBuyFuel:
      kv("player", i(v[0]));
      kv("fuel", kFuelNames[v[1]]);
      kv("price", i(v[2]));
      break;
    case EventType::kBuild:
      kv("player", i(v[0]));
      kv("city", i(v[1]));
      kv("cost", i(v[2]));
      kv("occupants", ListStr(list));
      break;
    case EventType::kIncome:
      kv("player", i(v[0]));
      kv("powered", i(v[1]));
      kv("income", i(v[2]));
      break;
    case EventType::kRun:
      kv("player", i(v[0]));
      kv("plant", i(v[1]));
      kv("burned", ListStr(list));
      break;
    case EventType::kRegions:
      kv("regions", ListStr(list));
      break;
    case EventType::kReturnFuel:
      kv("player", i(v[0]));
      kv("fuel", kFuelNames[v[1]]);
      kv("forced", i(v[2]));
      break;
    case EventType::kTrustPlace:
      kv("player", i(v[0]));
      kv("city", i(v[1]));
      break;
    case EventType::kTrustFuel:
      kv("plant", i(v[0]));
      kv("fuel", kFuelNames[v[1]]);
      break;
    case EventType::kTrustBlock:
      kv("city", i(v[0]));
      break;
    case EventType::kGameOver: {
      kv("winners", ListStr(list));
      std::string sc = "[";
      for (size_t k = 0; k < list2.size(); k += 2) {
        if (k) sc += ",";
        sc += ListStr({list2[k], list2[k + 1]});
      }
      kv("score", sc + "]");
      break;
    }
  }
  return s;
}

Event& State::Log(EventType type) {
  log_.push_back(Event{type, d_.step, d_.round, {}, {}, {}});
  return log_.back();
}

// Logging helper: evaluates its arguments only when the log is on.
#define PG_LOG(type, ...)                  \
  do {                                     \
    if (rules_->keep_log) {                \
      Event& e_ = Log(EventType::type);    \
      (void)e_;                            \
      __VA_ARGS__;                         \
    }                                      \
  } while (0)

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
int State::NumOrders(int n) {
  int k = 1;
  for (int i = 2; i <= n; ++i) k *= i;
  return k;
}

State::State(std::shared_ptr<const Rules> rules)
    : rules_(std::move(rules)), codec_(*rules_), n_(rules_->num_players), d_{} {
  const Rules& r = *rules_;
  d_.round = 1;
  d_.step = 1;
  d_.phase = Phase::kChance;
  // the playing zone is chosen, then player order is drawn, then the market dealt
  d_.region_set = static_cast<int8_t>(r.fixed_region_set);
  d_.chance_kind = d_.region_set >= 0 ? ChanceKind::kInitialOrder : ChanceKind::kRegions;
  d_.after_draw = Phase::kAuctionSelect;
  d_.discarder = -1;
  d_.market_cap = -1;
  d_.setup_draws = kMarketSize[1];
  d_.deck_plug = r.plug_mask;
  d_.deck_socket = r.socket_mask;
  d_.trust_houses = r.has_trust() ? kTrustHouses - kTrustStartHouses : 0;
  for (int p = 0; p < n_; ++p) {
    d_.order[p] = p;
    d_.money[p] = r.start_money;
  }
  for (int f = 0; f < kNumFuels; ++f) d_.fuel_market[f] = r.fuel_init[f];
}

int State::num_cities_of(int p) const { return Popcount(d_.cities[p]); }

int State::MaxCities() const {
  int m = 0;
  for (int p = 0; p < n_; ++p) m = std::max(m, Popcount(d_.cities[p]));
  return m;
}

bool State::Occupies(int c, int who) const {
  for (int i = 0; i < d_.num_occupants[c]; ++i)
    if (d_.occupants[c][i] == who) return true;
  return false;
}

std::vector<double> State::Returns() const {
  std::vector<double> out(n_, 0.0);
  if (d_.has_result)
    for (int p = 0; p < n_; ++p) out[p] = d_.result[p];
  return out;
}

std::vector<int> State::purchasable() const {
  std::vector<int> out;
  uint64_t m = d_.market;
  int k = d_.step < 3 ? 4 : 6;
  while (m && static_cast<int>(out.size()) < k) out.push_back(PopLowest(&m));
  return out;
}

int State::MinBid(int pl) const { return pl == d_.discount ? 1 : pl; }

int State::Starter() const {
  for (int i = 0; i < n_; ++i) {
    int p = d_.order[i];
    if (!(d_.done_auction & (1u << p))) return p;
  }
  return -1;
}

int State::CurrentPlayer() const {
  switch (d_.phase) {
    case Phase::kGameOver: return kTerminalPlayer;
    case Phase::kChance: return kChancePlayer;
    case Phase::kTrustSetup: return d_.trust_setup[kTrustStartHouses - d_.trust_setup_len];
    case Phase::kAuctionSelect: return Starter();
    case Phase::kAuctionBid: return d_.auction.ring[d_.auction.pos];
    case Phase::kAuctionDiscard:
    case Phase::kFuelDiscard: return d_.discarder;
    default: return d_.queue[d_.qpos];
  }
}

// ---- fuel helpers ---------------------------------------------------------
int State::Price(int f) const {
  int c = d_.fuel_market[f];
  if (c <= 0) return -1;
  return TrackPrice(f, kTrackLen[f] - c);
}

void State::Capacity(int p, std::array<int, kNumFuels>* cap, int* hybrid) const {
  cap->fill(0);
  *hybrid = 0;
  for (int j = 0; j < d_.num_plants[p]; ++j) {
    const Plant& pl = plant(d_.plants[p][j]);
    if (pl.kind == kKindHybrid) *hybrid += 2 * pl.need;
    else if (pl.kind != kKindEco) (*cap)[pl.kind] += 2 * pl.need;  // kind index == fuel index
  }
}

bool State::Fits(int p, const std::array<int16_t, kNumFuels>& s) const {
  std::array<int, kNumFuels> cap;
  int hybrid;
  Capacity(p, &cap, &hybrid);
  int over = std::max(0, s[kCoal] - cap[kCoal]) + std::max(0, s[kOil] - cap[kOil]);
  return over <= hybrid && s[kGarbage] <= cap[kGarbage] && s[kUranium] <= cap[kUranium];
}

std::vector<int> State::TrimOptions(int p) const {
  // Fuels p may return to get back within storage after a discard. Only coal vs
  // oil competing for hybrid space is a real choice.
  std::array<int, kNumFuels> cap;
  int hybrid;
  Capacity(p, &cap, &hybrid);
  const auto& s = d_.stored[p];
  int oc = std::max(0, s[kCoal] - cap[kCoal]);
  int oo = std::max(0, s[kOil] - cap[kOil]);
  std::vector<int> out;
  if (oc + oo <= hybrid) return out;
  if (oc > 0) out.push_back(kCoal);
  if (oo > 0) out.push_back(kOil);
  return out;
}

void State::ReturnFuel(int p, int f, bool forced) {
  --d_.stored[p][f];
  PG_LOG(kReturnFuel, e_.v[0] = p; e_.v[1] = f; e_.v[2] = forced);
}

void State::ContinueTrim() {
  // Return excess fuel after a discard; stop when the player must choose.
  int p = d_.discarder;
  std::array<int, kNumFuels> cap;
  int hybrid;
  Capacity(p, &cap, &hybrid);
  for (int f : {int(kGarbage), int(kUranium)})
    while (d_.stored[p][f] > cap[f]) ReturnFuel(p, f, /*forced=*/true);
  while (true) {
    std::vector<int> opts = TrimOptions(p);
    if (opts.empty()) break;
    if (opts.size() > 1) {
      d_.phase = Phase::kFuelDiscard;
      return;
    }
    ReturnFuel(p, opts[0], /*forced=*/true);
  }
  d_.discarder = -1;
  Refill(Phase::kAuctionSelect);
}

int State::BuildCost(int p, int c) const {
  // price of the lowest free space plus the cheapest connection (through any
  // cities in play) from one of the player's cities
  if (!(rules_->region_city_mask[d_.region_set] & Bit(c))) return -1;
  if (Popcount(d_.cities[p]) >= rules_->houses) return -1;
  int nocc = d_.num_occupants[c];
  if (nocc >= d_.step) return -1;
  if (Occupies(c, p)) return -1;
  int conn = 0;
  if (d_.cities[p]) {
    conn = 1 << 30;
    uint64_t m = d_.cities[p];
    const std::vector<int>& dist = rules_->region_dist[d_.region_set];
    const int n = rules_->map.num_cities;
    while (m) conn = std::min(conn, dist[PopLowest(&m) * n + c]);
  }
  return rules_->slot_costs[nocc] + conn;
}

std::vector<int> State::TrustSetupOptions() const {
  // The first Trust house goes anywhere in play, the others next to an earlier
  // one (one connection away). Falls back to any empty city.
  const MapSpec& m = rules_->map;
  uint64_t empty = 0, placed = 0;
  for (int c = 0; c < m.num_cities; ++c) {
    if (!(rules_->region_city_mask[d_.region_set] & Bit(c))) continue;
    if (d_.num_occupants[c] == 0) empty |= Bit(c);
  }
  for (int c = 0; c < m.num_cities; ++c)
    if (Occupies(c, trust_id())) placed |= Bit(c);
  if (!placed) return Bits(empty);
  uint64_t near = 0;
  for (int c : Bits(placed)) near |= m.neighbors[c];
  return (empty & near) ? Bits(empty & near) : Bits(empty);
}

// ---- chance ---------------------------------------------------------------
bool State::CardsLeft() const { return d_.top_pending || d_.real_plug + d_.real_socket > 0; }

std::vector<std::pair<int, double>> State::ChanceOutcomes() const {
  std::vector<std::pair<int, double>> out;
  if (d_.chance_kind == ChanceKind::kRegions) {
    int k = static_cast<int>(rules_->region_sets.size());
    for (int i = 0; i < k; ++i) out.emplace_back(i, 1.0 / k);
    return out;
  }
  if (d_.chance_kind == ChanceKind::kInitialOrder) {
    int k = NumOrders(n_);
    for (int i = 0; i < k; ++i) out.emplace_back(i, 1.0 / k);
    return out;
  }
  if (d_.chance_kind == ChanceKind::kSetup || d_.top_pending) {
    // the market is dealt from the plug plants; the top card is a plug
    int k = Popcount(d_.deck_plug);
    for (int num : Bits(d_.deck_plug)) out.emplace_back(num, 1.0 / k);
    return out;
  }
  // The next card is a real plug with probability real_plug / real, and then any
  // plug candidate equally likely (by symmetry); likewise for sockets. Plugs are
  // numbered below sockets, so this list is sorted.
  const double total = d_.real_plug + d_.real_socket;
  if (d_.real_plug) {
    double w = d_.real_plug / total / Popcount(d_.deck_plug);
    for (int num : Bits(d_.deck_plug)) out.emplace_back(num, w);
  }
  if (d_.real_socket) {
    double w = d_.real_socket / total / Popcount(d_.deck_socket);
    for (int num : Bits(d_.deck_socket)) out.emplace_back(num, w);
  }
  return out;
}

// ---- legal actions --------------------------------------------------------
std::vector<int> State::LegalActions() const {
  const Codec& c = codec_;
  std::vector<int> acts;
  switch (d_.phase) {
    case Phase::kGameOver:
      return acts;
    case Phase::kChance:
      for (const auto& o : ChanceOutcomes()) acts.push_back(o.first);
      return acts;
    case Phase::kTrustSetup:
      for (int city : TrustSetupOptions()) acts.push_back(c.kBuild0 + city);
      return acts;
    default:
      break;
  }
  int p = CurrentPlayer();
  int money = d_.money[p];
  switch (d_.phase) {
    case Phase::kAuctionSelect: {
      std::vector<int> buyable = purchasable();
      for (size_t k = 0; k < buyable.size(); ++k)
        if (money >= MinBid(buyable[k])) acts.push_back(c.kSelect0 + static_cast<int>(k));
      if (d_.round > 1 || acts.empty())  // everyone must buy in round 1 (if they can)
        acts.insert(acts.begin(), c.kPass);
      return acts;
    }
    case Phase::kAuctionBid: {
      const Auction& au = d_.auction;
      int hi = std::min(money, rules_->max_bid);
      if (au.high < 0) {
        for (int x = MinBid(au.plant); x <= hi; ++x) acts.push_back(c.kBid0 + x);
        return acts;
      }
      acts.push_back(c.kPass);
      for (int x = au.bid + 1; x <= hi; ++x) acts.push_back(c.kBid0 + x);
      return acts;
    }
    case Phase::kAuctionDiscard:
      // "They may not choose to scrap the just bought power plant" (the last one)
      for (int j = 0; j + 1 < d_.num_plants[p]; ++j) acts.push_back(c.kDiscard0 + j);
      return acts;
    case Phase::kFuelDiscard:
      for (int f : TrimOptions(p)) acts.push_back(c.kBuy0 + f);
      return acts;
    case Phase::kBuyFuel: {
      acts.push_back(c.kDone);
      for (int f = 0; f < kNumFuels; ++f) {
        int price = Price(f);
        if (price < 0 || money < price) continue;
        auto st = d_.stored[p];
        ++st[f];
        if (Fits(p, st)) acts.push_back(c.kBuy0 + f);
      }
      return acts;
    }
    case Phase::kBuild: {
      acts.push_back(c.kDone);
      for (int city = 0; city < rules_->map.num_cities; ++city) {
        int cost = BuildCost(p, city);
        if (cost >= 0 && cost <= money) acts.push_back(c.kBuild0 + city);
      }
      return acts;
    }
    case Phase::kBureaucracy: {
      acts.push_back(c.kDone);
      const auto& st = d_.stored[p];
      for (int j = 0; j < d_.num_plants[p]; ++j) {
        if (d_.ran[p] & (1u << j)) continue;
        const Plant& pl = plant(d_.plants[p][j]);
        int base = c.kRun0 + j * (kNumFuels + 1);
        if (pl.kind == kKindEco) {
          acts.push_back(base + kNoFuel);
        } else if (pl.kind == kKindHybrid) {
          for (int k = 0; k <= pl.need; ++k)  // k coal + (need - k) oil
            if (st[kCoal] >= k && st[kOil] >= pl.need - k) acts.push_back(base + k);
        } else if (st[pl.kind] >= pl.need) {
          acts.push_back(base + pl.kind);
        }
      }
      return acts;
    }
    default:
      throw std::logic_error("no legal actions defined for this phase");
  }
}

// ---- applying actions -----------------------------------------------------
void State::ApplyAction(int a) {
  Dispatch(a, /*forced=*/false);
  AutoAdvance();
}

void State::AutoAdvance() {
  // Apply forced PASS/DONE moves so agents only see real decisions.
  while (d_.phase != Phase::kChance && d_.phase != Phase::kGameOver) {
    std::vector<int> acts = LegalActions();
    if (acts.size() == 1 && (acts[0] == codec_.kPass || acts[0] == codec_.kDone))
      Dispatch(acts[0], /*forced=*/true);
    else
      break;
  }
}

void State::Dispatch(int a, bool forced) {
  switch (d_.phase) {
    case Phase::kChance: DoChance(a); break;
    case Phase::kTrustSetup: DoTrustSetup(a); break;
    case Phase::kAuctionSelect: DoSelect(a, forced); break;
    case Phase::kAuctionBid: DoBid(a, forced); break;
    case Phase::kAuctionDiscard: DoDiscard(a); break;
    case Phase::kFuelDiscard:
      ReturnFuel(d_.discarder, a - codec_.kBuy0, /*forced=*/false);
      ContinueTrim();
      break;
    case Phase::kBuyFuel: DoBuy(a); break;
    case Phase::kBuild: DoBuild(a); break;
    case Phase::kBureaucracy: DoRun(a); break;
    default: throw std::logic_error("cannot apply action in this phase");
  }
}

// -- chance
void State::DoChance(int a) {
  switch (d_.chance_kind) {
    case ChanceKind::kRegions:
      d_.region_set = static_cast<int8_t>(a);
      PG_LOG(kRegions, e_.list = rules_->region_sets[a]);
      d_.chance_kind = ChanceKind::kInitialOrder;
      return;
    case ChanceKind::kInitialOrder: {
      // a-th permutation of range(n) in lexicographic order (itertools.permutations)
      std::vector<int> perm(n_);
      std::iota(perm.begin(), perm.end(), 0);
      for (int i = 0; i < a; ++i) std::next_permutation(perm.begin(), perm.end());
      SetOrder(perm);
      d_.chance_kind = ChanceKind::kSetup;
      return;
    }
    case ChanceKind::kSetup:
      d_.deck_plug &= ~Bit(a);
      d_.market |= Bit(a);
      PG_LOG(kDraw, e_.v[0] = a);
      if (--d_.setup_draws == 0) FinishSetup();
      return;
    case ChanceKind::kPlant:
      break;
  }
  TakeFromStack(a);
  d_.market |= Bit(a);
  PG_LOG(kDraw, e_.v[0] = a);
  if (d_.after_draw == Phase::kAuctionSelect && d_.discount && a < d_.discount) {
    // the first replacement smaller than the discounted plant is removed
    // together with the discount token
    d_.market &= ~Bit(a);
    d_.removed |= Bit(a);
    d_.discount = 0;
    PG_LOG(kDiscountLost, e_.v[0] = a);
  }
  Refill(d_.after_draw);
}

void State::TakeFromStack(int a) {
  if (a <= kMaxPlug) {
    d_.deck_plug &= ~Bit(a);
    if (!d_.top_pending) --d_.real_plug;
  } else {
    d_.deck_socket &= ~Bit(a);
    --d_.real_socket;
  }
  d_.top_pending = false;
}

void State::FinishSetup() {
  // One more plug is set aside for the top of the stack; then plugs and sockets
  // are removed face down and the rest shuffled together.
  const Rules& r = *rules_;
  d_.top_pending = true;
  d_.real_plug = static_cast<int8_t>(Popcount(d_.deck_plug) - 1 - r.plug_removed);
  d_.real_socket = static_cast<int8_t>(Popcount(d_.deck_socket) - r.socket_removed);
  if (r.has_trust()) {
    // starting player 1 house, other player 2, starting 2, other 1
    const int s = d_.order[0], o = d_.order[1];
    d_.trust_setup = {int8_t(s), int8_t(o), int8_t(o), int8_t(s), int8_t(s), int8_t(o)};
    d_.trust_setup_len = kTrustStartHouses;
    d_.phase = Phase::kTrustSetup;
  } else {
    BeginAuctionPhase();
  }
}

int State::MarketTarget() const {
  // a pending step-3 card occupies the last slot of the future market
  int target = kMarketSize[d_.step] - (d_.step3_pending ? 1 : 0);
  if (d_.market_cap >= 0) target = std::min<int>(target, d_.market_cap);
  // the card removed a plant "without replacements"; step 3 starts next round
  if (d_.step3_next_round) target = std::min(target, Popcount(d_.market));
  return target;
}

void State::Refill(Phase after) {
  // Draw until the market is full, then move on to `after`. Stops at a chance
  // node whenever a hidden card must be drawn; DoChance resumes here.
  d_.after_draw = after;
  while (true) {
    const bool is_short = Popcount(d_.market) < MarketTarget();
    if (is_short && CardsLeft()) {
      d_.phase = Phase::kChance;
      d_.chance_kind = ChanceKind::kPlant;
      return;
    }
    if (is_short && d_.step < 3 && !d_.step3_pending && !d_.step3_removal_due &&
        !d_.step3_next_round) {
      Step3Card(after);
      continue;
    }
    if (d_.step3_removal_due) {
      Step3Removal();
      continue;
    }
    break;  // full, or the stack is exhausted
  }
  d_.market_cap = -1;
  d_.phase = after;
  Settle();
}

void State::Step3Card(Phase after) {
  // The Step 3 card comes up: shuffle the plants that were put under the stack;
  // they are the new stack.
  PG_LOG(kStep3Card);
  d_.removed |= d_.deck_plug | d_.deck_socket;  // the face-down setup removals
  d_.deck_plug = d_.bottom & rules_->plug_mask;
  d_.deck_socket = d_.bottom & rules_->socket_mask;
  d_.real_plug = static_cast<int8_t>(Popcount(d_.deck_plug));
  d_.real_socket = static_cast<int8_t>(Popcount(d_.deck_socket));
  d_.bottom = 0;
  if (after == Phase::kAuctionSelect) {
    // case 1, phase 2: the card waits at the end of the future market and
    // replacements keep coming until the phase ends
    d_.step3_pending = true;
    return;
  }
  // case 2, phase 5 (or the end of phase 2): card and lowest plant leave, no
  // replacements
  d_.market_cap = static_cast<int8_t>(Popcount(d_.market));
  d_.step3_starts = after == Phase::kFuelStart ? kStep3Now : kStep3NextRound;
  ScheduleStep3Removal();
}

void State::ScheduleStep3Removal() {
  if (d_.step == 1) StartStep2();  // Step 3 before Step 2: Step 2 changes first
  d_.step3_removal_due = true;
}

void State::Step3Removal() {
  // Remove the Step 3 card and the lowest plant; no replacements.
  d_.step3_removal_due = false;
  d_.step3_pending = false;
  if (d_.market) d_.removed |= Bit(PopLowest(&d_.market));
  d_.market_cap = static_cast<int8_t>(Popcount(d_.market));
  if (d_.step3_starts == kStep3Now) {
    d_.step = 3;
    PG_LOG(kStep3);
  } else {
    d_.step3_next_round = true;
  }
  d_.step3_starts = kStep3None;
}

void State::StartStep2() {
  // once: remove the lowest plant and replace it
  d_.step = 2;
  if (d_.market) d_.removed |= Bit(PopLowest(&d_.market));
  PG_LOG(kStep2);
}

void State::Settle() {
  switch (d_.phase) {
    case Phase::kAuctionSelect: AdvanceAuction(); break;
    case Phase::kFuelStart: StartFuel(); break;
    case Phase::kBureauStart: StartBureaucracy(); break;
    case Phase::kRoundStart: StartRound(); break;
    default: break;
  }
}

// -- Trust setup
void State::DoTrustSetup(int a) {
  int city = a - codec_.kBuild0;
  int p = CurrentPlayer();
  --d_.trust_setup_len;
  d_.occupants[city][d_.num_occupants[city]++] = static_cast<int8_t>(trust_id());
  PG_LOG(kTrustPlace, e_.v[0] = p; e_.v[1] = city);
  if (d_.trust_setup_len == 0) BeginAuctionPhase();
}

// -- auction
void State::BeginAuctionPhase() {
  d_.done_auction = 0;
  d_.bought_any = false;
  d_.auction_active = false;
  d_.trust_due = false;
  d_.trust_took = false;
  // the discount token goes on the smallest plant in the current market
  d_.discount = d_.market ? static_cast<int8_t>(LowestBit(d_.market)) : 0;
  if (d_.discount) PG_LOG(kDiscount, e_.v[0] = d_.discount);
  d_.phase = Phase::kAuctionSelect;
}

void State::AdvanceAuction() {
  d_.phase = Phase::kAuctionSelect;
  if (d_.trust_due) {
    TrustTake();
    return;
  }
  if (Starter() < 0) EndAuction();
}

void State::TrustTake() {
  // The Trust takes the biggest plant in the current market, for free; with 3
  // plants only if it beats its smallest, which it scraps.
  d_.trust_due = false;
  d_.trust_took = true;
  std::vector<int> current = purchasable();
  if (!current.empty()) {
    int best = current.back();
    int count = Popcount(d_.trust_plants);
    if (count < kTrustPlantLimit || best > LowestBit(d_.trust_plants)) {
      if (count >= kTrustPlantLimit) {
        int small = PopLowest(&d_.trust_plants);
        d_.removed |= Bit(small);
        PG_LOG(kTrustScrap, e_.v[0] = small);
      }
      d_.market &= ~Bit(best);
      d_.trust_plants |= Bit(best);
      if (best == d_.discount) d_.discount = 0;
      PG_LOG(kTrustTake, e_.v[0] = best);
      Refill(Phase::kAuctionSelect);
      return;
    }
  }
  AdvanceAuction();
}

void State::EndAuction() {
  if (d_.discount) {
    // nobody bought the discounted plant: it leaves and is replaced
    d_.market &= ~Bit(d_.discount);
    d_.removed |= Bit(d_.discount);
    PG_LOG(kDiscountScrapped, e_.v[0] = d_.discount);
    d_.discount = 0;
  }
  if (d_.step3_pending) {
    // after phase 2: remove the card and the lowest plant, no replacements;
    // step 3 starts in phase 3
    d_.step3_starts = kStep3Now;
    ScheduleStep3Removal();
  }
  Refill(Phase::kFuelStart);
}

void State::DoSelect(int a, bool forced) {
  int p = CurrentPlayer();
  if (a == codec_.kPass) {
    d_.done_auction |= 1u << p;
    PG_LOG(kPassRound, e_.v[0] = p; e_.v[1] = forced);
    if (rules_->has_trust() && !d_.trust_took && p == d_.order[0])
      d_.trust_due = true;  // after the first player opted out
    AdvanceAuction();
    return;
  }
  int pl = NthBit(d_.market, a - codec_.kSelect0);
  Auction& au = d_.auction;
  au = Auction{};
  au.plant = pl;
  au.selector = p;
  au.bid = 0;
  au.high = -1;
  au.pos = 0;
  au.ring_len = 0;
  // bidding goes clockwise in seat order (seat == player id), starting with the selector
  for (int i = 0; i < n_; ++i) {
    int q = (p + i) % n_;
    if (!(d_.done_auction & (1u << q))) au.ring[au.ring_len++] = q;
  }
  d_.auction_active = true;
  PG_LOG(kSelect, e_.v[0] = p; e_.v[1] = pl;
         e_.list.assign(au.ring.begin(), au.ring.begin() + au.ring_len));
  if (au.ring_len == 1) {  // the last player pays the minimum bid
    au.high = p;
    au.bid = MinBid(pl);
    Sell();
  } else {
    d_.phase = Phase::kAuctionBid;
  }
}

void State::DoBid(int a, bool forced) {
  int p = CurrentPlayer();
  Auction& au = d_.auction;
  if (a == codec_.kPass) {
    std::copy(au.ring.begin() + au.pos + 1, au.ring.begin() + au.ring_len,
              au.ring.begin() + au.pos);
    --au.ring_len;
    PG_LOG(kDrop, e_.v[0] = p; e_.v[1] = au.plant; e_.v[2] = au.bid; e_.v[3] = forced);
    if (au.pos >= au.ring_len) au.pos = 0;
    if (au.ring_len == 1) Sell();
  } else {
    au.bid = a - codec_.kBid0;
    au.high = p;
    au.pos = (au.pos + 1) % au.ring_len;
    PG_LOG(kBid, e_.v[0] = p; e_.v[1] = au.plant; e_.v[2] = au.bid);
  }
}

void State::Sell() {
  const Auction au = d_.auction;
  int buyer = au.high, price = au.bid, pl = au.plant;
  d_.money[buyer] -= price;
  d_.plants[buyer][d_.num_plants[buyer]++] = pl;
  d_.market &= ~Bit(pl);
  d_.done_auction |= 1u << buyer;
  d_.bought_any = true;
  PG_LOG(kSale, e_.v[0] = buyer; e_.v[1] = pl; e_.v[2] = price; e_.v[3] = au.selector;
         e_.list.assign(au.ring.begin(), au.ring.begin() + au.ring_len));
  d_.auction_active = false;
  if (pl == d_.discount) d_.discount = 0;
  if (pl == kNuclearPhaseOutPlant && rules_->map.nuclear_phase_out && !d_.uranium_stopped) {
    d_.uranium_stopped = true;
    PG_LOG(kUraniumStopped);
  }
  if (rules_->has_trust() && !d_.trust_took) d_.trust_due = true;  // after the first purchase
  if (d_.num_plants[buyer] > rules_->plant_limit) {
    d_.phase = Phase::kAuctionDiscard;
    d_.discarder = buyer;
  } else {
    Refill(Phase::kAuctionSelect);
  }
}

void State::DoDiscard(int a) {
  int p = d_.discarder;
  int j = a - codec_.kDiscard0;
  int num = d_.plants[p][j];
  for (int k = j; k + 1 < d_.num_plants[p]; ++k) d_.plants[p][k] = d_.plants[p][k + 1];
  --d_.num_plants[p];
  d_.removed |= Bit(num);
  PG_LOG(kDiscard, e_.v[0] = p; e_.v[1] = num);
  ContinueTrim();
}

// -- fuel
void State::StartFuel() {
  if (d_.round == 1) {
    // round 1: order is re-set once plants are bought
    std::vector<int> ord = RankedOrder();
    SetOrder(ord);
    PG_LOG(kReorder, e_.list = ord);
  }
  for (int i = 0; i < n_; ++i) d_.queue[i] = d_.order[n_ - 1 - i];
  d_.qlen = n_;
  d_.qpos = 0;
  d_.phase = Phase::kBuyFuel;
}

void State::DoBuy(int a) {
  int p = CurrentPlayer();
  if (a == codec_.kDone) {
    ++d_.qpos;
    if (rules_->has_trust() && d_.qpos == 1) TrustBuy();  // the Trust is second in order
    if (d_.qpos >= d_.qlen) StartBuild();
    return;
  }
  int f = a - codec_.kBuy0;
  int price = Price(f);
  d_.money[p] -= price;
  --d_.fuel_market[f];
  ++d_.stored[p][f];
  PG_LOG(kBuyFuel, e_.v[0] = p; e_.v[1] = f; e_.v[2] = price);
}

void State::TrustBuy() {
  // The Trust takes (for free) the fuel for one run of each plant, as much as is
  // available; hybrids alternate coal and oil, starting with coal.
  for (int num : Bits(d_.trust_plants)) {
    const Plant& pl = plant(num);
    for (int i = 0; i < pl.need; ++i) {
      int f = -1;
      if (pl.kind == kKindHybrid) {
        const int first = i % 2 == 0 ? kCoal : kOil, second = i % 2 == 0 ? kOil : kCoal;
        if (d_.fuel_market[first] > 0) f = first;
        else if (d_.fuel_market[second] > 0) f = second;
      } else if (d_.fuel_market[pl.kind] > 0) {
        f = pl.kind;
      }
      if (f < 0) break;
      --d_.fuel_market[f];
      ++d_.trust_stored[f];
      PG_LOG(kTrustFuel, e_.v[0] = num; e_.v[1] = f);
    }
  }
}

// -- build
void State::StartBuild() {
  for (int i = 0; i < n_; ++i) d_.queue[i] = d_.order[n_ - 1 - i];
  d_.qlen = n_;
  d_.qpos = 0;
  d_.phase = Phase::kBuild;
}

void State::DoBuild(int a) {
  int p = CurrentPlayer();
  if (a == codec_.kDone) {
    if (++d_.qpos >= d_.qlen) EndBuild();
    return;
  }
  int city = a - codec_.kBuild0;
  int cost = BuildCost(p, city);
  const bool was_empty = d_.num_occupants[city] == 0;
  d_.money[p] -= cost;
  d_.occupants[city][d_.num_occupants[city]++] = static_cast<int8_t>(p);
  d_.cities[p] |= Bit(city);
  PG_LOG(kBuild, e_.v[0] = p; e_.v[1] = city; e_.v[2] = cost;
         e_.list.assign(d_.occupants[city].begin(),
                        d_.occupants[city].begin() + d_.num_occupants[city]));
  if (rules_->has_trust() && was_empty && d_.trust_houses > 0) {
    // the Trust blocks the 15 space of every newly connected city
    d_.occupants[city][d_.num_occupants[city]++] = static_cast<int8_t>(trust_id());
    --d_.trust_houses;
    PG_LOG(kTrustBlock, e_.v[0] = city);
  }
}

void State::EndBuild() {
  // The game ends immediately after phase 4 once someone has the end-game
  // number of cities: no income is paid.
  if (MaxCities() >= rules_->end_cities) {
    EndGame();
    return;
  }
  // Step 2 starts at the beginning of phase 5.
  if (d_.step == 1 && MaxCities() >= rules_->step2_cities) StartStep2();
  Refill(Phase::kBureauStart);
}

// -- bureaucracy
void State::StartBureaucracy() {
  for (int i = 0; i < n_; ++i) d_.queue[i] = d_.order[i];
  d_.qlen = n_;
  d_.qpos = 0;
  d_.ran.fill(0);
  d_.powered.fill(0);
  d_.phase = Phase::kBureaucracy;
}

void State::DoRun(int a) {
  int p = CurrentPlayer();
  if (a == codec_.kDone) {
    int cap = 0;
    for (int j = 0; j < d_.num_plants[p]; ++j)
      if (d_.ran[p] & (1u << j)) cap += plant(d_.plants[p][j]).power;
    int powered = std::min(Popcount(d_.cities[p]), cap);
    const auto& income_table = rules_->income;
    int income = income_table[std::min<int>(powered, income_table.size() - 1)];
    d_.money[p] += income;
    d_.powered[p] = powered;
    PG_LOG(kIncome, e_.v[0] = p; e_.v[1] = powered; e_.v[2] = income);
    if (++d_.qpos >= d_.qlen) FinishRound();
    return;
  }
  int j = (a - codec_.kRun0) / (kNumFuels + 1), f = (a - codec_.kRun0) % (kNumFuels + 1);
  const Plant& pl = plant(d_.plants[p][j]);
  std::array<int, kNumFuels> burned{};
  if (pl.kind == kKindHybrid) {
    burned[kCoal] = f;
    burned[kOil] = pl.need - f;
  } else if (f != kNoFuel) {
    burned[f] = pl.need;
  }
  for (int g = 0; g < kNumFuels; ++g) d_.stored[p][g] -= burned[g];
  d_.ran[p] |= 1u << j;
  PG_LOG(kRun, e_.v[0] = p; e_.v[1] = pl.number; e_.list.assign(burned.begin(), burned.end()));
}

void State::FinishRound() {
  d_.trust_stored.fill(0);  // the Trust burns its fuel; it goes back to the supply
  if (d_.round >= rules_->max_rounds) {  // safety cap, not a rule
    EndGame();
    return;
  }
  const auto& add = rules_->resupply[d_.step - 1];
  for (int f = 0; f < kNumFuels; ++f) {
    if (f == kUranium && d_.uranium_stopped) continue;
    int held = 0;
    for (int p = 0; p < n_; ++p) held += d_.stored[p][f];
    int bank = rules_->fuel_totals[f] - d_.fuel_market[f] - held;
    int room = kTrackLen[f] - d_.fuel_market[f];
    d_.fuel_market[f] += std::max(0, std::min({add[f], bank, room}));
  }
  if (d_.market && !d_.step3_next_round) {
    if (d_.step < 3) d_.bottom |= Bit(PopHighest(&d_.market));  // highest plant under the step-3 card
    else d_.removed |= Bit(PopLowest(&d_.market));             // step 3: lowest plant leaves
  }
  Refill(Phase::kRoundStart);
}

void State::SetOrder(const std::vector<int>& order) {
  // Constant loop bound: GCC cannot prove n_ <= kMaxPlayers and warns on a plain copy.
  for (int i = 0; i < kMaxPlayers && i < static_cast<int>(order.size()); ++i)
    d_.order[i] = static_cast<int8_t>(order[i]);
}

std::vector<int> State::RankedOrder() const {
  std::vector<int> ord(n_);
  std::iota(ord.begin(), ord.end(), 0);
  auto biggest = [this](int p) {
    int b = 0;
    for (int j = 0; j < d_.num_plants[p]; ++j) b = std::max<int>(b, d_.plants[p][j]);
    return b;
  };
  std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) {
    int ca = Popcount(d_.cities[a]), cb = Popcount(d_.cities[b]);
    if (ca != cb) return ca > cb;
    return biggest(a) > biggest(b);
  });
  return ord;
}

void State::StartRound() {
  ++d_.round;
  if (d_.step3_next_round) {
    d_.step3_next_round = false;
    d_.step = 3;
    PG_LOG(kStep3);
  }
  SetOrder(RankedOrder());
  BeginAuctionPhase();
}

int State::MaxSupply(int p) const {
  // Most cities p can power with the plants and fuel they have.
  const auto& s = d_.stored[p];
  const int k = d_.num_plants[p];
  int best = 0;
  for (int subset = 0; subset < (1 << k); ++subset) {
    std::array<int, kNumFuels> need{};
    int hybrid = 0, power = 0;
    for (int j = 0; j < k; ++j) {
      if (!(subset & (1 << j))) continue;
      const Plant& pl = plant(d_.plants[p][j]);
      power += pl.power;
      if (pl.kind == kKindHybrid) hybrid += pl.need;
      else if (pl.kind != kKindEco) need[pl.kind] += pl.need;
    }
    bool ok = true;
    for (int f = 0; f < kNumFuels; ++f) ok = ok && need[f] <= s[f];
    if (!ok || hybrid > (s[kCoal] - need[kCoal]) + (s[kOil] - need[kOil])) continue;
    best = std::max(best, power);
  }
  return std::min(best, Popcount(d_.cities[p]));
}

void State::EndGame() {
  // most cities suppliable, then most money; remaining ties share the win
  for (int p = 0; p < n_; ++p) d_.powered[p] = static_cast<int16_t>(MaxSupply(p));
  auto score = [this](int p) { return std::array<int, 2>{d_.powered[p], d_.money[p]}; };
  std::array<int, 2> best = score(0);
  for (int p = 1; p < n_; ++p) best = std::max(best, score(p));
  std::vector<int> winners;
  for (int p = 0; p < n_; ++p)
    if (score(p) == best) winners.push_back(p);
  d_.result.fill(0.0);
  for (int p : winners) d_.result[p] = 1.0 / winners.size();
  d_.has_result = true;
  d_.phase = Phase::kGameOver;
  PG_LOG(kGameOver, e_.list = winners; for (int p = 0; p < n_; ++p) {
    auto s = score(p);
    e_.list2.insert(e_.list2.end(), s.begin(), s.end());
  });
}

// ---- observations -----------------------------------------------------------
int State::ObservationSize(const Rules& r) {
  const int n = r.num_players, m = n + (r.has_trust() ? 1 : 0);
  return kNumRestingPhases + (3 + 2) + 1 + (kNumFuels + 1) + kMarketSlots * 11 + 2 +
         r.map.num_cities * kMaxOccupants * m + r.map.num_cities +
         n * (3 + kNumFuels + r.plant_limit + 1) + (r.has_trust() ? kTrustPlantLimit + 1 : 0) +
         (2 + n + 1 + 1);
}

void State::ObservationTensor(int viewer, float* out) const {
  // Perfect-information observation, ego-centric (viewer is player 0). Same
  // layout as observation() in Python, concatenated in key order. With the
  // Trust, occupant index n is the Trust.
  const Rules& r = *rules_;
  const int n = n_, m = n + (r.has_trust() ? 1 : 0);
  float* o = out;
  std::fill(out, out + ObservationSize(r), 0.0f);
  o[static_cast<int>(d_.phase)] = 1.0f;
  o += kNumRestingPhases;
  o[d_.step - 1] = 1.0f;
  o[3] = d_.step3_pending ? 1.0f : 0.0f;
  o[4] = d_.step3_next_round ? 1.0f : 0.0f;
  o += 5;
  *o++ = static_cast<float>(d_.round) / r.max_rounds;
  for (int f = 0; f < kNumFuels; ++f) *o++ = static_cast<float>(d_.fuel_market[f]) / kTrackLen[f];
  *o++ = d_.uranium_stopped ? 1.0f : 0.0f;
  int buyable = static_cast<int>(purchasable().size());
  uint64_t mk = d_.market;
  for (int k = 0; k < kMarketSlots; ++k, o += 11) {
    if (!mk) continue;
    const Plant& pl = plant(PopLowest(&mk));
    o[0] = pl.number / 50.0f;
    o[1 + pl.kind] = 1.0f;
    o[7] = pl.need / 3.0f;
    o[8] = pl.power / 7.0f;
    o[9] = k < buyable ? 1.0f : 0.0f;
    o[10] = pl.number == d_.discount ? 1.0f : 0.0f;
  }
  *o++ = d_.top_pending ? 1.0f : 0.0f;
  *o++ = static_cast<float>(static_cast<double>(d_.real_plug + d_.real_socket) / r.plants.size());
  for (int c = 0; c < r.map.num_cities; ++c)
    for (int s = 0; s < kMaxOccupants; ++s, o += m)
      if (s < d_.num_occupants[c]) {
        int q = d_.occupants[c][s];
        o[q >= n ? q : ((q - viewer) % n + n) % n] = 1.0f;
      }
  for (int c = 0; c < r.map.num_cities; ++c)
    *o++ = d_.region_set >= 0 && (r.region_city_mask[d_.region_set] & Bit(c)) ? 1.0f : 0.0f;
  for (int i = 0; i < n; ++i) {
    int p = (viewer + i) % n;
    int pos = static_cast<int>(std::find(d_.order.begin(), d_.order.begin() + n, p) -
                               d_.order.begin());
    *o++ = d_.money[p] / 300.0f;
    *o++ = static_cast<float>(Popcount(d_.cities[p])) / r.end_cities;
    *o++ = static_cast<float>(pos) / n;
    for (int f = 0; f < kNumFuels; ++f) *o++ = d_.stored[p][f] / 24.0f;
    for (int j = 0; j <= r.plant_limit; ++j)
      *o++ = j < d_.num_plants[p] ? d_.plants[p][j] / 50.0f : 0.0f;
  }
  if (r.has_trust()) {
    std::vector<int> tp = Bits(d_.trust_plants);
    for (int j = 0; j < kTrustPlantLimit; ++j)
      *o++ = j < static_cast<int>(tp.size()) ? tp[j] / 50.0f : 0.0f;
    *o++ = static_cast<float>(d_.trust_houses) / kTrustHouses;
  }
  if (d_.auction_active) {
    const Auction& au = d_.auction;
    int high = au.high < 0 ? n : ((au.high - viewer) % n + n) % n;
    o[0] = au.plant / 50.0f;
    o[1] = au.bid / 300.0f;
    o[2 + high] = 1.0f;
    bool in_ring = std::find(au.ring.begin(), au.ring.begin() + au.ring_len, viewer) !=
                   au.ring.begin() + au.ring_len;
    o[2 + n + 1] = in_ring ? 1.0f : 0.0f;
  }
}

// ---- text -------------------------------------------------------------------
std::string State::ActionToString(int a) const {
  // Like Codec::Describe, but resolves plant slots and fuel choices against this state.
  const Codec& c = codec_;
  if (d_.phase == Phase::kChance) return "CHANCE " + std::to_string(a);
  int p = CurrentPlayer();
  if (d_.phase == Phase::kTrustSetup)
    return "PLACE TRUST HOUSE in " + rules_->map.city_names[a - c.kBuild0];
  if (a >= c.kSelect0 && a < c.kBid0) {
    int num = NthBit(d_.market, a - c.kSelect0);
    return "SELECT plant #" + std::to_string(num) +
           (num == d_.discount ? " (discounted: minimum bid 1)" : "");
  }
  if (a >= c.kDiscard0 && a < c.kRun0)
    return "DISCARD plant #" + std::to_string(d_.plants[p][a - c.kDiscard0]);
  if (d_.phase == Phase::kFuelDiscard && a >= c.kBuy0 && a < c.kBuild0)
    return std::string("RETURN ") + kFuelNames[a - c.kBuy0];
  if (a >= c.kBuild0 && a < c.kDiscard0) {
    int city = a - c.kBuild0, cost = BuildCost(p, city);
    return "BUILD " + rules_->map.city_names[city] + " for " +
           (cost < 0 ? std::string("None") : std::to_string(cost));
  }
  if (a >= c.kRun0) {
    int j = (a - c.kRun0) / (kNumFuels + 1), f = (a - c.kRun0) % (kNumFuels + 1);
    const Plant& pl = plant(d_.plants[p][j]);
    std::string how;
    if (pl.kind == kKindEco) {
      how = "no fuel";
    } else if (pl.kind == kKindHybrid) {
      how = std::to_string(f) + " coal + " + std::to_string(pl.need - f) + " oil";
    } else {
      std::string name = kFuelNames[f];
      std::transform(name.begin(), name.end(), name.begin(), ::tolower);
      how = std::to_string(pl.need) + " " + name;
    }
    return "RUN plant #" + std::to_string(pl.number) + " with " + how;
  }
  return c.Describe(a);
}

std::string State::Describe() const {
  static const char* kPhaseNames[] = {"CHANCE",          "TRUST_SETUP",  "AUCTION_SELECT",
                                      "AUCTION_BID",     "AUCTION_DISCARD", "FUEL_DISCARD",
                                      "BUY_FUEL",        "BUILD",        "BUREAUCRACY",
                                      "GAME_OVER"};
  const MapSpec& m = rules_->map;
  std::ostringstream os;
  os << "Round " << d_.round << ", step " << int(d_.step) << ", phase "
     << kPhaseNames[static_cast<int>(d_.phase)] << ", to act: " << CurrentPlayer() << "\n";
  if (d_.region_set >= 0) {
    os << "Regions in play: ";
    const auto& rs = rules_->region_sets[d_.region_set];
    for (size_t i = 0; i < rs.size(); ++i) os << (i ? ", " : "") << m.region_names[rs[i]];
    os << "\n";
  }
  os << "Turn order:";
  for (int i = 0; i < n_; ++i) os << " P" << int(d_.order[i]);
  os << "\nMarket: ";
  int buyable = static_cast<int>(purchasable().size()), k = 0;
  for (int num : Bits(d_.market)) {
    const Plant& pl = plant(num);
    if (k) os << "; ";
    os << "#" << num << " " << kKindNames[pl.kind] << " burns " << pl.need << " -> "
       << pl.power << " cities" << (k < buyable ? "" : " (future)")
       << (num == d_.discount ? " (discount: minimum bid 1)" : "");
    ++k;
  }
  if (d_.step3_pending) os << (k ? "; " : "") << "STEP 3 card (future)";
  os << "\nPlant stack: " << d_.real_plug + d_.real_socket + (d_.top_pending ? 1 : 0) << " cards"
     << (d_.top_pending ? " (top card is a plug plant, 03-15)" : "");
  os << "\nFuel market (coal/oil/garbage/uranium): ";
  for (int f = 0; f < kNumFuels; ++f) os << (f ? "/" : "") << d_.fuel_market[f];
  os << "  next prices: ";
  for (int f = 0; f < kNumFuels; ++f) {
    int pr = Price(f);
    os << (f ? "/" : "") << (pr < 0 ? std::string("None") : std::to_string(pr));
  }
  if (d_.uranium_stopped) os << "  (no more uranium resupply)";
  for (int p = 0; p < n_; ++p) {
    std::string names;
    for (int c : Bits(d_.cities[p])) names += (names.empty() ? "" : ", ") + m.city_names[c];
    os << "\nP" << p << ": $" << d_.money[p] << ", " << Popcount(d_.cities[p]) << " cities ["
       << names << "], plants "
       << RangeStr(d_.plants[p].begin(), d_.plants[p].begin() + d_.num_plants[p])
       << ", fuel " << RangeStr(d_.stored[p].begin(), d_.stored[p].end());
  }
  if (rules_->has_trust()) {
    std::string blocked;
    for (int c = 0; c < m.num_cities; ++c)
      if (Occupies(c, trust_id())) blocked += (blocked.empty() ? "" : ", ") + m.city_names[c];
    os << "\nTrust: plants " << ListStr(Bits(d_.trust_plants)) << ", houses left "
       << int(d_.trust_houses) << ", in [" << blocked << "]";
  }
  if (d_.auction_active) {
    const Auction& au = d_.auction;
    os << "\nAuction: plant #" << int(au.plant) << " bid " << au.bid << " high "
       << (au.high < 0 ? std::string("None") : std::to_string(au.high)) << " ring "
       << RangeStr(au.ring.begin(), au.ring.begin() + au.ring_len);
  }
  return os.str();
}

std::string State::Dump() const {
  // Canonical state; tools/difftest.py builds the identical string from Python.
  std::ostringstream os;
  auto per_player = [&](auto fn) {
    os << "[";
    for (int p = 0; p < n_; ++p) os << (p ? "," : "") << fn(p);
    os << "]";
  };
  auto flag = [](bool b) { return b ? "1" : "0"; };
  os << "ph=" << static_cast<int>(d_.phase) << " cp=" << CurrentPlayer() << " rd=" << d_.round
     << " st=" << int(d_.step) << " regions=";
  if (d_.region_set >= 0) os << ListStr(rules_->region_sets[d_.region_set]);
  else os << "-";
  os << " order=" << RangeStr(d_.order.begin(), d_.order.begin() + n_)
     << " money=" << RangeStr(d_.money.begin(), d_.money.begin() + n_) << " plants=";
  per_player([&](int p) {
    return RangeStr(d_.plants[p].begin(), d_.plants[p].begin() + d_.num_plants[p]);
  });
  os << " stored=";
  per_player([&](int p) { return RangeStr(d_.stored[p].begin(), d_.stored[p].end()); });
  os << " cities=";
  per_player([&](int p) { return ListStr(Bits(d_.cities[p])); });
  os << " occ=[";
  for (int c = 0; c < rules_->map.num_cities; ++c)
    os << (c ? "," : "")
       << RangeStr(d_.occupants[c].begin(), d_.occupants[c].begin() + d_.num_occupants[c]);
  static const char* const kStarts[] = {"-", "now", "next_round"};
  os << "] market=" << ListStr(Bits(d_.market)) << " dplug=" << ListStr(Bits(d_.deck_plug))
     << " dsock=" << ListStr(Bits(d_.deck_socket))
     << " real=" << ListStr({d_.real_plug, d_.real_socket}) << " top=" << flag(d_.top_pending)
     << " setup=" << int(d_.setup_draws)
     << " disc=" << (d_.discount ? std::to_string(d_.discount) : std::string("-"))
     << " s3p=" << flag(d_.step3_pending) << " s3due=" << flag(d_.step3_removal_due)
     << " s3starts=" << kStarts[d_.step3_starts] << " s3next=" << flag(d_.step3_next_round)
     << " cap=" << (d_.market_cap < 0 ? std::string("-") : std::to_string(d_.market_cap))
     << " bottom=" << ListStr(Bits(d_.bottom)) << " removed=" << ListStr(Bits(d_.removed))
     << " fuel=" << RangeStr(d_.fuel_market.begin(), d_.fuel_market.end())
     << " ustop=" << flag(d_.uranium_stopped) << " done=" << ListStr(Bits(d_.done_auction))
     << " bought=" << flag(d_.bought_any) << " auction=";
  if (d_.auction_active) {
    const Auction& au = d_.auction;
    os << "[" << int(au.plant) << "," << int(au.selector) << "," << au.bid << "," << int(au.high)
       << "," << int(au.pos) << "," << RangeStr(au.ring.begin(), au.ring.begin() + au.ring_len)
       << "]";
  } else {
    os << "-";
  }
  os << " disc_player=" << (d_.discarder < 0 ? std::string("-") : std::to_string(d_.discarder))
     << " queue=" << RangeStr(d_.queue.begin(), d_.queue.begin() + d_.qlen)
     << " qpos=" << int(d_.qpos) << " ran=";
  per_player([&](int p) { return ListStr(Bits(d_.ran[p])); });
  os << " powered=" << RangeStr(d_.powered.begin(), d_.powered.begin() + n_)
     << " trust=[" << ListStr(Bits(d_.trust_plants)) << ","
     << RangeStr(d_.trust_stored.begin(), d_.trust_stored.end()) << ","
     << int(d_.trust_houses) << "," << flag(d_.trust_due) << "," << flag(d_.trust_took) << ","
     << RangeStr(d_.trust_setup.begin() + (kTrustStartHouses - d_.trust_setup_len),
                 d_.trust_setup.end())
     << "] result=";
  char buf[32];
  if (d_.has_result) {
    os << "[";
    for (int p = 0; p < n_; ++p) {
      std::snprintf(buf, sizeof buf, "%.6f", d_.result[p]);
      os << (p ? "," : "") << buf;
    }
    os << "]";
  } else {
    os << "-";
  }
  os << " legal=" << ListStr(LegalActions()) << " probs=";
  if (IsChanceNode()) {
    os << "[";
    bool first = true;
    for (const auto& o : ChanceOutcomes()) {
      std::snprintf(buf, sizeof buf, "%.9f", o.second);
      os << (first ? "" : ",") << buf;
      first = false;
    }
    os << "]";
  } else {
    os << "-";
  }
  return os.str();
}

void State::CheckInvariants() const {
  const Rules& r = *rules_;
  uint64_t owned = 0;
  for (int p = 0; p < n_; ++p) {
    Invariant(d_.money[p] >= 0, "negative money");
    if (!(d_.phase == Phase::kAuctionDiscard && p == d_.discarder))
      Invariant(d_.num_plants[p] <= r.plant_limit, "plant limit");
    if (!(d_.phase == Phase::kFuelDiscard && p == d_.discarder))
      Invariant(Fits(p, d_.stored[p]), "storage overflow");
    Invariant(Popcount(d_.cities[p]) <= r.houses, "out of houses");
    if (d_.cities[p])
      Invariant(d_.region_set >= 0 && !(d_.cities[p] & ~r.region_city_mask[d_.region_set]),
                "city not in play");
    for (int j = 0; j < d_.num_plants[p]; ++j) {
      Invariant(!(owned & Bit(d_.plants[p][j])), "plant owned twice");
      owned |= Bit(d_.plants[p][j]);
    }
  }
  for (int f = 0; f < kNumFuels; ++f) {
    int total = d_.fuel_market[f] + d_.trust_stored[f];
    for (int p = 0; p < n_; ++p) total += d_.stored[p][f];
    Invariant(total <= r.fuel_totals[f], "fuel created from nothing");
  }
  for (int c = 0; c < r.map.num_cities; ++c) {
    int k = d_.num_occupants[c], players = 0;
    Invariant(k <= kMaxOccupants, "too many occupants");
    for (int i = 0; i < k; ++i) {
      players += d_.occupants[c][i] < n_;
      for (int j = i + 1; j < k; ++j)
        Invariant(d_.occupants[c][i] != d_.occupants[c][j], "duplicate occupant");
    }
    Invariant(players <= d_.step, "too many houses");
  }
  uint64_t all = 0;
  for (const Plant& pl : r.plants) all |= Bit(pl.number);
  const uint64_t parts[] = {d_.market, d_.deck_plug, d_.deck_socket, d_.bottom,
                            d_.removed, owned,     d_.trust_plants};
  uint64_t seen = 0;
  int count = 0;
  for (uint64_t x : parts) {
    seen |= x;
    count += Popcount(x);
  }
  Invariant(seen == all && count == Popcount(all), "plant conservation broken");
  int limit = kMarketSize[d_.step] - (d_.step3_pending ? 1 : 0);
  Invariant(Popcount(d_.market) <= limit, "market too large");
  Invariant(d_.real_plug >= 0 &&
                d_.real_plug <= Popcount(d_.deck_plug) - (d_.top_pending ? 1 : 0),
            "real plug count");
  Invariant(d_.real_socket >= 0 && d_.real_socket <= Popcount(d_.deck_socket),
            "real socket count");
  Invariant(!d_.discount || (d_.market & Bit(d_.discount)), "discount on a missing plant");
  Invariant(Popcount(d_.trust_plants) <= kTrustPlantLimit, "Trust plant limit");
}

}  // namespace powergrid
